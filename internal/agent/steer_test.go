package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// roles renders a message list compactly, for failure messages.
func roles(msgs []provider.Message) string {
	var b strings.Builder
	for i, m := range msgs {
		if i > 0 {
			b.WriteString(" ")
		}
		switch {
		case m.Role == provider.RoleAssistant && len(m.ToolCalls) > 0:
			b.WriteString("assistant(tools)")
		case m.Role == provider.RoleTool:
			b.WriteString("tool")
		default:
			b.WriteString(string(m.Role))
		}
	}
	return b.String()
}

// Steering must land after the round's tool results; the provider rejects a
// request whose tool results are not immediately after their call.
func TestSteeringIsInsertedAtATurnBoundary(t *testing.T) {
	ft := &fakeTool{name: "probe", readOnly: true}
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "c1", Name: "probe", Args: `{}`}}},
		{text: "done"},
	}}
	sink := &collectSink{}
	a := newAgent(t, client, sink, ft)

	// Steer while the first request is in flight, the moment steering is for.
	client.onStream = func() { a.Steer("use the other file") }

	if err := a.Run(context.Background(), "start"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if err := Validate(a.Messages()); err != nil {
		t.Fatalf("conversation invalid after steering: %v\n%s", err, roles(a.Messages()))
	}

	req := client.requests()[1]
	steerAt, toolAt := -1, -1
	for i, m := range req {
		if m.Role == provider.RoleTool {
			toolAt = i
		}
		if m.Role == provider.RoleUser && m.Content == "use the other file" {
			steerAt = i
		}
	}
	if steerAt < 0 {
		t.Fatalf("the steering message never reached the model: %s", roles(req))
	}
	if toolAt < 0 || steerAt < toolAt {
		t.Errorf("steering landed at %d, tool result at %d: %s", steerAt, toolAt, roles(req))
	}
}

func TestSteeringAfterTheModelStopsReopensTheTurn(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "Here is my answer."},
		{text: "Revised answer."},
	}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)

	steered := false
	client.onStream = func() {
		if !steered {
			steered = true
			a.Steer("no, the other way")
		}
	}

	if err := a.Run(context.Background(), "start"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if n := len(client.requests()); n != 2 {
		t.Fatalf("%d requests, want 2 -- the steering message should have re-opened the turn", n)
	}
	if got := sink.text(); !strings.Contains(got, "Revised answer.") {
		t.Errorf("text = %q, want the revised answer", got)
	}
	if a.Pending() != 0 {
		t.Errorf("%d steering messages left queued after the run", a.Pending())
	}
}

func TestSteeringQueuedBetweenTurnsIsDeliveredFirst(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "ok"}}}
	a := newAgent(t, client, &collectSink{})

	a.Steer("typed as the last turn ended")
	if a.Pending() != 1 {
		t.Fatalf("Pending = %d, want 1", a.Pending())
	}

	if err := a.Run(context.Background(), "the next prompt"); err != nil {
		t.Fatal(err)
	}

	req := client.requests()[0]
	var users []string
	for _, m := range req {
		if m.Role == provider.RoleUser {
			users = append(users, m.Content)
		}
	}
	want := []string{"typed as the last turn ended", "the next prompt"}
	if len(users) != 2 || users[0] != want[0] || users[1] != want[1] {
		t.Errorf("user messages = %q, want %q", users, want)
	}
}

func TestSteerIgnoresBlankInput(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	for _, s := range []string{"", "   ", "\n\t "} {
		if a.Steer(s) {
			t.Errorf("Steer(%q) queued a blank message", s)
		}
	}
	if a.Pending() != 0 {
		t.Errorf("Pending = %d, want 0", a.Pending())
	}
}

// Steering is announced so the scrollback shows why the model changed course.
func TestSteeringEmitsAnEvent(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "one"}, {text: "two"}}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)

	once := false
	client.onStream = func() {
		if !once {
			once = true
			a.Steer("correction")
		}
	}
	if err := a.Run(context.Background(), "start"); err != nil {
		t.Fatal(err)
	}

	var got string
	sink.mu.Lock()
	for _, e := range sink.events {
		if e.Kind == EvSteer {
			got = e.Text
		}
	}
	sink.mu.Unlock()
	if got != "correction" {
		t.Errorf("EvSteer text = %q, want %q", got, "correction")
	}
}

// Auto-compaction fires inside the run, at a turn boundary, and the run
// continues afterwards.
func TestAutoCompactionRunsInlineAndTheTurnContinues(t *testing.T) {
	// Summarisation runs at the top of the loop, before any agent turn; this
	// prose session folds into the checkpoint in two pieces.
	summary := scriptedTurn{text: "## Goal\nFinish the work.\n\n## Next Steps\n1. Carry on."}
	client := &scriptedClient{charsPerToken: 4, turns: []scriptedTurn{
		summary,
		summary,
		{text: "carrying on"},
	}}
	ft := &fakeTool{name: "probe", readOnly: true}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()), ft)
	sink := &collectSink{}

	// A 20k window with a 4k reserve: the budget is 16k.
	a := New(client, "m", exec, sink, Options{
		MaxIterations: 10, ContextLimit: 20000, WarnPercent: 80,
		AutoCompact: true, ReserveTokens: 4000, KeepRecentTokens: 1000,
	})
	a.SetSystem(strings.Repeat("system prompt. ", 100))
	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: strings.Repeat("earlier work. ", 4000)},
		{Role: provider.RoleAssistant, Content: strings.Repeat("earlier reply. ", 1000)},
	})

	if got := a.Usable(); got != 16000 {
		t.Fatalf("Usable = %d, want 16000", got)
	}
	if before := a.estimate(a.transcriptChars()); before < a.Usable() {
		t.Fatalf("test setup: used = %d, want it over the %d budget", before, a.Usable())
	}

	if err := a.Run(context.Background(), "keep going"); err != nil {
		t.Fatalf("Run returned %v; auto-compaction should have made room and continued", err)
	}

	var compacted *CompactResult
	sink.mu.Lock()
	for _, e := range sink.events {
		if e.Kind == EvCompacted {
			compacted = e.Compaction
		}
	}
	sink.mu.Unlock()

	if compacted == nil {
		t.Fatal("no compaction happened; the run should not have reached the model with a full window")
	}
	// Summarising is non-destructive: the checkpoint represents the dropped
	// prefix, so the request may cost a few tokens more than the tail alone.
	if compacted.SummarisedThrough <= 0 {
		t.Errorf("the checkpoint covers nothing: SummarisedThrough = %d", compacted.SummarisedThrough)
	}
	if len(a.Messages()) < 2 {
		t.Errorf("the transcript was destroyed: %d messages left", len(a.Messages()))
	}
	if after := a.ContextState().Projected; after >= a.Usable() {
		t.Errorf("still over budget after compaction: %d of %d", after, a.Usable())
	}
	if err := Validate(a.Messages()); err != nil {
		t.Errorf("conversation invalid after auto-compaction: %v", err)
	}
	if got := sink.text(); !strings.Contains(got, "carrying on") {
		t.Errorf("text = %q, want the run to have continued past the compaction", got)
	}
}

// The cut never lands on a tool result: a result separated from its call fails
// Validate.
func TestCutPointNeverCutsAToolResultFromItsCall(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	a.messages = []provider.Message{
		{Role: provider.RoleUser, Content: strings.Repeat("a", 4000)},
		{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{{ID: "t1", Name: "grep"}}},
		{Role: provider.RoleTool, ToolCallID: "t1", Content: strings.Repeat("b", 4000)},
		{Role: provider.RoleTool, ToolCallID: "t2", Content: strings.Repeat("c", 4000)},
		{Role: provider.RoleAssistant, Content: "done"},
		{Role: provider.RoleUser, Content: strings.Repeat("d", 4000)},
		{Role: provider.RoleAssistant, Content: "and again"},
	}
	for budget := 10; budget < 4000; budget += 137 {
		cut := a.startPoint(budget)
		if cut < len(a.messages) && a.messages[cut].Role == provider.RoleTool {
			t.Fatalf("budget %d: cut at %d lands on a tool result", budget, cut)
		}
		if cut == 0 {
			t.Fatalf("budget %d: cut at 0 leaves nothing to summarise", budget)
		}
		if cut > len(a.messages) {
			t.Fatalf("budget %d: cut at %d is past the end", budget, cut)
		}
	}
}
