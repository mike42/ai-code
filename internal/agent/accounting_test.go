package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

func toolExecutor(t *testing.T) *tool.LocalExecutor {
	t.Helper()
	return tool.NewLocalExecutor(tool.NewState(t.TempDir()))
}

// realisticSystemPrompt is large enough that the first response is a usable
// calibration sample.
var realisticSystemPrompt = strings.Repeat(
	"You are ai-code, a coding agent. Prefer the dedicated tools over shell commands. ", 60)

// bigTool stands in for grep or bash returning a large result.
type bigTool struct {
	name string
	out  string
}

func (b *bigTool) Name() string            { return b.name }
func (b *bigTool) Description() string     { return "returns a lot of output" }
func (b *bigTool) ReadOnly() bool          { return true }
func (b *bigTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (b *bigTool) Run(context.Context, *tool.State, json.RawMessage) tool.Result {
	return tool.Result{Content: b.out}
}

func TestContextUsedNeverFallsWhileTheConversationGrows(t *testing.T) {
	out := strings.Repeat("internal/render/screen.go:142: func (s *Screen) commitLocked\n", 650)

	bt := &bigTool{name: "grep", out: out}
	// Nine characters a token: repeated structured paths tokenise denser than
	// four.
	client := &scriptedClient{
		charsPerToken: 9,
		turns: []scriptedTurn{
			{calls: []provider.ToolCall{{ID: "c1", Name: "grep", Args: `{}`}}},
			{calls: []provider.ToolCall{{ID: "c2", Name: "grep", Args: `{}`}}},
			{text: "done"},
		},
	}

	a := newAgent(t, client, &collectSink{}, bt)
	a.SetSystem(realisticSystemPrompt)

	var series []int
	record := func(label string) {
		used := a.ContextState().Projected
		series = append(series, used)
		t.Logf("%-28s used=%d", label, used)
	}

	client.onStream = func() { record("about to send a request") }

	if err := a.Run(context.Background(), "search for it"); err != nil {
		t.Fatal(err)
	}
	record("after the run")

	for i := 1; i < len(series); i++ {
		if series[i] < series[i-1] {
			t.Errorf("context usage fell from %d to %d with nothing removed from the conversation; "+
				"series = %v", series[i-1], series[i], series)
		}
	}
}

func TestContextEstimateIsCloseToTheProvidersCount(t *testing.T) {
	out := strings.Repeat("internal/render/screen.go:142: func (s *Screen) commitLocked\n", 650)
	bt := &bigTool{name: "grep", out: out}

	client := &scriptedClient{
		charsPerToken: 9,
		turns: []scriptedTurn{
			{calls: []provider.ToolCall{{ID: "c1", Name: "grep", Args: `{}`}}},
			{calls: []provider.ToolCall{{ID: "c2", Name: "grep", Args: `{}`}}},
			{text: "done"},
		},
	}
	a := newAgent(t, client, &collectSink{}, bt)
	a.SetSystem(realisticSystemPrompt)

	var predicted []int
	client.onStream = func() { predicted = append(predicted, a.ContextState().Projected) }

	if err := a.Run(context.Background(), "search"); err != nil {
		t.Fatal(err)
	}

	// The first prediction has no calibration behind it; later ones do.
	for i, req := range client.requests() {
		if i == 0 {
			continue
		}
		actual := 0
		for _, m := range req {
			actual += messageChars(m)
		}
		actual = int(float64(actual) / 9)

		got := predicted[i]
		t.Logf("request %d: predicted %d, actual %d", i, got, actual)
		if diff := float64(got-actual) / float64(actual); diff > 0.15 || diff < -0.15 {
			t.Errorf("request %d: predicted %d tokens, the server counted %d (%.0f%% out)",
				i, got, actual, diff*100)
		}
	}
}

// The system prompt and tool schemas are sent on every request, so a resumed
// session must count them.
func TestResumedSessionCountsTheSystemPromptAndTools(t *testing.T) {
	ft := &fakeTool{name: "probe", readOnly: true}
	a := newAgent(t, &scriptedClient{}, &collectSink{}, ft)
	a.SetSystem(strings.Repeat("You are a coding agent. ", 400)) // ~9600 chars

	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: "hello"},
	})

	used := a.ContextState().Projected
	t.Logf("resumed session reports used=%d", used)
	if used < 2000 {
		t.Errorf("used = %d for a session whose system prompt alone is ~2400 tokens; "+
			"the system prompt and tool schemas are sent on every request and must be counted", used)
	}
}

func TestImplausibleUsageDoesNotPoisonTheEstimate(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})

	report := func(chars, promptTokens int) (provider.Message, bool) {
		m := provider.Message{Role: provider.RoleAssistant, Content: "ok"}
		ok := a.recordUsage(&m, chars, provider.Usage{PromptTokens: promptTokens})
		return m, ok
	}

	if _, ok := report(90000, 10000); !ok { // a normal request: 9 chars/token
		t.Fatal("a normal usage report was rejected")
	}
	if got := a.charsPerTokenNow(); got < 8.9 || got > 9.1 {
		t.Fatalf("ratio = %.2f after a normal response, want 9", got)
	}

	// The same prompt reported as 200 tokens, because 9800 of them were cached.
	m, ok := report(90000, 200)
	if ok {
		t.Error("a cache-hit usage report was believed")
	}
	if m.PromptTokens != 0 {
		t.Errorf("message carries prefill %d from a report that was not believable", m.PromptTokens)
	}
	if got := a.charsPerTokenNow(); got < 8.9 || got > 9.1 {
		t.Errorf("ratio = %.2f after a cache-hit usage report, want it unchanged at 9", got)
	}

	// 3.0 chars/token is inside the sanity band but far below the measured 9.
	if _, ok := report(90000, 3000); ok {
		t.Error("a report a third of the measured size was believed")
	}
	if got := a.charsPerTokenNow(); got < 8.9 || got > 9.1 {
		t.Errorf("ratio = %.2f after an under-report, want it unchanged at 9", got)
	}

	// A sample too small to learn from: the figure is accepted but the ratio
	// must not move.
	if _, ok := report(120, 40); !ok {
		t.Error("a small request's usage was rejected")
	}
	if got := a.charsPerTokenNow(); got < 8.9 || got > 9.1 {
		t.Errorf("ratio = %.2f after a tiny sample, want it unchanged at 9", got)
	}
}

func TestCompactionResetsTheGroundTruthFigures(t *testing.T) {
	client := &scriptedClient{charsPerToken: 9, turns: []scriptedTurn{
		{text: "ok"},
		{text: "## Goal\nDo the thing.\n\n## Next Steps\n1. Continue."},
	}}
	a := newAgent(t, client, &collectSink{})
	a.SetSystem(realisticSystemPrompt)

	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: strings.Repeat("some earlier work. ", 3000)},
		{Role: provider.RoleAssistant, Content: strings.Repeat("and the reply. ", 3000)},
	})
	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatal(err)
	}
	before := a.ContextState()
	if !before.Anchored {
		t.Fatal("usage should be ground truth after a response")
	}

	if _, err := a.Compact(context.Background(), 500); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	after := a.ContextState()
	t.Logf("compaction: %d -> %d tokens", before.Projected, after.Projected)
	if after.Projected >= before.Projected {
		t.Errorf("compaction left usage at %d, up from %d", after.Projected, before.Projected)
	}
	// The projection must match the request that will be sent now.
	if want := a.estimate(a.fixedChars() + charsOf(a.request().msgs)); after.Projected != want {
		t.Errorf("after compaction the projection is %d, but the next request is %d tokens",
			after.Projected, want)
	}
	if after.Anchored {
		t.Error("the projection claims a reported figure describes a prefix compaction just replaced")
	}
}

func TestLengthStopNamesWhoImposedTheCap(t *testing.T) {
	for _, tc := range []struct {
		name      string
		maxTokens int
		limit     int
		want      string
	}{
		{name: "the user's ceiling", maxTokens: 4096, limit: 65536, want: "agent.max_tokens"},
		{name: "no window known", maxTokens: 0, limit: 0, want: "server's own output limit"},
		{name: "the session is full", maxTokens: 0, limit: 65536, want: "/compact"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &scriptedClient{turns: []scriptedTurn{{text: "cut off", stop: provider.StopLength}}}
			sink := &collectSink{}
			exec := toolExecutor(t)
			a := New(client, "m", exec, sink, Options{MaxTokens: tc.maxTokens, ContextLimit: tc.limit})

			if err := a.Run(context.Background(), "go"); err != nil {
				t.Fatal(err)
			}
			found := false
			for _, n := range sink.notices() {
				if strings.Contains(n, tc.want) {
					found = true
				}
			}
			if !found {
				t.Errorf("max_tokens=%d limit=%d: notices %q, want one naming %q",
					tc.maxTokens, tc.limit, sink.notices(), tc.want)
			}
		})
	}
}

func TestLengthStopShortOfOurCapBlamesTheServer(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{
		text:  "cut off",
		stop:  provider.StopLength,
		usage: provider.Usage{PromptTokens: 100, CompletionTokens: 200},
	}}}
	sink := &collectSink{}
	a := New(client, "m", toolExecutor(t), sink, Options{ContextLimit: 65536})

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, n := range sink.notices() {
		if strings.Contains(n, "the server's own") {
			found = true
		}
	}
	if !found {
		t.Errorf("notices = %q, want one blaming the server for a 200-token stop under a ~49k cap",
			sink.notices())
	}
}

func TestContextUsedFallsWhenTheRequestShrinks(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.opts.ReserveTokens = DefaultReserveTokens
	a.opts.KeepRecentTokens = DefaultKeepRecentTokens
	a.charsPerToken = 4

	// A session well past its window: one repeated bulky tool result.
	bulk := strings.Repeat("match at offset 0x0000 in section .text\n", 400)
	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: "search the binary"})
	for i := 0; i < 140; i++ {
		a.messages = append(a.messages,
			provider.Message{Role: provider.RoleAssistant,
				ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "c", Content: bulk})
	}

	// Ground truth lives on the assistant message it describes, so the figure
	// and the messages cannot drift apart.
	last := &a.messages[len(a.messages)-2]
	last.PromptTokens = a.estimate(a.contextChars())
	last.Completion = 400
	before := a.ContextState().Projected

	a.setSummary(strings.Repeat("## Goal\nfind the prologues.\n", 40), a.startPoint(a.keepRecentTokens()))

	sent := a.messagesToSend()
	if len(sent) >= len(a.messages) {
		t.Fatalf("the request did not shrink: %d of %d messages, so this test proves nothing",
			len(sent), len(a.messages))
	}

	actual := a.estimate(a.fixedChars() + charsOf(sent))
	after := a.ContextState().Projected
	t.Logf("before=%d after=%d actual=%d (request %d of %d messages)",
		before, after, actual, len(sent), len(a.messages))

	if after >= before {
		t.Errorf("Used = %d after compaction, was %d before: the figure never fell", after, before)
	}
	// Wide tolerance: the estimate is meant to be close, not exact.
	if after > actual*3/2 {
		t.Errorf("Used = %d, but the request is really about %d tokens", after, actual)
	}
}

func TestAssemblyDoesNotResurrectSummarisedMessages(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.opts.ReserveTokens = DefaultReserveTokens
	a.opts.KeepRecentTokens = DefaultKeepRecentTokens
	a.charsPerToken = 4

	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: "search it"})
	for i := 0; i < 400; i++ {
		a.messages = append(a.messages,
			provider.Message{Role: provider.RoleAssistant,
				ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "c",
				Content: strings.Repeat("ordinary result line\n", 100)})
	}
	// The turn that ends the session: a result far larger than the tail budget.
	a.messages = append(a.messages,
		provider.Message{Role: provider.RoleAssistant,
			ToolCalls: []provider.ToolCall{{ID: "big", Name: "bash", Args: `{}`}}},
		provider.Message{Role: provider.RoleTool, ToolCallID: "big", IsError: true,
			Content: strings.Repeat("record 000000 at offset 000000 section\n", 12300)})

	a.summary = strings.Repeat("## Goal\nfind them.\n", 40)
	a.summarisedThrough = a.checkpointThrough(SummaryMaxTokens)
	if a.summarisedThrough != len(a.messages) {
		t.Fatalf("checkpoint covers %d of %d messages; this test needs it to cover all of them",
			a.summarisedThrough, len(a.messages))
	}

	sent := a.messagesToSend()
	for _, m := range sent {
		if m.ToolCallID == "big" {
			t.Fatalf("the summarised result is back in the request: %d messages, %d tokens",
				len(sent), a.estimate(a.fixedChars()+charsOf(sent)))
		}
	}
	if size := a.estimate(a.fixedChars() + charsOf(sent)); size > a.promptBudget() {
		t.Errorf("request is %d tokens, over the %d budget", size, a.promptBudget())
	}
}

// No single tool result may be larger than the conversation can carry.
func TestOversizedToolResultIsClampedOnArrival(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	a.opts.ContextLimit = 262144
	a.opts.KeepRecentTokens = DefaultKeepRecentTokens
	a.charsPerToken = 4

	huge := strings.Repeat("record 000000 at offset 000000 section\n", 12300)
	msgs := []provider.Message{
		{Role: provider.RoleTool, ToolCallID: "big", Content: huge},
		{Role: provider.RoleUser, Content: huge},
	}
	a.clampOversized(msgs)

	if limit := a.maxMessageChars(); len(msgs[0].Content) > limit+len(huge)/10 {
		t.Errorf("tool result is %d chars, want about %d", len(msgs[0].Content), limit)
	}
	if !strings.Contains(msgs[0].Content, "too large for the context window") {
		t.Error("the model was not told the result was cut")
	}
	// A pasted message is meant to be enormous; cutting it silently is worse
	// than the context it costs.
	if len(msgs[1].Content) != len(huge) {
		t.Error("a user message was clamped")
	}
}
