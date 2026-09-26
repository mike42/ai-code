package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// fillSession sets a two-message conversation sized to occupy roughly want
// tokens, so tests can stand the session at a chosen point in the window.
//
// Deterministic because a fresh agent has not calibrated yet: the ratio is
// defaultCharsPerToken until a response has been measured.
func fillSession(t *testing.T, a *Agent, want int) {
	t.Helper()
	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: "start"},
		{Role: provider.RoleAssistant, Content: ""},
	})
	pad := int(float64(want)*a.charsPerTokenNow()) - a.contextChars()
	if pad < 1 {
		t.Fatalf("a %d-token session is smaller than the system prompt and tool schemas", want)
	}
	a.messages[1].Content = strings.Repeat("x", pad)
	// The transcript, not the assembled request: once a session is over budget
	// the request is trimmed to fit, so measuring that would report the cap
	// back rather than how full the session actually is.
	if got := a.estimate(a.transcriptChars()); got < want-4 || got > want+4 {
		t.Fatalf("fillSession: used = %d, want about %d", got, want)
	}
}

// The cap has to track the window as the session fills. A flat number is wrong
// at both ends: too small it truncates a long answer for no reason connected to
// the session, too large it lets a thinking model generate straight through the
// compaction reserve and overflow mid-stream.
func TestMaxTokensShrinksAsTheSessionFills(t *testing.T) {
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

	if err := a.Run(context.Background(), "search for it"); err != nil {
		t.Fatal(err)
	}

	caps := client.maxTokens()
	t.Logf("caps sent: %v", caps)
	if len(caps) < 3 {
		t.Fatalf("only %d requests were made", len(caps))
	}
	for i := 1; i < len(caps); i++ {
		if caps[i] >= caps[i-1] {
			t.Errorf("cap went from %d to %d after 40k of tool output was appended; "+
				"caps = %v", caps[i-1], caps[i], caps)
		}
	}
}

// Whatever the cap is, prompt plus cap must fit. This is the property the whole
// change exists for: the request the model is answering cannot be allowed to
// have room to overrun the window it is being answered in.
func TestMaxTokensNeverExceedsTheRemainingWindow(t *testing.T) {
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

	if err := a.Run(context.Background(), "search"); err != nil {
		t.Fatal(err)
	}

	limit := a.opts.ContextLimit
	for i, req := range client.reqs {
		prompt := client.requestTokens(req)
		if req.MaxTokens <= 0 {
			t.Errorf("request %d sent no cap although the window is known", i)
			continue
		}
		t.Logf("request %d: prompt %d + cap %d = %d of %d", i, prompt, req.MaxTokens,
			prompt+req.MaxTokens, limit)
		if prompt+req.MaxTokens > limit {
			t.Errorf("request %d: prompt %d + cap %d exceeds the %d-token window",
				i, prompt, req.MaxTokens, limit)
		}
	}
}

// An explicit agent.max_tokens is a ceiling. Honouring it as written would
// reintroduce exactly the failure this change fixes: OpenCode sends 32000
// blindly and overruns a window that had a million tokens available.
func TestConfiguredMaxTokensIsACeilingNotAFloor(t *testing.T) {
	for _, tc := range []struct {
		name       string
		configured int
		limit      int
		want       func(sent int) bool
		desc       string
	}{
		{
			name: "below the room left", configured: 256, limit: 65536,
			want: func(sent int) bool { return sent == 256 },
			desc: "the configured ceiling, since it is the smaller of the two",
		},
		{
			name: "larger than the window", configured: 1000000, limit: 8192,
			want: func(sent int) bool { return sent > 0 && sent <= 8192-BudgetFor(8192).Reserve },
			desc: "the computed cap, because the configured one does not fit",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &scriptedClient{turns: []scriptedTurn{{text: "ok"}}}
			exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()))
			a := New(client, "m", exec, &collectSink{}, Options{
				MaxTokens: tc.configured, ContextLimit: tc.limit,
			})
			if err := a.Run(context.Background(), "go"); err != nil {
				t.Fatal(err)
			}
			sent := client.maxTokens()[0]
			if !tc.want(sent) {
				t.Errorf("max_tokens = %d for a configured %d in a %d window; want %s",
					sent, tc.configured, tc.limit, tc.desc)
			}
		})
	}
}

// An unknown window is a reason to let the server decide, not to guess at a
// number: it knows its own --n-predict and we do not.
func TestUnknownWindowSendsNoCap(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "ok"}}}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()))
	a := New(client, "m", exec, &collectSink{}, Options{ContextLimit: 0})

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if got := client.maxTokens()[0]; got != 0 {
		t.Errorf("max_tokens = %d with an unknown window, want 0 so the server decides", got)
	}
}

// A backend that advertises its own completion limit caps us too. Exceeding it
// is a request the backend rejects, which is worse than a short answer.
func TestBackendOutputLimitLowersTheCap(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "ok"}}}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()))
	a := New(client, "m", exec, &collectSink{}, Options{
		ContextLimit: 65536, MaxOutputTokens: 2048,
	})
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if got := client.maxTokens()[0]; got != 2048 {
		t.Errorf("max_tokens = %d, want the backend's advertised 2048", got)
	}
}

// Room to sit in is not room to answer in. A turn that fits but leaves a
// hundred tokens for the reply is a request that cannot produce anything
// useful, so compaction has to happen before it is sent rather than after it
// comes back truncated.
func TestNoRoomToAnswerCompactsInsteadOfSendingADoomedRequest(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "## Goal\nDo the thing.\n\n## Next Steps\n1. Continue."},
		{text: "carrying on"},
	}}
	sink := &collectSink{}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()))
	a := New(client, "m", exec, sink, Options{
		ContextLimit: 8192, AutoCompact: true, KeepRecentTokens: 100,
	})

	// Filled to within 300 tokens of the budget: enough to send, not enough to
	// answer. Derived, so the margin survives a change to the reserve.
	fillSession(t, a, a.Usable()-300)
	if room := a.Usable() - a.estimate(a.transcriptChars()); room >= minOutputTokens {
		t.Fatalf("test setup leaves %d tokens, which is above the %d floor", room, minOutputTokens)
	}

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	compacted := false
	for _, e := range sink.events {
		if e.Kind == EvCompacted {
			compacted = true
		}
	}
	if !compacted {
		t.Errorf("no compaction happened; notices = %q", sink.notices())
	}

	caps := client.maxTokens()
	t.Logf("caps sent: %v", caps)
	if len(caps) < 2 {
		t.Fatalf("expected a summarisation call and a turn, got %d requests", len(caps))
	}
	if got := caps[len(caps)-1]; got < minOutputTokens {
		t.Errorf("the turn after compaction was sent with a %d-token cap, under the %d floor",
			got, minOutputTokens)
	}
}

// Summarising is mechanical work. On a local model at 1-2 tokens/sec, letting a
// max-effort model think about it first costs minutes for a checkpoint that
// reads no better -- and it is charged exactly when the session is already
// stalled waiting for room.
func TestCompactionRunsWithThinkingOff(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "## Goal\nDo the thing.\n\n## Next Steps\n1. Continue."},
	}}
	a := newAgent(t, client, &collectSink{})
	// Large enough that a compaction actually pays for itself: a session
	// smaller than the checkpoint it would be given is declined outright.
	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: strings.Repeat("earlier work. ", 8000)},
		{Role: provider.RoleAssistant, Content: strings.Repeat("and the reply. ", 8000)},
	})

	if _, err := a.Compact(context.Background(), 100); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if got := client.lastFullRequest().Effort; got != provider.EffortNone {
		t.Errorf("summarisation ran at effort %q, want %q", got, provider.EffortNone)
	}
}
