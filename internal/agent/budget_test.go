package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// fillSession sizes a two-message conversation to roughly want tokens. The
// ratio is deterministic because a fresh agent has not calibrated yet.
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
	// The transcript, not the assembled request: an over-budget request is
	// trimmed to fit.
	if got := a.estimate(a.transcriptChars()); got < want-4 || got > want+4 {
		t.Fatalf("fillSession: used = %d, want about %d", got, want)
	}
}

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

// Prompt plus cap must fit the window: a request must not have room to overrun
// it.
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
// number.
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

	// Filled to leave under minOutputTokens of room to answer.
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

func TestCompactionRunsWithThinkingOff(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "## Goal\nDo the thing.\n\n## Next Steps\n1. Continue."},
	}}
	a := newAgent(t, client, &collectSink{})
	// Large enough that compaction pays for itself.
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
