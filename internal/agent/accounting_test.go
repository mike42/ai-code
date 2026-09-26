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

// realisticSystemPrompt stands in for ai-code's real one, which with the tool
// schemas is several thousand characters. Its size is the point: it is what
// makes the very first response a large enough sample to calibrate against.
var realisticSystemPrompt = strings.Repeat(
	"You are ai-code, a coding agent. Prefer the dedicated tools over shell commands. ", 60)

// bigTool returns a large result, the way grep or bash does.
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

// The reported number must never fall while the conversation is only growing.
//
// It did. The count mixes two measures: the provider's prompt_tokens, which is
// ground truth, and a chars/4 guess for everything appended since. When the
// guess overshoots -- and on tool output it overshoots badly, because code and
// paths tokenise denser than four characters a token -- the next response
// replaces it with the real figure and the display drops. Nothing was pruned;
// the earlier number was simply wrong and too high.
func TestContextUsedNeverFallsWhileTheConversationGrows(t *testing.T) {
	// 40k of realistic tool output. Roughly 4 chars/token by the old guess,
	// nearer 9 in reality for repeated structured text.
	out := strings.Repeat("internal/render/screen.go:142: func (s *Screen) commitLocked\n", 650)

	bt := &bigTool{name: "grep", out: out}
	// Nine characters a token: structured output with repeated paths is far
	// denser than the four the old estimator assumed, which is precisely the
	// case that made the count jump backwards.
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

// The estimate for what has been appended since the last response has to be
// close, not merely monotone. It is what decides when compaction fires, so a
// reading that runs 2x high compacts a session that had half its window left.
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

	// Capture what ai-code predicted just before each request, against what the
	// server then said the request actually cost.
	var predicted []int
	client.onStream = func() { predicted = append(predicted, a.ContextState().Projected) }

	if err := a.Run(context.Background(), "search"); err != nil {
		t.Fatal(err)
	}

	// The first prediction is made with no calibration yet, so it is allowed to
	// be poor; every one after has a measured ratio behind it.
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

// A resumed session reports what is really in the window, which includes the
// system prompt and every tool schema -- they are sent on every request and on
// this codebase they are thousands of tokens.
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

// Some llama.cpp builds report prompt_tokens as the tokens actually processed,
// which on a prompt-cache hit is a small fraction of the real prompt. Believing
// that figure would set the ratio to something absurd and wreck every later
// estimate, so an implausible reading is discarded and the previous one kept.
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

	// A plausible-looking ratio that is still far below what the conversation
	// can possibly be: 30,000 tokens reported for 90,000 characters reads as
	// 3.0 chars a token, inside the sanity band, but this model has been
	// measured at 9. Believing it would understate the window by two thirds.
	if _, ok := report(90000, 3000); ok {
		t.Error("a report a third of the measured size was believed")
	}
	if got := a.charsPerTokenNow(); got < 8.9 || got > 9.1 {
		t.Errorf("ratio = %.2f after an under-report, want it unchanged at 9", got)
	}

	// And a sample too small to learn anything from: accepted as a figure,
	// because a small request's size hardly matters either way, but it must
	// not move the ratio.
	if _, ok := report(120, 40); !ok {
		t.Error("a small request's usage was rejected")
	}
	if got := a.charsPerTokenNow(); got < 8.9 || got > 9.1 {
		t.Errorf("ratio = %.2f after a tiny sample, want it unchanged at 9", got)
	}
}

// Compaction is the one thing that legitimately reduces the count, and when it
// does the count must reflect the conversation that now exists rather than the
// one that was replaced.
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
	// It must match the request that will actually be sent now, not a
	// leftover figure describing one that no longer exists.
	if want := a.estimate(a.fixedChars() + charsOf(a.request().msgs)); after.Projected != want {
		t.Errorf("after compaction the projection is %d, but the next request is %d tokens",
			after.Projected, want)
	}
	if after.Anchored {
		t.Error("the projection claims a reported figure describes a prefix compaction just replaced")
	}
}

// A length stop has to name whose limit it was. The old message said only that
// "the model hit its output token cap", which sent people to their server
// settings -- while the cap was ai-code's own, applied to every request. Now
// that ai-code always sends a computed cap when it knows the window, there are
// three distinct culprits and each has a different remedy.
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

// A response that stopped well short of the cap ai-code sent was not stopped by
// that cap, and saying otherwise sends the user to edit a setting that had
// nothing to do with it.
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

// A request that got smaller has to report as smaller.
//
// The count is ground truth plus an estimate of whatever was appended since,
// and the estimate used to be clamped at zero. That is fine while a
// conversation only grows, and wrong the moment one shrinks: compaction
// replaces a long transcript with a short checkpoint, the difference goes
// negative, the clamp throws it away, and the figure stays pinned at the
// high-water mark. The user sees a compaction that freed nothing -- or, once
// the checkpoint adds its own characters back on the positive side, one that
// somehow made the session bigger.
func TestContextUsedFallsWhenTheRequestShrinks(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.opts.ReserveTokens = DefaultReserveTokens
	a.opts.KeepRecentTokens = DefaultKeepRecentTokens
	a.charsPerToken = 4

	// A session well past its window: one repeated bulky tool result, which is
	// the shape that gets a transcript there in a single turn.
	bulk := strings.Repeat("match at offset 0x0000 in section .text\n", 400)
	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: "search the binary"})
	for i := 0; i < 140; i++ {
		a.messages = append(a.messages,
			provider.Message{Role: provider.RoleAssistant,
				ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "c", Content: bulk})
	}

	// Ground truth from the last response, recorded where it belongs: on the
	// assistant message the request produced. The figure and the messages it
	// describes cannot drift apart, because they are the same object.
	last := &a.messages[len(a.messages)-2]
	last.PromptTokens = a.estimate(a.contextChars())
	last.Completion = 400
	before := a.ContextState().Projected

	// A compaction lands. The transcript is untouched; the assembled request
	// collapses to the checkpoint plus the tail.
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
	// Ground truth plus a signed estimate should land near the real size. A
	// wide tolerance: the estimate divides by a calibrated ratio, so it is
	// meant to be close, not exact.
	if after > actual*3/2 {
		t.Errorf("Used = %d, but the request is really about %d tokens", after, actual)
	}
}

// A message the checkpoint has already summarised must not come back.
//
// When a session ends on a tool result so large that splitPoint summarises
// the whole transcript, summarisedThrough reaches the end of the message list
// and firstKept's floor swallows its own loop. The fallback for "the window
// is too small for even the last message" then fired, walked back onto that
// last turn, and put the oversized result the checkpoint had just replaced
// into every request for the rest of the session -- where no further
// compaction could reach it, because summarising it is what put the floor
// there.
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
	// A person who pastes something enormous meant to; cutting it up silently
	// is worse than the context it costs.
	if len(msgs[1].Content) != len(huge) {
		t.Error("a user message was clamped")
	}
}
