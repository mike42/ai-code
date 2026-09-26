package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

// denseTokenizer counts a serialised request the way a byte-pair tokenizer
// does on code and paths.
func denseTokenizer(charsPerToken float64) func(provider.Request) int {
	return func(req provider.Request) int {
		b, _ := json.Marshal(struct {
			M []provider.Message `json:"messages"`
			T []provider.ToolDef `json:"tools"`
		}{req.Messages, req.Tools})
		return int(float64(len(b)) / charsPerToken)
	}
}

// toolChatter builds a session of the shape that fills a window: a user
// message, then rounds of bulky tool results.
func toolChatter(a *Agent, rounds int, out string) {
	a.messages = append(a.messages, provider.Message{
		Role: provider.RoleUser, Content: "find every caller of commitLocked"})
	for i := 0; i < rounds; i++ {
		if i > 0 && i%8 == 0 {
			a.messages = append(a.messages, provider.Message{
				Role: provider.RoleUser, Content: "keep going"})
		}
		a.messages = append(a.messages,
			provider.Message{Role: provider.RoleAssistant,
				ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{"q":"commitLocked"}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "c", Name: "grep", Content: out})
	}
	a.messages = append(a.messages, provider.Message{Role: provider.RoleAssistant, Content: "here they are"})
}

func bigSession(t *testing.T, client *scriptedClient, rounds int) *Agent {
	t.Helper()
	a := newAgent(t, client, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.opts.AutoCompact = true
	toolChatter(a, rounds, strings.Repeat(
		"internal/render/screen.go:142: func (s *Screen) commitLocked(b []byte) error\n", 90))
	return a
}

func TestKeptTailIsWithinTheBudgetItWasCutAgainst(t *testing.T) {
	client := &scriptedClient{
		tokenizer: denseTokenizer(2.9),
		turns:     []scriptedTurn{{text: "## Goal\nFind the callers.\n\n## Next Steps\n1. Carry on."}},
	}
	a := bigSession(t, client, 60)
	// Calibrate the way a live session does, from a real exchange.
	a.charsPerToken = float64(a.contextChars()) / float64(client.requestTokens(provider.Request{
		Messages: a.withSystem(a.messages), Tools: toolDefs(a.exec)}))

	budget := a.keepRecentTokens()
	cut := a.startPoint(budget)
	kept := a.estimateMessages(a.messages[cut:])
	t.Logf("ratio=%.2f budget=%d kept=%d (%d of %d messages)",
		a.charsPerToken, budget, kept, len(a.messages)-cut, len(a.messages))

	if cut == 0 {
		t.Fatal("the cut kept everything, so this proves nothing")
	}
	// One message of slack: the walk stops at a turn boundary.
	if kept > budget {
		t.Errorf("the cut kept %d tokens against a %d-token budget (%.0f%% over)",
			kept, budget, float64(kept-budget)/float64(budget)*100)
	}
}

func TestProjectionIsMonotoneUnderAppendAndFallsOnReclaim(t *testing.T) {
	client := &scriptedClient{
		tokenizer: denseTokenizer(3.1),
		turns:     []scriptedTurn{{text: "## Goal\nCarry on.\n\n## Next Steps\n1. More."}},
	}
	a := bigSession(t, client, 40)

	var series []int
	series = append(series, a.RequestTokens())
	for i := 0; i < 5; i++ {
		a.AppendUser(strings.Repeat("another thought. ", 200))
		series = append(series, a.RequestTokens())
	}
	for i := 1; i < len(series); i++ {
		if series[i] < series[i-1] {
			t.Errorf("the projection fell from %d to %d with only an append; series=%v",
				series[i-1], series[i], series)
		}
	}

	before := a.RequestTokens()
	if _, err := a.Compact(context.Background(), 0); err != nil {
		t.Fatal(err)
	}
	if after := a.RequestTokens(); after >= before {
		t.Errorf("compaction left the projection at %d, up from %d", after, before)
	}
}

func TestTheFigureDoesNotJumpBackAfterCompaction(t *testing.T) {
	client := &scriptedClient{
		tokenizer: denseTokenizer(3.0),
		turns: []scriptedTurn{
			{text: "understood"},
			{text: "## Goal\nFinish the search.\n\n## Next Steps\n1. Report."},
			{text: "here is where things stand"},
		},
	}
	sink := &collectSink{}
	a := bigSession(t, client, 45)
	a.sink = sink

	// A real turn first, so the session is anchored to a reported figure.
	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatal(err)
	}
	anchored := a.ContextState()
	if !anchored.Anchored {
		t.Fatal("no ground truth after a completed turn; the rest of this test is vacuous")
	}

	res, err := a.Compact(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	afterCompact := lastContext(t, sink)
	since := len(sink.contexts())
	t.Logf("before=%d reported after=%d published after=%d",
		anchored.Projected, res.TokensAfter, afterCompact.Projected)

	if afterCompact.Projected >= anchored.Projected {
		t.Fatalf("the published figure did not fall: %d then %d",
			anchored.Projected, afterCompact.Projected)
	}
	if res.TokensAfter != afterCompact.Projected {
		t.Errorf("the compaction reported %d but published %d: two figures for one request",
			res.TokensAfter, afterCompact.Projected)
	}

	// Every figure published after the next prompt must stay near the
	// post-compaction one.
	if err := a.Run(context.Background(), "where are we"); err != nil {
		t.Fatal(err)
	}
	for _, e := range sink.contexts()[since:] {
		if e.Projected > afterCompact.Projected*3/2 {
			t.Errorf("a figure of %d was published after a compaction that landed at %d",
				e.Projected, afterCompact.Projected)
		}
	}
}

// Every change to the next request publishes a figure, so no consumer holds a
// stale one.
func TestEveryChangePublishesAFigure(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(3.0), turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "c1", Name: "probe", Args: `{}`}}},
		{text: "done"},
	}}
	sink := &collectSink{}
	ft := &fakeTool{name: "probe", readOnly: true}
	a := newAgent(t, client, sink, ft)

	n := len(sink.contexts())
	a.AppendUser("hello")
	if len(sink.contexts()) == n {
		t.Error("appending a user message published no figure")
	}

	n = len(sink.contexts())
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if len(sink.contexts()) <= n+2 {
		t.Errorf("a turn with a tool round published %d figures; "+
			"the prompt, the response and the results are three changes",
			len(sink.contexts())-n)
	}

	n = len(sink.contexts())
	a.SetModel("other", 32768, 4096)
	if len(sink.contexts()) == n {
		t.Error("a model swap published no figure, so the window on screen is the old one")
	}
}

func TestReclaimingRemovesNoMessages(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(3.0),
		turns: []scriptedTurn{{text: "## Goal\nx\n\n## Next Steps\n1. y"}}}
	a := bigSession(t, client, 30)
	a.opts.ContextLimit = 65536 // small enough that compacting this session pays
	n := len(a.messages)

	if p := a.ClearOldOutput(); p.Results == 0 {
		t.Fatal("pruning found nothing to empty in a session of bulky tool results")
	}
	if got := len(a.messages); got != n {
		t.Errorf("pruning changed the transcript from %d messages to %d", n, got)
	}
	if _, err := a.Compact(context.Background(), 0); err != nil {
		t.Fatal(err)
	}
	if got := len(a.messages); got != n {
		t.Errorf("compaction changed the transcript from %d messages to %d", n, got)
	}
	if a.startAt <= 0 || a.startAt >= len(a.messages) {
		t.Errorf("cut = %d, which sends everything or nothing of a %d-message transcript",
			a.startAt, len(a.messages))
	}
}

func TestPruningIsIdempotentAndSparesTheRecentTurns(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(3.0)}
	a := bigSession(t, client, 30)

	first := a.ClearOldOutput()
	if first.Results == 0 {
		t.Fatal("nothing was pruned")
	}
	if second := a.ClearOldOutput(); second.Results != 0 {
		t.Errorf("a second pass pruned %d more results; the first was not complete",
			second.Results)
	}

	turns := 0
	for i := len(a.messages) - 1; i >= 0 && turns < recentTurnsKept; i-- {
		if a.messages[i].Role == provider.RoleUser {
			turns++
			continue
		}
		if a.messages[i].Role == provider.RoleTool && isCleared(a.messages[i]) {
			t.Fatalf("message %d was pruned, inside the last %d turns", i, recentTurnsKept)
		}
	}
}

// Pruning runs before summarising, so a tool-heavy session makes room without
// a model call.
func TestToolHeavySessionMakesRoomWithoutAModelCall(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(3.0),
		turns: []scriptedTurn{{text: "carrying on"}}}
	sink := &collectSink{}
	a := bigSession(t, client, 150)
	a.sink = sink
	a.opts.ContextLimit = 65536

	if a.fits() {
		t.Fatal("the session already fits, so the guard will not run")
	}
	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := len(client.requests()); got != 1 {
		t.Errorf("%d requests were sent; pruning should have made room without a "+
			"summarisation call, leaving just the turn", got)
	}
	for _, e := range sink.events {
		if e.Kind == EvCompacted {
			t.Error("the session summarised when emptying stale tool output would have done")
		}
	}
}

// makeRoom must never report success having freed nothing.
func TestASingleOversizedMessageFailsByName(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(3.0),
		turns: []scriptedTurn{{text: "## Goal\nx\n\n## Next Steps\n1. y"}, {text: "hello"}}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 32768
	a.opts.AutoCompact = true

	// A pasted file larger than the whole budget, as the newest turn; no
	// reclaim step can shed it, so the error must name it.
	a.messages = []provider.Message{
		{Role: provider.RoleUser, Content: "have a look at this"},
		{Role: provider.RoleAssistant, Content: "will do"},
		{Role: provider.RoleUser, Content: strings.Repeat(
			"func (s *Screen) commitLocked(b []byte) error { return s.w.Write(b) }\n", 8000)},
	}

	err := a.Run(context.Background(), "")
	if err == nil {
		t.Fatal("the run succeeded with a request that cannot fit; this is the wedge")
	}
	if !strings.Contains(err.Error(), "user message") {
		t.Errorf("the error does not name the message at fault: %v", err)
	}
	t.Logf("%v", err)
}

func TestASuccessfulReclaimAlwaysLeavesARequestThatFits(t *testing.T) {
	for _, rounds := range []int{20, 60, 150, 300} {
		client := &scriptedClient{tokenizer: denseTokenizer(3.0), turns: []scriptedTurn{
			{text: "## Goal\nx\n\n## Next Steps\n1. y"},
			{text: "## Goal\nx2\n\n## Next Steps\n1. y2"},
		}}
		a := bigSession(t, client, rounds)
		a.opts.ContextLimit = 65536
		a.AppendUser("carry on")

		err := a.makeRoom(context.Background(), a.plannedTokens())
		if err != nil {
			t.Logf("rounds=%d: makeRoom declined, which is allowed: %v", rounds, err)
			continue
		}
		if !a.fits() {
			t.Errorf("rounds=%d: makeRoom reported success but the next request is %d "+
				"of a %d-token budget", rounds, a.plannedTokens(), a.Usable())
		}
	}
}

func TestRepeatedCompactionDoesNotGrowTheSummarisationRequest(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(3.0), turns: []scriptedTurn{
		{text: "## Goal\nA\n\n## Next Steps\n1. a"},
		{text: "## Goal\nB\n\n## Next Steps\n1. b"},
		{text: "## Goal\nC\n\n## Next Steps\n1. c"},
	}}
	a := bigSession(t, client, 40)

	var sizes []int
	for i := 0; i < 3; i++ {
		before := len(client.requests())
		if _, err := a.Compact(context.Background(), 0); err != nil {
			t.Fatalf("compaction %d: %v", i+1, err)
		}
		sent := client.requests()[before:]
		if len(sent) != 1 {
			t.Fatalf("compaction %d made %d model calls, want 1", i+1, len(sent))
		}
		sizes = append(sizes, charsOf(sent[0]))
		// More work happens, and the session grows again.
		toolChatter(a, 20, strings.Repeat("another line of output\n", 90))
	}

	t.Logf("summarisation request sizes: %v", sizes)
	for i := 1; i < len(sizes); i++ {
		if sizes[i] > sizes[0]*2 {
			t.Errorf("compaction %d serialised %d characters against the first's %d; "+
				"the cost is growing with the session", i+1, sizes[i], sizes[0])
		}
	}
}

// A backend may refuse for length after the guard passes; recover once, not a
// dead end.
func TestABackendRefusalIsRecoveredFromOnce(t *testing.T) {
	client := &scriptedClient{
		tokenizer: denseTokenizer(3.0),
		turns: []scriptedTurn{
			{text: "recovered"},
			{text: "recovered"},
		},
	}
	sink := &collectSink{}
	a := bigSession(t, client, 40)
	a.sink = sink
	// The advertised window is a third larger than what the backend accepts;
	// the guard cannot see that.
	client.refuseOver = a.RequestTokens() * 2 / 3

	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := sink.text(); !strings.Contains(got, "recovered") {
		t.Errorf("the run did not recover: text = %q", got)
	}
	var warned bool
	for _, n := range sink.notices() {
		if strings.Contains(n, "rejected the request") {
			warned = true
		}
	}
	if !warned {
		t.Error("the refusal was recovered from silently; the user is owed the reason")
	}
}

// Otherwise the session is back at the window's edge within a turn or two.
func TestCompactionLandsWellClearOfTheWindow(t *testing.T) {
	client := &scriptedClient{tokenizer: denseTokenizer(2.9),
		turns: []scriptedTurn{{text: "## Goal\nx\n\n## Next Steps\n1. y"}}}
	a := bigSession(t, client, 200)

	if _, err := a.Compact(context.Background(), 0); err != nil {
		t.Fatal(err)
	}
	after, usable := a.RequestTokens(), a.Usable()
	t.Logf("landed at %d of a %d-token budget (%.0f%%)",
		after, usable, float64(after)/float64(usable)*100)
	if after > usable/2 {
		t.Errorf("compaction landed at %d, over half the %d-token budget: "+
			"the session will be back at the line almost immediately", after, usable)
	}
}

func lastContext(t *testing.T, s *collectSink) ContextState {
	t.Helper()
	all := s.contexts()
	if len(all) == 0 {
		t.Fatal("no context figure was ever published")
	}
	return all[len(all)-1]
}

func TestTheReportedSessionEndToEnd(t *testing.T) {
	const ratio = 2.9 // dense, the way a coding transcript really tokenises
	client := &scriptedClient{
		tokenizer: denseTokenizer(ratio),
		turns: []scriptedTurn{
			{text: "here is what I found"},
			{text: "## Goal\nFinish the search.\n\n## Progress\n### Done\n- [x] found them\n\n## Next Steps\n1. Report."},
			{text: "we are most of the way through the search"},
			{text: "and here is the rest"},
		},
	}
	sink := &collectSink{}
	a := newAgent(t, client, sink)
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.opts.AutoCompact = true
	toolChatter(a, 88, strings.Repeat(
		"internal/render/screen.go:142: func (s *Screen) commitLocked(b []byte) error\n", 90))

	// One real turn, so the session is anchored to a reported figure.
	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatal(err)
	}
	filled := a.ContextState()
	t.Logf("filled:     %6d of %d (%d%%), anchored=%v",
		filled.Projected, filled.Window, filled.Percent(), filled.Anchored)

	res, err := a.Compact(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	landed := a.ContextState()
	t.Logf("compacted:  %6d -> %6d (%d%% of window), %d messages -> %d",
		res.TokensBefore, res.TokensAfter, landed.Percent(),
		res.MessagesBefore, res.MessagesAfter)

	since := len(sink.contexts())
	if err := a.Run(context.Background(), "please explain where you are up to"); err != nil {
		t.Fatal(err)
	}
	settled := a.ContextState()
	t.Logf("settled:    %6d of %d (%d%%), anchored=%v",
		settled.Projected, settled.Window, settled.Percent(), settled.Anchored)

	peak := 0
	for _, c := range sink.contexts()[since:] {
		peak = max(peak, c.Projected)
	}
	t.Logf("peak published after the compaction: %d", peak)

	if landed.Projected > a.Usable()/2 {
		t.Errorf("landed at %d, over half the %d-token budget", landed.Projected, a.Usable())
	}
	tail := a.estimateMessages(a.messages[a.startAt:])
	if tail > a.keepRecentTokens() {
		t.Errorf("the kept tail is %d tokens against a %d-token budget",
			tail, a.keepRecentTokens())
	}

	if peak >= filled.Projected {
		t.Errorf("a figure of %d was published after compaction landed at %d, "+
			"back at the pre-compaction %d", peak, landed.Projected, filled.Projected)
	}

	reqs := client.reqs
	lastReq := reqs[len(reqs)-1]
	actual := client.requestTokens(lastReq)
	t.Logf("last request: backend counted %d, displayed %d", actual, settled.Projected)
	if diff := float64(settled.Projected-actual) / float64(actual); diff > 0.1 || diff < -0.1 {
		t.Errorf("displayed %d for a request the backend counted at %d (%.0f%% out)",
			settled.Projected, actual, diff*100)
	}
}
