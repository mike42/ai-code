package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// fakeTool records its calls and can block, so dispatch ordering and
// cancellation can be observed.
type fakeTool struct {
	name     string
	readOnly bool
	block    chan struct{}

	mu    sync.Mutex
	calls []string
	// running counts concurrent invocations, to prove parallelism.
	running, maxRunning int
}

func (f *fakeTool) Name() string            { return f.name }
func (f *fakeTool) Description() string     { return "test tool " + f.name }
func (f *fakeTool) ReadOnly() bool          { return f.readOnly }
func (f *fakeTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }

func (f *fakeTool) Run(ctx context.Context, st *tool.State, args json.RawMessage) tool.Result {
	f.mu.Lock()
	f.calls = append(f.calls, string(args))
	f.running++
	if f.running > f.maxRunning {
		f.maxRunning = f.running
	}
	f.mu.Unlock()

	defer func() {
		f.mu.Lock()
		f.running--
		f.mu.Unlock()
	}()

	if f.block != nil {
		select {
		case <-f.block:
		case <-ctx.Done():
			return tool.Result{Content: "cancelled", IsError: true}
		}
	}
	return tool.Result{Content: "ok from " + f.name}
}

func newAgent(t *testing.T, client provider.Client, sink Sink, tools ...tool.Tool) *Agent {
	t.Helper()
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()), tools...)
	return New(client, "test-model", exec, sink, Options{
		MaxIterations: 10, ParallelReads: true, LoopGuard: 3, ContextLimit: 65536,
	})
}

func TestLoopRunsToolsThenStops(t *testing.T) {
	ft := &fakeTool{name: "probe", readOnly: true}
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "c1", Name: "probe", Args: `{"x":1}`}}},
		{text: "All done."},
	}}
	sink := &collectSink{}
	a := newAgent(t, client, sink, ft)

	if err := a.Run(context.Background(), "do the thing"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(ft.calls) != 1 {
		t.Errorf("tool ran %d times, want 1", len(ft.calls))
	}
	if got := sink.text(); got != "All done." {
		t.Errorf("text = %q, want %q", got, "All done.")
	}

	if err := Validate(a.Messages()); err != nil {
		t.Errorf("conversation is invalid after a normal run: %v", err)
	}
}

func TestToolResultsFollowTheirCall(t *testing.T) {
	ft := &fakeTool{name: "probe", readOnly: true}
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{
			{ID: "a", Name: "probe", Args: `{}`},
			{ID: "b", Name: "probe", Args: `{}`},
		}},
		{text: "done"},
	}}
	a := newAgent(t, client, &collectSink{}, ft)
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}

	// Both tool results must follow their assistant message in call order.
	req := client.requests()[1]
	var idx int
	for i, m := range req {
		if m.Role == provider.RoleAssistant && len(m.ToolCalls) == 2 {
			idx = i
		}
	}
	if idx == 0 {
		t.Fatal("assistant message with tool calls not found in the second request")
	}
	if req[idx+1].ToolCallID != "a" || req[idx+2].ToolCallID != "b" {
		t.Errorf("tool results out of order: got %q then %q",
			req[idx+1].ToolCallID, req[idx+2].ToolCallID)
	}
}

func TestReadOnlyToolsRunConcurrently(t *testing.T) {
	release := make(chan struct{})
	ft := &fakeTool{name: "probe", readOnly: true, block: release}

	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{
			{ID: "a", Name: "probe", Args: `{}`},
			{ID: "b", Name: "probe", Args: `{}`},
			{ID: "c", Name: "probe", Args: `{}`},
		}},
		{text: "done"},
	}}
	a := newAgent(t, client, &collectSink{}, ft)

	done := make(chan error, 1)
	go func() { done <- a.Run(context.Background(), "go") }()

	deadline := time.After(2 * time.Second)
	for {
		ft.mu.Lock()
		n := ft.running
		ft.mu.Unlock()
		if n == 3 {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("only %d read-only tools ran concurrently, want 3", n)
		case <-time.After(time.Millisecond):
		}
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestWritingToolsAreSerialised(t *testing.T) {
	ft := &fakeTool{name: "mutate", readOnly: false}
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{
			{ID: "a", Name: "mutate", Args: `{}`},
			{ID: "b", Name: "mutate", Args: `{}`},
		}},
		{text: "done"},
	}}
	a := newAgent(t, client, &collectSink{}, ft)
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if ft.maxRunning > 1 {
		t.Errorf("%d writing tools ran at once; writes must be serialised", ft.maxRunning)
	}
}

func TestCancellationLeavesTheConversationSendable(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	ft := &fakeTool{name: "slow", readOnly: false, block: release}

	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{
			{ID: "a", Name: "slow", Args: `{}`},
			{ID: "b", Name: "slow", Args: `{}`},
		}},
	}}
	sink := &collectSink{}
	a := newAgent(t, client, sink, ft)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- a.Run(ctx, "go") }()

	deadline := time.After(2 * time.Second)
	for {
		ft.mu.Lock()
		n := len(ft.calls)
		ft.mu.Unlock()
		if n >= 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("tool never started")
		case <-time.After(time.Millisecond):
		}
	}
	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run returned an error on cancellation: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after cancellation")
	}

	if err := Validate(a.Messages()); err != nil {
		t.Fatalf("conversation is unsendable after cancellation: %v", err)
	}

	// Both calls must be answered, and the answers must say the work was
	// interrupted.
	answered := map[string]string{}
	for _, m := range a.Messages() {
		if m.Role == provider.RoleTool {
			answered[m.ToolCallID] = m.Content
		}
	}
	for _, id := range []string{"a", "b"} {
		body, ok := answered[id]
		if !ok {
			t.Errorf("tool call %q was never answered", id)
			continue
		}
		if !strings.Contains(strings.ToLower(body), "interrupt") {
			t.Errorf("result for %q should say it was interrupted, got: %s", id, body)
		}
	}
}

func TestRepairFixesAnOrphanedToolCall(t *testing.T) {
	messages := []provider.Message{
		{Role: provider.RoleUser, Content: "go"},
		{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
			{ID: "x", Name: "bash", Args: `{}`},
		}},
	}
	if err := Validate(messages); err == nil {
		t.Fatal("an orphaned tool call should fail validation")
	}

	repaired, n := Repair(messages)
	if n != 1 {
		t.Errorf("repaired %d calls, want 1", n)
	}
	if err := Validate(repaired); err != nil {
		t.Errorf("still invalid after repair: %v", err)
	}
	last := repaired[len(repaired)-1]
	if last.Role != provider.RoleTool || last.ToolCallID != "x" {
		t.Errorf("repair did not append a matching tool result: %+v", last)
	}
	if !strings.Contains(last.Content, "may have") {
		t.Errorf("the synthesised result should be honest that state may have changed, got: %s", last.Content)
	}
}

func TestValidateRejectsAToolResultWithNoCall(t *testing.T) {
	messages := []provider.Message{
		{Role: provider.RoleUser, Content: "hi"},
		{Role: provider.RoleTool, ToolCallID: "ghost", Content: "result"},
	}
	if err := Validate(messages); err == nil {
		t.Fatal("expected validation to reject an unmatched tool result")
	}
}

func TestLoopGuardStopsRepeatedIdenticalCalls(t *testing.T) {
	ft := &fakeTool{name: "spin", readOnly: true}
	same := scriptedTurn{calls: []provider.ToolCall{{ID: "z", Name: "spin", Args: `{"same":true}`}}}
	client := &scriptedClient{turns: []scriptedTurn{same, same, same, same, same, same}}
	sink := &collectSink{}
	a := newAgent(t, client, sink, ft)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, n := range sink.notices() {
		if strings.Contains(n, "repeated") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a loop-guard notice, got: %v", sink.notices())
	}
	if len(ft.calls) > 4 {
		t.Errorf("tool ran %d times; the guard should have stopped it sooner", len(ft.calls))
	}
	if err := Validate(a.Messages()); err != nil {
		t.Errorf("conversation invalid after the loop guard fired: %v", err)
	}
}

func TestContextAccountingUsesProviderUsage(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "hello", usage: provider.Usage{PromptTokens: 1000, CompletionTokens: 50}},
	}}
	a := newAgent(t, client, &collectSink{})
	if err := a.Run(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	cs := a.ContextState()
	if cs.Projected != 1050 {
		t.Errorf("used = %d, want 1050 taken from the provider's own usage report", cs.Projected)
	}
	if !cs.Anchored {
		t.Error("should not be marked estimated once real usage has been reported")
	}
	if cs.Percent() != 1 {
		t.Errorf("percent = %d, want 1", cs.Percent())
	}
}

// With auto_compact off, a full window is a hard stop, not an implicit rewrite.
func TestContextFullRefusesTheTurnAndKeepsTheSession(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "ok", usage: provider.Usage{PromptTokens: 64000, CompletionTokens: 2000}},
		{text: "second"},
	}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)

	if err := a.Run(context.Background(), "first"); err != nil {
		t.Fatal(err)
	}
	before := len(a.Messages())

	err := a.Run(context.Background(), "second request")
	if !errors.Is(err, ErrContextFull) {
		t.Fatalf("want ErrContextFull, got %v", err)
	}
	if !strings.Contains(err.Error(), "/compact") {
		t.Errorf("the error must tell the user what to do, got: %v", err)
	}
	if len(a.Messages()) <= before {
		t.Error("the user's message should still be recorded; the session stays intact")
	}
}

func TestContextWarningFiresOnceAtThreshold(t *testing.T) {
	const window = 65536
	usable := window - BudgetFor(window).Reserve
	trigger := usable * 80 / 100
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "a", usage: provider.Usage{PromptTokens: trigger + 1000, CompletionTokens: 100}},
		{text: "b", usage: provider.Usage{PromptTokens: trigger + 2000, CompletionTokens: 100}},
	}}
	sink := &collectSink{}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()))
	a := New(client, "m", exec, sink, Options{
		MaxIterations: 5, ContextLimit: window, WarnPercent: 80,
	})

	_ = a.Run(context.Background(), "one")
	_ = a.Run(context.Background(), "two")

	warnings := 0
	for _, n := range sink.notices() {
		if strings.Contains(n, "this session can use") {
			warnings++
		}
	}
	if warnings != 1 {
		t.Errorf("got %d context warnings, want exactly 1 -- repeating it every turn is noise", warnings)
	}
}

func TestSystemPromptIsFirstAndStable(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "ok"}, {text: "ok"}}}
	a := newAgent(t, client, &collectSink{})
	a.SetSystem("STABLE SYSTEM PROMPT")

	_ = a.Run(context.Background(), "one")
	_ = a.Run(context.Background(), "two")

	reqs := client.requests()
	if len(reqs) < 2 {
		t.Fatal("expected two requests")
	}
	for i, r := range reqs {
		if len(r) == 0 || r[0].Role != provider.RoleSystem {
			t.Fatalf("request %d does not begin with the system message", i)
		}
		if r[0].Content != "STABLE SYSTEM PROMPT" {
			t.Errorf("request %d system prompt changed to %q; an unstable prefix "+
				"silently destroys prompt caching", i, r[0].Content)
		}
	}
}

func TestExecutorFailureBecomesAToolResult(t *testing.T) {
	// A tool the executor does not know about must not break the loop.
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "a", Name: "nonexistent", Args: `{}`}}},
		{text: "recovered"},
	}}
	a := newAgent(t, client, &collectSink{})
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("an unknown tool should not fail the run: %v", err)
	}
	if err := Validate(a.Messages()); err != nil {
		t.Errorf("conversation invalid: %v", err)
	}

	var found bool
	for _, m := range a.Messages() {
		if m.Role == provider.RoleTool && strings.Contains(m.Content, "no tool named") {
			found = true
			if !strings.Contains(m.Content, "Available tools") {
				t.Error("the error should list what is available so the model can retry correctly")
			}
		}
	}
	if !found {
		t.Error("expected an error tool result naming the missing tool")
	}
}

// scriptLongRun builds n distinct tool-calling turns followed by a final answer.
// Distinct arguments keep the loop guard from firing.
func scriptLongRun(n int) []scriptedTurn {
	turns := make([]scriptedTurn, 0, n+1)
	for i := range n {
		turns = append(turns, scriptedTurn{calls: []provider.ToolCall{
			{ID: fmt.Sprintf("c%d", i), Name: "probe", Args: fmt.Sprintf(`{"i":%d}`, i)},
		}})
	}
	return append(turns, scriptedTurn{text: "done"})
}

func TestUnconfiguredRunIsNotCappedAtAHundredTurns(t *testing.T) {
	const turns = 250
	ft := &fakeTool{name: "probe", readOnly: true}
	client := &scriptedClient{turns: scriptLongRun(turns)}
	sink := &collectSink{}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()), ft)
	a := New(client, "m", exec, sink, Options{ContextLimit: 65536})

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(ft.calls) != turns {
		t.Fatalf("tool ran %d times, want %d -- the loop stopped short of the model finishing",
			len(ft.calls), turns)
	}
	if got := sink.text(); got != "done" {
		t.Errorf("text = %q, want %q", got, "done")
	}
	for _, n := range sink.notices() {
		if strings.Contains(n, "Stopped after") {
			t.Errorf("unconfigured run hit a turn cap: %q", n)
		}
	}
}

func TestConfiguredIterationLimitIsHonoured(t *testing.T) {
	ft := &fakeTool{name: "probe", readOnly: true}
	client := &scriptedClient{turns: scriptLongRun(20)}
	sink := &collectSink{}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()), ft)
	a := New(client, "m", exec, sink, Options{MaxIterations: 3, ContextLimit: 65536})

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(ft.calls) != 3 {
		t.Errorf("tool ran %d times, want 3", len(ft.calls))
	}

	var stopped bool
	for _, n := range sink.notices() {
		if strings.Contains(n, "configured limit of 3 turns") {
			stopped = true
		}
	}
	if !stopped {
		t.Errorf("no notice explaining the stop; got %q", sink.notices())
	}
}

func TestATurnThatSaysNothingSaysSo(t *testing.T) {
	for _, tc := range []struct {
		name string
		turn scriptedTurn
		want string
	}{
		{"nothing at all", scriptedTurn{}, "without producing any output"},
		{"thought but did not answer", scriptedTurn{reasoning: "hmm, let me consider"},
			"spent the whole turn thinking"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sink := &collectSink{}
			client := &scriptedClient{turns: []scriptedTurn{tc.turn}}
			a := New(client, "m", toolExecutor(t), sink, Options{ContextLimit: 65536})

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
				t.Errorf("notices = %q, want one containing %q", sink.notices(), tc.want)
			}
		})
	}
}

func TestAnOrdinaryTurnIsNotReportedAsEmpty(t *testing.T) {
	sink := &collectSink{}
	client := &scriptedClient{turns: []scriptedTurn{{text: "here is the answer"}}}
	a := New(client, "m", toolExecutor(t), sink, Options{ContextLimit: 65536})
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	for _, n := range sink.notices() {
		if strings.Contains(n, "without producing") || strings.Contains(n, "whole turn thinking") {
			t.Errorf("an ordinary turn was reported as empty: %q", n)
		}
	}
}
