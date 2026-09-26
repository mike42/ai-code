package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// writingTool stands in for edit or write: it reports a file change, and a
// read-only agent must never reach it.
type writingTool struct{ ran bool }

func (w *writingTool) Name() string            { return "write" }
func (w *writingTool) Description() string     { return "writes a file" }
func (w *writingTool) ReadOnly() bool          { return false }
func (w *writingTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (w *writingTool) Run(context.Context, *tool.State, json.RawMessage) tool.Result {
	w.ran = true
	return tool.Result{Content: "written", Files: []tool.FileStamp{{Path: "x.go"}}}
}

type readingTool struct{}

func (readingTool) Name() string            { return "read" }
func (readingTool) Description() string     { return "reads a file" }
func (readingTool) ReadOnly() bool          { return true }
func (readingTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (readingTool) Run(context.Context, *tool.State, json.RawMessage) tool.Result {
	return tool.Result{Content: "file contents"}
}

func mixedExecutor(t *testing.T) (*tool.LocalExecutor, *writingTool) {
	t.Helper()
	w := &writingTool{}
	return tool.NewLocalExecutor(tool.NewState(t.TempDir()), readingTool{}, w), w
}

// cloudClient is on-premises in every respect except the one that matters.
type cloudClient struct{ scriptedClient }

func (c *cloudClient) Class() provider.Class { return provider.ClassCloud }

func taskCall(prompt string) provider.ToolCall {
	return taskCallThinking(prompt, "")
}

func taskCallThinking(prompt, thinking string) provider.ToolCall {
	m := map[string]string{"description": "look", "prompt": prompt}
	if thinking != "" {
		m["thinking"] = thinking
	}
	args, _ := json.Marshal(m)
	return provider.ToolCall{ID: "c1", Name: "task", Args: string(args)}
}

// workerEfforts is the effort on every request that carried the worker prompt.
func workerEfforts(c *scriptedClient) []provider.Effort {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []provider.Effort
	for _, r := range c.reqs {
		for _, m := range r.Messages {
			if m.Role == provider.RoleSystem && strings.Contains(m.Content, "worker agent") {
				out = append(out, r.Effort)
			}
		}
	}
	return out
}

func mustSpawn(t *testing.T, a *Agent, c Child) *Agent {
	t.Helper()
	child, err := a.Spawn(c)
	if err != nil {
		t.Fatal(err)
	}
	return child
}

func attached(t *testing.T, client provider.Client, turns ...scriptedTurn) (*Agent, *TaskExecutor) {
	t.Helper()
	inner, _ := mixedExecutor(t)
	tasks := NewTaskExecutor(inner)
	a := New(client, "m", tasks, &collectSink{}, Options{ContextLimit: 65536})
	tasks.Attach(context.Background(), a)
	return a, tasks
}

// A sub-agent must not reach a cloud provider: the task tool is absent from
// the schema on cloud, not merely refused when called.
func TestTaskToolIsAbsentOnACloudProvider(t *testing.T) {
	onprem, _ := attached(t, &scriptedClient{})
	if !hasTool(onprem.exec.Definitions(), "task") {
		t.Fatal("the task tool should be offered on an on-premises provider")
	}

	cloud, _ := attached(t, &cloudClient{})
	if hasTool(cloud.exec.Definitions(), "task") {
		t.Error("the task tool is in the schema on a cloud provider")
	}
}

func TestTaskToolRefusesToRunOnCloudEvenIfNamed(t *testing.T) {
	a, tasks := attached(t, &cloudClient{})
	_ = a
	res, err := tasks.Execute(context.Background(), tool.Request{
		Name: "task", Args: json.RawMessage(`{"description":"x","prompt":"y"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.IsError {
		t.Errorf("a task call on a cloud provider produced %+v, want an error result", res)
	}
}

// A worker gets the executor from underneath, so the task tool is absent and
// workers cannot recurse.
func TestAWorkerCannotSpawnAWorker(t *testing.T) {
	a, _ := attached(t, &scriptedClient{})
	child, err := a.Spawn(Child{System: WorkerSystemPrompt})
	if err != nil {
		t.Fatal(err)
	}

	if hasTool(child.exec.Definitions(), "task") {
		t.Error("a child was handed the task tool, so workers can recurse")
	}
}

// The sandbox is the boundary, not the tool list: a worker may write.
func TestAWorkerCanWrite(t *testing.T) {
	a, _ := attached(t, &scriptedClient{})
	child, err := a.Spawn(Child{System: WorkerSystemPrompt})
	if err != nil {
		t.Fatal(err)
	}

	if !hasTool(child.exec.Definitions(), "write") {
		t.Error("a worker was denied the write tool")
	}
}

func TestConsultChildHasNoToolsAtAll(t *testing.T) {
	a, _ := attached(t, &scriptedClient{})
	child, err := a.Spawn(Child{NoTools: true})
	if err != nil {
		t.Fatal(err)
	}

	if n := len(child.exec.Definitions()); n != 0 {
		t.Errorf("a no-tools child was offered %d tools", n)
	}
}

// A child shares the parent's client and model; no field lets it choose
// otherwise.
func TestAChildRunsOnTheParentsModel(t *testing.T) {
	a, _ := attached(t, &scriptedClient{})
	a.SetModel("some-other-model", 65536, 0)
	child, err := a.Spawn(Child{})
	if err != nil {
		t.Fatal(err)
	}

	if child.Model() != "some-other-model" {
		t.Errorf("child model = %q, want the parent's", child.Model())
	}
	if child.client != a.client {
		t.Error("child got a different client from its parent")
	}
}

func TestAChildStartsWithNoConversation(t *testing.T) {
	a, _ := attached(t, &scriptedClient{})
	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: "the parent's business"},
	})
	if n := len(mustSpawn(t, a, Child{}).Messages()); n != 0 {
		t.Errorf("a child inherited %d messages", n)
	}
}

// The tool returns before the worker has done anything; the report arrives
// afterwards through the steering queue.
func TestTheWorkerReportArrivesAfterTheTurn(t *testing.T) {
	client := &scriptedClient{
		turns: []scriptedTurn{
			{calls: []provider.ToolCall{taskCall("where is the retry backoff")}},
			{text: "Started it.", stop: provider.StopEnd},
		},
		workerTurns: []scriptedTurn{
			{calls: []provider.ToolCall{{ID: "w1", Name: "read", Args: "{}"}}},
			{text: "internal/provider/openai.go:88 sets it.", stop: provider.StopEnd},
		},
	}
	a, tasks := attached(t, client)

	if err := a.Run(context.Background(), "find the backoff"); err != nil {
		t.Fatal(err)
	}

	var result string
	for _, m := range a.Messages() {
		if m.Role == provider.RoleTool && m.Name == "task" {
			result = m.Content
		}
	}
	if strings.Contains(result, "openai.go:88") {
		t.Errorf("the tool result carried the report (%q), so the turn waited for the worker", result)
	}
	if !strings.Contains(result, "started") {
		t.Errorf("the tool result is %q, want it to say the worker started", result)
	}

	tasks.Pool().Wait()
	queued := a.TakeSteering()
	if len(queued) != 1 {
		t.Fatalf("queued %d messages, want the one worker report", len(queued))
	}
	if !strings.Contains(queued[0], "openai.go:88") {
		t.Errorf("the report is %q, want the worker's finding", queued[0])
	}
	if !strings.Contains(queued[0], "<worker-report") {
		t.Errorf("the report is %q, want it tagged so the model does not read it as the user", queued[0])
	}
	for _, m := range a.Messages() {
		if m.Role == provider.RoleTool && m.Name == "read" {
			t.Error("the worker's own tool traffic leaked into the parent's conversation")
		}
	}
}

// A worker that failed still reports, because a parent told nothing waits for
// an answer that is never coming.
func TestAFailedWorkerStillReports(t *testing.T) {
	client := &scriptedClient{
		turns: []scriptedTurn{
			{calls: []provider.ToolCall{taskCall("look")}},
			{text: "Started it.", stop: provider.StopEnd},
		},
		workerTurns: []scriptedTurn{{text: "", stop: provider.StopEnd}},
	}
	a, tasks := attached(t, client)
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	tasks.Pool().Wait()

	queued := a.TakeSteering()
	if len(queued) != 1 {
		t.Fatalf("queued %d messages, want one", len(queued))
	}
	if !strings.Contains(queued[0], "no findings") {
		t.Errorf("the report is %q, want it to say the worker produced nothing", queued[0])
	}
}

// A worker outlives the turn that asked for it: the tool call's context ends
// with the turn.
func TestAWorkerOutlivesTheTurnThatStartedIt(t *testing.T) {
	release := make(chan struct{})
	client := &scriptedClient{
		turns: []scriptedTurn{
			{calls: []provider.ToolCall{taskCall("look")}},
			{text: "Started it.", stop: provider.StopEnd},
		},
		workerTurns: []scriptedTurn{{text: "found it", stop: provider.StopEnd}},
	}
	client.onWorkerStream = func() { <-release }
	a, tasks := attached(t, client)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	// The turn is over and the worker has not been allowed to answer yet.
	if n := tasks.Pool().Busy(); n != 1 {
		t.Fatalf("%d workers running after the turn ended, want 1 still going", n)
	}
	close(release)
	tasks.Pool().Wait()
	if got := a.TakeSteering(); len(got) != 1 {
		t.Fatalf("the worker delivered %d reports after the turn, want 1", len(got))
	}
}

// No hidden default: a worker inherits the session's thinking level unless the
// call says otherwise.
func TestAWorkerInheritsTheSessionThinkingLevel(t *testing.T) {
	for _, level := range []provider.Effort{
		provider.EffortUnset, provider.EffortNone, provider.EffortMedium, provider.EffortHigh,
	} {
		client := &scriptedClient{turns: []scriptedTurn{
			{calls: []provider.ToolCall{taskCall("look")}},
			{text: "found it", stop: provider.StopEnd},
			{text: "ok", stop: provider.StopEnd},
		}}
		a, tasks := attached(t, client)
		a.SetEffort(level)

		if err := a.Run(context.Background(), "go"); err != nil {
			t.Fatal(err)
		}
		tasks.Pool().Wait()
		got := workerEfforts(client)
		if len(got) == 0 {
			t.Fatalf("session at %q: no worker request was made", level)
		}
		for _, e := range got {
			if e != level {
				t.Errorf("session at %q: worker ran at %q", level, e)
			}
		}
	}
}

// The caller decides the worker's thinking level, in both directions.
func TestAWorkerHonoursAnExplicitThinkingLevel(t *testing.T) {
	cases := []struct {
		asked string
		want  provider.Effort
	}{
		{"none", provider.EffortNone},
		{"low", provider.EffortLow},
		{"high", provider.EffortHigh},
	}
	for _, c := range cases {
		client := &scriptedClient{turns: []scriptedTurn{
			{calls: []provider.ToolCall{taskCallThinking("look", c.asked)}},
			{text: "found it", stop: provider.StopEnd},
			{text: "ok", stop: provider.StopEnd},
		}}
		a, tasks := attached(t, client)
		// Set above what the call asks for, so an override is visible.
		a.SetEffort(provider.EffortHigh)

		if err := a.Run(context.Background(), "go"); err != nil {
			t.Fatal(err)
		}
		tasks.Pool().Wait()
		got := workerEfforts(client)
		if len(got) == 0 {
			t.Fatalf("thinking=%q: no worker request was made", c.asked)
		}
		for _, e := range got {
			if e != c.want {
				t.Errorf("thinking=%q: worker ran at %q, want %q", c.asked, e, c.want)
			}
		}
	}
}

func TestAnUnknownThinkingLevelIsAnErrorResult(t *testing.T) {
	a, tasks := attached(t, &scriptedClient{})
	res, err := tasks.Execute(context.Background(), tool.Request{
		Name: "task",
		Args: json.RawMessage(`{"description":"x","prompt":"y","thinking":"very hard"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.IsError || !strings.Contains(res.Content, "none, minimal, low, medium, high, xhigh, max") {
		t.Errorf("result = %+v, want an error naming the valid levels", res)
	}
	_ = a
}

func hasTool(defs []tool.Definition, name string) bool {
	for _, d := range defs {
		if d.Name == name {
			return true
		}
	}
	return false
}

// No slot count: a harness-imposed limit leaves workers sitting idle.
func TestEveryWorkerStartsAtOnce(t *testing.T) {
	const workers = 5

	arrived := make(chan struct{}, workers)
	release := make(chan struct{})

	var calls []provider.ToolCall
	for i := range workers {
		c := taskCall("look")
		c.ID = fmt.Sprintf("c%d", i)
		calls = append(calls, c)
	}
	client := &scriptedClient{
		turns: []scriptedTurn{
			{calls: calls},
			{text: "Started them.", stop: provider.StopEnd},
		},
		workerTurns: []scriptedTurn{{text: "found it", stop: provider.StopEnd}},
	}
	client.onWorkerStream = func() {
		arrived <- struct{}{}
		<-release
	}

	a, tasks := attached(t, client)
	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}

	for i := range workers {
		select {
		case <-arrived:
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d of %d workers reached the backend; the rest are being held back", i, workers)
		}
	}
	close(release)
	tasks.Pool().Wait()
}

func TestStuckSuggestsAfterTheSameErrorRepeats(t *testing.T) {
	var w stuckWatch
	for i := range stuckAfterTurns {
		w.observe([]tool.Result{{IsError: true, Content: "exit status 2: undefined: Foo"}})
		if i < stuckAfterTurns-1 {
			if s := w.suggestion(); s != "" {
				t.Fatalf("suggested after only %d turns: %q", i+1, s)
			}
		}
	}
	s := w.suggestion()
	if !strings.Contains(s, "/consult") {
		t.Errorf("suggestion = %q, want a pointer to /consult", s)
	}
	if again := w.suggestion(); again != "" {
		t.Errorf("suggested twice: %q -- a nudge every turn is a nag", again)
	}
}

// A long run of successful tool calls is investigation, not being stuck.
func TestReadingWithoutWritingIsNeverStuck(t *testing.T) {
	var w stuckWatch
	for range 200 {
		w.observe([]tool.Result{{Content: "some output"}})
	}
	if s := w.suggestion(); s != "" {
		t.Errorf("suggested %q for a long run of successful tool calls", s)
	}
}

// A failure that is fixed, or that alternates with success, is progress.
func TestAnErrorThatStopsRecurringIsNotStuck(t *testing.T) {
	var w stuckWatch
	for range stuckAfterTurns - 1 {
		w.observe([]tool.Result{{IsError: true, Content: "undefined: Foo"}})
	}
	w.observe([]tool.Result{{Content: "builds now"}})
	for range stuckAfterTurns - 1 {
		w.observe([]tool.Result{{IsError: true, Content: "undefined: Foo"}})
	}
	if s := w.suggestion(); s != "" {
		t.Errorf("suggested %q although the error cleared part-way through", s)
	}
}

// Different failures are progress; only the same wall repeating is the signal.
func TestDifferentErrorsDoNotCountAsRepeats(t *testing.T) {
	var w stuckWatch
	for i := range stuckAfterTurns + 2 {
		w.observe([]tool.Result{
			{IsError: true, Content: string(rune('a'+i)) + ": unrelated failure"},
		})
	}
	if s := w.suggestion(); s != "" {
		t.Errorf("suggested %q for a run of different errors", s)
	}
}

// The same wall with a different line number is the same wall.
func TestErrorsAreComparedIgnoringNumbers(t *testing.T) {
	var w stuckWatch
	for i := range stuckAfterTurns {
		w.observe([]tool.Result{{IsError: true,
			Content: "main.go:" + string(rune('1'+i)) + ": undefined: Foo"}})
	}
	if s := w.suggestion(); s == "" {
		t.Error("a repeating error with a moving line number was not recognised")
	}
}

// A long run must be reachable between iterations, not only when the model
// decides it is done.
func TestARunCanBeStoppedAtTheTurnBoundary(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "t1", Name: "read", Args: "{}"}}},
		{calls: []provider.ToolCall{{ID: "t2", Name: "read", Args: "{}"}}},
		{calls: []provider.ToolCall{{ID: "t3", Name: "read", Args: "{}"}}},
		{text: "done", stop: provider.StopEnd},
	}}
	a, _ := attached(t, client)

	asked := 0
	a.SetTurnBoundary(func(context.Context) bool {
		asked++
		return asked == 2
	})

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if asked != 2 {
		t.Errorf("the boundary was consulted %d times, want 2", asked)
	}
	if n := len(client.requests()); n != 2 {
		t.Errorf("the run made %d requests after being stopped at the second boundary, want 2", n)
	}
}

func TestAStoppedRunLeavesAWellFormedConversation(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "t1", Name: "read", Args: "{}"}}},
		{text: "carried on", stop: provider.StopEnd},
	}}
	a, _ := attached(t, client)
	a.SetTurnBoundary(func(context.Context) bool { return true })

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if err := Validate(a.withSystem(a.Messages())); err != nil {
		t.Fatalf("a stopped run left an unsendable conversation: %v", err)
	}

	a.SetTurnBoundary(nil)
	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatal(err)
	}
	if got := a.LastAssistantText(); got != "carried on" {
		t.Errorf("after resuming, last message = %q", got)
	}
}

func TestNoBoundaryCheckMeansNoChange(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "t1", Name: "read", Args: "{}"}}},
		{text: "done", stop: provider.StopEnd},
	}}
	a, _ := attached(t, client)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if got := a.LastAssistantText(); got != "done" {
		t.Errorf("the run did not finish normally: %q", got)
	}
}

// closableExecutor stands in for an executor whose fork is a live process, not
// a struct.
type closableExecutor struct {
	tool.Executor
	forks  []*closableExecutor
	mu     sync.Mutex
	closed bool
}

func (c *closableExecutor) Fork() (tool.Executor, error) {
	inner, err := c.Executor.Fork()
	if err != nil {
		return nil, err
	}
	f := &closableExecutor{Executor: inner}
	c.mu.Lock()
	c.forks = append(c.forks, f)
	c.mu.Unlock()
	return f, nil
}

func (c *closableExecutor) Close() error {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	return nil
}

func (c *closableExecutor) isClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// A worker closes the shell forked for it; left open, each finished worker
// keeps a process alive.
func TestAFinishedWorkerClosesItsShell(t *testing.T) {
	inner, _ := mixedExecutor(t)
	host := &closableExecutor{Executor: inner}
	tasks := NewTaskExecutor(host)
	client := &scriptedClient{
		turns: []scriptedTurn{
			{calls: []provider.ToolCall{taskCall("look")}},
			{text: "Started it.", stop: provider.StopEnd},
		},
		workerTurns: []scriptedTurn{{text: "found it", stop: provider.StopEnd}},
	}
	a := New(client, "m", tasks, &collectSink{}, Options{ContextLimit: 65536})
	tasks.Attach(context.Background(), a)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	tasks.Pool().Wait()

	host.mu.Lock()
	forks := append([]*closableExecutor(nil), host.forks...)
	host.mu.Unlock()
	if len(forks) != 1 {
		t.Fatalf("the worker took %d shells, want 1", len(forks))
	}
	if !forks[0].isClosed() {
		t.Error("the worker's shell is still open after it reported back")
	}
	if host.isClosed() {
		t.Error("the worker closed the session's own shell, not the one forked for it")
	}
}
