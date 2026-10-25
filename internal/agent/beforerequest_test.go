package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"ai-code/internal/tool"
)

// growingExec gains a tool when grow is called, as an executor does when an
// MCP server finishes starting.
type growingExec struct {
	tool.Executor
	extra []tool.Definition
}

func (g *growingExec) Definitions() []tool.Definition {
	return append(g.Executor.Definitions(), g.extra...)
}

func TestBeforeRequestRunsBeforeTheToolListIsRead(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "done"}}}
	exec := &growingExec{Executor: tool.NewLocalExecutor(tool.NewState(t.TempDir()))}
	a := New(client, "test-model", exec, &collectSink{}, Options{ContextLimit: 65536, LoopGuard: 3})
	calls := 0
	a.SetBeforeRequest(func(context.Context) error {
		calls++
		exec.extra = []tool.Definition{{Name: "hello_hello", Schema: json.RawMessage(`{}`)}}
		return nil
	})
	if err := a.Run(context.Background(), "hi"); err != nil {
		t.Fatal(err)
	}
	if calls == 0 {
		t.Fatal("hook never ran")
	}
	found := false
	for _, td := range client.reqs[0].Tools {
		found = found || td.Name == "hello_hello"
	}
	if !found {
		t.Fatalf("the tool added while waiting is missing from the request: %+v", client.reqs[0].Tools)
	}
}

func TestBeforeRequestCancelledSendsNothing(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "done"}}}
	a := newAgent(t, client, &collectSink{})
	ctx, cancel := context.WithCancel(context.Background())
	a.SetBeforeRequest(func(ctx context.Context) error {
		cancel()
		<-ctx.Done()
		return ctx.Err()
	})
	err := a.Run(ctx, "hi")
	if err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if len(client.reqs) != 0 {
		t.Fatalf("%d requests sent after cancelling the wait", len(client.reqs))
	}
}
