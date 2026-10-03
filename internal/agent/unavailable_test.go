package agent

import (
	"context"
	"errors"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

type unreachable struct{ tool.Executor }

func (u unreachable) Execute(context.Context, tool.Request) (tool.Result, error) {
	return tool.Result{}, &tool.Unavailable{Err: errors.New("no route to the remote machine")}
}

func TestARunStopsWhenTheToolsHaveNowhereToRun(t *testing.T) {
	inner, _ := mixedExecutor(t)
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "c1", Name: "read", Args: `{"path":"x"}`}}},
		{calls: []provider.ToolCall{{ID: "c2", Name: "read", Args: `{"path":"y"}`}}},
		{text: "gave up", stop: provider.StopEnd},
	}}
	a := New(client, "m", unreachable{inner}, &collectSink{}, Options{ContextLimit: 65536})

	err := a.Run(context.Background(), "go")
	var u *tool.Unavailable
	if !errors.As(err, &u) {
		t.Fatalf("Run returned %v, want the unavailable error", err)
	}
	client.mu.Lock()
	n := client.n
	client.mu.Unlock()
	if n != 1 {
		t.Errorf("%d requests were made, want 1: the model was asked again after the tools had nowhere to run", n)
	}
	msgs := a.Messages()
	if last := msgs[len(msgs)-1]; last.Role != provider.RoleTool {
		t.Errorf("the conversation ends with a %s message, want the tool result that closes the call", last.Role)
	}
}
