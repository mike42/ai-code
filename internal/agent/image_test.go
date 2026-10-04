package agent

import (
	"context"
	"encoding/json"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

type pictureTool struct{}

func (pictureTool) Name() string            { return "draw" }
func (pictureTool) Description() string     { return "draws" }
func (pictureTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (pictureTool) ReadOnly() bool          { return false }
func (pictureTool) Run(context.Context, *tool.State, json.RawMessage) tool.Result {
	return tool.Result{Content: "drew it", Images: []tool.Image{{MIME: "image/png", Data: []byte("png"), Width: 280, Height: 280}}}
}

func TestPicturesFromAToolReachTheModel(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{calls: []provider.ToolCall{{ID: "c1", Name: "draw", Args: `{}`}}},
		{text: "It is a fox."},
	}}
	a := newAgent(t, client, &collectSink{}, pictureTool{})
	if err := a.Run(context.Background(), "draw a fox and say what it is"); err != nil {
		t.Fatal(err)
	}
	var sent *provider.Message
	for _, m := range client.requests()[1] {
		if m.ToolCallID == "c1" {
			sent = &m
		}
	}
	if sent == nil || len(sent.Images) != 1 || string(sent.Images[0].Data) != "png" {
		t.Fatalf("the tool's picture was not in the next request: %+v", sent)
	}
	if got, without := messageChars(*sent), messageChars(provider.Message{Role: sent.Role, Name: sent.Name,
		ToolCallID: sent.ToolCallID, Content: sent.Content}); got-without != 100*4 {
		t.Errorf("a 280x280 picture added %d characters to the estimate, want 100 tokens' worth", got-without)
	}
}

func TestClearingOldOutputDropsPictures(t *testing.T) {
	client := &scriptedClient{}
	a := bigSession(t, client, 30)
	a.messages[2].Images = []provider.Image{{MIME: "image/png", Data: []byte("png"), Width: 1024, Height: 1024}}
	if a.messages[2].Role != provider.RoleTool {
		t.Fatalf("message 2 is a %s message", a.messages[2].Role)
	}
	a.ClearOldOutput()
	if !isCleared(a.messages[2]) || a.messages[2].Images != nil {
		t.Errorf("an old result was cleared but kept its picture: %+v", a.messages[2].Images)
	}
}
