package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

func withImage() []provider.Message {
	return []provider.Message{
		{Role: provider.RoleUser, Content: "look at capture.png"},
		{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{{ID: "c1", Name: "read", Args: `{}`}}},
		{Role: provider.RoleTool, ToolCallID: "c1", Name: "read", Content: "capture.png is an image",
			Images: []provider.Image{{MIME: "image/png", Data: []byte("png")}}},
	}
}

// A server that cannot take the image must not end the run: the model is told why.
func TestARefusedImageIsReplacedByTheServersAnswer(t *testing.T) {
	refusal := &provider.APIError{Status: 500, Message: "image input is not supported"}
	client := &scriptedClient{turns: []scriptedTurn{{err: refusal}, {text: "I cannot see it"}}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)
	a.SetMessages(withImage())

	if err := a.Run(context.Background(), ""); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(sink.text(), "cannot see it") {
		t.Fatalf("text = %q; the run stopped at the refusal", sink.text())
	}
	retry := client.seen[len(client.seen)-1]
	for _, m := range retry {
		if len(m.Images) > 0 {
			t.Fatal("the retry still carried the image")
		}
	}
	if tr := retry[len(retry)-1]; !strings.Contains(tr.Content, "image input is not supported") {
		t.Errorf("tool result = %q, want the server's answer in it", tr.Content)
	}
}

func TestOverloadIsNotMistakenForARefusedImage(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{err: &provider.APIError{Status: 503, Message: "loading"}}}}
	a := newAgent(t, client, &collectSink{})
	a.SetMessages(withImage())

	if err := a.Run(context.Background(), ""); err == nil {
		t.Fatal("a 503 should still be reported, not retried without the image")
	}
	if len(a.Messages()[2].Images) != 1 {
		t.Error("the image was dropped for an error unrelated to it")
	}
}

// Lemonade answers HTTP 200 and sends the refusal as the stream's first event.
func TestAnImageRefusedInsideTheStreamIsHandledTheSameWay(t *testing.T) {
	refusal := &provider.StreamError{ErrorType: "invalid_request_error",
		Message: "this server was started without the vision encoder, so it cannot read images"}
	client := &scriptedClient{turns: []scriptedTurn{{cutErr: refusal}, {text: "I cannot see it"}}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)
	a.SetMessages(withImage())

	if err := a.Run(context.Background(), ""); err != nil {
		t.Fatalf("Run: %v", err)
	}
	retry := client.seen[len(client.seen)-1]
	if tr := retry[len(retry)-1]; len(tr.Images) != 0 || !strings.Contains(tr.Content, "vision encoder") {
		t.Errorf("retried tool result = %+v, want the image replaced by the server's answer", tr)
	}
	if !strings.Contains(sink.text(), "cannot see it") {
		t.Errorf("text = %q; the run stopped at the refusal", sink.text())
	}
}
