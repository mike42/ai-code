package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

var errReset = errors.New("read tcp: connection reset by peer")

// A stream that dies mid-call is answered like a failed tool, and the model
// gets to try again in the same run.
func TestACutOffToolCallIsReportedToTheModelAndTheRunGoesOn(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "writing it", calls: []provider.ToolCall{{ID: "", Name: "probe", Args: `{"x":"hal`}},
			cutErr: errReset},
		{calls: []provider.ToolCall{{ID: "c2", Name: "probe", Args: `{"x":"half"}`}}},
		{text: "done"},
	}}
	ft := &fakeTool{name: "probe", readOnly: true}
	sink := &collectSink{}
	a := newAgent(t, client, sink, ft)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(sink.text(), "done") {
		t.Fatalf("text = %q; the run stopped at the cut-off instead of carrying on", sink.text())
	}
	if len(ft.calls) != 1 {
		t.Errorf("tool ran %d times, want once: the cut-off call must not run", len(ft.calls))
	}

	second := client.seen[1]
	if err := Validate(second); err != nil {
		t.Fatalf("the request after the cut-off is malformed: %v", err)
	}
	last := second[len(second)-1]
	if last.Role != provider.RoleTool || !last.IsError ||
		!strings.Contains(last.Content, "not run") || !strings.Contains(last.Content, "connection reset") {
		t.Errorf("model was not told the call failed and why: %+v", last)
	}
	call := second[len(second)-2].ToolCalls[0]
	if call.Args != "{}" || call.ID == "" {
		t.Errorf("cut-off call sent back as %+v; want an id and valid JSON arguments", call)
	}

	notices := strings.Join(sink.notices(), "\n")
	if !strings.Contains(notices, "1 tool call was not run") {
		t.Errorf("notices = %q, want a grammatical count", notices)
	}
	if strings.Contains(notices, "connection to the model server") {
		t.Errorf("notices = %q; the cause must be the reported error, not a guess", notices)
	}
}

func TestACutOffInTextAsksTheModelToCarryOn(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		{text: "halfway through a sent", cutErr: errReset},
		{text: "ence, finished"},
	}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(sink.text(), "finished") {
		t.Fatalf("text = %q; the run stopped at the cut-off", sink.text())
	}
	second := client.seen[1]
	if last := second[len(second)-1]; last.Role != provider.RoleUser || !strings.Contains(last.Content, "cut off") {
		t.Errorf("last message before the retry = %+v, want a note that the response was cut off", last)
	}
}

// Nothing arrived, so the server failed before the model did anything; going
// round again would only repeat it.
func TestACutOffBeforeAnyOutputStops(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{cutErr: errReset}, {text: "should not be asked"}}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)

	if err := a.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(client.seen) != 1 {
		t.Errorf("sent %d requests, want 1", len(client.seen))
	}
	if n := strings.Join(sink.notices(), "\n"); !strings.Contains(n, "connection reset") {
		t.Errorf("notices = %q, want the actual error", n)
	}
	if err := Validate(a.Messages()); err != nil {
		t.Errorf("conversation unsendable afterwards: %v", err)
	}
}
