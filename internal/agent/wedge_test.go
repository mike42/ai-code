package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

// A response cut off part-way through a tool call produces a call with no id --
// the id arrives in the first delta, and there was no first delta. Validate
// rejects it, and Repair only synthesises results for calls that *have* an id,
// so nothing could ever fix it. The conversation became permanently unsendable:
// every later turn failed instantly on the same message, and the session was
// dead until it was restarted.
//
// Repair has one job, and it is total: whatever it is handed, the result must
// pass Validate. Anything less means some state exists that the session cannot
// get out of.
func TestRepairAlwaysProducesASendableConversation(t *testing.T) {
	cases := []struct {
		name string
		msgs []provider.Message
	}{
		{
			name: "tool call truncated before its id arrived",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "read the file"},
				{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
					{ID: "", Name: "read", Args: `{"file_pa`},
				}},
			},
		},
		{
			name: "truncated before the name arrived either",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "go"},
				{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{{ID: "", Name: "", Args: "{"}}},
			},
		},
		{
			name: "one good call and one truncated",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "go"},
				{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
					{ID: "c1", Name: "read", Args: `{}`},
					{ID: "", Name: "grep", Args: `{"pat`},
				}},
				{Role: provider.RoleTool, ToolCallID: "c1", Content: "ok"},
			},
		},
		{
			name: "unanswered call, the cancellation case",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "go"},
				{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{{ID: "c1", Name: "read"}}},
			},
		},
		{
			name: "tool result with no call above it",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "go"},
				{Role: provider.RoleTool, ToolCallID: "c9", Content: "orphan"},
			},
		},
		{
			name: "tool result with no id at all",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "go"},
				{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{{ID: "c1", Name: "read"}}},
				{Role: provider.RoleTool, ToolCallID: "", Content: "who am I answering"},
			},
		},
		{
			name: "results separated from their call by an assistant message",
			msgs: []provider.Message{
				{Role: provider.RoleUser, Content: "go"},
				{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{{ID: "c1", Name: "read"}}},
				{Role: provider.RoleAssistant, Content: "thinking out loud"},
				{Role: provider.RoleTool, ToolCallID: "c1", Content: "late"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			repaired, n := Repair(tc.msgs)
			if err := Validate(repaired); err != nil {
				t.Errorf("Repair left the conversation unsendable (%d repairs):\n%v\n\nmessages: %s",
					n, err, roles(repaired))
			}
		})
	}
}

// The end-to-end shape of the bug: one truncated tool call, and every turn
// afterwards fails before it reaches the model.
func TestATruncatedToolCallDoesNotWedgeTheSession(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{
		// Cut off mid-call: no id, partial arguments.
		{calls: []provider.ToolCall{{ID: "", Name: "probe", Args: `{"x`}}, stop: provider.StopLength},
		{text: "recovered"},
		{text: "and still working"},
	}}
	ft := &fakeTool{name: "probe", readOnly: true}
	a := newAgent(t, client, &collectSink{}, ft)

	if err := a.Run(context.Background(), "first"); err != nil {
		t.Fatalf("the truncated turn itself failed: %v", err)
	}
	if err := Validate(a.Messages()); err != nil {
		t.Fatalf("conversation is unsendable after a truncated tool call: %v", err)
	}

	sink := &collectSink{}
	a.sink = sink
	if err := a.Run(context.Background(), "second"); err != nil {
		t.Fatalf("the next turn failed: %v", err)
	}
	if got := sink.text(); !strings.Contains(got, "working") {
		t.Errorf("text = %q, want the session to have carried on", got)
	}
}

// The session already on disk when this bug bit still contains the bad message.
// Resuming it has to recover, not fail the same way it did before -- otherwise
// the fix only helps sessions started after it.
func TestResumingAWedgedSessionRecovers(t *testing.T) {
	client := &scriptedClient{turns: []scriptedTurn{{text: "back in business"}}}
	sink := &collectSink{}
	a := newAgent(t, client, sink)

	// Exactly what a truncated turn wrote to the transcript.
	a.SetMessages([]provider.Message{
		{Role: provider.RoleUser, Content: "read the file"},
		{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
			{ID: "", Name: "read", Args: `{"file_pa`},
		}},
	})
	if err := Validate(a.Messages()); err == nil {
		t.Fatal("test setup: the resumed conversation should start out invalid")
	}

	if err := a.Run(context.Background(), "carry on"); err != nil {
		t.Fatalf("resuming a wedged session still fails: %v", err)
	}
	if got := sink.text(); !strings.Contains(got, "back in business") {
		t.Errorf("text = %q, want the session to have recovered", got)
	}
	if err := Validate(a.Messages()); err != nil {
		t.Errorf("conversation still invalid after the run: %v", err)
	}

	var repaired bool
	for _, n := range sink.notices() {
		if strings.Contains(n, "Repaired") {
			repaired = true
		}
	}
	if !repaired {
		t.Error("the recovery happened silently; it should say what it had to fix")
	}
}
