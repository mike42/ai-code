package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

func sessionForCompaction(t *testing.T, client *scriptedClient) *Agent {
	t.Helper()
	a := newAgent(t, client, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.charsPerToken = 4
	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: "do it"})
	for i := 0; i < 120; i++ {
		a.messages = append(a.messages,
			provider.Message{Role: provider.RoleAssistant,
				ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "c",
				Content: strings.Repeat("a line of output\n", 300)})
	}
	return a
}

// Every figure in a CompactResult measures the request; the transcript is append-only.
func TestCheckpointAndCompactionReportDifferentThings(t *testing.T) {
	client := &scriptedClient{charsPerToken: 4}
	a := sessionForCompaction(t, client)

	res, err := a.Summarise(context.Background(), a.SummaryCap())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rewrote {
		t.Error("Summarise reported rewriting the transcript, which it does not do")
	}
	if res.MessagesBefore != res.MessagesAfter {
		t.Errorf("a checkpoint reported %d messages -> %d; it removes none",
			res.MessagesBefore, res.MessagesAfter)
	}
	if res.TokensBefore != res.TokensAfter {
		t.Errorf("a checkpoint reported %d -> %d tokens; the session did not change",
			res.TokensBefore, res.TokensAfter)
	}
	if res.MessagesAfter != len(a.request().msgs) {
		t.Errorf("MessagesAfter = %d but the next request carries %d",
			res.MessagesAfter, len(a.request().msgs))
	}

	// Compact moves the boundary and must report a real saving without
	// removing messages.
	a2 := sessionForCompaction(t, &scriptedClient{charsPerToken: 4})
	transcript := len(a2.Messages())
	before := a2.RequestTokens()
	res2, err := a2.Compact(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !res2.Rewrote {
		t.Error("Compact did not report moving the boundary")
	}
	if res2.TokensBefore != before {
		t.Errorf("TokensBefore = %d, want the %d the next request would have cost",
			res2.TokensBefore, before)
	}
	if res2.TokensAfter >= res2.TokensBefore {
		t.Errorf("compaction reported no saving: %d -> %d", res2.TokensBefore, res2.TokensAfter)
	}
	if res2.MessagesAfter >= res2.MessagesBefore {
		t.Errorf("compaction reported %d messages -> %d", res2.MessagesBefore, res2.MessagesAfter)
	}
	if res2.MessagesAfter != len(a2.request().msgs) {
		t.Errorf("MessagesAfter = %d but the next request carries %d",
			res2.MessagesAfter, len(a2.request().msgs))
	}
	if got := len(a2.Messages()); got != transcript {
		t.Errorf("the transcript went from %d messages to %d; compaction removes nothing",
			transcript, got)
	}
}

// A zero keep-recent means "the derived size", not "keep nothing".
func TestCompactWithoutAnExplicitTailKeepsOne(t *testing.T) {
	a := sessionForCompaction(t, &scriptedClient{charsPerToken: 4})
	res, err := a.Compact(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.MessagesAfter < 3 {
		t.Errorf("compaction left %d messages: the checkpoint and nothing else",
			res.MessagesAfter)
	}
}
