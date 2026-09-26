package agent

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

// Compacting twice must not cost twice as much as compacting once.
//
// Every compaction used to re-serialise the session from message zero, so the
// summarisation prompt grew with the transcript -- 30k, 61k, 92k tokens for
// the first three compactions of one run. Quadratic over a session, and since
// the summarisation request is the one request nothing assembles or bounds,
// it eventually could not be sent at all. The session then had no way left to
// make room, which is the opposite of what compaction is for.
func TestCompactionCostDoesNotGrowWithTheSession(t *testing.T) {
	client := &scriptedClient{charsPerToken: 4}
	a := newAgent(t, client, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.charsPerToken = 4

	bulk := strings.Repeat("match at offset 0x0000 in section .text\n", 400)
	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: "search it"})
	grow := func(n int) {
		for i := 0; i < n; i++ {
			a.messages = append(a.messages,
				provider.Message{Role: provider.RoleAssistant,
					ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{}`}}},
				provider.Message{Role: provider.RoleTool, ToolCallID: "c", Content: bulk})
		}
	}

	// Tokens sent to summarise, per compaction, on a session that keeps
	// growing by the same amount each time.
	var cost []int
	compact := func() {
		before := len(client.reqs)
		if _, err := a.Summarise(context.Background(), a.SummaryCap()); err != nil {
			t.Fatal(err)
		}
		spent := 0
		for _, req := range client.reqs[before:] {
			for _, m := range req.Messages {
				spent += messageChars(m)
			}
		}
		cost = append(cost, spent/4)
	}

	for i := 0; i < 3; i++ {
		grow(60)
		compact()
	}
	t.Logf("summarisation cost per compaction: %v tokens", cost)

	// Each compaction covers the same amount of new conversation, so each
	// should cost about the same. Generous margin: the checkpoint being
	// folded in is itself part of the prompt and does vary a little.
	for i, c := range cost[1:] {
		if c > cost[0]*3/2 {
			t.Errorf("compaction %d cost %d tokens against %d for the first: still growing with the session",
				i+2, c, cost[0])
		}
	}
}

// Nothing new since the checkpoint means there is nothing to ask a model.
func TestSummarisingTwiceWithNoNewConversationSpendsNoCall(t *testing.T) {
	client := &scriptedClient{charsPerToken: 4}
	a := newAgent(t, client, &collectSink{})
	a.SetSystem(realisticSystemPrompt)
	a.opts.ContextLimit = 262144
	a.charsPerToken = 4

	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: "do it"})
	for i := 0; i < 40; i++ {
		a.messages = append(a.messages,
			provider.Message{Role: provider.RoleAssistant,
				ToolCalls: []provider.ToolCall{{ID: "c", Name: "grep", Args: `{}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "c",
				Content: strings.Repeat("a line of output\n", 200)})
	}
	if _, err := a.Summarise(context.Background(), a.SummaryCap()); err != nil {
		t.Fatal(err)
	}
	calls := len(client.reqs)

	if _, err := a.Summarise(context.Background(), a.SummaryCap()); err != nil {
		t.Fatal(err)
	}
	if got := len(client.reqs) - calls; got != 0 {
		t.Errorf("%d further model calls for a session that has not moved", got)
	}
}
