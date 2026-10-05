package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// instructionAllowance is the most a summary request may read fresh: the
// instruction appended to the cached request, and nothing else.
const instructionAllowance = 3000

// cachingServer stands in for a llama.cpp slot: it keeps the last prompt it
// processed, plus what it generated, and reads only the part of a new prompt
// that differs from it. readFresh is that part, per request, in characters.
type cachingServer struct {
	mu     sync.Mutex
	rounds int // tool rounds the model works through before it finishes
	// refuseSummary answers a summary request with a tool call, the way a
	// model that ignores the instruction does.
	refuseSummary bool
	slot          string
	readFresh     []int
	summaries     []int // indexes into readFresh of the summary requests
	sent          []provider.Request
}

func renderPrompt(req provider.Request) string {
	var b strings.Builder
	tools, _ := json.Marshal(req.Tools)
	b.Write(tools)
	for _, m := range req.Messages {
		calls, _ := json.Marshal(m.ToolCalls)
		fmt.Fprintf(&b, "<%s|%s|%s|%s>", m.Role, m.ToolCallID, m.Content, calls)
	}
	return b.String()
}

func isSummaryRequest(req provider.Request) bool {
	last := req.Messages[len(req.Messages)-1]
	return last.Role == provider.RoleUser && strings.Contains(last.Content, "checkpoint")
}

func (c *cachingServer) Name() string          { return "caching" }
func (c *cachingServer) Class() provider.Class { return provider.ClassOnPrem }
func (c *cachingServer) Models(context.Context) ([]provider.ModelInfo, error) {
	return nil, nil
}

func (c *cachingServer) Stream(_ context.Context, req provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	prompt := renderPrompt(req)
	common := 0
	for common < len(prompt) && common < len(c.slot) && prompt[common] == c.slot[common] {
		common++
	}
	c.readFresh = append(c.readFresh, len(prompt)-common)
	c.sent = append(c.sent, req)

	var t scriptedTurn
	switch {
	case isSummaryRequest(req) && c.refuseSummary:
		t = scriptedTurn{calls: []provider.ToolCall{{ID: "r", Name: "scan", Args: `{}`}}}
	case isSummaryRequest(req):
		c.summaries = append(c.summaries, len(c.readFresh)-1)
		t = scriptedTurn{text: "## Goal\nScan the tree.\n\n## Next Steps\n1. Keep scanning."}
	case c.rounds > 0:
		c.rounds--
		t = scriptedTurn{calls: []provider.ToolCall{{
			ID: fmt.Sprintf("c%d", c.rounds), Name: "scan",
			Args: fmt.Sprintf(`{"part":%d}`, c.rounds)}}}
	default:
		t = scriptedTurn{text: "done"}
	}
	c.slot = prompt + t.text
	t.usage = provider.Usage{PromptTokens: len(prompt) / 4, CompletionTokens: 50}
	return newScriptedStream(t), nil
}

// bulkyTool returns a fixed amount of output, the shape that fills a window.
type bulkyTool struct{ size int }

func (b *bulkyTool) Name() string            { return "scan" }
func (b *bulkyTool) Description() string     { return "scan part of the tree" }
func (b *bulkyTool) ReadOnly() bool          { return true }
func (b *bulkyTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (b *bulkyTool) Run(context.Context, *tool.State, json.RawMessage) tool.Result {
	return tool.Result{Content: strings.Repeat("internal/x/y.go:12: func z()\n", b.size/29)}
}

func cachingSession(t *testing.T, server *cachingServer) (*Agent, *collectSink) {
	t.Helper()
	sink := &collectSink{}
	exec := tool.NewLocalExecutor(tool.NewState(t.TempDir()), &bulkyTool{size: 6000})
	a := New(server, "m", exec, sink, Options{
		ContextLimit: 65536, AutoCompact: true, LoopGuard: 4,
	})
	a.SetSystem(realisticSystemPrompt)
	return a, sink
}

// A long tool-heavy run is summarised by the model while the session still
// fits, from the request the server has cached, and nothing is cleared.
func TestALongRunIsSummarisedFromTheCachedRequest(t *testing.T) {
	server := &cachingServer{rounds: 80}
	a, sink := cachingSession(t, server)

	if err := a.Run(context.Background(), "scan the whole tree"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(server.summaries) == 0 {
		t.Fatal("the run never summarised; the test did not fill the window")
	}
	for _, n := range sink.notices() {
		if strings.Contains(n, "Cleared the output") {
			t.Errorf("tool output was cleared although the model could summarise: %q", n)
		}
	}
	for _, i := range server.summaries {
		// What the previous request did not carry is read fresh: the newest
		// tool results and the instruction. Nothing before them is read again.
		added := len(renderPrompt(server.sent[i])) - len(renderPrompt(server.sent[i-1]))
		if fresh := server.readFresh[i]; fresh > added {
			t.Errorf("summary request %d read %d characters fresh, more than the %d added since the request before it",
				i, fresh, added)
		}
		if !isSummaryRequest(server.sent[i]) || isSummaryRequest(server.sent[i-1]) {
			t.Errorf("request %d is not a summary following a turn", i)
		}
	}
}

// The checkpoint written while idle is read from the cache too, and leaves the
// cache in place for the next prompt.
func TestAnIdleSummaryReadsFromTheCacheAndKeepsIt(t *testing.T) {
	server := &cachingServer{rounds: 3}
	a, _ := cachingSession(t, server)
	if err := a.Run(context.Background(), "scan a little"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if _, err := a.Summarise(context.Background(), SpeculativeSummaryMaxTokens); err != nil {
		t.Fatalf("Summarise: %v", err)
	}
	if fresh := server.readFresh[len(server.readFresh)-1]; fresh > instructionAllowance {
		t.Errorf("the idle summary read %d characters fresh; only the instruction should be new", fresh)
	}

	server.rounds = 0
	if err := a.Run(context.Background(), "thanks"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if fresh := server.readFresh[len(server.readFresh)-1]; fresh > 500 {
		t.Errorf("the prompt after an idle summary read %d characters fresh; the summary must not "+
			"displace the session from the cache", fresh)
	}
}

// Clearing tool output is the fallback when the model will not summarise,
// and the run still finishes.
func TestToolOutputIsClearedOnlyWhenSummarisingFails(t *testing.T) {
	server := &cachingServer{refuseSummary: true}
	a, sink := cachingSession(t, server)

	// Several prompts: output from the last two turns is never cleared.
	for i := 0; i < 6; i++ {
		server.rounds = 12
		if err := a.Run(context.Background(), fmt.Sprintf("scan part %d", i)); err != nil {
			t.Fatalf("Run %d: %v", i, err)
		}
	}
	failedAt, clearedAt := -1, -1
	for i, n := range sink.notices() {
		if failedAt < 0 && strings.Contains(n, "Summarising failed") {
			failedAt = i
		}
		if clearedAt < 0 && strings.Contains(n, "Cleared the output") {
			clearedAt = i
		}
	}
	if failedAt < 0 || clearedAt < failedAt {
		t.Errorf("want clearing only after a failed summary; notices = %q", sink.notices())
	}
}
