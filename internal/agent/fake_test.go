package agent

import (
	"context"
	"io"
	"strings"
	"sync"

	"ai-code/internal/provider"
)

// scriptedClient replays a fixed sequence of assistant turns, so the loop,
// cancellation and message shapes are tested deterministically.
type scriptedClient struct {
	mu    sync.Mutex
	turns []scriptedTurn
	n     int
	seen  [][]provider.Message
	// reqs keeps the whole request: max_tokens and the thinking level are
	// computed per turn.
	reqs []provider.Request
	// charsPerToken, when set, makes the client report usage computed from the
	// request it was given, at this density.
	charsPerToken float64
	// tokenizer, when set, counts a request instead of charsPerToken, so the
	// backend can disagree with the agent's own estimate.
	tokenizer func(provider.Request) int
	onStream  func()
	// refuseOver, when set, rejects any request whose counted size exceeds it,
	// the way a backend rejects a prompt longer than its window.
	refuseOver int

	// workerTurns, when set, is the script for requests carrying the worker
	// system prompt, with its own counter; workers run beside the parent.
	workerTurns []scriptedTurn
	workerN     int
	// onWorkerStream runs when a worker request is served, off the lock.
	onWorkerStream func()
	// onParentStream runs when a request that is not a worker's is served,
	// off the lock, so a test can hold the parent at a known point.
	onParentStream func()
}

func isWorkerRequest(req provider.Request) bool {
	for _, m := range req.Messages {
		if m.Role == provider.RoleSystem && strings.Contains(m.Content, "worker agent") {
			return true
		}
	}
	return false
}

// requestTokens measures a request the way a server would, at the client's
// configured density.
func (c *scriptedClient) requestTokens(req provider.Request) int {
	if c.tokenizer != nil {
		return c.tokenizer(req)
	}
	chars := 0
	for _, m := range req.Messages {
		chars += messageChars(m)
	}
	for _, t := range req.Tools {
		chars += len(t.Name) + len(t.Description) + len(t.Schema) + messageOverheadChars
	}
	return int(float64(chars) / c.charsPerToken)
}

type scriptedTurn struct {
	text      string
	reasoning string
	calls     []provider.ToolCall
	stop      provider.StopReason
	usage     provider.Usage
	err       error
}

func (c *scriptedClient) Name() string          { return "scripted" }
func (c *scriptedClient) Class() provider.Class { return provider.ClassOnPrem }

func (c *scriptedClient) Models(ctx context.Context) ([]provider.ModelInfo, error) {
	return []provider.ModelInfo{{ID: "test-model", ContextWindow: 65536, SupportsTools: true}}, nil
}

func (c *scriptedClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	// Refused before anything is recorded or the script advances: a rejected
	// request produced no turn.
	if c.refuseOver > 0 && c.requestTokens(req) > c.refuseOver {
		c.mu.Unlock()
		return nil, &provider.APIError{Status: 400,
			Message: "the request exceeds the available context length of this model"}
	}

	snapshot := make([]provider.Message, len(req.Messages))
	copy(snapshot, req.Messages)
	c.seen = append(c.seen, snapshot)
	c.reqs = append(c.reqs, req)

	var t scriptedTurn
	switch {
	case len(c.workerTurns) > 0 && isWorkerRequest(req):
		if c.workerN < len(c.workerTurns) {
			t = c.workerTurns[c.workerN]
			c.workerN++
		} else {
			t = scriptedTurn{text: "(worker script exhausted)", stop: provider.StopEnd}
		}
	case c.n < len(c.turns):
		t = c.turns[c.n]
		c.n++
	default:
		t = scriptedTurn{text: "(script exhausted)", stop: provider.StopEnd}
	}
	hook := c.onStream
	workerHook := c.onWorkerStream
	parentHook := c.onParentStream
	ratio := c.charsPerToken
	c.mu.Unlock()

	if workerHook != nil && isWorkerRequest(req) {
		workerHook()
	}
	if parentHook != nil && !isWorkerRequest(req) {
		parentHook()
	}

	if ratio > 0 || c.tokenizer != nil {
		t.usage.PromptTokens = c.requestTokens(req)
		if t.usage.CompletionTokens == 0 {
			// Measured the same way as the prompt: a server counts the tool
			// call it generated.
			den := ratio
			if den <= 0 {
				den = 4
			}
			t.usage.CompletionTokens = int(float64(messageChars(provider.Message{
				Role: provider.RoleAssistant, Content: t.text, ToolCalls: t.calls,
			})) / den)
		}
		t.usage.TotalTokens = t.usage.PromptTokens + t.usage.CompletionTokens
	}

	if hook != nil {
		hook()
	}
	if t.err != nil {
		return nil, t.err
	}
	return newScriptedStream(t), nil
}

func (c *scriptedClient) requests() [][]provider.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.seen
}

// maxTokens is the cap sent on each request, in order.
func (c *scriptedClient) maxTokens() []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]int, 0, len(c.reqs))
	for _, r := range c.reqs {
		out = append(out, r.MaxTokens)
	}
	return out
}

func (c *scriptedClient) lastFullRequest() provider.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.reqs) == 0 {
		return provider.Request{}
	}
	return c.reqs[len(c.reqs)-1]
}

func (c *scriptedClient) lastRequest() []provider.Message {
	reqs := c.requests()
	if len(reqs) == 0 {
		return nil
	}
	return reqs[len(reqs)-1]
}

type scriptedStream struct {
	turn   scriptedTurn
	events []provider.Event
	i      int
}

func newScriptedStream(t scriptedTurn) *scriptedStream {
	s := &scriptedStream{turn: t}
	if t.reasoning != "" {
		s.events = append(s.events, provider.Event{Kind: provider.EventReasoning, Text: t.reasoning})
	}
	if t.text != "" {
		// Deliver text in fragments, as a real stream does, so consumers cannot
		// accidentally depend on receiving whole lines.
		for _, chunk := range chunks(t.text, 7) {
			s.events = append(s.events, provider.Event{Kind: provider.EventText, Text: chunk})
		}
	}
	for i, tc := range t.calls {
		s.events = append(s.events, provider.Event{
			Kind: provider.EventToolCallStart, ToolIndex: i, ToolID: tc.ID, ToolName: tc.Name,
		})
		s.events = append(s.events, provider.Event{
			Kind: provider.EventToolCallArgs, ToolIndex: i, ToolID: tc.ID, ToolName: tc.Name, Text: tc.Args,
		})
	}
	if t.usage.PromptTokens > 0 || t.usage.CompletionTokens > 0 {
		s.events = append(s.events, provider.Event{Kind: provider.EventUsage, Usage: t.usage})
	}
	return s
}

func chunks(s string, n int) []string {
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if s != "" {
		out = append(out, s)
	}
	return out
}

func (s *scriptedStream) Recv() (provider.Event, error) {
	if s.i >= len(s.events) {
		return provider.Event{}, io.EOF
	}
	e := s.events[s.i]
	s.i++
	return e, nil
}

func (s *scriptedStream) Message() provider.Message {
	return provider.Message{
		Role:      provider.RoleAssistant,
		Content:   s.turn.text,
		Reasoning: s.turn.reasoning,
		ToolCalls: s.turn.calls,
	}
}

func (s *scriptedStream) Usage() provider.Usage { return s.turn.usage }

func (s *scriptedStream) StopReason() provider.StopReason {
	if s.turn.stop != "" {
		return s.turn.stop
	}
	if len(s.turn.calls) > 0 {
		return provider.StopToolCalls
	}
	return provider.StopEnd
}

func (s *scriptedStream) Close() error { return nil }

// collectSink accumulates events for assertions.
type collectSink struct {
	mu     sync.Mutex
	events []Event
}

func (c *collectSink) Emit(e Event) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, e)
}

func (c *collectSink) kinds() []EventKind {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]EventKind, 0, len(c.events))
	for _, e := range c.events {
		out = append(out, e.Kind)
	}
	return out
}

func (c *collectSink) text() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	s := ""
	for _, e := range c.events {
		if e.Kind == EvText {
			s += e.Text
		}
	}
	return s
}

// contexts returns every context figure published, in order.
func (c *collectSink) contexts() []ContextState {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []ContextState
	for _, e := range c.events {
		if e.Context != nil {
			out = append(out, *e.Context)
		}
	}
	return out
}

func (c *collectSink) notices() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, e := range c.events {
		if e.Kind == EvNotice || e.Kind == EvError {
			out = append(out, e.Text)
		}
	}
	return out
}
