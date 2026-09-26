// Package agent implements the loop: send messages, run the tools the model
// asks for, repeat until it stops asking.
//
// The loop never writes to a terminal; it emits a typed event stream.
package agent

import (
	"encoding/json"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

type EventKind string

const (
	EvTurnStart EventKind = "turn_start"
	EvText      EventKind = "text"
	EvReasoning EventKind = "reasoning"
	EvToolStart EventKind = "tool_start"
	EvToolEnd   EventKind = "tool_end"
	EvTurnEnd   EventKind = "turn_end"
	// EvSteer reports a message typed during the turn folded into the conversation.
	EvSteer EventKind = "steer"
	// EvCompacted reports that the session was summarised to free context.
	EvCompacted EventKind = "compacted"
	// EvContext carries the context figure and is its only route to a consumer:
	// nothing outside the agent computes one.
	EvContext EventKind = "context"
	EvUsage   EventKind = "usage"
	EvNotice  EventKind = "notice"
	EvError   EventKind = "error"
	EvDone    EventKind = "done"
)

type Level string

const (
	LevelInfo Level = "info"
	LevelWarn Level = "warn"
)

// Event is one thing that happened. A struct rather than an interface so the
// JSON renderer is a single json.Encoder call.
type Event struct {
	Kind EventKind `json:"kind"`
	Turn int       `json:"turn,omitempty"`

	// Text carries the delta for EvText and EvReasoning, and the message for
	// EvNotice, EvError and EvDone.
	Text  string `json:"text,omitempty"`
	Level Level  `json:"level,omitempty"`

	// Tool fields, set for EvToolStart and EvToolEnd.
	ToolID     string          `json:"tool_id,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"`
	ToolArgs   json.RawMessage `json:"tool_args,omitempty"`
	ToolResult *tool.Result    `json:"tool_result,omitempty"`

	Usage      *provider.Usage     `json:"usage,omitempty"`
	StopReason provider.StopReason `json:"stop_reason,omitempty"`

	// Context reports the context figure at the moment of the event.
	Context *ContextState `json:"context,omitempty"`

	// Compaction is set on EvCompacted, and carries what the compaction did.
	Compaction *CompactResult `json:"compaction,omitempty"`
}

// ContextState is the next request's occupancy against the window. Projected
// is derived from the newest reported prefill that still describes this
// request's prefix; Anchored says whether such a report existed.
type ContextState struct {
	Projected int  `json:"projected"`
	Window    int  `json:"window"`
	Anchored  bool `json:"anchored"`
}

func (c ContextState) Percent() int {
	if c.Window <= 0 {
		return 0
	}
	return c.Projected * 100 / c.Window
}

// Sink receives events. Implementations must be safe to call from the loop's
// goroutine and must not block for long: a slow sink stalls generation.
type Sink interface {
	Emit(Event)
}

// SinkFunc adapts a function to Sink.
type SinkFunc func(Event)

func (f SinkFunc) Emit(e Event) { f(e) }

// MultiSink fans out to several sinks.
type MultiSink []Sink

func (m MultiSink) Emit(e Event) {
	for _, s := range m {
		s.Emit(e)
	}
}
