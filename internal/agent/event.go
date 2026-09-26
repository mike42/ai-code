// Package agent implements the loop: send messages, run the tools the model
// asks for, repeat until it stops asking.
//
// The loop never writes to a terminal. It emits a typed event stream, and
// renderers consume it. That separation exists so the same loop drives an
// interactive terminal, a plain pipe, a JSON stream for scripting, and tests --
// and so that the tests can assert on what happened without parsing ANSI.
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
	// EvSteer reports that a message the user typed during the turn has been
	// folded into the conversation.
	EvSteer EventKind = "steer"
	// EvCompacted reports that the session was summarised to free context.
	EvCompacted EventKind = "compacted"
	// EvContext carries the context figure. It is emitted after every change
	// to what the next request will carry, and it is the only way that figure
	// reaches a consumer: nothing outside the agent computes one. Two caches
	// of the same number, one of them updated on three event kinds out of
	// twelve, is how /compact came to leave a stale occupancy on screen.
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

// Event is one thing that happened. It is a struct rather than an interface so
// that the JSON renderer is a single json.Encoder call and the wire format
// stays obvious.
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

	// Context reports window occupancy after a turn, for the status line.
	Context *ContextState `json:"context,omitempty"`

	// Compaction is set on EvCompacted, and carries what the compaction did.
	Compaction *CompactResult `json:"compaction,omitempty"`
}

// ContextState is what the next request will occupy, against the window it
// has to fit in.
//
// Projected is the quantity defined in accounting.go: never measured, always
// derived from the newest reported prefill that still describes this request's
// prefix. Anchored says whether such a report existed. An estimate is a fine
// thing to show; an estimate indistinguishable from a measurement is not, so
// the flag is rendered rather than kept for diagnostics.
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

// MultiSink fans out to several sinks, so a session transcript and a terminal
// renderer can consume the same stream.
type MultiSink []Sink

func (m MultiSink) Emit(e Event) {
	for _, s := range m {
		s.Emit(e)
	}
}
