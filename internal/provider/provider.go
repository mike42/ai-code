// Package provider defines the canonical message model and the client interface
// every backend implements. The message model follows the OpenAI
// chat-completions API; backend-specific richness lives in optional interfaces.
package provider

import (
	"context"
	"encoding/json"
	"fmt"
)

// Class separates providers that may receive data freely from those under
// compliance and privacy obligations. Switching a live session onto a cloud
// model requires explicit confirmation; see internal/agent.
type Class string

const (
	ClassOnPrem Class = "on-premises"
	ClassCloud  Class = "cloud"
)

func ParseClass(s string) (Class, error) {
	switch s {
	case string(ClassOnPrem), "onprem", "on-prem", "local":
		return ClassOnPrem, nil
	case string(ClassCloud):
		return ClassCloud, nil
	case "":
		return "", fmt.Errorf("provider class is required (\"cloud\" or \"on-premises\")")
	default:
		return "", fmt.Errorf("unknown provider class %q (want \"cloud\" or \"on-premises\")", s)
	}
}

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a single request from the model to run a tool. Args is raw JSON
// as the model produced it; validation happens at dispatch so a malformed call
// becomes a tool result the model can recover from.
type ToolCall struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	Args string `json:"args"`
}

// Message is one entry in the conversation.
//
// Invariant enforced by internal/agent/validate.go: every ToolCall in an
// assistant message is answered by exactly one RoleTool message carrying the
// matching ToolCallID, in the messages immediately following it.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content,omitempty"`

	// Reasoning holds a separate chain-of-thought channel (llama.cpp and
	// lemonade emit `reasoning_content` deltas); rendered dimmed, not sent back.
	Reasoning string `json:"reasoning,omitempty"`

	// Assistant only.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// RoleTool only.
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`

	// PromptTokens and Completion are what the backend reported for the request
	// that produced this assistant message: tokens read and tokens generated.
	// Zero elsewhere, and on any figure that failed the plausibility check.
	PromptTokens int `json:"prefill,omitempty"`
	Completion   int `json:"completion,omitempty"`
}

// ToolDef is a tool advertised to the model. Schema is a JSON Schema object.
type ToolDef struct {
	Name        string
	Description string
	Schema      json.RawMessage
}

type Request struct {
	Model       string
	Messages    []Message
	Tools       []ToolDef
	MaxTokens   int
	Temperature *float64
	TopP        *float64
	Stop        []string

	// Effort is the thinking level for this call, already resolved against the
	// model's detected shape; EffortUnset sends nothing.
	Effort Effort
	// TemplateKwargs carries chat_template_kwargs for backends that route
	// thinking controls through the template rather than reasoning_effort.
	TemplateKwargs map[string]any
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	CachedTokens     int `json:"cached_tokens"`
	TotalTokens      int `json:"total_tokens"`

	// ReasoningTokens is the part of CompletionTokens spent thinking; on a model
	// at maximum effort it can dwarf the visible answer.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	// CacheWriteTokens is what a provider charged to populate its prompt cache.
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	// Cost is what the provider says the call cost, in its own units, taken from
	// the response rather than a local price table that goes stale.
	Cost float64 `json:"cost,omitempty"`
}

type EventKind int

const (
	EventText EventKind = iota
	EventReasoning
	EventToolCallStart
	EventToolCallArgs
	EventUsage
)

// Event is one incremental update from a Stream; consumers that only want the
// finished message can read Stream.Message() after the stream ends.
type Event struct {
	Kind      EventKind
	Text      string
	ToolIndex int
	ToolID    string
	ToolName  string
	Usage     Usage
}

// StopReason describes why generation ended. "length" matters: a summary or a
// patch produced under a length stop is truncated and must not be trusted.
type StopReason string

const (
	StopEnd       StopReason = "end"
	StopToolCalls StopReason = "tool_calls"
	StopLength    StopReason = "length"
	StopAborted   StopReason = "aborted"

	StopError         StopReason = "error"
	StopContentFilter StopReason = "content_filter"
)

// Stream yields incremental events and accumulates the final assistant message.
// Reassembly of streamed tool-call argument fragments lives here, in one
// implementation, rather than in each consumer.
type Stream interface {
	// Recv returns the next event, or io.EOF when the stream is complete.
	Recv() (Event, error)
	// Message returns the accumulated assistant message, valid once Recv has
	// returned io.EOF; safe to call earlier to inspect partial state.
	Message() Message
	Usage() Usage
	StopReason() StopReason
	Close() error
}

// ModelInfo describes one model a provider can serve.
type ModelInfo struct {
	ID string

	// ContextWindow is the effective per-request context in tokens, not
	// llama.cpp's ctx_size: the KV cache is divided across --parallel slots.
	ContextWindow int

	// MaxContextWindow is the model's architectural limit, when known.
	MaxContextWindow int
	// CtxSize is the total KV allocation the backend was launched with.
	CtxSize int
	// Parallel is the number of concurrent slots sharing CtxSize.
	Parallel int

	// KVUnified reports whether slots share one KV pool; when they do, a slot is
	// not limited to CtxSize/Parallel. Not inferable from Parallel.
	KVUnified bool
	// PerSlotLimit is an explicit --kv-unified-per-slot, overriding the division
	// and KVUnified.
	PerSlotLimit int

	// Slots is the number of concurrent request slots observed on the backend, 0
	// when it does not report them; distinct from Parallel, what was requested.
	Slots int
	// SlotsBusy is how many of them were processing when last observed.
	SlotsBusy int

	// MaxOutputTokens is the backend's own cap on a single completion; zero means
	// unknown, not unlimited.
	MaxOutputTokens int

	SupportsTools bool
	Labels        []string
	Loaded        bool

	// Estimated marks a ContextWindow that was inferred rather than observed.
	Estimated bool
}

// Client is the minimum every backend implements.
type Client interface {
	Name() string
	Class() Class
	Stream(ctx context.Context, req Request) (Stream, error)
	Models(ctx context.Context) ([]ModelInfo, error)
}

// Health is the state of a backend that can only hold a limited number of
// models resident; on a single-slot server it shows when the model was evicted.
type Health struct {
	Ready       bool
	ModelLoaded string
	MaxLLMSlots int
	Busy        bool
	Streaming   bool
}

// Introspector is implemented by backends that can report and control model
// residency; when absent, the requested model is assumed servable.
type Introspector interface {
	Health(ctx context.Context) (*Health, error)
	Load(ctx context.Context, model string) error
	Unload(ctx context.Context, model string) error
}
