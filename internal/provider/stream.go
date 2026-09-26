package provider

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

type chunk struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			// OpenRouter names the same channel `reasoning` and additionally
			// sends structured blocks in `reasoning_details`; lemonade and
			// llama.cpp send `reasoning_content`. Accepting only one of them
			// silently discarded every thinking token from the other.
			Reasoning        string            `json:"reasoning"`
			ReasoningDetails []json.RawMessage `json:"reasoning_details"`
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
		// NativeFinishReason is the upstream provider's own word for why it
		// stopped. When the normalised reason is "error" it is the only thing
		// in the response that says what actually happened.
		NativeFinishReason string `json:"native_finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int     `json:"prompt_tokens"`
		CompletionTokens    int     `json:"completion_tokens"`
		TotalTokens         int     `json:"total_tokens"`
		Cost                float64 `json:"cost"`
		PromptTokensDetails *struct {
			CachedTokens     int `json:"cached_tokens"`
			CacheWriteTokens int `json:"cache_write_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		// Code is a string for some error families and a bare number for the
		// HTTP-shaped ones (402 and friends), so it cannot be typed.
		Code     json.RawMessage `json:"code"`
		Metadata *struct {
			ErrorType string `json:"error_type"`
		} `json:"metadata"`
	} `json:"error"`
}

// StreamError is a provider failure delivered inside the body of a 200
// response. Branching on HTTP status cannot distinguish these: the headers
// went out before generation started, so a context overflow that is fixable by
// compacting and a dead upstream both arrive as 200. ErrorType is the field
// that separates them.
type StreamError struct {
	Message   string
	Code      string
	ErrorType string
}

func (e *StreamError) Error() string {
	var b strings.Builder
	b.WriteString("provider error mid-stream")
	if e.ErrorType != "" {
		b.WriteString(" (" + e.ErrorType + ")")
	} else if e.Code != "" {
		b.WriteString(" (code " + e.Code + ")")
	}
	if e.Message != "" {
		b.WriteString(": " + e.Message)
	}
	return b.String()
}

// Recoverable reports whether retrying after shrinking the request can work.
func (e *StreamError) Recoverable() bool {
	return e.ErrorType == ErrorTypeContextLength
}

const ErrorTypeContextLength = "context_length_exceeded"

// ExtraUsage carries the usage fields provider.Usage has no room for.
type ExtraUsage struct {
	// Cost is what the provider charged for this call in its own units, not an
	// estimate derived from token counts and a price table.
	Cost             float64
	ReasoningTokens  int
	CacheWriteTokens int
}

// StreamExtras is the optional interface for wire detail the canonical Stream
// has nowhere to put. Backends that do not send these do not implement it.
type StreamExtras interface {
	NativeFinishReason() string
	ReasoningDetails() []json.RawMessage
	ExtraUsage() ExtraUsage
}

type toolAccum struct {
	id   string
	name string
	args strings.Builder
}

// openaiStream parses a server-sent-event stream and reassembles it into a
// single assistant message.
//
// Tool-call arguments arrive as arbitrary fragments of a JSON document, split
// at positions with no relationship to JSON structure (observed live: `{`,
// `"cmd":"`, `ls`, ` -`, `la`). Reassembly happens here so there is exactly one
// implementation of it in the codebase.
type openaiStream struct {
	resp *http.Response
	br   *bufio.Reader

	pending []Event
	tools   map[int]*toolAccum

	content   strings.Builder
	reasoning strings.Builder
	usage     Usage
	extra     ExtraUsage
	stop      StopReason
	native    string
	details   []json.RawMessage
	finished  bool
	err       error
}

var _ StreamExtras = (*openaiStream)(nil)

func (s *openaiStream) Recv() (Event, error) {
	for {
		if len(s.pending) > 0 {
			ev := s.pending[0]
			s.pending = s.pending[1:]
			return ev, nil
		}
		if s.finished {
			if s.err != nil {
				return Event{}, s.err
			}
			return Event{}, io.EOF
		}
		if err := s.readFrame(); err != nil {
			s.finished = true
			s.err = err
			// A stream that ends without an explicit [DONE] still yields
			// whatever was accumulated; the caller decides whether a partial
			// turn is salvageable.
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				s.err = nil
				continue
			}
			continue
		}
	}
}

func (s *openaiStream) readFrame() error {
	for {
		line, err := s.br.ReadString('\n')
		if err != nil {
			if len(strings.TrimSpace(line)) == 0 {
				return err
			}
			// Process the final partial line before reporting the error.
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" || strings.HasPrefix(line, ":") {
			if err != nil {
				return err
			}
			continue
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			// Some servers omit the space after the colon.
			if data, ok = strings.CutPrefix(line, "data:"); !ok {
				if err != nil {
					return err
				}
				continue
			}
		}
		if data == "[DONE]" {
			s.finalise()
			s.finished = true
			return nil
		}

		var c chunk
		if jsonErr := json.Unmarshal([]byte(data), &c); jsonErr != nil {
			// A malformed frame is worth reporting but not worth destroying an
			// otherwise good turn over.
			if err != nil {
				return err
			}
			continue
		}
		s.apply(c)
		return err
	}
}

func (s *openaiStream) apply(c chunk) {
	if c.Error != nil {
		e := &StreamError{Message: c.Error.Message, Code: errorCode(c.Error.Code)}
		if c.Error.Metadata != nil {
			e.ErrorType = c.Error.Metadata.ErrorType
		}
		if e.Message != "" || e.Code != "" || e.ErrorType != "" {
			s.err = e
			s.stop = StopError
			s.finished = true
			return
		}
	}

	if c.Usage != nil {
		s.usage = Usage{
			PromptTokens:     c.Usage.PromptTokens,
			CompletionTokens: c.Usage.CompletionTokens,
			TotalTokens:      c.Usage.TotalTokens,
		}
		s.extra = ExtraUsage{Cost: c.Usage.Cost}
		if c.Usage.PromptTokensDetails != nil {
			s.usage.CachedTokens = c.Usage.PromptTokensDetails.CachedTokens
			s.extra.CacheWriteTokens = c.Usage.PromptTokensDetails.CacheWriteTokens
		}
		if c.Usage.CompletionTokensDetails != nil {
			s.extra.ReasoningTokens = c.Usage.CompletionTokensDetails.ReasoningTokens
		}
		s.pending = append(s.pending, Event{Kind: EventUsage, Usage: s.usage})
	}

	for _, ch := range c.Choices {
		d := ch.Delta

		if ch.NativeFinishReason != "" {
			s.native = ch.NativeFinishReason
		}

		for _, r := range []string{d.ReasoningContent, d.Reasoning} {
			if r == "" {
				continue
			}
			s.reasoning.WriteString(r)
			s.pending = append(s.pending, Event{Kind: EventReasoning, Text: r})
		}
		s.details = append(s.details, d.ReasoningDetails...)
		if d.Content != "" {
			s.content.WriteString(d.Content)
			s.pending = append(s.pending, Event{Kind: EventText, Text: d.Content})
		}

		for _, tc := range d.ToolCalls {
			if s.tools == nil {
				s.tools = map[int]*toolAccum{}
			}
			acc, seen := s.tools[tc.Index]
			if !seen {
				acc = &toolAccum{}
				s.tools[tc.Index] = acc
			}
			if tc.ID != "" {
				acc.id = tc.ID
			}
			if tc.Function.Name != "" {
				acc.name = tc.Function.Name
			}
			if !seen || (tc.ID != "" && tc.Function.Name != "") {
				s.pending = append(s.pending, Event{
					Kind: EventToolCallStart, ToolIndex: tc.Index,
					ToolID: acc.id, ToolName: acc.name,
				})
			}
			if tc.Function.Arguments != "" {
				acc.args.WriteString(tc.Function.Arguments)
				s.pending = append(s.pending, Event{
					Kind: EventToolCallArgs, ToolIndex: tc.Index,
					ToolID: acc.id, ToolName: acc.name, Text: tc.Function.Arguments,
				})
			}
		}

		switch ch.FinishReason {
		case "stop":
			s.stop = StopEnd
		case "tool_calls", "function_call":
			s.stop = StopToolCalls
		case "length":
			s.stop = StopLength
		case "error":
			s.stop = StopError
		case "content_filter":
			s.stop = StopContentFilter
		}
	}
}

func (s *openaiStream) finalise() {
	if s.stop == "" {
		if len(s.tools) > 0 {
			s.stop = StopToolCalls
		} else {
			s.stop = StopEnd
		}
	}
}

func (s *openaiStream) Message() Message {
	m := Message{
		Role:      RoleAssistant,
		Content:   s.content.String(),
		Reasoning: s.reasoning.String(),
	}
	indices := make([]int, 0, len(s.tools))
	for i := range s.tools {
		indices = append(indices, i)
	}
	sort.Ints(indices)
	for _, i := range indices {
		acc := s.tools[i]
		args := acc.args.String()
		if strings.TrimSpace(args) == "" {
			// A tool call with no arguments is legal; the schema may have no
			// required fields. Send a valid empty object rather than "".
			args = "{}"
		}
		id := acc.id
		if id == "" {
			// The id arrives in the first delta of a call, so a response cut
			// off part-way through one produces a call without it. An id is
			// only the handle a result is matched to its call by, and it is our
			// own message that carries both, so synthesising one costs nothing
			// -- while leaving it empty produces a message the provider rejects
			// and that nothing downstream can answer.
			id = fmt.Sprintf("call_%d", i)
		}
		m.ToolCalls = append(m.ToolCalls, ToolCall{ID: id, Name: acc.name, Args: args})
	}
	return m
}

func (s *openaiStream) Usage() Usage           { return s.usage }
func (s *openaiStream) StopReason() StopReason { s.finalise(); return s.stop }

func (s *openaiStream) ExtraUsage() ExtraUsage              { return s.extra }
func (s *openaiStream) NativeFinishReason() string          { return s.native }
func (s *openaiStream) ReasoningDetails() []json.RawMessage { return s.details }

// errorCode flattens a code that arrives either quoted or bare.
func errorCode(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}

func (s *openaiStream) Close() error {
	if s.resp != nil && s.resp.Body != nil {
		return s.resp.Body.Close()
	}
	return nil
}
