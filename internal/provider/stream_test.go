package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// cassetteServer replays a recorded SSE response, split at arbitrary read
// boundaries.
func cassetteServer(t *testing.T, name string) *httptest.Server {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("cassette %s: %v", name, err)
	}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Write in small pieces so the parser is exercised across arbitrary
		// read boundaries.
		for i := 0; i < len(body); i += 37 {
			end := min(i+37, len(body))
			_, _ = w.Write(body[i:end])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
}

func clientFor(t *testing.T, url string) *OpenAI {
	t.Helper()
	c, err := NewOpenAI(Options{Name: "test", BaseURL: url, Class: ClassOnPrem})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func drain(t *testing.T, s Stream) []Event {
	t.Helper()
	var out []Event
	for {
		ev, err := s.Recv()
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		out = append(out, ev)
	}
}

func TestStreamReassemblesFragmentedToolCallArguments(t *testing.T) {
	// The recorded server splits the arguments JSON at positions with no
	// relationship to its structure: `{`, `"cmd":"`, `ls`, `"`, `}`. Anything
	// that parses fragments individually produces garbage.
	srv := cassetteServer(t, "toolcall.sse")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	msg := stream.Message()
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(msg.ToolCalls))
	}
	tc := msg.ToolCalls[0]
	if tc.Name != "bash" {
		t.Errorf("tool name = %q, want %q", tc.Name, "bash")
	}
	if tc.ID == "" {
		t.Error("tool call has no id; the id arrives on the first fragment only")
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Args), &args); err != nil {
		t.Fatalf("reassembled arguments are not valid JSON (%v): %q", err, tc.Args)
	}
	if _, ok := args["cmd"]; !ok {
		t.Errorf("reassembled arguments lack the cmd field: %q", tc.Args)
	}
	if stream.StopReason() != StopToolCalls {
		t.Errorf("stop reason = %q, want %q", stream.StopReason(), StopToolCalls)
	}
}

func TestStreamCapturesReasoningSeparatelyFromContent(t *testing.T) {
	// lemonade/llama.cpp emit a distinct reasoning_content channel; folding it
	// into the visible content would put the model's thinking in the
	// scrollback.
	srv := cassetteServer(t, "text.sse")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	events := drain(t, stream)
	var sawReasoning, sawText bool
	for _, e := range events {
		switch e.Kind {
		case EventReasoning:
			sawReasoning = true
		case EventText:
			sawText = true
		}
	}
	if !sawReasoning {
		t.Error("no reasoning events; this cassette contains reasoning_content deltas")
	}
	if !sawText {
		t.Error("no text events")
	}

	msg := stream.Message()
	if msg.Reasoning == "" {
		t.Error("reasoning was not accumulated")
	}
	if strings.Contains(msg.Content, msg.Reasoning) && msg.Reasoning != "" {
		t.Error("reasoning leaked into the visible content")
	}
}

func TestStreamReportsUsageFromTheFinalChunk(t *testing.T) {
	srv := cassetteServer(t, "text.sse")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	u := stream.Usage()
	if u.PromptTokens <= 0 {
		t.Errorf("prompt tokens = %d; usage is the ground truth for context "+
			"accounting and must be captured", u.PromptTokens)
	}
	if u.CompletionTokens <= 0 {
		t.Errorf("completion tokens = %d", u.CompletionTokens)
	}
}

func TestRequestSetsStreamOptionsIncludeUsage(t *testing.T) {
	// Without this the final usage chunk is never sent and context accounting
	// silently falls back to estimates for the whole session.
	var got wireRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	s, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	drain(t, s)

	if got.StreamOptions == nil || !got.StreamOptions.IncludeUsage {
		t.Error("stream_options.include_usage was not requested")
	}
}

func TestToolResultMessagesEncodeWithTheirCallID(t *testing.T) {
	var got wireRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	s, err := clientFor(t, srv.URL).Stream(context.Background(), Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleUser, Content: "hi"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "abc", Name: "bash", Args: `{"cmd":"ls"}`}}},
			{Role: RoleTool, ToolCallID: "abc", Name: "bash", Content: "file.txt"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	drain(t, s)

	if len(got.Messages) != 3 {
		t.Fatalf("got %d messages, want 3", len(got.Messages))
	}
	if got.Messages[1].Content != nil {
		t.Error("an assistant message with only tool calls should send null content, " +
			"which is what servers accept interoperably")
	}
	if len(got.Messages[1].ToolCalls) != 1 || got.Messages[1].ToolCalls[0].ID != "abc" {
		t.Error("assistant tool call was not encoded with its id")
	}
	if got.Messages[2].ToolCallID != "abc" {
		t.Errorf("tool result tool_call_id = %q, want %q", got.Messages[2].ToolCallID, "abc")
	}
}

func TestRetriesTransientFailuresThenSucceeds(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"model is loading"}}`)
			return
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c, err := NewOpenAI(Options{
		Name: "t", BaseURL: srv.URL, Class: ClassOnPrem,
		Retry: RetryPolicy{MaxAttempts: 4, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatalf("should have retried through the 503s: %v", err)
	}
	defer s.Close()
	if got := attempts.Load(); got != 3 {
		t.Errorf("made %d attempts, want 3", got)
	}
}

func TestDoesNotRetryAClientError(t *testing.T) {
	// A 400 means the request is wrong. Retrying it wastes time and, on a
	// single-slot server, occupies a slot other sessions are waiting for.
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"unknown model"}}`)
	}))
	defer srv.Close()

	c, err := NewOpenAI(Options{Name: "t", BaseURL: srv.URL, Class: ClassOnPrem,
		Retry: RetryPolicy{MaxAttempts: 4, BaseDelay: time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Stream(context.Background(), Request{Model: "m"}); err == nil {
		t.Fatal("expected an error")
	} else if !strings.Contains(err.Error(), "unknown model") {
		t.Errorf("the server's own message is the most actionable thing available "+
			"and should be surfaced, got: %v", err)
	}
	if got := attempts.Load(); got != 1 {
		t.Errorf("made %d attempts; a 400 must not be retried", got)
	}
}

func TestStreamStopsWhenContextIsCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f, _ := w.(http.Flusher)
		for i := 0; i < 1000; i++ {
			_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"x"}}]}`+"\n\n")
			if f != nil {
				f.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	s, err := clientFor(t, srv.URL).Stream(ctx, Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if _, err := s.Recv(); err != nil {
		t.Fatalf("first Recv: %v", err)
	}
	cancel()

	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			if _, err := s.Recv(); err != nil {
				return
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not stop after the context was cancelled")
	}
}

func TestTLSInsecureIsHonoured(t *testing.T) {
	// The self-hosted case other harnesses make hard: a server with a
	// certificate no public CA signed.
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	strict, err := NewOpenAI(Options{
		Name: "t", BaseURL: srv.URL, Class: ClassOnPrem,
		Retry: RetryPolicy{MaxAttempts: 1, BaseDelay: time.Millisecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := strict.Stream(context.Background(), Request{Model: "m"}); err == nil {
		t.Error("expected a certificate error without tls_insecure")
	}

	lax, err2 := NewOpenAI(Options{
		Name: "t", BaseURL: srv.URL, Class: ClassOnPrem,
		TLS: TLSOptions{Insecure: true},
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	s, err := lax.Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatalf("tls_insecure should have allowed this: %v", err)
	}
	s.Close()
}

func TestClientCertRequiresBothHalves(t *testing.T) {
	_, err := NewOpenAI(Options{
		Name: "t", BaseURL: "https://example.invalid", Class: ClassOnPrem,
		TLS: TLSOptions{ClientCert: "/tmp/cert.pem"},
	})
	if err == nil || !strings.Contains(err.Error(), "client_key") {
		t.Errorf("expected a clear error naming the missing half, got: %v", err)
	}
}

func TestModelsParsesLemonadeSlotArithmetic(t *testing.T) {
	// The shape /api/v1/models actually returns, including the parallel-slot
	// division that makes ctx_size misleading.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[
		 {"id":"Qwen3.6-27B","max_context_window":262144,"labels":["chat","tool-calling"],
		  "recipe_options":{"ctx_size":262144,"llamacpp_args":"--parallel 4 -ngl 999"}},
		 {"id":"Flux","labels":["image"]}]}`)
	}))
	defer srv.Close()

	models, err := clientFor(t, srv.URL).Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	// Tool-capable models sort first.
	m := models[0]
	if m.ID != "Qwen3.6-27B" {
		t.Fatalf("first model = %q, want the tool-capable one", m.ID)
	}
	if m.ContextWindow != 65536 {
		t.Errorf("context window = %d, want 65536 (262144 shared across 4 slots)", m.ContextWindow)
	}
	if !m.SupportsTools {
		t.Error("tool-calling label was not recognised")
	}
	if models[1].SupportsTools {
		t.Error("an image model should not be advertised as tool-capable")
	}
}

// A response cut off by the token cap is partial by definition. Compaction
// refuses to checkpoint one, so the stop reason has to survive parsing.
func TestStreamReportsLengthStop(t *testing.T) {
	srv := cassetteServer(t, "truncated.sse")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	if got := stream.StopReason(); got != StopLength {
		t.Errorf("stop reason = %q, want %q; this cassette hit the output cap "+
			"while still reasoning and produced no content at all", got, StopLength)
	}
	if stream.Message().Content != "" {
		t.Error("this cassette should have no visible content")
	}
	if stream.Message().Reasoning == "" {
		t.Error("reasoning should still have been captured")
	}
}

// The status line counts streamed deltas for its live token rate, because the
// authoritative count only arrives in the final usage chunk. That is sound only
// if one delta is one token, which this pins against real captured output.
func TestOneDeltaIsOneToken(t *testing.T) {
	cases := []struct {
		cassette string
		// toleranceLow allows for tokens the server bills but the renderer
		// never sees: the chat template's tool-call wrapper is billed but not
		// streamed, and argument fragments arrive as one EventToolCallStart.
		// Prose and reasoning are exact.
		toleranceLow float64
	}{
		{"text.sse", 0.01},
		{"truncated.sse", 0.01},
		{"toolcall.sse", 0.35},
	}

	for _, tc := range cases {
		t.Run(tc.cassette, func(t *testing.T) {
			srv := cassetteServer(t, tc.cassette)
			defer srv.Close()

			s, err := clientFor(t, srv.URL).Stream(context.Background(), Request{
				Model:    "Qwen3.6-27B",
				Messages: []Message{{Role: RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()

			deltas := 0
			for _, ev := range drain(t, s) {
				switch ev.Kind {
				case EventText, EventReasoning, EventToolCallStart:
					deltas++
				}
			}

			reported := s.Usage().CompletionTokens
			if reported == 0 {
				t.Fatal("cassette reported no completion_tokens")
			}
			err2 := float64(deltas-reported) / float64(reported)
			t.Logf("%d deltas vs %d completion_tokens (%+.1f%%)", deltas, reported, 100*err2)
			if err2 > 0.01 {
				t.Errorf("counted %d deltas but the server billed %d tokens: "+
					"counting deltas must never over-report", deltas, reported)
			}
			if -err2 > tc.toleranceLow {
				t.Errorf("counted %d deltas but the server billed %d tokens (%.1f%% low, "+
					"tolerance %.0f%%): one delta is no longer one token, so the tok/s "+
					"readout is understating", deltas, reported, -100*err2, 100*tc.toleranceLow)
			}
		})
	}
}

// sseServer serves a literal SSE body, for shapes no cassette happens to hold.
func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
}

func TestTruncatedToolCallStillGetsAnAnswerableID(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"file_pa"}}]},"finish_reason":"length"}]}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	msg := stream.Message()
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].ID == "" {
		t.Error("the tool call has no id; a conversation containing it can never be sent")
	}
	if stream.StopReason() != StopLength {
		t.Errorf("stop reason = %q, want %q", stream.StopReason(), StopLength)
	}
}

// The normal case must be untouched: an id the server sent is the id echoed back.
func TestServerSuppliedToolCallIDIsPreserved(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"read","arguments":"{}"}}]}}]}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	msg := stream.Message()
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_abc" {
		t.Errorf("tool calls = %+v, want the server's own id preserved", msg.ToolCalls)
	}
}

// OpenRouter puts thinking in `reasoning`, not `reasoning_content`; both names
// must reach the same accumulator.
func TestStreamAcceptsOpenRouterReasoningField(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"reasoning":"weigh","reasoning_details":[{"type":"reasoning.text","text":"weigh"}]}}]}`+"\n\n"+
		`data: {"choices":[{"delta":{"reasoning":" it up"}}]}`+"\n\n"+
		`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var reasoningEvents int
	for _, ev := range drain(t, stream) {
		if ev.Kind == EventReasoning {
			reasoningEvents++
		}
	}
	if reasoningEvents != 2 {
		t.Errorf("got %d reasoning events, want 2", reasoningEvents)
	}
	msg := stream.Message()
	if msg.Reasoning != "weigh it up" {
		t.Errorf("reasoning = %q, want %q", msg.Reasoning, "weigh it up")
	}
	if msg.Content != "done" {
		t.Errorf("content = %q, want %q", msg.Content, "done")
	}

	ex, ok := stream.(StreamExtras)
	if !ok {
		t.Fatal("stream does not expose StreamExtras")
	}
	details := ex.ReasoningDetails()
	if len(details) != 1 {
		t.Fatalf("got %d reasoning_details blocks, want 1", len(details))
	}
	// The blocks are round-tripped back to the provider verbatim on a later
	// turn, so they have to survive parsing unaltered.
	if !json.Valid(details[0]) || !strings.Contains(string(details[0]), "reasoning.text") {
		t.Errorf("reasoning_details block did not survive intact: %s", details[0])
	}
}

// The llama.cpp shape, alongside the OpenRouter one above: both names must
// reach the same accumulator.
func TestStreamAcceptsLlamaCppReasoningContentField(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"reasoning_content":"think"}}]}`+"\n\n"+
		`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	if got := stream.Message().Reasoning; got != "think" {
		t.Errorf("reasoning = %q, want %q", got, "think")
	}
	if got := stream.StopReason(); got != StopEnd {
		t.Errorf("stop reason = %q, want %q", got, StopEnd)
	}
}

// A content-filtered response is not a completed turn; reporting StopEnd would
// treat a refusal as the model's final answer.
func TestStreamReportsContentFilterStop(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"content":"par"},"finish_reason":"content_filter","native_finish_reason":"SAFETY"}]}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	if got := stream.StopReason(); got != StopContentFilter {
		t.Errorf("stop reason = %q, want %q; a filtered response must not look clean", got, StopContentFilter)
	}
	if got := stream.(StreamExtras).NativeFinishReason(); got != "SAFETY" {
		t.Errorf("native finish reason = %q, want %q", got, "SAFETY")
	}
}

func TestStreamReportsErrorFinishReason(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{},"finish_reason":"error","native_finish_reason":"upstream_502"}]}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	if got := stream.StopReason(); got != StopError {
		t.Errorf("stop reason = %q, want %q", got, StopError)
	}
	if got := stream.(StreamExtras).NativeFinishReason(); got != "upstream_502" {
		t.Errorf("native finish reason = %q, want %q; with finish_reason \"error\" "+
			"this is the only description of what went wrong", got, "upstream_502")
	}
}

// Mid-stream the HTTP status is always 200 -- the headers left before anything
// failed -- so error.metadata.error_type is the only thing that separates an
// overflow the harness can compact out of from a dead upstream.
func TestMidStreamErrorSurfacesRecoverableErrorType(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"content":"hi"}}]}`+"\n\n"+
		`data: {"error":{"message":"maximum context length exceeded","code":400,"metadata":{"error_type":"context_length_exceeded"}}}`+"\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var recvErr error
	for {
		if _, e := stream.Recv(); e != nil {
			recvErr = e
			break
		}
	}
	var se *StreamError
	if !errors.As(recvErr, &se) {
		t.Fatalf("error = %v (%T), want a *StreamError", recvErr, recvErr)
	}
	if se.ErrorType != ErrorTypeContextLength {
		t.Errorf("error type = %q, want %q", se.ErrorType, ErrorTypeContextLength)
	}
	if se.Code != "400" {
		t.Errorf("code = %q, want %q; OpenRouter sends it unquoted here", se.Code, "400")
	}
	if !se.Recoverable() {
		t.Error("a context overflow is recoverable by compacting and retrying")
	}
	if got := stream.StopReason(); got != StopError {
		t.Errorf("stop reason = %q, want %q", got, StopError)
	}
}

func TestMidStreamErrorWithUnknownTypeIsNotRecoverable(t *testing.T) {
	srv := sseServer(t, `data: {"error":{"message":"upstream is down","code":"provider_error","metadata":{"error_type":"provider_error"}}}`+"\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	var recvErr error
	for {
		if _, e := stream.Recv(); e != nil {
			recvErr = e
			break
		}
	}
	var se *StreamError
	if !errors.As(recvErr, &se) {
		t.Fatalf("error = %v (%T), want a *StreamError", recvErr, recvErr)
	}
	if se.Recoverable() {
		t.Error("a provider_error must not be retried as if it were an overflow")
	}
	if !strings.Contains(se.Error(), "upstream is down") {
		t.Errorf("the server's own message should be surfaced, got: %v", se)
	}
}

// usage.cost is what was actually charged. Deriving it from token counts and a
// price table gets caching, discounts and per-endpoint pricing wrong.
func TestStreamParsesCostAndReasoningTokens(t *testing.T) {
	srv := sseServer(t, `data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}],`+
		`"usage":{"prompt_tokens":100,"completion_tokens":40,"total_tokens":140,"cost":0.00123,`+
		`"completion_tokens_details":{"reasoning_tokens":31},`+
		`"prompt_tokens_details":{"cached_tokens":64,"cache_write_tokens":36}}}`+"\n\ndata: [DONE]\n\n")
	defer srv.Close()

	stream, err := clientFor(t, srv.URL).Stream(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	drain(t, stream)

	if u := stream.Usage(); u.PromptTokens != 100 || u.CompletionTokens != 40 || u.CachedTokens != 64 {
		t.Errorf("usage = %+v", u)
	}
	ex := stream.(StreamExtras).ExtraUsage()
	if ex.Cost != 0.00123 {
		t.Errorf("cost = %v, want 0.00123", ex.Cost)
	}
	if ex.ReasoningTokens != 31 {
		t.Errorf("reasoning tokens = %d, want 31", ex.ReasoningTokens)
	}
	if ex.CacheWriteTokens != 36 {
		t.Errorf("cache write tokens = %d, want 36", ex.CacheWriteTokens)
	}
}
