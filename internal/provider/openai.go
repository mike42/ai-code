package provider

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// TLSOptions covers the cases that make self-hosted backends painful: a
// private CA, a self-signed certificate, or mutual TLS.
type TLSOptions struct {
	// Insecure disables certificate verification entirely. Blunt, sometimes
	// exactly what a LAN service needs, and never hidden behind a rebuild.
	Insecure bool
	// CAFile is a PEM bundle to trust in addition to the system roots.
	CAFile string
	// CADir is a directory of PEM files to trust in addition to system roots.
	CADir string
	// ClientCert and ClientKey enable mutual TLS.
	ClientCert string
	ClientKey  string
}

// RetryPolicy governs transient failure handling. Retries happen only before
// any part of a response has been observed, so no retry duplicates output.
type RetryPolicy struct {
	MaxAttempts int
	BaseDelay   time.Duration
	MaxDelay    time.Duration
}

func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond, MaxDelay: 15 * time.Second}
}

// DefaultConnectTimeout bounds establishing a connection -- DNS, TCP and TLS.
// A server that is there may take as long as it takes.
const DefaultConnectTimeout = 30 * time.Second

// Dialect selects which of the two incompatible reasoning wire formats an
// endpoint speaks. Configured rather than sniffed, because getting it wrong is
// silent: an unrecognised `reasoning` object is passed through and ignored.
type Dialect string

const (
	// DialectOpenAI is reasoning_effort plus chat_template_kwargs: llama.cpp,
	// lemonade, vLLM, and the OpenAI API itself.
	DialectOpenAI Dialect = ""
	// DialectOpenRouter nests the same controls under a `reasoning` object and
	// names the response channel `reasoning` rather than `reasoning_content`.
	DialectOpenRouter Dialect = "openrouter"
)

// keyUse says whether one request carries the API key.
type keyUse bool

const (
	withKey    keyUse = true
	withoutKey keyUse = false
)

type Options struct {
	Name    string
	BaseURL string
	APIKey  string
	Class   Class
	TLS     TLSOptions
	Headers map[string]string
	Timeout time.Duration
	Retry   RetryPolicy
	// Dialect is the reasoning wire format; zero value is the OpenAI one.
	Dialect Dialect
	// SendReasoning replays prior assistant reasoning on later turns; off by
	// default as pure context cost.
	SendReasoning bool
}

// OpenAI is the generic OpenAI-compatible client; richer backends embed it and
// add their own introspection (see Lemonade).
type OpenAI struct {
	opts Options
	http *http.Client
}

func NewOpenAI(opts Options) (*OpenAI, error) {
	if opts.BaseURL == "" {
		return nil, fmt.Errorf("provider %q: base_url is required", opts.Name)
	}
	opts.BaseURL = strings.TrimRight(opts.BaseURL, "/")
	if opts.Retry.MaxAttempts == 0 {
		opts.Retry = DefaultRetryPolicy()
	}
	if opts.Timeout == 0 {
		opts.Timeout = DefaultConnectTimeout
	}
	tr, err := buildTransport(opts.TLS)
	if err != nil {
		return nil, fmt.Errorf("provider %q: %w", opts.Name, err)
	}
	// Bound connection setup only. Generation is bounded by the caller's context
	// and the token cap; a slow turn must not be failed by a timer.
	tr.DialContext = (&net.Dialer{Timeout: opts.Timeout, KeepAlive: 30 * time.Second}).DialContext
	tr.TLSHandshakeTimeout = opts.Timeout
	return &OpenAI{opts: opts, http: &http.Client{Transport: tr}}, nil
}

func buildTransport(t TLSOptions) (*http.Transport, error) {
	cfg := &tls.Config{InsecureSkipVerify: t.Insecure}

	if t.CAFile != "" || t.CADir != "" {
		pool, err := x509.SystemCertPool()
		if err != nil || pool == nil {
			pool = x509.NewCertPool()
		}
		var files []string
		if t.CAFile != "" {
			files = append(files, t.CAFile)
		}
		if t.CADir != "" {
			entries, err := os.ReadDir(t.CADir)
			if err != nil {
				return nil, fmt.Errorf("ca_dir %q: %w", t.CADir, err)
			}
			for _, e := range entries {
				if !e.IsDir() {
					files = append(files, filepath.Join(t.CADir, e.Name()))
				}
			}
		}
		for _, f := range files {
			pem, err := os.ReadFile(f)
			if err != nil {
				return nil, fmt.Errorf("ca_file %q: %w", f, err)
			}
			if !pool.AppendCertsFromPEM(pem) {
				return nil, fmt.Errorf("ca_file %q: no valid PEM certificates found", f)
			}
		}
		cfg.RootCAs = pool
	}

	if (t.ClientCert == "") != (t.ClientKey == "") {
		return nil, errors.New("client_cert and client_key must be set together")
	}
	if t.ClientCert != "" {
		cert, err := tls.LoadX509KeyPair(t.ClientCert, t.ClientKey)
		if err != nil {
			return nil, fmt.Errorf("client certificate: %w", err)
		}
		cfg.Certificates = []tls.Certificate{cert}
	}

	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.TLSClientConfig = cfg
	// Streaming responses must not be buffered by compression negotiation.
	tr.DisableCompression = true
	return tr, nil
}

func (c *OpenAI) Name() string    { return c.opts.Name }
func (c *OpenAI) Class() Class    { return c.opts.Class }
func (c *OpenAI) BaseURL() string { return c.opts.BaseURL }

type wireMessage struct {
	Role             string         `json:"role"`
	Content          any            `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	Reasoning        string         `json:"reasoning,omitempty"`
	ToolCalls        []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	Name             string         `json:"name,omitempty"`
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type wireRequest struct {
	Model         string         `json:"model"`
	Messages      []wireMessage  `json:"messages"`
	Tools         []wireTool     `json:"tools,omitempty"`
	MaxTokens     int            `json:"max_tokens,omitempty"`
	Temperature   *float64       `json:"temperature,omitempty"`
	TopP          *float64       `json:"top_p,omitempty"`
	Stop          []string       `json:"stop,omitempty"`
	Stream        bool           `json:"stream"`
	StreamOptions *streamOptions `json:"stream_options,omitempty"`

	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
	ChatTemplateKwa map[string]any `json:"chat_template_kwargs,omitempty"`
	Reasoning       *wireReasoning `json:"reasoning,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type wireReasoning struct {
	Effort string `json:"effort,omitempty"`
	// Enabled is how OpenRouter turns thinking off; there is no `effort:
	// "none"` there, that spelling is a llama.cpp-level disable.
	Enabled *bool `json:"enabled,omitempty"`
}

func (c *OpenAI) encode(req Request) wireRequest {
	w := wireRequest{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		Stop:          req.Stop,
		Stream:        true,
		StreamOptions: &streamOptions{IncludeUsage: true},
	}
	for _, m := range req.Messages {
		wm := wireMessage{Role: string(m.Role), Name: m.Name, ToolCallID: m.ToolCallID}
		// Some servers reject a missing content field, others an empty string;
		// null is the interoperable choice.
		if m.Content == "" && m.Role == RoleAssistant {
			wm.Content = nil
		} else {
			wm.Content = m.Content
		}
		if c.opts.SendReasoning && m.Role == RoleAssistant {
			if c.opts.Dialect == DialectOpenRouter {
				wm.Reasoning = m.Reasoning
			} else {
				wm.ReasoningContent = m.Reasoning
			}
		}
		for _, tc := range m.ToolCalls {
			var wtc wireToolCall
			wtc.ID = tc.ID
			wtc.Type = "function"
			wtc.Function.Name = tc.Name
			wtc.Function.Arguments = tc.Args
			wm.ToolCalls = append(wm.ToolCalls, wtc)
		}
		w.Messages = append(w.Messages, wm)
	}
	for _, t := range req.Tools {
		var wt wireTool
		wt.Type = "function"
		wt.Function.Name = t.Name
		wt.Function.Description = t.Description
		wt.Function.Parameters = t.Schema
		w.Tools = append(w.Tools, wt)
	}
	c.encodeEffort(&w, req)
	return w
}

func (c *OpenAI) encodeEffort(w *wireRequest, req Request) {
	// chat_template_kwargs is the caller's escape hatch, sent independently of
	// the effort level and never synthesised here.
	w.ChatTemplateKwa = req.TemplateKwargs

	if req.Effort == EffortUnset {
		return
	}
	if c.opts.Dialect == DialectOpenRouter {
		if req.Effort == EffortNone {
			// OpenRouter has no "none" level; disabling is a separate field.
			off := false
			w.Reasoning = &wireReasoning{Enabled: &off}
			return
		}
		w.Reasoning = &wireReasoning{Effort: string(req.Effort)}
		return
	}
	w.ReasoningEffort = string(req.Effort)
}

func (c *OpenAI) Stream(ctx context.Context, req Request) (Stream, error) {
	body, err := json.Marshal(c.encode(req))
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	resp, err := c.doWithRetry(ctx, "POST", "/chat/completions", body, withKey)
	if err != nil {
		return nil, err
	}
	return &openaiStream{resp: resp, br: bufio.NewReaderSize(resp.Body, 64*1024)}, nil
}

// doWithRetry retries transient failures with exponential backoff and jitter,
// honouring Retry-After; it never retries after the response body is read.
func (c *OpenAI) doWithRetry(ctx context.Context, method, path string, body []byte, key keyUse) (*http.Response, error) {
	var lastErr error
	for attempt := range c.opts.Retry.MaxAttempts {
		if attempt > 0 {
			delay := c.backoff(attempt, lastErr)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		req, err := http.NewRequestWithContext(ctx, method, c.opts.BaseURL+path, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		req.Header.Set("Accept", "text/event-stream")
		if key == withKey && c.opts.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)
		}
		for k, v := range c.opts.Headers {
			req.Header.Set(k, v)
		}

		resp, err := c.http.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			lastErr = err
			continue
		}
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		apiErr := readAPIError(resp)
		if !retryable(resp.StatusCode) {
			return nil, apiErr
		}
		lastErr = apiErr
	}
	return nil, fmt.Errorf("after %d attempts: %w", c.opts.Retry.MaxAttempts, lastErr)
}

func retryable(status int) bool {
	switch status {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	}
	return false
}

// APIError carries the server's own message, usually the most actionable thing
// a local backend offers (a model that is not loaded, a context overflow).
type APIError struct {
	Status     int
	Message    string
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("provider returned HTTP %d", e.Status)
	}
	return fmt.Sprintf("provider returned HTTP %d: %s", e.Status, e.Message)
}

func readAPIError(resp *http.Response) *APIError {
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))

	e := &APIError{Status: resp.StatusCode, Message: strings.TrimSpace(string(raw))}

	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &envelope) == nil {
		if envelope.Error.Message != "" {
			e.Message = envelope.Error.Message
		} else if envelope.Message != "" {
			e.Message = envelope.Message
		}
	}
	if ra := resp.Header.Get("Retry-After"); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil {
			e.RetryAfter = time.Duration(secs) * time.Second
		} else if t, err := http.ParseTime(ra); err == nil {
			e.RetryAfter = time.Until(t)
		}
	}
	return e
}

func (c *OpenAI) backoff(attempt int, lastErr error) time.Duration {
	var apiErr *APIError
	if errors.As(lastErr, &apiErr) && apiErr.RetryAfter > 0 {
		return min(apiErr.RetryAfter, c.opts.Retry.MaxDelay)
	}
	d := c.opts.Retry.BaseDelay * time.Duration(1<<uint(attempt-1))
	d = min(d, c.opts.Retry.MaxDelay)
	// Jitter, so sessions that back off together do not stay synchronised on a
	// single-slot server.
	return time.Duration(rand.Int63n(int64(d)) + int64(d)/2)
}

// IsContextOverflow reports whether an error is the backend refusing a request
// for being longer than the model's window. Best-effort: the API has no error
// code for this, so every server spells it differently in free text under 400.
func IsContextOverflow(err error) bool {
	var api *APIError
	if !errors.As(err, &api) || api.Status != http.StatusBadRequest {
		return false
	}
	m := strings.ToLower(api.Message)
	for _, s := range []string{
		"context length",
		"context window",
		"maximum context",
		"context_length_exceeded",
		"too many tokens",
		"prompt is too long",
		"exceeds the available context",
		"exceed context window",
		"n_ctx",
	} {
		if strings.Contains(m, s) {
			return true
		}
	}
	return false
}
