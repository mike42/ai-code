package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strconv"
	"strings"
	"sync"
)

// httpTransport speaks Streamable HTTP: every message is a POST, answered by
// a JSON body or an SSE stream of messages.
type httpTransport struct {
	conn    httpSink
	url     string
	headers map[string]string
	client  *http.Client

	base   context.Context
	cancel context.CancelFunc

	mu        sync.Mutex
	sessionID string
	version   string
}

// httpSink takes what comes back: a client's Conn, or the worker's bridge.
type httpSink interface {
	deliver(raw []byte)
	failRequest(id int64, err error)
}

func newHTTPTransport(sink httpSink, url string, headers map[string]string) *httpTransport {
	base, cancel := context.WithCancel(context.Background())
	return &httpTransport{conn: sink, url: url, headers: headers, client: &http.Client{}, base: base, cancel: cancel}
}

func (t *httpTransport) setVersion(v string) {
	t.mu.Lock()
	t.version = v
	t.mu.Unlock()
}

func (t *httpTransport) newRequest(ctx context.Context, method string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, t.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	t.mu.Lock()
	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}
	if t.version != "" {
		req.Header.Set("MCP-Protocol-Version", t.version)
	}
	t.mu.Unlock()
	return req, nil
}

// send posts one message. A request's answer is read in the background, so
// a long stream of progress does not hold up the caller; the stream belongs
// to the connection, not to ctx, and cancelling a call is said with
// notifications/cancelled.
func (t *httpTransport) send(ctx context.Context, msg []byte) error {
	req, err := t.newRequest(t.base, http.MethodPost, msg)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := t.client.Do(req)
	if err != nil {
		return unwrapURLError(err)
	}
	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		defer resp.Body.Close()
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		msg := strings.TrimSpace(string(snippet))
		if msg != "" {
			msg = ": " + firstLine(msg)
		}
		return fmt.Errorf("%s answered %s%s", t.url, resp.Status, msg)
	}
	id, isRequest := requestID(msg)
	if resp.StatusCode == http.StatusAccepted || !isRequest {
		resp.Body.Close()
		return nil
	}
	go t.readResponse(resp, id)
	return nil
}

func (t *httpTransport) readResponse(resp *http.Response, id int64) {
	defer resp.Body.Close()
	ct, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	var err error
	if ct == "text/event-stream" {
		err = readSSE(resp.Body, t.conn.deliver)
	} else {
		var body []byte
		if body, err = io.ReadAll(resp.Body); err == nil {
			t.conn.deliver(body)
		}
	}
	if err != nil && t.base.Err() == nil {
		t.conn.failRequest(id, fmt.Errorf("the response from %s broke off: %w", t.url, err))
		return
	}
	// A stream that ended without answering leaves nobody to answer.
	t.conn.failRequest(id, fmt.Errorf("%s ended the response without an answer", t.url))
}

func requestID(msg []byte) (int64, bool) {
	var m struct {
		ID     json.RawMessage `json:"id"`
		Method string          `json:"method"`
	}
	if json.Unmarshal(msg, &m) != nil || m.Method == "" || len(m.ID) == 0 {
		return 0, false
	}
	id, err := strconv.ParseInt(string(m.ID), 10, 64)
	return id, err == nil
}

// readSSE hands every event's data to deliver.
func readSSE(r io.Reader, deliver func([]byte)) error {
	br := bufio.NewReader(r)
	var data []string
	flush := func() {
		if len(data) > 0 {
			deliver([]byte(strings.Join(data, "\n")))
			data = nil
		}
	}
	for {
		line, err := br.ReadString('\n')
		line = strings.TrimRight(line, "\r\n")
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "data:"):
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
		if err != nil {
			flush()
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

func (t *httpTransport) close() error {
	t.mu.Lock()
	sid := t.sessionID
	t.mu.Unlock()
	if sid != "" {
		if req, err := t.newRequest(context.Background(), http.MethodDelete, nil); err == nil {
			if resp, err := t.client.Do(req); err == nil {
				resp.Body.Close()
			}
		}
	}
	t.cancel()
	return nil
}

func unwrapURLError(err error) error {
	var ue interface{ Unwrap() error }
	if errors.As(err, &ue) {
		if inner := ue.Unwrap(); inner != nil {
			s := inner.Error()
			if i := strings.LastIndex(s, ": "); i >= 0 && strings.Contains(s, "dial") {
				return errors.New(s[i+2:])
			}
			return inner
		}
	}
	return err
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
