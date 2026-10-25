// Package hello is a minimal MCP server for testing ai-code's client, over
// stdio or Streamable HTTP. The mcphello command wraps it.
package hello

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
)

type Options struct {
	Greeting string
	Root     string
	// Hold, when set, is an address dialled before answering initialize; the
	// answer waits until one byte arrives on that connection.
	Hold string
	// Token makes the HTTP handler require "Authorization: Bearer <Token>".
	Token string
	// ListChanges adds a goodbye tool after the first call and says so.
	ListChanges bool
	// Exit ends the process, for the crash tool; nil leaves crash out.
	Exit func(status int)
}

type msg struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   any             `json:"error,omitempty"`
}

type server struct {
	opt Options

	mu        sync.Mutex
	called    bool
	cancelled map[string]chan struct{}
}

func newServer(opt Options) *server {
	return &server{opt: opt, cancelled: map[string]chan struct{}{}}
}

func (s *server) tools() []map[string]any {
	t := []map[string]any{
		{
			"name":        "hello",
			"description": "Greets someone.",
			"inputSchema": json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`),
			"annotations": map[string]any{"readOnlyHint": true},
		},
		{
			"name":        "wait",
			"description": "Reports progress, then waits until cancelled.",
			"inputSchema": json.RawMessage(`{"type":"object"}`),
		},
	}
	if s.opt.Exit != nil {
		t = append(t, map[string]any{
			"name":        "crash",
			"description": "Exits with the given status.",
			"inputSchema": json.RawMessage(`{"type":"object","properties":{"status":{"type":"integer"}}}`),
		})
	}
	s.mu.Lock()
	changed := s.opt.ListChanges && s.called
	s.mu.Unlock()
	if changed {
		t = append(t, map[string]any{
			"name":        "goodbye",
			"description": "Says goodbye.",
			"inputSchema": json.RawMessage(`{"type":"object"}`),
		})
	}
	return t
}

// handle answers one message through send, which may be called more than
// once (progress, then the result).
func (s *server) handle(m msg, send func(msg)) {
	reply := func(result any) { send(msg{JSONRPC: "2.0", ID: m.ID, Result: result}) }
	fail := func(code int, text string) {
		send(msg{JSONRPC: "2.0", ID: m.ID, Error: map[string]any{"code": code, "message": text}})
	}
	switch m.Method {
	case "initialize":
		if s.opt.Hold != "" {
			if c, err := net.Dial("tcp", s.opt.Hold); err == nil {
				var b [1]byte
				_, _ = c.Read(b[:])
				c.Close()
			}
		}
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		_ = json.Unmarshal(m.Params, &p)
		reply(map[string]any{
			"protocolVersion": p.ProtocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{"listChanged": s.opt.ListChanges}},
			"serverInfo":      map[string]any{"name": "hello", "version": "1"},
		})
	case "notifications/initialized":
	case "notifications/cancelled":
		var p struct {
			RequestID json.RawMessage `json:"requestId"`
		}
		_ = json.Unmarshal(m.Params, &p)
		s.mu.Lock()
		if ch := s.cancelled[string(p.RequestID)]; ch != nil {
			close(ch)
			delete(s.cancelled, string(p.RequestID))
		}
		s.mu.Unlock()
	case "ping":
		reply(map[string]any{})
	case "tools/list":
		var p struct {
			Cursor string `json:"cursor"`
		}
		_ = json.Unmarshal(m.Params, &p)
		all := s.tools()
		// One tool per page, to exercise pagination.
		i, _ := strconv.Atoi(p.Cursor)
		page := map[string]any{"tools": all[i : i+1]}
		if i+1 < len(all) {
			page["nextCursor"] = strconv.Itoa(i + 1)
		}
		reply(page)
	case "tools/call":
		s.call(m, reply, fail, send)
	default:
		if len(m.ID) > 0 {
			fail(-32601, "no method "+m.Method)
		}
	}
}

func (s *server) call(m msg, reply func(any), fail func(int, string), send func(msg)) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Meta      struct {
			ProgressToken json.RawMessage `json:"progressToken"`
		} `json:"_meta"`
	}
	_ = json.Unmarshal(m.Params, &p)
	text := func(t string) map[string]any {
		return map[string]any{"content": []map[string]any{{"type": "text", "text": t}}}
	}
	first := false
	s.mu.Lock()
	if !s.called {
		s.called, first = true, true
	}
	s.mu.Unlock()
	defer func() {
		if first && s.opt.ListChanges {
			send(msg{JSONRPC: "2.0", Method: "notifications/tools/list_changed"})
		}
	}()

	switch p.Name {
	case "hello":
		var a struct {
			Name string `json:"name"`
		}
		_ = json.Unmarshal(p.Arguments, &a)
		greeting := s.opt.Greeting
		if greeting == "" {
			greeting = "Hello"
		}
		out := greeting + ", " + a.Name
		if s.opt.Root != "" {
			out += " (root " + s.opt.Root + ")"
		}
		reply(text(out))
	case "goodbye":
		reply(text("Goodbye"))
	case "wait":
		ch := make(chan struct{})
		s.mu.Lock()
		s.cancelled[string(m.ID)] = ch
		s.mu.Unlock()
		if len(p.Meta.ProgressToken) > 0 {
			send(msg{JSONRPC: "2.0", Method: "notifications/progress", Params: mustJSON(map[string]any{
				"progressToken": p.Meta.ProgressToken, "progress": 1, "total": 2, "message": "waiting",
			})})
		}
		<-ch
	case "crash":
		var a struct {
			Status int `json:"status"`
		}
		_ = json.Unmarshal(p.Arguments, &a)
		if s.opt.Exit != nil {
			s.opt.Exit(a.Status)
		}
	default:
		reply(map[string]any{"isError": true, "content": []map[string]any{{"type": "text", "text": "no tool " + p.Name}}})
	}
}

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// Serve speaks newline-delimited JSON-RPC over r and w until r ends.
func Serve(r io.Reader, w io.Writer, opt Options) error {
	s := newServer(opt)
	var wmu sync.Mutex
	send := func(m msg) {
		b, _ := json.Marshal(m)
		wmu.Lock()
		_, _ = w.Write(append(b, '\n'))
		wmu.Unlock()
	}
	br := bufio.NewReader(r)
	var wg sync.WaitGroup
	for {
		line, err := br.ReadBytes('\n')
		if line = bytes.TrimSpace(line); len(line) > 0 {
			var m msg
			if json.Unmarshal(line, &m) == nil {
				// Requests run concurrently so that wait cannot block a cancel.
				if m.Method == "tools/call" {
					wg.Add(1)
					go func() { defer wg.Done(); s.handle(m, send) }()
				} else {
					s.handle(m, send)
				}
			}
		}
		if err != nil {
			wg.Wait()
			return nil
		}
	}
}

// Handler serves Streamable HTTP. Requests are answered as SSE streams, or
// as plain JSON when jsonOnly is set.
func Handler(opt Options, jsonOnly bool) http.Handler {
	s := newServer(opt)
	var mu sync.Mutex
	sessions := map[string]bool{}
	next := 0
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if opt.Token != "" && r.Header.Get("Authorization") != "Bearer "+opt.Token {
			http.Error(w, "unauthorised", http.StatusUnauthorized)
			return
		}
		switch r.Method {
		case http.MethodDelete:
			mu.Lock()
			delete(sessions, r.Header.Get("Mcp-Session-Id"))
			mu.Unlock()
			return
		case http.MethodPost:
		default:
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var m msg
		if err := json.Unmarshal(body, &m); err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest)
			return
		}
		if m.Method == "initialize" {
			mu.Lock()
			next++
			sid := fmt.Sprintf("session-%d", next)
			sessions[sid] = true
			mu.Unlock()
			w.Header().Set("Mcp-Session-Id", sid)
		} else {
			mu.Lock()
			ok := sessions[r.Header.Get("Mcp-Session-Id")]
			mu.Unlock()
			if !ok {
				http.Error(w, "unknown session", http.StatusNotFound)
				return
			}
		}
		if len(m.ID) == 0 {
			s.handle(m, func(msg) {})
			w.WriteHeader(http.StatusAccepted)
			return
		}
		if jsonOnly {
			var out []msg
			s.handle(m, func(o msg) {
				if len(o.ID) > 0 {
					out = append(out, o)
				}
			})
			w.Header().Set("Content-Type", "application/json")
			if len(out) > 0 {
				_ = json.NewEncoder(w).Encode(out[len(out)-1])
			}
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		var wmu sync.Mutex
		s.handle(m, func(o msg) {
			b, _ := json.Marshal(o)
			wmu.Lock()
			defer wmu.Unlock()
			fmt.Fprintf(w, "event: message\ndata: %s\n\n", strings.TrimSpace(string(b)))
			if flusher != nil {
				flusher.Flush()
			}
		})
	})
}
