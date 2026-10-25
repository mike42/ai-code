package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
)

// transport carries whole JSON-RPC messages to the server. Messages coming
// back are handed to the Conn it was built for, through deliver.
type transport interface {
	send(ctx context.Context, msg []byte) error
	close() error
}

type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *rpcError) Error() string { return fmt.Sprintf("%s (code %d)", e.Message, e.Code) }

// Conn is one JSON-RPC 2.0 session. Calls may run concurrently; the ids keep
// their responses apart.
type Conn struct {
	t transport

	mu       sync.Mutex
	next     int64
	pending  map[int64]chan message
	progress map[string]func(note string)
	// gone is why the connection ended; set once.
	gone error

	notify func(method string, params json.RawMessage)
}

func newConn() *Conn {
	return &Conn{pending: map[int64]chan message{}, progress: map[string]func(string){}}
}

// deliver takes one message, or a batch, from the server.
func (c *Conn) deliver(raw []byte) {
	if len(raw) > 0 && raw[0] == '[' {
		var batch []json.RawMessage
		if json.Unmarshal(raw, &batch) == nil {
			for _, m := range batch {
				c.deliver(m)
			}
		}
		return
	}
	var m message
	if err := json.Unmarshal(raw, &m); err != nil {
		return
	}
	switch {
	case m.Method != "" && len(m.ID) > 0:
		c.answerServerRequest(m)
	case m.Method != "":
		c.handleNotification(m)
	default:
		id, err := strconv.ParseInt(string(m.ID), 10, 64)
		if err != nil {
			return
		}
		c.mu.Lock()
		ch := c.pending[id]
		delete(c.pending, id)
		c.mu.Unlock()
		if ch != nil {
			ch <- m
		}
	}
}

// answerServerRequest refuses everything a tools-only client does not offer:
// sampling, elicitation, roots. A ping is answered.
func (c *Conn) answerServerRequest(m message) {
	reply := message{JSONRPC: "2.0", ID: m.ID}
	if m.Method == "ping" {
		reply.Result = json.RawMessage(`{}`)
	} else {
		reply.Error = &rpcError{Code: -32601, Message: "ai-code does not support " + m.Method}
	}
	raw, _ := json.Marshal(reply)
	_ = c.t.send(context.Background(), raw)
}

func (c *Conn) handleNotification(m message) {
	if m.Method == "notifications/progress" {
		var p struct {
			Token    json.RawMessage `json:"progressToken"`
			Progress float64         `json:"progress"`
			Total    float64         `json:"total"`
			Message  string          `json:"message"`
		}
		if json.Unmarshal(m.Params, &p) != nil {
			return
		}
		c.mu.Lock()
		fn := c.progress[string(p.Token)]
		c.mu.Unlock()
		if fn != nil {
			fn(progressNote(p.Message, p.Progress, p.Total))
		}
		return
	}
	if c.notify != nil {
		c.notify(m.Method, m.Params)
	}
}

func progressNote(msg string, progress, total float64) string {
	frac := ""
	if total > 0 {
		frac = fmt.Sprintf("%g/%g", progress, total)
	} else if progress > 0 {
		frac = fmt.Sprintf("%g", progress)
	}
	switch {
	case msg != "" && frac != "":
		return msg + " (" + frac + ")"
	case msg != "":
		return msg
	default:
		return frac
	}
}

// fail ends the connection: every waiting call returns err.
func (c *Conn) fail(err error) {
	c.mu.Lock()
	if c.gone == nil {
		c.gone = err
	}
	pending := c.pending
	c.pending = map[int64]chan message{}
	c.mu.Unlock()
	for _, ch := range pending {
		close(ch)
	}
}

// failRequest ends one call, for a transport that learns of a failure per
// request (an HTTP error status).
func (c *Conn) failRequest(id int64, err error) {
	c.mu.Lock()
	ch := c.pending[id]
	delete(c.pending, id)
	c.mu.Unlock()
	if ch != nil {
		ch <- message{Error: &rpcError{Code: -32000, Message: err.Error()}}
	}
}

// Err is why the connection ended, or nil while it is up.
func (c *Conn) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.gone
}

// Call sends a request and waits for its response, with no time limit.
// Cancelling ctx tells the server with notifications/cancelled.
func (c *Conn) Call(ctx context.Context, method string, params any, result any) error {
	return c.call(ctx, method, params, result, nil)
}

func (c *Conn) call(ctx context.Context, method string, params any, result any, onProgress func(string)) error {
	c.mu.Lock()
	if c.gone != nil {
		err := c.gone
		c.mu.Unlock()
		return err
	}
	c.next++
	id := c.next
	ch := make(chan message, 1)
	c.pending[id] = ch
	token := ""
	if onProgress != nil {
		token = strconv.FormatInt(id, 10)
		c.progress[token] = onProgress
	}
	c.mu.Unlock()
	defer func() {
		if token != "" {
			c.mu.Lock()
			delete(c.progress, token)
			c.mu.Unlock()
		}
	}()

	raw, err := encodeRequest(id, method, params, token)
	if err != nil {
		return err
	}
	if err := c.t.send(ctx, raw); err != nil {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return err
	}

	select {
	case m, ok := <-ch:
		if !ok {
			return c.Err()
		}
		if m.Error != nil {
			return m.Error
		}
		if result != nil {
			if err := json.Unmarshal(m.Result, result); err != nil {
				return fmt.Errorf("%s: unreadable result: %w", method, err)
			}
		}
		return nil
	case <-ctx.Done():
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		_ = c.Notify(context.Background(), "notifications/cancelled",
			map[string]any{"requestId": id, "reason": "cancelled by the user"})
		return ctx.Err()
	}
}

func encodeRequest(id int64, method string, params any, progressToken string) ([]byte, error) {
	p, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	if progressToken != "" {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(p, &obj); err != nil || obj == nil {
			obj = map[string]json.RawMessage{}
		}
		obj["_meta"] = json.RawMessage(`{"progressToken":` + progressToken + `}`)
		if p, err = json.Marshal(obj); err != nil {
			return nil, err
		}
	}
	return json.Marshal(message{JSONRPC: "2.0", ID: json.RawMessage(strconv.FormatInt(id, 10)), Method: method, Params: p})
}

// Notify sends a notification; nothing comes back.
func (c *Conn) Notify(ctx context.Context, method string, params any) error {
	m := message{JSONRPC: "2.0", Method: method}
	if params != nil {
		p, err := json.Marshal(params)
		if err != nil {
			return err
		}
		m.Params = p
	}
	raw, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return c.t.send(ctx, raw)
}

// Close ends the connection and whatever carries it.
func (c *Conn) Close() error {
	c.fail(errors.New("stopped"))
	return c.t.close()
}
