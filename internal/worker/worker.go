// Package worker runs tools wherever a session's tools run. Protocol: JSON
// lines, a Hello, then one Result per Request; Cancel stops a running call.
package worker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"ai-code/internal/mcp"
	"ai-code/internal/tool"
)

type Settings struct {
	BashTimeout    time.Duration `json:"bash_timeout,omitempty"`
	BashMaxOutput  int           `json:"bash_max_output,omitempty"`
	Shell          string        `json:"shell,omitempty"`
	ReadMaxBytes   int           `json:"read_max_bytes,omitempty"`
	ReadMaxLines   int           `json:"read_max_lines,omitempty"`
	GrepMaxResults int           `json:"grep_max_results,omitempty"`
	// BashMaxOutputFunc, for tools in this process, is read at each command in
	// place of BashMaxOutput, which a worker elsewhere receives as a number.
	BashMaxOutputFunc func() int `json:"-"`
}

type Hello struct {
	Settings Settings `json:"settings"`
	// MCP, when set, makes this worker relay one MCP server instead of
	// serving tools: run here, or reached from here.
	MCP *mcp.Launch `json:"mcp,omitempty"`
}

// ExitStatus ends the worker with Code and nothing printed: a relayed MCP
// server's own status, which ai-code reports.
type ExitStatus struct{ Code int }

func (e ExitStatus) Error() string { return fmt.Sprintf("exit status %d", e.Code) }

func Tools(s Settings, cwd string) *tool.LocalExecutor {
	return tool.NewLocalExecutor(tool.NewState(cwd),
		&tool.BashTool{Timeout: s.BashTimeout, MaxOutputBytes: s.BashMaxOutput,
			MaxOutput: s.BashMaxOutputFunc, Shell: s.Shell},
		&tool.ReadTool{MaxBytes: s.ReadMaxBytes, MaxLines: s.ReadMaxLines},
		&tool.WriteTool{},
		&tool.EditTool{},
		&tool.GrepTool{MaxResults: s.GrepMaxResults},
		&tool.GlobTool{},
		&tool.LsTool{},
	)
}

// Serve reads requests concurrently so a cancel can reach a running call; EOF
// cancels it too.
func Serve(r io.Reader, w io.Writer, cwd string) error {
	return ServeWith(r, w, os.Stderr, cwd)
}

// ServeWith is Serve with the stderr a relayed MCP server writes to.
func ServeWith(r io.Reader, w io.Writer, errW io.Writer, cwd string) error {
	in := bufio.NewReader(r)
	dec := json.NewDecoder(in)

	var hello Hello
	if err := dec.Decode(&hello); err != nil {
		return fmt.Errorf("reading the opening message: %w", err)
	}
	if hello.MCP != nil {
		// What the decoder read past the opening message belongs to the server.
		rest := io.MultiReader(dec.Buffered(), in)
		if code := mcp.Relay(rest, w, errW, cwd, *hello.MCP); code != 0 {
			return ExitStatus{code}
		}
		return nil
	}

	out := bufio.NewWriter(w)
	defer out.Flush()
	enc := json.NewEncoder(out)
	exec := Tools(hello.Settings, cwd)

	var (
		mu      sync.Mutex
		running string
		stop    context.CancelFunc
	)
	cancelIf := func(match func(id string) bool) {
		mu.Lock()
		defer mu.Unlock()
		if stop != nil && match(running) {
			stop()
		}
	}

	reqs := make(chan tool.Request)
	readErr := make(chan error, 1)
	go func() {
		defer close(reqs)
		for {
			var req tool.Request
			if err := dec.Decode(&req); err != nil {
				cancelIf(func(string) bool { return true })
				if err != io.EOF {
					readErr <- err
				}
				return
			}
			if req.Cancel {
				cancelIf(func(id string) bool { return id == req.CallID })
				continue
			}
			reqs <- req
		}
	}()

	for req := range reqs {
		ctx, cancel := context.WithCancel(context.Background())
		mu.Lock()
		running, stop = req.CallID, cancel
		mu.Unlock()

		res, err := exec.Execute(ctx, req)

		mu.Lock()
		running, stop = "", nil
		mu.Unlock()
		cancel()

		if err != nil {
			res = tool.Errorf("executor failed: %v", err)
		}
		if err := enc.Encode(res); err != nil {
			return err
		}
		if err := out.Flush(); err != nil {
			return err
		}
	}
	select {
	case err := <-readErr:
		return err
	default:
		return nil
	}
}

type Client struct {
	enc *json.Encoder
	dec *json.Decoder
}

func NewClient(w io.Writer, r io.Reader, s Settings) (*Client, error) {
	c := &Client{enc: json.NewEncoder(w), dec: json.NewDecoder(bufio.NewReader(r))}
	if err := c.enc.Encode(Hello{Settings: s}); err != nil {
		return nil, ErrWrite{err}
	}
	return c, nil
}

func (c *Client) Call(ctx context.Context, req tool.Request) (tool.Result, error) {
	if err := c.enc.Encode(req); err != nil {
		return tool.Result{}, ErrWrite{err}
	}
	type reply struct {
		res tool.Result
		err error
	}
	got := make(chan reply, 1)
	go func() {
		var r reply
		r.err = c.dec.Decode(&r.res)
		got <- r
	}()
	select {
	case r := <-got:
		return r.res, r.err
	case <-ctx.Done():
	}
	_ = c.enc.Encode(tool.Request{CallID: req.CallID, Name: req.Name, Cancel: true})
	r := <-got
	return r.res, r.err
}

// ErrWrite: the request never reached the worker, so retrying is safe.
type ErrWrite struct{ Err error }

func (e ErrWrite) Error() string { return e.Err.Error() }
func (e ErrWrite) Unwrap() error { return e.Err }

func IsUnsent(err error) bool {
	var w ErrWrite
	return errors.As(err, &w)
}
