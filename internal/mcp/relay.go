package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"

	"ai-code/internal/tool"
)

// Launch is what a worker in the target environment is asked to run or
// reach. The argument list travels as a list in the worker's opening
// message, so it never passes through a shell, remote or local.
type Launch struct {
	Binary  string            `json:"binary,omitempty"`
	Args    []string          `json:"args,omitempty"`
	URL     string            `json:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
}

// LaunchFor is the Launch an entry describes.
func LaunchFor(e Entry) Launch {
	return Launch{Binary: e.Binary, Args: e.Args, URL: e.URL, Headers: e.Headers}
}

// Relay is the worker's side: it runs or reaches l from here and carries MCP
// messages between it and r/w, until r ends. It returns the exit status the
// worker should end with, which is the server's own.
func Relay(r io.Reader, w io.Writer, errW io.Writer, cwd string, l Launch) int {
	if l.URL != "" {
		return relayHTTP(r, w, errW, l)
	}
	return relayProcess(r, w, errW, cwd, l)
}

// relayProcess execs the server with its argument list and joins its stdio
// to the worker's. When the worker's stdin ends -- ai-code stopped the
// server, or went away -- the server's stdin is closed and its process group
// is asked to exit.
func relayProcess(r io.Reader, w io.Writer, errW io.Writer, cwd string, l Launch) int {
	path, err := exec.LookPath(l.Binary)
	if err != nil {
		fmt.Fprintf(errW, "%s: %v\n", l.Binary, unwrapLookPath(err))
		return 127
	}
	cmd := exec.Command(path, l.Args...)
	cmd.Dir = cwd
	cmd.Env = os.Environ()
	cmd.Stdout = w
	cmd.Stderr = errW
	tool.OwnProcessGroup(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		fmt.Fprintf(errW, "%s: %v\n", l.Binary, err)
		return 1
	}
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(errW, "%s: %v\n", l.Binary, unwrapLookPath(err))
		return 127
	}
	exited := make(chan struct{})
	go func() {
		_, _ = io.Copy(stdin, r)
		_ = stdin.Close()
		select {
		case <-exited:
		default:
			tool.TerminateProcessGroup(cmd)
		}
	}()
	err = cmd.Wait()
	close(exited)
	var ee *exec.ExitError
	switch {
	case err == nil:
		return 0
	case errors.As(err, &ee) && ee.ExitCode() >= 0:
		return ee.ExitCode()
	default:
		fmt.Fprintf(errW, "%s: %v\n", l.Binary, err)
		return 1
	}
}

// bridge turns Streamable HTTP back into newline-delimited messages on w.
type bridge struct {
	t *httpTransport

	mu      sync.Mutex
	w       io.Writer
	pending map[int64]bool
	initID  int64
}

func (b *bridge) write(raw []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, _ = b.w.Write(append(bytes.TrimSpace(raw), '\n'))
}

func (b *bridge) deliver(raw []byte) {
	var m message
	if json.Unmarshal(raw, &m) == nil && m.Method == "" && len(m.ID) > 0 {
		var id int64
		if json.Unmarshal(m.ID, &id) == nil {
			b.mu.Lock()
			delete(b.pending, id)
			isInit := id == b.initID
			b.mu.Unlock()
			if isInit {
				var res struct {
					ProtocolVersion string `json:"protocolVersion"`
				}
				if json.Unmarshal(m.Result, &res) == nil && res.ProtocolVersion != "" {
					b.t.setVersion(res.ProtocolVersion)
				}
			}
		}
	}
	b.write(raw)
}

// failRequest answers a request the server never answered, so the client is
// not left waiting; a request already answered is left alone.
func (b *bridge) failRequest(id int64, err error) {
	b.mu.Lock()
	open := b.pending[id]
	delete(b.pending, id)
	b.mu.Unlock()
	if !open {
		return
	}
	reply, _ := json.Marshal(message{JSONRPC: "2.0", ID: json.RawMessage(fmt.Sprint(id)),
		Error: &rpcError{Code: -32000, Message: err.Error()}})
	b.write(reply)
}

func relayHTTP(r io.Reader, w io.Writer, errW io.Writer, l Launch) int {
	b := &bridge{w: w, pending: map[int64]bool{}, initID: -1}
	b.t = newHTTPTransport(b, l.URL, l.Headers)
	defer b.t.close()
	br := bufio.NewReader(r)
	for {
		line, err := br.ReadBytes('\n')
		if line = bytes.TrimSpace(line); len(line) > 0 {
			b.forward(line, errW)
		}
		if err != nil {
			return 0
		}
	}
}

func (b *bridge) forward(line []byte, errW io.Writer) {
	id, isRequest := requestID(line)
	if isRequest {
		var m struct {
			Method string `json:"method"`
		}
		_ = json.Unmarshal(line, &m)
		b.mu.Lock()
		b.pending[id] = true
		if m.Method == "initialize" {
			b.initID = id
		}
		b.mu.Unlock()
	}
	if err := b.t.send(context.Background(), line); err != nil {
		if isRequest {
			b.failRequest(id, err)
		} else {
			fmt.Fprintf(errW, "%s: %v\n", b.t.url, err)
		}
	}
}
