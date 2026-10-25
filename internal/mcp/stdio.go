package mcp

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"ai-code/internal/tool"
)

// stdioTransport runs the server as a child process and speaks
// newline-delimited JSON over its stdin and stdout.
type stdioTransport struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
	wmu   sync.Mutex
	tail  *lastLine
	// exited is closed once the process has been reaped.
	exited chan struct{}
}

// startProcess execs binary with args, never through a shell.
func startProcess(c *Conn, binary string, args []string, env []string) (*stdioTransport, error) {
	path, err := exec.LookPath(binary)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", binary, unwrapLookPath(err))
	}
	if runtime.GOOS == "windows" {
		switch strings.ToLower(filepath.Ext(path)) {
		case ".bat", ".cmd":
			return nil, fmt.Errorf("%s is a batch file, which Windows runs through cmd.exe; ai-code runs MCP servers without a shell", path)
		}
	}
	cmd := exec.Command(path, args...)
	cmd.Env = env
	tool.OwnProcessGroup(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	t := &stdioTransport{cmd: cmd, stdin: stdin, tail: &lastLine{}, exited: make(chan struct{})}
	cmd.Stderr = t.tail
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("%s: %w", binary, err)
	}
	c.t = t
	go t.read(c, stdout)
	return t, nil
}

func unwrapLookPath(err error) error {
	var e *exec.Error
	if errors.As(err, &e) {
		if errors.Is(e.Err, exec.ErrNotFound) {
			return errors.New("not found")
		}
		return e.Err
	}
	return err
}

func (t *stdioTransport) read(c *Conn, stdout io.Reader) {
	r := bufio.NewReader(stdout)
	for {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); len(line) > 0 {
			c.deliver(line)
		}
		if err != nil {
			break
		}
	}
	werr := t.cmd.Wait()
	close(t.exited)
	c.fail(t.exitError(werr))
}

func (t *stdioTransport) exitError(werr error) error {
	msg := "exited"
	var ee *exec.ExitError
	switch {
	case errors.As(werr, &ee) && ee.ExitCode() >= 0:
		msg = fmt.Sprintf("exited (status %d)", ee.ExitCode())
	case errors.As(werr, &ee):
		msg = "exited (" + ee.String() + ")"
	case werr == nil:
		msg = "exited (status 0)"
	}
	if l := t.tail.String(); l != "" {
		msg += ": " + l
	}
	return errors.New(msg)
}

func (t *stdioTransport) send(ctx context.Context, msg []byte) error {
	t.wmu.Lock()
	defer t.wmu.Unlock()
	if _, err := t.stdin.Write(append(msg, '\n')); err != nil {
		select {
		case <-t.exited:
			return errors.New("the server has exited")
		default:
		}
		return err
	}
	return nil
}

// close ends the server: stdin closed so a well-behaved server exits by
// itself, and its process group terminated for one that does not. Only the
// process started here is signalled.
func (t *stdioTransport) close() error {
	t.wmu.Lock()
	_ = t.stdin.Close()
	t.wmu.Unlock()
	select {
	case <-t.exited:
	default:
		tool.TerminateProcessGroup(t.cmd)
	}
	return nil
}

// lastLine keeps the last non-empty line written to it: the server's stderr,
// where the reason for a failed start usually is.
type lastLine struct {
	mu   sync.Mutex
	part []byte
	last string
}

func (l *lastLine) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.part = append(l.part, p...)
	for {
		i := bytes.IndexByte(l.part, '\n')
		if i < 0 {
			break
		}
		if s := strings.TrimSpace(string(l.part[:i])); s != "" {
			l.last = s
		}
		l.part = l.part[i+1:]
	}
	return len(p), nil
}

func (l *lastLine) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if s := strings.TrimSpace(string(l.part)); s != "" {
		return s
	}
	return l.last
}
