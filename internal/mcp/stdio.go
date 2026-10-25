package mcp

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"ai-code/internal/tool"
)

// stdioTransport speaks newline-delimited JSON over a process's stdin and
// stdout: the server itself on the host runtime, or the worker relaying to
// it inside the target environment.
type stdioTransport struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
	wmu   sync.Mutex
	tail  *lastLine
	// direct: cmd is the server, so close may signal its process group. A
	// relaying worker ends its server itself when its stdin closes.
	direct bool
	// exited is closed once the process has been reaped.
	exited chan struct{}
}

// resolveBinary finds binary on this machine, refusing what would run through
// a shell.
func resolveBinary(binary string) (string, error) {
	path, err := exec.LookPath(binary)
	if err != nil {
		return "", fmt.Errorf("%s: %w", binary, unwrapLookPath(err))
	}
	if runtime.GOOS == "windows" {
		switch strings.ToLower(filepath.Ext(path)) {
		case ".bat", ".cmd":
			return "", fmt.Errorf("%s is a batch file, which Windows runs through cmd.exe; ai-code runs MCP servers without a shell", path)
		}
	}
	return path, nil
}

func unwrapLookPath(err error) error {
	var e *exec.Error
	if errors.As(err, &e) {
		if errors.Is(e.Err, exec.ErrNotFound) || errors.Is(e.Err, fs.ErrNotExist) {
			return errors.New("not found")
		}
		return e.Err
	}
	return err
}

// startCmd starts cmd, writes preface to it first if there is one, and reads
// its stdout as MCP messages for c.
func startCmd(c *Conn, cmd *exec.Cmd, preface []byte, direct bool) error {
	tool.OwnProcessGroup(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	t := &stdioTransport{cmd: cmd, stdin: stdin, tail: &lastLine{}, direct: direct, exited: make(chan struct{})}
	cmd.Stderr = t.tail
	if err := cmd.Start(); err != nil {
		return err
	}
	c.t = t
	go t.read(c, stdout)
	if len(preface) > 0 {
		if err := t.send(context.Background(), bytes.TrimRight(preface, "\n")); err != nil {
			return err
		}
	}
	return nil
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
// itself; a server run directly also has its own process group terminated.
// Only the process started here is signalled.
func (t *stdioTransport) close() error {
	t.wmu.Lock()
	_ = t.stdin.Close()
	t.wmu.Unlock()
	if t.direct {
		select {
		case <-t.exited:
		default:
			tool.TerminateProcessGroup(t.cmd)
		}
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
