package tool

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"
)

//go:embed bash.txt
var bashDescription string

// Sentinels the wrapper script prints after the user's command so ai-code can
// recover the shell's final state. Chosen to be implausible in real output.
const (
	stateMarker = "__ai_code_state_a37f91c4__"
	envMarker   = "__ai_code_env_a37f91c4__"
)

type BashTool struct {
	Timeout        time.Duration
	MaxOutputBytes int
	Shell          string
}

func (t *BashTool) Name() string        { return "bash" }
func (t *BashTool) Description() string { return bashDescription }

// ReadOnly is false unconditionally. Deciding whether a command mutates
// anything would mean parsing shell grammar, and getting it wrong means
// concurrent writes to the same file. Shell commands are always serialised.
func (t *BashTool) ReadOnly() bool { return false }

func (t *BashTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "command": {"type": "string", "description": "The shell command to run."},
    "timeout_seconds": {"type": "integer", "description": "Override the default timeout for this command."},
    "description": {"type": "string", "description": "A short present-tense description of what this runs, shown to the user."}
  },
  "required": ["command"]
}`)
}

type bashArgs struct {
	Command        string `json:"command"`
	TimeoutSeconds int    `json:"timeout_seconds"`
	Description    string `json:"description"`
}

func (t *BashTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a bashArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	if strings.TrimSpace(a.Command) == "" {
		return Errorf("command is empty.")
	}

	timeout := t.Timeout
	if a.TimeoutSeconds > 0 {
		timeout = time.Duration(a.TimeoutSeconds) * time.Second
	}
	if timeout <= 0 {
		timeout = 120 * time.Second
	}

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	shell := t.Shell
	if shell == "" {
		shell = defaultShell()
	}

	cmd := exec.Command(shell, "-c", wrapCommand(a.Command))
	cmd.Dir = st.Cwd()
	cmd.Env = buildEnv(st)
	setProcessGroup(cmd)

	// A command that reads stdin gets EOF rather than blocking forever on a
	// terminal that is not there.
	devNull, err := os.Open(os.DevNull)
	if err == nil {
		cmd.Stdin = devNull
		defer devNull.Close()
	}

	buf := newBoundedBuffer(t.MaxOutputBytes)
	cmd.Stdout = buf
	cmd.Stderr = buf

	start := time.Now()
	if err := cmd.Start(); err != nil {
		return Errorf("could not start the shell: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	var (
		waitErr   error
		timedOut  bool
		cancelled bool
	)
	select {
	case waitErr = <-done:
	case <-runCtx.Done():
		// SIGTERM first so the command can clean up, then SIGKILL. Both go to
		// the whole process group.
		killProcessGroup(cmd, true)
		select {
		case waitErr = <-done:
		case <-time.After(2 * time.Second):
			killProcessGroup(cmd, false)
			waitErr = <-done
		}
		timedOut = runCtx.Err() == context.DeadlineExceeded && ctx.Err() == nil
		cancelled = ctx.Err() != nil
	}

	elapsed := time.Since(start)
	output, state := splitState(buf.String())

	if state.cwd != "" && state.cwd != st.Cwd() {
		st.SetCwd(state.cwd)
	}
	applyEnvDelta(st, state.env)

	exitCode := cmd.ProcessState.ExitCode()

	// Every path out of here reports output, so every path has to be bounded.
	// The buffer deliberately holds several times the reporting budget so that
	// Truncate has both ends to choose between, which means an early return
	// that skips Truncate does not return a slightly large result -- it
	// returns the entire buffer, eight times the budget. A command that timed
	// out is the likeliest of all to have produced that much.
	truncated, wasTruncated := Truncate(trim(output), t.MaxOutputBytes)
	if buf.overflowed {
		wasTruncated = true
	}
	excerpt := truncated
	if wasTruncated {
		excerpt += truncationNote
	}

	switch {
	case cancelled:
		return Result{
			Content: fmt.Sprintf("Interrupted by user after %s. The command and everything it "+
				"spawned were killed; anything it had already changed is still changed, so "+
				"re-check rather than assuming it completed or did nothing.\n\nOutput before "+
				"it stopped:\n%s", elapsed.Round(time.Millisecond), excerpt),
			IsError:     true,
			Interrupted: true,
			Display:     fmt.Sprintf("Interrupted: %s", oneLine(a.Command)),
		}
	case timedOut:
		return Result{
			Content: fmt.Sprintf("Timed out after %s and was killed, along with everything it spawned.\n\n"+
				"Partial output:\n%s\n\nIf this command is genuinely long-running, redirect it to a "+
				"file and run it in the background, then poll the file.", timeout, excerpt),
			IsError: true,
			Display: fmt.Sprintf("Timed out (%s): %s", timeout, oneLine(a.Command)),
		}
	}

	var b strings.Builder
	if truncated == "" {
		b.WriteString("(no output)")
	} else {
		b.WriteString(truncated)
	}
	if exitCode != 0 {
		fmt.Fprintf(&b, "\n\nExit code: %d", exitCode)
		if waitErr != nil && exitCode == -1 {
			fmt.Fprintf(&b, " (%v)", waitErr)
		}
	}
	if wasTruncated {
		b.WriteString(truncationNote)
	}

	// The "$" marks the line as shell activity. Without it a description like
	// "Read main.go" is indistinguishable from the read tool's own line, and a
	// later "has not been read" error looks like a bug rather than the
	// consequence of having catted the file.
	display := "$ " + oneLine(a.Command)
	if a.Description != "" {
		display = "$ " + a.Description
	}
	if exitCode != 0 {
		display = fmt.Sprintf("%s (exit %d)", display, exitCode)
	}

	return Result{
		Content: b.String(),
		IsError: exitCode != 0,
		Display: display,
		Cwd:     st.Cwd(),
	}
}

func defaultShell() string {
	if runtime.GOOS == "windows" {
		return "powershell"
	}
	if sh := os.Getenv("SHELL"); sh != "" && !strings.HasSuffix(sh, "fish") {
		// fish is excluded because the wrapper below is POSIX shell syntax.
		return sh
	}
	return "/bin/bash"
}

// wrapCommand appends a state epilogue so ai-code can recover the working
// directory and environment the command left behind.
//
// This is what makes `cd` appear to persist across calls without holding a
// long-lived shell open. A persistent shell would give higher fidelity, but one
// hung command poisons every later call in the session, and it cannot be moved
// to a remote executor. Recovering the state after each call trades a little
// fidelity for a failure mode that stays contained.
func wrapCommand(user string) string {
	return user + "\n" +
		"__ai_code_rc=$?\n" +
		"printf '\\n%s\\n' '" + stateMarker + "'\n" +
		"pwd\n" +
		"printf '%s\\n' '" + envMarker + "'\n" +
		"env\n" +
		"exit $__ai_code_rc\n"
}

type shellState struct {
	cwd string
	env map[string]string
}

// splitState separates the command's real output from the epilogue.
func splitState(out string) (string, shellState) {
	idx := strings.LastIndex(out, stateMarker)
	if idx < 0 {
		// The command exited the shell directly, or was killed before the
		// epilogue ran. Output is still valid; state is simply unknown.
		return out, shellState{}
	}
	body := out[:idx]
	rest := out[idx+len(stateMarker):]

	st := shellState{env: map[string]string{}}
	envIdx := strings.Index(rest, envMarker)
	if envIdx < 0 {
		st.cwd = strings.TrimSpace(rest)
		return strings.TrimSuffix(body, "\n"), st
	}
	st.cwd = strings.TrimSpace(rest[:envIdx])
	for _, line := range strings.Split(rest[envIdx+len(envMarker):], "\n") {
		if k, v, ok := strings.Cut(line, "="); ok && k != "" {
			st.env[k] = v
		}
	}
	return strings.TrimSuffix(body, "\n"), st
}

// applyEnvDelta records variables the command exported or changed, so they are
// visible to the next call. Variables ai-code itself sets are excluded, as are the
// ones every process inherits and nobody means to propagate.
func applyEnvDelta(st *State, after map[string]string) {
	if len(after) == 0 {
		return
	}
	base := map[string]string{}
	for _, kv := range os.Environ() {
		if k, v, ok := strings.Cut(kv, "="); ok {
			base[k] = v
		}
	}
	for k, v := range after {
		if volatileEnv[k] {
			continue
		}
		if base[k] != v {
			st.SetEnv(k, v)
		}
	}
}

// volatileEnv lists variables that differ between any two processes and would
// otherwise be recorded as changes on every single call.
var volatileEnv = map[string]bool{
	"_": true, "PWD": true, "OLDPWD": true, "SHLVL": true,
	"__ai_code_rc": true, "RANDOM": true, "SECONDS": true,
}

func buildEnv(st *State) []string {
	env := os.Environ()

	// Non-interactive defaults. Without these, a pager waits for a keypress
	// nobody will press, and git blocks on a credential prompt with no
	// terminal to type into -- both of which look identical to a hang.
	env = append(env,
		"PAGER=cat",
		"GIT_PAGER=cat",
		"GIT_TERMINAL_PROMPT=0",
		"GH_PAGER=cat",
		"DEBIAN_FRONTEND=noninteractive",
		"PYTHONUNBUFFERED=1",
		"NO_COLOR=1",
		"TERM=dumb",
		"AI_CODE=1",
	)
	for k, v := range st.Env() {
		env = append(env, k+"="+v)
	}
	return env
}

// truncationNote tells the model it is looking at an excerpt. Without it a
// clipped result reads as the whole answer, and the model concludes the thing
// it was looking for is not there.
const truncationNote = "\n(output was truncated; rerun with a narrower command, a grep, or `| tail` if you need more)"

func trim(s string) string { return strings.TrimRight(s, "\n \t") }

func oneLine(cmd string) string {
	cmd = strings.TrimSpace(strings.ReplaceAll(cmd, "\n", " ; "))
	if len(cmd) > 90 {
		return cmd[:87] + "..."
	}
	return cmd
}

// ---------------------------------------------------------------------------

// boundedBuffer keeps the head and tail of a stream without ever holding more
// than a bounded amount in memory. A runaway command producing gigabytes of
// progress-bar output must not take the harness down with it.
type boundedBuffer struct {
	mu         sync.Mutex
	head       []byte
	tail       []byte
	limit      int
	overflowed bool
}

func newBoundedBuffer(limit int) *boundedBuffer {
	if limit <= 0 {
		limit = 60000
	}
	// Hold several times the reporting budget so Truncate still has material to
	// work with, but never unbounded.
	return &boundedBuffer{limit: limit * 8}
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// io.Copy treats a short count as an error, so the return value must always
	// be the length of the original slice regardless of how much was retained.
	n := len(p)

	headCap := b.limit / 2
	if len(b.head) < headCap {
		take := min(headCap-len(b.head), len(p))
		b.head = append(b.head, p[:take]...)
		p = p[take:]
		if len(p) == 0 {
			return n, nil
		}
	}

	b.tail = append(b.tail, p...)
	if tailCap := b.limit / 2; len(b.tail) > tailCap {
		b.overflowed = true
		b.tail = b.tail[len(b.tail)-tailCap:]
	}
	return n, nil
}

func (b *boundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.overflowed {
		return string(b.head) + string(b.tail)
	}
	return string(b.head) + "\n... output exceeded the in-memory limit; middle discarded ...\n" + string(b.tail)
}
