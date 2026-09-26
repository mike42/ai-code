//go:build unix

package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func bashTool() *BashTool {
	return &BashTool{Timeout: 20 * time.Second, MaxOutputBytes: 20000, Shell: "/bin/bash"}
}

func runBash(t *testing.T, st *State, cmd string) Result {
	t.Helper()
	return bashTool().Run(context.Background(), st, mustJSON(t, bashArgs{Command: cmd}))
}

func TestBashRunsAndCapturesBothStreams(t *testing.T) {
	st := NewState(t.TempDir())
	res := runBash(t, st, "echo out; echo err >&2")
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.Content)
	}
	for _, want := range []string{"out", "err"} {
		if !strings.Contains(res.Content, want) {
			t.Errorf("output missing %q: %s", want, res.Content)
		}
	}
}

func TestBashWorkingDirectoryPersistsAcrossCalls(t *testing.T) {
	// The whole point of recovering shell state after each call: `cd` appears
	// to persist even though every call is a fresh process.
	root := t.TempDir()
	sub := filepath.Join(root, "nested")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	st := NewState(root)

	if r := runBash(t, st, "cd nested"); r.IsError {
		t.Fatalf("cd failed: %s", r.Content)
	}
	res := runBash(t, st, "pwd")
	if !strings.Contains(res.Content, "nested") {
		t.Errorf("pwd = %q, want the directory from the previous call", res.Content)
	}
	if !strings.HasSuffix(st.Cwd(), "nested") {
		t.Errorf("tracked cwd = %q, want it to end in nested", st.Cwd())
	}
}

func TestBashExportedVariablesPersist(t *testing.T) {
	st := NewState(t.TempDir())
	if r := runBash(t, st, "export MY_TEST_VAR=hello123"); r.IsError {
		t.Fatalf("export failed: %s", r.Content)
	}
	res := runBash(t, st, "echo \"$MY_TEST_VAR\"")
	if !strings.Contains(res.Content, "hello123") {
		t.Errorf("exported variable did not survive: %q", res.Content)
	}
}

func TestBashStateMarkersNeverAppearInOutput(t *testing.T) {
	// The epilogue that recovers cwd and env must be invisible. If it leaks it
	// goes straight into the model's context on every single call.
	st := NewState(t.TempDir())
	res := runBash(t, st, "echo hello")

	for _, marker := range []string{stateMarker, envMarker, "PATH="} {
		if strings.Contains(res.Content, marker) {
			t.Errorf("state epilogue leaked into the output (%q):\n%s", marker, res.Content)
		}
	}
	if strings.TrimSpace(res.Content) != "hello" {
		t.Errorf("output = %q, want exactly \"hello\"", res.Content)
	}
}

func TestBashReportsNonZeroExit(t *testing.T) {
	st := NewState(t.TempDir())
	res := runBash(t, st, "echo failing; exit 3")
	if !res.IsError {
		t.Error("a non-zero exit should be reported as an error result")
	}
	if !strings.Contains(res.Content, "Exit code: 3") {
		t.Errorf("output should state the exit code: %s", res.Content)
	}
	if !strings.Contains(res.Content, "failing") {
		t.Error("output produced before the failure must still be reported")
	}
}

func TestBashTimesOutAndSaysSo(t *testing.T) {
	st := NewState(t.TempDir())
	bt := &BashTool{Timeout: 300 * time.Millisecond, MaxOutputBytes: 10000, Shell: "/bin/bash"}

	start := time.Now()
	res := bt.Run(context.Background(), st, mustJSON(t, bashArgs{Command: "echo starting; sleep 30"}))
	elapsed := time.Since(start)

	if !res.IsError {
		t.Fatal("a timed-out command should be an error result")
	}
	if elapsed > 5*time.Second {
		t.Errorf("took %v to give up on a 300ms timeout", elapsed)
	}
	if !strings.Contains(res.Content, "Timed out") {
		t.Errorf("message should say it timed out: %s", res.Content)
	}
	if !strings.Contains(res.Content, "starting") {
		t.Error("partial output before the timeout should still be reported")
	}
}

// The failure this guards: killing the shell's PID leaves its children running.
// A cancelled `make -j8` that orphans eight compilers keeps holding file
// handles and CPU while the user believes they stopped it.
func TestCancellationKillsTheWholeProcessGroup(t *testing.T) {
	dir := t.TempDir()
	pidFile := filepath.Join(dir, "child.pid")
	st := NewState(dir)

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan Result, 1)
	go func() {
		done <- bashTool().Run(ctx, st, mustJSON(t, bashArgs{
			// A grandchild that outlives a naive PID kill.
			Command: "sh -c 'echo $$ > " + pidFile + "; sleep 60' & sleep 60",
		}))
	}()

	// Wait for the grandchild to record its pid.
	var childPID int
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if b, err := os.ReadFile(pidFile); err == nil {
			if p, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil && p > 0 {
				childPID = p
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if childPID == 0 {
		t.Skip("could not observe the grandchild pid")
	}

	cancel()

	select {
	case res := <-done:
		if !res.IsError {
			t.Error("a cancelled command should be reported as an error result")
		}
		if !strings.Contains(res.Content, "Interrupted") {
			t.Errorf("message should say it was interrupted: %s", res.Content)
		}
		// Being honest that state may have changed matters: a killed command
		// may have written half a file, and "cancelled" alone reads as "nothing
		// happened".
		if !strings.Contains(res.Content, "already changed is still changed") {
			t.Errorf("message should be honest about partial effects: %s", res.Content)
		}
		if !res.Interrupted {
			t.Error("Interrupted should be set so the agent does not wrap this " +
				"message in a second, redundant explanation of the same event")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the command did not return after cancellation")
	}

	// The grandchild must be gone.
	gone := false
	for range 100 {
		if err := syscall.Kill(childPID, 0); err != nil {
			gone = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !gone {
		_ = syscall.Kill(childPID, syscall.SIGKILL)
		t.Errorf("grandchild %d survived cancellation; killing the PID is not enough, "+
			"the whole process group must go", childPID)
	}
}

func TestBashStdinIsClosedSoCommandsDoNotHang(t *testing.T) {
	// A command that reads stdin must get EOF, not block forever on a terminal
	// that is not there. A hang here is indistinguishable from a slow model.
	st := NewState(t.TempDir())
	bt := &BashTool{Timeout: 3 * time.Second, MaxOutputBytes: 10000, Shell: "/bin/bash"}

	start := time.Now()
	res := bt.Run(context.Background(), st, mustJSON(t, bashArgs{Command: "cat"}))
	if time.Since(start) > 2*time.Second {
		t.Errorf("`cat` blocked for %v; stdin should be /dev/null", time.Since(start))
	}
	if res.IsError && strings.Contains(res.Content, "Timed out") {
		t.Error("`cat` hit the timeout instead of receiving EOF")
	}
}

func TestBashDisablesPagers(t *testing.T) {
	st := NewState(t.TempDir())
	res := runBash(t, st, "echo \"PAGER=$PAGER GIT_PAGER=$GIT_PAGER PROMPT=$GIT_TERMINAL_PROMPT\"")
	for _, want := range []string{"PAGER=cat", "GIT_PAGER=cat", "PROMPT=0"} {
		if !strings.Contains(res.Content, want) {
			t.Errorf("expected %s in the environment: %s", want, res.Content)
		}
	}
}

func TestBashTruncatesRunawayOutput(t *testing.T) {
	st := NewState(t.TempDir())
	bt := &BashTool{Timeout: 30 * time.Second, MaxOutputBytes: 2000, Shell: "/bin/bash"}
	res := bt.Run(context.Background(), st, mustJSON(t, bashArgs{
		Command: "for i in $(seq 1 20000); do echo \"line $i of noise\"; done",
	}))
	if res.IsError {
		t.Fatalf("unexpected error: %s", res.Content)
	}
	if len(res.Content) > 6000 {
		t.Errorf("output is %d bytes despite a 2000 byte budget", len(res.Content))
	}
	if !strings.Contains(res.Content, "truncated") && !strings.Contains(res.Content, "elided") {
		t.Error("truncation should be stated so the model knows it has an excerpt")
	}
	// The tail is the part that matters; it must survive.
	if !strings.Contains(res.Content, "line 20000") {
		t.Error("the end of the output was lost; errors live at the end")
	}
}

func TestBashEmptyCommandIsRejected(t *testing.T) {
	st := NewState(t.TempDir())
	if res := runBash(t, st, "   "); !res.IsError {
		t.Error("an empty command should be rejected")
	}
}

func TestBashMalformedArgs(t *testing.T) {
	st := NewState(t.TempDir())
	res := bashTool().Run(context.Background(), st, json.RawMessage(`{"command":`))
	if !res.IsError || !strings.Contains(res.Content, "JSON") {
		t.Errorf("expected recoverable JSON feedback, got: %s", res.Content)
	}
}

// A command that times out must not put more into the context than one that
// finishes.
//
// It used to put eight times more. The timed-out and interrupted paths
// returned before Truncate was reached, and the only thing left bounding them
// was the in-memory buffer -- which is deliberately sized at several times the
// reporting budget precisely so that Truncate has both ends to choose
// between. A timeout is the case most likely to have filled it.
func TestTimedOutCommandIsTruncatedLikeAnyOther(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX shell syntax")
	}
	const budget = 20000
	bt := &BashTool{Timeout: time.Second, MaxOutputBytes: budget, Shell: "/bin/bash"}
	st := NewState(t.TempDir())

	spew := `for i in $(seq 1 20000); do echo "record 000000 at offset 000000"; done`
	run := func(cmd string) Result {
		raw, _ := json.Marshal(map[string]any{"command": cmd})
		return bt.Run(context.Background(), st, raw)
	}

	clean := run(spew)
	timedOut := run(spew + `; sleep 30`)

	if !strings.Contains(timedOut.Content, "Timed out") {
		t.Fatalf("the command did not time out, so this proves nothing: %.120s", timedOut.Content)
	}
	// Generous headroom for the surrounding prose, but nowhere near the 8x
	// buffer the bug let through.
	if max := budget * 2; len(timedOut.Content) > max {
		t.Errorf("timed-out result is %d bytes, want under %d (clean exit was %d)",
			len(timedOut.Content), max, len(clean.Content))
	}
	if !strings.Contains(timedOut.Content, "truncated") {
		t.Error("the model was not told the partial output is an excerpt")
	}
}
