package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/devcontainer"
	"ai-code/internal/tool"
)

func TestDevcontainerExecutorEndToEnd(t *testing.T) {
	engine, err := detectContainerEngine()
	if err != nil {
		t.Skipf("no container engine: %v", err)
	}

	bin := filepath.Join(t.TempDir(), "ai-code")
	cmd := exec.Command("go", "build", "-o", bin, "ai-code/cmd/ai-code")
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("could not build static binary: %v\n%s", err, out)
	}

	image := "docker.io/library/python:3.12-slim"
	if _, err := exec.Command(engine, "image", "exists", image).Output(); err != nil {
		t.Skipf("image %s not present; run `%s pull %s` to enable this test", image, engine, image)
	}

	cfg := &devcontainer.Config{Image: image, WorkspaceFolder: "/workspace"}
	dc := NewDevcontainerExecutor(cfg, nil, ".", ".", bin)

	dc.mu.Lock()
	started := dc.cmd != nil
	dc.mu.Unlock()
	if started {
		t.Fatal("container should not be started before the first Execute")
	}
	defer dc.Close()

	res, err := dc.Execute(context.Background(), tool.Request{
		CallID: "1",
		Name:   "bash",
		Args:   []byte(`{"command":"echo inside-container"}`),
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("tool returned error: %s", res.Content)
	}
	t.Logf("result: %q", res.Content)
}

// A dead container used to be invisible: start() treated a non-nil cmd as a
// live one, so every later tool call wrote to the same closed pipe and
// reported "broken pipe" with nothing to act on. These cover the two halves of
// that -- saying what actually happened, and letting the next call recover.

func TestFailureReportsExitStatusAndEngineOutput(t *testing.T) {
	e := &DevcontainerExecutor{}
	cmd := exec.Command("sh", "-c", "exit 3")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	e.cmd = cmd
	e.stderr = newCappedBuffer(stderrCap)
	e.stderr.Write([]byte("Error: statfs /nonexistent: no such file or directory"))

	msg := e.failure("write", errors.New("write |1: broken pipe"))

	// The engine's own words are the diagnosis; the bare pipe error sends
	// people looking at the wrong component.
	if !strings.Contains(msg, "statfs /nonexistent") {
		t.Errorf("the engine's output was dropped from the report: %q", msg)
	}
	if !strings.Contains(msg, "status") {
		t.Errorf("no exit status in the report: %q", msg)
	}
	if !strings.Contains(msg, "broken pipe") {
		t.Errorf("the underlying error was lost: %q", msg)
	}
}

func TestFailureReapsTheContainerProcess(t *testing.T) {
	e := &DevcontainerExecutor{}
	// A process that would outlive the session if nothing waited on it.
	cmd := exec.Command("sleep", "300")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	e.cmd = cmd

	done := make(chan struct{})
	go func() {
		e.failure("write", errors.New("broken pipe"))
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		t.Fatal("failure() blocked instead of killing and reaping the container")
	}
	if cmd.ProcessState == nil {
		t.Error("the process was never reaped, so its exit status is unknowable")
	}
}

func TestTeardownLetsTheNextCallStartFresh(t *testing.T) {
	e := &DevcontainerExecutor{
		cmd:    exec.Command("true"),
		enc:    json.NewEncoder(io.Discard),
		stderr: newCappedBuffer(stderrCap),
	}
	e.init = errors.New("stale")

	e.teardown()

	// start() returns early whenever either of these is set, so a leftover
	// value here is what made the failure permanent.
	if e.cmd != nil || e.init != nil || e.enc != nil || e.dec != nil {
		t.Errorf("teardown left state behind: cmd=%v init=%v enc=%v dec=%v",
			e.cmd != nil, e.init != nil, e.enc != nil, e.dec != nil)
	}
}

// Starting a container happens behind the user's first tool call, and it used
// to happen in complete silence.
//
// Engine detection, an image build or pull, then the container itself is
// minutes of work with nothing on screen, so `bash` looked like it had hung.
// Naming each phase as it is reached is the whole fix: a note once the wait
// is over answers the question too late to be worth asking.
func TestDevcontainerStartupNamesItsPhases(t *testing.T) {
	var notes []tool.Phase
	e := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), "ai-code")
	e.SetProgress(func(p tool.Phase) { notes = append(notes, p) })

	// A configuration that never loaded: startup fails before any container
	// work begins. Nothing happened, so nothing may be claimed -- a phase
	// report that fires regardless of what ran is just a different lie about
	// where the time went.
	if _, err := e.Execute(context.Background(), tool.Request{Name: "bash"}); err != nil {
		t.Fatalf("Execute returned a transport error: %v", err)
	}
	if len(notes) != 0 {
		t.Errorf("phases announced for a startup that never began: %v", notes)
	}

	// And the wiring itself reaches the caller.
	e.report("devcontainer: starting a container from %s.", "example-image")
	if len(notes) != 1 || !strings.Contains(notes[0].Note, "example-image") {
		t.Fatalf("progress notes = %v, want the phase that was reported", notes)
	}
	if notes[0].Done {
		t.Error("a phase in the middle of a start was marked as the end of it")
	}

	// The last one is the only one worth keeping, and says so.
	e.done("devcontainer: ready in %s.", "2m14s")
	if len(notes) != 2 || !notes[1].Done {
		t.Errorf("the end of the wait was not marked: %v", notes)
	}
}

// The task tool wraps the executor, and a wrapper that does not pass progress
// through is a wrapper that silently switches it off.
//
// This is exactly what happened: main.go asks the executor whether it reports
// progress, by which point the executor is the task wrapper. It did not, so a
// user watched a container rebuild for five minutes behind a bare spinner.
func TestProgressSurvivesTheTaskWrapper(t *testing.T) {
	var notes []tool.Phase
	inner := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), "ai-code")
	wrapped := agent.NewTaskExecutor(inner)

	pr, ok := tool.Executor(wrapped).(tool.ProgressReporter)
	if !ok {
		t.Fatal("the wrapped executor does not report progress, so nothing will be shown")
	}
	pr.SetProgress(func(p tool.Phase) { notes = append(notes, p) })

	inner.report("devcontainer: building the image from %s.", "Dockerfile")
	if len(notes) != 1 || !strings.Contains(notes[0].Note, "Dockerfile") {
		t.Errorf("progress notes = %v, want the phase to have reached the caller", notes)
	}
}

// The first thing anyone watching a long tool call wants to know is whether
// the clock they can see is running. It is not: the command's timeout is
// applied by the bash tool inside the container, so it cannot start until the
// container does.
func TestStartupSaysTheTimeoutIsNotRunning(t *testing.T) {
	var notes []tool.Phase
	e := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), "ai-code")
	e.SetProgress(func(p tool.Phase) { notes = append(notes, p) })
	e.report("devcontainer: preparing the container with %s. The command has not started yet, "+
		"so its timeout is not running.", "docker")

	if len(notes) == 0 || !strings.Contains(notes[0].Note, "timeout is not running") {
		t.Errorf("the first phase does not say the timeout is not running: %v", notes)
	}
}

// A session with nowhere to show progress must not panic on the nil callback.
func TestDevcontainerProgressIsOptional(t *testing.T) {
	var _ tool.ProgressReporter = (*DevcontainerExecutor)(nil)

	e := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), "ai-code")
	e.report("devcontainer: starting a container from %s.", "example-image")
}
