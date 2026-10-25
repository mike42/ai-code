package main

import (
	"context"
	"errors"
	"io"
	"os/exec"
	goruntime "runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/devcontainer"
	"ai-code/internal/tool"
	"ai-code/internal/worker"
	"ai-code/internal/workerbin"
)

func TestDevcontainerExecutorEndToEnd(t *testing.T) {
	engine, err := detectContainerEngine()
	if err != nil {
		t.Skipf("no container engine: %v", err)
	}

	requireWorker(t)

	image := "docker.io/library/python:3.12-slim"
	if _, err := exec.Command(engine, "image", "exists", image).Output(); err != nil {
		t.Skipf("image %s not present; run `%s pull %s` to enable this test", image, engine, image)
	}

	cfg := &devcontainer.Config{Image: image, WorkspaceFolder: "/workspace"}
	project := t.TempDir()
	removeOwn(t, engine, project)
	dc := NewDevcontainerExecutor(cfg, nil, project, project, noSettings)

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

// A dead container must report the engine's own output rather than only a pipe
// error, and the next call must be able to start fresh.
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

	// The engine's message is the diagnosis; the bare pipe error points at
	// the wrong component.
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
	// A process that would outlive the session if nothing reaped it.
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
		client: &worker.Client{},
		stderr: newCappedBuffer(stderrCap),
	}
	e.init = errors.New("stale")

	e.teardown()

	// start() returns early whenever any of these is set, so teardown must
	// clear them all.
	if e.cmd != nil || e.init != nil || e.client != nil {
		t.Errorf("teardown left state behind: cmd=%v init=%v client=%v",
			e.cmd != nil, e.init != nil, e.client != nil)
	}
}

// Engine detection, build or pull, then the container is minutes of work, so
// each phase is named as it starts.
func TestDevcontainerStartupNamesItsPhases(t *testing.T) {
	var notes []tool.Phase
	e := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), noSettings)
	e.SetProgress(func(p tool.Phase) { notes = append(notes, p) })

	// Configuration that never loaded means no container work began, so no
	// phases may be reported, and the run ends as unavailable.
	var unavailable *tool.Unavailable
	if _, err := e.Execute(context.Background(), tool.Request{Name: "bash"}); !errors.As(err, &unavailable) {
		t.Fatalf("Execute returned %v, want tool.Unavailable", err)
	}
	if len(notes) != 0 {
		t.Errorf("phases announced for a startup that never began: %v", notes)
	}

	e.report("devcontainer: starting a container from %s.", "example-image")
	if len(notes) != 1 || !strings.Contains(notes[0].Note, "example-image") {
		t.Fatalf("progress notes = %v, want the phase that was reported", notes)
	}
	if notes[0].Done {
		t.Error("a phase in the middle of a start was marked as the end of it")
	}

	e.done("devcontainer: ready in %s.", "2m14s")
	if len(notes) != 2 || !notes[1].Done {
		t.Errorf("the end of the wait was not marked: %v", notes)
	}
}

// The task tool wraps the executor, and main.go asks the wrapped value whether
// it reports progress, so the wrapper must pass SetProgress through.
func TestProgressSurvivesTheTaskWrapper(t *testing.T) {
	var notes []tool.Phase
	inner := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), noSettings)
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

// The command's timeout is applied by the bash tool inside the container, so
// it cannot run while the container is still starting.
func TestStartupSaysTheTimeoutIsNotRunning(t *testing.T) {
	var notes []tool.Phase
	e := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), noSettings)
	e.SetProgress(func(p tool.Phase) { notes = append(notes, p) })
	e.report("devcontainer: preparing the container with %s. The command has not started yet, "+
		"so its timeout is not running.", "docker")

	if len(notes) == 0 || !strings.Contains(notes[0].Note, "timeout is not running") {
		t.Errorf("the first phase does not say the timeout is not running: %v", notes)
	}
}

func TestDevcontainerProgressIsOptional(t *testing.T) {
	var _ tool.ProgressReporter = (*DevcontainerExecutor)(nil)

	e := NewDevcontainerExecutor(nil, nil, t.TempDir(), t.TempDir(), noSettings)
	e.report("devcontainer: starting a container from %s.", "example-image")
}

// requireWorker skips a test that starts a container when this build carries
// no worker for one. Workers are embedded by build.sh; a plain go test of a
// fresh checkout has none.
func requireWorker(t *testing.T) {
	t.Helper()
	if _, err := workerbin.For("linux", goruntime.GOARCH); err != nil {
		t.Skipf("%v (run ./build.sh once to embed them)", err)
	}
}

func keptTestSetup(t *testing.T) (engine, image string) {
	t.Helper()
	engine, err := detectContainerEngine()
	if err != nil {
		t.Skipf("no container engine: %v", err)
	}
	requireWorker(t)
	image = "docker.io/library/python:3.12-slim"
	if _, err := exec.Command(engine, "image", "exists", image).Output(); err != nil {
		t.Skipf("image %s not present; run `%s pull %s` to enable this test", image, engine, image)
	}
	return engine, image
}

// removeOwn removes the one container this test's project maps to, by name.
func removeOwn(t *testing.T, engine, project string) {
	t.Cleanup(func() { exec.Command(engine, "rm", "-f", containerName(project)).Run() })
}

func bash(t *testing.T, e *DevcontainerExecutor, cmd string) string {
	t.Helper()
	res, err := e.Execute(context.Background(), tool.Request{CallID: "1", Name: "bash",
		Args: []byte(`{"command":` + strconv.Quote(cmd) + `}`)})
	if err != nil {
		t.Fatalf("Execute(%q): %v", cmd, err)
	}
	return res.Content
}

// What a session installs outside the workspace is still there in the next
// session; a changed configuration starts over in a new container.
func TestContainerIsKeptBetweenSessions(t *testing.T) {
	engine, image := keptTestSetup(t)
	project := t.TempDir()
	removeOwn(t, engine, project)
	cfg := func(env string) *devcontainer.Config {
		return &devcontainer.Config{Image: image, WorkspaceFolder: "/workspace", ContainerEnv: map[string]string{"V": env}}
	}

	first := NewDevcontainerExecutor(cfg("1"), nil, project, project, noSettings)
	bash(t, first, "echo installed > /opt/marker")
	first.Close()

	second := NewDevcontainerExecutor(cfg("1"), nil, project, project, noSettings)
	if out := bash(t, second, "cat /opt/marker 2>&1 || echo missing"); !strings.Contains(out, "installed") {
		t.Fatalf("the next session lost what the last one put in /opt: %q", out)
	}
	second.Close()

	changed := NewDevcontainerExecutor(cfg("2"), nil, project, project, noSettings)
	defer changed.Close()
	if out := bash(t, changed, "cat /opt/marker 2>&1 || echo missing"); !strings.Contains(out, "missing") {
		t.Fatalf("a changed configuration reused the old container: %q", out)
	}
}

// A container that happens to have the name ai-code would use, but that
// ai-code did not make, is refused rather than replaced.
func TestForeignContainerWithTheSameNameIsLeftAlone(t *testing.T) {
	engine, image := keptTestSetup(t)
	project := t.TempDir()
	removeOwn(t, engine, project)
	if out, err := exec.Command(engine, "create", "--name", containerName(project), image, "true").CombinedOutput(); err != nil {
		t.Fatalf("creating the stand-in container: %v %s", err, out)
	}
	e := NewDevcontainerExecutor(&devcontainer.Config{Image: image, WorkspaceFolder: "/workspace"}, nil, project, project, noSettings)
	defer e.Close()
	_, err := e.Execute(context.Background(), tool.Request{CallID: "1", Name: "bash", Args: []byte(`{"command":"true"}`)})
	var unavailable *tool.Unavailable
	if !errors.As(err, &unavailable) || !strings.Contains(err.Error(), "not made by ai-code") {
		t.Fatalf("Execute = %v, want a refusal naming the foreign container", err)
	}
	if _, _, exists := inspectKept(engine, containerName(project)); !exists {
		t.Fatal("the foreign container was removed")
	}
}

// A note that is not the last phase stays on screen until a later one replaces
// it, and it hides the streaming reply and the thinking line while it does. So
// a call must never return with one still showing.
func TestACallLeavesNoUnfinishedNoteBehind(t *testing.T) {
	toWorker, workerIn := io.Pipe()
	workerOut, fromWorker := io.Pipe()
	go func() { _ = worker.Serve(toWorker, fromWorker, t.TempDir()) }()
	defer workerIn.Close()

	client, err := worker.NewClient(workerIn, workerOut, worker.Settings{})
	if err != nil {
		t.Fatalf("starting the in-process worker: %v", err)
	}
	var phases []tool.Phase
	e := &DevcontainerExecutor{cmd: exec.Command("true"), client: client, stderr: newCappedBuffer(stderrCap)}
	e.SetProgress(func(p tool.Phase) { phases = append(phases, p) })

	for i := range 2 {
		res, err := e.Execute(context.Background(), tool.Request{
			CallID: strconv.Itoa(i), Name: "bash", Args: []byte(`{"command":"echo hi"}`),
		})
		if err != nil || res.IsError {
			t.Fatalf("call %d: %v %s", i, err, res.Content)
		}
		if n := len(phases); n > 0 && !phases[n-1].Done {
			t.Fatalf("call %d returned with %q still showing", i, phases[n-1].Note)
		}
	}
}
