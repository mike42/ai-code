package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"ai-code/internal/devcontainer"
	"ai-code/internal/tool"
	"ai-code/internal/worker"
	"ai-code/internal/workerbin"
)

// DevcontainerExecutor runs tools inside the project's devcontainer, running
// the worker for the image's platform there over an attached pipe. The
// container starts on the first Execute call, not at startup.
type DevcontainerExecutor struct {
	// cfg is the parsed devcontainer.json.
	cfg *devcontainer.Config
	// cfgErr is a devcontainer.json that could not be parsed; reported when the
	// container is needed.
	cfgErr error
	// project is the host directory mounted into the container.
	project string
	// settings are sent to every worker; catalog advertises the same tools
	// before the container exists.
	settings worker.Settings
	catalog  *tool.LocalExecutor
	// workerPath is the worker on this machine, mounted into the container.
	workerPath string
	// workdir is the container path the daemon starts in.
	workdir string
	// progress names each startup phase as it is reached.
	progress tool.Progress
	// ready guards the one-off "ready" note; teardown clears it.
	ready bool

	// host names the executor that owns the container on a fork; a fork runs a
	// second daemon inside it, so agents share a filesystem.
	host *DevcontainerExecutor

	mu     sync.Mutex
	cmd    *exec.Cmd
	client *worker.Client
	init   error
	// answered: a worker that fails before its first answer is broken.
	answered bool
	// cid is the container this executor started, from --cidfile; engine is the
	// command that started it, and forks exec into cid.
	cid    string
	engine string
	// stderr holds the engine's captured output, for a failure report.
	stderr *cappedBuffer
}

// stderrCap bounds the podman startup output kept for a failure report.
const stderrCap = 64 * 1024

// cappedBuffer keeps only the tail of what is written to it.
type cappedBuffer struct {
	buf []byte
}

func newCappedBuffer(cap int) *cappedBuffer {
	return &cappedBuffer{buf: make([]byte, 0, cap)}
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	b.buf = append(b.buf, p...)
	if len(b.buf) > stderrCap {
		b.buf = append([]byte(nil), b.buf[len(b.buf)-stderrCap:]...)
	}
	return len(p), nil
}

func (b *cappedBuffer) String() string { return string(b.buf) }

// NewDevcontainerExecutor wires up a lazily-started devcontainer executor.
// project is the mount source; cwd is the launch directory.
func NewDevcontainerExecutor(cfg *devcontainer.Config, cfgErr error, project, cwd string, settings worker.Settings) *DevcontainerExecutor {
	e := &DevcontainerExecutor{cfg: cfg, cfgErr: cfgErr, project: project,
		settings: settings, catalog: worker.Tools(settings, "")}
	if cfg != nil {
		e.workdir = mountCwd(project, cwd, cfg.WorkspaceFolder)
	} else {
		e.workdir = mountCwd(project, cwd, "")
	}
	return e
}

// SetProgress implements tool.ProgressReporter.
func (e *DevcontainerExecutor) SetProgress(p tool.Progress) { e.progress = p }

// report names a startup phase, if anything is listening.
func (e *DevcontainerExecutor) report(format string, args ...any) {
	if e.progress != nil {
		e.progress(tool.Phase{Note: fmt.Sprintf(format, args...)})
	}
}

// done ends the wait and leaves one line behind explaining it.
func (e *DevcontainerExecutor) done(format string, args ...any) {
	if e.progress != nil {
		e.progress(tool.Phase{Note: fmt.Sprintf(format, args...), Done: true})
	}
}

// compactDuration formats an elapsed time for someone reading it once.
func compactDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.0fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}

// mountCwd maps the launch directory into the container's mount layout; a cwd
// outside the project falls back to the mount root.
func mountCwd(project, cwd, workspaceFolder string) string {
	if workspaceFolder == "" {
		workspaceFolder = "/workspace"
	}
	rel, err := filepath.Rel(project, cwd)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return workspaceFolder
	}
	return path.Join(workspaceFolder, filepath.ToSlash(rel))
}

// start brings the container up on first use, called only from Execute. The
// context is the turn's, so Ctrl-C during an image build stops the build; the
// container itself outlives the call.
func (e *DevcontainerExecutor) start(ctx context.Context) error {
	if e.init != nil || e.cmd != nil {
		return e.init
	}
	// A fork has no container of its own; it attaches a daemon to its host's.
	if e.host != nil {
		return e.startForked(ctx)
	}
	started := time.Now()

	if e.cfgErr != nil {
		e.init = fmt.Errorf("this project's devcontainer.json could not be read: %w", e.cfgErr)
		return e.init
	}
	if e.cfg == nil {
		e.init = fmt.Errorf("no devcontainer configuration was loaded")
		return e.init
	}

	engine, err := detectContainerEngine()
	if err != nil {
		e.init = err
		return err
	}
	// The command's timeout starts once the container is up, in the bash tool inside it.
	e.report("devcontainer: preparing the container with %s. The command has not started yet, "+
		"so its timeout is not running.", engine)

	image, err := e.resolveImage(ctx, engine)
	if err != nil {
		// An interrupted build is not a broken configuration, so do not cache it.
		if ctx.Err() != nil {
			return err
		}
		e.init = err
		return err
	}

	if e.workerPath, err = workerForImage(engine, image); err != nil {
		e.init = err
		return err
	}
	argv := e.cfg.RunArgv(e.project, e.workdir, e.workerPath, binInContainer)

	// --cidfile after RunArgv binds the id to the container actually created;
	// the name is fresh because the engine refuses an existing file.
	cidfile := filepath.Join(os.TempDir(), fmt.Sprintf("ai-code-cid-%d-%d", os.Getpid(), time.Now().UnixNano()))
	os.Remove(cidfile)
	argv = append(argv, "--cidfile", cidfile)
	argv = append(argv, image)
	e.engine = engine

	// The engine pulls a missing image during run, so only report a real pull.
	if !devcontainer.HaveImage(engine, image) {
		e.report("devcontainer: pulling %s. The first time, this can take several minutes.", image)
	}
	e.report("devcontainer: starting a container from %s.", image)
	defer func() {
		if e.init == nil && e.cmd != nil {
			e.done("devcontainer: ready in %s (%s). The tool call runs inside it from here.",
				compactDuration(time.Since(started)), image)
		}
	}()

	cmd := exec.CommandContext(context.Background(), engine, argv...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		e.init = fmt.Errorf("container stdin: %w", err)
		return e.init
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		e.init = fmt.Errorf("container stdout: %w", err)
		return e.init
	}
	// The engine's stderr is captured, never streamed: its multi-line progress
	// would corrupt the renderer's transient zone. Kept for a failure report.
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		e.init = fmt.Errorf("container stderr: %w", err)
		return e.init
	}
	stderr := newCappedBuffer(stderrCap)
	e.stderr = stderr

	if err := cmd.Start(); err != nil {
		e.init = fmt.Errorf("starting container (%s): %w", engine, err)
		return e.init
	}
	go func() {
		// Drain so a chatty engine cannot block on a full pipe.
		io.Copy(stderr, stderrPipe)
	}()

	e.cmd = cmd
	if e.client, err = worker.NewClient(stdin, stdout, e.settings); err != nil {
		e.init = fmt.Errorf("the worker in the container did not start: %w", err)
		return e.init
	}
	e.cid = readCIDFile(cidfile)
	return nil
}

// binInContainer is where the worker is mounted inside the container.
const binInContainer = "/ai-code-bin"

// readCIDFile waits briefly for the engine to record the container id, which
// it writes a moment after creation. An id that never appears only disables
// forks, which is reported where a fork is asked for.
func readCIDFile(path string) string {
	defer os.Remove(path)
	for range 100 {
		if b, err := os.ReadFile(path); err == nil {
			if id := strings.TrimSpace(string(b)); id != "" {
				return id
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	return ""
}

// Fork runs a second executor daemon inside the same container, so a worker
// sees everything its parent does. Nothing starts here: the exec happens on
// the fork's first tool call, against the parent's container.
func (e *DevcontainerExecutor) Fork() (tool.Executor, error) {
	host := e
	if e.host != nil {
		// A fork of a fork still belongs to the container's owner.
		host = e.host
	}
	return &DevcontainerExecutor{
		cfg:      e.cfg,
		cfgErr:   e.cfgErr,
		project:  e.project,
		settings: e.settings,
		catalog:  e.catalog,
		workdir:  e.workdir,
		host:     host,
	}, nil
}

// containerRunning asks the engine whether a container is up.
func containerRunning(engine, cid string) bool {
	out, err := exec.Command(engine, "inspect", "-f", "{{.State.Running}}", cid).Output()
	return err == nil && strings.TrimSpace(string(out)) == "true"
}

// startForked attaches a second daemon to the container the host owns.
func (e *DevcontainerExecutor) startForked(ctx context.Context) error {
	e.host.mu.Lock()
	err := e.host.start(ctx)
	cid, engine := e.host.cid, e.host.engine
	e.host.mu.Unlock()

	if err != nil {
		e.init = fmt.Errorf("the container this worker needs is unavailable: %w", err)
		return e.init
	}
	if cid == "" || engine == "" {
		e.init = fmt.Errorf(
			"this container's id could not be read from the engine, so a worker " +
				"cannot be given its own shell inside it")
		return e.init
	}
	// A started process is not a running container, so ask the engine.
	if !containerRunning(engine, cid) {
		e.host.mu.Lock()
		why := e.host.stderrTail()
		e.host.mu.Unlock()
		e.init = fmt.Errorf("the container this session started is not running, "+
			"so a worker cannot be given a shell in it%s", why)
		return e.init
	}

	argv := []string{"exec", "-i", "-w", e.workdir}
	if e.cfg.ContainerUser != "" {
		argv = append(argv, "--user", e.cfg.ContainerUser)
	} else if e.cfg.RemoteUser != "" {
		argv = append(argv, "--user", e.cfg.RemoteUser)
	}
	argv = append(argv, cid, binInContainer)

	cmd := exec.CommandContext(context.Background(), engine, argv...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		e.init = fmt.Errorf("worker shell stdin: %w", err)
		return e.init
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		e.init = fmt.Errorf("worker shell stdout: %w", err)
		return e.init
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		e.init = fmt.Errorf("worker shell stderr: %w", err)
		return e.init
	}
	stderr := newCappedBuffer(stderrCap)
	e.stderr = stderr

	if err := cmd.Start(); err != nil {
		e.init = fmt.Errorf("starting a worker shell in the container (%s): %w", engine, err)
		return e.init
	}
	go func() { io.Copy(stderr, stderrPipe) }()

	e.cmd = cmd
	if e.client, err = worker.NewClient(stdin, stdout, e.settings); err != nil {
		e.init = fmt.Errorf("the worker shell in the container did not start: %w", err)
		return e.init
	}
	return nil
}

// resolveImage produces the image to run: the configured one, or one built from
// its Dockerfile.
func (e *DevcontainerExecutor) resolveImage(ctx context.Context, engine string) (string, error) {
	switch e.cfg.Kind() {
	case devcontainer.KindImage:
		return e.cfg.Image, nil

	case devcontainer.KindBuild:
		tag, err := e.cfg.ImageTag()
		if err != nil {
			return "", err
		}
		// The tag changes whenever the Dockerfile does, so a tagged image is current.
		if devcontainer.HaveImage(engine, tag) {
			return tag, nil
		}
		e.report("devcontainer: building the image from %s. The first time, this can take several "+
			"minutes; Ctrl-C stops it.", filepath.Base(e.cfg.DockerfilePath()))
		out := newCappedBuffer(stderrCap)
		cmd := exec.CommandContext(ctx, engine, e.cfg.BuildArgv(engine, tag)...)
		cmd.Stdout, cmd.Stderr = out, out
		if err := cmd.Run(); err != nil {
			if ctx.Err() != nil {
				return "", fmt.Errorf("building %s from %s was interrupted",
					tag, e.cfg.DockerfilePath())
			}
			return "", fmt.Errorf("building %s from %s failed: %v\n\nBuild output:\n%s",
				tag, e.cfg.DockerfilePath(), err, strings.TrimSpace(out.String()))
		}
		return tag, nil

	case devcontainer.KindCompose:
		return "", fmt.Errorf("this devcontainer is defined by docker-compose (%q), which is not supported yet.\n\n"+
			"ai-code can run an \"image\" or a \"build\" configuration; run with --runtime host to work outside a container",
			e.cfg.Service)

	default:
		return "", fmt.Errorf("this devcontainer.json names no image, build or compose file, so there is nothing to run.\n\n" +
			"Add an \"image\" or a \"build\": {\"dockerfile\": ...}, or run with --runtime host to work outside a container")
	}
}

// Execute runs a tool in the container, starting it on first use. A dead
// container is restarted once per call: its closed pipe fails every later
// write identically, so a restart is the only recovery.
func (e *DevcontainerExecutor) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.start(ctx); err != nil {
		if ctx.Err() != nil {
			return tool.Errorf("devcontainer: interrupted while starting: %v", err), nil
		}
		return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("the devcontainer is unavailable: %v%s", err, e.stderrTail())}
	}

	res, err := e.client.Call(ctx, req)
	if worker.IsUnsent(err) {
		// Nothing was delivered, so re-running cannot double-execute.
		first := e.failure("write", err)
		e.teardown()
		if err := e.start(ctx); err != nil {
			return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("%s\n\nRestarting it failed too: %v%s", first, err, e.stderrTail())}
		}
		res, err = e.client.Call(ctx, req)
		if worker.IsUnsent(err) {
			return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("%s\n\nA freshly started container failed the same way: %v%s",
				first, err, e.stderrTail())}
		}
	}
	if err != nil {
		// Not retried: the request was delivered, so a write or edit may have run.
		msg := e.failure("read", err)
		answered := e.answered
		e.teardown()
		if !answered {
			return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("the worker in the container stopped before it answered anything: %s", msg)}
		}
		return tool.Errorf("%s", msg), nil
	}
	e.answered = true
	if !e.ready {
		// The first reply is the first evidence the container daemon answers.
		e.ready = true
		e.report("devcontainer: ready.")
	}
	return res, nil
}

// failure reports a dead container with the engine's exit status and stderr,
// not a bare "broken pipe".
func (e *DevcontainerExecutor) failure(op string, err error) string {
	status := ""
	if e.cmd != nil && e.cmd.Process != nil {
		// Reap it: nothing else waits on this process, so the status would be lost.
		_ = e.cmd.Process.Kill()
		_ = e.cmd.Wait()
		if st := e.cmd.ProcessState; st != nil {
			status = fmt.Sprintf(" (container exited with status %d)", st.ExitCode())
		}
	}
	return fmt.Sprintf("devcontainer %s failed: %v%s%s", op, err, status, e.stderrTail())
}

// teardown drops the dead container so the next start builds a fresh one.
func (e *DevcontainerExecutor) teardown() {
	e.ready = false
	e.cmd = nil
	e.client = nil
	e.init = nil
	e.answered = false
	e.stderr = nil
}

func (e *DevcontainerExecutor) Definitions() []tool.Definition { return e.catalog.Definitions() }

func (e *DevcontainerExecutor) IsReadOnly(name string) bool { return e.catalog.IsReadOnly(name) }

// Close tears down the container.
func (e *DevcontainerExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.cmd == nil {
		return nil
	}
	err := e.cmd.Process.Kill()
	_ = e.cmd.Wait()
	return err
}

// stderrTail returns the tail of podman's captured stderr, or "".
func (e *DevcontainerExecutor) stderrTail() string {
	if e.stderr == nil {
		return ""
	}
	s := strings.TrimSpace(e.stderr.String())
	if s == "" {
		return ""
	}
	// A single newline keeps the block out of the tool's own first line.
	return "\n\nContainer engine output:\n" + s
}

// detectContainerEngine picks podman or docker; probing is slow, so startup
// defers it.
func detectContainerEngine() (string, error) {
	for _, eng := range []string{"podman", "docker"} {
		if p, err := exec.LookPath(eng); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no container engine found (tried podman, docker)")
}

// workerForImage caches the worker for the image's platform, which is asked of
// the engine rather than assumed to be this machine's, and returns its path.
func workerForImage(engine, image string) (string, error) {
	out, err := exec.Command(engine, "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", image).Output()
	if err != nil {
		return "", fmt.Errorf("asking %s what platform %s is for: %w", filepath.Base(engine), image, err)
	}
	goos, goarch, ok := strings.Cut(strings.TrimSpace(string(out)), "/")
	if !ok || goos == "" || goarch == "" {
		return "", fmt.Errorf("%s reported the platform of %s as %q", filepath.Base(engine), image, strings.TrimSpace(string(out)))
	}
	bin, err := workerbin.For(goos, goarch)
	if err != nil {
		return "", fmt.Errorf("the image %s is %s/%s: %w", image, goos, goarch, err)
	}
	return cacheWorker(bin)
}

// cacheWorker writes bin to the cache, named by its content, and returns the path.
func cacheWorker(bin []byte) (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(bin)
	path := filepath.Join(dir, "ai-code", "workers", "ai-code-worker-"+hex.EncodeToString(sum[:])[:16])
	if fi, err := os.Stat(path); err == nil && fi.Size() == int64(len(bin)) {
		return path, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".worker-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(bin); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Chmod(0o755); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	return path, os.Rename(tmp.Name(), path)
}

// devcontainerID is the value of ${devcontainerId}: stable for a project, so
// that a named volume or label derived from it survives across sessions.
func devcontainerID(project string) string {
	sum := sha256.Sum256([]byte(project))
	return hex.EncodeToString(sum[:])[:16]
}
