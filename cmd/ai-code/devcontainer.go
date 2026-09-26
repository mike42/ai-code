package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"ai-code/internal/config"
	"ai-code/internal/devcontainer"
	"ai-code/internal/tool"
)

// DevcontainerExecutor runs tools inside the project's devcontainer.
//
// Startup is deferred to the first Execute call, so the user reaches a prompt
// without waiting on engine detection and a container start.
//
// It launches the same binary inside the container in --executor-daemon mode
// over an attached pipe and speaks JSON-lines to it, which remotes the whole
// tool set and keeps the file tools on the same filesystem as the shell.
type DevcontainerExecutor struct {
	// cfg is the parsed devcontainer.json. It decides the image, the mounts,
	// the user and everything else about the container.
	cfg *devcontainer.Config
	// cfgErr is a devcontainer.json that could not be parsed. Startup declined
	// to fail on it so the user got a session; it is reported here, where the
	// container is actually needed.
	cfgErr error
	// project is the host directory mounted into the container.
	project string
	// binary is the path to this ai-code binary on the host.
	binary string
	// workdir is the container path the daemon starts in -- the mounted
	// equivalent of the directory the user launched ai-code from.
	workdir string
	// progress names each startup phase as it is reached, so a tool call that
	// waits minutes for a container says why while it waits. A note after the
	// wait would not do: the point is to speak during it. nil in a session
	// with nowhere to show it. Set once, before the executor is used.
	progress tool.Progress
	// ready guards the one-off "ready" note, and is cleared by teardown so a
	// restarted container announces itself again.
	ready bool

	// host is set on a forked executor and names the one that owns the
	// container. A fork runs a second daemon inside that container rather
	// than starting a container of its own, so the two agents share a
	// filesystem exactly and differ only in which shell they are talking to.
	host *DevcontainerExecutor

	mu   sync.Mutex
	cmd  *exec.Cmd
	enc  *json.Encoder
	dec  *json.Decoder
	init error
	// cid is the container this executor started, once it has started one,
	// and engine the command that started it. Read by forks, which exec into
	// it. Taken from --cidfile rather than from a name we chose: the engine
	// reports the container it actually created, which stays correct even if
	// the project's runArgs renamed it.
	cid    string
	engine string
	// stderr holds the engine's captured output, for a failure report. It is
	// only meaningful after start has launched the container.
	stderr *cappedBuffer
}

// stderrCap bounds how much podman startup output we retain for a failure
// report. Enough to show the real error, not a dump of progress bars.
const stderrCap = 64 * 1024

// cappedBuffer keeps only the tail of what is written to it, so a chatty
// subprocess cannot accumulate unbounded output while we hold it for a failure
// report.
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
//
// project is the host directory mounted into the container; cwd is the
// directory the user launched ai-code from, which may be a subdirectory of
// project. The daemon starts in the mounted equivalent of cwd so the agent's
// first shell prompt lands where the user is, not at the mount root.
func NewDevcontainerExecutor(cfg *devcontainer.Config, cfgErr error, project, cwd, binary string) *DevcontainerExecutor {
	// The mount source must be absolute: it is passed to the container engine on
	// the host, where relative paths would resolve against the engine's cwd
	// rather than ours. Resolve it here so a `./ai-code` launch still works.
	if abs, err := filepath.Abs(binary); err == nil {
		binary = abs
	}
	e := &DevcontainerExecutor{cfg: cfg, cfgErr: cfgErr, project: project, binary: binary}
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

// mountCwd maps the user's launch directory into the container's mount layout.
// The project is mounted at workspaceFolder, so a cwd inside it becomes
// workspaceFolder/<relative path>. Anything that escapes the project (or that
// cannot be relativised) falls back to the mount root.
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

// start brings the container up on first use. It is not called from the startup
// path; only from Execute.
//
// The context is the turn's, so Ctrl-C during a multi-minute image build stops
// the build. It does not govern the container itself, which has to outlive the
// call that started it.
func (e *DevcontainerExecutor) start(ctx context.Context) error {
	if e.init != nil || e.cmd != nil {
		return e.init
	}
	// A fork has no container of its own to bring up; it attaches a second
	// daemon to the one its host owns.
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
	// The command's own timeout does not start until the container is up --
	// it is applied by the bash tool running inside it -- which is the first
	// thing anyone watching a four-minute tool call wants to know.
	e.report("devcontainer: preparing the container with %s. The command has not started yet, "+
		"so its timeout is not running.", engine)

	image, err := e.resolveImage(ctx, engine)
	if err != nil {
		// A build the user interrupted is not a broken configuration. Caching
		// it would make one Ctrl-C disable the container for the rest of the
		// session.
		if ctx.Err() != nil {
			return err
		}
		e.init = err
		return err
	}

	argv := e.cfg.RunArgv(e.project, e.workdir, e.binary, binInContainer)

	// --cidfile after RunArgv, so the id belongs to the container that was
	// actually created. The engine refuses to write to a file that exists, so
	// the name is fresh and the file is removed rather than truncated.
	cidfile := filepath.Join(os.TempDir(), fmt.Sprintf("ai-code-cid-%d-%d", os.Getpid(), time.Now().UnixNano()))
	os.Remove(cidfile)
	argv = append(argv, "--cidfile", cidfile)
	argv = append(argv, image, "--executor-daemon", e.workdir)
	e.engine = engine

	// Checked rather than assumed: the engine pulls a missing image as part of
	// run, and that is the long wait for an "image" configuration exactly as a
	// build is for a "build" one. Claiming a pull that did not happen would be
	// as unhelpful as saying nothing.
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
	// The engine's stderr is captured, never streamed to the terminal. Startup
	// and image-pull progress is noisy, multi-line and interleaved with nothing
	// the renderer owns -- streaming it would corrupt the transient zone (the
	// thinking marquee, the spinner) with raw terminal writes it cannot clean
	// up. The tail is kept and reported if startup fails; a successful run
	// discards it.
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
		// Drain so a chatty engine cannot block on a full pipe while the agent
		// waits for its tool result.
		io.Copy(stderr, stderrPipe)
	}()

	e.cmd = cmd
	e.enc = json.NewEncoder(stdin)
	e.dec = json.NewDecoder(bufio.NewReader(stdout))
	e.cid = readCIDFile(cidfile)
	return nil
}

// binInContainer is where this binary is mounted inside the container. The
// daemon, and every forked daemon, is that binary.
const binInContainer = "/ai-code-bin"

// readCIDFile waits briefly for the engine to record the container id.
//
// The engine writes it at creation, which is a moment after our process
// starts, so the first look is usually too early. An id that never appears is
// not a failure of the session -- the container is up and the first daemon is
// answering -- it only means forks cannot exec into it, which is reported at
// the point a fork is asked for.
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

// Fork runs a second executor daemon inside the same container.
//
// Not a second container: the point of a fork is an agent that sees exactly
// what its parent sees, and a second container would diverge on everything
// written outside the workspace mount -- installed packages, build caches,
// /tmp -- which is most of what a worker doing real work produces.
//
// The container is not started here. A fork is created while the parent may
// still be idle, and a worker that turns out to need no shell at all would
// pay minutes for nothing. The exec happens on the fork's first tool call,
// against whatever container the parent has by then, and Close ends it when
// the worker is done.
func (e *DevcontainerExecutor) Fork() (tool.Executor, error) {
	host := e
	if e.host != nil {
		// A fork of a fork still belongs to the container's owner. Chaining
		// would make the daemon depth follow the agent depth for no reason.
		host = e.host
	}
	return &DevcontainerExecutor{
		cfg:     e.cfg,
		cfgErr:  e.cfgErr,
		project: e.project,
		binary:  e.binary,
		workdir: e.workdir,
		host:    host,
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
	// Asked rather than assumed. The host reports a started *process*, which
	// is not a running container: a `run` that fails leaves the process
	// exiting a moment later, and the first thing anyone sees is a worker
	// exec'ing into an id that no longer exists. Ask the engine, and report
	// what it said about the container rather than what it said about the
	// exec that could not find it.
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
	argv = append(argv, cid, binInContainer, "--executor-daemon", e.workdir)

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
	e.enc = json.NewEncoder(stdin)
	e.dec = json.NewDecoder(bufio.NewReader(stdout))
	return nil
}

// resolveImage produces the image to run: the one the configuration names, or
// one built from its Dockerfile.
func (e *DevcontainerExecutor) resolveImage(ctx context.Context, engine string) (string, error) {
	switch e.cfg.Kind() {
	case devcontainer.KindImage:
		return e.cfg.Image, nil

	case devcontainer.KindBuild:
		tag, err := e.cfg.ImageTag()
		if err != nil {
			return "", err
		}
		// The engine's layer cache makes an unchanged rebuild cheap, but not
		// free, and the tag already changes whenever the Dockerfile does. An
		// image we have built before is therefore taken as current.
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

// Execute runs a tool in the container, starting it on first use.
//
// A container that has died is restarted rather than reported, once per call.
// The failure this recovers from is a write to a closed pipe, which is what a
// container that exited between two tool calls looks like from here -- and
// because start() treated a non-nil cmd as a live one, the old behaviour was
// to keep writing to that same dead pipe for the rest of the session. Every
// subsequent tool call failed identically and no amount of retrying by the
// model could clear it.
func (e *DevcontainerExecutor) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.start(ctx); err != nil {
		return tool.Errorf("devcontainer unavailable: %v%s", err, e.stderrTail()), nil
	}

	if err := e.enc.Encode(req); err != nil {
		// Nothing was delivered, so re-running it cannot double-execute
		// anything. Take the container down, bring a fresh one up and send it
		// once more before giving up.
		first := e.failure("write", err)
		e.teardown()
		if err := e.start(ctx); err != nil {
			return tool.Errorf("%s\n\nRestarting it failed too: %v%s", first, err, e.stderrTail()), nil
		}
		if err := e.enc.Encode(req); err != nil {
			return tool.Errorf("%s\n\nA freshly started container failed the same way: %v%s",
				first, err, e.stderrTail()), nil
		}
	}

	var res tool.Result
	if err := e.dec.Decode(&res); err != nil {
		// Deliberately not retried: the request was delivered, so the tool may
		// already have run. Re-sending a write or an edit because the reply
		// went missing is worse than reporting the failure. The container is
		// still torn down, so the next call starts a clean one.
		msg := e.failure("read", err)
		e.teardown()
		return tool.Errorf("%s", msg), nil
	}
	if !e.ready {
		// Announced here rather than after Start, because a started container
		// is not yet a working one -- this reply is the first evidence that
		// the agent inside it is answering.
		e.ready = true
		e.report("devcontainer: ready.")
	}
	return res, nil
}

// failure describes a dead container in terms of what actually happened to it.
//
// "broken pipe" on its own sends people looking at ai-code when the engine has
// printed the real reason -- a missing mount source, an image that will not
// run, an OOM kill -- and thrown it away. The exit status and the engine's own
// output are the diagnosis, so they go in the message.
func (e *DevcontainerExecutor) failure(op string, err error) string {
	status := ""
	if e.cmd != nil && e.cmd.Process != nil {
		// Reap it so the exit status is knowable. Nothing else waits on this
		// process, so without it a container that exited on its own stays a
		// zombie and its status is lost.
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
	e.enc = nil
	e.dec = nil
	e.init = nil
	e.stderr = nil
}

// Definitions advertises the same tools as the local executor. The container
// serves identical definitions from the same binary, so this is safe to
// advertise before the container starts.
func (e *DevcontainerExecutor) Definitions() []tool.Definition {
	// Build the definitions locally so the model sees them without waiting for
	// the container. The daemon builds the same set from the same binary.
	cfg, err := config.Load(e.project)
	if err != nil {
		return nil
	}
	exec, _ := buildTools(cfg, e.project, 0)
	return exec.Definitions()
}

// IsReadOnly reports whether a tool is safe to run concurrently. The container
// serves the same tools from the same binary.
func (e *DevcontainerExecutor) IsReadOnly(name string) bool {
	return isReadOnlyLocally(e.project, name)
}

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

// stderrTail returns the tail of podman's captured stderr, formatted as a
// diagnostic appendix. Empty when there was nothing to capture or the start
// failed before the pipe existed.
func (e *DevcontainerExecutor) stderrTail() string {
	if e.stderr == nil {
		return ""
	}
	s := strings.TrimSpace(e.stderr.String())
	if s == "" {
		return ""
	}
	// A single newline keeps the appended block out of the tool's own first
	// line.
	return "\n\nContainer engine output:\n" + s
}

// detectContainerEngine picks podman or docker. This is the potentially slow
// step, and it is why startup defers it: probing podman on a machine where it
// is not installed can take seconds, and the user should not wait.
func detectContainerEngine() (string, error) {
	for _, eng := range []string{"podman", "docker"} {
		if p, err := exec.LookPath(eng); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no container engine found (tried podman, docker)")
}

// isReadOnlyLocally mirrors the local executor's read-only map without holding
// a container. Used to keep concurrency decisions consistent before start.
func isReadOnlyLocally(project, name string) bool {
	cfg, err := config.Load(project)
	if err != nil {
		return false
	}
	exec, _ := buildTools(cfg, project, 0)
	return exec.IsReadOnly(name)
}

// devcontainerID is the value of ${devcontainerId}: stable for a project, so
// that a named volume or label derived from it survives across sessions.
func devcontainerID(project string) string {
	sum := sha256.Sum256([]byte(project))
	return hex.EncodeToString(sum[:])[:16]
}
