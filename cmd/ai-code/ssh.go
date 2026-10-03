package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"

	"ai-code/internal/runtime"
	"ai-code/internal/tool"
	"ai-code/internal/worker"
	"ai-code/internal/workerbin"
)

type SSHExecutor struct {
	rt        *runtime.Runtime
	sshConfig string
	settings  worker.Settings
	catalog   *tool.LocalExecutor
	progress  tool.Progress

	host *SSHExecutor

	// prepMu, not mu, so a fork can start while the host runs a long call.
	prepMu    sync.Mutex
	probed    *remoteFacts
	remoteBin string

	mu       sync.Mutex
	cmd      *exec.Cmd
	in       io.WriteCloser
	client   *worker.Client
	stderr   *cappedBuffer
	answered bool
}

func NewSSHExecutor(rt *runtime.Runtime, sshConfig string, settings worker.Settings) *SSHExecutor {
	return &SSHExecutor{rt: rt, sshConfig: sshConfig, settings: settings, catalog: worker.Tools(settings, "")}
}

func (e *SSHExecutor) SetProgress(p tool.Progress) { e.progress = p }

func (e *SSHExecutor) report(format string, args ...any) {
	if e.progress != nil {
		e.progress(tool.Phase{Note: fmt.Sprintf(format, args...)})
	}
}

// Forwarding is forced off whatever the ssh config says; BatchMode because
// nobody can answer a prompt.
func (e *SSHExecutor) sshArgv(remoteCmd string) []string {
	argv := []string{"-T",
		"-o", "BatchMode=yes",
		"-o", "ClearAllForwardings=yes",
		"-o", "ForwardAgent=no",
		"-o", "ForwardX11=no",
		"-o", "PermitLocalCommand=no",
	}
	if e.sshConfig != "" {
		argv = append(argv, "-F", e.sshConfig)
	}
	if e.rt.SSHPort != "" {
		argv = append(argv, "-p", e.rt.SSHPort)
	}
	if e.rt.SSHUser != "" {
		argv = append(argv, "-l", e.rt.SSHUser)
	}
	// The login shell may not be POSIX.
	return append(argv, "--", e.rt.SSHHost, "sh -c "+shQuote(remoteCmd))
}

func (e *SSHExecutor) run(ctx context.Context, remoteCmd string, stdin io.Reader) (string, error) {
	cmd := exec.CommandContext(ctx, "ssh", e.sshArgv(remoteCmd)...)
	tool.OwnProcessGroup(cmd)
	cmd.Stdin = stdin
	var out bytes.Buffer
	errOut := newCappedBuffer(stderrCap)
	cmd.Stdout, cmd.Stderr = &out, errOut
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(errOut.String()); msg != "" {
			return "", fmt.Errorf("%v: %s", err, msg)
		}
		return "", err
	}
	return out.String(), nil
}

func shQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func platformOf(uname string) (goos, goarch string, err error) {
	f := strings.Fields(uname)
	if len(f) != 2 {
		return "", "", fmt.Errorf("uname -s -m printed %q, which is not an operating system and a machine", strings.TrimSpace(uname))
	}
	switch strings.ToLower(f[0]) {
	case "linux":
		goos = "linux"
	case "darwin":
		goos = "darwin"
	case "freebsd":
		goos = "freebsd"
	case "openbsd":
		goos = "openbsd"
	case "netbsd":
		goos = "netbsd"
	default:
		return "", "", fmt.Errorf("the remote machine runs %s, which ai-code has no worker for", f[0])
	}
	switch f[1] {
	case "x86_64", "amd64":
		goarch = "amd64"
	case "aarch64", "arm64":
		goarch = "arm64"
	case "i386", "i686":
		goarch = "386"
	case "riscv64":
		goarch = "riscv64"
	default:
		return "", "", fmt.Errorf("the remote machine is %s on %s, which ai-code has no worker for", f[0], f[1])
	}
	return goos, goarch, nil
}

type remoteFacts struct {
	goos, goarch string
	cacheDir     string
	noCloud      string
}

// probeEnd stops a truncated reply reading as "no .nocloud".
const probeEnd = "ai-code-probe-end"

func probeScript(dir string) string {
	start := `"$HOME"`
	if dir != "" {
		start = shQuote(dir)
	}
	return `uname -s -m && printf '%s\n' "${XDG_CACHE_HOME:-$HOME/.cache}" && ` +
		`find_nocloud() { p=$1; while [ -n "$p" ]; do ` +
		`if [ -e "$p/.nocloud" ] || [ -L "$p/.nocloud" ]; then printf '%s\n' "$p/.nocloud"; return 0; fi; ` +
		`[ "$p" = / ] && return 1; p=$(dirname "$p"); done; return 1; } && ` +
		`d=` + start + ` && ` +
		`{ find_nocloud "$d" || { r=$(cd "$d" 2>/dev/null && pwd -P) && find_nocloud "$r"; } || printf '\n'; } && ` +
		`echo ` + probeEnd
}

func parseProbe(out string) (*remoteFacts, error) {
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) != 4 || lines[3] != probeEnd {
		return nil, fmt.Errorf("the remote machine's answer was not complete: %q", out)
	}
	goos, goarch, err := platformOf(lines[0])
	if err != nil {
		return nil, err
	}
	return &remoteFacts{goos: goos, goarch: goarch,
		cacheDir: strings.TrimSpace(lines[1]), noCloud: strings.TrimSpace(lines[2])}, nil
}

func (e *SSHExecutor) probe(ctx context.Context) (*remoteFacts, error) {
	owner := e.owner()
	owner.prepMu.Lock()
	defer owner.prepMu.Unlock()
	return owner.probeLocked(ctx)
}

func (e *SSHExecutor) probeLocked(ctx context.Context) (*remoteFacts, error) {
	if e.probed != nil {
		return e.probed, nil
	}
	e.report("ssh: connecting to %s.", e.rt.Remote)
	out, err := e.run(ctx, probeScript(e.rt.SSHDir), nil)
	if err != nil {
		return nil, fmt.Errorf("could not reach %s over ssh: %w", e.rt.Remote, err)
	}
	facts, err := parseProbe(out)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", e.rt.Remote, err)
	}
	e.probed = facts
	return facts, nil
}

func (e *SSHExecutor) owner() *SSHExecutor {
	if e.host != nil {
		return e.host
	}
	return e
}

func (e *SSHExecutor) NoCloud(ctx context.Context) (string, error) {
	facts, err := e.probe(ctx)
	if err != nil {
		return "", err
	}
	return facts.noCloud, nil
}

func (e *SSHExecutor) prepare(ctx context.Context) (string, error) {
	owner := e.owner()
	owner.prepMu.Lock()
	defer owner.prepMu.Unlock()
	if owner.remoteBin != "" {
		return owner.remoteBin, nil
	}
	facts, err := owner.probeLocked(ctx)
	if err != nil {
		return "", err
	}
	bin, err := workerbin.For(facts.goos, facts.goarch)
	if err != nil {
		return "", fmt.Errorf("%s is %s/%s: %w", e.rt.Remote, facts.goos, facts.goarch, err)
	}
	sum := sha256.Sum256(bin)
	dir := facts.cacheDir + "/ai-code/bin"
	remote := dir + "/ai-code-worker-" + hex.EncodeToString(sum[:])[:16]

	if _, err := e.run(ctx, "test -x "+shQuote(remote), nil); err != nil {
		e.report("ssh: copying the %s/%s worker (%d MB) to %s.", facts.goos, facts.goarch, len(bin)>>20, e.rt.Remote)
		// Renamed into place so concurrent sessions never run a partial file.
		tmp := shQuote(remote+".") + "$$"
		script := fmt.Sprintf("umask 077 && mkdir -p %s && cat > %s && chmod 700 %s && mv -f %s %s",
			shQuote(dir), tmp, tmp, tmp, shQuote(remote))
		if _, err := e.run(ctx, script, bytes.NewReader(bin)); err != nil {
			return "", fmt.Errorf("copying the worker to %s: %w", e.rt.Remote, err)
		}
	}
	owner.remoteBin = remote
	return remote, nil
}

func (e *SSHExecutor) start(ctx context.Context) error {
	if e.cmd != nil {
		return nil
	}
	started := time.Now()
	bin, err := e.prepare(ctx)
	if err != nil {
		return err
	}

	launch := "exec " + shQuote(bin)
	if e.rt.SSHDir != "" {
		launch = "cd " + shQuote(e.rt.SSHDir) + " && " + launch
	}
	cmd := exec.Command("ssh", e.sshArgv(launch)...)
	tool.OwnProcessGroup(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	e.stderr = newCappedBuffer(stderrCap)
	cmd.Stderr = e.stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting ssh: %w", err)
	}
	client, err := worker.NewClient(stdin, stdout, e.settings)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return fmt.Errorf("the worker on %s did not start: %v%s", e.rt.Remote, err, e.stderrTail())
	}
	e.cmd, e.in, e.client = cmd, stdin, client
	if e.host == nil && e.progress != nil {
		e.progress(tool.Phase{Note: fmt.Sprintf("ssh: connected to %s in %s.", e.rt.Remote,
			compactDuration(time.Since(started))), Done: true})
	}
	return nil
}

func (e *SSHExecutor) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.start(ctx); err != nil {
		if ctx.Err() != nil {
			return tool.Errorf("interrupted while connecting to %s", e.rt.Remote), nil
		}
		return tool.Result{}, &tool.Unavailable{Err: err}
	}
	res, err := e.client.Call(ctx, req)
	// Unsent: retrying on a new connection cannot run it twice.
	if worker.IsUnsent(err) {
		first := e.failure("write", err)
		e.teardown()
		if err := e.start(ctx); err != nil {
			return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("%s\n\nReconnecting failed: %w", first, err)}
		}
		res, err = e.client.Call(ctx, req)
		if worker.IsUnsent(err) {
			return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("%s\n\nA new connection failed the same way: %v%s",
				first, err, e.stderrTail())}
		}
	}
	if err != nil {
		// Not retried: the call may have run.
		msg := e.failure("read", err)
		answered := e.answered
		e.teardown()
		if !answered {
			return tool.Result{}, &tool.Unavailable{Err: fmt.Errorf("the worker on %s stopped before it answered anything: %s",
				e.rt.Remote, msg)}
		}
		return tool.Errorf("%s", msg), nil
	}
	e.answered = true
	return res, nil
}

func (e *SSHExecutor) failure(op string, err error) string {
	status := ""
	if e.cmd != nil && e.cmd.Process != nil {
		_ = e.cmd.Process.Kill()
		_ = e.cmd.Wait()
		if st := e.cmd.ProcessState; st != nil {
			status = fmt.Sprintf(" (ssh exited with status %d)", st.ExitCode())
		}
	}
	return fmt.Sprintf("the connection to %s failed on %s: %v%s%s", e.rt.Remote, op, err, status, e.stderrTail())
}

func (e *SSHExecutor) teardown() {
	e.cmd, e.in, e.client, e.answered = nil, nil, nil, false
}

func (e *SSHExecutor) stderrTail() string {
	if e.stderr == nil {
		return ""
	}
	if s := strings.TrimSpace(e.stderr.String()); s != "" {
		return "\n\nssh said:\n" + s
	}
	return ""
}

func (e *SSHExecutor) Fork() (tool.Executor, error) {
	host := e
	if e.host != nil {
		host = e.host
	}
	return &SSHExecutor{
		rt: e.rt, sshConfig: e.sshConfig, settings: e.settings, catalog: e.catalog,
		progress: e.progress, host: host,
	}, nil
}

func (e *SSHExecutor) Definitions() []tool.Definition { return e.catalog.Definitions() }

func (e *SSHExecutor) IsReadOnly(name string) bool { return e.catalog.IsReadOnly(name) }

func (e *SSHExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.cmd == nil {
		return nil
	}
	_ = e.in.Close()
	err := e.cmd.Process.Kill()
	_ = e.cmd.Wait()
	e.teardown()
	return err
}
