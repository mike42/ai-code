// Package runtime decides where ai-code's tools run: in the host process, in a
// devcontainer, or on a remote machine over SSH. The decision is fixed for the
// session, and running tools without a sandbox is explicit, never a fallback.
package runtime

import (
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"ai-code/internal/devcontainer"
)

type Kind string

const (
	KindDevcontainer Kind = "devcontainer"
	KindSSH          Kind = "ssh"
	KindHost         Kind = "host"
)

type Runtime struct {
	Kind Kind

	// Image is the devcontainer image, when Kind is KindDevcontainer and the
	// configuration names one; a Dockerfile build leaves this empty and carries
	// the build inputs in Config instead.
	Image string

	Config *devcontainer.Config

	// ConfigErr is set when a devcontainer.json was found but could not be
	// parsed; Resolve still returns the runtime and the error surfaces later.
	ConfigErr error

	Remote string
	// Empty user or port is left to ssh config; empty Dir is the remote home.
	SSHUser string
	SSHHost string
	SSHPort string
	SSHDir  string

	// Explicit records whether this runtime was chosen on the command line.
	Explicit bool
}

func (r *Runtime) Describe() string {
	switch r.Kind {
	case KindDevcontainer:
		if r.Image != "" {
			return "devcontainer:" + r.Image
		}
		if r.Config != nil && r.Config.Dockerfile() != "" {
			return "devcontainer:" + r.Config.Dockerfile()
		}
		return "devcontainer (no image configured)"
	case KindSSH:
		return r.sshURL()
	default:
		return "HOST - NO SANDBOX"
	}
}

func (r *Runtime) sshURL() string {
	return "ssh://" + r.Remote + r.SSHDir
}

// ShortLabel is a compact banner label: the sandbox kind is the signal.
func (r *Runtime) ShortLabel() string {
	switch r.Kind {
	case KindDevcontainer:
		return "devcontainer"
	case KindSSH:
		return r.sshURL()
	default:
		return "HOST - NO SANDBOX"
	}
}

func (r *Runtime) Sandboxed() bool { return r.Kind != KindHost }

func (r *Runtime) flagValue() string {
	switch r.Kind {
	case KindDevcontainer:
		return "devcontainer"
	case KindSSH:
		return r.sshURL()
	default:
		return "host"
	}
}

func (r *Runtime) FlagValue() string { return r.flagValue() }

// ErrNoRuntime is returned when no --runtime flag was given and no devcontainer
// was found; startup fails rather than falling back to the host.
type ErrNoRuntime struct{}

func (e ErrNoRuntime) Error() string {
	return `no execution environment was specified and this project defines no devcontainer.

Refusing to run tools without knowing where. Choose one explicitly:

    ai-code --runtime devcontainer     (requires .devcontainer/devcontainer.json)
    ai-code --runtime ssh://user@buildvm/home/user/work
                                       (a VM or remote machine as the sandbox)
    ai-code --runtime host             (NO SANDBOX - run directly on this machine)
`
}

// ErrNoDevcontainer is returned when --runtime devcontainer was requested but
// the project does not define one.
type ErrNoDevcontainer struct{}

func (e ErrNoDevcontainer) Error() string {
	return `--runtime devcontainer was requested but no devcontainer configuration was found.

Create one of:
    .devcontainer/devcontainer.json
    .devcontainer.json

or choose a different environment.`
}

func ParseFlag(value string) (*Runtime, error) {
	switch {
	case value == "":
		return nil, nil
	case value == "devcontainer":
		return &Runtime{Kind: KindDevcontainer, Explicit: true}, nil
	case value == "host":
		return &Runtime{Kind: KindHost, Explicit: true}, nil
	case strings.HasPrefix(value, "ssh://"):
		return parseSSH(value)
	default:
		return nil, fmt.Errorf(
			"invalid --runtime %q: must be \"devcontainer\", \"ssh://[USER@]HOST[:PORT][/DIR]\", or \"host\"", value)
	}
}

// parseSSH reads ssh://[USER@]HOST[:PORT][/DIR], DIR absolute on the remote.
func parseSSH(value string) (*Runtime, error) {
	usage := "--runtime ssh:// takes [USER@]HOST[:PORT][/DIR], e.g. ssh://agent@buildvm/home/agent/work"
	u, err := url.Parse(value)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", usage, err)
	}
	if u.Hostname() == "" {
		return nil, fmt.Errorf("%s: no host given", usage)
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("%s: %q has a query or fragment, which mean nothing here", usage, value)
	}
	if _, set := u.User.Password(); set {
		return nil, fmt.Errorf("%s: a password does not belong in the URL; use a key or ssh's own configuration", usage)
	}
	r := &Runtime{Kind: KindSSH, Explicit: true, SSHHost: u.Hostname(), SSHPort: u.Port()}
	if u.User != nil {
		r.SSHUser = u.User.Username()
	}
	if r.SSHPort != "" {
		if n, err := strconv.Atoi(r.SSHPort); err != nil || n < 1 || n > 65535 {
			return nil, fmt.Errorf("%s: %q is not a port", usage, r.SSHPort)
		}
	}
	if u.Path != "" && u.Path != "/" {
		r.SSHDir = path.Clean(u.Path)
	}
	r.Remote = u.Host
	if r.SSHUser != "" {
		r.Remote = r.SSHUser + "@" + u.Host
	}
	return r, nil
}

// DevcontainerPath returns the path to a devcontainer configuration file in dir
// or any of its parents, or "" if none exists.
func DevcontainerPath(dir string) string {
	if dir == "" {
		return ""
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	for p := abs; ; {
		for _, name := range []string{
			filepath.Join(p, ".devcontainer", "devcontainer.json"),
			filepath.Join(p, ".devcontainer.json"),
		} {
			if fi, err := os.Stat(name); err == nil && !fi.IsDir() {
				return name
			}
		}
		parent := filepath.Dir(p)
		if parent == p {
			break
		}
		p = parent
	}
	return ""
}

// Resolve decides where tools run from the CLI flag and the filesystem, without
// touching the container engine: that is deferred to the first tool call.
func Resolve(flag string, cwd string) (*Runtime, error) {
	partial, err := ParseFlag(flag)
	if err != nil {
		return nil, err
	}

	if partial != nil {
		switch partial.Kind {
		case KindDevcontainer:
			dcPath := DevcontainerPath(cwd)
			if dcPath == "" {
				return nil, ErrNoDevcontainer{}
			}
			// The configuration is read on the explicit path too: /restart
			// re-execs with --runtime devcontainer.
			loadInto(partial, dcPath)
			return partial, nil
		}
		return partial, nil
	}

	// No flag given: the only acceptable implicit decision is the safe one, a
	// configured devcontainer.
	if dcPath := DevcontainerPath(cwd); dcPath != "" {
		rt := &Runtime{Kind: KindDevcontainer}
		loadInto(rt, dcPath)
		return rt, nil
	}
	return nil, ErrNoRuntime{}
}

// loadInto parses the devcontainer configuration onto a runtime, recording a
// parse failure rather than failing startup; it surfaces when the container is
// needed.
func loadInto(rt *Runtime, dcPath string) {
	cfg, err := devcontainer.Load(dcPath)
	if err != nil {
		rt.ConfigErr = err
		return
	}
	rt.Config = cfg
	rt.Image = cfg.Image
}
