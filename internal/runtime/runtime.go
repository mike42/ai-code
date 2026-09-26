// Package runtime decides where ai-code's tools run: in the host process, in a
// devcontainer, or on a remote machine over SSH. The decision is fixed for the
// session, and running tools without a sandbox is explicit, never a fallback.
package runtime

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"ai-code/internal/devcontainer"
)

// Kind identifies where tools execute.
type Kind string

const (
	// KindDevcontainer runs tools inside the project's devcontainer.
	KindDevcontainer Kind = "devcontainer"
	// KindSSH runs tools on a remote machine, used as a sandbox.
	KindSSH Kind = "ssh"
	// KindHost runs tools directly on this machine, with no sandbox.
	KindHost Kind = "host"
)

// Runtime is a fully resolved execution environment. It is produced by Resolve
// and is immutable for the life of a session.
type Runtime struct {
	Kind Kind

	// Image is the devcontainer image, when Kind is KindDevcontainer and the
	// configuration names one; a Dockerfile build leaves this empty and carries
	// the build inputs in Config instead.
	Image string

	// Config is the parsed devcontainer.json, when Kind is KindDevcontainer.
	Config *devcontainer.Config

	// ConfigErr is set when a devcontainer.json was found but could not be
	// parsed; Resolve still returns the runtime and the error surfaces later.
	ConfigErr error

	// SSH destination ("user@host:port"), when Kind is KindSSH.
	Remote string

	// Explicit records whether this runtime was chosen on the command line.
	Explicit bool
}

// Describe returns a short human-readable label for the banner.
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
		return "ssh://" + r.Remote
	default:
		return "HOST - NO SANDBOX"
	}
}

// ShortLabel is a compact banner label: the sandbox kind is the signal.
func (r *Runtime) ShortLabel() string {
	switch r.Kind {
	case KindDevcontainer:
		return "devcontainer"
	case KindSSH:
		return "ssh://" + r.Remote
	default:
		return "HOST - NO SANDBOX"
	}
}

// Sandboxed reports whether this runtime provides isolation.
func (r *Runtime) Sandboxed() bool { return r.Kind != KindHost }

// flagValue returns the canonical --runtime value for this runtime.
func (r *Runtime) flagValue() string {
	switch r.Kind {
	case KindDevcontainer:
		return "devcontainer"
	case KindSSH:
		return "ssh://" + r.Remote
	default:
		return "host"
	}
}

// FlagValue returns the canonical --runtime value for this runtime, for
// recording in a session transcript.
func (r *Runtime) FlagValue() string { return r.flagValue() }

// ErrNoRuntime is returned when no --runtime flag was given and no devcontainer
// was found; startup fails rather than falling back to the host.
type ErrNoRuntime struct{}

func (e ErrNoRuntime) Error() string {
	return `no execution environment was specified and this project defines no devcontainer.

Refusing to run tools without knowing where. Choose one explicitly:

    ai-code --runtime devcontainer     (requires .devcontainer/devcontainer.json)
    ai-code --runtime ssh://buildvm    (a VM or remote machine as the sandbox)
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

// ParseFlag turns a --runtime value into a partial runtime. An empty value
// means "not specified" and is resolved against the filesystem by Resolve.
func ParseFlag(value string) (*Runtime, error) {
	switch {
	case value == "":
		return nil, nil
	case value == "devcontainer":
		return &Runtime{Kind: KindDevcontainer, Explicit: true}, nil
	case value == "host":
		return &Runtime{Kind: KindHost, Explicit: true}, nil
	case strings.HasPrefix(value, "ssh://"):
		dest := strings.TrimPrefix(value, "ssh://")
		if dest == "" {
			return nil, fmt.Errorf("--runtime ssh:// needs a host, e.g. ssh://buildvm")
		}
		return &Runtime{Kind: KindSSH, Remote: dest, Explicit: true}, nil
	default:
		return nil, fmt.Errorf(
			"invalid --runtime %q: must be \"devcontainer\", \"ssh://HOST\", or \"host\"", value)
	}
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

	dcPath := DevcontainerPath(cwd)
	hasDC := dcPath != ""

	if partial != nil {
		switch partial.Kind {
		case KindDevcontainer:
			if !hasDC {
				return nil, ErrNoDevcontainer{}
			}
			// The configuration is read on the explicit path too: /restart
			// re-execs with --runtime devcontainer.
			loadInto(partial, dcPath)
			return partial, nil
		case KindSSH:
			return nil, fmt.Errorf("--runtime ssh:// is not implemented yet; use devcontainer or host")
		}
		return partial, nil
	}

	// No flag given: the only acceptable implicit decision is the safe one, a
	// configured devcontainer.
	if hasDC {
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
