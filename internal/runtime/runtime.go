// Package runtime decides where ai-code's tools run: in the host process, in a
// devcontainer, or on a remote machine over SSH.
//
// The decision is made once, at startup, and is fixed for the session. It is
// also a *safety* decision, not a convenience one. Running tools without a
// sandbox is an explicit, named choice; it is never the default and never a
// fallback. If the user has not said where tools should run and no devcontainer
// is configured, startup fails rather than silently running on the host.
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

	// Devcontainer image name, when Kind is KindDevcontainer and the
	// configuration names one. A configuration that builds from a Dockerfile
	// leaves this empty and carries the build inputs in Config instead.
	Image string

	// Config is the parsed devcontainer.json, when Kind is KindDevcontainer.
	Config *devcontainer.Config

	// ConfigErr is set when a devcontainer.json was found but could not be
	// parsed. Resolve does not fail on it: the runtime decision is still
	// "devcontainer", and the error belongs where the container is started,
	// with the rest of the engine diagnostics.
	ConfigErr error

	// SSH destination ("user@host:port"), when Kind is KindSSH.
	Remote string

	// Explicit records whether the user chose this on the command line, as
	// opposed to it being the safe auto-detected default.
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

// ShortLabel is a compact banner label that omits the image name. The banner
// already names the provider, model and context; a fully-qualified container
// image (e.g. mcr.microsoft.com/devcontainers/python:3.12) makes the runtime
// line as long as the rest combined without saying much. The sandbox kind is
// the signal that matters.
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

// ErrNoRuntime is returned when the user gave no --runtime flag and no
// devcontainer could be found. It is deliberately an error rather than a
// fallback to the host: accidentally running unsandboxed is the one failure
// this feature exists to prevent.
type ErrNoRuntime struct{}

func (e ErrNoRuntime) Error() string {
	return `no execution environment was specified and this project defines no devcontainer.

Refusing to run tools without knowing where. Choose one explicitly:

    ai-code --runtime devcontainer     (requires .devcontainer/devcontainer.json)
    ai-code --runtime ssh://buildvm    (a VM or remote machine as the sandbox)
    ai-code --runtime host             (NO SANDBOX - run directly on this machine)
`
}

// ErrNoDevcontainer is returned when the user explicitly asked for a
// devcontainer but the project does not define one.
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
// or any of its parents, or "" if none exists. It is a cheap filesystem probe,
// so it is safe on the startup path.
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

// Resolve decides where tools run, from the CLI flag and the filesystem.
//
// Only two fast, local operations happen here: parsing the flag and probing for
// a devcontainer file. Nothing that could be slow -- container engine detection,
// starting a container, an SSH connection -- is attempted. Those are deferred to
// the first tool call, so the user reaches a prompt immediately.
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
			// The configuration has to be read here too, not only on the
			// implicit path below. /restart re-execs with an explicit
			// --runtime devcontainer, so skipping it meant a devcontainer
			// worked on first launch and every restarted session ran the
			// container engine with an empty image name.
			loadInto(partial, dcPath)
			return partial, nil
		case KindSSH:
			return nil, fmt.Errorf("--runtime ssh:// is not implemented yet; use devcontainer or host")
		}
		return partial, nil
	}

	// No flag given. The only acceptable implicit decision is the safe one:
	// a configured devcontainer. Anything else demands an explicit choice.
	if hasDC {
		rt := &Runtime{Kind: KindDevcontainer}
		loadInto(rt, dcPath)
		return rt, nil
	}
	return nil, ErrNoRuntime{}
}

// loadInto parses the devcontainer configuration onto a runtime.
//
// A parse failure is recorded rather than returned. Startup has already decided
// that tools run in a container; refusing to start at all because the file has
// a stray comma would deny the user the session in which they would fix it, and
// the error is reported the moment the container is needed.
func loadInto(rt *Runtime, dcPath string) {
	cfg, err := devcontainer.Load(dcPath)
	if err != nil {
		rt.ConfigErr = err
		return
	}
	rt.Config = cfg
	rt.Image = cfg.Image
}
