// Package devcontainer reads a devcontainer.json and turns it into the
// decisions a container engine needs: image, workspace mount, and the rest.
// Top-level keys are kept raw so anything unacted on is still reported.
package devcontainer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Config is a parsed devcontainer.json.
type Config struct {
	// Path is the file this was read from, and Dir the directory holding it.
	// Relative paths inside the configuration resolve against Dir.
	Path string
	Dir  string

	// Present holds every top-level key that appeared in the file, so that
	// Unsupported can name what was read but not acted on.
	Present map[string]json.RawMessage

	Name string `json:"name"`

	Image string `json:"image"`
	Build *Build `json:"build"`
	// DockerfileLegacy and ContextLegacy are the pre-"build" spellings. Both
	// are still in the published schema, so a real project can be using them.
	DockerfileLegacy string `json:"dockerFile"`
	ContextLegacy    string `json:"context"`

	// Recognised only so the configuration can be refused precisely, not run.
	DockerComposeFile json.RawMessage `json:"dockerComposeFile"`
	Service           string          `json:"service"`

	WorkspaceFolder string `json:"workspaceFolder"`
	WorkspaceMount  string `json:"workspaceMount"`

	Mounts       []Mount           `json:"mounts"`
	RunArgs      []string          `json:"runArgs"`
	ContainerEnv map[string]string `json:"containerEnv"`
	// envRewrites records host paths rewritten to container paths. See
	// EnvRewrites.
	envRewrites   []string
	RemoteEnv     map[string]string `json:"remoteEnv"`
	ContainerUser string            `json:"containerUser"`
	RemoteUser    string            `json:"remoteUser"`
	Init          *bool             `json:"init"`
	Privileged    *bool             `json:"privileged"`
	CapAdd        []string          `json:"capAdd"`
	SecurityOpt   []string          `json:"securityOpt"`

	Features map[string]json.RawMessage `json:"features"`
}

// Build is the "build" object: how to produce an image from a Dockerfile.
type Build struct {
	Dockerfile string            `json:"dockerfile"`
	Context    string            `json:"context"`
	Args       map[string]string `json:"args"`
	Target     string            `json:"target"`
	Options    []string          `json:"options"`
	CacheFrom  StringOrSlice     `json:"cacheFrom"`
}

// StringOrSlice accepts either a JSON string or an array of strings. Several
// devcontainer properties are specified that way.
type StringOrSlice []string

func (s *StringOrSlice) UnmarshalJSON(b []byte) error {
	var one string
	if err := json.Unmarshal(b, &one); err == nil {
		*s = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return err
	}
	*s = many
	return nil
}

// Mount is one entry of "mounts". The spec allows each entry to be either a
// string in the engine's own `source=...,target=...` form or an object with the
// same fields, and real configurations use both.
type Mount struct {
	Type     string `json:"type"`
	Source   string `json:"source"`
	Target   string `json:"target"`
	ReadOnly bool   `json:"readonly"`
	// raw is kept when the entry arrived as a string, so it passes to the
	// engine exactly as written rather than through fields this struct may not
	// model.
	raw string
}

func (m *Mount) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		m.raw = s
		// Parsed as well as kept, so Target is available for conflict checks.
		for _, part := range strings.Split(s, ",") {
			k, v, ok := strings.Cut(part, "=")
			if !ok {
				continue
			}
			switch strings.TrimSpace(k) {
			case "type":
				m.Type = v
			case "source", "src":
				m.Source = v
			case "target", "dst", "destination":
				m.Target = v
			}
		}
		return nil
	}
	type plain Mount
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*m = Mount(p)
	return nil
}

// String renders a mount in the `--mount` form both podman and docker accept.
func (m Mount) String() string {
	if m.raw != "" {
		return m.raw
	}
	typ := m.Type
	if typ == "" {
		typ = "bind"
	}
	parts := []string{"type=" + typ}
	if m.Source != "" {
		parts = append(parts, "source="+m.Source)
	}
	if m.Target != "" {
		parts = append(parts, "target="+m.Target)
	}
	if m.ReadOnly {
		parts = append(parts, "readonly")
	}
	return strings.Join(parts, ",")
}

// acted is every top-level key this build applies. Anything in the file and
// not in here is reported by Unsupported.
var acted = map[string]bool{
	"name": true, "image": true, "build": true,
	"dockerFile": true, "context": true,
	"workspaceFolder": true, "workspaceMount": true,
	"mounts": true, "runArgs": true,
	"containerEnv": true, "remoteEnv": true,
	"containerUser": true, "remoteUser": true,
	"init": true, "privileged": true, "capAdd": true, "securityOpt": true,
	// Editor-only by design; nothing for a terminal harness to do.
	"customizations": true, "forwardPorts": true, "portsAttributes": true,
	"otherPortsAttributes": true,
}

// supportNote explains a key that is read but not acted on, so the report can
// distinguish "editor-specific" from "not yet".
var supportNote = map[string]string{
	"customizations":       "editor-specific",
	"forwardPorts":         "editor-specific",
	"portsAttributes":      "editor-specific",
	"otherPortsAttributes": "editor-specific",
}

// EnvRewrites names the environment entries whose ${localWorkspaceFolder} was
// resolved to the container's path instead of the host's, so the rewrite is
// reported rather than quietly disagreeing with the file.
func (c *Config) EnvRewrites() []string { return c.envRewrites }

// Unsupported lists the top-level keys present in the file that this build does
// not act on, sorted. Editor-only keys are omitted: a terminal has no extension
// list to configure.
func (c *Config) Unsupported() []string {
	var out []string
	for k := range c.Present {
		if acted[k] || supportNote[k] != "" {
			continue
		}
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Kind is how the container is obtained.
type Kind int

const (
	// KindImage runs a named image.
	KindImage Kind = iota
	// KindBuild builds an image from a Dockerfile.
	KindBuild
	// KindCompose is a docker-compose configuration.
	KindCompose
	// KindNone is a configuration that names no way to obtain a container.
	KindNone
)

// Kind reports how this configuration expects its container to be obtained.
// Compose is checked first -- a compose file with an image key is still
// compose -- and image wins over build, as the schema says.
func (c *Config) Kind() Kind {
	switch {
	case len(c.DockerComposeFile) > 0:
		return KindCompose
	case c.Image != "":
		return KindImage
	case c.Dockerfile() != "":
		return KindBuild
	default:
		return KindNone
	}
}

// Dockerfile returns the configured Dockerfile path as written, preferring the
// "build" object over the legacy top-level key.
func (c *Config) Dockerfile() string {
	if c.Build != nil && c.Build.Dockerfile != "" {
		return c.Build.Dockerfile
	}
	return c.DockerfileLegacy
}

// DockerfilePath returns the absolute path to the Dockerfile, resolving
// relative paths against the directory holding devcontainer.json.
func (c *Config) DockerfilePath() string {
	df := c.Dockerfile()
	if df == "" {
		return ""
	}
	if filepath.IsAbs(df) {
		return df
	}
	return filepath.Join(c.Dir, filepath.FromSlash(df))
}

// ContextPath returns the absolute build context directory, defaulting to the
// directory holding devcontainer.json -- not the project root, and not the
// Dockerfile's directory.
func (c *Config) ContextPath() string {
	ctx := c.ContextLegacy
	if c.Build != nil && c.Build.Context != "" {
		ctx = c.Build.Context
	}
	if ctx == "" {
		return c.Dir
	}
	if filepath.IsAbs(ctx) {
		return ctx
	}
	return filepath.Join(c.Dir, filepath.FromSlash(ctx))
}

// BuildArgs returns the build arguments in a stable order, so that a hash taken
// over them does not change when Go's map iteration does.
func (c *Config) BuildArgs() []string {
	if c.Build == nil || len(c.Build.Args) == 0 {
		return nil
	}
	keys := make([]string, 0, len(c.Build.Args))
	for k := range c.Build.Args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+c.Build.Args[k])
	}
	return out
}

// Load reads and parses a devcontainer.json.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw, path)
}

// Parse turns the bytes of a devcontainer.json into a Config.
func Parse(raw []byte, path string) (*Config, error) {
	stripped := stripComments(raw)

	cfg := &Config{Path: path, Dir: filepath.Dir(path)}
	if err := json.Unmarshal(stripped, cfg); err != nil {
		return nil, fmt.Errorf("%s: %w%s", path, err, trailingCommaHint(stripped, err))
	}
	if err := json.Unmarshal(stripped, &cfg.Present); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return cfg, nil
}

// trailingCommaHint explains the one mistake the format disallows:
// devcontainer.json permits comments but not trailing commas, so the bare
// "invalid character '}'" is worth elaborating.
func trailingCommaHint(b []byte, err error) string {
	var se *json.SyntaxError
	if !asSyntaxError(err, &se) {
		return ""
	}
	off := int(se.Offset) - 1
	if off < 0 || off >= len(b) {
		return ""
	}
	if b[off] != '}' && b[off] != ']' {
		return ""
	}
	for i := off - 1; i >= 0; i-- {
		switch b[i] {
		case ' ', '\t', '\r', '\n':
			continue
		case ',':
			return "\n\ndevcontainer.json allows comments but not trailing commas; there is one just before this point"
		default:
			return ""
		}
	}
	return ""
}

func asSyntaxError(err error, target **json.SyntaxError) bool {
	se, ok := err.(*json.SyntaxError)
	if ok {
		*target = se
	}
	return ok
}

// stripComments removes // and /* */ comments, replacing each byte with a space
// so offsets and line numbers survive and encoding/json's error positions still
// point into the original file. Comments inside strings are not comments.
func stripComments(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)

	const (
		code = iota
		inString
		inLine
		inBlock
	)
	state := code
	for i := 0; i < len(b); i++ {
		c := b[i]
		switch state {
		case code:
			switch {
			case c == '"':
				state = inString
			case c == '/' && i+1 < len(b) && b[i+1] == '/':
				out[i], out[i+1] = ' ', ' '
				i++
				state = inLine
			case c == '/' && i+1 < len(b) && b[i+1] == '*':
				out[i], out[i+1] = ' ', ' '
				i++
				state = inBlock
			}
		case inString:
			if c == '\\' && i+1 < len(b) {
				i++
				continue
			}
			if c == '"' {
				state = code
			}
		case inLine:
			if c == '\n' {
				state = code
				continue // keep the newline, so line numbers survive
			}
			out[i] = ' '
		case inBlock:
			if c == '*' && i+1 < len(b) && b[i+1] == '/' {
				out[i], out[i+1] = ' ', ' '
				i++
				state = code
				continue
			}
			if c != '\n' { // keep newlines
				out[i] = ' '
			}
		}
	}
	return out
}
