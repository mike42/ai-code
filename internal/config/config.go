// Package config loads ai-code's layered TOML configuration.
//
// Layers, lowest precedence first:
//
//	built-in defaults
//	$XDG_CONFIG_HOME/ai-code/config.toml   (or ~/.config/ai-code/config.toml)
//	./.ai-code/config.toml                 (project, walking up to the repo root)
//	AI_CODE_* environment variables
//	command-line flags                  (applied by the caller)
//
// Later layers override earlier ones key by key: a project file that sets only
// `default_model` does not discard the user's provider definitions.
package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

type Config struct {
	DefaultProvider string `toml:"default_provider"`
	DefaultMode     string `toml:"default_mode"`
	// Editor overrides $VISUAL and $EDITOR for the Ctrl-G / /editor escape.
	Editor string `toml:"editor"`

	UI       UI                  `toml:"ui"`
	Agent    Agent               `toml:"agent"`
	Tools    Tools               `toml:"tools"`
	Provider map[string]Provider `toml:"provider"`
	Mode     map[string]Mode     `toml:"mode"`

	// Sources records which files contributed, for `ai-code config`.
	Sources []string `toml:"-"`
}

type UI struct {
	// Theme is a chroma style name for syntax highlighting, or "none".
	Theme string `toml:"theme"`
	// Reasoning: "off" hides it, "collapsed" shows a dimmed single line,
	// "full" streams it dimmed.
	Reasoning string `toml:"reasoning"`
	// Color: "auto", "always", "never". Auto respects NO_COLOR and TTY detection.
	Color string `toml:"color"`
	// StatusLine draws the single redrawable line above the prompt.
	StatusLine *bool `toml:"status_line"`
	// ContextWarnPercent is when the status line starts warning.
	ContextWarnPercent int `toml:"context_warn_percent"`
}

type Agent struct {
	// MaxIterations caps the turns one task may take. Zero -- the default --
	// is unlimited; loop_guard and auto_compact bound the runaway cases. Set
	// it only where a turn count is itself what you want to limit, such as a
	// metered provider.
	MaxIterations int `toml:"max_iterations"`
	// MaxTokens is a ceiling on one response. Zero -- the default -- lets the
	// per-turn budget derived from the window stand on its own. Set it only
	// where a fixed ceiling is what you want, such as a metered provider.
	MaxTokens   int      `toml:"max_tokens"`
	Temperature *float64 `toml:"temperature"`
	TopP        *float64 `toml:"top_p"`
	// ParallelReads dispatches read-only tools concurrently within one turn.
	ParallelReads *bool `toml:"parallel_reads"`
	// LoopGuard aborts when the model repeats an identical tool call this many
	// times in a row.
	LoopGuard int `toml:"loop_guard"`
	// ContextOverride forces a context window when a backend reports nothing
	// usable, or when you know better than it does.
	ContextOverride int `toml:"context_override"`

	// AutoCompact summarises the session automatically when it reaches the
	// reserve, instead of stopping and asking. Default true.
	AutoCompact *bool `toml:"auto_compact"`
	// CompactReserveTokens is the room kept free at the top of the context
	// window. It is the only reason ai-code ever stops a response: generation runs
	// until the model is done or until the window is this close to full, and
	// then the session is compacted rather than truncated.
	//
	// It has to hold one full response plus the summarisation call that
	// compaction itself makes, or compaction would have no room to run at the
	// moment it is needed.
	//
	// Zero -- the default -- derives it from the detected window. Set it only
	// to override the formula for every model at once.
	CompactReserveTokens int `toml:"compact_reserve_tokens"`
	// CompactKeepRecentTokens is how much of the tail survives compaction
	// verbatim. Recent turns are what the model is in the middle of, and a
	// paraphrase loses the detail still in play.
	//
	// Zero -- the default -- derives it from the detected window, as a share
	// of the request budget, so the same config behaves sensibly on a 32k
	// model and a 262k one. Setting it also caps one tool result at half the
	// value. Set it only to override the formula for every model at once.
	CompactKeepRecentTokens int `toml:"compact_keep_recent_tokens"`

	// AutoCheckpoint summarises the session on its own while the prompt sits
	// idle, so a later model swap, a narrower window or a restart has a
	// summary ready instead of stopping to make one. Default true.
	//
	// A switch rather than a sentinel value of AutoCheckpointIdleDelay, whose
	// zero has its own meaning. False schedules nothing and sends nothing.
	AutoCheckpoint *bool `toml:"auto_checkpoint"`
	// AutoCheckpointIdleDelay is how long the prompt must sit untouched before
	// that summary is written. It is abandoned the moment a key is pressed,
	// and zero means write it as soon as the prompt appears.
	//
	// The default is DefaultAutoCheckpointIdleDelay. Lower it if the server
	// has lemonade's auto_evict enabled: the checkpoint wants to land while
	// the KV cache is still warm, and downsize_idle_timeout (60s by default)
	// is when the server drops it. The two numbers look unrelated and are not.
	AutoCheckpointIdleDelay Duration `toml:"auto_checkpoint_idle_delay"`

	// Thinking is the level to start a session at, one of EffortLadder:
	// none, minimal, low, medium, high, xhigh, max. Empty sends no field and
	// leaves the server's default alone. /think changes it mid-session.
	Thinking string `toml:"thinking"`
}

// DefaultAutoCheckpointIdleDelay is long enough to sit out the pauses within
// a working session and short enough to catch someone walking away. Nothing
// depends on the exact number.
const DefaultAutoCheckpointIdleDelay = 120 * time.Second

type Tools struct {
	Bash BashTool `toml:"bash"`
	Read ReadTool `toml:"read"`
	Grep GrepTool `toml:"grep"`
	Deny []string `toml:"deny"`
}

type BashTool struct {
	Timeout Duration `toml:"timeout"`
	// MaxOutputBytes bounds one command's output. Zero -- the default --
	// derives it from the detected window, so a result can never be a large
	// share of the room the session has to think in. Set it only to override
	// the formula for every model at once.
	MaxOutputBytes int    `toml:"max_output_bytes"`
	Shell          string `toml:"shell"`
}

type ReadTool struct {
	MaxBytes int `toml:"max_bytes"`
	MaxLines int `toml:"max_lines"`
}

type GrepTool struct {
	// Backend: "auto" uses ripgrep when on PATH, else the built-in walker;
	// "go" forces the built-in; "ripgrep" requires rg and errors without it.
	Backend    string `toml:"backend"`
	MaxResults int    `toml:"max_results"`
}

type Provider struct {
	Kind         string            `toml:"kind"`
	BaseURL      string            `toml:"base_url"`
	APIKey       string            `toml:"api_key"`
	Class        string            `toml:"class"`
	DefaultModel string            `toml:"default_model"`
	Headers      map[string]string `toml:"headers"`
	// Timeout bounds establishing the connection only -- DNS, TCP and TLS. It
	// deliberately does not bound generation: a large local model can stream a
	// single answer for a very long time and cutting it off is never the
	// behaviour anyone wanted.
	Timeout Duration `toml:"timeout"`

	TLSInsecure bool   `toml:"tls_insecure"`
	CAFile      string `toml:"ca_file"`
	CADir       string `toml:"ca_dir"`
	ClientCert  string `toml:"client_cert"`
	ClientKey   string `toml:"client_key"`

	SendReasoning bool `toml:"send_reasoning"`
	// PropagateModelSwap opts this provider into IPC model-swap broadcasts.
	PropagateModelSwap *bool `toml:"propagate_model_swap"`
}

// Mode is prompt steering and nothing else. Every tool remains available in
// every mode: gathering information legitimately involves running commands, and
// a mode that blocks writes just makes the model narrate what it would have
// done. Modes shape intent, they do not police it.
type Mode struct {
	Description string `toml:"description"`
	Prompt      string `toml:"prompt"`
}

// Duration accepts Go duration strings ("120s", "2m") in TOML.
type Duration struct{ time.Duration }

func (d *Duration) UnmarshalText(text []byte) error {
	s := string(text)
	v, err := time.ParseDuration(s)
	if err != nil {
		// Bare numbers are seconds, which is what people write by accident.
		if n, nerr := strconv.Atoi(s); nerr == nil {
			d.Duration = time.Duration(n) * time.Second
			return nil
		}
		return fmt.Errorf("invalid duration %q: want a value like \"120s\" or \"2m\"", s)
	}
	d.Duration = v
	return nil
}

func (d Duration) MarshalText() ([]byte, error) { return []byte(d.String()), nil }

func boolPtr(b bool) *bool { return &b }

// Defaults returns the built-in configuration. ai-code is usable with no config
// file at all except for a provider definition, which cannot be guessed.
func Defaults() Config {
	return Config{
		DefaultMode: "build",
		UI: UI{
			Theme:              "auto",
			Reasoning:          "collapsed",
			Color:              "auto",
			StatusLine:         boolPtr(true),
			ContextWarnPercent: 80,
		},
		Agent: Agent{
			MaxIterations: 0, // unlimited. See Agent.MaxIterations.
			MaxTokens:     0, // no cap: the server decides. See Agent.MaxTokens.
			AutoCompact:   boolPtr(true),
			// Both zero: derived from the detected window. See
			// agent.BudgetFor. A number here overrides the formula globally.
			CompactReserveTokens:    0,
			CompactKeepRecentTokens: 0,
			AutoCheckpoint:          boolPtr(true),
			AutoCheckpointIdleDelay: Duration{DefaultAutoCheckpointIdleDelay},
			ParallelReads:           boolPtr(true),
			LoopGuard:               4,
		},
		Tools: Tools{
			Bash: BashTool{Timeout: Duration{120 * time.Second}, MaxOutputBytes: 0},
			Read: ReadTool{MaxBytes: 400_000, MaxLines: 2000},
			Grep: GrepTool{Backend: "auto", MaxResults: 200},
		},
		Provider: map[string]Provider{},
		Mode:     builtinModes(),
	}
}

func builtinModes() map[string]Mode {
	return map[string]Mode{
		"build": {
			Description: "Default. Investigate as needed, then make the change.",
			Prompt:      "",
		},
		"research": {
			Description: "Front-load investigation before proposing a change.",
			Prompt: strings.TrimSpace(`
Front-load your investigation. Read widely and run whatever commands help you
understand the system before you propose anything: reading files, running tests,
inspecting git history and writing scratch scripts are all fair game, and you do
not need permission for any of them.

State your understanding and your proposed approach before you change project
files. This is about sequencing, not permission -- you have every tool available.
`),
		},
		"review": {
			Description: "Critique the work rather than extending it.",
			Prompt: strings.TrimSpace(`
You are reviewing, not building. Read the code and report what you find:
correctness bugs first, then simplifications worth making. Run whatever commands
help you verify a suspicion -- an unverified review finding is a guess.

Report findings; do not fix them unless asked.
`),
		},
	}
}

// Load reads the configuration layers for a project directory.
func Load(projectDir string) (*Config, error) {
	cfg := Defaults()

	for _, path := range candidatePaths(projectDir) {
		if err := overlayFile(&cfg, path); err != nil {
			return nil, err
		}
	}
	if err := overlayEnv(&cfg); err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// UserConfigPath is where ai-code's own configuration lives.
func UserConfigPath() string {
	if dir := os.Getenv("AI_CODE_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "config.toml")
	}
	if dir := os.Getenv("XDG_CONFIG_HOME"); dir != "" {
		return filepath.Join(dir, "ai-code", "config.toml")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".config", "ai-code", "config.toml")
}

// noCloudMarker is the name of the file whose presence marks a directory tree
// as off-limits to cloud providers. It is deliberately checked as a file in
// the working directory and every parent up to the filesystem root, so a
// single .nocloud at the top of a repository, a home directory, or a
// container's root guards every project beneath it.
const noCloudMarker = ".nocloud"

// NoCloud reports whether a .nocloud file is present in dir or any of its
// parents. When it is true, the session must restrict itself to on-premises
// providers.
func NoCloud(dir string) bool {
	if dir == "" {
		return false
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	for p := abs; ; {
		if _, err := os.Stat(filepath.Join(p, noCloudMarker)); err == nil {
			return true
		}
		parent := filepath.Dir(p)
		if parent == p {
			break
		}
		p = parent
	}
	return false
}

func candidatePaths(projectDir string) []string {
	var paths []string
	if p := UserConfigPath(); p != "" {
		paths = append(paths, p)
	}
	if projectDir != "" {
		paths = append(paths, filepath.Join(projectDir, ".ai-code", "config.toml"))
	}
	return paths
}

// overlayFile merges one file over cfg. BurntSushi/toml assigns only the keys
// present in the document, so absent keys retain the lower layer's value --
// except for maps, where a whole entry would be replaced. Provider and Mode are
// therefore merged field by field.
func overlayFile(cfg *Config, path string) error {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}

	existingProviders := cfg.Provider
	existingModes := cfg.Mode
	cfg.Provider = map[string]Provider{}
	cfg.Mode = map[string]Mode{}

	if _, err := toml.Decode(string(raw), cfg); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}

	cfg.Provider = mergeProviders(existingProviders, cfg.Provider)
	cfg.Mode = mergeModes(existingModes, cfg.Mode)
	cfg.Sources = append(cfg.Sources, path)
	return nil
}

func mergeProviders(base, overlay map[string]Provider) map[string]Provider {
	out := make(map[string]Provider, len(base)+len(overlay))
	for k, v := range base {
		out[k] = v
	}
	for k, ov := range overlay {
		bv, exists := out[k]
		if !exists {
			out[k] = ov
			continue
		}
		out[k] = mergeProvider(bv, ov)
	}
	return out
}

func mergeProvider(base, ov Provider) Provider {
	m := base
	if ov.Kind != "" {
		m.Kind = ov.Kind
	}
	if ov.BaseURL != "" {
		m.BaseURL = ov.BaseURL
	}
	if ov.APIKey != "" {
		m.APIKey = ov.APIKey
	}
	if ov.Class != "" {
		m.Class = ov.Class
	}
	if ov.DefaultModel != "" {
		m.DefaultModel = ov.DefaultModel
	}
	if ov.Timeout.Duration != 0 {
		m.Timeout = ov.Timeout
	}
	if ov.CAFile != "" {
		m.CAFile = ov.CAFile
	}
	if ov.CADir != "" {
		m.CADir = ov.CADir
	}
	if ov.ClientCert != "" {
		m.ClientCert = ov.ClientCert
	}
	if ov.ClientKey != "" {
		m.ClientKey = ov.ClientKey
	}
	if ov.TLSInsecure {
		m.TLSInsecure = true
	}
	if ov.SendReasoning {
		m.SendReasoning = true
	}
	if ov.PropagateModelSwap != nil {
		m.PropagateModelSwap = ov.PropagateModelSwap
	}
	if len(ov.Headers) > 0 {
		if m.Headers == nil {
			m.Headers = map[string]string{}
		}
		for k, v := range ov.Headers {
			m.Headers[k] = v
		}
	}
	return m
}

func mergeModes(base, overlay map[string]Mode) map[string]Mode {
	out := make(map[string]Mode, len(base)+len(overlay))
	for k, v := range base {
		out[k] = v
	}
	for k, ov := range overlay {
		bv := out[k]
		if ov.Description != "" {
			bv.Description = ov.Description
		}
		if ov.Prompt != "" {
			bv.Prompt = ov.Prompt
		}
		out[k] = bv
	}
	return out
}

// overlayEnv applies AI_CODE_* overrides. Kept deliberately small: environment
// configuration is for the handful of things CI and scripts need to change.
func overlayEnv(cfg *Config) error {
	if v := os.Getenv("AI_CODE_PROVIDER"); v != "" {
		cfg.DefaultProvider = v
	}
	if v := os.Getenv("AI_CODE_MODE"); v != "" {
		cfg.DefaultMode = v
	}
	if v := os.Getenv("AI_CODE_EDITOR"); v != "" {
		cfg.Editor = v
	}
	if v := os.Getenv("AI_CODE_MODEL"); v != "" {
		p := cfg.DefaultProvider
		if p == "" {
			return errors.New("AI_CODE_MODEL is set but no default provider is configured")
		}
		pv := cfg.Provider[p]
		pv.DefaultModel = v
		cfg.Provider[p] = pv
	}
	if os.Getenv("NO_COLOR") != "" {
		cfg.UI.Color = "never"
	}
	return nil
}

func (c *Config) Validate() error {
	if len(c.Provider) == 0 {
		return nil // Reported with better guidance at the point of use.
	}
	names := make([]string, 0, len(c.Provider))
	for name := range c.Provider {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		p := c.Provider[name]
		if p.BaseURL == "" {
			return fmt.Errorf("provider %q: base_url is required", name)
		}
		switch p.Kind {
		case "", "openai", "lemonade", "openrouter":
		default:
			return fmt.Errorf("provider %q: unknown kind %q (want \"openai\", \"lemonade\" or \"openrouter\")", name, p.Kind)
		}
		if p.Class == "" {
			return fmt.Errorf("provider %q: class is required; set class = \"on-premises\" or class = \"cloud\". "+
				"ai-code uses this to decide when sending your session off-site needs confirmation", name)
		}
		if p.Class != "cloud" && p.Class != "on-premises" {
			return fmt.Errorf("provider %q: class must be \"cloud\" or \"on-premises\", got %q", name, p.Class)
		}
	}

	if c.DefaultProvider != "" {
		if _, ok := c.Provider[c.DefaultProvider]; !ok {
			return fmt.Errorf("default_provider %q is not defined; configured providers: %s",
				c.DefaultProvider, strings.Join(names, ", "))
		}
	}
	if c.DefaultMode != "" {
		if _, ok := c.Mode[c.DefaultMode]; !ok {
			return fmt.Errorf("default_mode %q is not defined", c.DefaultMode)
		}
	}
	return nil
}

// ResolveAPIKey expands the "env:NAME" indirection so keys need not sit in a
// config file. A literal value is returned unchanged.
func (p Provider) ResolveAPIKey() (string, error) {
	if name, ok := strings.CutPrefix(p.APIKey, "env:"); ok {
		v := os.Getenv(name)
		if v == "" {
			return "", fmt.Errorf("api_key refers to environment variable %s, which is unset", name)
		}
		return v, nil
	}
	return p.APIKey, nil
}

// ProviderNames returns configured provider names in a stable order.
func (c *Config) ProviderNames() []string {
	names := make([]string, 0, len(c.Provider))
	for n := range c.Provider {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// OnPremisesOnly returns a copy of the config restricted to providers whose
// class is on-premises. Used when a .nocloud file marks the tree as off-limits
// to cloud providers.
func (c *Config) OnPremisesOnly() *Config {
	out := *c
	out.Provider = map[string]Provider{}
	for n, p := range c.Provider {
		if p.Class == "on-premises" {
			out.Provider[n] = p
		}
	}
	if out.DefaultProvider != "" {
		if _, ok := out.Provider[out.DefaultProvider]; !ok {
			out.DefaultProvider = ""
		}
	}
	return &out
}

// ModeNames returns configured mode names in a stable order.
func (c *Config) ModeNames() []string {
	names := make([]string, 0, len(c.Mode))
	for n := range c.Mode {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
