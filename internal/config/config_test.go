package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// write creates a config file and returns the directory it lives in.
func write(t *testing.T, dir, rel, body string) {
	t.Helper()
	full := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func setupLayers(t *testing.T, userTOML, projectTOML string) *Config {
	t.Helper()
	userDir := t.TempDir()
	projDir := t.TempDir()
	t.Setenv("AI_CODE_CONFIG_DIR", userDir)

	if userTOML != "" {
		write(t, userDir, "config.toml", userTOML)
	}
	if projectTOML != "" {
		write(t, projDir, ".ai-code/config.toml", projectTOML)
	}
	cfg, err := Load(projDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return cfg
}

// A project file that sets one key must not wipe out the provider definitions
// in the home config.
func TestProjectLayerDoesNotDiscardUserProviders(t *testing.T) {
	cfg := setupLayers(t,
		`
default_provider = "home"

[provider.home]
kind = "lemonade"
base_url = "https://ai.example.internal/api/v1"
class = "on-premises"
tls_insecure = true
default_model = "Qwen3.6-27B"
`,
		`
default_mode = "research"
`)

	p, ok := cfg.Provider["home"]
	if !ok {
		t.Fatal("project layer discarded the user's provider definition")
	}
	if p.BaseURL != "https://ai.example.internal/api/v1" {
		t.Errorf("base_url = %q, want the user layer's value", p.BaseURL)
	}
	if !p.TLSInsecure {
		t.Error("tls_insecure was lost across the project layer")
	}
	if cfg.DefaultProvider != "home" {
		t.Errorf("default_provider = %q, want %q", cfg.DefaultProvider, "home")
	}
	if cfg.DefaultMode != "research" {
		t.Errorf("default_mode = %q, want the project layer's value", cfg.DefaultMode)
	}
}

func TestProjectLayerMergesIntoExistingProvider(t *testing.T) {
	// A project overriding just the model must keep the URL, class and TLS
	// settings from the user config.
	cfg := setupLayers(t,
		`
[provider.home]
kind = "lemonade"
base_url = "https://ai.example.internal/api/v1"
class = "on-premises"
tls_insecure = true
default_model = "Qwen3.6-27B"
`,
		`
[provider.home]
default_model = "Qwen3.8-27B-Coder"
`)

	p := cfg.Provider["home"]
	if p.DefaultModel != "Qwen3.8-27B-Coder" {
		t.Errorf("default_model = %q, want the project override", p.DefaultModel)
	}
	if p.BaseURL != "https://ai.example.internal/api/v1" {
		t.Errorf("base_url = %q, want it preserved from the user layer", p.BaseURL)
	}
	if p.Kind != "lemonade" {
		t.Errorf("kind = %q, want it preserved from the user layer", p.Kind)
	}
	if !p.TLSInsecure {
		t.Error("tls_insecure was lost when the project layer overrode one key")
	}
}

func TestDefaultsSurviveWhenUnset(t *testing.T) {
	cfg := setupLayers(t, `
[agent]
max_iterations = 7
`, "")

	if cfg.Agent.MaxIterations != 7 {
		t.Errorf("max_iterations = %d, want 7", cfg.Agent.MaxIterations)
	}
	if cfg.Agent.MaxTokens != Defaults().Agent.MaxTokens {
		t.Errorf("max_tokens = %d, want the default %d preserved",
			cfg.Agent.MaxTokens, Defaults().Agent.MaxTokens)
	}
	if cfg.Tools.Bash.Timeout.Duration != 120*time.Second {
		t.Errorf("bash timeout = %v, want the default", cfg.Tools.Bash.Timeout.Duration)
	}
}

func TestBuiltinModesAreAlwaysAvailable(t *testing.T) {
	cfg := setupLayers(t, "", "")
	for _, want := range []string{"build", "research", "review"} {
		if _, ok := cfg.Mode[want]; !ok {
			t.Errorf("built-in mode %q missing", want)
		}
	}
}

func TestUserModeOverridesBuiltinPromptOnly(t *testing.T) {
	cfg := setupLayers(t, `
[mode.research]
prompt = "my own steering text"
`, "")

	m := cfg.Mode["research"]
	if m.Prompt != "my own steering text" {
		t.Errorf("prompt = %q, want the user's override", m.Prompt)
	}
	if m.Description == "" {
		t.Error("description was cleared; unset keys should keep the built-in value")
	}
}

func TestDurationParsing(t *testing.T) {
	cfg := setupLayers(t, `
[tools.bash]
timeout = "2m"
`, "")
	if got := cfg.Tools.Bash.Timeout.Duration; got != 2*time.Minute {
		t.Errorf("timeout = %v, want 2m", got)
	}
}

func TestMissingClassIsRejectedWithGuidance(t *testing.T) {
	userDir := t.TempDir()
	t.Setenv("AI_CODE_CONFIG_DIR", userDir)
	write(t, userDir, "config.toml", `
[provider.home]
base_url = "https://ai.example.internal/api/v1"
`)
	_, err := Load(t.TempDir())
	if err == nil {
		t.Fatal("expected an error for a provider with no class")
	}
	// The message must say what to do, not just what is wrong.
	if !contains(err.Error(), "on-premises") || !contains(err.Error(), "cloud") {
		t.Errorf("error message does not name the valid values: %v", err)
	}
}

func TestUnknownDefaultProviderNamesTheAlternatives(t *testing.T) {
	userDir := t.TempDir()
	t.Setenv("AI_CODE_CONFIG_DIR", userDir)
	write(t, userDir, "config.toml", `
default_provider = "typo"

[provider.home]
base_url = "https://ai.example.internal/api/v1"
class = "on-premises"
`)
	_, err := Load(t.TempDir())
	if err == nil {
		t.Fatal("expected an error for an undefined default_provider")
	}
	if !contains(err.Error(), "home") {
		t.Errorf("error should list configured providers, got: %v", err)
	}
}

func TestResolveAPIKeyFromEnv(t *testing.T) {
	t.Setenv("SOME_KEY", "sk-secret")
	p := Provider{APIKey: "env:SOME_KEY"}
	got, err := p.ResolveAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	if got != "sk-secret" {
		t.Errorf("got %q, want the environment value", got)
	}

	p = Provider{APIKey: "env:DEFINITELY_UNSET_XYZ"}
	if _, err := p.ResolveAPIKey(); err == nil {
		t.Error("expected an error when the named variable is unset")
	}
}

func TestEnvOverrides(t *testing.T) {
	t.Setenv("AI_CODE_MODE", "review")
	cfg := setupLayers(t, `default_mode = "build"`, "")
	if cfg.DefaultMode != "review" {
		t.Errorf("default_mode = %q, want the AI_CODE_MODE override", cfg.DefaultMode)
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (func() bool {
		for i := 0; i+len(needle) <= len(haystack); i++ {
			if haystack[i:i+len(needle)] == needle {
				return true
			}
		}
		return false
	})()
}

func TestNoCloudDetectsInWorkingDir(t *testing.T) {
	dir := t.TempDir()
	if NoCloud(dir) {
		t.Fatal("NoCloud true on a clean directory")
	}
	if err := os.WriteFile(filepath.Join(dir, noCloudMarker), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}
	if !NoCloud(dir) {
		t.Fatal("NoCloud false when .nocloud is present in the working dir")
	}
}

func TestNoCloudDetectsInParents(t *testing.T) {
	root := t.TempDir()
	child := filepath.Join(root, "a", "b")
	if err := os.MkdirAll(child, 0o755); err != nil {
		t.Fatal(err)
	}
	// A .nocloud at the root must guard a deep descendant.
	if err := os.WriteFile(filepath.Join(root, noCloudMarker), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}
	if !NoCloud(child) {
		t.Fatal("NoCloud false when .nocloud is present in a parent dir")
	}

	// An entirely separate tree without a marker must not be flagged.
	other := t.TempDir()
	nested := filepath.Join(other, "a", "b")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	if NoCloud(nested) {
		t.Fatal("NoCloud true when no .nocloud is in the tree")
	}
}

func TestOnPremisesOnlyDropsCloudProviders(t *testing.T) {
	cfg := Defaults()
	cfg.DefaultProvider = "cloudy"
	cfg.Provider = map[string]Provider{
		"home":   {BaseURL: "https://ai.example.internal", Class: "on-premises"},
		"cloudy": {BaseURL: "https://api.example.com", Class: "cloud"},
	}

	restricted := cfg.OnPremisesOnly()
	if _, ok := restricted.Provider["cloudy"]; ok {
		t.Error("cloud provider survived OnPremisesOnly")
	}
	if _, ok := restricted.Provider["home"]; !ok {
		t.Error("on-premises provider was dropped by OnPremisesOnly")
	}
	if restricted.DefaultProvider != "" {
		t.Errorf("default_provider = %q, want empty after the cloud default was dropped", restricted.DefaultProvider)
	}
}
