package devcontainer

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestCommentsAreStripped(t *testing.T) {
	raw := []byte(`{
  // the image to run
  "image": "ubuntu:24.04", /* inline */
  "name": "with // comments in a string"
}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Image != "ubuntu:24.04" {
		t.Errorf("Image = %q", c.Image)
	}
	// A comment marker inside a string is part of the string, not a comment.
	if c.Name != "with // comments in a string" {
		t.Errorf("Name = %q", c.Name)
	}
}

// Scanning for the text `"image"` would match a key of that name nested
// anywhere else in the file.
func TestNestedImageKeyDoesNotWin(t *testing.T) {
	raw := []byte(`{
  "build": {"dockerfile": "Dockerfile"},
  "customizations": {"vscode": {"settings": {"image": "not-the-image"}}}
}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Image != "" {
		t.Errorf("Image = %q, want empty: the only \"image\" is nested under customizations", c.Image)
	}
	if c.Kind() != KindBuild {
		t.Errorf("Kind() = %v, want KindBuild", c.Kind())
	}
}

// A commented-out image must not be found either.
func TestCommentedImageIsNotTheImage(t *testing.T) {
	raw := []byte(`{
  // "image": "ubuntu:24.04",
  "build": {"dockerfile": "Dockerfile"}
}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Image != "" {
		t.Errorf("Image = %q, want empty: that line is a comment", c.Image)
	}
}

func TestLegacyDockerfileKey(t *testing.T) {
	raw := []byte(`{"dockerFile": "Dockerfile", "context": ".."}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Kind() != KindBuild {
		t.Errorf("Kind() = %v, want KindBuild", c.Kind())
	}
	if got, want := c.DockerfilePath(), filepath.FromSlash("/p/.devcontainer/Dockerfile"); got != want {
		t.Errorf("DockerfilePath() = %q, want %q", got, want)
	}
	if got, want := c.ContextPath(), filepath.FromSlash("/p"); got != want {
		t.Errorf("ContextPath() = %q, want %q", got, want)
	}
}

// Relative paths resolve against the directory holding devcontainer.json, not
// the project root and not the Dockerfile's own directory.
func TestBuildPathsResolveAgainstTheConfigDirectory(t *testing.T) {
	raw := []byte(`{"build": {"dockerfile": "docker/Dockerfile", "context": ".."}}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, want := c.DockerfilePath(), filepath.FromSlash("/p/.devcontainer/docker/Dockerfile"); got != want {
		t.Errorf("DockerfilePath() = %q, want %q", got, want)
	}
	if got, want := c.ContextPath(), filepath.FromSlash("/p"); got != want {
		t.Errorf("ContextPath() = %q, want %q", got, want)
	}
}

// With no "context", the context is the config's own directory.
func TestContextDefaultsToTheConfigDirectory(t *testing.T) {
	c, err := Parse([]byte(`{"build": {"dockerfile": "Dockerfile"}}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, want := c.ContextPath(), filepath.FromSlash("/p/.devcontainer"); got != want {
		t.Errorf("ContextPath() = %q, want %q", got, want)
	}
}

func TestComposeIsRecognisedAsCompose(t *testing.T) {
	raw := []byte(`{"dockerComposeFile": "compose.yml", "service": "app", "workspaceFolder": "/src"}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Kind() != KindCompose {
		t.Errorf("Kind() = %v, want KindCompose", c.Kind())
	}
}

func TestEmptyConfigNamesNothing(t *testing.T) {
	c, err := Parse([]byte(`{"name": "empty"}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Kind() != KindNone {
		t.Errorf("Kind() = %v, want KindNone", c.Kind())
	}
}

func TestMountsAcceptBothForms(t *testing.T) {
	raw := []byte(`{
  "image": "x",
  "mounts": [
    "source=/host/a,target=/c/a,type=bind",
    {"source": "/host/b", "target": "/c/b", "type": "volume"}
  ]
}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(c.Mounts) != 2 {
		t.Fatalf("got %d mounts, want 2", len(c.Mounts))
	}
	// The string form is passed through as written.
	if got := c.Mounts[0].String(); got != "source=/host/a,target=/c/a,type=bind" {
		t.Errorf("string mount rendered as %q", got)
	}
	// ...but is still parsed, so the target is known.
	if c.Mounts[0].Target != "/c/a" {
		t.Errorf("string mount target = %q, want /c/a", c.Mounts[0].Target)
	}
	if got := c.Mounts[1].String(); !strings.Contains(got, "type=volume") || !strings.Contains(got, "target=/c/b") {
		t.Errorf("object mount rendered as %q", got)
	}
}

func TestCacheFromAcceptsStringOrArray(t *testing.T) {
	for _, raw := range []string{
		`{"build": {"dockerfile": "Dockerfile", "cacheFrom": "a"}}`,
		`{"build": {"dockerfile": "Dockerfile", "cacheFrom": ["a"]}}`,
	} {
		c, err := Parse([]byte(raw), "/p/.devcontainer/devcontainer.json")
		if err != nil {
			t.Fatalf("Parse(%s): %v", raw, err)
		}
		if len(c.Build.CacheFrom) != 1 || c.Build.CacheFrom[0] != "a" {
			t.Errorf("Parse(%s): CacheFrom = %v", raw, c.Build.CacheFrom)
		}
	}
}

// Keys that were read but not acted on are named, so a half-applied
// configuration is never silent. Keys that are meaningless here are not.
func TestUnsupportedKeysAreNamed(t *testing.T) {
	raw := []byte(`{
  "image": "x",
  "features": {"ghcr.io/devcontainers/features/go:1": {}},
  "postCreateCommand": "make",
  "customizations": {"vscode": {}},
  "forwardPorts": [8080]
}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got := strings.Join(c.Unsupported(), ",")
	if got != "features,postCreateCommand" {
		t.Errorf("Unsupported() = %q, want %q", got, "features,postCreateCommand")
	}
}

// devcontainer.json allows comments but not trailing commas, which is a
// surprising enough combination to be worth explaining.
func TestTrailingCommaIsExplained(t *testing.T) {
	_, err := Parse([]byte("{\n  \"image\": \"x\",\n}"), "/p/.devcontainer/devcontainer.json")
	if err == nil {
		t.Fatal("want an error for a trailing comma")
	}
	if !strings.Contains(err.Error(), "trailing comma") {
		t.Errorf("error = %q, want it to mention trailing commas", err)
	}
}

// Comment stripping must not move anything: encoding/json reports byte offsets
// and they have to still point into the original file.
func TestStrippingPreservesOffsets(t *testing.T) {
	raw := []byte("{\n // c\n \"image\": \"x\"\n}")
	if got := len(stripComments(raw)); got != len(raw) {
		t.Errorf("stripped length = %d, want %d", got, len(raw))
	}
}

func TestBuildArgsAreStablyOrdered(t *testing.T) {
	raw := []byte(`{"build": {"dockerfile": "Dockerfile", "args": {"B": "2", "A": "1", "C": "3"}}}`)
	c, err := Parse(raw, "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	// Run it repeatedly: an unstable order would only show up sometimes, and
	// it would show up as an image tag that changes for no reason.
	for i := 0; i < 50; i++ {
		got := strings.Join(c.BuildArgs(), ",")
		if got != "A=1,B=2,C=3" {
			t.Fatalf("BuildArgs() = %q", got)
		}
	}
}
