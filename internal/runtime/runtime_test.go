package runtime

import (
	"ai-code/internal/devcontainer"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFlag(t *testing.T) {
	tests := []struct {
		in      string
		want    Kind
		wantErr bool
	}{
		{"", "", false},
		{"devcontainer", KindDevcontainer, false},
		{"host", KindHost, false},
		{"ssh://buildvm", KindSSH, false},
		{"ssh://user@vm:2222", KindSSH, false},
		{"ssh://", "", true},
		{"ssh://agent:secret@vm", "", true},
		{"ssh://vm:notaport", "", true},
		{"ssh://vm/work?x=1", "", true},
		{"bogus", "", true},
	}
	for _, tc := range tests {
		r, err := ParseFlag(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("ParseFlag(%q): expected error", tc.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseFlag(%q): unexpected error: %v", tc.in, err)
			continue
		}
		if tc.want == "" {
			if r != nil {
				t.Errorf("ParseFlag(%q): want nil, got %+v", tc.in, r)
			}
			continue
		}
		if r == nil || r.Kind != tc.want {
			t.Errorf("ParseFlag(%q): got %+v, want kind %q", tc.in, r, tc.want)
		}
	}
}

func TestResolveDefaultsToDevcontainer(t *testing.T) {
	dir := t.TempDir()
	dc := filepath.Join(dir, ".devcontainer")
	if err := os.MkdirAll(dc, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dc, "devcontainer.json"),
		[]byte(`{"image":"mcr.microsoft.com/devcontainers/base:ubuntu"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	r, err := Resolve("", dir)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if r.Kind != KindDevcontainer || r.Explicit {
		t.Errorf("got %+v, want implicit devcontainer", r)
	}
	if r.Image == "?" {
		t.Errorf("image not parsed: %q", r.Image)
	}
}

func TestResolveNoRuntimeWithoutConfig(t *testing.T) {
	dir := t.TempDir()
	if _, err := Resolve("", dir); err == nil {
		t.Fatal("expected an error when no flag and no devcontainer")
	} else if _, ok := err.(ErrNoRuntime); !ok {
		t.Errorf("want ErrNoRuntime, got %T: %v", err, err)
	}
}

func TestResolveHostIsExplicit(t *testing.T) {
	dir := t.TempDir()
	dc := filepath.Join(dir, ".devcontainer")
	if err := os.MkdirAll(dc, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dc, "devcontainer.json"),
		[]byte(`{"image":"x"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	r, err := Resolve("host", dir)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if r.Kind != KindHost || !r.Explicit {
		t.Errorf("got %+v, want explicit host", r)
	}
	if r.Sandboxed() {
		t.Error("host runtime should not be sandboxed")
	}
}

func TestResolveDevcontainerMissing(t *testing.T) {
	dir := t.TempDir()
	if _, err := Resolve("devcontainer", dir); err == nil {
		t.Fatal("expected an error for explicit devcontainer with no config")
	} else if _, ok := err.(ErrNoDevcontainer); !ok {
		t.Errorf("want ErrNoDevcontainer, got %T: %v", err, err)
	}
}

func TestDevcontainerPathSearchesParents(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	dc := filepath.Join(root, ".devcontainer.json")
	if err := os.WriteFile(dc, []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DevcontainerPath(sub); got != dc {
		t.Errorf("DevcontainerPath = %q, want %q", got, dc)
	}
}

func TestExplicitDevcontainerFlagStillResolvesTheImage(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".devcontainer"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := filepath.Join(dir, ".devcontainer", "devcontainer.json")
	if err := os.WriteFile(cfg, []byte(`{"image": "mcr.microsoft.com/devcontainers/base:ubuntu"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	explicit, err := Resolve("devcontainer", dir)
	if err != nil {
		t.Fatalf("explicit: %v", err)
	}
	implicit, err := Resolve("", dir)
	if err != nil {
		t.Fatalf("implicit: %v", err)
	}

	const want = "mcr.microsoft.com/devcontainers/base:ubuntu"
	if explicit.Image != want {
		t.Errorf("explicit --runtime devcontainer gave image %q, want %q", explicit.Image, want)
	}
	if explicit.Image != implicit.Image {
		t.Errorf("explicit gave %q but implicit gave %q", explicit.Image, implicit.Image)
	}
}

func TestDockerfileConfigIsABuildNotAMissingImage(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".devcontainer"), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := filepath.Join(dir, ".devcontainer", "devcontainer.json")
	if err := os.WriteFile(cfg, []byte(`{"build": {"dockerfile": "Dockerfile"}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	rt, err := Resolve("devcontainer", dir)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	// There is no image to name; "?" would be a valid-looking but wrong one.
	if rt.Image != "" {
		t.Errorf("Image = %q, want empty: this configuration names no image", rt.Image)
	}
	if rt.Config == nil {
		t.Fatal("Config is nil; the configuration was not parsed")
	}
	if got := rt.Config.Kind(); got != devcontainer.KindBuild {
		t.Errorf("Kind() = %v, want KindBuild", got)
	}
	if got := rt.Config.Dockerfile(); got != "Dockerfile" {
		t.Errorf("Dockerfile() = %q, want %q", got, "Dockerfile")
	}
	if got := rt.Describe(); strings.Contains(got, "no image") {
		t.Errorf("Describe() = %q, want it to describe the build", got)
	}
}

func TestParseSSHSplitsDestinationAndDirectory(t *testing.T) {
	tests := []struct {
		in                    string
		user, host, port, dir string
		remote, flag          string
	}{
		{"ssh://buildvm", "", "buildvm", "", "", "buildvm", "ssh://buildvm"},
		{"ssh://agent@vm:2222/home/agent/work/", "agent", "vm", "2222", "/home/agent/work",
			"agent@vm:2222", "ssh://agent@vm:2222/home/agent/work"},
		{"ssh://agent@vm/", "agent", "vm", "", "", "agent@vm", "ssh://agent@vm"},
		{"ssh://[::1]:22/srv", "", "::1", "22", "/srv", "[::1]:22", "ssh://[::1]:22/srv"},
	}
	for _, tc := range tests {
		r, err := ParseFlag(tc.in)
		if err != nil {
			t.Fatalf("ParseFlag(%q): %v", tc.in, err)
		}
		got := []string{r.SSHUser, r.SSHHost, r.SSHPort, r.SSHDir, r.Remote, r.FlagValue()}
		want := []string{tc.user, tc.host, tc.port, tc.dir, tc.remote, tc.flag}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("ParseFlag(%q) = %q, want %q", tc.in, got, want)
				break
			}
		}
	}
}

func TestResolveAcceptsSSH(t *testing.T) {
	r, err := Resolve("ssh://agent@vm/work", t.TempDir())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if r.Kind != KindSSH || r.SSHDir != "/work" {
		t.Errorf("Resolve = %+v", r)
	}
}
