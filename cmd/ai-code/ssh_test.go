package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"ai-code/internal/runtime"
	"ai-code/internal/worker"
)

func TestPlatformOf(t *testing.T) {
	for in, want := range map[string]string{
		"Linux x86_64\n": "linux/amd64",
		"Darwin arm64":   "darwin/arm64",
		"Linux aarch64":  "linux/arm64",
		"FreeBSD amd64":  "freebsd/amd64",
	} {
		goos, goarch, err := platformOf(in)
		if err != nil || goos+"/"+goarch != want {
			t.Errorf("platformOf(%q) = %s/%s, %v; want %s", in, goos, goarch, err, want)
		}
	}
	for _, in := range []string{"MINGW64_NT-10.0 x86_64", "Linux sparc64", "", "Linux"} {
		if _, _, err := platformOf(in); err == nil {
			t.Errorf("platformOf(%q) accepted", in)
		}
	}
}

func TestShQuoteSurvivesTheShell(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("no sh")
	}
	for _, s := range []string{"plain", "it's", `a "b" $HOME $(id) ;|&`, "new\nline"} {
		out, err := exec.Command("sh", "-c", "printf %s "+shQuote(s)).Output()
		if err != nil || string(out) != s {
			t.Errorf("%q came back as %q (%v)", s, out, err)
		}
	}
}

func TestSSHArgvKeepsTheHostOutOfOptionPosition(t *testing.T) {
	rt, err := runtime.ParseFlag("ssh://agent@vm:2222/srv/work")
	if err != nil {
		t.Fatal(err)
	}
	e := NewSSHExecutor(rt, "/cfg", worker.Settings{})
	got := strings.Join(e.sshArgv("true"), " ")
	want := "-T -o BatchMode=yes -o ClearAllForwardings=yes -o ForwardAgent=no -o ForwardX11=no " +
		"-o PermitLocalCommand=no -F /cfg -p 2222 -l agent -- vm sh -c 'true'"
	if got != want {
		t.Errorf("argv %q, want %q", got, want)
	}
}

func runProbe(t *testing.T, dir string) *remoteFacts {
	t.Helper()
	out, err := exec.Command("sh", "-c", probeScript(dir)).Output()
	if err != nil {
		t.Fatalf("probe script failed: %v", err)
	}
	facts, err := parseProbe(string(out))
	if err != nil {
		t.Fatalf("parseProbe(%q): %v", out, err)
	}
	return facts
}

func TestTheProbeFindsNoCloudAboveTheWorkingDirectory(t *testing.T) {
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("no sh")
	}
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	deep := filepath.Join(repo, "a", "b")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := runProbe(t, deep).noCloud; got != "" {
		t.Fatalf("no marker anywhere, but the probe found %q", got)
	}
	marker := filepath.Join(repo, ".nocloud")
	if err := os.WriteFile(marker, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := runProbe(t, deep).noCloud; got != marker {
		t.Errorf("marker in a parent: found %q, want %q", got, marker)
	}

	link := filepath.Join(t.TempDir(), "work")
	if err := os.Symlink(deep, link); err != nil {
		t.Skip("no symlinks here")
	}
	real, _ := filepath.EvalSymlinks(marker)
	if got := runProbe(t, link).noCloud; got != marker && got != real {
		t.Errorf("through a symlink: found %q, want %q", got, marker)
	}

	if got := runProbe(t, filepath.Join(repo, "not", "yet")).noCloud; got != marker {
		t.Errorf("missing directory: found %q, want %q", got, marker)
	}
}

func TestAnIncompleteProbeIsNotReadAsNoMarker(t *testing.T) {
	for _, out := range []string{
		"",
		"Linux x86_64\n/home/a/.cache\n",
		"Linux x86_64\n/home/a/.cache\n\n",
		"Linux x86_64\n/home/a/.cache\n\nsomething-else\n",
	} {
		if _, err := parseProbe(out); err == nil {
			t.Errorf("parseProbe(%q) accepted a reply that never finished", out)
		}
	}
	f, err := parseProbe("Darwin arm64\n/Users/a/.cache\n\n" + probeEnd + "\n")
	if err != nil || f.noCloud != "" || f.goos != "darwin" {
		t.Errorf("complete reply: %+v, %v", f, err)
	}
}
