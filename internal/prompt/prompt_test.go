package prompt

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The silent-5-to-10x-cost failure: an unstable system prompt means every
// request misses the provider's cache. It produces no error, so it is checked.
func TestSystemPromptIsByteIdenticalAcrossBuilds(t *testing.T) {
	opts := Options{
		Env: Env{
			Cwd: "/home/dev/project", OS: "linux", Arch: "amd64",
			Shell: "bash", IsGitRepo: true, GitRoot: "/home/dev/project", GitBranch: "main",
		},
		ModePrompt: "steering text",
		ToolNames:  []string{"bash", "edit", "glob", "grep", "ls", "read", "write"},
		Agents: []AgentsFile{
			{Path: "/home/dev/project/AGENTS.md", Content: "use tabs"},
		},
	}

	first := Build(opts)
	for i := range 50 {
		if got := Build(opts); got != first {
			t.Fatalf("build %d differs from the first; the prompt prefix is unstable "+
				"and prompt caching is silently broken", i)
		}
	}
	if strings.Contains(first, "20") && strings.Contains(first, ":") {
		// Weak but useful: a timestamp would show up as a date-like fragment.
		for _, bad := range []string{"AM", "PM", "UTC", "GMT"} {
			if strings.Contains(first, bad) {
				t.Errorf("prompt appears to contain a timestamp (%q), which breaks caching", bad)
			}
		}
	}
}

func TestToolOrderIsDeterministic(t *testing.T) {
	// Tool names come from a map somewhere upstream; iteration order must never
	// reach the prompt.
	a := SortedToolNames([]string{"write", "bash", "read"})
	b := SortedToolNames([]string{"read", "write", "bash"})
	if strings.Join(a, ",") != strings.Join(b, ",") {
		t.Errorf("tool ordering is not stable: %v vs %v", a, b)
	}
}

func TestAgentsFilesAreOrderedOutermostFirst(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "sub", "deeper")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(dir, body string) {
		if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(root, "root rules")
	write(nested, "nested rules")

	got := DiscoverAgents(nested, root)

	var bodies []string
	for _, f := range got {
		if strings.Contains(f.Path, root) {
			bodies = append(bodies, strings.TrimSpace(f.Content))
		}
	}
	if len(bodies) != 2 {
		t.Fatalf("found %d project files, want 2: %v", len(bodies), bodies)
	}
	if bodies[0] != "root rules" || bodies[1] != "nested rules" {
		t.Errorf("order = %v, want the outermost file first so the nested one reads "+
			"as an override", bodies)
	}
}

func TestAgentsBudgetIsEnforced(t *testing.T) {
	root := t.TempDir()
	huge := strings.Repeat("x", MaxAgentsBytes*2)
	if err := os.WriteFile(filepath.Join(root, "AGENTS.md"), []byte(huge), 0o644); err != nil {
		t.Fatal(err)
	}
	got := DiscoverAgents(root, root)

	total := 0
	for _, f := range got {
		total += len(f.Content)
	}
	if total > MaxAgentsBytes+200 {
		t.Errorf("instruction content is %d bytes, over the %d budget; an oversized "+
			"AGENTS.md would silently eat the context window", total, MaxAgentsBytes)
	}
	if len(got) > 0 && !strings.Contains(got[0].Content, "truncated") {
		t.Error("truncation should be stated, not silent")
	}
}

func TestClaudeMdIsNotRead(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "CLAUDE.md"), []byte("from CLAUDE.md"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, f := range DiscoverAgents(root, root) {
		if strings.HasPrefix(f.Path, root) {
			t.Errorf("picked up %s; AGENTS.md is the only project instruction file", f.Path)
		}
	}
}

func TestOnlyOneInstructionFilePerDirectory(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".ai-code"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"AGENTS.md", filepath.Join(".ai-code", "AGENTS.md")} {
		if err := os.WriteFile(filepath.Join(root, n), []byte("from "+n), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := DiscoverAgents(root, root)

	count := 0
	for _, f := range got {
		if strings.HasPrefix(f.Path, root) {
			count++
			if filepath.Base(filepath.Dir(f.Path)) == ".ai-code" {
				t.Errorf("picked %s; the root AGENTS.md should win", f.Path)
			}
		}
	}
	if count != 1 {
		t.Errorf("sent %d files from one directory, want 1", count)
	}
}

func TestDetectEnvFindsGitRoot(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	dir := t.TempDir()
	if out, err := exec.Command("git", "-C", dir, "init", "-q").CombinedOutput(); err != nil {
		t.Skipf("git init failed: %s", out)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}

	env := DetectEnv(sub)
	if !env.IsGitRepo {
		t.Fatal("git repository not detected from a subdirectory")
	}
	// macOS resolves temp dirs through a symlink, so compare the basenames.
	if filepath.Base(env.GitRoot) != filepath.Base(dir) {
		t.Errorf("git root = %q, want %q", env.GitRoot, dir)
	}
}

func TestPromptMentionsToolsAndMode(t *testing.T) {
	out := Build(Options{
		Env:        Env{Cwd: "/x", OS: "linux", Arch: "amd64"},
		ModePrompt: "RESEARCH STEERING",
		ToolNames:  []string{"bash", "read"},
	})
	if !strings.Contains(out, "RESEARCH STEERING") {
		t.Error("mode steering text is missing from the prompt")
	}
	if !strings.Contains(out, "bash, read") {
		t.Error("tool names are missing from the prompt")
	}
}

func TestRemoteEnvironmentAndCarriedNotes(t *testing.T) {
	out := Build(Options{
		Env:     Env{Cwd: "/home/agent/work", Remote: "agent@buildvm"},
		Carried: "Your last status update: parser done.",
	})
	for _, want := range []string{
		"Tools run on: agent@buildvm, over SSH.",
		"Working directory: /home/agent/work",
		"Platform: not known in advance",
		"Check for one before you start, and read it if it is there.",
		"## From your previous run\n\nYour last status update: parser done.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("prompt lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Platform: /") {
		t.Error("an unknown platform was printed as an empty one")
	}
	if strings.Contains(Build(Options{Env: Env{Cwd: "/x", OS: "linux", Arch: "amd64"}}), "previous run") {
		t.Error("an empty carried section was printed")
	}
}

func TestAppendedInstructions(t *testing.T) {
	out := Build(Options{Env: Env{Cwd: "/w", OS: "linux", Arch: "amd64"}, Appended: "You are the reviewer."})
	if !strings.Contains(out, "## Instructions for this session\n\nYou are the reviewer.\n") {
		t.Errorf("appended text missing:\n%s", out)
	}
	if strings.Contains(Build(Options{Env: Env{Cwd: "/w"}}), "Instructions for this session") {
		t.Error("an empty section was printed")
	}
	if strings.Contains(Build(Options{Env: Env{Cwd: "/w", OS: "linux"}}), "AGENTS.md in the working directory") {
		t.Error("a local session was told to look for an AGENTS.md it has already been given")
	}
}
