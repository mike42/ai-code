package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"ai-code/internal/agent"
	"ai-code/internal/prompt"
	"ai-code/internal/tool"
)

// Nothing about the host may reach a model whose tools run in a container.
//
// The reported symptom was a request to edit AGENTS.md failing: the model had
// been handed the file's path on the host, while every tool it has runs
// inside the container where that path does not exist. The disclosure and the
// broken edit are the same bug -- a path that names nothing the model can act
// on.
func TestNoHostPathsReachTheModelInAContainer(t *testing.T) {
	// A host layout with a username in it, and instruction files at both the
	// project root and a nested directory.
	home := t.TempDir()
	project := filepath.Join(home, "someone", "secret-project")
	nested := filepath.Join(project, "internal", "engine")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	for dir, body := range map[string]string{
		project: "# Project rules\nUse tabs.",
		nested:  "# Engine rules\nNo allocations in the hot path.",
	} {
		if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	const workspaceFolder = "/workspaces/secret-project"
	app := &App{
		cwd:         nested,
		projectRoot: project,
		hostGitRoot: project,
		showPath: func(hostPath string) (string, bool) {
			rel, err := filepath.Rel(project, hostPath)
			if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
				return "", false
			}
			return filepath.ToSlash(filepath.Join(workspaceFolder, rel)), true
		},
		env: prompt.Env{
			Cwd:       mountCwd(project, nested, workspaceFolder),
			OS:        "linux",
			Arch:      "amd64",
			IsGitRepo: true,
			GitRoot:   workspaceFolder,
		},
		exec:  tool.NewLocalExecutor(tool.NewState(nested)),
		agent: agent.New(nil, "m", tool.NewLocalExecutor(tool.NewState(nested)), nil, agent.Options{}),
	}
	app.rebuildSystemPrompt("")
	got := app.agent.System()

	// The whole point: no fragment of this machine's layout survives. The
	// project basename is not in this list -- it legitimately appears inside
	// the container path, which is the answer rather than the leak.
	for _, leak := range []string{home, project, nested, "someone"} {
		if strings.Contains(got, leak) {
			t.Errorf("system prompt leaks the host path %q:\n%s", leak, got)
		}
	}
	// And the instructions still arrive, named where the tools can reach them.
	for _, want := range []string{
		"Use tabs.",
		"No allocations in the hot path.",
		workspaceFolder + "/AGENTS.md",
		workspaceFolder + "/internal/engine/AGENTS.md",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("system prompt is missing %q:\n%s", want, got)
		}
	}
	// The shell belongs to the container, and claiming this machine's is a
	// wrong answer as well as a disclosure.
	if strings.Contains(got, "Shell:") {
		t.Errorf("system prompt asserts a shell it did not detect:\n%s", got)
	}
}

// On the host there is nothing to translate, and real paths must survive.
func TestHostSessionKeepsRealPaths(t *testing.T) {
	project := t.TempDir()
	if err := os.WriteFile(filepath.Join(project, "AGENTS.md"), []byte("# Rules\nUse tabs."), 0o644); err != nil {
		t.Fatal(err)
	}
	app := &App{
		cwd:         project,
		projectRoot: project,
		hostGitRoot: project,
		env:         prompt.Env{Cwd: project, OS: "linux", Arch: "amd64"},
		exec:        tool.NewLocalExecutor(tool.NewState(project)),
		agent:       agent.New(nil, "m", tool.NewLocalExecutor(tool.NewState(project)), nil, agent.Options{}),
	}
	app.rebuildSystemPrompt("")
	if got := app.agent.System(); !strings.Contains(got, filepath.Join(project, "AGENTS.md")) {
		t.Errorf("a host session lost its real AGENTS.md path:\n%s", got)
	}
}
