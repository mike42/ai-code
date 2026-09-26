package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func forkable(t *testing.T) (*LocalExecutor, string) {
	t.Helper()
	dir := t.TempDir()
	return NewLocalExecutor(NewState(dir),
		&ReadTool{MaxBytes: 1 << 20, MaxLines: 10000},
		&WriteTool{},
		&EditTool{},
	), dir
}

func run(t *testing.T, e Executor, name string, args any) Result {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	res, err := e.Execute(context.Background(), Request{Name: name, Args: raw})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// A fork keeps its own working directory. Two agents sharing one would move
// each other's cwd every time either ran `cd`.
func TestAForkHasItsOwnWorkingDirectory(t *testing.T) {
	parent, dir := forkable(t)
	child, err := parent.Fork()
	if err != nil {
		t.Fatal(err)
	}

	sub := filepath.Join(dir, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	child.(*LocalExecutor).state.SetCwd(sub)

	if got := parent.state.Cwd(); got != dir {
		t.Errorf("the parent's cwd moved to %q when the fork changed its own", got)
	}
}

// The important one, and the quiet one. A stamp records what one reader last
// saw. Shared, the child's read satisfies the parent's staleness check, and
// the parent's edit is applied to content it never looked at.
func TestAForkDoesNotSatisfyItsParentsStalenessCheck(t *testing.T) {
	parent, dir := forkable(t)
	path := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(path, []byte("original\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	child, err := parent.Fork()
	if err != nil {
		t.Fatal(err)
	}

	// The child reads and rewrites the file. The parent never read it.
	run(t, child, "read", map[string]any{"path": path})
	if res := run(t, child, "write", map[string]any{"path": path, "content": "changed\n"}); res.IsError {
		t.Fatalf("the fork could not write: %s", res.Content)
	}

	// The parent now tries to edit it. It must be refused: it has not read
	// this file, and the child's read is not its own.
	res := run(t, parent, "edit", map[string]any{
		"path": path, "old_string": "original", "new_string": "something else",
	})
	if !res.IsError {
		t.Fatal("the parent edited a file it never read, on the strength of the fork's read")
	}
	if !strings.Contains(res.Content, "has not been read") {
		t.Errorf("refused with %q, want it to say the file was not read in this session", res.Content)
	}
}

// A fork sees the same filesystem. Isolation is of working state, not of
// files: a worker is supposed to do real work in the same tree.
func TestAForkSeesTheSameFiles(t *testing.T) {
	parent, dir := forkable(t)
	path := filepath.Join(dir, "shared.txt")
	if res := run(t, parent, "write", map[string]any{"path": path, "content": "hello\n"}); res.IsError {
		t.Fatalf("parent write failed: %s", res.Content)
	}

	child, err := parent.Fork()
	if err != nil {
		t.Fatal(err)
	}
	res := run(t, child, "read", map[string]any{"path": path})
	if res.IsError || !strings.Contains(res.Content, "hello") {
		t.Errorf("the fork read %q, want the file its parent wrote", res.Content)
	}
}
