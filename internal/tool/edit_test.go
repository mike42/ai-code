package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// fixture writes a file and returns (state, absolute path).
func fixture(t *testing.T, name, content string) (*State, string) {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return NewState(dir), p
}

// readFirst simulates the model having read the file, which every edit requires.
func readFirst(t *testing.T, st *State, path string) {
	t.Helper()
	rt := &ReadTool{MaxBytes: 100000, MaxLines: 1000}
	res := rt.Run(context.Background(), st, mustJSON(t, readArgs{Path: path}))
	if res.IsError {
		t.Fatalf("setup read failed: %s", res.Content)
	}
}

const sample = `package main

import "fmt"

func main() {
	if err := run(); err != nil {
		fmt.Println(err)
	}
}
`

func TestEditAppliesExactMatch(t *testing.T) {
	st, path := fixture(t, "main.go", sample)
	readFirst(t, st, path)

	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path:      path,
		OldString: "\t\tfmt.Println(err)",
		NewString: "\t\tfmt.Fprintln(os.Stderr, err)",
	}))
	if res.IsError {
		t.Fatalf("edit failed: %s", res.Content)
	}
	got, _ := os.ReadFile(path)
	if !strings.Contains(string(got), "fmt.Fprintln(os.Stderr, err)") {
		t.Error("replacement was not written")
	}
	if strings.Contains(string(got), "fmt.Println(err)") {
		t.Error("original text is still present")
	}
}

func TestEditRefusesWithoutAPriorRead(t *testing.T) {
	st, path := fixture(t, "main.go", sample)
	// No read.
	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "func main() {", NewString: "func main() { // hi",
	}))
	if !res.IsError {
		t.Fatal("expected the edit to be refused")
	}
	if !strings.Contains(res.Content, "has not been read") {
		t.Errorf("message should say the file was not read, got: %s", res.Content)
	}
	got, _ := os.ReadFile(path)
	if string(got) != sample {
		t.Error("file was modified despite the refusal")
	}
}

func TestEditRefusesWhenFileChangedUnderneath(t *testing.T) {
	st, path := fixture(t, "main.go", sample)
	readFirst(t, st, path)

	// Something else rewrites the file: a formatter or a build step.
	if err := os.WriteFile(path, []byte(sample+"\n// appended by something else\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "func main() {", NewString: "func main() { // hi",
	}))
	if !res.IsError {
		t.Fatal("expected the edit to be refused after the file changed")
	}
	if !strings.Contains(res.Content, "changed on disk") {
		t.Errorf("message should say the file changed, got: %s", res.Content)
	}
	if !strings.Contains(res.Content, "NOT applied") {
		t.Errorf("message must be unambiguous that nothing was written, got: %s", res.Content)
	}
}

func TestEditDiagnosesWhitespaceMismatch(t *testing.T) {
	// The single most common edit failure: the model retypes a tab-indented
	// line using spaces. A bare "not found" gets the identical wrong retry.
	st, path := fixture(t, "main.go", sample)
	readFirst(t, st, path)

	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path:      path,
		OldString: "        fmt.Println(err)", // spaces, file has tabs
		NewString: "        fmt.Println(\"x\")",
	}))
	if !res.IsError {
		t.Fatal("expected a failure")
	}
	if !strings.Contains(res.Content, "whitespace") {
		t.Errorf("diagnosis should identify whitespace as the difference, got:\n%s", res.Content)
	}
	if !strings.Contains(res.Content, "line 7") {
		t.Errorf("diagnosis should name the line number, got:\n%s", res.Content)
	}
	if !strings.Contains(res.Content, "→") {
		t.Errorf("diagnosis should render tabs visibly, got:\n%s", res.Content)
	}
}

func TestEditDiagnosesAmbiguity(t *testing.T) {
	content := "a := 1\nb := 2\nvalue = compute()\nc := 3\nvalue = compute()\n"
	st, path := fixture(t, "x.go", content)
	readFirst(t, st, path)

	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "value = compute()", NewString: "value = compute2()",
	}))
	if !res.IsError {
		t.Fatal("expected an ambiguity error")
	}
	if !strings.Contains(res.Content, "appears 2 times") {
		t.Errorf("should report the occurrence count, got:\n%s", res.Content)
	}
	for _, want := range []string{"line 3", "line 5"} {
		if !strings.Contains(res.Content, want) {
			t.Errorf("should name each occurrence (%s), got:\n%s", want, res.Content)
		}
	}
	if !strings.Contains(res.Content, "replace_all") {
		t.Errorf("should offer replace_all as the resolution, got:\n%s", res.Content)
	}
	got, _ := os.ReadFile(path)
	if string(got) != content {
		t.Error("an ambiguous edit must not modify the file")
	}
}

func TestEditReplaceAll(t *testing.T) {
	content := "x = 1\nx = 1\nx = 1\n"
	st, path := fixture(t, "x.go", content)
	readFirst(t, st, path)

	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "x = 1", NewString: "x = 2", ReplaceAll: true,
	}))
	if res.IsError {
		t.Fatalf("edit failed: %s", res.Content)
	}
	got, _ := os.ReadFile(path)
	if strings.Contains(string(got), "x = 1") {
		t.Error("not all occurrences were replaced")
	}
	if !strings.Contains(res.Content, "3 occurrences") {
		t.Errorf("should report how many were replaced, got: %s", res.Content)
	}
}

func TestEditRejectsIdenticalStrings(t *testing.T) {
	st, path := fixture(t, "x.go", "abc\n")
	readFirst(t, st, path)
	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "abc", NewString: "abc",
	}))
	if !res.IsError {
		t.Fatal("expected an error for a no-op edit")
	}
}

func TestEditSecondEditInSameTurnSeesFreshStamp(t *testing.T) {
	// After a successful edit the tool must re-stamp, or a second edit to the
	// same file in the same turn would falsely report the file as changed.
	st, path := fixture(t, "x.go", "one\ntwo\nthree\n")
	readFirst(t, st, path)
	et := &EditTool{}

	if r := et.Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "one", NewString: "1",
	})); r.IsError {
		t.Fatalf("first edit failed: %s", r.Content)
	}
	r := et.Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "two", NewString: "2",
	}))
	if r.IsError {
		t.Fatalf("second edit failed: %s", r.Content)
	}
	got, _ := os.ReadFile(path)
	if string(got) != "1\n2\nthree\n" {
		t.Errorf("got %q, want both edits applied", string(got))
	}
}

func TestEditPreservesFileMode(t *testing.T) {
	st, path := fixture(t, "script.sh", "#!/bin/sh\necho hi\n")
	if err := os.Chmod(path, 0o755); err != nil {
		t.Fatal(err)
	}
	readFirst(t, st, path)

	if r := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path: path, OldString: "echo hi", NewString: "echo hello",
	})); r.IsError {
		t.Fatalf("edit failed: %s", r.Content)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o755 {
		t.Errorf("mode = %v, want 0755 preserved; an edit must not strip the executable bit", fi.Mode().Perm())
	}
}

func TestWriteRefusesToOverwriteUnreadFile(t *testing.T) {
	st, path := fixture(t, "important.txt", "valuable content\n")
	res := (&WriteTool{}).Run(context.Background(), st, mustJSON(t, writeArgs{
		Path: path, Content: "clobbered",
	}))
	if !res.IsError {
		t.Fatal("expected the overwrite to be refused")
	}
	got, _ := os.ReadFile(path)
	if string(got) != "valuable content\n" {
		t.Error("file was clobbered despite the refusal")
	}
}

func TestWriteCreatesNewFileAndParents(t *testing.T) {
	dir := t.TempDir()
	st := NewState(dir)
	target := filepath.Join(dir, "a", "b", "c.txt")

	res := (&WriteTool{}).Run(context.Background(), st, mustJSON(t, writeArgs{
		Path: target, Content: "hello\n",
	}))
	if res.IsError {
		t.Fatalf("write failed: %s", res.Content)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello\n" {
		t.Errorf("got %q", string(got))
	}
	if !strings.Contains(res.Content, "Created") {
		t.Errorf("should distinguish creation from overwrite, got: %s", res.Content)
	}
}

func TestReadRejectsBinaryWithGuidance(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "blob.bin")
	if err := os.WriteFile(p, []byte{0x7f, 'E', 'L', 'F', 0x00, 0x01, 0x02}, 0o644); err != nil {
		t.Fatal(err)
	}
	st := NewState(dir)
	res := (&ReadTool{MaxBytes: 10000, MaxLines: 100}).Run(context.Background(), st,
		mustJSON(t, readArgs{Path: p}))
	if !res.IsError {
		t.Fatal("expected binary content to be refused")
	}
	if !strings.Contains(res.Content, "binary") {
		t.Errorf("message should say why, got: %s", res.Content)
	}
}

func TestReadMissingFileSuggestsNeighbours(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"config.toml", "config.yaml"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	st := NewState(dir)
	res := (&ReadTool{MaxBytes: 10000, MaxLines: 100}).Run(context.Background(), st,
		mustJSON(t, readArgs{Path: filepath.Join(dir, "config.json")}))
	if !res.IsError {
		t.Fatal("expected a not-found error")
	}
	if !strings.Contains(res.Content, "config.toml") {
		t.Errorf("should suggest similarly named files, got: %s", res.Content)
	}
}

func TestMalformedArgsProduceRecoverableFeedback(t *testing.T) {
	st := NewState(t.TempDir())
	res := (&EditTool{}).Run(context.Background(), st, json.RawMessage(`{"path": "x", "old_string":`))
	if !res.IsError {
		t.Fatal("expected a parse error result")
	}
	if !strings.Contains(res.Content, "JSON") {
		t.Errorf("message should identify the problem as malformed JSON, got: %s", res.Content)
	}
}

// The stamp is keyed by absolute path, so a `cd` between the read and the edit
// must not lose it -- the model routinely reads a file from one directory and
// edits it by a different relative path from another.
func TestEditSurvivesWorkingDirectoryChange(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "cmd", "app")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(sub, "main.go")
	if err := os.WriteFile(path, []byte(sample), 0o644); err != nil {
		t.Fatal(err)
	}

	st := NewState(sub)
	readFirst(t, st, "main.go")

	st.SetCwd(root)
	res := (&EditTool{}).Run(context.Background(), st, mustJSON(t, editArgs{
		Path:      "cmd/app/main.go",
		OldString: "\t\tfmt.Println(err)",
		NewString: "\t\tfmt.Fprintln(os.Stderr, err)",
	}))
	if res.IsError {
		t.Fatalf("edit after a cd was refused: %s", res.Content)
	}
}
