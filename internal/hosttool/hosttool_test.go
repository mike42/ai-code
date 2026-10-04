package hosttool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"ai-code/internal/tool"
)

func newExec(t *testing.T, dir string, names ...string) *Executor {
	t.Helper()
	inner := tool.NewLocalExecutor(tool.NewState(t.TempDir()), &tool.LsTool{})
	e := Wrap(inner, dir, names)
	e.now = func() time.Time { return time.Date(2026, 10, 3, 12, 0, 0, 0, time.UTC) }
	return e
}

func call(t *testing.T, e *Executor, name, args string) tool.Result {
	t.Helper()
	res, err := e.Execute(context.Background(), tool.Request{Name: name, Args: json.RawMessage(args)})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func hasDef(e tool.Executor, name string) bool {
	for _, d := range e.Definitions() {
		if d.Name == name {
			return true
		}
	}
	return false
}

func TestOnlyEnabledToolsAreOffered(t *testing.T) {
	e := newExec(t, t.TempDir(), TodoName)
	if !hasDef(e, TodoName) || hasDef(e, StatusName) || !hasDef(e, "ls") {
		t.Errorf("definitions: %+v", e.Definitions())
	}
	res := call(t, e, StatusName, `{"title":"x","body":"y"}`)
	if !res.IsError {
		t.Error("a disabled host tool ran")
	}
}

func TestTodoPersistsAcrossRuns(t *testing.T) {
	dir := t.TempDir()
	first := newExec(t, dir, TodoName)
	res := call(t, first, TodoName, `{"todos":[
		{"content":"write the parser","status":"completed"},
		{"content":"test it","status":"in_progress"},
		{"content":"old idea","status":"cancelled"},
		{"content":"ship","status":"pending"}]}`)
	if res.IsError {
		t.Fatal(res.Content)
	}
	if res.Display != "Todo: 1 done, 2 to go -- now: test it" {
		t.Errorf("display %q", res.Display)
	}
	if !strings.Contains(res.Show, "- [>] **test it**") {
		t.Errorf("show %q", res.Show)
	}

	second := newExec(t, dir, TodoName)
	read := call(t, second, TodoName, `{}`)
	if !strings.Contains(read.Content, "- [ ] ship") || read.Show != "" {
		t.Errorf("read back: %+v", read)
	}
	if c := second.Carried(); !strings.Contains(c, "- [x] write the parser") {
		t.Errorf("carried: %q", c)
	}
}

func TestTodoRejectsBadLists(t *testing.T) {
	e := newExec(t, t.TempDir(), TodoName)
	for _, args := range []string{
		`{"todos":[{"content":"a","status":"in_progress"},{"content":"b","status":"in_progress"}]}`,
		`{"todos":[{"content":"a","status":"done"}]}`,
		`{"todos":[{"content":"  ","status":"pending"}]}`,
		`{"todos":`,
	} {
		if res := call(t, e, TodoName, args); !res.IsError {
			t.Errorf("%s was accepted", args)
		}
	}
}

func TestStatusKeepsHistoryAndLatest(t *testing.T) {
	dir := t.TempDir()
	e := newExec(t, dir, StatusName)
	res := call(t, e, StatusName, `{"title":"Parser  done","body":"All tests pass.\n\nNext: docs."}`)
	if res.IsError || res.Display != "Status: Parser done" || res.Show != "All tests pass.\n\nNext: docs." {
		t.Fatalf("%+v", res)
	}
	latest, err := os.ReadFile(filepath.Join(dir, "status.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(latest), "# Parser done\n\n_2026-10-03 12:00:00 UTC_\n\nAll tests pass.") {
		t.Errorf("status.md = %q", latest)
	}
	hist, _ := filepath.Glob(filepath.Join(dir, "status", "*.md"))
	if len(hist) != 1 {
		t.Errorf("history: %v", hist)
	}

	next := newExec(t, dir, StatusName)
	if c := next.Carried(); !strings.Contains(c, "# Parser done") {
		t.Errorf("carried: %q", c)
	}
	if c := newExec(t, dir, TodoName).Carried(); c != "" {
		t.Errorf("status was carried with the status tool disabled: %q", c)
	}
}

func TestWorkersGetNoHostTool(t *testing.T) {
	e := newExec(t, t.TempDir(), TodoName, StatusName)
	forked, err := e.Fork()
	if err != nil {
		t.Fatal(err)
	}
	if hasDef(forked, TodoName) || hasDef(e.Unwrapped(), StatusName) {
		t.Error("a worker's executor still has a host tool")
	}
}

func TestNotesPersistOutsideTheWorkspace(t *testing.T) {
	dir := t.TempDir()
	e := newExec(t, dir, NotesName)
	if res := call(t, e, NotesName, `{"action":"list"}`); res.IsError || res.Content != "No notes yet." {
		t.Fatalf("%+v", res)
	}
	if res := call(t, e, NotesName, `{"action":"write","name":"api-keys","content":"TOKEN=abc"}`); res.IsError {
		t.Fatal(res.Content)
	}
	b, err := os.ReadFile(filepath.Join(dir, "notes", "api-keys"))
	if err != nil || string(b) != "TOKEN=abc" {
		t.Fatalf("file %q %v", b, err)
	}

	next := newExec(t, dir, NotesName)
	if res := call(t, next, NotesName, `{"action":"read","name":"api-keys"}`); res.Content != "TOKEN=abc" {
		t.Errorf("reading the note in a new run gave: %+v", res)
	}
	c := next.Carried()
	if !strings.Contains(c, "api-keys") {
		t.Errorf("a new run is not told the note api-keys exists: %q", c)
	}
	if strings.Contains(c, "TOKEN") {
		t.Errorf("a new run is shown what is inside a note; it should only see the note's name: %q", c)
	}
	if res := call(t, next, NotesName, `{"action":"delete","name":"api-keys"}`); res.IsError {
		t.Fatal(res.Content)
	}
	if res := call(t, next, NotesName, `{"action":"read","name":"api-keys"}`); !res.IsError {
		t.Error("a deleted note was read")
	}
}

func TestNotesRefuseNamesThatLeaveTheirFolder(t *testing.T) {
	dir := t.TempDir()
	e := newExec(t, dir, NotesName)
	for _, name := range []string{"../escape", "a/b", ".hidden", "", "..", `a\b`} {
		args, _ := json.Marshal(map[string]string{"action": "write", "name": name, "content": "x"})
		if res := call(t, e, NotesName, string(args)); !res.IsError {
			t.Errorf("a note could be saved under the name %q", name)
		}
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "escape")); err == nil {
		t.Error("a note was written outside the notes folder")
	}
	if res := call(t, e, NotesName, `{"action":"write","name":"x"}`); !res.IsError {
		t.Error("write without content was accepted")
	}
}
