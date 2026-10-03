package session

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

func tempRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("AI_CODE_DATA_DIR", dir)
	return dir
}

func TestRoundTripThroughReplay(t *testing.T) {
	tempRoot(t)
	project := "/some/project"

	s, err := Create(Meta{Project: project, Model: "m", Provider: "home"})
	if err != nil {
		t.Fatal(err)
	}
	want := []provider.Message{
		{Role: provider.RoleUser, Content: "fix the bug"},
		{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
			{ID: "t1", Name: "read", Args: `{"path":"x.go"}`},
		}},
		{Role: provider.RoleTool, ToolCallID: "t1", Name: "read", Content: "file body"},
		{Role: provider.RoleAssistant, Content: "done"},
	}
	for _, m := range want {
		if err := s.AppendMessage(m); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	_, entries, err := Open(project, s.Meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	got := Messages(entries)

	if len(got) != len(want) {
		t.Fatalf("replayed %d messages, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Role != want[i].Role || got[i].Content != want[i].Content {
			t.Errorf("message %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	if len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].ID != "t1" {
		t.Errorf("tool calls did not survive the round trip: %+v", got[1].ToolCalls)
	}
	if got[2].ToolCallID != "t1" {
		t.Error("tool result lost its call id")
	}
}

func TestATruncatedFinalLineDoesNotLoseTheSession(t *testing.T) {
	tempRoot(t)
	project := "/p"

	s, err := Create(Meta{Project: project})
	if err != nil {
		t.Fatal(err)
	}
	_ = s.AppendMessage(provider.Message{Role: provider.RoleUser, Content: "first"})
	_ = s.AppendMessage(provider.Message{Role: provider.RoleAssistant, Content: "second"})
	path := s.Path()
	s.Close()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(raw, []byte(`{"type":"message","mess`)...), 0o600); err != nil {
		t.Fatal(err)
	}

	_, entries, err := Open(project, s.Meta.ID)
	if err != nil {
		t.Fatalf("a truncated final line should not fail the open: %v", err)
	}
	msgs := Messages(entries)
	if len(msgs) != 2 {
		t.Errorf("recovered %d messages, want the 2 complete ones", len(msgs))
	}
}

func TestCompactionResetsReplayedContext(t *testing.T) {
	tempRoot(t)
	s, err := Create(Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	_ = s.AppendMessage(provider.Message{Role: provider.RoleUser, Content: "old one"})
	_ = s.AppendMessage(provider.Message{Role: provider.RoleAssistant, Content: "old two"})
	_ = s.Append(Entry{Type: EntryCompaction, Summary: "## Goal\nthing", MessagesBefore: 2})
	_ = s.AppendMessage(provider.Message{Role: provider.RoleUser, Content: "after compaction"})
	s.Close()

	_, entries, err := Open("/p", s.Meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	msgs := Messages(entries)
	if len(msgs) != 1 || msgs[0].Content != "after compaction" {
		t.Errorf("replayed %+v, want only the post-compaction message", msgs)
	}

	raw, _ := os.ReadFile(s.Path())
	if !strings.Contains(string(raw), "old one") {
		t.Error("compaction removed history from the file; it should only shorten context")
	}
}

func TestListIsNewestFirstWithPreviews(t *testing.T) {
	tempRoot(t)
	for _, text := range []string{"first task", "second task"} {
		s, err := Create(Meta{Project: "/p", Model: "m"})
		if err != nil {
			t.Fatal(err)
		}
		_ = s.AppendMessage(provider.Message{Role: provider.RoleUser, Content: text})
		s.Close()
	}
	list, err := List("/p")
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("got %d sessions, want 2", len(list))
	}
	if list[0].Preview == "" {
		t.Error("listing should carry a preview of the opening request")
	}
	if list[0].Messages != 1 {
		t.Errorf("message count = %d, want 1", list[0].Messages)
	}
}

func TestSessionsAreOwnerReadableOnly(t *testing.T) {
	tempRoot(t)
	s, err := Create(Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	fi, err := os.Stat(s.Path())
	if err != nil {
		t.Fatal(err)
	}
	if perm := fi.Mode().Perm(); perm&0o077 != 0 {
		t.Errorf("session file mode is %v; it should not be group or world readable", perm)
	}
}

func TestDifferentProjectsAreKeptApart(t *testing.T) {
	root := tempRoot(t)
	for _, p := range []string{"/home/a/proj", "/home/b/proj"} {
		s, err := Create(Meta{Project: p})
		if err != nil {
			t.Fatal(err)
		}
		s.Close()
	}
	entries, err := os.ReadDir(filepath.Join(root, "sessions"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Errorf("got %d project directories, want 2 -- identically named projects "+
			"in different locations must not share a session directory", len(entries))
	}

	a, err := List("/home/a/proj")
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 1 {
		t.Errorf("listing for one project returned %d sessions", len(a))
	}
}

func TestIDsSortChronologically(t *testing.T) {
	a := NewID()
	b := NewID()
	if a >= b && a[:15] == b[:15] {
		return
	}
	if a > b {
		t.Errorf("ids do not sort by time: %q then %q", a, b)
	}
}

func TestInputsSurviveCompactionAndIncludeCommands(t *testing.T) {
	entries := []Entry{
		{Type: EntryInput, Input: "first prompt"},
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleUser, Content: "first prompt"}},
		{Type: EntryInput, Input: "/model qwen"},
		{Type: EntryCompaction, Summary: "earlier work"},
		{Type: EntryInput, Input: "second prompt"},
	}

	if msgs := Messages(entries); len(msgs) != 0 {
		t.Fatalf("setup: compaction should have cleared the messages, got %d", len(msgs))
	}

	got := Inputs(entries)
	want := []string{"first prompt", "/model qwen", "second prompt"}
	if len(got) != len(want) {
		t.Fatalf("Inputs() = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Inputs()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestInputsFallBackToUserMessagesForOlderSessions(t *testing.T) {
	entries := []Entry{
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleUser, Content: "old prompt"}},
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleAssistant, Content: "a reply"}},
		{Type: EntryMessage, Message: &provider.Message{
			Role: provider.RoleUser, Content: "<context-checkpoint>\nsummary\n</context-checkpoint>"}},
	}

	got := Inputs(entries)
	if len(got) != 1 || got[0] != "old prompt" {
		t.Errorf("Inputs() = %q, want just the typed prompt", got)
	}
}

func TestInputsPreferTheInputLogOverMessages(t *testing.T) {
	entries := []Entry{
		{Type: EntryInput, Input: "what I typed"},
		{Type: EntryMessage, Message: &provider.Message{
			Role: provider.RoleUser, Content: "what I typed, wrapped by ai-code"}},
	}
	got := Inputs(entries)
	if len(got) != 1 || got[0] != "what I typed" {
		t.Errorf("Inputs() = %q, want the raw line, not the wrapped message", got)
	}
}

func TestCheckpointLeavesTheTranscriptAlone(t *testing.T) {
	// The checkpoint is non-destructive: the summary is recorded beside the
	// conversation, and replay still yields every message.
	entries := []Entry{
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleUser, Content: "one"}},
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleAssistant, Content: "two"}},
		{Type: EntryCheckpoint, Summary: "## Goal\nthe thing", SummarisedThrough: 2},
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleUser, Content: "three"}},
	}

	if msgs := Messages(entries); len(msgs) != 3 {
		t.Errorf("replayed %d messages, want 3: a checkpoint must not reset the transcript", len(msgs))
	}

	summary, through, _ := Checkpoint(entries)
	if summary != "## Goal\nthe thing" || through != 2 {
		t.Errorf("Checkpoint() = %q, %d; want the recorded summary and 2", summary, through)
	}
}

func TestCheckpointTakesTheLastOneRecorded(t *testing.T) {
	entries := []Entry{
		{Type: EntryCheckpoint, Summary: "earlier", SummarisedThrough: 2},
		{Type: EntryCheckpoint, Summary: "later", SummarisedThrough: 6},
	}
	if summary, through, _ := Checkpoint(entries); summary != "later" || through != 6 {
		t.Errorf("Checkpoint() = %q, %d; want the most recent", summary, through)
	}
}

func TestCompactionClearsTheStandbyCheckpoint(t *testing.T) {
	// Legacy /compact summary is already replayed as the first message.
	entries := []Entry{
		{Type: EntryCheckpoint, Summary: "standby", SummarisedThrough: 2},
		{Type: EntryCompaction, Summary: "compacted"},
		{Type: EntryMessage, Message: &provider.Message{Role: provider.RoleUser, Content: "<context-checkpoint>compacted</context-checkpoint>"}},
	}
	if summary, through, _ := Checkpoint(entries); summary != "" || through != 0 {
		t.Errorf("Checkpoint() = %q, %d; want nothing after a compaction", summary, through)
	}
}

func TestCheckpointSurvivesAnOnDiskRoundTrip(t *testing.T) {
	tempRoot(t)
	s, err := Create(Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	_ = s.AppendMessage(provider.Message{Role: provider.RoleUser, Content: "one"})
	_ = s.AppendMessage(provider.Message{Role: provider.RoleAssistant, Content: "two"})
	_ = s.Append(Entry{Type: EntryCheckpoint, Summary: "## Goal\nship it", SummarisedThrough: 2})
	s.Close()

	_, entries, err := Open("/p", s.Meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	summary, through, _ := Checkpoint(entries)
	if summary != "## Goal\nship it" || through != 2 {
		t.Errorf("after reopening: Checkpoint() = %q, %d", summary, through)
	}
	if msgs := Messages(entries); len(msgs) != 2 {
		t.Errorf("after reopening: %d messages, want 2", len(msgs))
	}
}

func TestWorkspaceDirSeparatesWorkingDirectories(t *testing.T) {
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	a, _ := WorkspaceDir("ssh://agent@mac/Users/agent/a")
	b, _ := WorkspaceDir("ssh://agent@mac/Users/agent/b")
	local, _ := WorkspaceDir("/Users/agent/a")
	if a == b || a == local {
		t.Errorf("two working directories share notes: %s %s %s", a, b, local)
	}
	again, _ := WorkspaceDir("ssh://agent@mac/Users/agent/a")
	if again != a {
		t.Errorf("the same working directory moved: %s then %s", a, again)
	}
}

func TestStateDirHoldsSessionsAndNotes(t *testing.T) {
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	dir := t.TempDir()
	if err := SetStateDir(dir); err != nil {
		t.Fatal(err)
	}
	defer func() { stateDir = "" }()

	s, err := Create(Meta{Project: "/some/project"})
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	if _, err := os.Stat(filepath.Join(dir, "sessions", s.Meta.ID+".jsonl")); err != nil {
		t.Errorf("session not under --state-dir: %v", err)
	}
	if latest, err := Latest("/elsewhere"); err != nil || latest.ID != s.Meta.ID {
		t.Errorf("-c does not find it: %+v, %v", latest, err)
	}
	if w, _ := WorkspaceDir("/any"); w != dir {
		t.Errorf("notes go to %s, want %s", w, dir)
	}
}
