package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

// noNetClient stands in for a provider and fails the test if Tab so much as
// asks it a question. Completion is a keystroke: it reads the on-disk
// catalogue or it offers nothing.
type noNetClient struct {
	t     *testing.T
	name  string
	class provider.Class
}

func (c *noNetClient) Name() string          { return c.name }
func (c *noNetClient) Class() provider.Class { return c.class }

func (c *noNetClient) Stream(context.Context, provider.Request) (provider.Stream, error) {
	c.t.Error("Tab opened a stream to the provider")
	return nil, nil
}

func (c *noNetClient) Models(context.Context) ([]provider.ModelInfo, error) {
	c.t.Error("Tab asked the provider for its model catalogue")
	return nil, nil
}

// modelApp gives an App whose model cache holds exactly the given ids.
func modelApp(t *testing.T, class provider.Class, noCloud bool, ids ...string) *App {
	t.Helper()
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())

	var models []provider.ModelInfo
	for _, id := range ids {
		models = append(models, provider.ModelInfo{ID: id, SupportsTools: true})
	}
	if len(models) > 0 {
		saveModelCache("testprov", models)
	}
	return &App{
		client:  &noNetClient{t: t, name: "testprov", class: class},
		noCloud: noCloud,
	}
}

// Slash-command completion is the behaviour that already existed and must not
// change: candidates carry the leading slash and a trailing space.
func TestCommandCompletionUnchanged(t *testing.T) {
	a := &App{}

	at, got := a.complete("/to")
	if at != 0 {
		t.Errorf("replacement starts at %d, want the whole command word", at)
	}
	if !slices.Equal(got, []string{"/tokens ", "/tools "}) {
		t.Errorf("/to = %v, want /tokens and /tools", got)
	}

	if _, got := a.complete("/comp"); !slices.Equal(got, []string{"/compact "}) {
		t.Errorf("/comp = %v, want just /compact", got)
	}
	// /mode dispatches on its own but is also a prefix of /model.
	if _, got := a.complete("/mode"); !slices.Equal(got, []string{"/mode ", "/model "}) {
		t.Errorf("/mode = %v, want both /mode and /model", got)
	}
	if _, got := a.complete("/zzz"); got != nil {
		t.Errorf("/zzz = %v, want nothing", got)
	}
}

// Model ids differ at the end -- quantisation, size, a :variant -- so the
// common prefix is what a first Tab can safely insert, and the list is what
// the second one has to show.
func TestModelCompletionStopsAtTheSuffixDivergence(t *testing.T) {
	a := modelApp(t, provider.ClassOnPrem, false,
		"qwen3-coder-30b:free", "qwen3-coder-30b:nitro", "qwen3-coder-7b", "llama-3-8b")

	at, got := a.complete("/model qwen3-coder-3")
	if want := len("/model "); at != want {
		t.Errorf("replacement starts at %d, want %d (just after the command)", at, want)
	}
	want := []string{"qwen3-coder-30b:free", "qwen3-coder-30b:nitro"}
	if !slices.Equal(got, want) {
		t.Errorf("candidates = %v, want %v", got, want)
	}
	if common := commonPrefixOf(got); common != "qwen3-coder-30b:" {
		t.Errorf("common prefix = %q, want the id up to where the variants diverge", common)
	}
}

func TestModelCompletionListsEverythingForABareCommand(t *testing.T) {
	a := modelApp(t, provider.ClassOnPrem, false, "b-model", "a-model")

	at, got := a.complete("/model ")
	if at != len("/model ") {
		t.Errorf("replacement starts at %d", at)
	}
	if !slices.Equal(got, []string{"a-model", "b-model"}) {
		t.Errorf("bare /model = %v, want every id, sorted", got)
	}

	// Argument completion follows the dispatcher's own resolution rule, and
	// "/mod" is ambiguous between /mode and /model there too.
	if _, got := a.complete("/mod a"); got != nil {
		t.Errorf("/mod a = %v, want nothing for an ambiguous command", got)
	}
}

// A cold cache completes nothing and says nothing. The alternative -- fetching
// the catalogue on Tab -- is a keystroke that sometimes takes a second.
func TestModelCompletionIsSilentOnAColdCache(t *testing.T) {
	a := modelApp(t, provider.ClassOnPrem, false)
	if _, got := a.complete("/model q"); got != nil {
		t.Errorf("cold cache = %v, want nothing", got)
	}
}

// Under .nocloud a cloud model name must never be offered. Listing one and
// refusing it afterwards would still put the name on the screen as a choice.
func TestNoCloudHidesCloudModelsEntirely(t *testing.T) {
	a := modelApp(t, provider.ClassCloud, true, "gpt-secret-1", "gpt-secret-2")
	_, got := a.complete("/model ")
	if len(got) != 0 {
		t.Errorf("under .nocloud /model offered %v, want nothing", got)
	}
	if _, got := a.complete("/model gpt"); len(got) != 0 {
		t.Errorf("under .nocloud /model gpt offered %v, want nothing", got)
	}

	// The same cache on an on-premises provider still completes, so the test
	// above is about the marker and not about an empty cache.
	a.noCloud = false
	a.client = &noNetClient{t: t, name: "testprov", class: provider.ClassOnPrem}
	if _, got := a.complete("/model gpt"); len(got) != 2 {
		t.Errorf("without .nocloud = %v, want both ids", got)
	}
}

// completeTree builds a small project to complete against.
func completeTree(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, f := range []string{
		"internal/ui/editor.go", "internal/ui/steer.go", "internal/tool/ignore.go",
		"build/artefact.bin", "notes.md", ".hidden",
	} {
		p := filepath.Join(root, filepath.FromSlash(f))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("build/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestPathCompletion(t *testing.T) {
	root := completeTree(t)

	cases := []struct {
		name string
		line string
		at   int
		want []string
	}{
		{"directory gets a trailing separator", "look at inter", 8, []string{"internal/"}},
		{"one segment at a time", "look at internal/", 8, []string{"internal/tool/", "internal/ui/"}},
		{"files below a directory", "internal/ui/e", 0, []string{"internal/ui/editor.go"}},
		{"gitignored paths are not offered", "buil", 0, nil},
		{"dotfiles need the dot", "no", 0, []string{"notes.md"}},
		{"dotfiles appear once asked for", ".hid", 0, []string{".hidden"}},
		{"an empty line lists the working directory", "", 0, []string{"internal/", "notes.md"}},
		// The decision, pinned: Tab after a trailing space lists too. It is the
		// same keystroke at the same place in a word, and it may not depend on
		// whether prose happens to precede it.
		{"an empty word mid-line lists it too", "please look at ", 15, []string{"internal/", "notes.md"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			at, got := completePath(root, root, tc.line)
			if len(got) != 0 && at != tc.at {
				t.Errorf("replacement starts at %d, want %d", at, tc.at)
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("%q = %v, want %v", tc.line, got, tc.want)
			}
		})
	}
}

// A line that opens with an absolute path is not a slash command.
func TestAbsolutePathIsNotACommand(t *testing.T) {
	root := completeTree(t)
	a := &App{cwd: root, projectRoot: root}

	line := filepath.ToSlash(filepath.Join(root, "internal", "u"))
	at, got := a.complete(line)
	if at != 0 || len(got) != 1 || !strings.HasSuffix(got[0], "/internal/ui/") {
		t.Errorf("%q = (%d, %v), want the absolute directory completed", line, at, got)
	}
}

func commonPrefixOf(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	p := ss[0]
	for _, s := range ss[1:] {
		for !strings.HasPrefix(s, p) {
			p = p[:len(p)-1]
		}
	}
	return p
}

// The two near-empty lines must not be confused for one another: a bare prompt
// is a directory listing, a bare slash is still the command list.
func TestBareSlashStaysACommandList(t *testing.T) {
	root := completeTree(t)
	a := &App{cwd: root, projectRoot: root}

	at, got := a.complete("")
	if at != 0 || !slices.Equal(got, []string{"internal/", "notes.md"}) {
		t.Errorf("empty prompt = (%d, %v), want the working directory", at, got)
	}

	at, got = a.complete("/")
	if at != 0 || len(got) != len(commands) {
		t.Fatalf("/ = (%d, %v), want all %d commands", at, got, len(commands))
	}
	for _, c := range got {
		if !strings.HasPrefix(c, "/") || strings.Contains(c, "notes.md") {
			t.Errorf("/ offered %q, want a command name", c)
		}
	}
}

// A directory that cannot be read completes nothing. Tab is a keystroke: the
// only thing worse than no candidates is an error printed over the prompt.
func TestUnreadableDirectoryCompletesNothing(t *testing.T) {
	root := completeTree(t)

	missing := filepath.Join(root, "no-such-dir")
	if at, got := completePath(missing, root, ""); at != 0 || got != nil {
		t.Errorf("missing cwd = (%d, %v), want nothing", at, got)
	}

	if os.Geteuid() == 0 {
		t.Skip("root reads a 0000 directory regardless")
	}
	locked := filepath.Join(root, "locked")
	if err := os.Mkdir(locked, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(locked, 0o755) })
	if at, got := completePath(locked, root, ""); at != 0 || got != nil {
		t.Errorf("unreadable cwd = (%d, %v), want nothing", at, got)
	}
	if _, got := completePath(root, root, "locked/"); got != nil {
		t.Errorf("descending into an unreadable directory = %v, want nothing", got)
	}
}
