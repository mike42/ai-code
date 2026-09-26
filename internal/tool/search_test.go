package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// searchFixture builds a small tree with a .gitignore, mirroring the shape of a
// real repository: source worth searching, build output that is not.
func searchFixture(t *testing.T) *State {
	t.Helper()
	dir := t.TempDir()

	files := map[string]string{
		".gitignore":               "node_modules/\ndist/\n*.log\n",
		"main.go":                  "package main\n\nfunc main() {\n\tconnectDatabase()\n}\n",
		"db.go":                    "package main\n\nfunc connectDatabase() error {\n\treturn nil\n}\n",
		"internal/api/api.go":      "package api\n\nfunc Handler() {}\n",
		"internal/api/api_test.go": "package api\n\nfunc TestHandler(t *testing.T) {}\n",
		"dist/bundle.js":           "function connectDatabase(){}\n",
		"node_modules/lib/x.js":    "connectDatabase();\n",
		"debug.log":                "connectDatabase called\n",
		"README.md":                "# Project\n\nCall connectDatabase to start.\n",
	}
	for rel, body := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// A binary that would match almost any pattern if it were scanned.
	if err := os.WriteFile(filepath.Join(dir, "blob.bin"),
		append([]byte("connectDatabase"), 0x00, 0x01, 0x02), 0o644); err != nil {
		t.Fatal(err)
	}
	return NewState(dir)
}

func TestGrepSkipsIgnoredTreesBinariesAndDotfiles(t *testing.T) {
	st := searchFixture(t)
	res := (&GrepTool{MaxResults: 100}).Run(context.Background(), st,
		mustJSON(t, grepArgs{Pattern: "connectDatabase"}))
	if res.IsError {
		t.Fatalf("grep failed: %s", res.Content)
	}

	for _, want := range []string{"main.go", "db.go", "README.md"} {
		if !strings.Contains(res.Content, want) {
			t.Errorf("expected a match in %s:\n%s", want, res.Content)
		}
	}
	// These are the results that make a code search useless when they appear.
	for _, unwanted := range []string{"node_modules", "dist/", "debug.log", "blob.bin"} {
		if strings.Contains(res.Content, unwanted) {
			t.Errorf("search reached %s, which .gitignore excludes or which is binary:\n%s",
				unwanted, res.Content)
		}
	}
}

func TestGrepReportsLineNumbers(t *testing.T) {
	st := searchFixture(t)
	res := (&GrepTool{MaxResults: 100}).Run(context.Background(), st,
		mustJSON(t, grepArgs{Pattern: "func connectDatabase"}))
	if !strings.Contains(res.Content, "3:") {
		t.Errorf("expected the match reported at line 3:\n%s", res.Content)
	}
}

func TestGrepGlobFilter(t *testing.T) {
	st := searchFixture(t)
	res := (&GrepTool{MaxResults: 100}).Run(context.Background(), st,
		mustJSON(t, grepArgs{Pattern: "connectDatabase", Glob: "*.md"}))
	if !strings.Contains(res.Content, "README.md") {
		t.Errorf("glob filter excluded the file it should have kept:\n%s", res.Content)
	}
	if strings.Contains(res.Content, "main.go") {
		t.Errorf("glob filter did not exclude .go files:\n%s", res.Content)
	}
}

func TestGrepNoMatchesExplainsWhatWasSearched(t *testing.T) {
	// "No matches" alone leaves the model unable to distinguish an absent
	// symbol from a mistargeted search.
	st := searchFixture(t)
	res := (&GrepTool{MaxResults: 100}).Run(context.Background(), st,
		mustJSON(t, grepArgs{Pattern: "definitelyNotPresentXyz"}))
	if res.IsError {
		t.Fatalf("no matches is not an error: %s", res.Content)
	}
	if !strings.Contains(res.Content, "Searched") {
		t.Errorf("should report how many files were searched:\n%s", res.Content)
	}
	if !strings.Contains(res.Content, "gitignore") {
		t.Errorf("should state that ignored files were skipped:\n%s", res.Content)
	}
}

func TestGrepRejectsUnsupportedRegexWithGuidance(t *testing.T) {
	st := searchFixture(t)
	res := (&GrepTool{MaxResults: 100}).Run(context.Background(), st,
		mustJSON(t, grepArgs{Pattern: `(?=lookahead)`}))
	if !res.IsError {
		t.Fatal("expected an error for an unsupported construct")
	}
	if !strings.Contains(res.Content, "backreferences") {
		t.Errorf("message should explain RE2's limits and the way around them:\n%s", res.Content)
	}
}

func TestGrepFilesOnly(t *testing.T) {
	st := searchFixture(t)
	res := (&GrepTool{MaxResults: 100}).Run(context.Background(), st,
		mustJSON(t, grepArgs{Pattern: "connectDatabase", FilesOnly: true}))
	if strings.Contains(res.Content, "3:") {
		t.Errorf("files_only should not report line contents:\n%s", res.Content)
	}
	if !strings.Contains(res.Content, "main.go") {
		t.Errorf("expected main.go in the file list:\n%s", res.Content)
	}
}

func TestGlobMatchesAtAnyDepth(t *testing.T) {
	st := searchFixture(t)
	res := (&GlobTool{}).Run(context.Background(), st, mustJSON(t, globArgs{Pattern: "**/*_test.go"}))
	if res.IsError {
		t.Fatalf("glob failed: %s", res.Content)
	}
	if !strings.Contains(res.Content, "internal/api/api_test.go") {
		t.Errorf("** should match at any depth:\n%s", res.Content)
	}
	if strings.Contains(res.Content, "main.go\n") {
		t.Errorf("pattern matched files it should not have:\n%s", res.Content)
	}
}

func TestGlobBareExtensionMatchesBasename(t *testing.T) {
	// `*.go` is what people mean by "the Go files", at any depth.
	st := searchFixture(t)
	res := (&GlobTool{}).Run(context.Background(), st, mustJSON(t, globArgs{Pattern: "*.go"}))
	for _, want := range []string{"main.go", "internal/api/api.go"} {
		if !strings.Contains(res.Content, want) {
			t.Errorf("expected %s:\n%s", want, res.Content)
		}
	}
}

func TestGlobRespectsGitignore(t *testing.T) {
	st := searchFixture(t)
	res := (&GlobTool{}).Run(context.Background(), st, mustJSON(t, globArgs{Pattern: "**/*.js"}))
	if strings.Contains(res.Content, "node_modules") || strings.Contains(res.Content, "dist/") {
		t.Errorf("glob reached ignored directories:\n%s", res.Content)
	}
}

func TestGlobNoMatchExplainsThePatternSyntax(t *testing.T) {
	st := searchFixture(t)
	res := (&GlobTool{}).Run(context.Background(), st, mustJSON(t, globArgs{Pattern: "*.rs"}))
	if !strings.Contains(res.Content, "**") {
		t.Errorf("a no-match result should remind the model how depth matching works:\n%s", res.Content)
	}
}

func TestLsSeparatesDirectoriesFromFiles(t *testing.T) {
	st := searchFixture(t)
	res := (&LsTool{}).Run(context.Background(), st, mustJSON(t, lsArgs{}))
	if res.IsError {
		t.Fatalf("ls failed: %s", res.Content)
	}
	if !strings.Contains(res.Content, "internal/") {
		t.Errorf("directories should be marked with a trailing slash:\n%s", res.Content)
	}
	if strings.Contains(res.Content, ".gitignore") {
		t.Errorf("dotfiles should be hidden unless requested:\n%s", res.Content)
	}
}

func TestGlobFindsHiddenDirectoriesThePatternNames(t *testing.T) {
	dir := t.TempDir()
	for _, p := range []string{".devcontainer", "sub/.devcontainer"} {
		if err := os.MkdirAll(filepath.Join(dir, p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, p, "devcontainer.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// A hidden directory nobody asked for stays hidden.
	if err := os.MkdirAll(filepath.Join(dir, ".cache"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".cache", "devcontainer.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	st := NewState(dir)
	glob := func(pattern string) string {
		args, _ := json.Marshal(map[string]any{"pattern": pattern})
		return (&GlobTool{}).Run(context.Background(), st, args).Content
	}

	got := glob("**/.devcontainer/**")
	for _, want := range []string{".devcontainer/devcontainer.json", "sub/.devcontainer/devcontainer.json"} {
		if !strings.Contains(got, want) {
			t.Errorf("glob missed %s:\n%s", want, got)
		}
	}
	if strings.Contains(got, ".cache") {
		t.Errorf("glob returned a hidden directory the pattern did not name:\n%s", got)
	}

	// A wildcard has not asked for anything in particular, so it must not
	// turn into a search of every dot directory.
	if out := glob("**/*.json"); strings.Contains(out, ".cache") {
		t.Errorf("a wildcard pattern reached hidden files:\n%s", out)
	}
}
