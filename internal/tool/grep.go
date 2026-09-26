package tool

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

//go:embed grep.txt
var grepDescription string

type GrepTool struct {
	MaxResults int
}

func (t *GrepTool) Name() string        { return "grep" }
func (t *GrepTool) Description() string { return grepDescription }
func (t *GrepTool) ReadOnly() bool      { return true }

func (t *GrepTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "RE2 regular expression to search for."},
    "path": {"type": "string", "description": "Directory to search. Defaults to the working directory."},
    "glob": {"type": "string", "description": "Only search files whose name matches this glob, e.g. *.go"},
    "ignore_case": {"type": "boolean", "description": "Match case-insensitively."},
    "files_only": {"type": "boolean", "description": "List matching file paths instead of matching lines."},
    "max_results": {"type": "integer", "description": "Cap the number of matching lines returned."}
  },
  "required": ["pattern"]
}`)
}

type grepArgs struct {
	Pattern    string `json:"pattern"`
	Path       string `json:"path"`
	Glob       string `json:"glob"`
	IgnoreCase bool   `json:"ignore_case"`
	FilesOnly  bool   `json:"files_only"`
	MaxResults int    `json:"max_results"`
}

type grepHit struct {
	path string
	line int
	text string
}

func (t *GrepTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a grepArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	if a.Pattern == "" {
		return Errorf("pattern is required.")
	}

	expr := a.Pattern
	if a.IgnoreCase {
		expr = "(?i)" + expr
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return Errorf("the pattern is not a valid RE2 regular expression: %v\n\n"+
			"RE2 has no backreferences or lookaround. If you need those, rephrase the "+
			"pattern or use the bash tool with grep or perl.", err)
	}

	searchRoot := a.Path
	if searchRoot == "" {
		searchRoot = "."
	}
	root, err := resolve(st, searchRoot)
	if err != nil {
		return Errorf("%v", err)
	}
	if fi, err := os.Stat(root); err != nil {
		return Errorf("%s does not exist.", rel(st, root))
	} else if !fi.IsDir() {
		return Errorf("%s is a file, not a directory. Use the read tool to read it, "+
			"or pass its directory as `path` with a `glob` that selects it.", rel(st, root))
	}

	limit := a.MaxResults
	if limit <= 0 {
		limit = t.MaxResults
	}
	if limit <= 0 {
		limit = 200
	}

	var (
		hits      []grepHit
		filesSeen = map[string]bool{}
		scanned   int
		capped    bool
	)

	walkErr := walkFiles(walkOptions{root: root, respectGit: true, maxFiles: 200_000},
		func(abs, relPath string) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if a.Glob != "" {
				ok, _ := filepath.Match(a.Glob, filepath.Base(relPath))
				if !ok {
					return nil
				}
			}
			if len(hits) >= limit {
				capped = true
				return filepath.SkipAll
			}

			f, err := os.Open(abs)
			if err != nil {
				return nil
			}
			defer f.Close()

			// Sniff for binary content before scanning: a packed object or a
			// compiled binary will match almost any pattern and produce noise.
			head := make([]byte, 4096)
			n, _ := f.Read(head)
			if looksBinary(head[:n]) {
				return nil
			}
			if _, err := f.Seek(0, 0); err != nil {
				return nil
			}

			scanned++
			sc := bufio.NewScanner(f)
			sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
			lineNo := 0
			for sc.Scan() {
				lineNo++
				line := sc.Text()
				if !re.MatchString(line) {
					continue
				}
				filesSeen[relPath] = true
				if a.FilesOnly {
					return nil
				}
				hits = append(hits, grepHit{path: relPath, line: lineNo, text: line})
				if len(hits) >= limit {
					capped = true
					return filepath.SkipAll
				}
			}
			return nil
		})
	if walkErr != nil && ctx.Err() != nil {
		return Errorf("search was interrupted.")
	}

	if a.FilesOnly {
		return t.filesOnlyResult(st, a, root, filesSeen, scanned, capped)
	}
	return t.linesResult(st, a, root, hits, filesSeen, scanned, capped, limit)
}

func (t *GrepTool) filesOnlyResult(st *State, a grepArgs, root string, files map[string]bool, scanned int, capped bool) Result {
	names := make([]string, 0, len(files))
	for f := range files {
		names = append(names, f)
	}
	sort.Strings(names)

	if len(names) == 0 {
		return Result{
			Content: noMatchesMessage(a, rel(st, root), scanned),
			Display: fmt.Sprintf("grep %q: no matches", a.Pattern),
		}
	}
	var b strings.Builder
	for _, n := range names {
		b.WriteString(n)
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "\n%s matched (%d files searched).", plural(len(names), "file"), scanned)
	return Result{
		Content: b.String(),
		Display: fmt.Sprintf("grep %q: %s", a.Pattern, plural(len(names), "file")),
	}
}

func (t *GrepTool) linesResult(st *State, a grepArgs, root string, hits []grepHit, files map[string]bool, scanned int, capped bool, limit int) Result {
	if len(hits) == 0 {
		return Result{
			Content: noMatchesMessage(a, rel(st, root), scanned),
			Display: fmt.Sprintf("grep %q: no matches", a.Pattern),
		}
	}

	var b strings.Builder
	current := ""
	for _, h := range hits {
		if h.path != current {
			if current != "" {
				b.WriteByte('\n')
			}
			b.WriteString(h.path)
			b.WriteByte('\n')
			current = h.path
		}
		text := h.text
		if len(text) > 300 {
			text = text[:300] + " ..."
		}
		fmt.Fprintf(&b, "%d: %s\n", h.line, strings.TrimRight(text, " \t"))
	}

	fmt.Fprintf(&b, "\n%s in %s (%d files searched).",
		plural(len(hits), "match"), plural(len(files), "file"), scanned)
	if capped {
		fmt.Fprintf(&b, "\nStopped at the %d-result limit; narrow the pattern, "+
			"set `path` to a subdirectory, or use `files_only` to see the spread first.", limit)
	}

	out, _ := Truncate(b.String(), 40000)
	return Result{
		Content: out,
		Display: fmt.Sprintf("grep %q: %s in %s", a.Pattern, plural(len(hits), "match"), plural(len(files), "file")),
	}
}

// noMatchesMessage says what was actually searched. "No matches" alone leaves
// the model unable to tell an absent symbol from a mistargeted search.
func noMatchesMessage(a grepArgs, root string, scanned int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "No matches for %q.\n\nSearched %d files under %s",
		a.Pattern, scanned, root)
	if a.Glob != "" {
		fmt.Fprintf(&b, " matching %s", a.Glob)
	}
	b.WriteString(".\nFiles ignored by .gitignore, dotfiles and binaries were skipped.")
	if scanned == 0 {
		b.WriteString("\n\nNothing was searched at all -- check the path and the glob.")
	}
	return b.String()
}

// ---------------------------------------------------------------------------

//go:embed glob.txt
var globDescription string

type GlobTool struct{}

func (t *GlobTool) Name() string        { return "glob" }
func (t *GlobTool) Description() string { return globDescription }
func (t *GlobTool) ReadOnly() bool      { return true }

func (t *GlobTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "pattern": {"type": "string", "description": "Glob to match against paths, e.g. **/*.go or internal/**/*_test.go"},
    "path": {"type": "string", "description": "Directory to search under. Defaults to the working directory."},
    "all": {"type": "boolean", "description": "Include dotfiles and files ignored by .gitignore."}
  },
  "required": ["pattern"]
}`)
}

type globArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
	All     bool   `json:"all"`
}

func (t *GlobTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a globArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	if a.Pattern == "" {
		return Errorf("pattern is required.")
	}
	searchRoot := a.Path
	if searchRoot == "" {
		searchRoot = "."
	}
	root, err := resolve(st, searchRoot)
	if err != nil {
		return Errorf("%v", err)
	}

	var matches []string
	err = walkFiles(walkOptions{root: root, respectGit: !a.All, includeAll: a.All,
		namedHidden: literalHiddenSegments(a.Pattern), maxFiles: 200_000},
		func(abs, relPath string) error {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if globMatch(a.Pattern, relPath) {
				matches = append(matches, relPath)
			}
			return nil
		})
	if err != nil && ctx.Err() != nil {
		return Errorf("search was interrupted.")
	}

	if len(matches) == 0 {
		hint := "Patterns are matched against paths relative to that directory. " +
			"Use `**/` to match at any depth, for example `**/*_test.go`."
		if !a.All {
			// Explaining `**/` to someone whose pattern was fine, and whose
			// real problem is that the walker skipped a dot directory, is the
			// expensive kind of wrong message.
			hint += " Hidden files are skipped unless the pattern names them " +
				"or `all` is set."
		}
		return Result{
			Content: fmt.Sprintf("No files match %q under %s.\n\n%s",
				a.Pattern, rel(st, root), hint),
			Display: fmt.Sprintf("glob %q: no matches", a.Pattern),
		}
	}
	sort.Strings(matches)

	var b strings.Builder
	for _, m := range matches {
		b.WriteString(m)
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "\n%s.", plural(len(matches), "file"))

	out, _ := Truncate(b.String(), 20000)
	return Result{
		Content: out,
		Display: fmt.Sprintf("glob %q: %s", a.Pattern, plural(len(matches), "file")),
	}
}

// globMatch supports `**` for any depth, which filepath.Match does not.
// literalHiddenSegments picks out the dot-prefixed path segments a pattern
// names outright, as opposed to ones it might happen to match. `.devcontainer`
// counts; `.*` does not, because a wildcard has not asked for anything in
// particular and letting it through would quietly turn every search into a
// search of .git.
func literalHiddenSegments(pattern string) map[string]bool {
	var out map[string]bool
	for _, seg := range strings.Split(pattern, "/") {
		if !strings.HasPrefix(seg, ".") || strings.ContainsAny(seg, "*?[]") {
			continue
		}
		if out == nil {
			out = map[string]bool{}
		}
		out[seg] = true
	}
	return out
}

func globMatch(pattern, name string) bool {
	if !strings.Contains(pattern, "**") {
		if ok, _ := filepath.Match(pattern, name); ok {
			return true
		}
		// A pattern with no directory component matches on basename, which is
		// what people mean by `*.go`.
		if !strings.Contains(pattern, "/") {
			ok, _ := filepath.Match(pattern, filepath.Base(name))
			return ok
		}
		return false
	}

	before, after, _ := strings.Cut(pattern, "**")
	before = strings.TrimSuffix(before, "/")
	after = strings.TrimPrefix(after, "/")

	if before != "" {
		if !strings.HasPrefix(name, before+"/") && name != before {
			return false
		}
		name = strings.TrimPrefix(name, before+"/")
	}
	if after == "" {
		return true
	}
	// Try the remainder against every suffix of the path.
	segments := strings.Split(name, "/")
	for i := range segments {
		if globMatch(after, strings.Join(segments[i:], "/")) {
			return true
		}
	}
	return false
}
