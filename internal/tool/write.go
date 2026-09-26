package tool

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed write.txt
var writeDescription string

type WriteTool struct{}

func (t *WriteTool) Name() string        { return "write" }
func (t *WriteTool) Description() string { return writeDescription }
func (t *WriteTool) ReadOnly() bool      { return false }

func (t *WriteTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Path to write, absolute or relative to the working directory."},
    "content": {"type": "string", "description": "The complete new contents of the file."}
  },
  "required": ["path", "content"]
}`)
}

type writeArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

func (t *WriteTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a writeArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	path, err := resolve(st, a.Path)
	if err != nil {
		return Errorf("%v", err)
	}

	existing, readErr := os.ReadFile(path)
	isNew := os.IsNotExist(readErr)

	if !isNew {
		if readErr != nil {
			return Errorf("could not read the existing %s to check for changes: %v", rel(st, path), readErr)
		}
		// Overwriting destroys everything not in `content`. Requiring a prior
		// read is what stops a whole-file write silently discarding parts of a
		// file the model never looked at.
		if r := checkStale(st, path, existing); r != nil {
			return *r
		}
	}

	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Errorf("could not create directory %s: %v", dir, err)
		}
	}
	if err := writeFilePreservingMode(path, []byte(a.Content)); err != nil {
		return Errorf("could not write %s: %v", rel(st, path), err)
	}

	stamp := stampFile(path, []byte(a.Content))
	st.SetStamp(stamp)

	verb := "Wrote"
	if isNew {
		verb = "Created"
	}
	summary := fmt.Sprintf("%s %s (%d lines, %s).",
		verb, rel(st, path), stamp.Lines, humanBytes(len(a.Content)))
	if !isNew {
		before := countLines(existing)
		summary += fmt.Sprintf(" Previous contents (%d lines) were replaced.", before)
	}
	return Result{
		Content: summary,
		Display: fmt.Sprintf("%s %s (%d lines)", verb, rel(st, path), stamp.Lines),
		Files:   []FileStamp{stamp},
	}
}

// ---------------------------------------------------------------------------

//go:embed ls.txt
var lsDescription string

type LsTool struct{}

func (t *LsTool) Name() string        { return "ls" }
func (t *LsTool) Description() string { return lsDescription }
func (t *LsTool) ReadOnly() bool      { return true }

func (t *LsTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Directory to list. Defaults to the working directory."},
    "all": {"type": "boolean", "description": "Include entries whose name begins with a dot."}
  }
}`)
}

type lsArgs struct {
	Path string `json:"path"`
	All  bool   `json:"all"`
}

func (t *LsTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a lsArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	if a.Path == "" {
		a.Path = "."
	}
	path, err := resolve(st, a.Path)
	if err != nil {
		return Errorf("%v", err)
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Errorf("%s does not exist.", rel(st, path))
		}
		return Errorf("could not list %s: %v", rel(st, path), err)
	}

	var dirs, files []string
	for _, e := range entries {
		name := e.Name()
		if !a.All && strings.HasPrefix(name, ".") {
			continue
		}
		if e.IsDir() {
			dirs = append(dirs, name+"/")
			continue
		}
		info, err := e.Info()
		if err != nil {
			files = append(files, name)
			continue
		}
		files = append(files, fmt.Sprintf("%s  (%s)", name, humanBytes(int(info.Size()))))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s\n", rel(st, path))
	for _, d := range dirs {
		fmt.Fprintf(&b, "  %s\n", d)
	}
	for _, f := range files {
		fmt.Fprintf(&b, "  %s\n", f)
	}
	if len(dirs)+len(files) == 0 {
		b.WriteString("  (empty)\n")
	}

	return Result{
		Content: b.String(),
		Display: fmt.Sprintf("Listed %s (%d entries)", rel(st, path), len(dirs)+len(files)),
	}
}
