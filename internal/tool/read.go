package tool

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

//go:embed read.txt
var readDescription string

type ReadTool struct {
	MaxBytes int
	MaxLines int
}

func (t *ReadTool) Name() string        { return "read" }
func (t *ReadTool) Description() string { return readDescription }
func (t *ReadTool) ReadOnly() bool      { return true }

func (t *ReadTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Path to the file, absolute or relative to the working directory."},
    "offset": {"type": "integer", "description": "1-based line number to start from. Omit to read from the beginning."},
    "limit": {"type": "integer", "description": "Maximum number of lines to return."}
  },
  "required": ["path"]
}`)
}

type readArgs struct {
	Path   string `json:"path"`
	Offset int    `json:"offset"`
	Limit  int    `json:"limit"`
}

func (t *ReadTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a readArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	path, err := resolve(st, a.Path)
	if err != nil {
		return Errorf("%v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Errorf("%s does not exist.%s", rel(st, path), suggestNeighbours(path))
		}
		if os.IsPermission(err) {
			return Errorf("%s cannot be read: permission denied.", rel(st, path))
		}
		if fi, statErr := os.Stat(path); statErr == nil && fi.IsDir() {
			return Errorf("%s is a directory, not a file. Use the ls tool to list its contents.", rel(st, path))
		}
		return Errorf("could not read %s: %v", rel(st, path), err)
	}

	if looksBinary(content) {
		return Errorf("%s appears to be a binary file (%s) and was not read.\n\n"+
			"If you need to inspect it, use the bash tool with something like "+
			"`file`, `xxd | head` or `strings`.", rel(st, path), humanBytes(len(content)))
	}

	// Stamp the full file, not the excerpt, so a later edit can detect changes underneath.
	st.SetStamp(stampFile(path, content))

	if len(content) == 0 {
		return Result{
			Content: fmt.Sprintf("%s is empty (0 bytes).", rel(st, path)),
			Display: fmt.Sprintf("Read %s (empty)", rel(st, path)),
		}
	}

	lines := strings.Split(strings.TrimSuffix(string(content), "\n"), "\n")
	total := len(lines)

	start := 0
	if a.Offset > 0 {
		start = a.Offset - 1
	}
	if start >= total {
		return Errorf("%s has %d lines; offset %d is past the end.",
			rel(st, path), total, a.Offset)
	}

	limit := a.Limit
	if limit <= 0 {
		limit = t.MaxLines
	}
	end := min(start+limit, total)

	var b strings.Builder
	width := len(fmt.Sprint(end))
	for i := start; i < end; i++ {
		fmt.Fprintf(&b, "%*d\t%s\n", width, i+1, lines[i])
	}

	out, truncated := Truncate(b.String(), t.MaxBytes)

	var notes []string
	if end < total {
		notes = append(notes, fmt.Sprintf("showing lines %d-%d of %d; "+
			"read again with offset=%d for more", start+1, end, total, end+1))
	}
	if truncated {
		notes = append(notes, "output exceeded the size limit and was truncated")
	}
	if len(notes) > 0 {
		out += "\n(" + strings.Join(notes, "; ") + ")\n"
	}

	display := fmt.Sprintf("Read %s (%d lines)", rel(st, path), total)
	if end-start < total {
		display = fmt.Sprintf("Read %s (lines %d-%d of %d)", rel(st, path), start+1, end, total)
	}
	return Result{Content: out, Display: display, Files: []FileStamp{stampFile(path, content)}}
}

// suggestNeighbours names similar files in the same directory; a typo or a
// wrong extension is the usual cause and the model can fix it immediately.
func suggestNeighbours(path string) string {
	dir, base := splitPath(path)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var close []string
	stem := stemOf(base)
	for _, e := range entries {
		name := e.Name()
		if name == base {
			continue
		}
		if stemOf(name) == stem || strings.HasPrefix(name, stem) || strings.HasPrefix(stem, stemOf(name)) {
			close = append(close, name)
		}
		if len(close) >= 6 {
			break
		}
	}
	if len(close) == 0 {
		return ""
	}
	return "\n\nSimilarly named files in that directory: " + strings.Join(close, ", ")
}

func splitPath(p string) (dir, base string) {
	i := strings.LastIndexByte(p, '/')
	if i < 0 {
		return ".", p
	}
	return p[:i], p[i+1:]
}

func stemOf(name string) string {
	if i := strings.LastIndexByte(name, '.'); i > 0 {
		return name[:i]
	}
	return name
}
