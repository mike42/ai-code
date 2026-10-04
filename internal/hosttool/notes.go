package hosttool

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"ai-code/internal/tool"
)

const notesDir = "notes"

var noteName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$`)

var notesSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "action": {"type": "string", "enum": ["list", "read", "write", "delete"]},
    "name": {"type": "string", "description": "The note's name: letters, digits, dot, dash and underscore, e.g. api-keys or ci-notes.md. Needed for read, write and delete."},
    "content": {"type": "string", "description": "For write: the whole note, replacing what is there."}
  },
  "required": ["action"]
}`)

const notesDescription = `Keep private notes outside the workspace.

Notes are stored on the machine that started you, not where your commands run,
so nothing else working in or near your workspace can read them. They are still
here the next time you are started. Credentials and settings you have been given
for your work are kept here too: list your notes before you go looking for them.

Use it for things that must not sit in the workspace -- keys, tokens, account
details -- and for your own working knowledge between runs: what you checked,
what you learned about the systems you work on, what to look at next time.

Actions: list (names and sizes), read, write (replaces the whole note), delete.`

func (e *Executor) notePath(name string) (string, *tool.Result) {
	if !noteName.MatchString(name) {
		r := tool.Errorf("%q is not a usable note name: use letters, digits, dot, dash and underscore, starting with a letter or digit.", name)
		return "", &r
	}
	return filepath.Join(e.dir, notesDir, name), nil
}

func (e *Executor) notes(raw json.RawMessage) tool.Result {
	var args struct {
		Action  string  `json:"action"`
		Name    string  `json:"name"`
		Content *string `json:"content"`
	}
	if r := decode(raw, &args); r != nil {
		return *r
	}
	if args.Action == "list" {
		return e.listNotes()
	}
	path, bad := e.notePath(args.Name)
	switch args.Action {
	case "read", "write", "delete":
		if bad != nil {
			return *bad
		}
	default:
		return tool.Errorf("action must be list, read, write or delete, not %q.", args.Action)
	}
	switch args.Action {
	case "read":
		b, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			return tool.Errorf("there is no note named %q.", args.Name)
		}
		if err != nil {
			return tool.Errorf("could not read note %q: %v", args.Name, err)
		}
		return tool.Result{Content: string(b), Display: "Notes: read " + args.Name}
	case "write":
		if args.Content == nil {
			return tool.Errorf("write needs content: the whole note.")
		}
		if err := writeFile(path, []byte(*args.Content)); err != nil {
			return tool.Errorf("could not save note %q: %v", args.Name, err)
		}
		return tool.Result{Content: "Saved.", Display: "Notes: wrote " + args.Name}
	default:
		err := os.Remove(path)
		if errors.Is(err, os.ErrNotExist) {
			return tool.Errorf("there is no note named %q.", args.Name)
		}
		if err != nil {
			return tool.Errorf("could not delete note %q: %v", args.Name, err)
		}
		return tool.Result{Content: "Deleted.", Display: "Notes: deleted " + args.Name}
	}
}

type noteInfo struct {
	name string
	size int64
}

func (e *Executor) noteList() []noteInfo {
	entries, err := os.ReadDir(filepath.Join(e.dir, notesDir))
	if err != nil {
		return nil
	}
	var out []noteInfo
	for _, ent := range entries {
		if !ent.Type().IsRegular() || !noteName.MatchString(ent.Name()) {
			continue
		}
		info, err := ent.Info()
		if err != nil {
			continue
		}
		out = append(out, noteInfo{ent.Name(), info.Size()})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out
}

func (e *Executor) listNotes() tool.Result {
	list := e.noteList()
	if len(list) == 0 {
		return tool.Result{Content: "No notes yet.", Display: "Notes: none"}
	}
	var b strings.Builder
	for _, n := range list {
		fmt.Fprintf(&b, "%s (%d bytes)\n", n.name, n.size)
	}
	return tool.Result{Content: b.String(), Display: fmt.Sprintf("Notes: %d", len(list))}
}
