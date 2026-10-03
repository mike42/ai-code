package hosttool

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-code/internal/tool"
)

const todoFile = "todo.json"

const (
	Pending    = "pending"
	InProgress = "in_progress"
	Completed  = "completed"
	Cancelled  = "cancelled"
)

type TodoItem struct {
	Content string `json:"content"`
	Status  string `json:"status"`
}

type TodoList struct {
	Updated time.Time  `json:"updated"`
	Items   []TodoItem `json:"items"`
}

var todoSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "todos": {
      "type": "array",
      "description": "The whole list, in order, replacing what is there. Omit it to read the list back without changing it.",
      "items": {
        "type": "object",
        "properties": {
          "content": {"type": "string", "description": "What the task is, specific enough to act on."},
          "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]}
        },
        "required": ["content", "status"]
      }
    }
  }
}`)

const todoDescription = `Keep a todo list for the work in this workspace.

The list is kept outside the workspace and is still here the next time you are
started in it, so it is how unfinished work survives the end of a run. Call it
with no arguments to read it.

Each call with "todos" replaces the whole list, so send every item, in the
order you mean to work through them, each time.

Statuses: pending (not started), in_progress (being worked on now -- at most
one), completed (done and checked), cancelled (no longer needed).

Use it for work with three or more real steps, or when you are given several
tasks at once. Mark an item in_progress before you start it and completed as
soon as it is actually finished and verified, never on intent. If something is
blocked, leave it in_progress and add an item that says what blocks it. Skip it
for a single small change or a question.`

func (e *Executor) todoPath() string { return filepath.Join(e.dir, todoFile) }

func (e *Executor) loadTodos() (TodoList, error) {
	var list TodoList
	b, err := os.ReadFile(e.todoPath())
	if errors.Is(err, os.ErrNotExist) {
		return list, nil
	}
	if err != nil {
		return list, err
	}
	err = json.Unmarshal(b, &list)
	return list, err
}

func (e *Executor) todo(raw json.RawMessage) tool.Result {
	var args struct {
		Todos *[]TodoItem `json:"todos"`
	}
	if r := decode(raw, &args); r != nil {
		return *r
	}
	if args.Todos == nil {
		list, err := e.loadTodos()
		if err != nil {
			return tool.Errorf("the todo list at %s could not be read: %v", e.todoPath(), err)
		}
		if len(list.Items) == 0 {
			return tool.Result{Content: "The todo list is empty.", Display: "Todo: empty"}
		}
		return tool.Result{Content: todoText(list.Items), Display: todoSummary(list.Items)}
	}

	items := *args.Todos
	working := 0
	for i := range items {
		items[i].Content = strings.TrimSpace(items[i].Content)
		if items[i].Content == "" {
			return tool.Errorf("item %d has no content. Every item needs a description of the task.", i+1)
		}
		switch items[i].Status {
		case Pending, Completed, Cancelled:
		case InProgress:
			working++
		default:
			return tool.Errorf("item %d (%q) has status %q, which is not one of pending, in_progress, completed, cancelled.",
				i+1, items[i].Content, items[i].Status)
		}
	}
	if working > 1 {
		return tool.Errorf("%d items are in_progress. Only the one you are working on now should be; "+
			"set the others back to pending and send the list again.", working)
	}

	list := TodoList{Updated: e.now().UTC(), Items: items}
	b, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return tool.Errorf("encoding the todo list: %v", err)
	}
	if err := writeFile(e.todoPath(), append(b, '\n')); err != nil {
		return tool.Errorf("the todo list could not be saved to %s: %v", e.todoPath(), err)
	}
	return tool.Result{
		Content: "Saved.\n\n" + todoText(items),
		Display: todoSummary(items),
		Show:    todoText(items),
	}
}

func todoText(items []TodoItem) string {
	if len(items) == 0 {
		return "(empty)"
	}
	var b strings.Builder
	for _, it := range items {
		switch it.Status {
		case Completed:
			fmt.Fprintf(&b, "- [x] %s\n", it.Content)
		case InProgress:
			fmt.Fprintf(&b, "- [>] **%s**\n", it.Content)
		case Cancelled:
			fmt.Fprintf(&b, "- [-] %s (cancelled)\n", it.Content)
		default:
			fmt.Fprintf(&b, "- [ ] %s\n", it.Content)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func todoSummary(items []TodoItem) string {
	done, open := 0, 0
	current := ""
	for _, it := range items {
		switch it.Status {
		case Completed:
			done++
		case Cancelled:
		default:
			open++
		}
		if it.Status == InProgress {
			current = it.Content
		}
	}
	s := fmt.Sprintf("Todo: %d done, %d to go", done, open)
	if current != "" {
		s += " -- now: " + current
	}
	return s
}
