package hosttool

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"ai-code/internal/tool"
)

const (
	statusFile = "status.md"
	statusDir  = "status"
)

var statusSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "title": {"type": "string", "description": "One line: the headline of this update."},
    "body": {"type": "string", "description": "The update itself, in markdown."}
  },
  "required": ["title", "body"]
}`)

const statusDescription = `Report on your work to the person responsible for it.

Each update is saved where they read it, outside the workspace. Your latest
update is also given back to you the next time you are started here, so the
last one you write before you stop should say where the work stands: what is
done, what is not, what you found that matters, and what you would do next.

Write one when something important happens -- a result, a blocker, a decision
someone else needs to make -- and before you stop. Say what is true now; do not
narrate the steps that got you here.`

func (e *Executor) status(raw json.RawMessage) tool.Result {
	var args struct {
		Title string `json:"title"`
		Body  string `json:"body"`
	}
	if r := decode(raw, &args); r != nil {
		return *r
	}
	title := strings.Join(strings.Fields(args.Title), " ")
	body := strings.TrimSpace(args.Body)
	if title == "" || body == "" {
		return tool.Errorf("an update needs both a title and a body.")
	}

	now := e.now().UTC()
	doc := fmt.Sprintf("# %s\n\n_%s_\n\n%s\n", title, now.Format("2006-01-02 15:04:05 UTC"), body)

	history := filepath.Join(e.dir, statusDir, now.Format("20060102T150405.000000000Z")+".md")
	if err := writeFile(history, []byte(doc)); err != nil {
		return tool.Errorf("the status update could not be saved to %s: %v", history, err)
	}
	if err := writeFile(filepath.Join(e.dir, statusFile), []byte(doc)); err != nil {
		return tool.Errorf("the status update could not be saved to %s: %v", filepath.Join(e.dir, statusFile), err)
	}
	return tool.Result{
		Content: "Saved.",
		Display: "Status: " + title,
		Show:    body,
	}
}
