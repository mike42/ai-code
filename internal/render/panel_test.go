package render

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"ai-code/internal/agent"
	"ai-code/internal/tool"
)

func TestToolPanelIsCommittedNotRedrawn(t *testing.T) {
	var buf bytes.Buffer
	s := NewScreen(&buf, "never")
	s.isTTY = true
	r := NewInteractive(s, InteractiveOptions{})

	var items []string
	for i := range 60 {
		items = append(items, fmt.Sprintf("- [ ] item %d", i))
	}
	r.Emit(agent.Event{Kind: agent.EvToolEnd, ToolName: "todo", ToolResult: &tool.Result{
		Display: "Todo: 0 done, 60 to go",
		Show:    strings.Join(items, "\n"),
	}})

	if s.transient != 0 {
		t.Errorf("the panel left %d rows in the redrawn zone", s.transient)
	}
	out := buf.String()
	if strings.Contains(out, cursorUp) {
		t.Error("printing a panel moved the cursor up, which rewrites what is above it")
	}
	if !strings.HasPrefix(out, "● Todo: 0 done, 60 to go\n") {
		t.Errorf("summary line missing: %q", out[:min(len(out), 80)])
	}
	for i := range 60 {
		if n := strings.Count(out, fmt.Sprintf("item %d\n", i)); n != 1 {
			t.Fatalf("item %d appears %d times in the scrollback, want 1", i, n)
		}
	}
}
