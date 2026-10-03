package main

import (
	"strings"
	"testing"
)

func TestResolveTools(t *testing.T) {
	known := []string{"bash", "read", "task", "todo", "status_update"}
	off := map[string]bool{"todo": true, "status_update": true}

	c, err := resolveTools(known, off, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !c.allow("bash") || !c.allow("task") || c.allow("todo") || c.allow("status_update") {
		t.Errorf("defaults: %v", c.enabled)
	}

	c, err = resolveTools(known, off, splitToolList("todo,status_update"), splitToolList("task bash"))
	if err != nil {
		t.Fatal(err)
	}
	if !c.allow("todo") || !c.allow("status_update") || c.allow("task") || c.allow("bash") || !c.allow("read") {
		t.Errorf("chosen: %v", c.enabled)
	}

	if _, err := resolveTools(known, off, []string{"websearch"}, nil); err == nil ||
		!strings.Contains(err.Error(), "off unless enabled: status_update, todo") {
		t.Errorf("unknown tool: %v", err)
	}
	if _, err := resolveTools(known, off, []string{"bash"}, []string{"bash"}); err == nil {
		t.Error("a tool both enabled and disabled was accepted")
	}
}
