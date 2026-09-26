package main

import "testing"

// /verbose and /quiet change only how output is drawn; every other command
// waits for a turn boundary.
func TestRenderOnlyCommandsDoNotWaitForATurnBoundary(t *testing.T) {
	for _, line := range []string{"/verbose", "/quiet", "  /verbose  "} {
		if !renderOnlyCommand(line) {
			t.Errorf("%q should run immediately", line)
		}
	}
	// Anything that touches the conversation, the model or the process still
	// waits for the boundary.
	for _, line := range []string{"/compact", "/model foo", "/new", "/quit", "/mode review", "", "not a command"} {
		if renderOnlyCommand(line) {
			t.Errorf("%q must wait for the turn boundary", line)
		}
	}
}
