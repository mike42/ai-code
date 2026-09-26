package main

import "testing"

// /verbose and /quiet must not wait for a turn boundary.
//
// A slash command typed mid-turn is queued until the loop reaches a point
// where the conversation can safely change, which is after the next round of
// tool results. For a command that alters the conversation that is right. For
// these two it is useless: they change only how output is drawn, and on a
// model thinking at a few tokens a second the next boundary is minutes away
// -- by which time the thinking the user turned verbose on to read is over.
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
