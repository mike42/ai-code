package render

import (
	"io"
	"strings"
	"testing"

	"ai-code/internal/agent"
)

// statusTester is a renderer wired to a discarded screen: enough to drive
// Emit and read back the status line, without starting its paint loop.
func statusTester() *Interactive {
	return NewInteractive(NewScreen(io.Discard, "never"), InteractiveOptions{
		ShowStatus: true, WarnPercent: 80,
	})
}

// The status line adopts the context figure from wherever it arrives, and
// computes none of its own.
//
// It used to update on three event kinds out of twelve. A /compact emits none
// of them, so the pre-compaction occupancy stayed on screen until the next
// turn ended -- while the prompt marker, which asked the agent directly, had
// already moved. Two readers, two answers, one session.
func TestStatusLineAdoptsEveryPublishedFigure(t *testing.T) {
	r := statusTester()
	r.streaming = true

	r.Emit(agent.Event{Kind: agent.EvTurnEnd,
		Context: &agent.ContextState{Projected: 218_000, Window: 262_144, Anchored: true}})
	r.streaming = true // a finished turn stops the status line; the next one starts it
	if got := r.statusLocked(); !strings.Contains(got, "218.0k/262.1k ctx") {
		t.Fatalf("status = %q, want the figure from the finished turn", got)
	}

	// A bare context event, which is what making room emits.
	r.Emit(agent.Event{Kind: agent.EvContext,
		Context: &agent.ContextState{Projected: 70_000, Window: 262_144, Anchored: true}})
	if got := r.statusLocked(); !strings.Contains(got, "70.0k/262.1k ctx") {
		t.Errorf("status = %q, want the figure published after room was made", got)
	}
	if got := r.statusLocked(); strings.Contains(got, "218.0k") {
		t.Errorf("status = %q still shows the pre-compaction figure", got)
	}
}

// An estimate and a measurement look different, because they are.
func TestUnanchoredFigureIsMarked(t *testing.T) {
	r := statusTester()
	r.streaming = true

	r.Emit(agent.Event{Kind: agent.EvContext,
		Context: &agent.ContextState{Projected: 70_000, Window: 262_144, Anchored: false}})
	if got := r.statusLocked(); !strings.Contains(got, "~70.0k/262.1k ctx") {
		t.Errorf("status = %q, want an estimated figure marked as one", got)
	}

	r.Emit(agent.Event{Kind: agent.EvContext,
		Context: &agent.ContextState{Projected: 70_000, Window: 262_144, Anchored: true}})
	if got := r.statusLocked(); strings.Contains(got, "~") {
		t.Errorf("status = %q marks a reported figure as an estimate", got)
	}
}

// A context event commits nothing to the scrollback. One per tool result
// would fill it.
func TestContextEventWritesNothingToTheScrollback(t *testing.T) {
	r := statusTester()
	r.streaming = true
	before := r.screen
	r.Emit(agent.Event{Kind: agent.EvContext,
		Context: &agent.ContextState{Projected: 1, Window: 2, Anchored: true}})
	if r.screen != before {
		t.Error("a context event touched the screen")
	}
}
