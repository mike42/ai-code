package render

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"ai-code/internal/agent"
)

func newSteerRenderer(width int) (*Interactive, *bytes.Buffer) {
	var buf bytes.Buffer
	s := NewScreen(&buf, "never")
	s.isTTY, s.width = true, width

	r := &Interactive{
		screen: s, style: Style{}, showStatus: true, steerMark: "> ",
		streaming: true, turnStarted: time.Now(),
		stop: make(chan struct{}), done: make(chan struct{}),
	}
	return r, &buf
}

// The status line owns the bottom row until typing starts; the steering prompt
// takes it while typing, and clearing the line gives it back.
func TestSteeringPromptReplacesTheStatusLineAndGivesItBack(t *testing.T) {
	r, buf := newSteerRenderer(80)

	r.redraw()
	if got := Strip(buf.String()); !strings.Contains(got, "prefill") {
		t.Fatalf("bottom line = %q, want the status line before anything is typed", got)
	}

	buf.Reset()
	r.SetSteering("stop and check the tests", 24, true)
	got := Strip(buf.String())
	if !strings.Contains(got, "> stop and check the tests") {
		t.Errorf("bottom line = %q, want the steering prompt", got)
	}
	if strings.Contains(got, "prefill") || strings.Contains(got, "tok/s") {
		t.Errorf("bottom line = %q, want the status line replaced, not appended", got)
	}

	buf.Reset()
	r.SetSteering("", 0, false)
	got = Strip(buf.String())
	if !strings.Contains(got, "prefill") {
		t.Errorf("bottom line = %q, want the status line back once the line is cleared", got)
	}
	if strings.Contains(got, ">") {
		t.Errorf("bottom line = %q, want no prompt marker left behind", got)
	}
}

// The prompt is drawn in the transient zone, which is one row per line. Long
// input scrolls within the row rather than wrapping out of it.
func TestSteeringPromptScrollsRatherThanWrapping(t *testing.T) {
	r, buf := newSteerRenderer(40)
	text := strings.Repeat("abcdefghij", 12) // 120 columns of it

	r.SetSteering(text, len(text), true)
	out := buf.String()

	if strings.ContainsAny(out, "\n\r") {
		t.Errorf("steering prompt broke the line: %q", out)
	}
	if rows := countRows(out, 40); rows != 1 {
		t.Errorf("steering prompt occupies %d rows, want 1", rows)
	}
	if !strings.HasSuffix(Strip(stripCursorMoves(out)), "abcdefghij") {
		t.Errorf("steering prompt = %q, want the end of the text visible with the cursor", Strip(out))
	}
}

// The cursor has to sit where typing expects it, or editing anywhere but the
// end of the line is blind.
func TestSteeringPromptParksTheCursor(t *testing.T) {
	r, buf := newSteerRenderer(80)
	r.SetSteering("hello", 5, true)
	if got := buf.String(); strings.Contains(got, "\x1b[") && strings.Contains(got, "D") {
		if strings.Contains(got, "\x1b[1D") || strings.Contains(got, "\x1b[2D") {
			t.Errorf("cursor was moved back from the end of the line: %q", got)
		}
	}

	buf.Reset()
	r.SetSteering("hello", 2, true)
	// Marker is two columns, so the cursor belongs at column 4 of a 7-column
	// line: three columns back from the end.
	if got := buf.String(); !strings.Contains(got, "\x1b[3D") {
		t.Errorf("output = %q, want the cursor moved back three columns", got)
	}
}

// A steering prompt does not displace the line the model is streaming; it
// displaces the status line under it.
func TestSteeringPromptKeepsTheStreamingLineVisible(t *testing.T) {
	r, buf := newSteerRenderer(80)
	r.partial = []string{"func main() {"}

	r.SetSteering("wait", 4, true)
	got := Strip(buf.String())
	if !strings.Contains(got, "func main() {") {
		t.Errorf("output = %q, want the streaming line still shown", got)
	}
	if !strings.Contains(got, "> wait") {
		t.Errorf("output = %q, want the steering prompt below it", got)
	}
	if r.screen.transient != 2 {
		t.Errorf("transient rows = %d, want 2", r.screen.transient)
	}
}

// stripCursorMoves removes the trailing cursor-parking sequence so a test can
// assert on the text that was drawn.
func stripCursorMoves(s string) string {
	for {
		i := strings.LastIndex(s, "\x1b[")
		if i < 0 || !strings.HasSuffix(s, "D") {
			return s
		}
		s = s[:i]
	}
}

// Until the message is folded in, the status line the prompt just gave back is
// the only place that can acknowledge it.
func TestQueuedSteeringIsAcknowledgedInTheStatusLine(t *testing.T) {
	r, buf := newSteerRenderer(80)

	r.SetQueued(1)
	if got := Strip(buf.String()); !strings.Contains(got, "1 steering message queued") {
		t.Errorf("status = %q, want it to acknowledge the queued message", got)
	}

	r.SetQueued(2)
	buf.Reset()
	r.redraw()
	if got := Strip(buf.String()); !strings.Contains(got, "2 steering messages queued") {
		t.Errorf("status = %q, want a plural count", got)
	}

	// Folding one in clears it from the count.
	r.md = NewMarkdown(func(string) {}, func([]string) {}, Style{}, false, "", func() int { return 80 })
	r.Emit(agent.Event{Kind: agent.EvSteer, Text: "one of them"})
	buf.Reset()
	r.redraw()
	if got := Strip(buf.String()); !strings.Contains(got, "1 steering message queued") {
		t.Errorf("status = %q, want the count down to one", got)
	}
}

// A failed turn is a finished turn: a renderer left streaming would repaint the
// transient zone over the just-drawn prompt ten times a second.
func TestErrorEndsTheTurnForTheRenderer(t *testing.T) {
	r, buf := newSteerRenderer(80)
	r.md = NewMarkdown(func(string) {}, func([]string) {}, Style{}, false, "", func() int { return 80 })
	r.activeTool = "bash"

	r.Emit(agent.Event{Kind: agent.EvError, Text: "conversation is malformed and was not sent"})

	if r.streaming {
		t.Error("renderer still believes a turn is streaming after an error")
	}
	if r.activeTool != "" {
		t.Errorf("renderer still believes %q is running after an error", r.activeTool)
	}

	// And nothing is left in the transient zone to be repainted over the prompt.
	if r.screen.transient != 0 {
		t.Errorf("transient zone holds %d rows after an error, want 0", r.screen.transient)
	}
	buf.Reset()
	r.redraw()
	if got := Strip(buf.String()); strings.Contains(got, "bash") || strings.Contains(got, "prefill") {
		t.Errorf("a redraw after an error still paints a status line: %q", got)
	}
}
