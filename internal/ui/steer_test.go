package ui

import (
	"strings"
	"testing"
)

// steerTester drives a Steerer's key handling without a terminal, recording
// every render so the transitions between prompt and status line are visible.
type steerTester struct {
	*Steerer
	renders   []painted
	submitted []string
	interrupt int
}

type painted struct {
	display string
	col     int
	active  bool
}

func newSteerTester() *steerTester {
	st := &steerTester{Steerer: NewSteerer(-1)}
	st.Render = func(d string, c int, a bool) {
		st.renders = append(st.renders, painted{d, c, a})
	}
	st.Submit = func(text string) { st.submitted = append(st.submitted, text) }
	st.Interrupt = func() { st.interrupt++ }
	return st
}

func (st *steerTester) type_(s string) { st.feed([]byte(s)) }

// feed delivers bytes one at a time, which is the hard case: every multi-byte
// sequence arrives split.
func (st *steerTester) feed(b []byte) {
	var pending []byte
	for _, c := range b {
		pending = st.consume(append(pending, c))
	}
}

func (st *steerTester) last() painted {
	if len(st.renders) == 0 {
		return painted{}
	}
	return st.renders[len(st.renders)-1]
}

// The interaction the prompt exists for: typing raises it, clearing it puts it
// away again. active is what the renderer keys on, so it has to flip on the
// keystroke that empties the line and not a moment later.
func TestPromptAppearsOnTypingAndGoesAwayWhenTheLineIsCleared(t *testing.T) {
	st := newSteerTester()

	if len(st.renders) != 0 {
		t.Fatal("something was rendered before a key was pressed")
	}

	st.type_("hi")
	if got := st.last(); !got.active || got.display != "hi" {
		t.Errorf("after typing: %+v, want the prompt active showing \"hi\"", got)
	}

	st.feed([]byte{127}) // backspace
	if got := st.last(); !got.active || got.display != "h" {
		t.Errorf("after one backspace: %+v, want the prompt still active", got)
	}

	st.feed([]byte{127})
	if got := st.last(); got.active {
		t.Errorf("after clearing the line: %+v, want the status line back", got)
	}
}

func TestCtrlUClearsTheLineAndRestoresTheStatusLine(t *testing.T) {
	st := newSteerTester()
	st.type_("some steering")
	st.feed([]byte{21})
	if got := st.last(); got.active || got.display != "" {
		t.Errorf("after Ctrl-U: %+v, want an empty inactive prompt", got)
	}
}

func TestEnterSubmitsAndLeavesTheLineEmpty(t *testing.T) {
	st := newSteerTester()
	st.type_("use the other approach\r")

	if len(st.submitted) != 1 || st.submitted[0] != "use the other approach" {
		t.Fatalf("submitted = %q, want one message", st.submitted)
	}
	if got := st.last(); got.active || got.display != "" {
		t.Errorf("after Enter: %+v, want the status line back", got)
	}
	if st.interrupt != 0 {
		t.Error("submitting a steering message cancelled the turn")
	}
}

func TestEmptyEnterSubmitsNothing(t *testing.T) {
	st := newSteerTester()
	st.feed([]byte{'\r'})
	st.type_("   \r")
	if len(st.submitted) != 0 {
		t.Errorf("submitted = %q, want nothing for a blank line", st.submitted)
	}
}

// Ctrl-C clears with text on the line and cancels on an empty one; cancelling
// straight away would throw away a message without showing that it had.
func TestCtrlCClearsTheLineFirstAndCancelsOnlyWhenEmpty(t *testing.T) {
	st := newSteerTester()
	st.type_("never mind")

	st.feed([]byte{3})
	if st.interrupt != 0 {
		t.Error("Ctrl-C cancelled the turn while a steering message was typed")
	}
	if got := st.last(); got.active {
		t.Errorf("after Ctrl-C: %+v, want the line cleared", got)
	}

	st.feed([]byte{3})
	if st.interrupt != 1 {
		t.Errorf("interrupts = %d after Ctrl-C on an empty line, want 1", st.interrupt)
	}
}

func TestCtrlDDoesNotExitMidTurn(t *testing.T) {
	st := newSteerTester()
	st.feed([]byte{4})
	if st.interrupt != 0 || len(st.submitted) != 0 {
		t.Error("Ctrl-D did something during a turn; it should be ignored")
	}
}

// A pasted snippet is one message, not one per line. Without bracketed paste
// handling every newline in it submits, and a pasted stack trace becomes twelve
// steering messages.
func TestBracketedPasteIsOneMessage(t *testing.T) {
	st := newSteerTester()
	st.type_("look at ")
	st.type_("\x1b[200~func main() {\n\tprintln(\"hi\")\n}\x1b[201~")

	if len(st.submitted) != 0 {
		t.Fatalf("a paste submitted itself: %q", st.submitted)
	}
	st.feed([]byte{'\r'})

	if len(st.submitted) != 1 {
		t.Fatalf("submitted %d messages, want 1: %q", len(st.submitted), st.submitted)
	}
	want := "look at func main() {\n\tprintln(\"hi\")\n}"
	if st.submitted[0] != strings.ReplaceAll(want, "\t", " ") {
		t.Errorf("submitted %q, want %q", st.submitted[0], strings.ReplaceAll(want, "\t", " "))
	}
}

// Multi-line input cannot be drawn on the one row the prompt owns, so it is
// summarised there -- and echoed in full into the scrollback when it is sent.
func TestMultilineInputIsSummarisedOnThePromptLine(t *testing.T) {
	st := newSteerTester()
	st.type_("\x1b[200~alpha\nbeta\ngamma\x1b[201~")

	got := st.last()
	if strings.Contains(got.display, "\n") {
		t.Errorf("prompt display = %q, want a single line", got.display)
	}
	if !strings.Contains(got.display, "3 lines") {
		t.Errorf("prompt display = %q, want it to say how many lines", got.display)
	}
}

// An escape sequence split across reads must not be typed into the buffer as
// literal characters.
func TestSplitEscapeSequencesAreNotInsertedAsText(t *testing.T) {
	st := newSteerTester()
	st.type_("abc def")
	st.type_("\x1b[1;5D") // Ctrl-Left: back one word, to column 4
	st.type_("\x1b[D")    // Left: to column 3

	if got := st.last().display; got != "abc def" {
		t.Errorf("display = %q, want the escape sequences consumed, not inserted", got)
	}
	if got := st.last().col; got != 3 {
		t.Errorf("cursor column = %d, want 3 after Ctrl-Left then Left", got)
	}
}

func TestArrowsMoveTheCursorAndInsertionHappensThere(t *testing.T) {
	st := newSteerTester()
	st.type_("helo")
	st.type_("\x1b[D") // left, cursor between l and o
	st.type_("l")

	if got := st.last().display; got != "hello" {
		t.Errorf("display = %q, want %q", got, "hello")
	}
	st.feed([]byte{'\r'})
	if st.submitted[0] != "hello" {
		t.Errorf("submitted %q, want %q", st.submitted[0], "hello")
	}
}

// Multi-byte runes split across reads must survive as one character.
func TestSplitMultibyteRunes(t *testing.T) {
	st := newSteerTester()
	st.type_("naïve 日本")
	if got := st.last().display; got != "naïve 日本" {
		t.Errorf("display = %q, want %q", got, "naïve 日本")
	}
}

// Up and Down belong to the prompt's history, not here: pulling an old command
// over a message being written mid-turn loses the message.
func TestHistoryKeysAreIgnoredWhileSteering(t *testing.T) {
	st := newSteerTester()
	st.type_("half a thought")
	st.type_("\x1b[A\x1b[B")
	if got := st.last().display; got != "half a thought" {
		t.Errorf("display = %q, want it untouched by Up/Down", got)
	}
}

// Tab completes a path while a turn is running, as it does at the prompt. Only
// the ambiguous-match listing needs the screen; extending the word touches
// nothing but the buffer.
func TestTabCompletesWhileSteering(t *testing.T) {
	st := newSteerTester()
	st.Completions = func(line string) (int, []string) {
		at := strings.LastIndexByte(line, ' ') + 1
		var out []string
		for _, c := range []string{"internal/render/", "internal/runtime/"} {
			if strings.HasPrefix(c, line[at:]) {
				out = append(out, c)
			}
		}
		return at, out
	}

	// One match: the word is completed outright.
	st.type_("look at internal/rend\t")
	if got := st.last().display; got != "look at internal/render/" {
		t.Errorf("display = %q, want the completed path", got)
	}

	// Several matches: extended as far as they agree, and no further.
	st2 := newSteerTester()
	st2.Completions = st.Completions
	st2.type_("look at internal/r\t")
	if got := st2.last().display; got != "look at internal/r" {
		t.Errorf("display = %q, want it left alone: the candidates share no more", got)
	}

	// Tab is never inserted as a character, whether or not it completed.
	for _, r := range st.renders {
		if strings.ContainsRune(r.display, '\t') {
			t.Errorf("a tab reached the buffer: %q", r.display)
		}
	}
}

// Completion must not eat the rest of a sentence the cursor sits inside.
func TestTabCompletesOnlyTheWordBeforeTheCursor(t *testing.T) {
	st := newSteerTester()
	st.Completions = func(line string) (int, []string) {
		at := strings.LastIndexByte(line, ' ') + 1
		return at, []string{"cmd/ai-code/app.go"}
	}
	st.type_(" and hurry")
	// Home, then type a path at the start of the line.
	st.feed([]byte("\x1b[H"))
	st.type_("read cmd/ai\t")

	if got := st.last().display; got != "read cmd/ai-code/app.go and hurry" {
		t.Errorf("display = %q, want the tail preserved", got)
	}
}

// With nothing wired up, Tab is inert rather than a panic.
func TestTabWithoutCompletionsDoesNothing(t *testing.T) {
	st := newSteerTester()
	st.type_("plain\t")
	if got := st.last().display; got != "plain" {
		t.Errorf("display = %q, want the line untouched", got)
	}
}
