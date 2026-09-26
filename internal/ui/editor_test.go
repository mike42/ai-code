package ui

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
)

// pipeInto writes text to a pipe and returns the read end as an *os.File, so
// the editor takes its non-terminal path exactly as it does under a real shell
// pipeline.
func pipeInto(t *testing.T, text string) *os.File {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		defer w.Close()
		_, _ = w.WriteString(text)
	}()
	return r
}

// A fresh bufio.Reader per ReadLine reads ahead and discards the buffer, so
// every line after the first vanishes.
func TestPipedInputReadsEveryLine(t *testing.T) {
	in := pipeInto(t, "first\n/tokens\n/compact\nlast\n")
	defer in.Close()

	e := NewEditor(in, os.Stdout)

	var got []string
	for range 4 {
		line, err := e.ReadLine("> ")
		if err != nil {
			t.Fatalf("ReadLine after %d lines: %v", len(got), err)
		}
		got = append(got, line)
	}

	want := []string{"first", "/tokens", "/compact", "last"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("line %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestPipedInputReportsEOF(t *testing.T) {
	in := pipeInto(t, "only\n")
	defer in.Close()

	e := NewEditor(in, os.Stdout)
	if _, err := e.ReadLine("> "); err != nil {
		t.Fatal(err)
	}
	if _, err := e.ReadLine("> "); err != ErrEOF {
		t.Errorf("second read = %v, want ErrEOF", err)
	}
}

func TestHistoryIgnoresBlanksAndImmediateRepeats(t *testing.T) {
	e := NewEditor(os.Stdin, os.Stdout)
	for _, s := range []string{"one", "one", "  ", "two"} {
		e.AddHistory(s)
	}
	if len(e.history) != 2 {
		t.Errorf("history = %v, want [one two]", e.history)
	}
}

func TestStripCommentsMatchesGitConvention(t *testing.T) {
	got := stripComments("real line\n# a comment\n  # indented comment\nanother\n")
	if strings.Contains(got, "comment") {
		t.Errorf("comment lines survived: %q", got)
	}
	for _, want := range []string{"real line", "another"} {
		if !strings.Contains(got, want) {
			t.Errorf("content line %q was removed", want)
		}
	}
}

func TestLongestCommonPrefix(t *testing.T) {
	if got := longestCommonPrefix([]string{"/compact", "/config"}); got != "/co" {
		t.Errorf("got %q, want %q", got, "/co")
	}
	if got := longestCommonPrefix([]string{"/a", "/b"}); got != "/" {
		t.Errorf("got %q, want %q", got, "/")
	}
}

// captureEditor gives an Editor whose output can be inspected. The output is a
// regular file rather than a terminal, so redraw falls back to an 80-column
// width and the byte stream is deterministic.
func captureEditor(t *testing.T) (*Editor, func() string) {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "out-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })

	e := NewEditor(f, f)
	read := func() string {
		off, err := f.Seek(0, os.SEEK_CUR)
		if err != nil {
			t.Fatal(err)
		}
		b := make([]byte, off)
		if _, err := f.ReadAt(b, 0); err != nil {
			t.Fatal(err)
		}
		if err := f.Truncate(0); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Seek(0, os.SEEK_SET); err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	return e, read
}

// Typing at the end of the line must emit that character and nothing else: an
// erase-and-repaint leaves the terminal a blank line to draw between the two,
// which flickers.
func TestRedrawEmitsOnlyTheTypedCharacter(t *testing.T) {
	e, out := captureEditor(t)

	e.redraw("> ")
	out() // discard the initial paint

	for _, r := range "hello" {
		e.insert(r)
		e.redraw("> ")
		if got := out(); got != string(r) {
			t.Fatalf("typing %q emitted %q, want %q", r, got, string(r))
		}
	}
}

// A pure cursor movement must not reprint any text.
func TestRedrawCursorMoveEmitsOnlyMovement(t *testing.T) {
	e, out := captureEditor(t)
	for _, r := range "abcdef" {
		e.insert(r)
	}
	e.redraw("> ")
	out()

	e.cursor = 0
	e.redraw("> ")
	if got := out(); got != "\x1b[6D" {
		t.Errorf("home = %q, want a bare 6-column move left", got)
	}

	e.cursor = 6
	e.redraw("> ")
	if got := out(); got != "\x1b[6C" {
		t.Errorf("end = %q, want a bare 6-column move right", got)
	}
}

// Deleting must erase to end of line, but only the tail -- and the erase comes
// after the text is written, never before it.
func TestRedrawDeleteErasesTailOnly(t *testing.T) {
	e, out := captureEditor(t)
	for _, r := range "abc" {
		e.insert(r)
	}
	e.redraw("> ")
	out()

	e.deleteBackward()
	e.redraw("> ")
	got := out()
	if strings.Contains(got, "\x1b[2K") {
		t.Errorf("backspace = %q, want no full-line erase", got)
	}
	if !strings.Contains(got, "\x1b[K") {
		t.Errorf("backspace = %q, want an erase-to-end-of-line", got)
	}
	if strings.Contains(got, "> ") {
		t.Errorf("backspace = %q, want the prompt left alone", got)
	}
}

// Every redraw must reach the terminal as one write. Several writes give the
// terminal several chances to render a half-updated line.
func TestRedrawIsASingleWrite(t *testing.T) {
	e, out := captureEditor(t)
	e.redraw("> ")
	got := out()
	if !strings.HasPrefix(got, "\r> ") {
		t.Errorf("first paint = %q, want prompt written after a carriage return", got)
	}
	if strings.Contains(got, "\x1b[2K") {
		t.Errorf("first paint = %q, want no blanking erase", got)
	}
}

// Reading a fixed two bytes of an escape sequence leaves the rest in the
// reader, where it is read back as ordinary typing.
func TestEscapeSequencesAreFullyConsumed(t *testing.T) {
	cases := []struct {
		name string
		seq  string
	}{
		{"ctrl-left", "[1;5D"},
		{"ctrl-right", "[1;5C"},
		{"alt-left", "[1;3D"},
		{"home-tilde", "[1~"},
		{"end-tilde", "[4~"},
		{"delete", "[3~"},
		{"shift-f5", "[15;2~"},
		{"ss3-home", "OH"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := NewEditor(os.Stdin, os.Stdout)
			e.buf = []rune("one two")
			e.cursor = len(e.buf)

			r := bufio.NewReader(strings.NewReader(tc.seq + "REST"))
			e.handleEscape(r)

			rest, _ := io.ReadAll(r)
			if string(rest) != "REST" {
				t.Errorf("left %q unconsumed, want the sequence fully eaten", rest)
			}
		})
	}
}

func TestWordMotionAndKilling(t *testing.T) {
	e := NewEditor(os.Stdin, os.Stdout)
	e.buf = []rune("alpha beta gamma")
	e.cursor = len(e.buf)

	if got := e.wordStart(e.cursor); got != 11 {
		t.Errorf("wordStart = %d, want 11 (start of \"gamma\")", got)
	}
	e.cursor = 0
	if got := e.wordEnd(e.cursor); got != 5 {
		t.Errorf("wordEnd = %d, want 5 (end of \"alpha\")", got)
	}

	e.cursor = len(e.buf)
	e.killBackwardWord()
	if string(e.buf) != "alpha beta " {
		t.Errorf("after Alt-Backspace = %q", string(e.buf))
	}
	if e.kill != "gamma" {
		t.Errorf("kill = %q, want %q", e.kill, "gamma")
	}

	e.insertRunes([]rune(e.kill))
	if string(e.buf) != "alpha beta gamma" {
		t.Errorf("after Ctrl-Y = %q", string(e.buf))
	}
	if e.cursor != len(e.buf) {
		t.Errorf("cursor = %d, want %d", e.cursor, len(e.buf))
	}
}

// Ctrl-W is whitespace-delimited even though Alt-Backspace is not: Ctrl-W on a
// path should take the whole path, not one segment of it.
func TestCtrlWTakesTheWholePath(t *testing.T) {
	e := NewEditor(os.Stdin, os.Stdout)
	e.buf = []rune("cat internal/ui/editor.go")
	e.cursor = len(e.buf)

	e.deleteWord()
	if string(e.buf) != "cat " {
		t.Errorf("Ctrl-W = %q, want %q", string(e.buf), "cat ")
	}

	e2 := NewEditor(os.Stdin, os.Stdout)
	e2.buf = []rune("cat internal/ui/editor.go")
	e2.cursor = len(e2.buf)
	e2.killBackwardWord()
	if string(e2.buf) != "cat internal/ui/editor." {
		t.Errorf("Alt-Backspace = %q, want it to stop at the segment", string(e2.buf))
	}
}

// Pasted terminal output routinely carries colour codes. Left in the buffer
// they would be written straight back out by redraw, corrupting the line, and
// then sent to the model as noise.
func TestPasteStripsControlSequences(t *testing.T) {
	got := string(sanitisePaste("\x1b[31mred\x1b[0m\tand\nmore\x07"))
	want := "red and\nmore"
	if got != want {
		t.Errorf("sanitisePaste = %q, want %q", got, want)
	}
}

func TestReverseSearchFindsAndSubmits(t *testing.T) {
	e, _ := captureEditor(t)
	e.AddHistory("go build ./...")
	e.AddHistory("git status")
	e.AddHistory("go test ./internal/ui")

	// "go" matches the newest entry; a second Ctrl-R steps further back.
	r := bufio.NewReader(strings.NewReader("go\x12\r"))
	if !e.reverseSearch(r, "> ") {
		t.Fatal("Enter should submit the match")
	}
	if string(e.buf) != "go build ./..." {
		t.Errorf("match = %q, want the second-newest \"go\" entry", string(e.buf))
	}
}

func TestReverseSearchCtrlCRestoresTheBuffer(t *testing.T) {
	e, _ := captureEditor(t)
	e.AddHistory("git status")
	e.buf = []rune("half-typed")
	e.cursor = len(e.buf)

	r := bufio.NewReader(strings.NewReader("git\x03"))
	if e.reverseSearch(r, "> ") {
		t.Fatal("Ctrl-C should not submit")
	}
	if string(e.buf) != "half-typed" {
		t.Errorf("buffer = %q, want what was being typed before the search", string(e.buf))
	}
}

func TestTranspose(t *testing.T) {
	e := NewEditor(os.Stdin, os.Stdout)
	e.buf = []rune("teh")
	e.cursor = 3
	e.transpose()
	if string(e.buf) != "the" {
		t.Errorf("transpose = %q, want %q", string(e.buf), "the")
	}
}

// Completion replaces the word it was given and nothing else. The buffer is a
// prompt, so swapping the whole buffer for the candidate would delete the rest
// of the sentence around the path.
func TestCompleteSplicesAWordAndKeepsTheRestOfTheLine(t *testing.T) {
	e, _ := captureEditor(t)
	e.buf = []rune("read internal/ui/ed and tell me what it does")
	e.cursor = len("read internal/ui/ed")
	e.Completions = func(line string) (int, []string) {
		if line != "read internal/ui/ed" {
			t.Errorf("callback got %q, want the line up to the cursor", line)
		}
		return len("read "), []string{"internal/ui/editor.go"}
	}

	e.complete()

	want := "read internal/ui/editor.go and tell me what it does"
	if string(e.buf) != want {
		t.Errorf("buffer = %q, want %q", string(e.buf), want)
	}
	if e.cursor != len("read internal/ui/editor.go") {
		t.Errorf("cursor = %d, want the end of the completed word", e.cursor)
	}
}

// Model ids share a long prefix and differ in a suffix -- a quantisation, a
// size, a :variant. The first Tab must insert exactly as much as is certain
// and stop at the divergence.
func TestCompleteInsertsTheCommonPrefixOnly(t *testing.T) {
	e, _ := captureEditor(t)
	e.buf = []rune("/model qwen")
	e.cursor = len(e.buf)
	e.Completions = func(string) (int, []string) {
		return len("/model "), []string{"qwen3-30b:free", "qwen3-30b:nitro"}
	}

	e.complete()

	if want := "/model qwen3-30b:"; string(e.buf) != want {
		t.Errorf("buffer = %q, want %q", string(e.buf), want)
	}
}

// Once the word already is the common prefix there is nothing to insert, so
// the candidates are listed instead. Reading that list is how the suffix gets
// chosen, which is why it is columns and not one space-joined line.
func TestCompleteListsCandidatesInColumns(t *testing.T) {
	e, out := captureEditor(t)
	e.buf = []rune("/model qwen3-30b:")
	e.cursor = len(e.buf)
	e.Completions = func(string) (int, []string) {
		return len("/model "), []string{"qwen3-30b:free", "qwen3-30b:nitro"}
	}

	e.complete()

	if string(e.buf) != "/model qwen3-30b:" {
		t.Errorf("buffer = %q, want it left alone", string(e.buf))
	}
	got := out()
	if got != "\r\nqwen3-30b:free   qwen3-30b:nitro\r\n" {
		t.Errorf("listing = %q", got)
	}
	// Something other than redraw has written to the terminal, so the next
	// redraw has to repaint rather than diff against a stale picture.
	if e.painted {
		t.Error("the listing did not invalidate the painted line")
	}
}

func TestCompleteListingIsCapped(t *testing.T) {
	e, out := captureEditor(t)
	var many []string
	for i := range maxListed + 5 {
		many = append(many, fmt.Sprintf("file%03d.go", i))
	}
	e.buf = []rune("file0")
	e.cursor = len(e.buf)
	e.Completions = func(string) (int, []string) { return 0, many }

	e.complete()

	got := out()
	if !strings.Contains(got, "... and 5 more") {
		t.Errorf("listing = %q, want the overflow noted rather than printed", got)
	}
	if strings.Contains(got, "file060.go") {
		t.Error("listing printed past the cap")
	}
}

// A candidate list the callback declines to offer must leave the line exactly
// as it was -- no bell, no blank line, no repaint.
func TestCompleteWithNoCandidatesDoesNothing(t *testing.T) {
	e, out := captureEditor(t)
	e.buf = []rune("hello wor")
	e.cursor = len(e.buf)
	e.Completions = func(string) (int, []string) { return 6, nil }

	e.complete()

	if string(e.buf) != "hello wor" || out() != "" {
		t.Errorf("buffer = %q, output = %q, want both untouched", string(e.buf), out())
	}
}

// Redraw scrolls a long line horizontally instead of wrapping it, so submit
// must put the whole line in the scrollback, not only the tail on screen.
func TestSubmitPutsTheWholeLineInTheScrollback(t *testing.T) {
	e, out := captureEditor(t)
	text := "START-" + strings.Repeat("abcdefghij ", 30) + "-END"
	for _, r := range text {
		e.insert(r)
	}
	e.redraw("> ")
	out()

	e.finalise("> ", text)
	got := out()

	if !strings.Contains(got, text) {
		t.Errorf("submitted line = %q, want the whole prompt", got)
	}
	if n := strings.Count(got, "-END"); n != 1 {
		t.Errorf("the prompt appears %d times on submit, want once", n)
	}
	if !strings.HasSuffix(got, "\r\n") {
		t.Errorf("submitted line = %q, want it to end the line", got)
	}
}

// A short line is already drawn where it will stay. Rewriting it in place is
// what keeps it from being echoed a second time by the app.
func TestSubmitDoesNotRepeatAShortLine(t *testing.T) {
	e, out := captureEditor(t)
	e.finalise("> ", "fix the failing test")
	got := out()
	if n := strings.Count(got, "fix the failing test"); n != 1 {
		t.Errorf("submit emitted %q, want the line exactly once", got)
	}
	if strings.Contains(got, "\x1b[2K") {
		t.Errorf("submit = %q, want no blanking erase", got)
	}
}

// Multi-line input stays summarised on the prompt line. The real text goes to
// the scrollback through render.Echo, and a raw newline here would split the
// row the erase arithmetic below counts on.
func TestSubmitKeepsMultilineInputSummarised(t *testing.T) {
	e, out := captureEditor(t)
	e.finalise("> ", "first line\nsecond line\nthird line")
	got := out()

	if strings.Contains(got, "second line") {
		t.Errorf("submit = %q, want multi-line content summarised, not drawn", got)
	}
	if !strings.Contains(got, "[3 lines, Ctrl-G to edit]") {
		t.Errorf("submit = %q, want the line-count summary", got)
	}
	if body := strings.TrimSuffix(got, "\r\n"); strings.ContainsAny(body, "\n") {
		t.Errorf("submit = %q, want a single physical row", got)
	}
}

// An enormous paste with its newlines stripped out is still one line, and
// reproducing all of it would scroll the session away.
func TestSubmitBoundsAnEnormousLine(t *testing.T) {
	e, out := captureEditor(t)
	huge := strings.Repeat("x", 100_000)
	e.finalise("> ", "START "+huge+" END")
	got := out()

	if len(got) > 4000 {
		t.Errorf("submit emitted %d bytes for a 100k-character line, want it capped", len(got))
	}
	for _, want := range []string{"START ", " END", "omitted"} {
		if !strings.Contains(got, want) {
			t.Errorf("submit = %q, want it to contain %q", got[:200], want)
		}
	}
}

// Piped input has no line to finalise: the terminal never drew one, and an
// escape sequence in a pipe or a CI log is corruption.
func TestPipedSubmitWritesNothingToTheTerminal(t *testing.T) {
	e, out := captureEditor(t)
	in := pipeInto(t, "do the thing\n")
	defer in.Close()
	e.in = in

	if line, err := e.ReadLine("> "); err != nil || line != "do the thing" {
		t.Fatalf("ReadLine = %q, %v", line, err)
	}
	if got := out(); got != "" {
		t.Errorf("piped read wrote %q to the terminal, want nothing", got)
	}
}
