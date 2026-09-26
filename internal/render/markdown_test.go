package render

import (
	"strings"
	"testing"
)

// capture collects committed lines and the latest partial line, which is
// exactly the two-stream contract the terminal renderer consumes.
type capture struct {
	committed []string
	partial   []string
}

func newCapture(color bool, theme string) (*capture, *Markdown) {
	return newCaptureWidth(color, theme, 80)
}

func newCaptureWidth(color bool, theme string, width int) (*capture, *Markdown) {
	c := &capture{}
	md := NewMarkdown(
		func(l string) { c.committed = append(c.committed, l) },
		func(p []string) { c.partial = p },
		NewStyle(color), color, theme, func() int { return width },
	)
	return c, md
}

// feed writes text in small chunks, as a real stream arrives, so the renderer
// cannot depend on receiving whole lines.
func feed(md *Markdown, text string, chunk int) {
	for len(text) > chunk {
		md.Write(text[:chunk])
		text = text[chunk:]
	}
	md.Write(text)
}

func TestCommittedOutputHasNoTrailingWhitespace(t *testing.T) {
	// The pi complaint, mechanically enforced: selecting output must not pick
	// up trailing spaces.
	c, md := newCapture(true, "auto")
	feed(md, "Some prose with trailing spaces   \n\n```go\nfunc x() {   \n}\n```\nmore   \n", 5)
	md.Flush()

	for i, line := range c.committed {
		if line != strings.TrimRight(line, " \t") {
			t.Errorf("committed line %d has trailing whitespace: %q", i, line)
		}
	}
}

func TestCommittedContentHasNoLeftGutter(t *testing.T) {
	// The other half of the pi complaint: code must be selectable at its real
	// indentation, with nothing prepended.
	c, md := newCapture(false, "none")
	feed(md, "```go\nfunc main() {\n\tprintln(\"hi\")\n}\n```\n", 4)
	md.Flush()

	want := []string{"func main() {", "\tprintln(\"hi\")", "}"}
	var got []string
	for _, l := range c.committed {
		if s := Strip(l); strings.TrimSpace(s) != "" {
			got = append(got, s)
		}
	}
	if len(got) != len(want) {
		t.Fatalf("got %d code lines %q, want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("line %d = %q, want %q -- code must be committed at its own "+
				"indentation with no prefix", i, got[i], want[i])
		}
	}
}

func TestCodeIsSyntaxHighlightedLineByLine(t *testing.T) {
	c, md := newCapture(true, "auto")
	feed(md, "```go\nfunc main() {\n\tx := 1\n}\n```\n", 3)
	md.Flush()

	var codeLines []string
	for _, l := range c.committed {
		if strings.Contains(Strip(l), "func main") || strings.Contains(Strip(l), "x := 1") {
			codeLines = append(codeLines, l)
		}
	}
	if len(codeLines) == 0 {
		t.Fatal("no code lines were committed")
	}
	for _, l := range codeLines {
		if !strings.Contains(l, esc) {
			t.Errorf("code line was committed unhighlighted: %q", Strip(l))
		}
	}
}

func TestNothingIsCommittedUntilItsNewlineArrives(t *testing.T) {
	// The core guarantee. A line stays revisable in the transient zone until
	// its final form is known, so the scrollback never contains text that
	// would have needed restyling.
	c, md := newCapture(true, "auto")

	md.Write("```go\nfunc ")
	if len(c.committed) != 0 {
		t.Fatalf("committed %q before any line was complete", c.committed)
	}
	md.Write("main() {")
	if len(c.committed) != 0 {
		t.Fatalf("committed %q before the newline arrived", c.committed)
	}
	md.Write("\n")
	if len(c.committed) != 1 {
		t.Fatalf("expected exactly one committed line, got %q", c.committed)
	}
}

func TestPartialLineIsPreviewedAsItWillCommit(t *testing.T) {
	// The markers are never shown, so they never have to be taken away. The
	// transient zone is repainted every frame, so guessing there is free.
	c, md := newCapture(true, "auto")
	md.Write("Some **bold te")

	if len(c.partial) != 1 {
		t.Fatalf("partial = %q, want one row", c.partial)
	}
	if got := Strip(c.partial[0]); got != "Some bold te" {
		t.Errorf("partial = %q, want the markers already resolved away", got)
	}
	if !strings.Contains(c.partial[0], codeBold) {
		t.Errorf("partial = %q, want the open span styled speculatively", c.partial[0])
	}
	if c.committed != nil {
		t.Errorf("nothing may be committed before the newline, got %q", c.committed)
	}
}

func TestPreviewHoldsBackADelimiterStillBeingTyped(t *testing.T) {
	// A lone "*" on its way to "**" must not appear for one frame and vanish.
	c, md := newCapture(true, "auto")
	for _, text := range []string{"text ", "*", "*", "b"} {
		md.Write(text)
		if strings.Contains(Strip(c.partial[0]), "*") {
			t.Fatalf("partial = %q, want the pending delimiter held back", Strip(c.partial[0]))
		}
	}
	if got := Strip(c.partial[0]); got != "text b" {
		t.Errorf("partial = %q, want %q", got, "text b")
	}
}

func TestPreviewDoesNotShiftWhenTheLineLands(t *testing.T) {
	// The whole point: what is previewed is what gets committed, so the text
	// does not move when the newline arrives.
	c, md := newCaptureWidth(true, "auto", 40)
	const line = "A sentence with **bold** and `code` in it that is long enough to wrap once.\n"

	feed(md, strings.TrimSuffix(line, "\n"), 6)
	preview := append([]string(nil), c.partial...)
	md.Write("\n")
	md.Flush()

	if len(c.committed) != len(preview) {
		t.Fatalf("committed %d rows but previewed %d: %q vs %q",
			len(c.committed), len(preview), c.committed, preview)
	}
	for i := range preview {
		if Strip(c.committed[i]) != Strip(preview[i]) {
			t.Errorf("row %d moved on commit: previewed %q, committed %q",
				i, Strip(preview[i]), Strip(c.committed[i]))
		}
	}
}

func TestFenceDelimitersAreNotCommitted(t *testing.T) {
	c, md := newCapture(false, "none")
	feed(md, "before\n```\ncode\n```\nafter\n", 3)
	md.Flush()

	for _, l := range c.committed {
		if strings.Contains(Strip(l), "```") {
			t.Errorf("fence delimiter leaked into the output: %q", Strip(l))
		}
	}
}

func TestUnmatchedBacktickDoesNotSwallowTheLine(t *testing.T) {
	// Prose about shell commands routinely contains a lone backtick.
	c, md := newCapture(true, "auto")
	feed(md, "use the `rg command to search\n", 4)
	md.Flush()

	if len(c.committed) == 0 {
		t.Fatal("nothing committed")
	}
	got := Strip(c.committed[0])
	if got != "use the `rg command to search" {
		t.Errorf("got %q, want the line intact with its literal backtick", got)
	}
}

func TestUnterminatedFenceIsClosedOnFlush(t *testing.T) {
	c, md := newCapture(false, "none")
	md.Write("```go\nfunc x() {\n")
	md.Flush()

	// A second, independent stream must not still be in code mode.
	md.Write("plain prose\n")
	md.Flush()

	last := Strip(c.committed[len(c.committed)-1])
	if last != "plain prose" {
		t.Errorf("last line = %q; the renderer stayed in code mode across a flush", last)
	}
}

func TestConsecutiveBlankLinesAreCollapsed(t *testing.T) {
	c, md := newCapture(false, "none")
	feed(md, "one\n\n\n\n\ntwo\n", 2)
	md.Flush()

	blanks := 0
	for _, l := range c.committed {
		if strings.TrimSpace(Strip(l)) == "" {
			blanks++
		}
	}
	if blanks > 1 {
		t.Errorf("committed %d blank lines; runs should collapse to one", blanks)
	}
}

func TestEveryStyledLineTerminatesItsEscapes(t *testing.T) {
	// An unterminated colour run bleeds into whatever is pasted after it.
	c, md := newCapture(true, "auto")
	feed(md, "# Heading\n\nSome **bold** text and `code`.\n\n```go\nx := 1\n```\n", 3)
	md.Flush()

	for i, l := range c.committed {
		if !strings.Contains(l, esc) {
			continue
		}
		if HasOpenSGR(l) {
			t.Errorf("line %d ends with an unterminated escape run, which bleeds "+
				"styling into the next line and into anything pasted after it: %q", i, l)
		}
	}
}

func TestChunkBoundariesDoNotAffectOutput(t *testing.T) {
	// The same markdown must render identically no matter how the stream is
	// split, which is the property that makes streaming safe.
	const src = "# Title\n\nSome text with `code`.\n\n```go\nfunc f() int {\n\treturn 42\n}\n```\n\nDone.\n"

	var reference []string
	for _, chunk := range []int{1, 2, 3, 7, 13, 64, 1000} {
		c, md := newCapture(true, "auto")
		feed(md, src, chunk)
		md.Flush()
		if reference == nil {
			reference = c.committed
			continue
		}
		if len(c.committed) != len(reference) {
			t.Fatalf("chunk size %d produced %d lines, want %d", chunk, len(c.committed), len(reference))
		}
		for i := range reference {
			if c.committed[i] != reference[i] {
				t.Errorf("chunk size %d line %d = %q, want %q",
					chunk, i, Strip(c.committed[i]), Strip(reference[i]))
			}
		}
	}
}

func TestPlainModeEmitsNoEscapeSequences(t *testing.T) {
	c, md := newCapture(false, "none")
	feed(md, "# Title\n\n**bold** and `code`\n\n```go\nx := 1\n```\n", 5)
	md.Flush()

	for _, l := range c.committed {
		if strings.Contains(l, esc) {
			t.Errorf("colour disabled but line contains an escape: %q", l)
		}
	}
}

const longProse = "The renderer commits every finished line straight to the scrollback, " +
	"which is what keeps the terminal's own selection and search working on it.\n"

func TestProseWrapsWithinTheTerminalWidth(t *testing.T) {
	const width = 40
	c, md := newCaptureWidth(true, "auto", width)
	feed(md, longProse, 7)
	md.Flush()

	if len(c.committed) < 2 {
		t.Fatalf("expected the paragraph to wrap, got %d line(s)", len(c.committed))
	}
	for i, l := range c.committed {
		if n := len([]rune(Strip(l))); n > width {
			t.Errorf("committed line %d is %d columns wide, want <= %d: %q", i, n, width, Strip(l))
		}
	}
}

func TestWrappingNeverSplitsAWord(t *testing.T) {
	c, md := newCaptureWidth(false, "none", 40)
	feed(md, longProse, 7)
	md.Flush()

	// Every word in the source must survive intact in some committed line.
	for _, word := range strings.Fields(longProse) {
		found := false
		for _, l := range c.committed {
			if strings.Contains(Strip(l), word) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("word %q was split across the wrap", word)
		}
	}
}

func TestWrappingLeavesCodeBlocksByteForByte(t *testing.T) {
	// The hard constraint: a block copied out of the scrollback must be the
	// code that was written, at any terminal width.
	const src = "```go\nfunc handle(w http.ResponseWriter, r *http.Request, logger *slog.Logger) error {\n" +
		"\treturn writeJSON(w, http.StatusOK, map[string]string{\"status\": \"ok\", \"detail\": \"all good\"})\n}\n```\n"
	want := []string{
		"func handle(w http.ResponseWriter, r *http.Request, logger *slog.Logger) error {",
		"\treturn writeJSON(w, http.StatusOK, map[string]string{\"status\": \"ok\", \"detail\": \"all good\"})",
		"}",
	}

	for _, width := range []int{20, 40, 200} {
		c, md := newCaptureWidth(false, "none", width)
		feed(md, src, 9)
		md.Flush()

		var got []string
		for _, l := range c.committed {
			if s := Strip(l); strings.TrimSpace(s) != "" {
				got = append(got, s)
			}
		}
		if len(got) != len(want) {
			t.Fatalf("width %d: code block came out as %d lines, want %d: %q", width, len(got), len(want), got)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("width %d: code line %d = %q, want %q", width, i, got[i], want[i])
			}
		}
	}
}

func TestWrappedListItemsHangUnderTheirMarker(t *testing.T) {
	c, md := newCaptureWidth(false, "none", 30)
	feed(md, "- the first item runs past the edge of the terminal\n"+
		"12. the twelfth item does too and keeps going\n"+
		"> a quoted line that is also much too long to fit\n", 6)
	md.Flush()

	var got []string
	for _, l := range c.committed {
		got = append(got, Strip(l))
	}
	wantIndents := map[string]string{
		"the first item":   "  ",
		"the twelfth item": "    ",
		"a quoted line":    "  ",
	}
	for marker, indent := range wantIndents {
		var conts []string
		seen := false
		for _, l := range got {
			if strings.Contains(l, marker) {
				seen = true
				continue
			}
			if seen && strings.HasPrefix(l, " ") {
				conts = append(conts, l)
			} else if seen {
				break
			}
		}
		if len(conts) == 0 {
			t.Fatalf("%q did not wrap; lines: %q", marker, got)
		}
		for _, l := range conts {
			if !strings.HasPrefix(l, indent) || strings.HasPrefix(l, indent+" ") {
				t.Errorf("continuation of %q = %q, want it indented by %d spaces", marker, l, len(indent))
			}
		}
	}
}

func TestWrappingInsideAStyledSpanClosesAndReopensIt(t *testing.T) {
	c, md := newCaptureWidth(true, "auto", 30)
	feed(md, "Some text and **a bold run that is long enough to be broken in half** after it.\n", 5)
	md.Flush()

	bold := 0
	for i, l := range c.committed {
		if HasOpenSGR(l) {
			t.Errorf("wrapped line %d leaves an escape run open: %q", i, l)
		}
		if strings.Contains(l, codeBold) {
			bold++
		}
	}
	if bold < 2 {
		t.Errorf("the bold run was not reopened after the wrap; %d of %d lines carry it", bold, len(c.committed))
	}
}

func TestCodeIsPreviewedVerbatim(t *testing.T) {
	// Speculation is for prose. A fenced line commits byte for byte, so the
	// preview must not dress it up either.
	c, md := newCaptureWidth(true, "auto", 40)
	feed(md, "```go\n", 3)
	md.Write("\tx := map[string]string{\"a\": \"b\"} // **not bold** and `not code`")

	if len(c.partial) != 1 {
		t.Fatalf("partial = %q, want one unwrapped row", c.partial)
	}
	want := "\tx := map[string]string{\"a\": \"b\"} // **not bold** and `not code`"
	if c.partial[0] != want {
		t.Errorf("partial = %q, want it verbatim: %q", c.partial[0], want)
	}
}

func TestTableRowsAreNotWrapped(t *testing.T) {
	// Breaking a row at a space puts cells on their own lines and the columns
	// stop meaning anything.
	c, md := newCaptureWidth(false, "none", 24)
	feed(md, "| File | Added | Removed |\n|---|---:|---:|\n| internal/render/markdown.go | 66 | 0 |\n", 8)
	md.Flush()

	want := []string{
		"| File | Added | Removed |",
		"|---|---:|---:|",
		"| internal/render/markdown.go | 66 | 0 |",
	}
	if len(c.committed) != len(want) {
		t.Fatalf("table came out as %d lines, want %d: %q", len(c.committed), len(want), c.committed)
	}
	for i := range want {
		if Strip(c.committed[i]) != want[i] {
			t.Errorf("row %d = %q, want %q", i, Strip(c.committed[i]), want[i])
		}
	}
}

func TestPreviewIsBoundedByTerminalHeight(t *testing.T) {
	// A transient zone taller than the screen cannot be erased: the top of it
	// has scrolled beyond the cursor's reach, and the erase would clear
	// committed output instead.
	rows := []string{"one", "two", "three", "four", "five", "six", "seven", "eight"}
	for _, height := range []int{4, 10, 24, 80} {
		got := boundPreview(rows, height)
		if len(got) > max(height/2, 1) {
			t.Errorf("height %d: preview is %d rows, want at most %d", height, len(got), max(height/2, 1))
		}
		if got[len(got)-1] != "eight" {
			t.Errorf("height %d: preview ends at %q, want the tail still visible", height, got[len(got)-1])
		}
	}
}

// ---------------------------------------------------------------------------
// Tables
//
// Models emit cells of wildly different widths, so the columns never line up
// on their own. Alignment needs the whole table before it can measure it,
// which is the one place this renderer buffers instead of committing a line as
// soon as it is complete -- so most of what follows is about making sure the
// buffer always empties again.
// ---------------------------------------------------------------------------

// renderTableAt feeds a source in small chunks, so a table that only looks
// right when it arrives whole is caught.
func renderTableAt(t *testing.T, width int, color bool, src string) []string {
	t.Helper()
	c, md := newCaptureWidth(color, "none", width)
	feed(md, src, 7)
	md.Flush()
	out := make([]string, 0, len(c.committed))
	for _, l := range c.committed {
		out = append(out, Strip(l))
	}
	return out
}

// isBoxed reports whether every line of a rendered table is the same width on
// screen. That is the whole claim alignment makes, and it is the one property
// a change of border style cannot quietly break.
func isBoxed(lines []string) (int, bool) {
	if len(lines) == 0 {
		return 0, false
	}
	w := visibleWidth(lines[0])
	for _, l := range lines {
		if visibleWidth(l) != w {
			return w, false
		}
	}
	return w, true
}

func TestTableColumnsAreAligned(t *testing.T) {
	got := renderTableAt(t, 80, false, "| File | Added | Removed |\n|---|---:|---:|\n"+
		"| internal/render/markdown.go | 66 | 0 |\n| x.go | 1 | 12 |\n")

	want := []string{
		"┌─────────────────────────────┬───────┬─────────┐",
		"│ File                        │ Added │ Removed │",
		"├─────────────────────────────┼───────┼─────────┤",
		"│ internal/render/markdown.go │    66 │       0 │",
		"│ x.go                        │     1 │      12 │",
		"└─────────────────────────────┴───────┴─────────┘",
	}
	if len(got) != len(want) {
		t.Fatalf("got %d lines, want %d:\n got %q\nwant %q", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("line %d =\n  %q\nwant\n  %q", i, got[i], want[i])
		}
	}
}

func TestTableAlignmentMarkersAreHonoured(t *testing.T) {
	// The width comes from the content, not from how many dashes the model
	// happened to type in the delimiter row. The drawn border carries no
	// colons, so the padding is the only thing left saying which way a column
	// is set -- which makes this the test that it is applied at all.
	got := renderTableAt(t, 80, false, "| Left | Middle | Right |\n|:-|:-:|-:|\n| a | b | c |\n")

	want := []string{
		"┌──────┬────────┬───────┐",
		"│ Left │ Middle │ Right │",
		"├──────┼────────┼───────┤",
		"│ a    │   b    │     c │",
		"└──────┴────────┴───────┘",
	}
	for i := range want {
		if i >= len(got) || got[i] != want[i] {
			t.Fatalf("got:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	}
}

func TestTableAtTheEndOfTheStreamIsFlushed(t *testing.T) {
	// Nothing arrives after the last row, so only Flush can release it. A
	// buffered table that is never flushed is output the model produced and
	// the user never sees -- far worse than a misaligned one.
	got := renderTableAt(t, 80, false, "| a | bb |\n|---|----|\n| 1 | 2 |")
	if _, ok := isBoxed(got); !ok || len(got) == 0 {
		t.Fatalf("table at end of stream was not flushed as a block: %q", got)
	}
	if !strings.Contains(strings.Join(got, "\n"), "1") {
		t.Errorf("the last row went missing: %q", got)
	}
}

func TestTableIsFlushedBeforeTheLineThatEndedIt(t *testing.T) {
	// Order matters: the paragraph that follows a table must not overtake it.
	got := renderTableAt(t, 80, false, "| a | bb |\n|---|----|\n| 1 | 2 |\nafter the table\n")
	last := got[len(got)-1]
	if !strings.Contains(last, "after the table") {
		t.Errorf("the line that ended the table is not last: %q", got)
	}
	if _, ok := isBoxed(got[:len(got)-1]); !ok {
		t.Errorf("the table was not aligned before the following line: %q", got)
	}
}

func TestTableIsFlushedWhenAFenceOpens(t *testing.T) {
	got := renderTableAt(t, 80, false, "| a | bb |\n|---|----|\n| 1 | 2 |\n```\nx := 1\n```\n")
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "┌") || !strings.Contains(joined, "└") {
		t.Errorf("the table was not drawn before the fence: %q", got)
	}
	if !strings.Contains(joined, "x := 1") {
		t.Errorf("the fenced code went missing: %q", got)
	}
}

func TestTableMeasuresWideCharactersInColumns(t *testing.T) {
	// A CJK rune occupies two columns. Measuring in runes makes every row a
	// different width on screen while looking correct in a diff.
	got := renderTableAt(t, 80, false, "| 名前 | 説明 |\n|---|---|\n| 日本語テキスト | wide |\n| ascii | ok |\n")
	if w, ok := isBoxed(got); !ok {
		t.Errorf("rows are not the same display width (first is %d): %q", w, got)
	}
}

func TestStyledCellsAreMeasuredOnWhatIsDisplayed(t *testing.T) {
	// With colour on, **bold** becomes four visible characters wrapped in
	// escapes. Measuring the source counts the markers and the columns drift
	// by two for every styled cell.
	got := renderTableAt(t, 80, true, "| Model | Notes |\n|---|---|\n| a | **bold** cell |\n| bb | plain text |\n")
	if w, ok := isBoxed(got); !ok {
		t.Errorf("styled cells were measured on the source (first row %d): %q", w, got)
	}
}

func TestRaggedRowsKeepEveryCell(t *testing.T) {
	// Models routinely emit rows with too few or too many cells. A spec
	// parser drops the extra; it is content, so it gets a column.
	got := renderTableAt(t, 80, false, "| A | B |\n|---|---|\n| ragged |\n| x | y | z |\n")
	joined := strings.Join(got, "\n")
	for _, cell := range []string{"ragged", "x", "y", "z"} {
		if !strings.Contains(joined, cell) {
			t.Errorf("cell %q was dropped: %q", cell, got)
		}
	}
	if w, ok := isBoxed(got); !ok {
		t.Errorf("ragged rows broke the alignment (first row %d): %q", w, got)
	}
}

func TestTableTooWideForTheTerminalShrinksWhileItStillReads(t *testing.T) {
	// Shrinking beats falling back while the cells still say something.
	got := renderTableAt(t, 40, false, "| Model | Context | Notes |\n|---|---|---|\n"+
		"| some-long-model-name | 262144 | a fairly long note |\n")
	w, ok := isBoxed(got)
	if !ok {
		t.Fatalf("narrow table is not aligned: %q", got)
	}
	if w > 40 {
		// An aligned layout wider than the terminal is wrapped by the terminal
		// at an arbitrary column, which scrambles the columns far worse than
		// never having aligned them.
		t.Errorf("laid out to %d columns in a 40-column terminal: %q", w, got)
	}
	if !strings.Contains(strings.Join(got, "\n"), "…") {
		t.Errorf("over-long cells were not marked as truncated: %q", got)
	}
}

func TestTableIsCommittedExactlyOnce(t *testing.T) {
	// flushTable clears the buffer before committing, because commit is a
	// callback into the owner of the screen and a re-entrant call would find
	// the table still sitting there.
	got := renderTableAt(t, 80, false, "| a | bb |\n|---|----|\n| 1 | 2 |\n\ntail\n")
	if n := strings.Count(strings.Join(got, "\n"), "┌"); n != 1 {
		t.Errorf("table drawn %d times, want once: %q", n, got)
	}
}

func TestRowsAreNotCommittedWhileTheTableIsStillArriving(t *testing.T) {
	// The buffer is the point: a row committed early cannot be widened when a
	// later row turns out to be longer.
	c, md := newCaptureWidth(false, "none", 80)
	feed(md, "| a | bb |\n|---|----|\n| 1 | 2 |\n", 7)
	if len(c.committed) != 0 {
		t.Errorf("rows were committed before the table ended: %q", c.committed)
	}
	md.Flush()
	if len(c.committed) == 0 {
		t.Error("Flush did not release the buffered table")
	}
}
