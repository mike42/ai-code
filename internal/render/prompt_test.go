package render

import (
	"strings"
	"testing"
)

// A single-line prompt is already on screen after the marker, so echoing it as
// a block would show it twice.
func TestSingleLineEchoIsOneLine(t *testing.T) {
	got := Echo(Style{}, "prompt", "fix the failing test")
	body := nonBlank(got)
	if len(body) != 1 {
		t.Fatalf("echo = %q, want a single line", got)
	}
	if body[0] != "prompt: fix the failing test" {
		t.Errorf("echo = %q", body[0])
	}
}

// The line editor never draws multi-line input -- it summarises it and points
// at Ctrl-G -- so at the moment it is submitted the terminal has never shown
// what was sent. It goes into the scrollback here instead.
func TestMultilineEchoShowsEveryLineBetweenRules(t *testing.T) {
	text := "Refactor this:\n\n    func f() {}\n\nand keep the tests passing."
	got := Echo(Style{}, "prompt", text)

	joined := strings.Join(got, "\n")
	for _, want := range []string{"func f() {}", "and keep the tests passing."} {
		if !strings.Contains(joined, want) {
			t.Errorf("echo is missing %q:\n%s", want, joined)
		}
	}
	if !strings.Contains(joined, "prompt · 5 lines") {
		t.Errorf("echo has no labelled rule:\n%s", joined)
	}
	if n := strings.Count(joined, "──"); n < 2 {
		t.Errorf("echo has %d rules, want one above and one below:\n%s", n, joined)
	}
}

// Indentation, not a gutter character: a vertical bar down the left edge is
// picked up by selection.
func TestMultilineEchoDoesNotPutAGutterInTheTextColumns(t *testing.T) {
	got := Echo(Style{}, "prompt", "line one\nline two")
	for _, l := range got {
		body := strings.TrimSpace(Strip(l))
		if body == "" || strings.HasPrefix(body, "──") {
			continue
		}
		if strings.ContainsAny(body[:1], "│|┃╎") {
			t.Errorf("echoed line %q starts with a gutter character", l)
		}
	}
	if got[2] != "  line one" {
		t.Errorf("body line = %q, want it indented by two spaces", got[2])
	}
}

func TestLongEchoIsCapped(t *testing.T) {
	text := strings.TrimRight(strings.Repeat("x\n", 500), "\n")
	got := Echo(Style{}, "prompt", text)

	if len(got) > echoMaxLines+6 {
		t.Errorf("echo is %d lines for a 500-line prompt, want it capped", len(got))
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "prompt · 500 lines") {
		t.Errorf("echo does not report the real length:\n%s", joined)
	}
	if !strings.Contains(joined, "460 more lines not shown") {
		t.Errorf("echo does not say what it left out:\n%s", joined)
	}
}

// A long line is bounded from the middle: the opening words say what was asked
// and the closing ones carry the path or the constraint added last.
func TestBoundLineCutsTheMiddleAndKeepsBothEnds(t *testing.T) {
	text := "START " + strings.Repeat("filler ", 5000) + "END"
	got := BoundLine(text)

	if !strings.HasPrefix(got, "START ") {
		t.Errorf("bounded line = %q..., want the opening words kept", got[:40])
	}
	if !strings.HasSuffix(got, "END") {
		t.Errorf("bounded line ends %q, want the closing words kept", got[len(got)-40:])
	}
	if !strings.Contains(got, "omitted") {
		t.Error("bounded line does not say that anything was left out")
	}
	if w := visibleWidth(got); w > echoMaxColumns+64 {
		t.Errorf("bounded line is %d columns, want it capped near %d", w, echoMaxColumns)
	}
}

func TestBoundLineLeavesAnOrdinaryPromptAlone(t *testing.T) {
	text := "fix the failing test in internal/render/prompt_test.go"
	if got := BoundLine(text); got != text {
		t.Errorf("BoundLine changed a short prompt to %q", got)
	}
}

// The cap is in columns rather than runes. A prompt in CJK is half as many
// runes for the same number of rows on screen, and rows are what flood.
func TestBoundLineCountsColumnsNotRunes(t *testing.T) {
	got := BoundLine(strings.Repeat("\u6f22", echoMaxColumns))
	if w := visibleWidth(got); w > echoMaxColumns+64 {
		t.Errorf("bounded line is %d columns, want it capped near %d", w, echoMaxColumns)
	}
}

// A single line still echoes as a single line once bounded -- the block
// framing is for multi-line input, where the rules mark where it starts and
// stops.
func TestLongSingleLineEchoStaysOneLine(t *testing.T) {
	got := Echo(Style{}, "prompt", strings.Repeat("x", 100_000))
	body := nonBlank(got)
	if len(body) != 1 {
		t.Fatalf("echo = %d lines, want one", len(body))
	}
	if !strings.Contains(body[0], "omitted") {
		t.Error("a 100k-character prompt was echoed in full")
	}
	if w := visibleWidth(body[0]); w > echoMaxColumns+64 {
		t.Errorf("echoed line is %d columns, want it capped", w)
	}
}

func TestIsMultiline(t *testing.T) {
	cases := map[string]bool{
		"one line":       false,
		"trailing\n":     false,
		"  padded  ":     false,
		"two\nlines":     true,
		"\n\nembedded\n": false,
	}
	for in, want := range cases {
		if got := IsMultiline(in); got != want {
			t.Errorf("IsMultiline(%q) = %v, want %v", in, got, want)
		}
	}
}

func nonBlank(lines []string) []string {
	var out []string
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}
