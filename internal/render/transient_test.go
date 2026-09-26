package render

import (
	"bytes"
	"strings"
	"testing"

	"ai-code/internal/agent"
)

// countRows reports how many physical rows a terminal of the given width would
// use for a run of output, which is the number eraseTransient has to agree with.
func countRows(s string, width int) int {
	rows, col := 1, 0
	inEsc := false
	for _, r := range Strip(s) {
		_ = inEsc
		switch {
		case r == '\n':
			rows++
			col = 0
		case r == '\r':
			col = 0
		case r == '\t':
			col += 8 - col%8
		default:
			col += runeWidth(r)
		}
		if col > width {
			rows++
			col = runeWidth(r)
		}
	}
	return rows
}

// Truncating by rune count does not bound columns: tabs and wide runes overflow
// the margin and wrap, so eraseTransient must clear exactly what was drawn.
func TestTransientLineNeverOccupiesMoreThanOneRow(t *testing.T) {
	cases := []struct {
		name string
		line string
	}{
		{"tabs", "thinking: \tif err != nil {\n\t\treturn err\n\t}"},
		{"wide runes", "thinking: " + strings.Repeat("日本語テキスト", 30)},
		{"emoji", "thinking: " + strings.Repeat("🙂", 120)},
		{"carriage return", "thinking: first\rsecond\rthird"},
		{"plain overflow", "thinking: " + strings.Repeat("x", 300)},
		{"styled overflow", "\x1b[2mthinking: " + strings.Repeat("y", 300) + "\x1b[0m"},
	}

	for _, width := range []int{40, 80, 120} {
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				var buf bytes.Buffer
				s := NewScreen(&buf, "never")
				s.isTTY, s.width = true, width

				s.SetTransient(tc.line)
				got := buf.String()

				if strings.ContainsAny(got, "\n\r") {
					t.Errorf("width %d: transient output contains a line break: %q", width, got)
				}
				if rows := countRows(got, width); rows != 1 {
					t.Errorf("width %d: transient line occupies %d rows, want 1: %q", width, rows, got)
				}
				if s.transient != 1 {
					t.Errorf("width %d: screen counted %d transient rows, want 1", width, s.transient)
				}
			})
		}
	}
}

// The erase must clear exactly what was drawn, or the leftover stays in the
// scrollback.
func TestTransientIsFullyErasedBeforeACommit(t *testing.T) {
	var buf bytes.Buffer
	s := NewScreen(&buf, "never")
	s.isTTY, s.width = true, 40

	s.SetTransient("\tthinking about "+strings.Repeat("日", 40), "⠋  12s")
	if s.transient != 2 {
		t.Fatalf("transient = %d, want 2", s.transient)
	}
	buf.Reset()

	s.Commit("func main() {")
	got := buf.String()

	// One clear for the bottom row, then up-and-clear for the row above it.
	if n := strings.Count(got, clearLine); n != 2 {
		t.Errorf("%d line erases before the commit, want 2: %q", n, got)
	}
	if n := strings.Count(got, cursorUp); n != 1 {
		t.Errorf("%d cursor-up before the commit, want 1: %q", n, got)
	}
	if !strings.HasSuffix(got, "func main() {\n") {
		t.Errorf("committed output = %q, want it to end with the committed line alone", got)
	}
	if strings.Contains(got, "thinking about") {
		t.Errorf("the thinking line was re-emitted into the scrollback: %q", got)
	}
}

func TestRuneWidth(t *testing.T) {
	cases := []struct {
		r    rune
		want int
	}{
		{'a', 1},
		{'…', 1},
		{'─', 1},
		{'⠋', 1}, // the spinner: one column in every terminal, as documented
		{'日', 2},
		{'🙂', 2},
		{'́', 0}, // combining acute
		{'\n', 0},
	}
	for _, c := range cases {
		if got := runeWidth(c.r); got != c.want {
			t.Errorf("runeWidth(%q) = %d, want %d", c.r, got, c.want)
		}
	}
}

// The marquee must show the newest text, so it keeps a tail: the screen would
// truncate a fixed window from the front.
func TestThinkingMarqueeShowsTheTail(t *testing.T) {
	text := strings.Repeat("old ", 100) + "the newest thought"

	got := marquee(text, 40)
	if !strings.HasSuffix(got, "the newest thought") {
		t.Errorf("marquee = %q, want it to end with the most recent text", got)
	}
	if w := visibleWidth(got); w > 40 {
		t.Errorf("marquee width = %d, want at most 40", w)
	}
}

func TestMarqueeFlattensWhitespace(t *testing.T) {
	got := marquee("first line\n\tif x {\r\n\t\treturn\n}", 200)
	if strings.ContainsAny(got, "\n\r\t") {
		t.Errorf("marquee = %q, want no control characters", got)
	}
	if want := "first line if x { return }"; got != want {
		t.Errorf("marquee = %q, want %q", got, want)
	}
}

func TestMarqueeCountsWideRunesAsTwoColumns(t *testing.T) {
	got := marquee(strings.Repeat("日", 50), 10)
	if n := len([]rune(got)); n != 5 {
		t.Errorf("marquee kept %d wide runes for 10 columns, want 5", n)
	}
}

// Trimming reasoning by byte offset can split a multi-byte rune, so the tail
// is trimmed on rune boundaries.
func TestReasoningTrimIsRuneSafe(t *testing.T) {
	r := &Interactive{reasonMode: "collapsed", style: Style{}}
	for range 200 {
		r.Emit(agent.Event{Kind: agent.EvReasoning, Text: "日本語のテキストです。"})
	}
	if !utf8Valid(r.reasoning) {
		t.Errorf("retained reasoning is not valid UTF-8: %q", r.reasoning)
	}
	if n := len([]rune(r.reasoning)); n != reasoningKeep {
		t.Errorf("retained %d runes, want %d", n, reasoningKeep)
	}
}

func utf8Valid(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}
