// Package render turns the agent's event stream into terminal output.
//
// Committed output is written once and never revised, so the terminal's own
// search, selection and copy keep working. The transient zone -- the streaming
// line and the status line -- is redrawn freely and always erased before the
// next commit.
package render

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/term"
)

const (
	esc        = "\x1b"
	sgrReset   = esc + "[0m"
	clearLine  = "\r" + esc + "[2K"
	cursorUp   = esc + "[A"
	hideCursor = esc + "[?25l"
	showCursor = esc + "[?25h"
)

// Screen owns the boundary between committed output and the transient zone.
type Screen struct {
	mu sync.Mutex

	w      *bufio.Writer
	out    io.Writer
	isTTY  bool
	color  bool
	width  int
	height int
	fd     int

	// transient counts the physical lines currently drawn below the committed
	// output. Every transient line is truncated to one row so the count stays
	// exact.
	transient int
	// atLineStart tracks whether the cursor sits in column zero of a fresh line.
	atLineStart bool
}

func NewScreen(out io.Writer, colorMode string) *Screen {
	s := &Screen{
		out:         out,
		w:           bufio.NewWriterSize(out, 16*1024),
		width:       80,
		height:      24,
		atLineStart: true,
		fd:          -1,
	}
	if f, ok := out.(*os.File); ok {
		s.fd = int(f.Fd())
		s.isTTY = term.IsTerminal(s.fd)
		if w, h, err := term.GetSize(s.fd); err == nil && w > 0 {
			s.width = w
			if h > 0 {
				s.height = h
			}
		}
	}
	switch colorMode {
	case "always":
		s.color = true
	case "never":
		s.color = false
	default:
		s.color = s.isTTY && os.Getenv("NO_COLOR") == ""
	}
	return s
}

func (s *Screen) IsTTY() bool { return s.isTTY }
func (s *Screen) Color() bool { return s.color }

func (s *Screen) Width() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.width
}

// Height is the number of rows the terminal has. The transient zone is bounded
// by it: a taller zone cannot be erased, because the cursor cannot walk back up
// past rows that have scrolled away.
func (s *Screen) Height() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.height
}

// Resize updates the known width. Committed text is left for the terminal to
// reflow, which is the advantage of not owning the screen.
func (s *Screen) Resize() {
	if s.fd < 0 {
		return
	}
	if w, h, err := term.GetSize(s.fd); err == nil && w > 0 {
		s.mu.Lock()
		s.width = w
		if h > 0 {
			s.height = h
		}
		s.mu.Unlock()
	}
}

// Commit writes one finished line to the scrollback, with trailing whitespace
// stripped and an SGR reset before the newline: a coloured trailing space lands
// in the clipboard, and an open colour run bleeds into a paste.
func (s *Screen) Commit(line string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.eraseTransientLocked()
	s.commitLocked(line)
	s.w.Flush()
}

// CommitBlock writes several lines at once.
func (s *Screen) CommitBlock(lines ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.eraseTransientLocked()
	for _, l := range lines {
		s.commitLocked(l)
	}
	s.w.Flush()
}

func (s *Screen) commitLocked(line string) {
	line = strings.TrimRight(line, " \t")
	s.w.WriteString(line)
	// Close any run the line left open, so styling never bleeds past the
	// newline; closed lines get nothing appended, which keeps the committed
	// bytes minimal.
	if s.color && HasOpenSGR(line) {
		s.w.WriteString(sgrReset)
	}
	s.w.WriteByte('\n')
	s.atLineStart = true
}

// SetTransient replaces the transient zone. Lines are flattened to a single
// physical row each and truncated, so the wrap count stays predictable.
func (s *Screen) SetTransient(lines ...string) {
	s.setTransient(-1, lines)
}

// SetTransientCursor is SetTransient with the cursor parked at a given column
// of the last line, which the steering prompt needs: otherwise the cursor sits
// at the end whatever the arrow keys have done, and editing is blind.
func (s *Screen) SetTransientCursor(col int, lines ...string) {
	s.setTransient(col, lines)
}

func (s *Screen) setTransient(cursorCol int, lines []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.isTTY {
		// Without a terminal there is nowhere to redraw; transient content is
		// dropped, which is what a pipe or a CI log should receive.
		return
	}
	s.eraseTransientLocked()

	endCol := 0
	for i, l := range lines {
		l = truncateVisible(flattenTransient(l), s.width-1)
		s.w.WriteString(l)
		if s.color && HasOpenSGR(l) {
			s.w.WriteString(sgrReset)
		}
		if i < len(lines)-1 {
			s.w.WriteByte('\n')
		} else {
			endCol = visibleWidth(l)
		}
	}
	if cursorCol >= 0 && cursorCol < endCol {
		fmt.Fprintf(s.w, "%s[%dD", esc, endCol-cursorCol)
	}
	s.transient = len(lines)
	s.w.Flush()
}

func (s *Screen) ClearTransient() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.eraseTransientLocked()
	s.w.Flush()
}

func (s *Screen) eraseTransientLocked() {
	if s.transient == 0 || !s.isTTY {
		return
	}
	s.w.WriteString(clearLine)
	for range s.transient - 1 {
		s.w.WriteString(cursorUp)
		s.w.WriteString(clearLine)
	}
	s.transient = 0
	s.atLineStart = true
}

func (s *Screen) HideCursor() {
	if !s.isTTY {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.w.WriteString(hideCursor)
	s.w.Flush()
}

func (s *Screen) ShowCursor() {
	if !s.isTTY {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.w.WriteString(showCursor)
	s.w.Flush()
}

// Control writes a terminal mode sequence through the same writer as
// everything else, so it cannot land in the middle of a line the renderer is
// part-way through emitting.
func (s *Screen) Control(seq string) {
	if !s.isTTY {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.w.WriteString(seq)
	s.w.Flush()
}

func (s *Screen) Flush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.w.Flush()
}

// truncateVisible shortens a string to n columns, ignoring ANSI escapes and
// closing any open sequence at the cut. Columns, not runes: wide runes are two
// columns, and a line truncated by rune count wraps, which breaks the erase.
func truncateVisible(s string, n int) string {
	if n <= 0 {
		return ""
	}
	var (
		b       strings.Builder
		visible int
		inEsc   bool
		sawEsc  bool
	)
	for _, r := range s {
		if inEsc {
			b.WriteRune(r)
			if isEscFinal(r) {
				inEsc = false
			}
			continue
		}
		if r == 0x1b {
			inEsc, sawEsc = true, true
			b.WriteRune(r)
			continue
		}
		w := runeWidth(r)
		if visible+w > n {
			if sawEsc {
				b.WriteString(sgrReset)
			}
			return b.String()
		}
		b.WriteRune(r)
		visible += w
	}
	return b.String()
}

// visibleWidth counts columns, ignoring escape sequences.
func visibleWidth(s string) int {
	n, inEsc := 0, false
	for _, r := range s {
		if inEsc {
			if isEscFinal(r) {
				inEsc = false
			}
			continue
		}
		if r == 0x1b {
			inEsc = true
			continue
		}
		n += runeWidth(r)
	}
	return n
}

func isEscFinal(r rune) bool {
	return r == 'm' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z')
}

// flattenTransient folds a string onto one physical row, because the erase walks
// up a counted number of rows; newlines and tabs become spaces. SGR sequences
// cost no columns and are kept.
func flattenTransient(s string) string {
	if strings.IndexFunc(s, isTransientUnsafe) < 0 {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	inEsc := false
	for _, r := range s {
		switch {
		case inEsc:
			b.WriteRune(r)
			if isEscFinal(r) {
				inEsc = false
			}
		case r == 0x1b:
			inEsc = true
			b.WriteRune(r)
		case r == '\n' || r == '\r' || r == '\t':
			b.WriteByte(' ')
		case r < 0x20 || r == 0x7f:
			// Any other C0 control moves the cursor or changes terminal state.
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func isTransientUnsafe(r rune) bool {
	return r != 0x1b && (r < 0x20 || r == 0x7f)
}

// runeWidth returns the number of columns a rune occupies: combining marks take
// none, East Asian wide and fullwidth blocks take two, everything else one. It
// does not resolve emoji ZWJ sequences, where terminals disagree anyway.
func runeWidth(r rune) int {
	switch {
	case r == 0:
		return 0
	case r < 0x20 || r == 0x7f:
		// Controls never reach here from the transient path, but committed
		// text is measured too.
		return 0
	case r < 0x7f:
		return 1
	case unicode.In(r, unicode.Mn, unicode.Me, unicode.Cf):
		return 0
	case isWide(r):
		return 2
	}
	return 1
}

// wideRanges are the East Asian Wide and Fullwidth blocks, plus the emoji
// blocks that terminals render double-width.
var wideRanges = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x1100, 0x115f, 1}, // Hangul Jamo initial consonants
		{0x2e80, 0x303e, 1}, // CJK radicals, Kangxi, CJK symbols
		{0x3041, 0x33ff, 1}, // Hiragana .. CJK compatibility
		{0x3400, 0x4dbf, 1}, // CJK extension A
		{0x4e00, 0x9fff, 1}, // CJK unified ideographs
		{0xa000, 0xa4cf, 1}, // Yi
		{0xac00, 0xd7a3, 1}, // Hangul syllables
		{0xf900, 0xfaff, 1}, // CJK compatibility ideographs
		{0xfe10, 0xfe19, 1}, // vertical forms
		{0xfe30, 0xfe6f, 1}, // CJK compatibility forms
		{0xff00, 0xff60, 1}, // fullwidth forms
		{0xffe0, 0xffe6, 1}, // fullwidth signs
	},
	R32: []unicode.Range32{
		{0x1f300, 0x1f64f, 1}, // symbols, pictographs, emoticons
		{0x1f900, 0x1f9ff, 1}, // supplemental symbols and pictographs
		{0x20000, 0x3fffd, 1}, // CJK extensions B..
	},
}

func isWide(r rune) bool { return unicode.Is(wideRanges, r) }
