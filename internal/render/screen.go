// Package render turns the agent's event stream into terminal output.
//
// The central constraint: everything that reaches the scrollback buffer is
// written exactly once and never revised. ai-code does not use the alternate
// screen, never enables mouse reporting, and never redraws text that has
// already scrolled. That is what keeps the terminal's own search, selection and
// copy working, and it is why the renderer is built around committing whole
// lines rather than around a screen it owns.
//
// Two zones exist:
//
//	committed  - real scrollback. Written once, styled, never touched again.
//	transient  - the last few lines: the line currently streaming, and the
//	             status line. Redrawn freely, and always erased before anything
//	             is committed, so it never ends up in the scrollback.
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
	// output. Every transient line is truncated to the terminal width so that
	// this count cannot be thrown off by wrapping.
	transient int
	// atLineStart tracks whether the cursor sits in column zero of a fresh
	// committed line.
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
// by it: a zone taller than the screen cannot be erased, because the cursor
// cannot walk back up past rows that have already scrolled away.
func (s *Screen) Height() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.height
}

// Resize updates the known width. Only the transient zone is affected;
// committed text is left for the terminal to reflow, which it does natively
// and which is the entire advantage of not owning the screen.
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

// Commit writes one finished line to the scrollback.
//
// The line is emitted with trailing whitespace stripped and an explicit SGR
// reset before the newline. Both matter for selection: a coloured trailing
// space is invisible on screen but lands in the clipboard, and an unterminated
// colour run bleeds into whatever the user pastes.
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
	// newline. Already-closed lines get nothing appended, which keeps the
	// committed bytes minimal and the golden tests meaningful.
	if s.color && HasOpenSGR(line) {
		s.w.WriteString(sgrReset)
	}
	s.w.WriteByte('\n')
	s.atLineStart = true
}

// SetTransient replaces the transient zone. Lines are flattened to a single
// physical row each and truncated to the terminal width, so the wrap count
// stays predictable.
func (s *Screen) SetTransient(lines ...string) {
	s.setTransient(-1, lines)
}

// SetTransientCursor is SetTransient with the cursor parked at a given column
// of the last line rather than after its final character. The steering prompt
// needs it: without it the cursor sits at the end of the line whatever the user
// has done with the arrow keys, and editing anywhere but the end is blind.
func (s *Screen) SetTransientCursor(col int, lines ...string) {
	s.setTransient(col, lines)
}

func (s *Screen) setTransient(cursorCol int, lines []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.isTTY {
		// Without a terminal there is nowhere to redraw. Transient content is
		// simply dropped, which is what a pipe or a CI log should receive.
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

// truncateVisible shortens a string to n columns, ignoring ANSI escape
// sequences when counting and closing any open sequence at the cut.
//
// Columns, not runes. A transient line is truncated so that it occupies one
// physical row, and that guarantee is what lets eraseTransient know how far to
// move the cursor up. Counting runes breaks it for exactly the text the
// thinking line carries most often -- CJK, emoji, box drawing -- because those
// runes are two columns wide, so a "79 rune" line lands in column 100 and
// wraps.
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

// flattenTransient makes a string safe to place in the transient zone.
//
// Every transient line must occupy exactly one physical row, because the erase
// that precedes the next commit walks up a counted number of rows. A newline
// splits the row in two and the erase then misses the top half, stranding it in
// the scrollback -- which is how a thinking trace containing code ends up
// printed permanently, a fragment at a time, between the lines it was meant to
// be scrolling behind. A tab is worse than a newline here rather than better:
// it is one rune that renders as up to eight columns, so it defeats the width
// arithmetic without being visible in the string at all.
//
// SGR sequences are kept: they are the styling, and they cost no columns.
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

// runeWidth returns the number of columns a rune occupies.
//
// This is wcwidth, cut down to what a terminal renderer actually needs:
// combining marks take no space of their own, the East Asian wide and
// fullwidth blocks take two, and everything else takes one. It does not try to
// resolve emoji ZWJ sequences -- no terminal agrees on those anyway -- and it
// is deliberately conservative, because over-counting truncates a line early
// while under-counting wraps it, and only one of those corrupts the display.
func runeWidth(r rune) int {
	switch {
	case r == 0:
		return 0
	case r < 0x20 || r == 0x7f:
		// Controls never reach here from the transient path (flattenTransient
		// removes them) but committed text is measured too.
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
