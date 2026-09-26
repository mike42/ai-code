// Package ui is the interactive input layer: a line editor, the escape to a
// real text editor, and the slash commands.
//
// Mouse reporting stays off, so text selection keeps working; bracketed paste is
// on. The alternate screen is never used, so scrollback survives and the
// terminal's own search covers the whole session.
package ui

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/term"

	"ai-code/internal/render"
)

var (
	// ErrInterrupt is returned when Ctrl-C is pressed at the prompt.
	ErrInterrupt = errors.New("interrupted")
	// ErrEOF is returned on Ctrl-D at an empty prompt.
	ErrEOF = errors.New("end of input")
)

const (
	bracketedPasteOn  = "\x1b[?2004h"
	bracketedPasteOff = "\x1b[?2004l"
)

type Editor struct {
	in  *os.File
	out *os.File

	history    []string
	historyPos int
	// Completions supplies Tab candidates for the line up to the cursor: the
	// byte offset where they begin, and the candidates, which are words rather
	// than whole lines. The editor replaces from there to the cursor.
	Completions func(line string) (start int, candidates []string)
	// EditorCommand overrides $VISUAL and $EDITOR.
	EditorCommand string
	// OnActivity is called once per ReadLine, on the first key pressed; the
	// mark of a person returning to the keyboard, which background work against
	// an abandoned prompt is cancelled from.
	OnActivity func()

	buf    []rune
	cursor int
	saved  string

	// preload is text the next ReadLine starts with. Steering hands over here:
	// a half-typed sentence belongs in the prompt that replaces the steering
	// line, not in the bin.
	preload string

	// kill holds the most recently killed text so Ctrl-Y can put it back, the
	// way bash does. One slot rather than a ring.
	kill string

	// State for the incremental redraw. lastVisible, lastCursor, lastWidth and
	// lastPrompt describe what is on screen, so redraw can compute the smallest
	// edit from it.
	painted    bool
	lastPrompt string
	// live is a prompt supplied from outside the read loop, by a worker
	// finishing or another window swapping the model; without it the next
	// keystroke repaints ReadLine's stale prompt and commits that.
	live        string
	lastVisible []rune
	lastCursor  int
	lastWidth   int

	// header is the replaceable line above the prompt; see SetHeader.
	header      string
	headerDrawn bool

	// paintMu serialises painting: the read loop paints after each key, and
	// Refresh paints from whichever goroutine noticed the prompt is wrong;
	// interleaved escape sequences corrupt the line.
	paintMu sync.Mutex

	// piped is created once and reused: a fresh bufio.Reader per call would
	// read ahead into its buffer and then discard it, silently swallowing every
	// line after the first.
	piped *bufio.Reader
}

func NewEditor(in, out *os.File) *Editor {
	return &Editor{in: in, out: out, historyPos: -1}
}

func (e *Editor) AddHistory(line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	if n := len(e.history); n > 0 && e.history[n-1] == line {
		return
	}
	e.history = append(e.history, line)
	if len(e.history) > 500 {
		e.history = e.history[len(e.history)-500:]
	}
}

// Preload seeds the next ReadLine with text, cursor at the end.
func (e *Editor) Preload(text string) { e.preload = text }

func (e *Editor) isTTY() bool { return term.IsTerminal(int(e.in.Fd())) }

// IsTTY reports whether ReadLine will edit a line or merely read one, which
// callers arranging something around the prompt need to know.
func (e *Editor) IsTTY() bool { return e.isTTY() }

// ReadLine reads one submission; without a terminal it falls back to plain line
// reading, which is how piped input drives the program.
func (e *Editor) ReadLine(prompt string) (string, error) {
	if !e.isTTY() {
		return e.readPiped()
	}

	old, err := term.MakeRaw(int(e.in.Fd()))
	if err != nil {
		return e.readPiped()
	}
	defer term.Restore(int(e.in.Fd()), old)

	fmt.Fprint(e.out, bracketedPasteOn)
	defer fmt.Fprint(e.out, bracketedPasteOff)

	e.buf = append(e.buf[:0], []rune(e.preload)...)
	e.cursor = len(e.buf)
	e.preload = ""
	e.historyPos = -1
	e.invalidate()

	// This call's prompt supersedes anything left over from the last one.
	e.paintMu.Lock()
	e.live = ""
	e.paintMu.Unlock()

	e.redraw(prompt)

	reader := bufio.NewReaderSize(e.in, 4096)
	active := false
	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			return "", ErrEOF
		}
		if !active {
			active = true
			if e.OnActivity != nil {
				// Before the key is acted on, so the abort starts while the
				// character is drawn.
				e.OnActivity()
			}
		}

		switch r {
		case '\r', '\n':
			e.finalise(prompt, string(e.buf))
			return string(e.buf), nil

		case 3: // Ctrl-C
			fmt.Fprint(e.out, "\r\n")
			return "", ErrInterrupt

		case 4: // Ctrl-D
			if len(e.buf) == 0 {
				fmt.Fprint(e.out, "\r\n")
				return "", ErrEOF
			}
			e.deleteForward()

		case 1: // Ctrl-A
			e.cursor = 0
		case 5: // Ctrl-E
			e.cursor = len(e.buf)
		case 2: // Ctrl-B
			if e.cursor > 0 {
				e.cursor--
			}
		case 6: // Ctrl-F
			if e.cursor < len(e.buf) {
				e.cursor++
			}
		case 11: // Ctrl-K
			e.kill = string(e.buf[e.cursor:])
			e.buf = e.buf[:e.cursor]
		case 21: // Ctrl-U
			e.kill = string(e.buf[:e.cursor])
			e.buf = append([]rune{}, e.buf[e.cursor:]...)
			e.cursor = 0
		case 23: // Ctrl-W
			e.deleteWord()
		case 25: // Ctrl-Y
			e.insertRunes([]rune(e.kill))
		case 20: // Ctrl-T
			e.transpose()
		case 16: // Ctrl-P
			e.historyPrev()
		case 14: // Ctrl-N
			e.historyNext()
		case 18: // Ctrl-R
			if e.reverseSearch(reader, prompt) {
				// The search prompt is on the line, so the finalising repaint
				// is what puts the recalled command into the scrollback
				// looking typed.
				e.finalise(prompt, string(e.buf))
				return string(e.buf), nil
			}
		case 12: // Ctrl-L
			fmt.Fprint(e.out, "\x1b[2J\x1b[H")
			e.invalidate()

		case 7: // Ctrl-G -- hand off to a real editor
			text, err := e.launchEditor(string(e.buf))
			term.Restore(int(e.in.Fd()), old)
			if err == nil && strings.TrimSpace(text) != "" {
				e.finalise(prompt, text)
				return text, nil
			}
			_, _ = term.MakeRaw(int(e.in.Fd()))
			e.invalidate()
			if err != nil {
				fmt.Fprintf(e.out, "\r\n%s\r\n", err)
			}

		case '\t':
			e.complete()

		case 127, 8: // Backspace
			e.deleteBackward()

		case 27: // Escape sequence
			e.handleEscape(reader)

		default:
			if r >= 32 {
				e.insert(r)
			}
		}
		e.redraw(prompt)
	}
}

func (e *Editor) readPiped() (string, error) {
	if e.piped == nil {
		e.piped = bufio.NewReader(e.in)
	}
	line, err := e.piped.ReadString('\n')
	if err != nil {
		if line == "" {
			return "", ErrEOF
		}
		return strings.TrimRight(line, "\r\n"), nil
	}
	return strings.TrimRight(line, "\r\n"), nil
}

// handleEscape consumes one escape sequence and applies it. The whole sequence
// must be consumed even when it is ignored: reading a fixed two bytes leaves the
// tail in the reader, where it is read back as keystrokes.
func (e *Editor) handleEscape(r *bufio.Reader) {
	b1, _, err := r.ReadRune()
	if err != nil {
		return
	}
	switch b1 {
	case '[':
		e.handleCSI(r)
	case 'O':
		// SS3, used for cursor keys in application mode: one final byte.
		if f, _, err := r.ReadRune(); err == nil {
			e.cursorKey(f, 0)
		}
	// ESC-prefixed keys are how a terminal reports Alt.
	case 'b':
		e.cursor = e.wordStart(e.cursor)
	case 'f':
		e.cursor = e.wordEnd(e.cursor)
	case 'd':
		e.killForwardWord()
	case 127, 8:
		e.killBackwardWord()
	}
}

// handleCSI reads a CSI sequence to its final byte and dispatches on it.
func (e *Editor) handleCSI(r *bufio.Reader) {
	var params []rune
	var final rune
	for {
		ch, _, err := r.ReadRune()
		if err != nil {
			return
		}
		if ch >= 0x40 && ch <= 0x7e { // final byte
			final = ch
			break
		}
		params = append(params, ch)
		if len(params) > 32 {
			return // malformed; stop rather than read forever
		}
	}

	p := string(params)
	// A modifier arrives as a second parameter: "1;5D" is Ctrl-Left, "1;3D"
	// Alt-Left; both are word motions here.
	mod := 0
	if i := strings.IndexByte(p, ';'); i >= 0 {
		mod, _ = strconv.Atoi(p[i+1:])
		p = p[:i]
	}

	switch final {
	case 'A', 'B', 'C', 'D', 'H', 'F':
		e.cursorKey(final, mod)
	case '~':
		switch p {
		case "1", "7": // Home
			e.cursor = 0
		case "4", "8": // End
			e.cursor = len(e.buf)
		case "3": // Delete
			if isWordMod(mod) {
				e.killForwardWord()
			} else {
				e.deleteForward()
			}
		case "200": // bracketed paste begins
			e.readPaste(r)
		}
	}
}

func (e *Editor) cursorKey(final rune, mod int) {
	switch final {
	case 'A':
		e.historyPrev()
	case 'B':
		e.historyNext()
	case 'C':
		if isWordMod(mod) {
			e.cursor = e.wordEnd(e.cursor)
		} else if e.cursor < len(e.buf) {
			e.cursor++
		}
	case 'D':
		if isWordMod(mod) {
			e.cursor = e.wordStart(e.cursor)
		} else if e.cursor > 0 {
			e.cursor--
		}
	case 'H':
		e.cursor = 0
	case 'F':
		e.cursor = len(e.buf)
	}
}

// isWordMod reports whether a CSI modifier means Ctrl or Alt, the two that turn
// an arrow key into a word motion. The encoding is 1 + shift(1)/alt(2)/ctrl(4).
func isWordMod(mod int) bool {
	m := mod - 1
	return m > 0 && m&(2|4) != 0
}

// readPaste consumes a bracketed-paste payload verbatim, so a pasted multi-line
// snippet does not submit on its first newline.
func (e *Editor) readPaste(r *bufio.Reader) {
	const end = "\x1b[201~"
	var b strings.Builder
	for {
		ch, _, err := r.ReadRune()
		if err != nil {
			break
		}
		b.WriteRune(ch)
		if strings.HasSuffix(b.String(), end) {
			break
		}
	}
	text := strings.TrimSuffix(b.String(), end)
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	e.insertRunes(sanitisePaste(text))
}

// sanitisePaste drops control characters from pasted text: an escape sequence
// would be sent straight back to the terminal by redraw, corrupting the line,
// and then to the model as noise. Tabs become spaces; newlines survive.
func sanitisePaste(s string) []rune {
	rs := []rune(s)
	out := make([]rune, 0, len(rs))
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == 0x1b:
			// Drop the whole sequence, not just the escape byte -- otherwise
			// "\x1b[31m" leaves a literal "[31m" behind in the buffer.
			i = skipEscape(rs, i)
		case r == '\n':
			out = append(out, r)
		case r == '\t':
			out = append(out, ' ')
		case r < 32 || r == 127:
		default:
			out = append(out, r)
		}
	}
	return out
}

// skipEscape returns the index of the last rune of the sequence starting at i,
// so the caller's loop increment lands just past it.
func skipEscape(rs []rune, i int) int {
	if i+1 >= len(rs) {
		return len(rs)
	}
	switch rs[i+1] {
	case '[': // CSI: parameters, then a final byte in @..~
		for j := i + 2; j < len(rs); j++ {
			if rs[j] >= 0x40 && rs[j] <= 0x7e {
				return j
			}
		}
		return len(rs)
	case ']': // OSC: runs to BEL, or to ESC \
		for j := i + 2; j < len(rs); j++ {
			if rs[j] == 0x07 {
				return j
			}
			if rs[j] == 0x1b && j+1 < len(rs) && rs[j+1] == '\\' {
				return j + 1
			}
		}
		return len(rs)
	default:
		return i + 1
	}
}

func (e *Editor) insert(r rune) { e.insertRunes([]rune{r}) }

// insertRunes splices text in at the cursor; the tail is copied out first,
// since the append would overwrite the runes still to be appended after it.
func (e *Editor) insertRunes(rs []rune) {
	if len(rs) == 0 {
		return
	}
	tail := append([]rune(nil), e.buf[e.cursor:]...)
	e.buf = append(e.buf[:e.cursor], rs...)
	e.buf = append(e.buf, tail...)
	e.cursor += len(rs)
}

func (e *Editor) transpose() {
	if len(e.buf) < 2 {
		return
	}
	i := min(e.cursor, len(e.buf)-1)
	if i == 0 {
		return
	}
	e.buf[i-1], e.buf[i] = e.buf[i], e.buf[i-1]
	if e.cursor < len(e.buf) {
		e.cursor++
	}
}

func (e *Editor) deleteBackward() {
	if e.cursor == 0 {
		return
	}
	e.buf = append(e.buf[:e.cursor-1], e.buf[e.cursor:]...)
	e.cursor--
}

func (e *Editor) deleteForward() {
	if e.cursor >= len(e.buf) {
		return
	}
	e.buf = append(e.buf[:e.cursor], e.buf[e.cursor+1:]...)
}

// deleteWord is Ctrl-W: kill back to the previous whitespace. bash uses
// whitespace here and word characters for Alt-Backspace, and the difference
// matters: Ctrl-W on a path takes the whole path, not one segment.
func (e *Editor) deleteWord() {
	i := e.cursor
	for i > 0 && e.buf[i-1] == ' ' {
		i--
	}
	for i > 0 && e.buf[i-1] != ' ' {
		i--
	}
	e.killRange(i, e.cursor)
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
}

// wordStart and wordEnd match Alt-B and Alt-F: skip non-word characters, then
// move over the word. Free functions so the steering prompt gets the same
// motions from the same code.
func wordStart(buf []rune, i int) int {
	for i > 0 && !isWordRune(buf[i-1]) {
		i--
	}
	for i > 0 && isWordRune(buf[i-1]) {
		i--
	}
	return i
}

func wordEnd(buf []rune, i int) int {
	for i < len(buf) && !isWordRune(buf[i]) {
		i++
	}
	for i < len(buf) && isWordRune(buf[i]) {
		i++
	}
	return i
}

func (e *Editor) wordStart(i int) int { return wordStart(e.buf, i) }
func (e *Editor) wordEnd(i int) int   { return wordEnd(e.buf, i) }

func (e *Editor) killBackwardWord() { e.killRange(e.wordStart(e.cursor), e.cursor) }
func (e *Editor) killForwardWord()  { e.killRange(e.cursor, e.wordEnd(e.cursor)) }

// killRange removes [from,to) and remembers it for Ctrl-Y.
func (e *Editor) killRange(from, to int) {
	if from >= to {
		return
	}
	e.kill = string(e.buf[from:to])
	e.buf = append(e.buf[:from], e.buf[to:]...)
	e.cursor = from
}

func (e *Editor) historyPrev() {
	if len(e.history) == 0 {
		return
	}
	if e.historyPos == -1 {
		e.saved = string(e.buf)
		e.historyPos = len(e.history)
	}
	if e.historyPos == 0 {
		return
	}
	e.historyPos--
	e.buf = []rune(e.history[e.historyPos])
	e.cursor = len(e.buf)
}

func (e *Editor) historyNext() {
	if e.historyPos == -1 {
		return
	}
	e.historyPos++
	if e.historyPos >= len(e.history) {
		e.historyPos = -1
		e.buf = []rune(e.saved)
	} else {
		e.buf = []rune(e.history[e.historyPos])
	}
	e.cursor = len(e.buf)
}

func (e *Editor) complete() {
	if e.Completions == nil {
		return
	}
	line := string(e.buf[:e.cursor])
	start, matches := e.Completions(line)
	if len(matches) == 0 || start < 0 || start > len(line) {
		return
	}
	at := utf8.RuneCountInString(line[:start])
	word := line[start:]

	if len(matches) == 1 {
		e.replaceWord(at, matches[0])
		return
	}
	if common := longestCommonPrefix(matches); len(common) > len(word) {
		e.replaceWord(at, common)
		return
	}
	e.listMatches(matches)
}

// replaceWord swaps the runes in [at,cursor) for text. Only that span is
// touched: a prompt is prose, and completing a path mid-sentence must not eat
// the rest of the sentence.
func (e *Editor) replaceWord(at int, text string) {
	tail := append([]rune(nil), e.buf[e.cursor:]...)
	e.buf = append(e.buf[:at], []rune(text)...)
	e.cursor = len(e.buf)
	e.buf = append(e.buf, tail...)
}

// maxListed caps an ambiguous-match listing; a directory of a thousand files
// would otherwise scroll the session away on one keystroke.
const maxListed = 60

// listMatches prints the candidates above the prompt, in columns: reading the
// list is how a suffix is chosen when there are dozens of them.
func (e *Editor) listMatches(matches []string) {
	width := 80
	if w, _, err := term.GetSize(int(e.out.Fd())); err == nil && w > 0 {
		width = w
	}

	shown := matches
	extra := 0
	if len(shown) > maxListed {
		extra = len(shown) - maxListed
		shown = shown[:maxListed]
	}

	widest := 0
	for _, m := range shown {
		widest = max(widest, len([]rune(strings.TrimRight(m, " "))))
	}
	cols := max(1, width/(widest+2))
	rows := (len(shown) + cols - 1) / cols

	var b strings.Builder
	b.WriteString("\r\n")
	for r := range rows {
		for c := range cols {
			// Column-major, like ls: the eye reads a sorted list down, not
			// across.
			i := c*rows + r
			if i >= len(shown) {
				continue
			}
			name := strings.TrimRight(shown[i], " ")
			b.WriteString(name)
			if c+1 < cols && i+rows < len(shown) {
				b.WriteString(strings.Repeat(" ", widest+2-len([]rune(name))))
			}
		}
		b.WriteString("\r\n")
	}
	if extra > 0 {
		fmt.Fprintf(&b, "... and %d more\r\n", extra)
	}
	_, _ = io.WriteString(e.out, b.String())
	e.invalidate()
}

func longestCommonPrefix(ss []string) string {
	if len(ss) == 0 {
		return ""
	}
	p := ss[0]
	for _, s := range ss[1:] {
		for !strings.HasPrefix(s, p) {
			p = p[:len(p)-1]
			if p == "" {
				return ""
			}
		}
	}
	return p
}

// Refresh repaints the line with a new prompt, for a caller that has learned
// something the prompt should say. Safe from another goroutine; a no-op before
// anything has been painted.
func (e *Editor) Refresh(prompt string) {
	if !e.isTTY() {
		return
	}
	e.paintMu.Lock()
	defer e.paintMu.Unlock()
	e.live = prompt
	if !e.painted {
		return
	}
	e.paint(prompt)
}

// SetHeader puts one replaceable line directly above the prompt. While the
// prompt is up it is rewritten in place, so a model that changes leaves one
// line; on submit it becomes scrollback, above the exchange it describes.
func (e *Editor) SetHeader(text, prompt string) {
	if !e.isTTY() {
		return
	}
	e.paintMu.Lock()
	defer e.paintMu.Unlock()
	if text == e.header {
		return
	}

	var b strings.Builder
	if e.headerDrawn {
		// Back up over the prompt line onto the header and overwrite it; the
		// line cannot be removed once drawn, so returning to the original
		// model blanks it.
		b.WriteString("\r\x1b[2K\x1b[1A")
	}
	b.WriteString("\r\x1b[2K")
	b.WriteString(text)
	b.WriteString("\r\n")
	fmt.Fprint(e.out, b.String())

	e.header, e.headerDrawn = text, true
	e.live = prompt
	e.invalidate()
	e.paint(prompt)
}

// ForgetHeader stops the line above from being rewritten; it now belongs to the
// exchange below it.
func (e *Editor) ForgetHeader() {
	e.paintMu.Lock()
	defer e.paintMu.Unlock()
	e.header, e.headerDrawn = "", false
}

// EmitAbove puts lines into the scrollback above the prompt and repaints it
// with whatever was typed intact; anything else written from another goroutine
// lands in the middle of the line being typed.
func (e *Editor) EmitAbove(prompt string, lines ...string) {
	if !e.isTTY() || len(lines) == 0 {
		return
	}
	e.paintMu.Lock()
	defer e.paintMu.Unlock()

	var b strings.Builder
	if e.painted {
		b.WriteString("\r\x1b[2K")
	}
	for _, l := range lines {
		b.WriteString(l)
		b.WriteString("\r\n")
	}
	fmt.Fprint(e.out, b.String())
	e.live = prompt
	e.invalidate()
	e.paint(prompt)
}

// redraw repaints the prompt line as a single write, carrying only the
// difference from what is on screen. Text wider than the terminal scrolls
// horizontally rather than wrapping, whose behaviour varies by terminal.
func (e *Editor) redraw(prompt string) {
	e.paintMu.Lock()
	defer e.paintMu.Unlock()
	e.paint(prompt)
}

func (e *Editor) paint(prompt string) {
	if e.live != "" {
		prompt = e.live
	}
	width := 80
	if w, _, err := term.GetSize(int(e.out.Fd())); err == nil && w > 0 {
		width = w
	}

	display := string(e.buf)
	cursorCol := e.cursor
	if s := summarise(display); s != display {
		display = s
		cursorCol = len([]rune(display))
	}

	promptW := render.VisibleWidth(prompt)
	avail := width - promptW - 1
	if avail < 10 {
		avail = 10
	}
	runes := []rune(display)
	start := 0
	if cursorCol > avail {
		start = cursorCol - avail
	}
	end := min(start+avail, len(runes))
	visible := runes[start:end]
	cursorScreen := promptW + (cursorCol - start)

	var b strings.Builder
	if !e.painted || prompt != e.lastPrompt || width != e.lastWidth {
		// Nothing reliable is known about the line, so repaint it whole; the
		// erase comes after the text, so the line is never momentarily blank.
		b.WriteString("\r")
		b.WriteString(prompt)
		b.WriteString(string(visible))
		b.WriteString("\x1b[K")
		moveTo(&b, promptW+len(visible), cursorScreen)
	} else if i := commonPrefix(e.lastVisible, visible); i == len(e.lastVisible) && i == len(visible) {
		// Text unchanged; this was a cursor movement.
		moveTo(&b, e.lastCursor, cursorScreen)
	} else {
		from := promptW + i
		moveTo(&b, e.lastCursor, from)
		b.WriteString(string(visible[i:]))
		endCol := promptW + len(visible)
		if len(visible) < len(e.lastVisible) {
			b.WriteString("\x1b[K")
		}
		moveTo(&b, endCol, cursorScreen)
	}

	if b.Len() > 0 {
		_, _ = io.WriteString(e.out, b.String())
	}

	e.painted = true
	e.lastPrompt = prompt
	e.lastWidth = width
	e.lastVisible = append(e.lastVisible[:0], visible...)
	e.lastCursor = cursorScreen
}

// summarise replaces multi-line content with a one-line stand-in: it is not
// edited inline (Ctrl-G edits it properly), and the line has to stay one
// physical row, since the erase depends on a counted row count.
func summarise(s string) string {
	i := strings.IndexByte(s, '\n')
	if i < 0 {
		return s
	}
	first := s[:i]
	if len(first) > 40 {
		first = first[:40]
	}
	return fmt.Sprintf("%s… [%d lines, Ctrl-G to edit]", first, strings.Count(s, "\n")+1)
}

// finalise leaves the completed input on the line before moving off it: redraw
// only shows a scrolled window into the buffer, and without this the scrollback
// would keep its tail. Wrapping is safe here, since the line is never redrawn.
func (e *Editor) finalise(prompt, text string) {
	// Prefer a prompt repainted while the line was being typed: otherwise the
	// marker committed to the scrollback says what was true when it first
	// appeared, not when it was submitted.
	if e.live != "" {
		prompt = e.live
	}
	var b strings.Builder
	b.WriteString("\r")
	b.WriteString(prompt)
	b.WriteString(render.BoundLine(summarise(text)))
	// The erase comes after the text: first would blank the visible line. It
	// clears the tail of the last row left by the scrolled window.
	b.WriteString("\x1b[K")
	b.WriteString("\r\n")
	_, _ = io.WriteString(e.out, b.String())
	e.invalidate()
}

// invalidate declares that something other than redraw has written to the
// terminal, so the next redraw repaints rather than diffing against a stale
// picture.
func (e *Editor) invalidate() { e.painted = false }

// moveTo emits the shortest cursor movement between two columns. Relative
// movement composes with whatever else is on the line, unlike absolute
// positioning.
func moveTo(b *strings.Builder, from, to int) {
	switch {
	case to > from:
		fmt.Fprintf(b, "\x1b[%dC", to-from)
	case to < from:
		fmt.Fprintf(b, "\x1b[%dD", from-to)
	}
}

func commonPrefix(a, b []rune) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// LaunchEditor opens an editor on the given text and returns what was saved.
func (e *Editor) LaunchEditor(initial string) (string, error) { return e.launchEditor(initial) }

// launchEditor opens $VISUAL or $EDITOR on the current buffer, the way git does
// for a commit message; it is the intended path for anything longer than a
// sentence.
func (e *Editor) launchEditor(initial string) (string, error) {
	editor := e.EditorCommand
	if editor == "" {
		editor = os.Getenv("VISUAL")
	}
	if editor == "" {
		editor = os.Getenv("EDITOR")
	}
	if editor == "" {
		editor = firstAvailable("nvim", "vim", "nano", "vi")
	}
	if editor == "" {
		return "", errors.New("no editor found: set editor in config.toml, or $VISUAL or $EDITOR")
	}

	tmp, err := os.CreateTemp("", "ai-code-prompt-*.md")
	if err != nil {
		return "", err
	}
	path := tmp.Name()
	defer os.Remove(path)

	body := initial
	if strings.TrimSpace(body) == "" {
		body = ""
	}
	if _, err := tmp.WriteString(body); err != nil {
		tmp.Close()
		return "", err
	}
	tmp.Close()

	// The editor takes over the terminal; ai-code must not hold raw mode or
	// draw while it runs.
	fields := strings.Fields(editor)
	cmd := exec.Command(fields[0], append(fields[1:], path)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("editor %s exited with an error: %w", fields[0], err)
	}

	out, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimRight(stripComments(string(out)), "\n"), nil
}

// stripComments removes lines beginning with '#', matching the convention
// people already know from git commit messages.
func stripComments(s string) string {
	var b strings.Builder
	for _, line := range strings.Split(s, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String()
}

func firstAvailable(names ...string) string {
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return filepath.Base(p)
		}
	}
	return ""
}

// reverseSearch is bash's Ctrl-R: type to walk backwards through history for a
// substring. Ctrl-R again steps further back, Enter submits the match, Escape or
// an arrow key keeps it for editing, and Ctrl-C restores the typed line.
func (e *Editor) reverseSearch(r *bufio.Reader, prompt string) (submit bool) {
	origBuf := append([]rune(nil), e.buf...)
	origCursor := e.cursor

	var query []rune
	from := len(e.history) // exclusive: where the next search starts looking
	found := -1

	// search walks backwards from the given index, leaving found and from
	// untouched if there is no further match, so holding Ctrl-R past the last
	// hit stays put.
	search := func(start int) {
		q := strings.ToLower(string(query))
		for i := start - 1; i >= 0; i-- {
			if q == "" || strings.Contains(strings.ToLower(e.history[i]), q) {
				found, from = i, i
				return
			}
		}
	}
	apply := func() {
		if found >= 0 {
			e.buf = []rune(e.history[found])
			e.cursor = len(e.buf)
		}
	}
	restart := func() {
		from, found = len(e.history), -1
		search(from)
		apply()
	}
	draw := func() {
		e.redraw(fmt.Sprintf("(reverse-i-search)`%s': ", string(query)))
	}

	search(from)
	apply()
	draw()

	for {
		ch, _, err := r.ReadRune()
		if err != nil {
			return false
		}
		switch ch {
		case '\r', '\n':
			e.invalidate()
			return true

		case 3: // Ctrl-C: abandon the search entirely
			e.buf, e.cursor = origBuf, origCursor
			e.invalidate()
			return false

		case 7: // Ctrl-G: keep the match, back to normal editing
			e.invalidate()
			return false

		case 27:
			// Escape leaves the search, and may also start an arrow key, which
			// in bash both exits and moves: hand the rest of the sequence to
			// the normal parser rather than reading it back as literal keys.
			if r.Buffered() > 0 {
				e.handleEscape(r)
			}
			e.invalidate()
			return false

		case 18: // Ctrl-R: next match, further back
			search(from)
			apply()

		case 127, 8:
			if len(query) > 0 {
				query = query[:len(query)-1]
				restart()
			}

		default:
			if ch >= 32 {
				query = append(query, ch)
				restart()
			}
		}
		draw()
	}
}
