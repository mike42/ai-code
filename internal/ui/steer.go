package ui

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"golang.org/x/term"
)

// Steerer reads the terminal while a turn is running, so the model can be
// corrected without waiting for it to stop. It never writes to the terminal:
// it maintains a buffer and reports it through Render, which owns the zone.
type Steerer struct {
	// Render reports the current line. display is already a single line, with
	// multi-line input summarised; cursorCol is a column within it; active is
	// false when nothing is typed, the cue to restore the status line.
	Render func(display string, cursorCol int, active bool)
	// Submit receives a finished message. It is called from the read goroutine.
	Submit func(text string)
	// Interrupt is called for Ctrl-C on an empty line.
	Interrupt func()
	// Completions supplies Tab candidates for the line up to the cursor, in the
	// same shape as Editor.Completions. Called off the lock; it reads the
	// filesystem.
	Completions func(line string) (start int, candidates []string)

	// Control writes a terminal mode sequence: bracketed paste has to be on
	// while steering reads, or a pasted snippet submits itself a line at a
	// time. It goes through the caller, serialised with everything else written.
	Control func(seq string)

	fd  int
	mu  sync.Mutex
	buf []rune
	cur int

	stop    chan struct{}
	done    chan struct{}
	started bool
}

// NewSteerer returns a Steerer reading the given terminal file descriptor.
func NewSteerer(fd int) *Steerer {
	return &Steerer{fd: fd, stop: make(chan struct{}), done: make(chan struct{})}
}

// SteeringAvailable reports whether steering can run on this terminal.
func SteeringAvailable(fd int) bool { return steeringSupported && term.IsTerminal(fd) }

// Start puts the terminal into character-at-a-time input mode and begins
// reading. Stop must be called before anything else reads stdin.
func (s *Steerer) Start() error {
	if !SteeringAvailable(s.fd) {
		return fmt.Errorf("steering unavailable on this input")
	}
	restore, err := rawInput(s.fd)
	if err != nil {
		return err
	}
	s.started = true
	s.control(bracketedPasteOn)

	go func() {
		defer close(s.done)
		defer restore()

		buf := make([]byte, 1024)
		var pending []byte
		for {
			select {
			case <-s.stop:
				return
			default:
			}

			n, err := readInput(s.fd, buf)
			if err != nil {
				return
			}
			if n == 0 {
				continue // VTIME expiry: nothing typed
			}
			pending = s.consume(append(pending, buf[:n]...))
		}
	}()
	return nil
}

// Stop restores the terminal and returns whatever was typed but not submitted;
// a half-typed sentence belongs in the prompt that replaces this one.
func (s *Steerer) Stop() string {
	if !s.started {
		return ""
	}
	s.started = false
	close(s.stop)
	<-s.done
	s.control(bracketedPasteOff)

	s.mu.Lock()
	defer s.mu.Unlock()
	left := string(s.buf)
	s.buf, s.cur = nil, 0
	return left
}

// maxPending bounds the bytes held waiting for a keystroke to complete; a
// bracketed paste can be a whole file, so the bound is generous.
const maxPending = 4 << 20

func (s *Steerer) control(seq string) {
	if s.Control != nil {
		s.Control(seq)
	}
}

// consume processes as many complete keystrokes as the bytes allow, returning
// the incomplete tail: a rune, an escape sequence or a paste can be split across
// reads, and treating the halves as separate keys mangles all three.
func (s *Steerer) consume(b []byte) []byte {
	for len(b) > 0 {
		n, ok := s.key(b)
		if ok {
			b = b[n:]
			continue
		}
		// Incomplete. A truncated rune is at most four bytes, so anything
		// longer that is not an escape began with a byte that cannot start one:
		// drop it rather than stall on it forever.
		if b[0] != 0x1b && len(b) > 4 {
			b = b[1:]
			continue
		}
		if len(b) > maxPending {
			return nil
		}
		return b
	}
	return nil
}

// key handles one keystroke, returning how many bytes it consumed; ok is false
// when b holds only part of one.
func (s *Steerer) key(b []byte) (int, bool) {
	switch b[0] {
	case '\r', '\n':
		s.submit()
		return 1, true

	case 3: // Ctrl-C
		// On an empty line this cancels the turn; with something typed it
		// forgets the typing, so Ctrl-C twice still cancels and never discards
		// a queued message silently.
		if s.discard() && s.Interrupt != nil {
			s.Interrupt()
		}
		return 1, true

	case 4: // Ctrl-D
		// Not end-of-input here: holding a key down while a turn runs is not a
		// way to exit ai-code.
		return 1, true

	case 21: // Ctrl-U
		s.edit(func() { s.buf, s.cur = append([]rune{}, s.buf[s.cur:]...), 0 })
		return 1, true
	case 11: // Ctrl-K
		s.edit(func() { s.buf = s.buf[:s.cur] })
		return 1, true
	case 23: // Ctrl-W
		s.edit(s.deleteWordLocked)
		return 1, true
	case 1: // Ctrl-A
		s.edit(func() { s.cur = 0 })
		return 1, true
	case 5: // Ctrl-E
		s.edit(func() { s.cur = len(s.buf) })
		return 1, true
	case 2: // Ctrl-B
		s.edit(func() { s.cur = max(0, s.cur-1) })
		return 1, true
	case 6: // Ctrl-F
		s.edit(func() { s.cur = min(len(s.buf), s.cur+1) })
		return 1, true

	case '\t':
		s.complete()
		return 1, true

	case 127, 8: // Backspace
		s.edit(func() {
			if s.cur > 0 {
				s.buf = append(s.buf[:s.cur-1], s.buf[s.cur:]...)
				s.cur--
			}
		})
		return 1, true

	case 27: // Escape sequence
		return s.escape(b)
	}

	if b[0] < 32 {
		return 1, true // any other control: ignored rather than inserted
	}

	r, size := decodeRune(b)
	switch {
	case size == 0:
		return 0, false // truncated multi-byte rune; wait for the rest
	case r < 0:
		return 1, true // continuation byte with no lead: noise, not input
	}
	s.edit(func() { s.insertLocked([]rune{r}) })
	return size, true
}

// escape handles the CSI sequences worth having at a steering prompt: the
// motions and bracketed paste. Anything else is consumed and discarded, since an
// unconsumed tail is read back as literal keystrokes.
func (s *Steerer) escape(b []byte) (int, bool) {
	if len(b) < 2 {
		return 0, false
	}
	if b[1] != '[' && b[1] != 'O' {
		return 2, true // Alt-key or a stray escape
	}
	if b[1] == 'O' {
		if len(b) < 3 {
			return 0, false
		}
		s.motion(rune(b[2]), 0)
		return 3, true
	}

	i := 2
	for i < len(b) && (b[i] < 0x40 || b[i] > 0x7e) {
		i++
	}
	if i >= len(b) {
		return 0, false
	}
	params, final := string(b[2:i]), b[i]
	i++

	if final == '~' && strings.HasPrefix(params, "200") {
		// Bracketed paste: the payload runs to the closing marker, taken as a
		// block so a pasted snippet does not submit on its first newline.
		const end = "\x1b[201~"
		j := strings.Index(string(b[i:]), end)
		if j < 0 {
			return 0, false
		}
		text := string(b[i : i+j])
		text = strings.ReplaceAll(strings.ReplaceAll(text, "\r\n", "\n"), "\r", "\n")
		s.edit(func() { s.insertLocked(sanitisePaste(text)) })
		return i + j + len(end), true
	}

	// A modifier arrives as a second parameter: "1;5D" is Ctrl-Left, "1;3D"
	// Alt-Left. Both are word motions, as they are at the prompt.
	mod := 0
	if j := strings.IndexByte(params, ';'); j >= 0 {
		mod, _ = strconv.Atoi(params[j+1:])
		params = params[:j]
	}

	switch final {
	case 'A', 'B', 'C', 'D', 'H', 'F':
		s.motion(rune(final), mod)
	case '~':
		switch params {
		case "1", "7":
			s.edit(func() { s.cur = 0 })
		case "4", "8":
			s.edit(func() { s.cur = len(s.buf) })
		case "3":
			s.edit(func() {
				if s.cur < len(s.buf) {
					s.buf = append(s.buf[:s.cur], s.buf[s.cur+1:]...)
				}
			})
		}
	}
	return i, true
}

func (s *Steerer) motion(final rune, mod int) {
	word := isWordMod(mod)
	switch final {
	case 'C':
		s.edit(func() {
			if word {
				s.cur = wordEnd(s.buf, s.cur)
			} else {
				s.cur = min(len(s.buf), s.cur+1)
			}
		})
	case 'D':
		s.edit(func() {
			if word {
				s.cur = wordStart(s.buf, s.cur)
			} else {
				s.cur = max(0, s.cur-1)
			}
		})
	case 'H':
		s.edit(func() { s.cur = 0 })
	case 'F':
		s.edit(func() { s.cur = len(s.buf) })
	}
	// Up and Down are ignored: history belongs to the prompt, and stepping
	// through it would replace a half-written message.
}

// complete extends the word before the cursor, the way Tab does at the prompt.
// The ambiguous-match listing cannot happen here: it prints above the prompt,
// and the Steerer must not write to a terminal the renderer is using.
func (s *Steerer) complete() {
	if s.Completions == nil {
		return
	}
	s.mu.Lock()
	line := string(s.buf[:s.cur])
	s.mu.Unlock()

	start, matches := s.Completions(line)
	if len(matches) == 0 || start < 0 || start > len(line) {
		return
	}

	text := matches[0]
	if len(matches) > 1 {
		// Nothing shared beyond what is typed: at the prompt this lists the
		// candidates, and here there is nowhere to list them.
		common := longestCommonPrefix(matches)
		if len(common) <= len(line)-start {
			return
		}
		text = common
	}

	at := utf8.RuneCountInString(line[:start])
	s.edit(func() {
		// The candidates were gathered off the lock; if the line changed
		// since, the offsets describe a line that no longer exists.
		if s.cur > len(s.buf) || at > s.cur || string(s.buf[:s.cur]) != line {
			return
		}
		tail := append([]rune(nil), s.buf[s.cur:]...)
		s.buf = append(s.buf[:at], []rune(text)...)
		s.cur = len(s.buf)
		s.buf = append(s.buf, tail...)
	})
}

func (s *Steerer) insertLocked(rs []rune) {
	tail := append([]rune(nil), s.buf[s.cur:]...)
	s.buf = append(append(s.buf[:s.cur], rs...), tail...)
	s.cur += len(rs)
}

func (s *Steerer) deleteWordLocked() {
	i := s.cur
	for i > 0 && s.buf[i-1] == ' ' {
		i--
	}
	for i > 0 && s.buf[i-1] != ' ' {
		i--
	}
	s.buf = append(s.buf[:i], s.buf[s.cur:]...)
	s.cur = i
}

// edit applies a mutation and reports the result; the render callback runs
// outside the lock, since it reaches the screen and the screen has a lock.
func (s *Steerer) edit(f func()) {
	s.mu.Lock()
	f()
	display, col := summariseLine(s.buf, s.cur)
	active := len(s.buf) > 0
	s.mu.Unlock()

	if s.Render != nil {
		s.Render(display, col, active)
	}
}

// discard clears the buffer, reporting whether it was already empty.
func (s *Steerer) discard() (wasEmpty bool) {
	s.mu.Lock()
	wasEmpty = len(s.buf) == 0
	s.buf, s.cur = nil, 0
	s.mu.Unlock()
	if s.Render != nil {
		s.Render("", 0, false)
	}
	return wasEmpty
}

func (s *Steerer) submit() {
	s.mu.Lock()
	text := string(s.buf)
	s.buf, s.cur = nil, 0
	s.mu.Unlock()

	if s.Render != nil {
		s.Render("", 0, false)
	}
	if strings.TrimSpace(text) != "" && s.Submit != nil {
		s.Submit(text)
	}
}

// decodeRune reads one UTF-8 rune, returning size 0 if b holds only part of it.
func decodeRune(b []byte) (rune, int) {
	need := 1
	switch {
	case b[0] < 0x80:
	case b[0]&0xe0 == 0xc0:
		need = 2
	case b[0]&0xf0 == 0xe0:
		need = 3
	case b[0]&0xf8 == 0xf0:
		need = 4
	default:
		return -1, 1 // continuation byte with no lead
	}
	if len(b) < need {
		return 0, 0
	}
	rs := []rune(string(b[:need]))
	return rs[0], need
}

// summariseLine reduces a buffer to one displayable line and a column in it,
// the same way the line editor does: multi-line input is described, not
// rendered, and echoed in full on submit.
func summariseLine(buf []rune, cur int) (string, int) {
	s := string(buf)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		first := []rune(s[:i])
		if len(first) > 40 {
			first = first[:40]
		}
		display := fmt.Sprintf("%s… [%d lines]", string(first), strings.Count(s, "\n")+1)
		return display, len([]rune(display))
	}
	return s, cur
}
