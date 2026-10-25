package ui

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"

	"golang.org/x/sys/unix"

	"ai-code/internal/agent"
	"ai-code/internal/render"
)

// openPTY gives a terminal pair: the editor and screen write to the slave as
// they would to a real terminal, kernel output processing included, and the
// test reads what the terminal would receive from the master.
func openPTY(t *testing.T) (master, slave *os.File) {
	t.Helper()
	m, err := os.OpenFile("/dev/ptmx", os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Skipf("no pty: %v", err)
	}
	if err := unix.IoctlSetPointerInt(int(m.Fd()), unix.TIOCSPTLCK, 0); err != nil {
		t.Fatal(err)
	}
	n, err := unix.IoctlGetUint32(int(m.Fd()), unix.TIOCGPTN)
	if err != nil {
		t.Fatal(err)
	}
	s, err := os.OpenFile(fmt.Sprintf("/dev/pts/%d", n), os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.IoctlSetWinsize(int(s.Fd()), unix.TIOCSWINSZ, &unix.Winsize{Row: 24, Col: 80}); err != nil {
		t.Fatal(err)
	}
	return m, s
}

// vt is just enough of a terminal to lay out what the editor and screen
// write: carriage return, line feed, erases and horizontal cursor motion.
type vt struct {
	mu   sync.Mutex
	cond *sync.Cond
	rows [][]rune
	row  int
	col  int
	raw  strings.Builder
}

func newVT() *vt {
	v := &vt{rows: [][]rune{nil}}
	v.cond = sync.NewCond(&v.mu)
	return v
}

func (v *vt) feed(b []byte) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.raw.Write(b)
	s := []rune(string(b))
	for i := 0; i < len(s); i++ {
		switch r := s[i]; {
		case r == '\r':
			v.col = 0
		case r == '\n':
			v.row++
			for len(v.rows) <= v.row {
				v.rows = append(v.rows, nil)
			}
		case r == 0x1b && i+1 < len(s) && s[i+1] == '[':
			j := i + 2
			for j < len(s) && (s[j] < 0x40 || s[j] > 0x7e) {
				j++
			}
			if j < len(s) {
				v.csi(string(s[i+2:j]), s[j])
			}
			i = j
		case r >= 0x20:
			line := v.rows[v.row]
			for len(line) <= v.col {
				line = append(line, ' ')
			}
			line[v.col] = r
			v.rows[v.row] = line
			v.col++
		}
	}
	v.cond.Broadcast()
}

func (v *vt) csi(params string, final rune) {
	n, err := strconv.Atoi(strings.TrimPrefix(params, "?"))
	if err != nil {
		n = 0
	}
	switch final {
	case 'K':
		line := v.rows[v.row]
		if n == 2 {
			v.rows[v.row] = nil
		} else if v.col < len(line) {
			v.rows[v.row] = line[:v.col]
		}
	case 'D':
		v.col = max(0, v.col-max(n, 1))
	case 'C':
		v.col += max(n, 1)
	}
}

// screen is the visible text, one string per row, trailing blanks trimmed.
func (v *vt) screenLocked() []string {
	out := make([]string, len(v.rows))
	for i, r := range v.rows {
		out[i] = strings.TrimRight(string(r), " ")
	}
	return out
}

// waitFor blocks until the screen satisfies ok.
func (v *vt) waitFor(ok func(rows []string, row, col int) bool) []string {
	v.mu.Lock()
	defer v.mu.Unlock()
	for !ok(v.screenLocked(), v.row, v.col) {
		v.cond.Wait()
	}
	return v.screenLocked()
}

// syncMark is a control sequence no terminal acts on, so it changes nothing
// on screen.
const syncMark = "\x1b[7777q"

func lastNonEmpty(rows []string) (int, string) {
	for i := len(rows) - 1; i >= 0; i-- {
		if rows[i] != "" {
			return i, rows[i]
		}
	}
	return -1, ""
}

// A notice arriving while a line is half typed -- an MCP server finishing its
// start, a devcontainer coming up -- goes into the scrollback above the
// prompt, starting in column zero, and the prompt is drawn again below it
// with the typed text and the cursor where they were.
func TestNoticesWhileTypingLandAboveThePrompt(t *testing.T) {
	master, slave := openPTY(t)
	term := newVT()
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := master.Read(buf)
			if n > 0 {
				term.feed(buf[:n])
			}
			if err != nil {
				return
			}
		}
	}()
	t.Cleanup(func() { slave.Close(); master.Close() })

	screen := render.NewScreen(slave, "never")
	if !screen.IsTTY() {
		t.Fatal("the pty slave is not a terminal")
	}
	r := render.NewInteractive(screen, render.InteractiveOptions{})
	ed := NewEditor(slave, slave)

	screen.Attach(ed)
	got := make(chan string, 1)
	go func() {
		line, err := ed.ReadLine("> ")
		if err != nil {
			t.Error(err)
		}
		got <- line
	}()

	term.waitFor(func(rows []string, _, _ int) bool { _, l := lastNonEmpty(rows); return l == ">" })
	master.WriteString("helo")
	term.waitFor(func(rows []string, _, _ int) bool { _, l := lastNonEmpty(rows); return l == "> helo" })
	// Ctrl-B: the cursor sits inside the word, which the repaint must keep.
	master.WriteString("\x02")
	term.waitFor(func(_ []string, _, col int) bool { return col == len("> hel") })

	r.Emit(agent.Event{Kind: agent.EvNotice, Text: "MCP ghidra failed: handshake: exited (status 127): ghidra-mcp: not found"})
	r.SetNotice("devcontainer: building the image")

	// Both calls have written everything by the time they return; a marker
	// written after them arrives after all of it.
	slave.WriteString(syncMark)
	rows := term.waitFor(func([]string, int, int) bool { return strings.Contains(term.raw.String(), syncMark) })
	at, _ := lastNonEmpty(rows)
	want := []string{
		"",
		"note  MCP ghidra failed: handshake: exited (status 127): ghidra-mcp: not found",
		"> helo",
	}
	if at < 2 || strings.Join(rows[at-2:at+1], "\n") != strings.Join(want, "\n") {
		t.Fatalf("screen =\n%s\nwant it to end with\n%s\nbytes: %q",
			strings.Join(rows, "\n"), strings.Join(want, "\n"), term.raw.String())
	}
	if strings.Contains(strings.Join(rows, "\n"), "building the image") {
		t.Errorf("the status line was drawn over the prompt:\n%s", strings.Join(rows, "\n"))
	}

	// Typing carries on at the cursor that was restored.
	master.WriteString("l\r")
	if line := <-got; line != "hello" {
		t.Errorf("ReadLine = %q, want the typed line with the edit applied", line)
	}
	screen.Detach()

	// Back on a cooked terminal, a commit lands on a fresh line as usual.
	screen.Commit("after")
	rows = term.waitFor(func(rows []string, _, _ int) bool { _, l := lastNonEmpty(rows); return l == "after" })
	if at, _ := lastNonEmpty(rows); rows[at-1] != "> hello" {
		t.Errorf("screen =\n%s\nwant the submitted line, then the commit below it", strings.Join(rows, "\n"))
	}
}
