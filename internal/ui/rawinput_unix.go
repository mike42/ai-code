//go:build linux || darwin || dragonfly || freebsd || netbsd || openbsd

package ui

import (
	"errors"

	"golang.org/x/sys/unix"
)

// rawInput switches the terminal to character-at-a-time input without touching
// output processing, and returns a function that puts it back.
//
// Not term.MakeRaw, which also clears OPOST: the renderer streams the reply
// through this same terminal, and without the "\n" to CRLF translation its
// output descends the screen in a staircase.
//
// Only the input side changes, so Ctrl-C arrives as byte 3 rather than SIGINT
// and the caller must handle it -- which is the point, since steering
// distinguishes "cancel the turn" from "clear what I have typed".
//
// VMIN/VTIME make the read time out rather than block, so reading can stop
// when the turn ends. A blocked read holds the next prompt's first keystroke
// hostage until a second one releases it.
func rawInput(fd int) (restore func(), err error) {
	old, err := unix.IoctlGetTermios(fd, ioctlGetTermios)
	if err != nil {
		return nil, err
	}

	n := *old
	n.Lflag &^= unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN
	n.Iflag &^= unix.IXON | unix.ICRNL | unix.INLCR | unix.IGNCR | unix.BRKINT | unix.ISTRIP
	// Oflag is left exactly as it was: OPOST must survive.
	n.Cc[unix.VMIN] = 0
	n.Cc[unix.VTIME] = 1 // deciseconds

	if err := unix.IoctlSetTermios(fd, ioctlSetTermios, &n); err != nil {
		return nil, err
	}
	return func() { _ = unix.IoctlSetTermios(fd, ioctlSetTermios, old) }, nil
}

// readInput reads whatever is available, returning 0 at the VTIME timeout.
//
// unix.Read rather than os.File.Read, because os.File reports a zero-length
// read of a file as io.EOF -- and under VMIN=0 a zero-length read is the
// ordinary case, once every tenth of a second. Going through os.File would
// make every idle tick indistinguishable from the user pressing Ctrl-D.
func readInput(fd int, buf []byte) (int, error) {
	for {
		n, err := unix.Read(fd, buf)
		if errors.Is(err, unix.EINTR) {
			continue
		}
		if n < 0 {
			n = 0
		}
		return n, err
	}
}

const steeringSupported = true
