//go:build linux || darwin || dragonfly || freebsd || netbsd || openbsd

package ui

import (
	"errors"

	"golang.org/x/sys/unix"
)

// rawInput switches the terminal to character-at-a-time input without touching
// output processing, and returns a restore function. Not term.MakeRaw, which
// clears OPOST too and leaves output in a staircase; VMIN/VTIME time reads out.
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
// unix.Read rather than os.File.Read: os.File reports a zero-length read as
// io.EOF, and under VMIN=0 that is the ordinary case once per tenth of a second.
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
