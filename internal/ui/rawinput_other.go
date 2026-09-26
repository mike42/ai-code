//go:build !(linux || darwin || dragonfly || freebsd || netbsd || openbsd)

package ui

import "errors"

// Steering needs character-at-a-time input with output processing left intact,
// which is a termios operation. The Windows console equivalent belongs with the
// rest of the Windows port rather than ahead of it, so steering is simply off
// here: the turn runs, the status line shows, and input is taken at the prompt
// as before.
func rawInput(fd int) (func(), error) {
	return nil, errors.New("steering is not supported on this platform yet")
}

func readInput(fd int, buf []byte) (int, error) { return 0, errors.New("unsupported") }

const steeringSupported = false
