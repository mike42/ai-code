//go:build !(linux || darwin || dragonfly || freebsd || netbsd || openbsd)

package ui

import "errors"

// rawInput is unsupported here: steering needs termios, and the Windows console
// equivalent belongs with the rest of the Windows port.
func rawInput(fd int) (func(), error) {
	return nil, errors.New("steering is not supported on this platform yet")
}

func readInput(fd int, buf []byte) (int, error) { return 0, errors.New("unsupported") }

const steeringSupported = false
