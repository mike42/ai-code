//go:build !unix

package main

import "errors"

// execSelf is unavailable without execve: relaunching would hand the terminal
// back to the shell mid-session.
func execSelf(binary string, argv []string) error {
	return errors.New("restarting in place needs exec, which this platform does not have")
}
