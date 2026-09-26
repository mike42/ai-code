//go:build !unix

package main

import "errors"

// Windows has no execve. Relaunching means spawning a child and exiting, which
// hands the terminal back to the shell mid-session, so /restart stays off until
// that is done properly.
func execSelf(binary string, argv []string) error {
	return errors.New("restarting in place needs exec, which this platform does not have")
}
