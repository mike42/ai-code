//go:build unix

package main

import (
	"os"
	"syscall"
)

// execSelf replaces this process with a fresh copy of the binary. On success it
// does not return: the pid, terminal and open descriptors stay as they were.
func execSelf(binary string, argv []string) error {
	return syscall.Exec(binary, argv, os.Environ())
}
