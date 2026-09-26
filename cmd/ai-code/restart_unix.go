//go:build unix

package main

import (
	"os"
	"syscall"
)

// execSelf replaces this process with a fresh copy of the binary.
//
// This is execve, so on success it does not return: the process image is gone,
// the pid, the terminal, the working directory and every open descriptor stay
// exactly as they were. That is what makes a restart invisible to the terminal
// the session is running in.
func execSelf(binary string, argv []string) error {
	return syscall.Exec(binary, argv, os.Environ())
}
