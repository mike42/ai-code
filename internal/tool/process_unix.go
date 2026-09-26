//go:build unix

package tool

import (
	"os/exec"
	"syscall"
)

// setProcessGroup puts the command in its own process group so that cancelling
// it kills the whole tree.
//
// Killing the PID alone is not enough and the difference is not academic: a
// `make -j8` or a `npm test` spawns children that keep running, keep holding
// file handles, and keep writing to a pipe nobody is reading. Every subsequent
// tool call then contends with orphans from a command the user thought they
// cancelled.
func setProcessGroup(c *exec.Cmd) {
	if c.SysProcAttr == nil {
		c.SysProcAttr = &syscall.SysProcAttr{}
	}
	c.SysProcAttr.Setpgid = true
}

// killProcessGroup sends a signal to the entire group. The negative PID is what
// makes it a group operation.
func killProcessGroup(c *exec.Cmd, graceful bool) {
	if c.Process == nil {
		return
	}
	sig := syscall.SIGKILL
	if graceful {
		sig = syscall.SIGTERM
	}
	pgid, err := syscall.Getpgid(c.Process.Pid)
	if err != nil {
		_ = c.Process.Signal(sig)
		return
	}
	_ = syscall.Kill(-pgid, sig)
}
