//go:build unix

package tool

import (
	"os/exec"
	"syscall"
)

// setProcessGroup puts the command in its own process group so that cancelling
// it kills the whole tree. Killing the PID alone leaves children running,
// holding file handles and writing to a pipe nobody is reading.
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
