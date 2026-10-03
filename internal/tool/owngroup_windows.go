//go:build windows

package tool

import (
	"os/exec"
	"syscall"
)

// A new process group keeps the console's Ctrl-C from reaching c.
func OwnProcessGroup(c *exec.Cmd) {
	if c.SysProcAttr == nil {
		c.SysProcAttr = &syscall.SysProcAttr{}
	}
	c.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
}
