//go:build !unix

package tool

import "os/exec"

// No process groups on Windows: a cancelled command kills only its direct
// child.
func setProcessGroup(c *exec.Cmd) {}

func killProcessGroup(c *exec.Cmd, graceful bool) {
	if c.Process != nil {
		_ = c.Process.Kill()
	}
}
