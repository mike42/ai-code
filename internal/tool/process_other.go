//go:build !unix

package tool

import "os/exec"

// Windows has no process groups in the POSIX sense. Killing the job object
// would be the equivalent and is deferred to the milestone that brings up
// Windows properly; until then a cancelled command kills only its direct child.
func setProcessGroup(c *exec.Cmd) {}

func killProcessGroup(c *exec.Cmd, graceful bool) {
	if c.Process != nil {
		_ = c.Process.Kill()
	}
}
