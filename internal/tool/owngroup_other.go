//go:build !unix && !windows

package tool

import "os/exec"

func OwnProcessGroup(c *exec.Cmd) {}
