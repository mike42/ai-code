//go:build unix

package coord

import (
	"os"
	"syscall"
)

// canDetectLiveness reports whether a dead instance can be told from a live
// one. See the constant in lock_other.go for what happens when it cannot.
const canDetectLiveness = true

// lockExclusive takes an exclusive advisory lock and returns a release
// function, or false if another process already holds it.
//
// The lock lives on the open file description, so the kernel drops it when
// the process dies however it dies -- crash, SIGKILL, power loss. That is the
// whole reason liveness is a lock rather than a heartbeat or a kill(pid, 0):
// a heartbeat needs a timeout to decide someone is gone, and a pid can be
// reused or belong to another namespace.
func lockExclusive(f *os.File) (bool, error) {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if err == syscall.EWOULDBLOCK {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// isHeld reports whether some process holds an exclusive lock on this file.
// Asked by taking a shared lock: it is granted only when no exclusive one
// exists, and is released immediately.
// lockBlocking waits for the exclusive lock rather than reporting it taken.
func lockBlocking(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
}

func unlock(f *os.File) { _ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }

func isHeld(f *os.File) bool {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_SH|syscall.LOCK_NB); err != nil {
		return true
	}
	_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	return false
}
