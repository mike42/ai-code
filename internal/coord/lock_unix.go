//go:build unix

package coord

import (
	"os"
	"syscall"
)

// canDetectLiveness reports whether a dead instance can be told from a live
// one.
const canDetectLiveness = true

// lockExclusive takes an exclusive advisory lock without blocking. The kernel
// drops it when the process dies, however it dies, which is why liveness is a
// lock rather than a heartbeat or a pid check.
func lockExclusive(f *os.File) (bool, error) {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		if err == syscall.EWOULDBLOCK {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// lockBlocking waits for the exclusive lock rather than reporting it taken.
func lockBlocking(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
}

func unlock(f *os.File) { _ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }

// isHeld reports whether some process holds an exclusive lock on this file:
// a shared lock is granted only when no exclusive one exists.
func isHeld(f *os.File) bool {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_SH|syscall.LOCK_NB); err != nil {
		return true
	}
	_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	return false
}
