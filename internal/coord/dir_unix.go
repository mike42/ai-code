//go:build unix

package coord

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
)

// privateDir creates dir and checks that it, and the directory holding it,
// belong to this user alone: anyone who can write there can talk to every
// instance.
func privateDir(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	for _, d := range []string{filepath.Dir(dir), dir} {
		fi, err := os.Lstat(d)
		if err != nil {
			return err
		}
		st, ok := fi.Sys().(*syscall.Stat_t)
		if !fi.IsDir() || !ok || int(st.Uid) != os.Getuid() || fi.Mode().Perm()&0o077 != 0 {
			return fmt.Errorf("%s is not a directory private to this user", d)
		}
	}
	return nil
}

func processGone(pid int) bool {
	return errors.Is(syscall.Kill(pid, 0), syscall.ESRCH)
}
