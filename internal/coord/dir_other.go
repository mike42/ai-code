//go:build !unix

package coord

import "os"

// privateDir creates dir under the per-user temporary directory, which is
// where DefaultDir puts it on these platforms.
func privateDir(dir string) error { return os.MkdirAll(dir, 0o700) }

// processGone is never sure, so stale sockets are left in place; each costs a
// refused connection at startup.
func processGone(int) bool { return false }
