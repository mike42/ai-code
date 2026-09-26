//go:build !unix

package coord

import "os"

// canDetectLiveness is false without flock, so coordination switches itself
// off rather than guess. Assuming everything is live blocks swaps forever;
// assuming everything is dead silently disables the feature.
const canDetectLiveness = false

func lockExclusive(*os.File) (bool, error) { return false, nil }

func lockBlocking(*os.File) error { return errNoLiveness }

func unlock(*os.File) {}

func isHeld(*os.File) bool { return false }
