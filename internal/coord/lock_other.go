//go:build !unix

package coord

import "os"

// canDetectLiveness is false without flock, and coordination switches itself
// off rather than guess.
//
// Guessing in either direction is worse than not running. Assume everything
// is live and a window that crashed blocks model swaps forever, on the
// evidence of a file nobody owns. Assume everything is dead and the feature
// silently never fires. Off is the only honest third option, and it leaves
// the harness exactly as it was before any of this existed.
const canDetectLiveness = false

func lockExclusive(*os.File) (bool, error) { return false, nil }

func lockBlocking(*os.File) error { return errNoLiveness }

func unlock(*os.File) {}

func isHeld(*os.File) bool { return false }
