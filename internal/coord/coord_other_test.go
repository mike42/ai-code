//go:build !unix

package coord

import "testing"

// Without flock coordination reports nothing and records nothing, so a
// dead window can never block a swap on the strength of a file nobody owns.
func TestCoordinationIsOffWithoutLiveness(t *testing.T) {
	dir := t.TempDir()
	if _, err := Register(dir, Peer{Model: "m"}); err == nil {
		t.Error("Register succeeded on a platform that cannot prove liveness")
	}
	if got := Peers(dir); got != nil {
		t.Errorf("Peers() = %+v, want nothing", got)
	}
	if got := NotReady(dir, "m"); got != nil {
		t.Errorf("BusyOn() = %+v, want nothing", got)
	}
}
