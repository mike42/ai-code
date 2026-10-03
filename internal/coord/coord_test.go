package coord

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

const server = "http://ai.example.internal:8000/v1"

// Fake pids, far above any real pid_max, so stale-socket cleanup leaves them alone.
var nextPID = 900_000_000

func joined(t *testing.T, dir string, p Peer) *Bus {
	t.Helper()
	nextPID++
	p.PID = nextPID
	b, err := Join(dir, p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Close)
	return b
}

func shortDir(t *testing.T) string {
	t.Helper()
	d, err := os.MkdirTemp("", "co")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(d) })
	return filepath.Join(d, "peers")
}

// until waits for cond, woken only by the bus's own change notifications.
func until(t *testing.T, b *Bus, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		ch := b.Changed()
		if cond() {
			return
		}
		select {
		case <-ch:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func TestInstancesSeeEachOtherAndEachChange(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Cwd: "/a", Server: server, Model: "m"})
	b := joined(t, dir, Peer{Cwd: "/b", Server: server, Model: "m"})

	until(t, a, "a to see b", func() bool { return len(a.Peers()) == 1 })
	until(t, b, "b to see a", func() bool { return len(b.Peers()) == 1 })

	b.Set(func(p *Peer) { p.State = StateWorking })
	until(t, a, "b's state", func() bool { return a.Peers()[0].State == StateWorking })

	b.Close()
	until(t, a, "b to be gone", func() bool { return len(a.Peers()) == 0 })
}

func TestASwapWaitsForEveryInstanceOnTheServerToAnswer(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Server: server, Model: "m"})
	b := joined(t, dir, Peer{Server: server, Model: "m"})
	elsewhere := joined(t, dir, Peer{Server: "http://other.example.internal/v1", Model: "m"})
	until(t, a, "both peers introduced", func() bool {
		_, ok1 := a.Peer(b.Self().PID)
		_, ok2 := a.Peer(elsewhere.Self().PID)
		return ok1 && ok2
	})

	done := make(chan error, 1)
	go func() { done <- a.Request(context.Background(), server, "x", 8192, nil) }()

	until(t, b, "the swap to reach b", func() bool { return len(b.Pending()) == 1 })
	if len(elsewhere.Pending()) != 0 {
		t.Error("an instance on another server was asked to stop")
	}
	select {
	case err := <-done:
		t.Fatalf("the swap went ahead before b answered (err %v)", err)
	case <-time.After(50 * time.Millisecond):
	}

	s := b.Pending()[0]
	if s.Into != "x" || s.IntoWindow != 8192 {
		t.Errorf("b saw %+v", s)
	}
	b.Ready(s.ID)
	if err := <-done; err != nil {
		t.Fatal(err)
	}

	a.Done(server, "x")
	until(t, b, "the load to be announced", func() bool { return b.Loaded() == "x" && len(b.Pending()) == 0 })
	if a.Loaded() != "x" {
		t.Errorf("the requester records %q as loaded", a.Loaded())
	}
	if l := b.Load(); l.Model != "x" || l.By != a.Self().PID {
		t.Errorf("b records %q loaded by pid %d, want x by %d", l.Model, l.By, a.Self().PID)
	}
}

func TestAnInstanceThatExitsNoLongerHoldsASwap(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Server: server, Model: "m"})
	b := joined(t, dir, Peer{Server: server, Model: "m"})
	until(t, a, "a to see b", func() bool { return len(a.Peers()) == 1 })

	done := make(chan error, 1)
	go func() { done <- a.Request(context.Background(), server, "x", 0, nil) }()
	until(t, b, "the swap to reach b", func() bool { return len(b.Pending()) == 1 })

	b.Close()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestARequesterThatExitsReleasesEveryoneWaiting(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Server: server, Model: "m"})
	b := joined(t, dir, Peer{Server: server, Model: "m"})
	until(t, a, "a to see b", func() bool { return len(a.Peers()) == 1 })

	go a.Request(context.Background(), server, "x", 0, nil)
	until(t, b, "the swap to reach b", func() bool { return len(b.Pending()) == 1 })
	a.Close()
	until(t, b, "the swap to die with a", func() bool { return len(b.Pending()) == 0 })
	if b.Loaded() != "" {
		t.Errorf("a swap that never completed left %q loaded", b.Loaded())
	}
}

func TestOfTwoSwapsTheEarlierGoesFirst(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Server: server, Model: "m"})
	b := joined(t, dir, Peer{Server: server, Model: "n"})
	until(t, a, "a to see b", func() bool { return len(a.Peers()) == 1 })
	until(t, b, "b to see a", func() bool { return len(b.Peers()) == 1 })

	first := make(chan error, 1)
	go func() { first <- a.Request(context.Background(), server, "x", 0, nil) }()
	until(t, b, "a's swap to reach b", func() bool { return len(b.Pending()) == 1 })

	if err := b.Request(context.Background(), server, "y", 0, nil); !errors.Is(err, ErrPreempted) {
		t.Fatalf("the later swap returned %v, want ErrPreempted", err)
	}
	until(t, a, "b's withdrawal", func() bool { return len(a.Pending()) == 0 })

	b.Ready(b.Pending()[0].ID)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
}

func TestANewcomerLearnsAPendingSwapAndWhatIsLoaded(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Server: server, Model: "m"})
	if err := a.Request(context.Background(), server, "x", 0, nil); err != nil {
		t.Fatal(err)
	}
	a.Done(server, "x")
	b := joined(t, dir, Peer{Server: server, Model: "x"})
	until(t, a, "a to see b", func() bool { return len(a.Peers()) == 1 })

	done := make(chan error, 1)
	go func() { done <- a.Request(context.Background(), server, "y", 0, nil) }()
	until(t, b, "the swap to reach b", func() bool { return len(b.Pending()) == 1 })

	c := joined(t, dir, Peer{Server: server, Model: "m"})
	until(t, c, "c to learn the swap", func() bool { return len(c.Pending()) == 1 })
	if c.Loaded() != "x" {
		t.Errorf("c learned %q as loaded, want x", c.Loaded())
	}
	b.Ready(b.Pending()[0].ID)
	select {
	case err := <-done:
		t.Fatalf("the swap went ahead without the newcomer (err %v)", err)
	case <-time.After(50 * time.Millisecond):
	}
	c.Ready(c.Pending()[0].ID)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

// Instances that start together each dial the other; both must settle on one
// connection and lose nothing.
func TestADoubleConnectionSettlesOnOne(t *testing.T) {
	dir := shortDir(t)
	a := joined(t, dir, Peer{Server: server, Model: "m"})
	b := joined(t, dir, Peer{Server: server, Model: "m"})
	until(t, a, "a to see b", func() bool { return len(a.Peers()) == 1 })
	until(t, b, "b to see a", func() bool { return len(b.Peers()) == 1 })

	// b dialled a when it joined; now a dials b too. a has the lower pid, so
	// both must keep a's connection.
	nc, err := net.Dial("unix", socketPath(dir, b.Self().PID))
	if err != nil {
		t.Fatal(err)
	}
	go a.serve(nc, true)

	settled := func(x *Bus, peer int, dialled bool) func() bool {
		return func() bool {
			x.mu.Lock()
			defer x.mu.Unlock()
			c := x.byPID[peer]
			return len(x.conns) == 1 && c != nil && c.dialed == dialled && x.conns[c]
		}
	}
	until(t, a, "a to keep its own connection", settled(a, b.Self().PID, true))
	until(t, b, "b to keep a's connection", settled(b, a.Self().PID, false))

	b.Set(func(p *Peer) { p.State = StateParked })
	until(t, a, "b's state over the survivor", func() bool { return a.Peers()[0].State == StateParked })
	a.Set(func(p *Peer) { p.State = StateWorking })
	until(t, b, "a's state over the survivor", func() bool { return b.Peers()[0].State == StateWorking })
}

func TestASocketLeftByADeadProcessIsCleared(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a dead process cannot be told from a live one here, so its socket stays")
	}
	dir := shortDir(t)
	if err := privateDir(dir); err != nil {
		t.Fatal(err)
	}
	stale := socketPath(dir, 2_000_000_000)
	ln, err := net.Listen("unix", stale)
	if err != nil {
		t.Fatal(err)
	}
	ln.(*net.UnixListener).SetUnlinkOnClose(false)
	ln.Close()

	joined(t, dir, Peer{Server: server})
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Errorf("the stale socket is still there (%v)", err)
	}
}

func TestASharedDirectoryIsRefused(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the per-user temporary directory is the boundary here")
	}
	dir := shortDir(t)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0o777); err != nil {
		t.Fatal(err)
	}
	if _, err := Join(dir, Peer{PID: 900_000_001}); err == nil {
		t.Error("joined through a directory other users can write to")
	}
}

func TestANilBusIsAnInstanceAlone(t *testing.T) {
	var b *Bus
	if err := b.Request(context.Background(), server, "x", 0, nil); err != nil {
		t.Fatal(err)
	}
	b.Done(server, "x")
	b.Set(func(*Peer) {})
	if b.Loaded() != "" || b.Pending() != nil || b.Peers() != nil {
		t.Error("a nil bus reported something")
	}
}
