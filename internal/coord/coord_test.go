//go:build unix

package coord

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestMain lets the test binary re-exec itself as a peer that registers and
// blocks. flock is per open file description, so a child holding the lock is
// the only way to test liveness honestly.
func TestMain(m *testing.M) {
	if dir := os.Getenv("AI_CODE_TEST_SETINTENT_DIR"); dir != "" {
		_ = Decide(dir, func() { _ = SetIntent(dir, "written-elsewhere") })
		os.Exit(0)
	}
	if dir := os.Getenv("AI_CODE_TEST_DECIDE_DIR"); dir != "" {
		_ = Decide(dir, func() { fmt.Println("in") })
		os.Exit(0)
	}
	if dir := os.Getenv("AI_CODE_TEST_PEER_DIR"); dir != "" {
		reg, err := Register(dir, Peer{Model: os.Getenv("AI_CODE_TEST_PEER_MODEL"), Cwd: "/tmp/peer"})
		if err != nil {
			os.Exit(2)
		}
		if os.Getenv("AI_CODE_TEST_PEER_BUSY") == "1" {
			reg.SetBusy(true)
		}
		if st := os.Getenv("AI_CODE_TEST_PEER_STATE"); st != "" {
			reg.SetState(State(st))
		}
		fmt.Println("ready")
		// Not select{}: with every goroutine blocked the runtime calls that a
		// deadlock and aborts, which quietly unregisters the peer the test is
		// about to look for.
		time.Sleep(time.Hour)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// startPeer runs a second process that registers and stays alive until killed.
func startPeer(t *testing.T, dir, model string, busy bool) *exec.Cmd {
	return startPeerState(t, dir, model, busy, "")
}

func startPeerState(t *testing.T, dir, model string, busy bool, state string) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(),
		"AI_CODE_TEST_PEER_DIR="+dir,
		"AI_CODE_TEST_PEER_MODEL="+model,
		"AI_CODE_TEST_PEER_BUSY="+map[bool]string{true: "1", false: "0"}[busy],
		"AI_CODE_TEST_PEER_STATE="+state)
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})

	buf := make([]byte, 6)
	if _, err := out.Read(buf); err != nil {
		t.Fatalf("peer never became ready: %v", err)
	}
	return cmd
}

func TestAPeerIsVisibleToOtherInstances(t *testing.T) {
	dir := t.TempDir()
	peer := startPeer(t, dir, "big-model", false)

	got := Peers(dir)
	if len(got) != 1 {
		t.Fatalf("saw %d peers, want 1", len(got))
	}
	if got[0].PID != peer.Process.Pid || got[0].Model != "big-model" {
		t.Errorf("peer = %+v, want pid %d on big-model", got[0], peer.Process.Pid)
	}
}

// Liveness is a lock, not a heartbeat: a killed process frees its entry at
// once, with no timeout to wait out and no pid to misidentify.
func TestAKilledPeerFreesItsEntryImmediately(t *testing.T) {
	dir := t.TempDir()
	peer := startPeer(t, dir, "big-model", true)

	if len(Peers(dir)) != 1 {
		t.Fatal("setup: the peer was not registered")
	}

	if err := peer.Process.Signal(syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	_, _ = peer.Process.Wait()

	start := time.Now()
	if n := len(Peers(dir)); n != 0 {
		t.Errorf("a SIGKILLed peer is still listed (%d entries)", n)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("took %v to notice; it should be immediate", elapsed)
	}

	// It tidies up after itself, so the directory does not accumulate.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if pid, err := strconv.Atoi(filepath.Base(e.Name()[:len(e.Name())-len(filepath.Ext(e.Name()))])); err == nil {
			if pid == peer.Process.Pid {
				t.Errorf("%s was left behind", e.Name())
			}
		}
	}
}

// A session in a turn is what a swap has to wait out, and the state is how
// it says so.
func TestAWorkingSessionIsWhatASwapWaitsFor(t *testing.T) {
	dir := t.TempDir()
	startPeerState(t, dir, "big-model", true, string(StateWorking))
	startPeerState(t, dir, "other-model", true, string(StateWorking))

	if got := NotReady(dir, "big-model"); len(got) != 1 {
		t.Errorf("waiting on %d sessions for big-model, want 1", len(got))
	}
	if got := NotReady(dir, "nobody-model"); len(got) != 0 {
		t.Errorf("waiting on %d sessions for a model nobody has", len(got))
	}
}

// A session must not list itself, or a swap waits for itself to park.
func TestSelfIsNotAPeer(t *testing.T) {
	dir := t.TempDir()
	reg, err := Register(dir, Peer{Model: "mine"})
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	if got := Peers(dir); len(got) != 0 {
		t.Errorf("listed self as a peer: %+v", got)
	}
}

func TestCloseRemovesTheEntry(t *testing.T) {
	dir := t.TempDir()
	reg, err := Register(dir, Peer{Model: "mine"})
	if err != nil {
		t.Fatal(err)
	}
	reg.Close()

	if got := Peers(dir); len(got) != 0 {
		t.Errorf("entry survived Close: %+v", got)
	}
}

// Every method has to work on a nil receiver, because a session that could
// not register still has to run.
func TestANilRegistrationIsUsable(t *testing.T) {
	var r *Registration
	r.SetBusy(true)
	r.SetModel("p", "m")
	r.Update(func(*Peer) { t.Error("Update ran on a nil registration") })
	r.Close()
	if got := r.Self(); got.PID != 0 {
		t.Errorf("Self() on nil = %+v", got)
	}
}

func TestRegisterFailsGracefullyOnAnUnusableDirectory(t *testing.T) {
	file := filepath.Join(t.TempDir(), "notadir")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Register(filepath.Join(file, "peers"), Peer{}); err == nil {
		t.Error("registering under a regular file should fail rather than panic")
	}
}

// The announcement must reach a session while the model is still loaded,
// because writing a summary needs the model.
func TestAPeerSeesTheAnnouncementBeforeTheModelGoes(t *testing.T) {
	dir := t.TempDir()
	reg, err := Register(dir, Peer{Model: "big-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	if _, ok := PendingSwap(dir); ok {
		t.Fatal("a swap was pending before one was announced")
	}

	// A second process announces; ours must see it.
	peer := startPeer(t, dir, "big-model", false)
	writeNotice(t, dir, SwapNotice{
		PID: peer.Process.Pid, Model: "big-model", Into: "small-model", IntoWindow: 8192,
	})

	got, ok := PendingSwap(dir)
	if !ok {
		t.Fatal("the announcement was not visible to a peer")
	}
	if got.Into != "small-model" || got.IntoWindow != 8192 {
		t.Errorf("notice = %+v, want the target model and its window", got)
	}
}

// Without the target window a peer cannot tell whether its transcript will
// still fit, which is half of whether it needs a summary at all.
func TestTheAnnouncementCarriesTheTargetWindow(t *testing.T) {
	dir := t.TempDir()
	// Registered first: an announcement whose author holds no lock reads as
	// one left behind by a dead process, which is the behaviour tested below.
	reg, err := Register(dir, Peer{Model: "big-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	a, err := AnnounceSwap(dir, "big-model", "small-model", 8192)
	if err != nil {
		t.Fatal(err)
	}
	defer a.End()

	// PendingSwap hides our own announcement; read it as another pid would.
	got, ok := PendingSwapAny(dir)
	if !ok {
		t.Fatal("no announcement was published")
	}
	if got.IntoWindow != 8192 {
		t.Errorf("IntoWindow = %d, want 8192", got.IntoWindow)
	}
}

func TestNobodyParksForTheirOwnAnnouncement(t *testing.T) {
	dir := t.TempDir()
	a, err := AnnounceSwap(dir, "big-model", "small-model", 8192)
	if err != nil {
		t.Fatal(err)
	}
	defer a.End()

	if _, ok := PendingSwap(dir); ok {
		t.Error("the announcing session was told to park for its own swap")
	}
}

// One crash mid-swap must not leave every other window parking forever
// against a model change that will never happen.
func TestAnAnnouncementFromADeadProcessIsIgnored(t *testing.T) {
	dir := t.TempDir()
	peer := startPeer(t, dir, "big-model", false)
	writeNotice(t, dir, SwapNotice{PID: peer.Process.Pid, Model: "big-model", Into: "small-model"})

	if _, ok := PendingSwap(dir); !ok {
		t.Fatal("setup: the announcement should be live")
	}
	_ = peer.Process.Kill()
	_, _ = peer.Process.Wait()

	if _, ok := PendingSwap(dir); ok {
		t.Error("an announcement outlived the process that made it")
	}
}

func TestEndWithdrawsTheAnnouncement(t *testing.T) {
	dir := t.TempDir()
	reg, err := Register(dir, Peer{Model: "big-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	a, err := AnnounceSwap(dir, "big-model", "small-model", 0)
	if err != nil {
		t.Fatal(err)
	}
	a.End()

	if _, ok := PendingSwapAny(dir); ok {
		t.Error("the announcement survived End")
	}
}

// A swap waits on exactly one thing: every session on the model having
// guaranteed it will stop calling for it.
func TestNotParkedIsWhatASwapWaitsFor(t *testing.T) {
	dir := t.TempDir()
	startPeerState(t, dir, "big-model", false, string(StateWorking))
	startPeerState(t, dir, "big-model", false, string(StateReady))
	startPeerState(t, dir, "other-model", false, string(StateWorking))

	waiting := NotReady(dir, "big-model")
	if len(waiting) != 1 {
		t.Fatalf("waiting on %d sessions, want 1", len(waiting))
	}
	if waiting[0].State != StateWorking {
		t.Errorf("waiting on a session in state %q", waiting[0].State)
	}
}

// A session with no claim on the model is not something a swap waits for,
// and is not listed during someone else's swap either.
func TestAFreeSessionIsNotWaitedFor(t *testing.T) {
	dir := t.TempDir()
	startPeerState(t, dir, "big-model", false, string(StateFree))

	if got := NotReady(dir, "big-model"); len(got) != 0 {
		t.Errorf("a free session was waited for: %+v", got)
	}
	if StateFree.Describe() != "" {
		t.Errorf("free renders as %q; it should never be shown", StateFree.Describe())
	}
}

// The words are the ones local-coordination.md §8 prescribes, and no others.
func TestStatesReadAsPlainEnglish(t *testing.T) {
	for state, want := range map[State]string{
		StateWorking:     "finishing its turn",
		StateSummarising: "writing a summary",
		StateReady:       "ready",
		StateFree:        "",
	} {
		if got := state.Describe(); got != want {
			t.Errorf("%q renders as %q, want %q", state, got, want)
		}
	}
}

func writeNotice(t *testing.T, dir string, n SwapNotice) {
	t.Helper()
	b, err := json.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, swapFile), b, 0o600); err != nil {
		t.Fatal(err)
	}
}

// PendingSwapAny reads the announcement without the self-exclusion, so a test
// can inspect one it published itself.
func PendingSwapAny(dir string) (SwapNotice, bool) {
	b, err := os.ReadFile(filepath.Join(dir, swapFile))
	if err != nil {
		return SwapNotice{}, false
	}
	var n SwapNotice
	if json.Unmarshal(b, &n) != nil {
		return SwapNotice{}, false
	}
	return n, true
}

// A session must never appear among the sessions it is waiting for, on any
// of the paths that list them.
func TestASessionIsNeverAmongItsOwnPeers(t *testing.T) {
	dir := t.TempDir()
	reg, err := Register(dir, Peer{Model: "big-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()
	reg.SetState(StateWorking)

	if got := OnModel(dir, "big-model"); len(got) != 0 {
		t.Errorf("OnModel listed this session: %+v", got)
	}
	if got := NotReady(dir, "big-model"); len(got) != 0 {
		t.Errorf("NotParked listed this session, so a swap would wait for itself: %+v", got)
	}

	a, err := AnnounceSwap(dir, "big-model", "small-model", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer a.End()
	if _, ok := PendingSwap(dir); ok {
		t.Error("the announcing session was asked to park for its own swap")
	}
}

// The announcement alone cannot close the race: a swap reads the session list
// while a session marks itself working, so both sides decide under the lock.
func TestDecideSerialisesConcurrentDeciders(t *testing.T) {
	dir := t.TempDir()

	var (
		mu       sync.Mutex
		inside   int
		overlaps int
		order    []int
	)
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = Decide(dir, func() {
				mu.Lock()
				inside++
				if inside > 1 {
					overlaps++
				}
				order = append(order, i)
				mu.Unlock()

				// Long enough that an unserialised section would overlap.
				time.Sleep(time.Millisecond)

				mu.Lock()
				inside--
				mu.Unlock()
			})
		}()
	}
	wg.Wait()

	if overlaps != 0 {
		t.Errorf("%d deciders ran at the same time; the section is not exclusive", overlaps)
	}
	if len(order) != 16 {
		t.Errorf("%d of 16 deciders ran", len(order))
	}
}

// Two processes, not two goroutines: flock is per open file description, so
// exclusion within one process proves nothing about the case that matters.
func TestDecideExcludesAnotherProcess(t *testing.T) {
	dir := t.TempDir()

	held := make(chan struct{})
	released := make(chan struct{})
	go func() {
		_ = Decide(dir, func() {
			close(held)
			<-released
		})
	}()
	<-held

	// A second process must not get in while the first holds it.
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "AI_CODE_TEST_DECIDE_DIR="+dir)
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _, _ = cmd.Process.Wait() })

	got := make(chan struct{})
	go func() {
		buf := make([]byte, 4)
		if _, err := out.Read(buf); err == nil {
			close(got)
		}
	}()

	select {
	case <-got:
		close(released)
		t.Fatal("another process entered the section while it was held")
	case <-time.After(150 * time.Millisecond):
	}

	close(released)
	select {
	case <-got:
	case <-time.After(5 * time.Second):
		t.Fatal("the section was never released to the other process")
	}
}

// A platform with no flock must not block: it has no exclusion to offer, and
// silently never running the caller's work would be worse than not having it.
func TestDecideStillRunsTheWorkWithoutLiveness(t *testing.T) {
	ran := false
	_ = Decide(t.TempDir(), func() { ran = true })
	if !ran {
		t.Error("Decide did not run its section")
	}
}

func TestASessionThatHasNeverRunATurnIsNotWaitedFor(t *testing.T) {
	dir := t.TempDir()
	startPeerState(t, dir, "big-model", false, "")

	peers := Peers(dir)
	if len(peers) != 1 {
		t.Fatalf("saw %d peers, want 1", len(peers))
	}
	if peers[0].State != StateFree {
		t.Errorf("a fresh registration published state %q, want %q", peers[0].State, StateFree)
	}
	if got := NotReady(dir, "big-model"); len(got) != 0 {
		t.Errorf("a swap would wait forever for a session that has done nothing: %+v", got)
	}
}

// Only the two states that mean "holding the model right now" are waited for.
func TestOnlyActiveStatesAreWaitedFor(t *testing.T) {
	for _, c := range []struct {
		state State
		wait  bool
	}{
		{StateWorking, true},
		{StateSummarising, true},
		{StateReady, false},
		{StateFree, false},
		{State(""), false},
		{State("something-added-later"), false},
	} {
		dir := t.TempDir()
		startPeerState(t, dir, "big-model", false, string(c.state))
		got := len(NotReady(dir, "big-model")) == 1
		if got != c.wait {
			t.Errorf("state %q: waited=%v, want %v", c.state, got, c.wait)
		}
	}
}

// The server stays the authority on what is loaded; this records the intended
// model for the other windows while a load is in flight.
func TestTheIntentIsReadableWithoutAskingAnyone(t *testing.T) {
	dir := t.TempDir()

	if _, ok := CurrentIntent(dir); ok {
		t.Fatal("a fresh directory claims the user has asked for something")
	}

	if err := SetIntent(dir, "big-model"); err != nil {
		t.Fatal(err)
	}
	got, ok := CurrentIntent(dir)
	if !ok || got.Model != "big-model" {
		t.Fatalf("LoadedModel() = %+v, %v", got, ok)
	}
	if got.PID != os.Getpid() {
		t.Errorf("record says pid %d asked, want %d", got.PID, os.Getpid())
	}

	// Asking for something else replaces it rather than accumulating.
	if err := SetIntent(dir, "other-model"); err != nil {
		t.Fatal(err)
	}
	if got, _ := CurrentIntent(dir); got.Model != "other-model" {
		t.Errorf("after a second request the record says %q", got.Model)
	}
}

// The windows are separate processes sharing a filesystem, so a record written
// by one must be readable by another.
func TestAnotherProcessSeesTheIntent(t *testing.T) {
	dir := t.TempDir()
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "AI_CODE_TEST_SETINTENT_DIR="+dir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("helper failed: %v: %s", err, out)
	}

	got, ok := CurrentIntent(dir)
	if !ok || got.Model != "written-elsewhere" {
		t.Errorf("LoadedModel() = %+v, %v; want the other process's record", got, ok)
	}
	if got.PID == os.Getpid() {
		t.Error("the record credits this process with a request it did not make")
	}
}

func TestAnAnnouncementHoldsEveryOtherSession(t *testing.T) {
	dir := t.TempDir()
	reg, err := Register(dir, Peer{Model: "old-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	peer := startPeer(t, dir, "old-model", false)
	writeNotice(t, dir, SwapNotice{
		PID: peer.Process.Pid, Model: "old-model", Into: "new-model", IntoWindow: 8192,
	})

	// On the outgoing model.
	if _, ok := PendingSwap(dir); !ok {
		t.Fatal("a session on the outgoing model saw no announcement")
	}

	// And having already adopted the incoming one, which is the case a
	// check against the current model would have missed.
	reg.SetModel("home", "new-model")
	if _, ok := PendingSwap(dir); !ok {
		t.Error("a session that already adopted the incoming model saw no announcement")
	}

	// Once the load is done the announcement goes and the prompt is free.
	if err := os.Remove(filepath.Join(dir, swapFile)); err != nil {
		t.Fatal(err)
	}
	if _, ok := PendingSwap(dir); ok {
		t.Error("the announcement outlived the swap")
	}
}

func TestAnEmptyDirectoryPathTouchesNothing(t *testing.T) {
	dir := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	defer func() { _ = os.Chdir(wd) }()

	if _, err := Register("", Peer{Model: "m"}); err == nil {
		t.Error("Register succeeded with no shared directory")
	}
	if got := Peers(""); got != nil {
		t.Errorf("Peers() = %+v", got)
	}
	if _, ok := PendingSwap(""); ok {
		t.Error("PendingSwap found an announcement with no shared directory")
	}
	if err := SetIntent("", "m"); err != nil {
		t.Errorf("SetIntent should be a silent no-op, got %v", err)
	}
	if _, ok := CurrentIntent(""); ok {
		t.Error("CurrentIntent found something with no shared directory")
	}
	if _, err := AnnounceSwap("", "a", "b", 0); err == nil {
		t.Error("AnnounceSwap succeeded with no shared directory")
	}
	ran := false
	_ = Decide("", func() { ran = true })
	if !ran {
		t.Error("Decide skipped its section instead of running it unserialised")
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("files written into the working directory: %v", names)
	}
}
