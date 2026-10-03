package main

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-code/internal/coord"
	"ai-code/internal/provider"
)

const testServer = "http://ai.example.internal:8000/v1"

var testPID = 910_000_000

// sharedApps puts a on a bus as the main loop would, and returns a second,
// bare instance on the same server for the test to drive.
func sharedApps(t *testing.T, a *App) *coord.Bus {
	t.Helper()
	return sharedAppsAnswering(t, a, true)
}

// sharedAppsAnswering leaves out the goroutine that answers swaps when answer
// is false, so a test can read what the window would be told.
func sharedAppsAnswering(t *testing.T, a *App, answer bool) *coord.Bus {
	t.Helper()
	base, err := os.MkdirTemp("", "co")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(base) })
	dir := filepath.Join(base, "peers")

	testPID++
	bus, err := coord.Join(dir, coord.Peer{PID: testPID, Cwd: "/work/a", Server: testServer, Model: a.model.ID})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bus.Close)
	a.share = newModelShare(bus, time.Now().UnixNano())
	a.client = a.share.wrap(a.client)
	a.agent.SetClient(a.client)
	a.agent.SetTurnBoundary(a.atBoundary)

	if answer {
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			defer close(done)
			a.answerSwaps(ctx)
		}()
		t.Cleanup(func() { cancel(); <-done })
	}

	testPID++
	other, err := coord.Join(dir, coord.Peer{PID: testPID, Cwd: "/work/b", Server: testServer, Model: "other-model"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(other.Close)
	until(t, other, "the instances to meet", func() bool { return len(other.Peers()) == 1 })
	until(t, bus, "the instances to meet", func() bool { return len(bus.Peers()) == 1 })
	return other
}

// until waits for cond, woken only by the bus's change notifications.
func until(t *testing.T, b *coord.Bus, what string, cond func() bool) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		changed := b.Changed()
		if cond() {
			return
		}
		select {
		case <-changed:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func request(b *coord.Bus, into string, window int) chan error {
	done := make(chan error, 1)
	go func() { done <- b.Request(context.Background(), testServer, into, window, nil) }()
	return done
}

func answered(t *testing.T, done chan error, what string) {
	t.Helper()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("%s was never answered", what)
	}
}

func unanswered(t *testing.T, done chan error, what string) {
	t.Helper()
	select {
	case err := <-done:
		t.Fatalf("%s was answered early (err %v)", what, err)
	case <-time.After(50 * time.Millisecond):
	}
}

// scriptClient answers request i with a tool call while i < calls, then with
// text. A request waits on hold[i] when there is one. It counts health probes,
// which taking turns must never need.
type scriptClient struct {
	countingClient
	calls  int
	hold   map[int]chan struct{}
	health int
}

func (c *scriptClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	i := c.streams
	c.streams++
	gate := c.hold[i]
	c.mu.Unlock()
	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if i < c.calls {
		return &callStream{id: "t" + string(rune('a'+i))}, nil
	}
	return &textStream{text: "## Goal\nship it"}, nil
}

func (c *scriptClient) Health(context.Context) (*provider.Health, error) {
	c.mu.Lock()
	c.health++
	c.mu.Unlock()
	return &provider.Health{}, nil
}
func (c *scriptClient) Load(context.Context, string) error   { return nil }
func (c *scriptClient) Unload(context.Context, string) error { return nil }

func (c *scriptClient) count() (streams, health int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.streams, c.health
}

type callStream struct{ id string }

func (s *callStream) Recv() (provider.Event, error) { return provider.Event{}, io.EOF }
func (s *callStream) Message() provider.Message {
	return provider.Message{Role: provider.RoleAssistant,
		ToolCalls: []provider.ToolCall{{ID: s.id, Name: "read", Args: "{}"}}}
}
func (s *callStream) Usage() provider.Usage           { return provider.Usage{} }
func (s *callStream) StopReason() provider.StopReason { return provider.StopToolCalls }
func (s *callStream) Close() error                    { return nil }

func scriptedApp(t *testing.T, c *scriptClient, sessionChars int) *App {
	t.Helper()
	a, _ := swapApp(t, 262144, sessionChars)
	a.client = c
	a.agent.SetClient(c)
	return a
}

func TestAnIdleInstanceMakesWayAtOnce(t *testing.T) {
	c := &scriptClient{}
	a := scriptedApp(t, c, 0)
	other := sharedApps(t, a)

	answered(t, request(other, "x", 8192), "a swap asked of an idle instance")
	if streams, health := c.count(); streams != 0 || health != 0 {
		t.Errorf("making way cost %d requests and %d health probes, want none", streams, health)
	}
}

// The user's scenario: one instance needs another model; a long run elsewhere
// parks at the end of its turn, stays parked while the other model works, and
// carries on by itself when its own model is loaded again.
func TestARunParksAtTheEndOfItsTurnAndCarriesOnWhenItsModelReturns(t *testing.T) {
	first := make(chan struct{})
	c := &scriptClient{calls: 2, hold: map[int]chan struct{}{0: first}}
	a := scriptedApp(t, c, 0)
	other := sharedApps(t, a)
	other.Done(testServer, "wide-model")
	until(t, a.share.bus, "the load to be heard", func() bool { return a.share.bus.Loaded() == "wide-model" })

	run := make(chan error, 1)
	go func() { run <- a.runTurn(context.Background(), "go") }()
	until(t, other, "the run to start", func() bool {
		p := other.Peers()
		return len(p) == 1 && p[0].State == coord.StateWorking
	})
	for {
		if s, _ := c.count(); s == 1 {
			break
		}
		time.Sleep(time.Millisecond)
	}

	// Wide enough for this conversation, so parking needs no summary.
	swap := request(other, "x", 262144)
	unanswered(t, swap, "a swap during a request")

	close(first)
	answered(t, swap, "a swap once the turn ended")
	if s, _ := c.count(); s != 1 {
		t.Fatalf("the run sent %d requests, want 1: it should have parked after its first turn", s)
	}
	until(t, other, "the run to say it parked", func() bool { return other.Peers()[0].State == coord.StateParked })
	other.Done(testServer, "x")

	// Another model is loaded, so the run stays parked.
	select {
	case err := <-run:
		t.Fatalf("the parked run ended (err %v)", err)
	case <-time.After(50 * time.Millisecond):
	}
	if s, _ := c.count(); s != 1 {
		t.Fatalf("a parked run sent a request for a model that is not loaded")
	}

	answered(t, request(other, "wide-model", 0), "the swap back")
	other.Done(testServer, "wide-model")

	select {
	case err := <-run:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the run never carried on")
	}
	if streams, health := c.count(); streams != 3 || health != 0 {
		t.Errorf("the run made %d requests and %d health probes, want 3 and none", streams, health)
	}
}

func TestASendAsksForItsModelWhenAnotherIsLoaded(t *testing.T) {
	c := &scriptClient{}
	a := scriptedApp(t, c, 0)
	other := sharedApps(t, a)
	other.Done(testServer, "other-model")
	until(t, a.share.bus, "the load to be heard", func() bool { return a.share.bus.Loaded() == "other-model" })

	other.Set(func(p *coord.Peer) { p.State = coord.StateWorking })
	run := make(chan error, 1)
	go func() { run <- a.runTurn(context.Background(), "go") }()

	until(t, other, "the request for wide-model", func() bool {
		p := other.Pending()
		return len(p) == 1 && p[0].Into == "wide-model"
	})
	if s, _ := c.count(); s != 0 {
		t.Fatal("the send went out before the other instance made way")
	}
	other.Ready(other.Pending()[0].ID)

	select {
	case err := <-run:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the send never went out")
	}
	until(t, other, "the load to be announced", func() bool { return other.Loaded() == "wide-model" })
}

// A request outside the main run, such as a background worker's, waits while
// another model is loaded rather than loading its own back.
func TestAWorkerRequestWaitsForItsModel(t *testing.T) {
	c := &scriptClient{}
	a := scriptedApp(t, c, 0)
	other := sharedApps(t, a)
	other.Done(testServer, "other-model")
	until(t, a.share.bus, "the load to be heard", func() bool { return a.share.bus.Loaded() == "other-model" })

	sent := make(chan struct{})
	go func() {
		st, err := a.client.Stream(context.Background(), provider.Request{Model: "wide-model"})
		if err == nil {
			st.Close()
		}
		close(sent)
	}()
	select {
	case <-sent:
		t.Fatal("a worker's request went out for a model that is not loaded")
	case <-time.After(50 * time.Millisecond):
	}
	other.Done(testServer, "wide-model")
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("the worker's request never went out")
	}
}

// A swap to a model too small for this conversation is answered only after
// the summary is written, while the model that can write it is still loaded.
func TestASwapWaitsForTheSummaryAStrandedSessionNeeds(t *testing.T) {
	gate := make(chan struct{})
	c := &scriptClient{hold: map[int]chan struct{}{0: gate}}
	a := scriptedApp(t, c, 200000)
	other := sharedApps(t, a)

	swap := request(other, "narrow-model", 1024)
	until(t, other, "the summary to start", func() bool {
		p := other.Peers()
		return len(p) == 1 && p[0].State == coord.StateSummarising
	})
	unanswered(t, swap, "a swap while its summary was being written")
	close(gate)
	answered(t, swap, "a swap once its summary was written")

	if summary, _ := a.agent.Summary(); summary == "" {
		t.Error("no summary was kept")
	}
}

func TestAModelSwitchWaitsForAnotherInstanceToMakeWay(t *testing.T) {
	c := &scriptClient{}
	a := scriptedApp(t, c, 0)
	other := sharedApps(t, a)
	other.Set(func(p *coord.Peer) { p.Model = "wide-model" })

	var wg sync.WaitGroup
	wg.Add(1)
	switched := make(chan struct{})
	go func() {
		defer wg.Done()
		a.swapTo(context.Background(), a.client, testServer, provider.ModelInfo{ID: "x"}, 0)
		close(switched)
	}()
	until(t, other, "the swap to arrive", func() bool { return len(other.Pending()) == 1 })
	select {
	case <-switched:
		t.Fatal("the switch went ahead before the other instance made way")
	case <-time.After(50 * time.Millisecond):
	}
	other.Ready(other.Pending()[0].ID)
	wg.Wait()
	until(t, other, "the load to be announced", func() bool { return other.Loaded() == "x" })
}

func TestAnUnsharedProviderNeverWaits(t *testing.T) {
	c := &scriptClient{}
	a := scriptedApp(t, c, 0)
	other := sharedApps(t, a)
	a.share.bus.Set(func(p *coord.Peer) { p.Server = "" })
	other.Done(testServer, "other-model")
	other.Set(func(p *coord.Peer) { p.State = coord.StateWorking })

	if err := a.runTurn(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if s, _ := c.count(); s != 1 {
		t.Errorf("sent %d requests, want 1", s)
	}
}

// modelClient records the model each request was sent to.
type modelClient struct {
	countingClient
	sent []string
}

func (c *modelClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	c.streams++
	c.sent = append(c.sent, req.Model)
	c.mu.Unlock()
	return &textStream{text: "hello"}, nil
}

func (c *modelClient) Models(context.Context) ([]provider.ModelInfo, error) {
	return []provider.ModelInfo{{ID: "wide-model", ContextWindow: 262144}, {ID: "other-model", ContextWindow: 262144}}, nil
}

func (c *modelClient) models() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.sent...)
}

// The reported churn: an idle window whose model another window swapped out
// must not load it back with its next prompt. It is told, and the prompt goes
// to the model now loaded.
func TestAnIdleWindowsNextPromptGoesToTheModelAnotherWindowLoaded(t *testing.T) {
	a, _ := swapApp(t, 262144, 0)
	c := &modelClient{}
	a.client = c
	a.agent.SetClient(c)
	other := sharedAppsAnswering(t, a, false)

	other.Done(testServer, "wide-model")
	until(t, a.share.bus, "wide-model to be loaded", func() bool { return a.share.bus.Loaded() == "wide-model" })
	a.noticeChanges()

	swap := request(other, "other-model", 262144)
	until(t, a.share.bus, "the swap to arrive", func() bool { return len(a.share.bus.Pending()) == 1 })
	if told := a.noticeChanges(); len(told) != 1 || !strings.Contains(told[0], "is switching the server to other-model") {
		t.Errorf("when the swap began the idle window was told %q", told)
	}
	a.answerPending(context.Background())
	answered(t, swap, "the other window's swap")
	other.Done(testServer, "other-model")
	until(t, a.share.bus, "the swap to finish", func() bool { return a.share.bus.Loaded() == "other-model" })

	told := a.noticeChanges()
	if len(told) == 0 || !strings.Contains(strings.Join(told, "\n"), "/model wide-model") {
		t.Errorf("the idle window was told %q; it should say how to switch back", told)
	}

	// Bounded: the old behaviour asks for wide-model back, which nothing answers.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out := captureErr(t, func() {
		if err := a.handleInput(ctx, "hello"); err != nil {
			t.Error(err)
		}
	})
	if ctx.Err() != nil {
		t.Fatal("the idle window asked for its old model back")
	}
	if !strings.Contains(out, "this prompt goes to other-model") {
		t.Errorf("the send did not say where it went:\n%s", out)
	}
	if got := c.models(); len(got) != 1 || got[0] != "other-model" {
		t.Errorf("the prompt was sent to %v, want other-model", got)
	}
	if len(other.Pending()) != 0 {
		t.Error("the idle window asked for its old model back")
	}
	if a.model.ID != "other-model" || a.share.bus.Self().Model != "other-model" {
		t.Errorf("the window is on %q (published %q)", a.model.ID, a.share.bus.Self().Model)
	}
}

// A model loaded before this window existed was never taken from it: a send
// asks for the window's own model.
func TestALoadBeforeJoiningIsNotADisplacement(t *testing.T) {
	a, _ := swapApp(t, 262144, 0)
	c := &modelClient{}
	a.client = c
	a.agent.SetClient(c)
	other := sharedApps(t, a)
	// As if this window joined after the load.
	a.share.mu.Lock()
	a.share.joined = time.Now().Add(time.Hour).UnixNano()
	a.share.mu.Unlock()
	other.Done(testServer, "other-model")
	until(t, a.share.bus, "the load to be heard", func() bool { return a.share.bus.Loaded() == "other-model" })

	go func() {
		until(t, other, "the request for wide-model", func() bool { return len(other.Pending()) == 1 })
		other.Ready(other.Pending()[0].ID)
	}()
	if err := a.handleInput(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	if got := c.models(); len(got) != 1 || got[0] != "wide-model" {
		t.Errorf("the prompt was sent to %v, want wide-model", got)
	}
}

// A summary on a model the last round did not use prefills the whole
// conversation into a slot.
func TestNoSummaryIsSentToAModelTheLastRoundDidNotUse(t *testing.T) {
	a, client := swapApp(t, 262144, 200000)
	a.lastRoundModel = "narrow-model"

	runIdle(t, a, 0)
	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("the idle summary went to a cold model (%d calls)", streams)
	}

	sharedApps(t, a)
	if a.needsSummaryFor(coord.Swap{Into: "x", IntoWindow: 1024}) {
		t.Error("a swap asked for a summary on a cold model")
	}
}

func captureErr(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stderr
	os.Stderr = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	f()
	w.Close()
	os.Stderr = saved
	return <-done
}
