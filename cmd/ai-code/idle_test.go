package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-code/internal/config"
	"ai-code/internal/coord"
	"ai-code/internal/provider"
	"ai-code/internal/session"
)

// blockingClient holds a stream open until the context is cancelled, which is
// what a real summarisation looks like from the caller's side: a long call
// that only ends when the model finishes or someone gives up on it.
type blockingClient struct {
	countingClient
	entered chan struct{}
	once    sync.Once
}

func (c *blockingClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	c.streams++
	c.mu.Unlock()
	c.once.Do(func() { close(c.entered) })
	<-ctx.Done()
	return nil, ctx.Err()
}

// summaryClient answers a summarisation the way a model would, and keeps the
// requests so what actually went on the wire can be asserted on.
type summaryClient struct {
	countingClient
	text string
	reqs []provider.Request
}

func (c *summaryClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	c.streams++
	c.reqs = append(c.reqs, req)
	c.mu.Unlock()
	return &textStream{text: c.text}, nil
}

type textStream struct {
	text string
	sent bool
}

func (s *textStream) Recv() (provider.Event, error) {
	if s.sent {
		return provider.Event{}, io.EOF
	}
	s.sent = true
	return provider.Event{Kind: provider.EventText, Text: s.text}, nil
}
func (s *textStream) Message() provider.Message {
	return provider.Message{Role: provider.RoleAssistant, Content: s.text}
}
func (s *textStream) Usage() provider.Usage           { return provider.Usage{} }
func (s *textStream) StopReason() provider.StopReason { return provider.StopEnd }
func (s *textStream) Close() error                    { return nil }

// health is an Introspector whose answers the checkpoint has to respect.
type health struct {
	countingClient
	h      provider.Health
	models []provider.ModelInfo
}

func (c *health) Models(context.Context) ([]provider.ModelInfo, error) {
	c.mu.Lock()
	c.countingClient.models++
	c.mu.Unlock()
	return c.models, nil
}

func (c *health) Health(context.Context) (*provider.Health, error) { return &c.h, nil }
func (c *health) Load(context.Context, string) error               { return nil }
func (c *health) Unload(context.Context, string) error             { return nil }

// A keystroke cancels the context, and the checkpoint has to be gone by the
// time the prompt returns -- everything after it touches the same agent.
func TestIdleCheckpointAbortsOnCancellation(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &blockingClient{entered: make(chan struct{})}
	a.client = client
	a.agent.SetClient(client)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		a.watchWhileIdle(ctx, ctx, 0, nil)
	}()

	select {
	case <-client.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the checkpoint never reached the model")
	}

	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the checkpoint outlived its cancellation; the prompt would have waited on it")
	}

	if summary, _ := a.agent.Summary(); summary != "" {
		t.Errorf("an abandoned checkpoint was stored anyway: %q", summary)
	}
}

// Zero is not how the feature is turned off. It means now.
func TestZeroIdleCheckpointsImmediately(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &summaryClient{text: "## Goal\nship it"}
	a.client = client
	a.agent.SetClient(client)

	runIdle(t, a, 0)

	if summary, _ := a.agent.Summary(); !strings.Contains(summary, "ship it") {
		t.Errorf("a zero interval should checkpoint at once; got %q", summary)
	}
}

func TestIdleCheckpointWaitsOutTheInterval(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &summaryClient{text: "## Goal\nship it"}
	a.client = client
	a.agent.SetClient(client)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		a.watchWhileIdle(ctx, ctx, time.Hour, nil)
	}()
	cancel()
	<-done

	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("the model was called %d times during the idle wait, want none", streams)
	}
}

func TestIdleCheckpointSkipsAColdSession(t *testing.T) {
	a, _ := swapApp(t, 262144, 0)
	client := &summaryClient{text: "## Goal\nship it"}
	a.client = client
	a.agent.SetClient(client)

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("an empty session produced %d model calls, want none", streams)
	}
}

// The invariant is not "send nothing" -- it is "cause no load". A session
// that finds a different model resident adopts it, so anything it sends
// afterwards names the model that is already there and evicts nobody.
func TestIdleCheckpointNeverSendsForAModelThatIsNotLoaded(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &health{h: provider.Health{ModelLoaded: "someone-elses-model", Ready: true}}
	a.client = client
	a.agent.SetClient(client)

	runIdle(t, a, 0)

	if a.model.ID != "someone-elses-model" {
		t.Fatalf("session still wants %q; a request would load it back over the resident model",
			a.model.ID)
	}
	if got := a.agent.Model(); got != "someone-elses-model" {
		t.Errorf("the agent would send %q, not the resident model", got)
	}
}

func TestIdleCheckpointSkipsWhileTheModelIsBusy(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &health{h: provider.Health{ModelLoaded: "wide-model", Busy: true}}
	a.client = client
	a.agent.SetClient(client)

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("checkpointed while the model was busy (%d calls)", streams)
	}
}

func TestIdleCheckpointRunsWhenTheBackendIsFree(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &health{h: provider.Health{ModelLoaded: "wide-model", Ready: true}}
	a.client = client
	a.agent.SetClient(client)

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams == 0 {
		t.Error("a free backend holding our own model should have been checkpointed")
	}
}

func TestCheckpointIsPersistedOnceAndRestored(t *testing.T) {
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	a, _ := swapApp(t, 262144, 200000)
	client := &summaryClient{text: "## Goal\nship it"}
	a.client = client
	a.agent.SetClient(client)

	s, err := session.Create(session.Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	a.sess = s

	runIdle(t, a, 0)
	// A second pass with nothing new must not append the same text again.
	a.recordCheckpoint()
	s.Close()

	_, entries, err := session.Open("/p", s.Meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, e := range entries {
		if e.Type == session.EntryCheckpoint {
			n++
		}
	}
	if n != 1 {
		t.Errorf("wrote %d checkpoint entries, want exactly 1", n)
	}

	summary, through, _ := session.Checkpoint(entries)
	if !strings.Contains(summary, "ship it") {
		t.Errorf("restored summary = %q", summary)
	}
	if through <= 0 {
		t.Errorf("restored coverage = %d, want the message count the summary accounts for", through)
	}
}

// Turning the speculative checkpoint off must not stop the swap watch: they
// share a goroutine, and a window that stops answering announcements makes
// every other window's swap hang.
func TestTurningTheCheckpointOffLeavesTheSwapWatchRunning(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	d := config.Defaults()
	off := false
	d.Agent.AutoCheckpoint = &off
	a.cfg = &d
	client := &summaryClient{text: "## Goal\nship it"}
	a.client = client
	a.agent.SetClient(client)

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("auto_checkpoint = false still wrote a speculative summary (%d calls)", streams)
	}
}

// runIdle runs the idle loop until it acts, then stops it and waits for the
// goroutine to finish. The loop itself returns only when cancelled, because
// it goes on watching for another window's swap announcement long after the
// speculative checkpoint is done.
//
// Progress is watched through the client, not the agent. The agent is written
// by the loop's goroutine, and joining before the caller reads it is what
// makes that read safe.
func runIdle(t *testing.T, a *App, idle time.Duration) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		a.watchWhileIdle(ctx, ctx, idle, nil)
	}()

	if counter, ok := a.client.(interface{ calls() (int, int) }); ok {
		// Everything under this is an in-memory fake, so work that is going
		// to happen happens at once. The wait is for the scheduler.
		for range 200 {
			if streams, _ := counter.calls(); streams > 0 {
				break
			}
			time.Sleep(time.Millisecond)
		}
	}

	cancel()
	<-done
}

// The failure this guards is the one the whole subsystem exists to stop: a
// prompt typed before another window swapped, sent to a model that is no
// longer loaded, making the server load it back and evict theirs.
func TestASendAdoptsTheLoadedModelInsteadOfReloadingItsOwn(t *testing.T) {
	a, _ := swapApp(t, 262144, 0)
	client := &health{h: provider.Health{ModelLoaded: "someone-elses-model", Ready: true}}
	client.models = []provider.ModelInfo{{ID: "someone-elses-model", ContextWindow: 65536}}
	a.client = client
	a.agent.SetClient(client)

	if a.model.ID == "someone-elses-model" {
		t.Fatal("setup: the session should start on a different model")
	}
	captureOut(t, func() { a.adoptModelChange(context.Background(), nil) })

	if a.model.ID != "someone-elses-model" {
		t.Errorf("the session still wants %q; sending would evict the other window",
			a.model.ID)
	}
	if got := a.agent.Model(); got != "someone-elses-model" {
		t.Errorf("the agent would still send %q", got)
	}
}

// One invariant covers every case: the line shows the difference between
// what the scrollback says and what is loaded. These are the cases that got
// it wrong one at a time.
func TestModelChangeIsReportedAgainstWhatTheScrollbackSays(t *testing.T) {
	newApp := func(t *testing.T, known string, loaded string) (*App, *health) {
		t.Helper()
		a, _ := swapApp(t, 262144, 0)
		a.model = provider.ModelInfo{ID: known, ContextWindow: 262144}
		a.knownModel = known
		c := &health{h: provider.Health{ModelLoaded: loaded, Ready: true}}
		c.models = []provider.ModelInfo{
			{ID: known, ContextWindow: 262144},
			{ID: loaded, ContextWindow: 65536},
		}
		a.client = c
		a.agent.SetClient(c)
		return a, c
	}

	t.Run("a session that has sent nothing still reports, because its banner claimed a model", func(t *testing.T) {
		a, _ := newApp(t, "banner-model", "other-model")
		out := captureOut(t, func() { a.adoptModelChange(context.Background(), nil) })
		if !strings.Contains(out, "other-model") {
			t.Errorf("no report on a session that had only shown a banner:\n%s", out)
		}
	})

	t.Run("no difference, no report", func(t *testing.T) {
		a, _ := newApp(t, "same-model", "same-model")
		out := captureOut(t, func() { a.adoptModelChange(context.Background(), nil) })
		if n := len(nonBlank(out)); n != 0 {
			t.Errorf("reported %d lines with nothing changed:\n%s", n, out)
		}
	})

	// At a live prompt the line is a replaceable header, so nothing is
	// committed and the scrollback still names the model it started on.
	// Coming back to it is therefore no difference at all.
	t.Run("away and back at a prompt leaves nothing to say", func(t *testing.T) {
		a, _ := newApp(t, "home-model", "other-model")

		a.model = provider.ModelInfo{ID: "other-model"}
		if a.modelChangeLine() == "" {
			t.Error("no line while on a model the scrollback does not name")
		}

		a.model = provider.ModelInfo{ID: "home-model"}
		if got := a.modelChangeLine(); got != "" {
			t.Errorf("returning to the model the scrollback names gave %q", got)
		}
	})

	// Once a line has been committed it is what the scrollback says, so a
	// change back from there is a real difference and is reported.
	t.Run("a committed line becomes what the scrollback says", func(t *testing.T) {
		a, c := newApp(t, "home-model", "other-model")
		captureOut(t, func() { a.adoptModelChange(context.Background(), nil) })
		if a.knownModel != "other-model" {
			t.Fatalf("after committing, the scrollback says %q", a.knownModel)
		}

		c.h.ModelLoaded = "home-model"
		back := captureOut(t, func() { a.adoptModelChange(context.Background(), nil) })
		if !strings.Contains(back, "was other-model") {
			t.Errorf("a change back was not measured against the committed line:\n%s", back)
		}
	})

	t.Run("one line per difference, not per change", func(t *testing.T) {
		a, c := newApp(t, "home-model", "a-model")
		out := captureOut(t, func() {
			a.adoptModelChange(context.Background(), nil)
			c.h.ModelLoaded = "b-model"
			c.models = append(c.models, provider.ModelInfo{ID: "b-model", ContextWindow: 65536})
			a.adoptModelChange(context.Background(), nil)
		})
		// Without a prompt to sit above, each difference commits -- but each
		// is measured against the last thing said, never against the start.
		if strings.Count(out, "was home-model") > 1 {
			t.Errorf("a later change was measured against the original model:\n%s", out)
		}
	})
}

// The proactive path: a session at an idle prompt picks the change up from
// its own watch loop, before anything is typed and before anything is sent.
//
// If this ever stops working the pre-send guard would silently cover for it,
// and the harness would be back to noticing swaps only once it had already
// sent something -- so this is asserted separately from that guard.
func TestTheIdleWatchAdoptsWithoutWaitingForASend(t *testing.T) {
	a, _ := swapApp(t, 262144, 0)
	client := &health{h: provider.Health{ModelLoaded: "big-model", Ready: true}}
	client.models = []provider.ModelInfo{{ID: "big-model", ContextWindow: 262144}}
	a.client = client
	a.agent.SetClient(client)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		a.watchWhileIdle(ctx, ctx, time.Hour, nil)
	}()
	for range 200 {
		if _, models := client.calls(); models > 0 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	<-done

	if a.model.ID != "big-model" {
		t.Errorf("the idle watch left the session on %q; nothing was typed or sent",
			a.model.ID)
	}
	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("adopting took %d model calls, want none", streams)
	}
}

// deadClient fails every call. It stands in for the server being busy
// loading a large model, which is precisely when the old health-and-
// catalogue questions timed out and a session carried on naming a model
// that had gone.
type deadClient struct{ countingClient }

func (c *deadClient) Health(context.Context) (*provider.Health, error) {
	return nil, errors.New("timeout")
}
func (c *deadClient) Load(context.Context, string) error   { return errors.New("timeout") }
func (c *deadClient) Unload(context.Context, string) error { return errors.New("timeout") }
func (c *deadClient) Models(context.Context) ([]provider.ModelInfo, error) {
	c.mu.Lock()
	c.countingClient.models++
	c.mu.Unlock()
	return nil, errors.New("timeout")
}

// The point of sharing intent: on one machine, with one user, a window
// learns which model to use from a local file and never has to ask. A
// request naming the incoming model is queued behind the load; one naming
// the outgoing model drags it back.
func TestASessionAdoptsWithTheServerCompletelyUnreachable(t *testing.T) {
	dir := t.TempDir()
	a, _ := swapApp(t, 262144, 0)
	a.coordDir = dir
	a.knownModel = "old-model"
	a.model = provider.ModelInfo{ID: "old-model", ContextWindow: 262144}
	a.client = &deadClient{}
	a.agent.SetClient(a.client)
	a.agent.SetModel("old-model", 262144, 0)

	// The user asked for something else, in another window.
	if err := coord.SetIntent(dir, "new-model"); err != nil {
		t.Fatal(err)
	}

	out := captureOut(t, func() {
		if !a.adoptIntended(nil) {
			t.Error("the session did not follow the user's stated intent")
		}
	})

	if a.model.ID != "new-model" {
		t.Errorf("session still wants %q; its next request would load it back", a.model.ID)
	}
	if got := a.agent.Model(); got != "new-model" {
		t.Errorf("the agent would send %q", got)
	}
	if !strings.Contains(out, "new-model") {
		t.Errorf("the change was not reported:\n%s", out)
	}
	// The window is unknown with no catalogue, so the one in force stands
	// rather than collapsing to a default.
	if got := a.agent.ContextState().Window; got != 262144 {
		t.Errorf("context limit = %d, want the previous window kept", got)
	}
}

// With nothing asked for, it leaves the session alone rather than guessing.
func TestNoRecordedIntentMeansNoChange(t *testing.T) {
	a, _ := swapApp(t, 262144, 0)
	a.coordDir = t.TempDir()
	before := a.model.ID

	if a.adoptIntended(nil) {
		t.Error("adopted something with nothing asked for")
	}
	if a.model.ID != before {
		t.Errorf("session moved to %q with nothing asked for", a.model.ID)
	}
}
