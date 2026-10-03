package main

import (
	"context"
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

// blockingClient holds a stream open until the context is cancelled.
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

// summaryClient answers a summarisation and records the requests.
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

// The checkpoint must be gone by the time the prompt returns; everything
// after it touches the same agent.
func TestIdleCheckpointAbortsOnCancellation(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	client := &blockingClient{entered: make(chan struct{})}
	a.client = client
	a.agent.SetClient(client)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		a.idleCheckpoint(ctx, 0)
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
		a.idleCheckpoint(ctx, time.Hour)
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

// A summary sent while another instance's model is loaded would load this
// one's back over it.
func TestIdleCheckpointNeverSendsForAModelThatIsNotLoaded(t *testing.T) {
	a, client := swapApp(t, 262144, 200000)
	other := sharedApps(t, a)
	other.Done(testServer, "someone-elses-model")
	until(t, a.share.bus, "the load to be heard", func() bool { return a.share.bus.Loaded() != "" })

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("checkpointed over another instance's model (%d calls)", streams)
	}
	if a.model.ID != "wide-model" {
		t.Errorf("the session moved itself onto %q", a.model.ID)
	}
}

func TestIdleCheckpointSkipsWhileAnotherInstanceIsWorking(t *testing.T) {
	a, client := swapApp(t, 262144, 200000)
	other := sharedApps(t, a)
	other.Set(func(p *coord.Peer) { p.State = coord.StateWorking })
	until(t, a.share.bus, "the other instance's state", func() bool {
		peers := a.share.bus.Peers()
		return len(peers) == 1 && peers[0].State == coord.StateWorking
	})

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("checkpointed while another instance was using the model (%d calls)", streams)
	}
}

func TestIdleCheckpointRunsWhenTheBackendIsFree(t *testing.T) {
	a, client := swapApp(t, 262144, 200000)
	sharedApps(t, a)

	runIdle(t, a, 0)

	if streams, _ := client.calls(); streams == 0 {
		t.Error("an idle server should have been used for the checkpoint")
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

func TestTurningTheCheckpointOffSendsNothing(t *testing.T) {
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

// runIdle runs the idle checkpoint to completion.
func runIdle(t *testing.T, a *App, idle time.Duration) {
	t.Helper()
	a.idleCheckpoint(context.Background(), idle)
}
