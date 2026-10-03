package main

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"ai-code/internal/agent"
	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

type sentCounter struct {
	class    provider.Class
	streams  atomic.Int32
	listings atomic.Int32
}

func (c *sentCounter) Name() string          { return "prov" }
func (c *sentCounter) Class() provider.Class { return c.class }
func (c *sentCounter) Models(context.Context) ([]provider.ModelInfo, error) {
	c.listings.Add(1)
	return nil, nil
}
func (c *sentCounter) Stream(context.Context, provider.Request) (provider.Stream, error) {
	c.streams.Add(1)
	return nil, errors.New("sent")
}

type introspectingClient struct{ sentCounter }

func (*introspectingClient) Health(context.Context) (*provider.Health, error) { return nil, nil }
func (*introspectingClient) Load(context.Context, string) error               { return nil }
func (*introspectingClient) Unload(context.Context, string) error             { return nil }

func TestARemoteNoCloudStopsTheFirstRequestToACloudProvider(t *testing.T) {
	var found []string
	checks := 0
	gate := &cloudGate{
		remote: true,
		check:  func(context.Context) (string, error) { checks++; return "/home/u/project/.nocloud", nil },
		found:  func(p string) { found = append(found, p) },
	}
	cloud := &sentCounter{class: provider.ClassCloud}
	c := gate.wrap(cloud)

	for range 2 {
		_, err := c.Stream(context.Background(), provider.Request{})
		if err == nil || !strings.Contains(err.Error(), "nothing was sent") ||
			!strings.Contains(err.Error(), "/home/u/project/.nocloud") {
			t.Errorf("Stream = %v, want a refusal naming the marker", err)
		}
	}
	if n := cloud.streams.Load(); n != 0 {
		t.Fatalf("%d requests reached the cloud provider", n)
	}
	if checks != 1 || len(found) != 1 {
		t.Errorf("checked %d times and announced %v; want one of each", checks, found)
	}
	if !gate.forbidsCloud() {
		t.Error("the session was not restricted to on-premises providers")
	}

	onprem := &sentCounter{class: provider.ClassOnPrem}
	_, _ = gate.wrap(onprem).Stream(context.Background(), provider.Request{})
	if onprem.streams.Load() != 1 {
		t.Error("an on-premises request was held back by the marker")
	}
}

func TestAnUncheckedRemoteSendsNothing(t *testing.T) {
	fail := true
	gate := &cloudGate{remote: true, check: func(context.Context) (string, error) {
		if fail {
			return "", errors.New("connection refused")
		}
		return "", nil
	}}
	for _, class := range []provider.Class{provider.ClassCloud, provider.ClassOnPrem} {
		inner := &sentCounter{class: class}
		_, err := gate.wrap(inner).Stream(context.Background(), provider.Request{})
		var u *tool.Unavailable
		if !errors.As(err, &u) || inner.streams.Load() != 0 {
			t.Errorf("%s: Stream = %v after %d sends, want tool.Unavailable and nothing sent", class, err, inner.streams.Load())
		}
	}
	fail = false
	cloud := &sentCounter{class: provider.ClassCloud}
	_, _ = gate.wrap(cloud).Stream(context.Background(), provider.Request{})
	if cloud.streams.Load() != 1 {
		t.Error("a check that later succeeded with no marker still held the request back")
	}
}

func TestConcurrentRequestsCheckOnce(t *testing.T) {
	var checks atomic.Int32
	gate := &cloudGate{remote: true, check: func(context.Context) (string, error) { checks.Add(1); return "", nil }}
	c := gate.wrap(&sentCounter{class: provider.ClassOnPrem})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = c.Stream(context.Background(), provider.Request{}) }()
	}
	wg.Wait()
	if n := checks.Load(); n != 1 {
		t.Errorf("checked %d times", n)
	}
}

func TestTheGateKeepsResidencyControls(t *testing.T) {
	c := (&cloudGate{}).wrap(&introspectingClient{sentCounter{class: provider.ClassOnPrem}})
	if _, ok := c.(provider.Introspector); !ok {
		t.Error("a backend's residency controls were hidden by the gate")
	}
}

func TestNoCloudHoldsAtTheRequest(t *testing.T) {
	cloud := &sentCounter{class: provider.ClassCloud}
	if _, err := (&cloudGate{static: true}).wrap(cloud).Stream(context.Background(), provider.Request{}); err == nil ||
		cloud.streams.Load() != 0 {
		t.Errorf("Stream = %v, %d sent", err, cloud.streams.Load())
	}
}

func TestTheRefusalEndsTheRun(t *testing.T) {
	gate := &cloudGate{remote: true, check: func(context.Context) (string, error) { return "", errors.New("no route") }}
	cloud := &sentCounter{class: provider.ClassCloud}
	a := agent.New(gate.wrap(cloud), "m", noTools{}, agent.SinkFunc(func(agent.Event) {}),
		agent.Options{ContextLimit: 65536})
	err := a.Run(context.Background(), "summarise the secret source")
	var u *tool.Unavailable
	if !errors.As(err, &u) {
		t.Fatalf("Run = %v, want tool.Unavailable", err)
	}
	if cloud.streams.Load() != 0 {
		t.Error("the prompt reached the cloud provider")
	}
}

func TestACloudCatalogueWaitsForTheCheck(t *testing.T) {
	checks := 0
	marker := ""
	gate := &cloudGate{remote: true, check: func(context.Context) (string, error) { checks++; return marker, nil }}
	cloud := &sentCounter{class: provider.ClassCloud}
	c := gate.wrap(cloud)

	if _, err := c.Models(context.Background()); err == nil || cloud.listings.Load() != 0 || checks != 0 {
		t.Fatalf("before the check: err=%v listings=%d checks=%d", err, cloud.listings.Load(), checks)
	}
	onprem := &sentCounter{class: provider.ClassOnPrem}
	if _, err := gate.wrap(onprem).Models(context.Background()); err != nil || onprem.listings.Load() != 1 {
		t.Errorf("an on-premises catalogue was held back: %v", err)
	}

	_ = gate.settle(context.Background())
	if _, err := c.Models(context.Background()); err != nil || cloud.listings.Load() != 1 {
		t.Errorf("after a clean check: err=%v listings=%d", err, cloud.listings.Load())
	}

	marked := &cloudGate{remote: true, check: func(context.Context) (string, error) { return "/p/.nocloud", nil }}
	_ = marked.settle(context.Background())
	cloud2 := &sentCounter{class: provider.ClassCloud}
	if _, err := marked.wrap(cloud2).Models(context.Background()); err == nil || cloud2.listings.Load() != 0 {
		t.Errorf("a marked tree's cloud catalogue was fetched")
	}
}
