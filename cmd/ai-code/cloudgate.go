package main

import (
	"context"
	"fmt"
	"sync"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// cloudGate holds every model request until the remote working directory has
// been checked for .nocloud, and sends nothing if it cannot be checked.
type cloudGate struct {
	static bool
	remote bool
	check  func(context.Context) (string, error)
	found  func(path string)

	mu      sync.Mutex
	checked bool
	marker  string
}

func (g *cloudGate) settle(ctx context.Context) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.checked || !g.remote {
		return nil
	}
	if g.check == nil {
		return &tool.Unavailable{Err: fmt.Errorf("nothing was sent: the remote working directory cannot be checked for .nocloud")}
	}
	marker, err := g.check(ctx)
	if err != nil {
		return &tool.Unavailable{Err: fmt.Errorf(
			"nothing was sent: whether the remote working directory forbids cloud providers could not be checked: %w", err)}
	}
	g.checked, g.marker = true, marker
	if marker != "" && g.found != nil {
		g.found(marker)
	}
	return nil
}

func (g *cloudGate) pending() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.remote && !g.checked
}

func (g *cloudGate) forbidsCloud() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.static || g.marker != ""
}

func (g *cloudGate) refusal() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.marker != "" {
		return fmt.Errorf("nothing was sent: %s marks the remote working directory, so this session "+
			"is restricted to on-premises providers. Choose one with /provider", g.marker)
	}
	return fmt.Errorf("nothing was sent: %v", errCloudForbiddenHere)
}

func (g *cloudGate) wrap(c provider.Client) provider.Client {
	gc := gatedClient{Client: c, gate: g}
	if in, ok := c.(provider.Introspector); ok {
		return gatedIntrospector{gatedClient: gc, Introspector: in}
	}
	return gc
}

type gatedClient struct {
	provider.Client
	gate *cloudGate
}

func (c gatedClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	if err := c.gate.settle(ctx); err != nil {
		return nil, err
	}
	if c.Class() == provider.ClassCloud && c.gate.forbidsCloud() {
		return nil, c.gate.refusal()
	}
	return c.Client.Stream(ctx, req)
}

// A cloud catalogue waits for the check but never triggers it: listing
// happens at startup, which must not wait on ssh.
func (c gatedClient) Models(ctx context.Context) ([]provider.ModelInfo, error) {
	if c.Class() == provider.ClassCloud {
		if c.gate.forbidsCloud() {
			return nil, c.gate.refusal()
		}
		if c.gate.pending() {
			return nil, fmt.Errorf("%s was not asked for its models: the remote working directory "+
				"has not yet been checked for .nocloud, which happens when the first request is sent", c.Name())
		}
	}
	return c.Client.Models(ctx)
}

type gatedIntrospector struct {
	gatedClient
	provider.Introspector
}
