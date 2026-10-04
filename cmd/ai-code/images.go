package main

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"ai-code/internal/provider"
)

// checkImageModel refuses a session whose image model the provider does not
// list as one.
func checkImageModel(ctx context.Context, c provider.Client, providerName, id string) error {
	if _, ok := unwrapClient(c).(provider.ImageGenerator); !ok {
		return fmt.Errorf("--image-gen-model: provider %s cannot make pictures", providerName)
	}
	models, err := c.Models(ctx)
	if err != nil {
		return fmt.Errorf("--image-gen-model: listing the models of %s: %w", providerName, err)
	}
	var images []string
	for _, m := range models {
		if m.ID == id {
			if provider.IsImageModel(m) {
				return nil
			}
			return fmt.Errorf("--image-gen-model: %s on %s is not an image model; it is labelled %s",
				id, providerName, strings.Join(m.Labels, ", "))
		}
		if slices.Contains(m.Labels, "image") {
			images = append(images, m.ID)
		}
	}
	msg := fmt.Sprintf("--image-gen-model: %s has no model named %q", providerName, id)
	if len(images) > 0 {
		msg += ". Its image models: " + strings.Join(images, ", ")
	}
	return fmt.Errorf("%s", msg)
}

// imageGenerator is the current provider, behind the same .nocloud check a
// chat request passes.
func (a *App) imageGenerator(ctx context.Context) (provider.ImageGenerator, error) {
	if a.gate != nil {
		if err := a.gate.settle(ctx); err != nil {
			return nil, err
		}
		if a.client.Class() == provider.ClassCloud && a.gate.forbidsCloud() {
			return nil, a.gate.refusal()
		}
	}
	gen, ok := unwrapClient(a.client).(provider.ImageGenerator)
	if !ok {
		return nil, fmt.Errorf("provider %s cannot make pictures", a.providerName)
	}
	return gen, nil
}

func unwrapClient(c provider.Client) provider.Client {
	for {
		switch w := c.(type) {
		case gatedClient:
			c = w.Client
		case gatedIntrospector:
			c = w.Client
		case sharedClient:
			c = w.Client
		case sharedIntrospector:
			c = w.Client
		default:
			return c
		}
	}
}
