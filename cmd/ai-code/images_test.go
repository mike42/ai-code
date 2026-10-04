package main

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
)

type catalogueClient struct {
	countingClient
	models []provider.ModelInfo
}

func (c *catalogueClient) Models(context.Context) ([]provider.ModelInfo, error) { return c.models, nil }

func (c *catalogueClient) GenerateImage(context.Context, provider.ImageRequest) (provider.Image, error) {
	return provider.Image{}, nil
}

func TestTheImageModelMustBeOneTheProviderListsAsAnImageModel(t *testing.T) {
	c := &catalogueClient{models: []provider.ModelInfo{
		{ID: "Qwen", Labels: []string{"chat", "tool-calling"}},
		{ID: "Flux-A", Labels: []string{"image"}},
		{ID: "Flux-B", Labels: []string{"custom", "image"}},
		{ID: "Unlabelled"},
	}}
	gate := &cloudGate{}
	wrapped := gate.wrap(c)
	for _, tc := range []struct{ model, want string }{
		{"Flux-A", ""},
		{"Unlabelled", ""},
		{"Qwen", "Qwen on home is not an image model; it is labelled chat, tool-calling"},
		{"Flux-C", `home has no model named "Flux-C". Its image models: Flux-A, Flux-B`},
	} {
		err := checkImageModel(context.Background(), wrapped, "home", tc.model)
		if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%s: got %v, want %q", tc.model, err, tc.want)
		}
	}
}

func TestAProviderThatCannotDrawIsRefusedAtStartup(t *testing.T) {
	err := checkImageModel(context.Background(), (&cloudGate{}).wrap(&countingClient{}), "plain", "Flux")
	if err == nil || !strings.Contains(err.Error(), "provider plain cannot make pictures") {
		t.Errorf("got %v", err)
	}
}

func TestImagesUseTheCurrentProviderThroughEveryWrapper(t *testing.T) {
	c := &catalogueClient{}
	share := &modelShare{}
	a := &App{client: share.wrap((&cloudGate{}).wrap(c)), providerName: "home"}
	gen, err := a.imageGenerator(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if gen != provider.ImageGenerator(c) {
		t.Errorf("got %T, want the provider itself", gen)
	}
}

type cloudCatalogue struct{ catalogueClient }

func (c *cloudCatalogue) Class() provider.Class { return provider.ClassCloud }

func TestNoCloudStopsPicturesAsItStopsChat(t *testing.T) {
	gate := &cloudGate{static: true}
	a := &App{client: gate.wrap(&cloudCatalogue{}), gate: gate, providerName: "cloudy"}
	if _, err := a.imageGenerator(context.Background()); err == nil || !strings.Contains(err.Error(), "nothing was sent") {
		t.Errorf("a cloud image model was usable under --no-cloud: %v", err)
	}
}
