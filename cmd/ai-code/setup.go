package main

import (
	"ai-code/internal/agent"
	"ai-code/internal/worker"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"ai-code/internal/config"
	"ai-code/internal/provider"
	"ai-code/internal/session"
)

func buildClient(name string, p config.Provider) (provider.Client, error) {
	key, err := p.ResolveAPIKey()
	if err != nil {
		return nil, fmt.Errorf("provider %q: %w", name, err)
	}
	class, err := provider.ParseClass(p.Class)
	if err != nil {
		return nil, fmt.Errorf("provider %q: %w", name, err)
	}

	opts := provider.Options{
		Name:    name,
		BaseURL: p.BaseURL,
		APIKey:  key,
		Class:   class,
		Headers: p.Headers,
		Timeout: p.Timeout.Duration,
		TLS: provider.TLSOptions{
			Insecure:   p.TLSInsecure,
			CAFile:     p.CAFile,
			CADir:      p.CADir,
			ClientCert: p.ClientCert,
			ClientKey:  p.ClientKey,
		},
		SendReasoning: p.SendReasoning,
	}

	switch p.Kind {
	case "lemonade":
		return provider.NewLemonade(opts)
	case "openrouter":
		// The llama.cpp reasoning spelling OpenRouter silently ignores: without
		// this, thinking levels would appear to work and do nothing.
		opts.Dialect = provider.DialectOpenRouter
		return provider.NewOpenAI(opts)
	default:
		return provider.NewOpenAI(opts)
	}
}

// toolSettings are sent to the worker; the remote machine's own configuration is
// never read. window is read when the settings are built, and, for tools in
// this process, again each time a command's output is bounded, since the
// model and so its window may be chosen or changed after the tools exist.
func toolSettings(cfg *config.Config, window func() int) worker.Settings {
	s := worker.Settings{
		BashTimeout:    cfg.Tools.Bash.Timeout.Duration,
		BashMaxOutput:  cfg.Tools.Bash.MaxOutputBytes,
		Shell:          cfg.Tools.Bash.Shell,
		ReadMaxBytes:   cfg.Tools.Read.MaxBytes,
		ReadMaxLines:   cfg.Tools.Read.MaxLines,
		GrepMaxResults: cfg.Tools.Grep.MaxResults,
	}
	if s.BashMaxOutput <= 0 {
		derive := func() int { return agent.BudgetFor(window()).MaxToolResultBytes() }
		s.BashMaxOutput = derive()
		s.BashMaxOutputFunc = derive
	}
	return s
}

// modelCache is the last catalogue the provider sent, for Tab completion, which
// must not wait on the network. Nothing that picks a model reads it.
type modelCache struct {
	Fetched time.Time            `json:"fetched"`
	Models  []provider.ModelInfo `json:"models"`
}

func modelCachePath(providerName string) (string, error) {
	root, err := session.Root()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "cache", "models-"+sanitise(providerName)+".json"), nil
}

func sanitise(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '/' || r == '\\' || r == ':' || r == ' ' {
			return '-'
		}
		return r
	}, s)
}

func loadModelCache(providerName string) *modelCache {
	path, err := modelCachePath(providerName)
	if err != nil {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var c modelCache
	if json.Unmarshal(raw, &c) != nil {
		return nil
	}
	return &c
}

func saveModelCache(providerName string, models []provider.ModelInfo) {
	path, err := modelCachePath(providerName)
	if err != nil {
		return
	}
	if os.MkdirAll(filepath.Dir(path), 0o700) != nil {
		return
	}
	raw, err := json.Marshal(modelCache{Fetched: time.Now(), Models: models})
	if err != nil {
		return
	}
	_ = os.WriteFile(path, raw, 0o600)
}

// liveModels asks the provider for its catalogue. Every decision about which
// model to use reads this; the copy it leaves on disk is for Tab completion
// only, which must not wait on the network.
func liveModels(ctx context.Context, c provider.Client) ([]provider.ModelInfo, error) {
	models, err := c.Models(ctx)
	if err != nil {
		return nil, fmt.Errorf("%s did not list its models: %w", c.Name(), err)
	}
	saveModelCache(c.Name(), models)
	return models, nil
}

// modelChoice is the model a session uses and why.
type modelChoice struct {
	Model provider.ModelInfo
	// Why says where the choice came from, for the line announcing it.
	Why string
	// Explicit marks a model the user named. Only that may replace one the
	// server has loaded.
	Explicit bool
}

// chooseModel picks the model from what is true now, never from a default:
//
//  1. the one the user named (--model, AI_CODE_MODEL)
//  2. the one the server has loaded
//  3. with nothing loaded, the one last used in this project
//
// Anything else is an error that says so; there is no fallback, because a
// guessed model is a model swap on the user's server.
func chooseModel(ctx context.Context, c provider.Client, requested, previous string) (modelChoice, error) {
	models, err := liveModels(ctx, c)
	if err != nil {
		return modelChoice{}, err
	}
	byID := map[string]provider.ModelInfo{}
	for _, m := range models {
		byID[m.ID] = m
	}

	if requested != "" {
		m, ok := byID[requested]
		if !ok {
			return modelChoice{}, fmt.Errorf("model %q is not available on %s.\n\nAvailable models:\n%s",
				requested, c.Name(), formatModelList(models, 15))
		}
		return modelChoice{Model: m, Why: "named with --model", Explicit: true}, nil
	}

	intro, ok := c.(provider.Introspector)
	if !ok {
		if m, ok := byID[previous]; ok && previous != "" {
			return modelChoice{Model: m, Why: "last used here"}, nil
		}
		return modelChoice{}, noModel("%s cannot say which model is loaded and none was used here before: "+
			"choose one with /model NAME", c.Name())
	}

	h, err := intro.Health(ctx)
	if err != nil {
		return modelChoice{}, fmt.Errorf("could not ask %s which model is loaded: %w", c.Name(), err)
	}
	var loaded []provider.ModelInfo
	for _, id := range h.Loaded {
		if m, ok := byID[id]; ok {
			loaded = append(loaded, m)
		}
	}
	switch {
	case len(loaded) == 1:
		return modelChoice{Model: loaded[0], Why: "loaded on " + c.Name()}, nil
	case len(loaded) > 1:
		names := make([]string, len(loaded))
		for i, m := range loaded {
			if m.ID == previous {
				return modelChoice{Model: m, Why: "loaded on " + c.Name() + ", last used here"}, nil
			}
			names[i] = m.ID
		}
		return modelChoice{}, noModel("%s has %s loaded: choose one with /model NAME",
			c.Name(), joinAnd(names))
	}
	if m, ok := byID[previous]; ok && previous != "" {
		return modelChoice{Model: m, Why: "last used here; nothing is loaded on " + c.Name()}, nil
	}
	return modelChoice{}, noModel("nothing is loaded on %s and no model was used here before: "+
		"choose one with /model NAME", c.Name())
}

func formatModelList(models []provider.ModelInfo, limit int) string {
	sorted := append([]provider.ModelInfo(nil), models...)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].SupportsTools != sorted[j].SupportsTools {
			return sorted[i].SupportsTools
		}
		return sorted[i].ID < sorted[j].ID
	})

	var b strings.Builder
	for i, m := range sorted {
		if i >= limit {
			fmt.Fprintf(&b, "  ... and %d more\n", len(sorted)-limit)
			break
		}
		b.WriteString("  " + m.ID)
		var notes []string
		if m.ContextWindow > 0 {
			notes = append(notes, fmt.Sprintf("%s ctx", compactInt(m.ContextWindow)))
		}
		if m.Loaded {
			notes = append(notes, "loaded")
		}
		if !m.SupportsTools {
			notes = append(notes, "no tool calling")
		}
		if len(notes) > 0 {
			fmt.Fprintf(&b, "  (%s)", strings.Join(notes, ", "))
		}
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

func compactInt(n int) string {
	switch {
	case n >= 1000000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%dk", n/1000)
	default:
		return fmt.Sprint(n)
	}
}

// windowIsDivided reports whether a single request gets less than the server's
// whole KV allocation. Parallel > 1 is not sufficient: with a unified KV cache
// several slots share one pool and each may still reach the full ctx_size.
func windowIsDivided(m provider.ModelInfo) bool {
	return m.CtxSize > 0 && m.ContextWindow > 0 && m.ContextWindow < m.CtxSize
}

// slotsAreContended reports the other shape: the window is the whole pool, but
// only while the other slots are idle.
func slotsAreContended(m provider.ModelInfo) bool {
	return m.KVUnified && m.Parallel > 1 && m.CtxSize > 0 && !windowIsDivided(m)
}

// slotSummary states how the backend carved up its KV cache, or "" when it did
// not. The count is the server's decision, so this reports it and suggests
// nothing.
func slotSummary(m provider.ModelInfo) string {
	switch {
	case windowIsDivided(m):
		return fmt.Sprintf("%s across %d slots", compactInt(m.CtxSize), m.Parallel)
	case slotsAreContended(m):
		return fmt.Sprintf("%s shared across %d slots", compactInt(m.CtxSize), m.Parallel)
	}
	return ""
}

func contextLimitFor(m provider.ModelInfo, override int) (int, string) {
	if override > 0 {
		return override, "configured override"
	}
	if m.ContextWindow > 0 {
		if windowIsDivided(m) {
			// The number people misread: one request gets a fraction of it.
			return m.ContextWindow, fmt.Sprintf(
				"%s allocation split across %d slots", compactInt(m.CtxSize), m.Parallel)
		}
		if slotsAreContended(m) {
			return m.ContextWindow, fmt.Sprintf(
				"%s allocation shared by %d slots", compactInt(m.CtxSize), m.Parallel)
		}
		return m.ContextWindow, "reported by the server"
	}
	return provider.DefaultUnknownContext, "unknown; using a conservative default"
}
