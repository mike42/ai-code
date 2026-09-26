package main

import (
	"ai-code/internal/agent"
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
	"ai-code/internal/tool"
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

// buildTools assembles the tool set. The executor sorts by name, so the
// advertised list is deterministic and the provider's prompt cache is not
// invalidated by map iteration. window is the detected context window, or 0.
func buildTools(cfg *config.Config, cwd string, window int) (*tool.LocalExecutor, *tool.State) {
	st := tool.NewState(cwd)
	maxOutput := cfg.Tools.Bash.MaxOutputBytes
	if maxOutput <= 0 {
		maxOutput = agent.BudgetFor(window).MaxToolResultBytes()
	}
	return tool.NewLocalExecutor(st,
		&tool.BashTool{
			Timeout:        cfg.Tools.Bash.Timeout.Duration,
			MaxOutputBytes: maxOutput,
			Shell:          cfg.Tools.Bash.Shell,
		},
		&tool.ReadTool{MaxBytes: cfg.Tools.Read.MaxBytes, MaxLines: cfg.Tools.Read.MaxLines},
		&tool.WriteTool{},
		&tool.EditTool{},
		&tool.GrepTool{MaxResults: cfg.Tools.Grep.MaxResults},
		&tool.GlobTool{},
		&tool.LsTool{},
	), st
}

const modelCacheTTL = 6 * time.Hour

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

func loadModelCache(providerName string) (*modelCache, bool) {
	path, err := modelCachePath(providerName)
	if err != nil {
		return nil, false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var c modelCache
	if json.Unmarshal(raw, &c) != nil {
		return nil, false
	}
	return &c, time.Since(c.Fetched) < modelCacheTTL
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

func fetchModels(ctx context.Context, c provider.Client, allowCache bool) ([]provider.ModelInfo, error) {
	if allowCache {
		if cached, fresh := loadModelCache(c.Name()); fresh && len(cached.Models) > 0 {
			return cached.Models, nil
		}
	}
	models, err := c.Models(ctx)
	if err != nil {
		// Stale cache beats no information: an unreachable server should not
		// stop work from starting.
		if cached, _ := loadModelCache(c.Name()); cached != nil && len(cached.Models) > 0 {
			return cached.Models, nil
		}
		return nil, err
	}
	saveModelCache(c.Name(), models)
	return models, nil
}

// resolveModel decides which model to use, in order:
//
//  1. --model on the command line
//  2. default_model for the provider
//  3. whatever the server currently has resident (single-slot servers)
//  4. the first tool-capable model it offers
func resolveModel(ctx context.Context, c provider.Client, requested, configured string) (provider.ModelInfo, error) {
	models, err := fetchModels(ctx, c, true)
	if err != nil {
		if requested != "" || configured != "" {
			// A named model still works without a catalogue; the window stays
			// unknown until a response reports usage.
			name := requested
			if name == "" {
				name = configured
			}
			return provider.ModelInfo{ID: name, SupportsTools: true}, nil
		}
		return provider.ModelInfo{}, fmt.Errorf(
			"could not reach %s to list models: %w", c.Name(), err)
	}

	byID := map[string]provider.ModelInfo{}
	for _, m := range models {
		byID[m.ID] = m
	}

	for _, want := range []string{requested, configured} {
		if want == "" {
			continue
		}
		if m, ok := byID[want]; ok {
			return m, nil
		}
		return provider.ModelInfo{}, fmt.Errorf(
			"model %q is not available on %s.\n\nAvailable models:\n%s",
			want, c.Name(), formatModelList(models, 15))
	}

	if intro, ok := c.(provider.Introspector); ok {
		if h, err := intro.Health(ctx); err == nil && h.ModelLoaded != "" {
			if m, ok := byID[h.ModelLoaded]; ok {
				return m, nil
			}
		}
	}

	for _, m := range models {
		if m.SupportsTools {
			return m, nil
		}
	}
	if len(models) > 0 {
		return models[0], nil
	}
	return provider.ModelInfo{}, fmt.Errorf("%s offers no models", c.Name())
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
