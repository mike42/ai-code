package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"sort"
	"strings"
	"time"
)

// modelEntry is the union of the fields the supported backends put in /models.
// Unknown fields are ignored and absent fields stay zero.
type modelEntry struct {
	ID string `json:"id"`

	// OpenRouter and several gateways.
	ContextLength int `json:"context_length"`
	TopProvider   *struct {
		ContextLength int `json:"context_length"`
	} `json:"top_provider"`
	SupportedParameters []string `json:"supported_parameters"`

	// lemonade.
	MaxContextWindow int      `json:"max_context_window"`
	Labels           []string `json:"labels"`
	Downloaded       *bool    `json:"downloaded"`
	RecipeOptions    *struct {
		CtxSize      int    `json:"ctx_size"`
		LlamaCppArgs string `json:"llamacpp_args"`
	} `json:"recipe_options"`
}

func (c *OpenAI) getJSON(ctx context.Context, path string, out any, key keyUse) error {
	resp, err := c.doWithRetry(ctx, "GET", path, nil, key)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}
	return nil
}

// catalogueKey says whether listing models must carry the API key. Not on
// OpenRouter: the catalogue is public, so sending the key would attribute a
// routine startup listing to the account. Everywhere else the key stays on.
func (c *OpenAI) catalogueKey() keyUse {
	if c.opts.Dialect == DialectOpenRouter {
		return withoutKey
	}
	return withKey
}

func (c *OpenAI) Models(ctx context.Context) ([]ModelInfo, error) {
	var body struct {
		Data []modelEntry `json:"data"`
	}
	if err := c.getJSON(ctx, "/models", &body, c.catalogueKey()); err != nil {
		return nil, err
	}
	out := make([]ModelInfo, 0, len(body.Data))
	for _, e := range body.Data {
		out = append(out, e.toModelInfo())
	}
	sortModels(out)
	return out, nil
}

func (e modelEntry) toModelInfo() ModelInfo {
	m := ModelInfo{
		ID:               e.ID,
		MaxContextWindow: e.MaxContextWindow,
		Labels:           e.Labels,
	}
	if m.MaxContextWindow == 0 {
		m.MaxContextWindow = e.ContextLength
	}
	if m.MaxContextWindow == 0 && e.TopProvider != nil {
		m.MaxContextWindow = e.TopProvider.ContextLength
	}

	if e.RecipeOptions != nil {
		m.CtxSize = e.RecipeOptions.CtxSize
		args := ParseLlamaCppArgs(e.RecipeOptions.LlamaCppArgs)
		// An explicit --ctx-size on the command line wins over the recipe's
		// nominal value.
		if args.CtxSize > 0 {
			m.CtxSize = args.CtxSize
		}
		// Both absent and auto mean "the server decides", a different window
		// from "one slot".
		m.Parallel = args.Parallel
		m.KVUnified = args.UnifiedKV()
		m.PerSlotLimit = args.PerSlotLimit
	}

	m.ContextWindow, m.Estimated = EffectiveContext(m.contextInputs(0))

	m.SupportsTools = slices.Contains(e.Labels, "tool-calling") ||
		slices.Contains(e.SupportedParameters, "tools")
	// No capability metadata at all: assume capable rather than refuse.
	if len(e.Labels) == 0 && len(e.SupportedParameters) == 0 {
		m.SupportsTools = true
	}
	return m
}

func sortModels(m []ModelInfo) {
	sort.Slice(m, func(i, j int) bool {
		// Tool-capable models first: on a coding harness, the rest are noise.
		if m[i].SupportsTools != m[j].SupportsTools {
			return m[i].SupportsTools
		}
		return strings.ToLower(m[i].ID) < strings.ToLower(m[j].ID)
	})
}

// slotEntry is one element of llama.cpp's GET /slots array; field names and
// shape vary between releases, so everything is optional.
type slotEntry struct {
	ID int `json:"id"`
	// NCtx is the per-request window this slot actually has: observed, not
	// inferred.
	NCtx         int  `json:"n_ctx"`
	IsProcessing bool `json:"is_processing"`

	NPromptTokens          int `json:"n_prompt_tokens"`
	NPromptTokensProcessed int `json:"n_prompt_tokens_processed"`
}

// SlotsInfo is the aggregate of what GET /slots reported.
type SlotsInfo struct {
	// Count is the observed slot count, the resolved answer to --parallel auto.
	Count int
	Busy  int
	// NCtx is the smallest per-slot window seen; a heterogeneous server errs
	// towards compacting early rather than a mid-turn overflow.
	NCtx int
	// MaxPromptTokens is the largest prompt currently resident in any slot.
	MaxPromptTokens int
}

// slotsProbeTimeout bounds the /slots probe; a slow no is the same as a no on a
// startup path.
const slotsProbeTimeout = 3 * time.Second

// Slots reads GET /slots, which llama.cpp exposes and most other backends do
// not. Absence is normal: the endpoint is off under --no-slots or returns
// 501/404, and the context chain falls through, so callers ignore the error.
func (c *OpenAI) Slots(ctx context.Context) (*SlotsInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, slotsProbeTimeout)
	defer cancel()

	var raw []slotEntry
	if err := c.getJSON(ctx, "/slots", &raw, withKey); err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		// An empty array is not zero slots; it would claim the server can
		// serve nobody.
		return nil, fmt.Errorf("/slots reported no slots")
	}

	info := &SlotsInfo{Count: len(raw)}
	for _, s := range raw {
		if s.IsProcessing {
			info.Busy++
		}
		if s.NCtx > 0 && (info.NCtx == 0 || s.NCtx < info.NCtx) {
			info.NCtx = s.NCtx
		}
		info.MaxPromptTokens = max(info.MaxPromptTokens, s.NPromptTokens, s.NPromptTokensProcessed)
	}
	return info, nil
}

// Lemonade adds residency introspection to the generic client; on a single-slot
// server, knowing which model is resident separates a working session from a
// silent swap.
type Lemonade struct {
	*OpenAI
}

func NewLemonade(opts Options) (*Lemonade, error) {
	c, err := NewOpenAI(opts)
	if err != nil {
		return nil, err
	}
	return &Lemonade{OpenAI: c}, nil
}

func (l *Lemonade) Models(ctx context.Context) ([]ModelInfo, error) {
	models, err := l.OpenAI.Models(ctx)
	if err != nil {
		return nil, err
	}
	// Mark what is resident; a health failure still leaves a useful model list.
	resident := -1
	if h, err := l.Health(ctx); err == nil && h.ModelLoaded != "" {
		for i := range models {
			if models[i].ID == h.ModelLoaded {
				models[i].Loaded = true
				resident = i
			}
		}
	}

	// Slots belong to the process serving the resident model, so they say
	// nothing about the rest of the catalogue.
	if resident >= 0 {
		if s, err := l.Slots(ctx); err == nil {
			m := &models[resident]
			m.Slots, m.SlotsBusy = s.Count, s.Busy
			// An observed count resolves `--parallel auto`, which the launch
			// arguments cannot.
			if m.Parallel < 1 {
				m.Parallel = s.Count
			}
			m.ContextWindow, m.Estimated = EffectiveContext(m.contextInputs(s.NCtx))
		}
	}
	return models, nil
}

func (l *Lemonade) Health(ctx context.Context) (*Health, error) {
	var body struct {
		ModelLoaded string `json:"model_loaded"`
		MaxModels   struct {
			LLM int `json:"llm"`
		} `json:"max_models"`
		AllModelsLoaded []struct {
			ModelName   string `json:"model_name"`
			Loaded      bool   `json:"loaded"`
			IsBusy      bool   `json:"is_busy"`
			IsStreaming bool   `json:"is_streaming"`
			Status      string `json:"status"`
		} `json:"all_models_loaded"`
	}
	if err := l.getJSON(ctx, "/health", &body, withKey); err != nil {
		return nil, err
	}

	h := &Health{
		ModelLoaded: body.ModelLoaded,
		MaxLLMSlots: body.MaxModels.LLM,
	}
	for _, m := range body.AllModelsLoaded {
		if m.ModelName == body.ModelLoaded {
			h.Ready = m.Loaded && m.Status == "ready"
			h.Busy = m.IsBusy
			h.Streaming = m.IsStreaming
		}
	}
	return h, nil
}

func (l *Lemonade) Load(ctx context.Context, model string) error {
	body, _ := json.Marshal(map[string]string{"model_name": model})
	resp, err := l.doWithRetry(ctx, "POST", "/load", body, withKey)
	if err != nil {
		return fmt.Errorf("loading model %q: %w", model, err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	return nil
}

func (l *Lemonade) Unload(ctx context.Context, model string) error {
	body, _ := json.Marshal(map[string]string{"model_name": model})
	resp, err := l.doWithRetry(ctx, "POST", "/unload", body, withKey)
	if err != nil {
		return fmt.Errorf("unloading model %q: %w", model, err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	return nil
}

var (
	_ Client       = (*OpenAI)(nil)
	_ Client       = (*Lemonade)(nil)
	_ Introspector = (*Lemonade)(nil)
	_              = http.StatusOK
)
