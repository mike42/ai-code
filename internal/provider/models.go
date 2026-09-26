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

// modelEntry is the union of the fields the backends ai-code speaks to put in
// /models. Unknown fields are ignored; absent fields stay zero and the context
// computation degrades gracefully.
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

// catalogueKey says whether listing models must carry the API key.
//
// On OpenRouter it must not. The catalogue is public and answers without one,
// while sending the key turns a routine listing -- done at startup whenever
// the six-hour cache has expired, before the user has typed anything -- into
// an authenticated, attributable contact with a third party. No user content
// is in it either way; what the key adds is the account it is charged to.
//
// Everywhere else the key stays on. A self-hosted endpoint may well refuse to
// list anything without it, and withholding it from your own server buys no
// privacy. The question is asked of the dialect rather than the base URL for
// the reason given on Dialect: which protocol an endpoint speaks is
// configuration, not something to infer from a hostname.
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
		// An explicit --ctx-size on the command line is what the backend was
		// actually launched with and wins over the recipe's nominal value.
		if args.CtxSize > 0 {
			m.CtxSize = args.CtxSize
		}
		// Left at 0 when --parallel is absent and ParallelAuto when it is auto:
		// both mean "the server decides", which is a different statement from
		// "one slot" and produces a different window.
		m.Parallel = args.Parallel
		m.KVUnified = args.UnifiedKV()
		m.PerSlotLimit = args.PerSlotLimit
	}

	m.ContextWindow, m.Estimated = EffectiveContext(m.contextInputs(0))

	m.SupportsTools = slices.Contains(e.Labels, "tool-calling") ||
		slices.Contains(e.SupportedParameters, "tools")
	// A backend that advertises no capability metadata at all is assumed
	// capable; refusing to talk to it would be worse than trying and failing.
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

// ---------------------------------------------------------------------------
// slots
// ---------------------------------------------------------------------------

// slotEntry is one element of llama.cpp's GET /slots array. Field names and
// the surrounding shape have both moved between releases, so everything is
// optional and a missing field costs us a fallback rather than an error.
type slotEntry struct {
	ID int `json:"id"`
	// NCtx is the per-request window this slot actually has. It is the one
	// number in the whole context chain that nothing has to infer.
	NCtx         int  `json:"n_ctx"`
	IsProcessing bool `json:"is_processing"`

	// Prompt-token counts: how full the slot currently is. Not yet surfaced,
	// but this endpoint is the only place they exist and reading them here
	// keeps the eventual dispatch decision (is there a slot with a warm prefix
	// and room to spare?) from needing a second protocol.
	NPromptTokens          int `json:"n_prompt_tokens"`
	NPromptTokensProcessed int `json:"n_prompt_tokens_processed"`
}

// SlotsInfo is the aggregate of what GET /slots reported.
type SlotsInfo struct {
	// Count is the observed slot count -- the resolved answer to `--parallel
	// auto`, which nothing else on the wire tells us.
	Count int
	Busy  int
	// NCtx is the smallest per-slot window seen. Slots are normally uniform;
	// taking the minimum means a heterogeneous server errs towards compacting
	// early rather than towards a mid-turn overflow.
	NCtx int
	// MaxPromptTokens is the largest prompt currently resident in any slot.
	MaxPromptTokens int
}

// slotsProbeTimeout bounds the /slots probe. It is enrichment on a path the
// user is waiting on, and doWithRetry will happily spend several backoffs on a
// server that answers 503; a slow no is the same as a no here.
const slotsProbeTimeout = 3 * time.Second

// Slots reads GET /slots, which llama.cpp exposes and most other backends do
// not.
//
// Absence is the normal case, not a failure: the endpoint is off under
// --no-slots, returns 501 on some builds, 404 on anything that is not
// llama.cpp, and lemonade proxies some upstream endpoints but not others, so
// reachability is a runtime discovery rather than a property of the configured
// provider. Every one of those degrades to the next source in the context
// chain, which is why callers ignore the error instead of surfacing it.
func (c *OpenAI) Slots(ctx context.Context) (*SlotsInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, slotsProbeTimeout)
	defer cancel()

	var raw []slotEntry
	if err := c.getJSON(ctx, "/slots", &raw, withKey); err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		// A well-formed empty array tells us nothing; treating it as "zero
		// slots" would claim the server can serve nobody.
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

// ---------------------------------------------------------------------------
// lemonade
// ---------------------------------------------------------------------------

// Lemonade adds residency introspection to the generic client. On a server with
// a single LLM slot, knowing which model is actually resident is the difference
// between a working session and a silent model swap.
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
	// Mark what is actually resident. A health failure is not fatal here: the
	// model list is still useful without residency information.
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
	// nothing about the rest of the catalogue -- and without /health we do not
	// know which entry that is. Best-effort, same as residency above.
	if resident >= 0 {
		if s, err := l.Slots(ctx); err == nil {
			m := &models[resident]
			m.Slots, m.SlotsBusy = s.Count, s.Busy
			// An observed count resolves `--parallel auto`, the one case where
			// the launch arguments genuinely cannot tell us the divisor.
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
