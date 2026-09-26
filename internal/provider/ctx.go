package provider

import (
	"strconv"
	"strings"
)

// DefaultUnknownContext stands in when a backend reports no window.
// Conservative: overestimating fails mid-turn, underestimating compacts sooner.
const DefaultUnknownContext = 8192

// ParallelAuto is what llama.cpp's `--parallel -1` means: the server decides
// the slot count at runtime. Not 1: an auto count also turns unified KV on by
// default.
const ParallelAuto = -1

// ContextInputs describes how a backend may have carved up its KV cache.
// Every field is optional: zero means "not known", and the chain falls through
// to the next source rather than assuming a value.
type ContextInputs struct {
	// MaxContextWindow is the model's architectural limit. It only ever caps.
	MaxContextWindow int
	// CtxSize is the total KV allocation the server was launched with -- the
	// pool, not a per-request number.
	CtxSize int
	// Parallel is the requested slot count: 0 if `--parallel` was absent,
	// ParallelAuto if explicitly auto, otherwise the requested number.
	Parallel int
	// KVUnified reports whether the slots share one pool; not derivable from
	// Parallel, see LlamaCppArgs.UnifiedKV.
	KVUnified bool
	// PerSlotLimit is an explicit `--kv-unified-per-slot N`.
	PerSlotLimit int
	// SlotNCtx is a per-slot window observed from GET /slots -- the only input
	// measured rather than derived from launch flags.
	SlotNCtx int
}

// EffectiveContext computes the per-request context window in tokens and
// reports whether the result was observed or inferred. Precedence, strongest
// first:
//
//  1. GET /slots -> slots[k].n_ctx     the running server's own answer
//  2. --kv-unified-per-slot N          an explicit per-slot window
//  3. unified KV                       the whole pool, contended
//  4. split KV                         pool / slot count
//  5. architectural limit alone        a guess, flagged as one
//  6. nothing known                    0; caller substitutes a default
//
// A derived number is capped by the model's architectural limit; an observed
// slot window is not.
func EffectiveContext(in ContextInputs) (window int, estimated bool) {
	atMost := func(n int) int {
		if in.MaxContextWindow > 0 && n > in.MaxContextWindow {
			return in.MaxContextWindow
		}
		return n
	}

	switch {
	case in.SlotNCtx > 0:
		return in.SlotNCtx, false

	case in.PerSlotLimit > 0:
		// Overrides both the division and the unified/split question, so it is
		// checked before either.
		return atMost(in.PerSlotLimit), false

	case in.CtxSize > 0 && in.KVUnified:
		// A shared pool means one request may reach the whole ctx_size, but
		// only while the other slots are idle: a ceiling, not a reservation,
		// so it stays flagged as an estimate.
		return atMost(in.CtxSize), true

	case in.CtxSize > 0 && in.Parallel > 0:
		return atMost(in.CtxSize / in.Parallel), false

	case in.CtxSize > 0:
		// Split KV with an unresolved slot count: `--no-kv-unified` with an
		// auto `--parallel`. The divisor is whatever the server chose at
		// startup, so the pool is an upper bound.
		return atMost(in.CtxSize), true

	case in.MaxContextWindow > 0:
		// No allocation info: the architectural limit is the best guess, but
		// the server may have been launched with less.
		return in.MaxContextWindow, true

	default:
		return 0, false
	}
}

// contextInputs rebuilds the chain inputs from a ModelInfo so a later
// observation can re-run the chain without the caller reassembling the fields.
func (m ModelInfo) contextInputs(slotNCtx int) ContextInputs {
	return ContextInputs{
		MaxContextWindow: m.MaxContextWindow,
		CtxSize:          m.CtxSize,
		Parallel:         m.Parallel,
		KVUnified:        m.KVUnified,
		PerSlotLimit:     m.PerSlotLimit,
		SlotNCtx:         slotNCtx,
	}
}

// LlamaCppArgs holds the launch flags that decide the per-request window.
// Zero values mean "flag absent", which is distinct from the flag's default.
type LlamaCppArgs struct {
	CtxSize int
	// Parallel is 0 when `--parallel` was absent, ParallelAuto when it was
	// explicitly -1, and the requested slot count otherwise.
	Parallel int
	// KVUnified is only meaningful when KVUnifiedSet is true; use UnifiedKV.
	KVUnified    bool
	KVUnifiedSet bool
	PerSlotLimit int
}

// UnifiedKV applies llama.cpp's default rule: unified KV is enabled if and
// only if the slot count is auto. A bare `--parallel 4` silently switches to
// split and quarters the window, so this is not derivable from Parallel.
func (a LlamaCppArgs) UnifiedKV() bool {
	if a.KVUnifiedSet {
		return a.KVUnified
	}
	return a.Parallel < 1
}

// ParseLlamaCppArgs extracts the flags that affect the usable context window
// from a llama.cpp command line in either flag form; quoted values are skipped
// rather than misparsed. Prefer the resolved launch command over a recipe.
func ParseLlamaCppArgs(args string) LlamaCppArgs {
	var out LlamaCppArgs
	fields := splitArgs(args)
	for i := 0; i < len(fields); i++ {
		f := fields[i]

		name, inlineVal, hasInline := strings.Cut(f, "=")
		value := func() string {
			if hasInline {
				return inlineVal
			}
			if i+1 < len(fields) {
				i++
				return fields[i]
			}
			return ""
		}

		switch name {
		case "--ctx-size", "-c", "--n-ctx":
			if n, err := strconv.Atoi(value()); err == nil && n > 0 {
				out.CtxSize = n
			}
		case "--parallel", "-np":
			// A negative value is auto and must survive: it carries the signal
			// that the cache is unified, worth a factor of the slot count.
			if n, err := strconv.Atoi(value()); err == nil {
				switch {
				case n > 0:
					out.Parallel = n
				case n < 0:
					out.Parallel = ParallelAuto
				}
			}
		case "--kv-unified", "-kvu":
			out.KVUnified, out.KVUnifiedSet = true, true
			if hasInline {
				out.KVUnified = truthy(inlineVal)
			} else if i+1 < len(fields) && isBoolLiteral(fields[i+1]) {
				// llama.cpp has been migrating boolean flags to explicit
				// on/off values, so accept a value only when the next token is
				// unambiguously one.
				i++
				out.KVUnified = truthy(fields[i])
			}
		case "--no-kv-unified":
			out.KVUnified, out.KVUnifiedSet = false, true
		case "--kv-unified-per-slot":
			if n, err := strconv.Atoi(value()); err == nil && n > 0 {
				out.PerSlotLimit = n
			}
		}
	}
	return out
}

func isBoolLiteral(s string) bool {
	switch strings.ToLower(s) {
	case "on", "off", "true", "false", "1", "0", "yes", "no", "enabled", "disabled":
		return true
	}
	return false
}

func truthy(s string) bool {
	switch strings.ToLower(s) {
	case "off", "false", "0", "no", "disabled":
		return false
	}
	return true
}

// splitArgs splits shell-like, respecting single and double quotes. It only
// has to avoid treating the interior of a quoted JSON blob as separate flags.
func splitArgs(s string) []string {
	var (
		out   []string
		cur   strings.Builder
		quote rune
	)
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				cur.WriteRune(r)
			}
		case r == '\'' || r == '"':
			quote = r
		case r == ' ' || r == '\t' || r == '\n':
			flush()
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}
