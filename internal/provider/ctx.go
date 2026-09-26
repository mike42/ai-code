package provider

import (
	"strconv"
	"strings"
)

// DefaultUnknownContext is used when a backend tells us nothing about its
// window. Deliberately conservative: overestimating the window produces a hard
// API error mid-turn, while underestimating merely compacts sooner.
const DefaultUnknownContext = 8192

// ParallelAuto is what llama.cpp's `--parallel -1` means: the server decides
// the slot count at runtime. It is emphatically not 1 -- treating it as 1 both
// invents a divisor and hides the second consequence of an auto slot count,
// which is that unified KV is then on by default.
const ParallelAuto = -1

// ContextInputs is everything we may have learned about how a backend carved
// up its KV cache. Passed as a struct because the chain below keeps acquiring
// sources as more of the protocol is understood, and six positional ints at a
// call site are indistinguishable from each other to a reader.
//
// Every field is optional. Zero means "not known", and the chain falls through
// to the next source rather than assuming a value.
type ContextInputs struct {
	// MaxContextWindow is the model's architectural limit. It only ever caps.
	MaxContextWindow int
	// CtxSize is the total KV allocation the server was launched with -- the
	// pool, which is not by itself a per-request number.
	CtxSize int
	// Parallel is the requested slot count: 0 if `--parallel` was absent,
	// ParallelAuto if it was explicitly auto, otherwise the requested number.
	Parallel int
	// KVUnified reports whether the slots share one pool. See
	// LlamaCppArgs.UnifiedKV for why this cannot be derived from Parallel at
	// the point of use.
	KVUnified bool
	// PerSlotLimit is an explicit `--kv-unified-per-slot N`.
	PerSlotLimit int
	// SlotNCtx is a per-slot window observed from GET /slots -- the only input
	// here that was measured rather than derived from launch flags.
	SlotNCtx int
}

// EffectiveContext computes the per-request context window in tokens, and
// reports whether the answer was observed or merely inferred.
//
// llama.cpp's ctx_size is the total KV allocation, not a per-request window.
// A split cache divides it across `--parallel` slots; a unified one lets a
// single request reach the lot. Reading ctx_size directly overstates the
// window by the slot count and fails mid-session.
//
// Precedence, strongest first:
//
//  1. GET /slots -> slots[k].n_ctx     the running server's own answer
//  2. --kv-unified-per-slot N          an explicit per-slot window
//  3. unified KV                       the whole pool, contended
//  4. split KV                         pool / slot count
//  5. architectural limit alone        a guess, flagged as one
//  6. nothing known                    0; caller substitutes a default
//
// A ctx_size above the model's architectural limit is legal and observed, so
// every derived number is capped by that limit. An observed slot window is
// not: it came from the process that will service the request.
//
// Returns 0 when nothing is known; callers substitute DefaultUnknownContext.
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
		// This flag overrides both the division and the unified/split question,
		// so it is checked before either.
		return atMost(in.PerSlotLimit), false

	case in.CtxSize > 0 && in.KVUnified:
		// A shared pool means one request may reach the whole ctx_size, but
		// only while the other slots are idle. It is a ceiling rather than a
		// reservation, so it is marked as an estimate even though the number
		// itself came from the launch arguments.
		return atMost(in.CtxSize), true

	case in.CtxSize > 0 && in.Parallel > 0:
		return atMost(in.CtxSize / in.Parallel), false

	case in.CtxSize > 0:
		// Split KV with an unresolved slot count: `--no-kv-unified` alongside an
		// auto `--parallel`. The divisor is whatever the server chose at
		// startup and we have not observed it, so the pool is an upper bound.
		return atMost(in.CtxSize), true

	case in.MaxContextWindow > 0:
		// No allocation info: the architectural limit is the best guess, but the
		// server may well have been launched with less.
		return in.MaxContextWindow, true

	default:
		return 0, false
	}
}

// contextInputs rebuilds the chain inputs from a ModelInfo that has already
// been populated from the catalogue, so a later observation (a slot window
// arriving from /slots after the model list was built) can re-run the chain
// without the caller reassembling six fields by hand.
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
// Zero values mean "flag absent", which is distinct from the flag's default:
// the defaults are applied by UnifiedKV and by EffectiveContext, where the
// other flags needed to interpret them are also in scope.
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

// UnifiedKV applies llama.cpp's own default rule: unified KV is enabled if and
// only if the slot count is auto.
//
// This is why KVUnified is not derivable from Parallel at the point of use and
// has to be resolved here: a command line with neither flag has a unified
// cache and a full-size window, while adding a bare `--parallel 4` silently
// switches the cache to split and quarters every request's window. Nothing on
// the command line says so.
func (a LlamaCppArgs) UnifiedKV() bool {
	if a.KVUnifiedSet {
		return a.KVUnified
	}
	return a.Parallel < 1
}

// ParseLlamaCppArgs extracts the flags that affect the usable context window
// from a llama.cpp command line.
//
// Prefer the resolved launch command (lemonade's /health exposes
// all_models_loaded[].launch_command as an array with `auto` already resolved)
// over the recipe's nominal argument string, which may still contain
// placeholders.
//
// Recognises both long and short forms, and both `--flag N` and `--flag=N`.
// Quoted argument values (lemonade embeds e.g. --chat-template-kwargs '{...}')
// are skipped rather than misparsed.
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
			// A negative value is auto and must survive: dropping it (as an
			// earlier version did by only accepting n > 0) loses the signal
			// that the cache is unified, which is worth a factor of the slot
			// count in the other direction.
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
				// Bare in every build seen so far, but llama.cpp has been
				// migrating boolean flags to an explicit on/off value (-fa did
				// exactly that), so accept a value only when the next token
				// unambiguously is one.
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

// splitArgs performs shell-like splitting that respects single and double
// quotes. It is not a full shell parser and does not need to be: it only has to
// avoid treating the interior of a quoted JSON blob as separate flags.
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
