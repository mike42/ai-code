package provider

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEffectiveContext(t *testing.T) {
	tests := []struct {
		name          string
		in            ContextInputs
		want          int
		wantEstimated bool
	}{
		{"split KV divides the pool across the requested slots",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, Parallel: 4},
			65536, false},
		{"unified KV does not divide: every slot may reach the whole pool",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, Parallel: 4, KVUnified: true},
			262144, true},
		{"one slot gets the pool either way",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, Parallel: 1},
			262144, false},
		{"--kv-unified-per-slot overrides the division",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, Parallel: 4, PerSlotLimit: 32768},
			32768, false},
		{"--kv-unified-per-slot overrides unified KV too",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, KVUnified: true, PerSlotLimit: 32768},
			32768, false},
		{"ctx_size above the model's architectural limit is capped",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 655360, Parallel: 1},
			262144, false},
		{"the cap applies after the slot division",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 655360, Parallel: 2},
			262144, false},
		{"the cap applies to an oversized per-slot limit",
			ContextInputs{MaxContextWindow: 131072, PerSlotLimit: 262144},
			131072, false},
		{"an observed slot window beats every derived source",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, Parallel: 4,
				KVUnified: true, PerSlotLimit: 32768, SlotNCtx: 65536},
			65536, false},
		{"an observed slot window is not second-guessed by the catalogue limit",
			// The server allocated it; the catalogue is the thing more likely
			// to be stale.
			ContextInputs{MaxContextWindow: 131072, SlotNCtx: 262144},
			262144, false},
		{"--parallel -1 is auto, which makes the cache unified by default",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144,
				Parallel: ParallelAuto, KVUnified: true},
			262144, true},
		{"--no-kv-unified with an auto slot count: divisor unknown, pool is an upper bound",
			ContextInputs{MaxContextWindow: 262144, CtxSize: 262144, Parallel: ParallelAuto},
			262144, true},
		{"parallel absent is not a divisor of zero",
			ContextInputs{MaxContextWindow: 131072, CtxSize: 131072},
			131072, true},
		{"no allocation info falls back to the model limit, as an estimate",
			ContextInputs{MaxContextWindow: 131072, Parallel: 4},
			131072, true},
		{"no model limit uses the per-slot allocation",
			ContextInputs{CtxSize: 131072, Parallel: 2},
			65536, false},
		{"nothing known returns 0 so the caller can substitute a default",
			ContextInputs{},
			0, false},
		{"integer division truncates rather than over-promising",
			ContextInputs{CtxSize: 100000, Parallel: 3},
			33333, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, estimated := EffectiveContext(tt.in)
			if got != tt.want || estimated != tt.wantEstimated {
				t.Errorf("EffectiveContext(%+v) = (%d, estimated %v), want (%d, estimated %v)",
					tt.in, got, estimated, tt.want, tt.wantEstimated)
			}
		})
	}
}

func TestParseLlamaCppArgs(t *testing.T) {
	tests := []struct {
		name string
		args string
		want LlamaCppArgs
	}{
		{
			// Verbatim from a lemonade /api/v1/health response. The
			// quoted JSON in --chat-template-kwargs must not be parsed as flags,
			// and the trailing bare -kvu must not swallow -ngl.
			name: "live lemonade args with embedded quoted JSON",
			args: `--batch-size 4096 --cache-ram 65536 --chat-template-kwargs '{"preserve_thinking":true}' --min-p 0.00 --parallel 4 --poll 0 --repeat-penalty 1.0 --spec-draft-n-max 3 --temp 1.0 --threads 4 --top-k 20 --top-p 0.95 --ubatch-size 4096 -fa on -kvu -ngl 999`,
			want: LlamaCppArgs{Parallel: 4, KVUnified: true, KVUnifiedSet: true},
		},
		{"long form with equals", "--ctx-size=32768 --parallel=2",
			LlamaCppArgs{CtxSize: 32768, Parallel: 2}},
		{"short forms", "-c 16384 -np 8",
			LlamaCppArgs{CtxSize: 16384, Parallel: 8}},
		{"ctx size alone is reported", "--ctx-size 4096",
			LlamaCppArgs{CtxSize: 4096}},
		{"absent flags report zero", "-ngl 999 -fa on", LlamaCppArgs{}},
		{"empty string", "", LlamaCppArgs{}},
		{"trailing flag with no value is ignored", "--parallel", LlamaCppArgs{}},
		{"non-numeric value is ignored", "--parallel abc", LlamaCppArgs{}},
		{"a quoted value containing --parallel is not a flag",
			`--chat-template-kwargs "--parallel 99" --parallel 2`,
			LlamaCppArgs{Parallel: 2}},

		{"--parallel -1 is auto, not one and not absent", "--parallel -1",
			LlamaCppArgs{Parallel: ParallelAuto}},
		{"-np -1 likewise", "-np -1", LlamaCppArgs{Parallel: ParallelAuto}},
		{"long kv-unified", "--kv-unified",
			LlamaCppArgs{KVUnified: true, KVUnifiedSet: true}},
		{"--no-kv-unified is an explicit off, distinct from absent", "--no-kv-unified",
			LlamaCppArgs{KVUnified: false, KVUnifiedSet: true}},
		{"-kvu does not consume the following flag", "-kvu --ctx-size 8192",
			LlamaCppArgs{CtxSize: 8192, KVUnified: true, KVUnifiedSet: true}},
		{"-kvu does take an on/off value where a build wants one", "-kvu off -c 8192",
			LlamaCppArgs{CtxSize: 8192, KVUnified: false, KVUnifiedSet: true}},
		{"--kv-unified=false", "--kv-unified=false",
			LlamaCppArgs{KVUnified: false, KVUnifiedSet: true}},
		{"per-slot limit", "--kv-unified-per-slot 32768 --parallel 4",
			LlamaCppArgs{Parallel: 4, PerSlotLimit: 32768}},
		{"per-slot limit with equals", "--kv-unified-per-slot=16384",
			LlamaCppArgs{PerSlotLimit: 16384}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ParseLlamaCppArgs(tt.args); got != tt.want {
				t.Errorf("ParseLlamaCppArgs(%q) = %+v, want %+v", tt.args, got, tt.want)
			}
		})
	}
}

func TestUnifiedKVDefault(t *testing.T) {
	// llama.cpp enables unified KV iff the slot count is auto, so the default
	// flips on the presence of --parallel alone. Getting this backwards is a
	// factor-of-the-slot-count error in whichever direction it lands.
	tests := []struct {
		args string
		want bool
	}{
		{"", true},
		{"-ngl 999", true},
		{"--parallel -1", true},
		{"--parallel 4", false},
		{"--parallel 1", false},
		{"--parallel 4 -kvu", true},
		{"--parallel -1 --no-kv-unified", false},
	}
	for _, tt := range tests {
		t.Run(tt.args, func(t *testing.T) {
			if got := ParseLlamaCppArgs(tt.args).UnifiedKV(); got != tt.want {
				t.Errorf("ParseLlamaCppArgs(%q).UnifiedKV() = %v, want %v", tt.args, got, tt.want)
			}
		})
	}
}

// End to end, from an argv string to the number ai-code plans against. The two
// cases differ only in one three-letter flag and by a factor of four.
func TestArgsToWindow(t *testing.T) {
	const (
		archLimit = 262144
		pool      = 262144
	)
	tests := []struct {
		name string
		args string
		want int
	}{
		{"split across four slots", `--ctx-size 262144 --parallel 4`, 65536},
		{"the same server with a unified cache", `--ctx-size 262144 --parallel 4 -kvu`, 262144},
		{"an explicit per-slot window wins over both", `--ctx-size 262144 --parallel 4 -kvu --kv-unified-per-slot 32768`, 32768},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := ParseLlamaCppArgs(tt.args)
			got, _ := EffectiveContext(ContextInputs{
				MaxContextWindow: archLimit,
				CtxSize:          max(a.CtxSize, pool),
				Parallel:         a.Parallel,
				KVUnified:        a.UnifiedKV(),
				PerSlotLimit:     a.PerSlotLimit,
			})
			if got != tt.want {
				t.Fatalf("effective context = %d, want %d", got, tt.want)
			}
		})
	}
}

// A slot window observed from /slots ends the chain: it is what the process
// serving the request actually has.
func TestObservedSlotWindowWins(t *testing.T) {
	a := ParseLlamaCppArgs(`--ctx-size 262144 --parallel 4 -kvu`)
	in := ContextInputs{
		MaxContextWindow: 262144,
		CtxSize:          a.CtxSize,
		Parallel:         a.Parallel,
		KVUnified:        a.UnifiedKV(),
		SlotNCtx:         40960,
	}
	got, estimated := EffectiveContext(in)
	if got != 40960 || estimated {
		t.Fatalf("EffectiveContext = (%d, estimated %v), want (40960, estimated false): "+
			"the launch flags said 262144, the server says otherwise and it is the one serving",
			got, estimated)
	}
}

// lemonadeTestServer answers /models with one llama.cpp-backed entry, /health
// with that model resident, and hands /slots to the caller so each test can
// decide what -- if anything -- that endpoint does.
func lemonadeTestServer(t *testing.T, slots http.HandlerFunc) *Lemonade {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			_, _ = io.WriteString(w, `{"data":[
			 {"id":"m","max_context_window":262144,"labels":["chat","tool-calling"],
			  "recipe_options":{"ctx_size":262144,"llamacpp_args":"--parallel 4 -ngl 999"}}]}`)
		case "/health":
			_, _ = io.WriteString(w, `{"model_loaded":"m","max_models":{"llm":1},
			 "all_models_loaded":[{"model_name":"m","loaded":true,"status":"ready"}]}`)
		case "/slots":
			slots(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)

	c, err := NewLemonade(Options{Name: "test", BaseURL: srv.URL, Class: ClassOnPrem})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestSlotsObservationOverridesArgvArithmetic(t *testing.T) {
	// Two slots, both narrower than the 262144/4 the launch arguments imply.
	// The server is the authority on what it will actually accept.
	c := lemonadeTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `[
		 {"id":0,"n_ctx":40960,"is_processing":true,"n_prompt_tokens":1200},
		 {"id":1,"n_ctx":40960,"is_processing":false,"n_prompt_tokens":17}]`)
	})
	models, err := c.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	m := models[0]
	if m.ContextWindow != 40960 || m.Estimated {
		t.Errorf("context window = %d (estimated %v), want 40960 observed; argv arithmetic said 65536",
			m.ContextWindow, m.Estimated)
	}
	if m.Slots != 2 || m.SlotsBusy != 1 {
		t.Errorf("slots = %d busy %d, want 2 busy 1", m.Slots, m.SlotsBusy)
	}
}

// Every way /slots can be missing is ordinary, and none of them may cost the
// user either an error or the rest of the model metadata.
func TestSlotsAbsenceDegradesSilently(t *testing.T) {
	cases := map[string]http.HandlerFunc{
		"404 on a backend that is not llama.cpp": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		},
		"501 from a build with the endpoint compiled out": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotImplemented)
		},
		"an error object where an array was expected": func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.WriteString(w, `{"error":{"message":"slots endpoint is disabled"}}`)
		},
		"a well-formed but empty array": func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.WriteString(w, `[]`)
		},
		"HTML from a proxy that does not forward this path": func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.WriteString(w, `<html><body>404</body></html>`)
		},
	}
	for name, h := range cases {
		t.Run(name, func(t *testing.T) {
			models, err := lemonadeTestServer(t, h).Models(context.Background())
			if err != nil {
				t.Fatalf("model listing failed because /slots did not answer: %v", err)
			}
			m := models[0]
			if m.ContextWindow != 65536 {
				t.Errorf("context window = %d, want the argv fallback 65536", m.ContextWindow)
			}
			if m.Slots != 0 {
				t.Errorf("slots = %d, want 0: nothing was observed", m.Slots)
			}
			if !m.Loaded {
				t.Error("residency from /health was lost along with /slots")
			}
		})
	}
}

// An auto slot count is the one case the launch arguments cannot answer, so an
// observed count is the only way Parallel is ever anything but a guess.
func TestSlotsResolveAutoParallel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			_, _ = io.WriteString(w, `{"data":[
			 {"id":"m","max_context_window":131072,
			  "recipe_options":{"ctx_size":131072,"llamacpp_args":"--parallel -1 -ngl 999"}}]}`)
		case "/health":
			_, _ = io.WriteString(w, `{"model_loaded":"m","all_models_loaded":[{"model_name":"m","loaded":true,"status":"ready"}]}`)
		case "/slots":
			_, _ = io.WriteString(w, `[{"id":0},{"id":1},{"id":2}]`)
		}
	}))
	defer srv.Close()

	c, err := NewLemonade(Options{Name: "test", BaseURL: srv.URL, Class: ClassOnPrem})
	if err != nil {
		t.Fatal(err)
	}
	models, err := c.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	m := models[0]
	if m.Slots != 3 || m.Parallel != 3 {
		t.Errorf("slots = %d, parallel = %d, want 3 and 3", m.Slots, m.Parallel)
	}
	// These slots report no n_ctx, and an auto slot count means unified KV, so
	// the window is the whole pool rather than a third of it.
	if m.ContextWindow != 131072 {
		t.Errorf("context window = %d, want 131072: a unified pool is not divided by the slot count",
			m.ContextWindow)
	}
}
