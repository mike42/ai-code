package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// encodeBody marshals a request exactly as Stream would, without a network.
func encodeBody(t *testing.T, opts Options, req Request) map[string]any {
	t.Helper()
	if opts.BaseURL == "" {
		opts.BaseURL = "http://provider.invalid/v1"
	}
	if opts.Name == "" {
		opts.Name = "test"
	}
	c, err := NewOpenAI(opts)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	raw, err := json.Marshal(c.encode(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return out
}

func assertAbsent(t *testing.T, body map[string]any, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if v, ok := body[k]; ok {
			t.Errorf("%q should be absent, got %#v", k, v)
		}
	}
}

func TestEncodeEffort(t *testing.T) {
	tests := []struct {
		name    string
		dialect Dialect
		effort  Effort
		want    map[string]any
		absent  []string
	}{{
		name:   "unset sends nothing",
		effort: EffortUnset,
		absent: []string{"reasoning_effort", "reasoning", "chat_template_kwargs"},
	}, {
		name:   "off is the hard disable",
		effort: EffortNone,
		want:   map[string]any{"reasoning_effort": "none"},
		absent: []string{"reasoning"},
	}, {
		name:   "low passes through",
		effort: EffortLow,
		want:   map[string]any{"reasoning_effort": "low"},
	}, {
		name:   "max is not a spec value and degrades to high",
		effort: EffortHigh,
		want:   map[string]any{"reasoning_effort": "high"},
	}, {
		name:    "openrouter unset sends nothing",
		dialect: DialectOpenRouter,
		effort:  EffortUnset,
		absent:  []string{"reasoning_effort", "reasoning"},
	}, {
		name:    "openrouter nests the effort",
		dialect: DialectOpenRouter,
		effort:  EffortHigh,
		want:    map[string]any{"reasoning": map[string]any{"effort": "high"}},
		absent:  []string{"reasoning_effort"},
	}, {
		name:    "openrouter max degrades to high",
		dialect: DialectOpenRouter,
		effort:  EffortHigh,
		want:    map[string]any{"reasoning": map[string]any{"effort": "high"}},
	}, {
		name:    "openrouter off disables rather than sending none",
		dialect: DialectOpenRouter,
		effort:  EffortNone,
		want:    map[string]any{"reasoning": map[string]any{"enabled": false}},
		absent:  []string{"reasoning_effort"},
	}}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := encodeBody(t, Options{Dialect: tc.dialect, Class: ClassOnPrem}, Request{
				Model:    "m",
				Messages: []Message{{Role: RoleUser, Content: "hi"}},
				Effort:   tc.effort,
			})
			for k, want := range tc.want {
				got, ok := body[k]
				if !ok {
					t.Fatalf("%q missing from body %v", k, body)
				}
				if !jsonEqual(got, want) {
					t.Errorf("%q = %#v, want %#v", k, got, want)
				}
			}
			assertAbsent(t, body, tc.absent...)
		})
	}
}

func TestEncodeTemplateKwargs(t *testing.T) {
	body := encodeBody(t, Options{Class: ClassOnPrem}, Request{
		Model:          "m",
		Messages:       []Message{{Role: RoleUser, Content: "hi"}},
		Effort:         EffortNone,
		TemplateKwargs: map[string]any{"enable_thinking": false, "reasoning_format": "deepseek"},
	})
	want := map[string]any{"enable_thinking": false, "reasoning_format": "deepseek"}
	if !jsonEqual(body["chat_template_kwargs"], want) {
		t.Errorf("chat_template_kwargs = %#v, want %#v", body["chat_template_kwargs"], want)
	}

	// The kwargs must not depend on an effort level being set: they are how a
	// caller reaches modes the ladder cannot name.
	body = encodeBody(t, Options{Class: ClassOnPrem}, Request{
		Model:          "m",
		Messages:       []Message{{Role: RoleUser, Content: "hi"}},
		TemplateKwargs: map[string]any{"enable_thinking": false},
	})
	if !jsonEqual(body["chat_template_kwargs"], map[string]any{"enable_thinking": false}) {
		t.Errorf("kwargs dropped without an effort level: %#v", body["chat_template_kwargs"])
	}
	assertAbsent(t, body, "reasoning_effort")
}

func TestEncodeReasoningRoundTrip(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: "go"},
		{Role: RoleAssistant, Reasoning: "prior thought", ToolCalls: []ToolCall{
			{ID: "c1", Name: "ls", Args: "{}"},
		}},
		{Role: RoleTool, ToolCallID: "c1", Name: "ls", Content: "a.go"},
	}

	for _, tc := range []struct {
		name    string
		opts    Options
		field   string
		present bool
	}{
		{"off omits it", Options{Class: ClassOnPrem}, "reasoning_content", false},
		{"on uses reasoning_content", Options{Class: ClassOnPrem, SendReasoning: true}, "reasoning_content", true},
		{"openrouter off omits it", Options{Class: ClassCloud, Dialect: DialectOpenRouter}, "reasoning", false},
		{"openrouter on uses reasoning", Options{Class: ClassCloud, Dialect: DialectOpenRouter, SendReasoning: true}, "reasoning", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := encodeBody(t, tc.opts, Request{Model: "m", Messages: msgs})
			out, _ := body["messages"].([]any)
			if len(out) != len(msgs) {
				t.Fatalf("got %d messages, want %d", len(out), len(msgs))
			}
			assistant, _ := out[1].(map[string]any)
			got, ok := assistant[tc.field]
			if tc.present {
				if !ok || got != "prior thought" {
					t.Errorf("assistant %q = %#v, want %q", tc.field, got, "prior thought")
				}
			} else if ok {
				t.Errorf("assistant %q should be absent, got %#v", tc.field, got)
			}
			// The other dialect's spelling must never appear alongside it.
			other := "reasoning"
			if tc.field == "reasoning" {
				other = "reasoning_content"
			}
			if v, ok := assistant[other]; ok {
				t.Errorf("wrong dialect field %q present: %#v", other, v)
			}
			for _, i := range []int{0, 2} {
				m, _ := out[i].(map[string]any)
				if _, ok := m["reasoning_content"]; ok {
					t.Errorf("non-assistant message %d carries reasoning_content", i)
				}
				if _, ok := m["reasoning"]; ok {
					t.Errorf("non-assistant message %d carries reasoning", i)
				}
			}
		})
	}
}

func jsonEqual(a, b any) bool {
	x, err1 := json.Marshal(a)
	y, err2 := json.Marshal(b)
	return err1 == nil && err2 == nil && string(x) == string(y)
}

// The OpenRouter catalogue is public, and the key is what turns a routine
// listing into an attributable contact with a third party. It must not ride
// along on a call that does not need it.
func TestTheOpenRouterCatalogueIsReadAnonymously(t *testing.T) {
	cases := []struct {
		name    string
		dialect Dialect
		want    string
	}{
		{"openrouter", DialectOpenRouter, ""},
		{"any other endpoint", DialectOpenAI, "Bearer secret-key"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var seen string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/models" {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				seen = r.Header.Get("Authorization")
				_, _ = io.WriteString(w, `{"data":[{"id":"m","context_length":1024}]}`)
			}))
			t.Cleanup(srv.Close)

			c, err := NewOpenAI(Options{
				Name: "test", BaseURL: srv.URL, Class: ClassCloud,
				APIKey: "secret-key", Dialect: tc.dialect,
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Models(context.Background()); err != nil {
				t.Fatal(err)
			}
			if seen != tc.want {
				t.Errorf("Authorization on /models = %q, want %q", seen, tc.want)
			}
		})
	}
}

// Withholding the key from the catalogue must not withhold it from the one
// call that is actually the user's account being spent.
func TestInferenceStillSendsTheKeyOnOpenRouter(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	c, err := NewOpenAI(Options{
		Name: "test", BaseURL: srv.URL, Class: ClassCloud,
		APIKey: "secret-key", Dialect: DialectOpenRouter,
	})
	if err != nil {
		t.Fatal(err)
	}
	st, err := c.Stream(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := st.Recv(); err != nil {
			break
		}
	}
	if seen != "Bearer secret-key" {
		t.Errorf("Authorization on /chat/completions = %q, want the key", seen)
	}
}
