package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/provider"
	"ai-code/internal/render"
	"ai-code/internal/session"
)

// fakeLemonade is a model server that holds what it has loaded and records
// every way of changing that. A chat request for a model it does not hold
// loads it, as Lemonade does: that is the swap these tests exist to catch.
type fakeLemonade struct {
	mu        sync.Mutex
	catalogue []string
	loaded    []string
	loads     []string // POST /load
	swaps     []string // chat requests that evicted something to load their model
	chats     []string // the model each chat request named
	healthErr bool
	// holdHealth, when set, makes /health wait until it is closed.
	holdHealth chan struct{}
}

func (f *fakeLemonade) serve(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/models":
			f.mu.Lock()
			var data []map[string]any
			for _, id := range f.catalogue {
				data = append(data, map[string]any{"id": id, "labels": []string{"tool-calling"}})
			}
			f.mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
		case r.URL.Path == "/health":
			if f.holdHealth != nil {
				<-f.holdHealth
			}
			f.mu.Lock()
			defer f.mu.Unlock()
			if f.healthErr {
				// Not a 5xx, which the client retries with backoff.
				http.Error(w, "no health endpoint", http.StatusNotFound)
				return
			}
			var all []map[string]any
			first := ""
			for _, id := range f.loaded {
				if first == "" {
					first = id
				}
				all = append(all, map[string]any{"model_name": id, "loaded": true, "status": "ready"})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"model_loaded": first, "all_models_loaded": all})
		case r.URL.Path == "/load":
			var body struct {
				Model string `json:"model_name"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			f.mu.Lock()
			f.loads = append(f.loads, body.Model)
			f.loaded = []string{body.Model}
			f.mu.Unlock()
		case r.URL.Path == "/chat/completions":
			var body struct {
				Model string `json:"model"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			f.mu.Lock()
			f.chats = append(f.chats, body.Model)
			if !slices.Contains(f.loaded, body.Model) {
				if len(f.loaded) > 0 {
					f.swaps = append(f.swaps, fmt.Sprintf("%s over %s", body.Model, strings.Join(f.loaded, ",")))
				}
				f.loaded = []string{body.Model}
			}
			f.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func (f *fakeLemonade) record() (loads, swaps, chats []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.loads), slices.Clone(f.swaps), slices.Clone(f.chats)
}

// lemonadeApp is a fresh launch against f, with the real Lemonade client and
// a data directory of its own. Nothing about the model is set: that is what
// the launch has to find out.
func lemonadeApp(t *testing.T, f *fakeLemonade) *App {
	t.Helper()
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	pc := config.Provider{Kind: "lemonade", BaseURL: f.serve(t), Class: "on-premises"}
	client, err := buildClient("home", pc)
	if err != nil {
		t.Fatal(err)
	}
	d := config.Defaults()
	a := &App{
		cfg:            &d,
		client:         client,
		exec:           noTools{},
		providerName:   "home",
		providerCfg:    pc,
		screen:         render.NewScreen(io.Discard, "never"),
		sessionProject: "/work/project",
	}
	a.agent = agent.New(client, "", noTools{}, agent.SinkFunc(a.noteContext), agent.Options{})
	return a
}

// usedHere records a past session in the project in which model answered.
func usedHere(t *testing.T, a *App, model string) {
	t.Helper()
	s, err := session.Create(session.Meta{Project: a.sessionProject, Provider: a.providerName})
	if err != nil {
		t.Fatal(err)
	}
	_ = s.Append(session.Entry{Type: session.EntryModel, Provider: a.providerName, Model: model})
	_ = s.AppendMessage(provider.Message{Role: provider.RoleUser, Content: "hi"})
	_ = s.AppendMessage(provider.Message{Role: provider.RoleAssistant, Content: "hello"})
	s.Close()
}

// The bug report: a model is loaded, ai-code is opened, "hello" is typed, and
// that must not load a different model. The catalogue lists another model
// first, and the cached copy of it predates the loaded one -- the two things
// that made the old startup pick the wrong model.
func TestHelloWithAModelLoadedCausesNoSwap(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"aaa-first-in-list", "the-loaded-one"}, loaded: []string{"the-loaded-one"}}
	a := lemonadeApp(t, f)
	saveModelCache("home", []provider.ModelInfo{{ID: "aaa-first-in-list", SupportsTools: true}})

	a.startModelPick(context.Background(), "")
	if err := a.handleInput(context.Background(), "hello"); err != nil {
		t.Fatalf("hello: %v", err)
	}

	loads, swaps, chats := f.record()
	if len(loads) > 0 || len(swaps) > 0 {
		t.Errorf("saying hello loaded %v and swapped %v; want the loaded model used as it is", loads, swaps)
	}
	if !slices.Equal(chats, []string{"the-loaded-one"}) {
		t.Errorf("chat requests named %v, want only the loaded model", chats)
	}
}

// /model names a model; one the server lists is available, whatever an old
// copy of the list on disk says.
func TestModelCommandFindsAModelTheCacheHasNotSeen(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"old", "new-today"}, loaded: []string{"old"}}
	a := lemonadeApp(t, f)
	saveModelCache("home", []provider.ModelInfo{{ID: "old", SupportsTools: true}})
	a.adoptModel(modelChoice{Model: provider.ModelInfo{ID: "old"}})

	if err := a.cmdModel(context.Background(), "new-today"); err != nil {
		t.Fatalf("/model new-today: %v", err)
	}
	if a.model.ID != "new-today" {
		t.Errorf("model = %q after /model new-today", a.model.ID)
	}
}

// --model is a demand, checked against what the server lists now.
func TestNamedModelIsCheckedAgainstTheLiveList(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"old", "new-today"}, loaded: []string{"old"}}
	a := lemonadeApp(t, f)
	saveModelCache("home", []provider.ModelInfo{{ID: "old", SupportsTools: true}})

	c, err := chooseModel(context.Background(), a.client, "new-today", "")
	if err != nil {
		t.Fatalf("--model new-today: %v", err)
	}
	if c.Model.ID != "new-today" || !c.Explicit {
		t.Errorf("choice = %+v, want new-today as the user's own demand", c)
	}
	if _, err := chooseModel(context.Background(), a.client, "no-such-model", ""); err == nil {
		t.Error("--model accepted a model the server does not list")
	}
}

// With nothing loaded and nothing used here before, there is no model, and a
// prompt is refused rather than sent somewhere; the text comes back to the
// prompt.
func TestNothingLoadedAndNoHistoryRefusesToSend(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"a", "b"}}
	a := lemonadeApp(t, f)

	a.startModelPick(context.Background(), "")
	err := a.handleInput(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "/model NAME") {
		t.Fatalf("hello = %v, want a refusal that says how to choose", err)
	}
	if loads, swaps, chats := f.record(); len(loads)+len(swaps)+len(chats) > 0 {
		t.Errorf("a refused prompt reached the server: loads %v, swaps %v, chats %v", loads, swaps, chats)
	}
	if a.pendingInput != "hello" {
		t.Errorf("the typed text is gone: pending input %q", a.pendingInput)
	}
}

// With nothing loaded, the model last used in this project comes back. No
// model is evicted to load it.
func TestNothingLoadedRevivesTheModelLastUsedHere(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"aaa-first", "my-model"}}
	a := lemonadeApp(t, f)
	usedHere(t, a, "my-model")

	a.startModelPick(context.Background(), "")
	if err := a.handleInput(context.Background(), "hello"); err != nil {
		t.Fatalf("hello: %v", err)
	}
	if _, swaps, chats := f.record(); len(swaps) > 0 || !slices.Equal(chats, []string{"my-model"}) {
		t.Errorf("chats %v, swaps %v; want my-model, loaded into an empty server", chats, swaps)
	}
}

// A server that cannot say what it holds gets no guess.
func TestAnUnansweredHealthCheckPicksNothing(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"a"}, loaded: []string{"a"}, healthErr: true}
	a := lemonadeApp(t, f)
	usedHere(t, a, "a")

	if c, err := chooseModel(context.Background(), a.client, "", "a"); err == nil {
		t.Errorf("chose %q without knowing what is loaded", c.Model.ID)
	}
	a.startModelPick(context.Background(), "")
	if err := a.handleInput(context.Background(), "hello"); err == nil {
		t.Error("a prompt was sent while the server could not say what it holds")
	}
	if _, _, chats := f.record(); len(chats) > 0 {
		t.Errorf("chat requests %v reached the server", chats)
	}
}

// Several models loaded: the one used here is taken if it is among them;
// otherwise the user chooses.
func TestSeveralLoadedTakesTheOneUsedHereOrAsks(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"a", "b", "c"}, loaded: []string{"a", "c"}}
	a := lemonadeApp(t, f)

	if _, err := chooseModel(context.Background(), a.client, "", ""); !isNoModel(err) {
		t.Errorf("two loaded, none used here: %v, want to be asked", err)
	}
	if _, err := chooseModel(context.Background(), a.client, "", "b"); !isNoModel(err) {
		t.Errorf("two loaded, a third used here: %v, want to be asked", err)
	}
	c, err := chooseModel(context.Background(), a.client, "", "c")
	if err != nil || c.Model.ID != "c" {
		t.Errorf("two loaded, c used here: %q, %v; want c", c.Model.ID, err)
	}
}

// The prompt comes first. Asking the server happens behind it, and only a
// send waits for the answer.
func TestThePromptDoesNotWaitForTheServer(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"m"}, loaded: []string{"m"}, holdHealth: make(chan struct{})}
	a := lemonadeApp(t, f)

	a.startModelPick(context.Background(), "") // returns with /health still unanswered

	sent := make(chan error, 1)
	go func() { sent <- a.handleInput(context.Background(), "hello") }()
	select {
	case err := <-sent:
		t.Fatalf("the prompt was handled before the server said what it holds: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(f.holdHealth)
	if err := <-sent; err != nil {
		t.Fatalf("hello: %v", err)
	}
	if _, _, chats := f.record(); !slices.Equal(chats, []string{"m"}) {
		t.Errorf("chats = %v, want the loaded model once the server answered", chats)
	}
}

// Something else loaded a different model since this session chose one. The
// prompt goes to what is loaded rather than loading its model back.
func TestThePromptFollowsTheServerRatherThanLoadingBack(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"mine", "theirs"}, loaded: []string{"theirs"}}
	a := lemonadeApp(t, f)
	a.adoptModel(modelChoice{Model: provider.ModelInfo{ID: "mine"}})

	if err := a.handleInput(context.Background(), "hello"); err != nil {
		t.Fatalf("hello: %v", err)
	}
	loads, swaps, chats := f.record()
	if len(loads)+len(swaps) > 0 || !slices.Equal(chats, []string{"theirs"}) {
		t.Errorf("loads %v, swaps %v, chats %v; want the prompt sent to theirs", loads, swaps, chats)
	}
	if a.model.ID != "theirs" {
		t.Errorf("model = %q, want the session on what the server holds", a.model.ID)
	}
}

// A model the user named may replace the loaded one; that is what naming it
// asks for. Once loaded it is an ordinary choice again.
func TestANamedModelMayReplaceTheLoadedOne(t *testing.T) {
	f := &fakeLemonade{catalogue: []string{"loaded", "wanted"}, loaded: []string{"loaded"}}
	a := lemonadeApp(t, f)

	a.startModelPick(context.Background(), "wanted")
	if err := a.handleInput(context.Background(), "hello"); err != nil {
		t.Fatalf("hello: %v", err)
	}
	if _, _, chats := f.record(); !slices.Equal(chats, []string{"wanted"}) {
		t.Errorf("chats = %v, want the named model", chats)
	}
	if err := a.handleInput(context.Background(), "again"); err != nil {
		t.Fatalf("again: %v", err)
	}
	if a.modelExplicit {
		t.Error("still marked as a demand after the server loaded it")
	}
}
