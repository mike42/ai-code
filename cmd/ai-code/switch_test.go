package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/provider"
	"ai-code/internal/render"
	"ai-code/internal/tool"
)

// countingClient records what the swap path asked of the provider. Counting
// rather than failing: a client that refuses to be called can only report the
// violation as a panic somewhere unrelated.
type countingClient struct {
	mu      sync.Mutex
	streams int
	models  int
}

func (c *countingClient) Name() string          { return "counting" }
func (c *countingClient) Class() provider.Class { return provider.ClassOnPrem }

func (c *countingClient) Stream(context.Context, provider.Request) (provider.Stream, error) {
	c.mu.Lock()
	c.streams++
	c.mu.Unlock()
	return &emptyStream{}, nil
}

func (c *countingClient) Models(context.Context) ([]provider.ModelInfo, error) {
	c.mu.Lock()
	c.models++
	c.mu.Unlock()
	return nil, nil
}

func (c *countingClient) calls() (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.streams, c.models
}

type emptyStream struct{}

func (s *emptyStream) Recv() (provider.Event, error) { return provider.Event{}, io.EOF }
func (s *emptyStream) Message() provider.Message {
	return provider.Message{Role: provider.RoleAssistant, Content: "done"}
}
func (s *emptyStream) Usage() provider.Usage           { return provider.Usage{} }
func (s *emptyStream) StopReason() provider.StopReason { return provider.StopEnd }
func (s *emptyStream) Close() error                    { return nil }

type noTools struct{}

func (noTools) Definitions() []tool.Definition { return nil }

func (n noTools) Fork() (tool.Executor, error) { return n, nil }
func (noTools) Execute(context.Context, tool.Request) (tool.Result, error) {
	return tool.Result{}, nil
}

// swapApp is an App wired to a counting client, with a session of roughly
// sessionChars characters already in it.
func swapApp(t *testing.T, oldLimit, sessionChars int) (*App, *countingClient) {
	t.Helper()
	d := config.Defaults()
	cfg := &d
	client := &countingClient{}

	app := &App{
		cfg:          cfg,
		client:       client,
		exec:         noTools{},
		providerName: "local",
		model:        provider.ModelInfo{ID: "wide-model", ContextWindow: oldLimit},
		screen:       render.NewScreen(io.Discard, "never"),
	}
	// Wired as main.go wires it: the App learns the context figure from the
	// event stream, never by asking the agent.
	app.agent = agent.New(client, "wide-model", noTools{},
		agent.SinkFunc(app.noteContext), agent.Options{
			ContextLimit:  oldLimit,
			ReserveTokens: cfg.Agent.CompactReserveTokens,
		})
	app.agent.SetMessages(conversation(sessionChars))
	return app, client
}

// conversation builds a well-formed exchange of about n characters.
func conversation(n int) []provider.Message {
	body := strings.Repeat("the quick brown fox jumps over the lazy dog. ", 40)
	var msgs []provider.Message
	for chars := 0; chars < n; chars += 2 * len(body) {
		i := len(msgs)
		msgs = append(msgs,
			provider.Message{Role: provider.RoleUser, Content: fmt.Sprintf("request %d: %s", i, body)},
			provider.Message{Role: provider.RoleAssistant, Content: fmt.Sprintf("reply %d: %s", i, body)})
	}
	return msgs
}

func narrowModel(limit int) provider.ModelInfo {
	return provider.ModelInfo{ID: "narrow-model", ContextWindow: limit}
}

// swap runs switchTo with the given keystrokes on stdin and returns everything
// it wrote.
func swap(t *testing.T, a *App, model provider.ModelInfo, answer string) string {
	t.Helper()

	in, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := in.WriteString(answer); err != nil {
		t.Fatal(err)
	}
	if _, err := in.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	stdin, stdout, stderr := os.Stdin, os.Stdout, os.Stderr
	os.Stdin, os.Stdout, os.Stderr = in, w, w

	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()

	err = a.switchTo(context.Background(), a.client, "local", config.Provider{}, model, false)

	os.Stdin, os.Stdout, os.Stderr = stdin, stdout, stderr
	_ = w.Close()
	out := <-done
	_ = r.Close()
	if err != nil {
		t.Fatalf("switchTo: %v", err)
	}
	return out
}

const askMarker = "What the next request should carry"

// A wider window takes nothing away, so there is nothing to decide.
func TestSwapToRoomierWindowAsksNothing(t *testing.T) {
	a, _ := swapApp(t, 32768, 120000)

	out := swap(t, a, provider.ModelInfo{ID: "wider-model", ContextWindow: 262144}, "")

	if strings.Contains(out, askMarker) {
		t.Errorf("a roomier window should not prompt:\n%s", out)
	}
	if a.model.ID != "wider-model" {
		t.Errorf("model is %q, want wider-model", a.model.ID)
	}
}

func TestSwapThatStillFitsAsksNothing(t *testing.T) {
	a, _ := swapApp(t, 262144, 4000)

	out := swap(t, a, narrowModel(131072), "")

	if strings.Contains(out, askMarker) {
		t.Errorf("a window the session still fits should not prompt:\n%s", out)
	}
	if a.model.ID != "narrow-model" {
		t.Errorf("model is %q, want narrow-model", a.model.ID)
	}
}

func TestNarrowerWindowStatesConsequences(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	a.agent.SetSummary("## Goal\nfinish the thing", 20)

	plan := a.agent.PlanResume(32768)
	if plan.Fits {
		t.Fatal("the fixture is meant not to fit a 32k window")
	}

	out := swap(t, a, narrowModel(32768), "c\n")

	if !strings.Contains(out, askMarker) {
		t.Fatalf("expected a choice:\n%s", out)
	}
	// Each option has to state what it sends, what is left to work in, and
	// how much of the session stops being sent.
	for _, want := range []string{
		tokenCount(plan.Session),
		tokenCount(plan.Checkpoint.Prompt),
		tokenCount(plan.Checkpoint.Free),
		tokenCount(plan.Transcript.Prompt),
		tokenCount(plan.Transcript.Free),
		fmt.Sprintf("last %d messages", plan.Checkpoint.Kept),
		fmt.Sprintf("%d earlier messages", plan.Checkpoint.Dropped),
		fmt.Sprintf("%d before them are dropped outright", plan.Transcript.Dropped),
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the choice does not state %q:\n%s", want, out)
		}
	}
	if plan.Checkpoint.Free <= 0 || plan.Transcript.Free <= 0 {
		t.Errorf("both options should leave room to answer in: %+v", plan)
	}
}

func TestTranscriptChoiceDropsTheCheckpoint(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	a.agent.SetSummary("## Goal\nfinish the thing", 20)

	swap(t, a, narrowModel(32768), "t\n")

	for _, m := range a.agent.MessagesFitting(8000) {
		if strings.Contains(m.Content, "<context-checkpoint>") {
			t.Fatal("the transcript choice still sent the checkpoint")
		}
	}
}

// With no checkpoint stored, offering one would mean a model call; the dialog
// says so instead.
func TestCheckpointOptionAbsentWithoutOne(t *testing.T) {
	a, client := swapApp(t, 262144, 200000)

	out := swap(t, a, narrowModel(32768), "t\n")

	if !strings.Contains(out, "no checkpoint has been written") {
		t.Errorf("expected the checkpoint option to explain its absence:\n%s", out)
	}
	if !strings.Contains(out, "[t/w]") || strings.Contains(out, "[c/t/w]") {
		t.Errorf("c should not be an accepted key when there is no checkpoint:\n%s", out)
	}
	if streams, _ := client.calls(); streams != 0 {
		t.Errorf("the swap made %d provider calls; it must make none", streams)
	}
}

// A swap makes no provider call: a summary here would run on the outgoing
// model while the caller waits.
func TestSwapMakesNoProviderCall(t *testing.T) {
	for _, answer := range []string{"c\n", "t\n", "w\n", ""} {
		a, client := swapApp(t, 262144, 200000)
		a.agent.SetSummary("## Goal\nfinish the thing", 20)

		swap(t, a, narrowModel(32768), answer)

		streams, models := client.calls()
		if streams != 0 || models != 0 {
			t.Errorf("answer %q: %d streams and %d catalogue calls, want none",
				answer, streams, models)
		}
	}
}

func TestWaitStaysOnTheCurrentModel(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)
	a.agent.SetSummary("## Goal\nfinish the thing", 20)

	out := swap(t, a, narrowModel(32768), "w\n")

	if a.model.ID != "wide-model" || a.agent.Model() != "wide-model" {
		t.Errorf("wait left the session on %q/%q, want wide-model", a.model.ID, a.agent.Model())
	}
	if got := a.agent.ContextState().Window; got != 262144 {
		t.Errorf("wait changed the window to %d", got)
	}
	if !strings.Contains(out, "Staying on") {
		t.Errorf("wait should say what it did:\n%s", out)
	}
}

func TestUnreadableAnswerStaysPut(t *testing.T) {
	a, _ := swapApp(t, 262144, 200000)

	swap(t, a, narrowModel(32768), "")

	if a.model.ID != "wide-model" {
		t.Errorf("an unreadable answer switched to %q", a.model.ID)
	}
}

func TestAutoResume(t *testing.T) {
	interrupted := func() []provider.Message {
		return append(conversation(4000),
			provider.Message{Role: provider.RoleAssistant, ToolCalls: []provider.ToolCall{
				{ID: "1", Name: "read", Args: `{"path":"x"}`}}},
			provider.Message{Role: provider.RoleTool, ToolCallID: "1", Content: "interrupted"})
	}

	cases := []struct {
		name     string
		model    string
		midTurn  bool
		wantRuns int
	}{
		{"same model, stopped mid-turn", "wide-model", true, 1},
		{"same model, turn finished", "wide-model", false, 0},
		{"different model, stopped mid-turn", "other-model", true, 0},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a, client := swapApp(t, 262144, 4000)
			if c.midTurn {
				a.agent.SetMessages(interrupted())
			}

			swap(t, a, provider.ModelInfo{ID: c.model, ContextWindow: 262144}, "")

			if streams, _ := client.calls(); streams != c.wantRuns {
				t.Errorf("%d turns ran, want %d", streams, c.wantRuns)
			}
		})
	}
}

// cloudProviderApp is a session on a local provider, with a cloud provider
// configured but not selected. The returned counter is every HTTP request
// that cloud provider received.
func cloudProviderApp(t *testing.T) (*App, *int32) {
	t.Helper()
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		_, _ = io.WriteString(w, `{"data":[{"id":"remote-model","context_length":1024}]}`)
	}))
	t.Cleanup(srv.Close)

	app, _ := swapApp(t, 65536, 4000)
	app.cfg.Provider = map[string]config.Provider{
		"remote": {
			Kind:         "openrouter",
			BaseURL:      srv.URL,
			Class:        string(provider.ClassCloud),
			APIKey:       "secret-key",
			DefaultModel: "remote-model",
		},
	}
	// A cached catalogue would answer the request this test counts, so point
	// the data directory somewhere empty.
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	return app, &hits
}

// answering runs fn with the given keystrokes on stdin.
func answering(t *testing.T, keys string, fn func()) {
	t.Helper()
	in, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := in.WriteString(keys); err != nil {
		t.Fatal(err)
	}
	if _, err := in.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	saved := os.Stdin
	os.Stdin = in
	defer func() { os.Stdin = saved }()
	fn()
}

// The refusal must come before any request reaches the provider: reading its
// catalogue would already be the contact .nocloud forbids.
func TestANoCloudTreeContactsNothingBeforeRefusing(t *testing.T) {
	app, hits := cloudProviderApp(t)
	app.noCloud = true

	err := app.cmdProvider(context.Background(), "remote")
	if err == nil {
		t.Fatal("a .nocloud tree accepted a cloud provider")
	}
	if n := atomic.LoadInt32(hits); n != 0 {
		t.Errorf("the provider received %d requests before being refused, want none", n)
	}
}

// "Nothing was sent" must hold, including the catalogue read needed to name
// the model in the prompt.
func TestDecliningTheCloudSwitchSendsNothing(t *testing.T) {
	app, hits := cloudProviderApp(t)

	var err error
	answering(t, "n\n", func() { err = app.cmdProvider(context.Background(), "remote") })
	if err != nil {
		t.Fatalf("cmdProvider: %v", err)
	}
	if n := atomic.LoadInt32(hits); n != 0 {
		t.Errorf("the provider received %d requests although the move was declined, want none", n)
	}
	if app.providerName != "local" {
		t.Errorf("the session moved to %q after declining", app.providerName)
	}
}
