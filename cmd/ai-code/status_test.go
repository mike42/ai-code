package main

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/render"
)

// statusApp builds an App whose output can be read back as the user sees it.
// Without a terminal a.out falls through to stdout, so stdout is what the
// test has to capture.
func statusApp(t *testing.T, model provider.ModelInfo) *App {
	t.Helper()
	a, _ := swapApp(t, model.ContextWindow, 200000)
	a.screen = render.NewScreen(io.Discard, "never")
	a.cwd = "/work"
	a.model = model
	return a
}

func captureOut(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved := os.Stdout
	os.Stdout = w

	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()

	f()

	os.Stdout = saved
	w.Close()
	out := <-done
	r.Close()
	return out
}

func dividedModel() provider.ModelInfo {
	return provider.ModelInfo{
		ID: "a-model", ContextWindow: 262144, CtxSize: 786432, Parallel: 3,
	}
}

// The startup banner reports the machine; it does not advise on it. How many
// slots to run is a trade-off already made, usually for a reason ai-code
// cannot see, and re-litigating it every session is noise.
func TestBannerStatesTheSlotsWithoutAdvice(t *testing.T) {
	a := statusApp(t, dividedModel())
	out := captureOut(t, a.banner)

	for _, advice := range []string{"--parallel", "--kv-unified", "Reduce", "if you want"} {
		if strings.Contains(out, advice) {
			t.Errorf("the banner tells the user how to reconfigure their server (%q):\n%s", advice, out)
		}
	}
	for _, fact := range []string{"a-model", "262k", "786k", "3 slots"} {
		if !strings.Contains(out, fact) {
			t.Errorf("the banner omits %q, which is one of the facts it is for:\n%s", fact, out)
		}
	}
	if n := len(nonBlank(out)); n != 2 {
		t.Errorf("the banner is %d lines, want 2:\n%s", n, out)
	}
}

func TestBannerSaysNothingAboutSlotsWhenThereIsOne(t *testing.T) {
	a := statusApp(t, provider.ModelInfo{ID: "a-model", ContextWindow: 262144})
	out := captureOut(t, a.banner)
	if strings.Contains(out, "slot") {
		t.Errorf("an undivided window should not mention slots:\n%s", out)
	}
}

// /tokens answers how full the context is and when something happens about
// it; the two numbers are the whole answer, and anything else is derivable
// from them or a restatement of the config file.
func TestTokensIsTwoLines(t *testing.T) {
	a := statusApp(t, dividedModel())
	var err error
	out := captureOut(t, func() { err = a.cmdTokens(context.Background(), "") })
	if err != nil {
		t.Fatal(err)
	}

	if n := len(nonBlank(out)); n > 2 {
		t.Errorf("/tokens printed %d lines, want at most 2:\n%s", n, out)
	}
	if !strings.Contains(out, "of 262k for the next request") {
		t.Errorf("/tokens does not report the next request against the window:\n%s", out)
	}
	if !strings.Contains(out, "summarises at") {
		t.Errorf("/tokens does not say when auto-compaction happens:\n%s", out)
	}
	for _, gone := range []string{"Messages", "Reserve", "Output", "Budget"} {
		if strings.Contains(out, gone) {
			t.Errorf("/tokens still prints the %s line:\n%s", gone, out)
		}
	}
}

func TestTokensSaysWhenItWillStopInsteadOfSummarising(t *testing.T) {
	a := statusApp(t, dividedModel())
	off := false
	a.cfg.Agent.AutoCompact = &off

	out := captureOut(t, func() { _ = a.cmdTokens(context.Background(), "") })
	if !strings.Contains(out, "stops at") {
		t.Errorf("with auto_compact off, /tokens should say it stops:\n%s", out)
	}
}

func nonBlank(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, l)
		}
	}
	return out
}

// The marker stays a marker. Which model answered is recorded in the
// scrollback when it changes, where it stays readable next to the prompts it
// applies to; the prompt line is not the place for it.
func TestThePromptMarkerCarriesNoModelName(t *testing.T) {
	a := statusApp(t, provider.ModelInfo{ID: "small-model", ContextWindow: 65536})
	a.modeName = "build"

	if got := render.Strip(a.promptString()); got != "> " {
		t.Errorf("prompt = %q, want the bare marker", got)
	}

	a.model = provider.ModelInfo{ID: "some-other-model", ContextWindow: 262144}
	if got := render.Strip(a.promptString()); strings.Contains(got, "some-other-model") {
		t.Errorf("prompt = %q, want no model name on it", got)
	}
}

// A slash command typed mid-turn is a command, not something to send to the
// model as a correction.
func TestCommandsThatEndTheRun(t *testing.T) {
	for _, line := range []string{"/model foo", "/compact", "/new", "/mode review", "/prov openai"} {
		if !commandEndsTheRun(line) {
			t.Errorf("%q should end the run: continuing would answer the old question with the new setup", line)
		}
	}
	for _, line := range []string{"/think high", "/tokens", "/verbose", "/quiet", "/help", "/tools"} {
		if commandEndsTheRun(line) {
			t.Errorf("%q should not end the run", line)
		}
	}
	if commandEndsTheRun("/nonsense") {
		t.Error("an unknown command should not end the run; it reports an error instead")
	}
}

func TestQueuedCommandsRunInOrderAndClear(t *testing.T) {
	a := statusApp(t, provider.ModelInfo{ID: "a-model", ContextWindow: 65536})

	a.queueCommand("/think high")
	a.queueCommand("/tokens")
	if n := a.pendingCommands(); n != 2 {
		t.Fatalf("queued %d commands, want 2", n)
	}

	captureOut(t, func() { a.runQueuedCommands(context.Background()) })

	if n := a.pendingCommands(); n != 0 {
		t.Errorf("%d commands left queued after running them", n)
	}
	if got := a.agent.Effort(); got != provider.EffortHigh {
		t.Errorf("the queued /think did not take effect: effort = %q", got)
	}
}

// One that changes the setup stops the run; the rest let it carry on.
func TestAQueuedModelChangeStopsTheRun(t *testing.T) {
	a := statusApp(t, provider.ModelInfo{ID: "a-model", ContextWindow: 65536})

	a.queueCommand("/think low")
	var stop bool
	captureOut(t, func() { stop = a.runQueuedCommands(context.Background()) })
	if stop {
		t.Error("/think should not stop the run")
	}

	a.queueCommand("/mode review")
	captureOut(t, func() { stop = a.runQueuedCommands(context.Background()) })
	if !stop {
		t.Error("/mode should stop the run")
	}
}

// A line submitted during a swap is a prompt, not a correction to a turn
// that is not running. Steering's usual action would queue it against a turn
// that does not exist, where it would sit unanswered.
func TestALineSubmittedDuringASwapBecomesTheNextPrompt(t *testing.T) {
	a := statusApp(t, provider.ModelInfo{ID: "a-model", ContextWindow: 65536})

	a.queuePrompt("  what changed in the parser?  ")
	a.queuePrompt("")
	a.queuePrompt("and why")

	got, ok := a.takeDeferredPrompt()
	if !ok || got != "what changed in the parser?" {
		t.Errorf("first deferred prompt = %q, %v", got, ok)
	}
	got, ok = a.takeDeferredPrompt()
	if !ok || got != "and why" {
		t.Errorf("second deferred prompt = %q, %v; a blank line is not a prompt", got, ok)
	}
	if _, ok := a.takeDeferredPrompt(); ok {
		t.Error("more prompts came back than were submitted")
	}
	if n := a.agent.Pending(); n != 0 {
		t.Errorf("%d went to the steering queue, where nothing would have run them", n)
	}
}
