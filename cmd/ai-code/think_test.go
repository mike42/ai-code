package main

import (
	"context"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/session"
)

func thinkApp(t *testing.T) *App {
	t.Helper()
	return statusApp(t, provider.ModelInfo{ID: "a-model", ContextWindow: 262144})
}

// /think changes effort without a model swap, effective on the next request.
func TestThinkChangesTheLevelSentOnTheWire(t *testing.T) {
	a := thinkApp(t)
	if a.agent.Effort() != provider.EffortUnset {
		t.Fatalf("setup: started at %q", a.agent.Effort())
	}

	captureOut(t, func() { _ = a.cmdThink(context.Background(), "high") })
	if got := a.agent.Effort(); got != provider.EffortHigh {
		t.Errorf("after /think high the agent asks for %q", got)
	}

	captureOut(t, func() { _ = a.cmdThink(context.Background(), "none") })
	if got := a.agent.Effort(); got != provider.EffortNone {
		t.Errorf("after /think none the agent asks for %q", got)
	}
}

func TestThinkAcceptsTheUsualWordsAndRejectsTherest(t *testing.T) {
	a := thinkApp(t)
	for _, word := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		var err error
		captureOut(t, func() { err = a.cmdThink(context.Background(), word) })
		if err != nil {
			t.Errorf("/think %s: %v", word, err)
		}
	}
	for _, word := range []string{"quite hard", "off", "ultra", "banana"} {
		var err error
		captureOut(t, func() { err = a.cmdThink(context.Background(), word) })
		if err == nil {
			t.Errorf("/think %s was accepted; the levels are the OpenAI ones only", word)
		}
	}
}

func TestTheLevelAskedForIsTheLevelSent(t *testing.T) {
	a := thinkApp(t)
	captureOut(t, func() { _ = a.cmdThink(context.Background(), "high") })
	if got := a.agent.Effort(); got != provider.EffortHigh {
		t.Errorf("effort = %q, want the level as asked", got)
	}
}

// A level chosen mid-session must survive /restart, like the model and mode.
func TestThinkSurvivesRestart(t *testing.T) {
	a := thinkApp(t)
	a.flags = &flags{}
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	sess, err := session.Create(session.Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	a.sess = sess
	captureOut(t, func() { _ = a.cmdThink(context.Background(), "high") })

	argv := strings.Join(a.restartArgv("ai-code"), " ")
	if !strings.Contains(argv, "--think high") {
		t.Errorf("restart argv = %q, want the thinking level carried over", argv)
	}
}

// Summarising is mechanical: max-effort thinking would cost minutes on a
// local model while the session is already stuck.
func TestCompactionIgnoresTheSessionThinkingLevel(t *testing.T) {
	a := thinkApp(t)
	captureOut(t, func() { _ = a.cmdThink(context.Background(), "high") })

	client := &summaryClient{text: "## Goal\nship it"}
	a.client = client
	a.agent.SetClient(client)
	runIdle(t, a, 0)

	client.mu.Lock()
	defer client.mu.Unlock()
	if len(client.reqs) == 0 {
		t.Fatal("no summarisation request was made")
	}
	for _, r := range client.reqs {
		if r.Effort != provider.EffortNone {
			t.Errorf("a summarisation went out at effort %q, want off", r.Effort)
		}
	}
}

// A note appears only when the model is known to do something other than what
// was asked.
func TestThinkSaysNothingWhenThereIsNothingToSay(t *testing.T) {
	a := thinkApp(t)

	out := captureOut(t, func() { _ = a.cmdThink(context.Background(), "high") })
	if n := len(nonBlank(out)); n != 1 {
		t.Errorf("/think high printed %d lines, want just the level:\n%s", n, out)
	}
	if n := len(nonBlank(captureOut(t, func() { _ = a.cmdThink(context.Background(), "") }))); n != 2 {
		t.Errorf("/think printed %d lines, want the level and the choices", n)
	}
}
