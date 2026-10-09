package main

import (
	"context"
	"testing"
	"time"

	"ai-code/internal/provider"
	"ai-code/internal/session"
)

// cancelMidRequest starts a turn whose first request hangs, queues steering
// while it does, and cancels it as the given caller would.
func cancelMidRequest(t *testing.T, fromSteeringPrompt bool, steering ...string) (*App, *scriptClient) {
	t.Helper()
	c := &scriptClient{hold: map[int]chan struct{}{0: make(chan struct{})}}
	a := scriptedApp(t, c, 0)

	run := make(chan error, 1)
	go func() { run <- a.runTurn(context.Background(), "go") }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		if s, _ := c.count(); s == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the turn never sent its request")
		}
		time.Sleep(time.Millisecond)
	}

	for _, text := range steering {
		a.agent.Steer(text)
	}
	if fromSteeringPrompt {
		a.steerInterrupted.Store(true)
	}
	a.cancelTurn()

	select {
	case err := <-run:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the cancelled turn never ended")
	}
	return a, c
}

func userMessages(a *App) []string {
	var out []string
	for _, m := range a.agent.Messages() {
		if m.Role == provider.RoleUser {
			out = append(out, m.Content)
		}
	}
	return out
}

// The user's scenario: a long tool call, steering typed during it, Ctrl-C at
// the steering prompt to make the model listen. The steering is sent at once.
func TestCtrlCAtTheSteeringPromptSendsWhatWasQueued(t *testing.T) {
	a, c := cancelMidRequest(t, true, "use the other file", "and skip the tests")

	if s, _ := c.count(); s != 2 {
		t.Fatalf("made %d requests, want 2: the cancelled one and one carrying the steering", s)
	}
	got := userMessages(a)
	want := []string{"go", "use the other file", "and skip the tests"}
	if len(got) < len(want) {
		t.Fatalf("user messages %q, want %q", got, want)
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("user messages %q, want %q", got, want)
		}
	}
	if n := a.agent.Pending(); n != 0 {
		t.Errorf("%d steering messages still queued after being sent", n)
	}
}

// A cancel that is not the user asking to be heard (a time limit, SIGINT)
// starts nothing new, but the steering is kept for the next turn.
func TestOtherCancelsKeepQueuedSteering(t *testing.T) {
	a, c := cancelMidRequest(t, false, "use the other file")

	if s, _ := c.count(); s != 1 {
		t.Fatalf("made %d requests, want only the cancelled one", s)
	}
	if n := a.agent.Pending(); n != 1 {
		t.Fatalf("%d steering messages queued after the cancel, want 1 kept", n)
	}
}

// Ctrl-C with nothing queued is a plain stop: no second request.
func TestCtrlCWithNothingQueuedJustStops(t *testing.T) {
	_, c := cancelMidRequest(t, true)
	if s, _ := c.count(); s != 1 {
		t.Fatalf("made %d requests, want only the cancelled one", s)
	}
}

func TestLinesTypedWhileSteeringReachTheSavedHistory(t *testing.T) {
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	a, _ := swapApp(t, 262144, 0)
	s, err := session.Create(session.Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	a.sess = s

	a.rememberTyped("first steer")
	a.rememberTyped("cleared with ctrl-c")
	a.recordTyped()
	s.Close()

	_, entries, err := session.Open("/p", s.Meta.ID)
	if err != nil {
		t.Fatal(err)
	}
	got := session.Inputs(entries)
	if len(got) != 2 || got[0] != "first steer" || got[1] != "cleared with ctrl-c" {
		t.Errorf("saved history %q, want both lines in order", got)
	}
}

// A second steering prompt in the same turn, as a model swap opens, must not
// wipe what the first one left half-typed.
func TestUnsentTextSurvivesALaterSteeringPrompt(t *testing.T) {
	a := &App{}
	a.keepPending("half a sentence")
	a.keepPending("")
	if a.pendingInput != "half a sentence" {
		t.Fatalf("pending input %q after an empty prompt closed", a.pendingInput)
	}
	a.keepPending("more")
	if a.pendingInput != "half a sentence\nmore" {
		t.Errorf("pending input %q, want both pieces", a.pendingInput)
	}
}
