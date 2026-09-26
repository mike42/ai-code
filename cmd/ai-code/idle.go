package main

import (
	"context"
	"strings"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/provider"
	"ai-code/internal/session"
	"ai-code/internal/ui"
)

// checkpointProbeTimeout bounds the health check that decides whether the
// backend is free enough for a checkpoint.
//
// Not a limit on generation, which is never bounded by wall clock here: it
// bounds a liveness question whose answer stops being useful the moment it is
// slow, because a server that cannot say whether it is busy within a few
// seconds is busy. Same reasoning as the /slots probe.
const checkpointProbeTimeout = 3 * time.Second

// startIdleCheckpoint arranges for the session to be summarised if the prompt
// is left untouched, and returns the function that calls that off.
//
// The summary replaces nothing and is read only if a later window turns out
// to be too narrow for the real messages. Writing it while idle keeps the
// server's prefix cache warm for it, and puts the cost somewhere other than
// in front of a waiting user.
//
// The returned stop function waits for the goroutine, because the caller
// touches the agent the checkpoint is writing to. The keystroke has already
// cancelled it by then, so the wait is not perceptible.
func (a *App) startIdleCheckpoint(ctx context.Context, editor *ui.Editor) func() {
	if !a.watchEnabled(editor) {
		return func() {}
	}

	// Two lifetimes, because a keystroke means two different things to the
	// two jobs here. It aborts the speculative summary -- that is a model
	// call nobody asked for and the person is back. It must not stop the
	// watch: a window part-way through typing still has to answer another
	// window's swap, or that swap waits for a session that will never reply.
	watch, stopWatch := context.WithCancel(ctx)
	work, stopWork := context.WithCancel(watch)
	done := make(chan struct{})

	editor.OnActivity = stopWork
	go func() {
		defer close(done)
		a.watchWhileIdle(watch, work, a.cfg.Agent.AutoCheckpointIdleDelay.Duration, editor)
	}()

	return func() {
		stopWatch()
		<-done
		editor.OnActivity = nil
	}
}

// watchEnabled reports whether this session takes part at all. The swap
// watch and the speculative checkpoint share a goroutine, so turning the
// checkpoint off with agent.auto_checkpoint leaves the watch running.
func (a *App) watchEnabled(editor *ui.Editor) bool {
	// A piped session has no prompt to watch and no keystroke to abort with,
	// so anything started here could only collide with the next line of the
	// script.
	return editor != nil && editor.IsTTY()
}

// watchWhileIdle runs for as long as the prompt is up.
//
// Two clocks and two lifetimes. The speculative checkpoint is lazy work
// nobody asked for, bounded by `work`, which a keystroke ends. The watch for
// another window's swap is not optional and is bounded by `watch`, which
// ends only when the line is submitted: a window blocked on this session
// needs an answer whether or not someone is typing here.
func (a *App) watchWhileIdle(watch, work context.Context, idle time.Duration, editor *ui.Editor) {
	due := time.Now().Add(idle)
	written := false

	for {
		a.prepareForSwap(watch, editor)
		// The shared intent first: a local file saying what this user last
		// asked for. The server is consulted only about changes nobody here
		// asked for, and consulting it is optional.
		if !a.adoptIntended(editor) {
			a.adoptModelChange(watch, editor)
		}

		if !written && work.Err() == nil && !time.Now().Before(due) {
			written = true
			a.speculativeCheckpoint(work)
		}

		select {
		case <-watch.Done():
			return
		case <-time.After(swapPollInterval):
		}
	}
}

// speculativeCheckpoint writes the summary nobody asked for.
func (a *App) speculativeCheckpoint(ctx context.Context) {
	if a.cfg.Agent.AutoCheckpoint != nil && !*a.cfg.Agent.AutoCheckpoint {
		return
	}
	if !a.agent.CheckpointWorthwhile(agent.SpeculativeSummaryMaxTokens) {
		return
	}
	if !a.backendFree(ctx) {
		return
	}
	if _, err := a.agent.Summarise(ctx, agent.SpeculativeSummaryMaxTokens); err != nil {
		// Silent: nobody asked for this summary, the next idle period retries,
		// and a failure that matters is reported by the path that needs one.
		return
	}
	a.recordCheckpoint()
}

// backendFree reports whether a summary can be written without taking the
// model away from something else.
//
// Unknown counts as free: most backends expose nothing to ask, and refusing to
// checkpoint on every one of them would mean the feature only ever worked on
// lemonade.
func (a *App) backendFree(ctx context.Context) bool {
	in, ok := a.client.(provider.Introspector)
	if !ok {
		return true
	}
	ctx, cancel := context.WithTimeout(ctx, checkpointProbeTimeout)
	defer cancel()

	h, err := in.Health(ctx)
	if err != nil {
		return true
	}
	if h.ModelLoaded != "" && h.ModelLoaded != a.model.ID {
		// Someone else's model is resident. Sending anything now would evict it
		// and load ours back, and a swap per turn between two instances is
		// exactly the thrash a checkpoint exists to soften. Causing it for a
		// summary nobody asked for would be self-defeating.
		return false
	}
	return !h.Busy && !h.Streaming
}

// recordCheckpoint persists the standby summary the agent is holding, so a
// /restart or a --resume starts with it instead of making a new one.
//
// Safe to call after anything that might have produced one: it appends only
// when the agent holds something different from what is on disk.
func (a *App) recordCheckpoint() {
	if a.sess == nil {
		return
	}
	summary, through := a.agent.Summary()
	if strings.TrimSpace(summary) == "" || summary == a.recordedSummary {
		return
	}
	if err := a.sess.Append(session.Entry{
		Type:              session.EntryCheckpoint,
		Time:              time.Now(),
		Summary:           summary,
		SummarisedThrough: through,
	}); err != nil {
		return
	}
	a.recordedSummary = summary
}
