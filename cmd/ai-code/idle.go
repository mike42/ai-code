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
// backend is free: a server slow to answer counts as busy.
const checkpointProbeTimeout = 3 * time.Second

// startIdleCheckpoint summarises the session if the prompt is left untouched,
// and returns the stop function, which waits for the goroutine before the
// caller touches the agent.
func (a *App) startIdleCheckpoint(ctx context.Context, editor *ui.Editor) func() {
	if !a.watchEnabled(editor) {
		return func() {}
	}

	// Two lifetimes: a keystroke ends the summary but keeps the swap watch
	// running.
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

// watchEnabled reports whether this session takes part.
func (a *App) watchEnabled(editor *ui.Editor) bool {
	// A piped session has no prompt to watch.
	return editor != nil && editor.IsTTY()
}

// watchWhileIdle runs for as long as the prompt is up. The checkpoint is bounded
// by work, which a keystroke ends; the swap watch is bounded by watch, which
// ends only when the line is submitted.
func (a *App) watchWhileIdle(watch, work context.Context, idle time.Duration, editor *ui.Editor) {
	due := time.Now().Add(idle)
	written := false

	for {
		a.prepareForSwap(watch, editor)
		// The shared intent first: a local file, no server call.
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
		// Silent: the next idle period retries; a real failure is reported where needed.
		return
	}
	a.recordCheckpoint()
}

// backendFree reports whether a summary can be written without taking the model
// away; unknown counts as free.
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
		// Someone else's model is resident: a send now would evict it.
		return false
	}
	return !h.Busy && !h.Streaming
}

// recordCheckpoint persists the standby summary so a /restart or a --resume
// starts with it. It appends only when the summary differs from the last.
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
