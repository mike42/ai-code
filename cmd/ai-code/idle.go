package main

import (
	"context"
	"strings"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/coord"
	"ai-code/internal/session"
	"ai-code/internal/ui"
)

// startIdleCheckpoint summarises the session if the prompt is left untouched,
// so a later model swap finds the summary already written. It returns the stop
// function, which waits for the goroutine before the caller touches the agent.
func (a *App) startIdleCheckpoint(ctx context.Context, editor *ui.Editor) func() {
	if !a.watchEnabled(editor) {
		return func() {}
	}

	work, stop := context.WithCancel(ctx)
	done := make(chan struct{})
	editor.OnActivity = stop
	go func() {
		defer close(done)
		a.idleCheckpoint(work, a.cfg.Agent.AutoCheckpointIdleDelay.Duration)
	}()

	return func() {
		stop()
		<-done
		editor.OnActivity = nil
	}
}

// idleCheckpoint waits out the idle delay, then writes the summary.
func (a *App) idleCheckpoint(ctx context.Context, delay time.Duration) {
	idle := time.NewTimer(delay)
	defer idle.Stop()
	select {
	case <-ctx.Done():
		return
	case <-idle.C:
	}
	a.speculativeCheckpoint(ctx)
}

// watchEnabled reports whether this session takes part.
func (a *App) watchEnabled(editor *ui.Editor) bool {
	// A piped session has no prompt to watch.
	return editor != nil && editor.IsTTY()
}

// speculativeCheckpoint writes the summary nobody asked for.
func (a *App) speculativeCheckpoint(ctx context.Context) {
	if a.cfg.Agent.AutoCheckpoint != nil && !*a.cfg.Agent.AutoCheckpoint {
		return
	}
	if !a.agentMu.TryLock() {
		return
	}
	defer a.agentMu.Unlock()
	if !a.modelIsWarm() || !a.agent.CheckpointWorthwhile(agent.SpeculativeSummaryMaxTokens) {
		return
	}
	if !a.backendFree() {
		return
	}
	if _, err := a.agent.Summarise(ctx, agent.SpeculativeSummaryMaxTokens); err != nil {
		// Silent: the next idle period retries; a real failure is reported where needed.
		return
	}
	a.recordCheckpoint()
}

// backendFree reports whether a summary can be written without taking the
// model from another instance or queueing behind one.
func (a *App) backendFree() bool {
	s := a.share
	if s == nil || s.bus.Self().Server == "" {
		return true
	}
	self := s.bus.Self()
	if len(s.bus.Pending()) > 0 {
		return false
	}
	if l := s.bus.Loaded(); l != "" && l != self.Model {
		return false
	}
	for _, p := range s.bus.Peers() {
		if p.State == coord.StateWorking || p.State == coord.StateSummarising {
			return false
		}
	}
	return true
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
