package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/coord"
	"ai-code/internal/provider"
	"ai-code/internal/render"
	"ai-code/internal/ui"
)

// swapPollInterval is how often each side re-reads the shared files. A sampling
// rate over local files, not a deadline: nothing gives up when it elapses.
const swapPollInterval = 500 * time.Millisecond

// announceSwap tells the other windows that this one is about to load a
// different model, and waits for any that need to save themselves first. The
// wait is not modal: typing during it is queued and runs once the model loads.
func (a *App) announceSwap(parent context.Context, client provider.Client, into provider.ModelInfo, window int) {
	// The terminal is opened first and closed last, around wait and load.
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	stopSteering := a.steerWith(cancel, a.queuePrompt)
	defer func() { a.pendingInput = stopSteering() }()
	defer a.status("")

	if a.model.ID != into.ID {
		a.waitForPeers(ctx, into, window)
	}
	// After the wait, before the announcement is withdrawn: peers stay out of
	// the way until the weights are in place.
	a.loadModel(ctx, client, into)
}

// waitForPeers holds the swap until every peer has saved itself.
func (a *App) waitForPeers(ctx context.Context, into provider.ModelInfo, window int) {
	// Publishing the announcement and reading who must act are one decision.
	var (
		notice *coord.Announcement
		err    error
		peers  []coord.Peer
	)
	_ = coord.Decide(a.coordDir, func() {
		// The intent is recorded before the weights move, so other windows
		// name the incoming model and the server queues their sends behind it.
		_ = coord.SetIntent(a.coordDir, into.ID)
		coord.Trace("intent: %s -> %s", a.model.ID, into.ID)

		peers = coord.OnModel(a.coordDir, a.model.ID)
		coord.Trace("announce %s -> %s: %d peers to wait for", a.model.ID, into.ID, len(peers))
		for _, p := range peers {
			coord.Trace("  peer pid=%d %s state=%q", p.PID, p.Model, p.State)
		}
		if len(peers) == 0 {
			return
		}
		notice, err = coord.AnnounceSwap(a.coordDir, a.model.ID, into.ID, window)
	})
	if len(peers) == 0 || err != nil || notice == nil {
		// Nobody to tell, or nobody listening.
		return
	}
	defer notice.End()

	for {
		unfinished := coord.NotReady(a.coordDir, a.model.ID)
		if len(unfinished) == 0 {
			return
		}
		a.status(describeWaiting(unfinished))

		select {
		case <-ctx.Done():
			// No clock bounds this wait: only Ctrl-C ends it, and cutting it
			// off can destroy work.
			a.note(fmt.Sprintf("Loading %s without waiting for %s.",
				into.ID, listPeers(unfinished)))
			return
		case <-time.After(swapPollInterval):
		}
	}
}

// describeWaiting is the one line shown while a swap waits.
func describeWaiting(peers []coord.Peer) string {
	parts := make([]string, 0, len(peers))
	for _, p := range peers {
		what := p.State.Describe()
		if what == "" {
			what = "starting"
		}
		parts = append(parts, p.Describe()+" "+what)
	}
	return "waiting · " + strings.Join(parts, " · ")
}

// status puts one line in the transient zone, replacing whatever was there.
func (a *App) status(text string) {
	if a.interactive != nil {
		a.interactive.SetNotice(text)
	}
}

// loadModel loads the weights, so /model X means the model has changed rather
// than that the next request will. Lemonade loads on demand, which would defer
// the eviction peers were warned about to an unrelated later moment.
func (a *App) loadModel(ctx context.Context, client provider.Client, into provider.ModelInfo) {
	in, ok := client.(provider.Introspector)
	if !ok {
		// Nothing to load explicitly: the request names the model.
		return
	}
	a.status("loading " + into.ID)
	coord.Trace("POST /load %s", into.ID)
	if err := in.Load(ctx, into.ID); err != nil {
		coord.Trace("load %s failed: %v", into.ID, err)
		a.note("The server did not load " + into.ID + ": " + err.Error() +
			". The next request will try again.")
		return
	}
	coord.Trace("loaded %s", into.ID)
}

// listPeers names sessions the way a person tells one window from another.
func listPeers(peers []coord.Peer) string {
	names := make([]string, 0, len(peers))
	for _, p := range peers {
		names = append(names, p.Describe())
	}
	switch len(names) {
	case 0:
		return "anything"
	case 1:
		return names[0]
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}

// prepareForSwap writes a summary when another window is about to take the
// model and this conversation will not fit the incoming window. Called only
// between turns, where a summary can be written.
func (a *App) prepareForSwap(ctx context.Context, editor *ui.Editor) {
	say := func(line string) {
		if editor != nil {
			editor.EmitAbove(a.promptString(), line)
			return
		}
		a.note(line)
	}

	notice, ok := coord.PendingSwap(a.coordDir)
	if !ok {
		if a.peers.Self().State == coord.StateReady {
			a.peers.SetState(coord.StateFree)
		}
		return
	}
	// Answered on the announcement alone: this session may already have adopted
	// the incoming model, so a check against the current one would skip a save.
	if a.peers.Self().State == coord.StateReady {
		return
	}

	if !a.strandedBy(notice.IntoWindow) {
		a.peers.SetState(coord.StateReady)
		return
	}

	a.peers.SetState(coord.StateSummarising)
	say(fmt.Sprintf("Another window is loading %s, which this conversation will not fit. "+
		"Writing a summary while %s is still here…", notice.Into, a.model.ID))

	if _, err := a.agent.Summarise(ctx, agent.SpeculativeSummaryMaxTokens); err != nil {
		a.peers.SetNoSummary()
		say("Could not write a summary: " + err.Error())
		return
	}
	a.recordCheckpoint()
	a.peers.SetState(coord.StateReady)
	say("Summary written before " + a.model.ID + " was unloaded.")
}

// strandedBy reports whether losing the current model would leave the session
// unable to continue. A standby summary counts: assembly falls back to it.
func (a *App) strandedBy(intoWindow int) bool {
	if intoWindow <= 0 {
		// Whether it fits cannot be decided; saving is the recoverable mistake.
		return true
	}
	if summary, _ := a.agent.Summary(); summary != "" {
		return false
	}
	return a.agent.TranscriptTokens() > intoWindow
}

// adoptModelChange notices that the loaded model is no longer this session's
// and carries on with the new one, reporting whether anything changed. It
// neither waits nor reloads: reloading is the eviction that starts the thrash.
func (a *App) adoptModelChange(ctx context.Context, editor *ui.Editor) bool {
	in, ok := a.client.(provider.Introspector)
	if !ok {
		return false
	}

	// Bounded: a server slow to say what is loaded is busy loading.
	probe, cancel := context.WithTimeout(ctx, checkpointProbeTimeout)
	h, err := in.Health(probe)
	cancel()
	if err != nil {
		coord.Trace("adopt: health failed: %v", err)
		return false
	}
	if h.ModelLoaded == "" || h.ModelLoaded == a.model.ID {
		return false
	}
	coord.Trace("adopt: loaded=%s, this session wants %s", h.ModelLoaded, a.model.ID)

	// The catalogue is not bounded by that budget: a failed read is no reason
	// to keep naming a model that is not loaded.
	loaded := provider.ModelInfo{ID: h.ModelLoaded}
	if models, err := a.client.Models(ctx); err == nil {
		for _, m := range models {
			if m.ID == h.ModelLoaded {
				loaded = m
				break
			}
		}
	} else {
		coord.Trace("adopt: catalogue unreadable (%v); adopting %s with no window info",
			err, h.ModelLoaded)
	}

	limit, _ := contextLimitFor(loaded, a.cfg.Agent.ContextOverride)
	if limit <= 0 {
		// Window unknown: keep the one in force rather than a bad default.
		limit = a.agent.Window()
	}
	a.model = loaded
	a.agent.SetModel(loaded.ID, limit, loaded.MaxOutputTokens)
	a.peers.SetModel(a.providerName, loaded.ID)
	coord.Trace("adopt: now on %s", loaded.ID)

	a.showModelChange(editor)
	return true
}

// showModelChange shows the difference between the scrollback's model and the
// loaded one: rewritten in place before a prompt, or blanked when they cancel.
func (a *App) showModelChange(editor *ui.Editor) {
	line := a.modelChangeLine()

	if editor != nil {
		editor.SetHeader(line, a.promptString())
		return
	}
	// No prompt to sit above: this guards a send, so the line enters scrollback.
	if line != "" {
		a.out(line)
		a.knownModel = a.model.ID
	}
}

// modelChangeLine reports that difference, or "" when there is none.
func (a *App) modelChangeLine() string {
	if a.knownModel == "" || a.model.ID == a.knownModel {
		return ""
	}
	style := render.NewStyle(a.screen.Color())
	limit, _ := contextLimitFor(a.model, a.cfg.Agent.ContextOverride)
	return style.Warn("── now on "+a.model.ID) +
		style.Dim(fmt.Sprintf("  (was %s, %s ctx)", a.knownModel, compactInt(limit)))
}

func describePeers(peers []coord.Peer) string {
	switch len(peers) {
	case 0:
		return "no other session"
	case 1:
		return "one other session"
	}
	return fmt.Sprintf("%d other sessions", len(peers))
}

// stopForSwap is consulted between loop iterations and reports whether this
// run should end so another window can have the model. Stopping at a boundary
// leaves a well-formed conversation the next prompt continues from.
func (a *App) stopForSwap(ctx context.Context) bool {
	// Commands typed during the turn run here too, in the order they were typed.
	stop := a.runQueuedCommands(ctx)

	notice, ok := coord.PendingSwap(a.coordDir)
	if !ok || notice.Model != a.model.ID {
		return stop || a.modelWentAway(ctx)
	}

	a.note(fmt.Sprintf(
		"Another window is loading %s, so this run is stopping here. "+
			"Send anything to carry on, on whichever model is loaded.", notice.Into))
	a.prepareForSwap(ctx, nil)
	return true
}

// claimModel takes what is loaded and declares this session working on it,
// under the same lock a swap decides with. Either a swap sees this session and
// waits, or the announcement is seen here and the send yields.
func (a *App) claimModel(ctx context.Context) {
	for {
		yield, waitingFor := false, ""
		_ = coord.Decide(a.coordDir, func() {
			// Any announcement holds this back: before the load lands, a request
			// sent now loads whatever it names.
			if n, ok := coord.PendingSwap(a.coordDir); ok {
				coord.Trace("claim: holding, pid=%d is loading %s", n.PID, n.Into)
				yield, waitingFor = true, n.Into
				return
			}
			if !a.adoptIntended(nil) {
				// No load recorded yet, so record this session's intent.
				if _, known := coord.CurrentIntent(a.coordDir); !known {
					_ = coord.SetIntent(a.coordDir, a.model.ID)
					coord.Trace("claim: first use, recording %s as the intent", a.model.ID)
				}
			}
			a.peers.SetState(coord.StateWorking)
			coord.Trace("claim: working on %s", a.model.ID)
		})
		if !yield {
			return
		}

		a.prepareForSwap(ctx, nil)
		a.status("holding this prompt until " + waitingFor + " is loaded")
		ok := a.awaitSwap(ctx)
		a.status("")
		if !ok {
			return
		}
	}
}

// awaitSwap waits for the announcement to be withdrawn, once the weights are in
// place.
func (a *App) awaitSwap(ctx context.Context) bool {
	for {
		if _, ok := coord.PendingSwap(a.coordDir); !ok {
			return true
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(swapPollInterval):
		}
	}
}

// modelWentAway reports whether the model this run has been using is still
// the one loaded. The claim is made once per turn, but a run sends many
// requests, and sending again after a change loads the old model back.
func (a *App) modelWentAway(ctx context.Context) bool {
	in, ok := a.client.(provider.Introspector)
	if !ok {
		return false
	}
	probe, cancel := context.WithTimeout(ctx, checkpointProbeTimeout)
	defer cancel()

	h, err := in.Health(probe)
	if err != nil || h.ModelLoaded == "" || h.ModelLoaded == a.model.ID {
		return false
	}
	coord.Trace("boundary: %s is loaded, this run wants %s -- stopping", h.ModelLoaded, a.model.ID)
	a.note(fmt.Sprintf(
		"%s was loaded while this was running, so it is stopping here rather than "+
			"loading %s back. Send anything to carry on.", h.ModelLoaded, a.model.ID))
	return true
}

// adoptIntended moves this session onto the model last asked for in any
// window. One local file, no network call, so it works while the server is
// busy loading.
func (a *App) adoptIntended(editor *ui.Editor) bool {
	l, ok := coord.CurrentIntent(a.coordDir)
	if !ok || l.Model == a.model.ID {
		return false
	}
	coord.Trace("adopt: %s was asked for (by pid %d); this session was on %s",
		l.Model, l.PID, a.model.ID)
	a.adoptLoaded(l.Model, editor)
	return true
}

// adoptLoaded points this session at a model by name, filling in its window
// from the catalogue where it can.
func (a *App) adoptLoaded(id string, editor *ui.Editor) {
	loaded := provider.ModelInfo{ID: id}
	if models, err := a.client.Models(context.Background()); err == nil {
		for _, m := range models {
			if m.ID == id {
				loaded = m
				break
			}
		}
	}
	limit, _ := contextLimitFor(loaded, a.cfg.Agent.ContextOverride)
	if loaded.ContextWindow <= 0 && a.cfg.Agent.ContextOverride <= 0 {
		// Catalogue unreadable: the window in force beats the unknown default.
		if current := a.agent.Window(); current > 0 {
			limit = current
		}
	}
	a.model = loaded
	a.agent.SetModel(loaded.ID, limit, loaded.MaxOutputTokens)
	a.peers.SetModel(a.providerName, loaded.ID)
	coord.Trace("adopt: now on %s (window %d)", loaded.ID, limit)
	a.showModelChange(editor)
}
