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
// different model, and waits for any that need to save themselves first.
//
// There is no confirmation. The user typed /model X and their intent is not
// in doubt; peers report what they are doing, they do not vote. The only
// decision left is whether to keep waiting, and Ctrl-C on an empty line ends
// that, as it ends every other wait in this harness.
//
// The wait is not modal. Typing during it is queued and runs on the new model
// once it loads, which is what typing after /model X meant.
func (a *App) announceSwap(parent context.Context, client provider.Client, into provider.ModelInfo, window int) {
	// The terminal is opened first and closed last, around everything: the
	// wait for peers and the load itself. Loading is the slow half -- minutes
	// on a large model -- so opening it only for the wait, as an earlier
	// version did by letting defers unwind in the wrong order, meant typing
	// was possible for the fast part and not the slow one.
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	stopSteering := a.steerWith(cancel, a.queuePrompt)
	defer func() { a.pendingInput = stopSteering() }()
	defer a.status("")

	if a.model.ID != into.ID {
		a.waitForPeers(ctx, into, window)
	}
	// After the wait, and before the announcement is withdrawn: peers stay
	// out of the way until the weights are actually in place.
	a.loadModel(ctx, client, into)
}

// waitForPeers holds the swap until every other session on this model has
// done whatever it needed to before losing it.
func (a *App) waitForPeers(ctx context.Context, into provider.ModelInfo, window int) {
	// Publishing the announcement and reading who has to act on it are one
	// decision. Split them and a session can mark itself working in the gap,
	// after this has already decided nobody was.
	var (
		notice *coord.Announcement
		err    error
		peers  []coord.Peer
	)
	_ = coord.Decide(a.coordDir, func() {
		// The intent is recorded here, before the weights move. Every other
		// window then names the incoming model on anything it sends during
		// the load, which the server queues behind the load rather than
		// answering by dragging the old model back.
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
			// No clock bounds this wait: a session mid-turn on a slow model
			// can take minutes, and cutting it off destroys the work the
			// waiting exists to protect. Ending it is the user's call.
			a.note(fmt.Sprintf("Loading %s without waiting for %s.",
				into.ID, listPeers(unfinished)))
			return
		case <-time.After(swapPollInterval):
		}
	}
}

// describeWaiting is the one line shown while a swap waits, naming each
// session by directory and what it is doing.
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
// Empty clears it. Nothing reaches the scrollback.
func (a *App) status(text string) {
	if a.interactive != nil {
		a.interactive.SetNotice(text)
	}
}

// loadModel loads the weights, so that /model X means the model has changed
// rather than that the next request will change it.
//
// Without this the swap is only a note to self: lemonade loads on demand, so
// the eviction the other windows were just warned about would happen at some
// unrelated later moment, after the warning had been withdrawn.
func (a *App) loadModel(ctx context.Context, client provider.Client, into provider.ModelInfo) {
	in, ok := client.(provider.Introspector)
	if !ok {
		// Nothing to load explicitly: this backend serves whatever is named
		// on the request itself.
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

// listPeers names sessions by directory, which is how a person tells one
// window from another.
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

// prepareForSwap is what a session does when another window announces it is
// taking the model.
//
// Usually nothing. A session whose conversation still fits the incoming
// window simply continues on the new model, so there is nothing to save and
// nothing to protect. The one case that needs work is a conversation too big
// for the new window: without a summary written now, while the old model is
// still loaded, that session is stranded, because summarising later needs
// the model that is about to go.
//
// Called only where a summary can actually be written: between turns, and at
// the boundary at the end of one. Never during, because summarising is
// itself a request and the model is busy with the turn.
//
// editor, when non-nil, is the live prompt this is running underneath: its
// line has to be lifted out of the way before anything is written, or the
// output lands in the middle of what is being typed.
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
	// Answered on the strength of there being an announcement, not on which
	// model this session currently thinks it is on. It may already have
	// adopted the incoming one -- the intent is published before the load --
	// and a check against the current model would then skip the save for a
	// conversation that still needs it.
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

// strandedBy reports whether losing the current model for one with this
// window would leave the session unable to continue.
//
// A summary already in hand counts: assembly falls back to it, so the
// session is fine even though the raw transcript is too big.
func (a *App) strandedBy(intoWindow int) bool {
	if intoWindow <= 0 {
		// Whether it fits cannot be decided, and saving is the recoverable
		// mistake of the two.
		return true
	}
	if summary, _ := a.agent.Summary(); summary != "" {
		return false
	}
	return a.agent.TranscriptTokens() > intoWindow
}

// adoptModelChange notices that the loaded model is no longer this session's
// and carries on with the new one. It reports whether anything changed.
//
// It neither waits nor reloads. Reloading is the eviction that starts the
// swap-per-turn thrash; waiting strands a session behind a decision the user
// already made in another window. What it changed is said by the prompt,
// which names the model until that model has answered here -- so the change
// is visible before a prompt is composed rather than underneath the reply to
// one.
func (a *App) adoptModelChange(ctx context.Context, editor *ui.Editor) bool {
	in, ok := a.client.(provider.Introspector)
	if !ok {
		return false
	}

	// The health question is short and is allowed to be bounded: a server
	// that cannot say what is loaded within a few seconds is busy loading
	// something, which is itself the answer.
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

	// The catalogue is not bounded by that budget, and failing to read it is
	// not a reason to carry on believing a model that is not loaded.
	//
	// It used to share the three-second probe with the health call, on a
	// path that runs precisely while the server is busiest -- mid-load of
	// somebody else's model. Timing out left the session still naming the
	// old model, and its next request loaded that model straight back: the
	// churn, caused by the code meant to prevent it, and silent because the
	// error was discarded.
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
		// Nothing known about the new model's window. Keep the one in force
		// rather than dropping to a default that may be wildly wrong.
		limit = a.agent.Window()
	}
	a.model = loaded
	a.agent.SetModel(loaded.ID, limit, loaded.MaxOutputTokens)
	a.peers.SetModel(a.providerName, loaded.ID)
	coord.Trace("adopt: now on %s", loaded.ID)

	a.showModelChange(editor)
	return true
}

// showModelChange shows the difference between what the scrollback says this
// session is on and what is actually loaded.
//
// That one comparison covers every case rather than a list of them. Three
// changes before a single prompt are one difference, so the line is
// rewritten in place rather than appended. A change and a change back is no
// difference, so the line is blanked. A session that has sent nothing still
// reports, because its banner made a claim that is no longer true.
func (a *App) showModelChange(editor *ui.Editor) {
	line := a.modelChangeLine()

	if editor != nil {
		editor.SetHeader(line, a.promptString())
		return
	}
	// No prompt to sit above: this is the guard before a send, so the line
	// goes straight into the scrollback and becomes what it says.
	if line != "" {
		a.out(line)
		a.knownModel = a.model.ID
	}
}

// modelChangeLine is the difference between what the scrollback says and
// what is loaded, or "" when there is none.
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

// stopForSwap is consulted between loop iterations. It reports whether this
// run should end so another window can have the model.
//
// A long agentic run is the case that needs it: waiting for the run to
// finish means waiting for the model to decide it is done, which may be an
// hour of tool calls away, and the window that asked for the model is
// blocked the whole time. Stopping at a boundary leaves a well-formed
// conversation that the next prompt continues from.
func (a *App) stopForSwap(ctx context.Context) bool {
	// Commands typed during the turn run here too, in the order they were
	// typed, alongside the steering the loop has already folded in.
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

// claimModel is what a session does immediately before sending: under the
// same lock a swap decides with, take what is loaded and declare itself
// working on it.
//
// The declaration is the point. A swap that reads the session list after
// this sees a session in flight and waits for it; a swap that got there
// first has published an announcement this sees, and the send yields to it
// instead of racing. Either order is correct, and there is no third.
func (a *App) claimModel(ctx context.Context) {
	for {
		yield, waitingFor := false, ""
		_ = coord.Decide(a.coordDir, func() {
			// Any announcement at all holds this back, not just one naming
			// the model this session is on. Between the announcement and the
			// load there is a window in which the weights have not moved
			// yet: a request sent into it is answered by loading whatever it
			// names, right now, which is either the outgoing model coming
			// back or the incoming one arriving before the sessions using
			// the old one have saved themselves.
			//
			// So the prompt waits, exactly as a run stopped at a turn
			// boundary waits, and goes out once the model it expects is
			// there.
			if n, ok := coord.PendingSwap(a.coordDir); ok {
				coord.Trace("claim: holding, pid=%d is loading %s", n.PID, n.Into)
				yield, waitingFor = true, n.Into
				return
			}
			if !a.adoptIntended(nil) {
				// Nothing has recorded a load yet, so this session's own
				// request is about to become one -- lemonade loads whatever
				// a request names. Saying so now, under the lock, is what
				// makes the next window's decision correct.
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

// awaitSwap waits for the announcement to be withdrawn, which happens once
// the weights are in place. It reports whether it is worth looking again.
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
// the one loaded.
//
// A turn claims the model once, before its first request. A run with tool
// calls makes many more after that, and between any two of them the weights
// can change -- a load queued behind the first request completing while the
// second is being prepared, or something outside ai-code entirely. Sending
// the next request anyway is what makes the server load the old model back,
// evicting whatever replaced it: churn caused by this session, several
// requests after it last looked.
//
// Checked at the loop boundary because that is where the run can stop and
// leave a conversation worth continuing.
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

// adoptIntended moves this session onto the model this user last asked for,
// in whichever window they asked. It reads one local file and makes no
// network call, which is what lets it be right while the server is busy
// loading and slowest to answer.
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

// adoptLoaded points this session at a model by name, filling in what it can
// about the window from the catalogue but never depending on being able to.
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
		// The catalogue was unreadable, so contextLimitFor fell back to its
		// unknown-window default. Keeping the window already in force is the
		// better guess: it at least belonged to a real model on this server.
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
