package main

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"ai-code/internal/provider"
	"ai-code/internal/render"
	"ai-code/internal/session"
)

// modelPick is the answer to "which model", worked out in the background
// while the prompt is already up. The goroutine writes choice and err before
// closing done; only the main loop reads them, and only after done.
type modelPick struct {
	requested string
	done      chan struct{}
	choice    modelChoice
	err       error
	applied   bool
}

// startModelPick asks the provider which model to use without holding up the
// prompt, and puts one line in the scrollback when it knows.
func (a *App) startModelPick(ctx context.Context, requested string) {
	p := &modelPick{requested: requested, done: make(chan struct{})}
	a.pick = p
	client, name, project := a.client, a.providerName, a.sessionProject
	go func() {
		p.choice, p.err = chooseModel(ctx, client, requested, session.PreviousModel(project, name))
		close(p.done)
		if ctx.Err() != nil {
			return
		}
		if p.err != nil {
			a.note(p.err.Error())
			return
		}
		a.out(a.modelLine(p.choice))
	}()
}

// modelLine announces the model and what the server says about its window.
func (a *App) modelLine(c modelChoice) string {
	style := render.NewStyle(a.screen.Color())
	limit, _ := contextLimitFor(c.Model, a.cfg.Agent.ContextOverride)
	ctx := compactInt(limit) + " ctx"
	if s := slotSummary(c.Model); s != "" {
		ctx += " (" + s + ")"
	}
	return fmt.Sprintf("%s  %s   %s   %s", style.Dim("model"), style.Bold(c.Model.ID),
		style.Dim(c.Why), style.Dim(ctx))
}

// applyModelPick takes the background answer once it exists. It runs on the
// main loop, under agentMu, which the background goroutine never takes: a send
// waiting for the answer already holds it.
func (a *App) applyModelPick() {
	p := a.pick
	if p == nil || p.applied {
		return
	}
	select {
	case <-p.done:
	default:
		return
	}
	p.applied = true
	if p.err == nil {
		a.adoptModel(p.choice)
	}
}

// adoptModel makes c the session's model without loading anything: it is
// already loaded, or nothing is and the first request loads it.
func (a *App) adoptModel(c modelChoice) {
	limit, _ := contextLimitFor(c.Model, a.cfg.Agent.ContextOverride)
	a.model = c.Model
	a.agent.SetModel(c.Model.ID, limit, c.Model.MaxOutputTokens)
	a.modelExplicit = c.Explicit
	if a.share != nil {
		a.share.chose(serverKey(a.client, a.providerCfg), c.Model.ID)
	}
	a.recordModel()
}

// recordModel notes the model in the session file, which is where the next
// launch looks for the model last used here.
func (a *App) recordModel() {
	if a.sess == nil || a.model.ID == "" {
		return
	}
	_ = a.sess.Append(session.Entry{Type: session.EntryModel, Provider: a.providerName, Model: a.model.ID})
}

// readyToSend is what a prompt waits for before anything is sent: a model,
// and a server that is not holding a different one this session did not ask
// for. Ctrl-C ends the wait.
func (a *App) readyToSend(parent context.Context) error {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	restore := installInterrupt(cancel)
	defer restore()
	defer a.status("")

	if err := a.settleModel(ctx); err != nil {
		return err
	}
	return a.checkServer(ctx)
}

// settleModel waits for the background answer, and when it found no model,
// asks again: the server may be up now, or have something loaded.
func (a *App) settleModel(ctx context.Context) error {
	if a.model.ID != "" {
		return nil
	}
	requested := ""
	if p := a.pick; p != nil {
		requested = p.requested
		select {
		case <-p.done:
		default:
			a.status("waiting for " + a.providerName + " to say which model is loaded")
			select {
			case <-p.done:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		a.applyModelPick()
		if a.model.ID != "" {
			return nil
		}
	}
	a.status("asking " + a.providerName + " which model is loaded")
	c, err := chooseModel(ctx, a.client, requested, session.PreviousModel(a.sessionProject, a.providerName))
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("%w; nothing was sent", err)
	}
	a.adoptModel(c)
	a.out(a.modelLine(c))
	return nil
}

// checkServer stops a prompt from loading a model over one the server already
// has, unless the user asked for that model by name. Instances agree among
// themselves through coord, but what the server holds is only known by asking
// it, so it is asked once per prompt, never on a timer.
func (a *App) checkServer(ctx context.Context) error {
	intro, ok := a.client.(provider.Introspector)
	if !ok {
		return nil
	}
	h, err := intro.Health(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("could not ask %s which model is loaded, so nothing was sent: %w", a.providerName, err)
	}
	if slices.Contains(h.Loaded, a.model.ID) {
		a.modelExplicit = false
		return nil
	}
	if len(h.Loaded) == 0 || a.modelExplicit {
		// Nothing is evicted, or the user asked for this swap; claim makes it,
		// after the other instances have made way.
		return nil
	}
	if len(h.Loaded) > 1 {
		return fmt.Errorf("%s has %s loaded, not %s, so nothing was sent: choose one with /model NAME",
			a.providerName, joinAnd(h.Loaded), a.model.ID)
	}
	models, err := liveModels(ctx, a.client)
	if err != nil {
		return fmt.Errorf("%w; nothing was sent", err)
	}
	for _, m := range models {
		if m.ID == h.Loaded[0] {
			was := a.model.ID
			a.note(fmt.Sprintf("%s has %s loaded now, so this prompt goes to it rather than loading %s back. "+
				"/model %s switches back.", a.providerName, m.ID, was, was))
			return a.changeModel(ctx, a.client, a.providerName, a.providerCfg, m, false, false)
		}
	}
	return fmt.Errorf("%s has %s loaded, which it does not list, so nothing was sent: choose one with /model NAME",
		a.providerName, h.Loaded[0])
}

func joinAnd(ss []string) string {
	switch len(ss) {
	case 0:
		return ""
	case 1:
		return ss[0]
	}
	out := ss[0]
	for _, s := range ss[1 : len(ss)-1] {
		out += ", " + s
	}
	return out + " and " + ss[len(ss)-1]
}

// noModelError is chooseModel's answer that there is nothing to pick, as
// opposed to a server that could not be asked.
type noModelError struct{ msg string }

func (e *noModelError) Error() string { return e.msg }

func noModel(format string, args ...any) error {
	return &noModelError{msg: fmt.Sprintf(format, args...)}
}

func isNoModel(err error) bool {
	var nm *noModelError
	return errors.As(err, &nm)
}
