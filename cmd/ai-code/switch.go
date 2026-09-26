package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/provider"
	"ai-code/internal/render"
)

// cmdModel shows or changes the model.
func (a *App) cmdModel(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())

	if args == "" {
		models, err := fetchModels(ctx, a.client, true)
		if err != nil {
			return err
		}
		lines := []string{"", fmt.Sprintf("  Current: %s on %s", style.Bold(a.model.ID), a.providerName), ""}
		for _, l := range strings.Split(formatModelList(models, 40), "\n") {
			lines = append(lines, style.Dim(l))
		}
		lines = append(lines, "", style.Dim("Switch with /model <name>."), "")
		a.out(lines...)
		return nil
	}

	models, err := fetchModels(ctx, a.client, true)
	if err != nil {
		return err
	}
	var target *provider.ModelInfo
	for i := range models {
		if models[i].ID == args {
			target = &models[i]
			break
		}
	}
	if target == nil {
		return fmt.Errorf("model %q is not available on %s.\n\nAvailable:\n%s",
			args, a.providerName, formatModelList(models, 20))
	}
	return a.switchTo(ctx, a.client, a.providerName, a.providerCfg, *target, false)
}

// cmdProvider shows or changes the provider.
func (a *App) cmdProvider(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())

	if args == "" {
		lines := []string{""}
		for _, name := range a.cfg.ProviderNames() {
			p := a.cfg.Provider[name]
			marker := "  "
			if name == a.providerName {
				marker = "* "
			}
			cls := p.Class
			if cls == string(provider.ClassCloud) {
				cls = style.Warn(cls)
			}
			lines = append(lines, fmt.Sprintf("  %s%-12s %s  %s", marker, name, p.BaseURL, style.Dim(cls)))
		}
		a.out(append(lines, "")...)
		return nil
	}

	pc, ok := a.cfg.Provider[args]
	if !ok {
		return fmt.Errorf("no provider named %q; configured: %s",
			args, strings.Join(a.cfg.ProviderNames(), ", "))
	}
	// Both gates run before a client exists, because building one and asking
	// it for its catalogue is a network call to the provider being considered.
	// Asking afterwards repairs neither case: in a .nocloud tree the call is
	// the contact the marker forbids, and in front of the confirmation it
	// makes "Nothing was sent" untrue.
	class, err := provider.ParseClass(pc.Class)
	if err != nil {
		return fmt.Errorf("provider %q: %w", args, err)
	}
	if a.noCloud && class == provider.ClassCloud {
		return errCloudForbiddenHere
	}
	confirmed := false
	if class == provider.ClassCloud && len(a.agent.Messages()) > 0 {
		if !a.confirmCloudSwitch(args, class, pc.DefaultModel) {
			a.out("", "Cancelled. Nothing was sent.", "")
			return nil
		}
		confirmed = true
	}

	client, err := buildClient(args, pc)
	if err != nil {
		return err
	}
	model, err := resolveModel(ctx, client, "", pc.DefaultModel)
	if err != nil {
		return err
	}
	return a.switchTo(ctx, client, args, pc, model, confirmed)
}

// errCloudForbiddenHere is what a .nocloud tree answers. No amount of
// confirmation changes what the marker promises.
var errCloudForbiddenHere = errors.New(
	"a .nocloud file is present, so this session is restricted to on-premises providers")

// switchTo moves the live session onto a different model or provider.
//
// Moving an in-progress session to a cloud provider transmits everything it
// contains -- source, command output, whatever the model has been shown -- to a
// third party. That is a decision with compliance and privacy consequences, so
// it is confirmed explicitly and the confirmation states what would actually be
// sent. Moving to an on-premises provider asks nothing: the whole point of
// running your own is not being interrogated about it.
// confirmed says the caller already put the move to the user and was told to
// go ahead, which is what a caller that had to ask before it could build a
// client does.
func (a *App) switchTo(ctx context.Context, client provider.Client, name string, pc config.Provider, model provider.ModelInfo, confirmed bool) error {
	style := render.NewStyle(a.screen.Color())

	// In a .nocloud tree, cloud providers are not merely confirmed away -- they
	// are unavailable. No amount of confirmation changes what the marker
	// promises, and the marker exists precisely so an internal code-base can
	// never end up on a cloud model even by explicit choice.
	if a.noCloud && client.Class() == provider.ClassCloud {
		return errCloudForbiddenHere
	}

	movingOffPremises := !confirmed && client.Class() == provider.ClassCloud &&
		len(a.agent.Messages()) > 0

	if movingOffPremises {
		if !a.confirmCloudSwitch(name, client.Class(), model.ID) {
			a.out("", "Cancelled. Nothing was sent.", "")
			return nil
		}
	}

	newLimit, source := contextLimitFor(model, a.cfg.Agent.ContextOverride)
	oldLimit := a.agent.Window()
	used := a.contextState().Projected
	// Both halves are computed before anything is reassigned: the choice below
	// may be to stay put, and there is no undoing a swap that has happened.
	sameModel := model.ID == a.agent.Model() && name == a.providerName
	plan := a.agent.PlanResume(newLimit)

	// A smaller window is the case that bites, and the only one worth
	// interrupting for. A wider one changes nothing the session was already
	// doing, and a window it still fits in needs no decision -- a dialog after
	// every /model would be worse than the problem it warns about.
	narrower := newLimit > 0 && oldLimit > 0 && newLimit < oldLimit
	fromTranscript := false
	if narrower && !plan.Fits {
		switch a.chooseResume(model, newLimit, plan) {
		case "w":
			a.out("", fmt.Sprintf("Staying on %s. Nothing was sent and the session is untouched.",
				style.Bold(a.model.ID)), "")
			return nil
		case "t":
			fromTranscript = true
		}
	}

	// Everything above could still decide not to swap. From here it happens,
	// so this is the point to tell the other windows and then actually load
	// the weights -- in that order, because the load is the eviction they
	// are being warned about.
	a.announceSwap(ctx, client, model, newLimit)

	a.client = client
	a.providerName = name
	a.providerCfg = pc
	a.model = model
	a.agent.SetClient(client)
	a.agent.SetModel(model.ID, newLimit, model.MaxOutputTokens)
	a.peers.SetModel(name, model.ID)
	a.agent.SetResumeFromTranscript(fromTranscript)

	// The scrollback now says this, so it is what a later change is measured
	// against.
	a.knownModel = model.ID
	a.out(fmt.Sprintf("Now using %s on %s (%s, %s ctx — %s).",
		style.Bold(model.ID), name, client.Class(), compactInt(newLimit), style.Dim(source)))

	switch {
	case narrower && !plan.Fits:
		// The choice already stated its own numbers; repeating them here
		// would read as a second decision to make.
	case !plan.Fits:
		p := plan.Checkpoint
		if !p.Available {
			p = plan.Transcript
		}
		a.note(fmt.Sprintf(
			"This session is about %s tokens, so the next request carries %s of it -- the last "+
				"%d messages, leaving about %s to answer in. Nothing is deleted; /compact frees the rest.",
			tokenCount(plan.Session), tokenCount(p.Prompt), p.Kept, tokenCount(p.Free)))
	case newLimit > 0 && used*100/max(newLimit, 1) >= a.cfg.UI.ContextWarnPercent:
		a.note(fmt.Sprintf("This session already fills %d%% of the new window.",
			used*100/newLimit))
	}

	// Coming back to the model the session stopped on is not a change of
	// anything -- the window is what it was -- so the interrupted turn can
	// carry on without a question that has one answer. Both halves matter:
	// picking up a finished session would send a request nobody asked for.
	if sameModel && a.agent.StoppedMidTurn() {
		a.out(style.Dim("Same model, and the last turn did not finish. Resuming it."), "")
		return a.runTurn(ctx, "")
	}
	return nil
}

// chooseResume asks how to carry on in a window the session no longer fits.
//
// It states what each way costs rather than naming it, because the names are
// interchangeable to anyone who has not read the assembly code and the numbers
// are not: how much room is left to work in, and how much of the session stops
// being sent, are the whole of the decision. No option summarises on the spot
// -- that would be a model call, on the model being left behind, at the moment
// the user is waiting to get on with something else.
func (a *App) chooseResume(model provider.ModelInfo, newLimit int, plan agent.ResumeChoice) string {
	style := render.NewStyle(a.screen.Color())

	lines := []string{
		"",
		style.Warn(fmt.Sprintf("%s has a narrower window than this session.", model.ID)),
		"",
		fmt.Sprintf("  This session   about %s tokens, %d messages",
			tokenCount(plan.Session), len(a.agent.Messages())),
		fmt.Sprintf("  %-14s %s window, %s of it usable after the reserve",
			model.ID, compactInt(newLimit), tokenCount(plan.Usable)),
		"",
		"What the next request should carry:",
		"",
	}

	if c := plan.Checkpoint; c.Available {
		lines = append(lines,
			fmt.Sprintf("  c  The checkpoint and the last %d messages: sends about %s, leaving",
				c.Kept, tokenCount(c.Prompt)),
			fmt.Sprintf("     about %s to answer in. The %d earlier messages go as a summary",
				tokenCount(c.Free), c.Dropped),
			lossLine(c))
	} else {
		lines = append(lines,
			"  c  Unavailable: no checkpoint has been written for this session yet, and",
			"     writing one now is a model call you would sit and wait through.")
	}

	t := plan.Transcript
	lines = append(lines,
		fmt.Sprintf("  t  The last %d messages and nothing else: sends about %s, leaving about",
			t.Kept, tokenCount(t.Prompt)),
		fmt.Sprintf("     %s to answer in. The %d before them are dropped outright, and older",
			tokenCount(t.Free), t.Dropped),
		"     turns keep falling off as you work -- nothing will be summarised.",
		fmt.Sprintf("  w  Stay on %s and change nothing. Nothing is sent either way until", a.model.ID),
		"     you type again.",
		"")

	// The keys offered are the keys that work. Listing c when there is no
	// checkpoint invites an answer that silently becomes something else.
	valid := "ctw"
	if !plan.Checkpoint.Available {
		valid = "tw"
	}
	lines = append(lines,
		style.Dim("The transcript is never deleted; it comes back whole on a wider model."),
		style.Dim(fmt.Sprintf("[%s] ", strings.Join(strings.Split(valid, ""), "/"))))
	a.out(lines...)
	// Anything unreadable or unrecognised stays put, for the reason readYesNo
	// does: the answer that changes nothing is the one that cannot be wrong.
	return readChoice(valid, "w")
}

// lossLine says what a checkpoint does not cover, which is the one thing about
// it that is easy to assume away: a checkpoint written five turns ago speaks
// for the session as it was five turns ago, and nothing speaks for the rest.
func lossLine(c agent.ResumePlan) string {
	if c.Unrepresented == 0 {
		return "     instead, and the summary covers all of them."
	}
	return fmt.Sprintf("     instead, and %d of them are newer than it, so nothing covers those.",
		c.Unrepresented)
}

// tokenCount keeps a figure someone is about to make a decision on at one
// decimal place. compactInt rounds 3.6k down to 3k, and on a window this
// narrow that is a sixth of the room being misreported.
func tokenCount(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprint(n)
}

// model is the model being moved to, and "" when the provider has not been
// asked which one that is -- which is the case whenever asking would itself
// be the first contact with it.
func (a *App) confirmCloudSwitch(name string, class provider.Class, model string) bool {
	style := render.NewStyle(a.screen.Color())
	msgs := a.agent.Messages()
	read, modified := agent.FilesTouched(msgs)
	used := a.contextState().Projected

	if model == "" {
		model = "chosen from " + name + "'s catalogue"
	}
	lines := []string{
		"",
		style.Warn("This moves a live session to a cloud provider."),
		"",
		fmt.Sprintf("  Provider   %s (%s)", name, class),
		fmt.Sprintf("  Model      %s", model),
		fmt.Sprintf("  Would send %d messages, about %s tokens", len(msgs), compactInt(used)),
	}
	if n := len(read) + len(modified); n > 0 {
		lines = append(lines, fmt.Sprintf("  Files      %d in context", n))
		for _, f := range firstN(append(append([]string{}, modified...), read...), 8) {
			lines = append(lines, "             "+style.Dim(f))
		}
		if n > 8 {
			lines = append(lines, "             "+style.Dim(fmt.Sprintf("... and %d more", n-8)))
		}
	}
	lines = append(lines, "",
		style.Dim("Everything above leaves this machine. Continue? [y/N] "))
	a.out(lines...)

	return readYesNo()
}

func firstN(s []string, n int) []string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// readChoice reads a single letter from the offered set, falling back to the
// given answer on anything else.
func readChoice(valid, fallback string) string {
	r := bufio.NewReader(os.Stdin)
	line, err := r.ReadString('\n')
	if err != nil {
		return fallback
	}
	if s := strings.ToLower(strings.TrimSpace(line)); len(s) == 1 && strings.Contains(valid, s) {
		return s
	}
	return fallback
}

// readYesNo reads a single confirmation line. Deliberately requires an explicit
// "y": the default on an ambiguous answer is not to send anything.
func readYesNo() bool {
	r := bufio.NewReader(os.Stdin)
	line, err := r.ReadString('\n')
	if err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	}
	return false
}
