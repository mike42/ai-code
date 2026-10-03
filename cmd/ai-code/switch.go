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
			if a.cloudForbidden() && p.Class == string(provider.ClassCloud) {
				continue
			}
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
	// Both gates run before a client exists: building one and asking for its
	// catalogue is a network call to the provider being considered.
	class, err := provider.ParseClass(pc.Class)
	if err != nil {
		return fmt.Errorf("provider %q: %w", args, err)
	}
	if class == provider.ClassCloud && a.gate != nil {
		// Whether the remote working directory forbids this has to be known
		// before the provider is contacted at all, catalogue included.
		if err := a.gate.settle(ctx); err != nil {
			return err
		}
	}
	if a.cloudForbidden() && class == provider.ClassCloud {
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
	if a.gate != nil {
		client = a.gate.wrap(client)
	}
	client = a.share.wrap(client)
	model, err := resolveModel(ctx, client, "", pc.DefaultModel)
	if err != nil {
		return err
	}
	return a.switchTo(ctx, client, args, pc, model, confirmed)
}

// errCloudForbiddenHere is what a .nocloud tree answers.
var errCloudForbiddenHere = errors.New(
	"--no-cloud or a .nocloud file applies, so this session is restricted to on-premises providers")

// switchTo moves the live session onto a different model or provider. A move
// to a cloud provider is confirmed explicitly, with the payload stated;
// confirmed says the caller already asked and was told to go ahead.
func (a *App) switchTo(ctx context.Context, client provider.Client, name string, pc config.Provider, model provider.ModelInfo, confirmed bool) error {
	style := render.NewStyle(a.screen.Color())

	// In a .nocloud tree a cloud provider is unavailable, not merely confirmed
	// away.
	if a.cloudForbidden() && client.Class() == provider.ClassCloud {
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
	// Computed before anything is reassigned: the choice below may be to stay put.
	sameModel := model.ID == a.agent.Model() && name == a.providerName
	plan := a.agent.PlanResume(newLimit)

	// Only a narrower window that no longer fits is worth interrupting for.
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

	// From here the swap happens: the other instances make way, then the
	// weights load, because the load is the eviction they make way for.
	server := serverKey(client, pc)
	a.swapTo(ctx, client, server, model, newLimit)

	a.client = client
	a.providerName = name
	a.providerCfg = pc
	a.model = model
	a.agent.SetClient(client)
	a.agent.SetModel(model.ID, newLimit, model.MaxOutputTokens)
	if a.share != nil {
		a.share.chose(server, model.ID)
	}
	a.agent.SetResumeFromTranscript(fromTranscript)

	a.out(fmt.Sprintf("Now using %s on %s (%s, %s ctx — %s).",
		style.Bold(model.ID), name, client.Class(), compactInt(newLimit), style.Dim(source)))

	switch {
	case narrower && !plan.Fits:
		// The choice already stated its own numbers.
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

	// Same model with an unfinished turn: resume it without asking.
	if sameModel && a.agent.StoppedMidTurn() {
		a.out(style.Dim("Same model, and the last turn did not finish. Resuming it."), "")
		return a.runTurn(ctx, "")
	}
	return nil
}

// chooseResume asks how to carry on in a window the session no longer fits.
// Options are stated as costs -- room left to work in, messages no longer
// sent -- not by name. No option summarises on the spot.
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

	// Only offer keys that work: listing c without a checkpoint invites a wrong
	// answer.
	valid := "ctw"
	if !plan.Checkpoint.Available {
		valid = "tw"
	}
	lines = append(lines,
		style.Dim("The transcript is never deleted; it comes back whole on a wider model."),
		style.Dim(fmt.Sprintf("[%s] ", strings.Join(strings.Split(valid, ""), "/"))))
	a.out(lines...)
	// Anything unrecognised falls back to the answer that changes nothing.
	return readChoice(valid, "w")
}

// lossLine says how many messages are newer than the checkpoint, which nothing
// covers.
func lossLine(c agent.ResumePlan) string {
	if c.Unrepresented == 0 {
		return "     instead, and the summary covers all of them."
	}
	return fmt.Sprintf("     instead, and %d of them are newer than it, so nothing covers those.",
		c.Unrepresented)
}

// tokenCount keeps one decimal place, which compactInt rounding would hide on
// a narrow window.
func tokenCount(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprint(n)
}

// confirmCloudSwitch asks before moving a live session to a cloud provider.
// model is the model being moved to, or "" when the provider has not been asked
// because asking would itself be the first contact with it.
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

// readYesNo reads a single confirmation line, and requires an explicit "y": an
// ambiguous answer sends nothing.
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
