package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"ai-code/internal/agent"
	"ai-code/internal/provider"
	"ai-code/internal/render"
)

// cmdConsult asks the model to argue against the session's own work. It
// inherits the conversation, runs at the strongest thinking, and uses no
// tools; the answer is appended so the main loop sees the critique.
func (a *App) cmdConsult(ctx context.Context, args string) error {
	if len(a.agent.Messages()) == 0 {
		return errors.New("there is nothing to consult about yet")
	}
	if a.client.Class() == provider.ClassCloud {
		return errors.New("sub-agents do not run on a cloud provider. Switch to an on-premises model first")
	}

	style := render.NewStyle(a.screen.Color())
	a.echoPrompt("/consult " + args)
	a.out(style.Dim("Reviewing the session…"))

	// No tools: a reviewer that reads files is doing the work again.
	child, err := a.agent.Spawn(agent.Child{
		System:  consultSystemPrompt,
		Effort:  provider.EffortHigh,
		NoTools: true,
		Sink:    a.interactiveSink(),
	})
	if err != nil {
		return err
	}

	// The transcript is fitted to the loaded model's window, as a turn would be.
	limit, _ := contextLimitFor(a.model, a.cfg.Agent.ContextOverride)
	transcript := serialiseForReview(a.agent.MessagesFitting(limit))

	focus := strings.TrimSpace(args)
	if focus == "" {
		focus = "the work as a whole"
	}
	prompt := fmt.Sprintf("<session>\n%s\n</session>\n\nFocus on: %s", transcript, focus)

	return a.runChild(ctx, child, prompt, "consultation")
}

// runChild drives a child agent with the interrupt handling a turn gets, and
// folds its answer back where the caller asked for it.
func (a *App) runChild(ctx context.Context, child *agent.Agent, input, foldAs string) error {
	ctx, cancel := context.WithCancel(ctx)
	restore := installInterrupt(cancel)
	defer restore()
	defer cancel()

	if err := child.Run(ctx, input); err != nil {
		return err
	}
	answer := strings.TrimSpace(child.LastAssistantText())
	if answer == "" {
		return errors.New("it returned nothing")
	}
	if foldAs != "" {
		a.agent.AppendUser(fmt.Sprintf("<%s>\n%s\n</%s>", foldAs, answer, foldAs))
		a.recordNew()
	}
	return nil
}

// interactiveSink is where a child's output goes: the renderer the main loop
// writes to.
func (a *App) interactiveSink() agent.Sink {
	if a.interactive != nil {
		return a.interactive
	}
	return nil
}

const consultSystemPrompt = `You are reviewing a coding session that another agent is part-way through. Your job is to find what is wrong with it.

Assume the work contains a mistake and look for it. The agent that did it has
already convinced itself; agreeing is only useful if you tried hard not to.

Look for, in order:
- a misunderstanding of the request, or a requirement quietly dropped
- code that is wrong, not merely inelegant: wrong condition, wrong order,
  unhandled case, a claim the evidence in the session does not support
- an approach that works but will not survive the next change

You have no tools, so do not propose reading anything. Work from the session.
Where you are unsure, say what would settle it.

Be specific and short. Name files and functions. If the work is sound, say so
in one line and name the strongest remaining risk -- do not invent a problem
to justify the exercise.`

// serialiseForReview flattens the conversation into something to read, not
// continue.
func serialiseForReview(msgs []provider.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case provider.RoleUser:
			fmt.Fprintf(&b, "\n## User\n%s\n", m.Content)
		case provider.RoleAssistant:
			if m.Content != "" {
				fmt.Fprintf(&b, "\n## Agent\n%s\n", m.Content)
			}
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(&b, "\n## Agent ran %s\n%s\n", tc.Name, truncate(tc.Args, 500))
			}
		case provider.RoleTool:
			fmt.Fprintf(&b, "\n## Result\n%s\n", truncate(m.Content, 2000))
		}
	}
	return b.String()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + fmt.Sprintf("\n… (%d more characters)", len(s)-n)
}
