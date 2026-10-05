package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"ai-code/internal/agent"
	"ai-code/internal/render"
	"ai-code/internal/session"
)

// cmdCompact asks the model to summarise the session at a break chosen by
// hand. Old tool output is left alone: clearing it first would change the
// request the server has cached, and the summary would be read from scratch.
func (a *App) cmdCompact(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())

	if len(a.agent.Messages()) < 2 {
		return fmt.Errorf("there is nothing to compact yet")
	}

	before := a.contextState().Projected

	// Said only when there will be a wait: deciding not to compact is free.
	if a.agent.CompactionWorthwhile(a.cfg.Agent.CompactKeepRecentTokens) {
		a.out("", style.Dim("Summarising the older messages…"))
	}

	// The tail stays verbatim, under the automatic path's token budget.
	res, err := a.agent.Compact(ctx, a.cfg.Agent.CompactKeepRecentTokens)
	if errors.Is(err, agent.ErrNothingToFree) {
		// Nothing to free: save a summary without changing what gets sent.
		res, err = a.agent.Summarise(ctx, a.agent.SummaryCap())
	}
	if err != nil {
		return err
	}
	res.TokensBefore = before

	if a.sess != nil {
		// Appended is the boundary the next request starts from, so a resume
		// assembles the same way.
		_ = a.sess.Append(session.Entry{
			Type:              session.EntryCheckpoint,
			Summary:           res.Summary,
			SummarisedThrough: res.SummarisedThrough,
			Cut:               res.SummarisedThrough,
			MessagesBefore:    res.MessagesBefore,
			TokensBefore:      res.TokensBefore,
		})
		a.recordedSummary = res.Summary
	}

	// Both figures are the same quantity measured the same way.
	saved := max(res.TokensBefore-res.TokensAfter, 0)
	if !res.Rewrote {
		a.out(
			fmt.Sprintf("Nothing to free. The next request is %s, which is already smaller "+
				"than the %s of recent messages a compaction keeps word for word anyway.",
				compactInt(res.TokensAfter), compactInt(a.agent.KeepRecentTokens())),
			style.Dim("A summary was saved in case you switch to a model with a smaller window."),
			"")
		return nil
	}

	a.out(
		fmt.Sprintf("Next request: %s → %s of a %s window (%s freed; summarised the first %d messages, kept %d).",
			compactInt(res.TokensBefore), compactInt(res.TokensAfter),
			compactInt(a.contextState().Window), compactInt(saved),
			res.SummarisedThrough, res.MessagesAfter),
		style.Dim("Nothing was deleted. The session on disk keeps every message; only what gets sent is shorter."),
		"")

	if args == "show" {
		lines := []string{""}
		for _, l := range strings.Split(res.Summary, "\n") {
			lines = append(lines, style.Dim(l))
		}
		a.out(append(lines, "")...)
	}
	return nil
}
