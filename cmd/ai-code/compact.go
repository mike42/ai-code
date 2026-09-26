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

// cmdCompact makes room in the context, at a natural break rather than at the
// moment the window happens to fill.
//
// The same path the agent takes automatically when the session reaches the
// reserve -- pruning first, then summarising if that was not enough -- so a
// session compacted by hand and one compacted automatically end up in the
// same shape. A summary taken when a piece of work is finished is a better
// checkpoint than one taken mid-edit, which is the whole reason to run it
// yourself.
func (a *App) cmdCompact(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())

	if len(a.agent.Messages()) < 2 {
		return fmt.Errorf("there is nothing to compact yet")
	}

	before := a.contextState().Projected

	// Free tool output first. It costs no model call, and on a session heavy
	// with reads and greps it is most of the saving -- so the user sees the
	// number move before the summarisation they are waiting on has started.
	cleared := a.agent.ClearOldOutput()

	// Said before the wait, not after it, and only when there is going to be
	// one: deciding it is not worth compacting costs no model call.
	if a.agent.CompactionWorthwhile(a.cfg.Agent.CompactKeepRecentTokens) {
		a.out("", style.Dim("Summarising the older messages…"))
	}

	// Keep the tail verbatim: it is what the model is in the middle of, and
	// paraphrasing it loses the detail still in play. Budgeted in tokens, and
	// with the same budget the automatic path uses.
	res, err := a.agent.Compact(ctx, a.cfg.Agent.CompactKeepRecentTokens)
	if errors.Is(err, agent.ErrNothingToFree) {
		// The session is smaller than the recent messages a compaction would
		// keep anyway, so it would add a summary in front of everything and
		// leave a bigger request than it started with. The user asked for a
		// summary at a natural break, though, and that part still stands:
		// save one without changing what gets sent, ready for a smaller model.
		res, err = a.agent.Summarise(ctx, a.agent.SummaryCap())
	}
	if err != nil {
		return err
	}
	res.TokensBefore = before
	res.Cleared, res.ClearedTokens = cleared.Results, cleared.Tokens

	if a.sess != nil {
		// Nothing was replaced, so nothing is re-recorded: the transcript in
		// the file is already whole and stays that way. What is appended is
		// the boundary the next request starts from, so a resumed session
		// assembles from the same place this one does.
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

	// Both figures are the same quantity measured the same way: what the next
	// request would have cost, and what it will cost now. Reporting a
	// provider-anchored "before" against an estimated "after" is how a
	// compaction came to announce a saving that was really the gap between
	// two rulers.
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

	var did []string
	if res.Cleared == 1 {
		did = append(did, "pruned 1 tool result")
	} else if res.Cleared > 1 {
		did = append(did, fmt.Sprintf("pruned %d tool results", res.Cleared))
	}
	did = append(did, fmt.Sprintf("summarised the first %d messages, kept %d",
		res.SummarisedThrough, res.MessagesAfter))

	a.out(
		fmt.Sprintf("Next request: %s → %s of a %s window (%s freed; %s).",
			compactInt(res.TokensBefore), compactInt(res.TokensAfter),
			compactInt(a.contextState().Window), compactInt(saved),
			strings.Join(did, ", ")),
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
