package agent

import (
	"fmt"
	"strings"

	"ai-code/internal/provider"
)

// Pruning: reclaiming context without a model call.
//
// Most of the weight in a coding session is tool output, and most of it is
// dead the moment the model has read it -- a directory listing from forty
// turns ago, the grep that found the file already open. Summarising it costs
// a whole model call, which on a local backend is minutes; emptying it costs
// nothing.
//
// So this runs first, and a compaction only happens when it was not enough.
// On a session heavy with tool results that turns most compactions into an
// operation with no latency at all.

// clearedMarker opens the text left in place of removed output. The call and
// its arguments stay, because "we ran this" is the part the model reasons
// from; only the output goes.
const clearedMarker = "[Output removed to free context"

func clearedNotice(chars int) string {
	return fmt.Sprintf("%s: %s of output from a completed call.]", clearedMarker, compactChars(chars))
}

func isCleared(m provider.Message) bool {
	return strings.HasPrefix(m.Content, clearedMarker)
}

// Cleared is what one pass recovered.
type Cleared struct {
	Results int
	Tokens  int
}

// Prune empties the output of completed tool results that are old enough to
// be safe to lose.
//
// Three things are protected, and each for its own reason:
//
//   - the last two turns, whatever their size, because that is what the model
//     is working from right now;
//   - the most recent results up to half the verbatim tail budget, because a
//     tail of nothing but "output removed" is a tail that cannot be continued
//     from, whatever its token count says;
//   - anything already pruned, which also ends the walk: everything older has
//     been through this before.
//
// Derived from the tail budget rather than fixed, so it stays coherent across
// window sizes for the same reason every other size here is derived.
func (a *Agent) ClearOldOutput() Cleared {
	budget := a.budget().KeepOutput
	var res Cleared
	protected, turns := 0, 0

	for i := len(a.messages) - 1; i >= 0; i-- {
		m := &a.messages[i]
		if m.Role == provider.RoleUser {
			turns++
			continue
		}
		if m.Role != provider.RoleTool || m.Content == "" {
			continue
		}
		if isCleared(*m) {
			break
		}
		if turns < recentTurnsKept {
			protected += a.estimate(messageChars(*m))
			continue
		}
		if protected < budget {
			protected += a.estimate(messageChars(*m))
			continue
		}
		chars := len(m.Content)
		m.Content = clearedNotice(chars)
		res.Results++
		res.Tokens += a.estimate(chars - len(m.Content))
	}

	if res.Results > 0 {
		// The messages that were sent have changed underneath every figure
		// recorded against them, so ground truth is invalidated the same way
		// a compaction invalidates it.
		a.requestChanged()
		a.publish()
	}
	return res
}

// recentTurnsKept is how much of the recent conversation is never pruned. Two,
// because one is only the exchange in progress: the model regularly reads a
// file in one turn and edits it in the next, and emptying the read between
// them is how a session starts re-reading what it already has.
const recentTurnsKept = 2

// compactChars formats a byte count for a message a person reads.
func compactChars(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d bytes", n)
	}
}
