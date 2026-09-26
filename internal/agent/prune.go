package agent

import (
	"fmt"
	"strings"

	"ai-code/internal/provider"
)

// Pruning reclaims context without a model call.

// clearedMarker opens the text left in place of removed output; the call and
// its arguments stay, only the output goes.
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

// ClearOldOutput empties the output of completed tool results old enough to
// lose, protecting three things: the last two turns, which the model is
// working from; the most recent results up to half the tail budget, so the
// tail can still be continued from; and anything already pruned, which ends
// the walk.
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
		// The messages changed under every recorded figure, so ground truth is
		// invalidated as a compaction would invalidate it.
		a.requestChanged()
		a.publish()
	}
	return res
}

// recentTurnsKept is how much of the recent conversation is never pruned.
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
