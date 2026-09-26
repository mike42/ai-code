package agent

import (
	"fmt"
	"strings"

	"ai-code/internal/tool"
)

// stuckAfterTurns is how many turns of the same failure it takes before the
// suggestion is worth making.
const stuckAfterTurns = 6

// stuckWatch notices the same failure coming back turn after turn, which the
// loop guard cannot see because the calls that produce it differ.
type stuckWatch struct {
	// lastError is the most recent error result; repeats how many turns running
	// it has come back.
	lastError string
	repeats   int
	suggested bool
}

// observe folds in one turn's tool results.
func (w *stuckWatch) observe(results []tool.Result) {
	if len(results) == 0 {
		return
	}

	errText := ""
	for _, r := range results {
		if r.IsError && errText == "" {
			errText = normaliseError(r.Content)
		}
	}

	switch {
	case errText == "":
		w.lastError, w.repeats = "", 0
	case errText == w.lastError:
		w.repeats++
	default:
		w.lastError, w.repeats = errText, 1
	}
}

// suggestion is the one line to show, or "" for nothing; it fires once per session.
func (w *stuckWatch) suggestion() string {
	if w.suggested || w.repeats < stuckAfterTurns {
		return ""
	}
	w.suggested = true
	return fmt.Sprintf(
		"The same error has come back %d turns running. /consult asks the model to "+
			"argue against its own approach, which is sometimes what breaks this.", w.repeats)
}

// normaliseError strips the parts of a message that differ between identical
// failures.
func normaliseError(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	var b strings.Builder
	for _, r := range s {
		if r >= '0' && r <= '9' {
			continue
		}
		b.WriteRune(r)
	}
	return strings.TrimSpace(b.String())
}
