package agent

import "ai-code/internal/provider"

// Token accounting has three quantities: the prefill a backend reported for a
// sent request, recorded on the assistant message; the projection of the next
// request, derived from it; and a separate transcript estimate.

// promptTokensOf normalises a usage report into "tokens the model had to read".
// Cached tokens are a subset of prompt_tokens, not an addition: they still
// occupy the window, so they belong in the figure the window is measured against.
func promptTokensOf(u provider.Usage) int {
	if u.PromptTokens > 0 {
		return u.PromptTokens
	}
	// Some servers report only a total; the completion is the part that is not prefill.
	if u.TotalTokens > u.CompletionTokens {
		return u.TotalTokens - u.CompletionTokens
	}
	return 0
}

// defaultCharsPerToken is the ratio used until a response has been measured.
const defaultCharsPerToken = 4.0

// charsPerTokenNow returns the calibrated ratio, or the default.
func (a *Agent) charsPerTokenNow() float64 {
	if a.charsPerToken > 0 {
		return a.charsPerToken
	}
	return defaultCharsPerToken
}

// estimate is the only token estimator in the system; every caller that needs
// the size of something not yet sent goes through here, so no second ratio can
// disagree with it. The result carries the sign of its argument.
func (a *Agent) estimate(chars int) int {
	return int(float64(chars) / a.charsPerTokenNow())
}

func (a *Agent) estimateMessages(msgs []provider.Message) int {
	return a.estimate(charsOf(msgs))
}

const (
	// minCalibrationChars is the smallest sample worth learning from: below it
	// the per-message constants outweigh the tokenizer.
	minCalibrationChars = 2000
	// The plausible range for a tokenizer over text a coding agent sends.
	minCharsPerToken = 1.5
	maxCharsPerToken = 12.0
)

// believableSize reports whether a reported figure can be believed as the
// full size of a request of this many characters. The ratio band catches a
// non-count; the estimate comparison catches a low prompt-cache hit.
func (a *Agent) believableSize(chars, reported int) bool {
	if reported <= 0 {
		return false
	}
	if chars < minCalibrationChars {
		// Too small to judge, and too small to matter either way.
		return true
	}
	if r := float64(chars) / float64(reported); r < minCharsPerToken || r > maxCharsPerToken {
		return false
	}
	// Only once a ratio has been learned: before that the estimate is the nominal four.
	if a.charsPerToken > 0 && reported*2 < a.estimate(chars) {
		return false
	}
	return true
}

// recordUsage attaches ground truth to the assistant message it describes and
// learns the ratio from the same measurement. False means the report was not
// believable and no prefill was recorded.
func (a *Agent) recordUsage(m *provider.Message, chars int, u provider.Usage) bool {
	p := promptTokensOf(u)
	if !a.believableSize(chars, p) {
		return false
	}
	if chars >= minCalibrationChars {
		a.charsPerToken = float64(chars) / float64(p)
	}
	m.PromptTokens = p
	m.Completion = u.CompletionTokens
	return true
}

// requestChanged records that everything in front of the transcript has moved,
// so no earlier prefill still describes the request prefix. It is a position,
// not a flag: a later response re-anchors simply by being newer.
func (a *Agent) requestChanged() { a.requestChangedAt = len(a.messages) }

// lastReportedAt is the newest message whose reported prefill still describes
// the prefix of the request about to be sent, or -1.
func (a *Agent) lastReportedAt(first int) int {
	floor := max(first, a.requestChangedAt)
	for i := len(a.messages) - 1; i >= floor; i-- {
		if a.messages[i].Role == provider.RoleAssistant && a.messages[i].PromptTokens > 0 {
			return i
		}
	}
	return -1
}

// RequestTokens is what the next request will occupy.
func (a *Agent) RequestTokens() int {
	n, _ := a.project()
	return n
}

// project measures what will actually be sent.
func (a *Agent) project() (tokens int, anchored bool) { return a.sizeOf(a.request()) }

// plannedTokens measures the request the cut describes, before any emergency
// trim; a trimmed request always fits, so triggering off one triggers off nothing.
func (a *Agent) plannedTokens() int {
	n, _ := a.sizeOf(a.plannedRequest())
	return n
}

func (a *Agent) sizeOf(r request) (tokens int, anchored bool) {
	if r.exact {
		if i := a.lastReportedAt(r.first); i >= 0 {
			n := a.messages[i].PromptTokens + a.messages[i].Completion
			n += a.estimateMessages(a.messages[i+1:])
			return max(n, 0), true
		}
	}
	return max(a.estimate(a.fixedChars()+charsOf(r.msgs)), 0), false
}

// ContextState is what the display needs, produced only here and reached through events.
func (a *Agent) ContextState() ContextState {
	n, anchored := a.project()
	return ContextState{Projected: n, Window: a.opts.ContextLimit, Anchored: anchored}
}

// publish emits the context figure after every change to what the next request carries.
func (a *Agent) publish() {
	a.emit(Event{Kind: EvContext, Context: ptr(a.ContextState())})
}

// TranscriptTokens estimates what the whole conversation would cost if all of
// it were sent. Not a projection and not anchored: no reported figure covers
// messages that are largely not in any request.
func (a *Agent) TranscriptTokens() int {
	return max(a.estimate(a.transcriptChars()), 0)
}
