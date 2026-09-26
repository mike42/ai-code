package agent

import "ai-code/internal/provider"

// Token accounting.
//
// There are exactly two quantities in this system, and nothing else may be
// called "tokens used".
//
// A *prefill* is what the backend reported for a request that was actually
// sent: the tokens the model had to read. It is ground truth, and it is
// recorded on the assistant message that request produced
// (provider.Message.PromptTokens), never in a field on the agent.
//
// A *projection* is what the next request will cost. It is never measured,
// always derived: the prefill of the newest response that still describes the
// current request prefix, plus an estimate of everything appended since.
//
// Both describe a request. Neither describes the transcript on disk. The one
// place a whole-session figure is still wanted -- costing a move to a narrower
// window -- asks for it by name, through TranscriptTokens, and gets an
// estimate that says so.

// promptTokensOf normalises a usage report into "tokens the model had to read".
//
// Every backend ai-code speaks to serves the OpenAI chat-completions shape,
// where prompt_tokens already includes any prefix served from cache and
// prompt_tokens_details.cached_tokens is a subset of it rather than an
// addition. Cached tokens are cheaper, not absent: they still occupy the
// window, so they belong in the figure the window is measured against.
func promptTokensOf(u provider.Usage) int {
	if u.PromptTokens > 0 {
		return u.PromptTokens
	}
	// Some servers report only a total. The completion is the part of it that
	// is not prefill, so the subtraction is exact where both are present.
	if u.TotalTokens > u.CompletionTokens {
		return u.TotalTokens - u.CompletionTokens
	}
	return 0
}

// defaultCharsPerToken is the ratio used until a response has been measured.
// English prose is near four; code and paths are denser, which is why the
// number is calibrated rather than trusted.
const defaultCharsPerToken = 4.0

// charsPerTokenNow returns the calibrated ratio, or the default.
func (a *Agent) charsPerTokenNow() float64 {
	if a.charsPerToken > 0 {
		return a.charsPerToken
	}
	return defaultCharsPerToken
}

// estimate is the only token estimator in the system.
//
// Every caller that needs the size of something not yet sent goes through
// here: the cut point, assembly, the summarisation budget, the display. A
// second estimator with its own ratio is how the tail came to be cut a third
// larger than the budget it was cut against, and the fix is not a better
// second ratio, it is that there is no second ratio.
//
// The result carries the sign of its argument: a negative size is a
// conversation that got smaller, and callers subtract.
func (a *Agent) estimate(chars int) int {
	return int(float64(chars) / a.charsPerTokenNow())
}

func (a *Agent) estimateMessages(msgs []provider.Message) int {
	return a.estimate(charsOf(msgs))
}

const (
	// minCalibrationChars is the smallest sample worth learning from. Below it
	// the per-message overhead constants are a large share of the total, so the
	// ratio measures the approximation rather than the tokenizer. Every real
	// request clears this easily: the system prompt and the tool schemas alone
	// are several thousand characters before the conversation starts.
	minCalibrationChars = 2000
	// The plausible range for a byte-pair tokenizer over text a coding agent
	// sends.
	minCharsPerToken = 1.5
	maxCharsPerToken = 12.0
)

// believableSize reports whether a reported figure can be believed as the
// full size of a request of this many characters.
//
// Two rules, because they catch different lies. The ratio band catches a
// figure that is not a token count at all. The comparison against the current
// estimate catches the one that matters: llama.cpp can be built to report only
// the tokens it actually processed, so a prompt-cache hit arrives as a
// prompt_tokens far below the real prefill. That reading passes the ratio band
// comfortably -- 40% of a true 3.0 ratio reads as 7.5, well inside it -- and
// believing it would halve the displayed occupancy and suppress a compaction
// that was due.
//
// Deliberately one-sided. A figure above the estimate is conservative: it can
// only bring a compaction forward, and the estimate is the thing more likely
// to be wrong. A figure below it hides an overflow.
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
	// Only once a ratio has been learned, because before that the estimate is
	// the nominal four and is itself the less trustworthy number.
	if a.charsPerToken > 0 && reported*2 < a.estimate(chars) {
		return false
	}
	return true
}

// recordUsage attaches ground truth to the assistant message it describes, and
// learns the ratio from the same measurement.
//
// Returns false when the report was not believable, in which case the message
// carries no prefill and projection falls back to estimating it -- the safe
// direction, because an estimate of a request that was sent is still bounded
// by the messages that are visibly in it.
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

// requestChanged records that everything in front of the conversation has
// moved, so no prefill recorded before now still describes a request prefix.
//
// Called by every operation that changes what a request carries ahead of the
// newest messages: compaction, pruning, a model swap, replacing the
// transcript. It is a position rather than a flag, so a response that arrives
// afterwards re-anchors by simply being newer, and nothing has to remember to
// clear anything.
func (a *Agent) requestChanged() { a.requestChangedAt = len(a.messages) }

// lastReportedAt is the newest message whose reported prefill still describes the
// prefix of the request about to be sent, or -1 when there is none.
func (a *Agent) lastReportedAt(first int) int {
	floor := max(first, a.requestChangedAt)
	for i := len(a.messages) - 1; i >= floor; i-- {
		if a.messages[i].Role == provider.RoleAssistant && a.messages[i].PromptTokens > 0 {
			return i
		}
	}
	return -1
}

// Projected is what the next request will occupy.
func (a *Agent) RequestTokens() int {
	n, _ := a.project()
	return n
}

// project measures what will actually be sent.
func (a *Agent) project() (tokens int, anchored bool) { return a.sizeOf(a.request()) }

// plannedTokens measures the request the cut describes, before any
// emergency trim. This is the guard's quantity: a trimmed request fits by
// construction, so triggering off one is triggering off nothing.
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

// ContextState is what the display needs. It has one producer -- this method,
// reached only through the event stream -- and no consumer recomputes it.
func (a *Agent) ContextState() ContextState {
	n, anchored := a.project()
	return ContextState{Projected: n, Window: a.opts.ContextLimit, Anchored: anchored}
}

// publish emits the context figure. Called after every change to what the next
// request carries, so no consumer ever holds a value the agent has moved on
// from.
func (a *Agent) publish() {
	a.emit(Event{Kind: EvContext, Context: ptr(a.ContextState())})
}

// TranscriptTokens estimates what the whole conversation would cost if all of
// it were sent.
//
// Not a projection and deliberately not anchored: it describes messages that
// are largely not in any request, so no reported figure covers them. Its one
// caller is the cost of moving to a narrower window, where an estimate is what
// the question deserves.
func (a *Agent) TranscriptTokens() int {
	return max(a.estimate(a.transcriptChars()), 0)
}
