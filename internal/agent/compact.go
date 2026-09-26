package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"ai-code/internal/provider"
)

// Compaction summarises a session so it fits back inside the context window.
//
// The prompt structure and the serialisation approach below are adapted from
// pi (github.com/badlogic/pi, MIT, (c) 2025 Mario Zechner),
// packages/coding-agent/src/core/compaction/{compaction,utils}.ts: a fixed
// section format, a flat transcript rather than replayed messages, and file
// lists carried across the boundary.
//
// What the format defends first is the user's original intent and any
// correction they made.

const summarisationSystemPrompt = `You are a context summarization assistant. Read the conversation you are given and produce a structured summary in the exact format requested.

Do NOT continue the conversation. Do NOT answer any question that appears in it. Output only the summary.`

const summarisationPrompt = `The transcript above is a conversation to summarise. Produce a context checkpoint that another model will use to continue this work with none of the original messages available.

Use this EXACT format:

## Goal
[What is the user trying to accomplish, in their terms. Multiple items if the session covered several tasks.]

## Constraints & Preferences
- [Constraints, preferences and requirements the user stated, including corrections they made to earlier work]
- [Or "(none)" if none were stated]

## Progress
### Done
- [x] [Completed work]

### In Progress
- [ ] [What is underway right now]

### Blocked
- [Anything preventing progress, or "(none)"]

## Key Decisions
- **[Decision]**: [Why]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, commands or references needed to continue]
- [Or "(none)"]

Keep every section concise. Preserve exact file paths, function names, commands and error messages verbatim -- an approximate path is worse than no path.`

// updateSummarisationPrompt is used when a checkpoint already exists.
//
// Taken from pi's UPDATE_SUMMARIZATION_INSTRUCTIONS, for the reason its
// existence implies: the second compaction of a session sees only the messages
// since the first, so a prompt that just says "summarise this" quietly drops
// everything before it. The rules that carry state forward -- preserve, move
// items from in-progress to done, update rather than restate -- are what make
// repeated compaction converge instead of erode.
const updateSummarisationPrompt = `The transcript above is NEW conversation to fold into the existing checkpoint in <previous-summary> tags.

RULES:
- PRESERVE every fact from the previous summary that is still true
- ADD the new progress, decisions and context from the transcript
- MOVE items from "In Progress" to "Done" when the transcript shows them completed
- UPDATE "Next Steps" to reflect what is left, not what was left before
- PRESERVE exact file paths, function names, commands and error messages verbatim
- REMOVE only what the transcript shows is no longer relevant

Use this EXACT format:

## Goal
[Preserve existing goals; add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add any newly stated]
- [Or "(none)"]

## Progress
### Done
- [x] [Previously done items AND newly completed ones]

### In Progress
- [ ] [What is underway now]

### Blocked
- [Current blockers only -- drop the ones that were resolved, or "(none)"]

## Key Decisions
- **[Decision]**: [Why] (keep all previous, add new)

## Next Steps
1. [Ordered, based on the current state]

## Critical Context
- [Carry forward what is still needed, add what is new]
- [Or "(none)"]

Keep every section concise. Preserve exact file paths, function names, commands and error messages verbatim -- an approximate path is worse than no path.`

// SummaryMaxTokens bounds a checkpoint the user asked for. The reserve is
// sized to hold it.
const SummaryMaxTokens = 4096

// SpeculativeSummaryMaxTokens bounds a checkpoint nobody asked for: one
// written while the session is idle, purely so that a later model swap is
// cheap, and thrown away unread if no swap happens.
//
// Far smaller than SummaryMaxTokens because on a local model the cap is the
// wall-clock cost, not a token cost. At a few tokens a second, 4096 tokens is
// minutes of generation -- long enough that the checkpoint is still being
// written when the swap it was meant to make cheap arrives, which defeats the
// entire purpose of writing it early.
const SpeculativeSummaryMaxTokens = 800

// startPoint finds where to cut the conversation, keeping roughly
// keepRecentTokens of the tail verbatim.
//
// A token budget rather than a message count. Message count is the wrong unit
// because messages are not a consistent size: six messages is a couple of
// hundred tokens after a short exchange and forty thousand after a round of
// greps, so a fixed count either keeps too little to continue from or too much
// to fit.
//
// The unit of the walk is a turn -- a user message and everything that
// followed it -- rather than a message. A turn is what the summary prompt is
// written to describe, and it is the boundary at which the kept tail reads as
// a conversation rather than as the back half of one. Cutting mid-turn strands
// an assistant message whose reasoning was summarised away from it.
//
// Measured with the agent's own estimator, which is the point of it being a
// method. The free function it replaced divided by a fixed four characters a
// token while every other part of the system divided by the calibrated ratio,
// so the tail it kept was larger than the budget it was cut against by
// whatever the two ratios differed by -- around a third on a coding session.
func (a *Agent) startPoint(keepRecentTokens int) int {
	msgs := a.messages
	if keepRecentTokens <= 0 || len(msgs) == 0 {
		return len(msgs)
	}

	starts := turnStarts(msgs)
	if len(starts) == 0 {
		// No user message anywhere: nothing divides this into turns, so fall
		// back to splitting it as one.
		return a.startInsideTurn(0, len(msgs), keepRecentTokens)
	}

	// Whole turns from the newest backwards, while they fit.
	used, cut := 0, len(msgs)
	for i := len(starts) - 1; i >= 0; i-- {
		end := len(msgs)
		if i+1 < len(starts) {
			end = starts[i+1]
		}
		size := a.estimateMessages(msgs[starts[i]:end])
		if used+size > keepRecentTokens {
			break
		}
		used += size
		cut = starts[i]
	}

	if cut == len(msgs) {
		// The newest turn alone is larger than the whole tail budget. Keep as
		// much of its end as fits rather than none of it: the model is in the
		// middle of that turn, and handing it back a summary of work it has
		// not finished is worse than handing it back the recent half verbatim.
		last := starts[len(starts)-1]
		cut = a.startInsideTurn(last, len(msgs), keepRecentTokens)
	}

	// Everything kept and nothing to summarise is not a compaction. Leave the
	// oldest turn on the summary side so the call has material.
	if cut == 0 {
		if len(starts) > 1 {
			cut = starts[1]
		} else {
			cut = a.startInsideTurn(0, len(msgs), keepRecentTokens)
		}
	}
	return legalStart(msgs, cut)
}

// startInsideTurn cuts inside a single turn, at the earliest point whose
// remainder fits the budget.
//
// Forwards from the start rather than backwards from the end, so the cut is
// the one that keeps the most: the first position that fits is the earliest
// one, and everything after it fits too.
func (a *Agent) startInsideTurn(start, end, budget int) int {
	for i := start + 1; i < end; i++ {
		if a.estimateMessages(a.messages[i:end]) <= budget {
			return legalStart(a.messages, i)
		}
	}
	// Not even the last message fits. Keep it anyway -- the guard's
	// escalation deals with a single message too large for the window, and it
	// can say which one, which this cannot.
	return legalStart(a.messages, end-1)
}

// legalStart walks forwards off any tool result, onto the next message that can
// begin a conversation.
//
// A result separated from the call that produced it is exactly the
// message-shape violation Validate exists to catch.
func legalStart(msgs []provider.Message, cut int) int {
	for cut < len(msgs) && msgs[cut].Role == provider.RoleTool {
		cut++
	}
	return cut
}

// turnStarts indexes every message that begins a turn.
//
// A turn starts at a user message. Steering appends one mid-exchange, which
// makes it a legal cut point too -- it is a place the conversation could have
// started, which is the only property a cut needs.
func turnStarts(msgs []provider.Message) []int {
	var out []int
	for i, m := range msgs {
		if m.Role == provider.RoleUser {
			out = append(out, i)
		}
	}
	return out
}

// MessagesFitting returns the conversation to send to a model that has this
// many tokens of room: the whole transcript when it fits, and otherwise the
// summary followed by as much of the recent conversation as is left over.
//
// The decision belongs at the moment of sending, not of summarising, because
// the window belongs to whichever model is loaded. Swapping to a narrower one
// carries less; swapping back still has every message.
//
// The window is a whole request's worth of room, so the system prompt and tool
// schemas are charged against it here.
func (a *Agent) MessagesFitting(window int) []provider.Message {
	if window <= 0 || a.estimate(a.transcriptChars()) <= window {
		return a.messages
	}
	return a.assemble(window, a.resumeSummary())
}

// resumeSummary is the checkpoint assembly may use, which is none when the
// user chose the transcript over it.
func (a *Agent) resumeSummary() string {
	if a.resumeFromTranscript {
		return ""
	}
	return a.summary
}

// assemble builds a request of at most window tokens from the given checkpoint
// and as much of the tail as is left over.
func (a *Agent) assemble(window int, summary string) []provider.Message {
	head := a.summaryHead(summary)
	first := a.firstKept(window, a.estimate(charsOf(head)))

	out := make([]provider.Message, 0, len(head)+len(a.messages)-first)
	out = append(out, head...)
	out = append(out, a.messages[first:]...)
	// The cut lands wherever the arithmetic put it, which is regularly between
	// an assistant's tool call and the result answering it. A result with no
	// call above it is rejected by the provider, not ignored by it.
	return dropOrphanToolResults(out)
}

// firstKept is the index of the earliest transcript message a window of this
// size has room for, once a checkpoint of headTokens sits in front of it.
//
// It stops at summarisedThrough rather than walking past it: the checkpoint
// above already says what those messages said, and repeating it spends the
// window twice on the same content. Reaching that floor means the request
// covers the whole session -- the recent part verbatim, the rest in summary.
func (a *Agent) firstKept(window, headTokens int) int {
	floor := 0
	if headTokens > 0 && a.summarisedThrough > 0 {
		floor = min(a.summarisedThrough, len(a.messages))
	}

	room := window - a.estimate(a.fixedChars()) - headTokens
	first := len(a.messages)
	for i := len(a.messages) - 1; i >= floor; i-- {
		room -= a.estimate(messageChars(a.messages[i]))
		if room < 0 {
			break
		}
		first = i
	}

	if first == len(a.messages) && floor == len(a.messages) {
		// Every message is already accounted for by the checkpoint, so the
		// checkpoint alone is a complete request rather than an empty one.
		//
		// Without this the fallback below fires instead and walks back onto
		// the last turn -- the very content the checkpoint was written to
		// replace. One oversized tool result then reappears in every request
		// for the rest of the session, and no amount of compacting removes it,
		// because summarising it is precisely what put the floor here.
		return first
	}

	if first == len(a.messages) && len(a.messages) > 0 {
		// A window too small for even the last message still has to produce a
		// request: an empty one is a 400 from every provider. Send the last
		// turn over budget instead -- it may be refused, but a request that
		// cannot be built is refused every time.
		first = len(a.messages) - 1
		for first > 0 && a.messages[first].Role == provider.RoleTool {
			first--
		}
	}
	return first
}

// summaryHead is the checkpoint as it goes on the wire: the original request
// verbatim, then the summary. Empty when there is no summary.
//
// The first user message is preserved beside the summary because a summary is
// a lossy paraphrase, and the one thing that must survive intact is what the
// user actually asked for.
func (a *Agent) summaryHead(summary string) []provider.Message {
	if strings.TrimSpace(summary) == "" {
		return nil
	}
	var out []provider.Message
	if first := firstUserMessage(a.messages); first != "" && a.estimate(len(first)) <= a.preservedRequestBudget() {
		if !strings.HasPrefix(first, preservedRequestPreamble) {
			// Already-compacted sessions carry the preamble in their first
			// message; a second copy of it reads as a stutter.
			first = preservedRequestPreamble + first
		}
		out = append(out, provider.Message{Role: provider.RoleUser, Content: first})
	}
	return append(out, provider.Message{
		Role: provider.RoleUser,
		Content: "<context-checkpoint>\nEarlier turns were summarised to free context. " +
			"This is the state of the work:\n\n" + summary + "\n</context-checkpoint>",
	})
}

const preservedRequestPreamble = "The original request that started this session, " +
	"preserved verbatim across compaction:\n\n"

// preservedRequestBudget bounds the opening request kept verbatim above a
// checkpoint.
//
// It is kept because it is the anchor the session is working towards and a
// summary of it reads second-hand. But an opening request can be a pasted
// file, and preserving that verbatim puts the very thing compaction was called
// to remove into every later request. Past this share the summary speaks for
// it instead.
func (a *Agent) preservedRequestBudget() int {
	return a.promptBudget() / 4
}

// promptBudget is how much of the window an assembled request may fill.
//
// Not all of it: outputBudget hands the model whatever is left of the budget,
// so a request assembled right up to the line would cap every answer for the
// rest of the session at the floor. A quarter of the budget held back is a
// reply's worth on any window wide enough to be working in.
func (a *Agent) promptBudget() int {
	return a.promptBudgetIn(a.opts.ContextLimit)
}

func (a *Agent) promptBudgetIn(limit int) int {
	return a.usableIn(limit) * 3 / 4
}

// request is what the next call will carry.
//
// exact says the list is precisely the checkpoint head followed by
// a.messages[first:], with nothing dropped in between. That is the ordinary
// case, and it is what makes a prefill recorded earlier still describe this
// request's prefix -- so it is the condition projection checks before
// believing one. When assembly has had to trim (a window that shrank under a
// running session), the prefix is a shape nothing was ever measured against,
// and the projection says so rather than guessing.
type request struct {
	msgs  []provider.Message
	first int
	exact bool
}

// intended is the request the cut describes: the checkpoint head followed by
// everything from the cut onwards, with nothing dropped to make it fit.
//
// This is what the guard measures, and the distinction is load-bearing. Ask
// for the request that will actually be sent and the answer has already been
// trimmed to fit, so it always fits, so the guard never fires and the session
// spends the rest of its life quietly dropping its oldest turns instead of
// summarising them.
//
// A compaction moves the cut; it does not rewrite a.messages. So the normal
// path here is a slice and a concatenation, with no arithmetic at all.
func (a *Agent) plannedRequest() request {
	first := 0
	var head []provider.Message
	if a.startAt > 0 && !a.resumeFromTranscript {
		first = min(a.startAt, len(a.messages))
		head = a.summaryHead(a.summary)
	}

	msgs := make([]provider.Message, 0, len(head)+len(a.messages)-first)
	msgs = append(msgs, head...)
	msgs = append(msgs, a.messages[first:]...)

	// A cut never lands on a tool result, so this is a no-op on every path
	// that chose one. It fires only for a transcript that arrived broken, and
	// then the prefix is not one anything measured.
	repaired := dropOrphanToolResults(msgs)
	exact := len(repaired) == len(msgs)
	msgs = repaired

	return request{msgs: msgs, first: first, exact: exact}
}

// request is what will actually go on the wire.
//
// Normally the intended one: by the time a request is built the guard has
// already made room for it. It differs only where the guard deliberately did
// not act -- the user who chose to lose their oldest turns rather than spend a
// narrow window on a checkpoint -- and where the window shrank under a running
// session. Then the oldest messages are dropped to fit, and the anchor goes
// with them: this prefix is not one any response was measured against.
func (a *Agent) request() request {
	r := a.plannedRequest()
	usable := a.Usable()
	if usable <= 0 {
		return r
	}
	// Room to answer in, not merely room to sit in -- the same threshold the
	// guard triggers on. Held to the plain budget instead, the two disagree by
	// exactly the output floor: the guard makes room because the reply will
	// not fit, then assembly hands back everything because it does, and
	// nothing is ever freed.
	if a.estimate(a.fixedChars()+charsOf(r.msgs)) <= usable-minOutputTokens {
		return r
	}
	return request{msgs: a.MessagesFitting(a.promptBudget()), exact: false}
}

// messagesToSend is what the next request carries.
func (a *Agent) messagesToSend() []provider.Message { return a.request().msgs }

// summaryCovers reports whether the checkpoint already in hand accounts for
// everything a request of this size would have to leave out. When it does,
// summarising again buys nothing and costs a whole model call -- which on a
// local model is the difference between a pause and a coffee break.
func (a *Agent) summaryCovers(window int) bool {
	head := a.summaryHead(a.summary)
	if len(head) == 0 || a.summarisedThrough <= 0 {
		return false
	}
	rest := a.messages[min(a.summarisedThrough, len(a.messages)):]
	return a.estimate(a.fixedChars()+charsOf(head)+charsOf(rest)) <= window
}

// ResumePlan is what one way of continuing in a narrower window would
// actually carry, in the numbers a person needs to choose between them.
type ResumePlan struct {
	// Available is false for the checkpoint plan when no checkpoint has been
	// written. Offering it anyway would mean summarising on the spot, which is
	// a model call at the one moment the user is waiting to get on with
	// something.
	Available bool
	// Prompt is what the assembled request would occupy; Free is what is left
	// of the budget for the model to answer in, which is the number that
	// decides whether the session is still workable.
	Prompt, Free int
	// Kept and Dropped split the transcript where the window forces it.
	Kept, Dropped int
	// Unrepresented is how many of the dropped messages nothing at all speaks
	// for -- all of them without a checkpoint, and with one, those written
	// after it was taken.
	Unrepresented int
}

// ResumeChoice is what a window of a given size leaves a session able to do.
type ResumeChoice struct {
	// Fits is true when the whole transcript still goes on the wire with room
	// to answer, which means there is nothing to decide.
	Fits bool
	// Session is the whole transcript's cost, whether or not it would be sent,
	// and Usable is what the window offers once the reserve is held back --
	// the two numbers whose gap is the reason there is anything to decide.
	Session, Usable        int
	Checkpoint, Transcript ResumePlan
}

// PlanResume costs each way of continuing in a window of contextLimit tokens.
//
// Arithmetic only, deliberately: this is called at the moment a model is
// swapped, and a swap that had to summarise first would stall on the model
// being swapped away from -- for minutes, on the small local models that are
// exactly the ones being swapped to.
func (a *Agent) PlanResume(contextLimit int) ResumeChoice {
	c := ResumeChoice{Session: a.TranscriptTokens(), Usable: a.usableIn(contextLimit)}

	usable := c.Usable
	// Room to answer in, not merely room to sit in: the same threshold
	// guardContext and messagesToSend use, so "it fits" here means the next
	// request really will carry everything.
	if usable <= 0 || c.Session <= usable-minOutputTokens {
		c.Fits = true
		return c
	}

	c.Checkpoint = a.resumePlan(contextLimit, a.summary)
	c.Checkpoint.Available = strings.TrimSpace(a.summary) != ""
	c.Transcript = a.resumePlan(contextLimit, "")
	return c
}

func (a *Agent) resumePlan(contextLimit int, summary string) ResumePlan {
	head := a.summaryHead(summary)
	msgs := a.assemble(a.promptBudgetIn(contextLimit), summary)

	kept := len(msgs) - len(head)
	dropped := len(a.messages) - kept
	unrepresented := dropped
	if len(head) > 0 {
		unrepresented = max(dropped-a.summarisedThrough, 0)
	}

	prompt := max(a.estimate(a.fixedChars()+charsOf(msgs)), 0)
	return ResumePlan{
		Available:     true,
		Prompt:        prompt,
		Free:          max(a.usableIn(contextLimit)-prompt, 0),
		Kept:          kept,
		Dropped:       dropped,
		Unrepresented: unrepresented,
	}
}

// CompactResult reports what a compaction did.
//
// Every figure here measures the *session*: the whole transcript, before and
// after. That needs saying because there are three plausible quantities in
// play -- the transcript, the request that would be sent now, and the window
// -- and reporting one where the reader assumes another is how a checkpoint
// that removed nothing came to be announced as "166 messages -> 4".
type CompactResult struct {
	Summary string
	// SummarisedThrough is how many messages from the start of the transcript
	// the summary accounts for, and -- when Rewrote is true -- the boundary
	// the next request starts from. A caller persists it so a resumed session
	// starts from the same place.
	SummarisedThrough int
	// Rewrote distinguishes the two things that both produce a checkpoint.
	// Summarise writes a summary beside the transcript and changes nothing
	// about what is sent; Compact also moves the boundary the next request
	// starts from. Neither removes a message. Without this a caller cannot
	// tell "freed nothing" from "nothing needed freeing", and both were being
	// printed as the former.
	Rewrote bool
	// TokensBefore and TokensAfter are projections: what the next request
	// would have cost, and what it will cost now. The same quantity measured
	// the same way, so the difference between them is a saving rather than
	// the gap between an estimate and a measurement.
	TokensBefore int
	TokensAfter  int
	// MessagesBefore and MessagesAfter count what the request carries, not
	// what the transcript holds -- the transcript only ever grows. Equal when
	// Rewrote is false.
	MessagesBefore int
	MessagesAfter  int
	// Cleared is how many tool results were emptied before any model call was
	// made, and ClearedTokens what that alone recovered.
	Cleared       int
	ClearedTokens int
}

// Compact makes room by summarising the older part of the session.
//
// Nothing is deleted. The summary is written, a boundary is recorded, and
// assembly starts the next request from that boundary -- so the transcript
// stays whole, a later swap to a wider model can still reach the messages,
// and a figure recorded against the old shape is invalidated by position
// rather than by having to remember to clear it.
//
// keepRecentTokens of zero means the size derived from the window, the same
// one the automatic path uses, so /compact and auto-compaction leave the
// session in the same shape.
func (a *Agent) Compact(ctx context.Context, keepRecentTokens int) (*CompactResult, error) {
	if len(a.messages) < 2 {
		return nil, errors.New("there is nothing to compact yet")
	}
	if keepRecentTokens <= 0 {
		keepRecentTokens = a.keepRecentTokens()
	}

	// The intended request throughout, on both sides of the comparison: the
	// one that would be sent has already been trimmed to fit, so measuring
	// that would report a saving against a number the trim had already made
	// small.
	before := a.plannedTokens()
	beforeCount := len(a.plannedRequest().msgs)
	cut := a.startPoint(keepRecentTokens)

	// Would it help? A checkpoint is not free space: it is text, and it goes
	// in front of everything the cut kept. On a session already smaller than
	// its own tail budget the cut moves a message or two and the summary adds
	// more than that back, so the "compaction" ends with a larger request
	// than it started with.
	//
	// Costed against a checkpoint as large as one is allowed to be, because
	// the real one does not exist yet and guessing low here would spend the
	// model call to find out.
	if !a.worthSummarising(cut, a.SummaryCap()) {
		return nil, ErrNothingToFree
	}

	summary, err := a.summariseThrough(ctx, cut, a.SummaryCap())
	if err != nil {
		return nil, err
	}

	// And now against the checkpoint that actually got written, because a
	// model does not always produce the short one it was asked for. Nothing
	// is installed unless it leaves the next request smaller: "reclaiming
	// never increases it" is not a tendency, it is the invariant that makes
	// the figure on screen worth reading.
	a.setSummary(summary, cut)
	after := a.plannedTokens()
	if after >= before {
		// Keep the summary as a standby for a narrower window -- it cost a
		// model call and it is still the best checkpoint available -- but
		// take back the boundary, so the next request is the one that was
		// already smaller.
		a.startAt = 0
		a.requestChanged()
		a.publish()
		return &CompactResult{
			Summary:           summary,
			SummarisedThrough: cut,
			Rewrote:           false,
			TokensBefore:      before,
			TokensAfter:       a.plannedTokens(),
			MessagesBefore:    beforeCount,
			MessagesAfter:     len(a.plannedRequest().msgs),
		}, nil
	}

	return &CompactResult{
		Summary:           summary,
		SummarisedThrough: cut,
		Rewrote:           true,
		TokensBefore:      before,
		TokensAfter:       after,
		MessagesBefore:    beforeCount,
		MessagesAfter:     len(a.plannedRequest().msgs),
	}, nil
}

// ErrNothingToFree is a compaction declined because it would not make the
// next request any smaller. Not a failure: the session is simply not one that
// has anything to give up, and spending a model call to prove it is worse
// than saying so.
var ErrNothingToFree = errors.New("there is nothing to free: the session is already smaller than the recent messages a compaction would keep anyway")

// CompactionWorthwhile reports whether compacting now would leave a smaller
// request than the one already assembled, so a caller can say what it is
// about to do before it starts doing it.
func (a *Agent) CompactionWorthwhile(keepRecentTokens int) bool {
	if len(a.messages) < 2 {
		return false
	}
	if keepRecentTokens <= 0 {
		keepRecentTokens = a.keepRecentTokens()
	}
	return a.worthSummarising(a.startPoint(keepRecentTokens), a.SummaryCap())
}

// worthSummarising reports whether cutting here, and paying for a checkpoint of
// at most summaryTokens, would leave a smaller request than the one now.
func (a *Agent) worthSummarising(cut, summaryTokens int) bool {
	if cut <= 0 || cut > len(a.messages) {
		return false
	}
	kept := a.estimate(a.fixedChars() + charsOf(a.messages[cut:]))
	return kept+summaryTokens < a.plannedTokens()
}

// setSummary installs a checkpoint and the boundary it stands in for.
//
// The one place the sent prefix moves, which is why it is the one place that
// has to invalidate ground truth -- and it does so by position, so a response
// arriving afterwards re-anchors the session simply by being newer.
func (a *Agent) setSummary(summary string, cut int) {
	a.summary = summary
	a.summarisedThrough = cut
	a.startAt = cut
	a.warned = false
	a.requestChanged()
	a.publish()
}

// checkpointThrough is how far into the transcript a checkpoint written now
// would reach.
//
// Costed against a checkpoint as long as it is allowed to be, because the one
// being written does not exist yet. A checkpoint that stops short of what the
// next request has to leave out is stale on arrival, and the session pays for
// a second model call to find that out.
func (a *Agent) checkpointThrough(maxTokens int) int {
	through := a.startPoint(a.keepRecentTokens())
	if cut := a.firstKept(a.promptBudget(), maxTokens); cut > through {
		through = cut
	}
	return through
}

// CheckpointWorthwhile reports whether a checkpoint written now would cover
// anything the one already in hand does not.
//
// No separate threshold for how much growth counts as material: the cut moves
// in token-sized steps, so it only advances once a full tail's worth of new
// conversation has pushed it along.
func (a *Agent) CheckpointWorthwhile(maxTokens int) bool {
	if len(a.messages) < 2 {
		return false
	}
	return a.checkpointThrough(maxTokens) > a.summarisedThrough
}

// Summarise records a checkpoint of the conversation without changing it.
//
// The summary is a spare, not a replacement. Nothing is sent differently until
// a window turns out to be too narrow for the real messages, and a swap back
// to a roomier model still has every one of them. That is what makes it
// reasonable to write a summary the session may never need -- which is the
// whole point of doing it while the user is idle rather than while they wait.
//
// maxTokens caps the summary: see SummaryMaxTokens and
// SpeculativeSummaryMaxTokens.
func (a *Agent) Summarise(ctx context.Context, maxTokens int) (*CompactResult, error) {
	if len(a.messages) < 2 {
		return nil, errors.New("there is nothing to summarise yet")
	}

	through := a.checkpointThrough(maxTokens)
	summary, err := a.summariseThrough(ctx, through, maxTokens)
	if err != nil {
		return nil, err
	}

	a.summary = summary
	a.summarisedThrough = through
	a.warned = false

	// Nothing moved: the transcript is exactly as it was and the checkpoint
	// sits beside it. Reporting the assembled request as "messages after"
	// made a non-destructive checkpoint read as the destruction of every
	// message it did not include.
	session := a.TranscriptTokens()
	return &CompactResult{
		Summary:           summary,
		SummarisedThrough: through,
		Rewrote:           false,
		TokensBefore:      session,
		TokensAfter:       session,
		MessagesBefore:    len(a.messages),
		MessagesAfter:     len(a.messages),
	}, nil
}

// summariseThrough summarises messages[:through], folding in the checkpoint
// already in hand.
func (a *Agent) summariseThrough(ctx context.Context, through, maxTokens int) (string, error) {
	end := min(through, len(a.messages))
	covered := a.messages[:end]

	// Only the conversation the checkpoint does not already account for.
	//
	// updateSummarisationPrompt has always opened "The transcript above is
	// NEW conversation to fold into the existing checkpoint", and until now
	// that was simply false: every compaction re-serialised the session from
	// message zero. The cost of compacting therefore grew with the session --
	// measured at 30k, 61k and 92k tokens for the first three compactions of
	// one run -- so the mechanism that keeps a long session alive got more
	// expensive exactly as the session got longer, and eventually could not
	// run at all. It also handed the model its own summary's source material
	// a second time under an instruction saying it was new.
	start := 0
	summary := a.summary
	if strings.TrimSpace(summary) != "" {
		start = min(a.summarisedThrough, end)
	}

	fresh := a.messages[start:end]
	if strings.TrimSpace(serialiseConversation(fresh)) == "" {
		if strings.TrimSpace(summary) != "" {
			// Nothing has happened since the checkpoint was written, so it is
			// already the answer. Saying so beats spending a model call to be
			// told the same thing back.
			return summary, nil
		}
		return "", errors.New("there is nothing to compact yet")
	}

	// Folded in pieces when there is more new conversation than one request
	// can carry. Normally there is exactly one piece: the chunk is as large
	// as the window allows, so an ordinary compaction still costs one call.
	for _, chunk := range a.summarisationChunks(fresh, maxTokens) {
		next, err := a.summarise(ctx, serialiseConversation(chunk), summary, maxTokens)
		if err != nil {
			return "", err
		}
		summary = next
	}

	// The file lists stay over everything the checkpoint covers, not just the
	// new part: it is a local scan with no model call, and narrowing it would
	// quietly drop every file touched before the last checkpoint.
	return summary + collectFileOperations(covered), nil
}

// minSummarisationChunk is the least new conversation worth one summarisation
// request. Below it the instructions outweigh the material and folding would
// take more calls than the content justifies.
const minSummarisationChunk = 1024

// summarisationChunks splits new conversation into pieces that each fit in a
// request, alongside the checkpoint they are being folded into.
//
// The summarisation call is the one request nothing else assembles or bounds,
// so without this a single compaction covering more than a window's worth of
// new material fails -- and it fails at the moment the session has no other
// way to make room.
func (a *Agent) summarisationChunks(msgs []provider.Message, maxTokens int) [][]provider.Message {
	// The whole usable window, not the prompt budget: the prompt budget holds
	// a quarter back so a conversation request leaves room to reply in, and
	// this request's reply is already subtracted as maxTokens. Charging it
	// twice would chop the material into more pieces than the window needs,
	// and every extra piece is another model call.
	budget := a.Usable() - maxTokens -
		a.estimate(len(a.summary)+len(updateSummarisationPrompt)+len(summarisationSystemPrompt))
	if a.Usable() <= 0 {
		// No window to fit anything to; one piece and let the server decide.
		return [][]provider.Message{msgs}
	}
	if budget < minSummarisationChunk {
		budget = minSummarisationChunk
	}
	limit := int(float64(budget) * a.charsPerTokenNow())

	var out [][]provider.Message
	start, size := 0, 0
	for i := range msgs {
		n := len(serialiseConversation(msgs[i : i+1]))
		if size > 0 && size+n > limit {
			out = append(out, msgs[start:i])
			start, size = i, 0
		}
		size += n
	}
	return append(out, msgs[start:])
}

func (a *Agent) summarise(ctx context.Context, transcript, previous string, maxTokens int) (string, error) {
	prompt := "<transcript>\n" + transcript + "\n</transcript>\n\n"
	instructions := summarisationPrompt
	if strings.TrimSpace(previous) != "" {
		// A second compaction summarises only what has happened since the
		// first, so without this the checkpoint would forget the beginning of
		// the session every time it ran -- which on a long session is exactly
		// when the beginning matters most.
		prompt += "<previous-summary>\n" + previous + "\n</previous-summary>\n\n"
		instructions = updateSummarisationPrompt
	}

	req := provider.Request{
		Model: a.model,
		Messages: []provider.Message{
			{Role: provider.RoleSystem, Content: summarisationSystemPrompt},
			{Role: provider.RoleUser, Content: prompt + instructions},
		},
		// An explicit cap rather than the turn budget: a checkpoint that grows
		// without bound defeats the purpose of making one, and this call runs
		// inside the reserve, which is sized to hold it.
		MaxTokens: maxTokens,
		// Summarising is mechanical. At 1-2 tokens/sec on a local model, letting
		// a max-effort model reason about it first costs minutes of wall clock
		// for a checkpoint that reads no better, and it is charged at exactly
		// the moment the session is already stuck waiting for room.
		Effort: provider.EffortNone,
	}

	stream, err := a.client.Stream(ctx, req)
	if err != nil {
		return "", fmt.Errorf("summarising the session: %w", err)
	}
	defer stream.Close()

	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", fmt.Errorf("summarising the session: %w", err)
		}
	}

	// A summary cut off by the token cap is partial by definition. Persisting
	// it would silently make the checkpoint wrong, so refuse instead. (This
	// check mirrors pi's getSummarizationFailure, and for the same reason.)
	if stream.StopReason() == provider.StopLength {
		return "", errors.New("the summary hit the output token cap and is incomplete, so it was discarded. " +
			"The session is unchanged. Try /compact again, or start a new session")
	}

	summary := strings.TrimSpace(stream.Message().Content)
	if summary == "" {
		return "", errors.New("the model returned an empty summary; the session is unchanged")
	}
	return summary, nil
}

// serialiseConversation flattens messages into a transcript.
//
// Handing the model a transcript rather than a message list is deliberate: as a
// message list it reads as a conversation to continue, and the model answers
// the last question instead of summarising. As tagged text it reads as data.
func serialiseConversation(messages []provider.Message) string {
	const toolResultMax = 2000

	var parts []string
	for _, m := range messages {
		switch m.Role {
		case provider.RoleUser:
			if s := strings.TrimSpace(m.Content); s != "" {
				parts = append(parts, "[User]: "+s)
			}
		case provider.RoleAssistant:
			if s := strings.TrimSpace(m.Content); s != "" {
				parts = append(parts, "[Assistant]: "+s)
			}
			if len(m.ToolCalls) > 0 {
				var calls []string
				for _, tc := range m.ToolCalls {
					calls = append(calls, fmt.Sprintf("%s(%s)", tc.Name, compactArgs(tc.Args)))
				}
				parts = append(parts, "[Assistant tool calls]: "+strings.Join(calls, "; "))
			}
		case provider.RoleTool:
			s := strings.TrimSpace(m.Content)
			if s == "" {
				continue
			}
			if len(s) > toolResultMax {
				s = s[:toolResultMax] + fmt.Sprintf("\n[... %d more characters truncated]", len(s)-toolResultMax)
			}
			prefix := "[Tool result]: "
			if m.IsError {
				prefix = "[Tool error]: "
			}
			parts = append(parts, prefix+s)
		}
	}
	return strings.Join(parts, "\n\n")
}

func compactArgs(raw string) string {
	var m map[string]any
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		if len(raw) > 200 {
			return raw[:200] + "..."
		}
		return raw
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, k := range keys {
		v := fmt.Sprint(m[k])
		if len(v) > 120 {
			v = v[:120] + "..."
		}
		parts = append(parts, fmt.Sprintf("%s=%q", k, v))
	}
	return strings.Join(parts, ", ")
}

// collectFileOperations records which files the session read and which it
// changed, so that survives compaction even when the prose summary omits it.
func collectFileOperations(messages []provider.Message) string {
	read := map[string]bool{}
	modified := map[string]bool{}

	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			var args struct {
				Path string `json:"path"`
			}
			if json.Unmarshal([]byte(tc.Args), &args) != nil || args.Path == "" {
				continue
			}
			switch tc.Name {
			case "read":
				read[args.Path] = true
			case "write", "edit":
				modified[args.Path] = true
			}
		}
	}
	for p := range modified {
		delete(read, p)
	}

	var b strings.Builder
	if len(read) > 0 {
		fmt.Fprintf(&b, "\n\n<read-files>\n%s\n</read-files>", strings.Join(sortedKeys(read), "\n"))
	}
	if len(modified) > 0 {
		fmt.Fprintf(&b, "\n\n<modified-files>\n%s\n</modified-files>", strings.Join(sortedKeys(modified), "\n"))
	}
	return b.String()
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func firstUserMessage(messages []provider.Message) string {
	for _, m := range messages {
		if m.Role == provider.RoleUser && strings.TrimSpace(m.Content) != "" {
			return m.Content
		}
	}
	return ""
}

// dropOrphanToolResults removes tool results whose originating call did not
// survive, which would otherwise make the very next request invalid.
func dropOrphanToolResults(messages []provider.Message) []provider.Message {
	valid := map[string]bool{}
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			valid[tc.ID] = true
		}
	}
	out := make([]provider.Message, 0, len(messages))
	for _, m := range messages {
		if m.Role == provider.RoleTool && !valid[m.ToolCallID] {
			continue
		}
		out = append(out, m)
	}
	return out
}

// FilesTouched reports the files a conversation has read or modified.
//
// Used by the cloud-switch confirmation: before a session is sent to an
// off-site provider, the user should be able to see which of their files are
// actually in that context, not just how many tokens it is.
func FilesTouched(messages []provider.Message) (read, modified []string) {
	r := map[string]bool{}
	m := map[string]bool{}
	for _, msg := range messages {
		for _, tc := range msg.ToolCalls {
			var args struct {
				Path string `json:"path"`
			}
			if json.Unmarshal([]byte(tc.Args), &args) != nil || args.Path == "" {
				continue
			}
			switch tc.Name {
			case "read":
				r[args.Path] = true
			case "write", "edit":
				m[args.Path] = true
			}
		}
	}
	for p := range m {
		delete(r, p)
	}
	return sortedKeys(r), sortedKeys(m)
}
