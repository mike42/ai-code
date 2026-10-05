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
// The checkpoint format is adapted from pi (github.com/badlogic/pi, MIT,
// (c) 2025 Mario Zechner), packages/coding-agent/src/core/compaction: a fixed
// section format, and file lists carried across the boundary.
//
// The model writes the checkpoint from the request it is already working
// from, with one instruction appended. Every token before the instruction is
// the prefix the server processed on the previous turn, so on a server that
// keeps it cached the summary costs only its own output. Any other prompt
// shape -- another system prompt, a flattened transcript -- has to be read
// from scratch, which on a large session and slow hardware takes many minutes.

const checkpointInstruction = `Stop work on the task for a moment. Everything above is about to be replaced by a context checkpoint, and you are the one writing it. Another model will continue this work from the checkpoint with none of the original messages available.

Do NOT call any tools. Do NOT continue the task or answer anything above. Output only the checkpoint.

If the conversation above begins with an earlier <context-checkpoint>, carry forward everything in it that is still true, and fold the newer conversation into it.

Use this EXACT format:

## Goal
[What the user is trying to accomplish, in their terms. Multiple items if the session covered several tasks.]

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

// SummaryMaxTokens bounds an on-demand checkpoint; the reserve is sized to hold it.
const SummaryMaxTokens = 4096

// SpeculativeSummaryMaxTokens bounds a checkpoint nobody asked for: written
// while the session is idle so a later model swap is cheap, and far smaller
// than SummaryMaxTokens because on a local model the cap is wall-clock cost.
const SpeculativeSummaryMaxTokens = 800

// startPoint finds the cut that keeps keepRecentTokens of whole turns
// verbatim, measured with the calibrated estimator.
func (a *Agent) startPoint(keepRecentTokens int) int {
	msgs := a.messages
	if keepRecentTokens <= 0 || len(msgs) == 0 {
		return len(msgs)
	}

	starts := turnStarts(msgs)
	if len(starts) == 0 {
		// No user message anywhere: nothing divides this into turns, so split
		// it as one.
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
		// The newest turn alone is larger than the tail budget. Keep as much of
		// its end as fits: the model is in the middle of that turn.
		last := starts[len(starts)-1]
		cut = a.startInsideTurn(last, len(msgs), keepRecentTokens)
	}

	// Keep the oldest turn on the summary side so the call has material.
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
func (a *Agent) startInsideTurn(start, end, budget int) int {
	for i := start + 1; i < end; i++ {
		if a.estimateMessages(a.messages[i:end]) <= budget {
			return legalStart(a.messages, i)
		}
	}
	// Keep the last message even when it does not fit; the guard's escalation
	// can name the single message too large for the window, which this cannot.
	return legalStart(a.messages, end-1)
}

// legalStart walks forwards off any tool result, onto the next message that can
// begin a conversation; a stranded result is the message-shape violation
// Validate exists to catch.
func legalStart(msgs []provider.Message, cut int) int {
	for cut < len(msgs) && msgs[cut].Role == provider.RoleTool {
		cut++
	}
	return cut
}

// turnStarts indexes every message that begins a turn: a user message, which
// steering also appends mid-exchange, making it a legal cut point too.
func turnStarts(msgs []provider.Message) []int {
	var out []int
	for i, m := range msgs {
		if m.Role == provider.RoleUser {
			out = append(out, i)
		}
	}
	return out
}

// MessagesFitting decides at the moment of sending, because the window belongs
// to the model loaded: the whole transcript when it fits, otherwise the
// summary followed by as much recent conversation as is left.
func (a *Agent) MessagesFitting(window int) []provider.Message {
	if window <= 0 || a.estimate(a.transcriptChars()) <= window {
		return a.messages
	}
	return a.assemble(window, a.resumeSummary())
}

// resumeSummary is the checkpoint assembly may use, none when resumeFromTranscript is set.
func (a *Agent) resumeSummary() string {
	if a.resumeFromTranscript {
		return ""
	}
	return a.summary
}

// assemble builds a request of at most window tokens from a checkpoint and the tail.
func (a *Agent) assemble(window int, summary string) []provider.Message {
	head := a.summaryHead(summary)
	first := a.firstKept(window, a.estimate(charsOf(head)))

	out := make([]provider.Message, 0, len(head)+len(a.messages)-first)
	out = append(out, head...)
	out = append(out, a.messages[first:]...)
	// The cut can land between a tool call and its result; drop stranded
	// results, which the provider rejects.
	return dropOrphanToolResults(out)
}

// firstKept is the earliest transcript message a request of this size has room
// for, once a checkpoint of headTokens sits in front. It stops at
// summarisedThrough, which the checkpoint already covers.
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
		// Every message is covered by the checkpoint, so the checkpoint alone
		// is a complete request; walking back onto the last turn would re-send
		// the content the checkpoint replaced.
		return first
	}

	if first == len(a.messages) && len(a.messages) > 0 {
		// A window too small for even the last message still has to produce a
		// request; send the last turn over budget rather than nothing.
		first = len(a.messages) - 1
		for first > 0 && a.messages[first].Role == provider.RoleTool {
			first--
		}
	}
	return first
}

// summaryHead is the checkpoint as it goes on the wire: the original request
// verbatim, then the summary. A summary is lossy, so what the session was
// asked for stays intact.
func (a *Agent) summaryHead(summary string) []provider.Message {
	if strings.TrimSpace(summary) == "" {
		return nil
	}
	var out []provider.Message
	if first := firstUserMessage(a.messages); first != "" && a.estimate(len(first)) <= a.preservedRequestBudget() {
		if !strings.HasPrefix(first, preservedRequestPreamble) {
			// A second copy of the preamble reads as a stutter.
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
// checkpoint. It is the anchor the session works towards, but an opening
// request can be a pasted file, so past this share the summary speaks for it.
func (a *Agent) preservedRequestBudget() int {
	return a.promptBudget() / 4
}

// promptBudget is how much of the window an assembled request may fill.
// outputBudget hands the model what is left, so a request assembled to the
// line would cap every answer for the rest of the session.
func (a *Agent) promptBudget() int {
	return a.promptBudgetIn(a.opts.ContextLimit)
}

func (a *Agent) promptBudgetIn(limit int) int {
	return a.usableIn(limit) * 3 / 4
}

// request is what the next call will carry. exact means the list is precisely
// the checkpoint head followed by a.messages[first:], which is what lets an
// earlier prefill still describe this request's prefix.
type request struct {
	msgs  []provider.Message
	first int
	exact bool
}

// plannedRequest is the request the cut describes: the checkpoint head followed
// by everything from the cut, with nothing dropped to make it fit. This is what
// the guard measures; a request trimmed to fit always fits.
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

	// A cut never lands on a tool result, so the repair fires only on a broken
	// transcript.
	repaired := dropOrphanToolResults(msgs)
	exact := len(repaired) == len(msgs)
	msgs = repaired

	return request{msgs: msgs, first: first, exact: exact}
}

// request is what will actually go on the wire. It differs from the planned
// one only when the window shrank under a running session or
// resumeFromTranscript is set; then the oldest messages are dropped to fit.
func (a *Agent) request() request {
	r := a.plannedRequest()
	usable := a.Usable()
	if usable <= 0 {
		return r
	}
	// The same threshold the guard triggers on: held to the plain budget, the
	// two disagree by exactly the output floor.
	if a.estimate(a.fixedChars()+charsOf(r.msgs)) <= usable-minOutputTokens {
		return r
	}
	return request{msgs: a.MessagesFitting(a.promptBudget()), exact: false}
}

// messagesToSend is what the next request carries.
func (a *Agent) messagesToSend() []provider.Message { return a.request().msgs }

// summaryCovers reports whether the checkpoint in hand already accounts for
// everything a request of this size would have to leave out.
func (a *Agent) summaryCovers(window int) bool {
	head := a.summaryHead(a.summary)
	if len(head) == 0 || a.summarisedThrough <= 0 {
		return false
	}
	rest := a.messages[min(a.summarisedThrough, len(a.messages)):]
	return a.estimate(a.fixedChars()+charsOf(head)+charsOf(rest)) <= window
}

// ResumePlan is what one way of continuing in a narrower window would carry.
type ResumePlan struct {
	// Available is false for the checkpoint plan when no checkpoint exists;
	// offering it would mean summarising while the swap waits.
	Available bool
	// Prompt is what the request would occupy; Free is what is left to answer in.
	Prompt, Free int
	// Kept and Dropped split the transcript where the window forces it.
	Kept, Dropped int
	// Unrepresented is how many dropped messages nothing speaks for: all of
	// them without a checkpoint, and those written after it with one.
	Unrepresented int
}

// ResumeChoice is what a window of a given size leaves a session able to do.
type ResumeChoice struct {
	// Fits is true when the whole transcript still goes on the wire with room
	// to answer, so there is nothing to decide.
	Fits bool
	// Session is the whole transcript's cost, whether or not it would be sent,
	// and Usable is what the window offers once the reserve is held back.
	Session, Usable        int
	Checkpoint, Transcript ResumePlan
}

// PlanResume costs each way of continuing in a window of contextLimit tokens.
// Arithmetic only: a swap must not stall on a summarisation call.
func (a *Agent) PlanResume(contextLimit int) ResumeChoice {
	c := ResumeChoice{Session: a.TranscriptTokens(), Usable: a.usableIn(contextLimit)}

	usable := c.Usable
	// The same threshold the guard and assembly use, so "fits" means it really fits.
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

// CompactResult reports what a compaction did. Every figure measures the
// session: the whole transcript, before and after, not the request and not
// the window.
type CompactResult struct {
	Summary string
	// SummarisedThrough is how many messages from the start the summary
	// accounts for, and -- when Rewrote is true -- the boundary the next
	// request starts from.
	SummarisedThrough int
	// Rewrote distinguishes the two checkpoint writers: Summarise changes
	// nothing about what is sent, Compact also moves the boundary. Neither
	// removes a message.
	Rewrote bool
	// TokensBefore and TokensAfter are projections: the request's cost before and after.
	TokensBefore int
	TokensAfter  int
	// MessagesBefore and MessagesAfter count what the request carries.
	MessagesBefore int
	MessagesAfter  int
	// Cleared is how many tool results were emptied before any model call;
	// ClearedTokens is what that alone recovered.
	Cleared       int
	ClearedTokens int
}

// Compact makes room by summarising the older part of the session. Nothing is
// deleted: the summary and the boundary are recorded, and assembly starts from
// the boundary. A keepRecentTokens of zero means the size derived from the window.
func (a *Agent) Compact(ctx context.Context, keepRecentTokens int) (*CompactResult, error) {
	if len(a.messages) < 2 {
		return nil, errors.New("there is nothing to compact yet")
	}
	if keepRecentTokens <= 0 {
		keepRecentTokens = a.keepRecentTokens()
	}

	// The intended request on both sides: the one that would be sent has
	// already been trimmed to fit.
	before := a.plannedTokens()
	beforeCount := len(a.plannedRequest().msgs)
	cut := a.startPoint(keepRecentTokens)

	// A checkpoint is not free space: it goes in front of everything the cut
	// kept, so a summary larger than what it replaces makes the request bigger.
	// Costed against the biggest allowed, since the real one does not exist yet.
	if !a.worthSummarising(cut, a.SummaryCap()) {
		return nil, ErrNothingToFree
	}

	summary, err := a.summariseInPlace(ctx, a.SummaryCap())
	if err != nil {
		return nil, err
	}

	// Measure against the checkpoint that got written: nothing is installed
	// unless it leaves the next request smaller.
	prevStart := a.startAt
	a.setSummary(summary, cut)
	after := a.plannedTokens()
	if after >= before {
		// Keep the summary as a standby, but take back the boundary: the
		// request without it was already smaller.
		a.startAt = prevStart
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

// ErrNothingToFree reports a compaction declined because it would not make the
// next request smaller. Not a failure: nothing needed freeing.
var ErrNothingToFree = errors.New("there is nothing to free: the session is already smaller than the recent messages a compaction would keep anyway")

// CompactionWorthwhile reports whether compacting now would leave a smaller request.
func (a *Agent) CompactionWorthwhile(keepRecentTokens int) bool {
	if len(a.messages) < 2 {
		return false
	}
	if keepRecentTokens <= 0 {
		keepRecentTokens = a.keepRecentTokens()
	}
	return a.worthSummarising(a.startPoint(keepRecentTokens), a.SummaryCap())
}

// worthSummarising reports whether the cut and its checkpoint leave a smaller request.
func (a *Agent) worthSummarising(cut, summaryTokens int) bool {
	if cut <= 0 || cut > len(a.messages) {
		return false
	}
	kept := a.estimate(a.fixedChars() + charsOf(a.messages[cut:]))
	return kept+summaryTokens < a.plannedTokens()
}

// setSummary installs a checkpoint and the boundary it stands in for. It is
// the one place the sent prefix moves, so it invalidates ground truth.
func (a *Agent) setSummary(summary string, cut int) {
	a.summary = summary
	a.summaryWrittenAt = len(a.messages)
	a.summarisedThrough = cut
	a.startAt = cut
	a.requestChanged()
	a.publish()
}

// checkpointThrough is how far into the transcript a checkpoint written now
// would reach, costed against one as long as it is allowed to be.
func (a *Agent) checkpointThrough(maxTokens int) int {
	through := a.startPoint(a.keepRecentTokens())
	if cut := a.firstKept(a.promptBudget(), maxTokens); cut > through {
		through = cut
	}
	return through
}

// CheckpointWorthwhile reports whether a checkpoint written now would cover
// anything the one in hand does not.
func (a *Agent) CheckpointWorthwhile(maxTokens int) bool {
	if len(a.messages) < 2 {
		return false
	}
	return a.checkpointThrough(maxTokens) > a.summarisedThrough
}

// Summarise records a checkpoint of the conversation without changing it. The
// summary is a spare: nothing is sent differently until a window is too narrow
// for the real messages. maxTokens caps the summary.
func (a *Agent) Summarise(ctx context.Context, maxTokens int) (*CompactResult, error) {
	if len(a.messages) < 2 {
		return nil, errors.New("there is nothing to summarise yet")
	}

	through := a.checkpointThrough(maxTokens)
	summary, err := a.summariseInPlace(ctx, maxTokens)
	if err != nil {
		return nil, err
	}

	a.summary = summary
	a.summarisedThrough = through
	a.summaryWrittenAt = len(a.messages)

	// Nothing moved: the transcript is exactly as it was and the figure is
	// equal on both sides.
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

// summariseInPlace asks the model for a checkpoint of everything it can see,
// sending the next request exactly as it would go out with the instruction
// appended. The recorded boundary is the caller's; the text may describe more
// than that, which costs nothing.
func (a *Agent) summariseInPlace(ctx context.Context, maxTokens int) (string, error) {
	if strings.TrimSpace(a.summary) != "" && a.summaryWrittenAt >= len(a.messages) {
		// Nothing has happened since the checkpoint was written.
		return a.summary, nil
	}

	// The planned request is the one the server last processed, plus what has
	// been appended since. Only when the summary would not fit beside it does
	// the trimmed request stand in, and that one is not cached.
	sending := a.plannedRequest().msgs
	if a.Usable() > 0 && !a.summaryCallFits() {
		sending = a.messagesToSend()
	}
	messages := a.withSystem(sending)
	if Validate(messages) != nil {
		// streamTurn would send the repaired list, so that is the cached one.
		messages, _ = Repair(messages)
	}
	messages = append(messages[:len(messages):len(messages)],
		provider.Message{Role: provider.RoleUser, Content: checkpointInstruction})

	req := provider.Request{
		Model:    a.model,
		Messages: messages,
		// The tool definitions are part of the rendered prompt; leaving them
		// out would change its first tokens and lose the cache.
		Tools:       toolDefs(a.exec),
		Temperature: a.opts.Temperature,
		TopP:        a.opts.TopP,
		// A cap rather than the turn budget: a checkpoint that grows without
		// bound defeats the purpose of making one.
		MaxTokens: maxTokens,
		// Thinking would spend the cap before the checkpoint is written.
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

	msg := stream.Message()
	if len(msg.ToolCalls) > 0 {
		return "", errors.New("the model called a tool instead of writing the summary; the session is unchanged")
	}
	// A summary cut off by the cap is partial; mirrors pi's getSummarizationFailure.
	if stream.StopReason() == provider.StopLength {
		return "", errors.New("the summary hit the output token cap and is incomplete, so it was discarded. " +
			"The session is unchanged")
	}
	summary := strings.TrimSpace(msg.Content)
	if summary == "" {
		return "", errors.New("the model returned an empty summary; the session is unchanged")
	}
	return summary + collectFileOperations(a.messages), nil
}

// summaryCallTokens is what asking for a checkpoint adds to the request: the
// instruction and the longest checkpoint allowed.
func (a *Agent) summaryCallTokens() int {
	return a.estimate(len(checkpointInstruction)) + a.SummaryCap()
}

// collectFileOperations records which files the session read and changed, for the checkpoint.
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

// dropOrphanToolResults removes tool results whose originating call did not survive.
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

// FilesTouched reports the files a conversation has read or modified, for the
// cloud-switch confirmation.
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
