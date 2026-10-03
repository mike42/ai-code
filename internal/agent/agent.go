package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

type Options struct {
	// MaxIterations caps the turns one Run may take; 0 is unlimited.
	MaxIterations int
	// MaxTokens is a ceiling on one response, below the window-derived cap; 0 means none.
	MaxTokens int
	// MaxOutputTokens is the backend's limit on a single completion, 0 when unknown.
	// It only ever lowers the cap, never raises it.
	MaxOutputTokens int
	Temperature     *float64
	TopP            *float64
	ParallelReads   bool
	LoopGuard       int
	// ContextLimit is the effective window in tokens, 0 when unknown.
	ContextLimit int
	// WarnPercent is the occupancy at which the agent emits a context warning.
	WarnPercent int

	// AutoCompact summarises the session when it reaches the reserve instead of stopping.
	AutoCompact bool
	// ReserveTokens is the room kept free at the top of the window: one response
	// plus the summarisation call compaction makes.
	ReserveTokens int
	// KeepRecentTokens is how much of the tail survives compaction verbatim.
	KeepRecentTokens int

	// Effort is the thinking level to request. EffortUnset sends nothing.
	Effort provider.Effort
}

type Agent struct {
	// unavailable is set by a tool call that found nowhere to run, and ends
	// the run once that round of results is in. Calls run concurrently.
	unavailableMu sync.Mutex
	unavailable   *tool.Unavailable

	client   provider.Client
	model    string
	exec     tool.Executor
	sink     Sink
	opts     Options
	system   string
	messages []provider.Message

	// charsSent is the size of the request in flight, so the response can be divided by it.
	charsSent int
	// charsPerToken is the measured ratio; defaultCharsPerToken until a response is seen.
	charsPerToken float64
	// requestChangedAt is the transcript length when the request prefix last moved.
	requestChangedAt int
	// toolChars caches the size of the tool definitions, constant per session.
	toolChars int
	warned    bool

	readOnly func(name string) bool

	// steer holds messages typed while the turn was running; they are appended
	// only at turn boundaries, because a user message may not come between an
	// assistant message that requested tools and those tools' results.
	steerMu sync.Mutex
	steerQ  []string

	// summary is a standby checkpoint kept beside the transcript, not in place
	// of it; assembly reaches for it only when the messages do not fit. Each
	// new checkpoint folds in the previous one.
	summary string
	// startAt is where the next request begins: earlier messages are not sent,
	// the summary stands in for them. Zero means everything is sent.
	startAt int
	// summarisedThrough is how many messages from the start of the transcript
	// the summary accounts for.
	summarisedThrough int

	// resumeFromTranscript carries recent messages verbatim and lets older ones
	// fall off, rather than spend a small window on a checkpoint.
	resumeFromTranscript bool

	// stuck watches for a session going in circles without repeating a call exactly.
	stuck stuckWatch

	// atBoundary is consulted between iterations, where the message list is complete.
	atBoundary func(context.Context) bool
}

func New(client provider.Client, model string, exec tool.Executor, sink Sink, opts Options) *Agent {
	if opts.LoopGuard <= 0 {
		opts.LoopGuard = 4
	}
	if opts.WarnPercent <= 0 {
		opts.WarnPercent = 80
	}
	a := &Agent{client: client, model: model, exec: exec, sink: sink, opts: opts}

	// Concurrency is only safe for tools that declare themselves read-only.
	if ro, ok := exec.(interface{ IsReadOnly(string) bool }); ok {
		a.readOnly = ro.IsReadOnly
	} else {
		a.readOnly = func(string) bool { return false }
	}
	// An opening figure, so a consumer has something before the first turn.
	a.publish()
	return a
}

// Window is the context length of the model now loaded, 0 when unknown.
func (a *Agent) Window() int { return a.opts.ContextLimit }

func (a *Agent) SetSystem(prompt string) {
	a.system = prompt
	// Part of every request, so changing it invalidates recorded prefills.
	a.requestChanged()
	a.publish()
}

// System returns the assembled system prompt.
func (a *Agent) System() string { return a.system }

// budget is the sizing derived from the window the agent is on now. Read
// rather than stored, so a model swap re-derives it.
func (a *Agent) budget() Budget { return BudgetFor(a.opts.ContextLimit) }

// reserveTokens is the derived reserve unless a configured value overrides it.
func (a *Agent) reserveTokens() int {
	if a.opts.ReserveTokens > 0 {
		return a.opts.ReserveTokens
	}
	return a.budget().Reserve
}

// KeepRecentTokens is how much of the tail a compaction keeps verbatim.
func (a *Agent) KeepRecentTokens() int { return a.keepRecentTokens() }

func (a *Agent) keepRecentTokens() int {
	if a.opts.KeepRecentTokens > 0 {
		return a.opts.KeepRecentTokens
	}
	return a.budget().KeepRecent
}

// SummaryCap is the most a checkpoint of this session may run to. Derived
// from the reserve in force, because the checkpoint must fit in it.
func (a *Agent) SummaryCap() int {
	return clamp(a.reserveTokens()/2, minSummaryTokens, SummaryMaxTokens)
}

// contextChars sizes the request ai-code would send now: system prompt, tool
// definitions, and the messages that would go with them.
func (a *Agent) contextChars() int {
	return a.fixedChars() + charsOf(a.messagesToSend())
}

// fixedChars is what every request carries whatever the conversation is.
func (a *Agent) fixedChars() int {
	return len(a.system) + messageOverheadChars + a.toolDefChars()
}

func charsOf(msgs []provider.Message) int {
	n := 0
	for _, m := range msgs {
		n += messageChars(m)
	}
	return n
}

// transcriptChars is the size of the whole conversation, sent or not.
func (a *Agent) transcriptChars() int { return a.fixedChars() + charsOf(a.messages) }

func (a *Agent) toolDefChars() int {
	if a.toolChars == 0 && a.exec != nil {
		for _, d := range a.exec.Definitions() {
			a.toolChars += len(d.Name) + len(d.Description) + len(d.Schema) + messageOverheadChars
		}
	}
	return a.toolChars
}

func (a *Agent) Messages() []provider.Message { return a.messages }

// SetMessages replaces the conversation and clears any checkpoint that
// described the one it replaces.
func (a *Agent) SetMessages(m []provider.Message) {
	a.messages = m
	a.summary = ""
	a.summarisedThrough = 0
	a.startAt = 0
	a.resumeFromTranscript = false
	a.requestChanged()
	a.publish()
}

// Summary returns the standby checkpoint and how many messages from the start
// of the transcript it accounts for.
func (a *Agent) Summary() (string, int) { return a.summary, a.summarisedThrough }

// SetSummary restores a checkpoint from a saved session. Call it after
// SetMessages, which clears one.
func (a *Agent) SetSummary(summary string, summarisedThrough int) {
	a.SetCheckpoint(summary, summarisedThrough, 0)
}

// SetCheckpoint restores a checkpoint and the boundary a compaction chose. A
// cut of zero restores a standby checkpoint; a non-zero one restores a
// compaction, so a resumed session sends the same prefix the original did.
func (a *Agent) SetCheckpoint(summary string, summarisedThrough, cut int) {
	clampIndex := func(n int) int { return clamp(n, 0, len(a.messages)) }
	a.summary = summary
	a.summarisedThrough = clampIndex(summarisedThrough)
	a.startAt = clampIndex(cut)
	// Whatever was recorded against these messages was recorded under a
	// different system prompt and tool set.
	a.requestChanged()
	a.publish()
}

func (a *Agent) Model() string { return a.model }

// SetTurnBoundary installs a check that runs between loop iterations, after
// the tool results and before the next request. That is the only point at
// which a stop leaves a well-formed message list behind.
func (a *Agent) SetTurnBoundary(fn func(context.Context) bool) { a.atBoundary = fn }

// SetModel points the agent at a different model. The backend's own output
// limit travels with it: it is a property of the model being swapped to.
func (a *Agent) SetModel(model string, contextLimit, maxOutputTokens int) {
	a.model = model
	a.opts.ContextLimit = contextLimit
	a.opts.MaxOutputTokens = maxOutputTokens
	a.warned = false
	// A different model tokenizes differently, so the learned ratio and every
	// recorded prefill are stale.
	a.charsPerToken = 0
	defer func() {
		a.requestChanged()
		a.publish()
	}()
	// A choice about how to survive one window must not outlive it.
	a.resumeFromTranscript = false
}

// SetResumeFromTranscript records that losing the oldest turns outright is
// preferred to spending a narrow window on a checkpoint. Call it after
// SetModel, which clears it.
func (a *Agent) SetResumeFromTranscript(v bool) { a.resumeFromTranscript = v }

// StoppedMidTurn reports whether the transcript ends part-way through an
// exchange: a request with no reply, or a tool round not read back.
func (a *Agent) StoppedMidTurn() bool {
	if len(a.messages) == 0 {
		return false
	}
	last := a.messages[len(a.messages)-1]
	switch last.Role {
	case provider.RoleUser, provider.RoleTool:
		return true
	case provider.RoleAssistant:
		return len(last.ToolCalls) > 0
	}
	return false
}

func (a *Agent) emit(e Event) {
	if a.sink != nil {
		a.sink.Emit(e)
	}
}

// AppendUser adds a user message without running a turn.
func (a *Agent) AppendUser(text string) {
	a.messages = append(a.messages, provider.Message{Role: provider.RoleUser, Content: text})
	a.publish()
}

// Steer queues a message to be folded into the running turn. Safe to call
// from another goroutine, and from outside a turn: anything queued while
// nothing runs stays queued until TakeSteering collects it.
func (a *Agent) Steer(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	a.steerMu.Lock()
	a.steerQ = append(a.steerQ, text)
	a.steerMu.Unlock()
	return true
}

// Pending reports how many steering messages are waiting.
func (a *Agent) Pending() int {
	a.steerMu.Lock()
	defer a.steerMu.Unlock()
	return len(a.steerQ)
}

// TakeSteering removes and returns everything queued.
func (a *Agent) TakeSteering() []string {
	a.steerMu.Lock()
	defer a.steerMu.Unlock()
	q := a.steerQ
	a.steerQ = nil
	return q
}

// applySteering appends anything queued and reports whether it did.
func (a *Agent) applySteering() bool {
	q := a.TakeSteering()
	for _, text := range q {
		a.AppendUser(text)
		a.emit(Event{Kind: EvSteer, Text: text})
	}
	return len(q) > 0
}

var ErrContextFull = errors.New("context window is full")

// Run performs turns until the model stops requesting tools. The context
// governs the HTTP stream, every running tool, and the loop. Cancelling it
// leaves the conversation in a sendable state.
func (a *Agent) Run(ctx context.Context, userInput string) error {
	// Anything queued before the turn began goes in ahead of the new input, in order.
	a.applySteering()
	if userInput != "" {
		a.AppendUser(userInput)
	}

	var recent []string
	// One retry for the whole run, not one per turn: a wrong window is wrong
	// for every turn.
	overflowRetried := false

	for turn := 1; a.opts.MaxIterations <= 0 || turn <= a.opts.MaxIterations; turn++ {
		if err := ctx.Err(); err != nil {
			return a.finishInterrupted()
		}
		if err := a.guardContext(ctx); err != nil {
			return err
		}

		a.emit(Event{Kind: EvTurnStart, Turn: turn})

		assistant, stop, _, reasoned, err := a.streamTurn(ctx, turn)
		if err != nil {
			if ctx.Err() != nil {
				return a.finishInterrupted()
			}
			// The backend refused for length despite the guard clearing it, so
			// the window is not what the model reported. Make room and try
			// once, not twice: the second attempt would fail the same way.
			if provider.IsContextOverflow(err) && !overflowRetried {
				overflowRetried = true
				a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: "The server rejected the request " +
					"as too long, even though it fits the window the server itself reported. " +
					"Making room and trying once more."})
				if rerr := a.makeRoom(ctx, a.RequestTokens()); rerr != nil {
					a.emit(Event{Kind: EvError, Text: rerr.Error()})
					return rerr
				}
				turn--
				continue
			}
			a.emit(Event{Kind: EvError, Text: err.Error()})
			return err
		}

		// The reported figures travel on the message itself, attached in streamTurn.
		a.messages = append(a.messages, assistant)
		a.publish()

		if len(assistant.ToolCalls) == 0 {
			// A finished turn that said nothing looks like a hung one: with no
			// output and no visible reasoning there is no reply to read.
			a.noticeEmptyTurn(assistant, reasoned)

			// Take up anything queued while the model was talking rather than
			// dropping it.
			if a.applySteering() {
				continue
			}
			a.emit(Event{Kind: EvDone, Text: string(stop), Context: ptr(a.ContextState())})
			return nil
		}

		if sig := signature(assistant.ToolCalls); sig != "" {
			recent = append(recent, sig)
			// Only the tail is ever read, and a run has no turn limit to bound this.
			if n := len(recent) - a.opts.LoopGuard; n > 0 {
				recent = recent[n:]
			}
			if looping(recent, a.opts.LoopGuard) {
				a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: fmt.Sprintf(
					"The same tool call was repeated %d times in a row; stopping to avoid a loop.",
					a.opts.LoopGuard)})
				a.appendInterruptResults(assistant.ToolCalls, nil,
					"Stopped: this identical tool call has been repeated several times without progress. "+
						"Try a different approach.")
				a.emit(Event{Kind: EvDone, Text: "loop-guard"})
				return nil
			}
		}

		results, raw := a.dispatch(ctx, assistant.ToolCalls)
		a.clampOversized(results)
		a.messages = append(a.messages, results...)
		a.publish()

		// The results are in, so the conversation is well formed, and there
		// is nowhere for the tools to run. Another request would only hand
		// the model an error it cannot do anything about.
		a.unavailableMu.Lock()
		unavailable := a.unavailable
		a.unavailable = nil
		a.unavailableMu.Unlock()
		if unavailable != nil {
			a.emit(Event{Kind: EvDone, Text: "unavailable", Context: ptr(a.ContextState())})
			return unavailable
		}

		a.stuck.observe(raw)
		if s := a.stuck.suggestion(); s != "" {
			a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: s})
		}

		if ctx.Err() != nil {
			return a.finishInterrupted()
		}

		// After the tool results, before the next request: the one point in the
		// loop where the message list is complete and a user message is legal.
		a.applySteering()

		if a.atBoundary != nil && a.atBoundary(ctx) {
			// The only place a run can stop cleanly.
			a.emit(Event{Kind: EvDone, Text: "stopped", Context: ptr(a.ContextState())})
			return nil
		}
	}

	// Reachable only when MaxIterations was set; the default is unlimited.
	a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: fmt.Sprintf(
		"Stopped after the configured limit of %d turns, without the model finishing.",
		a.opts.MaxIterations)})
	return nil
}

// streamTurn sends one request and consumes the response stream. reasoned
// reports that the model emitted thinking, which collapsed mode never shows,
// so a turn that only reasoned is distinguishable from one that did nothing.
func (a *Agent) streamTurn(ctx context.Context, turn int) (msg provider.Message, stop provider.StopReason, usageKnown, reasoned bool, err error) {
	// What goes on the wire is decided here rather than held in a.messages:
	// the window belongs to the loaded model, the transcript to the session.
	sending := a.messagesToSend()
	messages := a.withSystem(sending)
	if err := Validate(messages); err != nil {
		// Self-heal rather than dead-end: a missing tool result is recoverable
		// and refusing to proceed helps nobody.
		repaired, n := Repair(messages)
		if err2 := Validate(repaired); err2 != nil {
			return provider.Message{}, "", false, false, err
		}
		a.messages, _ = Repair(a.messages)
		messages = repaired
		a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: fmt.Sprintf(
			"Repaired %s left by an interrupted or truncated turn.", pluralise(n, "problem"))})
	}

	outCap, capFrom := a.outputBudget()
	req := provider.Request{
		Model:       a.model,
		Messages:    messages,
		Tools:       toolDefs(a.exec),
		MaxTokens:   outCap,
		Temperature: a.opts.Temperature,
		TopP:        a.opts.TopP,
		Effort:      a.opts.Effort,
	}

	// Measured before the send, so the response can be divided by it.
	a.charsSent = a.contextChars()

	stream, err := a.client.Stream(ctx, req)
	if err != nil {
		return provider.Message{}, "", false, false, err
	}
	defer stream.Close()

	for {
		ev, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// A stream that dies mid-turn leaves a partial assistant message.
			// Partial text is kept; partial tool calls are discarded by
			// finishPartial, a truncated arguments blob being worse than none.
			if ctx.Err() != nil {
				return provider.Message{}, "", false, false, ctx.Err()
			}
			return a.finishPartial(stream), provider.StopAborted, false, reasoned, nil
		}

		switch ev.Kind {
		case provider.EventText:
			a.emit(Event{Kind: EvText, Text: ev.Text, Turn: turn})
		case provider.EventReasoning:
			reasoned = true
			a.emit(Event{Kind: EvReasoning, Text: ev.Text, Turn: turn})
		case provider.EventToolCallStart:
			// Announced once, by dispatch, when the call actually starts running.
		case provider.EventUsage:
			u := ev.Usage
			a.emit(Event{Kind: EvUsage, Usage: &u, Turn: turn})
		}
	}

	usage := stream.Usage()
	msg = stream.Message()
	stop = stream.StopReason()

	// Ground truth is attached to the message the request produced, and only
	// if it can be believed: a cache-hit prefill is left unmarked rather than
	// anchoring the session to a figure that would hide an overflow.
	usageKnown = a.recordUsage(&msg, a.charsSent, usage)

	if stop == provider.StopLength {
		a.emit(Event{Kind: EvNotice, Level: LevelWarn,
			Text: a.lengthStopMessage(outCap, capFrom, usage.CompletionTokens)})
	}

	a.emit(Event{Kind: EvTurnEnd, Turn: turn, Usage: &usage, StopReason: stop,
		Context: ptr(a.ContextState())})
	return msg, stop, usageKnown, reasoned, nil
}

// noticeEmptyTurn reports a turn that ended with nothing to read.
func (a *Agent) noticeEmptyTurn(assistant provider.Message, reasoned bool) {
	if strings.TrimSpace(assistant.Content) != "" {
		return
	}
	text := "The model ended its turn without producing any output or calling a tool. " +
		"Send another message to carry on."
	if reasoned {
		text = "The model spent the whole turn thinking and then stopped without answering. " +
			"Run /verbose to see the reasoning for the next one. Send another message to carry on."
	}
	a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: text})
}

func (a *Agent) finishPartial(stream provider.Stream) provider.Message {
	msg := stream.Message()
	if len(msg.ToolCalls) > 0 {
		a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: fmt.Sprintf(
			"The response stream ended early; %s were incomplete and were discarded.",
			pluralise(len(msg.ToolCalls), "tool call"))})
		msg.ToolCalls = nil
	}
	if strings.TrimSpace(msg.Content) == "" {
		msg.Content = "(the response was cut off before any output arrived)"
	}
	return msg
}

// dispatch runs the requested tools and returns their results in call order.
// Read-only calls in a batch run concurrently; anything that writes or
// executes is serialised, and bash is never treated as read-only.
func (a *Agent) dispatch(ctx context.Context, calls []provider.ToolCall) ([]provider.Message, []tool.Result) {
	results := make([]provider.Message, len(calls))
	// Indexed by call, so concurrent runs write distinct slots without a lock.
	raw := make([]tool.Result, len(calls))
	completed := make([]bool, len(calls))

	run := func(i int, tc provider.ToolCall) {
		a.emit(Event{Kind: EvToolStart, ToolID: tc.ID, ToolName: tc.Name,
			ToolArgs: []byte(tc.Args)})

		res, err := a.exec.Execute(ctx, tool.Request{
			CallID: tc.ID, Name: tc.Name, Args: []byte(tc.Args),
		})
		var unavailable *tool.Unavailable
		if errors.As(err, &unavailable) {
			a.unavailableMu.Lock()
			if a.unavailable == nil {
				a.unavailable = unavailable
			}
			a.unavailableMu.Unlock()
		}
		if err != nil {
			// The executor failed, not the tool; it still has to become a
			// tool result, because an unanswered call breaks the next request.
			res = tool.Result{
				Content: fmt.Sprintf("The tool could not be run: %v", err),
				IsError: true,
			}
		}

		// A tool that handles cancellation may return an ordinary-looking
		// result; remarking it as interrupted stops the model reading it as
		// completed. Tools that already reported the interruption are left alone.
		if ctx.Err() != nil && !res.Interrupted {
			res = tool.Result{
				IsError: true,
				Display: "Interrupted: " + tc.Name,
				Content: "Interrupted by user while this tool was running, so anything it " +
					"touches may have been partially modified. Re-check rather than assuming " +
					"it completed or that it did nothing.\n\nWhat it reported before stopping:\n" +
					res.Content,
			}
		}

		results[i] = provider.Message{
			Role:       provider.RoleTool,
			ToolCallID: tc.ID,
			Name:       tc.Name,
			Content:    res.Content,
			IsError:    res.IsError,
		}
		completed[i] = true
		raw[i] = res
		r := res
		a.emit(Event{Kind: EvToolEnd, ToolID: tc.ID, ToolName: tc.Name, ToolResult: &r})
	}

	i := 0
	for i < len(calls) {
		if ctx.Err() != nil {
			break
		}

		// Longest run of consecutive read-only calls.
		j := i
		if a.opts.ParallelReads {
			for j < len(calls) && a.readOnly(calls[j].Name) {
				j++
			}
		}

		if j-i > 1 {
			var wg sync.WaitGroup
			for k := i; k < j; k++ {
				wg.Add(1)
				go func(k int) {
					defer wg.Done()
					run(k, calls[k])
				}(k)
			}
			wg.Wait()
			i = j
			continue
		}

		run(i, calls[i])
		i++
	}

	// Calls not reached because the context was cancelled still need an answer.
	for k := range calls {
		if !completed[k] {
			results[k] = provider.Message{
				Role:       provider.RoleTool,
				ToolCallID: calls[k].ID,
				Name:       calls[k].Name,
				IsError:    true,
				Content: "Interrupted by user before this tool ran. Nothing it would have done " +
					"has happened, but earlier tools in the same batch may have completed.",
			}
		}
	}
	return results, raw
}

func (a *Agent) appendInterruptResults(calls []provider.ToolCall, done map[string]bool, msg string) {
	for _, tc := range calls {
		if done[tc.ID] {
			continue
		}
		a.messages = append(a.messages, provider.Message{
			Role: provider.RoleTool, ToolCallID: tc.ID, Name: tc.Name,
			IsError: true, Content: msg,
		})
	}
}

// finishInterrupted leaves the conversation in a state that can be sent again.
func (a *Agent) finishInterrupted() error {
	repaired, n := Repair(a.messages)
	a.messages = repaired
	if n > 0 {
		a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: fmt.Sprintf(
			"Interrupted. %s marked as interrupted so the session stays usable.",
			pluralise(n, "unfinished tool call"))})
	}
	a.emit(Event{Kind: EvDone, Text: "interrupted", Context: ptr(a.ContextState())})
	return nil
}

const (
	// DefaultReserveTokens is the headroom kept free at the top of the window:
	// one full response plus the summarisation call compaction makes.
	DefaultReserveTokens = 16384
	// DefaultKeepRecentTokens is how much of the tail survives verbatim.
	DefaultKeepRecentTokens = 20000
	// minOutputTokens is the least room worth starting a turn with: under this
	// a turn stops mid-arguments and finishPartial throws the call away.
	minOutputTokens = 512
)

// Usable is the window minus the reserve, 0 when the window is unknown.
func (a *Agent) Usable() int { return a.usableIn(a.opts.ContextLimit) }

// usableIn is Usable for a window the agent is not on yet.
func (a *Agent) usableIn(limit int) int {
	if limit <= 0 {
		return 0
	}
	reserve := a.reserveTokens()
	// A reserve wider than the window would compact on every turn and never
	// converge, so half the window is the most that can be held back.
	if reserve > limit/2 {
		reserve = limit / 2
	}
	return limit - reserve
}

// budgetSource records which limit produced the cap, for a length-stop message.
type budgetSource int

const (
	// budgetServer means ai-code sent no cap at all.
	budgetServer budgetSource = iota
	budgetContext
	budgetConfig
	budgetBackend
)

// outputBudget is the max_tokens for this turn: the room actually left.
// Recomputed each turn, because a thinking model can spend tens of thousands
// of reasoning tokens mid-stream. Zero means no cap, which an unknown window gets.
func (a *Agent) outputBudget() (int, budgetSource) {
	n, src := 0, budgetServer
	if usable := a.Usable(); usable > 0 {
		n, src = usable-a.RequestTokens(), budgetContext
		if n < minOutputTokens {
			// The estimate moved under the guard; the output floor beats an
			// unanswerable request.
			n = minOutputTokens
		}
	}
	// Both only ever lower the cap: a ceiling is not permission to overrun the window.
	if a.opts.MaxTokens > 0 && (n == 0 || a.opts.MaxTokens < n) {
		n, src = a.opts.MaxTokens, budgetConfig
	}
	if a.opts.MaxOutputTokens > 0 && (n == 0 || a.opts.MaxOutputTokens < n) {
		n, src = a.opts.MaxOutputTokens, budgetBackend
	}
	return n, src
}

// lengthStopMessage names whose cap stopped the response and what to change.
func (a *Agent) lengthStopMessage(sent int, src budgetSource, completion int) string {
	// A stop well short of the cap was not ai-code's cap, whatever was sent.
	if sent == 0 || (completion > 0 && completion < sent*9/10) {
		if sent == 0 {
			return "The response was cut off by the server's own output limit, not by ai-code: " +
				"the context window here is unknown, so no cap was sent. Check the server's --n-predict."
		}
		return fmt.Sprintf(
			"The response was cut off after %d tokens although ai-code allowed %d, so the limit "+
				"was the server's own. Check its --n-predict.", completion, sent)
	}

	switch src {
	case budgetConfig:
		return fmt.Sprintf(
			"The response was cut off at %d tokens by agent.max_tokens in your config, not because "+
				"the model had finished. Raise it or set it to 0: ai-code caps every request at the "+
				"room left in the window anyway, so removing your own cap cannot overflow the context.",
			sent)
	case budgetBackend:
		return fmt.Sprintf(
			"The response was cut off at %d tokens, which is the most this backend accepts for one "+
				"completion. Ask for the work in smaller pieces.", sent)
	default:
		limit := a.opts.ContextLimit
		return fmt.Sprintf(
			"The response was cut off at %s tokens, which was all the room left in a %s window "+
				"with %s held back for summarising. This is ai-code's cap, not your server's: "+
				"the session is nearly full. Run /compact and ask again.",
			compactTokens(sent), compactTokens(limit), compactTokens(limit-a.Usable()))
	}
}

// guardContext keeps the session inside the window, compacting when it
// reaches the reserve. It runs only on a complete message list: before a new
// prompt, and after a round of tool results.
func (a *Agent) guardContext(ctx context.Context) error {
	window := a.opts.ContextLimit
	usable := a.Usable()
	if usable <= 0 {
		return nil // unknown window: nothing to guard against
	}

	// The projection of the request the cut describes -- not of the one that
	// would be sent, already trimmed to fit, and not of the transcript, which
	// only grows.
	projected := a.plannedTokens()

	// Room to answer in, not merely room to sit in: the output floor is part
	// of the trigger.
	if usable-projected >= minOutputTokens {
		// The warning is a fraction of the budget, not the window: against the
		// window it would fire after the compaction it warns about.
		if trigger := usable * a.opts.WarnPercent / 100; projected >= trigger && !a.warned {
			a.warned = true
			a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: fmt.Sprintf(
				"The next request is %s of the %s this session can use (%s window, %s kept free for the reply). "+
					"ai-code will make room by itself when it fills up.",
				compactTokens(projected), compactTokens(usable),
				compactTokens(window), compactTokens(window-usable))})
		}
		return nil
	}

	return a.makeRoom(ctx, projected)
}

// fits reports whether the next request leaves room to answer in.
func (a *Agent) fits() bool {
	usable := a.Usable()
	return usable <= 0 || usable-a.plannedTokens() >= minOutputTokens
}

// makeRoom makes room for the next request, cheapest first: prune, then
// summarise, then drop the oldest turns. Never returns having freed nothing.
func (a *Agent) makeRoom(ctx context.Context, before int) error {
	window := a.opts.ContextLimit
	usable := a.Usable()

	// 1. Clear old tool output: no model call, no waiting.
	if p := a.ClearOldOutput(); p.Results > 0 {
		// Report what the next request saved, not the text removed: most of
		// it was never in the request.
		saved := max(before-a.plannedTokens(), 0)
		a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: fmt.Sprintf(
			"Cleared the output of %s the model had already read, which takes %s off the next request. "+
				"The calls themselves are still there.",
			pluralise(p.Results, "old tool call"), compactTokens(saved))})
		if a.fits() {
			return nil
		}
	}

	// A checkpoint that already covers what the next request leaves out is as
	// good as a fresh one, and costs no model call.
	if a.resumeFromTranscript {
		return nil
	}

	if !a.opts.AutoCompact {
		return fmt.Errorf("%w: the next request is %s and the window is %s, with %s kept free for the reply.\n\n"+
			"Run /compact to summarise what has happened so far, or start again with /new. "+
			"Either way the session on disk keeps every message. Set agent.auto_compact = true "+
			"to have ai-code do this for you",
			ErrContextFull, compactTokens(a.plannedTokens()), compactTokens(window),
			compactTokens(window-usable))
	}

	// 2. Summarise. One attempt: summarising is itself a request that has to
	// fit, and nothing has been removed if it fails.
	a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: fmt.Sprintf(
		"The next request would be %s, and the window is %s. Summarising the older messages to make room. "+
			"Nothing is deleted -- the session on disk keeps every message.",
		compactTokens(a.plannedTokens()), compactTokens(window))})

	res, err := a.Compact(ctx, 0)
	switch {
	case errors.Is(err, ErrNothingToFree):
		// Nothing left a summary could stand in for; the escalation below is
		// exactly that case.
	case err != nil:
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("%w: the next request is %s, the window is %s, and summarising failed: %w.\n\n"+
			"Start again with /new. The session on disk keeps every message",
			ErrContextFull, compactTokens(a.plannedTokens()), compactTokens(window), err)
	default:
		res.Cleared, res.ClearedTokens = 0, 0
		a.emit(Event{Kind: EvCompacted, Text: res.Summary, Compaction: res,
			Context: ptr(a.ContextState())})
		if a.fits() {
			return nil
		}
	}

	// 3. The kept tail is still too large, so one turn inside it is. Drop the
	// oldest, one at a time, and never the newest: the checkpoint was written
	// before it, so dropping it loses work outright.
	lastTurn := a.lastTurnStart()
	dropped := 0
	for a.startAt < lastTurn && !a.fits() {
		next := a.nextTurnAfter(a.startAt)
		if next > lastTurn {
			break
		}
		a.startAt = legalStart(a.messages, next)
		dropped++
		a.requestChanged()
		a.publish()
	}
	if dropped > 0 && a.fits() {
		a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: fmt.Sprintf(
			"The summary was not enough on its own, so the %s after it are no longer sent either. "+
				"The summary covers them, and the session on disk keeps every message. "+
				"The next request is %s.",
			pluralise(dropped, "oldest remaining turn"), compactTokens(a.plannedTokens()))})
		return nil
	}
	if a.fits() {
		return nil
	}

	// 4. One message is larger than the whole budget. Say which.
	return fmt.Errorf("%w: %s",
		ErrContextFull, a.oversizedMessage(usable))
}

// lastTurnStart is where the newest turn begins, the boundary escalation will not cross.
func (a *Agent) lastTurnStart() int {
	for i := len(a.messages) - 1; i >= 0; i-- {
		if a.messages[i].Role == provider.RoleUser {
			return i
		}
	}
	return 0
}

// nextTurnAfter is the start of the first turn beginning strictly after i.
func (a *Agent) nextTurnAfter(i int) int {
	for j := i + 1; j < len(a.messages); j++ {
		if a.messages[j].Role == provider.RoleUser {
			return j
		}
	}
	return len(a.messages)
}

// oversizedMessage names the single entry no amount of summarising can shed.
func (a *Agent) oversizedMessage(usable int) string {
	worst, size := -1, 0
	for i := a.startAt; i < len(a.messages); i++ {
		if n := a.estimate(messageChars(a.messages[i])); n > size {
			worst, size = i, n
		}
	}
	if worst < 0 {
		return fmt.Sprintf("the next request is %s and only %s can be used, "+
			"and there is nothing left to remove.\n\n"+
			"Start again with /new; the session on disk keeps every message",
			compactTokens(a.plannedTokens()), compactTokens(usable))
	}
	m := a.messages[worst]
	what := string(m.Role) + " message"
	if m.Role == provider.RoleTool {
		what = "result of the " + m.Name + " tool"
	}
	return fmt.Sprintf("the next request is %s and only %s can be used. The %s in it is %s "+
		"on its own -- more than a whole session has room for.\n\n"+
		"Summarising cannot get rid of it. Start again with /new; "+
		"the session on disk keeps every message",
		compactTokens(a.plannedTokens()), compactTokens(usable), what, compactTokens(size))
}

// compactTokens formats a token count for a message a person reads.
func compactTokens(n int) string {
	switch {
	case n >= 1000000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	default:
		return fmt.Sprint(n)
	}
}

func (a *Agent) withSystem(messages []provider.Message) []provider.Message {
	if a.system == "" {
		return messages
	}
	out := make([]provider.Message, 0, len(messages)+1)
	out = append(out, provider.Message{Role: provider.RoleSystem, Content: a.system})
	return append(out, messages...)
}

func toolDefs(e tool.Executor) []provider.ToolDef {
	defs := e.Definitions()
	out := make([]provider.ToolDef, 0, len(defs))
	for _, d := range defs {
		out = append(out, provider.ToolDef{
			Name: d.Name, Description: d.Description, Schema: d.Schema,
		})
	}
	return out
}

// signature identifies a batch of tool calls for loop detection.
func signature(calls []provider.ToolCall) string {
	var b strings.Builder
	for _, tc := range calls {
		b.WriteString(tc.Name)
		b.WriteByte('(')
		b.WriteString(tc.Args)
		b.WriteString(") ")
	}
	return b.String()
}

func looping(recent []string, n int) bool {
	if len(recent) < n {
		return false
	}
	last := recent[len(recent)-1]
	for i := len(recent) - n; i < len(recent); i++ {
		if recent[i] != last {
			return false
		}
	}
	return true
}

// maxMessageChars bounds one message in the transcript. The unit is the
// verbatim tail, because a message larger than that can never be kept beside
// the conversation it belongs to; half the usable window caps it again.
func (a *Agent) maxMessageChars() int {
	limit := a.budget().MaxToolResult
	if a.opts.KeepRecentTokens > 0 {
		// An override of the tail budget carries the result cap with it.
		limit = max(a.opts.KeepRecentTokens/4, minToolResultTokens)
	}
	if usable := a.Usable(); usable > 0 && limit > usable/2 {
		limit = usable / 2
	}
	return int(float64(limit) * a.charsPerTokenNow())
}

// clampOversized trims tool results too large to live in the transcript. It
// runs as results enter, the last moment before they become history. Only
// tool results: cutting a pasted message up is worse than the context cost.
func (a *Agent) clampOversized(msgs []provider.Message) {
	limit := a.maxMessageChars()
	if limit <= 0 {
		return
	}
	for i := range msgs {
		if msgs[i].Role != provider.RoleTool || len(msgs[i].Content) <= limit {
			continue
		}
		trimmed, _ := tool.Truncate(msgs[i].Content, limit)
		msgs[i].Content = trimmed + "\n(this result was too large for the context window, so only the " +
			"beginning and end are shown; narrow the command or filter its output if you need the rest)"
	}
}

// messageOverheadChars approximates the role markers and JSON scaffolding a
// message costs beyond its own text; calibration absorbs any error in it.
const messageOverheadChars = 32

// messageChars is the size of one message as it will be sent. One definition,
// used by the running count, by calibration and by the compaction budget, so
// those three can never disagree about how big a conversation is.
func messageChars(m provider.Message) int {
	n := len(m.Content) + len(m.Role) + len(m.Name) + len(m.ToolCallID) + messageOverheadChars
	for _, tc := range m.ToolCalls {
		n += len(tc.Name) + len(tc.Args) + len(tc.ID) + messageOverheadChars
	}
	return n
}

// pluralise handles the sibilant endings a bare "+s" gets wrong.
func pluralise(n int, word string) string {
	if n == 1 {
		return "1 " + word
	}
	switch {
	case strings.HasSuffix(word, "s"), strings.HasSuffix(word, "x"),
		strings.HasSuffix(word, "ch"), strings.HasSuffix(word, "sh"):
		return fmt.Sprintf("%d %ses", n, word)
	default:
		return fmt.Sprintf("%d %ss", n, word)
	}
}

func ptr[T any](v T) *T { return &v }

// SetClient swaps the backend mid-session, as /provider does. Messages are
// canonical, so a switch is a change of destination rather than a translation.
func (a *Agent) SetClient(c provider.Client) { a.client = c }

// Client returns the current backend.
func (a *Agent) Client() provider.Client { return a.client }
