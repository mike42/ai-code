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
	// MaxIterations caps the turns one Run may take. 0 -- the default -- is
	// unlimited: the runaway cases a turn count looked like it was covering
	// are covered by looping() and guardContext() instead.
	MaxIterations int
	// MaxTokens is a ceiling on one response, not the cap ai-code sends. The
	// cap is derived from the window every turn (see outputBudget); this only
	// lowers it further. 0 means no ceiling of the user's own.
	MaxTokens int
	// MaxOutputTokens is the backend's own limit on a single completion, from
	// ModelInfo.MaxOutputTokens. 0 when unknown -- which is not the same as
	// unlimited, so it only ever lowers the cap, never raises it.
	MaxOutputTokens int
	Temperature     *float64
	TopP            *float64
	ParallelReads   bool
	LoopGuard       int
	// ContextLimit is the effective window in tokens, 0 when unknown.
	ContextLimit int
	// WarnPercent is the occupancy at which the agent emits a context warning.
	WarnPercent int

	// AutoCompact summarises the session when it reaches the reserve rather
	// than stopping and asking.
	AutoCompact bool
	// ReserveTokens is the room kept free at the top of the window: enough for
	// one full response plus the summarisation call compaction itself makes.
	ReserveTokens int
	// KeepRecentTokens is how much of the tail survives compaction verbatim.
	KeepRecentTokens int

	// Effort is the thinking level to request. EffortUnset sends nothing.
	Effort provider.Effort
}

type Agent struct {
	client   provider.Client
	model    string
	exec     tool.Executor
	sink     Sink
	opts     Options
	system   string
	messages []provider.Message

	// Token accounting lives in accounting.go. Ground truth is recorded on the
	// assistant message it describes; the only accounting state here is the
	// ratio, which is a property of the model rather than of the messages, and
	// the position that says which recorded figures still apply.
	//
	// charsSent is the size of the request currently in flight, kept so the
	// response can be divided by it.
	charsSent int
	// charsPerToken is the measured ratio, defaultCharsPerToken until a
	// response has been seen.
	charsPerToken float64
	// requestChangedAt is the length of the transcript when what sits in front
	// of it last moved. A prefill recorded before that point described a
	// prefix that no longer exists. See requestChanged.
	requestChangedAt int
	// toolChars caches the size of the tool definitions, which are sent on
	// every request and do not change during a session.
	toolChars int
	warned    bool

	readOnly func(name string) bool

	// steer holds messages typed while the turn was running. They are appended
	// at turn boundaries rather than injected where they arrive, because the
	// message list has a shape the provider enforces: a user message may not
	// come between an assistant message that requested tools and the results of
	// those tools. Waiting for the boundary is not a limitation to be worked
	// around -- it is the only point at which an insertion is well-formed.
	steerMu sync.Mutex
	steerQ  []string

	// summary is a standby copy of the conversation so far, kept beside the
	// transcript rather than in place of it. Nothing sends it while the real
	// messages fit; MessagesFitting reaches for it only when they do not,
	// which is what makes it safe to write one speculatively and throw it
	// away. It is also fed back into the next summarisation, so a second one
	// updates the checkpoint rather than starting again from a shrinking
	// window of history.
	summary string
	// startAt is where the next request begins. Messages before it are not
	// sent; the summary stands in for them. Zero means everything is sent.
	//
	// Separate from summarisedThrough because a summary written while the
	// session was idle covers messages without changing what gets sent. Only
	// /compact and the automatic path move this.
	//
	// Nothing is deleted when it moves: the session on disk stays whole, so
	// a wider model can still reach the older messages.
	startAt int
	// summarisedThrough is how many messages from the start of the transcript
	// the summary accounts for. Without it there is no way to tell whether a
	// checkpoint still covers what a request has to leave out, and the choice
	// is between summarising every turn and silently dropping detail.
	summarisedThrough int

	// resumeFromTranscript is the user's answer to a window that shrank under
	// them: carry recent messages verbatim and let the older ones fall off,
	// rather than spend part of a small window on a checkpoint. See
	// SetResumeFromTranscript.
	resumeFromTranscript bool

	// stuck watches for a session going in circles without repeating a call
	// exactly, which is the shape the loop guard cannot see.
	stuck stuckWatch

	// atBoundary is consulted between iterations, where the message list is
	// complete. Returning true ends the run cleanly. See SetTurnBoundary.
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
	// An opening figure, so a consumer that only ever adopts what it is told
	// has something to show before the first turn rather than a zero.
	a.publish()
	return a
}

// Window is the context length of the model now loaded, 0 when unknown.
//
// A property of the model rather than of the accounting, which is why it is
// asked for directly instead of being read off the context figure.
func (a *Agent) Window() int { return a.opts.ContextLimit }

func (a *Agent) SetSystem(prompt string) {
	a.system = prompt
	// Part of every request, so changing it changes the prefix that any
	// recorded figure described.
	a.requestChanged()
	a.publish()
}

// System returns the assembled system prompt, for a caller that needs to
// inspect what the model is actually being told.
func (a *Agent) System() string { return a.system }

// budget is the sizing derived from the window the agent is on now.
//
// Read rather than stored, so a model swap re-derives it. Stored, every
// caller of SetModel would have to remember to recompute, and the one that
// forgot would leave a 262k session's sizes on a 32k model.
func (a *Agent) budget() Budget { return BudgetFor(a.opts.ContextLimit) }

// reserveTokens and keepRecentTokens are the derived sizes unless the user
// set them. A configured value is a global override and is taken as given:
// someone who writes a number in a config file has said they know better than
// the formula, which is a thing they are allowed to be right about.
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

// SummaryCap is the most a checkpoint of this session may run to.
//
// Derived from the reserve actually in force rather than from the window,
// because the reserve is what has to hold the checkpoint: someone who
// overrides the reserve downwards has lowered the room a summary has to fit
// in, whether or not they were thinking about summaries at the time.
func (a *Agent) SummaryCap() int {
	return clamp(a.reserveTokens()/2, minSummaryTokens, SummaryMaxTokens)
}

// contextChars sizes the request ai-code would send now: system prompt, tool
// definitions, and the messages that would go with them -- which is the whole
// transcript only while the whole transcript fits. The first two are thousands
// of tokens and go out every turn, so leaving them out understates a session.
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

// transcriptChars is the size of the whole conversation, whether or not all of
// it would be sent.
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

// SetMessages replaces the conversation. Any checkpoint goes with it: a
// summary of the session /new just left behind would be handed to the model as
// the state of the one it is starting.
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
// of the transcript it accounts for. Both are empty until something has
// summarised the session.
func (a *Agent) Summary() (string, int) { return a.summary, a.summarisedThrough }

// SetSummary puts back a checkpoint recorded by an earlier run. Call it after
// SetMessages, which clears one.
//
// A resumed session with no checkpoint is the ordinary case, not an error:
// every session recorded before checkpoints existed is one, and so is every
// session that never filled its window.
func (a *Agent) SetSummary(summary string, summarisedThrough int) {
	a.SetCheckpoint(summary, summarisedThrough, 0)
}

// SetCheckpoint puts back a checkpoint recorded by an earlier run, including
// the boundary a compaction chose.
//
// A cut of zero restores a checkpoint that stood beside the transcript and
// changed nothing about what was sent; a non-zero one restores a compaction,
// so the resumed session assembles its requests from the same place the
// original did rather than replaying everything the compaction made room by
// setting aside.
func (a *Agent) SetCheckpoint(summary string, summarisedThrough, cut int) {
	clampIndex := func(n int) int { return clamp(n, 0, len(a.messages)) }
	a.summary = summary
	a.summarisedThrough = clampIndex(summarisedThrough)
	a.startAt = clampIndex(cut)
	// Whatever was recorded against these messages was recorded under a
	// different system prompt and a different tool set, in another process.
	a.requestChanged()
	a.publish()
}

func (a *Agent) Model() string { return a.model }

// SetModel points the agent at a different model. The backend's own output
// limit travels with it: it is a property of the model being swapped to, and
// leaving the previous one in place would cap every later request at a number
// belonging to a model that is no longer loaded.
// SetTurnBoundary installs a check that runs between loop iterations, after
// the tool results and before the next request.
//
// That is the only point at which a run can be stopped and leave a
// well-formed message list behind, which is what makes the stop recoverable.
// A long agentic run reaches it many times; it reaches the end of the run
// only when the model decides to finish, which may be a long way off.
func (a *Agent) SetTurnBoundary(fn func(context.Context) bool) { a.atBoundary = fn }

func (a *Agent) SetModel(model string, contextLimit, maxOutputTokens int) {
	a.model = model
	a.opts.ContextLimit = contextLimit
	a.opts.MaxOutputTokens = maxOutputTokens
	a.warned = false
	// A different model tokenizes differently, so the learned ratio belongs
	// to the one being left behind, and every figure recorded under it
	// described a request this model will never see.
	a.charsPerToken = 0
	defer func() {
		a.requestChanged()
		a.publish()
	}()
	// A choice about how to survive one window must not outlive it. Carried
	// into a wider model it would keep auto-compaction switched off for a
	// session that has room for it again.
	a.resumeFromTranscript = false
}

// SetResumeFromTranscript records that the user would rather lose the oldest
// turns outright than spend a narrow window on a checkpoint.
//
// Call it after SetModel, which clears it. It changes two things: assembly
// stops reaching for the summary, and the automatic compaction that would
// otherwise write one is skipped -- a checkpoint nothing will send is a model
// call, and on a local model that is minutes, for nothing.
func (a *Agent) SetResumeFromTranscript(v bool) { a.resumeFromTranscript = v }

// StoppedMidTurn reports whether the transcript ends part-way through an
// exchange: a request with no reply, or a tool round the model never got to
// read back.
//
// A session in that state has work outstanding that nobody has to be asked
// about, which is what makes resuming it automatically safe.
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

// Steer queues a message to be folded into the running turn.
//
// Safe to call from another goroutine, and from outside a turn: anything queued
// while nothing is running stays queued until TakeSteering collects it, so a
// message typed in the instant a turn ends is not lost to the race.
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

// Run performs turns until the model stops requesting tools.
//
// The context governs everything: the HTTP stream, every running tool, and the
// loop itself. Cancelling it leaves the conversation in a sendable state, which
// is the whole reason cancellation is threaded rather than bolted on.
func (a *Agent) Run(ctx context.Context, userInput string) error {
	// Anything queued before the turn began -- typed in the moment the previous
	// one was ending -- goes in ahead of the new input, in the order it was
	// actually typed.
	a.applySteering()
	if userInput != "" {
		a.AppendUser(userInput)
	}

	var recent []string
	// One retry for the whole run, not one per turn: a window that is wrong
	// is wrong for every turn, and retrying each of them turns one bad
	// reading into a session that never makes progress.
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
			// The backend refused the request for length despite the guard
			// clearing it. That means the window is not what the model
			// reported, or a prefix the server adds was not counted. Make
			// room and try once. Not twice: the second attempt would run
			// against the state the first one already failed on.
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

		// The reported figures travel on the message itself, attached in
		// streamTurn, so nothing here has to remember that they describe this
		// point in the conversation and not a later one.
		a.messages = append(a.messages, assistant)
		a.publish()

		if len(assistant.ToolCalls) == 0 {
			// A finished turn that said nothing is indistinguishable from a
			// hung one: EvDone paints nothing, and reasoning never reaches
			// the scrollback, so the session appears to have stopped
			// answering. Whatever else is wrong, the user is owed the
			// difference between "no reply" and "no response".
			a.noticeEmptyTurn(assistant, reasoned)

			// The model has finished, but the user may have typed something
			// while it was talking. Taking it up here rather than dropping it
			// is the whole point of steering: you correct the course as you
			// read the reply, and the correction lands without you having to
			// wait for the prompt and repeat yourself.
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
			// Something outside the loop needs it to stop, and this is the
			// only place it can stop cleanly. A long agentic run may be many
			// iterations from the user's prompt, so waiting for the run to
			// end is waiting indefinitely.
			a.emit(Event{Kind: EvDone, Text: "stopped", Context: ptr(a.ContextState())})
			return nil
		}
	}

	// Only reachable when agent.max_iterations was set deliberately; the
	// default is unlimited.
	a.emit(Event{Kind: EvNotice, Level: LevelWarn, Text: fmt.Sprintf(
		"Stopped after the configured limit of %d turns, without the model finishing.",
		a.opts.MaxIterations)})
	return nil
}

// streamTurn sends one request and consumes the response stream.
//
// reasoned reports that the model emitted thinking, which in collapsed mode
// never reaches the scrollback -- so it is the difference between a turn that
// said nothing and a turn that did nothing.
func (a *Agent) streamTurn(ctx context.Context, turn int) (msg provider.Message, stop provider.StopReason, usageKnown, reasoned bool, err error) {
	// What goes on the wire is decided here rather than held in a.messages,
	// because the window belongs to whichever model is loaded now and the
	// transcript belongs to the session.
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

	// Whether the model thought without answering. Reasoning is not committed
	// to the scrollback in collapsed mode, so a turn that only reasons leaves
	// nothing behind at all, and the difference matters to what we tell the
	// user about it.
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
			// Partial text is kept because it is often the useful part; partial
			// tool calls are discarded by finishPartial, because a truncated
			// arguments blob is worse than no call at all.
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
			// Deliberately not announced here. A tool call is announced once,
			// by dispatch, when it actually starts running -- announcing it
			// again as it streams in produces a duplicate line for every call
			// and tells the user nothing they are not about to see.
		case provider.EventUsage:
			u := ev.Usage
			a.emit(Event{Kind: EvUsage, Usage: &u, Turn: turn})
		}
	}

	usage := stream.Usage()
	msg = stream.Message()
	stop = stream.StopReason()

	// Ground truth is attached to the message the request produced, and only
	// if it can be believed. A backend that reports a cache hit as a tiny
	// prompt_tokens leaves the message unmarked rather than anchoring the
	// session to a figure that would hide an overflow.
	usageKnown = a.recordUsage(&msg, a.charsSent, usage)

	if stop == provider.StopLength {
		a.emit(Event{Kind: EvNotice, Level: LevelWarn,
			Text: a.lengthStopMessage(outCap, capFrom, usage.CompletionTokens)})
	}

	a.emit(Event{Kind: EvTurnEnd, Turn: turn, Usage: &usage, StopReason: stop,
		Context: ptr(a.ContextState())})
	return msg, stop, usageKnown, reasoned, nil
}

// noticeEmptyTurn reports a turn that ended with nothing for the user to read.
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
//
// Read-only tools in the same batch run concurrently, which is most of the
// latency win, while anything that writes or executes is serialised. Working
// out whether a shell command mutates state would mean parsing shell grammar,
// so bash is simply never concurrent.
func (a *Agent) dispatch(ctx context.Context, calls []provider.ToolCall) ([]provider.Message, []tool.Result) {
	results := make([]provider.Message, len(calls))
	// Indexed by call, so concurrent runs write to distinct slots and no lock
	// is needed. The messages lose the fields stuck detection reads.
	raw := make([]tool.Result, len(calls))
	completed := make([]bool, len(calls))

	run := func(i int, tc provider.ToolCall) {
		a.emit(Event{Kind: EvToolStart, ToolID: tc.ID, ToolName: tc.Name,
			ToolArgs: []byte(tc.Args)})

		res, err := a.exec.Execute(ctx, tool.Request{
			CallID: tc.ID, Name: tc.Name, Args: []byte(tc.Args),
		})
		if err != nil {
			// The executor itself failed, not the tool. Still has to become a
			// tool result: an unanswered call breaks the next request.
			res = tool.Result{
				Content: fmt.Sprintf("The tool could not be run: %v", err),
				IsError: true,
			}
		}

		// A tool that handles cancellation itself may return an ordinary-looking
		// result. Reporting that verbatim would tell the model the work
		// completed. Whatever it managed to say is kept, but the interruption
		// is stated first and the result is marked as an error.
		//
		// Tools that already reported the interruption themselves are left
		// alone: they said it better, with detail this layer does not have.
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

		// Gather the longest run of consecutive read-only calls.
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

	// Anything not reached, because the context was cancelled partway through,
	// still needs an answer.
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

// Default sizes for the context reserve. See Options.
const (
	// DefaultReserveTokens is the headroom kept free at the top of the window.
	// It must hold one full response plus the summarisation call that
	// compaction makes, because compaction runs at the moment the reserve is
	// reached and needs room to do it.
	DefaultReserveTokens = 16384
	// DefaultKeepRecentTokens is how much of the tail survives verbatim.
	DefaultKeepRecentTokens = 20000
	// minOutputTokens is the least room worth starting a turn with. A tool call
	// carrying a path and a small edit is a few hundred tokens before the model
	// has said anything, so under this the turn stops mid-arguments and
	// finishPartial throws the call away: a wasted round trip rather than a
	// short answer. Reaching it means the session is full, not that the reply
	// should be terse.
	minOutputTokens = 512
)

// Usable is the window minus the reserve: the point at which the session is
// compacted. Zero when the window is unknown.
func (a *Agent) Usable() int { return a.usableIn(a.opts.ContextLimit) }

// usableIn is Usable for a window the agent is not on yet, so a model change
// can be costed before it is made rather than discovered after it.
func (a *Agent) usableIn(limit int) int {
	if limit <= 0 {
		return 0
	}
	reserve := a.reserveTokens()
	// A reserve wider than the window itself would compact on every turn and
	// never converge. Half the window is the most that can sensibly be held
	// back. The derived reserve is an eighth, so this now only ever catches a
	// configured override that was set for a roomier model than the one
	// actually loaded.
	if reserve > limit/2 {
		reserve = limit / 2
	}
	return limit - reserve
}

// budgetSource records which limit produced the cap, so a length stop can name
// the component the user has to go and change.
type budgetSource int

const (
	// budgetServer means ai-code sent no cap at all.
	budgetServer budgetSource = iota
	budgetContext
	budgetConfig
	budgetBackend
)

// outputBudget is the max_tokens for this turn: the room actually left.
//
// Recomputed rather than fixed, because a thinking model can spend tens of
// thousands of reasoning tokens against the same budget and overflow the
// window mid-stream. Zero means no cap, which is what an unknown window gets.
func (a *Agent) outputBudget() (int, budgetSource) {
	n, src := 0, budgetServer
	if usable := a.Usable(); usable > 0 {
		n, src = usable-a.RequestTokens(), budgetContext
		if n < minOutputTokens {
			// guardContext compacts before it gets this far, so arriving here
			// means the estimate moved under us between the two. Eating a little
			// of the reserve beats sending a request that cannot answer.
			n = minOutputTokens
		}
	}
	// Both of these only ever lower the cap. The user asked for a ceiling, not
	// permission to overrun the window; the backend's own limit is a fact about
	// what it will accept.
	if a.opts.MaxTokens > 0 && (n == 0 || a.opts.MaxTokens < n) {
		n, src = a.opts.MaxTokens, budgetConfig
	}
	if a.opts.MaxOutputTokens > 0 && (n == 0 || a.opts.MaxOutputTokens < n) {
		n, src = a.opts.MaxOutputTokens, budgetBackend
	}
	return n, src
}

// lengthStopMessage names whose cap stopped the response and what to do about
// it. The bare version sent people to their server settings when the limit was
// ai-code's own, which is the single most expensive kind of wrong message: it
// costs an hour of debugging the wrong component.
func (a *Agent) lengthStopMessage(sent int, src budgetSource, completion int) string {
	// A stop well short of what we asked for was not our cap, whatever we sent.
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

// guardContext keeps the session inside the window, compacting when it reaches
// the reserve.
//
// Both call sites matter and they are the two pi identified: before a new user
// prompt, and after a round of tool results has been appended but before the
// next request. Those are the points at which the message list is complete and
// a rewrite of it is well-formed. Checking anywhere else means either
// compacting a conversation that is missing its tool results, or discovering
// the overflow as a provider error mid-stream.
func (a *Agent) guardContext(ctx context.Context) error {
	window := a.opts.ContextLimit
	usable := a.Usable()
	if usable <= 0 {
		return nil // unknown window: nothing to guard against
	}

	// The projection of the request the cut describes -- not of the one that
	// would be sent, which has already been trimmed to fit and so always
	// fits, and not of the transcript, which since compaction records a
	// boundary rather than deleting anything goes on growing for the rest of
	// the session.
	projected := a.plannedTokens()

	// Room to answer in, not merely room to sit in. A session that fits but
	// leaves nothing for the reply produces a request that cannot succeed, so
	// the output floor is part of the trigger rather than a check further down.
	if usable-projected >= minOutputTokens {
		// The warning is a fraction of the *budget*, not of the window. Measured
		// against the window it is dead on any model whose reserve is a large
		// share of it: 80% of 64k is 52k, which is past the 49k budget, so the
		// heads-up would arrive after the thing it was warning about.
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

// makeRoom makes room for the next request, cheapest mechanism first.
//
// The order is the whole design. Pruning costs nothing and clears the trigger
// outright on a session whose weight is tool output, which is most coding
// sessions; summarising costs a model call, which on a local backend is
// minutes. Doing them the other way round -- or only ever doing the second --
// is what made compaction feel like a treadmill.
//
// Every step strictly shrinks the request, so the ladder terminates on the
// number of entries rather than on a step count. What it must never do is
// return nil having freed nothing: that is the wedge, where a session spends
// the rest of its life sending a request it has already been told is too big.
func (a *Agent) makeRoom(ctx context.Context, before int) error {
	window := a.opts.ContextLimit
	usable := a.Usable()

	// 1. Clear old tool output. No model call, no waiting.
	if p := a.ClearOldOutput(); p.Results > 0 {
		// What the next request saved, not how much text was removed. Most
		// of a long session is not in the next request at all, so the amount
		// removed can be several times the whole window -- a true number
		// that tells the reader nothing and looks like a bug.
		saved := max(before-a.plannedTokens(), 0)
		a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: fmt.Sprintf(
			"Cleared the output of %s the model had already read, which takes %s off the next request. "+
				"The calls themselves are still there.",
			pluralise(p.Results, "old tool call"), compactTokens(saved))})
		if a.fits() {
			return nil
		}
	}

	// A checkpoint that already accounts for everything the next request has
	// to leave out is as good as a fresh one, and costs no model call. Neither
	// is one the user has said not to send: assembly is already dropping the
	// oldest turns on their instruction, so the request fits without it.
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

	// 2. Summarise. One attempt: it is itself a request, and retrying it runs
	// into the same wall. A failure leaves the session exactly as it was,
	// because nothing has been removed.
	a.emit(Event{Kind: EvNotice, Level: LevelInfo, Text: fmt.Sprintf(
		"The next request would be %s, and the window is %s. Summarising the older messages to make room. "+
			"Nothing is deleted -- the session on disk keeps every message.",
		compactTokens(a.plannedTokens()), compactTokens(window))})

	res, err := a.Compact(ctx, 0)
	switch {
	case errors.Is(err, ErrNothingToFree):
		// Nothing left that a summary could stand in for. Not a failure --
		// the escalation below is exactly the case this describes.
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

	// 3. The kept tail is still too large, which means one turn inside it is.
	// Drop the oldest of them, one at a time.
	//
	// Never the newest, whatever it costs. The checkpoint covers everything
	// up to the cut, so a turn dropped from in front of it is at worst
	// summarised rather than verbatim -- but the newest turn is what the
	// model is in the middle of, and the summary was written before it, so
	// dropping that one loses work outright and leaves a tool call with no
	// result behind it.
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

	// 4. One message is larger than the whole budget. Say which, because
	// nothing the session can do to itself will change that.
	return fmt.Errorf("%w: %s",
		ErrContextFull, a.oversizedMessage(usable))
}

// lastTurnStart is where the newest turn begins, which is the boundary
// escalation will not cross.
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

// oversizedMessage names the single entry that no amount of summarising can
// shed, so the report is actionable rather than a restatement of the arithmetic.
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

// maxMessageChars bounds one message in the transcript.
//
// The unit is the verbatim tail, because a message larger than the whole tail
// budget can never be kept beside the conversation it belongs to: whatever
// else compaction drops, that one message still does not fit. Half the usable
// window caps it again on narrow windows, where the configured tail may be a
// large share of everything there is.
func (a *Agent) maxMessageChars() int {
	limit := a.budget().MaxToolResult
	if a.opts.KeepRecentTokens > 0 {
		// An override of the tail budget carries the result cap with it, so
		// the two cannot be configured into disagreeing.
		limit = max(a.opts.KeepRecentTokens/4, minToolResultTokens)
	}
	if usable := a.Usable(); usable > 0 && limit > usable/2 {
		limit = usable / 2
	}
	return int(float64(limit) * a.charsPerTokenNow())
}

// clampOversized trims a tool result too large to live in the transcript.
//
// The tools bound their own output, so this is a backstop rather than the
// first line of defence -- but it is the only one that holds for every tool
// at once, including whichever one gets a new early-return path next. It runs
// where the results enter the transcript, which is the last moment anything
// can be done about them: afterwards they are history, and rewriting history
// is what compaction is deliberately not allowed to do.
//
// Only tool results. A user who pastes something enormous has said what they
// meant to say, and silently cutting it up is worse than the context cost;
// compaction bounds the opening request separately, see
// preservedRequestBudget.
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

// estimateMessage approximates a message's token cost.
//
// Only ever used for content appended since the last response, because the
// provider's own prompt_tokens covers everything before that. Four bytes per
// token is close enough for code and prose, and the per-message constant covers
// role and delimiter overhead.
// messageOverheadChars approximates the role markers and JSON scaffolding a
// message costs beyond its own text. The exact value barely matters: it is
// included in the figure calibration divides, so a systematic error in it is
// absorbed by the measured ratio rather than accumulated.
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

// SetClient swaps the backend mid-session.
//
// Used by /provider. The conversation is unaffected: messages are canonical and
// every backend ai-code speaks to takes the same shape, so a switch is a change of
// destination rather than a translation.
func (a *Agent) SetClient(c provider.Client) { a.client = c }

// Client returns the current backend.
func (a *Agent) Client() provider.Client { return a.client }
