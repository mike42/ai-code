package render

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"ai-code/internal/agent"
)

// Interactive is the terminal renderer.
//
// It owns the transient zone and composes it from two things: the line
// currently streaming, and the status line. Everything else it writes goes
// straight to the scrollback, once.
type Interactive struct {
	screen *Screen
	style  Style
	md     *Markdown

	showStatus bool
	warnPct    int
	steerMark  string

	mu           sync.Mutex
	partial      []string
	turnStarted  time.Time
	firstTokenAt time.Time
	tokens       int
	samples      []rateSample
	activeTool   string
	// reasonTrace is everything the model has thought this turn, kept so that
	// turning /verbose on part-way through can show the thinking that has
	// already happened rather than only what comes next. Reset each turn and
	// otherwise unbounded, because one turn's reasoning is already bounded by
	// the output cap the turn was sent with.
	reasonTrace strings.Builder
	// busy is set for as long as a turn is running, which is not the same as
	// streaming: the agent also works between requests -- summarising to free
	// context, most of all -- and during that the status line has to keep
	// moving or the session looks hung. See SetBusy.
	busy         bool
	toolStarted  time.Time
	ctxState     agent.ContextState
	streaming    bool
	spinnerIndex int
	reasoning    string
	// notice is transient progress from outside the turn loop, such as a
	// model swap. See SetNotice.
	notice     string
	reasonMode string
	sawText    bool

	// verbose is the /verbose toggle: full thinking, tool arguments and tool
	// output, all committed. It is read on the agent's goroutine as events
	// arrive and written on the input goroutine when the command is run, so it
	// lives under the same mutex as the rest of the display state.
	verbose bool
	// reasonBuf holds the tail of a thinking delta that has not reached a
	// newline yet, and reasonOpen whether a thinking block is currently being
	// committed. Only used in verbose mode.
	reasonBuf  string
	reasonOpen bool

	// Steering state. steerActive is what decides whether the bottom line is a
	// prompt or the status line -- not whether steerText is empty, because the
	// two differ for exactly one keystroke: the moment the user deletes the
	// last character, which is when the status line has to come back.
	steerText   string
	steerCol    int
	steerActive bool
	// steerQueued counts messages submitted but not yet folded in. Between
	// those two moments the model is still working from the old instruction,
	// and the wait can be a whole tool call long -- so the status line has to
	// say the message was received, or Enter looks like it did nothing.
	steerQueued int

	// dirty coalesces transient-zone repaints. A fast model emits hundreds of
	// deltas a second, and repainting on each one floods the terminal with
	// escape sequences -- which is what makes streaming output impossible to
	// read and pointlessly expensive over ssh. Deltas set this flag; the ticker
	// does the repaint at a fixed rate.
	dirty bool

	stop chan struct{}
	done chan struct{}
}

// rateSample is one reading of the running token count.
//
// The rate is computed over a trailing window rather than the whole turn. A
// cumulative average is the wrong statistic for a live readout: it is dominated
// by everything that came before, so when the true rate changes the displayed
// number converges as 1/t. That looks exactly like generation slowing down on
// its own, while on screen it plainly is not.
type rateSample struct {
	at time.Time
	n  int
}

const rateWindow = 5 * time.Second

// spinner uses braille dots: they occupy one column in every terminal and do
// not shift the line width as they animate.
var spinner = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

type InteractiveOptions struct {
	ShowStatus  bool
	Theme       string
	Reasoning   string // off | collapsed | full
	WarnPercent int
	// Verbose is the starting detail level, from -v. /verbose and /quiet move
	// it afterwards.
	Verbose bool
	// SteerPrompt is the marker drawn in place of the status line while the
	// user is typing during a turn. Passed in rather than defined here so that
	// there is one definition of the prompt marker in the program.
	SteerPrompt string
}

func NewInteractive(s *Screen, opts InteractiveOptions) *Interactive {
	style := NewStyle(s.Color())
	r := &Interactive{
		screen:     s,
		style:      style,
		showStatus: opts.ShowStatus && s.IsTTY(),
		warnPct:    opts.WarnPercent,
		reasonMode: opts.Reasoning,
		verbose:    opts.Verbose,
		steerMark:  opts.SteerPrompt,
		stop:       make(chan struct{}),
		done:       make(chan struct{}),
	}
	if r.steerMark == "" {
		r.steerMark = "> "
	}
	r.md = NewMarkdown(
		func(line string) { r.screen.Commit(line) },
		r.setPartial,
		style, s.Color(), opts.Theme, s.Width,
	)
	return r
}

func (r *Interactive) setPartial(lines []string) {
	r.mu.Lock()
	r.partial = lines
	r.dirty = true
	streaming := r.streaming
	r.mu.Unlock()

	// When no ticker is running there is nothing to coalesce against, so paint
	// immediately rather than leaving the line invisible.
	if !streaming {
		r.redraw()
	}
}

// Start begins the status-line ticker.
//
// The redraw rate is capped deliberately. A fast model emitting hundreds of
// tokens a second would otherwise repaint the status line hundreds of times a
// second, which is precisely the churn that makes a streaming terminal
// impossible to read.
func (r *Interactive) Start() {
	go func() {
		defer close(r.done)
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-r.stop:
				return
			case <-t.C:
				r.mu.Lock()
				active := r.streaming || r.activeTool != "" || r.busy
				if active {
					r.spinnerIndex++
				}
				if r.streaming {
					r.sampleLocked(time.Now())
				}
				paint := active || r.dirty
				r.dirty = false
				r.mu.Unlock()
				if paint {
					r.redraw()
				}
			}
		}
	}()
}

// Control forwards a terminal mode sequence to the screen.
func (r *Interactive) Control(seq string) { r.screen.Control(seq) }

func (r *Interactive) Close() {
	select {
	case <-r.stop:
	default:
		close(r.stop)
	}
	<-r.done
	r.screen.ClearTransient()
	r.screen.Flush()
}

func (r *Interactive) Emit(e agent.Event) {
	// Any event that is not another thinking delta ends the thinking block, so
	// the half-line held back waiting for its newline is committed here --
	// before whatever this event commits, which is the only order in which the
	// scrollback reads as it happened.
	if e.Kind != agent.EvReasoning {
		r.flushReasoning()
	}

	// One consumer rule: the context figure is adopted wherever it arrives and
	// is never computed here. The renderer used to update it on three event
	// kinds out of twelve, so a /compact -- which emits none of them -- left
	// the pre-compaction occupancy on the status line until the next turn
	// ended.
	if e.Context != nil {
		r.mu.Lock()
		r.ctxState = *e.Context
		r.mu.Unlock()
	}

	switch e.Kind {
	case agent.EvContext:
		// Carried above. Nothing is drawn for it on its own: the status line
		// repaints on its own timer, and committing a line per context change
		// would put one in the scrollback for every tool result.
		return
	case agent.EvTurnStart:
		r.mu.Lock()
		r.turnStarted = time.Now()
		r.firstTokenAt = time.Time{}
		r.tokens = 0
		r.samples = r.samples[:0]
		r.streaming = true
		r.sawText = false
		r.reasonTrace.Reset()
		r.mu.Unlock()

	case agent.EvText:
		r.mu.Lock()
		first := !r.sawText
		r.sawText = true
		r.reasoning = ""
		r.countTokenLocked()
		r.mu.Unlock()
		if first {
			r.commitBlankIfNeeded()
		}
		r.md.Write(e.Text)

	case agent.EvReasoning:
		r.handleReasoning(e.Text)

	case agent.EvToolStart:
		if e.ToolName == "" {
			return
		}
		r.md.Flush()
		r.mu.Lock()
		r.activeTool = e.ToolName
		r.toolStarted = time.Now()
		verbose := r.verbose
		r.mu.Unlock()
		if verbose {
			r.commitToolCall(e)
		}
		r.redraw()

	case agent.EvToolEnd:
		r.mu.Lock()
		r.activeTool = ""
		r.mu.Unlock()
		r.commitToolResult(e)

	case agent.EvTurnEnd:
		r.md.Flush()
		r.mu.Lock()
		r.streaming = false
		r.mu.Unlock()
		r.redraw()

	case agent.EvSteer:
		// Into the scrollback, always -- including for a single line. What the
		// user typed mid-turn only ever existed in the transient zone, and the
		// next committed line erases that. Without this the model visibly
		// changes course in response to nothing.
		r.md.Flush()
		r.mu.Lock()
		r.steerQueued = max(0, r.steerQueued-1)
		r.mu.Unlock()
		if label, ok := workerReportLabel(e.Text); ok {
			// A worker's report was committed in full when it arrived, so
			// repeating the body here would print the same findings twice.
			// The line still has to appear: it is the moment the model was
			// given them, and without it the model changes course in
			// response to nothing.
			r.screen.CommitBlock(Echo(r.style, "steering", "worker report: "+label)...)
			return
		}
		r.screen.CommitBlock(Echo(r.style, "steering", e.Text)...)

	case agent.EvCompacted:
		r.md.Flush()
		if c := e.Compaction; c != nil {
			line := fmt.Sprintf(
				"checkpoint  written, covering %d of %d messages. Nothing was removed; "+
					"the session continues from it when the window is short.",
				c.SummarisedThrough, c.MessagesBefore)
			if c.Rewrote {
				line = fmt.Sprintf(
					"compacted  %s → %s tokens, %d messages → %d. The full transcript is still on disk.",
					compactNum(c.TokensBefore), compactNum(c.TokensAfter),
					c.MessagesBefore, c.MessagesAfter)
			}
			r.screen.CommitBlock("", r.style.Dim(line))
		}

	case agent.EvNotice:
		r.md.Flush()
		prefix := r.style.Dim("note")
		if e.Level == agent.LevelWarn {
			prefix = r.style.Warn("warning")
		}
		r.screen.CommitBlock("", prefix+"  "+e.Text)

	case agent.EvError:
		// A failed turn is a finished turn. Without this the renderer stays in
		// its streaming state for the rest of the session: the ticker keeps
		// repainting the transient zone ten times a second, on top of the
		// prompt the line editor has drawn underneath it, and the prompt looks
		// like it is being wiped out as fast as it appears.
		r.md.Flush()
		r.mu.Lock()
		r.streaming = false
		r.activeTool = ""
		r.mu.Unlock()
		r.screen.CommitBlock("", r.style.Error("error")+"  "+e.Text)
		r.screen.ClearTransient()

	case agent.EvDone:
		r.md.Flush()
		r.mu.Lock()
		r.streaming = false
		r.activeTool = ""
		r.mu.Unlock()
		r.screen.ClearTransient()
	}
}

// handleReasoning keeps the thinking channel out of the scrollback.
//
// Reasoning is long, repetitive and rarely worth re-reading. In "collapsed"
// mode it appears only in the transient zone, so the user can watch the model
// think without any of it landing in their scrollback or their clipboard.
// /verbose is the escape hatch from that, and it overrides ui.reasoning: the
// command asks for everything, and a configured default is a default.
func (r *Interactive) handleReasoning(text string) {
	// Counted before the mode check: reasoning tokens are generated at the same
	// rate whether or not they are shown, and on a reasoning model they are
	// most of the response.
	r.mu.Lock()
	r.countTokenLocked()
	verbose := r.verbose
	r.mu.Unlock()

	if verbose {
		r.commitReasoning(text)
		return
	}
	if r.reasonMode == "off" {
		return
	}
	if r.reasonMode == "full" {
		r.md.Write(text)
		return
	}
	r.mu.Lock()
	r.reasonTrace.WriteString(text)
	// Only the tail is ever displayed, so only the marquee keeps the tail --
	// but the trim is by rune, not by byte. Slicing a UTF-8 string at a byte offset lands in
	// the middle of a multi-byte rune about half the time it is used on
	// non-ASCII text, and the orphaned continuation bytes reach the terminal as
	// a replacement character or, on some terminals, as nothing at all while
	// still consuming a column.
	r.reasoning = tailRunes(r.reasoning+text, reasoningKeep)
	r.dirty = true
	r.mu.Unlock()
}

// reasoningKeep is how much of the thinking channel is retained for the
// marquee. It only has to cover the widest line any terminal will show.
const reasoningKeep = 512

func tailRunes(s string, n int) string {
	if len(s) <= n { // bytes >= runes, so this is a safe fast path
		return s
	}
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[len(rs)-n:])
}

func (r *Interactive) commitBlankIfNeeded() {
	r.screen.Commit("")
}

// SetVerbose switches the detail level and reports the level that was in
// effect before, so the caller can say the session was already there.
func (r *Interactive) SetVerbose(v bool) bool {
	// Flushed before the switch, not after: the half-line still buffered was
	// produced under the old level and is only committed under the verbose one.
	r.flushReasoning()

	r.mu.Lock()
	was := r.verbose
	r.verbose = v
	// The marquee and the committed thinking are the same text, so the
	// retained tail goes with the switch. Keeping it would show the thinking
	// twice going into verbose, and would strand a frozen tail in the
	// transient zone for the rest of the turn coming out of it -- verbose
	// never refreshes the marquee, so nothing would ever move it on.
	r.reasoning = ""
	// Turning verbose on part-way through a turn is a request to read the
	// thinking, and most of it has usually already happened -- at a few
	// tokens a second there is time to decide you want it only once it is
	// well underway. Showing just the remainder answers a question nobody
	// asked.
	var caught string
	if v && !was {
		caught = r.reasonTrace.String()
		r.reasonTrace.Reset()
	}
	r.dirty = true
	r.mu.Unlock()

	if lines := strings.Split(strings.TrimRight(caught, "\n"), "\n"); caught != "" {
		r.commitReasonLines(true, lines)
	}

	r.redraw()
	return was
}

// commitReasoning puts the thinking channel into the scrollback a line at a
// time.
//
// Deltas do not arrive on line boundaries, so the tail of one is held back
// until its newline. Committing it early would split a sentence across two
// scrollback lines that can never be rejoined, since a committed line is never
// revised.
func (r *Interactive) commitReasoning(text string) {
	r.mu.Lock()
	r.reasonBuf += text
	cut := strings.LastIndexByte(r.reasonBuf, '\n')
	if cut < 0 {
		r.mu.Unlock()
		return
	}
	done := r.reasonBuf[:cut]
	r.reasonBuf = r.reasonBuf[cut+1:]
	open := r.reasonOpen
	r.reasonOpen = true
	r.mu.Unlock()

	r.commitReasonLines(!open, strings.Split(done, "\n"))
}

// flushReasoning commits whatever thinking is buffered and closes the block.
func (r *Interactive) flushReasoning() {
	r.mu.Lock()
	rest, open := r.reasonBuf, r.reasonOpen
	r.reasonBuf, r.reasonOpen = "", false
	r.mu.Unlock()

	if strings.TrimSpace(rest) == "" {
		return
	}
	r.commitReasonLines(!open, []string{rest})
}

func (r *Interactive) commitReasonLines(header bool, lines []string) {
	var out []string
	if header {
		out = append(out, "", r.style.Dim("thinking"))
	}
	for _, l := range lines {
		// Indented and dimmed rather than prefixed with a glyph: this is model
		// output, and a gutter character would land in any selection of it.
		out = append(out, "  "+r.style.Dim(strings.TrimRight(l, " \t")))
	}
	r.screen.CommitBlock(out...)
}

// commitToolCall shows what the model actually asked for, which the one-line
// summary at the end of the call necessarily leaves out.
func (r *Interactive) commitToolCall(e agent.Event) {
	out := []string{"", r.style.Dim("tool") + "  " + e.ToolName}
	for _, l := range argLines(e.ToolArgs) {
		out = append(out, "  "+r.style.Dim(l))
	}
	r.screen.CommitBlock(out...)
}

// argLines renders a tool's arguments for display, indented if they are JSON
// and verbatim if they are not -- arguments that failed to parse are exactly
// the ones worth seeing unaltered.
func argLines(args []byte) []string {
	if len(args) == 0 {
		return nil
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, args, "", "  "); err != nil {
		return strings.Split(strings.TrimRight(string(args), "\n"), "\n")
	}
	return strings.Split(buf.String(), "\n")
}

func (r *Interactive) commitToolResult(e agent.Event) {
	res := e.ToolResult
	if res == nil {
		return
	}
	display := res.Display
	if display == "" {
		display = e.ToolName
	}

	glyph := r.style.Tool("●")
	if res.IsError {
		glyph = r.style.Error("●")
	}
	// The glyph sits on ai-code's own status line, never on model content or code,
	// so a selection of the output never picks up a gutter.
	r.screen.Commit(glyph + " " + display)

	r.mu.Lock()
	verbose := r.verbose
	r.mu.Unlock()

	switch {
	case verbose:
		// No cap. The tool has already bounded its own output, and a second
		// trim here would make /verbose a claim the renderer does not keep.
		for _, line := range contentLines(res.Content) {
			r.screen.Commit("  " + r.style.Dim(line))
		}
	case res.IsError:
		for _, line := range firstLines(res.Content, 6) {
			r.screen.Commit("  " + r.style.Dim(line))
		}
	}
}

func contentLines(s string) []string {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

func firstLines(s string, n int) []string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > n {
		lines = append(lines[:n], fmt.Sprintf("... %d more lines", len(lines)-n))
	}
	return lines
}

// SetSteering reports what the user has typed during the turn.
//
// While active it takes the bottom line of the transient zone, in place of the
// status line. That is the whole interaction: the status line is what ai-code has
// to say and the prompt is what the user has to say, they occupy the same row,
// and the one that is showing is the one that currently matters. Clearing the
// line gives the row back.
// SetQueued reports how many steering messages are waiting to be folded in.
func (r *Interactive) SetQueued(n int) {
	r.mu.Lock()
	r.steerQueued = n
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

// SetBusy marks a turn as running.
//
// The status line used to be driven entirely by EvTurnStart/EvTurnEnd and the
// tool events, which between them do not cover the whole turn. The gap is
// everything the agent does between requests, and the longest thing in it is
// compaction: a full summarisation call, minutes of it on a local model, with
// no event of its own. Through all of that the spinner stopped and the
// elapsed counter froze, which is exactly what a hung session looks like.
func (r *Interactive) SetBusy(v bool) {
	r.mu.Lock()
	r.busy = v
	if v {
		r.turnStarted = time.Now()
	}
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

// SetNotice puts one line of progress in the transient zone, where it is
// replaced by the next one rather than committed to the scrollback.
//
// For work that is worth watching and not worth keeping: a model swap
// reports several times a second and leaves one summary behind, where
// committing each update would scroll a screen of history for an event with
// a one-line outcome.
func (r *Interactive) SetNotice(text string) {
	r.mu.Lock()
	r.notice = text
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

func (r *Interactive) SetSteering(display string, cursorCol int, active bool) {
	r.mu.Lock()
	r.steerText, r.steerCol, r.steerActive = display, cursorCol, active
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

// boundPreview limits how tall the streaming line is allowed to make the
// transient zone.
//
// The zone is erased by walking the cursor back up over it, so a zone taller
// than the screen is one that cannot be erased: the top of it has already
// scrolled beyond the cursor's reach, and what the erase would clear instead is
// committed output. Half the screen leaves room for the status line and for
// enough context above to read. Past that the tail is shown, because the tail
// is the part still arriving.
func boundPreview(lines []string, height int) []string {
	limit := max(height/2, 1)
	if len(lines) <= limit {
		return lines
	}
	return lines[len(lines)-limit:]
}

// redraw repaints the transient zone: the streaming line, then the status line
// or the steering prompt.
func (r *Interactive) redraw() {
	if !r.screen.IsTTY() {
		return
	}
	width := r.screen.Width()

	r.mu.Lock()
	partial := r.partial
	reasoning := r.reasoning
	notice := r.notice
	status := r.statusLocked()
	steerLine, steerCol := "", -1
	if r.steerActive {
		steerLine, steerCol = r.steerLocked(width - 1)
	}
	r.dirty = false
	r.mu.Unlock()

	var lines []string
	if notice != "" {
		lines = append(lines, r.style.Dim(marquee(notice, width-1)))
	} else if len(partial) > 0 {
		lines = append(lines, boundPreview(partial, r.screen.Height())...)
	} else if reasoning != "" {
		const label = "thinking: "
		lines = append(lines, r.style.Dim(label+marquee(reasoning, width-1-len(label))))
	}
	switch {
	case steerCol >= 0:
		lines = append(lines, steerLine)
	case r.showStatus && status != "":
		lines = append(lines, status)
	}
	r.screen.SetTransientCursor(steerCol, lines...)
}

// steerLocked lays out the steering prompt, scrolling the text horizontally so
// the cursor stays on screen. Scrolling rather than wrapping, because a
// transient line that wraps is one the erase can no longer clean up.
func (r *Interactive) steerLocked(width int) (string, int) {
	markW := visibleWidth(r.steerMark)
	avail := max(width-markW, 8)

	rs := []rune(r.steerText)
	cur := min(max(r.steerCol, 0), len(rs))

	// Walk left from the cursor until the window is full, then fill whatever
	// room is left to the right. Anchoring on the cursor rather than on either
	// end is what keeps it visible in both directions: a long line typed
	// straight through scrolls with the cursor at the margin, and moving back
	// into the middle of it scrolls the other way.
	start, before := cur, 0
	for start > 0 {
		w := runeWidth(rs[start-1])
		if before+w > avail {
			break
		}
		before += w
		start--
	}
	end, total := cur, before
	for end < len(rs) {
		w := runeWidth(rs[end])
		if total+w > avail {
			break
		}
		total += w
		end++
	}
	return r.style.Bold(r.steerMark) + string(rs[start:end]), markW + before
}

// marquee renders a growing text as a single line that scrolls, showing the
// newest end of it.
//
// All whitespace is folded here, not just newlines: the transient zone
// guarantees one physical row per line, and a stray tab or carriage return
// breaks the row count that the next erase depends on. Folding before the
// width arithmetic also keeps it honest, since the tail is then chosen from
// text that is one column per character.
func marquee(s string, width int) string {
	if width <= 0 {
		return ""
	}
	rs := []rune(collapseSpace(s))
	used, i := 0, len(rs)
	for i > 0 {
		w := runeWidth(rs[i-1])
		if used+w > width {
			break
		}
		used += w
		i--
	}
	return string(rs[i:])
}

// collapseSpace folds every run of whitespace into a single space and drops the
// control characters that would otherwise move the cursor.
func collapseSpace(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	space := false
	for _, r := range s {
		switch {
		case r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\v' || r == '\f':
			space = b.Len() > 0
		case r < 0x20 || r == 0x7f:
			// Dropped outright: no width, but plenty of effect.
		default:
			if space {
				b.WriteByte(' ')
				space = false
			}
			b.WriteRune(r)
		}
	}
	return b.String()
}

// countTokenLocked records the arrival of one streamed delta. Caller holds the
// mutex.
//
// One delta is one token, measured rather than assumed: the cassettes in
// internal/provider/testdata match at 120/120 and 709/712. So the live rate
// costs nothing and needs no per-model tokenizer.
//
// A tool call's scaffolding is counted by the server but never streamed, so a
// turn ending in one undercounts by roughly 20 tokens. This drives the rate
// readout only; the context total comes from the server's usage report.
func (r *Interactive) countTokenLocked() {
	if r.firstTokenAt.IsZero() {
		r.firstTokenAt = time.Now()
	}
	r.tokens++
}

// sampleLocked pushes a reading and discards those outside the window, keeping
// the one immediately before it so the window stays full width.
func (r *Interactive) sampleLocked(now time.Time) {
	r.samples = append(r.samples, rateSample{now, r.tokens})
	cut := now.Add(-rateWindow)
	drop := 0
	for drop+1 < len(r.samples) && r.samples[drop+1].at.Before(cut) {
		drop++
	}
	r.samples = append(r.samples[:0], r.samples[drop:]...)
}

// rateLocked returns tokens per second over the trailing window.
func (r *Interactive) rateLocked() (float64, bool) {
	if len(r.samples) < 2 {
		return 0, false
	}
	first, last := r.samples[0], r.samples[len(r.samples)-1]
	dt := last.at.Sub(first.at).Seconds()
	n := last.n - first.n
	if dt < 0.5 || n <= 0 {
		return 0, false
	}
	return float64(n) / dt, true
}

// statusLocked builds the single redrawable line. Caller holds the mutex.
func (r *Interactive) statusLocked() string {
	if !r.streaming && r.activeTool == "" && !r.busy {
		return ""
	}

	var parts []string

	// While the thinking marquee is on screen it is itself the waiting
	// indicator; a spinner beside it is redundant noise. The spinner returns
	// as soon as text starts streaming or a tool runs.
	if r.reasoning == "" {
		sp := spinner[r.spinnerIndex%len(spinner)]
		if r.activeTool != "" {
			parts = append(parts, fmt.Sprintf("%s %s", sp, r.activeTool))
		} else {
			parts = append(parts, sp)
		}
	}

	if !r.turnStarted.IsZero() {
		parts = append(parts, fmt.Sprintf("%ds", int(time.Since(r.turnStarted).Seconds())))
	}
	if r.steerQueued > 0 {
		word := "message"
		if r.steerQueued > 1 {
			word += "s"
		}
		parts = append(parts, fmt.Sprintf("%d steering %s queued", r.steerQueued, word))
	}
	if r.ctxState.Window > 0 {
		// The tilde is the difference between a figure the backend reported
		// and one derived from character counts. Both are useful; showing
		// them identically is not.
		mark := ""
		if !r.ctxState.Anchored {
			mark = "~"
		}
		ctx := fmt.Sprintf("%s%s/%s ctx",
			mark, compactNum(r.ctxState.Projected), compactNum(r.ctxState.Window))
		if r.ctxState.Percent() >= r.warnPct {
			ctx = r.style.Warn(ctx)
		}
		parts = append(parts, ctx)
	}

	switch {
	case r.activeTool != "":
		// No tokens are being generated while a tool runs, so there is no rate
		// to report. Showing the last one would be a number that is true of the
		// past and false of the present.
	case r.streaming && r.firstTokenAt.IsZero():
		// Prompt processing. On a large context this is most of the wait, and a
		// bare spinner gives no clue which of the two phases you are in.
		parts = append(parts, "prefill")
	default:
		if rate, ok := r.rateLocked(); ok {
			parts = append(parts, fmt.Sprintf("%.0f tok/s", rate))
		}
	}

	return r.style.Dim(strings.Join(parts, "  ·  "))
}

func compactNum(n int) string {
	switch {
	case n >= 1000000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	default:
		return fmt.Sprint(n)
	}
}

// workerReportLabel picks the label out of a folded worker report.
//
// It reads the tag that internal/agent writes around a report. A steer that
// is not one is left alone, so an unrecognised shape degrades to showing the
// text rather than to showing nothing.
func workerReportLabel(text string) (string, bool) {
	if !strings.HasPrefix(text, "<worker-report") {
		return "", false
	}
	const key = `label="`
	i := strings.Index(text, key)
	if i < 0 {
		return "", false
	}
	rest := text[i+len(key):]
	j := strings.Index(rest, `"`)
	if j < 0 {
		return "", false
	}
	return rest[:j], true
}
