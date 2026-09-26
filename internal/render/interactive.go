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

// Interactive is the terminal renderer. It owns the transient zone -- the line
// currently streaming and the status line -- and writes everything else to the
// scrollback, once.
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
	// reasonTrace is everything the model has thought this turn, so /verbose
	// turned on part-way through can show it. Reset each turn.
	reasonTrace strings.Builder
	// busy is set for the whole turn, not just streaming: the agent also works
	// between requests, and the status line has to keep moving. See SetBusy.
	busy         bool
	toolStarted  time.Time
	ctxState     agent.ContextState
	streaming    bool
	spinnerIndex int
	reasoning    string
	// notice is transient progress from outside the turn loop, such as a model swap. See SetNotice.
	notice     string
	reasonMode string
	sawText    bool

	// verbose is the /verbose toggle: full thinking, tool arguments and output,
	// all committed. Read on the agent goroutine, written on the input
	// goroutine, so it lives under the same mutex as the display state.
	verbose bool
	// reasonBuf holds the tail of a thinking delta that has not reached a
	// newline; reasonOpen whether a thinking block is being committed.
	reasonBuf  string
	reasonOpen bool

	// Steering state. steerActive decides whether the bottom line is a prompt
	// or the status line -- not whether steerText is empty, because the two
	// differ for the one keystroke that deletes the last character.
	steerText   string
	steerCol    int
	steerActive bool
	// steerQueued counts messages submitted but not yet folded in. The status
	// line has to say they were received, or Enter looks like it did nothing.
	steerQueued int

	// dirty coalesces transient-zone repaints: deltas set it, and the ticker
	// repaints at a fixed rate rather than once per delta.
	dirty bool

	stop chan struct{}
	done chan struct{}
}

// rateSample is one reading of the running token count; the rate is computed
// over a trailing window, since a cumulative average converges as 1/t after a
// change and reads as generation slowing down.
type rateSample struct {
	at time.Time
	n  int
}

const rateWindow = 5 * time.Second

// spinner uses braille dots: one column in every terminal, so the line width
// does not shift as they animate.
var spinner = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

type InteractiveOptions struct {
	ShowStatus  bool
	Theme       string
	Reasoning   string // off | collapsed | full
	WarnPercent int
	// Verbose is the starting detail level, from -v; /verbose and /quiet move it.
	Verbose bool
	// SteerPrompt is the marker drawn in place of the status line while typing
	// during a turn. Passed in so the program has one definition of it.
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

// Start begins the status-line ticker. The redraw rate is capped so a fast
// model does not churn the terminal with hundreds of repaints a second.
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
	// Any event that is not a thinking delta ends the thinking block: commit
	// the half-line held for its newline before whatever this event commits,
	// so the scrollback order matches arrival.
	if e.Kind != agent.EvReasoning {
		r.flushReasoning()
	}

	// The context figure is adopted wherever it arrives; it is never computed
	// here.
	if e.Context != nil {
		r.mu.Lock()
		r.ctxState = *e.Context
		r.mu.Unlock()
	}

	switch e.Kind {
	case agent.EvContext:
		// Carried above; nothing is drawn for it on its own, or every tool
		// result would put a line in the scrollback.
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
		// Always into the scrollback, even a single line: it only ever existed
		// in the transient zone, which the next committed line erases.
		r.md.Flush()
		r.mu.Lock()
		r.steerQueued = max(0, r.steerQueued-1)
		r.mu.Unlock()
		if label, ok := workerReportLabel(e.Text); ok {
			// The report body was committed in full when it arrived, so only
			// the moment it was given to the model belongs here.
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
		// A failed turn is a finished turn: clear the streaming state, or the
		// ticker keeps repainting over the prompt the line editor has drawn.
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

// handleReasoning keeps the thinking channel out of the scrollback. In
// "collapsed" mode it appears only in the transient zone; /verbose overrides
// ui.reasoning.
func (r *Interactive) handleReasoning(text string) {
	// Counted before the mode check: reasoning tokens are generated whether or
	// not they are shown.
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
	// Only the tail is displayed, but the trim is by rune, not by byte: an
	// orphaned continuation byte reaches the terminal as a replacement
	// character, or as nothing while still consuming a column.
	r.reasoning = tailRunes(r.reasoning+text, reasoningKeep)
	r.dirty = true
	r.mu.Unlock()
}

// reasoningKeep is how much of the thinking channel the marquee retains; it
// covers the widest line any terminal will show.
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

// SetVerbose switches the detail level and reports the previous level.
func (r *Interactive) SetVerbose(v bool) bool {
	// Flushed before the switch: the buffered half-line was produced under the
	// old level and is only committed under the verbose one.
	r.flushReasoning()

	r.mu.Lock()
	was := r.verbose
	r.verbose = v
	// The marquee and the committed thinking are the same text, so the
	// retained tail goes with the switch; otherwise it would show twice going
	// into verbose and freeze in the transient zone coming out.
	r.reasoning = ""
	// Verbose turned on part-way through asks to read the thinking, most of
	// which has already happened; commit the trace, not just the remainder.
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
// time, holding back the tail of a delta until its newline: committed lines
// are never revised or rejoined.
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
		// Indented and dimmed rather than given a glyph: a gutter would land
		// in any selection of model output.
		out = append(out, "  "+r.style.Dim(strings.TrimRight(l, " \t")))
	}
	r.screen.CommitBlock(out...)
}

// commitToolCall shows the arguments, which the call's one-line summary at the
// end leaves out.
func (r *Interactive) commitToolCall(e agent.Event) {
	out := []string{"", r.style.Dim("tool") + "  " + e.ToolName}
	for _, l := range argLines(e.ToolArgs) {
		out = append(out, "  "+r.style.Dim(l))
	}
	r.screen.CommitBlock(out...)
}

// argLines renders a tool's arguments for display, indented if they are JSON
// and verbatim if not: arguments that failed to parse are the ones worth seeing
// unaltered.
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
	// The glyph sits on ai-code's own line, never on model content or code, so
	// a selection of the output never picks up a gutter.
	r.screen.Commit(glyph + " " + display)

	r.mu.Lock()
	verbose := r.verbose
	r.mu.Unlock()

	switch {
	case verbose:
		// No cap: the tool has already bounded its own output.
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

// SetQueued reports how many steering messages are waiting to be folded in.
func (r *Interactive) SetQueued(n int) {
	r.mu.Lock()
	r.steerQueued = n
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

// SetBusy marks a turn as running. It covers the gap between requests, where
// no event marks progress: compaction above all, which can take minutes.
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

// SetNotice puts one line of progress in the transient zone, replaced by the
// next rather than committed: worth watching, not worth keeping.
func (r *Interactive) SetNotice(text string) {
	r.mu.Lock()
	r.notice = text
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

// SetSteering reports the line being typed during the turn. While active it
// takes the bottom line in place of the status line; clearing it gives the row
// back.
func (r *Interactive) SetSteering(display string, cursorCol int, active bool) {
	r.mu.Lock()
	r.steerText, r.steerCol, r.steerActive = display, cursorCol, active
	r.dirty = true
	r.mu.Unlock()
	r.redraw()
}

// boundPreview caps the streaming line at half the screen. The transient zone
// is erased by walking the cursor back up over it, so a taller zone cannot be
// erased; past the cap the tail is shown, since that is what is still arriving.
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

// steerLocked lays out the steering prompt, scrolling horizontally so the
// cursor stays on screen; a wrapped transient line cannot be erased cleanly.
func (r *Interactive) steerLocked(width int) (string, int) {
	markW := visibleWidth(r.steerMark)
	avail := max(width-markW, 8)

	rs := []rune(r.steerText)
	cur := min(max(r.steerCol, 0), len(rs))

	// Walk left from the cursor until the window is full, then fill what is
	// left to the right. Anchoring on the cursor keeps it visible in both
	// directions, whether the line scrolls at the margin or back in the middle.
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

// marquee renders growing text as one scrolling line showing its newest end.
// All whitespace is folded first: the transient zone guarantees one physical
// row per line, and a stray tab or carriage return breaks the erase count.
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

// collapseSpace folds runs of whitespace into one space and drops the control
// characters that would otherwise move the cursor.
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

// countTokenLocked records the arrival of one streamed delta; the caller holds
// the mutex. One delta is one token, so the live rate needs no tokenizer; a
// turn ending in a tool call undercounts by roughly 20.
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

	// The thinking marquee is itself the waiting indicator; the spinner
	// returns once text streams or a tool runs.
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
		// ~ marks a figure derived from character counts rather than reported
		// by the backend.
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
		// No tokens are generated while a tool runs, so there is no rate to
		// report.
	case r.streaming && r.firstTokenAt.IsZero():
		// Prompt processing, which on a large context is most of the wait.
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

// workerReportLabel reads the tag internal/agent writes around a folded worker
// report; an unrecognised shape degrades to showing the text.
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
