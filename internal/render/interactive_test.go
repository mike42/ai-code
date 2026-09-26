package render

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

func newRateTester() *Interactive {
	return &Interactive{streaming: true, showStatus: true, style: Style{}}
}

func TestRateDoesNotCarryOverFromThePreviousRequest(t *testing.T) {
	r := newRateTester()
	base := time.Now()

	// A finished request that produced plenty of tokens.
	r.firstTokenAt = base
	r.tokens = 800
	r.sampleLocked(base.Add(4 * time.Second))

	// The next one begins. Nothing has been generated yet.
	r.Emit(agent.Event{Kind: agent.EvTurnStart})

	if r.tokens != 0 {
		t.Errorf("token count = %d at the start of a request, want 0", r.tokens)
	}
	if len(r.samples) != 0 {
		t.Errorf("%d samples survived into the next request, want 0", len(r.samples))
	}
	if _, ok := r.rateLocked(); ok {
		t.Error("a rate was reported before a single token had arrived")
	}
	if got := r.statusLocked(); strings.Contains(got, "tok/s") {
		t.Errorf("status = %q, want no rate before the first token", got)
	}
}

// A cumulative average converges as 1/t and sinks while output keeps flowing,
// so the reading tracks a trailing window instead.
func TestRateTracksTheCurrentSpeedNotTheAverage(t *testing.T) {
	r := newRateTester()
	base := time.Now()
	r.firstTokenAt = base

	// Ten seconds at 200 tok/s.
	for i := 1; i <= 10; i++ {
		r.tokens = i * 200
		r.sampleLocked(base.Add(time.Duration(i) * time.Second))
	}
	fast, ok := r.rateLocked()
	if !ok {
		t.Fatal("no rate after ten seconds of streaming")
	}
	if fast < 180 || fast > 220 {
		t.Errorf("rate = %.0f, want about 200", fast)
	}

	// It drops to 20 tok/s. Within the window the reading must follow it down
	// to roughly the new speed, not linger near the old average.
	for i := 11; i <= 20; i++ {
		r.tokens += 20
		r.sampleLocked(base.Add(time.Duration(i) * time.Second))
	}
	slow, _ := r.rateLocked()
	if slow < 15 || slow > 25 {
		t.Errorf("rate = %.0f after slowing to 20 tok/s, want about 20", slow)
	}
}

// Prefill can be most of the wait on a long context, so it is shown separately
// and excluded from the rate.
func TestPrefillIsShownAndExcludedFromTheRate(t *testing.T) {
	r := newRateTester()
	r.turnStarted = time.Now().Add(-30 * time.Second)

	if got := r.statusLocked(); !strings.Contains(got, "prefill") {
		t.Errorf("status = %q, want it to say the request is still in prefill", got)
	}

	// First token arrives after a long wait; the rate clock starts here.
	r.countTokenLocked()
	first := r.firstTokenAt
	if first.Before(r.turnStarted.Add(29 * time.Second)) {
		t.Error("the rate clock started before the first token")
	}

	for i := 1; i <= 10; i++ {
		r.tokens = i * 100
		r.sampleLocked(first.Add(time.Duration(i) * 100 * time.Millisecond))
	}
	rate, ok := r.rateLocked()
	if !ok {
		t.Fatal("no rate reported")
	}
	if rate < 900 || rate > 1100 {
		t.Errorf("rate = %.0f, want about 1000 -- the 30s of prefill must not be in it", rate)
	}
}

// No tokens arrive while a tool runs, so a leftover rate would state the past
// as if it were the present.
func TestNoRateIsShownWhileAToolRuns(t *testing.T) {
	r := newRateTester()
	base := time.Now()
	r.firstTokenAt = base
	for i := 1; i <= 10; i++ {
		r.tokens = i * 100
		r.sampleLocked(base.Add(time.Duration(i) * 100 * time.Millisecond))
	}
	if got := r.statusLocked(); !strings.Contains(got, "tok/s") {
		t.Fatalf("status = %q, want a rate while streaming", got)
	}

	r.activeTool = "bash"
	if got := r.statusLocked(); strings.Contains(got, "tok/s") {
		t.Errorf("status = %q, want no rate while a tool is running", got)
	}
}

// Reasoning tokens cost the same time whether or not they are displayed, so
// they count towards the rate; otherwise it reads near zero through a long
// think.
func TestReasoningDeltasCountTowardsTheRate(t *testing.T) {
	for _, mode := range []string{"off", "collapsed", "full"} {
		t.Run(mode, func(t *testing.T) {
			r := newRateTester()
			r.reasonMode = mode
			r.md = NewMarkdown(func(string) {}, func([]string) {}, Style{}, false, "", func() int { return 80 })

			for range 50 {
				r.Emit(agent.Event{Kind: agent.EvReasoning, Text: "x"})
			}
			if r.tokens != 50 {
				t.Errorf("counted %d reasoning tokens in mode %q, want 50", r.tokens, mode)
			}
		})
	}
}

// The window must not grow without bound over a long generation.
func TestRateWindowStaysBounded(t *testing.T) {
	r := newRateTester()
	base := time.Now()
	for i := range 6000 { // ten minutes at the 10Hz tick
		r.tokens = i
		r.sampleLocked(base.Add(time.Duration(i) * 100 * time.Millisecond))
	}
	if len(r.samples) > 60 {
		t.Errorf("%d samples retained, want the trailing window only (~50)", len(r.samples))
	}
}

// newVerbosityTester wires a renderer to a buffer standing in for the
// scrollback. The ticker is not started, so every repaint here is the one the
// event caused.
func newVerbosityTester(t *testing.T, tty bool) (*Interactive, *Screen, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	s := NewScreen(&buf, "never")
	s.isTTY, s.width, s.height = tty, 80, 24
	r := NewInteractive(s, InteractiveOptions{Reasoning: "collapsed", SteerPrompt: "> "})
	return r, s, &buf
}

func think(r *Interactive, text string) {
	r.Emit(agent.Event{Kind: agent.EvReasoning, Text: text})
}

// In the default mode none of the reasoning may reach the scrollback, since a
// selection or a copy picks that up.
func TestQuietModeKeepsThinkingOutOfTheScrollback(t *testing.T) {
	r, _, buf := newVerbosityTester(t, false)

	think(r, "I should read the loop first.\nThen the renderer.\n")
	if r.reasoning == "" {
		t.Error("the marquee was not fed in quiet mode")
	}

	// Checked after the answer starts too: the thinking must not arrive late,
	// on the event that ends the block.
	r.Emit(agent.Event{Kind: agent.EvText, Text: "Done.\n"})
	if got := Strip(buf.String()); strings.Contains(got, "read the loop") {
		t.Errorf("thinking reached the scrollback in quiet mode: %q", got)
	}
}

func TestVerboseModeCommitsThinkingToTheScrollback(t *testing.T) {
	r, _, buf := newVerbosityTester(t, false)
	r.SetVerbose(true)

	think(r, "I should read the loop first.\nThen the ")
	think(r, "renderer.") // no trailing newline: the tail must still arrive
	r.Emit(agent.Event{Kind: agent.EvToolStart, ToolName: "read"})

	got := Strip(buf.String())
	for _, want := range []string{"thinking", "I should read the loop first.", "Then the renderer."} {
		if !strings.Contains(got, want) {
			t.Errorf("committed output = %q, want it to contain %q", got, want)
		}
	}
}

// Showing the marquee and the committed thinking would render the same words
// twice.
func TestVerboseThinkingIsNotAlsoInTheMarquee(t *testing.T) {
	r, s, buf := newVerbosityTester(t, true)
	r.SetVerbose(true)

	think(r, "weighing two approaches\n")
	if r.reasoning != "" {
		t.Errorf("marquee text = %q, want none while thinking is being committed", r.reasoning)
	}

	buf.Reset()
	r.redraw()
	if got := Strip(buf.String()); strings.Contains(got, "thinking:") {
		t.Errorf("transient zone = %q, want no marquee in verbose mode", got)
	}
	if s.transient > 1 {
		t.Errorf("transient zone is %d rows, want at most 1", s.transient)
	}
}

func TestToolArgumentsAppearOnlyInVerbose(t *testing.T) {
	args := []byte(`{"command":"go test ./...","description":"Run the tests"}`)

	r, _, buf := newVerbosityTester(t, false)
	r.Emit(agent.Event{Kind: agent.EvToolStart, ToolName: "bash", ToolArgs: args})
	if got := Strip(buf.String()); strings.Contains(got, "go test") {
		t.Errorf("quiet mode showed the tool arguments: %q", got)
	}

	r, _, buf = newVerbosityTester(t, false)
	r.SetVerbose(true)
	r.Emit(agent.Event{Kind: agent.EvToolStart, ToolName: "bash", ToolArgs: args})
	got := Strip(buf.String())
	for _, want := range []string{"bash", "go test ./..."} {
		if !strings.Contains(got, want) {
			t.Errorf("verbose tool call = %q, want it to contain %q", got, want)
		}
	}
}

func TestFullToolOutputIsShownOnSuccessOnlyInVerbose(t *testing.T) {
	content := strings.Join([]string{"line one", "line two", "line three"}, "\n")
	end := agent.Event{Kind: agent.EvToolEnd, ToolName: "read", ToolResult: &tool.Result{
		Display: "Read main.go (3 lines)",
		Content: content,
	}}

	r, _, buf := newVerbosityTester(t, false)
	r.Emit(end)
	got := Strip(buf.String())
	if !strings.Contains(got, "Read main.go (3 lines)") {
		t.Errorf("quiet mode = %q, want the one-line summary", got)
	}
	if strings.Contains(got, "line two") {
		t.Errorf("quiet mode printed the tool output on success: %q", got)
	}

	r, _, buf = newVerbosityTester(t, false)
	r.SetVerbose(true)
	r.Emit(end)
	got = Strip(buf.String())
	for _, want := range []string{"Read main.go (3 lines)", "line one", "line two", "line three"} {
		if !strings.Contains(got, want) {
			t.Errorf("verbose mode = %q, want it to contain %q", got, want)
		}
	}
}

// Quiet mode caps error output at six lines; /verbose must not move that cap.
func TestErrorOutputIsUnchangedInQuietModeAndFullInVerbose(t *testing.T) {
	var lines []string
	for i := 1; i <= 20; i++ {
		lines = append(lines, fmt.Sprintf("error line %d", i))
	}
	end := agent.Event{Kind: agent.EvToolEnd, ToolName: "bash", ToolResult: &tool.Result{
		Display: "bash failed", Content: strings.Join(lines, "\n"), IsError: true,
	}}

	r, _, buf := newVerbosityTester(t, false)
	r.Emit(end)
	got := Strip(buf.String())
	if !strings.Contains(got, "error line 6") || strings.Contains(got, "error line 7") {
		t.Errorf("quiet error output = %q, want the first six lines only", got)
	}
	if !strings.Contains(got, "... 14 more lines") {
		t.Errorf("quiet error output = %q, want it to say how much was elided", got)
	}

	r, _, buf = newVerbosityTester(t, false)
	r.SetVerbose(true)
	r.Emit(end)
	got = Strip(buf.String())
	if !strings.Contains(got, "error line 20") {
		t.Errorf("verbose error output = %q, want every line", got)
	}
	if strings.Contains(got, "more lines") {
		t.Errorf("verbose error output was still elided: %q", got)
	}
}

// Toggling mid-turn: what has already happened stays as it was rendered, and
// the next event is rendered at the new level.
func TestTogglingVerbosityTakesEffectOnTheNextEvent(t *testing.T) {
	r, s, buf := newVerbosityTester(t, true)

	think(r, "quiet thought\n")
	if got := Strip(buf.String()); strings.Contains(got, "quiet thought") {
		t.Fatalf("thinking was committed before /verbose: %q", got)
	}

	if was := r.SetVerbose(true); was {
		t.Error("SetVerbose reported the session was already verbose")
	}
	think(r, "loud thought\n")
	if got := Strip(buf.String()); !strings.Contains(got, "loud thought") {
		t.Errorf("committed output = %q, want the thinking after /verbose", got)
	}

	if was := r.SetVerbose(false); !was {
		t.Error("SetVerbose did not report the previous verbose state")
	}
	buf.Reset()
	think(r, "quiet again\n")
	if got := Strip(buf.String()); strings.Contains(got, "quiet again") {
		t.Errorf("thinking was still committed after /quiet: %q", got)
	}

	// The transient zone must survive the round trip: a row drawn and not counted
	// is a row the next commit cannot erase.
	r.redraw()
	if s.transient > 1 {
		t.Errorf("transient zone is %d rows after toggling, want at most 1", s.transient)
	}
}

// A half-written thinking line that is buffered when the mode changes must not
// surface later, under a mode that does not show thinking at all.
func TestASwitchToQuietFlushesTheBufferedThinkingLine(t *testing.T) {
	r, _, buf := newVerbosityTester(t, false)
	r.SetVerbose(true)
	think(r, "a line with no newline yet")

	r.SetVerbose(false)
	if got := Strip(buf.String()); !strings.Contains(got, "a line with no newline yet") {
		t.Errorf("committed output = %q, want the buffered line flushed by the switch", got)
	}

	buf.Reset()
	r.Emit(agent.Event{Kind: agent.EvText, Text: "hello\n"})
	if got := Strip(buf.String()); strings.Contains(got, "no newline yet") {
		t.Errorf("the buffered thinking line resurfaced in quiet mode: %q", got)
	}
}

// The status line must keep moving for the whole turn, including between
// requests, where compaction runs.
func TestStatusKeepsMovingBetweenRequests(t *testing.T) {
	r, _, _ := newVerbosityTester(t, true)
	r.showStatus = true

	r.SetBusy(true)
	// A request runs and finishes; the turn has not.
	r.Emit(agent.Event{Kind: agent.EvTurnStart})
	r.Emit(agent.Event{Kind: agent.EvTurnEnd, Usage: &provider.Usage{}})

	r.mu.Lock()
	between := r.statusLocked()
	active := r.streaming || r.activeTool != "" || r.busy
	r.mu.Unlock()

	if between == "" {
		t.Error("the status line went blank between requests, which is where compaction runs")
	}
	if !active {
		t.Error("the ticker would not repaint between requests, so the spinner freezes")
	}

	// The turn really ending does put it away.
	r.SetBusy(false)
	r.mu.Lock()
	after := r.statusLocked()
	r.mu.Unlock()
	if after != "" {
		t.Errorf("status line = %q after the turn ended, want it gone", after)
	}
}

// /verbose mid-turn must show the reasoning already generated, not only the
// retained marquee tail.
func TestVerboseMidTurnShowsTheThinkingAlreadyDone(t *testing.T) {
	r, _, buf := newVerbosityTester(t, false)

	r.Emit(agent.Event{Kind: agent.EvTurnStart})
	think(r, "First I will check the parser.\n")
	think(r, "The tokeniser looks wrong.\n")
	// Past the marquee's retained tail, so this cannot pass by accident.
	think(r, strings.Repeat("filler reasoning that pushes the tail along.\n", 40))
	think(r, "Now the conclusion.\n")

	if got := buf.String(); strings.Contains(got, "First I will check the parser") {
		t.Fatal("quiet mode committed thinking to the scrollback")
	}

	r.SetVerbose(true)

	got := buf.String()
	for _, want := range []string{
		"First I will check the parser",
		"The tokeniser looks wrong",
		"Now the conclusion",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("switching to verbose did not show %q:\n%s", want, got)
		}
	}

	// The next turn starts clean, or every dump repeats the last one.
	r.Emit(agent.Event{Kind: agent.EvTurnStart})
	r.mu.Lock()
	left := r.reasonTrace.Len()
	r.mu.Unlock()
	if left != 0 {
		t.Errorf("%d bytes of the previous turn's thinking carried over", left)
	}
}
