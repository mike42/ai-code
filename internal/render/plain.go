package render

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"ai-code/internal/agent"
)

// Plain renders to a non-terminal destination: a pipe, a file, a CI log. It is
// a first-class consumer of the same event stream, not a degraded mode; nothing
// here moves a cursor or emits an escape sequence.
type Plain struct {
	mu      sync.Mutex
	w       io.Writer
	verbose bool
	atStart bool
	// inReason tracks whether the thinking channel is mid-block, so its header
	// is written once.
	inReason bool
}

func NewPlain(w io.Writer, verbose bool) *Plain {
	return &Plain{w: w, verbose: verbose, atStart: true}
}

// SetVerbose switches the detail level and reports the level that was in
// effect before. /verbose reaches this renderer when stdout is redirected but
// the session is still being driven from a keyboard.
func (p *Plain) SetVerbose(v bool) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	was := p.verbose
	p.verbose = v
	return was
}

func (p *Plain) Emit(e agent.Event) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if e.Kind != agent.EvReasoning && p.inReason {
		p.newlineIfNeeded()
		p.inReason = false
	}

	switch e.Kind {
	case agent.EvReasoning:
		if !p.verbose {
			return
		}
		if !p.inReason {
			p.newlineIfNeeded()
			fmt.Fprintln(p.w, "[thinking]")
			p.inReason = true
		}
		fmt.Fprint(p.w, e.Text)
		p.atStart = strings.HasSuffix(e.Text, "\n")

	case agent.EvText:
		fmt.Fprint(p.w, e.Text)
		p.atStart = strings.HasSuffix(e.Text, "\n")

	case agent.EvToolStart:
		if e.ToolName == "" {
			return
		}
		p.newlineIfNeeded()
		if p.verbose {
			fmt.Fprintf(p.w, "[tool] %s %s\n", e.ToolName, string(e.ToolArgs))
		}

	case agent.EvToolEnd:
		if e.ToolResult == nil {
			return
		}
		p.newlineIfNeeded()
		display := e.ToolResult.Display
		if display == "" {
			display = e.ToolName
		}
		fmt.Fprintf(p.w, "[tool] %s\n", display)
		if show := strings.TrimRight(e.ToolResult.Show, "\n"); show != "" {
			for _, l := range strings.Split(show, "\n") {
				fmt.Fprintf(p.w, "       %s\n", l)
			}
		}
		if e.ToolResult.IsError || p.verbose {
			for _, l := range strings.Split(strings.TrimRight(e.ToolResult.Content, "\n"), "\n") {
				fmt.Fprintf(p.w, "       %s\n", l)
			}
		}

	case agent.EvSteer:
		p.newlineIfNeeded()
		fmt.Fprintf(p.w, "[steering] %s\n", e.Text)

	case agent.EvCompacted:
		p.newlineIfNeeded()
		if c := e.Compaction; c != nil {
			if c.Rewrote {
				fmt.Fprintf(p.w, "[compacted] %d -> %d tokens, %d -> %d messages\n",
					c.TokensBefore, c.TokensAfter, c.MessagesBefore, c.MessagesAfter)
			} else {
				fmt.Fprintf(p.w, "[checkpoint] covering %d of %d messages; nothing removed\n",
					c.SummarisedThrough, c.MessagesBefore)
			}
		}

	case agent.EvNotice:
		p.newlineIfNeeded()
		fmt.Fprintf(p.w, "[%s] %s\n", e.Level, e.Text)

	case agent.EvError:
		p.newlineIfNeeded()
		fmt.Fprintf(p.w, "[error] %s\n", e.Text)

	case agent.EvDone:
		p.newlineIfNeeded()
	}
}

func (p *Plain) newlineIfNeeded() {
	if !p.atStart {
		fmt.Fprintln(p.w)
		p.atStart = true
	}
}

// JSON emits one event per line, for scripting and for tests that want to
// assert on the event stream rather than on rendered text.
type JSON struct {
	mu  sync.Mutex
	enc *json.Encoder
}

func NewJSON(w io.Writer) *JSON {
	return &JSON{enc: json.NewEncoder(w)}
}

func (j *JSON) Emit(e agent.Event) {
	j.mu.Lock()
	defer j.mu.Unlock()
	_ = j.enc.Encode(e)
}
