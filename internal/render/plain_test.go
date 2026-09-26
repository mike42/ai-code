package render

import (
	"bytes"
	"strings"
	"testing"

	"ai-code/internal/agent"
)

func plainEvents(p *Plain) {
	p.Emit(agent.Event{Kind: agent.EvReasoning, Text: "weighing it up\n"})
	p.Emit(agent.Event{Kind: agent.EvToolStart, ToolName: "bash",
		ToolArgs: []byte(`{"command":"go test ./..."}`)})
	p.Emit(agent.Event{Kind: agent.EvText, Text: "done\n"})
}

// -v and /verbose have to mean the same thing in both renderers, or the flag
// and the command are two features wearing one name.
func TestPlainVerboseShowsThinkingAndArguments(t *testing.T) {
	var buf bytes.Buffer
	plainEvents(NewPlain(&buf, true))

	for _, want := range []string{"[thinking]", "weighing it up", "go test ./..."} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("verbose output = %q, want it to contain %q", buf.String(), want)
		}
	}
}

func TestPlainQuietDropsThinkingAndArguments(t *testing.T) {
	var buf bytes.Buffer
	plainEvents(NewPlain(&buf, false))

	for _, unwanted := range []string{"thinking", "weighing it up", "go test"} {
		if strings.Contains(buf.String(), unwanted) {
			t.Errorf("quiet output = %q, want no %q", buf.String(), unwanted)
		}
	}
}

func TestPlainVerbosityIsSwitchable(t *testing.T) {
	var buf bytes.Buffer
	p := NewPlain(&buf, false)
	if was := p.SetVerbose(true); was {
		t.Error("SetVerbose reported the renderer was already verbose")
	}
	plainEvents(p)
	if !strings.Contains(buf.String(), "weighing it up") {
		t.Errorf("output = %q, want thinking after the switch", buf.String())
	}
}
