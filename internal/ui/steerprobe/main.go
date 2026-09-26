// Command steerprobe drives the renderer and the steering prompt under a real
// terminal, checking that OPOST survives and that a transient line stays on one
// row. Test-only; not part of the shipped CLI.
package main

import (
	"fmt"
	"os"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/render"
	"ai-code/internal/ui"
)

func main() {
	screen := render.NewScreen(os.Stdout, "always")
	r := render.NewInteractive(screen, render.InteractiveOptions{
		ShowStatus: true, Reasoning: "collapsed", WarnPercent: 80, SteerPrompt: "> ",
	})
	r.Start()

	st := ui.NewSteerer(int(os.Stdin.Fd()))
	var got []string
	st.Render = r.SetSteering
	st.Submit = func(s string) { got = append(got, s) }
	st.Interrupt = func() { got = append(got, "<INTERRUPT>") }
	st.Control = r.Control
	if err := st.Start(); err != nil {
		fmt.Printf("steer start: %v\n", err)
		return
	}

	r.Emit(agent.Event{Kind: agent.EvTurnStart})
	// Reasoning carrying tabs, wide characters and carriage returns.
	for range 90 {
		r.Emit(agent.Event{Kind: agent.EvReasoning,
			Text: "\tso I should check whether 日本語 handling is right\n\t\treturn nil\r"})
		time.Sleep(30 * time.Millisecond)
	}
	// Then some code, which is where stray newlines appear.
	for _, chunk := range []string{
		"Here is the fix.\n\n```go\n",
		"func main() {\n\tprintln(\"hello\")\n",
		"\tfor i := range 10 {\n\t\tprintln(i)\n\t}\n}\n```\n",
		"That should do it.\n",
	} {
		r.Emit(agent.Event{Kind: agent.EvText, Text: chunk})
		time.Sleep(400 * time.Millisecond)
	}
	r.Emit(agent.Event{Kind: agent.EvDone})

	left := st.Stop()
	r.SetSteering("", 0, false)
	r.Close()

	fmt.Printf("SUBMITTED=%q LEFTOVER=%q\n", got, left)
}
