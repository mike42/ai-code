// Command steerprobe drives the renderer and the steering prompt under a real
// terminal. Test-only; not part of the shipped CLI.
//
// It exists because the two things steering depends on cannot be reached from a
// unit test. Whether the terminal is left able to translate "\n" -- steering
// changes the input side of the termios and must not touch the output side --
// and whether a transient line stays on one row are both properties of a real
// tty, not of the bytes the renderer produces.
//
// Drive it through a pty and replay the escapes:
//
//	( sleep 1; printf 'stop, check the tests'; sleep 1; printf '\r'; sleep 5 ) |
//	    script -q -c "COLUMNS=80 steerprobe" /dev/null
//
// What to look for: the committed code is flush left with no stray fragments of
// the thinking trace between its lines, the thinking line stays on one row while
// carrying tabs and wide characters, and the bottom line alternates between the
// status line and the "> " prompt as the buffer fills and empties.
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
	// Reasoning full of exactly what used to break the transient zone.
	for range 90 {
		r.Emit(agent.Event{Kind: agent.EvReasoning,
			Text: "\tso I should check whether 日本語 handling is right\n\t\treturn nil\r"})
		time.Sleep(30 * time.Millisecond)
	}
	// Then some code, which is where the stray newlines showed up.
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
