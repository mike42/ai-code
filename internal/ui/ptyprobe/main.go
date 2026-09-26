// Command ptyprobe drives the line editor under a real terminal so the escape
// sequences it writes can be inspected. Test-only; not part of the shipped CLI.
package main

import (
	"fmt"
	"os"

	"ai-code/internal/ui"
)

func main() {
	e := ui.NewEditor(os.Stdin, os.Stdout)
	e.AddHistory("go build ./...")
	e.AddHistory("git status")
	// The idle checkpoint hangs off this, and it is the one part of the read
	// loop that cannot be reached without a terminal. Printing it here is how
	// "a keystroke aborts with no perceptible delay" gets checked at all.
	e.OnActivity = func() { fmt.Print("[activity]") }
	for {
		line, err := e.ReadLine("> ")
		if err != nil {
			fmt.Printf("\r\nERR %v\r\n", err)
			return
		}
		fmt.Printf("GOT[%s]\r\n", line)
	}
}
