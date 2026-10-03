// Command ai-code-worker serves ai-code's tools on stdin/stdout. ai-code starts it.
package main

import (
	"fmt"
	"os"

	"ai-code/internal/worker"
)

func main() {
	cwd, err := os.Getwd()
	if err == nil {
		err = worker.Serve(os.Stdin, os.Stdout, cwd)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai-code-worker:", err)
		os.Exit(1)
	}
}
