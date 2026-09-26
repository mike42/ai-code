//go:build unix

package main

import (
	"os"
	"os/signal"
	"syscall"

	"ai-code/internal/render"
)

// watchResize repaints when the terminal changes size.
func watchResize(s *render.Screen) func() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGWINCH)
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-ch:
				s.Resize()
			case <-done:
				return
			}
		}
	}()
	return func() {
		signal.Stop(ch)
		close(done)
	}
}
