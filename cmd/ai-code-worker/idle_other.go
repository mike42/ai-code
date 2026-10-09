//go:build !unix

package main

import (
	"os"
	"os/signal"
	"syscall"
)

// idle runs until SIGTERM or an interrupt. Windows has no zombies and does not
// hand orphans to another process, so there is nothing to reap.
func idle() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGTERM, os.Interrupt)
	<-ch
}
