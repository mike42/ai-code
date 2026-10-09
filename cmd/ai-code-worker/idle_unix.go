//go:build unix

package main

import (
	"os"
	"os/signal"
	"syscall"
)

// idle runs until SIGTERM or an interrupt. As the container's PID 1 it adopts
// every process orphaned inside it -- a shell's `cmd &`, a daemon that forks
// away -- and must reap them: a zombie keeps its slot in the container's
// process limit, and once that is full nothing in the container can fork.
func idle() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGTERM, os.Interrupt, syscall.SIGCHLD)
	reapUntilStopped(ch)
}

// reapUntilStopped reaps before each wait, so orphans that exited before the
// first SIGCHLD was caught are collected too; SIGCHLDs coalesce, so each pass
// collects everything that has exited.
func reapUntilStopped(ch <-chan os.Signal) {
	for {
		reapExited()
		if <-ch != syscall.SIGCHLD {
			return
		}
	}
}

func reapExited() {
	for {
		var ws syscall.WaitStatus
		pid, err := syscall.Wait4(-1, &ws, syscall.WNOHANG, nil)
		if err == syscall.EINTR {
			continue
		}
		if err != nil || pid <= 0 {
			return
		}
	}
}
