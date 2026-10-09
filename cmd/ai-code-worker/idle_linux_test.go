package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// A background job outlives the shell that started it and is handed to the
// container's PID 1. Unreaped, it stays a zombie holding a slot in the
// container's process limit until fork fails for everything in it.
func TestTheIdleWorkerReapsOrphans(t *testing.T) {
	if os.Getenv("AI_CODE_IDLE_HELPER") == "1" {
		orphanThenIdle()
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestTheIdleWorkerReapsOrphans$")
	cmd.Env = append(os.Environ(), "AI_CODE_IDLE_HELPER=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
}

// orphanThenIdle runs in its own process, which stands in for PID 1 by
// becoming a subreaper; an orphan's exit status is never collected by Go.
func orphanThenIdle() {
	fail := func(f string, a ...any) {
		os.Stderr.WriteString(strings.TrimSpace(fmt.Sprintf(f, a...)) + "\n")
		os.Exit(1)
	}
	const prSetChildSubreaper = 36
	if _, _, e := syscall.RawSyscall(syscall.SYS_PRCTL, prSetChildSubreaper, 1, 0); e != 0 {
		err := error(e)
		fail("becoming a subreaper: %v", err)
	}
	out, err := exec.Command("sh", "-c", "sleep 0.1 >/dev/null 2>&1 & echo $!").Output()
	if err != nil {
		fail("starting the orphan: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		fail("reading the orphan's pid from %q: %v", out, err)
	}
	for state(pid) != "Z" {
		if state(pid) == "" {
			fail("orphan %d vanished before it was reaped; it was not ours", pid)
		}
		time.Sleep(10 * time.Millisecond)
	}

	ch := make(chan os.Signal, 1)
	ch <- syscall.SIGTERM
	reapUntilStopped(ch)

	if s := state(pid); s != "" {
		fail("orphan %d is still present in state %s after idle returned", pid, s)
	}
}

func state(pid int) string {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return ""
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 || i+2 >= len(s) {
		return ""
	}
	return s[i+2 : i+3]
}
