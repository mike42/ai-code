package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"ai-code/internal/tool"

	"golang.org/x/term"
)

const (
	exitInterrupted = 2
	exitKilled      = 3
	exitUnavailable = 4
)

type exitStatus struct {
	code int
	msg  string
}

func (e *exitStatus) Error() string { return e.msg }

type timeLimits struct {
	steerAfter      time.Duration
	steerPrompt     string
	interruptAfter  time.Duration
	interruptPrompt string
	exitAfter       time.Duration
}

const defaultSteerPrompt = "Time for this run is nearly up. Finish the step you are on, " +
	"leave the work in a state someone else can pick up, say where it stands, and stop."

func (l timeLimits) any() bool {
	return l.steerAfter > 0 || l.interruptAfter > 0 || l.exitAfter > 0
}

func (l timeLimits) check(print bool) error {
	if !l.any() {
		if l.steerPrompt != "" || l.interruptPrompt != "" {
			return fmt.Errorf("--steer-prompt and --interrupt-prompt need --steer-after and --interrupt-after")
		}
		return nil
	}
	if !print {
		return fmt.Errorf("--steer-after, --interrupt-after and --exit-after are for unattended runs and need -p")
	}
	if l.steerPrompt != "" && l.steerAfter == 0 {
		return fmt.Errorf("--steer-prompt needs --steer-after")
	}
	if l.interruptAfter > 0 && l.interruptPrompt == "" {
		return fmt.Errorf("--interrupt-after needs --interrupt-prompt: what to do in the time left is " +
			"particular to the job, so there is no default")
	}
	if l.interruptPrompt != "" && l.interruptAfter == 0 {
		return fmt.Errorf("--interrupt-prompt needs --interrupt-after")
	}
	order := []struct {
		name string
		d    time.Duration
	}{{"--steer-after", l.steerAfter}, {"--interrupt-after", l.interruptAfter}, {"--exit-after", l.exitAfter}}
	prev, prevName := time.Duration(0), ""
	for _, o := range order {
		if o.d == 0 {
			continue
		}
		if prev > 0 && o.d <= prev {
			return fmt.Errorf("%s (%s) must come after %s (%s)", o.name, o.d, prevName, prev)
		}
		prev, prevName = o.d, o.name
	}
	return nil
}

func parseLimit(flag, v string) (time.Duration, error) {
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("%s takes a duration such as 45m or 1h30m, not %q", flag, v)
	}
	return d, nil
}

type limitTimers struct {
	app    *App
	limits timeLimits
	timers []*time.Timer

	mu          sync.Mutex
	interrupted bool
}

func (a *App) startLimits(l timeLimits, started time.Time, restoreTerm func()) *limitTimers {
	t := &limitTimers{app: a, limits: l}
	after := func(d time.Duration, fn func()) {
		if d > 0 {
			t.timers = append(t.timers, time.AfterFunc(time.Until(started.Add(d)), fn))
		}
	}
	after(l.steerAfter, func() {
		prompt := l.steerPrompt
		if prompt == "" {
			prompt = defaultSteerPrompt
		}
		a.note(fmt.Sprintf("--steer-after %s reached; asking the model to wrap up.", l.steerAfter))
		a.agent.Steer(prompt)
	})
	after(l.interruptAfter, func() {
		t.mu.Lock()
		t.interrupted = true
		t.mu.Unlock()
		a.note(fmt.Sprintf("--interrupt-after %s reached; stopping the turn.", l.interruptAfter))
		a.cancelTurn()
		if a.workers != nil {
			a.workers.Cancel()
		}
	})
	after(l.exitAfter, func() {
		restoreTerm()
		fmt.Fprintf(os.Stderr, "\nai-code: --exit-after %s reached; stopped without wrapping up\n", l.exitAfter)
		os.Exit(exitKilled)
	})
	return t
}

func (t *limitTimers) stop() {
	for _, tm := range t.timers {
		tm.Stop()
	}
}

func (t *limitTimers) wasInterrupted() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.interrupted
}

// os.Exit skips the deferred cleanup, so a hard exit repeats it: erase the
// status line, restore the input mode, bracketed paste off, cursor on.
func saveTerminal(clearStatus func()) func() {
	fd := int(os.Stdin.Fd())
	var state *term.State
	if term.IsTerminal(fd) {
		state, _ = term.GetState(fd)
	}
	return func() {
		clearStatus()
		if state != nil {
			_ = term.Restore(fd, state)
			fmt.Fprint(os.Stdout, "\x1b[?2004l\x1b[?25h")
		}
	}
}

func (a *App) runPrint(ctx context.Context, opening string, lim *limitTimers) error {
	err := a.printTurns(ctx, opening, lim)
	var unavailable *tool.Unavailable
	if errors.As(err, &unavailable) {
		return &exitStatus{code: exitUnavailable, msg: unavailable.Error()}
	}
	return err
}

func (a *App) printTurns(ctx context.Context, opening string, lim *limitTimers) error {
	a.agentMu.Lock()
	defer a.agentMu.Unlock()
	err := a.runTurn(ctx, opening)
	if err == nil && !lim.wasInterrupted() {
		err = a.drainWorkers(ctx)
	}
	if !lim.wasInterrupted() {
		return err
	}
	a.stopWorkers()
	if err := a.runTurn(ctx, lim.limits.interruptPrompt); err != nil {
		return err
	}
	return &exitStatus{code: exitInterrupted, msg: fmt.Sprintf("stopped at --interrupt-after %s", lim.limits.interruptAfter)}
}
