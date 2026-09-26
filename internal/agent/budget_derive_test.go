package agent

import "testing"

// The sizes have to stay in a sane relationship to each other at every window,
// not just at the 262k one they were originally tuned against.
func TestBudgetScalesWithTheWindow(t *testing.T) {
	for _, window := range []int{8192, 16384, 32768, 65536, 131072, 262144, 1 << 20} {
		b := BudgetFor(window)
		usable := window - b.Reserve
		prompt := usable * 3 / 4
		t.Logf("window=%-8d reserve=%-6d usable=%-7d prompt=%-7d keepRecent=%-7d summary=%-5d result=%d (%d B)",
			window, b.Reserve, usable, prompt, b.KeepRecent, b.Summary, b.MaxToolResult, b.MaxToolResultBytes())

		if b.Reserve >= window/2 {
			t.Errorf("window %d: reserve %d is half the window or more", window, b.Reserve)
		}
		// The point of the whole exercise: whatever survives compaction must
		// leave the session room to grow before the next one.
		if free := prompt - b.KeepRecent - b.Summary; free < prompt/3 {
			t.Errorf("window %d: only %d of %d prompt tokens free after a compaction",
				window, free, prompt)
		}
		if b.KeepRecent >= prompt {
			t.Errorf("window %d: tail %d cannot fit in a %d prompt", window, b.KeepRecent, prompt)
		}
		// The tail has to hold several exchanges, not one result and a stub --
		// except at the bottom, where the floor on a usable excerpt wins and
		// the window is simply too small to have both.
		if b.MaxToolResult > minToolResultTokens && b.MaxToolResult > b.KeepRecent/4 {
			t.Errorf("window %d: one result (%d) claims more than a quarter of the tail (%d)",
				window, b.MaxToolResult, b.KeepRecent)
		}
		if b.Summary >= b.Reserve {
			t.Errorf("window %d: summary %d leaves the reserve %d nothing to answer in",
				window, b.Summary, b.Reserve)
		}
	}
}

// A configured value overrides the formula, globally, for every model.
func TestConfiguredSizesOverrideTheFormula(t *testing.T) {
	derived := newAgent(t, &scriptedClient{}, &collectSink{})
	derived.opts.ContextLimit = 32768
	derived.opts.ReserveTokens = 0
	derived.opts.KeepRecentTokens = 0

	set := newAgent(t, &scriptedClient{}, &collectSink{})
	set.opts.ContextLimit = 32768
	set.opts.ReserveTokens = 9000
	set.opts.KeepRecentTokens = 4000

	if derived.reserveTokens() == 9000 || derived.keepRecentTokens() == 4000 {
		t.Fatal("the derived agent picked up the other one's overrides")
	}
	if got := set.reserveTokens(); got != 9000 {
		t.Errorf("reserve = %d, want the configured 9000", got)
	}
	if got := set.keepRecentTokens(); got != 4000 {
		t.Errorf("keepRecent = %d, want the configured 4000", got)
	}
	// An overridden tail carries the result cap with it, so the two cannot be
	// configured into disagreeing.
	if got := set.maxMessageChars(); got != int(4000/4*defaultCharsPerToken) {
		t.Errorf("maxMessageChars = %d, want a quarter of the configured tail", got)
	}
}

// A model swap re-derives, rather than leaving a wide model's sizes on a
// narrow one.
func TestSwappingModelsRederivesTheBudget(t *testing.T) {
	a := newAgent(t, &scriptedClient{}, &collectSink{})
	a.SetModel("wide", 262144, 0)
	wide := a.keepRecentTokens()
	a.SetModel("narrow", 32768, 0)
	narrow := a.keepRecentTokens()

	if narrow >= wide {
		t.Errorf("tail is %d on a 32k model and %d on a 262k one", narrow, wide)
	}
	if narrow >= a.Usable() {
		t.Errorf("tail %d does not fit the %d budget it was derived from", narrow, a.Usable())
	}
}
