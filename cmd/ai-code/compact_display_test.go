package main

import (
	"context"
	"strings"
	"testing"
)

// The reported bug, at the level it was seen: /compact printed a figure, and
// the status line above the next prompt still showed the one from before.
//
// Two caches of one number. The prompt marker asked the agent directly and
// was right; the renderer held a copy it only updated on three event kinds
// out of twelve, and /compact emitted none of them. Nothing invalidated it
// until the next turn ended, so the session looked to be back where it had
// started.
func TestCompactLeavesNoStaleFigureBehind(t *testing.T) {
	a, _ := swapApp(t, 65536, 200000)

	before := a.contextState().Projected
	if before == 0 {
		t.Fatal("the App was never told a context figure, so this proves nothing")
	}

	out := captureOut(t, func() {
		if err := a.cmdCompact(context.Background(), ""); err != nil {
			t.Fatalf("cmdCompact: %v", err)
		}
	})

	after := a.contextState().Projected
	t.Logf("%s\nApp figure: %d -> %d", strings.TrimSpace(out), before, after)

	if after >= before {
		t.Errorf("the figure the prompt will show is %d, unchanged from %d before the compaction",
			after, before)
	}
	// The number printed and the number the prompt will show are the same
	// number, so they have to agree to the digit.
	if !strings.Contains(out, compactInt(after)) {
		t.Errorf("/compact printed %q but the prompt will show %s",
			strings.TrimSpace(out), compactInt(after))
	}
}

// A session with room to spare must not be made larger by compacting it.
//
// A checkpoint is text, and it goes in front of everything the cut kept. On a
// session already smaller than its own tail budget the cut moves a message or
// two and the summary adds more than that back, so the operation ends with a
// bigger request than it started with -- which is how a compaction came to be
// reported as freeing nothing while the figure went up.
func TestCompactOnASmallSessionIsHonest(t *testing.T) {
	a, _ := swapApp(t, 262144, 2000)
	before := a.contextState().Projected

	out := captureOut(t, func() {
		if err := a.cmdCompact(context.Background(), ""); err != nil {
			t.Fatalf("cmdCompact: %v", err)
		}
	})
	after := a.contextState().Projected
	t.Logf("%s\nApp figure: %d -> %d", strings.TrimSpace(out), before, after)

	if after > before {
		t.Errorf("compacting a small session grew the next request from %d to %d",
			before, after)
	}
}
