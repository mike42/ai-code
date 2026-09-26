package main

import (
	"context"
	"strings"
	"testing"
)

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

// A checkpoint is text added in front of the kept tail, so compacting a
// session already smaller than its tail budget can grow the next request.
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
