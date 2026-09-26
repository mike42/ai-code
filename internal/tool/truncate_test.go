package tool

import (
	"fmt"
	"strings"
	"testing"
)

func TestTruncateKeepsBothEnds(t *testing.T) {
	// The case that motivates head+tail: a long build log whose only useful
	// line is the error at the very bottom.
	var b strings.Builder
	b.WriteString("BUILD STARTED\n")
	for i := range 5000 {
		fmt.Fprintf(&b, "compiling module %d\n", i)
	}
	b.WriteString("FATAL: undefined reference to `frobnicate'\n")
	input := b.String()

	out, truncated := Truncate(input, 2000)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if len(out) > 2200 {
		t.Errorf("output is %d bytes, well over the 2000 budget", len(out))
	}
	if !strings.Contains(out, "BUILD STARTED") {
		t.Error("head was lost")
	}
	if !strings.Contains(out, "FATAL: undefined reference") {
		t.Error("tail was lost -- this is the line the model actually needs")
	}
	if !strings.Contains(out, "elided") {
		t.Error("no elision marker; the model cannot tell it has an excerpt")
	}
}

func TestTruncateReportsHowMuchWasRemoved(t *testing.T) {
	input := strings.Repeat("some line of output\n", 1000)
	out, _ := Truncate(input, 500)
	if !strings.Contains(out, "lines elided") {
		t.Errorf("marker should state the line count, got: %q", out)
	}
	if !strings.Contains(out, "KB") && !strings.Contains(out, "B)") {
		t.Errorf("marker should state the byte count, got: %q", out)
	}
}

func TestTruncateLeavesShortOutputAlone(t *testing.T) {
	input := "just a little output\n"
	out, truncated := Truncate(input, 1000)
	if truncated {
		t.Error("short output should not be truncated")
	}
	if out != input {
		t.Errorf("output was modified: %q", out)
	}
}

func TestTruncateHandlesOneEnormousLine(t *testing.T) {
	// Minified JSON has no line structure to split on.
	input := strings.Repeat("x", 100_000)
	out, truncated := Truncate(input, 1000)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if len(out) > 1200 {
		t.Errorf("output is %d bytes, over budget", len(out))
	}
	if !strings.Contains(out, "elided") {
		t.Error("no elision marker")
	}
}

func TestTruncateZeroBudgetIsNoOp(t *testing.T) {
	input := "content"
	out, truncated := Truncate(input, 0)
	if truncated || out != input {
		t.Error("a zero budget should disable truncation rather than empty the output")
	}
}

func TestPluralisation(t *testing.T) {
	// "2 matchs" in tool output is small, but it is the kind of small that
	// makes a tool feel unfinished.
	cases := map[string]string{
		"match": "2 matches", "file": "2 files", "line": "2 lines",
		"box": "2 boxes", "class": "2 classes", "entry": "2 entries",
	}
	for word, want := range cases {
		if got := plural(2, word); got != want {
			t.Errorf("plural(2, %q) = %q, want %q", word, got, want)
		}
	}
	if got := plural(1, "match"); got != "1 match" {
		t.Errorf("plural(1, \"match\") = %q", got)
	}
}
