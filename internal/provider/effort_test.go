package provider

import (
	"strings"
	"testing"
)

// The vocabulary is the OpenAI one exactly. What a level means varies by
// model; which levels exist does not.
func TestParseEffortAcceptsTheSpecLevels(t *testing.T) {
	for in, want := range map[string]Effort{
		"none":      EffortNone,
		"minimal":   EffortMinimal,
		"low":       EffortLow,
		"medium":    EffortMedium,
		"high":      EffortHigh,
		"xhigh":     EffortXHigh,
		"max":       EffortMax,
		"HIGH":      EffortHigh,
		"  medium ": EffortMedium,
		"":          EffortUnset,
	} {
		got, err := ParseEffort(in)
		if err != nil {
			t.Errorf("ParseEffort(%q): %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("ParseEffort(%q) = %q, want %q", in, got, want)
		}
	}
}

// A near miss is a typo to report, not a level to guess at. Guessing produces
// a request that succeeds at a thinking level nobody asked for.
func TestParseEffortRejectsEverythingElse(t *testing.T) {
	for _, in := range []string{"off", "ultra", "turbo", "banana", "min", "med", "normal", "default", "disable"} {
		if got, err := ParseEffort(in); err == nil {
			t.Errorf("ParseEffort(%q) = %q, want an error naming the valid levels", in, got)
		}
	}
	_, err := ParseEffort("turbo")
	if err == nil || !strings.Contains(err.Error(), "none, minimal, low, medium, high, xhigh, max") {
		t.Errorf("error = %v, want the valid levels listed", err)
	}
}

// Unset is not a level. It means the field is absent, so the server's own
// default stands -- which is a different request from asking for "none".
func TestUnsetIsNotALevel(t *testing.T) {
	for _, e := range EffortLadder {
		if e == EffortUnset {
			t.Error("EffortUnset is in the ladder; it is the absence of a level")
		}
	}
}
