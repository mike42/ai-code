package provider

import (
	"fmt"
	"strings"
)

// Effort is the thinking level requested for one call. It maps to
// reasoning_effort at the llama.cpp level and reasoning.effort on OpenRouter.
// Backends silently ignore unknown values, so the vocabulary is closed here.
type Effort string

const (
	// EffortUnset sends nothing, leaving the server's default; EffortNone asks
	// for the field to be sent.
	EffortUnset Effort = ""
	// EffortNone disables thinking. llama.cpp implements this itself, so it
	// works whether or not the model has levels of its own.
	EffortNone    Effort = "none"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	// EffortXHigh and EffortMax sit above the OpenAI ladder; xAI takes "xhigh",
	// GLM-5.2 takes "max", and a model that does not know one ignores it.
	EffortXHigh Effort = "xhigh"
	EffortMax   Effort = "max"
)

// EffortLadder is the set of levels, weakest first.
var EffortLadder = []Effort{
	EffortNone, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax,
}

// EffortNames lists the levels for a message to a person or a model.
func EffortNames() string {
	names := make([]string, len(EffortLadder))
	for i, e := range EffortLadder {
		names[i] = string(e)
	}
	return strings.Join(names, ", ")
}

// ParseEffort accepts an exact level or the empty string; it does not guess at
// spelling variants.
func ParseEffort(s string) (Effort, error) {
	e := Effort(strings.ToLower(strings.TrimSpace(s)))
	if e == EffortUnset {
		return EffortUnset, nil
	}
	for _, valid := range EffortLadder {
		if e == valid {
			return e, nil
		}
	}
	return "", fmt.Errorf("unknown thinking level %q (want %s)", s, EffortNames())
}
