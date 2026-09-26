package provider

import (
	"fmt"
	"strings"
)

// Effort is the thinking level requested for one call.
//
// On a local setup this is the main capability knob there is: a model swap
// costs tens of seconds to minutes, so moving the resident model between
// "answer immediately" and "think hard" is the only cheap way to change how
// much work goes into a turn. It maps onto reasoning_effort at the llama.cpp
// level and onto reasoning.effort on OpenRouter.
//
// The vocabulary is fixed and the meaning is not. Which levels a model
// distinguishes is its own business -- several commonly collapse onto one
// behaviour, and a model with no reasoning ignores the field entirely -- but
// the set of words is closed.
//
// It has to be closed here because nothing downstream closes it. Neither
// lemonade nor llama.cpp validates reasoning_effort: the value is handed to
// the chat template, and one the template does not recognise is silently
// ignored. Measured against a live server, "banana" is accepted with a 200.
// So a typo that reached the wire would produce a successful request at
// whatever thinking level the model happens to default to.
type Effort string

const (
	// EffortUnset sends nothing, leaving the server's own default in place.
	// Distinct from EffortNone, which asks for the field to be sent.
	EffortUnset Effort = ""
	// EffortNone disables thinking. llama.cpp implements this itself, so it
	// works whether or not the model has levels of its own.
	EffortNone    Effort = "none"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	// EffortXHigh and EffortMax are above the OpenAI ladder. Both are real:
	// xAI takes "xhigh", and GLM-5.2 takes "max" over an OpenAI-compatible
	// endpoint. Neither is universal, and a model that does not know one
	// ignores it.
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

// ParseEffort accepts a level exactly, and the empty string for "send
// nothing". Spelling variants are not accepted: a rejected word is a typo the
// caller can fix, where a guess at what they meant is a thinking level
// quietly different from the one they asked for.
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
