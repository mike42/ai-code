package main

import (
	"context"
	"fmt"
	"strings"

	"ai-code/internal/provider"
	"ai-code/internal/render"
)

// cmdThink shows or changes how hard the model thinks.
//
// On a local setup this is the only capability knob that is cheap. Swapping
// model costs tens of seconds to minutes of loading; moving the resident
// model between "answer now" and "think hard" costs nothing and takes effect
// on the next turn.
func (a *App) cmdThink(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())

	if strings.TrimSpace(args) == "" {
		a.out("",
			fmt.Sprintf("  Thinking: %s", style.Bold(effortLabel(a.agent.Effort()))),
			"  "+style.Dim(strings.ReplaceAll(provider.EffortNames(), ", ", " · ")),
			"")
		return nil
	}

	want, err := provider.ParseEffort(args)
	if err != nil {
		return err
	}
	a.agent.SetEffort(want)
	if a.flags != nil {
		a.flags.think = string(want)
	}
	a.out("", fmt.Sprintf("Thinking: %s.", style.Bold(effortLabel(want))), "")
	return nil
}

func effortLabel(e provider.Effort) string {
	if e == provider.EffortUnset {
		return "the model's default"
	}
	return string(e)
}
