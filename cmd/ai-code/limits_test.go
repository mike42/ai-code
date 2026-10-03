package main

import (
	"strings"
	"testing"
)

func TestTimeLimitFlags(t *testing.T) {
	ok := [][]string{
		{"-p", "--steer-after", "10m", "--interrupt-after", "20m", "--interrupt-prompt", "wrap up", "--exit-after", "30m", "go"},
		{"-p", "--exit-after", "1h", "go"},
		{"-p", "--steer-after", "5m", "--steer-prompt", "nearly done", "go"},
		{"go"},
	}
	for _, args := range ok {
		if _, err := parseFlags(args); err != nil {
			t.Errorf("%v: %v", args, err)
		}
	}
	bad := map[string][]string{
		"need -p":                  {"--exit-after", "1h"},
		"needs --interrupt-prompt": {"-p", "--interrupt-after", "1h"},
		"needs --interrupt-after":  {"-p", "--interrupt-prompt", "x", "--exit-after", "1h"},
		"needs --steer-after":      {"-p", "--steer-prompt", "x", "--exit-after", "1h"},
		"must come after":          {"-p", "--steer-after", "30m", "--exit-after", "20m"},
		"takes a duration":         {"-p", "--exit-after", "90"},
	}
	for want, args := range bad {
		_, err := parseFlags(args)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%v: got %v, want an error saying %q", args, err, want)
		}
	}
	f, _ := parseFlags([]string{"-p", "--interrupt-after", "90s", "--interrupt-prompt", "commit now", "x"})
	if f.limits.interruptAfter.Seconds() != 90 || f.limits.interruptPrompt != "commit now" {
		t.Errorf("parsed %+v", f.limits)
	}
}
