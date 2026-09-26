package agent

import (
	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// Child describes an agent spawned by another one. A child runs on the client
// and model its parent is already using, so nothing in a config file can route
// a sub-agent elsewhere.
type Child struct {
	// System replaces the parent's system prompt entirely; a child starts with no conversation.
	System string
	// Effort is the thinking level for every request the child makes.
	Effort provider.Effort
	// NoTools gives the child no tools, for reasoning over what it was handed.
	NoTools bool
	// Sink receives the child's events; nil discards them.
	Sink Sink
}

// Spawn creates a child agent with a fresh conversation. It shares the
// parent's client, model, executor and window arithmetic, not its messages.
// A child never receives the task tool, which is what bounds recursion.
func (a *Agent) Spawn(c Child) (*Agent, error) {
	exec := a.exec
	if inner, ok := exec.(interface{ Unwrapped() tool.Executor }); ok {
		exec = inner.Unwrapped()
	}
	// Fork gives the child its own cwd, environment and file stamps, so the two
	// agents cannot satisfy each other's staleness checks. A failed fork is reported.
	if !c.NoTools {
		forked, err := exec.Fork()
		if err != nil {
			return nil, err
		}
		exec = forked
	} else {
		exec = tool.None()
	}

	opts := a.opts
	opts.Effort = c.Effort
	opts.AutoCompact = true

	child := New(a.client, a.model, exec, c.Sink, opts)
	child.SetSystem(c.System)
	return child, nil
}

// Close releases what this agent's executor holds open. Only a child's
// executor is ever its own: Spawn forks one, and a fork can be a live process.
// Whoever forked it closes it.
func (a *Agent) Close() error {
	if c, ok := a.exec.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

// Effort is the thinking level this agent asks for; unset sends nothing.
func (a *Agent) Effort() provider.Effort { return a.opts.Effort }

// SetEffort changes the thinking level for every later request.
func (a *Agent) SetEffort(e provider.Effort) { a.opts.Effort = e }

// LastAssistantText is the final thing the model said, which for a child is its whole answer.
func (a *Agent) LastAssistantText() string {
	for i := len(a.messages) - 1; i >= 0; i-- {
		if m := a.messages[i]; m.Role == provider.RoleAssistant && m.Content != "" {
			return m.Content
		}
	}
	return ""
}

// Turns is how many times this agent has been round the loop.
func (a *Agent) Turns() int {
	n := 0
	for _, m := range a.messages {
		if m.Role == provider.RoleAssistant {
			n++
		}
	}
	return n
}
