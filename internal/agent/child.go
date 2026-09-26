package agent

import (
	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// Child describes an agent spawned by another one.
//
// There is deliberately no provider or model field, and there never will be.
// A child runs on the client and model its parent is already using, so no
// value in a configuration file can route a sub-agent anywhere the user did
// not send the session itself.
type Child struct {
	// System replaces the parent's system prompt entirely. A child starts with
	// no conversation, so the prompt is all the context it has.
	System string
	// Effort is the thinking level for every request the child makes.
	Effort provider.Effort
	// NoTools gives it none at all, for a child meant to reason over what it
	// was handed rather than go and look.
	NoTools bool
	// Sink receives the child's events. Nil discards them, which is what a
	// child whose only output is its final answer wants.
	Sink Sink
}

// Spawn creates a child agent with a fresh conversation.
//
// It shares the parent's client, model and executor, and inherits the window
// arithmetic, so a child on a 4k model compacts on the same terms the parent
// does. It does not share messages, accounting or the standby checkpoint.
//
// A child never receives the task tool, because the executor it is handed is
// the one underneath it. That is what bounds recursion: not a depth counter
// that has to be threaded through and can be got wrong, but a tool that is
// absent from the schema a child is shown.
func (a *Agent) Spawn(c Child) (*Agent, error) {
	exec := a.exec
	if inner, ok := exec.(interface{ Unwrapped() tool.Executor }); ok {
		exec = inner.Unwrapped()
	}
	// Its own working directory, environment and file stamps. A child runs
	// beside its parent rather than after it, so shared state is two agents
	// moving each other's cwd and -- quietly -- satisfying each other's
	// staleness checks. See tool.Executor.Fork.
	//
	// A fork that fails is reported rather than worked around. Handing the
	// child its parent's shell would leave both agents editing against each
	// other's reads, which is the failure this call exists to prevent, and it
	// would do it invisibly.
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
	// The parent's checkpoint machinery is about surviving a model swap in a
	// long-lived session. A child is finished before one can happen.
	opts.AutoCompact = true

	child := New(a.client, a.model, exec, c.Sink, opts)
	child.SetSystem(c.System)
	return child, nil
}

// Close releases what this agent's executor holds open.
//
// Only a child's executor is ever its own: Spawn forks one, and a fork can be
// a live process -- in a devcontainer it is a second `exec` daemon inside the
// container, started on the fork's first tool call and otherwise kept for the
// life of the session. A worker that ends without this leaves that daemon
// behind, and a long session of workers ends up unable to fork at all.
//
// Whoever forked the executor closes it. Calling this on a session's own
// agent would close the executor the session is still using.
func (a *Agent) Close() error {
	if c, ok := a.exec.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

// Effort is the thinking level this agent asks for. Unset sends nothing and
// lets the model do whatever it does by default.
func (a *Agent) Effort() provider.Effort { return a.opts.Effort }

// SetEffort changes the thinking level for every later request.
//
// It takes effect on the next turn and touches nothing already said, so the
// conversation carries across the change. Compaction is unaffected: it sends
// its own level, because summarising is mechanical and paying max-effort
// thinking for it is minutes of wall clock on a local model.
func (a *Agent) SetEffort(e provider.Effort) { a.opts.Effort = e }

// LastAssistantText is the final thing the model said, which for a child is
// its whole answer.
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
