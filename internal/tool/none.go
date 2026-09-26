package tool

import "context"

// None is an executor with no tools, for an agent meant to think rather than
// act. An empty schema is what stops the model reaching for one, and the
// error covers a model that names a tool from memory anyway.
func None() Executor { return noExecutor{} }

type noExecutor struct{}

func (noExecutor) Definitions() []Definition { return nil }

func (noExecutor) Execute(context.Context, Request) (Result, error) {
	return Errorf("no tools are available here. Answer from what you were given."), nil
}

func (noExecutor) IsReadOnly(string) bool { return true }

// Fork is the same nothing. A child with no tools has no working state to
// keep apart from anyone else's.
func (n noExecutor) Fork() (Executor, error) { return n, nil }
