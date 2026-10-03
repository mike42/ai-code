package tool

import "context"

type Filtered struct {
	inner Executor
	allow func(name string) bool
}

func Filter(inner Executor, allow func(name string) bool) *Filtered {
	return &Filtered{inner: inner, allow: allow}
}

func (f *Filtered) Definitions() []Definition {
	var out []Definition
	for _, d := range f.inner.Definitions() {
		if f.allow(d.Name) {
			out = append(out, d)
		}
	}
	return out
}

func (f *Filtered) Execute(ctx context.Context, req Request) (Result, error) {
	if !f.allow(req.Name) {
		var names []string
		for _, d := range f.Definitions() {
			names = append(names, d.Name)
		}
		return Errorf("no tool named %q is available. Available tools: %s",
			req.Name, joinNames(names)), nil
	}
	return f.inner.Execute(ctx, req)
}

func (f *Filtered) Fork() (Executor, error) {
	inner, err := f.inner.Fork()
	if err != nil {
		return nil, err
	}
	return Filter(inner, f.allow), nil
}

func (f *Filtered) IsReadOnly(name string) bool {
	if ro, ok := f.inner.(interface{ IsReadOnly(string) bool }); ok {
		return ro.IsReadOnly(name)
	}
	return false
}

func (f *Filtered) SetProgress(p Progress) {
	if pr, ok := f.inner.(ProgressReporter); ok {
		pr.SetProgress(p)
	}
}

func (f *Filtered) Close() error {
	if c, ok := f.inner.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}
