package tool

import (
	"context"
	"testing"
)

type closeCounter struct {
	*LocalExecutor
	closed *int
}

func (c closeCounter) Close() error { *c.closed++; return nil }

func (c closeCounter) Fork() (Executor, error) {
	inner, _ := c.LocalExecutor.Fork()
	return closeCounter{inner.(*LocalExecutor), c.closed}, nil
}

func TestFilterHidesAndRefusesDisabledTools(t *testing.T) {
	local := NewLocalExecutor(NewState(t.TempDir()), &BashTool{}, &ReadTool{}, &LsTool{})
	closed := 0
	f := Filter(closeCounter{local, &closed}, func(name string) bool { return name != "bash" })

	for _, d := range f.Definitions() {
		if d.Name == "bash" {
			t.Fatal("bash is disabled but still advertised")
		}
	}
	res, err := f.Execute(context.Background(), Request{Name: "bash", Args: []byte(`{"command":"echo ran"}`)})
	if err != nil || !res.IsError {
		t.Fatalf("calling a disabled tool: %+v, %v", res, err)
	}

	forked, err := f.Fork()
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range forked.Definitions() {
		if d.Name == "bash" {
			t.Fatal("a fork got back a tool its parent was denied")
		}
	}
	if !forked.(*Filtered).IsReadOnly("read") {
		t.Error("IsReadOnly was not passed through")
	}
	_ = forked.(*Filtered).Close()
	if closed != 1 {
		t.Errorf("Close reached the wrapped executor %d times, want 1", closed)
	}
}
