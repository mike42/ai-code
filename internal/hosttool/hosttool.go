// Package hosttool provides tools that always run, and keep their files, on
// this machine.
package hosttool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-code/internal/tool"
)

const (
	TodoName   = "todo"
	StatusName = "status_update"
)

func Names() []string { return []string{StatusName, TodoName} }

type Executor struct {
	inner   tool.Executor
	dir     string
	enabled map[string]bool
	now     func() time.Time
}

func Wrap(inner tool.Executor, dir string, names []string) *Executor {
	e := &Executor{inner: inner, dir: dir, enabled: map[string]bool{}, now: time.Now}
	for _, n := range names {
		e.enabled[n] = true
	}
	return e
}

func (e *Executor) Definitions() []tool.Definition {
	defs := e.inner.Definitions()
	if e.enabled[StatusName] {
		defs = append(defs, tool.Definition{Name: StatusName, Description: statusDescription, Schema: statusSchema})
	}
	if e.enabled[TodoName] {
		defs = append(defs, tool.Definition{Name: TodoName, Description: todoDescription, Schema: todoSchema})
	}
	return defs
}

func (e *Executor) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	if !e.enabled[req.Name] {
		return e.inner.Execute(ctx, req)
	}
	start := e.now()
	var res tool.Result
	switch req.Name {
	case TodoName:
		res = e.todo(req.Args)
	case StatusName:
		res = e.status(req.Args)
	}
	res.Duration = e.now().Sub(start).Round(time.Millisecond).String()
	return res, nil
}

func (e *Executor) Unwrapped() tool.Executor {
	if u, ok := e.inner.(interface{ Unwrapped() tool.Executor }); ok {
		return u.Unwrapped()
	}
	return e.inner
}

func (e *Executor) Fork() (tool.Executor, error) { return e.inner.Fork() }

func (e *Executor) IsReadOnly(name string) bool {
	if e.enabled[name] {
		return false
	}
	if ro, ok := e.inner.(interface{ IsReadOnly(string) bool }); ok {
		return ro.IsReadOnly(name)
	}
	return false
}

func (e *Executor) SetProgress(p tool.Progress) {
	if pr, ok := e.inner.(tool.ProgressReporter); ok {
		pr.SetProgress(p)
	}
}

func (e *Executor) Close() error {
	if c, ok := e.inner.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

func (e *Executor) Carried() string {
	var parts []string
	if e.enabled[StatusName] {
		if b, err := os.ReadFile(filepath.Join(e.dir, statusFile)); err == nil && len(strings.TrimSpace(string(b))) > 0 {
			parts = append(parts, "Your last status update, from a previous run in this workspace:\n\n"+
				"<status-update>\n"+strings.TrimSpace(string(b))+"\n</status-update>")
		}
	}
	if e.enabled[TodoName] {
		if list, err := e.loadTodos(); err == nil && len(list.Items) > 0 {
			parts = append(parts, "Your todo list, as you left it:\n\n"+todoText(list.Items))
		}
	}
	return strings.Join(parts, "\n\n")
}

func writeFile(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return nil
}

func decode(raw json.RawMessage, dst any) *tool.Result {
	if len(raw) == 0 {
		raw = []byte("{}")
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		r := tool.Errorf("could not parse the arguments as JSON: %v\n\nRe-issue the call with valid JSON matching the tool's schema.", err)
		return &r
	}
	return nil
}
