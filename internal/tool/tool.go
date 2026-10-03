// Package tool defines the tools ai-code exposes to the model and the
// value-passing executor boundary they run behind: Request and Result are JSON
// round-trippable, and working state lives with the executor, not the agent.
package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Request is one tool invocation. It crosses the executor boundary as JSON.
type Request struct {
	CallID string          `json:"call_id"`
	Name   string          `json:"name"`
	Args   json.RawMessage `json:"args"`
	// Cancel, on a request sent while CallID is still running in an executor
	// daemon, stops that call. It carries no tool to run.
	Cancel bool `json:"cancel,omitempty"`
}

// Result is the outcome. A tool that fails returns a Result with IsError set
// and an actionable message, not a Go error: an error result is feedback the
// model can act on, while an error escaping into the agent loop corrupts it.
type Result struct {
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`

	// Display is a one-line summary for the renderer, so the terminal can show
	// "Read internal/agent/loop.go (142 lines)" instead of the full payload.
	Display string `json:"display,omitempty"`

	// Show is markdown for the person watching, rendered in full beneath the
	// Display line at every verbosity. It never reaches the model, which
	// already has Content.
	Show string `json:"show,omitempty"`

	// Interrupted marks a result the tool itself produced in response to
	// cancellation. The agent wraps cancelled results but skips that when this
	// is set, since the tool has already said it.
	Interrupted bool `json:"interrupted,omitempty"`

	Cwd      string      `json:"cwd,omitempty"`
	Files    []FileStamp `json:"files,omitempty"`
	Duration string      `json:"duration,omitempty"`
}

// FileStamp identifies a file's content at the moment a tool saw it. The edit
// tool refuses when the recorded stamp no longer matches, stopping edits of a
// version that no longer exists.
type FileStamp struct {
	Path    string    `json:"path"`
	Hash    string    `json:"hash"`
	Size    int64     `json:"size"`
	ModTime time.Time `json:"mod_time"`
	Lines   int       `json:"lines,omitempty"`
}

// Definition is a tool as advertised to the model.
type Definition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema"`
	ReadOnly    bool            `json:"read_only"`
}

// Executor runs tools. The error return is reserved for the executor itself
// failing -- a broken transport to a remote host, say. A tool that fails
// returns a Result with IsError set and a nil error.
type Executor interface {
	Definitions() []Definition
	Execute(ctx context.Context, req Request) (Result, error)
	// Fork returns an executor over the same tools and filesystem, with its
	// own working directory, environment overlay and file stamps. Two agents
	// sharing them would corrupt each other's cwd and staleness checks.
	Fork() (Executor, error)
}

// Progress names what an executor is doing, while it is doing it. It is local
// wiring between an executor and the terminal, not a field of Request or
// Result: channels and callbacks stay out of the JSON boundary.
type Progress func(Phase)

// Phase is one step of a slow start, and whether it was the last. Only the
// final phase is worth keeping: it is the answer to why a tool call took
// minutes, long after the spinner is gone.
type Phase struct {
	Note string
	Done bool
}

// ProgressReporter is an executor whose first call can block long enough to
// need reporting, such as bringing a container up. It names the phases it
// actually reached, so silence means nothing happened.
type ProgressReporter interface {
	SetProgress(Progress)
}

// Tool is one tool's implementation.
type Tool interface {
	Name() string
	Description() string
	Schema() json.RawMessage
	// ReadOnly reports whether this tool can run concurrently with other
	// read-only tools. Anything that writes or runs a command says false.
	ReadOnly() bool
	Run(ctx context.Context, st *State, args json.RawMessage) Result
}

// State is the executor's view of the workspace. It lives with the executor
// rather than the agent because in a remote configuration this is genuinely
// remote state: the working directory of a shell on another machine.
type State struct {
	mu sync.Mutex

	cwd   string
	env   map[string]string
	files map[string]FileStamp
}

func NewState(cwd string) *State {
	return &State{cwd: cwd, env: map[string]string{}, files: map[string]FileStamp{}}
}

func (s *State) Cwd() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cwd
}

func (s *State) SetCwd(dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cwd = dir
}

// Env returns a copy of the tracked environment overlay.
func (s *State) Env() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]string, len(s.env))
	for k, v := range s.env {
		out[k] = v
	}
	return out
}

func (s *State) SetEnv(k, v string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.env[k] = v
}

func (s *State) Stamp(path string) (FileStamp, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, ok := s.files[path]
	return f, ok
}

func (s *State) SetStamp(f FileStamp) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.files[f.Path] = f
}

func (s *State) ForgetStamp(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.files, path)
}

type LocalExecutor struct {
	state *State
	tools map[string]Tool
	order []string
}

func NewLocalExecutor(state *State, tools ...Tool) *LocalExecutor {
	e := &LocalExecutor{state: state, tools: map[string]Tool{}}
	for _, t := range tools {
		e.tools[t.Name()] = t
		e.order = append(e.order, t.Name())
	}
	// Deterministic ordering: a map-order reshuffle would invalidate the cached prompt prefix.
	sort.Strings(e.order)
	return e
}

func (e *LocalExecutor) Definitions() []Definition {
	out := make([]Definition, 0, len(e.order))
	for _, name := range e.order {
		t := e.tools[name]
		out = append(out, Definition{
			Name:        t.Name(),
			Description: t.Description(),
			Schema:      t.Schema(),
			ReadOnly:    t.ReadOnly(),
		})
	}
	return out
}

func (e *LocalExecutor) Execute(ctx context.Context, req Request) (Result, error) {
	t, ok := e.tools[req.Name]
	if !ok {
		return Errorf("no tool named %q is available. Available tools: %s",
			req.Name, joinNames(e.order)), nil
	}

	start := time.Now()
	res := t.Run(ctx, e.state, req.Args)
	res.Duration = time.Since(start).Round(time.Millisecond).String()
	res.Cwd = e.state.Cwd()
	return res, nil
}

// State is the working state the tools act on.
func (e *LocalExecutor) State() *State { return e.state }

// IsReadOnly reports whether a named tool is safe to run concurrently.
func (e *LocalExecutor) IsReadOnly(name string) bool {
	t, ok := e.tools[name]
	return ok && t.ReadOnly()
}

func joinNames(names []string) string {
	out := ""
	for i, n := range names {
		if i > 0 {
			out += ", "
		}
		out += n
	}
	return out
}

// Errorf builds an error Result. The model reads these messages and has no
// other chance to recover, so they say what went wrong and what to do.
func Errorf(format string, args ...any) Result {
	msg := fmt.Sprintf(format, args...)
	return Result{Content: msg, IsError: true, Display: firstLine(msg)}
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return s[:i]
		}
	}
	return s
}

// decodeArgs unmarshals tool arguments, turning a malformed call into feedback
// rather than a failure: models do emit invalid JSON, and a precise error gets
// a correct retry.
func decodeArgs(raw json.RawMessage, dst any) *Result {
	if len(raw) == 0 {
		raw = []byte("{}")
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		r := Errorf("could not parse the arguments as JSON: %v\n\nReceived: %s\n\n"+
			"Re-issue the call with valid JSON matching the tool's schema.", err, truncateMiddle(string(raw), 500))
		return &r
	}
	return nil
}

func (e *LocalExecutor) Fork() (Executor, error) {
	tools := make([]Tool, 0, len(e.order))
	for _, name := range e.order {
		tools = append(tools, e.tools[name])
	}
	return NewLocalExecutor(NewState(e.state.Cwd()), tools...), nil
}

// Unavailable is an executor error saying the place tools run cannot be used
// at all: the remote machine cannot be reached, the container will not start,
// there is no worker for its platform. Nothing the model does can change
// that, so a run that meets it stops rather than handing the model an error
// to retry against.
type Unavailable struct {
	Err error
}

func (e *Unavailable) Error() string { return e.Err.Error() }
func (e *Unavailable) Unwrap() error { return e.Err }
