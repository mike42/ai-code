// Package tool defines the tools ai-code exposes to the model, and the boundary
// they run behind.
//
// That boundary is deliberately a value-passing interface rather than a set of
// direct calls. Today there is one implementation, LocalExecutor, which runs
// tools in this process. The interface is shaped so a future implementation can
// run them somewhere else entirely -- in a container, a sandbox, or over SSH to
// a remote development host -- without the agent noticing. Two rules keep that
// door open, and both are enforced now rather than promised later:
//
//   - Request and Result are JSON round-trippable. No pointers into agent state,
//     no channels, no io.Reader.
//   - Tools never reach into the agent. Everything they need arrives in the
//     request; everything they change comes back in the result. Working
//     directory, environment and file stamps live with the executor, because in
//     a remote setup that is where the filesystem actually is.
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
}

// Result is the outcome. A tool that fails returns a Result with IsError set
// and an actionable message; it does not return a Go error. The distinction
// matters: an error escaping into the agent loop is how conversations get
// corrupted, whereas an error *result* is feedback the model can act on, which
// is the only self-correction channel it has.
type Result struct {
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`

	// Display is a one-line summary for the renderer, so the terminal can show
	// "Read internal/agent/loop.go (142 lines)" instead of the full payload.
	Display string `json:"display,omitempty"`

	// Interrupted marks a result the tool itself produced in response to
	// cancellation. The agent wraps cancelled results with an explanation, but
	// skips doing so when this is set: a tool that already handled the
	// interruption has said it better, and two stacked explanations of the same
	// event read as a bug.
	Interrupted bool `json:"interrupted,omitempty"`

	// Metadata the agent uses for its own bookkeeping.
	Cwd      string      `json:"cwd,omitempty"`
	Files    []FileStamp `json:"files,omitempty"`
	Duration string      `json:"duration,omitempty"`
}

// FileStamp identifies a file's content at the moment a tool saw it. The edit
// tool refuses to apply a change when the stamp it recorded no longer matches,
// which is what stops the model confidently editing a version of a file that no
// longer exists.
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
	// Fork returns an executor over the same tools and the same filesystem,
	// with its own working state: working directory, environment overlay and
	// file stamps.
	//
	// It exists because agents run beside each other. Two agents sharing one
	// working state move each other's cwd, and -- quietly, which is worse --
	// satisfy each other's staleness checks, so one agent's edit is applied
	// to content the other read. Forking is what makes a concurrent agent
	// correct, not what makes it safe: a fork sees the same files and may
	// write to all of them.
	//
	// Every executor answers, and the answer is a separate shell of whatever
	// kind that executor has:
	//
	//   - local: a new State over the same tools.
	//   - devcontainer: a second executor daemon in the same container, so
	//     the filesystem is identical and only the shell is new.
	//   - remote: a second connection to the same host.
	//
	// It is a method rather than an optional interface because the wrong
	// answer is silence: an executor that could not fork but said nothing
	// would hand two agents one shell and corrupt their edits invisibly. An
	// executor that genuinely cannot returns an error saying so, and the
	// caller reports it rather than sharing.
	Fork() (Executor, error)
}

// Progress names what an executor is doing, while it is doing it.
//
// Deliberately not a field of Request or Result. Those cross the executor
// boundary as JSON, and this package's own rule keeps channels and callbacks
// out of them. Progress never crosses that boundary: it is local wiring
// between an executor and whatever is drawing the terminal, so a remote
// executor reports only the phases its local half went through.
type Progress func(Phase)

// Phase is one step of a slow start, and whether it was the last.
//
// Done separates the two things the reader needs at different moments. While
// the wait is on, each phase replaces the one before it and none of it is
// worth keeping. Once it is over, exactly one line is worth keeping, because
// it is the answer to "why did that tool call take four minutes" long after
// the spinner has gone.
type Phase struct {
	Note string
	Done bool
}

// ProgressReporter is an executor whose first call can block long enough that
// the user needs to be told why.
//
// Bringing a container up is minutes of engine work sitting behind a tool
// call, and from the terminal that is indistinguishable from `bash` having
// hung. An executor that can be slow to start accepts a Progress and names
// each phase as it reaches it -- phases it actually reached, so that silence
// means nothing happened rather than nothing was reported.
type ProgressReporter interface {
	SetProgress(Progress)
}

// Tool is one tool's implementation.
type Tool interface {
	Name() string
	Description() string
	Schema() json.RawMessage
	// ReadOnly reports whether this tool can safely run concurrently with other
	// read-only tools in the same turn. Anything that writes to the filesystem
	// or runs a command says false and is serialised.
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

// ---------------------------------------------------------------------------
// LocalExecutor
// ---------------------------------------------------------------------------

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
	// Deterministic ordering matters more than it looks: tool definitions sit
	// in the cached prefix of every request, and a map-iteration reshuffle
	// silently invalidates the provider's prompt cache on every single turn.
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

// Errorf builds an error Result. Every message here is read by the model and is
// its only chance to recover, so they say what went wrong AND what to do.
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
// rather than a failure. Models do emit invalid JSON; telling them precisely
// what was wrong gets a correct retry far more often than a bare parse error.
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

// Fork returns an executor over the same tools with independent state.
//
// Tools are stateless -- Run takes the State it works against -- so a fork
// shares them and differs only in what they act on.
//
// A concurrent agent needs its own working directory, environment overlay and
// file stamps. Sharing cwd means one agent's `cd` moves another's. Sharing
// stamps is worse and quieter: a stamp records what one reader last saw, so a
// shared map lets agent B pass checkStale on the strength of agent A's read,
// and B's edit is applied to content B never looked at. Separate stamps turn
// that into the ordinary "changed on disk since it was read" refusal, which is
// already the right answer between two writers.
func (e *LocalExecutor) Fork() (Executor, error) {
	tools := make([]Tool, 0, len(e.order))
	for _, name := range e.order {
		tools = append(tools, e.tools[name])
	}
	return NewLocalExecutor(NewState(e.state.Cwd()), tools...), nil
}
