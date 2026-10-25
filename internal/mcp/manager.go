package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"ai-code/internal/tool"
)

// Entry is one named server as configured: a binary to run or a URL.
type Entry struct {
	Name string
	// Source is "global", "local" or "cli"; Hides is the source of a lower
	// entry of the same name that this one replaces.
	Source string
	Hides  string
	// Path is the config file, empty for cli.
	Path string
	// Value is the --mcp text after "=", as typed, for cli entries.
	Value string

	URL     string
	Headers map[string]string
	Binary  string
	Args    []string
}

// Target is the binary or URL, which is what identifies a server to the user.
func (e Entry) Target() string {
	if e.URL != "" {
		return e.URL
	}
	return e.Binary
}

type State string

const (
	Inactive State = "inactive"
	Starting State = "starting"
	Active   State = "active"
	Failed   State = "failed"
)

// Report says a start has finished, one way or the other.
type Report struct {
	Name string
	Err  error
	// Tools and SchemaChars describe what the model now sees.
	Tools       int
	SchemaChars int
}

// Row is one line of /mcp list.
type Row struct {
	Entry
	State       State
	Detail      string
	Tools       int
	SchemaChars int
}

type exposed struct {
	name     string
	original string
	def      tool.Definition
}

type server struct {
	entry  Entry
	state  State
	detail string
	client *Client
	tools  []exposed
	cancel context.CancelFunc
	// listChanged: the server's tool list changed since it was fetched.
	listChanged bool
}

// Manager runs the session's MCP servers. Only started servers contribute
// tools, appended after the built-in ones in the order they were started.
type Manager struct {
	resolve func() (map[string]Entry, error)
	env     []string

	mu       sync.Mutex
	servers  map[string]*server
	order    []string
	reserved map[string]bool
	// changed is closed and replaced whenever a start finishes.
	changed     chan struct{}
	report      func(Report)
	held        []Report
	unsupported string
}

// NewManager resolves entries through resolve each time it needs them, so
// /mcp list and start see the config files as they are now.
func NewManager(resolve func() (map[string]Entry, error), env []string) *Manager {
	return &Manager{resolve: resolve, env: env, servers: map[string]*server{}, changed: make(chan struct{})}
}

// SetUnsupported makes every start fail with msg: the runtime cannot run
// servers, and the host is never a fallback for a sandboxed session.
func (m *Manager) SetUnsupported(msg string) {
	m.mu.Lock()
	m.unsupported = msg
	m.mu.Unlock()
}

// SetReserved names the tools a server's tools must not collide with.
func (m *Manager) SetReserved(names []string) {
	m.mu.Lock()
	m.reserved = map[string]bool{}
	for _, n := range names {
		m.reserved[n] = true
	}
	m.mu.Unlock()
}

// SetReporter delivers every finished start, including any that finished
// before it was set.
func (m *Manager) SetReporter(fn func(Report)) {
	m.mu.Lock()
	m.report = fn
	held := m.held
	m.held = nil
	m.mu.Unlock()
	for _, r := range held {
		fn(r)
	}
}

func (m *Manager) emit(r Report) {
	m.mu.Lock()
	fn := m.report
	if fn == nil {
		m.held = append(m.held, r)
	}
	m.mu.Unlock()
	if fn != nil {
		fn(r)
	}
}

// Entry resolves one name.
func (m *Manager) Entry(name string) (Entry, error) {
	entries, err := m.resolve()
	if err != nil {
		return Entry{}, err
	}
	e, ok := entries[name]
	if !ok {
		return Entry{}, fmt.Errorf("no MCP server named %s; /mcp list shows them", name)
	}
	return e, nil
}

// Start begins starting name in the background and returns at once.
func (m *Manager) Start(name string) error {
	e, err := m.Entry(name)
	if err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.unsupported != "" {
		return errors.New(m.unsupported)
	}
	if s := m.servers[name]; s != nil {
		return fmt.Errorf("%s is already %s; /mcp restart %s starts it again", name, s.state, name)
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &server{entry: e, state: Starting, cancel: cancel}
	m.servers[name] = s
	m.order = append(slices.DeleteFunc(m.order, func(n string) bool { return n == name }), name)
	go m.run(ctx, s)
	return nil
}

func (m *Manager) run(ctx context.Context, s *server) {
	e := s.entry
	notify := func(method string) {
		if method == "notifications/tools/list_changed" {
			m.mu.Lock()
			s.listChanged = true
			m.mu.Unlock()
		}
	}
	var (
		cl  *Client
		err error
	)
	if e.URL != "" {
		cl, err = Dial(ctx, e.URL, e.Headers, notify)
	} else {
		cl, err = Connect(ctx, e.Binary, e.Args, m.env, notify)
	}
	var tools []Tool
	if err == nil {
		if tools, err = cl.Tools(ctx); err != nil {
			cl.Close()
		}
	}

	m.mu.Lock()
	if m.servers[e.Name] != s {
		// Stopped while starting.
		m.mu.Unlock()
		if err == nil {
			cl.Close()
		}
		return
	}
	var ex []exposed
	if err == nil {
		if ex, err = m.expose(e.Name, tools); err != nil {
			cl.Close()
		}
	}
	r := Report{Name: e.Name, Err: err}
	if err != nil {
		s.state, s.detail = Failed, err.Error()
	} else {
		s.state, s.client, s.tools = Active, cl, ex
		r.Tools, r.SchemaChars = len(ex), schemaChars(ex)
	}
	close(m.changed)
	m.changed = make(chan struct{})
	m.mu.Unlock()
	m.emit(r)
}

// expose names a server's tools for the model as <server>_<tool>. Called
// with m.mu held.
func (m *Manager) expose(server string, tools []Tool) ([]exposed, error) {
	taken := map[string]bool{}
	for _, n := range m.order {
		if s := m.servers[n]; s != nil {
			for _, t := range s.tools {
				taken[t.name] = true
			}
		}
	}
	var out []exposed
	for _, t := range tools {
		name := toolName(server, t.Name)
		switch {
		case m.reserved[name]:
			return nil, fmt.Errorf("its tool %s would be called %s, which is a built-in tool", t.Name, name)
		case taken[name]:
			return nil, fmt.Errorf("its tool %s would be called %s, which another MCP tool already is", t.Name, name)
		}
		taken[name] = true
		schema := t.InputSchema
		if len(schema) == 0 || string(schema) == "null" {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		out = append(out, exposed{name: name, original: t.Name, def: tool.Definition{
			Name: name, Description: t.Description, Schema: schema, ReadOnly: t.Annotations.ReadOnlyHint,
		}})
	}
	return out, nil
}

// toolName fits a tool name to what model APIs accept: [A-Za-z0-9_-], at
// most 64 characters.
func toolName(server, name string) string {
	var b strings.Builder
	b.WriteString(server)
	b.WriteByte('_')
	for _, r := range name {
		if r < utf8.RuneSelf && (r == '_' || r == '-' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	s := b.String()
	if len(s) > 64 {
		s = s[:64]
	}
	return s
}

func schemaChars(ex []exposed) int {
	n := 0
	for _, t := range ex {
		n += len(t.def.Name) + len(t.def.Description) + len(t.def.Schema)
	}
	return n
}

// Stop ends name, starting or running, and takes its tools away.
func (m *Manager) Stop(name string) error {
	m.mu.Lock()
	s := m.servers[name]
	if s == nil {
		m.mu.Unlock()
		return fmt.Errorf("%s is not running", name)
	}
	delete(m.servers, name)
	m.order = slices.DeleteFunc(m.order, func(n string) bool { return n == name })
	close(m.changed)
	m.changed = make(chan struct{})
	m.mu.Unlock()
	s.cancel()
	if s.client != nil {
		s.client.Close()
	}
	return nil
}

// Restart stops name if it is running and starts it again, fetching its
// tool list afresh. It keeps its place in the start order.
func (m *Manager) Restart(name string) error {
	if _, err := m.Entry(name); err != nil {
		return err
	}
	m.mu.Lock()
	pos := slices.Index(m.order, name)
	m.mu.Unlock()
	_ = m.Stop(name)
	if err := m.Start(name); err != nil {
		return err
	}
	if pos >= 0 {
		m.mu.Lock()
		m.order = slices.DeleteFunc(m.order, func(n string) bool { return n == name })
		m.order = slices.Insert(m.order, min(pos, len(m.order)), name)
		m.mu.Unlock()
	}
	return nil
}

// Starting names the servers still starting, in start order.
func (m *Manager) Starting() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []string
	for _, n := range m.order {
		if m.servers[n].state == Starting {
			out = append(out, n)
		}
	}
	return out
}

// Settle waits, without a time limit, until no server is starting, calling
// waiting with the names it is waiting for each time they change and with
// nil when it stops waiting.
func (m *Manager) Settle(ctx context.Context, waiting func([]string)) error {
	shown := false
	defer func() {
		if shown {
			waiting(nil)
		}
	}()
	for {
		m.mu.Lock()
		changed := m.changed
		m.mu.Unlock()
		names := m.Starting()
		if len(names) == 0 {
			return nil
		}
		waiting(names)
		shown = true
		select {
		case <-changed:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// List describes every configured name, with its state.
func (m *Manager) List() ([]Row, error) {
	entries, err := m.resolve()
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	names := make([]string, 0, len(entries))
	for n := range entries {
		names = append(names, n)
	}
	for n := range m.servers {
		if _, ok := entries[n]; !ok {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	var rows []Row
	for _, n := range names {
		row := Row{Entry: entries[n], State: Inactive}
		if s := m.servers[n]; s != nil {
			// A running server is shown as it was started.
			row.Entry, row.State, row.Detail = s.entry, s.state, s.detail
			row.Tools, row.SchemaChars = len(s.tools), schemaChars(s.tools)
			if s.state == Active && s.client.Err() != nil {
				row.State, row.Detail = Failed, s.client.Err().Error()
			} else if s.listChanged {
				row.Detail = "its tool list changed; /mcp restart " + n + " fetches it"
			}
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// Running reports whether name is starting, active or failed after starting.
func (m *Manager) Running(name string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.servers[name] != nil
}

// RestartArgs are the --mcp flags that bring back the servers asked for and
// not stopped, in start order: a failed one too, since it was asked for.
func (m *Manager) RestartArgs() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []string
	for _, n := range m.order {
		e := m.servers[n].entry
		if e.Source == "cli" {
			out = append(out, "--mcp", n+"="+e.Value)
		} else {
			out = append(out, "--mcp", n)
		}
	}
	return out
}

// Definitions are the tools of started servers, in start order. A server
// that has exited keeps its tools until it is stopped or restarted, so the
// request does not change under the model.
func (m *Manager) Definitions() []tool.Definition {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []tool.Definition
	for _, n := range m.order {
		for _, t := range m.servers[n].tools {
			out = append(out, t.def)
		}
	}
	return out
}

func (m *Manager) lookup(name string) (*server, exposed, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, n := range m.order {
		s := m.servers[n]
		for _, t := range s.tools {
			if t.name == name {
				return s, t, true
			}
		}
	}
	return nil, exposed{}, false
}

func (m *Manager) call(ctx context.Context, req tool.Request, progress tool.Progress) (tool.Result, bool) {
	s, t, ok := m.lookup(req.Name)
	if !ok {
		return tool.Result{}, false
	}
	var note func(string)
	if progress != nil {
		note = func(n string) {
			if n != "" {
				progress(tool.Phase{Note: s.entry.Name + ": " + n})
			}
		}
	}
	res := s.client.Call(ctx, t.original, req.Args, note)
	if err := s.client.Err(); err != nil && res.IsError && !res.Interrupted {
		res.Content, res.Display = s.entry.Name+" "+err.Error(), ""
	}
	if res.Display == "" {
		res.Display = req.Name
		if line := firstLine(strings.TrimSpace(res.Content)); line != "" {
			if utf8.RuneCountInString(line) > 100 {
				line = string([]rune(line)[:100]) + "…"
			}
			res.Display += "  " + line
		}
	}
	return res, true
}

// Close stops every server.
func (m *Manager) Close() {
	m.mu.Lock()
	names := slices.Clone(m.order)
	m.mu.Unlock()
	for _, n := range names {
		_ = m.Stop(n)
	}
}

// Executor adds the started servers' tools to an executor chain.
type Executor struct {
	inner    tool.Executor
	m        *Manager
	progress tool.Progress
}

func Wrap(inner tool.Executor, m *Manager) *Executor { return &Executor{inner: inner, m: m} }

func (e *Executor) Definitions() []tool.Definition {
	return append(e.inner.Definitions(), e.m.Definitions()...)
}

func (e *Executor) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	if !req.Cancel {
		if res, ok := e.m.call(ctx, req, e.progress); ok {
			return res, nil
		}
	}
	return e.inner.Execute(ctx, req)
}

func (e *Executor) Fork() (tool.Executor, error) {
	inner, err := e.inner.Fork()
	if err != nil {
		return nil, err
	}
	return Wrap(inner, e.m), nil
}

func (e *Executor) IsReadOnly(name string) bool {
	if _, t, ok := e.m.lookup(name); ok {
		return t.def.ReadOnly
	}
	if ro, ok := e.inner.(interface{ IsReadOnly(string) bool }); ok {
		return ro.IsReadOnly(name)
	}
	return false
}

func (e *Executor) SetProgress(p tool.Progress) {
	e.progress = p
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
