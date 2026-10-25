package mcp

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"ai-code/internal/tool"
)

type nullExec struct{}

func (nullExec) Definitions() []tool.Definition {
	return []tool.Definition{{Name: "bash", Schema: json.RawMessage(`{}`)}}
}
func (nullExec) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	return tool.Result{Content: "inner " + req.Name}, nil
}
func (nullExec) Fork() (tool.Executor, error) { return nullExec{}, nil }

type reports struct {
	mu  sync.Mutex
	got []Report
	ch  chan Report
}

func newManager(t *testing.T, entries map[string]Entry) (*Manager, *reports) {
	t.Helper()
	m := NewManager(func() (map[string]Entry, error) { return entries, nil }, os.Environ())
	m.SetReserved([]string{"bash", "hello_hello2"})
	r := &reports{ch: make(chan Report, 16)}
	m.SetReporter(func(rep Report) {
		r.mu.Lock()
		r.got = append(r.got, rep)
		r.mu.Unlock()
		r.ch <- rep
	})
	t.Cleanup(m.Close)
	return m, r
}

func helloEntry(name string, args ...string) Entry {
	return Entry{Name: name, Source: "local", Binary: helloBin, Args: args}
}

func defNames(defs []tool.Definition) string {
	var n []string
	for _, d := range defs {
		n = append(n, d.Name)
	}
	return strings.Join(n, ",")
}

func TestManagerStartsInBackgroundAndAppendsInStartOrder(t *testing.T) {
	m, r := newManager(t, map[string]Entry{"b": helloEntry("b"), "a": helloEntry("a")})
	exe := Wrap(nullExec{}, m)
	if err := m.Start("b"); err != nil {
		t.Fatal(err)
	}
	if err := m.Start("a"); err != nil {
		t.Fatal(err)
	}
	if err := m.Settle(context.Background(), func([]string) {}); err != nil {
		t.Fatal(err)
	}
	if got := defNames(exe.Definitions()); got != "bash,b_hello,b_wait,b_crash,a_hello,a_wait,a_crash" {
		t.Fatalf("tools = %s", got)
	}
	<-r.ch
	rep := <-r.ch
	if rep.Err != nil || rep.Tools != 3 || rep.SchemaChars == 0 {
		t.Fatalf("report = %+v", rep)
	}
	res, err := exe.Execute(context.Background(), tool.Request{Name: "a_hello", Args: json.RawMessage(`{"name":"you"}`)})
	if err != nil || res.Content != "Hello, you" || res.Display != "a_hello  Hello, you" {
		t.Fatalf("result = %+v, %v", res, err)
	}
	if !exe.IsReadOnly("a_hello") || exe.IsReadOnly("a_wait") {
		t.Fatal("readOnlyHint not applied")
	}
	if res, _ := exe.Execute(context.Background(), tool.Request{Name: "bash"}); res.Content != "inner bash" {
		t.Fatalf("built-in call = %+v", res)
	}
}

func TestManagerSettleWaitsForAHeldStartAndStopReleasesIt(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	m, _ := newManager(t, map[string]Entry{"slow": helloEntry("slow", "--hold", ln.Addr().String())})
	if err := m.Start("slow"); err != nil {
		t.Fatal(err)
	}
	held, err := ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()

	waiting := make(chan []string, 4)
	done := make(chan error, 1)
	go func() { done <- m.Settle(context.Background(), func(n []string) { waiting <- n }) }()
	if got := <-waiting; !reflect.DeepEqual(got, []string{"slow"}) {
		t.Fatalf("waiting for %q", got)
	}
	if err := m.Stop("slow"); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if got := <-waiting; got != nil {
		t.Fatalf("waiting not cleared: %q", got)
	}
	if m.Running("slow") || len(m.Definitions()) != 0 {
		t.Fatal("stopped server still present")
	}
}

func TestManagerSettleIsCancellable(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	m, _ := newManager(t, map[string]Entry{"slow": helloEntry("slow", "--hold", ln.Addr().String())})
	m.Start("slow")
	held, _ := ln.Accept()
	defer held.Close()
	ctx, cancel := context.WithCancel(context.Background())
	err = m.Settle(ctx, func([]string) { cancel() })
	if err != context.Canceled {
		t.Fatalf("err = %v", err)
	}
}

func TestManagerCrashKeepsToolsAndNamesTheServer(t *testing.T) {
	m, _ := newManager(t, map[string]Entry{"h": helloEntry("h")})
	exe := Wrap(nullExec{}, m)
	m.Start("h")
	m.Settle(context.Background(), func([]string) {})
	before := defNames(exe.Definitions())
	res, _ := exe.Execute(context.Background(), tool.Request{Name: "h_crash", Args: json.RawMessage(`{"status":1}`)})
	if !res.IsError || res.Content != "h exited (status 1)" {
		t.Fatalf("crash result = %+v", res)
	}
	res, _ = exe.Execute(context.Background(), tool.Request{Name: "h_hello", Args: json.RawMessage(`{"name":"x"}`)})
	if !res.IsError || res.Content != "h exited (status 1)" || res.Display != "h_hello  h exited (status 1)" {
		t.Fatalf("after crash = %+v", res)
	}
	if after := defNames(exe.Definitions()); after != before {
		t.Fatalf("tools changed after crash: %s -> %s", before, after)
	}
	rows, _ := m.List()
	if rows[0].State != Failed || rows[0].Detail != "exited (status 1)" {
		t.Fatalf("row = %+v", rows[0])
	}
	if err := m.Restart("h"); err != nil {
		t.Fatal(err)
	}
	m.Settle(context.Background(), func([]string) {})
	if res, _ := exe.Execute(context.Background(), tool.Request{Name: "h_hello", Args: json.RawMessage(`{"name":"x"}`)}); res.IsError {
		t.Fatalf("after restart = %+v", res)
	}
}

func TestManagerRefusesAClashWithABuiltIn(t *testing.T) {
	m, r := newManager(t, map[string]Entry{"hello": helloEntry("hello")})
	m.SetReserved([]string{"hello_wait"})
	m.Start("hello")
	rep := <-r.ch
	if rep.Err == nil || !strings.Contains(rep.Err.Error(), "hello_wait, which is a built-in tool") {
		t.Fatalf("report = %+v", rep)
	}
	if len(m.Definitions()) != 0 {
		t.Fatal("clashing tools exposed")
	}
}

func TestManagerFailedStartIsReportedAndListed(t *testing.T) {
	m, r := newManager(t, map[string]Entry{"gone": {Name: "gone", Source: "global", Binary: "/no/such/mcp-server"}})
	m.Start("gone")
	rep := <-r.ch
	if rep.Err == nil || !strings.Contains(rep.Err.Error(), "/no/such/mcp-server") {
		t.Fatalf("report = %+v", rep)
	}
	rows, _ := m.List()
	if rows[0].State != Failed || !strings.Contains(rows[0].Detail, "/no/such/mcp-server") {
		t.Fatalf("row = %+v", rows[0])
	}
}

func TestManagerRestartArgs(t *testing.T) {
	entries := map[string]Entry{
		"cli":     {Name: "cli", Source: "cli", Value: helloBin + ` --greeting "good day"`, Binary: helloBin, Args: []string{"--greeting", "good day"}},
		"local":   helloEntry("local"),
		"stopped": helloEntry("stopped"),
		"broken":  {Name: "broken", Source: "global", Binary: "/no/such/mcp-server"},
		"idle":    helloEntry("idle"),
	}
	m, _ := newManager(t, entries)
	for _, n := range []string{"local", "cli", "stopped", "broken"} {
		if err := m.Start(n); err != nil {
			t.Fatal(err)
		}
	}
	m.Settle(context.Background(), func([]string) {})
	m.Stop("stopped")
	want := []string{"--mcp", "local", "--mcp", "cli=" + helloBin + ` --greeting "good day"`, "--mcp", "broken"}
	if got := m.RestartArgs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %q\nwant %q", got, want)
	}
	m.Restart("local")
	if got := m.RestartArgs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("restart moved local: %q", got)
	}
}

func TestManagerUnsupportedRuntime(t *testing.T) {
	m, _ := newManager(t, map[string]Entry{"h": helloEntry("h")})
	m.SetUnsupported("not here")
	if err := m.Start("h"); err == nil || err.Error() != "not here" {
		t.Fatalf("err = %v", err)
	}
}

func TestManagerHoldsReportsUntilAReporterIsSet(t *testing.T) {
	m := NewManager(func() (map[string]Entry, error) { return map[string]Entry{"h": helloEntry("h")}, nil }, os.Environ())
	defer m.Close()
	m.Start("h")
	m.Settle(context.Background(), func([]string) {})
	got := make(chan Report, 1)
	m.SetReporter(func(r Report) { got <- r })
	if r := <-got; r.Name != "h" || r.Err != nil {
		t.Fatalf("report = %+v", r)
	}
}

func TestToolNameFitsAPIRules(t *testing.T) {
	if got := toolName("lldb", "set.break point"); got != "lldb_set_break_point" {
		t.Fatal(got)
	}
	if got := toolName("s", strings.Repeat("x", 80)); len(got) != 64 {
		t.Fatal(len(got))
	}
}
