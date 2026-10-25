package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http/httptest"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"sync"
	"testing"

	"ai-code/internal/mcp/hello"
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

// workerLauncher starts the real worker binary on this machine, in place of
// `docker exec ... worker` or `ssh ... worker`, with the opening message the
// worker protocol defines. launched records every command it made.
func workerLauncher(launched *[]*exec.Cmd, mu *sync.Mutex) Launcher {
	return func(ctx context.Context, e Entry) (*exec.Cmd, []byte, error) {
		cmd := exec.Command(workerBin)
		mu.Lock()
		*launched = append(*launched, cmd)
		mu.Unlock()
		preface, err := json.Marshal(map[string]any{"settings": map[string]any{}, "mcp": LaunchFor(e)})
		return cmd, append(preface, '\n'), err
	}
}

func TestManagerRunsServersThroughTheLauncher(t *testing.T) {
	entries := map[string]Entry{
		"h":    {Name: "h", Source: "cli", Binary: helloBin, Args: []string{"--greeting", `good "day"`, "--root=/my dir"}},
		"gone": {Name: "gone", Source: "local", Binary: "/no/such/mcp-server"},
	}
	m, r := newManager(t, entries)
	var (
		launched []*exec.Cmd
		mu       sync.Mutex
	)
	m.SetLauncher(workerLauncher(&launched, &mu))
	exe := Wrap(nullExec{}, m)
	m.Start("h")
	m.Settle(context.Background(), func([]string) {})
	if rep := <-r.ch; rep.Err != nil {
		t.Fatal(rep.Err)
	}
	mu.Lock()
	if len(launched) != 1 || len(launched[0].Args) != 1 {
		t.Fatalf("the launcher's command carried arguments: %q", launched[0].Args)
	}
	mu.Unlock()
	res, _ := exe.Execute(context.Background(), tool.Request{Name: "h_hello", Args: json.RawMessage(`{"name":"you"}`)})
	if res.Content != `good "day", you (root /my dir)` {
		t.Fatalf("arguments did not arrive as a list: %+v", res)
	}

	// Progress and cancellation cross the worker too.
	ctx, cancel := context.WithCancel(context.Background())
	var notes []string
	exe.SetProgress(func(p tool.Phase) { notes = append(notes, p.Note); cancel() })
	if res, _ := exe.Execute(ctx, tool.Request{Name: "h_wait"}); !res.Interrupted {
		t.Fatalf("wait = %+v", res)
	}
	if len(notes) != 1 || notes[0] != "h: waiting (1/2)" {
		t.Fatalf("progress = %q", notes)
	}
	exe.SetProgress(nil)

	// The server's own exit status comes back through the worker.
	res, _ = exe.Execute(context.Background(), tool.Request{Name: "h_crash", Args: json.RawMessage(`{"status":3}`)})
	if !res.IsError || res.Content != "h exited (status 3)" {
		t.Fatalf("crash = %+v", res)
	}

	m.Start("gone")
	rep := <-r.ch
	if rep.Err == nil || !strings.Contains(rep.Err.Error(), "exited (status 127): /no/such/mcp-server: not found") {
		t.Fatalf("missing binary: %v", rep.Err)
	}
}

func TestManagerReachesURLsThroughTheLauncher(t *testing.T) {
	srv := httptest.NewServer(hello.Handler(hello.Options{Token: "secret"}, false))
	defer srv.Close()
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	refused := "http://" + ln.Addr().String() + "/mcp"
	ln.Close()
	entries := map[string]Entry{
		"web":  {Name: "web", Source: "local", URL: srv.URL, Headers: map[string]string{"Authorization": "Bearer secret"}},
		"down": {Name: "down", Source: "local", URL: refused},
	}
	m, r := newManager(t, entries)
	var (
		launched []*exec.Cmd
		mu       sync.Mutex
	)
	m.SetLauncher(workerLauncher(&launched, &mu))
	exe := Wrap(nullExec{}, m)
	m.Start("web")
	if rep := <-r.ch; rep.Err != nil || rep.Tools != 2 {
		t.Fatalf("report = %+v", rep)
	}
	res, _ := exe.Execute(context.Background(), tool.Request{Name: "web_hello", Args: json.RawMessage(`{"name":"net"}`)})
	if res.IsError || res.Content != "Hello, net" {
		t.Fatalf("result = %+v", res)
	}
	ctx, cancel := context.WithCancel(context.Background())
	exe.SetProgress(func(tool.Phase) { cancel() })
	if res, _ := exe.Execute(ctx, tool.Request{Name: "web_wait"}); !res.Interrupted {
		t.Fatalf("wait = %+v", res)
	}
	exe.SetProgress(nil)
	if res, _ := exe.Execute(context.Background(), tool.Request{Name: "web_hello", Args: json.RawMessage(`{"name":"again"}`)}); res.Content != "Hello, again" {
		t.Fatalf("after cancel = %+v", res)
	}

	m.Start("down")
	rep := <-r.ch
	if rep.Err == nil || !strings.Contains(rep.Err.Error(), "refused") {
		t.Fatalf("refused: %v", rep.Err)
	}
}

// A server that ignores the end of its stdin is still ended when the worker's
// stdin closes: the worker terminates the server's process group.
func TestRelayEndsAServerThatIgnoresItsStdin(t *testing.T) {
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan int, 1)
	go func() { done <- Relay(inR, outW, io.Discard, t.TempDir(), Launch{Binary: helloBin}); outW.Close() }()
	go io.Copy(io.Discard, outR)
	for _, m := range []string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"wait"}}`,
	} {
		inW.Write([]byte(m + "\n"))
	}
	inW.Close()
	if code := <-done; code == 0 {
		t.Fatalf("relay ended with status 0; the server should have been terminated")
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
