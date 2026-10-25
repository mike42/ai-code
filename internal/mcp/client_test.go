package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"ai-code/internal/mcp/hello"
)

// helloBin is the mcphello command, built once for the package's tests.
var helloBin string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "mcphello")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	helloBin = filepath.Join(dir, "mcphello")
	if runtime.GOOS == "windows" {
		helloBin += ".exe"
	}
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if out, err := exec.Command(goTool, "build", "-o", helloBin, "./mcphello").CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "building mcphello: %v\n%s", err, out)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func connectHello(t *testing.T, line string, notify func(string)) *Client {
	t.Helper()
	words, err := Split(line)
	if err != nil {
		t.Fatal(err)
	}
	cl, err := Connect(context.Background(), helloBin, words, os.Environ(), notify)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cl.Close() })
	return cl
}

func toolNames(ts []Tool) []string {
	var n []string
	for _, t := range ts {
		n = append(n, t.Name)
	}
	return n
}

func TestStdioListsEveryPage(t *testing.T) {
	cl := connectHello(t, "", nil)
	if cl.ServerName != "hello" || cl.Version != ProtocolVersion {
		t.Fatalf("server %q version %q", cl.ServerName, cl.Version)
	}
	tools, err := cl.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(toolNames(tools), ","); got != "hello,wait,crash" {
		t.Fatalf("tools = %s", got)
	}
	if !tools[0].Annotations.ReadOnlyHint || tools[1].Annotations.ReadOnlyHint {
		t.Fatal("readOnlyHint not carried")
	}
	want := `{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`
	if string(tools[0].InputSchema) != want {
		t.Fatalf("schema altered: %s", tools[0].InputSchema)
	}
}

func TestStdioArgumentsArriveAsSplit(t *testing.T) {
	cl := connectHello(t, `--greeting "good morning" --root=/my\ dir`, nil)
	res := cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"world"}`), nil)
	if res.IsError || res.Content != "good morning, world (root /my dir)" {
		t.Fatalf("result = %+v", res)
	}
}

func TestStdioEnvironmentReachesTheServer(t *testing.T) {
	cl, err := Connect(context.Background(), helloBin, nil, append(os.Environ(), "HELLO_ROOT=/work"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	res := cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"x"}`), nil)
	if !strings.Contains(res.Content, "(root /work)") {
		t.Fatalf("result = %+v", res)
	}
}

func TestStdioProgressAndCancel(t *testing.T) {
	cl := connectHello(t, "", nil)
	ctx, cancel := context.WithCancel(context.Background())
	var notes []string
	var mu sync.Mutex
	res := cl.Call(ctx, "wait", nil, func(n string) {
		mu.Lock()
		notes = append(notes, n)
		mu.Unlock()
		cancel()
	})
	if !res.Interrupted {
		t.Fatalf("result = %+v", res)
	}
	if len(notes) != 1 || notes[0] != "waiting (1/2)" {
		t.Fatalf("progress = %q", notes)
	}
	// The server took the cancellation and is still serving.
	if r := cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"again"}`), nil); r.Content != "Hello, again" {
		t.Fatalf("after cancel: %+v", r)
	}
}

func TestStdioCrashIsAnErrorResult(t *testing.T) {
	cl := connectHello(t, "", nil)
	res := cl.Call(context.Background(), "crash", json.RawMessage(`{"status":3}`), nil)
	if !res.IsError || !strings.Contains(res.Content, "exited (status 3)") {
		t.Fatalf("result = %+v", res)
	}
	if cl.Err() == nil {
		t.Fatal("connection still up after the server exited")
	}
	res = cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"x"}`), nil)
	if !res.IsError || !strings.Contains(res.Content, "exited (status 3)") {
		t.Fatalf("call after crash = %+v", res)
	}
}

func TestStdioExitBeforeHandshakeNamesStatusAndStderr(t *testing.T) {
	_, err := Connect(context.Background(), helloBin, []string{"--no-such-flag"}, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "exited (status 2)") || !strings.Contains(err.Error(), "bearer token") {
		t.Fatalf("err = %v", err)
	}
}

func TestStdioMissingBinary(t *testing.T) {
	_, err := Connect(context.Background(), filepath.Join(t.TempDir(), "nope"), nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("err = %v", err)
	}
}

func TestWindowsBatchFilesAreRefused(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("batch files run through cmd.exe only on Windows")
	}
	for _, ext := range []string{".cmd", ".bat"} {
		path := filepath.Join(t.TempDir(), "server"+ext)
		if err := os.WriteFile(path, []byte("@echo off\r\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		_, err := Connect(context.Background(), path, nil, nil, nil)
		if err == nil || !strings.Contains(err.Error(), "without a shell") {
			t.Errorf("%s: err = %v", ext, err)
		}
	}
}

func TestStdioListChangedIsReported(t *testing.T) {
	got := make(chan string, 1)
	cl := connectHello(t, "--list-changes", func(m string) { got <- m })
	cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"x"}`), nil)
	if m := <-got; m != "notifications/tools/list_changed" {
		t.Fatalf("notification %q", m)
	}
	tools, _ := cl.Tools(context.Background())
	if !strings.Contains(strings.Join(toolNames(tools), ","), "goodbye") {
		t.Fatal("new tool not listed")
	}
}

func TestStdioHeldHandshakeWaitsForRelease(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	done := make(chan error, 1)
	go func() {
		cl, err := Connect(context.Background(), helloBin, []string{"--hold", ln.Addr().String()}, nil, nil)
		if err == nil {
			cl.Close()
		}
		done <- err
	}()
	held, err := ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		t.Fatalf("handshake finished while held: %v", err)
	default:
	}
	held.Write([]byte{1})
	held.Close()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestStdioCancelledHandshake(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := Connect(ctx, helloBin, []string{"--hold", ln.Addr().String()}, nil, nil)
		done <- err
	}()
	held, err := ln.Accept()
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	cancel()
	if err := <-done; err == nil {
		t.Fatal("cancelled handshake succeeded")
	}
}

func TestHTTPOverSSEAndJSON(t *testing.T) {
	for _, jsonOnly := range []bool{false, true} {
		t.Run(fmt.Sprint("json=", jsonOnly), func(t *testing.T) {
			var deleted bool
			var mu sync.Mutex
			h := hello.Handler(hello.Options{Token: "secret"}, jsonOnly)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodDelete && r.Header.Get("Mcp-Session-Id") != "" {
					mu.Lock()
					deleted = true
					mu.Unlock()
				}
				if r.Method == http.MethodPost && !strings.Contains(string(mustRead(r)), `"initialize"`) &&
					r.Header.Get("MCP-Protocol-Version") != ProtocolVersion {
					http.Error(w, "no protocol version header", http.StatusBadRequest)
					return
				}
				h.ServeHTTP(w, r)
			}))
			defer srv.Close()

			cl, err := Dial(context.Background(), srv.URL, map[string]string{"Authorization": "Bearer secret"}, nil)
			if err != nil {
				t.Fatal(err)
			}
			tools, err := cl.Tools(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(toolNames(tools), ","); got != "hello,wait" {
				t.Fatalf("tools = %s", got)
			}
			res := cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"web"}`), nil)
			if res.IsError || res.Content != "Hello, web" {
				t.Fatalf("result = %+v", res)
			}
			cl.Close()
			mu.Lock()
			defer mu.Unlock()
			if !deleted {
				t.Fatal("session not ended with DELETE")
			}
		})
	}
}

// mustRead reads a request body and puts it back for the handler.
func mustRead(r *http.Request) []byte {
	b, _ := io.ReadAll(r.Body)
	r.Body = io.NopCloser(bytes.NewReader(b))
	return b
}

func TestHTTPProgressAndCancel(t *testing.T) {
	srv := httptest.NewServer(hello.Handler(hello.Options{}, false))
	defer srv.Close()
	cl, err := Dial(context.Background(), srv.URL, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	ctx, cancel := context.WithCancel(context.Background())
	res := cl.Call(ctx, "wait", nil, func(string) { cancel() })
	if !res.Interrupted {
		t.Fatalf("result = %+v", res)
	}
	if r := cl.Call(context.Background(), "hello", json.RawMessage(`{"name":"x"}`), nil); r.Content != "Hello, x" {
		t.Fatalf("after cancel: %+v", r)
	}
}

func TestHTTPRefusalNamesTheStatus(t *testing.T) {
	srv := httptest.NewServer(hello.Handler(hello.Options{Token: "secret"}, false))
	defer srv.Close()
	_, err := Dial(context.Background(), srv.URL, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("err = %v", err)
	}
}

func TestHTTPConnectionRefused(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := ln.Addr().String()
	ln.Close()
	_, err := Dial(context.Background(), "http://"+addr+"/mcp", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "refused") {
		t.Fatalf("err = %v", err)
	}
}
