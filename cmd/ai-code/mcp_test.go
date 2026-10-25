package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"sync"
	"testing"

	"ai-code/internal/config"
	"ai-code/internal/devcontainer"
	"ai-code/internal/mcp"
	"ai-code/internal/provider"
	"ai-code/internal/runtime"
	"ai-code/internal/session"
	"ai-code/internal/tool"
	"ai-code/internal/worker"
)

var (
	binsOnce  sync.Once
	helloPath string
	// workerPath is the ai-code worker, standing in for one started by
	// docker exec or ssh in the target environment.
	workerPath string
	binsErr    error
)

func buildBins(t *testing.T) {
	t.Helper()
	binsOnce.Do(func() {
		dir, err := os.MkdirTemp("", "mcphello")
		if err != nil {
			binsErr = err
			return
		}
		exe := ""
		if goruntime.GOOS == "windows" {
			exe = ".exe"
		}
		helloPath = filepath.Join(dir, "mcphello"+exe)
		workerPath = filepath.Join(dir, "ai-code-worker"+exe)
		goTool := filepath.Join(goruntime.GOROOT(), "bin", "go")
		for out, pkg := range map[string]string{helloPath: "ai-code/internal/mcp/mcphello", workerPath: "ai-code/cmd/ai-code-worker"} {
			if msg, err := exec.Command(goTool, "build", "-o", out, pkg).CombinedOutput(); err != nil {
				binsErr = err
				t.Logf("%s", msg)
				return
			}
		}
	})
	if binsErr != nil {
		t.Fatal(binsErr)
	}
}

// helloBinary is the mcphello stub, built once for these tests.
func helloBinary(t *testing.T) string {
	buildBins(t)
	return helloPath
}

// mcpApp is an App whose global config is in its own directory and whose
// working directory holds the local config.
func mcpApp(t *testing.T, cli map[string]mcp.Entry) (a *App, global, local string) {
	t.Helper()
	globalDir := t.TempDir()
	t.Setenv("AI_CODE_CONFIG_DIR", globalDir)
	a = statusApp(t, provider.ModelInfo{ID: "a-model", ContextWindow: 262144})
	a.cwd = t.TempDir()
	a.flags = &flags{}
	if cli == nil {
		cli = map[string]mcp.Entry{}
	}
	a.mcp = mcp.NewManager(mcpResolver(a.cwd, cli), os.Environ())
	a.mcp.SetReporter(func(mcp.Report) {})
	t.Cleanup(a.mcp.Close)
	return a, filepath.Join(globalDir, "config.toml"), config.ProjectConfigPath(a.cwd)
}

func settle(t *testing.T, a *App) {
	t.Helper()
	if err := a.mcp.Settle(context.Background(), func([]string) {}); err != nil {
		t.Fatal(err)
	}
}

func TestParseMCPFlags(t *testing.T) {
	cfg := &config.Config{MCP: map[string]config.MCPServer{"lldb": {Binary: "/usr/bin/lldb-mcp", Source: "local"}}}
	s, err := parseMCPFlags([]string{
		"lldb",
		`hello=/opt/mcp/hello --greeting "good morning"`,
		"blender=http://10.0.0.5:9999",
	}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(s.start, ",") != "lldb,hello,blender" {
		t.Fatalf("start = %q", s.start)
	}
	h := s.cli["hello"]
	if h.Binary != "/opt/mcp/hello" || strings.Join(h.Args, "|") != "--greeting|good morning" || h.Value != `/opt/mcp/hello --greeting "good morning"` {
		t.Fatalf("hello = %+v", h)
	}
	if b := s.cli["blender"]; b.URL != "http://10.0.0.5:9999" || b.Binary != "" {
		t.Fatalf("blender = %+v", b)
	}
	if _, ok := s.cli["lldb"]; ok {
		t.Fatal("a bare name became a cli entry")
	}

	for value, want := range map[string]string{
		"nope":            "no MCP server named nope is configured",
		"x=":              "needs a URL, or a binary",
		`x=/bin/a "b`:     `unterminated " at column 8`,
		"bad_name=/bin/a": "letters, digits and -",
	} {
		if _, err := parseMCPFlags([]string{value}, cfg); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("--mcp %s: err = %v, want %q", value, err, want)
		}
	}
	if _, err := parseMCPFlags([]string{"lldb", "lldb=/x"}, cfg); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Errorf("duplicate: %v", err)
	}
}

func TestMCPListShowsTheWinnerAndWhereItLives(t *testing.T) {
	cli := map[string]mcp.Entry{"probe": {Name: "probe", Source: "cli", Value: "/opt/probe", Binary: "/opt/probe"}}
	a, global, local := mcpApp(t, cli)
	writeConfig(t, global, "[mcp.lldb]\nbinary = \"/usr/bin/lldb-mcp\"\n\n[mcp.probe]\nurl = \"http://10.0.0.9/mcp\"\n\n[mcp.blender]\nurl = \"http://10.0.0.5:9999\"\n")
	writeConfig(t, local, "[mcp.lldb]\nbinary = \"/opt/lldb-mcp\"\n")

	out := captureOut(t, func() {
		if err := a.cmdMCP(context.Background(), "list"); err != nil {
			t.Error(err)
		}
	})
	for _, want := range []string{
		"blender  global",
		"lldb     local, hides global  inactive  /opt/lldb-mcp",
		"probe    cli, hides global",
		"global  " + global,
		"local   " + local,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("list lacks %q:\n%s", want, out)
		}
	}
}

func TestMCPAddSavesThenStartsAndRemoveStopsThenDeletes(t *testing.T) {
	bin := helloBinary(t)
	a, _, local := mcpApp(t, nil)
	writeConfig(t, local, "# mine\n[ui]\ntheme = \"none\"\n")

	if err := a.cmdMCP(context.Background(), "add hello="+bin+` --greeting "good day"`); err != nil {
		t.Fatal(err)
	}
	settle(t, a)
	if got := names(a.mcp.Definitions()); got != "hello_hello,hello_wait,hello_crash" {
		t.Fatalf("tools after add = %s", got)
	}
	text, _ := os.ReadFile(local)
	if !strings.HasPrefix(string(text), "# mine\n[ui]\ntheme = \"none\"\n") || !strings.Contains(string(text), `args = ["--greeting", "good day"]`) {
		t.Fatalf("config after add:\n%s", text)
	}

	captureStderr(t, func() {
		if err := a.cmdMCP(context.Background(), "remove hello"); err != nil {
			t.Fatal(err)
		}
	})
	if a.mcp.Running("hello") || len(a.mcp.Definitions()) != 0 {
		t.Fatal("removed server still running")
	}
	text, _ = os.ReadFile(local)
	if string(text) != "# mine\n[ui]\ntheme = \"none\"\n\n" {
		t.Fatalf("config after remove:\n%q", text)
	}
}

func TestMCPRemoveRefusesGlobalAndCLIEntries(t *testing.T) {
	cli := map[string]mcp.Entry{"probe": {Name: "probe", Source: "cli", Binary: "/opt/probe"}}
	a, global, _ := mcpApp(t, cli)
	writeConfig(t, global, "[mcp.lldb]\nbinary = \"/usr/bin/lldb-mcp\"\n")
	if err := a.cmdMCP(context.Background(), "remove lldb"); err == nil || !strings.Contains(err.Error(), "global config, "+global) {
		t.Errorf("global: %v", err)
	}
	if err := a.cmdMCP(context.Background(), "remove probe"); err == nil || !strings.Contains(err.Error(), "command line") {
		t.Errorf("cli: %v", err)
	}
	if err := a.cmdMCP(context.Background(), "add probe=/x"); err == nil || !strings.Contains(err.Error(), "command line") {
		t.Errorf("add over cli: %v", err)
	}
}

func TestRestartBringsBackTheServersThatAreRunning(t *testing.T) {
	bin := helloBinary(t)
	cli := map[string]mcp.Entry{"one": {Name: "one", Source: "cli", Value: bin + " --greeting 'hi there'", Binary: bin, Args: []string{"--greeting", "hi there"}}}
	a, _, local := mcpApp(t, cli)
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	sess, err := session.Create(session.Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	a.sess = sess
	writeConfig(t, local, "[mcp.two]\nbinary = "+tomlQuote(bin)+"\n\n[mcp.three]\nbinary = "+tomlQuote(bin)+"\n\n[mcp.idle]\nbinary = "+tomlQuote(bin)+"\n")

	for _, n := range []string{"one", "two", "three"} {
		if err := a.mcp.Start(n); err != nil {
			t.Fatal(err)
		}
	}
	settle(t, a)
	captureStderr(t, func() {
		if err := a.cmdMCP(context.Background(), "stop three"); err != nil {
			t.Fatal(err)
		}
	})
	argv := strings.Join(a.restartArgv("ai-code"), " ")
	want := "--mcp one=" + bin + " --greeting 'hi there' --mcp two"
	if !strings.HasSuffix(argv, want) {
		t.Fatalf("argv = %q\nwant suffix %q", argv, want)
	}

	captureStderr(t, func() {
		if err := a.cmdMCP(context.Background(), "remove two"); err != nil {
			t.Fatal(err)
		}
	})
	if argv := strings.Join(a.restartArgv("ai-code"), " "); strings.Contains(argv, "--mcp two") {
		t.Fatalf("a removed server is carried: %q", argv)
	}
}

func TestRestartArgvUnchangedWithoutMCP(t *testing.T) {
	a, _, _ := mcpApp(t, nil)
	t.Setenv("AI_CODE_DATA_DIR", t.TempDir())
	sess, err := session.Create(session.Meta{Project: "/p"})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()
	a.sess = sess
	if argv := strings.Join(a.restartArgv("ai-code"), " "); strings.Contains(argv, "--mcp") {
		t.Fatalf("argv = %q", argv)
	}
}

func TestMCPCommandsDuringATurn(t *testing.T) {
	a, _, _ := mcpApp(t, nil)
	if !a.mcpImmediate("/mcp") || !a.mcpImmediate("/mcp list") {
		t.Error("listing should never wait for the turn")
	}
	if a.mcpImmediate("/mcp stop lldb") {
		t.Error("a stop during a running request should wait for the boundary")
	}
	a.mcpHolding.Store(true)
	if !a.mcpImmediate("/mcp stop lldb") {
		t.Error("a stop while the request is held should run at once")
	}
	if a.mcpImmediate("/model x") {
		t.Error("not an /mcp command")
	}
}

// The worker in the target environment is started with nothing but its own
// path; the server's binary and arguments travel in its opening message.
func TestMCPWorkerCommandsNameOnlyTheWorker(t *testing.T) {
	rt, err := runtime.ParseFlag("ssh://dev@build-box/srv/proj")
	if err != nil {
		t.Fatal(err)
	}
	ssh := NewSSHExecutor(rt, "", worker.Settings{})
	ssh.remoteBin = "/home/dev/.cache/ai-code/bin/ai-code-worker-0123"
	cmd, err := ssh.MCPCommand(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	last := cmd.Args[len(cmd.Args)-1]
	if last != `sh -c 'cd '\''/srv/proj'\'' && exec '\''/home/dev/.cache/ai-code/bin/ai-code-worker-0123'\'''` {
		t.Fatalf("ssh remote command = %q", last)
	}

	dc := &DevcontainerExecutor{cfg: &devcontainer.Config{RemoteUser: "dev"},
		cmd: &exec.Cmd{}, cid: "c1", engine: "podman", entry: "/ai-code-workers/w", workdir: "/workspace"}
	dc.publishUp()
	fork := &DevcontainerExecutor{host: dc, workdir: "/workspace/sub"}
	// A tool call holds mu for as long as it runs; starting an MCP server in
	// the running container must not wait for it.
	dc.mu.Lock()
	cmd, err = fork.MCPCommand(context.Background())
	dc.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(cmd.Args, " "); got != "podman exec -i -w /workspace/sub --user dev c1 /ai-code-workers/w" {
		t.Fatalf("devcontainer command = %q", got)
	}

	launch := mcpLauncher(func(context.Context) (*exec.Cmd, error) { return exec.Command("worker"), nil })
	_, preface, err := launch(context.Background(), mcp.Entry{Binary: "/opt/mcp/lldb", Args: []string{"--root", "/my dir", `it's "quoted"`}})
	if err != nil {
		t.Fatal(err)
	}
	var hello worker.Hello
	if err := json.Unmarshal(preface, &hello); err != nil {
		t.Fatal(err)
	}
	if hello.MCP == nil || hello.MCP.Binary != "/opt/mcp/lldb" || strings.Join(hello.MCP.Args, "|") != `--root|/my dir|it's "quoted"` {
		t.Fatalf("opening message = %s", preface)
	}
}

// End to end through the real worker, as a devcontainer or ssh session runs
// it: /mcp add, the tool call, the server's working directory.
func TestMCPRunsInTheTargetEnvironmentThroughTheWorker(t *testing.T) {
	buildBins(t)
	a, _, _ := mcpApp(t, nil)
	target := t.TempDir()
	opened := 0
	a.mcp.SetLauncher(mcpLauncher(func(context.Context) (*exec.Cmd, error) {
		opened++
		cmd := exec.Command(workerPath)
		cmd.Dir = target
		return cmd, nil
	}))
	if err := a.cmdMCP(context.Background(), "add hello="+helloPath+` --greeting "good day"`); err != nil {
		t.Fatal(err)
	}
	settle(t, a)
	exe := mcp.Wrap(a.exec, a.mcp)
	res, err := exe.Execute(context.Background(), tool.Request{Name: "hello_hello", Args: json.RawMessage(`{"name":"sandbox"}`)})
	if err != nil || res.Content != "good day, sandbox" {
		t.Fatalf("result = %+v, %v", res, err)
	}
	if opened != 1 {
		t.Fatalf("worker opened %d times", opened)
	}

	a.mcp.SetLauncher(mcpLauncher(func(context.Context) (*exec.Cmd, error) {
		return nil, errors.New("the devcontainer is unavailable: no engine")
	}))
	got := make(chan mcp.Report, 1)
	a.mcp.SetReporter(func(r mcp.Report) { got <- r })
	if err := a.mcp.Restart("hello"); err != nil {
		t.Fatal(err)
	}
	if r := <-got; r.Err == nil || !strings.Contains(r.Err.Error(), "devcontainer is unavailable") {
		t.Fatalf("report = %+v", r)
	}
}

func TestMCPUsage(t *testing.T) {
	a, _, local := mcpApp(t, nil)
	out := captureOut(t, func() { _ = a.cmdMCP(context.Background(), "") })
	if !strings.Contains(out, "/mcp add NAME=VALUE        save to "+local+", then start") {
		t.Fatalf("usage:\n%s", out)
	}
	if err := a.cmdMCP(context.Background(), "frobnicate x"); err == nil {
		t.Fatal("unknown verb accepted")
	}
}

func writeConfig(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func tomlQuote(s string) string {
	return `"` + strings.ReplaceAll(strings.ReplaceAll(s, `\`, `\\`), `"`, `\"`) + `"`
}

func names(defs []tool.Definition) string {
	var n []string
	for _, d := range defs {
		n = append(n, d.Name)
	}
	return strings.Join(n, ",")
}

func captureStderr(t *testing.T, f func()) {
	t.Helper()
	saved := os.Stderr
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer devnull.Close()
	os.Stderr = devnull
	defer func() { os.Stderr = saved }()
	f()
}
