package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeFile(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestMCPEntriesReplaceRatherThanMerge(t *testing.T) {
	global := t.TempDir()
	project := t.TempDir()
	t.Setenv("AI_CODE_CONFIG_DIR", global)
	writeFile(t, filepath.Join(global, "config.toml"), `
[mcp.lldb]
binary = "/usr/bin/lldb-mcp"
args = ["--global-only"]

[mcp.blender]
url = "http://10.0.0.5:9999"
headers = { Authorization = "Bearer x" }
`)
	writeFile(t, ProjectConfigPath(project), `
[mcp.lldb]
binary = "/opt/lldb-mcp"
`)
	cfg, err := Load(project)
	if err != nil {
		t.Fatal(err)
	}
	lldb := cfg.MCP["lldb"]
	if lldb.Binary != "/opt/lldb-mcp" || lldb.Args != nil || lldb.Source != "local" || lldb.Hides != "global" || lldb.Path != ProjectConfigPath(project) {
		t.Fatalf("lldb = %+v", lldb)
	}
	blender := cfg.MCP["blender"]
	if blender.URL != "http://10.0.0.5:9999" || blender.Source != "global" || blender.Hides != "" || blender.Headers["Authorization"] != "Bearer x" {
		t.Fatalf("blender = %+v", blender)
	}
}

func TestMCPValidation(t *testing.T) {
	cases := map[string]string{
		"[mcp.a]\nurl = \"http://x\"\nbinary = \"/b\"\n":      "not both",
		"[mcp.a]\nargs = [\"x\"]\n":                           "needs url or binary",
		"[mcp.a]\nurl = \"http://x\"\nargs = [\"x\"]\n":       "args go with binary",
		"[mcp.a]\nbinary = \"/b\"\nheaders = { A = \"b\" }\n": "headers go with url",
		"[mcp.a]\nurl = \"ftp://x\"\n":                        "not an http",
		"[mcp.\"a_b\"]\nbinary = \"/b\"\n":                    "letters, digits and -",
	}
	for text, want := range cases {
		project := t.TempDir()
		t.Setenv("AI_CODE_CONFIG_DIR", t.TempDir())
		writeFile(t, ProjectConfigPath(project), text)
		_, err := Load(project)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: err = %v, want %q", text, err, want)
		}
	}
}

func TestAddMCPKeepsTheFileAndCreatesIt(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".ai-code", "config.toml")
	if err := AddMCP(path, "hello", MCPServer{Binary: `C:\tools\hello.exe`, Args: []string{"--greeting", `say "hi" <now>`}}); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	want := "[mcp.hello]\nbinary = \"C:\\\\tools\\\\hello.exe\"\nargs = [\"--greeting\", \"say \\\"hi\\\" <now>\"]\n"
	if string(got) != want {
		t.Fatalf("file =\n%s\nwant\n%s", got, want)
	}

	existing := "# keep me\ndefault_mode = \"build\" # and me\n\n[ui]\ntheme = \"none\""
	writeFile(t, path, existing)
	if err := AddMCP(path, "web", MCPServer{URL: "http://10.0.0.5:9999"}); err != nil {
		t.Fatal(err)
	}
	got, _ = os.ReadFile(path)
	if !strings.HasPrefix(string(got), existing+"\n") {
		t.Fatalf("existing text changed:\n%s", got)
	}
	if !strings.HasSuffix(string(got), "\n\n[mcp.web]\nurl = \"http://10.0.0.5:9999\"\n") {
		t.Fatalf("table not appended:\n%s", got)
	}

	if err := AddMCP(path, "web", MCPServer{URL: "http://other"}); err == nil || !strings.Contains(err.Error(), "already defines mcp.web") {
		t.Fatalf("duplicate add: %v", err)
	}
}

func TestAddedEntryLoadsBack(t *testing.T) {
	project := t.TempDir()
	t.Setenv("AI_CODE_CONFIG_DIR", t.TempDir())
	want := MCPServer{Binary: "/opt/x y/hello", Args: []string{"a b", `c\d`, ""}}
	if err := AddMCP(ProjectConfigPath(project), "hello", want); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(project)
	if err != nil {
		t.Fatal(err)
	}
	got := cfg.MCP["hello"]
	if got.Binary != want.Binary || !reflect.DeepEqual(got.Args, want.Args) {
		t.Fatalf("loaded %+v, want %+v", got, want)
	}
}

func TestRemoveMCPLeavesEverythingElse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	text := `# top comment
[mcp.keep]
binary = "/a"

[mcp.lldb] # trailing comment
binary = "/usr/bin/lldb-mcp"
args = ["--x"]

[mcp.lldb.headers]
X = "y"

# about the ui
[ui]
theme = "none"
`
	writeFile(t, path, text)
	if err := RemoveMCP(path, "lldb"); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	want := `# top comment
[mcp.keep]
binary = "/a"

[ui]
theme = "none"
`
	if string(got) != want {
		t.Fatalf("file =\n%s\nwant\n%s", got, want)
	}
}

func TestRemoveMCPRefusesFormsItCannotEdit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	text := "mcp.lldb.binary = \"/usr/bin/lldb-mcp\"\n"
	writeFile(t, path, text)
	if err := RemoveMCP(path, "lldb"); err == nil || !strings.Contains(err.Error(), "remove it by hand") {
		t.Fatalf("err = %v", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != text {
		t.Fatal("file changed")
	}
}
