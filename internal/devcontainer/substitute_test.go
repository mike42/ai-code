package devcontainer

import (
	"strings"
	"testing"
)

func TestSubstituteWorkspaceVariables(t *testing.T) {
	v := Vars{
		LocalWorkspaceFolder:     "/home/dev/proj",
		ContainerWorkspaceFolder: "/workspaces/proj",
		ID:                       "abc123",
	}
	cases := map[string]string{
		"${localWorkspaceFolder}":                  "/home/dev/proj",
		"${localWorkspaceFolderBasename}":          "proj",
		"${containerWorkspaceFolder}":              "/workspaces/proj",
		"${containerWorkspaceFolderBasename}":      "proj",
		"${devcontainerId}":                        "abc123",
		"source=${localWorkspaceFolder},type=bind": "source=/home/dev/proj,type=bind",
		"no variables here":                        "no variables here",
	}
	for in, want := range cases {
		if got := Substitute(in, v); got != want {
			t.Errorf("Substitute(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSubstituteLocalEnv(t *testing.T) {
	t.Setenv("AI_CODE_TEST_VAR", "set-value")
	v := Vars{}
	cases := map[string]string{
		"${localEnv:AI_CODE_TEST_VAR}":            "set-value",
		"${localEnv:AI_CODE_TEST_VAR:fallback}":   "set-value",
		"${localEnv:AI_CODE_TEST_UNSET:fallback}": "fallback",
		// No default and unset expands to nothing, as upstream does.
		"${localEnv:AI_CODE_TEST_UNSET}": "",
	}
	for in, want := range cases {
		if got := Substitute(in, v); got != want {
			t.Errorf("Substitute(%q) = %q, want %q", in, got, want)
		}
	}
}

// ${containerEnv:...} cannot be known before the container exists. Expanding it
// to nothing would turn a configuration error into a mysterious one.
func TestContainerEnvIsLeftAlone(t *testing.T) {
	got := Substitute("${containerEnv:PATH}:/extra", Vars{})
	if got != "${containerEnv:PATH}:/extra" {
		t.Errorf("Substitute() = %q, want it left verbatim", got)
	}
}

func TestUnknownVariableIsLeftAlone(t *testing.T) {
	if got := Substitute("${noSuchThing}", Vars{}); got != "${noSuchThing}" {
		t.Errorf("Substitute() = %q, want it left verbatim", got)
	}
}

func TestUnterminatedReferenceDoesNotPanic(t *testing.T) {
	if got := Substitute("${localWorkspaceFolder", Vars{LocalWorkspaceFolder: "/p"}); got != "${localWorkspaceFolder" {
		t.Errorf("Substitute() = %q", got)
	}
}

// workspaceFolder has to be resolved before anything that refers to it.
func TestWorkspaceFolderResolvesBeforeContainerWorkspaceFolder(t *testing.T) {
	c, err := Parse([]byte(`{
  "image": "x",
  "workspaceFolder": "/src/${localWorkspaceFolderBasename}",
  "mounts": ["source=/host/cache,target=${containerWorkspaceFolder}/.cache,type=bind"]
}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	c.SubstituteAll("/home/dev/proj", "id")
	if c.WorkspaceFolder != "/src/proj" {
		t.Fatalf("WorkspaceFolder = %q", c.WorkspaceFolder)
	}
	if got := c.Mounts[0].String(); !strings.Contains(got, "target=/src/proj/.cache") {
		t.Errorf("mount = %q, want the resolved workspace folder in it", got)
	}
}

// The spec's default, which is not the /workspace this harness used to hardcode.
func TestWorkspaceFolderDefault(t *testing.T) {
	c, err := Parse([]byte(`{"image": "x"}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	c.SubstituteAll("/home/dev/my-proj", "id")
	if c.WorkspaceFolder != "/workspaces/my-proj" {
		t.Errorf("WorkspaceFolder = %q, want /workspaces/my-proj", c.WorkspaceFolder)
	}
}

func TestWorkspaceMountDefaultBindsTheProject(t *testing.T) {
	c, err := Parse([]byte(`{"image": "x"}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	c.SubstituteAll("/home/dev/proj", "id")
	got := c.WorkspaceMountSpec("/home/dev/proj")
	if !strings.Contains(got, "source=/home/dev/proj") || !strings.Contains(got, "target=/workspaces/proj") {
		t.Errorf("WorkspaceMountSpec() = %q", got)
	}
}

// An explicit workspaceMount wins, variables and all.
func TestExplicitWorkspaceMountIsHonoured(t *testing.T) {
	c, err := Parse([]byte(`{
  "image": "x",
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind"
}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	c.SubstituteAll("/home/dev/proj", "id")
	want := "source=/home/dev/proj,target=/workspace,type=bind"
	if got := c.WorkspaceMountSpec("/home/dev/proj"); got != want {
		t.Errorf("WorkspaceMountSpec() = %q, want %q", got, want)
	}
}

// runArgs is how a project reaches the engine directly; on rootless podman it
// is what keeps written files owned by the user.
func TestRunArgvCarriesRunArgsAndUser(t *testing.T) {
	c, err := Parse([]byte(`{
  "image": "x",
  "workspaceFolder": "/workspace",
  "runArgs": ["--userns=keep-id"],
  "remoteUser": "ubuntu",
  "containerEnv": {"B": "2", "A": "1"}
}`), "/p/.devcontainer/devcontainer.json")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	c.SubstituteAll("/home/dev/proj", "id")
	argv := strings.Join(c.RunArgv("/home/dev/proj", "/workspace", "/host/ai-code", "/ai-code-bin"), " ")

	for _, want := range []string{
		"--userns=keep-id",
		"--user ubuntu",
		"-e A=1 -e B=2", // sorted, so the command line is stable
		"--entrypoint /ai-code-bin",
		"-v /host/ai-code:/ai-code-bin:ro",
	} {
		if !strings.Contains(argv, want) {
			t.Errorf("RunArgv() = %q\n  missing %q", argv, want)
		}
	}
}

// An image with its own ENTRYPOINT would otherwise receive the daemon's
// arguments instead of being replaced by it.
func TestEntrypointIsOverridden(t *testing.T) {
	c, _ := Parse([]byte(`{"image": "x", "workspaceFolder": "/w"}`), "/p/.devcontainer/devcontainer.json")
	c.SubstituteAll("/home/dev/proj", "id")
	argv := c.RunArgv("/home/dev/proj", "/w", "/host/ai-code", "/ai-code-bin")
	found := false
	for i, a := range argv {
		if a == "--entrypoint" && i+1 < len(argv) && argv[i+1] == "/ai-code-bin" {
			found = true
		}
	}
	if !found {
		t.Errorf("RunArgv() = %v, want --entrypoint /ai-code-bin", argv)
	}
}

// A host path in containerEnv is both a disclosure and a wrong answer.
//
// ${localWorkspaceFolder} resolves to the directory on the machine running
// the engine. Put into an environment variable the container reads, it names
// a path that does not exist there -- and hands whatever runs inside the
// sandbox the layout and username of the host. The container's own workspace
// folder is what such a config means.
func TestHostPathsDoNotReachContainerEnvironment(t *testing.T) {
	c := &Config{
		Image: "alpine",
		// Deliberately not the host basename, so a rewritten basename is
		// distinguishable from one that happened to match.
		WorkspaceFolder: "/workspaces/app",
		ContainerEnv: map[string]string{
			"PROJECT_DIR": "${localWorkspaceFolder}",
			"PROJECT":     "${localWorkspaceFolderBasename}",
			"UNRELATED":   "keep-me",
		},
		RemoteEnv: map[string]string{"ALSO_DIR": "${localWorkspaceFolder}/src"},
	}
	c.SubstituteAll("/home/someone/proj", "id")

	for _, got := range []string{
		c.ContainerEnv["PROJECT_DIR"], c.ContainerEnv["PROJECT"],
		c.ContainerEnv["UNRELATED"], c.RemoteEnv["ALSO_DIR"],
	} {
		if strings.Contains(got, "/home/someone") || strings.Contains(got, "someone") {
			t.Errorf("a host path reached the container environment: %q", got)
		}
	}
	if got := c.ContainerEnv["PROJECT_DIR"]; got != "/workspaces/app" {
		t.Errorf("PROJECT_DIR = %q, want the container's workspace folder", got)
	}
	if got := c.ContainerEnv["PROJECT"]; got != "app" {
		t.Errorf("PROJECT = %q, want the container folder's basename", got)
	}
	if got := c.RemoteEnv["ALSO_DIR"]; got != "/workspaces/app/src" {
		t.Errorf("ALSO_DIR = %q, want the container path", got)
	}
	if got := c.ContainerEnv["UNRELATED"]; got != "keep-me" {
		t.Errorf("UNRELATED = %q, want it untouched", got)
	}

	// Rewriting silently would be worse than the leak, so it is recorded.
	want := []string{"containerEnv.PROJECT", "containerEnv.PROJECT_DIR", "remoteEnv.ALSO_DIR"}
	if got := c.EnvRewrites(); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("EnvRewrites() = %v, want %v", got, want)
	}

	// The mount itself still needs the real host path -- that one is an
	// argument to the engine on the host, not a value read inside.
	if got := c.WorkspaceMountSpec("/home/someone/proj"); !strings.Contains(got, "source=/home/someone/proj") {
		t.Errorf("the bind mount lost its host source: %q", got)
	}

	// And nothing in the launch arguments carries the host path except that
	// mount, which cannot work any other way.
	argv := c.RunArgv("/home/someone/proj", "/workspaces/app", "/host/ai-code", "/ai-code-bin")
	for i, a := range argv {
		if !strings.Contains(a, "/home/someone") {
			continue
		}
		if strings.HasPrefix(a, "type=bind,source=/home/someone/proj") {
			continue
		}
		t.Errorf("argv[%d] = %q carries the host path", i, a)
	}
}
