package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"ai-code/internal/devcontainer"
	"ai-code/internal/runtime"
	"ai-code/internal/tool"
)

// Unit tests over argv cannot prove the image really builds, the daemon starts
// inside it, or the workspace lands where the configuration said. This runs the
// repository's own example, so the example cannot rot away from the code.
func TestDockerfileDevcontainerRunsTools(t *testing.T) {
	engine, err := detectContainerEngine()
	if err != nil {
		t.Skipf("no container engine: %v", err)
	}

	example, err := filepath.Abs("../../examples/sandbox-demo-dockerfile")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(example); err != nil {
		t.Skipf("example not present: %v", err)
	}

	rt, err := runtime.Resolve("devcontainer", example)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if rt.ConfigErr != nil {
		t.Fatalf("parsing the example's devcontainer.json: %v", rt.ConfigErr)
	}
	if got := rt.Config.Kind(); got != devcontainer.KindBuild {
		t.Fatalf("Kind() = %v, want KindBuild", got)
	}
	rt.Config.SubstituteAll(example, devcontainerID(example))

	if rt.Config.WorkspaceFolder != "/workspace" {
		t.Errorf("WorkspaceFolder = %q, want /workspace", rt.Config.WorkspaceFolder)
	}
	if rt.Config.RemoteUser != "ubuntu" {
		t.Errorf("RemoteUser = %q, want ubuntu", rt.Config.RemoteUser)
	}
	if len(rt.Config.RunArgs) != 1 || rt.Config.RunArgs[0] != "--userns=keep-id" {
		t.Errorf("RunArgs = %v, want [--userns=keep-id]", rt.Config.RunArgs)
	}

	// The example's image is a different userspace from the one that built
	// this test, so the binary must be static.
	bin := filepath.Join(t.TempDir(), "ai-code")
	build := exec.Command("go", "build", "-o", bin, "ai-code/cmd/ai-code")
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("could not build static binary: %v\n%s", err, out)
	}

	dc := NewDevcontainerExecutor(rt.Config, nil, example, example, bin)
	defer dc.Close()

	// Build and container start are deferred to the first tool call, never to
	// startup.
	args, _ := json.Marshal(map[string]string{"command": "pwd && id -un && cat /etc/os-release | head -1"})
	res, err := dc.Execute(context.Background(), tool.Request{
		CallID: "1", Name: "bash", Args: args,
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.IsError {
		t.Fatalf("tool failed inside the built container:\n%s", res.Content)
	}

	// Tools run as remoteUser, not as the image's default, in the configured
	// workspaceFolder.
	if !strings.Contains(res.Content, "/workspace") {
		t.Errorf("working directory is not the configured workspaceFolder:\n%s", res.Content)
	}
	if !strings.Contains(res.Content, "ubuntu") {
		t.Errorf("tools are not running as the configured remoteUser:\n%s", res.Content)
	}
	if !strings.Contains(res.Content, "Ubuntu") {
		t.Errorf("container is not the image the Dockerfile builds:\n%s", res.Content)
	}

	// The build leaves a tagged image behind for a second session to reuse.
	tag, err := rt.Config.ImageTag()
	if err != nil {
		t.Fatalf("ImageTag: %v", err)
	}
	if !devcontainer.HaveImage(engine, tag) {
		t.Errorf("image %s was not left behind for the next session", tag)
	}
}
