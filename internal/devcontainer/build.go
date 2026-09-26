package devcontainer

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"sort"
)

// ImageTag is the local tag for the image built from this configuration.
//
// It is derived from the build inputs that change what the image *is* -- the
// Dockerfile's contents, the build arguments, the target stage -- so two
// projects never collide and an edited Dockerfile produces a different tag.
//
// It deliberately does not hash the build context's contents. Walking a large
// context on every launch to decide whether to skip a build the engine would
// have answered from its own layer cache in the same time is the wrong trade;
// the build is run unconditionally and the engine's cache makes the no-change
// case cheap.
func (c *Config) ImageTag() (string, error) {
	df := c.DockerfilePath()
	if df == "" {
		return "", fmt.Errorf("no Dockerfile configured")
	}
	content, err := os.ReadFile(df)
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", df, err)
	}
	h := sha256.New()
	h.Write(content)
	for _, a := range c.BuildArgs() {
		h.Write([]byte("\x00arg\x00" + a))
	}
	if c.Build != nil {
		h.Write([]byte("\x00target\x00" + c.Build.Target))
		for _, o := range c.Build.Options {
			h.Write([]byte("\x00opt\x00" + o))
		}
	}
	h.Write([]byte("\x00ctx\x00" + c.ContextPath()))
	return "ai-code-devcontainer:" + hex.EncodeToString(h.Sum(nil))[:16], nil
}

// BuildArgv is the engine command line that builds this configuration's image.
func (c *Config) BuildArgv(engine, tag string) []string {
	argv := []string{"build", "-t", tag, "-f", c.DockerfilePath()}
	for _, a := range c.BuildArgs() {
		argv = append(argv, "--build-arg", a)
	}
	if c.Build != nil {
		if c.Build.Target != "" {
			argv = append(argv, "--target", c.Build.Target)
		}
		for _, cf := range c.Build.CacheFrom {
			argv = append(argv, "--cache-from", cf)
		}
		argv = append(argv, c.Build.Options...)
	}
	return append(argv, c.ContextPath())
}

// HaveImage reports whether the engine already holds an image by this tag.
func HaveImage(engine, tag string) bool {
	cmd := exec.Command(engine, "image", "exists", tag)
	if err := cmd.Run(); err == nil {
		return true
	}
	// `image exists` is podman's; docker answers the same question by inspect.
	cmd = exec.Command(engine, "image", "inspect", tag)
	cmd.Stdout, cmd.Stderr = nil, nil
	return cmd.Run() == nil
}

// RunArgv assembles the engine arguments that start a container for this
// configuration, short of the image name and the command to run in it.
//
// binaryMount is the host path of the ai-code binary and the container path to
// mount it at; the daemon inside the container is that binary.
func (c *Config) RunArgv(localWorkspaceFolder, workdir, hostBinary, containerBinary string) []string {
	argv := []string{"run", "--rm", "-i"}

	argv = append(argv, "--mount", c.WorkspaceMountSpec(localWorkspaceFolder))
	for _, m := range c.Mounts {
		argv = append(argv, "--mount", m.String())
	}
	// The binary is a separate read-only bind: its host path does not exist
	// inside the container, and nothing in there should be able to rewrite the
	// thing it is executing.
	argv = append(argv, "-v", hostBinary+":"+containerBinary+":ro")

	argv = append(argv, "-w", workdir)

	if c.ContainerUser != "" {
		argv = append(argv, "--user", c.ContainerUser)
	} else if c.RemoteUser != "" {
		// Without lifecycle commands there is no stage that runs as a different
		// user from the tools, so remoteUser and containerUser collapse to the
		// same thing: who the agent's commands run as. Honouring remoteUser
		// here is what keeps files written into the bind mount owned by the
		// user who launched ai-code.
		argv = append(argv, "--user", c.RemoteUser)
	}

	// Sorted, so the command line is stable between launches and a diff of
	// two failures is about what changed rather than about map ordering.
	for _, kv := range sortedEnv(c.ContainerEnv) {
		argv = append(argv, "-e", kv)
	}
	for _, kv := range sortedEnv(c.RemoteEnv) {
		argv = append(argv, "-e", kv)
	}

	if c.Init != nil && *c.Init {
		argv = append(argv, "--init")
	}
	if c.Privileged != nil && *c.Privileged {
		argv = append(argv, "--privileged")
	}
	for _, capability := range c.CapAdd {
		argv = append(argv, "--cap-add", capability)
	}
	for _, so := range c.SecurityOpt {
		argv = append(argv, "--security-opt", so)
	}

	// runArgs last, so a project can override anything decided above.
	argv = append(argv, c.RunArgs...)

	// The image's own ENTRYPOINT must not wrap the daemon: an image that sets
	// one would otherwise receive "/ai-code-bin --executor-daemon ..." as
	// arguments to something else entirely.
	argv = append(argv, "--entrypoint", containerBinary)
	return argv
}

// sortedEnv renders an environment map as KEY=VALUE in key order.
func sortedEnv(m map[string]string) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, k+"="+m[k])
	}
	return out
}
