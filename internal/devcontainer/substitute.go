package devcontainer

import (
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// Vars are the values a devcontainer.json's ${...} references resolve against.
type Vars struct {
	// LocalWorkspaceFolder is the host directory being mounted.
	LocalWorkspaceFolder string
	// ContainerWorkspaceFolder is the resolved workspaceFolder inside the
	// container. It is itself substituted first, since a file may write
	// workspaceFolder in terms of the local one.
	ContainerWorkspaceFolder string
	// ID is the value of ${devcontainerId}: stable for a given project so a
	// named volume or label survives across sessions.
	ID string
	// ContainerSide marks a value that will be read from inside the
	// container, where a host path is both useless and a disclosure. See
	// expand.
	ContainerSide bool
}

// Substitute expands the variable references a devcontainer.json may contain.
//
// ${containerEnv:...} is deliberately left as written: it can only be resolved
// by asking a container that does not exist yet, and silently expanding it to
// an empty string would turn a configuration error into a mysterious one.
func Substitute(s string, v Vars) string {
	if !strings.Contains(s, "${") {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		if s[i] != '$' || i+1 >= len(s) || s[i+1] != '{' {
			b.WriteByte(s[i])
			i++
			continue
		}
		end := strings.IndexByte(s[i:], '}')
		if end < 0 {
			b.WriteString(s[i:])
			break
		}
		ref := s[i+2 : i+end]
		b.WriteString(expand(ref, v, s[i:i+end+1]))
		i += end + 1
	}
	return b.String()
}

func expand(ref string, v Vars, whole string) string {
	name, arg, hasArg := strings.Cut(ref, ":")
	switch name {
	case "localWorkspaceFolder":
		if v.ContainerSide {
			// A host path in an environment variable the container reads
			// names nothing that exists there, so a config using it this way
			// is already broken -- and it hands a sandboxed agent the
			// directory layout and username of the machine hosting it. The
			// container's own workspace folder is what the author meant.
			return v.ContainerWorkspaceFolder
		}
		return v.LocalWorkspaceFolder
	case "localWorkspaceFolderBasename":
		if v.ContainerSide {
			return path.Base(v.ContainerWorkspaceFolder)
		}
		return filepath.Base(v.LocalWorkspaceFolder)
	case "containerWorkspaceFolder":
		return v.ContainerWorkspaceFolder
	case "containerWorkspaceFolderBasename":
		return path.Base(v.ContainerWorkspaceFolder)
	case "devcontainerId":
		return v.ID
	case "localEnv":
		key, def, hasDef := strings.Cut(arg, ":")
		if got, ok := os.LookupEnv(key); ok && got != "" {
			return got
		}
		if hasDef {
			return def
		}
		// An unset variable with no default expands to nothing, which is what
		// the reference implementation does.
		return ""
	case "containerEnv":
		// Unresolvable before the container exists. Left verbatim.
		_ = hasArg
		return whole
	}
	return whole
}

// SubstituteAll expands variables throughout a configuration, in place.
//
// workspaceFolder is resolved first because ${containerWorkspaceFolder}
// elsewhere in the file means whatever workspaceFolder ended up being.
func (c *Config) SubstituteAll(localWorkspaceFolder, id string) {
	v := Vars{LocalWorkspaceFolder: localWorkspaceFolder, ID: id}

	c.WorkspaceFolder = Substitute(c.WorkspaceFolder, v)
	if c.WorkspaceFolder == "" {
		// The spec's default: /workspaces/<basename of the host folder>.
		c.WorkspaceFolder = path.Join("/workspaces", filepath.Base(localWorkspaceFolder))
	}
	v.ContainerWorkspaceFolder = c.WorkspaceFolder

	c.WorkspaceMount = Substitute(c.WorkspaceMount, v)
	c.Image = Substitute(c.Image, v)
	c.RemoteUser = Substitute(c.RemoteUser, v)
	c.ContainerUser = Substitute(c.ContainerUser, v)

	for i := range c.Mounts {
		m := &c.Mounts[i]
		m.raw = Substitute(m.raw, v)
		m.Source = Substitute(m.Source, v)
		m.Target = Substitute(m.Target, v)
	}
	for i, a := range c.RunArgs {
		c.RunArgs[i] = Substitute(a, v)
	}
	// Read inside the container, so host paths are rewritten rather than
	// passed through. Recorded, because silently meaning something other than
	// what the file says is worse than the leak it prevents.
	inside := v
	inside.ContainerSide = true
	c.envRewrites = nil
	for k, val := range c.ContainerEnv {
		c.ContainerEnv[k] = Substitute(val, inside)
		if c.ContainerEnv[k] != Substitute(val, v) {
			c.envRewrites = append(c.envRewrites, "containerEnv."+k)
		}
	}
	for k, val := range c.RemoteEnv {
		c.RemoteEnv[k] = Substitute(val, inside)
		if c.RemoteEnv[k] != Substitute(val, v) {
			c.envRewrites = append(c.envRewrites, "remoteEnv."+k)
		}
	}
	sort.Strings(c.envRewrites)
	if c.Build != nil {
		for k, val := range c.Build.Args {
			c.Build.Args[k] = Substitute(val, v)
		}
		c.Build.Dockerfile = Substitute(c.Build.Dockerfile, v)
		c.Build.Context = Substitute(c.Build.Context, v)
	}
	c.DockerfileLegacy = Substitute(c.DockerfileLegacy, v)
	c.ContextLegacy = Substitute(c.ContextLegacy, v)
}

// WorkspaceMountSpec returns the --mount argument that binds the project into
// the container, honouring an explicit workspaceMount and otherwise building
// the default bind the spec describes.
func (c *Config) WorkspaceMountSpec(localWorkspaceFolder string) string {
	if c.WorkspaceMount != "" {
		return c.WorkspaceMount
	}
	return "type=bind,source=" + localWorkspaceFolder + ",target=" + c.WorkspaceFolder
}
