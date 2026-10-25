package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/BurntSushi/toml"
)

var mcpName = regexp.MustCompile(`^[A-Za-z0-9-]+$`)

// ValidMCPName reports whether name can prefix the tool names the model sees.
func ValidMCPName(name string) error {
	if !mcpName.MatchString(name) {
		return fmt.Errorf("MCP server name %q: use letters, digits and -", name)
	}
	return nil
}

// ValidateMCP checks one [mcp.<name>] entry.
func ValidateMCP(name string, m MCPServer) error {
	if err := ValidMCPName(name); err != nil {
		return err
	}
	switch {
	case m.URL != "" && m.Binary != "":
		return fmt.Errorf("mcp.%s: give url or binary, not both", name)
	case m.URL == "" && m.Binary == "":
		return fmt.Errorf("mcp.%s: needs url or binary", name)
	case m.URL != "" && len(m.Args) > 0:
		return fmt.Errorf("mcp.%s: args go with binary, not url", name)
	case m.Binary != "" && len(m.Headers) > 0:
		return fmt.Errorf("mcp.%s: headers go with url, not binary", name)
	case m.URL != "" && !IsMCPURL(m.URL):
		return fmt.Errorf("mcp.%s: url %q is not an http:// or https:// URL", name, m.URL)
	}
	return nil
}

// IsMCPURL reports whether s is a network URL to connect to rather than a
// binary to run.
func IsMCPURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != ""
}

// ProjectConfigPath is the project layer's file, which /mcp add and remove edit.
func ProjectConfigPath(projectDir string) string {
	return filepath.Join(projectDir, ".ai-code", "config.toml")
}

// AddMCP appends an [mcp.<name>] table to the file at path, creating it if
// needed. The existing text is left exactly as it was.
func AddMCP(path, name string, m MCPServer) error {
	if err := ValidateMCP(name, m); err != nil {
		return err
	}
	old, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if defined, err := definesMCP(old, name); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	} else if defined {
		return fmt.Errorf("%s already defines mcp.%s", path, name)
	}
	var b strings.Builder
	b.Write(old)
	if len(old) > 0 && !bytes.HasSuffix(old, []byte("\n")) {
		b.WriteString("\n")
	}
	if len(old) > 0 {
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "[mcp.%s]\n", name)
	if m.URL != "" {
		fmt.Fprintf(&b, "url = %s\n", tomlString(m.URL))
	} else {
		fmt.Fprintf(&b, "binary = %s\n", tomlString(m.Binary))
		if len(m.Args) > 0 {
			quoted := make([]string, len(m.Args))
			for i, a := range m.Args {
				quoted[i] = tomlString(a)
			}
			fmt.Fprintf(&b, "args = [%s]\n", strings.Join(quoted, ", "))
		}
	}
	return writeChecked(path, []byte(b.String()), name, true)
}

// RemoveMCP deletes the [mcp.<name>] table, and any [mcp.<name>.*] tables
// under it, from the file at path. Everything else is left as it was.
func RemoveMCP(path, name string) error {
	old, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	lines := strings.SplitAfter(string(old), "\n")
	var out []string
	removing, found := false, false
	for _, l := range lines {
		if h, ok := tableHeader(l); ok {
			removing = isMCPTable(h, name)
			found = found || removing
		}
		if !removing {
			out = append(out, l)
		}
	}
	if !found {
		return fmt.Errorf("%s defines mcp.%s in a form /mcp remove cannot edit; remove it by hand", path, name)
	}
	return writeChecked(path, []byte(strings.Join(out, "")), name, false)
}

// tableHeader returns the dotted key of a [table] line.
func tableHeader(line string) (string, bool) {
	t := strings.TrimSpace(line)
	if i := strings.Index(t, "#"); i >= 0 && !strings.Contains(t[:i], `"`) {
		t = strings.TrimSpace(t[:i])
	}
	if !strings.HasPrefix(t, "[") || strings.HasPrefix(t, "[[") || !strings.HasSuffix(t, "]") {
		return "", false
	}
	return strings.TrimSpace(t[1 : len(t)-1]), true
}

func isMCPTable(header, name string) bool {
	var parts []string
	for _, p := range strings.Split(header, ".") {
		parts = append(parts, strings.Trim(strings.TrimSpace(p), `"'`))
	}
	return len(parts) >= 2 && parts[0] == "mcp" && parts[1] == name
}

func definesMCP(raw []byte, name string) (bool, error) {
	var doc struct {
		MCP map[string]toml.Primitive `toml:"mcp"`
	}
	if _, err := toml.Decode(string(raw), &doc); err != nil {
		return false, err
	}
	_, ok := doc.MCP[name]
	return ok, nil
}

// writeChecked writes text only if it still parses and now does (or no
// longer does) define mcp.<name>.
func writeChecked(path string, text []byte, name string, want bool) error {
	defined, err := definesMCP(text, name)
	if err != nil {
		return fmt.Errorf("editing %s would leave it unreadable (%v); nothing was written", path, err)
	}
	if defined != want {
		return fmt.Errorf("could not edit mcp.%s in %s; nothing was written", name, path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, text, 0o644)
}

// tomlString quotes s as a TOML basic string. JSON's escapes are a subset of
// TOML's once HTML escaping is off.
func tomlString(s string) string {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s)
	return strings.TrimRight(b.String(), "\n")
}
