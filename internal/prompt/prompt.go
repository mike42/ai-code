// Package prompt assembles the system prompt. It must be byte-identical across
// every request in a session: providers cache on an exact prefix match, and one
// unstable byte turns every later request into a cache miss that shows up as
// latency or cost, never as an error.
package prompt

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// Env is the stable environment description gathered once per session.
type Env struct {
	Cwd       string
	OS        string
	Arch      string
	Shell     string
	GitBranch string
	GitRoot   string
	IsGitRepo bool
}

type Options struct {
	Env Env
	// ModePrompt is the steering text for the active mode; modes shape how the
	// model works, they never remove tools.
	ModePrompt string
	// Agents holds discovered AGENTS.md content, outermost first.
	Agents []AgentsFile
	// ToolNames is used only for the summary line, in the advertised order.
	ToolNames []string
}

const base = `You are ai-code, a coding agent running in a terminal on a developer's machine.

You have direct access to the filesystem and a shell. Use them. When you need to
know something about this project -- how a function is used, whether a test
passes, what a config file contains -- find out rather than guessing or asking.
A command you can run yourself is faster than a question you make the user answer.

## How to work

Read before you edit. The edit tool enforces this, and it is not bureaucracy:
editing a file you have not read means editing what you imagine it contains.
Only the read tool satisfies it: cat, sed and head in the shell show you a file
but record nothing, so an edit after one of those is refused. Search and explore
with the shell; open anything you intend to change with the read tool.

Prefer targeted edits to whole-file rewrites. A write replaces everything,
including the parts you never looked at.

When a tool fails, read what it says. The error messages here are written to
tell you exactly what went wrong and what to do about it -- a failed edit names
the line it nearly matched and the whitespace that differed. Reissuing the same
call unchanged will fail the same way.

Verify your work when there is a way to. If the project has tests, run them. If
it compiles, compile it. Reporting a change as done without checking it is the
single least useful thing you can do.

## What to say

Be concise. The user is reading your output in a terminal, streamed, while they
wait. Say what you did and what you found; skip the preamble, the summary of the
request, and the offer to help further.

Do not narrate routine tool use -- the user can see the tool calls. Explain when
something is surprising, when you made a judgement call, or when you are about
to do something with consequences.

Use markdown sparingly: it renders, but headers and bullet lists for a two-line
answer are noise. Code and file paths in backticks; real code in fenced blocks
with a language tag so it is highlighted.

When you report a problem you did not fix, say so explicitly rather than leaving
it implied.`

// Build assembles the system prompt. The section order is fixed.
func Build(o Options) string {
	var b strings.Builder
	b.WriteString(base)

	b.WriteString("\n\n## Environment\n\n")
	b.WriteString(o.Env.describe())

	if len(o.ToolNames) > 0 {
		fmt.Fprintf(&b, "\nTools available: %s.\n", strings.Join(o.ToolNames, ", "))
	}

	if strings.TrimSpace(o.ModePrompt) != "" {
		b.WriteString("\n## Current mode\n\n")
		b.WriteString(strings.TrimSpace(o.ModePrompt))
		b.WriteString("\n")
	}

	if len(o.Agents) > 0 {
		b.WriteString("\n## Project instructions\n\n")
		b.WriteString("These come from AGENTS.md files in this project. They are the " +
			"project's own conventions and they take precedence over your general habits. " +
			"Where two files conflict, the more deeply nested one wins.\n")
		for _, a := range o.Agents {
			if a.Path == "" {
				// No path to give: a user-level file outside anything the
				// tools can reach.
				fmt.Fprintf(&b, "\n<agents-file>\n%s\n</agents-file>\n",
					strings.TrimSpace(a.Content))
				continue
			}
			fmt.Fprintf(&b, "\n<agents-file path=%q>\n%s\n</agents-file>\n",
				a.Path, strings.TrimSpace(a.Content))
		}
	}

	return b.String()
}

func (e Env) describe() string {
	var b strings.Builder
	fmt.Fprintf(&b, "Working directory: %s\n", e.Cwd)
	fmt.Fprintf(&b, "Platform: %s/%s\n", e.OS, e.Arch)
	if e.Shell != "" {
		fmt.Fprintf(&b, "Shell: %s\n", e.Shell)
	}
	if e.IsGitRepo {
		fmt.Fprintf(&b, "Git repository: yes (root %s", e.GitRoot)
		if e.GitBranch != "" {
			fmt.Fprintf(&b, ", branch %s", e.GitBranch)
		}
		b.WriteString(")\n")
	} else {
		b.WriteString("Git repository: no\n")
	}
	return b.String()
}

// DetectEnv gathers environment facts once, at session start; re-reading the
// branch would rewrite the cached prefix on every request.
func DetectEnv(cwd string) Env {
	e := Env{
		Cwd:   cwd,
		OS:    goos(),
		Arch:  goarch(),
		Shell: filepath.Base(os.Getenv("SHELL")),
	}
	if root, ok := gitRoot(cwd); ok {
		e.IsGitRepo = true
		e.GitRoot = root
		e.GitBranch = gitBranch(cwd)
	}
	return e
}

func gitRoot(dir string) (string, bool) {
	for d := dir; ; {
		if fi, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			_ = fi
			return d, true
		}
		parent := filepath.Dir(d)
		if parent == d {
			return "", false
		}
		d = parent
	}
}

func gitBranch(dir string) string {
	cmd := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// AgentsFile is one discovered instruction file.
type AgentsFile struct {
	Path    string
	Content string
}

// DefaultAgentsNames are the filenames ai-code looks for, in preference order
// within a single directory. AGENTS.md is the cross-tool convention; the
// .ai-code copy is for projects that keep tool files out of the root.
var DefaultAgentsNames = []string{"AGENTS.md", ".ai-code/AGENTS.md"}

// MaxAgentsBytes caps the instruction budget, which otherwise silently consumes
// every request's window.
const MaxAgentsBytes = 32 * 1024

// DiscoverAgents finds instruction files from the git root down to the working
// directory, outermost first, so more specific files appear later and read as
// overrides. A user-level file is included first of all.
func DiscoverAgents(cwd, gitRoot string) []AgentsFile {
	var dirs []string

	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, ".config", "ai-code"))
	}

	stop := gitRoot
	if stop == "" {
		stop = cwd
	}
	var chain []string
	for d := cwd; ; {
		chain = append(chain, d)
		if d == stop || filepath.Dir(d) == d {
			break
		}
		parent := filepath.Dir(d)
		if len(parent) < len(stop) {
			break
		}
		d = parent
	}
	for i := len(chain) - 1; i >= 0; i-- {
		dirs = append(dirs, chain[i])
	}

	var (
		out   []AgentsFile
		total int
		seen  = map[string]bool{}
	)
	for _, dir := range dirs {
		for _, name := range DefaultAgentsNames {
			path := filepath.Join(dir, name)
			if seen[path] {
				continue
			}
			seen[path] = true

			data, err := os.ReadFile(path)
			if err != nil || len(strings.TrimSpace(string(data))) == 0 {
				continue
			}
			if total+len(data) > MaxAgentsBytes {
				remaining := MaxAgentsBytes - total
				if remaining < 512 {
					return out
				}
				data = append(data[:remaining],
					[]byte("\n\n[truncated: the project instruction budget was reached]")...)
			}
			total += len(data)
			out = append(out, AgentsFile{Path: path, Content: string(data)})
			// Only the first matching name per directory is sent.
			break
		}
	}
	return out
}

// SortedToolNames returns names in a deterministic order for the prompt.
func SortedToolNames(names []string) []string {
	out := append([]string(nil), names...)
	sort.Strings(out)
	return out
}
