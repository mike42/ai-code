package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/mcp"
	"ai-code/internal/render"
	"ai-code/internal/worker"
)

// mcpSetup is what --mcp asked for: session-only entries, and every name to
// start, in flag order.
type mcpSetup struct {
	cli   map[string]mcp.Entry
	start []string
}

// parseMCPFlags resolves --mcp values against the loaded config. A name no
// source defines stops ai-code here, before anything starts.
func parseMCPFlags(values []string, cfg *config.Config) (*mcpSetup, error) {
	s := &mcpSetup{cli: map[string]mcp.Entry{}}
	seen := map[string]bool{}
	for _, v := range values {
		name, value, hasValue := strings.Cut(v, "=")
		if err := config.ValidMCPName(name); err != nil {
			return nil, fmt.Errorf("--mcp %s: %w", v, err)
		}
		if seen[name] {
			return nil, fmt.Errorf("--mcp %s is given twice", name)
		}
		seen[name] = true
		if hasValue {
			e, err := parseMCPValue(name, value)
			if err != nil {
				return nil, fmt.Errorf("--mcp %s: %w", name, err)
			}
			if c, ok := cfg.MCP[name]; ok {
				e.Hides = c.Source
			}
			s.cli[name] = e
		} else if _, ok := cfg.MCP[name]; !ok {
			return nil, fmt.Errorf("--mcp %s: no MCP server named %s is configured; give one with --mcp %s=VALUE", name, name, name)
		}
		s.start = append(s.start, name)
	}
	return s, nil
}

// parseMCPValue reads NAME=VALUE's VALUE: a network URL to connect to, or a
// binary and its arguments.
func parseMCPValue(name, value string) (mcp.Entry, error) {
	e := mcp.Entry{Name: name, Source: "cli", Value: value}
	if config.IsMCPURL(value) {
		e.URL = value
		return e, nil
	}
	words, err := mcp.Split(value)
	if err != nil {
		return e, err
	}
	if len(words) == 0 {
		return e, errors.New("needs a URL, or a binary and its arguments")
	}
	e.Binary, e.Args = words[0], words[1:]
	return e, nil
}

func entryFromConfig(name string, m config.MCPServer) mcp.Entry {
	return mcp.Entry{
		Name: name, Source: m.Source, Hides: m.Hides, Path: m.Path,
		URL: m.URL, Headers: m.Headers, Binary: m.Binary, Args: m.Args,
	}
}

// mcpResolver reads the config files afresh each time, so /mcp sees edits
// made since launch, and lays the session's cli entries over them.
func mcpResolver(cwd string, cli map[string]mcp.Entry) func() (map[string]mcp.Entry, error) {
	return func() (map[string]mcp.Entry, error) {
		cfg, err := config.Load(cwd)
		if err != nil {
			return nil, err
		}
		out := map[string]mcp.Entry{}
		for name, m := range cfg.MCP {
			out[name] = entryFromConfig(name, m)
		}
		for name, e := range cli {
			if c, ok := cfg.MCP[name]; ok {
				e.Hides = c.Source
			} else {
				e.Hides = ""
			}
			out[name] = e
		}
		return out, nil
	}
}

// mcpLauncher starts each server through a worker that open starts in the
// target environment, as every tool runs there. The binary and its argument
// list, or the URL and headers, go in the worker's opening message.
func mcpLauncher(open func(context.Context) (*exec.Cmd, error)) mcp.Launcher {
	return func(ctx context.Context, e mcp.Entry) (*exec.Cmd, []byte, error) {
		cmd, err := open(ctx)
		if err != nil {
			return nil, nil, err
		}
		launch := mcp.LaunchFor(e)
		preface, err := json.Marshal(worker.Hello{MCP: &launch})
		if err != nil {
			return nil, nil, err
		}
		return cmd, append(preface, '\n'), nil
	}
}

// mcpReport is the one line a finished start leaves in the scrollback.
func (a *App) mcpReport(r mcp.Report) {
	if r.Err != nil {
		a.note(fmt.Sprintf("MCP %s failed: %v", r.Name, r.Err))
		return
	}
	line := fmt.Sprintf("MCP %s started: %s, %s of schema", r.Name,
		plural(r.Tools, "tool"), a.tokenFigure(r.SchemaChars))
	a.mcpMu.Lock()
	atLaunch := a.mcpAtLaunch[r.Name]
	delete(a.mcpAtLaunch, r.Name)
	a.mcpMu.Unlock()
	if !atLaunch {
		line += a.reReadClause()
	}
	a.note(line)
}

// reReadClause states what changing the tool list costs: the server re-reads
// the whole session on the next request. Nothing is cached before the first
// request, unless a resumed session may have left its prefix on the server.
func (a *App) reReadClause() string {
	if !a.mcpSent.Load() && len(a.resumed) == 0 {
		return ""
	}
	n := a.contextState().Projected
	if n == 0 {
		return ""
	}
	return fmt.Sprintf("; the next request re-reads %s tokens", compactTokens(n))
}

func (a *App) tokenFigure(chars int) string {
	return compactTokens(agent.RoughTokens(chars)) + " tokens"
}

func compactTokens(n int) string {
	if n < 1000 {
		return fmt.Sprint(n)
	}
	return fmt.Sprintf("%.1fk", float64(n)/1000)
}

func plural(n int, word string) string {
	if n == 1 {
		return "1 " + word
	}
	return fmt.Sprintf("%d %ss", n, word)
}

// mcpSettle holds a request until every requested server has finished
// starting, naming them on the status line meanwhile.
func (a *App) mcpSettle(ctx context.Context) error {
	err := a.mcp.Settle(ctx, func(names []string) {
		a.mcpHolding.Store(len(names) > 0)
		if a.interactive == nil {
			if len(names) > 0 {
				fmt.Fprintf(os.Stderr, "note: waiting for %s\n", strings.Join(names, ", "))
			}
			return
		}
		if len(names) == 0 {
			a.interactive.SetNotice("")
		} else {
			a.interactive.SetNotice("waiting for " + strings.Join(names, ", "))
		}
	})
	if err == nil {
		a.mcpSent.Store(true)
	}
	return err
}

const mcpUsage = `Usage:
  /mcp list                  every configured server, where it is defined, and its state
  /mcp add NAME=VALUE        save to %s, then start
  /mcp remove NAME           stop, then delete from %s
  /mcp start NAME
  /mcp stop NAME
  /mcp restart NAME          start again and fetch its tool list afresh
VALUE is an http(s):// URL, or a binary and its arguments.`

func (a *App) cmdMCP(ctx context.Context, args string) error {
	verb, rest, _ := strings.Cut(strings.TrimSpace(args), " ")
	rest = strings.TrimSpace(rest)
	local := config.ProjectConfigPath(a.cwd)
	switch verb {
	case "":
		a.out(strings.Split("\n"+fmt.Sprintf(mcpUsage, local, local), "\n")...)
		return nil
	case "list":
		return a.mcpList()
	case "add":
		return a.mcpAdd(rest, local)
	}
	if rest == "" || strings.ContainsAny(rest, " \t") {
		return fmt.Errorf("/mcp %s takes one server name", verb)
	}
	switch verb {
	case "remove":
		return a.mcpRemove(rest)
	case "start":
		return a.mcp.Start(rest)
	case "stop":
		if err := a.mcp.Stop(rest); err != nil {
			return err
		}
		a.note("MCP " + rest + " stopped" + a.reReadClause())
		return nil
	case "restart":
		return a.mcp.Restart(rest)
	}
	return fmt.Errorf("/mcp has no %q; /mcp alone shows usage", verb)
}

func (a *App) mcpAdd(arg, local string) error {
	if a.cwd == "" {
		return errors.New("/mcp add saves to the project's config, and a remote session has none here")
	}
	name, value, ok := strings.Cut(arg, "=")
	if !ok || name == "" {
		return errors.New("/mcp add takes NAME=VALUE")
	}
	if err := config.ValidMCPName(name); err != nil {
		return err
	}
	if e, err := a.mcp.Entry(name); err == nil && e.Source == "cli" {
		return fmt.Errorf("%s is defined on the command line for this session", name)
	}
	e, err := parseMCPValue(name, value)
	if err != nil {
		return err
	}
	if err := config.AddMCP(local, name, config.MCPServer{URL: e.URL, Binary: e.Binary, Args: e.Args}); err != nil {
		return err
	}
	return a.mcp.Start(name)
}

func (a *App) mcpRemove(name string) error {
	e, err := a.mcp.Entry(name)
	if err != nil {
		return err
	}
	switch e.Source {
	case "cli":
		return fmt.Errorf("%s is defined on the command line for this session; /mcp stop %s stops it", name, name)
	case "global":
		return fmt.Errorf("%s is defined in the global config, %s; /mcp remove edits only %s", name, e.Path, config.ProjectConfigPath(a.cwd))
	}
	if a.mcp.Running(name) {
		_ = a.mcp.Stop(name)
	}
	if err := config.RemoveMCP(e.Path, name); err != nil {
		return err
	}
	msg := "MCP " + name + " removed from " + e.Path
	if after, err := a.mcp.Entry(name); err == nil {
		msg += "; the " + after.Source + " entry of that name now applies"
	}
	a.note(msg)
	return nil
}

func (a *App) mcpList() error {
	rows, err := a.mcp.List()
	if err != nil {
		return err
	}
	style := render.NewStyle(a.screen.Color())
	if len(rows) == 0 {
		a.out("", style.Dim("No MCP servers are configured. /mcp alone shows usage."))
		return nil
	}
	table := [][]string{{"name", "source", "status", "detail"}}
	paths := map[string]string{}
	for _, r := range rows {
		source := r.Source
		if r.Hides != "" {
			source += ", hides " + r.Hides
		}
		if r.Path != "" {
			paths[r.Source] = r.Path
		}
		detail := r.Target()
		if r.State == mcp.Active {
			detail += "  " + plural(r.Tools, "tool") + ", " + a.tokenFigure(r.SchemaChars)
		}
		if r.Detail != "" {
			detail += "  " + r.Detail
		}
		table = append(table, []string{r.Name, source, string(r.State), detail})
	}
	widths := make([]int, 3)
	for _, row := range table {
		for i := range widths {
			widths[i] = max(widths[i], len(row[i]))
		}
	}
	lines := []string{""}
	for i, row := range table {
		line := fmt.Sprintf("%-*s  %-*s  %-*s  %s", widths[0], row[0], widths[1], row[1], widths[2], row[2], row[3])
		if i == 0 {
			line = style.Dim(line)
		}
		lines = append(lines, line)
	}
	if len(paths) > 0 {
		lines = append(lines, "")
		for _, src := range []string{"global", "local"} {
			if p, ok := paths[src]; ok {
				lines = append(lines, style.Dim(fmt.Sprintf("%-6s  %s", src, p)))
			}
		}
	}
	a.out(lines...)
	return nil
}

// mcpImmediate reports whether a /mcp line typed during a turn runs at once
// rather than at the turn boundary: listing always, and anything while the
// request is held waiting for servers, since no request is in flight then.
func (a *App) mcpImmediate(line string) bool {
	if a.mcp == nil {
		return false
	}
	fields := strings.Fields(line)
	if len(fields) == 0 || fields[0] != "/mcp" {
		return false
	}
	return len(fields) == 1 || fields[1] == "list" || a.mcpHolding.Load()
}
