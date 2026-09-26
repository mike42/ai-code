// Command ai-code is a coding agent for the terminal.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/coord"
	"ai-code/internal/prompt"
	"ai-code/internal/provider"
	"ai-code/internal/render"
	"ai-code/internal/runtime"
	"ai-code/internal/session"
	"ai-code/internal/tool"
)

var version = "0.1.0-dev"

type flags struct {
	provider     string
	model        string
	mode         string
	think        string
	runtime      string
	resume       string
	continueLast bool
	print        bool
	jsonOut      bool
	verbose      bool
	noStatus     bool
	help         bool
	version      bool
	args         []string
}

func main() {
	if err := run(); err != nil {
		if errors.Is(err, errQuit) {
			return
		}
		var restart *restartRequest
		if errors.As(err, &restart) {
			// Last thing this process does: every deferred teardown has run by
			// now, which is why the request travels up here.
			err = execSelf(restart.binary, restart.argv)
			fmt.Fprintln(os.Stderr, "ai-code: could not restart: "+err.Error())
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "ai-code: "+err.Error())
		os.Exit(1)
	}
}

var errQuit = errors.New("quit")

// restartRequest asks main to replace this process with a fresh copy of the
// binary, once everything this one holds open has been let go.
type restartRequest struct {
	binary string
	argv   []string
}

func (r *restartRequest) Error() string { return "restart" }

func run() error {
	// Internal mode: serve the tool executor as a JSON-lines server on stdio
	// inside a sandbox, checked before flag parsing.
	if len(os.Args) > 2 && os.Args[1] == "--executor-daemon" {
		if cwd, err := os.Getwd(); err == nil {
			return runExecutorDaemon(cwd)
		}
	}

	f, err := parseFlags(os.Args[1:])
	if err != nil {
		return err
	}
	if f.help {
		fmt.Print(usage)
		return nil
	}
	if f.version {
		fmt.Printf("ai-code %s\n", version)
		return nil
	}

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}

	// A .nocloud file in the working directory or a parent marks the tree
	// off-limits to cloud providers.
	nocloud := config.NoCloud(cwd)
	if nocloud {
		fmt.Fprintln(os.Stderr, "ai-code: .nocloud detected -- restricting to on-premises providers")
	}

	if len(f.args) > 0 {
		switch f.args[0] {
		case "models":
			return cmdModels(f, cwd, nocloud)
		case "config":
			return cmdConfig(cwd, nocloud)
		case "sessions":
			return cmdSessions(cwd)
		}
	}

	cfg, err := config.Load(cwd)
	if err != nil {
		return err
	}

	if nocloud {
		cfg = cfg.OnPremisesOnly()
		if len(cfg.Provider) == 0 {
			return fmt.Errorf("a .nocloud file is present here, so this session is restricted to on-premises providers, " +
				"but none are configured. Add a provider with class = \"on-premises\" and retry.")
		}
	}
	return startSession(f, cfg, cwd, nocloud)
}

const usage = `ai-code -- a coding agent for the terminal

Usage:
  ai-code [flags] [prompt]        Start a session, optionally with an opening prompt
  ai-code models                  List the models a provider offers
  ai-code config                  Show the resolved configuration and where it came from
  ai-code sessions                List saved sessions for this project

Flags:
  -p, --print          Run non-interactively and exit when the model stops
      --json           Emit the event stream as JSON, one object per line
  -c, --continue       Resume the most recent session for this directory
      --resume ID      Resume a specific session
      --provider NAME  Use a configured provider
      --model NAME     Use a specific model
      --mode NAME      Start in a mode (build, research, review, or your own)
      --runtime WHERE  Where tool commands run: devcontainer, ssh://HOST, or host
      --no-status      Do not draw the status line
  -v, --verbose        Start verbose: full thinking, tool arguments and output
  -h, --help           This text
      --version        Print the version

Configuration lives in $XDG_CONFIG_HOME/ai-code/config.toml and .ai-code/config.toml.
Run "ai-code config" to see what is currently in effect.
`

func parseFlags(args []string) (*flags, error) {
	f := &flags{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		next := func() (string, error) {
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s needs a value", a)
			}
			i++
			return args[i], nil
		}
		var err error
		switch a {
		case "-h", "--help":
			f.help = true
		case "--version":
			f.version = true
		case "-p", "--print":
			f.print = true
		case "--json":
			f.jsonOut, f.print = true, true
		case "-v", "--verbose":
			f.verbose = true
		case "--no-status":
			f.noStatus = true
		case "-c", "--continue":
			f.continueLast = true
		case "--resume":
			f.resume, err = next()
		case "--provider":
			f.provider, err = next()
		case "--model":
			f.model, err = next()
		case "--mode":
			f.mode, err = next()
		case "--think":
			f.think, err = next()
		case "--runtime":
			f.runtime, err = next()
		default:
			if strings.HasPrefix(a, "-") && len(a) > 1 {
				return nil, fmt.Errorf("unknown flag %s (try --help)", a)
			}
			f.args = append(f.args, a)
		}
		if err != nil {
			return nil, err
		}
	}
	return f, nil
}

// startSession builds everything and hands off to the REPL or a single run.
func startSession(f *flags, cfg *config.Config, cwd string, nocloud bool) error {
	providerName := f.provider
	if providerName == "" {
		providerName = cfg.DefaultProvider
	}
	if providerName == "" && len(cfg.Provider) == 1 {
		providerName = cfg.ProviderNames()[0]
	}
	if providerName == "" {
		return noProviderError(cfg)
	}
	pc, ok := cfg.Provider[providerName]
	if !ok {
		return fmt.Errorf("no provider named %q; configured: %s",
			providerName, strings.Join(cfg.ProviderNames(), ", "))
	}

	client, err := buildClient(providerName, pc)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	model, err := resolveModel(ctx, client, f.model, pc.DefaultModel)
	if err != nil {
		return err
	}
	limit, limitSource := contextLimitFor(model, cfg.Agent.ContextOverride)

	modeName := f.mode
	if modeName == "" {
		modeName = cfg.DefaultMode
	}
	mode, ok := cfg.Mode[modeName]
	if !ok {
		return fmt.Errorf("no mode named %q; available: %s",
			modeName, strings.Join(cfg.ModeNames(), ", "))
	}

	// The project root (git root, or cwd) is what a devcontainer mounts and
	// tools run against.
	env := prompt.DetectEnv(cwd)
	projectRoot := env.GitRoot
	if projectRoot == "" {
		projectRoot = cwd
	}
	// Kept before anything rewrites env for a container: AGENTS.md files are
	// read off this machine's disk.
	hostGitRoot := env.GitRoot
	// nil means the model's paths and this machine's are the same thing.
	var showPath func(string) (string, bool)

	// Where tools run is a safety decision, fixed for the session. Slow work is
	// deferred to the first tool call.
	rt, err := runtime.Resolve(f.runtime, cwd)
	if err != nil {
		return err
	}

	// The local executor runs tools in this process; the devcontainer executor
	// starts a container on the first tool call.
	var exe tool.Executor
	var toolState *tool.State
	switch rt.Kind {
	case runtime.KindDevcontainer:
		binary, err := exec.LookPath(os.Args[0])
		if err != nil {
			return fmt.Errorf("could not find the ai-code binary to run in the container: %w", err)
		}
		// Expand ${...} references before anything reads the config;
		// workspaceFolder is resolved here.
		if rt.Config != nil {
			rt.Config.SubstituteAll(projectRoot, devcontainerID(projectRoot))
		}
		dc := NewDevcontainerExecutor(rt.Config, rt.ConfigErr, projectRoot, cwd, binary)
		exe = dc
		defer dc.Close()

		// Tools run inside the container, where the project is mounted at
		// workspaceFolder; telling the model host paths would be wrong.
		workspaceFolder := "/workspace"
		if rt.Config != nil {
			workspaceFolder = rt.Config.WorkspaceFolder
		}
		env.Cwd = mountCwd(projectRoot, cwd, workspaceFolder)
		if env.IsGitRepo {
			env.GitRoot = workspaceFolder
		}
		// The shell and platform belong to the container: the model picks
		// commands on the strength of them.
		env.Shell = ""
		env.OS = "linux"
		showPath = func(hostPath string) (string, bool) {
			rel, err := filepath.Rel(projectRoot, hostPath)
			if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
				// Outside the mount: nothing in the container corresponds.
				return "", false
			}
			return path.Join(workspaceFolder, filepath.ToSlash(rel)), true
		}

		if rt.Config != nil {
			if rewritten := rt.Config.EnvRewrites(); len(rewritten) > 0 {
				fmt.Fprintf(os.Stderr, "devcontainer.json: ${localWorkspaceFolder} in %s resolved to "+
					"the container's workspace folder, not this machine's path\n",
					strings.Join(rewritten, ", "))
			}
		}

		// A configuration key read but not acted on is reported here, rather
		// than silently ignored.
		if rt.Config != nil {
			if missing := rt.Config.Unsupported(); len(missing) > 0 {
				fmt.Fprintf(os.Stderr, "devcontainer.json: not applied by this build: %s\n",
					strings.Join(missing, ", "))
			}
		}
	default: // host
		local, st := buildTools(cfg, cwd, limit)
		exe, toolState = local, st
	}

	thinking, err := provider.ParseEffort(f.think)
	if err != nil {
		return err
	}
	if f.think == "" {
		if thinking, err = provider.ParseEffort(cfg.Agent.Thinking); err != nil {
			return fmt.Errorf("agent.thinking: %w", err)
		}
	}
	// The task tool sits outside the executor boundary because a worker needs
	// the agent; dormant until Attach.
	tasks := agent.NewTaskExecutor(exe)
	exe = tasks

	// Non-interactive output consumes the same event stream as the terminal.
	screen := render.NewScreen(os.Stdout, cfg.UI.Color)
	defer watchResize(screen)()
	var (
		sink        agent.Sink
		interactive *render.Interactive
		verbosity   interface{ SetVerbose(bool) bool }
	)
	switch {
	case f.jsonOut:
		sink = render.NewJSON(os.Stdout)
	case !screen.IsTTY():
		// Piped or redirected: plain text, no escapes, no cursor movement.
		plain := render.NewPlain(os.Stdout, f.verbose)
		sink, verbosity = plain, plain
	default:
		// A terminal gets the full renderer even for --print; the output
		// destination decides, not the mode.
		showStatus := !f.noStatus && (cfg.UI.StatusLine == nil || *cfg.UI.StatusLine)
		interactive = render.NewInteractive(screen, render.InteractiveOptions{
			ShowStatus:  showStatus,
			Theme:       cfg.UI.Theme,
			Reasoning:   cfg.UI.Reasoning,
			WarnPercent: cfg.UI.ContextWarnPercent,
			SteerPrompt: promptMarker,
			Verbose:     f.verbose,
		})
		interactive.Start()
		defer interactive.Close()
		sink, verbosity = interactive, interactive
	}

	app := &App{
		cfg:          cfg,
		cwd:          cwd,
		projectRoot:  projectRoot,
		hostGitRoot:  hostGitRoot,
		showPath:     showPath,
		client:       client,
		providerName: providerName,
		providerCfg:  pc,
		model:        model,
		// The banner is about to say this, so a later change is measured
		// against it.
		knownModel:  model.ID,
		modeName:    modeName,
		exec:        exe,
		toolState:   toolState,
		screen:      screen,
		interactive: interactive,
		verbosity:   verbosity,
		env:         env,
		limitSource: limitSource,
		noCloud:     nocloud,
		rt:          rt,
		flags:       f,
	}

	// An executor that blocks for minutes on its first call reports through
	// the renderer's own zone.
	if pr, ok := exe.(tool.ProgressReporter); ok {
		pr.SetProgress(app.progress)
	}

	// The session file records everything, so a crash costs at most the turn
	// in progress.
	if err := app.openSession(f); err != nil {
		return err
	}
	defer app.closeSession()

	// The renderer is the only event sink; messages reach the transcript from
	// the agent's message list, which holds exactly what was sent. The App is a
	// second consumer of the stream, for the prompt marker's context figure.
	app.agent = agent.New(client, model.ID, exe,
		agent.MultiSink{sink, agent.SinkFunc(app.noteContext)}, agent.Options{
			MaxIterations:   cfg.Agent.MaxIterations,
			MaxTokens:       cfg.Agent.MaxTokens,
			MaxOutputTokens: model.MaxOutputTokens,
			Temperature:     cfg.Agent.Temperature,
			TopP:            cfg.Agent.TopP,
			ParallelReads:   cfg.Agent.ParallelReads == nil || *cfg.Agent.ParallelReads,
			LoopGuard:       cfg.Agent.LoopGuard,
			ContextLimit:    limit,
			WarnPercent:     cfg.UI.ContextWarnPercent,

			Effort: thinking,

			AutoCompact:      cfg.Agent.AutoCompact == nil || *cfg.Agent.AutoCompact,
			ReserveTokens:    cfg.Agent.CompactReserveTokens,
			KeepRecentTokens: cfg.Agent.CompactKeepRecentTokens,
		})
	// After Attach, so the prompt recites the task tool.
	tasks.Attach(ctx, app.agent)
	app.workers = tasks.Pool()
	app.wireWorkers()
	// Workers outlive turns, so nothing else ends them; without this one would
	// keep writing after the prompt is gone.
	defer app.workers.Cancel()
	// The loop boundary is the only place a long agentic run can be stopped
	// from outside.
	app.agent.SetTurnBoundary(app.stopForSwap)

	// A failed registration is not worth reporting: the session works without it.
	app.coordDir = coord.DefaultDir()
	if reg, err := coord.Register(app.coordDir, coord.Peer{
		Session:  sessionID(app),
		Provider: providerName,
		Model:    model.ID,
		Cwd:      cwd,
	}); err == nil {
		app.peers = reg
		defer reg.Close()
	}
	app.rebuildSystemPrompt(mode.Prompt)
	if len(app.resumed) > 0 {
		app.agent.SetMessages(app.resumed)
		// After the messages, because the checkpoint is an index into them.
		app.agent.SetCheckpoint(app.resumedSummary, app.resumedThrough, app.resumedCut)
	}

	opening := strings.Join(f.args, " ")

	if f.print {
		if opening == "" {
			data, _ := readAllStdin()
			opening = strings.TrimSpace(data)
		}
		if opening == "" {
			return errors.New("nothing to do: pass a prompt, or pipe one in")
		}
		if err := app.runTurn(ctx, opening); err != nil {
			return err
		}
		return app.drainWorkers(ctx)
	}

	return app.repl(ctx, cancel, opening)
}

func noProviderError(cfg *config.Config) error {
	path := config.UserConfigPath()
	return fmt.Errorf(`no provider is configured.

ai-code does not ship with a provider, because it does not favour one. Add yours to
%s, for example:

    default_provider = "home"

    [provider.home]
    kind         = "lemonade"
    base_url     = "https://ai.example.internal/api/v1"
    class        = "on-premises"
    tls_insecure = true

kind may be "lemonade", "openai" (any OpenAI-compatible endpoint) or "openrouter".
class is required: ai-code asks for confirmation before sending a live session to a
"cloud" provider, and never asks for an "on-premises" one`, path)
}

func readAllStdin() (string, error) {
	fi, err := os.Stdin.Stat()
	if err != nil || fi.Mode()&os.ModeCharDevice != 0 {
		return "", nil
	}
	var b strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := os.Stdin.Read(buf)
		b.Write(buf[:n])
		if err != nil {
			break
		}
	}
	return b.String(), nil
}

// installInterrupt wires Ctrl-C to context cancellation rather than to process
// death. A killed process leaves the transcript with an unanswered tool call;
// a cancelled context lets the agent close those out properly.
func installInterrupt(cancel context.CancelFunc) func() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		select {
		case <-ch:
			cancel()
		case <-done:
		}
	}()
	return func() {
		signal.Stop(ch)
		close(done)
	}
}

func cmdSessions(cwd string) error {
	env := prompt.DetectEnv(cwd)
	root := env.GitRoot
	if root == "" {
		root = cwd
	}
	list, err := session.List(root)
	if err != nil {
		return err
	}
	if len(list) == 0 {
		fmt.Printf("No sessions recorded for %s\n", root)
		return nil
	}
	fmt.Printf("Sessions for %s\n\n", root)
	for _, s := range list {
		fmt.Printf("  %s  %s  %d messages  %s\n",
			s.ID, s.Modified.Format("2006-01-02 15:04"), s.Messages, s.Model)
		if s.Preview != "" {
			fmt.Printf("      %s\n", s.Preview)
		}
	}
	fmt.Printf("\nResume with: ai-code --resume <id>\n")
	return nil
}

func cmdConfig(cwd string, nocloud bool) error {
	cfg, err := config.Load(cwd)
	if err != nil {
		return err
	}
	if nocloud {
		fmt.Println("a .nocloud file is present -- cloud providers are unavailable for this tree.")
		cfg = cfg.OnPremisesOnly()
	}
	fmt.Println("Configuration files, in the order they were applied:")
	if len(cfg.Sources) == 0 {
		fmt.Println("  (none found -- built-in defaults only)")
		fmt.Printf("  would read: %s\n", config.UserConfigPath())
		fmt.Printf("  would read: %s\n", filepath.Join(cwd, ".ai-code", "config.toml"))
	}
	for _, s := range cfg.Sources {
		fmt.Printf("  %s\n", s)
	}

	fmt.Printf("\nDefault provider: %s\n", orNone(cfg.DefaultProvider))
	fmt.Printf("Default mode:     %s\n", orNone(cfg.DefaultMode))

	fmt.Printf("\nProviders:\n")
	if len(cfg.Provider) == 0 {
		fmt.Println("  (none configured)")
	}
	for _, name := range cfg.ProviderNames() {
		p := cfg.Provider[name]
		kind := p.Kind
		if kind == "" {
			kind = "openai"
		}
		fmt.Printf("  %-12s %s  kind=%s class=%s", name, p.BaseURL, kind, p.Class)
		if p.TLSInsecure {
			fmt.Print("  tls=insecure")
		}
		if p.DefaultModel != "" {
			fmt.Printf("  model=%s", p.DefaultModel)
		}
		fmt.Println()
	}

	fmt.Printf("\nModes:\n")
	for _, name := range cfg.ModeNames() {
		fmt.Printf("  %-12s %s\n", name, cfg.Mode[name].Description)
	}
	return nil
}

func orNone(s string) string {
	if s == "" {
		return "(not set)"
	}
	return s
}

func cmdModels(f *flags, cwd string, nocloud bool) error {
	cfg, err := config.Load(cwd)
	if err != nil {
		return err
	}
	if nocloud {
		cfg = cfg.OnPremisesOnly()
	}
	name := f.provider
	if name == "" {
		name = cfg.DefaultProvider
	}
	if name == "" && len(cfg.Provider) == 1 {
		name = cfg.ProviderNames()[0]
	}
	pc, ok := cfg.Provider[name]
	if !ok {
		if nocloud {
			return fmt.Errorf("a .nocloud file is present here, so only on-premises providers are available. " +
				"Add a provider with class = \"on-premises\" and retry.")
		}
		return noProviderError(cfg)
	}
	client, err := buildClient(name, pc)
	if err != nil {
		return err
	}

	models, err := fetchModels(context.Background(), client, false)
	if err != nil {
		return err
	}
	fmt.Printf("%s (%s)\n\n", name, pc.BaseURL)
	fmt.Println(formatModelList(models, 500))

	// A large ctx_size split across several parallel slots gives each request a
	// fraction of it.
	for _, m := range models {
		if windowIsDivided(m) {
			fmt.Printf("\nNote: %s splits a %s KV cache across %d slots,\n"+
				"so a single request gets %s, not %s.\n",
				m.ID, compactInt(m.CtxSize), m.Parallel,
				compactInt(m.ContextWindow), compactInt(m.CtxSize))
		}
	}
	return nil
}

// sessionID is the session's id when it is being recorded, and "" otherwise.
func sessionID(a *App) string {
	if a.sess == nil {
		return ""
	}
	return a.sess.Meta.ID
}
