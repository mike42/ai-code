package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/coord"
	"ai-code/internal/prompt"
	"ai-code/internal/provider"
	"ai-code/internal/render"
	"ai-code/internal/runtime"
	"ai-code/internal/session"
	"ai-code/internal/tool"
	"ai-code/internal/ui"
)

// App holds the wiring for one interactive session.
type App struct {
	cfg          *config.Config
	cwd          string
	projectRoot  string
	client       provider.Client
	providerName string
	providerCfg  config.Provider
	model        provider.ModelInfo
	modeName     string
	limitSource  string
	// noCloud is true when a .nocloud file was found in the working directory
	// or a parent. Cloud providers are then unavailable for the whole session.
	noCloud bool

	// hostGitRoot is the repository root on this machine. env.GitRoot may
	// hold the container's path instead, and AGENTS.md files are read from
	// the host's filesystem either way.
	hostGitRoot string
	// showPath maps a host path to the one the model should be told about.
	// nil when they are the same, which is every session that is not running
	// its tools in a container. The bool is false for a file the container
	// cannot see at all: its content still reaches the model, but naming a
	// path the model cannot act on is worse than naming none.
	showPath func(string) (string, bool)

	exec      tool.Executor
	toolState *tool.State
	agent     *agent.Agent

	// workers is the set of sub-agents running beside this session. They
	// outlive the turn that started them, so they are the session's and not
	// the turn's.
	workers *agent.Workers
	// editor is the live prompt, set for as long as the REPL owns the
	// terminal. A worker finishing while the user is typing reaches the
	// screen through it; nil in a piped session, where there is no prompt to
	// write above.
	editorMu sync.Mutex
	editor   *ui.Editor

	// rt is the resolved execution environment for this session. It is fixed at
	// startup and never changes.
	rt *runtime.Runtime

	screen      *render.Screen
	interactive *render.Interactive
	env         prompt.Env

	// ctxState is the context figure as last published by the agent. See
	// noteContext: it is adopted, never computed.
	ctxMu    sync.Mutex
	ctxState agent.ContextState

	// verbosity is the renderer's detail level, driven by /verbose and /quiet.
	// Both the terminal and the plain renderer offer it; the JSON stream does
	// not, and there it is nil.
	verbosity interface{ SetVerbose(bool) bool }

	sess    *session.Session
	resumed []provider.Message
	// resumedInputs is the command history recovered from the session file, so
	// a /restart or a --resume does not land the user at a prompt with no Up
	// arrow.
	resumedInputs []string
	recorded      int
	// recordedSummary is the standby checkpoint already on disk, so the same
	// text is not appended again every turn.
	recordedSummary string

	// resumedSummary is the standby checkpoint recovered from the session
	// file, and resumedThrough how much of the transcript it accounts for.
	// Without these a restart throws away the summary it was most likely to
	// need -- the swap that prompted the restart is the next thing to happen.
	resumedSummary string
	resumedThrough int
	// resumedCut is the boundary a compaction in the recorded session chose,
	// zero when the checkpoint was a standby one. Without it a resumed
	// session would assemble from the whole replayed transcript and undo the
	// compaction it is resuming from.
	resumedCut int

	// queuedCmds are slash commands typed while a turn was running. They run
	// at the turn boundary; see startSteering.
	// steerMu guards steering, which says whether the terminal is already
	// open for input somewhere up the stack.
	steerMu  sync.Mutex
	steering bool

	cmdMu      sync.Mutex
	queuedCmds []string
	// deferredPrompts are lines submitted while a model swap was running, to
	// be run once it finishes, on whichever model it loaded.
	deferredPrompts []string
	// turnErr is an errQuit or a restart raised by one of them, which has to
	// leave the turn before it can be acted on.
	turnErr error

	// knownModel is the model the scrollback last said this session was on:
	// the startup banner, a /model line, or a change notice that a prompt
	// has since been submitted underneath.
	//
	// One invariant governs the whole report: the line above the prompt
	// shows the difference between what the scrollback says and what is
	// true. Equal means nothing is shown, which is why swapping away and
	// back leaves no trace, and why a session that has sent nothing still
	// reports -- its banner made a claim that is no longer true.
	knownModel string

	// pendingInput is text the user was part-way through typing when the last
	// turn ended. It seeds the next prompt.
	pendingInput string

	// flags is what this process was launched with, kept so /restart can hand
	// the display options on to its replacement.
	flags *flags

	// peers is what this window publishes about itself for the others to
	// read, and coordDir is the directory they share. Both are usable when
	// nil or empty: coordination is an improvement on a session that works
	// without it.
	peers    *coord.Registration
	coordDir string
}

func (a *App) openSession(f *flags) error {
	id := f.resume
	if f.continueLast && id == "" {
		latest, err := session.Latest(a.projectRoot)
		if err != nil {
			return err
		}
		id = latest.ID
	}

	if id != "" {
		s, entries, err := session.Open(a.projectRoot, id)
		if err != nil {
			return err
		}
		a.sess = s
		a.resumed = session.Messages(entries)
		a.recorded = len(a.resumed)
		a.resumedInputs = session.Inputs(entries)
		a.resumedSummary, a.resumedThrough, a.resumedCut = session.Checkpoint(entries)
		a.recordedSummary = a.resumedSummary

		if s.Meta.Project != "" && s.Meta.Project != a.projectRoot {
			// A moved or renamed repository. Say so rather than silently
			// resuming against paths that may no longer exist.
			a.note(fmt.Sprintf("This session was recorded in %s, which is not where you are now. "+
				"File paths in its history may be stale.", s.Meta.Project))
		}
		return nil
	}

	s, err := session.Create(session.Meta{
		Project:  a.projectRoot,
		Provider: a.providerName,
		Model:    a.model.ID,
		Mode:     a.modeName,
		Runtime:  a.runtimeLabel(),
		Version:  version,
	})
	if err != nil {
		return err
	}
	a.sess = s
	return nil
}

func (a *App) closeSession() {
	if a.sess != nil {
		_ = a.sess.Close()
	}
}

// recordInput logs a submitted line so command history survives a restart.
//
// Separate from recordNew because a slash command never reaches the agent's
// message list at all, and the prompts that do are wrapped and later dropped
// by compaction. Neither round-trips as something to press Up and re-run.
func (a *App) recordInput(line string) {
	if a.sess == nil || strings.TrimSpace(line) == "" {
		return
	}
	_ = a.sess.Append(session.Entry{Type: session.EntryInput, Time: time.Now(), Input: line})
}

// recordNew appends messages added since the last call.
func (a *App) recordNew() {
	if a.sess == nil {
		return
	}
	msgs := a.agent.Messages()
	for i := a.recorded; i < len(msgs); i++ {
		_ = a.sess.AppendMessage(msgs[i])
	}
	a.recorded = len(msgs)
}

func (a *App) note(text string) {
	if a.interactive != nil {
		a.interactive.Emit(agent.Event{Kind: agent.EvNotice, Level: agent.LevelInfo, Text: text})
		return
	}
	fmt.Fprintf(os.Stderr, "note: %s\n", text)
}

// progress reports what a slow tool call is waiting for.
//
// Into the transient zone, where each line replaces the last and the status
// line sits underneath it, so a container start that takes minutes says what
// it is doing the whole time without scrolling a screen of history for it.
// The scrollback gets one line at the end instead; see noteReady.
//
// Without a terminal there is no transient zone, so it goes to stderr, where
// a piped run still shows why it paused.
func (a *App) progress(p tool.Phase) {
	if a.interactive == nil {
		fmt.Fprintf(os.Stderr, "note: %s\n", p.Note)
		return
	}
	if p.Done {
		a.interactive.SetNotice("")
		a.note(p.Note)
		return
	}
	a.interactive.SetNotice(p.Note)
}

func (a *App) out(lines ...string) {
	if a.screen != nil && a.screen.IsTTY() {
		a.screen.CommitBlock(lines...)
		return
	}
	for _, l := range lines {
		fmt.Println(render.Strip(l))
	}
}

// runTurn executes one exchange, with Ctrl-C bound to cancellation and the
// terminal open for steering.
func (a *App) runTurn(parent context.Context, input string) error {
	ctx, cancel := context.WithCancel(parent)
	restore := installInterrupt(cancel)
	defer restore()
	defer cancel()

	// Under the same lock a swap decides with: take whatever is loaded and
	// declare this session working on it, so a swap either waits for this
	// turn or this turn yields to the swap. Nothing in between.
	a.claimModel(ctx)

	a.peers.SetBusy(true)
	defer func() {
		a.peers.SetBusy(false)
		a.peers.SetState(coord.StateFree)
	}()

	stopSteering := a.startSteering(cancel)

	// For the whole turn, not just the parts that emit events. Compaction
	// runs between requests and emits nothing while it works.
	if a.interactive != nil {
		a.interactive.SetBusy(true)
		defer a.interactive.SetBusy(false)
	}

	err := a.agent.Run(ctx, input)

	// Anything half-typed when the turn ended goes to the next prompt rather
	// than into the void.
	a.pendingInput = stopSteering()

	if ctx.Err() != nil {
		// The turn was cancelled. A steering message queued but not yet folded
		// in belongs to the course the user has just abandoned, so it is
		// dropped rather than delivered with whatever they type next.
		if n := len(a.agent.TakeSteering()); n > 0 {
			noun := "steering message"
			if n > 1 {
				noun += "s"
			}
			a.note(fmt.Sprintf("Discarded %d %s queued while the cancelled turn was running.", n, noun))
		}
	}

	a.recordNew()
	// An automatic compaction inside the turn leaves a checkpoint behind.
	a.recordCheckpoint()

	// The turn is over and the message list is complete: the loop boundary
	// where a new user message is legal, and the first moment since the turn
	// began that a summary can be written. A swap announced while this turn
	// was running has been waiting for exactly this.
	a.prepareForSwap(parent, nil)

	// A /quit or /restart typed during the turn could not act from inside
	// it; this is where it leaves.
	if a.turnErr != nil {
		err, a.turnErr = a.turnErr, nil
	}
	return err
}

// startSteering opens the terminal for input while the turn runs, and returns a
// function that closes it and yields whatever was typed but not submitted.
//
// Steering is best-effort. If the terminal cannot be put into the mode it needs
// -- not a tty, an unsupported platform, an ioctl that fails -- the turn runs
// exactly as it did before, with the status line and nothing else. It is not
// worth failing a request over.
func (a *App) startSteering(cancel context.CancelFunc) func() string {
	return a.steerWith(cancel, nil)
}

// steerWith opens the terminal with a caller-supplied action for a submitted
// line. Nil means the usual one: fold it into the running turn.
func (a *App) steerWith(cancel context.CancelFunc, submit func(string)) func() string {
	if a.interactive == nil || !ui.SteeringAvailable(int(os.Stdin.Fd())) {
		return func() string { return "" }
	}

	// One reader of the terminal at a time. A /model typed mid-turn runs at
	// the loop boundary, inside a turn that already has the terminal open,
	// and a second reader on the same descriptor means two goroutines
	// racing for each keystroke.
	a.steerMu.Lock()
	if a.steering {
		a.steerMu.Unlock()
		return func() string { return "" }
	}
	a.steering = true
	a.steerMu.Unlock()

	st := ui.NewSteerer(int(os.Stdin.Fd()))
	st.Render = a.interactive.SetSteering
	if submit != nil {
		st.Submit = submit
	} else {
		st.Submit = a.steerSubmit
	}
	st.Interrupt = cancel
	st.Control = a.interactive.Control
	// The same candidates the prompt offers. A path is no easier to type
	// while a turn is running than it is at the prompt.
	st.Completions = a.complete

	if err := st.Start(); err != nil {
		a.steerMu.Lock()
		a.steering = false
		a.steerMu.Unlock()
		return func() string { return "" }
	}
	return func() string {
		left := st.Stop()
		a.interactive.SetSteering("", 0, false)
		a.steerMu.Lock()
		a.steering = false
		a.steerMu.Unlock()
		return left
	}
}

func (a *App) steerSubmit(text string) {
	{
		// A slash command typed mid-turn is a command, the same as one typed
		// at the prompt. It runs at the turn boundary rather than here,
		// because that is where the conversation is in a state a command can
		// safely change -- and it is the same point steering is folded in,
		// so the two stay in the order they were typed.
		if strings.HasPrefix(strings.TrimSpace(text), "/") {
			// Most commands wait for the turn boundary, below. These two do
			// not: they change only how output is drawn, touching neither the
			// conversation nor the model, and waiting defeats the point of
			// them. A turn boundary arrives after the next round of tool
			// results, which on a model thinking at a few tokens a second is
			// minutes away -- and the thinking the user turned verbose on to
			// read is over by then.
			if renderOnlyCommand(text) {
				if err := a.command(context.Background(), text); err != nil {
					a.reportError(err)
				}
				return
			}
			a.queueCommand(text)
			a.interactive.SetQueued(a.agent.Pending() + a.pendingCommands())
			return
		}
		if !a.agent.Steer(text) {
			return
		}
		// The status line acknowledges it; the scrollback does not. The agent
		// emits EvSteer when the message is actually folded into the
		// conversation, and that is the honest moment to commit a line saying
		// so -- until then the model is still working from the old
		// instruction, and a line in the scrollback would be read as the point
		// at which it changed course.
		a.interactive.SetQueued(a.agent.Pending() + a.pendingCommands())
	}
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

func (a *App) repl(ctx context.Context, cancel context.CancelFunc, opening string) error {
	editor := ui.NewEditor(os.Stdin, os.Stdout)
	editor.EditorCommand = a.cfg.Editor
	editor.Completions = a.complete
	// Published for as long as this loop owns the terminal, so a worker
	// finishing between turns has somewhere to print.
	a.setEditor(editor)
	defer a.setEditor(nil)

	a.banner()

	// Before anything typed this run, so Up walks back through the whole
	// session rather than only the part since the last restart.
	for _, h := range a.resumedInputs {
		editor.AddHistory(h)
	}

	if opening != "" {
		editor.AddHistory(opening)
		a.recordInput(opening)
		if err := a.handleInput(ctx, opening); err != nil {
			switch {
			case errors.Is(err, errQuit):
				return nil
			case isRestart(err):
				return err
			default:
				a.reportError(err)
			}
		}
	}

	for {
		// A prompt typed during a model swap runs now, on the model the swap
		// loaded, without being retyped.
		if line, ok := a.takeDeferredPrompt(); ok {
			a.echoPrompt(line)
			if err := a.handleInput(ctx, line); err != nil {
				switch {
				case errors.Is(err, errQuit):
					return nil
				case isRestart(err):
					return err
				default:
					a.reportError(err)
				}
			}
			continue
		}

		if a.pendingInput != "" {
			editor.Preload(a.pendingInput)
			a.pendingInput = ""
		}
		// Built before the background work starts, because it reads the same
		// agent state that work is about to write to.
		prompt := a.promptString()
		stopIdle := a.startIdleCheckpoint(ctx, editor)
		line, err := editor.ReadLine(prompt)
		stopIdle()

		switch {
		case errors.Is(err, ui.ErrEOF):
			a.out("")
			return nil
		case errors.Is(err, ui.ErrInterrupt):
			// Ctrl-C on an empty prompt is the only gesture that reaches a
			// background worker: the turn that started it has long since
			// ended, so there is no run to interrupt instead.
			a.stopWorkers()
			continue
		case err != nil:
			return err
		}

		// Whatever the header said now describes the exchange below it: it
		// stops being rewritten, stays where it is, and becomes what the
		// scrollback says this session is on.
		editor.ForgetHeader()
		a.knownModel = a.model.ID

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		editor.AddHistory(line)
		a.recordInput(line)

		if err := a.handleInput(ctx, line); err != nil {
			switch {
			case errors.Is(err, errQuit):
				return nil
			case isRestart(err):
				return err
			default:
				a.reportError(err)
			}
		}
	}
}

func (a *App) reportError(err error) {
	if a.interactive != nil {
		a.interactive.Emit(agent.Event{Kind: agent.EvError, Text: err.Error()})
		return
	}
	fmt.Fprintln(os.Stderr, "error: "+err.Error())
}

func (a *App) handleInput(ctx context.Context, line string) error {
	if strings.HasPrefix(line, "/") {
		return a.command(ctx, line)
	}

	a.echoTypedPrompt(line)
	return a.runTurn(ctx, line)
}

// echoPrompt puts a prompt into the scrollback before the answer to it
// arrives. Without this the transcript has a gap exactly where the question
// should be.
//
// It is for input the terminal was never left showing: /editor composes
// somewhere else entirely and returns with the prompt line untouched.
func (a *App) echoPrompt(text string) {
	if a.screen == nil || !a.screen.IsTTY() {
		return
	}
	a.out(render.Echo(render.NewStyle(a.screen.Color()), "prompt", text)...)
}

// echoTypedPrompt is echoPrompt for input that came back from the line editor,
// which finishes drawing a single line where it will stay but can only
// summarise a multi-line one.
func (a *App) echoTypedPrompt(text string) {
	if !render.IsMultiline(text) {
		return
	}
	a.echoPrompt(text)
}

// runtimeLabel is the canonical --runtime value for the session's execution
// environment, for the session transcript. It stays fixed for the session.
func (a *App) runtimeLabel() string {
	if a.rt == nil {
		return "host"
	}
	return a.rt.FlagValue()
}

func (a *App) banner() {
	style := render.NewStyle(a.screen.Color())
	limit, _ := contextLimitFor(a.model, a.cfg.Agent.ContextOverride)

	cls := ""
	if a.client.Class() == provider.ClassCloud {
		// Only cloud needs flagging: it is the one case where the provider name
		// does not already say where the data goes. "home · on-premises" is
		// redundant, so the class is omitted for on-premises providers.
		cls = " · " + style.Warn(string(a.client.Class()))
	}

	rtLine := ""
	if a.rt != nil {
		if a.rt.Sandboxed() {
			rtLine = "   " + style.Dim("runtime: "+a.rt.ShortLabel())
		} else {
			// The unsandboxed case is deliberately louder. Running tools directly
			// on the host is always an explicit choice, and the banner should not
			// let it fade into the noise of the other status text.
			rtLine = "   " + style.Warn("runtime: "+a.rt.ShortLabel())
		}
	}

	// The slot arrangement qualifies the context figure, so it belongs beside
	// it rather than on a line of its own.
	ctx := compactInt(limit) + " ctx"
	if s := slotSummary(a.model); s != "" {
		ctx += " (" + s + ")"
	}
	if e := a.agent.Effort(); e != provider.EffortUnset {
		ctx += "   thinking " + effortLabel(e)
	}

	a.out(
		fmt.Sprintf("%s %s   %s   %s%s   %s%s",
			style.Bold("ai-code"), style.Dim(version),
			a.model.ID,
			style.Dim(a.providerName),
			cls,
			style.Dim(ctx),
			rtLine),
		style.Dim(fmt.Sprintf("%s · /help for commands, Ctrl-G to open an editor",
			a.cwd)),
		"",
	)
}

// noteContext adopts a context figure from the event stream.
//
// The App is a consumer like the renderer, not a second source: asking the
// agent directly is how the prompt marker and the status line came to
// disagree after a /compact, one of them reading a value the other had never
// been told about.
func (a *App) noteContext(e agent.Event) {
	if e.Context == nil {
		return
	}
	a.ctxMu.Lock()
	a.ctxState = *e.Context
	a.ctxMu.Unlock()
}

func (a *App) contextState() agent.ContextState {
	a.ctxMu.Lock()
	defer a.ctxMu.Unlock()
	return a.ctxState
}

func (a *App) promptString() string {
	style := render.NewStyle(a.screen.Color())
	cs := a.contextState()

	// ASCII ">", not a chevron. The prompt marker is the one character that is
	// on screen in every session and in every screenshot and paste of one, and
	// U+203A is exactly the sort of character that a font falls back on, a
	// terminal mismeasures, or a copy into a bug report turns into a question
	// mark. Nothing about it was worth that.
	marker := promptMarker
	if a.modeName != "build" {
		marker = a.modeName + " " + promptMarker
	}
	workers := ""
	if m := a.workerMarker(); m != "" {
		workers = style.Dim(m)
	}
	if cs.Window > 0 && cs.Percent() >= a.cfg.UI.ContextWarnPercent {
		return workers + style.Warn(fmt.Sprintf("[%d%%] ", cs.Percent())) + style.Bold(marker)
	}
	return workers + style.Bold(marker)
}

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

// promptMarker is the prompt, and the steering prompt shown mid-turn.
const promptMarker = "> "

type command struct {
	name    string
	summary string
	run     func(a *App, ctx context.Context, args string) error
}

var commands []command

func init() {
	commands = []command{
		{"help", "Show this list", (*App).cmdHelp},
		{"model", "Show or switch the model", (*App).cmdModel},
		{"provider", "Show or switch the provider", (*App).cmdProvider},
		{"mode", "Show or switch the mode", (*App).cmdMode},
		{"compact", "Summarise the session to free context", (*App).cmdCompact},
		{"consult", "Ask the model to argue against the work so far", (*App).cmdConsult},
		{"think", "Show or change how hard the model thinks", (*App).cmdThink},
		{"tokens", "Show context usage", (*App).cmdTokens},
		{"tools", "List the tools available to the model", (*App).cmdTools},
		{"sessions", "List saved sessions for this project", (*App).cmdSessionList},
		{"new", "Start a fresh session, keeping this one on disk", (*App).cmdNew},
		{"editor", "Compose a prompt in $EDITOR", (*App).cmdEditor},
		{"verbose", "Show full thinking, tool arguments and tool output", (*App).cmdVerbose},
		{"quiet", "Back to the thinking marquee and one-line tool summaries", (*App).cmdQuiet},
		{"clear", "Clear the screen (scrollback is preserved)", (*App).cmdClear},
		{"restart", "Restart ai-code and resume this session", (*App).cmdRestart},
		{"quit", "Exit", (*App).cmdQuit},
	}
	sort.Slice(commands, func(i, j int) bool { return commands[i].name < commands[j].name })
}

func (a *App) command(ctx context.Context, line string) error {
	name, args, _ := strings.Cut(strings.TrimPrefix(line, "/"), " ")
	args = strings.TrimSpace(args)

	matches := matchCommands(strings.ToLower(name))
	switch len(matches) {
	case 0:
		return fmt.Errorf("no command /%s; /help lists them", name)
	case 1:
		return matches[0].run(a, ctx, args)
	default:
		var names []string
		for _, m := range matches {
			names = append(names, "/"+m.name)
		}
		return fmt.Errorf("/%s is ambiguous: %s", name, strings.Join(names, ", "))
	}
}

// matchCommands resolves a typed name to the commands it could mean: an exact
// name wins outright, otherwise every command it prefixes.
func matchCommands(name string) []command {
	var matches []command
	for _, c := range commands {
		if c.name == name {
			return []command{c}
		}
		if strings.HasPrefix(c.name, name) {
			matches = append(matches, c)
		}
	}
	return matches
}

func (a *App) cmdHelp(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())
	lines := []string{"", style.Bold("Commands")}
	for _, c := range commands {
		lines = append(lines, fmt.Sprintf("  %-12s %s", "/"+c.name, style.Dim(c.summary)))
	}
	lines = append(lines,
		"",
		style.Bold("Keys"),
		"  "+fmt.Sprintf("%-12s %s", "Ctrl-G", style.Dim("open $EDITOR for a long prompt")),
		"  "+fmt.Sprintf("%-12s %s", "Ctrl-C", style.Dim("cancel the running turn (press again at the prompt to clear)")),
		"  "+fmt.Sprintf("%-12s %s", "Ctrl-D", style.Dim("exit")),
		"  "+fmt.Sprintf("%-12s %s", "Tab", style.Dim("complete a command, a file path, or a model after /model")),
		"  "+fmt.Sprintf("%-12s %s", "Up/Down", style.Dim("history")),
		"",
		style.Bold("Steering"),
		"  "+style.Dim("Type while the model is working. The status line becomes a prompt;"),
		"  "+style.Dim("Enter queues what you wrote and it is folded in at the next turn,"),
		"  "+style.Dim("without cancelling. Clear the line and the status line comes back."),
		"  "+style.Dim("Ctrl-C clears a typed line; on an empty one it cancels the turn."),
		"")
	a.out(lines...)
	return nil
}

func (a *App) cmdTools(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())
	lines := []string{""}
	for _, d := range a.exec.Definitions() {
		kind := "writes"
		if d.ReadOnly {
			kind = "read-only"
		}
		summary := strings.SplitN(strings.TrimSpace(d.Description), "\n", 2)[0]
		lines = append(lines, fmt.Sprintf("  %-8s %-11s %s", d.Name, style.Dim(kind), style.Dim(summary)))
	}
	lines = append(lines, "",
		style.Dim("Every tool is available in every mode. Modes change how the model works, not what it can do."),
		"")
	a.out(lines...)
	return nil
}

// cmdTokens reports how full the context is and when something happens about
// it. Those are the only two numbers here, because the status line is absent
// between turns and the prompt marker is silent below the warning threshold.
func (a *App) cmdTokens(ctx context.Context, args string) error {
	cs := a.contextState()
	style := render.NewStyle(a.screen.Color())

	limit, source := contextLimitFor(a.model, a.cfg.Agent.ContextOverride)
	usable := a.agent.Usable()

	head := fmt.Sprintf("  %s for the next request", compactInt(cs.Projected))
	if limit > 0 {
		head = fmt.Sprintf("  %s of %s for the next request (%d%%)",
			compactInt(cs.Projected), compactInt(limit), cs.Percent())
	}
	if usable > 0 {
		what := "summarises"
		if a.cfg.Agent.AutoCompact != nil && !*a.cfg.Agent.AutoCompact {
			what = "stops"
		}
		head += style.Dim(fmt.Sprintf(" · %s at %s", what, compactInt(usable)))
	}

	// The second line says where the window came from, and is omitted when
	// there is nothing notable about that.
	var notes []string
	if limit <= 0 {
		notes = append(notes, "no window reported, so the server decides")
	} else if s := slotSummary(a.model); s != "" {
		notes = append(notes, s)
	} else if source != "" {
		notes = append(notes, source)
	}
	if !cs.Anchored {
		notes = append(notes, "estimated -- the server has not reported a size for this request yet")
	}
	if summary, _ := a.agent.Summary(); strings.TrimSpace(summary) != "" {
		notes = append(notes, "summary saved")
	}

	lines := []string{"", head}
	if len(notes) > 0 {
		lines = append(lines, "  "+style.Dim(strings.Join(notes, " · ")))
	}
	a.out(append(lines, "")...)
	return nil
}

func (a *App) cmdMode(ctx context.Context, args string) error {
	style := render.NewStyle(a.screen.Color())
	if args == "" {
		lines := []string{"", fmt.Sprintf("  Current mode: %s", style.Bold(a.modeName)), ""}
		for _, n := range a.cfg.ModeNames() {
			marker := "  "
			if n == a.modeName {
				marker = "* "
			}
			lines = append(lines, fmt.Sprintf("  %s%-10s %s", marker, n, style.Dim(a.cfg.Mode[n].Description)))
		}
		a.out(append(lines, "")...)
		return nil
	}
	mode, ok := a.cfg.Mode[args]
	if !ok {
		return fmt.Errorf("no mode named %q; available: %s", args, strings.Join(a.cfg.ModeNames(), ", "))
	}
	a.modeName = args
	a.rebuildSystemPrompt(mode.Prompt)
	a.out("", fmt.Sprintf("Mode is now %s. %s", style.Bold(args), style.Dim(mode.Description)), "")
	return nil
}

func (a *App) rebuildSystemPrompt(modePrompt string) {
	var names []string
	for _, d := range a.exec.Definitions() {
		names = append(names, d.Name)
	}
	a.agent.SetSystem(prompt.Build(prompt.Options{
		Env:        a.env,
		ModePrompt: modePrompt,
		Agents:     a.projectInstructions(),
		ToolNames:  prompt.SortedToolNames(names),
	}))
}

// projectInstructions finds the AGENTS.md files for this session and names
// them the way the model can use.
//
// Discovery walks this machine's filesystem, because that is where the files
// are, and it is bounded by this machine's git root -- not by env.GitRoot,
// which inside a container holds the mounted path and would send the walk up
// past the project into the user's home directory.
//
// The paths then have to be translated, because the model is told container
// paths everywhere else and acts through tools that run there. Handing it
// "/home/someone/proj/AGENTS.md" and a tool that can only see
// "/workspace/AGENTS.md" is how a request to edit the file turns into a
// file-not-found the model cannot explain.
func (a *App) projectInstructions() []prompt.AgentsFile {
	files := prompt.DiscoverAgents(a.cwd, a.hostGitRoot)
	if a.showPath == nil {
		return files
	}
	for i := range files {
		if p, ok := a.showPath(files[i].Path); ok {
			files[i].Path = p
		} else {
			files[i].Path = ""
		}
	}
	return files
}

func (a *App) cmdClear(ctx context.Context, args string) error {
	// Clears the visible screen only. The scrollback is deliberately left
	// intact: it is the session's history and the reason ai-code does not use the
	// alternate screen.
	if a.screen.IsTTY() {
		fmt.Print("\x1b[2J\x1b[H")
	}
	return nil
}

func (a *App) cmdQuit(ctx context.Context, args string) error { return errQuit }

// cmdRestart replaces the running process with a fresh copy of the binary and
// resumes this session in it.
//
// It exists for one workflow: rebuild ai-code, then pick up the conversation on
// the new binary instead of abandoning it. The exec happens in main, not here,
// because exec replaces the process image and a defer that has not run by then
// never runs at all -- including the one that flushes this session to disk.
func (a *App) cmdRestart(ctx context.Context, args string) error {
	if a.sess == nil {
		return errors.New("this session is not being recorded, so there would be nothing to resume")
	}

	// Resolving the name rather than asking for this process's own image is the
	// whole point: after a rebuild the name refers to the new file, while
	// /proc/self/exe still refers to the inode this process started from.
	binary, err := exec.LookPath(os.Args[0])
	if err != nil {
		return fmt.Errorf("could not find the ai-code binary to restart: %w", err)
	}

	a.recordNew()
	style := render.NewStyle(a.screen.Color())
	a.out("", style.Dim(fmt.Sprintf("Restarting %s, resuming session %s", binary, a.sess.Meta.ID)))

	return &restartRequest{binary: binary, argv: a.restartArgv(binary)}
}

// restartArgv describes the session as it stands now rather than as it was
// launched, so a provider, model or mode changed part-way through survives the
// restart instead of reverting to whatever was on the original command line.
func (a *App) restartArgv(binary string) []string {
	argv := []string{binary,
		"--resume", a.sess.Meta.ID,
		"--provider", a.providerName,
		"--model", a.model.ID,
		"--mode", a.modeName,
	}
	if a.rt != nil {
		argv = append(argv, "--runtime", a.rt.FlagValue())
	}
	if e := a.agent.Effort(); e != provider.EffortUnset {
		argv = append(argv, "--think", string(e))
	}
	if a.flags != nil {
		if a.flags.verbose {
			argv = append(argv, "--verbose")
		}
		if a.flags.noStatus {
			argv = append(argv, "--no-status")
		}
	}
	return argv
}

func isRestart(err error) bool {
	var r *restartRequest
	return errors.As(err, &r)
}

func (a *App) cmdSessionList(ctx context.Context, args string) error {
	list, err := session.List(a.projectRoot)
	if err != nil {
		return err
	}
	style := render.NewStyle(a.screen.Color())
	lines := []string{""}
	for i, s := range list {
		if i >= 15 {
			lines = append(lines, style.Dim(fmt.Sprintf("  ... and %d more", len(list)-15)))
			break
		}
		marker := "  "
		if a.sess != nil && s.ID == a.sess.Meta.ID {
			marker = "* "
		}
		lines = append(lines, fmt.Sprintf("  %s%s  %s  %s",
			marker, s.ID, style.Dim(fmt.Sprintf("%d msgs", s.Messages)), style.Dim(s.Preview)))
	}
	lines = append(lines, "", style.Dim("Resume with: ai-code --resume <id>"), "")
	a.out(lines...)
	return nil
}

func (a *App) cmdEditor(ctx context.Context, args string) error {
	e := ui.NewEditor(os.Stdin, os.Stdout)
	e.EditorCommand = a.cfg.Editor
	text, err := e.LaunchEditor(args)
	if err != nil {
		return err
	}
	if strings.TrimSpace(text) == "" {
		return nil
	}
	a.echoPrompt(text)
	return a.runTurn(ctx, text)
}

func (a *App) cmdVerbose(ctx context.Context, args string) error { return a.setVerbose(true) }
func (a *App) cmdQuiet(ctx context.Context, args string) error   { return a.setVerbose(false) }

func (a *App) setVerbose(on bool) error {
	if a.verbosity == nil {
		return errors.New("the JSON stream already carries every event in full; there is nothing to expand")
	}
	was := a.verbosity.SetVerbose(on)

	// Written back to the flags because /restart rebuilds the process from
	// them: without this the restart silently undoes the toggle.
	if a.flags != nil {
		a.flags.verbose = on
	}

	what := "Verbose: full thinking, tool arguments and tool output."
	if !on {
		what = "Quiet: thinking marquee and one-line tool summaries."
	}
	if was == on {
		what = "Already " + strings.ToLower(what)
	}
	a.out("", what, "")
	return nil
}

func (a *App) cmdNew(ctx context.Context, args string) error {
	a.closeSession()
	s, err := session.Create(session.Meta{
		Project: a.projectRoot, Provider: a.providerName,
		Model: a.model.ID, Mode: a.modeName, Version: version,
	})
	if err != nil {
		return err
	}
	a.sess = s
	a.recorded = 0
	a.agent.SetMessages(nil)
	a.out("", "Started a new session. The previous one is on disk and can be resumed.", "")
	return nil
}

// queuePrompt holds a line submitted while a model swap is in progress.
//
// It is a prompt, not a correction to a running turn: nothing is running.
// Steering's usual action would queue it against a turn that does not
// exist, where it would sit unanswered until something else started one.
// Running it after the swap, on the model being loaded, is what typing
// during a swap meant.
func (a *App) queuePrompt(line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	a.cmdMu.Lock()
	a.deferredPrompts = append(a.deferredPrompts, line)
	n := len(a.deferredPrompts)
	a.cmdMu.Unlock()
	if a.interactive != nil {
		a.interactive.SetQueued(n)
	}
}

// takeDeferredPrompt removes the next line submitted during a swap.
func (a *App) takeDeferredPrompt() (string, bool) {
	a.cmdMu.Lock()
	defer a.cmdMu.Unlock()
	if len(a.deferredPrompts) == 0 {
		return "", false
	}
	line := a.deferredPrompts[0]
	a.deferredPrompts = a.deferredPrompts[1:]
	return line, true
}

// renderOnlyCommand reports whether a command can run in the middle of a
// turn because it changes nothing the turn can see.
func renderOnlyCommand(line string) bool {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return false
	}
	switch strings.TrimPrefix(fields[0], "/") {
	case "verbose", "quiet":
		return true
	}
	return false
}

// queueCommand holds a slash command typed during a turn.
func (a *App) queueCommand(line string) {
	a.cmdMu.Lock()
	a.queuedCmds = append(a.queuedCmds, strings.TrimSpace(line))
	a.cmdMu.Unlock()
}

func (a *App) pendingCommands() int {
	a.cmdMu.Lock()
	defer a.cmdMu.Unlock()
	return len(a.queuedCmds)
}

// runQueuedCommands executes what was typed during the turn, at the boundary
// where the conversation is complete. It reports whether the run should end:
// a command that changes the model, or quits, cannot sensibly be followed by
// another iteration of the run it interrupted.
func (a *App) runQueuedCommands(ctx context.Context) bool {
	a.cmdMu.Lock()
	queued := a.queuedCmds
	a.queuedCmds = nil
	a.cmdMu.Unlock()

	stop := false
	for _, line := range queued {
		a.echoPrompt(line)
		err := a.command(ctx, line)
		switch {
		case err == nil:
		case errors.Is(err, errQuit), isRestart(err):
			// Neither can happen inside a turn, so the run ends and the error
			// is handed back to the loop that knows what to do with it.
			a.turnErr = err
			return true
		default:
			a.reportError(err)
		}
		if commandEndsTheRun(line) {
			stop = true
		}
	}
	return stop
}

// commandEndsTheRun reports whether continuing the run after this command
// would be answering the old question with the new setup.
func commandEndsTheRun(line string) bool {
	name, _, _ := strings.Cut(strings.TrimPrefix(line, "/"), " ")
	matches := matchCommands(strings.ToLower(name))
	if len(matches) != 1 {
		return false
	}
	switch matches[0].name {
	case "model", "provider", "compact", "new", "mode":
		return true
	}
	return false
}
