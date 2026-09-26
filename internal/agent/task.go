package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// WorkerSystemPrompt is what a worker is told it is: a job with no one to ask,
// reporting findings rather than conclusions.
const WorkerSystemPrompt = `You are a worker agent carrying out one job for another agent.

Your brief is everything you will be given. Nobody can answer a question or
approve anything, so do not stop to ask: where the brief leaves a choice open,
take the most reasonable reading, do the work, and say in your reply what you
assumed. Stopping to ask ends the job with nothing done.

You have the full tool set and the job may well require changing things. Do
what the brief asks. Stay inside it: other agents are working in this same
tree at the same time, so touching files your brief did not name is how two
of you collide.

Then report, briefly and specifically:
- what you did, and what you changed, with exact paths
- exact file paths and line numbers for anything you found
- the relevant code or output, quoted rather than paraphrased
- what you looked for and did not find, which is often the answer
- anything you assumed, and anything you could not finish

Do not restate the brief, do not narrate your search, and do not offer to do
more. Your reply goes back to the agent that sent you as your whole report,
and you will not be asked a follow-up.`

const taskToolName = "task"

var taskSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "Three to six words naming what this worker is finding out, for the user to read while it runs."
    },
    "prompt": {
      "type": "string",
      "description": "The complete instruction. The worker starts with no conversation and cannot ask you anything, so name the files, symbols or behaviour it should look at and say what a useful answer contains."
    },
    "thinking": {
      "type": "string",
      "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
      "description": "How hard the worker should think. Omit to match this session. Lower it to none or minimal for a question that is really a search -- finding callers, checking whether a pattern holds -- where thinking buys nothing and costs wall-clock time. Raise it for a question that needs the answer worked out rather than located, such as why two pieces of code disagree."
    }
  },
  "required": ["description", "prompt"]
}`)

const taskDescription = `Hand a self-contained job to another agent, which runs in the background.

This call returns immediately, before the worker has done anything. The worker
then runs beside you on its own context, its own shell and its own working
directory, with the same tools you have -- it can read, search, run commands
and write files. When it finishes, its report arrives as a message in this
conversation.

Use it for work that is worth doing while you do something else: tracing how a
symbol is used across a tree, running a long build or test suite, carrying out
a change you have already specified well enough to describe in full. The
searching and the output stay in the worker; only the report reaches you.

Because it runs in the background:
- Do not wait for it, and do not ask whether it is done. Nothing you can say
  will make it finish sooner, and the answer reaches you either way.
- Start the ones you want, then carry on with work that does not depend on
  them. Ending your turn is correct when nothing else is outstanding -- the
  report will arrive.
- Several at once is normal and is the point. Every worker you start begins
  straight away; the backend decides in what order they generate.

It runs on this same model and machine and cannot ask you anything, so the
prompt must be complete: name the files, symbols or behaviour, say what a
useful answer contains, and say what to change if it is changing something.

Two workers editing the same file will collide, so give each one work that
does not overlap the others or what you are doing yourself.

Do not use it for something you can finish in one or two tool calls.`

// TaskExecutor adds the task tool to an executor. It is a wrapper rather than
// an ordinary tool because a worker needs the agent, and tools live on the far
// side of the container/SSH boundary; the one that cannot cross stays here.
type TaskExecutor struct {
	inner tool.Executor

	mu     sync.Mutex
	parent *Agent
	pool   *Workers
}

func NewTaskExecutor(inner tool.Executor) *TaskExecutor {
	return &TaskExecutor{inner: inner}
}

// Attach supplies the agent whose children this will spawn, and the session
// context those children run under. The context is the session's, not a
// turn's: a worker outlives the tool call that started it.
func (t *TaskExecutor) Attach(ctx context.Context, parent *Agent) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.parent = parent
	t.pool = NewWorkers(ctx)
}

// Pool is the set of running workers; nil until Attach.
func (t *TaskExecutor) Pool() *Workers {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.pool
}

// Unwrapped is the executor without the task tool, which is what a child is given.
func (t *TaskExecutor) Unwrapped() tool.Executor { return t.inner }

// Fork forks what is underneath and does not re-wrap it, so a worker never gets the task tool.
func (t *TaskExecutor) Fork() (tool.Executor, error) { return t.inner.Fork() }

func (t *TaskExecutor) agent() *Agent {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.parent
}

// available reports whether the task tool may be offered at all. Never on a
// cloud provider: a worker's prompt and codebase must not reach a third party
// on the model's initiative. Absent from the schema rather than refused.
func (t *TaskExecutor) available() bool {
	a := t.agent()
	return a != nil && a.client != nil && a.client.Class() != provider.ClassCloud
}

func (t *TaskExecutor) Definitions() []tool.Definition {
	defs := t.inner.Definitions()
	if !t.available() {
		return defs
	}
	return append(defs, tool.Definition{
		Name:        taskToolName,
		Description: taskDescription,
		Schema:      taskSchema,
		ReadOnly:    true,
	})
}

func (t *TaskExecutor) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	if req.Name != taskToolName {
		return t.inner.Execute(ctx, req)
	}
	if !t.available() {
		return tool.Errorf("%s is not available in this session.", taskToolName), nil
	}
	return t.runWorker(ctx, req), nil
}

// SetProgress passes the progress callback through to the executor being wrapped.
func (t *TaskExecutor) SetProgress(p tool.Progress) {
	if pr, ok := t.inner.(tool.ProgressReporter); ok {
		pr.SetProgress(p)
	}
}

func (t *TaskExecutor) IsReadOnly(name string) bool {
	if name == taskToolName {
		return true
	}
	if ro, ok := t.inner.(interface{ IsReadOnly(string) bool }); ok {
		return ro.IsReadOnly(name)
	}
	return false
}

func (t *TaskExecutor) runWorker(ctx context.Context, req tool.Request) tool.Result {
	var args struct {
		Description string `json:"description"`
		Prompt      string `json:"prompt"`
		Thinking    string `json:"thinking"`
	}
	if err := json.Unmarshal(nonEmptyJSON(req.Args), &args); err != nil {
		return tool.Errorf("could not parse the arguments as JSON: %v", err)
	}
	if strings.TrimSpace(args.Prompt) == "" {
		return tool.Errorf("prompt is required: say what the worker should find out, " +
			"in full, because it starts with no conversation.")
	}
	label := strings.TrimSpace(args.Description)
	if label == "" {
		label = "worker"
	}

	parent := t.agent()

	// Unset inherits the session's level: a worker thinks as hard as the
	// conversation that sent it.
	effort := parent.Effort()
	if strings.TrimSpace(args.Thinking) != "" {
		e, err := provider.ParseEffort(args.Thinking)
		if err != nil {
			return tool.Errorf("%v", err)
		}
		effort = e
	}

	pool := t.Pool()
	if pool == nil {
		return tool.Errorf("%s is not available in this session.", taskToolName)
	}

	prompt := args.Prompt
	w := pool.Start(label, func(wctx context.Context, w *Worker) Report {
		rep := Report{}
		defer func() { parent.Steer(frameReport(w, rep)) }()

		child, err := parent.Spawn(Child{
			System: WorkerSystemPrompt,
			Effort: effort,
			Sink:   w.Sink(),
		})
		if err != nil {
			rep.Why = fmt.Sprintf("could not be given its own shell: %v", err)
			return rep
		}
		// The shell was forked for this worker and nothing else holds it.
		defer child.Close()

		if err := child.Run(wctx, prompt); err != nil {
			rep.Turns = child.Turns()
			if wctx.Err() != nil {
				rep.Why = "interrupted before it reported back"
			} else {
				rep.Why = fmt.Sprintf("failed: %v", err)
			}
			return rep
		}
		rep.Turns = child.Turns()
		if rep.Text = strings.TrimSpace(child.LastAssistantText()); rep.Text == "" {
			rep.Why = "returned nothing"
		}
		return rep
	})

	return tool.Result{
		Content: fmt.Sprintf("Worker %d started: %q.\n\n"+
			"It is running beside this session on its own context and will report back "+
			"as a message when it finishes. Do not wait for it and do not ask about it: "+
			"carry on with work that does not depend on its answer, or end your turn. "+
			"If nothing else is outstanding, ending the turn is correct -- its findings "+
			"will reach you.", w.ID, label),
		Display: fmt.Sprintf("%s (started)", label),
	}
}

// frameReport is how a worker's answer re-enters the parent conversation,
// tagged so the model does not read it as a typed message.
func frameReport(w *Worker, rep Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<worker-report id=%q label=%q turns=%q", fmt.Sprint(w.ID), w.Label,
		fmt.Sprint(rep.Turns))
	if rep.Why != "" {
		fmt.Fprintf(&b, " outcome=%q", rep.Why)
	}
	b.WriteString(">\n")
	if rep.Text != "" {
		b.WriteString(rep.Text)
	} else {
		fmt.Fprintf(&b, "This worker produced no findings (%s). Decide whether to look "+
			"into it directly; do not simply start another worker on the same question.",
			rep.Why)
	}
	b.WriteString("\n</worker-report>")
	return b.String()
}

func nonEmptyJSON(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	return raw
}

var (
	_ tool.Executor         = (*TaskExecutor)(nil)
	_ tool.ProgressReporter = (*TaskExecutor)(nil)
)
