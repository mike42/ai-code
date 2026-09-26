package main

import (
	"context"
	"fmt"
	"strings"

	"ai-code/internal/agent"
	"ai-code/internal/render"
	"ai-code/internal/ui"
)

// wireWorkers connects the worker pool to the terminal. Reports reach the
// conversation through the steer queue whether or not anything is watching;
// this only wires the display.
func (a *App) wireWorkers() {
	if a.workers == nil {
		return
	}
	a.workers.OnReport = a.showWorkerReport
	a.workers.OnChange = a.refreshWorkerLine
}

// setEditor records the live prompt, or clears it.
func (a *App) setEditor(e *ui.Editor) {
	a.editorMu.Lock()
	a.editor = e
	a.editorMu.Unlock()
}

func (a *App) liveEditor() *ui.Editor {
	a.editorMu.Lock()
	defer a.editorMu.Unlock()
	return a.editor
}

// showWorkerReport puts a finished worker into the scrollback. It writes only
// when the prompt is up: the transient zone has one owner, and mid-turn the
// report is announced by the steer event instead.
func (a *App) showWorkerReport(rep agent.Report) {
	ed := a.liveEditor()
	if ed == nil || a.isSteering() {
		return
	}
	style := render.NewStyle(a.screen.Color())

	head := fmt.Sprintf("%s  %s", style.Bold("worker done"), rep.Label)
	detail := fmt.Sprintf("%d turns · %s", rep.Turns, compactDuration(rep.Elapsed))
	if rep.Why != "" {
		head = fmt.Sprintf("%s  %s", style.Warn("worker "+rep.Why), rep.Label)
	}

	lines := []string{"", head + "  " + style.Dim(detail)}
	for _, l := range strings.Split(strings.TrimRight(rep.Text, "\n"), "\n") {
		if l == "" {
			lines = append(lines, "")
			continue
		}
		lines = append(lines, "  "+l)
	}
	lines = append(lines, "", style.Dim("Its findings go to the model with your next message."))
	ed.EmitAbove(a.promptString(), lines...)
}

// refreshWorkerLine redraws the prompt so its worker marker is current.
func (a *App) refreshWorkerLine() {
	if ed := a.liveEditor(); ed != nil {
		ed.Refresh(a.promptString())
	}
}

// workerMarker is what the prompt says about workers running behind it. It is
// on the prompt, not the header, which the model-change notice owns.
func (a *App) workerMarker() string {
	if a.workers == nil {
		return ""
	}
	active := a.workers.Active()
	if len(active) == 0 {
		return ""
	}
	return fmt.Sprintf("[%d working] ", len(active))
}

// drainWorkers finishes the background work a non-interactive run started.
// There is no prompt to carry on at, so returning early would cancel running
// workers and lose their work; the loop ends when the pool is idle.
func (a *App) drainWorkers(ctx context.Context) error {
	for a.workers != nil && a.workers.Busy() > 0 {
		a.workers.Wait()
		if a.agent.Pending() == 0 {
			return nil
		}
		if err := a.runTurn(ctx, ""); err != nil {
			return err
		}
	}
	return nil
}

// stopWorkers ends everything running in the background. Each cancelled worker
// still reports, so the model is not left waiting for an answer.
func (a *App) stopWorkers() {
	if a.workers == nil {
		return
	}
	n := a.workers.Busy()
	if n == 0 {
		return
	}
	noun := "worker"
	if n != 1 {
		noun = "workers"
	}
	a.out("", fmt.Sprintf("Stopping %d %s.", n, noun))
	a.workers.Cancel()
}

func (a *App) isSteering() bool {
	a.steerMu.Lock()
	defer a.steerMu.Unlock()
	return a.steering
}
