package agent

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Worker is one investigation running beside the session. It outlives the turn
// that started it: the session's context governs it, not the turn's.
type Worker struct {
	ID    int
	Label string
	Began time.Time

	cancel context.CancelFunc
	done   chan struct{}

	mu    sync.Mutex
	turns int
}

// WorkerStatus is a snapshot for the status line.
type WorkerStatus struct {
	ID      int
	Label   string
	Turns   int
	Elapsed time.Duration
}

func (w *Worker) Status() WorkerStatus {
	w.mu.Lock()
	defer w.mu.Unlock()
	return WorkerStatus{
		ID:      w.ID,
		Label:   w.Label,
		Turns:   w.turns,
		Elapsed: time.Since(w.Began).Round(time.Second),
	}
}

// Turns is how many times round the loop it has been.
func (w *Worker) Turns() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.turns
}

// Sink is where the worker's own events go. It writes nothing: the renderer's
// transient zone has one owner, and a worker's output reaches the screen as a
// status line and as its report.
func (w *Worker) Sink() Sink {
	return SinkFunc(func(e Event) {
		if e.Kind != EvTurnStart {
			return
		}
		w.mu.Lock()
		w.turns = e.Turn
		w.mu.Unlock()
	})
}

// Report is a finished worker's answer.
type Report struct {
	ID      int
	Label   string
	Turns   int
	Elapsed time.Duration
	// Text is the worker's report, empty when it did not get that far.
	Text string
	// Why is set when the worker did not report: cancelled, or failed.
	Why string
}

// Workers is every background worker belonging to one session.
type Workers struct {
	base context.Context

	mu      sync.Mutex
	next    int
	running map[int]*Worker

	// OnReport receives each finished worker, once, without the pool locked.
	OnReport func(Report)
	// OnChange reports whenever the set of running workers changes.
	OnChange func()
}

func NewWorkers(base context.Context) *Workers {
	return &Workers{base: base, running: map[int]*Worker{}}
}

// Start registers a worker and runs fn in its own goroutine, without waiting.
func (p *Workers) Start(label string, fn func(context.Context, *Worker) Report) *Worker {
	ctx, cancel := context.WithCancel(p.base)

	p.mu.Lock()
	p.next++
	w := &Worker{
		ID:     p.next,
		Label:  label,
		Began:  time.Now(),
		cancel: cancel,
		done:   make(chan struct{}),
	}
	p.running[w.ID] = w
	p.mu.Unlock()
	p.changed()

	go func() {
		defer close(w.done)
		rep := fn(ctx, w)

		p.mu.Lock()
		delete(p.running, w.ID)
		p.mu.Unlock()
		p.changed()

		rep.ID, rep.Label = w.ID, w.Label
		rep.Elapsed = time.Since(w.Began).Round(time.Second)
		if p.OnReport != nil {
			p.OnReport(rep)
		}
		cancel()
	}()

	return w
}

// Active is a snapshot of what is running, oldest first.
func (p *Workers) Active() []WorkerStatus {
	p.mu.Lock()
	ws := make([]*Worker, 0, len(p.running))
	for _, w := range p.running {
		ws = append(ws, w)
	}
	p.mu.Unlock()

	out := make([]WorkerStatus, 0, len(ws))
	for _, w := range ws {
		out = append(out, w.Status())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Busy is how many workers are running.
func (p *Workers) Busy() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.running)
}

// Wait blocks until every worker running now has finished; later ones are not stopped.
func (p *Workers) Wait() {
	p.mu.Lock()
	ws := make([]*Worker, 0, len(p.running))
	for _, w := range p.running {
		ws = append(ws, w)
	}
	p.mu.Unlock()

	for _, w := range ws {
		<-w.done
	}
}

// Cancel stops every running worker and waits for each to report, so a
// cancelled worker has delivered its report before the caller stops listening.
func (p *Workers) Cancel() {
	p.mu.Lock()
	ws := make([]*Worker, 0, len(p.running))
	for _, w := range p.running {
		ws = append(ws, w)
	}
	p.mu.Unlock()

	for _, w := range ws {
		w.cancel()
	}
	for _, w := range ws {
		<-w.done
	}
}

func (p *Workers) changed() {
	if p.OnChange != nil {
		p.OnChange()
	}
}
