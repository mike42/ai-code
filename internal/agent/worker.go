package agent

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Worker is one investigation running beside the session.
//
// It outlives the turn that started it. The turn's context governs the tool
// call that asked for a worker, and that call returns the moment the worker is
// registered; binding the worker to it would kill the worker as soon as the
// turn ended, which is the thing this exists to avoid. The context a worker
// runs under comes from the session instead.
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

// Sink is where the worker's own events go.
//
// It writes nothing. The renderer's transient zone has one owner, and a worker
// streaming its text into the parent's scrollback would interleave with the
// turn the user is reading -- three of them at once would be unreadable even
// if it were safe. What a worker produces reaches the screen twice: as a
// status line while it runs, and as its report when it finishes.
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
//
// The zero value is not usable: a worker's lifetime is the session's, so the
// pool has to be told what that is.
type Workers struct {
	base context.Context

	mu      sync.Mutex
	next    int
	running map[int]*Worker

	// OnReport receives each finished worker. It is called from the worker's
	// own goroutine, once, and never with the pool locked.
	OnReport func(Report)
	// OnChange is called whenever the set of running workers changes, so a
	// status line can be redrawn without polling.
	OnChange func()
}

func NewWorkers(base context.Context) *Workers {
	return &Workers{base: base, running: map[int]*Worker{}}
}

// Start registers a worker and runs fn in its own goroutine.
//
// fn is handed the context the worker lives under and the worker itself, and
// returns what to report. Start does not wait for any of it.
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

// Busy is how many workers are running, for a caller that only needs to know
// whether any are.
func (p *Workers) Busy() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.running)
}

// Wait blocks until every worker running now has finished.
//
// It does not stop anything starting afterwards, because nothing here can
// prevent that: the model may be mid-turn and about to ask for another.
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

// Cancel stops every running worker and waits for each to report.
//
// Waiting matters: a cancelled worker still delivers a report saying it was
// cancelled, and a caller that is tearing the session down needs that to have
// happened before it stops listening.
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
