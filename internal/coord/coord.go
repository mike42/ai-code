// Package coord lets the ai-code instances belonging to one person on one
// machine see each other. It assumes a single user and desktop session, where
// a shared directory of files under flock is enough; there is no second party.
package coord

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// errNoLiveness is why coordination is off on a platform with no flock.
var errNoLiveness = errors.New(
	"this platform cannot prove another instance is still alive, so instances are not coordinated")

// Peer is one running instance, as it last described itself.
type Peer struct {
	PID      int       `json:"pid"`
	Session  string    `json:"session"`
	Provider string    `json:"provider"`
	Model    string    `json:"model"`
	Cwd      string    `json:"cwd"`
	Busy     bool      `json:"busy"`
	Since    time.Time `json:"since"`

	// State is what this instance is doing about the model, in the words a
	// person reads. See State.
	State State `json:"state"`
	// NoSummary is set when this session needed a summary before losing the
	// model and could not write one. The swap proceeds anyway; the person
	// swapping is told.
	NoSummary bool `json:"no_summary"`
}

// State is what a session is doing about the model it holds. Each value says
// what is being waited for.
type State string

const (
	// StateFree is no claim on the model. Never shown: a session with no
	// claim should not be noise during someone else's swap.
	StateFree State = "free"
	// StateWorking is a turn in flight.
	StateWorking State = "working"
	// StateSummarising holds a slot and is generating.
	StateSummarising State = "summarising"
	// StateReady means this session has done whatever it needed to before
	// losing the model; usually that is nothing.
	StateReady State = "ready"
)

// Describe renders a state for someone reading a swap in progress.
func (s State) Describe() string {
	switch s {
	case StateWorking:
		return "finishing its turn"
	case StateSummarising:
		return "writing a summary"
	case StateReady:
		return "ready"
	}
	return ""
}

// Describe names a peer the way a person would recognise it: by the directory
// it is working in, because that is what distinguishes one window from
// another.
func (p Peer) Describe() string {
	dir := p.Cwd
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if rel, err := filepath.Rel(home, dir); err == nil && !strings.HasPrefix(rel, "..") {
			dir = "~/" + rel
		}
	}
	if dir == "" {
		dir = fmt.Sprintf("pid %d", p.PID)
	}
	return dir
}

// usable reports whether the shared directory can be worked with. An empty
// path would make filepath.Join produce relative paths, scattering files
// through the project.
func usable(dir string) bool { return canDetectLiveness && dir != "" }

// DefaultDir is the directory the windows share: XDG_RUNTIME_DIR, per-user and
// cleared on logout, with a temp-directory fallback for systems without it.
func DefaultDir() string {
	if d := os.Getenv("XDG_RUNTIME_DIR"); d != "" {
		return filepath.Join(d, "ai-code", "peers")
	}
	return filepath.Join(os.TempDir(), fmt.Sprintf("ai-code-%d", os.Getuid()), "peers")
}

// Registration is this instance's entry. Every method is safe to call on a
// nil receiver, so a caller that could not register does not have to check.
type Registration struct {
	mu   sync.Mutex
	dir  string
	lock *os.File
	self Peer
}

// Register announces this instance and holds the lock that proves it alive.
// Failure is never worth stopping for: coordination is an improvement on a
// session that otherwise works.
func Register(dir string, self Peer) (*Registration, error) {
	if !usable(dir) {
		return nil, errNoLiveness
	}

	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	self.PID = os.Getpid()
	self.Since = time.Now()
	if self.State == "" {
		self.State = StateFree
	}

	lockPath := filepath.Join(dir, fmt.Sprintf("%d.lock", self.PID))
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	held, err := lockExclusive(f)
	if err != nil || !held {
		f.Close()
		if err != nil {
			return nil, err
		}
		// A live process owns this pid's entry, so the pid was reused; its lock is authoritative.
		return nil, fmt.Errorf("another process holds the registration for pid %d", self.PID)
	}

	r := &Registration{dir: dir, lock: f, self: self}
	if err := r.write(); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

// Update changes what this instance publishes about itself.
func (r *Registration) Update(fn func(*Peer)) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	fn(&r.self)
	_ = r.write()
}

// SetBusy publishes whether a turn is running. It is the field a swap
// actually consults.
func (r *Registration) SetBusy(busy bool) {
	r.Update(func(p *Peer) { p.Busy = busy })
}

// SetModel publishes a model change.
func (r *Registration) SetModel(provider, model string) {
	r.Update(func(p *Peer) { p.Provider, p.Model = provider, model })
}

// write persists the entry. Caller holds the mutex.
func (r *Registration) write() error {
	b, err := json.Marshal(r.self)
	if err != nil {
		return err
	}
	path := filepath.Join(r.dir, fmt.Sprintf("%d.json", r.self.PID))
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Close removes this instance's entry. The lock would be dropped by the
// kernel anyway; this only saves the next reader a stale file to clean up.
func (r *Registration) Close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	pid := r.self.PID
	os.Remove(filepath.Join(r.dir, fmt.Sprintf("%d.json", pid)))
	if r.lock != nil {
		r.lock.Close()
		os.Remove(filepath.Join(r.dir, fmt.Sprintf("%d.lock", pid)))
	}
}

// Self is this instance's entry as last published.
func (r *Registration) Self() Peer {
	if r == nil {
		return Peer{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.self
}

// Peers lists the other live instances, and deletes the entries of any that
// died without cleaning up. Excluding self is not optional: a session that
// waits on its own pid presents as a hang rather than an error.
func Peers(dir string) []Peer {
	if !usable(dir) {
		return nil
	}

	selfPID := os.Getpid()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}

	var out []Peer
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		var p Peer
		if json.Unmarshal(b, &p) != nil || p.PID == 0 || p.PID == selfPID {
			continue
		}
		if !alive(dir, p.PID) {
			os.Remove(filepath.Join(dir, name))
			os.Remove(filepath.Join(dir, fmt.Sprintf("%d.lock", p.PID)))
			continue
		}
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Since.Before(out[j].Since) })
	return out
}

func alive(dir string, pid int) bool {
	f, err := os.Open(filepath.Join(dir, fmt.Sprintf("%d.lock", pid)))
	if err != nil {
		// No lock file: an entry written without one, or one already cleaned up; nobody holds it.
		return false
	}
	defer f.Close()
	return isHeld(f)
}

// SwapNotice is one instance telling the others it is about to load a
// different model, so the sessions on the current one can park first. Finding
// out afterwards is too late: writing a summary needs the model.
type SwapNotice struct {
	// PID is who announced it, so nobody parks for their own swap.
	PID int `json:"pid"`
	// Model is the one being unloaded. Sessions on any other are unaffected.
	Model string `json:"model"`
	// Into and IntoWindow describe what is being loaded instead. The window
	// tells a peer whether its transcript will still fit, half of whether it
	// needs a summary at all.
	Into       string    `json:"into"`
	IntoWindow int       `json:"into_window"`
	Since      time.Time `json:"since"`
}

const swapFile = "swap.json"

// Announcement is a published swap. End it once the swap is done, or peers
// will keep parking for a model change that already happened.
type Announcement struct {
	dir string
}

// AnnounceSwap publishes an imminent model change.
func AnnounceSwap(dir, model, into string, intoWindow int) (*Announcement, error) {
	if !usable(dir) {
		return nil, errNoLiveness
	}
	n := SwapNotice{
		PID: os.Getpid(), Model: model, Into: into,
		IntoWindow: intoWindow, Since: time.Now(),
	}
	b, err := json.Marshal(n)
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, swapFile)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, path); err != nil {
		return nil, err
	}
	return &Announcement{dir: dir}, nil
}

// End withdraws the announcement.
func (a *Announcement) End() {
	if a == nil {
		return
	}
	os.Remove(filepath.Join(a.dir, swapFile))
}

// PendingSwap reads the current announcement, if there is one. One left behind
// by a process that died is cleared rather than honoured, or a crash mid-swap
// would leave every window parking forever.
func PendingSwap(dir string) (SwapNotice, bool) {
	selfPID := os.Getpid()
	if !usable(dir) {
		return SwapNotice{}, false
	}
	b, err := os.ReadFile(filepath.Join(dir, swapFile))
	if err != nil {
		return SwapNotice{}, false
	}
	var n SwapNotice
	if json.Unmarshal(b, &n) != nil || n.Model == "" || n.PID == selfPID {
		return SwapNotice{}, false
	}
	if !alive(dir, n.PID) {
		os.Remove(filepath.Join(dir, swapFile))
		return SwapNotice{}, false
	}
	return n, true
}

// SetState publishes what this session is doing about its model.
func (r *Registration) SetState(s State) {
	r.Update(func(p *Peer) { p.State = s })
}

// SetNoSummary records that a summary was needed and could not be written.
func (r *Registration) SetNoSummary() {
	r.Update(func(p *Peer) { p.NoSummary, p.State = true, StateReady })
}

// NotReady lists the live sessions on a model that still have something to do
// before it goes; an empty result means the swap is safe. A session not
// actively holding the model counts as ready.
func NotReady(dir, model string) []Peer {
	var out []Peer
	for _, p := range Peers(dir) {
		if p.Model != model {
			continue
		}
		switch p.State {
		case StateWorking, StateSummarising:
			out = append(out, p)
		}
	}
	return out
}

// OnModel lists every live session using a model.
func OnModel(dir, model string) []Peer {
	var out []Peer
	for _, p := range Peers(dir) {
		if p.Model == model {
			out = append(out, p)
		}
	}
	return out
}

const decisionFile = "model.lock"

// Decide runs fn while holding the machine-wide lock on model decisions. Both
// a swap and a session about to send decide here, so whoever takes the lock
// second sees what the first published.
func Decide(dir string, fn func()) error {
	if !usable(dir) {
		// Without flock there is no mutual exclusion; the work still happens, unserialised.
		fn()
		return nil
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		fn()
		return err
	}
	f, err := os.OpenFile(filepath.Join(dir, decisionFile), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		fn()
		return err
	}
	defer f.Close()

	if err := lockBlocking(f); err != nil {
		fn()
		return err
	}
	defer unlock(f)

	fn()
	return nil
}

// Intent is the model this machine last asked for, written by whichever window
// asked. The server remains the authority on what is loaded; this records what
// was wanted, useful while a load is still in progress.
type Intent struct {
	Model string    `json:"model"`
	PID   int       `json:"pid"`
	At    time.Time `json:"at"`
}

const intentFile = "intent.json"

// SetIntent records the model asked for. Call it with the decision lock held
// and before the load, so every window agrees on the target while it is on its
// way.
func SetIntent(dir, model string) error {
	if !usable(dir) || model == "" {
		return nil
	}
	b, err := json.Marshal(Intent{Model: model, PID: os.Getpid(), At: time.Now()})
	if err != nil {
		return err
	}
	path := filepath.Join(dir, intentFile)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// CurrentIntent reads that record. Absent means nothing has been asked for
// through ai-code, which says nothing about what the server has loaded.
func CurrentIntent(dir string) (Intent, bool) {
	if !usable(dir) {
		return Intent{}, false
	}
	b, err := os.ReadFile(filepath.Join(dir, intentFile))
	if err != nil {
		return Intent{}, false
	}
	var i Intent
	if json.Unmarshal(b, &i) != nil || i.Model == "" {
		return Intent{}, false
	}
	return i, true
}
