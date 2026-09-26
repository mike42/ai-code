// Package coord lets the ai-code instances belonging to one person on one
// machine see each other.
//
// Everything here assumes a single user and a single desktop session: several
// windows, one of them in front and the others running work that was started
// and walked away from. There is no second party, so there is nothing to
// authenticate, negotiate or arbitrate. That assumption is why a directory of
// files under flock is enough, where anything crossing a trust boundary would
// need a daemon and a protocol.
//
// Four small files in one shared directory, and that is the whole of it:
//
//   - <pid>.json   what one window is doing, and <pid>.lock, held open for
//     as long as that window lives, so a dead one is known
//     immediately rather than after a timeout
//   - intent.json  the model the user last asked for, in any window
//   - swap.json    a swap about to happen, while it is happening
//   - model.lock   held for the moment a window decides about the model,
//     so two windows cannot decide at the same time
//
// It stays silent. The files are read at exactly one moment -- a model swap
// that would take the model out from under a session using it -- and nothing
// here prints, polls on a timer, or adds a command. Another window is one
// alt-tab away, so a feature that merely listed them would be telling the
// user what their own screen already shows.
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
	// model and could not write one. The swap proceeds anyway -- one broken
	// window must not hold the machine -- but the person swapping is told.
	NoSummary bool `json:"no_summary"`
}

// State is what a session is doing about the model it holds.
//
// The words are the ones in local-coordination.md §8, which exists because an
// earlier draft invented PARK/COMPACT/DEGRADED and they read as precise
// without being so. Each says what is being waited for.
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
	// losing the model, and the swap may proceed. Usually that is nothing at
	// all: a session whose conversation still fits the incoming window just
	// carries on with the new model.
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

// usable reports whether the shared directory can be worked with at all.
//
// An empty path is the case to catch: filepath.Join("", "x") is the
// *relative* path "x", so every read and write would land in whatever
// directory the process happens to be in. A session that failed to work out
// where the shared directory is must take no part in coordination, not
// scatter files through the user's project.
func usable(dir string) bool { return canDetectLiveness && dir != "" }

// DefaultDir is the directory the windows share.
//
// XDG_RUNTIME_DIR is the right home for it: per-user, cleared on logout, and
// on a tmpfs, so a machine that lost power comes back with nothing in it
// rather than with stale files. The fallback is only for systems without it.
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
//
// A failure is returned but is never worth stopping for: coordination is an
// improvement on a session that otherwise works, and a read-only runtime
// directory should not stop someone using the harness.
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
		// A live process already owns this pid's entry, which on a sane system
		// means our own pid was reused after an unclean exit. Its lock is
		// authoritative; ours is not.
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
// died without cleaning up.
//
// "Other" is not optional and is not a parameter. Every caller wants it, and
// a caller that passed the wrong pid produced a session waiting for itself to
// park -- which presents as a hang rather than as an error.
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
		// No lock file: the entry was written by something that never held one,
		// or the lock was already cleaned up. Either way nobody is holding it.
		return false
	}
	defer f.Close()
	return isHeld(f)
}

// ---------------------------------------------------------------------------
// Announcing a swap
// ---------------------------------------------------------------------------

// SwapNotice is one instance telling the others it is about to load a
// different model, so the sessions on the current one can park first.
//
// The announcement is the whole point of coordinating. Finding out afterwards
// that the model has gone is too late by construction: writing a summary
// needs the model, so a session that learns about the swap after it happened
// has already lost the ability to save itself. Noticing a missing model is a
// safety net for changes made outside ai-code, never the path between
// instances.
type SwapNotice struct {
	// PID is who announced it, so nobody parks for their own swap.
	PID int `json:"pid"`
	// Model is the one being unloaded. Sessions on any other are unaffected.
	Model string `json:"model"`
	// Into and IntoWindow describe what is being loaded instead. The window
	// is what lets a peer decide whether its transcript will still fit, which
	// is half of whether it needs a summary at all.
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

// PendingSwap reads the current announcement, if there is one.
//
// One left behind by a process that died is cleared rather than honoured:
// otherwise a single crash mid-swap would leave every other window parking
// against a model change that will never happen.
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

// NotReady lists the live sessions on a model that still have something to
// do before it goes. An empty result means the swap is safe.
//
// Anything that is not actively holding the model counts as ready. Waiting
// is for sessions that said they are doing something; a session that has
// published no state at all -- one that has never run a turn, so has never
// had a state to publish -- has no claim, and treating its silence as a
// claim makes every swap around it wait for an answer that never comes.
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

// ---------------------------------------------------------------------------
// The decision lock
// ---------------------------------------------------------------------------

const decisionFile = "model.lock"

// Decide runs fn while holding the machine-wide lock on model decisions.
//
// It closes the one race the announcement alone cannot. A swap reads the list
// of sessions to wait for; a session about to send marks itself as working.
// Interleave those two and the swap loads while a request is on its way to
// the model it is unloading -- which the server answers by loading that model
// straight back. That is the churn, and no amount of checking health first
// removes it, because the check and the send are not one action.
//
// Both sides do their deciding here instead, so the interleaving cannot
// happen: whoever takes the lock second sees what the first one published.
// The section is a few file operations on a local filesystem with no network
// in it, so blocking is measured in microseconds.
//
// Every instance is the same user on the same machine, which is what makes a
// file lock sufficient; nothing here would survive a trust boundary.
func Decide(dir string, fn func()) error {
	if !usable(dir) {
		// Without flock there is no mutual exclusion to be had. The caller's
		// work still happens; it is simply not serialised against anyone.
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

// ---------------------------------------------------------------------------
// Which model the user wants
// ---------------------------------------------------------------------------

// Intent is the model this machine's user last asked for, written by
// whichever window they asked in.
//
// The server remains the authority on what is actually loaded; nothing here
// can know that, and a record claiming to would be wrong the moment someone
// loaded a model by hand. What this is instead is a more up-to-date record
// of what was *wanted*, and because every instance belongs to the same
// person on the same machine, they can act on it together.
//
// Which makes it more useful than the server's answer on the path that
// matters. A request issued while a model is still loading should name the
// model being loaded, not the one still resident: lemonade queues it behind
// the load and answers it from the new weights. Asking the server would give
// the old name, and sending that is what drags the old model back.
//
// It stands until the user asks for something else.
type Intent struct {
	Model string    `json:"model"`
	PID   int       `json:"pid"`
	At    time.Time `json:"at"`
}

const intentFile = "intent.json"

// SetIntent records the model the user has asked for. Call it with the
// decision lock held, and before the load rather than after: the point is
// that every window agrees on the target while it is still on its way.
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

// CurrentIntent reads that record. Absent means nobody has asked for
// anything through ai-code since these files were created, which says
// nothing about what the server has loaded.
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
