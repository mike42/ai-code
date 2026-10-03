// Package coord connects the ai-code instances of one user on one machine, so
// that instances sharing a model server that holds one model at a time take
// turns instead of evicting each other. Each instance listens on a unix socket
// in a private directory and holds a connection to every other one. Every
// change travels as a message, and a closed connection means that instance has
// gone; nothing is polled.
package coord

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// State is what an instance is doing with the model, as a person waiting on it
// would want it described.
type State string

const (
	StateIdle        State = "idle"
	StateWorking     State = "working"
	StateSummarising State = "summarising"
	StateParked      State = "parked"
)

func (s State) Describe() string {
	switch s {
	case StateWorking:
		return "finishing its turn"
	case StateSummarising:
		return "writing a summary"
	case StateParked:
		return "paused"
	case "":
		return "starting"
	}
	return ""
}

// Peer is one instance as it last described itself. Server is "" for an
// instance whose provider is not shared, such as a cloud one.
type Peer struct {
	PID    int    `json:"pid"`
	Cwd    string `json:"cwd"`
	Server string `json:"server"`
	Model  string `json:"model"`
	State  State  `json:"state"`
}

// Describe names a peer by its working directory, which is what tells one
// window from another.
func (p Peer) Describe() string {
	dir := p.Cwd
	if home, err := os.UserHomeDir(); err == nil && home != "" && dir != "" {
		if rel, err := filepath.Rel(home, dir); err == nil && !strings.HasPrefix(rel, "..") {
			dir = "~/" + rel
		}
	}
	switch {
	case dir != "":
		return dir
	case p.PID == 0:
		return "a window"
	}
	return fmt.Sprintf("pid %d", p.PID)
}

// SwapID orders swaps: the earlier request goes first, and every instance
// agrees which that is.
type SwapID struct {
	PID int   `json:"pid"`
	Seq int64 `json:"seq"`
}

func (s SwapID) precedes(o SwapID) bool {
	if s.Seq != o.Seq {
		return s.Seq < o.Seq
	}
	return s.PID < o.PID
}

// Swap is one instance asking every other on Server to stop sending requests
// for any model but Into, so that Into can be loaded.
type Swap struct {
	ID         SwapID `json:"id"`
	Server     string `json:"server"`
	Into       string `json:"into"`
	IntoWindow int    `json:"into_window"`
}

type loadedAt struct {
	Model string `json:"model"`
	Seq   int64  `json:"seq"`
	By    int    `json:"by"`
}

type message struct {
	Kind   string              `json:"kind"`
	Peer   *Peer               `json:"peer,omitempty"`
	Swap   *Swap               `json:"swap,omitempty"`
	Ready  []SwapID            `json:"ready,omitempty"`
	Loaded map[string]loadedAt `json:"loaded,omitempty"`
}

const (
	kindHello    = "hello"
	kindPeer     = "peer"
	kindSwap     = "swap"
	kindReady    = "ready"
	kindWithdraw = "withdraw"
	kindLoaded   = "loaded"
	kindRetire   = "retire"
)

// ErrPreempted is returned by Request when an earlier swap on the same server
// is pending; the request has been withdrawn.
var ErrPreempted = errors.New("an earlier model swap is pending")

// Bus is this instance's connection to the others. Every method is safe on a
// nil receiver, which behaves as an instance alone on the machine.
type Bus struct {
	ln net.Listener

	mu        sync.Mutex
	self      Peer
	conns     map[*conn]bool
	byPID     map[int]*conn
	foreign   map[SwapID]Swap
	mine      *Swap
	readyFrom map[int]bool
	readied   map[SwapID]bool
	loaded    map[string]loadedAt
	changed   chan struct{}
	closed    bool
}

// DefaultDir is the directory the instances share: per user, and cleared on
// logout where XDG_RUNTIME_DIR exists.
func DefaultDir() string {
	if d := os.Getenv("XDG_RUNTIME_DIR"); d != "" {
		return filepath.Join(d, "ai-code", "peers")
	}
	return filepath.Join(os.TempDir(), fmt.Sprintf("ai-code-%d", os.Getuid()), "peers")
}

// Join listens for other instances and connects to every one already there.
// self.PID is this process's unless set. An error means this platform or
// directory cannot carry messages, and the instance runs uncoordinated.
func Join(dir string, self Peer) (*Bus, error) {
	pid := self.PID
	if pid == 0 {
		pid = os.Getpid()
	}
	if err := privateDir(dir); err != nil {
		return nil, err
	}
	path := socketPath(dir, pid)
	// sockaddr_un holds about 104 bytes on macOS.
	if len(path) > 100 {
		return nil, fmt.Errorf("socket path %s is too long", path)
	}
	// The pid is this process's, so a socket under its name was left by a dead one.
	_ = os.Remove(path)
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}

	self.PID = pid
	if self.State == "" {
		self.State = StateIdle
	}
	b := &Bus{
		ln:      ln,
		self:    self,
		conns:   map[*conn]bool{},
		byPID:   map[int]*conn{},
		foreign: map[SwapID]Swap{},
		readied: map[SwapID]bool{},
		loaded:  map[string]loadedAt{},
		changed: make(chan struct{}),
	}
	go b.accept()

	// Listening comes first, so an instance starting at the same moment either
	// appears here or finds this one when it looks.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		other, ok := socketPID(e.Name())
		if !ok || other == pid {
			continue
		}
		p := filepath.Join(dir, e.Name())
		nc, err := net.DialTimeout("unix", p, time.Second)
		if err != nil {
			if processGone(other) {
				_ = os.Remove(p)
			}
			continue
		}
		go b.serve(nc, true)
	}
	return b, nil
}

func socketPath(dir string, pid int) string {
	return filepath.Join(dir, strconv.Itoa(pid)+".sock")
}

func socketPID(name string) (int, bool) {
	s, ok := strings.CutSuffix(name, ".sock")
	if !ok {
		return 0, false
	}
	pid, err := strconv.Atoi(s)
	return pid, err == nil && pid > 0
}

func (b *Bus) accept() {
	for {
		nc, err := b.ln.Accept()
		if err != nil {
			return
		}
		go b.serve(nc, false)
	}
}

// Close leaves; the others see the connection close.
func (b *Bus) Close() {
	if b == nil {
		return
	}
	b.mu.Lock()
	b.closed = true
	conns := make([]*conn, 0, len(b.conns))
	for c := range b.conns {
		conns = append(conns, c)
	}
	b.mu.Unlock()
	b.ln.Close()
	for _, c := range conns {
		c.nc.Close()
	}
}

// Changed is closed at the next change to anything the bus reports. Take it
// before reading, so a change in between is not missed.
func (b *Bus) Changed() <-chan struct{} {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.changed
}

func (b *Bus) notifyLocked() {
	close(b.changed)
	b.changed = make(chan struct{})
}

func (b *Bus) broadcastLocked(m message) {
	for c := range b.conns {
		c.send(m)
	}
}

func (b *Bus) helloLocked() message {
	self := b.self
	m := message{Kind: kindHello, Peer: &self, Swap: b.mine, Loaded: map[string]loadedAt{}}
	for id := range b.readied {
		m.Ready = append(m.Ready, id)
	}
	for k, v := range b.loaded {
		m.Loaded[k] = v
	}
	return m
}

// Self is this instance as last published.
func (b *Bus) Self() Peer {
	if b == nil {
		return Peer{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.self
}

// Set changes what this instance publishes about itself.
func (b *Bus) Set(fn func(*Peer)) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	before := b.self
	fn(&b.self)
	if b.self == before {
		return
	}
	self := b.self
	b.broadcastLocked(message{Kind: kindPeer, Peer: &self})
	b.notifyLocked()
}

// Peers lists the other instances on this instance's server.
func (b *Bus) Peers() []Peer {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []Peer
	for _, c := range b.byPID {
		if c.peer.Server == b.self.Server && b.self.Server != "" {
			out = append(out, c.peer)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

// Peer is the instance with this pid, if it is connected.
func (b *Bus) Peer(pid int) (Peer, bool) {
	if b == nil {
		return Peer{}, false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	c, ok := b.byPID[pid]
	if !ok {
		return Peer{}, false
	}
	return c.peer, true
}

// Loaded is the model last loaded on this instance's server through any
// instance, or "" when none has said.
func (b *Bus) Loaded() string {
	if b == nil {
		return ""
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.loaded[b.self.Server].Model
}

// Load says which model was last loaded on this instance's server, by which
// instance, and when it was asked for.
type Load struct {
	Model string
	By    int
	Seq   int64
}

func (b *Bus) Load() Load {
	if b == nil {
		return Load{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	l := b.loaded[b.self.Server]
	return Load{Model: l.Model, By: l.By, Seq: l.Seq}
}

// Pending lists other instances' swaps on this instance's server, earliest
// first.
func (b *Bus) Pending() []Swap {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.pendingLocked(b.self.Server)
}

func (b *Bus) pendingLocked(server string) []Swap {
	if server == "" {
		return nil
	}
	var out []Swap
	for _, s := range b.foreign {
		if s.Server == server {
			out = append(out, s)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID.precedes(out[j].ID) })
	return out
}

// PendingOn is Pending for a server this instance is not on yet.
func (b *Bus) PendingOn(server string) []Swap {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.pendingLocked(server)
}

// Ready tells the requester of a swap that this instance has stopped using
// the model and will send nothing more for it until it is loaded again.
func (b *Bus) Ready(id SwapID) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.foreign[id]; !ok || b.readied[id] {
		return
	}
	b.readied[id] = true
	b.broadcastLocked(message{Kind: kindReady, Swap: &Swap{ID: id}})
}

// Request asks every instance on server to stop using any model but into, and
// returns once all have said they have. Progress is called with those still
// to answer whenever that changes. On ErrPreempted the request is already
// withdrawn; on any other error it stays pending until Withdraw or Done.
func (b *Bus) Request(ctx context.Context, server, into string, window int, progress func([]Peer)) error {
	if b == nil || server == "" {
		return nil
	}
	b.mu.Lock()
	s := Swap{ID: SwapID{PID: b.self.PID, Seq: time.Now().UnixNano()},
		Server: server, Into: into, IntoWindow: window}
	b.mine = &s
	b.readyFrom = map[int]bool{}
	b.broadcastLocked(message{Kind: kindSwap, Swap: &s})
	b.notifyLocked()
	b.mu.Unlock()

	var last string
	for {
		b.mu.Lock()
		ch := b.changed
		for _, f := range b.pendingLocked(server) {
			if f.ID.precedes(s.ID) {
				b.withdrawLocked()
				b.mu.Unlock()
				return ErrPreempted
			}
		}
		waiting := b.waitingLocked()
		b.mu.Unlock()

		if len(waiting) == 0 {
			return nil
		}
		if progress != nil {
			if key := fmt.Sprint(waiting); key != last {
				last = key
				progress(waiting)
			}
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// waitingLocked lists the instances that have not answered this instance's
// swap; one that has not introduced itself yet may be on the same server.
func (b *Bus) waitingLocked() []Peer {
	var out []Peer
	for c := range b.conns {
		switch {
		case !c.introduced:
			out = append(out, Peer{})
		case b.byPID[c.peer.PID] != c:
		case c.peer.Server == b.mine.Server && !b.readyFrom[c.peer.PID]:
			out = append(out, c.peer)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

// Withdraw cancels this instance's swap without loading anything.
func (b *Bus) Withdraw() {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.withdrawLocked()
}

func (b *Bus) withdrawLocked() {
	if b.mine == nil {
		return
	}
	b.broadcastLocked(message{Kind: kindWithdraw, Swap: b.mine})
	b.mine = nil
	b.notifyLocked()
}

// Done announces that into is now the model on server, which releases
// everyone waiting for it, and ends this instance's swap if it had one.
func (b *Bus) Done(server, into string) {
	if b == nil || server == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	s := Swap{ID: SwapID{PID: b.self.PID, Seq: time.Now().UnixNano()}, Server: server, Into: into}
	if b.mine != nil {
		s.ID = b.mine.ID
	}
	b.mine = nil
	b.loaded[server] = loadedAt{Model: into, Seq: s.ID.Seq, By: b.self.PID}
	b.broadcastLocked(message{Kind: kindLoaded, Swap: &s})
	b.notifyLocked()
}

// conn is one connection to another instance. Writes are queued, so a slow
// reader never holds up the bus, and they leave in the order they were made.
type conn struct {
	nc         net.Conn
	dialed     bool
	introduced bool
	retired    bool
	peer       Peer

	qmu    sync.Mutex
	q      []message
	ending bool
	wake   chan struct{}
}

func (c *conn) send(m message) {
	c.qmu.Lock()
	c.q = append(c.q, m)
	c.qmu.Unlock()
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

// retire says this connection is a duplicate, then ends it.
func (c *conn) retire() {
	c.send(message{Kind: kindRetire})
	c.end()
}

// end stops writing once the queue is drained; the other side then reads EOF.
func (c *conn) end() {
	c.qmu.Lock()
	c.ending = true
	c.qmu.Unlock()
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *conn) writeLoop() {
	enc := json.NewEncoder(c.nc)
	for range c.wake {
		c.qmu.Lock()
		q, ending := c.q, c.ending
		c.q = nil
		c.qmu.Unlock()
		for _, m := range q {
			if enc.Encode(m) != nil {
				return
			}
		}
		if ending {
			if cw, ok := c.nc.(interface{ CloseWrite() error }); ok {
				_ = cw.CloseWrite()
			}
			return
		}
	}
}

func (b *Bus) serve(nc net.Conn, dialed bool) {
	c := &conn{nc: nc, dialed: dialed, wake: make(chan struct{}, 1)}
	go c.writeLoop()
	defer nc.Close()
	defer c.end()

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	// Queued in the same critical section that adds the connection, so the
	// introduction and every later message reach it in order.
	c.send(b.helloLocked())
	b.conns[c] = true
	b.notifyLocked()
	b.mu.Unlock()

	dec := json.NewDecoder(bufio.NewReader(nc))
	for {
		var m message
		if err := dec.Decode(&m); err != nil {
			break
		}
		b.handle(c, m)
	}

	b.mu.Lock()
	delete(b.conns, c)
	if c.introduced && !c.retired && b.byPID[c.peer.PID] == c {
		b.forgetLocked(c.peer.PID)
	}
	b.notifyLocked()
	b.mu.Unlock()
}

// forgetLocked drops an instance that has gone: its swaps die with it.
func (b *Bus) forgetLocked(pid int) {
	delete(b.byPID, pid)
	for id := range b.foreign {
		if id.PID == pid {
			delete(b.foreign, id)
			delete(b.readied, id)
		}
	}
}

func (b *Bus) handle(c *conn, m message) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if m.Kind == kindHello {
		if m.Peer == nil || c.introduced {
			return
		}
		c.introduced = true
		c.peer = *m.Peer
		pid := c.peer.PID
		if old, ok := b.byPID[pid]; ok {
			// Two instances that started together each dialled the other. Both
			// keep the connection dialled by the lower pid, and stop writing to
			// the other one so it drains and closes.
			keepNew := c.dialed == (b.self.PID < pid)
			if !keepNew {
				c.retired = true
				delete(b.conns, c)
				c.retire()
				return
			}
			old.retired = true
			delete(b.conns, old)
			old.retire()
		}
		b.byPID[pid] = c
		// A hello is the sender's whole state, so it replaces what was known.
		for id := range b.foreign {
			if id.PID == pid {
				delete(b.foreign, id)
			}
		}
		if m.Swap != nil {
			b.foreign[m.Swap.ID] = *m.Swap
		}
		if b.mine != nil {
			for _, id := range m.Ready {
				if id == b.mine.ID {
					b.readyFrom[pid] = true
				}
			}
		}
		for server, l := range m.Loaded {
			if l.Seq > b.loaded[server].Seq {
				b.loaded[server] = l
			}
		}
		b.notifyLocked()
		return
	}

	if m.Kind == kindRetire {
		// The peer is still there, on the connection whose introduction may
		// not have arrived yet.
		c.retired = true
		delete(b.conns, c)
		b.notifyLocked()
		return
	}
	if !c.introduced || c.retired || b.byPID[c.peer.PID] != c {
		return
	}
	pid := c.peer.PID
	switch m.Kind {
	case kindPeer:
		if m.Peer != nil {
			c.peer = *m.Peer
		}
	case kindSwap:
		if m.Swap != nil && m.Swap.ID.PID == pid {
			b.foreign[m.Swap.ID] = *m.Swap
		}
	case kindWithdraw:
		if m.Swap != nil {
			delete(b.foreign, m.Swap.ID)
			delete(b.readied, m.Swap.ID)
		}
	case kindLoaded:
		if m.Swap != nil {
			delete(b.foreign, m.Swap.ID)
			delete(b.readied, m.Swap.ID)
			if m.Swap.ID.Seq > b.loaded[m.Swap.Server].Seq {
				b.loaded[m.Swap.Server] = loadedAt{Model: m.Swap.Into, Seq: m.Swap.ID.Seq, By: pid}
			}
		}
	case kindReady:
		if m.Swap != nil && b.mine != nil && m.Swap.ID == b.mine.ID {
			b.readyFrom[pid] = true
		}
	default:
		return
	}
	b.notifyLocked()
}
