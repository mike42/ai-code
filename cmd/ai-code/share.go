package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"ai-code/internal/agent"
	"ai-code/internal/config"
	"ai-code/internal/coord"
	"ai-code/internal/provider"
)

// modelShare is this instance's part in taking turns on a model server with
// the other instances using it. The main run holds the model from its claim
// until it parks at a boundary or finishes; any other request holds it while
// it streams. Another instance's swap is answered once nothing holds it.
type modelShare struct {
	bus *coord.Bus

	mu          sync.Mutex
	running     bool
	parked      bool
	inflight    int
	summarising bool
	prompting   bool
	kick        chan struct{}

	// joined is when this instance arrived: only a load after it can have
	// taken a model away from it.
	joined int64
	// seen is the last load noticed, and displaced the model another
	// instance loaded over this one's, until either one is chosen.
	seen      coord.Load
	displaced string
	told      map[coord.SwapID]bool
}

func newModelShare(bus *coord.Bus, joined int64) *modelShare {
	return &modelShare{bus: bus, kick: make(chan struct{}, 1), joined: joined, told: map[coord.SwapID]bool{}}
}

// serverKey names the server a client's models share; "" opts out, as for a
// cloud provider, where loading one model evicts nothing.
func serverKey(c provider.Client, pc config.Provider) string {
	if c.Class() == provider.ClassCloud {
		return ""
	}
	return strings.TrimRight(pc.BaseURL, "/")
}

type holdKey struct{}

// holding marks the main run's requests, which its claim already covers.
func holding(ctx context.Context) context.Context { return context.WithValue(ctx, holdKey{}, true) }

func holds(ctx context.Context) bool { return ctx.Value(holdKey{}) != nil }

func (s *modelShare) poke() {
	select {
	case s.kick <- struct{}{}:
	default:
	}
}

// against lists the pending swaps that would take model away.
func (s *modelShare) against(model string) []coord.Swap {
	var out []coord.Swap
	for _, sw := range s.bus.Pending() {
		if sw.Into != model {
			out = append(out, sw)
		}
	}
	return out
}

// usable reports whether a request for model can go now. Caller holds mu.
func (s *modelShare) usable(model string) bool {
	if s.bus.Self().Server == "" {
		return true
	}
	if len(s.against(model)) > 0 {
		return false
	}
	l := s.bus.Loaded()
	return l == "" || l == model
}

func (s *modelShare) setState(st coord.State) {
	s.bus.Set(func(p *coord.Peer) { p.State = st })
}

// admit holds back a request that is not the main run's until its model can
// be used, and counts it while it streams.
func (s *modelShare) admit(ctx context.Context, model string) (func(), error) {
	if s == nil || holds(ctx) {
		return func() {}, nil
	}
	for {
		changed := s.bus.Changed()
		s.mu.Lock()
		if s.usable(model) {
			s.inflight++
			s.mu.Unlock()
			var once sync.Once
			return func() {
				once.Do(func() {
					s.mu.Lock()
					s.inflight--
					s.mu.Unlock()
					s.poke()
				})
			}, nil
		}
		s.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (s *modelShare) wrap(c provider.Client) provider.Client {
	if s == nil {
		return c
	}
	sc := sharedClient{Client: c, share: s}
	if in, ok := c.(provider.Introspector); ok {
		return sharedIntrospector{sharedClient: sc, Introspector: in}
	}
	return sc
}

type sharedClient struct {
	provider.Client
	share *modelShare
}

func (c sharedClient) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	release, err := c.share.admit(ctx, req.Model)
	if err != nil {
		return nil, err
	}
	st, err := c.Client.Stream(ctx, req)
	if err != nil {
		release()
		return nil, err
	}
	return &releasingStream{Stream: st, release: release}, nil
}

type sharedIntrospector struct {
	sharedClient
	provider.Introspector
}

type releasingStream struct {
	provider.Stream
	release func()
}

func (r *releasingStream) Close() error {
	err := r.Stream.Close()
	r.release()
	return err
}

// claim takes the model for a run, asking the other instances to make way
// when another model is loaded.
func (a *App) claim(ctx context.Context) error {
	s := a.share
	if s == nil {
		return nil
	}
	defer a.status("")
	for {
		changed := s.bus.Changed()
		self := s.bus.Self()
		s.mu.Lock()
		if self.Server == "" || (len(s.against(self.Model)) == 0 && s.bus.Loaded() == self.Model) {
			s.running = true
			s.mu.Unlock()
			s.setState(coord.StateWorking)
			return nil
		}
		blocked := s.against(self.Model)
		s.mu.Unlock()

		if len(blocked) > 0 {
			a.status("holding this prompt until " + blocked[0].Into + " is loaded")
			select {
			case <-changed:
			case <-ctx.Done():
				return ctx.Err()
			}
			continue
		}

		coord.Trace("claim: loaded=%q, asking for %s", s.bus.Loaded(), self.Model)
		err := s.bus.Request(ctx, self.Server, self.Model, a.agent.Window(), a.showWaiting)
		switch {
		case errors.Is(err, coord.ErrPreempted):
			continue
		case err != nil:
			s.bus.Withdraw()
			return err
		}
		s.bus.Done(self.Server, self.Model)
	}
}

// finishRun gives the model back at the end of a run.
func (a *App) finishRun() {
	s := a.share
	if s == nil {
		return
	}
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
	s.setState(coord.StateIdle)
	s.poke()
}

// atBoundary runs between loop iterations, where the message list is complete.
// When another instance needs the server for a different model, the run parks
// here and carries on once its own model is loaded again.
func (a *App) atBoundary(ctx context.Context) bool {
	if a.runQueuedCommands(ctx) {
		return true
	}
	s := a.share
	if s == nil {
		return false
	}
	self := s.bus.Self()
	s.mu.Lock()
	if s.usable(self.Model) {
		s.mu.Unlock()
		return false
	}
	against := s.against(self.Model)
	s.mu.Unlock()

	if len(against) > 0 {
		a.summariseForSwap(ctx, against[0], a.note)
	}
	s.mu.Lock()
	s.running, s.parked = false, true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.parked = false
		s.mu.Unlock()
	}()
	s.setState(coord.StateParked)
	s.poke()

	coord.Trace("park: %s waits for its model", self.Model)
	why := s.bus.Loaded() + " is loaded"
	if len(against) > 0 {
		why = a.describeRequester(against[0]) + " is loading " + against[0].Into
	}
	a.note(fmt.Sprintf("Paused: %s. This run carries on when %s is loaded again; Ctrl-C stops it here.",
		why, self.Model))

	for {
		changed := s.bus.Changed()
		s.mu.Lock()
		if len(s.against(self.Model)) == 0 && s.bus.Loaded() == self.Model {
			s.running = true
			s.mu.Unlock()
			break
		}
		s.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return true
		}
	}
	s.setState(coord.StateWorking)
	coord.Trace("park: %s is back", self.Model)
	a.note(self.Model + " is loaded again; carrying on.")
	return false
}

// answerSwaps answers other instances' swaps for as long as the session runs.
func (a *App) answerSwaps(ctx context.Context) {
	s := a.share
	for {
		changed := s.bus.Changed()
		a.answerPending(ctx)
		a.tellIdle(a.noticeChanges())
		select {
		case <-changed:
		case <-s.kick:
		case <-ctx.Done():
			return
		}
	}
}

func (a *App) answerPending(ctx context.Context) {
	s := a.share
	for _, sw := range s.bus.Pending() {
		if sw.Into != s.bus.Self().Model {
			a.summariseIdle(ctx, sw)
		}
		s.mu.Lock()
		if sw.Into == s.bus.Self().Model || !(s.running || s.inflight > 0 || s.summarising) {
			s.bus.Ready(sw.ID)
		}
		s.mu.Unlock()
	}
}

// noticeChanges follows what the other instances do to this one's model and
// returns what a person at this window should be told.
func (a *App) noticeChanges() []string {
	s := a.share
	self := s.bus.Self()
	if self.Server == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var lines []string
	for _, sw := range s.bus.Pending() {
		if sw.Into == self.Model || s.told[sw.ID] {
			continue
		}
		s.told[sw.ID] = true
		lines = append(lines, fmt.Sprintf("── %s is switching the server to %s.", a.describeRequester(sw), sw.Into))
	}

	l := s.bus.Load()
	if l == s.seen {
		return lines
	}
	s.seen = l
	switch {
	case l.Model == "":
	case l.Model == self.Model:
		if s.displaced != "" {
			s.displaced = ""
			lines = append(lines, "── "+l.Model+" is loaded again.")
		}
	case l.By != self.PID && l.Seq > s.joined:
		s.displaced = l.Model
		lines = append(lines, fmt.Sprintf("── %s switched the server to %s. Your next prompt goes to %s; /model %s switches back.",
			a.describePID(l.By), l.Model, l.Model, self.Model))
	}
	return lines
}

// tellIdle shows lines above the prompt while one is up; a window in a run
// says what happened through its own notes instead.
func (a *App) tellIdle(lines []string) {
	s := a.share
	s.mu.Lock()
	prompting := s.prompting && !s.running && !s.parked
	s.mu.Unlock()
	if len(lines) == 0 || !prompting {
		return
	}
	if ed := a.liveEditor(); ed != nil {
		ed.EmitAbove(a.promptString(), lines...)
	}
}

func (a *App) setPrompting(on bool) {
	if s := a.share; s != nil {
		s.mu.Lock()
		s.prompting = on
		s.mu.Unlock()
		s.poke()
	}
}

// beforeSend runs before a typed prompt is sent. A swap in progress is waited
// out, since sending now would load this window's model back over it; when
// another window has since loaded a different model, the prompt goes to that
// one.
func (a *App) beforeSend(parent context.Context) error {
	s := a.share
	if s == nil {
		return nil
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	restore := installInterrupt(cancel)
	defer restore()
	defer a.status("")

	for {
		changed := s.bus.Changed()
		against := s.against(s.bus.Self().Model)
		if len(against) == 0 {
			break
		}
		a.status("holding this prompt until " + against[0].Into + " is loaded")
		select {
		case <-changed:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	for _, line := range a.noticeChanges() {
		a.note(line)
	}

	s.mu.Lock()
	model := s.displaced
	s.mu.Unlock()
	if model == "" {
		return nil
	}
	a.note(fmt.Sprintf("%s loaded %s in place of %s, so this prompt goes to %s. /model %s switches back.",
		a.describePID(s.bus.Load().By), model, s.bus.Self().Model, model, s.bus.Self().Model))
	info := provider.ModelInfo{ID: model}
	if models, err := a.client.Models(ctx); err == nil {
		for _, m := range models {
			if m.ID == model {
				info = m
				break
			}
		}
	}
	return a.switchTo(ctx, a.client, a.providerName, a.providerCfg, info, false)
}

// displacedNote says, after a run, that its model was taken while it ran.
func (a *App) displacedNote() string {
	s := a.share
	if s == nil {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.displaced == "" {
		return ""
	}
	return fmt.Sprintf("%s is loaded now. Your next prompt goes to %s; /model %s switches back.",
		s.displaced, s.displaced, s.bus.Self().Model)
}

func (s *modelShare) chose(server, model string) {
	s.bus.Set(func(p *coord.Peer) { p.Server, p.Model = server, model })
	s.mu.Lock()
	s.displaced = ""
	s.seen = s.bus.Load()
	s.mu.Unlock()
}

// summariseIdle writes the summary a swap calls for while this instance is
// between runs. When the main loop holds the agent it is busy with something
// that sends nothing, and the swap is answered without one.
func (a *App) summariseIdle(ctx context.Context, sw coord.Swap) {
	s := a.share
	if !a.agentMu.TryLock() {
		return
	}
	defer a.agentMu.Unlock()

	s.mu.Lock()
	if s.running || s.inflight > 0 || s.summarising || !a.needsSummaryFor(sw) {
		s.mu.Unlock()
		return
	}
	s.summarising = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.summarising = false
		s.mu.Unlock()
		s.setState(coord.StateIdle)
	}()

	say := a.note
	if ed := a.liveEditor(); ed != nil {
		say = func(line string) { ed.EmitAbove(a.promptString(), line) }
	}
	a.summariseForSwap(ctx, sw, say)
}

// summariseForSwap writes a summary while the current model is still loaded,
// when the incoming one could not hold this conversation.
func (a *App) summariseForSwap(ctx context.Context, sw coord.Swap, say func(string)) {
	if !a.needsSummaryFor(sw) {
		return
	}
	model := a.share.bus.Self().Model
	a.share.setState(coord.StateSummarising)
	say(fmt.Sprintf("%s is loading %s, which this conversation will not fit. Writing a summary while %s is still loaded…",
		a.describeRequester(sw), sw.Into, model))
	if _, err := a.agent.Summarise(holding(ctx), agent.SpeculativeSummaryMaxTokens); err != nil {
		say("Could not write a summary: " + err.Error())
		return
	}
	a.recordCheckpoint()
	say("Summary written before " + model + " was unloaded.")
}

// needsSummaryFor reports whether this conversation would be stranded on the
// incoming model. A standby summary counts: assembly falls back to it.
func (a *App) needsSummaryFor(sw coord.Swap) bool {
	if a.agent.TranscriptTokens() == 0 || !a.modelIsWarm() {
		return false
	}
	if sw.IntoWindow <= 0 {
		// Whether it fits cannot be decided; saving is the recoverable mistake.
		return true
	}
	if summary, _ := a.agent.Summary(); summary != "" {
		return false
	}
	return a.agent.TranscriptTokens() > sw.IntoWindow
}

// modelIsWarm reports whether the last round ran on the model the agent would
// send to now. A summary sent to any other model is a prefill of the whole
// conversation, holding a slot for nothing.
func (a *App) modelIsWarm() bool {
	return a.lastRoundModel != "" && a.lastRoundModel == a.agent.Model()
}

func (a *App) describeRequester(sw coord.Swap) string { return a.describePID(sw.ID.PID) }

func (a *App) describePID(pid int) string {
	if p, ok := a.share.bus.Peer(pid); ok {
		return p.Describe()
	}
	return "another window"
}

// swapTo makes into the model on server, after every other instance there has
// stopped using the one it replaces. The wait is not modal: typing during it
// is queued and runs once the model loads.
func (a *App) swapTo(parent context.Context, client provider.Client, server string, into provider.ModelInfo, window int) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	stopSteering := a.steerWith(cancel, a.queuePrompt)
	defer func() { a.pendingInput = stopSteering() }()
	defer a.status("")

	s := a.share
	if s != nil && server != "" && server == s.bus.Self().Server && s.bus.Loaded() == into.ID {
		return
	}
	if s != nil && server != "" {
		a.waitForPeers(ctx, server, into, window)
	}
	a.loadModel(parent, client, into)
	if s != nil {
		s.bus.Done(server, into.ID)
	}
}

func (a *App) waitForPeers(ctx context.Context, server string, into provider.ModelInfo, window int) {
	bus := a.share.bus
	var waiting []coord.Peer
	progress := func(w []coord.Peer) {
		waiting = w
		a.showWaiting(w)
	}
	for {
		err := bus.Request(ctx, server, into.ID, window, progress)
		if err == nil {
			return
		}
		if errors.Is(err, coord.ErrPreempted) {
			for len(bus.PendingOn(server)) > 0 && ctx.Err() == nil {
				changed := bus.Changed()
				if len(bus.PendingOn(server)) == 0 {
					break
				}
				a.status("waiting for another window's model swap")
				select {
				case <-changed:
				case <-ctx.Done():
				}
			}
			if ctx.Err() == nil {
				continue
			}
		}
		// Only Ctrl-C ends the wait; cutting it off can cost another window its work.
		a.note(fmt.Sprintf("Loading %s without waiting for %s.", into.ID, listPeers(waiting)))
		return
	}
}

func (a *App) showWaiting(peers []coord.Peer) {
	a.status(describeWaiting(peers))
}

// describeWaiting is the one line shown while a swap waits.
func describeWaiting(peers []coord.Peer) string {
	parts := make([]string, 0, len(peers))
	for _, p := range peers {
		parts = append(parts, p.Describe()+" "+p.State.Describe())
	}
	return "waiting · " + strings.Join(parts, " · ")
}

// status puts one line in the transient zone, replacing whatever was there.
func (a *App) status(text string) {
	if a.interactive != nil {
		a.interactive.SetNotice(text)
	}
}

// loadModel loads the weights, so /model X means the model has changed rather
// than that the next request will. Lemonade loads on demand, which would defer
// the eviction peers were told about to an unrelated later moment.
func (a *App) loadModel(ctx context.Context, client provider.Client, into provider.ModelInfo) {
	in, ok := client.(provider.Introspector)
	if !ok {
		return
	}
	a.status("loading " + into.ID)
	coord.Trace("POST /load %s", into.ID)
	if err := in.Load(ctx, into.ID); err != nil {
		coord.Trace("load %s failed: %v", into.ID, err)
		a.note("The server did not load " + into.ID + ": " + err.Error() +
			". The next request will try again.")
	}
}

// listPeers names sessions the way a person tells one window from another.
func listPeers(peers []coord.Peer) string {
	names := make([]string, 0, len(peers))
	for _, p := range peers {
		names = append(names, p.Describe())
	}
	switch len(names) {
	case 0:
		return "anything"
	case 1:
		return names[0]
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}
