// Package session persists a conversation as an append-only JSONL file.
//
// The transcript is the source of truth, not a log kept alongside some other
// state. Rebuilding a session means replaying its entries, which makes resume,
// fork and inspection fall out for free, and makes a crash lose at most the
// turn in progress. It is also greppable, which matters more in practice than
// it sounds: when something goes wrong, the record of what was actually sent is
// the first thing anyone wants.
package session

import (
	"bufio"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"ai-code/internal/provider"
)

type EntryType string

const (
	EntryMeta       EntryType = "meta"
	EntryMessage    EntryType = "message"
	EntryCompaction EntryType = "compaction"
	EntryNote       EntryType = "note"
	// EntryInput is a line the user submitted, prompt or slash command,
	// exactly as typed. It exists because neither of the other two records it
	// faithfully: a slash command never becomes a message at all, and a prompt
	// that does is later wrapped, summarised or dropped by compaction. Command
	// history has to outlive all of that.
	EntryInput EntryType = "input"
	// EntryCheckpoint records a summary that did NOT replace anything. This is
	// the non-destructive counterpart of EntryCompaction: the transcript is
	// still whole, and the summary sits beside it as a spare for a model swap
	// or a narrower window. Replay must therefore not treat it as a reset.
	EntryCheckpoint EntryType = "checkpoint"
)

type Entry struct {
	Type EntryType `json:"type"`
	Time time.Time `json:"time"`

	Message *provider.Message `json:"message,omitempty"`
	Meta    *Meta             `json:"meta,omitempty"`
	Note    string            `json:"note,omitempty"`
	Input   string            `json:"input,omitempty"`

	// Compaction records what a /compact replaced, so the pre-compaction
	// history stays on disk even though it left the context. Checkpoint
	// records a summary that replaced nothing.
	Summary string `json:"summary,omitempty"`
	// SummarisedThrough is how many messages from the start of the replayed
	// conversation the summary accounts for. Meaningless without the message
	// list it indexes, which is why it is stored with the summary rather than
	// recomputed on resume.
	SummarisedThrough int `json:"summarised_through,omitempty"`
	MessagesBefore    int `json:"messages_before,omitempty"`
	TokensBefore      int `json:"tokens_before,omitempty"`
	// Cut is the boundary a compaction chose: the next request starts from
	// this message. Zero on a checkpoint written speculatively, which covers
	// messages without deciding anything about what is sent.
	Cut int `json:"cut,omitempty"`
}

// Meta is written once at the head of every session.
type Meta struct {
	ID         string    `json:"id"`
	Started    time.Time `json:"started"`
	Project    string    `json:"project"`
	Provider   string    `json:"provider"`
	Model      string    `json:"model"`
	Mode       string    `json:"mode"`
	ForkedFrom string    `json:"forked_from,omitempty"`
	Runtime    string    `json:"runtime,omitempty"`
	Version    string    `json:"version"`
}

type Session struct {
	Meta Meta
	path string
	f    *os.File
	w    *bufio.Writer
}

// Root is the directory sessions live under.
func Root() (string, error) {
	if d := os.Getenv("AI_CODE_DATA_DIR"); d != "" {
		return d, nil
	}
	if d := os.Getenv("XDG_DATA_HOME"); d != "" {
		return filepath.Join(d, "ai-code"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".local", "share", "ai-code"), nil
}

// projectKey derives a stable directory name from a project path.
//
// The absolute path is also recorded in the metadata, because hashing the path
// means a moved repository looks like a different project. Keeping the original
// path lets ai-code say so rather than silently starting from nothing.
func projectKey(project string) string {
	sum := sha256.Sum256([]byte(project))
	base := filepath.Base(project)
	base = strings.Map(func(r rune) rune {
		if r == '/' || r == '\\' || r == ':' {
			return '-'
		}
		return r
	}, base)
	return fmt.Sprintf("%s-%s", base, hex.EncodeToString(sum[:4]))
}

func dirFor(project string) (string, error) {
	root, err := Root()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "sessions", projectKey(project)), nil
}

// NewID returns a lexically sortable identifier: sorting session files by name
// sorts them by time, which is what every listing wants.
func NewID() string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return fmt.Sprintf("%s-%s", time.Now().UTC().Format("20060102T150405"), hex.EncodeToString(b[:]))
}

// Create starts a new session file.
func Create(meta Meta) (*Session, error) {
	if meta.ID == "" {
		meta.ID = NewID()
	}
	if meta.Started.IsZero() {
		meta.Started = time.Now()
	}
	dir, err := dirFor(meta.Project)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("creating session directory: %w", err)
	}

	path := filepath.Join(dir, meta.ID+".jsonl")
	// 0600: a transcript contains whatever the model was shown, which routinely
	// includes source, configuration and anything a command printed.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("creating session file: %w", err)
	}

	s := &Session{Meta: meta, path: path, f: f, w: bufio.NewWriter(f)}
	if err := s.Append(Entry{Type: EntryMeta, Time: meta.Started, Meta: &meta}); err != nil {
		f.Close()
		return nil, err
	}
	return s, nil
}

// Open reopens an existing session for appending.
func Open(project, id string) (*Session, []Entry, error) {
	dir, err := dirFor(project)
	if err != nil {
		return nil, nil, err
	}
	path := filepath.Join(dir, id+".jsonl")

	entries, meta, err := readFile(path)
	if err != nil {
		return nil, nil, err
	}

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, nil, fmt.Errorf("reopening session: %w", err)
	}
	return &Session{Meta: meta, path: path, f: f, w: bufio.NewWriter(f)}, entries, nil
}

func (s *Session) Path() string { return s.path }

// Append writes one entry and flushes it. Flushing every entry costs a syscall
// per message and buys crash-safety, which is the right trade for a file whose
// whole purpose is surviving an unexpected exit.
func (s *Session) Append(e Entry) error {
	if e.Time.IsZero() {
		e.Time = time.Now()
	}
	raw, err := json.Marshal(e)
	if err != nil {
		return err
	}
	if _, err := s.w.Write(append(raw, '\n')); err != nil {
		return err
	}
	return s.w.Flush()
}

func (s *Session) AppendMessage(m provider.Message) error {
	return s.Append(Entry{Type: EntryMessage, Message: &m})
}

func (s *Session) Close() error {
	if s.w != nil {
		_ = s.w.Flush()
	}
	if s.f != nil {
		return s.f.Close()
	}
	return nil
}

// readFile parses a transcript. A truncated final line is tolerated: a crash
// mid-write should cost the last entry, not the whole session.
func readFile(path string) ([]Entry, Meta, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, Meta{}, fmt.Errorf("no session %s", filepath.Base(path))
		}
		return nil, Meta{}, err
	}
	defer f.Close()

	var (
		entries []Entry
		meta    Meta
	)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 256*1024), 32*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var e Entry
		if err := json.Unmarshal(line, &e); err != nil {
			// Almost certainly a partial final line from an interrupted write.
			continue
		}
		if e.Type == EntryMeta && e.Meta != nil {
			meta = *e.Meta
		}
		entries = append(entries, e)
	}
	return entries, meta, sc.Err()
}

// Messages replays entries into a conversation.
//
// Compaction entries reset the conversation: everything before one is
// represented by its summary, exactly as it was in context when the session was
// live. The full history stays in the file for anyone who wants to read it.
func Messages(entries []Entry) []provider.Message {
	var out []provider.Message
	for _, e := range entries {
		switch e.Type {
		case EntryMessage:
			if e.Message != nil {
				out = append(out, *e.Message)
			}
		case EntryCompaction:
			out = out[:0]
		}
	}
	return out
}

// Checkpoint returns the summary a resumed session should start with, how
// many of the replayed messages it accounts for, and the boundary the next
// request starts from.
//
// Cut is zero for a checkpoint written speculatively, which stands beside the
// transcript without changing what is sent.
//
// EntryCompaction is the pre-append-only form, written by versions that
// replaced the message list instead of recording a boundary. Messages replays
// one by clearing the conversation, exactly as it was in context when the
// session was live, so the summary is already the first message there and
// handing it back as a checkpoint too would put the same text in front of the
// model twice. Files already on disk are the only thing that produces one.
func Checkpoint(entries []Entry) (summary string, through, cut int) {
	for _, e := range entries {
		switch e.Type {
		case EntryCheckpoint:
			summary, through, cut = e.Summary, e.SummarisedThrough, e.Cut
		case EntryCompaction:
			summary, through, cut = "", 0, 0
		}
	}
	return summary, through, cut
}

// Inputs returns the lines the user submitted, oldest first, for restoring
// command history on resume.
//
// Deliberately not cleared by a compaction entry the way Messages is:
// compaction is about what the model is shown, and it would be surprising for
// reclaiming context to also erase what you can press Up to reach.
//
// Sessions recorded before EntryInput existed fall back to their user
// messages, so a session already on disk still gets most of its history back.
// The fallback skips the wrappers compaction introduces, which are ai-code's
// words rather than anything the user typed.
func Inputs(entries []Entry) []string {
	var out []string
	for _, e := range entries {
		if e.Type == EntryInput && strings.TrimSpace(e.Input) != "" {
			out = append(out, e.Input)
		}
	}
	if len(out) > 0 {
		return out
	}
	for _, e := range entries {
		if e.Type != EntryMessage || e.Message == nil || e.Message.Role != provider.RoleUser {
			continue
		}
		c := e.Message.Content
		if strings.TrimSpace(c) == "" || strings.Contains(c, "<context-checkpoint>") {
			continue
		}
		out = append(out, c)
	}
	return out
}

// Info summarises a session for listings.
type Info struct {
	ID       string
	Path     string
	Project  string
	Model    string
	Started  time.Time
	Modified time.Time
	Messages int
	Preview  string
	Size     int64
}

// List returns the sessions recorded for a project, newest first.
func List(project string) ([]Info, error) {
	dir, err := dirFor(project)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}

	var out []Info
	for _, de := range entries {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".jsonl") {
			continue
		}
		path := filepath.Join(dir, de.Name())
		fi, err := de.Info()
		if err != nil {
			continue
		}
		items, meta, err := readFile(path)
		if err != nil {
			continue
		}

		info := Info{
			ID:       strings.TrimSuffix(de.Name(), ".jsonl"),
			Path:     path,
			Project:  meta.Project,
			Model:    meta.Model,
			Started:  meta.Started,
			Modified: fi.ModTime(),
			Size:     fi.Size(),
		}
		for _, it := range items {
			if it.Type == EntryMessage && it.Message != nil {
				info.Messages++
				if info.Preview == "" && it.Message.Role == provider.RoleUser {
					info.Preview = firstLine(it.Message.Content, 70)
				}
			}
		}
		out = append(out, info)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out, nil
}

// Latest returns the most recently modified session for a project.
func Latest(project string) (*Info, error) {
	all, err := List(project)
	if err != nil {
		return nil, err
	}
	if len(all) == 0 {
		return nil, errors.New("no previous session for this project")
	}
	return &all[0], nil
}

func firstLine(s string, max int) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > max {
		s = s[:max-1] + "…"
	}
	return s
}
