// Package session persists a conversation as an append-only JSONL file. The
// transcript is the source of truth: replaying its entries rebuilds the
// session, so a crash loses at most the turn in progress.
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
	// EntryInput is a line submitted exactly as typed, prompt or slash command.
	// Neither messages nor notes record it faithfully: a slash command never
	// becomes a message, and a prompt is later wrapped or dropped.
	EntryInput EntryType = "input"
	// EntryCheckpoint records a summary that replaced nothing: the
	// non-destructive counterpart of EntryCompaction, which replay must not
	// treat as a reset.
	EntryCheckpoint EntryType = "checkpoint"
)

type Entry struct {
	Type EntryType `json:"type"`
	Time time.Time `json:"time"`

	Message *provider.Message `json:"message,omitempty"`
	Meta    *Meta             `json:"meta,omitempty"`
	Note    string            `json:"note,omitempty"`
	Input   string            `json:"input,omitempty"`

	// Summary is what a compaction replaced, so the pre-compaction history stays
	// on disk.
	Summary string `json:"summary,omitempty"`
	// SummarisedThrough is how many messages from the start the summary accounts
	// for, stored with the summary rather than recomputed on resume.
	SummarisedThrough int `json:"summarised_through,omitempty"`
	MessagesBefore    int `json:"messages_before,omitempty"`
	TokensBefore      int `json:"tokens_before,omitempty"`
	// Cut is the boundary a compaction chose: the next request starts from this
	// message. Zero on a speculative checkpoint.
	Cut int `json:"cut,omitempty"`
}

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

// projectKey derives a stable directory name from a project path. The path
// itself is also recorded in the metadata, so a moved repository is identifiable.
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
	if stateDir != "" {
		return filepath.Join(stateDir, "sessions"), nil
	}
	root, err := Root()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "sessions", projectKey(project)), nil
}

var stateDir string

// SetStateDir puts this process's sessions and workspace files under dir.
func SetStateDir(dir string) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	stateDir = abs
	return nil
}

// WorkspaceDir is per working directory (local path or ssh:// URL), so agents
// in different directories never share files.
func WorkspaceDir(where string) (string, error) {
	if stateDir != "" {
		return stateDir, nil
	}
	root, err := Root()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "workspaces", projectKey(where)), nil
}

// NewID returns a lexically sortable identifier: name order is time order.
func NewID() string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return fmt.Sprintf("%s-%s", time.Now().UTC().Format("20060102T150405"), hex.EncodeToString(b[:]))
}

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
	// 0600: a transcript holds whatever the model was shown, including source
	// and configuration.
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

// Append writes one entry and flushes it: a syscall per message buys
// crash-safety on a file whose purpose is surviving an unexpected exit.
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

// readFile tolerates a truncated final line from a crash mid-write.
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
			continue
		}
		if e.Type == EntryMeta && e.Meta != nil {
			meta = *e.Meta
		}
		entries = append(entries, e)
	}
	return entries, meta, sc.Err()
}

// Messages replays entries into a conversation; a compaction entry resets it,
// with the summary standing in for everything before it.
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

// Checkpoint returns the summary a resumed session should start with, how many
// of the replayed messages it accounts for, and the boundary the next request
// starts from. Cut is zero for a speculative checkpoint. An EntryCompaction
// clears the result: Messages already replays it, so it would arrive twice.
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

// Inputs returns the lines submitted, oldest first, for restoring command
// history on resume. A compaction entry does not clear it the way Messages
// does; sessions without EntryInput entries fall back to their user messages.
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
