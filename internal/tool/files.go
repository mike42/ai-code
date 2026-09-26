package tool

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

// resolve turns a possibly-relative path into an absolute one against the
// executor's tracked working directory, and expands a leading ~.
func resolve(st *State, path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, strings.TrimPrefix(path, "~"))
		}
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path), nil
	}
	return filepath.Clean(filepath.Join(st.Cwd(), path)), nil
}

// rel renders a path relative to the working directory when that is shorter,
// so messages read like the paths a person would type.
func rel(st *State, path string) string {
	if r, err := filepath.Rel(st.Cwd(), path); err == nil && !strings.HasPrefix(r, "..") {
		return r
	}
	return path
}

func stampFile(path string, content []byte) FileStamp {
	sum := sha256.Sum256(content)
	st := FileStamp{
		Path:  path,
		Hash:  hex.EncodeToString(sum[:]),
		Size:  int64(len(content)),
		Lines: countLines(content),
	}
	if fi, err := os.Stat(path); err == nil {
		st.ModTime = fi.ModTime()
	} else {
		st.ModTime = time.Now()
	}
	return st
}

func countLines(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	n := 0
	for _, c := range b {
		if c == '\n' {
			n++
		}
	}
	if b[len(b)-1] != '\n' {
		n++
	}
	return n
}

// looksBinary reports whether content should not be handed to the model as
// text: a NUL byte or invalid UTF-8 in the first few KB is decisive enough.
func looksBinary(b []byte) bool {
	probe := b
	if len(probe) > 8000 {
		probe = probe[:8000]
	}
	for _, c := range probe {
		if c == 0 {
			return true
		}
	}
	return !utf8.Valid(probe)
}

// checkStale reports whether a file changed since a tool last read it, the
// guard against editing content that no longer exists.
func checkStale(st *State, path string, current []byte) *Result {
	prev, ok := st.Stamp(path)
	if !ok {
		r := Errorf("%s has not been read in this session.\n\n"+
			"Use the read tool on it first. Viewing it with cat, sed or head in the bash "+
			"tool does not count: only the read tool records the content an edit is "+
			"checked against. Editing without reading is how edits get applied to "+
			"content that is no longer there.", rel(st, path))
		return &r
	}
	now := stampFile(path, current)
	if now.Hash != prev.Hash {
		st.ForgetStamp(path)
		r := Errorf("%s has changed on disk since it was read "+
			"(was %d bytes, now %d bytes).\n\n"+
			"The edit was NOT applied. Read the file again and reissue the edit "+
			"against its current contents.", rel(st, path), prev.Size, now.Size)
		return &r
	}
	return nil
}
