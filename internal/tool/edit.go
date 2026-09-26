package tool

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

//go:embed edit.txt
var editDescription string

type EditTool struct{}

func (t *EditTool) Name() string        { return "edit" }
func (t *EditTool) Description() string { return editDescription }
func (t *EditTool) ReadOnly() bool      { return false }

func (t *EditTool) Schema() json.RawMessage {
	return json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "Path to the file to edit."},
    "old_string": {"type": "string", "description": "Exact text to replace, including indentation. Must be unique in the file unless replace_all is set."},
    "new_string": {"type": "string", "description": "Text to replace it with."},
    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."}
  },
  "required": ["path", "old_string", "new_string"]
}`)
}

type editArgs struct {
	Path       string `json:"path"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all"`
}

func (t *EditTool) Run(ctx context.Context, st *State, raw json.RawMessage) Result {
	var a editArgs
	if r := decodeArgs(raw, &a); r != nil {
		return *r
	}
	path, err := resolve(st, a.Path)
	if err != nil {
		return Errorf("%v", err)
	}
	if a.OldString == a.NewString {
		return Errorf("old_string and new_string are identical, so this edit would do nothing.")
	}
	if a.OldString == "" {
		return Errorf("old_string is empty. To create a file or replace it entirely, use the write tool.")
	}

	content, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Errorf("%s does not exist. Use the write tool to create it.%s",
				rel(st, path), suggestNeighbours(path))
		}
		return Errorf("could not read %s: %v", rel(st, path), err)
	}
	if r := checkStale(st, path, content); r != nil {
		return *r
	}

	text := string(content)
	count := strings.Count(text, a.OldString)

	switch {
	case count == 0:
		return diagnoseNoMatch(st, path, text, a.OldString)
	case count > 1 && !a.ReplaceAll:
		return diagnoseAmbiguous(st, path, text, a.OldString, count)
	}

	var updated string
	if a.ReplaceAll {
		updated = strings.ReplaceAll(text, a.OldString, a.NewString)
	} else {
		updated = strings.Replace(text, a.OldString, a.NewString, 1)
	}

	if err := writeFilePreservingMode(path, []byte(updated)); err != nil {
		return Errorf("could not write %s: %v", rel(st, path), err)
	}
	st.SetStamp(stampFile(path, []byte(updated)))

	added, removed := countChangedLines(a.OldString, a.NewString)
	where := ""
	if count == 1 {
		where = fmt.Sprintf(" at line %d", lineOf(text, strings.Index(text, a.OldString)))
	}

	occurrences := ""
	if a.ReplaceAll && count > 1 {
		occurrences = fmt.Sprintf(" (%d occurrences)", count)
	}

	return Result{
		Content: fmt.Sprintf("Edited %s%s%s. +%d -%d lines.\n\n%s",
			rel(st, path), where, occurrences, added, removed,
			contextAround(updated, a.NewString)),
		Display: fmt.Sprintf("Edited %s (+%d -%d)", rel(st, path), added, removed),
		Files:   []FileStamp{stampFile(path, []byte(updated))},
	}
}

func writeFilePreservingMode(path string, content []byte) error {
	mode := os.FileMode(0o644)
	if fi, err := os.Stat(path); err == nil {
		mode = fi.Mode().Perm()
	}
	return os.WriteFile(path, content, mode)
}

// ---------------------------------------------------------------------------
// Failure diagnosis
//
// These messages are the whole game. The model cannot see the file; when an
// edit fails, this text is the only information it gets about why. A bare "not
// found" produces a retry of the identical wrong string. Naming the actual
// difference produces a correct retry on the next attempt.
// ---------------------------------------------------------------------------

func diagnoseNoMatch(st *State, path, text, old string) Result {
	name := rel(st, path)

	// By far the most common cause: the text is there, but the indentation
	// differs -- tabs retyped as spaces, or a different nesting depth.
	if line, actual, ok := findWhitespaceVariant(text, old); ok {
		return Errorf(
			"old_string was not found in %s, but a match differing only in whitespace exists at line %d.\n\n"+
				"You supplied:\n%s\n\nThe file contains:\n%s\n\n"+
				"(%s shown as %s, trailing space as %s)\n\n"+
				"Reissue the edit using the file's exact text.",
			name, line, indentBlock(visible(old)), indentBlock(visible(actual)),
			"tab", "→", "·")
	}

	// Next most common: the anchor line exists but the surrounding lines the
	// model included do not match.
	if line, ok := findAnchorLine(text, old); ok {
		return Errorf(
			"old_string was not found in %s.\n\n"+
				"Its first line does appear, at line %d, but the lines around it differ "+
				"from what you supplied. Here is what the file actually has:\n\n%s\n\n"+
				"Reissue the edit using this text.",
			name, line, indentBlock(excerpt(text, line, countLinesStr(old)+2)))
	}

	// Nothing recognisable. Say so plainly and point at the recovery step
	// rather than leaving the model to guess.
	return Errorf(
		"old_string was not found in %s, and no near match was located.\n\n"+
			"You supplied:\n%s\n\n"+
			"The file has %d lines. Read it again and copy the exact text you intend to "+
			"replace -- do not retype it.",
		name, indentBlock(visible(old)), countLinesStr(text))
}

func diagnoseAmbiguous(st *State, path, text, old string, count int) Result {
	var lines []int
	for i, idx := 0, 0; i < count && idx <= len(text); i++ {
		off := strings.Index(text[idx:], old)
		if off < 0 {
			break
		}
		lines = append(lines, lineOf(text, idx+off))
		idx += off + len(old)
	}

	var b strings.Builder
	for _, ln := range lines {
		fmt.Fprintf(&b, "  line %d:  %s\n", ln, strings.TrimSpace(nthLine(text, ln)))
	}

	return Errorf(
		"old_string appears %d times in %s, so the edit is ambiguous and was NOT applied.\n\n"+
			"Occurrences:\n%s\n"+
			"Either extend old_string with enough surrounding lines to identify one "+
			"occurrence uniquely, or set replace_all to change all %d.",
		count, rel(st, path), b.String(), count)
}

// findWhitespaceVariant looks for the supplied text with leading and trailing
// whitespace on each line normalised away. Returns the line number and the
// file's actual text for that region.
func findWhitespaceVariant(text, old string) (line int, actual string, ok bool) {
	textLines := strings.Split(text, "\n")
	oldLines := strings.Split(strings.TrimSuffix(old, "\n"), "\n")
	if len(oldLines) == 0 {
		return 0, "", false
	}

	norm := func(s string) string { return strings.TrimSpace(s) }
	target := make([]string, len(oldLines))
	for i, l := range oldLines {
		target[i] = norm(l)
	}

	for i := 0; i+len(target) <= len(textLines); i++ {
		match := true
		for j := range target {
			if norm(textLines[i+j]) != target[j] {
				match = false
				break
			}
		}
		if match {
			return i + 1, strings.Join(textLines[i:i+len(target)], "\n"), true
		}
	}
	return 0, "", false
}

// findAnchorLine locates the first non-blank line of old_string in the file,
// when it occurs exactly once. A unique anchor tells the model where it was
// aiming even though the block as a whole did not match.
func findAnchorLine(text, old string) (int, bool) {
	var anchor string
	for _, l := range strings.Split(old, "\n") {
		if strings.TrimSpace(l) != "" {
			anchor = strings.TrimSpace(l)
			break
		}
	}
	if anchor == "" || len(anchor) < 4 {
		return 0, false
	}
	textLines := strings.Split(text, "\n")
	found, at := 0, 0
	for i, l := range textLines {
		if strings.TrimSpace(l) == anchor {
			found++
			at = i + 1
		}
	}
	if found == 1 {
		return at, true
	}
	return 0, false
}

// visible renders whitespace the model would otherwise be unable to see.
func visible(s string) string {
	var b strings.Builder
	for _, line := range strings.Split(s, "\n") {
		line = strings.ReplaceAll(line, "\t", "→")
		trimmed := strings.TrimRight(line, " ")
		b.WriteString(trimmed)
		b.WriteString(strings.Repeat("·", len(line)-len(trimmed)))
		b.WriteByte('\n')
	}
	return strings.TrimSuffix(b.String(), "\n")
}

func indentBlock(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = "    " + l
	}
	return strings.Join(lines, "\n")
}

func excerpt(text string, startLine, n int) string {
	lines := strings.Split(text, "\n")
	from := max(startLine-1, 0)
	to := min(from+n, len(lines))
	var b strings.Builder
	for i := from; i < to; i++ {
		fmt.Fprintf(&b, "%d\t%s\n", i+1, lines[i])
	}
	return strings.TrimSuffix(b.String(), "\n")
}

// contextAround shows the edited region so the model can confirm the result
// without spending another read call.
func contextAround(text, needle string) string {
	idx := strings.Index(text, needle)
	if idx < 0 {
		return ""
	}
	start := lineOf(text, idx)
	n := countLinesStr(needle) + 4
	return excerpt(text, max(start-2, 1), n)
}

func lineOf(text string, offset int) int {
	if offset < 0 {
		return 0
	}
	return strings.Count(text[:offset], "\n") + 1
}

func nthLine(text string, n int) string {
	lines := strings.Split(text, "\n")
	if n-1 < 0 || n-1 >= len(lines) {
		return ""
	}
	return lines[n-1]
}

func countLinesStr(s string) int {
	if s == "" {
		return 0
	}
	return strings.Count(strings.TrimSuffix(s, "\n"), "\n") + 1
}

func countChangedLines(old, new string) (added, removed int) {
	return countLinesStr(new), countLinesStr(old)
}
