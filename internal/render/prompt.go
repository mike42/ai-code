package render

import (
	"fmt"
	"strings"
)

// echoMaxLines caps how much of a long input is echoed. A pasted file is a
// perfectly ordinary prompt and reproducing all of it would push the rest of
// the session out of view for no gain -- the full text is in the transcript
// either way.
const echoMaxLines = 40

// echoMaxColumns is the same cap for input that arrives as one enormous line,
// where counting lines measures nothing. A paste with the newlines stripped out
// of it is common enough -- a log line, a URL, a stack trace joined up by
// whatever it was copied from -- and reproducing all of it costs as many rows
// as a long multi-line prompt does. Twenty rows of an eighty-column terminal.
const echoMaxColumns = 1600

// BoundLine caps a single line of submitted input for display.
//
// The cut is taken from the middle because the ends are what identify a
// prompt: the opening words say what was asked, and the closing ones carry the
// path or the constraint that was added last. Head-truncation loses the first,
// tail-truncation the second, and either leaves two long prompts looking
// identical in the scrollback.
func BoundLine(text string) string {
	if visibleWidth(text) <= echoMaxColumns {
		return text
	}
	rs := []rune(text)
	head := columnPrefix(rs, echoMaxColumns*2/3)
	tail := columnSuffix(rs, echoMaxColumns-echoMaxColumns*2/3)
	if tail < head {
		tail = head
	}
	return fmt.Sprintf("%s …[%s omitted]… %s",
		string(rs[:head]), pluralise(tail-head, "character"), string(rs[tail:]))
}

// columnPrefix returns the number of leading runes that fit in n columns.
func columnPrefix(rs []rune, n int) int {
	w := 0
	for i, r := range rs {
		if w += runeWidth(r); w > n {
			return i
		}
	}
	return len(rs)
}

// columnSuffix returns the index at which the trailing n columns begin.
func columnSuffix(rs []rune, n int) int {
	w := 0
	for i := len(rs) - 1; i >= 0; i-- {
		if w += runeWidth(rs[i]); w > n {
			return i + 1
		}
	}
	return 0
}

// Echo frames submitted input for the scrollback.
//
// Two cases need it, for the same reason: the text was never drawn where it
// will stay. A multi-line prompt is summarised by the line editor rather than
// rendered, and Ctrl-G and /editor compose it somewhere else entirely, so at
// the moment it is sent the terminal has never shown what was actually said. A
// steering message typed during a turn was only ever in the transient zone,
// which is erased by the next thing committed. Either way the session reads as
// an answer to a question nobody can see.
//
// Multi-line text is framed with rules above and below rather than with a full
// box, because a box means a vertical gutter character at the start of every
// line, and a gutter is picked up by selection. Anything ai-code draws around text
// the user may want to copy back out has to stay out of the columns that text
// occupies.
func Echo(s Style, label, text string) []string {
	text = strings.TrimRight(text, "\n")
	if !IsMultiline(text) {
		return []string{"", s.Dim(label+":") + " " + BoundLine(text), ""}
	}

	body := strings.Split(text, "\n")
	truncated := 0
	if len(body) > echoMaxLines {
		truncated = len(body) - echoMaxLines
		body = body[:echoMaxLines]
	}

	head := fmt.Sprintf("%s · %s", label, pluralise(len(body)+truncated, "line"))
	out := []string{"", s.Dim(rule(head, 60))}
	for _, l := range body {
		out = append(out, "  "+l)
	}
	if truncated > 0 {
		out = append(out, "  "+s.Dim(fmt.Sprintf("... %s not shown", pluralise(truncated, "more line"))))
	}
	return append(out, s.Dim(rule("", 60)), "")
}

// rule draws a horizontal rule with an optional label set into it.
func rule(label string, width int) string {
	if label == "" {
		return strings.Repeat("─", width)
	}
	head := "── " + label + " "
	if n := width - visibleWidth(head); n > 0 {
		return head + strings.Repeat("─", n)
	}
	return head
}

func pluralise(n int, noun string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, noun)
	}
	return fmt.Sprintf("%d %ss", n, noun)
}

// IsMultiline reports whether text spans more than one line. A single-line
// prompt needs no echo: the line editor completes it in place on submit and it
// stays there, so echoing would print it twice.
func IsMultiline(text string) bool { return strings.Contains(strings.TrimSpace(text), "\n") }
