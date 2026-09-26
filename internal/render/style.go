package render

import "strings"

// Style holds the ANSI palette: dim for anything secondary, one accent for tool
// activity, otherwise the terminal's default foreground. Restraint keeps
// streaming output readable.
type Style struct {
	enabled bool
}

func NewStyle(enabled bool) Style { return Style{enabled: enabled} }

const (
	codeDim     = esc + "[2m"
	codeBold    = esc + "[1m"
	codeItalic  = esc + "[3m"
	codeCyan    = esc + "[36m"
	codeGreen   = esc + "[32m"
	codeYellow  = esc + "[33m"
	codeRed     = esc + "[31m"
	codeMagenta = esc + "[35m"
)

func (s Style) wrap(code, text string) string {
	if !s.enabled || text == "" {
		return text
	}
	return code + text + sgrReset
}

func (s Style) Dim(t string) string     { return s.wrap(codeDim, t) }
func (s Style) Bold(t string) string    { return s.wrap(codeBold, t) }
func (s Style) Italic(t string) string  { return s.wrap(codeItalic, t) }
func (s Style) Heading(t string) string { return s.wrap(codeBold, t) }
func (s Style) Code(t string) string    { return s.wrap(codeCyan, t) }
func (s Style) Bullet(t string) string  { return s.wrap(codeDim, t) }
func (s Style) Quote(t string) string   { return s.wrap(codeDim, t) }
func (s Style) Tool(t string) string    { return s.wrap(codeCyan, t) }
func (s Style) Ok(t string) string      { return s.wrap(codeGreen, t) }
func (s Style) Warn(t string) string    { return s.wrap(codeYellow, t) }
func (s Style) Error(t string) string   { return s.wrap(codeRed, t) }
func (s Style) Accent(t string) string  { return s.wrap(codeMagenta, t) }

// Strip removes every escape sequence. Used by the golden tests that assert on
// what a selection would actually contain.
func Strip(s string) string {
	var b strings.Builder
	inEsc := false
	for _, r := range s {
		if inEsc {
			if r == 'm' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
				inEsc = false
			}
			continue
		}
		if r == 0x1b {
			inEsc = true
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// HasOpenSGR reports whether a line ends with an unterminated colour or
// attribute run: an escape left open bleeds into whatever is printed or pasted
// next. A line ending with a reset is not the same test.
func HasOpenSGR(s string) bool {
	open := false
	for i := 0; i < len(s); i++ {
		if s[i] != 0x1b || i+1 >= len(s) || s[i+1] != '[' {
			continue
		}
		j := i + 2
		for j < len(s) && (s[j] == ';' || (s[j] >= '0' && s[j] <= '9')) {
			j++
		}
		if j < len(s) && s[j] == 'm' {
			params := s[i+2 : j]
			open = !(params == "" || params == "0")
		}
		i = j
	}
	return open
}

// VisibleWidth counts the columns a string occupies, ignoring escapes.
func VisibleWidth(s string) int { return visibleWidth(s) }
