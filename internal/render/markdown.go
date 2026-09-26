package render

import (
	"strings"
	"unicode/utf8"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
)

// Markdown renders streamed markdown one committed line at a time. Block
// structure is decided per line, so a finished line is styled once and never
// revised; the line still arriving is previewed in the transient zone.
type Markdown struct {
	// commit receives finished lines; partial the line still arriving.
	// Decoupling them from Screen keeps the renderer testable and lets one
	// owner compose the transient zone from the partial line and the status line.
	commit  func(string)
	partial func([]string)

	style Style
	color bool
	width func() int

	line strings.Builder

	inFence   bool
	fenceMark string
	fenceLang string
	fenceBody []string

	// tableBuf holds a table's rows until its extent is known; tableRaw means
	// the buffer was given up on and the rest of this table streams unaligned.
	tableBuf []string
	tableRaw bool

	chromaStyle *chroma.Style
	// lastCommitBlank records whether the last line committed was blank, so
	// runs of blank lines collapse.
	lastCommitBlank bool
}

func NewMarkdown(commit func(string), partial func([]string), style Style, color bool, theme string, width func() int) *Markdown {
	if width == nil {
		width = func() int { return 80 }
	}
	m := &Markdown{
		commit: commit, partial: partial,
		style: style, color: color, width: width,
		lastCommitBlank: true,
	}
	if color {
		m.chromaStyle = resolveChromaStyle(theme)
	}
	return m
}

// resolveChromaStyle picks a syntax theme. "auto" avoids a background colour,
// which looks wrong in every terminal but the one it was designed for.
func resolveChromaStyle(name string) *chroma.Style {
	switch name {
	case "", "auto":
		if s := styles.Get("average"); s != nil {
			return s
		}
		return styles.Fallback
	case "none", "off":
		return nil
	default:
		if s := styles.Get(name); s != nil {
			return s
		}
		return styles.Fallback
	}
}

// Write feeds a chunk of streamed markdown.
func (m *Markdown) Write(text string) {
	for {
		i := strings.IndexByte(text, '\n')
		if i < 0 {
			m.line.WriteString(text)
			break
		}
		m.line.WriteString(text[:i])
		m.commitLine(m.line.String())
		m.line.Reset()
		text = text[i+1:]
	}
	m.preview()
}

// preview shows the in-progress line in the transient zone, styled and wrapped
// as it will appear. A wrong guess costs one frame here, not a corrupted
// scrollback line, and code is previewed verbatim.
func (m *Markdown) preview() {
	if m.partial == nil {
		return
	}
	line := m.line.String()
	if len(m.tableBuf) > 0 {
		// Buffered rows exist nowhere else yet, so the transient zone stands
		// in for the scrollback until the table can be laid out, shown as the
		// model wrote them: widths are not known until the last row arrives.
		rows := m.tableBuf
		if len(rows) > tablePreviewRows {
			rows = rows[len(rows)-tablePreviewRows:]
		}
		rows = append([]string(nil), rows...)
		if line != "" {
			rows = append(rows, line)
		}
		m.partial(rows)
		return
	}
	if line == "" {
		m.partial(nil)
		return
	}
	if m.inFence || m.tableRaw || isTableRow(line) {
		m.partial([]string{line})
		return
	}
	if _, _, ok := fenceDelimiter(strings.TrimRight(line, " \t")); ok {
		// The delimiter is never committed, so it is never previewed either.
		m.partial(nil)
		return
	}
	styled, cont := m.speculate(line)
	m.partial(wrapStyled(styled, cont, m.proseWidth()))
}

// proseWidth is one column short of the terminal: the preview truncates every
// transient row to width-1 to keep the erase count exact, and wrapping both
// paths to the same budget keeps the committed line identical to the preview.
func (m *Markdown) proseWidth() int {
	return max(m.width()-1, 1)
}

// Flush commits whatever is left when the stream ends.
func (m *Markdown) Flush() {
	if m.line.Len() > 0 {
		m.commitLine(m.line.String())
		m.line.Reset()
	}
	if m.inFence {
		// The model ended mid-fence; close it rather than leaving the renderer
		// in code mode for whatever comes next.
		m.inFence = false
		m.fenceBody = nil
	}
	m.flushTable()
	if m.partial != nil {
		m.partial(nil)
	}
}

func (m *Markdown) commitLine(line string) {
	trimmed := strings.TrimRight(line, " \t")

	if !m.inFence {
		if isTableRow(trimmed) {
			m.bufferTableRow(trimmed)
			return
		}
		// Whatever ends the table is committed after it: the rows it ends are
		// still sitting in the buffer.
		m.flushTable()
	}

	if fence, lang, ok := fenceDelimiter(trimmed); ok {
		if !m.inFence {
			m.inFence, m.fenceMark, m.fenceLang = true, fence, lang
			m.fenceBody = m.fenceBody[:0]
			return // The delimiter itself is not shown; the styling conveys it.
		}
		if strings.HasPrefix(trimmed, m.fenceMark) {
			m.inFence = false
			m.fenceBody = nil
			return
		}
	}

	if m.inFence {
		m.commitCodeLine(trimmed)
		return
	}

	blank := strings.TrimSpace(trimmed) == ""
	if blank && m.lastCommitBlank {
		// Collapse runs of blank lines. Models emit them liberally and they
		// cost real vertical space in an inline renderer.
		return
	}
	m.lastCommitBlank = blank

	styled, cont := m.styleProse(trimmed)
	for _, l := range wrapStyled(styled, cont, m.proseWidth()) {
		m.commit(l)
	}
}

// commitCodeLine highlights one line of a fenced block. The whole block is
// re-lexed each time, so multi-line constructs highlight correctly, and only
// the finished line is emitted, which keeps output append-only.
func (m *Markdown) commitCodeLine(line string) {
	m.fenceBody = append(m.fenceBody, line)
	m.lastCommitBlank = false

	if m.chromaStyle == nil || len(m.fenceBody) > 2000 {
		// Very large blocks fall back to plain text rather than re-lexing once
		// per line.
		m.commit(line)
		return
	}

	styled, ok := m.highlightLast()
	if !ok {
		m.commit(line)
		return
	}
	m.commit(styled)
}

func (m *Markdown) highlightLast() (string, bool) {
	lexer := lexers.Get(m.fenceLang)
	if lexer == nil {
		lexer = lexers.Analyse(strings.Join(m.fenceBody, "\n"))
	}
	if lexer == nil {
		return "", false
	}

	source := strings.Join(m.fenceBody, "\n")
	iter, err := lexer.Tokenise(nil, source)
	if err != nil {
		return "", false
	}

	// Split the token stream into lines.
	lineTokens := [][]chroma.Token{{}}
	for _, tok := range iter.Tokens() {
		parts := strings.Split(tok.Value, "\n")
		for i, p := range parts {
			if i > 0 {
				lineTokens = append(lineTokens, []chroma.Token{})
			}
			if p != "" {
				last := len(lineTokens) - 1
				lineTokens[last] = append(lineTokens[last], chroma.Token{Type: tok.Type, Value: p})
			}
		}
	}

	target := len(m.fenceBody) - 1
	if target >= len(lineTokens) {
		return "", false
	}

	formatter := formatters.Get("terminal256")
	if formatter == nil {
		return "", false
	}
	var b strings.Builder
	if err := formatter.Format(&b, m.chromaStyle, chroma.Literator(lineTokens[target]...)); err != nil {
		return "", false
	}
	return strings.TrimRight(b.String(), "\n"), true
}

// styleProse styles one complete line and reports the indent a wrapped
// continuation carries, so text stays aligned under its bullet, number or quote
// rather than resetting to column zero.
func (m *Markdown) styleProse(line string) (styled, contIndent string) {
	return m.styleProseWith(line, m.inline)
}

// speculate renders a line that is still arriving as though it were complete:
// an open span is styled through to the end, and a trailing fragment that may
// still grow is held back, so a delimiter is never shown and then taken away.
func (m *Markdown) speculate(line string) (styled, contIndent string) {
	if !m.color {
		return m.styleProse(line)
	}
	return m.styleProseWith(holdBack(line), m.inlineOpen)
}

// holdBack drops a trailing fragment that would appear for one frame and then
// vanish: a lone "*" on its way to "**", a closing delimiter, or the "#" of a
// heading that has not reached its space yet.
func holdBack(line string) string {
	if t := strings.TrimLeft(line, " \t"); t != "" && strings.Trim(t, "#") == "" {
		return ""
	}
	return strings.TrimRight(line, "*`")
}

func (m *Markdown) styleProseWith(line string, inline func(string) string) (styled, contIndent string) {
	trimmedLeft := strings.TrimLeft(line, " \t")
	indent := line[:len(line)-len(trimmedLeft)]

	cont := indent
	switch {
	case strings.HasPrefix(trimmedLeft, "> "),
		strings.HasPrefix(trimmedLeft, "- "), strings.HasPrefix(trimmedLeft, "* "),
		strings.HasPrefix(trimmedLeft, "+ "):
		cont = indent + "  "
	}
	if marker, _, ok := orderedListItem(trimmedLeft); ok {
		cont = indent + strings.Repeat(" ", len(marker)+1)
	}

	if !m.color {
		return line, cont
	}

	switch {
	case strings.HasPrefix(trimmedLeft, "#"):
		level := 0
		for level < len(trimmedLeft) && trimmedLeft[level] == '#' {
			level++
		}
		if level <= 6 && level < len(trimmedLeft) && trimmedLeft[level] == ' ' {
			return indent + m.style.Heading(strings.TrimSpace(trimmedLeft[level:])), cont
		}

	case strings.HasPrefix(trimmedLeft, "> "):
		return indent + m.style.Quote(trimmedLeft), cont

	case isHorizontalRule(trimmedLeft):
		return m.style.Dim(strings.Repeat("─", min(m.width()-1, 60))), cont

	case strings.HasPrefix(trimmedLeft, "- "), strings.HasPrefix(trimmedLeft, "* "),
		strings.HasPrefix(trimmedLeft, "+ "):
		return indent + m.style.Bullet("•") + " " + inline(trimmedLeft[2:]), cont
	}

	if marker, rest, ok := orderedListItem(trimmedLeft); ok {
		return indent + m.style.Bullet(marker) + " " + inline(rest), cont
	}

	return indent + inline(trimmedLeft), cont
}

// inline styles spans within a line: code and bold.
func (m *Markdown) inline(s string) string {
	s = m.spanReplace(s, "`", m.style.Code)
	s = m.spanReplace(s, "**", m.style.Bold)
	return s
}

// spanReplace styles paired delimiters, leaving an unmatched delimiter alone:
// an unmatched backtick is common in prose about shell commands and must not
// swallow the rest of the line.
func (m *Markdown) spanReplace(s, delim string, style func(string) string) string {
	if !strings.Contains(s, delim) {
		return s
	}
	parts := strings.Split(s, delim)
	if len(parts) < 3 {
		return s
	}
	var b strings.Builder
	for i, p := range parts {
		switch {
		case i%2 == 0:
			b.WriteString(p)
		case i == len(parts)-1:
			// Trailing unmatched delimiter: restore it literally.
			b.WriteString(delim)
			b.WriteString(p)
		default:
			b.WriteString(style(p))
		}
	}
	return b.String()
}

func fenceDelimiter(line string) (mark, lang string, ok bool) {
	trimmed := strings.TrimLeft(line, " ")
	for _, d := range []string{"```", "~~~"} {
		if strings.HasPrefix(trimmed, d) {
			return d, strings.TrimSpace(strings.TrimPrefix(trimmed, d)), true
		}
	}
	return "", "", false
}

func isHorizontalRule(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) < 3 {
		return false
	}
	for _, d := range []byte{'-', '*', '_'} {
		if strings.Trim(s, string(d)) == "" {
			return true
		}
	}
	return false
}

func orderedListItem(s string) (marker, rest string, ok bool) {
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 || i+1 >= len(s) {
		return "", "", false
	}
	if (s[i] == '.' || s[i] == ')') && s[i+1] == ' ' {
		return s[:i+1], s[i+2:], true
	}
	return "", "", false
}

// cell is one unit of a styled line: a visible rune, or a zero-width escape
// sequence that must travel with the text around it.
type cell struct {
	text  string
	width int
	space bool
}

func splitCells(s string) []cell {
	cells := make([]cell, 0, len(s))
	for i := 0; i < len(s); {
		if s[i] == 0x1b {
			j := i + 1
			for j < len(s) {
				c := s[j]
				j++
				if c == 'm' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
					break
				}
			}
			cells = append(cells, cell{text: s[i:j]})
			i = j
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		cells = append(cells, cell{text: s[i : i+size], width: 1, space: r == ' '})
		i += size
	}
	return cells
}

// wrapStyled breaks an already-styled prose line at word boundaries, returning
// the lines to commit. A break inside a span closes the run and reopens it, so
// no escape bleeds into what follows; code is never wrapped.
func wrapStyled(line, contIndent string, width int) []string {
	if width <= 0 {
		return []string{line}
	}
	cells := splitCells(line)
	visible := 0
	for _, c := range cells {
		visible += c.width
	}
	if visible <= width {
		return []string{line}
	}

	// openAt[i] is the escape sequence in force immediately before cell i.
	openAt := make([]string, len(cells)+1)
	open := ""
	for i, c := range cells {
		openAt[i] = open
		if c.width == 0 {
			if c.text == sgrReset {
				open = ""
			} else {
				open = c.text
			}
		}
	}
	openAt[len(cells)] = open

	var out []string
	start, first := 0, true
	for start < len(cells) {
		prefix := ""
		if !first {
			prefix = contIndent
		}
		avail := max(width-len([]rune(prefix)), 1)

		col, lastSpace, overflow := 0, -1, -1
		for i := start; i < len(cells); i++ {
			if cells[i].width == 0 {
				continue
			}
			if col+cells[i].width > avail {
				overflow = i
				break
			}
			col += cells[i].width
			if cells[i].space {
				lastSpace = i
			}
		}

		if overflow < 0 {
			out = append(out, prefix+openAt[start]+join(cells[start:]))
			break
		}

		// Break at the last space that fits. Without one the word is longer
		// than the terminal is wide, so it has to be cut somewhere.
		cut := overflow
		if lastSpace > start {
			cut = lastSpace
		}
		seg := prefix + openAt[start] + join(cells[start:cut])
		if openAt[cut] != "" {
			seg += sgrReset
		}
		out = append(out, seg)

		start = cut
		for start < len(cells) && cells[start].space {
			start++
		}
		first = false
	}
	return out
}

func join(cells []cell) string {
	var b strings.Builder
	for _, c := range cells {
		b.WriteString(c.text)
	}
	return b.String()
}

// inlineOpen is inline for a line that is still arriving: a span whose closing
// delimiter has not turned up is styled on the assumption that it will. When it
// does not, the committed line restores the literal delimiter.
func (m *Markdown) inlineOpen(s string) string {
	s = m.spanReplaceOpen(s, "`", m.style.Code)
	s = m.spanReplaceOpen(s, "**", m.style.Bold)
	return s
}

func (m *Markdown) spanReplaceOpen(s, delim string, style func(string) string) string {
	if !strings.Contains(s, delim) {
		return s
	}
	var b strings.Builder
	for i, p := range strings.Split(s, delim) {
		if i%2 == 0 {
			b.WriteString(p)
			continue
		}
		b.WriteString(style(p))
	}
	return b.String()
}

// tableMaxRows bounds the buffer. Past it rows are committed unaligned, so a
// pathological input degrades rather than growing without limit.
const tableMaxRows = 2000

// tablePreviewRows bounds how much of the buffer the transient zone echoes: the
// zone is repainted on every token and erased by counted cursor moves, so only
// the tail is worth showing.
const tablePreviewRows = 40

// tableMinCol is the narrowest a column may be squeezed to before the table is
// given up on and committed raw: a cell of two characters and an ellipsis
// destroys information the reader cannot recover.
const tableMinCol = 8

// tableAlign is the delimiter row's marker for one column: ":---", "---:" or
// ":---:". The default is left.
type tableAlign struct{ left, right bool }

// bufferTableRow holds a row until the table's extent is known, since a
// column's width depends on every row.
func (m *Markdown) bufferTableRow(row string) {
	if m.tableRaw {
		m.commitTableRowRaw(row)
		return
	}
	m.tableBuf = append(m.tableBuf, row)
	if len(m.tableBuf) <= tableMaxRows {
		return
	}
	rows := m.tableBuf
	m.tableBuf, m.tableRaw = nil, true
	for _, r := range rows {
		m.commitTableRowRaw(r)
	}
}

// commitTableRowRaw emits a row whole when the table cannot be aligned. A table
// row is structure, not prose: breaking it at a space moves cells onto their own
// lines and the columns stop meaning anything.
func (m *Markdown) commitTableRowRaw(row string) {
	styled, _ := m.styleProse(row)
	m.commit(styled)
	m.lastCommitBlank = false
}

func (m *Markdown) flushTable() {
	m.tableRaw = false
	if len(m.tableBuf) == 0 {
		return
	}
	rows := m.tableBuf
	// Cleared before committing: commit re-enters through the owner of the
	// screen, and a buffered table would be emitted a second time.
	m.tableBuf = nil
	for _, l := range m.renderTable(rows) {
		m.commit(l)
	}
	m.lastCommitBlank = false
}

func (m *Markdown) renderTable(rows []string) []string {
	aligns, ok := tableAlignments(rows)
	if !ok {
		// No delimiter row, so this is a run of lines that merely start with a
		// pipe. There are no columns to line up.
		return m.rawTable(rows)
	}

	cols := len(aligns)
	grid := make([][]string, 0, len(rows)-1)
	for i, r := range rows {
		if i == 1 {
			continue // The delimiter is redrawn from the widths it cannot know.
		}
		cells := splitTableCells(r)
		for j := range cells {
			cells[j] = m.styleCell(cells[j])
		}
		cols = max(cols, len(cells))
		grid = append(grid, cells)
	}
	if cols == 0 {
		return m.rawTable(rows)
	}
	for len(aligns) < cols {
		// A row with more cells than the delimiter declared gets a column: the
		// extra is content, not something to drop.
		aligns = append(aligns, tableAlign{})
	}

	widths := make([]int, cols)
	for _, cells := range grid {
		for j, c := range cells {
			// Measured on the styled text: visibleWidth skips the escapes,
			// and counting the markers as content would drift the columns.
			widths[j] = max(widths[j], visibleWidth(c))
		}
	}
	for j := range widths {
		widths[j] = max(widths[j], 1)
	}

	indent := rows[0][:len(rows[0])-len(strings.TrimLeft(rows[0], " "))]
	widths, ok = fitTable(widths, m.proseWidth()-len(indent))
	if !ok {
		return m.rawTable(rows)
	}

	out := make([]string, 0, len(grid)+3)
	out = append(out, indent+tableRule(widths, tableTop))
	for i, cells := range grid {
		out = append(out, indent+tableRow(cells, widths, aligns))
		if i == 0 {
			out = append(out, indent+tableRule(widths, tableMiddle))
		}
	}
	out = append(out, indent+tableRule(widths, tableBottom))
	return out
}

func (m *Markdown) rawTable(rows []string) []string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		styled, _ := m.styleProse(r)
		out = append(out, styled)
	}
	return out
}

// styleCell applies inline markdown to a cell. With colour off the source is
// left alone, as styleProseWith does with a whole line, so the markers still
// convey the emphasis.
func (m *Markdown) styleCell(s string) string {
	if !m.color {
		return s
	}
	return m.inline(s)
}

// fitTable decides the final column widths, reporting false when the table
// should not be aligned: every line has to fit one row, and a terminal-wrapped
// table is scrambled far worse than an unaligned one.
func fitTable(nat []int, avail int) ([]int, bool) {
	overhead := 1 + 3*len(nat)
	total, floor := overhead, overhead
	for _, w := range nat {
		total += w
		floor += min(w, tableMinCol)
	}
	if total <= avail {
		return nat, true
	}
	if floor > avail {
		return nil, false
	}

	out := fitColumns(nat, avail-overhead)
	got := overhead
	for _, w := range out {
		got += w
	}
	if got > avail {
		// Arithmetic guard: an over-budget layout must never reach the
		// terminal, and raw output is always safe.
		return nil, false
	}
	return out, true
}

// fitColumns shares a budget out between columns: a column narrower than its
// fair share is paid in full and its change handed back to the rest, so short
// columns do not sit wide while long ones are cut to pieces.
func fitColumns(nat []int, budget int) []int {
	out := make([]int, len(nat))
	done := make([]bool, len(nat))
	remaining, left := budget, len(nat)

	for left > 0 {
		fair := remaining / left
		progress := false
		for i, w := range nat {
			if done[i] || w > fair {
				continue
			}
			out[i], done[i], progress = w, true, true
			remaining -= w
			left--
		}
		if progress {
			continue
		}
		for i := range nat {
			if done[i] {
				continue
			}
			out[i] = max(fair, min(nat[i], tableMinCol))
			remaining -= out[i]
			left--
		}
	}
	return out
}

func tableRow(cells []string, widths []int, aligns []tableAlign) string {
	var b strings.Builder
	b.WriteString(tableVertical)
	for i, w := range widths {
		cell := ""
		if i < len(cells) {
			cell = cells[i]
		}
		b.WriteByte(' ')
		b.WriteString(padCell(cell, w, aligns[i]))
		b.WriteByte(' ')
		b.WriteString(tableVertical)
	}
	return b.String()
}

func padCell(s string, w int, a tableAlign) string {
	n := visibleWidth(s)
	if n > w {
		s = truncateCell(s, w)
		n = visibleWidth(s)
	}
	gap := w - n
	switch {
	case a.left && a.right:
		l := gap / 2
		return strings.Repeat(" ", l) + s + strings.Repeat(" ", gap-l)
	case a.right:
		return strings.Repeat(" ", gap) + s
	}
	return s + strings.Repeat(" ", gap)
}

func truncateCell(s string, w int) string {
	if w <= 1 {
		return truncateVisible(s, w)
	}
	return truncateVisible(s, w-1) + "…"
}

// Every glyph is one column wide, so the width arithmetic above is unchanged.
const (
	tableVertical   = "\u2502"
	tableHorizontal = "\u2500"
)

// tableRuleStyle is one of the three horizontal rules, by its corners.
type tableRuleStyle struct{ left, join, right string }

var (
	tableTop    = tableRuleStyle{"\u250c", "\u252c", "\u2510"}
	tableMiddle = tableRuleStyle{"\u251c", "\u253c", "\u2524"}
	tableBottom = tableRuleStyle{"\u2514", "\u2534", "\u2518"}
)

// tableRule draws a horizontal rule in box drawing rather than the markdown the
// model wrote: the pipes and dashes are markup, not content. Alignment is shown
// by the cell padding, not by colons.
func tableRule(widths []int, st tableRuleStyle) string {
	var b strings.Builder
	b.WriteString(st.left)
	for i, w := range widths {
		if i > 0 {
			b.WriteString(st.join)
		}
		b.WriteString(strings.Repeat(tableHorizontal, w+2))
	}
	b.WriteString(st.right)
	return b.String()
}

// tableAlignments reads the delimiter row a table must have as its second line.
func tableAlignments(rows []string) ([]tableAlign, bool) {
	if len(rows) < 2 {
		return nil, false
	}
	cells := splitTableCells(rows[1])
	if len(cells) == 0 {
		return nil, false
	}
	aligns := make([]tableAlign, len(cells))
	for i, c := range cells {
		body := strings.TrimSuffix(strings.TrimPrefix(c, ":"), ":")
		if body == "" || strings.Trim(body, "-") != "" {
			return nil, false
		}
		aligns[i] = tableAlign{
			left:  strings.HasPrefix(c, ":"),
			right: strings.HasSuffix(c, ":"),
		}
	}
	return aligns, true
}

// splitTableCells breaks a row on its unescaped pipes. The outer pair is the
// table's border rather than empty cells, but only when present: models drop the
// trailing pipe often enough that requiring it would lose the last column.
func splitTableCells(line string) []string {
	s := strings.TrimSpace(line)
	var (
		cells []string
		b     strings.Builder
	)
	for i := 0; i < len(s); i++ {
		switch {
		case s[i] == '\\' && i+1 < len(s) && s[i+1] == '|':
			b.WriteByte('|')
			i++
		case s[i] == '|':
			cells = append(cells, strings.TrimSpace(b.String()))
			b.Reset()
		default:
			b.WriteByte(s[i])
		}
	}
	cells = append(cells, strings.TrimSpace(b.String()))

	if len(cells) > 0 && cells[0] == "" {
		cells = cells[1:]
	}
	if len(cells) > 0 && cells[len(cells)-1] == "" {
		cells = cells[:len(cells)-1]
	}
	return cells
}

// isTableRow reports whether a line is part of a markdown table.
func isTableRow(line string) bool {
	return strings.HasPrefix(strings.TrimLeft(line, " \t"), "|")
}
