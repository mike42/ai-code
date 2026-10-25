// Package mcp is ai-code's Model Context Protocol client: tools only, over
// stdio or Streamable HTTP.
package mcp

import (
	"fmt"
	"strings"
)

// Split turns a command line into words the way a POSIX shell splits them,
// and does nothing else a shell does: no expansion, globbing, redirection or
// command separators. The result is exec'd directly.
//
// Outside quotes, a backslash escapes only whitespace, a quote or another
// backslash, so a Windows path such as C:\tools\x.exe arrives intact.
func Split(s string) ([]string, error) {
	var (
		words  []string
		cur    strings.Builder
		inWord bool
	)
	r := []rune(s)
	for i := 0; i < len(r); i++ {
		c := r[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			if inWord {
				words = append(words, cur.String())
				cur.Reset()
				inWord = false
			}
		case c == '\'':
			inWord = true
			end := indexRune(r, i+1, '\'')
			if end < 0 {
				return nil, fmt.Errorf("unterminated ' at column %d", i+1)
			}
			cur.WriteString(string(r[i+1 : end]))
			i = end
		case c == '"':
			inWord = true
			start := i
			closed := false
			for i++; i < len(r); i++ {
				if r[i] == '\\' && i+1 < len(r) && (r[i+1] == '"' || r[i+1] == '\\') {
					i++
					cur.WriteRune(r[i])
					continue
				}
				if r[i] == '"' {
					closed = true
					break
				}
				cur.WriteRune(r[i])
			}
			if !closed {
				return nil, fmt.Errorf("unterminated \" at column %d", start+1)
			}
		case c == '\\':
			inWord = true
			if i+1 < len(r) && strings.ContainsRune(" \t\n\r'\"\\", r[i+1]) {
				i++
				cur.WriteRune(r[i])
			} else {
				cur.WriteRune(c)
			}
		default:
			inWord = true
			cur.WriteRune(c)
		}
	}
	if inWord {
		words = append(words, cur.String())
	}
	return words, nil
}

func indexRune(r []rune, from int, c rune) int {
	for i := from; i < len(r); i++ {
		if r[i] == c {
			return i
		}
	}
	return -1
}
