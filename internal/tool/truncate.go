package tool

import (
	"fmt"
	"strings"
)

// Truncate shortens tool output that would otherwise eat the context window,
// keeping the start and the end: build output and stack traces pay off at the
// end.
func Truncate(s string, maxBytes int) (string, bool) {
	return truncateShare(s, maxBytes, 0.4)
}

func truncateShare(s string, maxBytes int, headShare float64) (string, bool) {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s, false
	}

	lines := strings.Split(s, "\n")
	headBudget := int(float64(maxBytes) * headShare)
	tailBudget := maxBytes - headBudget

	var (
		head    []string
		tail    []string
		used    int
		headEnd int
	)
	for i, ln := range lines {
		if used+len(ln)+1 > headBudget {
			headEnd = i
			break
		}
		head = append(head, ln)
		used += len(ln) + 1
		headEnd = i + 1
	}

	used = 0
	tailStart := len(lines)
	for i := len(lines) - 1; i >= headEnd; i-- {
		if used+len(lines[i])+1 > tailBudget {
			break
		}
		tail = append([]string{lines[i]}, tail...)
		used += len(lines[i]) + 1
		tailStart = i
	}

	elidedLines := tailStart - headEnd
	if elidedLines <= 0 {
		// Budgets met in the middle: nothing elided, but one very long line exceeded maxBytes.
		return truncateMiddle(s, maxBytes), true
	}

	elidedBytes := len(s) - lenLines(head) - lenLines(tail)
	marker := fmt.Sprintf("\n... %s elided (%s) ...\n", plural(elidedLines, "line"), humanBytes(elidedBytes))

	return strings.Join(head, "\n") + marker + strings.Join(tail, "\n"), true
}

// truncateMiddle handles content with no useful line structure, such as a
// single enormous line of minified JSON.
func truncateMiddle(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	if maxBytes < 64 {
		return s[:maxBytes]
	}
	head := maxBytes * 2 / 5
	tail := maxBytes - head - 48
	if tail < 0 {
		tail = 0
	}
	return s[:head] +
		fmt.Sprintf("\n... %s elided ...\n", humanBytes(len(s)-head-tail)) +
		s[len(s)-tail:]
}

func lenLines(lines []string) int {
	n := 0
	for _, l := range lines {
		n += len(l) + 1
	}
	return n
}

func plural(n int, word string) string {
	if n == 1 {
		return "1 " + word
	}
	return fmt.Sprintf("%d %s", n, pluralise(word))
}

// pluralise handles sibilant endings that a bare "+s" gets wrong.
func pluralise(word string) string {
	switch {
	case strings.HasSuffix(word, "s"), strings.HasSuffix(word, "x"),
		strings.HasSuffix(word, "z"), strings.HasSuffix(word, "ch"),
		strings.HasSuffix(word, "sh"):
		return word + "es"
	case strings.HasSuffix(word, "y") && len(word) > 1 && !isVowel(word[len(word)-2]):
		return word[:len(word)-1] + "ies"
	default:
		return word + "s"
	}
}

func isVowel(b byte) bool {
	switch b {
	case 'a', 'e', 'i', 'o', 'u':
		return true
	}
	return false
}

func humanBytes(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%d B", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1f KB", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1f MB", float64(n)/(1024*1024))
	}
}
