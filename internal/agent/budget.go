package agent

// Budget is the sizes a context window implies.
type Budget struct {
	// Reserve is held back at the top of the window.
	Reserve int
	// Summary caps a checkpoint.
	Summary int
	// KeepRecent is how much of the tail survives compaction verbatim.
	KeepRecent int
	// MaxToolResult bounds one tool result, in tokens.
	MaxToolResult int
	// KeepOutput is how much recent tool output is kept whole when reclaiming
	// space without a model call.
	KeepOutput int
}

// BudgetFor derives the sizes for a window of this many tokens, 0 when unknown.
func BudgetFor(window int) Budget {
	if window <= 0 {
		return Budget{
			Reserve:       DefaultReserveTokens,
			Summary:       SummaryMaxTokens,
			KeepRecent:    DefaultKeepRecentTokens,
			MaxToolResult: DefaultKeepRecentTokens / 2,
			KeepOutput:    DefaultKeepRecentTokens / 2,
		}
	}

	// The reserve holds one response plus the summarisation call compaction makes.
	reserve := clamp(window/8, minReserveTokens, DefaultReserveTokens)
	prompt := (window - reserve) * 3 / 4

	// A share self-scales; a ceiling would break that at every window size.
	keepRecent := max(prompt/3, minKeepRecentTokens)

	return Budget{
		Reserve: reserve,
		// Half the reserve: the reserve also holds the response.
		Summary:    clamp(reserve/2, minSummaryTokens, SummaryMaxTokens),
		KeepRecent: keepRecent,
		// A quarter of the tail, so several exchanges fit rather than two.
		MaxToolResult: max(keepRecent/4, minToolResultTokens),
		// Half the tail, so pruning never empties more than compaction would keep.
		KeepOutput: max(keepRecent/2, minToolResultTokens),
	}
}

// MaxToolResultBytes is MaxToolResult at four characters a token.
func (b Budget) MaxToolResultBytes() int { return b.MaxToolResult * 4 }

const (
	// The floors for reserve, tail and checkpoint: a short answer plus a short
	// checkpoint, one tool result plus the exchange around it, and the sections
	// the summary format asks for.
	minReserveTokens    = 2048
	minKeepRecentTokens = 1024
	minSummaryTokens    = 512
	// minToolResultTokens: below this a result is a hint that output existed.
	minToolResultTokens = 512
)

func clamp(v, lo, hi int) int { return min(max(v, lo), hi) }
