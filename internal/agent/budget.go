package agent

// Budget is the set of sizes a context window implies.
//
// These were absolute constants, tuned against one 262k deployment and
// written down as though they were universal. They are not: at 32k the
// reserve alone claimed half the window, the verbatim tail was larger than
// any request could carry, and one bash result could not fit in a request at
// all. The window is already detected per slot, so it is the one number the
// rest can be derived from.
//
// Not every size here scales, and the ones that do not are the interesting
// part. minOutputTokens and SpeculativeSummaryMaxTokens are absolute on
// purpose -- the first is bounded by what a tool call costs, the second by
// how long a person will wait for a checkpoint nobody asked for, and neither
// of those changes because the window did.
type Budget struct {
	// Reserve is held back at the top of the window.
	Reserve int
	// Summary caps a checkpoint.
	Summary int
	// KeepRecent is how much of the tail survives compaction verbatim.
	KeepRecent int
	// MaxToolResult bounds one tool result, in tokens.
	MaxToolResult int
	// Prune is how much recent tool output is kept whole when reclaiming
	// space without a model call. See Agent.Prune.
	KeepOutput int
}

// BudgetFor derives the sizes for a window of this many tokens. A window of
// zero is one the backend would not name, where nothing can be derived and
// the fallbacks stand in.
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

	// The reserve is the one size with a justified ceiling: it exists to hold
	// one response plus the summarisation call compaction makes, and both of
	// those are bounded by what they are rather than by the window. Past
	// SummaryMaxTokens*4 no session can spend it, so holding more back is
	// window given away for nothing.
	reserve := clamp(window/8, minReserveTokens, DefaultReserveTokens)
	prompt := (window - reserve) * 3 / 4

	// No ceiling on either of these. A share already self-scales: a third of
	// the prompt budget leaves two thirds free for the session to grow into,
	// at every window size, which is the property a ceiling would break. Cap
	// the tail at a fixed 32k on a 1M window and the model works from a 4k
	// checkpoint with 700k of the window standing empty -- more compactions,
	// each one costing a full summarisation call, to save room nothing wanted.
	keepRecent := max(prompt/3, minKeepRecentTokens)

	return Budget{
		Reserve: reserve,
		// Halved again because the reserve holds the response as well as the
		// checkpoint, and a checkpoint that fills the whole reserve leaves
		// compaction no room to answer in once it has run.
		Summary:    clamp(reserve/2, minSummaryTokens, SummaryMaxTokens),
		KeepRecent: keepRecent,
		// A quarter, so the tail holds several exchanges rather than two. This
		// is the divisor that matters: at half, one wide grep and one build log
		// fill everything compaction just made room for. At a quarter a 262k
		// window derives 61,440 bytes, which is within 3% of the 60,000 the
		// constant was hand-tuned to on that same window -- the formula
		// reproducing a known-good number it was not fitted to.
		MaxToolResult: max(keepRecent/4, minToolResultTokens),
		// Half the tail. The tail is what compaction keeps verbatim, so
		// protecting half of it means pruning can never empty out more than
		// half of what a compaction would go on to preserve -- which is the
		// property that keeps the two mechanisms from fighting.
		KeepOutput: max(keepRecent/2, minToolResultTokens),
	}
}

// MaxToolResultBytes is MaxToolResult for a tool that counts bytes rather
// than tokens. Nominal four characters a token: a tool cannot know the
// model's real ratio, and it does not need to -- clampOversized bounds the
// transcript exactly, using the calibrated ratio, whatever the tools let by.
func (b Budget) MaxToolResultBytes() int { return b.MaxToolResult * 4 }

const (
	// The floors are what each size is for, at the point it stops being for
	// anything. A reserve below this cannot hold both a short answer and a
	// short checkpoint; a tail below this cannot hold one tool result and the
	// exchange around it, so keeping it buys nothing a checkpoint would not
	// say better; a summary below this cannot fill the sections its own
	// format asks for.
	minReserveTokens    = 2048
	minKeepRecentTokens = 1024
	minSummaryTokens    = 512
	// A result trimmed below this is not an excerpt, it is a hint that output
	// existed.
	minToolResultTokens = 512
)

func clamp(v, lo, hi int) int { return min(max(v, lo), hi) }
