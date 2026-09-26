//go:build !unix

package main

import "ai-code/internal/render"

// Windows has no SIGWINCH. A console resize is discovered by polling
// GetConsoleScreenBufferInfo, which is a separate piece of work; until then
// the layout is correct at startup and stale after a resize.
func watchResize(*render.Screen) func() { return func() {} }
