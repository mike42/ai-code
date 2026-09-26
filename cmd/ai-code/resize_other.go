//go:build !unix

package main

import "ai-code/internal/render"

// watchResize does nothing here: Windows has no SIGWINCH, so a console
// resize is not noticed.
func watchResize(*render.Screen) func() { return func() {} }
