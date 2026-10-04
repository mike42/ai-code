package main

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// The worker runs in containers and on remote machines, which must never
// reach a model server themselves: inference and image generation happen on
// the machine running ai-code. Without the network packages, it cannot.
func TestTheWorkerCannotReachTheNetwork(t *testing.T) {
	out, err := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), "list", "-deps", ".").Output()
	if err != nil {
		t.Fatalf("listing the worker's packages: %v", err)
	}
	for _, pkg := range strings.Fields(string(out)) {
		if pkg == "net" || strings.HasPrefix(pkg, "net/") || pkg == "ai-code/internal/provider" {
			t.Errorf("the worker is built with %s", pkg)
		}
	}
}
