package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	airt "ai-code/internal/runtime"
	"ai-code/internal/tool"
	"ai-code/internal/worker"
)

type phases struct {
	mu  sync.Mutex
	all []tool.Phase
}

func (p *phases) add(ph tool.Phase) { p.mu.Lock(); p.all = append(p.all, ph); p.mu.Unlock() }

func (p *phases) output() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []string
	for _, ph := range p.all {
		if ph.Output {
			out = append(out, ph.Note)
		}
	}
	return out
}

func TestOutputLinesStreamsWholeLinesAndShowsAPartialOne(t *testing.T) {
	var got phases
	o := newOutputLines(got.add, "building;")

	o.Write([]byte("STEP 1/3\nSTEP 2/3\r\nEnter pass"))
	if last := got.all[len(got.all)-1]; last.Output || last.Note != "building; Enter pass" {
		t.Errorf("partial line shown as %+v, want it in the notice", last)
	}
	o.Write([]byte("phrase: \n"))
	o.flush()

	want := []string{"STEP 1/3", "STEP 2/3", "Enter passphrase: "}
	if g := got.output(); strings.Join(g, "|") != strings.Join(want, "|") {
		t.Errorf("committed lines = %q, want %q", g, want)
	}
	if f := o.failure(); strings.Contains(f, "STEP") {
		t.Errorf("failure() = %q; output already shown must not be repeated", f)
	}
}

func TestOutputLinesWithoutAListenerKeepsTheTailForTheError(t *testing.T) {
	o := newOutputLines(nil, "")
	o.Write([]byte("E: Unable to locate package nope\n"))
	if f := o.failure(); !strings.Contains(f, "Unable to locate package") {
		t.Errorf("failure() = %q, want the build output", f)
	}
}

// The build's output must reach the screen while it runs, not only on failure.
func TestImageBuildOutputIsShownAsItArrives(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the stand-in engine is a shell script")
	}
	dir := t.TempDir()
	dc := filepath.Join(dir, ".devcontainer")
	if err := os.MkdirAll(dc, 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dc, "Dockerfile"), []byte("FROM scratch\n"), 0o644)
	os.WriteFile(filepath.Join(dc, "devcontainer.json"), []byte(`{"build":{"dockerfile":"Dockerfile"}}`), 0o644)

	engine := filepath.Join(dir, "engine")
	os.WriteFile(engine, []byte(`#!/bin/sh
case "$1" in
image) exit 1 ;;
build) echo "STEP 1/2: FROM scratch"; echo "STEP 2/2: RUN apt-get update" >&2; exit "${FAIL:-0}" ;;
esac
`), 0o755)

	rt, err := airt.Resolve("devcontainer", dir)
	if err != nil || rt.ConfigErr != nil {
		t.Fatalf("Resolve: %v %v", err, rt.ConfigErr)
	}

	for _, fail := range []string{"0", "1"} {
		t.Setenv("FAIL", fail)
		var got phases
		e := NewDevcontainerExecutor(rt.Config, nil, dir, dir, worker.Settings{})
		e.SetProgress(got.add)

		_, err := e.resolveImage(context.Background(), engine)
		if (err != nil) != (fail == "1") {
			t.Fatalf("FAIL=%s: resolveImage error = %v", fail, err)
		}
		want := []string{"STEP 1/2: FROM scratch", "STEP 2/2: RUN apt-get update"}
		if g := got.output(); strings.Join(g, "|") != strings.Join(want, "|") {
			t.Errorf("FAIL=%s: lines shown = %q, want %q", fail, g, want)
		}
		if err != nil && strings.Contains(err.Error(), "STEP") {
			t.Errorf("error repeats output already shown: %v", err)
		}
	}
}
