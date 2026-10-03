package worker

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

	"ai-code/internal/tool"
)

func pipeWorker(t *testing.T, s Settings) (*Client, func()) {
	t.Helper()
	reqR, reqW := io.Pipe()
	resR, resW := io.Pipe()
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = Serve(reqR, resW, t.TempDir())
		resW.Close()
	}()
	c, err := NewClient(reqW, resR, s)
	if err != nil {
		t.Fatal(err)
	}
	return c, func() { reqW.Close(); <-done }
}

func bash(id, command string) tool.Request {
	args, _ := json.Marshal(map[string]string{"command": command})
	return tool.Request{CallID: id, Name: "bash", Args: args}
}

func TestACancelledCallStopsTheCommand(t *testing.T) {
	c, stop := pipeWorker(t, Settings{})
	defer stop()

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(200*time.Millisecond, cancel)
	start := time.Now()
	res, err := c.Call(ctx, bash("c1", "sleep 30"))
	if err != nil {
		t.Fatal(err)
	}
	if took := time.Since(start); took > 10*time.Second {
		t.Fatalf("the command ran on for %s after it was cancelled", took)
	}
	if !res.Interrupted {
		t.Errorf("result does not say it was interrupted: %+v", res)
	}

	res, err = c.Call(context.Background(), bash("c2", "echo next"))
	if err != nil || res.IsError || !strings.Contains(res.Content, "next") {
		t.Fatalf("call after a cancel: %+v, %v", res, err)
	}
}

func TestAStaleCancelDoesNotStopTheNextCall(t *testing.T) {
	c, stop := pipeWorker(t, Settings{})
	defer stop()
	if err := c.enc.Encode(tool.Request{CallID: "gone", Cancel: true}); err != nil {
		t.Fatal(err)
	}
	res, err := c.Call(context.Background(), bash("c3", "sleep 0.3; echo finished"))
	if err != nil || res.Interrupted || res.IsError {
		t.Fatalf("%+v, %v", res, err)
	}
}

func TestTheWorkerStopsItsCommandWhenTheSessionGoes(t *testing.T) {
	reqR, reqW := io.Pipe()
	done := make(chan error, 1)
	go func() { done <- Serve(reqR, io.Discard, t.TempDir()) }()
	enc := json.NewEncoder(reqW)
	_ = enc.Encode(Hello{})
	_ = enc.Encode(bash("c", "sleep 30"))
	time.Sleep(200 * time.Millisecond)
	reqW.Close()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the worker kept running a command after its session went away")
	}
}

func TestTheWorkerUsesTheSettingsItIsSent(t *testing.T) {
	c, stop := pipeWorker(t, Settings{BashMaxOutput: 2000})
	defer stop()
	res, err := c.Call(context.Background(), bash("c", "head -c 100000 /dev/zero | tr '\\0' x"))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Content) > 4000 {
		t.Errorf("output was %d bytes with a 2000-byte limit sent", len(res.Content))
	}
}
