package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"

	"ai-code/internal/config"
	"ai-code/internal/tool"
)

// runExecutorDaemon serves the tool executor over stdin/stdout as JSON-lines,
// the daemon side of a DevcontainerExecutor. It builds the full tool set, so
// every tool runs against the sandbox's filesystem.
func runExecutorDaemon(cwd string) error {
	cfg, err := config.Load(cwd)
	if err != nil {
		return err
	}
	exec, _ := buildTools(cfg, cwd, 0)

	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	enc := json.NewEncoder(out)
	dec := json.NewDecoder(in)

	for {
		var req tool.Request
		err := dec.Decode(&req)
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		res, err := exec.Execute(context.Background(), req)
		if err != nil {
			// Transport-level failure: report it as a result so the agent
			// loop can recover rather than hang.
			res = tool.Errorf("executor failed: %v", err)
		}
		if err := enc.Encode(res); err != nil {
			return err
		}
		out.Flush()
	}
}
