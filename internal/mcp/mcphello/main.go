// Command mcphello is a hello-world MCP server for trying ai-code's MCP
// support by hand and for its tests. It speaks stdio unless --http is given.
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"

	"ai-code/internal/mcp/hello"
)

func main() {
	var (
		opt      hello.Options
		addr     string
		jsonOnly bool
	)
	flag.StringVar(&opt.Greeting, "greeting", "", "greeting used by the hello tool")
	flag.StringVar(&opt.Root, "root", os.Getenv("HELLO_ROOT"), "echoed by the hello tool")
	flag.StringVar(&opt.Hold, "hold", "", "dial this address before answering initialize and wait for one byte")
	flag.StringVar(&opt.Token, "token", "", "with --http, require this bearer token")
	flag.BoolVar(&opt.ListChanges, "list-changes", false, "add a goodbye tool after the first call")
	flag.StringVar(&addr, "http", "", "serve Streamable HTTP on this address instead of stdio")
	flag.BoolVar(&jsonOnly, "json", false, "with --http, answer with JSON bodies instead of SSE")
	flag.Parse()
	opt.Exit = os.Exit

	if addr != "" {
		if err := http.ListenAndServe(addr, hello.Handler(opt, jsonOnly)); err != nil {
			fmt.Fprintln(os.Stderr, "mcphello:", err)
			os.Exit(1)
		}
		return
	}
	if err := hello.Serve(os.Stdin, os.Stdout, opt); err != nil {
		fmt.Fprintln(os.Stderr, "mcphello:", err)
		os.Exit(1)
	}
}
