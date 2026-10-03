// Package workerbin embeds the gzipped workers that build.sh puts in bin/.
package workerbin

import (
	"bytes"
	"compress/gzip"
	"embed"
	"fmt"
	"io"
	"io/fs"
	"sort"
	"strings"
	"sync"
)

//go:embed all:bin
var files embed.FS

const prefix, suffix = "ai-code-worker-", ".gz"

func Platforms() []string {
	entries, _ := fs.ReadDir(files, "bin")
	var out []string
	for _, e := range entries {
		n := e.Name()
		if strings.HasPrefix(n, prefix) && strings.HasSuffix(n, suffix) {
			p := strings.TrimSuffix(strings.TrimPrefix(n, prefix), suffix)
			if os, arch, ok := strings.Cut(p, "-"); ok {
				out = append(out, os+"/"+arch)
			}
		}
	}
	sort.Strings(out)
	return out
}

var (
	mu    sync.Mutex
	cache = map[string][]byte{}
)

func For(goos, goarch string) ([]byte, error) {
	key := goos + "-" + goarch
	mu.Lock()
	defer mu.Unlock()
	if b, ok := cache[key]; ok {
		return b, nil
	}
	gz, err := files.ReadFile("bin/" + prefix + key + suffix)
	if err != nil {
		have := Platforms()
		if len(have) == 0 {
			return nil, fmt.Errorf("this build of ai-code carries no workers, so its tools cannot run "+
				"anywhere but here; it needs building with build.sh to run them on %s/%s", goos, goarch)
		}
		return nil, fmt.Errorf("this build of ai-code carries workers for %s, not %s/%s",
			strings.Join(have, ", "), goos, goarch)
	}
	r, err := gzip.NewReader(bytes.NewReader(gz))
	if err != nil {
		return nil, fmt.Errorf("the %s/%s worker in this build is damaged: %w", goos, goarch, err)
	}
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("the %s/%s worker in this build is damaged: %w", goos, goarch, err)
	}
	cache[key] = b
	return b, nil
}
