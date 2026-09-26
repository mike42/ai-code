package coord

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// Tracing writes what every instance decided about the model to one shared
// file, so a churn that only appears with two windows open can be read back
// afterwards rather than reasoned about.
//
// Off unless AI_CODE_TRACE names a file. Lines are appended with O_APPEND in
// single writes, which the kernel keeps whole, so several processes share
// one file without interleaving mid-line.
var (
	traceMu   sync.Mutex
	traceFile *os.File
	traceOnce sync.Once
)

func Trace(format string, args ...any) {
	traceOnce.Do(func() {
		path := os.Getenv("AI_CODE_TRACE")
		if path == "" {
			return
		}
		f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			return
		}
		traceFile = f
	})
	if traceFile == nil {
		return
	}
	line := fmt.Sprintf("%s pid=%-7d %s\n",
		time.Now().Format("15:04:05.000"), os.Getpid(), fmt.Sprintf(format, args...))

	traceMu.Lock()
	_, _ = traceFile.WriteString(line)
	traceMu.Unlock()
}
