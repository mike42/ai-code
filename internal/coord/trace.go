package coord

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// Trace writes what every instance decided about the model to one shared file,
// so churn that appears only with two windows open can be read back later. Off
// unless AI_CODE_TRACE names a file; writes are single, whole O_APPEND lines.
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
