#!/bin/bash
set -euo pipefail
PATH="$PATH:/usr/local/go/bin" CGO_ENABLED=0 go build -o ~/.local/bin/ai-code ./cmd/ai-code
