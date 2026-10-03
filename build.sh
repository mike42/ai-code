#!/bin/sh
# Builds the workers, then ai-code with them embedded.
# Env: GOOS/GOARCH (ai-code host), OUT (default dist), VERSION (optional).
set -eu
cd "$(dirname "$0")"
OUT=${OUT:-dist}
WORKERS="linux/amd64 linux/arm64 darwin/amd64 darwin/arm64"
BIN=internal/workerbin/bin

rm -f "$BIN"/ai-code-worker-*.gz
for p in $WORKERS; do
	os=${p%/*} arch=${p#*/}
	CGO_ENABLED=0 GOOS=$os GOARCH=$arch go build -trimpath -ldflags='-s -w' \
		-o "$BIN/ai-code-worker-$os-$arch" ./cmd/ai-code-worker
	gzip -9 -n -f "$BIN/ai-code-worker-$os-$arch"
done

LDFLAGS="-s -w"
if [ -n "${VERSION:-}" ]; then
	LDFLAGS="$LDFLAGS -X main.version=$VERSION"
fi
mkdir -p "$OUT"
CGO_ENABLED=0 go build -trimpath -ldflags "$LDFLAGS" -o "$OUT/ai-code$(go env GOEXE)" ./cmd/ai-code
echo "built $OUT/ai-code$(go env GOEXE) with workers for: $WORKERS"
