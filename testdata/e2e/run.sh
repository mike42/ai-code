#!/usr/bin/env bash
#
# End-to-end proof that background workers do not block the session.
#
# Builds ai-code, starts a mock LLM, drives the real binary through a PTY on a
# fixed clock, and writes terminal snapshots to build/out. Nothing here touches
# a real provider: XDG_CONFIG_HOME and XDG_RUNTIME_DIR are redirected, so
# neither the real provider config nor the cross-window coordination directory
# is read.
#
# Usage: ./run.sh [slots|devcontainer|all]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ROOT="$HERE/build"
PORT="${MOCK_PORT:-8099}"
WHICH="${1:-all}"

command -v go >/dev/null || { echo "go is not on PATH" >&2; exit 1; }

mkdir -p "$ROOT"/{out,xdg}

# The coordination directory lives under XDG_RUNTIME_DIR, and a session that
# reads the real one adopts whatever model another window last asked for. So
# it is redirected -- but only to a subdirectory of the real runtime dir, not
# to somewhere convenient: podman keeps its network namespaces there too, and
# pointing it at an ordinary directory fails the container with
# "pasta failed ... Couldn't open network namespace".
XRUN="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ai-code-e2e"
mkdir -p "$XRUN"

# --- python -----------------------------------------------------------------
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "== creating venv"
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" -q install pyte
fi
PY="$ROOT/.venv/bin/python"

# --- binary -----------------------------------------------------------------
echo "== building ai-code"
( cd "$REPO" && go build -o "$ROOT/ai-code" ./cmd/ai-code )

# --- fixtures ---------------------------------------------------------------
write_config() {
  mkdir -p "$1/.ai-code"
  cat > "$1/.ai-code/config.toml" <<EOF
default_provider = "mock"
default_mode     = "build"

[ui]
color       = "always"
status_line = true

[agent]
auto_checkpoint = false

[provider.mock]
kind          = "lemonade"
base_url      = "http://127.0.0.1:$PORT/api/v1"
class         = "on-premises"
default_model = "Mock-Coder-30B"
EOF
}

mkdir -p "$ROOT/proj"
echo "print('hello')" > "$ROOT/proj/hello.py"
write_config "$ROOT/proj"

rm -rf "$ROOT/dcproj"
cp -r "$REPO/examples/sandbox-demo" "$ROOT/dcproj"
write_config "$ROOT/dcproj"

# --- mock -------------------------------------------------------------------
stop_mock() {
  # The bracket keeps this pattern from matching the shell running it.
  for p in $(pgrep -f "mockllm[.]py" 2>/dev/null || true); do kill "$p" 2>/dev/null || true; done
  sleep 0.5
}
start_mock() {
  stop_mock
  : > "$ROOT/mock.log"
  ( cd "$HERE" && MOCK_LOG="$ROOT/mock.log" MOCK_PORT="$PORT" \
      setsid python3 mockllm.py > "$ROOT/mock.stderr" 2>&1 < /dev/null & )
  for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$PORT/api/v1/health" >/dev/null && return 0
    sleep 0.25
  done
  echo "mock did not come up; see $ROOT/mock.stderr" >&2
  exit 1
}
trap stop_mock EXIT

# --- scenarios --------------------------------------------------------------
run_scenario() {
  local name="$1"
  echo "== $name"
  sed -e "s#@ROOT@#$ROOT#g" -e "s#@BIN@#$ROOT/ai-code#g" -e "s#@XRUN@#$XRUN#g" \
    "$HERE/scenarios/$name.json" > "$ROOT/$name.spec.json"
  start_mock
  ( cd "$HERE" && "$PY" drive.py "$ROOT/$name.spec.json" > "$ROOT/out/$name.log" 2>&1 )
  cp "$ROOT/mock.log" "$ROOT/out/$name.mock.log"
  echo "   snapshots: $ROOT/out/$name.snapshots.html"
  echo "              $ROOT/out/$name.snapshots.ansi   (cat it)"
  echo "   server log: $ROOT/out/$name.mock.log"
}

case "$WHICH" in
  all) run_scenario slots; run_scenario devcontainer ;;
  *)   run_scenario "$WHICH" ;;
esac

echo
echo "== done. Review the snapshots; the mock log is the concurrency evidence."
