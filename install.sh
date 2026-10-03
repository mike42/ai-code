#!/bin/bash
set -euo pipefail
PATH="$PATH:/usr/local/go/bin" OUT=~/.local/bin "$(dirname "$0")/build.sh"
