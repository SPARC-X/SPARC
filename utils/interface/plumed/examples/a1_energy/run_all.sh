#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/../_common/env.sh"
cd "$ROOT"
export PORT="${PORT:-32411}"
python3 run.py
python3 analyze.py
