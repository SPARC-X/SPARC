#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/../_common/env.sh"
cd "$ROOT"
export PORT_U="${PORT_U:-32431}"
export PORT_B="${PORT_B:-32432}"
python3 run.py
python3 analyze.py
