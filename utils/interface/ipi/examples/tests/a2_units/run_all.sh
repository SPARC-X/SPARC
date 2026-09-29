#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT=31422
# shellcheck disable=SC1091
source "$ROOT/../_common/run_md.sh"
python3 check_units.py
