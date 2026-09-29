#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v i-pi >/dev/null 2>&1; then
  echo "i-pi is not on PATH. Load it with your module system or Python environment, then retry." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
i-pi input.xml 2>&1 | tee i-pi.log
