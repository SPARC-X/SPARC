#!/usr/bin/env bash
# Terminal A: start i-PI (MD server) first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v i-pi >/dev/null 2>&1; then
  echo "i-pi is not on PATH. Load it with your module system or Python environment, then retry." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
echo "i-PI listening on localhost:31415 (cwd=$ROOT)"
exec i-pi input.xml 2>&1 | tee i-pi.log
