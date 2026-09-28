#!/usr/bin/env bash
# Rebuild SPARC from the sources in this repository. Used by A3.
# Honors the caller's makefile overrides (USE_SOCKET, USE_PLUMED, PLUMED_ROOT).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../../../../../" && pwd)/src"
jobs="$(nproc 2>/dev/null || echo 1)"
echo "PWD=$(pwd)  make -j${jobs}"
make -j"$jobs"
ls -la ../lib/sparc
strings ../lib/sparc | grep -E 'Socket server requested EXIT|unknown message' | head
