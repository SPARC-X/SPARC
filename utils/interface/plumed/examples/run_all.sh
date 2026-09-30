#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/_common/env.sh"

if command -v fuser >/dev/null 2>&1; then
  for p in 32410 32411 32421 32431 32432; do
    fuser -k "${p}/tcp" >/dev/null 2>&1 || true
  done
fi

tests=(
  a0_socket_smoke
  a1_energy
  b1_cv
  b2_umbrella
)

rc=0
for d in "${tests[@]}"; do
  echo
  echo "======== $d ========"
  if (cd "$ROOT/$d" && chmod +x run_all.sh && ./run_all.sh); then
    echo "OK $d"
  else
    echo "FAIL $d" >&2
    rc=1
  fi
done
exit "$rc"
