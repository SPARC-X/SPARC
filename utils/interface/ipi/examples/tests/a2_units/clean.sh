#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
rm -f i-pi.log sparc.log simulation.out simulation.pos_au_0.xyz simulation.pos_aa_0.xyz \
      RESTART units_table.txt Al.out Al.out_* Al.static Al.static_*
rm -rf __pycache__
echo "cleaned $ROOT"
