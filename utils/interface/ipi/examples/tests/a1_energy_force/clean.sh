#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
rm -f i-pi.log sparc.log simulation.out simulation.pos_0.xyz simulation.frc_0.xyz \
      RESTART compare_table.txt Al.out Al.static Al.static_01 Al.out_01
rm -rf sp_frame* __pycache__
echo "cleaned $ROOT"
