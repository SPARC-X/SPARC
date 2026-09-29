#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT=31441

echo "== C1 unbiased"
rm -f simulation.out simulation.pos_0.xyz COLVAR
XML=input_unbiased.xml
export PORT XML
# shellcheck disable=SC1091
source "$ROOT/../_common/run_md.sh"
cp simulation.pos_0.xyz unbiased.pos_0.xyz
cp simulation.out unbiased.out

echo "== C1 biased (ffplumed)"
rm -f simulation.out simulation.pos_0.xyz COLVAR
XML=input_biased.xml
export PORT XML
# shellcheck disable=SC1091
source "$ROOT/../_common/run_md.sh"
cp simulation.pos_0.xyz biased.pos_0.xyz
cp simulation.out biased.out

python3 analyze.py
