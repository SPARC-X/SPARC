#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT=31431
# Equal physical time (40 fs): 0.5 fs → 80 steps, 1.0 fs → 40 steps, 2.0 fs → 20 steps.

python3 - <<PY
from make_xml import xml_nve
from pathlib import Path
for dt, n in (("0.5", 80), ("1.0", 40), ("2.0", 20)):
    Path(f"input_dt{dt}.xml").write_text(xml_nve(31431, dt, n))
PY

for dt in 0.5 1.0 2.0; do
  echo "== B1 dt=${dt} fs"
  rm -f simulation.out simulation.pos_0.xyz RESTART Al.out Al.static Al.static_01
  XML="input_dt${dt}.xml"
  export PORT XML
  # shellcheck disable=SC1091
  source "$ROOT/../_common/run_md.sh"
  cp simulation.out "out_dt${dt}.dat"
done

python3 analyze.py
