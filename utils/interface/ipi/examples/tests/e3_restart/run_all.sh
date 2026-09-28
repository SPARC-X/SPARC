#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
B2="$ROOT/../b2_nvt"
if [[ ! -f "$B2/RESTART" ]]; then
  echo "Need $B2/RESTART — run b2_nvt first." >&2
  exit 1
fi
cp "$B2/RESTART" .
python3 - <<'PY'
from pathlib import Path
p = Path("RESTART")
t = p.read_text()
t = t.replace("<total_steps>50</total_steps>", "<total_steps>55</total_steps>")
p.write_text(t)
PY
cp "$B2/Al.inpt" "$B2/Al.ion" .
PSP_SRC="$ROOT/../../../../../../psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8"
cp "$PSP_SRC" .
# Continue only a few extra steps: rewrite total_steps in a sidecar if needed.
# i-pi RESTART continues from the checkpoint as stored.

PORT=31432
XML=RESTART
export PORT XML
# shellcheck disable=SC1091
source "$ROOT/../_common/run_md.sh"
python3 analyze.py
