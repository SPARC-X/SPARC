#!/usr/bin/env bash
# One-shot A1: i-PI + SPARC MD, then independent single-points, then compare.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/../../../../_common/env.sh"
PORT=31415
PSP_SRC="$SPARC_REPO/psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8"

if ! command -v i-pi >/dev/null 2>&1; then
  echo "i-pi is not on PATH. Load it with your module system or Python environment, then retry." >&2
  exit 1
fi
if [[ ! -f 13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 ]]; then
  cp "$PSP_SRC" .
fi

echo "== A1: starting i-PI on port $PORT"
i-pi input.xml > i-pi.log 2>&1 &
IPI_PID=$!

python3 -c '
import socket, sys, time
port = int(sys.argv[1])
for _ in range(120):
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        raise SystemExit(0)
    except OSError:
        time.sleep(0.25)
print("i-PI did not open port", port, file=sys.stderr)
raise SystemExit(1)
' "$PORT"

echo "== A1: starting SPARC client"
sparc_mpi "$NP" "$SPARC" -socket "localhost:${PORT}" -name Al > sparc.log 2>&1 &
SPARC_PID=$!

set +e
wait "$IPI_PID"
IPI_RC=$?
wait "$SPARC_PID"
SPARC_RC=$?
set -e

echo "== A1: i-PI exit $IPI_RC  SPARC exit $SPARC_RC (SPARC may be nonzero on EXIT)"
if [[ ! -f simulation.out ]]; then
  echo "Missing simulation.out. See i-pi.log / sparc.log" >&2
  exit 1
fi

echo "== A1: independent SPARC single-points"
python3 C_run_singlepoints.py --sparc "$SPARC" --np "$NP"

echo "== A1: compare"
python3 D_compare.py
