#!/usr/bin/env bash
# Shared i-PI + SPARC launcher. Source from a test directory after setting:
#   PORT, NP, SPARC (optional), XML (default input.xml)
set -euo pipefail
_RUNMD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_RUNMD/../../../../_common/env.sh"
PORT="${PORT:-31415}"
XML="${XML:-input.xml}"
PSP_SRC="$SPARC_REPO/psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8"

if ! command -v i-pi >/dev/null 2>&1; then
  echo "i-pi is not on PATH. Load it with your module system or Python environment, then retry." >&2
  exit 1
fi
if [[ ! -f 13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 ]]; then
  cp "$PSP_SRC" .
fi

echo "== MD: i-PI $XML on port $PORT"
i-pi "$XML" > i-pi.log 2>&1 &
IPI_PID=$!

python3 -c '
import socket, sys, time
port = int(sys.argv[1])
for _ in range(180):
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

echo "== MD: SPARC client np=$NP"
sparc_mpi "$NP" "$SPARC" -socket "localhost:${PORT}" -name Al > sparc.log 2>&1 &
SPARC_PID=$!

set +e
wait "$IPI_PID"
IPI_RC=$?
wait "$SPARC_PID"
SPARC_RC=$?
set -e

echo "== MD: i-PI exit $IPI_RC  SPARC exit $SPARC_RC"
if [[ ! -f simulation.out ]]; then
  echo "Missing simulation.out. See i-pi.log / sparc.log" >&2
  exit 1
fi
export IPI_RC SPARC_RC
