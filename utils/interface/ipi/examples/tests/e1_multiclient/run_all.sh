#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT=31461
NP_CLIENT="${NP_CLIENT:-2}"
# shellcheck disable=SC1091
source "$ROOT/../../../../_common/env.sh"
PSP_SRC="$SPARC_REPO/psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8"

if ! command -v i-pi >/dev/null 2>&1; then
  echo "i-pi is not on PATH. Load it with your module system or Python environment, then retry." >&2
  exit 1
fi

for c in client_a client_b; do
  rm -rf "$c"
  mkdir -p "$c"
  cp Al.inpt Al.ion "$c/"
  cp "$PSP_SRC" "$c/"
done

echo "== E1: i-PI nbeads=2"
i-pi input.xml > i-pi.log 2>&1 &
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
raise SystemExit("i-PI did not open port")
' "$PORT"

echo "== E1: two SPARC clients np=$NP_CLIENT"
( cd client_a && sparc_mpi "$NP_CLIENT" "$SPARC" -socket "localhost:${PORT}" -name Al > sparc.log 2>&1 ) &
PA=$!
( cd client_b && sparc_mpi "$NP_CLIENT" "$SPARC" -socket "localhost:${PORT}" -name Al > sparc.log 2>&1 ) &
PB=$!

set +e
wait "$IPI_PID"
IPI_RC=$?
wait "$PA"
wait "$PB"
set -e
echo "== E1: i-PI exit $IPI_RC"
python3 analyze.py
