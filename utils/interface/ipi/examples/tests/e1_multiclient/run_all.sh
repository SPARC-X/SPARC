#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT=31461
NP_CLIENT="${NP_CLIENT:-2}"
if [[ -z "${SPARC_REPO:-}" ]]; then
  _d="$ROOT"
  while [[ "$_d" != "/" && ! -f "$_d/src/makefile" ]]; do
    _d="$(dirname "$_d")"
  done
  export SPARC_REPO="$_d"
fi
unset _d
export SPARC="${SPARC:-$SPARC_REPO/lib/sparc}"
export NP="${NP:-4}"
export MPIEXEC="${MPIEXEC:-mpirun -np}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
_plumed_lib=""
if [[ -n "${PLUMED_ROOT:-}" && -d "${PLUMED_ROOT}/lib" ]]; then
  _plumed_lib="${PLUMED_ROOT}/lib"
elif [[ -n "${PLUMED_KERNEL:-}" && -e "${PLUMED_KERNEL}" ]]; then
  _plumed_lib="$(cd "$(dirname "${PLUMED_KERNEL}")" && pwd)"
elif [[ -d "${HOME}/opt/plumed/lib" ]]; then
  _plumed_lib="${HOME}/opt/plumed/lib"
  export PLUMED_ROOT="${HOME}/opt/plumed"
fi
if [[ -n "$_plumed_lib" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_plumed_lib}:"*) ;;
    *) export LD_LIBRARY_PATH="${_plumed_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
  if [[ -z "${PLUMED_KERNEL:-}" && -f "${_plumed_lib}/libplumedKernel.so" ]]; then
    export PLUMED_KERNEL="${_plumed_lib}/libplumedKernel.so"
  fi
fi
unset _plumed_lib
sparc_mpi() {
  # shellcheck disable=SC2086
  $MPIEXEC "$@"
}
if [[ ! -f "$SPARC" ]]; then
  echo "SPARC binary not found: $SPARC" >&2
  echo "Set SPARC= to a binary built with USE_SOCKET=1." >&2
  exit 1
fi
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
