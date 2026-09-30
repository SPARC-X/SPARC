#!/usr/bin/env bash
# Source from a PLUMED example directory. Sets PYTHONPATH and the SPARC /
# PLUMED / MPI defaults. Intentionally no `set -e`: check_env.sh sources this.
#
# Overrides: SPARC, SPARC_REPO, NP, MPIEXEC, PLUMED_ROOT, PLUMED_KERNEL.
# SPARC_ENV_STRICT=0 keeps going when the SPARC binary is missing.
_EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "$_EX_DIR/.." && pwd)"

if [[ -z "${SPARC_REPO:-}" ]]; then
  _d="$_EX_DIR"
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

# MPIEXEC is split on spaces on purpose (for example "srun -n").
sparc_mpi() {
  # shellcheck disable=SC2086
  $MPIEXEC "$@"
}

if [[ "${SPARC_ENV_STRICT:-1}" != "0" && ! -f "$SPARC" ]]; then
  echo "SPARC binary not found: $SPARC" >&2
  echo "Set SPARC= to a binary built with USE_SOCKET=1." >&2
  exit 1
fi

_PLUMED="$(cd "$_ROOT/.." && pwd)"
export PYTHONPATH="${_PLUMED}${PYTHONPATH:+:$PYTHONPATH}"
