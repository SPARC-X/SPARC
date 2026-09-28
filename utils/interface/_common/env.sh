#!/usr/bin/env bash
# Shared runtime for the socket examples. Source this file.
# It does not activate a module or a conda env, and it does not assume
# WSL or a cluster. Put i-pi, python3, and an MPI launcher on PATH first.
#
# Overrides:
#   SPARC          socket binary (default: <repo>/lib/sparc)
#   SPARC_REPO     repository root
#   NP             MPI ranks (default 4)
#   MPIEXEC        words before the rank count
#                  default: "mpirun -np"
#                  Slurm:   MPIEXEC="srun -n"
#   PLUMED_ROOT    install prefix; used only when lib/ exists
#   PLUMED_KERNEL  libplumedKernel.so; left unchanged if already set
#   SPARC_ENV_STRICT  0 keeps going when the SPARC binary is missing
_IFACE_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SPARC_REPO:-}" ]]; then
  export SPARC_REPO="$(cd "$_IFACE_COMMON/../../.." && pwd)"
fi
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
