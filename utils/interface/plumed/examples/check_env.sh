#!/usr/bin/env bash
# Report the tools these examples need. Does not assume WSL or a cluster.
set +e
ROOT="$(cd "$(dirname "$0")" && pwd)"
export SPARC_ENV_STRICT=0
# shellcheck disable=SC1091
source "$ROOT/_common/env.sh"

rc=0
echo "=== host ==="
uname -srm 2>/dev/null || uname -a
echo "=== SPARC ==="
echo "SPARC_REPO=$SPARC_REPO"
echo "SPARC=$SPARC"
if [[ -f "$SPARC" ]]; then
  ls -la "$SPARC"
else
  echo "missing SPARC binary. Set SPARC= to a USE_SOCKET=1 build."
  rc=1
fi
echo "=== MPI ==="
echo "MPIEXEC=$MPIEXEC"
echo "NP=$NP"
_mpi_bin="${MPIEXEC%% *}"
if command -v "$_mpi_bin" >/dev/null 2>&1; then
  command -v "$_mpi_bin"
  "$_mpi_bin" --version 2>&1 | head -n 3
else
  echo "MPI launcher not on PATH: $_mpi_bin"
  echo "Set MPIEXEC (default \"mpirun -np\"; on Slurm, MPIEXEC=\"srun -n\")."
  rc=1
fi
echo "=== PLUMED ==="
echo "PLUMED_ROOT=${PLUMED_ROOT:-}"
echo "PLUMED_KERNEL=${PLUMED_KERNEL:-}"
if [[ -n "${PLUMED_KERNEL:-}" && -f "${PLUMED_KERNEL}" ]]; then
  ls -la "$PLUMED_KERNEL"
else
  echo "PLUMED_KERNEL is not a file."
  echo "Set PLUMED_KERNEL or PLUMED_ROOT, or load the site PLUMED module, before a run that calls PLUMED."
fi
if command -v plumed >/dev/null 2>&1; then
  command -v plumed
else
  echo "plumed executable is not on PATH (the Python module can still work)."
fi
echo "=== python ==="
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is not on PATH"
  rc=1
else
  python3 - <<'PY'
import importlib, sys
print("python", sys.executable, sys.version.split()[0])
failed = False
for name in ("numpy", "plumed"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name}: OK  {getattr(mod, '__file__', '')}")
    except Exception as exc:
        print(f"  {name}: FAIL  {exc}")
        failed = True
raise SystemExit(1 if failed else 0)
PY
  if [[ $? -ne 0 ]]; then
    rc=1
  fi
fi
echo "=== i-pi ==="
if command -v i-pi >/dev/null 2>&1; then
  command -v i-pi
else
  echo "i-pi is not on PATH (required for ipi/examples, not for plumed/examples)."
fi
if [[ "$rc" -eq 0 ]]; then
  echo "DONE"
else
  echo "DONE (missing pieces above)"
fi
exit "$rc"
