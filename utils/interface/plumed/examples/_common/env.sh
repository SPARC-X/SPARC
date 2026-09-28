#!/usr/bin/env bash
# Source from a PLUMED example directory. Sets PYTHONPATH and the shared
# SPARC / PLUMED / MPI defaults. Intentionally no `set -e`: check_env.sh
# sources this file.
_EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "$_EX_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$_EX_DIR/../../../_common/env.sh"
_PLUMED="$(cd "$_ROOT/.." && pwd)"
export PYTHONPATH="${_PLUMED}${PYTHONPATH:+:$PYTHONPATH}"
