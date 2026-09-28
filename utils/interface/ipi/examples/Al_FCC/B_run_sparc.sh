#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/../../../_common/env.sh"
sparc_mpi "$NP" "$SPARC" -socket localhost:31415 -name Al 2>&1 | tee sparc.log
