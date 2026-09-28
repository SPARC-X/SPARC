#!/usr/bin/env bash
# Terminal B: SPARC force client. Start after i-PI is listening.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/../../../../_common/env.sh"
PSP_SRC="$SPARC_REPO/psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8"

if [[ ! -f 13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 ]]; then
  cp "$PSP_SRC" .
fi

echo "SPARC client -> localhost:31415  ($SPARC, np=$NP)"
sparc_mpi "$NP" "$SPARC" -socket localhost:31415 -name Al 2>&1 | tee sparc.log
