#!/usr/bin/env python3
"""D1/D2: MTS evaluation counts and conserved quantity."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
from sparc_files import latest_sparc_static


def main():
    static = ""
    static_path = latest_sparc_static(HERE)
    if static_path is not None:
        static = static_path.read_text(errors="replace")
    n_sparc = len(re.findall(r"socket step\s+\d+", static, flags=re.I))
    n_plumed = 0
    colvar = HERE / "COLVAR"
    if colvar.is_file():
        n_plumed = sum(
            1
            for line in colvar.read_text(errors="replace").splitlines()
            if line.strip() and not line.startswith("#")
        )
    log = (HERE / "i-pi.log").read_text(errors="replace")
    m = re.search(r"Calculating \(forward loop\)\s+(\d+)", log)
    if m:
        n_plumed = max(n_plumed, int(m.group(1)))
    rows = []
    with (HERE / "simulation.out").open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    cons = a[:, 2]
    n_outer = len(a) - 1
    ratio = n_plumed / max(n_sparc, 1)
    ok_ratio = n_plumed >= 2 * n_sparc and n_sparc >= n_outer
    dcons = float(cons[-1] - cons[0])
    ok_c = np.all(np.isfinite(cons)) and abs(dcons) < 0.2
    ok = ok_ratio and ok_c
    lines = [
        "D1/D2 MTS",
        f"outer MD rows={len(a)}  SPARC socket steps={n_sparc}  PLUMED COLVAR={n_plumed}",
        f"PLUMED/SPARC ratio={ratio:.2f} (expect ~4)",
        f"Δconserved={dcons:.4e} Ha",
        f"inner evaluates more often than SPARC: {'PASS' if ok_ratio else 'FAIL'}",
        f"conserved finite: {'PASS' if ok_c else 'FAIL'}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "mts_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
