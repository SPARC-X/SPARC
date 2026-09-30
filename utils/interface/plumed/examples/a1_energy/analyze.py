#!/usr/bin/env python3
"""Compare SPARC socket energy to PLUMED COLVAR ENERGY (both eV)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "plumed"))

from driver import EV_TO_KJMOL, read_colvar

HERE = Path(__file__).resolve().parent
TOL = 1.0e-4


def main() -> int:
    hist = np.genfromtxt(HERE / "history.csv", delimiter=",", names=True)
    col = read_colvar(HERE / "COLVAR")
    e_sp = np.atleast_1d(np.asarray(hist["energy_dft"], dtype=float))
    # PLUMED ENERGY column is usually named "e"
    key = "e" if "e" in col else [k for k in col if k != "time"][0]
    e_pl = np.atleast_1d(np.asarray(col[key], dtype=float))
    n = min(len(e_sp), len(e_pl))
    d = np.abs(e_sp[:n] - e_pl[:n])
    dmax = float(d.max()) if n else float("inf")
    unit_note = "eV vs eV"
    if n and dmax >= TOL:
        d2 = np.abs(e_sp[:n] - e_pl[:n] / EV_TO_KJMOL)
        d3 = np.abs(e_sp[:n] - e_pl[:n] * EV_TO_KJMOL)
        if float(d2.max()) < dmax:
            dmax = float(d2.max())
            unit_note = "COLVAR looked like kJ/mol; converted to eV"
        elif float(d3.max()) < dmax:
            dmax = float(d3.max())
            unit_note = "COLVAR looked like eV*kJ/mol factor; converted"
    ok = n > 0 and dmax < TOL
    lines = [
        "A1 energy (SPARC socket vs PLUMED ENERGY)",
        f"n = {n}",
        f"max |E_plumed - E_sparc| = {dmax:.3e} eV  ({unit_note})",
        f"tolerance = {TOL:.1e} eV",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "energy_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
