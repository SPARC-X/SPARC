#!/usr/bin/env python3
"""A0: one SPARC socket single-point (DFT only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "plumed"))

from driver import SparcSocketSession, load_al_fcc, write_history_csv

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "32410"))


def main() -> int:
    atoms = load_al_fcc(HERE.parent / "_common" / "init.xyz")
    with SparcSocketSession(HERE, port=PORT) as sparc:
        e, f = sparc.get_energy_forces(atoms)
    ok = np.isfinite(e) and np.isfinite(f).all()
    text = [
        "A0 socket smoke",
        f"energy = {e:.8f} eV",
        f"max |F| = {np.max(np.abs(f)):.6e} eV/Å",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    out = "\n".join(text) + "\n"
    print(out, end="")
    (HERE / "smoke_table.txt").write_text(out)
    write_history_csv(
        HERE / "history.csv",
        {"step": [0], "energy_dft": [e], "max_force": [float(np.max(np.abs(f)))]},
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
