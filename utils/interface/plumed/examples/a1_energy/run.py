#!/usr/bin/env python3
"""A1: a few socket single-points; PLUMED ENERGY must match SPARC."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "plumed"))

from driver import PlumedHandle, SparcPlumedEngine, SparcSocketSession, load_al_fcc, write_history_csv

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "32411"))
NIMG = 3
DT_FS = 1.0
T_K = 300.0


def main() -> int:
    for name in ("COLVAR", "plumed.log"):
        p = HERE / name
        if p.exists():
            p.unlink()

    atoms0 = load_al_fcc(HERE.parent / "_common" / "init.xyz")
    images = []
    for i in range(NIMG):
        at = atoms0.copy()
        if i:
            at.rattle(0.04, seed=i)
        images.append(at)

    with SparcSocketSession(HERE, port=PORT) as sparc:
        with PlumedHandle(
            natoms=len(atoms0.symbols),
            timestep_fs=DT_FS,
            temperature_K=T_K,
            plumed_dat=HERE / "plumed.dat",
            log_file=str(HERE / "plumed.log"),
        ) as plumed:
            eng = SparcPlumedEngine(sparc, plumed)
            for at in images:
                eng.evaluate(at)
            write_history_csv(HERE / "history.csv", eng.history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
