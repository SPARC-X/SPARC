#!/usr/bin/env python3
"""B1: unbiased NVT; print Al–Al distance via PLUMED."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "plumed"))

from driver import (
    PlumedHandle,
    SparcPlumedEngine,
    SparcSocketSession,
    load_al_fcc,
    pair_distance,
    run_nvt,
    write_history_csv,
)

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "32421"))
NSTEPS = int(os.environ.get("NSTEPS", "8"))
DT_FS = float(os.environ.get("DT_FS", "1.0"))
T_K = 300.0


def _format_xyz(atoms) -> str:
    lines = [f"{len(atoms.symbols)}", f"d12={pair_distance(atoms):.8f}"]
    for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
        lines.append(f"{s} {x:18.10f} {y:18.10f} {z:18.10f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    for name in ("COLVAR", "plumed.log", "md.xyz"):
        p = HERE / name
        if p.exists():
            p.unlink()

    atoms = load_al_fcc(HERE.parent / "_common" / "init.xyz")
    xyz_lines = []

    def on_step(_n, at):
        xyz_lines.append(_format_xyz(at))

    with SparcSocketSession(HERE, port=PORT) as sparc:
        with PlumedHandle(
            natoms=len(atoms.symbols),
            timestep_fs=DT_FS,
            temperature_K=T_K,
            plumed_dat=HERE / "plumed.dat",
            log_file=str(HERE / "plumed.log"),
        ) as plumed:
            eng = SparcPlumedEngine(sparc, plumed)

            def forces_fn(at):
                _e, f = eng.evaluate(at)
                return f

            run_nvt(
                atoms,
                forces_fn,
                nsteps=NSTEPS,
                timestep_fs=DT_FS,
                temperature_K=T_K,
                on_step=on_step,
            )
            write_history_csv(HERE / "history.csv", eng.history)

    (HERE / "md.xyz").write_text("".join(xyz_lines))
    print(f"B1 finished, last d12 = {pair_distance(atoms):.4f} Å")
    return 0


if __name__ == "__main__":
    sys.exit(main())
