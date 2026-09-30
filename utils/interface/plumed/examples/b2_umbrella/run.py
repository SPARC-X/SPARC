#!/usr/bin/env python3
"""B2: unbiased vs PLUMED umbrella on Al–Al distance, SPARC socket DFT."""

from __future__ import annotations

import os
import shutil
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
    run_verlet,
    write_history_csv,
)

HERE = Path(__file__).resolve().parent
NSTEPS = int(os.environ.get("NSTEPS", "15"))
DT_FS = float(os.environ.get("DT_FS", "1.0"))
T_K = 300.0


def _format_xyz(atoms) -> str:
    lines = [f"{len(atoms.symbols)}", f"d12={pair_distance(atoms):.8f}"]
    for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
        lines.append(f"{s} {x:18.10f} {y:18.10f} {z:18.10f}")
    return "\n".join(lines) + "\n"


def one_run(tag: str, plumed_dat: Path, port: int) -> None:
    work = HERE / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    shutil.copy(plumed_dat, work / "plumed.dat")

    atoms = load_al_fcc(HERE.parent / "_common" / "init.xyz")
    xyz_lines = []

    def on_step(_n, at):
        xyz_lines.append(_format_xyz(at))

    prev = Path.cwd()
    os.chdir(work)
    try:
        with SparcSocketSession(work, port=port) as sparc:
            with PlumedHandle(
                natoms=len(atoms.symbols),
                timestep_fs=DT_FS,
                temperature_K=T_K,
                plumed_dat=work / "plumed.dat",
                log_file=str(work / "plumed.log"),
            ) as plumed:
                eng = SparcPlumedEngine(sparc, plumed)

                def forces_fn(at):
                    _e, f = eng.evaluate(at)
                    return f

                run_verlet(
                    atoms,
                    forces_fn,
                    nsteps=NSTEPS,
                    timestep_fs=DT_FS,
                    temperature_K=0.0,
                    seed=1,
                    on_step=on_step,
                )
                write_history_csv(work / "history.csv", eng.history)
    finally:
        os.chdir(prev)

    (work / "md.xyz").write_text("".join(xyz_lines))
    shutil.copy(work / "md.xyz", HERE / f"{tag}.xyz")
    if (work / "COLVAR").is_file():
        shutil.copy(work / "COLVAR", HERE / f"{tag}.COLVAR")


def main() -> int:
    one_run("unbiased", HERE / "plumed_unbiased.dat", int(os.environ.get("PORT_U", "32431")))
    one_run("biased", HERE / "plumed_biased.dat", int(os.environ.get("PORT_B", "32432")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
