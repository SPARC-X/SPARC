#!/usr/bin/env python3
"""Write native SPARC single-point jobs for each i-PI trajectory frame and run them."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from a1_io import HERE, PSP_NAME, read_ipi_xyz, write_sparc_inpt, write_sparc_ion


def default_sparc():
    repo = HERE.parents[5]  # SPARC repository root
    return os.environ.get("SPARC", str(repo / "lib" / "sparc"))


def default_psp():
    local = HERE / PSP_NAME
    if local.is_file():
        return local
    return HERE.parents[5] / "psps" / PSP_NAME


def run_frame(workdir: Path, sparc: str, np_proc: int):
    log = workdir / "sparc.log"
    mpi = [tok for tok in os.environ.get("MPIEXEC", "mpirun -np").split() if tok]
    cmd = mpi + [str(np_proc), sparc, "-name", "Al"]
    print("  ", " ".join(cmd), f"(cwd={workdir.name})")
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    if proc.returncode != 0:
        raise SystemExit(
            f"SPARC single-point failed in {workdir} (exit {proc.returncode}). "
            f"See {log}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pos",
        type=Path,
        default=HERE / "simulation.pos_0.xyz",
        help="i-PI position trajectory (Bohr)",
    )
    p.add_argument("--np", type=int, default=int(os.environ.get("NP", "4")))
    p.add_argument("--sparc", default=default_sparc())
    p.add_argument("--psp", type=Path, default=default_psp())
    p.add_argument(
        "--frames",
        default="all",
        help="Comma-separated frame indices, or 'all'",
    )
    args = p.parse_args()

    if not args.pos.is_file():
        raise SystemExit(
            f"Missing {args.pos}. Run A_run_ipi.sh + B_run_sparc.sh first "
            "(or ./run_all.sh)."
        )
    if not Path(args.sparc).is_file():
        raise SystemExit(f"SPARC binary not found: {args.sparc}")
    if not args.psp.is_file():
        raise SystemExit(f"Pseudopotential not found: {args.psp}")

    frames = read_ipi_xyz(args.pos)
    if args.frames == "all":
        indices = list(range(len(frames)))
    else:
        indices = [int(x) for x in args.frames.split(",") if x.strip()]

    print(f"Independent SPARC single-points for {len(indices)} frame(s)")
    for i in indices:
        fr = frames[i]
        workdir = HERE / f"sp_frame{i:02d}"
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir()
        write_sparc_ion(workdir / "Al.ion", fr["symbols"], fr["xyz"])
        write_sparc_inpt(workdir / "Al.inpt", fr["cellpar"])
        shutil.copy(args.psp, workdir / PSP_NAME)
        run_frame(workdir, args.sparc, args.np)
        static = workdir / "Al.static"
        if not static.is_file():
            raise SystemExit(f"No {static} after SPARC run")
        print(f"  frame {i}: wrote {static}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
