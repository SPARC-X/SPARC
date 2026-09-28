#!/usr/bin/env python3
"""Compare i-PI socket energies/forces to independent SPARC single-points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from a1_io import (
    E_TOL_HA,
    F_TOL_HA_BOHR,
    HERE,
    parse_sparc_static,
    read_ipi_properties,
    read_ipi_xyz,
)

sys.path.insert(0, str(HERE.parent / "_common"))
from sparc_files import latest_sparc_static


def load_ipi(here: Path):
    prop_path = here / "simulation.out"
    pos_path = here / "simulation.pos_0.xyz"
    frc_candidates = sorted(here.glob("simulation.frc*.xyz"))
    if not prop_path.is_file():
        raise SystemExit(f"Missing {prop_path}")
    prop = read_ipi_properties(prop_path)
    # columns: step, time, conserved, temperature, potential(Ha)
    e_ipi = prop[:, 4]
    f_ipi = None
    if frc_candidates:
        f_ipi = [fr["xyz"] for fr in read_ipi_xyz(frc_candidates[0])]
    if not pos_path.is_file():
        raise SystemExit(f"Missing {pos_path}")
    return e_ipi, f_ipi


def load_socket_static(here: Path):
    path = latest_sparc_static(here)
    if path is None:
        return None, None
    return parse_sparc_static(path), path


def load_sp_frames(here: Path, n: int):
    energies = []
    forces = []
    found = []
    for i in range(n):
        path = here / f"sp_frame{i:02d}" / "Al.static"
        if not path.is_file():
            energies.append(None)
            forces.append(None)
            continue
        blocks = parse_sparc_static(path)
        energies.append(blocks[-1]["energy"])
        forces.append(blocks[-1]["forces"])
        found.append(i)
    if not found:
        raise SystemExit(
            "No sp_frameXX/Al.static files. Run python3 C_run_singlepoints.py first."
        )
    return energies, forces, found


def fmt(x, width=14):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return f"{'n/a':>{width}}"
    return f"{x:{width}.6e}"


def force_stats(fa, fb):
    if fa is None or fb is None:
        return None, None
    if fa.shape != fb.shape:
        return None, None
    d = fa - fb
    rmse = float(np.sqrt(np.mean(d * d)))
    maxabs = float(np.max(np.abs(d)))
    return rmse, maxabs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=HERE)
    p.add_argument("--e-tol", type=float, default=E_TOL_HA)
    p.add_argument("--f-tol", type=float, default=F_TOL_HA_BOHR)
    args = p.parse_args()
    here = args.dir.resolve()

    e_ipi, f_ipi = load_ipi(here)
    n = len(e_ipi)
    socket, socket_path = load_socket_static(here)
    e_sp, f_sp, sp_found = load_sp_frames(here, n)

    e_sock = [None] * n
    f_sock = [None] * n
    if socket:
        for blk in socket:
            if blk["step"] is None:
                continue
            idx = blk["step"] - 1
            if 0 <= idx < n:
                e_sock[idx] = blk["energy"]
                f_sock[idx] = blk["forces"]
        if all(x is None for x in e_sock) and len(socket) == n:
            for i, blk in enumerate(socket):
                e_sock[i] = blk["energy"]
                f_sock[i] = blk["forces"]

    header = (
        f"{'frame':>5}  {'E_ipi':>14}  {'E_socket':>14}  {'E_sp':>14}  "
        f"{'dE_ipi-sp':>12}  {'Fmax_ipi-sp':>12}  {'Fmax_sock-sp':>12}"
    )
    lines = [
        "A1 energy / force consistency",
        f"directory: {here}",
        f"tolerances: |dE| < {args.e_tol:.1e} Ha,  max|dF| < {args.f_tol:.1e} Ha/Bohr",
    ]
    if socket_path is not None:
        lines.append(f"socket static: {socket_path.name}")
    lines += [
        "",
        header,
        "-" * len(header),
    ]

    e_diffs = []
    f_diffs = []
    f_sock_diffs = []

    for i in range(n):
        de = None
        if e_sp[i] is not None:
            de = e_ipi[i] - e_sp[i]
            e_diffs.append(abs(de))
        fi = f_ipi[i] if f_ipi is not None and i < len(f_ipi) else None
        _, fmax_sp = force_stats(fi, f_sp[i])
        _, fmax_sock_sp = force_stats(f_sock[i], f_sp[i])
        if fmax_sp is not None:
            f_diffs.append(fmax_sp)
        if fmax_sock_sp is not None:
            f_sock_diffs.append(fmax_sock_sp)
        lines.append(
            f"{i:5d}  {fmt(e_ipi[i])}  {fmt(e_sock[i])}  {fmt(e_sp[i])}  "
            f"{fmt(de, 12)}  {fmt(fmax_sp, 12)}  {fmt(fmax_sock_sp, 12)}"
        )

    lines.append("")
    if e_diffs:
        lines.append(f"max |E_ipi - E_sp|          = {max(e_diffs):.6e} Ha")
    if f_diffs:
        lines.append(f"max |F_ipi - F_sp|          = {max(f_diffs):.6e} Ha/Bohr")
    if f_sock_diffs:
        lines.append(f"max |F_socket - F_sp|       = {max(f_sock_diffs):.6e} Ha/Bohr")
    if socket and e_sock[0] is not None:
        de_is = [
            abs(e_ipi[i] - e_sock[i])
            for i in range(n)
            if e_sock[i] is not None
        ]
        if de_is:
            lines.append(f"max |E_ipi - E_socket|      = {max(de_is):.6e} Ha")

    ok_e = bool(e_diffs) and max(e_diffs) < args.e_tol
    ok_f = bool(f_diffs) and max(f_diffs) < args.f_tol
    if f_ipi is None:
        lines.append("WARNING: no simulation.frc_0.xyz; force vs i-PI skipped")
        ok_f = False

    lines.append("")
    lines.append(f"energy vs independent SPARC SP : {'PASS' if ok_e else 'FAIL'}")
    lines.append(f"force  vs independent SPARC SP : {'PASS' if ok_f else 'FAIL'}")
    if not sp_found:
        lines.append("FAIL: no single-point frames")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    (here / "compare_table.txt").write_text(text)

    return 0 if (ok_e and ok_f) else 1


if __name__ == "__main__":
    sys.exit(main())
