#!/usr/bin/env python3
"""A2: Å file vs Bohr on the wire vs SPARC lattice print."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
from sparc_files import latest_sparc_static


def ipi_factors():
    from ipi.utils.units import unit_to_user, unit_to_internal

    return {
        "angstrom_per_bohr": unit_to_user("length", "angstrom", 1.0),
        "bohr_per_angstrom": unit_to_internal("length", "angstrom", 1.0),
    }


def parse_cell_abc(comment: str):
    m = re.search(
        r"CELL\(abcABC\):\s*"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        comment,
    )
    if not m:
        raise ValueError(comment)
    return np.array([float(x) for x in m.groups()])


def read_all_xyz(path: Path):
    frames = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        comment = lines[i + 1]
        xyz = np.zeros((n, 3))
        for k in range(n):
            xyz[k] = [float(x) for x in lines[i + 2 + k].split()[1:4]]
        cell = parse_cell_abc(comment)
        pos_u = "atomic_unit"
        cell_u = "atomic_unit"
        mpos = re.search(r"positions\{([^}]+)\}", comment)
        mcell = re.search(r"cell\{([^}]+)\}", comment)
        if mpos:
            pos_u = mpos.group(1)
        if mcell:
            cell_u = mcell.group(1)
        frames.append((cell, xyz, pos_u, cell_u))
        i += 2 + n
    return frames


def read_sparc_frames(path: Path):
    text = path.read_text(errors="replace")
    blocks = text.split("Atom positions")
    frames = []
    for b in blocks[1:]:
        frac = []
        in_frac = False
        for line in b.splitlines():
            if "Fractional coordinates" in line:
                in_frac = True
                continue
            if in_frac:
                if line.strip().startswith("Lattice"):
                    in_frac = False
                    continue
                parts = line.split()
                if len(parts) == 3:
                    frac.append([float(x) for x in parts])
        m = re.search(r"Lattice \(Bohr\):\s*\n\s*([-+0-9.eE]+)", b)
        if m is None or not frac:
            continue
        a = float(m.group(1))
        frac = np.array(frac)
        frames.append((a, frac * a))
    return frames


def wrap(d, L):
    return d - L * np.round(d / L)


def to_bohr(vec, unit, bohr_per_aa):
    if unit in ("atomic_unit", "atomic_units", "bohr"):
        return vec
    if unit == "angstrom":
        return vec * bohr_per_aa
    raise ValueError(unit)


def main():
    u = ipi_factors()
    bohr_per_aa = u["bohr_per_angstrom"]
    lines = ["A2 units sanity", f"directory: {HERE}", ""]
    lines.append(f"  1 Å = {bohr_per_aa:.12f} Bohr")
    lines.append("")

    au = read_all_xyz(HERE / "simulation.pos_au_0.xyz")
    aa = read_all_xyz(HERE / "simulation.pos_aa_0.xyz")
    static = latest_sparc_static(HERE)
    if static is None:
        raise SystemExit("No Al.static* from the SPARC socket run.")
    sp = read_sparc_frames(static)
    lines.append(f"SPARC static: {static.name}  ({len(sp)} frames)")
    lines.append(f"i-PI frames: {len(au)} au, {len(aa)} Å")
    lines.append("")

    n = min(len(au), len(aa), len(sp))
    if n < 1:
        raise SystemExit("No overlapping frames to compare.")

    max_cell_aa_au = 0.0
    max_cell_sp_au = 0.0
    max_pos = 0.0
    table = []
    for i in range(n):
        cell_au, xyz_au, pos_u_au, cell_u_au = au[i]
        cell_aa, xyz_aa, pos_u_aa, cell_u_aa = aa[i]
        a_sp, r_sp = sp[i]
        a_au = float(to_bohr(cell_au, cell_u_au, bohr_per_aa)[0])
        a_aa = float(cell_aa[0])
        a_aa_b = float(to_bohr(cell_aa, cell_u_aa, bohr_per_aa)[0])
        da = a_sp - a_au
        max_cell_aa_au = max(max_cell_aa_au, abs(a_aa_b - a_au))
        max_cell_sp_au = max(max_cell_sp_au, abs(da))
        r_au = to_bohr(xyz_au, pos_u_au, bohr_per_aa)
        d = wrap(r_au - r_sp, a_sp)
        maxdr = float(np.max(np.abs(d)))
        max_pos = max(max_pos, maxdr)
        table.append((i, a_aa, a_au, a_sp, da, maxdr))
        lines.append(
            f"  iter {i}: a_Å={a_aa:.5f}  a_iPI={a_au:.8f}  "
            f"a_SP={a_sp:.9f}  da={da:.3e}  max|dr|={maxdr:.3e}"
        )

    checks = [
        ("cell Å-file vs a.u.-file (both→Bohr)", max_cell_aa_au, 1e-5),
        ("SPARC lattice Bohr vs i-PI pos_au", max_cell_sp_au, 1e-5),
        ("positions i-PI au vs SPARC", max_pos, 1e-4),
    ]
    lines.append("")
    lines.append("Numerical checks:")
    ok_all = True
    for name, val, tol in checks:
        ok = val < tol
        ok_all = ok_all and ok
        lines.append(f"  {'PASS' if ok else 'FAIL'}: {name}: {val:.3e}  (tol {tol:.1e})")

    lines.append("")
    lines.append("On the wire (i-PI driver protocol) everything is atomic units.")
    lines.append("i-PI reads Å from xyz; SPARC prints the Bohr cell it received.")
    lines.append("")
    lines.append(f"overall: {'PASS' if ok_all else 'FAIL'}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "units_table.txt").write_text(text)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
