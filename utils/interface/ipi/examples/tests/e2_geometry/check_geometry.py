#!/usr/bin/env python3
"""E2: i-PI xyz vs SPARC socket-printed geometries (Bohr)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A1 = HERE.parent / "a1_energy_force"
sys.path.insert(0, str(HERE.parent / "_common"))
from sparc_files import latest_sparc_static


def read_ipi_xyz(path: Path):
    frames = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line)
            comment = f.readline()
            xyz = np.zeros((n, 3))
            for i in range(n):
                p = f.readline().split()
                xyz[i] = [float(x) for x in p[1:4]]
            m = re.search(
                r"CELL\(abcABC\):\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
                comment,
            )
            cell = np.array([float(x) for x in m.groups()]) if m else None
            frames.append((xyz, cell))
    return frames


def parse_sparc_static(path: Path):
    text = path.read_text(errors="replace")
    blocks = []
    frac, lat, step = [], None, None
    in_frac = False
    in_lat = False
    lat_rows = []
    for raw in text.splitlines():
        line = raw.strip()
        m = re.search(r"socket step\s+(\d+)", line, flags=re.I)
        if m:
            if frac and lat is not None:
                blocks.append((step, np.array(frac), lat))
            step = int(m.group(1))
            frac, lat_rows, in_frac, in_lat = [], [], False, False
            continue
        if line.startswith("Fractional coordinates"):
            in_frac, in_lat = True, False
            continue
        if line.startswith("Lattice"):
            in_frac, in_lat = False, True
            lat_rows = []
            continue
        if in_frac:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    frac.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    continue
                except ValueError:
                    in_frac = False
            else:
                in_frac = False
        if in_lat:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    lat_rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(lat_rows) == 3:
                        lat = np.array(lat_rows)
                        in_lat = False
                    continue
                except ValueError:
                    in_lat = False
            else:
                in_lat = False
    if frac and lat is not None:
        blocks.append((step, np.array(frac), lat))
    return blocks


def wrap_diff(a, b, cell_len):
    d = a - b
    d -= np.round(d / cell_len) * cell_len
    return d


def main():
    pos = A1 / "simulation.pos_0.xyz"
    static = latest_sparc_static(A1)
    if not pos.is_file() or static is None:
        raise SystemExit("Run a1_energy_force first (need pos xyz + Al.static*).")
    ipi = read_ipi_xyz(pos)
    sp = parse_sparc_static(static)
    n = min(len(ipi), len(sp))
    maxp, maxc = 0.0, 0.0
    for i in range(n):
        xyz, cell = ipi[i]
        _step, frac, lat = sp[i]
        cart = frac @ lat
        L = np.linalg.norm(lat, axis=1)
        d = wrap_diff(xyz, cart, L)
        maxp = max(maxp, float(np.max(np.abs(d))))
        if cell is not None:
            maxc = max(maxc, float(np.max(np.abs(cell - L))))
    ok = maxp < 1e-4 and maxc < 1e-5
    lines = [
        "E2 geometry",
        f"SPARC static: {static.name}",
        f"frames compared={n}",
        f"max |r_ipi - r_sparc| (wrapped, Bohr) = {maxp:.3e}",
        f"max |cell_ipi - cell_sparc| (Bohr) = {maxc:.3e}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "geometry_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
