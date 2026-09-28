"""Parsers and SPARC input writers for A1 energy/force comparison."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PSP_NAME = "13_Al_3_1.9_1.9_pbe_n_v1.0.psp8"

E_TOL_HA = 5.0e-5
F_TOL_HA_BOHR = 1.0e-4

INPT_BODY = """LATVEC:
{latvec}
CELL: {cell[0]:.10f} {cell[1]:.10f} {cell[2]:.10f}
KPOINT_GRID: 1 1 1
KPOINT_SHIFT: 0.5 0.5 0.5
BC: P P P

MESH_SPACING: 0.7

EXCHANGE_CORRELATION: GGA_PBE
TOL_SCF: 1e-5
SMEARING: 0.001
ELEC_TEMP_TYPE: fd
FIX_RAND: 1

MD_FLAG: 0
RELAX_FLAG: 0

PRINT_ATOMS: 1
PRINT_FORCES: 1
PRINT_DENSITY: 0
CALC_STRESS: 1
PRINT_EIGEN: 0
PRINT_ORBITAL: 0
"""


def cellpar_to_cell(a, b, c, alpha, beta, gamma):
    """abcABC (angles in degrees) -> 3x3 row lattice vectors."""
    deg = np.pi / 180.0
    al, be, ga = alpha * deg, beta * deg, gamma * deg
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    czsq = max(c * c - cx * cx - cy * cy, 0.0)
    vc = np.array([cx, cy, np.sqrt(czsq)])
    return np.vstack([va, vb, vc])


def parse_cell_comment(comment: str):
    """Read CELL(abcABC) from an i-PI xyz comment line."""
    m = re.search(
        r"CELL\(abcABC\):\s*"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        comment,
    )
    if not m:
        raise ValueError(f"No CELL(abcABC) in comment: {comment!r}")
    return tuple(float(x) for x in m.groups())


def read_ipi_xyz(path: Path):
    """Return list of dicts: symbols, xyz (n,3), cellpar, comment."""
    path = Path(path)
    frames = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n = int(line)
            comment = f.readline()
            symbols = []
            xyz = np.zeros((n, 3), dtype=float)
            for i in range(n):
                parts = f.readline().split()
                symbols.append(parts[0])
                xyz[i] = [float(x) for x in parts[1:4]]
            frames.append(
                {
                    "symbols": symbols,
                    "xyz": xyz,
                    "cellpar": parse_cell_comment(comment),
                    "comment": comment.strip(),
                }
            )
    return frames


def read_ipi_properties(path: Path):
    """Parse simulation.out; return ndarray with columns as in the file."""
    rows = []
    with Path(path).open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(x) for x in s.split()])
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return np.array(rows, dtype=float)


def parse_sparc_static(path: Path):
    """Parse one or more energy/force blocks from a SPARC .static file.

    Socket runs append one block per request ('socket step N', 1-based).
    A native single-point has a single block.
    """
    text = Path(path).read_text(errors="replace")
    blocks = []
    energy = None
    forces = []
    in_forces = False
    step = None

    def flush():
        nonlocal energy, forces, in_forces, step
        if energy is None:
            return
        blocks.append(
            {
                "step": step,
                "energy": energy,
                "forces": np.array(forces, dtype=float) if forces else None,
            }
        )
        energy = None
        forces = []
        in_forces = False
        step = None

    for raw in text.splitlines():
        line = raw.strip()
        mstep = re.search(r"socket step\s+(\d+)", line, flags=re.I)
        if mstep:
            flush()
            step = int(mstep.group(1))
            continue
        if line.startswith("Total free energy (Ha):"):
            if energy is not None and not mstep:
                # new block without a socket-step header
                flush()
                step = None
            energy = float(line.split(":")[1])
            in_forces = False
            continue
        if line.startswith("Atomic forces"):
            in_forces = True
            forces = []
            continue
        if in_forces:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    forces.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    continue
                except ValueError:
                    in_forces = False
            else:
                in_forces = False
    flush()
    if not blocks:
        raise ValueError(f"No energy blocks in {path}")
    return blocks


def latvec_and_cell_lengths(cell):
    """SPARC LATVEC (unit rows) and CELL (lengths) from a 3x3 lattice."""
    lengths = np.linalg.norm(cell, axis=1)
    latvec = cell / lengths[:, None]
    return latvec, lengths


def write_sparc_ion(path: Path, symbols, xyz, psp_name=PSP_NAME):
    if len(set(symbols)) != 1:
        raise ValueError("A1 writer only handles a single atom type")
    atom = symbols[0]
    lines = [
        f"ATOM_TYPE: {atom}",
        f"N_TYPE_ATOM: {len(symbols)}",
        "COORD:",
    ]
    for row in xyz:
        lines.append(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}")
    lines.append("")
    lines.append(f"PSEUDO_POT: {psp_name}")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def write_sparc_inpt(path: Path, cellpar):
    cell = cellpar_to_cell(*cellpar)
    latvec, lengths = latvec_and_cell_lengths(cell)
    lat_txt = "\n".join(
        f" {latvec[i, 0]:.16f}  {latvec[i, 1]:.16f}  {latvec[i, 2]:.16f}"
        for i in range(3)
    )
    Path(path).write_text(INPT_BODY.format(latvec=lat_txt, cell=lengths))
