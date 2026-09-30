"""Atomic configuration and the unit conversions shared by the driver.

Python and PLUMED see eV, Å, and fs. The SPARC socket wire is Hartree and Bohr.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# Socket wire, matching src/socket/driver.h.
HARTREE_TO_EV = 27.211386024367243
BOHR_TO_ANG = 0.52917721067
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG

# PLUMED's internal units are kJ/mol, nm, and ps.
EV_TO_KJMOL = 96.4853321233100184
ANG_TO_NM = 0.1
FS_TO_PS = 0.001
KB_EV = 8.617333262145e-5  # eV/K

AL_MASS_AMU = 26.9815385
# (eV/Å)/amu -> Å/fs^2
EV_AMU_TO_ANG_FS2 = 0.009648533593658412


class Structure:
    """Positions, cell, masses, and velocities. Lengths are in Å."""

    def __init__(
        self,
        symbols: Iterable[str],
        positions: np.ndarray,
        cell: np.ndarray,
        masses: Optional[np.ndarray] = None,
    ) -> None:
        self.symbols = list(symbols)
        self.positions = np.array(positions, dtype=np.float64)
        self.cell = np.array(cell, dtype=np.float64)
        n = len(self.symbols)
        if masses is None:
            self.masses = np.full(n, AL_MASS_AMU, dtype=np.float64)
        else:
            self.masses = np.array(masses, dtype=np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64)

    def copy(self) -> "Structure":
        """Return an independent copy, including velocities."""
        other = Structure(self.symbols, self.positions.copy(), self.cell.copy(), self.masses.copy())
        other.velocities = self.velocities.copy()
        return other

    def get_positions(self) -> np.ndarray:
        """Cartesian positions in Å, shape (n, 3)."""
        return self.positions

    def get_masses(self) -> np.ndarray:
        """Atomic masses in amu."""
        return self.masses

    def get_cell(self) -> np.ndarray:
        """Cell vectors in Å, stored as rows, shape (3, 3)."""
        return self.cell

    def rattle(self, stdev: float, seed: int) -> None:
        """Add Gaussian noise of the given standard deviation (Å) to positions."""
        rng = np.random.default_rng(seed)
        self.positions += rng.normal(0.0, stdev, self.positions.shape)


def pair_distance(atoms: Structure, i: int = 0, j: int = 1) -> float:
    """Distance in Å between atoms i and j, without the minimum-image convention."""
    return float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))


def write_history_csv(path: Path, history: dict) -> None:
    """Write a column-oriented dict of equal-length sequences as CSV."""
    keys = list(history.keys())
    n = len(history[keys[0]]) if keys else 0
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in range(n):
            writer.writerow([history[key][row] for key in keys])


def read_colvar(path: Path) -> Dict[str, np.ndarray]:
    """Read a PLUMED COLVAR file. The ``#! FIELDS`` line supplies the column names."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    fields: List[str] = []
    rows = []
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith("#!"):
                if "FIELDS" in text:
                    fields = text.split("FIELDS", 1)[1].split()
                continue
            if text.startswith("#"):
                continue
            rows.append([float(item) for item in text.split()])
    data = np.array(rows, dtype=float) if rows else np.zeros((0, len(fields)))
    if not fields:
        fields = [f"c{i}" for i in range(data.shape[1])]
    return {name: data[:, i] for i, name in enumerate(fields)}
