#!/usr/bin/env python3
"""B2: Al–Al Cartesian distance unbiased vs PLUMED umbrella (API driver)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "plumed"))

from driver import read_colvar

HERE = Path(__file__).resolve().parent
NN = 4.05 / np.sqrt(2.0)
AT = 3.20


def _d12_xyz(path: Path) -> np.ndarray:
    dists = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n = int(line)
        i += 2
        xyz = []
        for _k in range(n):
            parts = lines[i].split()
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1
        a = np.array(xyz)
        dists.append(float(np.linalg.norm(a[0] - a[1])))
    return np.array(dists)


def _d12_colvar(path: Path) -> np.ndarray:
    col = read_colvar(path)
    key = "d" if "d" in col else [k for k in col if k != "time"][0]
    return np.array(col[key], dtype=float)


def main() -> int:
    u = _d12_xyz(HERE / "unbiased.xyz")
    b = _d12_xyz(HERE / "biased.xyz")
    um, bm = float(u.mean()), float(b.mean())
    ok_u = abs(um - NN) < abs(um - AT)
    ok_shift = bm > um + 0.01
    ok = ok_u and ok_shift and len(u) >= 2 and len(b) >= 2
    extra = ""
    colp = HERE / "biased.COLVAR"
    if colp.is_file():
        try:
            c = _d12_colvar(colp)
            extra = f"COLVAR biased mean={float(c.mean()):.4f} Å (NOPBC)\n"
        except Exception:
            extra = ""
    lines = [
        "B2 umbrella (SPARC socket + PLUMED API)",
        f"FCC nn={NN:.3f} Å  restraint AT={AT:.2f} Å",
        f"unbiased mean d12={um:.4f} Å  std={u.std():.4f}  n={len(u)}",
        f"biased   mean d12={bm:.4f} Å  std={b.std():.4f}  n={len(b)}",
        extra.rstrip(),
        f"unbiased nearer to lattice nn than to AT: {'PASS' if ok_u else 'FAIL'}",
        f"biased mean > unbiased + 0.01 Å: {'PASS' if ok_shift else 'FAIL'}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join([x for x in lines if x != ""]) + "\n"
    print(text, end="")
    (HERE / "umbrella_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
