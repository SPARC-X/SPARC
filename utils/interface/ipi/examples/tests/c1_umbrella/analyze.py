#!/usr/bin/env python3
"""C1/C2: Al–Al distance unbiased vs PLUMED umbrella."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NN = 4.05 / np.sqrt(2.0)  # ~2.863 Å
AT = 3.20


def read_xyz_d12(path: Path):
    dists = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n = int(line)
            _ = f.readline()
            xyz = []
            for _i in range(n):
                parts = f.readline().split()
                xyz.append([float(x) for x in parts[1:4]])
            a = np.array(xyz)
            dists.append(float(np.linalg.norm(a[0] - a[1])))
    return np.array(dists)


def main():
    u = read_xyz_d12(HERE / "unbiased.pos_0.xyz")
    b = read_xyz_d12(HERE / "biased.pos_0.xyz")
    um, bm = float(u.mean()), float(b.mean())
    ok_u = abs(um - NN) < abs(um - AT)
    ok_shift = bm > um + 0.01
    ok = ok_u and ok_shift
    lines = [
        "C1/C2 umbrella",
        f"FCC nn={NN:.3f} Å  restraint AT={AT:.2f} Å",
        f"unbiased mean d12={um:.4f} Å  std={u.std():.4f}  n={len(u)}",
        f"biased   mean d12={bm:.4f} Å  std={b.std():.4f}  n={len(b)}",
        f"unbiased nearer to lattice nn than to AT: {'PASS' if ok_u else 'FAIL'}",
        f"biased mean > unbiased + 0.01 Å: {'PASS' if ok_shift else 'FAIL'}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "umbrella_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
