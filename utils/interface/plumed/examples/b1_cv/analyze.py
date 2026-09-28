#!/usr/bin/env python3
"""B1: COLVAR distance should sit near the FCC nearest-neighbour length."""

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


def main() -> int:
    col = read_colvar(HERE / "COLVAR")
    key = "d" if "d" in col else [k for k in col if k != "time"][0]
    d = np.array(col[key], dtype=float)
    mean = float(d.mean()) if len(d) else float("nan")
    ok_n = len(d) >= 2
    ok_nn = abs(mean - NN) < abs(mean - AT)
    ok = ok_n and ok_nn
    lines = [
        "B1 unbiased CV",
        f"FCC nn={NN:.3f} Å",
        f"n={len(d)}  mean d12={mean:.4f} Å  std={float(d.std()) if len(d) else float('nan'):.4f}",
        f"COLVAR rows >= 2: {'PASS' if ok_n else 'FAIL'}",
        f"mean nearer to lattice nn than to 3.20 Å: {'PASS' if ok_nn else 'FAIL'}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "cv_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
