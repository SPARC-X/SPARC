#!/usr/bin/env python3
"""B2: NVT mean temperature after a short equilibration window."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGET = 300.0
EQ = 10
T_TOL = 150.0
DE_TOL = 0.1


def main():
    path = HERE / "simulation.out"
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    t_ps, cons, temp = a[:, 1], a[:, 2], a[:, 3]
    prod = temp[EQ:] if len(temp) > EQ else temp
    tmean = float(prod.mean())
    tstd = float(prod.std())
    dcons = float(cons[-1] - cons[0])
    ok_t = abs(tmean - TARGET) < T_TOL
    ok_c = abs(dcons) < DE_TOL
    ok_f = tstd > 1.0
    ok = ok_t and ok_c and ok_f
    lines = [
        "B2 NVT",
        f"nsteps={len(temp)}  discard first {EQ}",
        f"T_mean={tmean:.1f} K  T_std={tstd:.1f} K  target={TARGET:.0f} K",
        f"Δconserved={dcons:.4e} Ha",
        f"T within ±{T_TOL:.0f} K: {'PASS' if ok_t else 'FAIL'}",
        f"conserved stable: {'PASS' if ok_c else 'FAIL'}",
        f"T fluctuates: {'PASS' if ok_f else 'FAIL'}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "nvt_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
