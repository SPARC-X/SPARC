#!/usr/bin/env python3
"""B1: conserved-quantity drift vs timestep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DTS = ("0.5", "1.0", "2.0")


def load(dt: str):
    path = HERE / f"out_dt{dt}.dat"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(x) for x in s.split()])
    a = np.array(rows)
    # step, time_ps, conserved, T, pot, kin
    return a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def main():
    lines = ["B1 NVE conserved drift", ""]
    lines.append(
        f"{'dt_fs':>6}  {'n':>4}  {'dE_cons_Ha':>12}  {'slope_Ha/ps':>12}  "
        f"{'T_mean':>8}  {'T_std':>8}"
    )
    slopes = {}
    deltas = {}
    ok = True
    for dt in DTS:
        t, cons, temp, pot, kin = load(dt)
        dE = float(cons[-1] - cons[0])
        slope = float(np.polyfit(t, cons, 1)[0]) if len(t) > 1 else 0.0
        slopes[dt] = slope
        deltas[dt] = dE
        if abs(dE) >= 0.05:
            ok = False
        lines.append(
            f"{dt:>6}  {len(t):4d}  {dE:12.4e}  {slope:12.4e}  "
            f"{temp.mean():8.1f}  {temp.std():8.1f}"
        )
    lines.append("")
    lines.append("Potential + kinetic should trade off; conserved is their extended sum.")
    if abs(slopes["2.0"]) + 1e-12 < 0.2 * abs(slopes["0.5"]):
        lines.append(
            "NOTE: |drift| at 2 fs is much smaller than at 0.5 fs; "
            "SCF noise likely dominates this short run."
        )
    lines.append("")
    lines.append(f"overall: {'PASS' if ok else 'FAIL'}  (|Δconserved| < 0.05 Ha)")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "nve_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
