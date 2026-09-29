#!/usr/bin/env python3
"""B3: NPT volume response and finite pressure/temperature."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
from sparc_files import latest_sparc_static


def sparc_pressure_gpa(static_text: str):
    """SPARC pres = -trace(stress)/3. The printed block is the stress tensor."""
    match = re.search(r"Stress \(GPa\):\s*\n(.*)\n(.*)\n(.*)", static_text)
    if match is None:
        return None
    rows = []
    for line in match.groups():
        nums = [float(x) for x in line.split()]
        if len(nums) != 3:
            return None
        rows.append(nums)
    return -(rows[0][0] + rows[1][1] + rows[2][2]) / 3.0


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
    vol = a[:, 5]
    pres = a[:, 6]
    temp = a[:, 3]
    v0 = float(vol[0])
    vtail = float(vol[-10:].mean()) if len(vol) >= 10 else float(vol[-1])
    vstd = float(vol.std())
    finite = np.all(np.isfinite(vol)) and np.all(vol > 0) and np.all(np.isfinite(temp))
    moved = vstd / max(v0, 1e-12) > 0.01
    static_path = latest_sparc_static(HERE)
    static = static_path.read_text(errors="replace") if static_path is not None else None
    has_stress = static is not None and "Stress (GPa)" in static
    p0 = float(pres[0])
    p_sparc = sparc_pressure_gpa(static) if static is not None else None
    # Kinetic stress at the first step is far below the ~10 GPa electronic pressure.
    sign_ok = p_sparc is not None and abs(p0 - p_sparc) < 2.0
    ok = finite and moved and has_stress and sign_ok
    lines = [
        "B3 NPT",
        f"nsteps={len(vol)}  V0={v0:.4f} Å^3  V_tail={vtail:.4f} Å^3  V_std={vstd:.4f}",
        f"P_md[0]={p0:.3f} GPa  P_sparc={p_sparc if p_sparc is None else round(p_sparc, 3)} GPa  "
        f"P_md mean={float(pres.mean()):.3f} GPa  T_mean={float(temp.mean()):.1f} K",
        f"volume finite/positive: {'PASS' if finite else 'FAIL'}",
        f"volume fluctuates (barostat active): {'PASS' if moved else 'FAIL'}",
        f"SPARC printed stress: {'PASS' if has_stress else 'FAIL'}",
        f"i-PI pressure matches SPARC pres=-trace(stress)/3: {'PASS' if sign_ok else 'FAIL'}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "npt_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
