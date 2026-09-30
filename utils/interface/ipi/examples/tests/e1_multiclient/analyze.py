#!/usr/bin/env python3
"""E1: two clients connected; PIMD steps completed."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    log = (HERE / "i-pi.log").read_text(errors="replace")
    n_hs = log.count("Handshaking was successful")
    n_assign = log.count("Assigning")
    rows = 0
    out = HERE / "simulation.out"
    if out.is_file():
        rows = sum(
            1
            for line in out.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    ok = n_hs >= 2 and rows >= 6
    lines = [
        "E1 multi-client",
        f"successful handshakes={n_hs}  force assignments={n_assign}  out rows={rows}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "multiclient_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
