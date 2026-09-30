#!/usr/bin/env python3
"""E3: i-PI restart produced further MD output."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    log = (HERE / "i-pi.log").read_text(errors="replace") if (HERE / "i-pi.log").is_file() else ""
    out = HERE / "simulation.out"
    rows = 0
    if out.is_file():
        rows = sum(
            1
            for line in out.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    ok_log = "Exiting cleanly" in log or "I-PI reports success" in log
    ok = rows >= 2 and (HERE / "RESTART").is_file()
    lines = [
        "E3 restart",
        f"simulation.out rows={rows}",
        f"i-PI clean exit text: {ok_log}",
        f"overall: {'PASS' if ok else 'FAIL'}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "restart_table.txt").write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
