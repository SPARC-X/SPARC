#!/usr/bin/env python3
"""A3: SPARC should treat i-PI EXIT as a clean shutdown."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    log = (HERE / "sparc.log").read_text(errors="replace")
    rc = os.environ.get("SPARC_RC", "")
    unknown = "Getting an unknown message from server" in log
    clean = "Socket server requested EXIT" in log
    lines = [
        "A3 EXIT shutdown",
        f"SPARC_RC={rc!r}",
        f"clean EXIT message present: {clean}",
        f"unknown-message error present: {unknown}",
    ]
    if clean and not unknown:
        lines.append("overall: PASS")
        rc_out = 0
    elif unknown and not clean:
        lines.append(
            "overall: FAIL (binary still old, or EXIT header not recognised)."
        )
        lines.append(
            "Source already handles EXIT in driver.c; rebuild lib/sparc and re-run."
        )
        rc_out = 1
    else:
        lines.append("overall: FAIL (ambiguous log)")
        rc_out = 1
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "exit_check.txt").write_text(text)
    return rc_out


if __name__ == "__main__":
    sys.exit(main())
