"""Pick SPARC output files when a rerun rotates names (Al.static → Al.static_01)."""

from __future__ import annotations

import re
from pathlib import Path

_STATIC_NAME = re.compile(r"^Al\.static(?:_(\d+))?$")


def latest_sparc_static(here: Path) -> Path | None:
    """Newest Al.static / Al.static_NN in *here*.

    Prefers modification time; if times tie, the higher SPARC rotation
    index wins (``Al.static_01`` over ``Al.static``).
    """
    cands = [p for p in here.iterdir() if p.is_file() and _STATIC_NAME.match(p.name)]
    if not cands:
        return None

    def key(path: Path):
        m = _STATIC_NAME.match(path.name)
        idx = int(m.group(1)) if m and m.group(1) else 0
        return (path.stat().st_mtime, idx)

    return max(cands, key=key)
