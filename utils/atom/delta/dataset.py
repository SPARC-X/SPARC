from __future__ import annotations
import numpy as np
from typing import Dict, Any

def save_delta_npz(path: str, snapshot, ref_v: Dict[str, np.ndarray], other_v: Dict[str, np.ndarray]) -> None:
    """Persist Δ-learning data to NPZ."""
    np.savez_compressed(
        path,
        r=snapshot.r_quad, rho=snapshot.rho,
        vx_ref=ref_v.get("vx"), vc_ref=ref_v.get("vc"),
        vx_other=other_v.get("vx"), vc_other=other_v.get("vc"),
        meta=snapshot.meta
    )
