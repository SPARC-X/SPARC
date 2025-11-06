from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from .snapshot import DensitySnapshot

class SnapshotBuilder:
    """Build DensitySnapshot from SCF state. Placeholder."""
    def __init__(self, ops_dict: Dict[str, Any]):
        self.ops = ops_dict

    def build(self, Z: int, rho: np.ndarray, eigvals: Optional[np.ndarray] = None,
              eigvecs: Optional[np.ndarray] = None, meta: Optional[Dict[str, Any]] = None,
              include_tau: bool = False) -> DensitySnapshot:
        r = self.ops.get("r_quad"); w = self.ops.get("w_quad")
        D = self.ops.get("D")
        if r is None or w is None or D is None:
            raise RuntimeError("Operators dict missing r_quad/w_quad/D.")
        # Minimal grad_rho (exact formula depends on your D operator)
        grad_rho = None
        tau = None
        V_H = None
        return DensitySnapshot(
            Z=Z, r_quad=r, w_quad=w, rho=rho, grad_rho=grad_rho, tau=tau,
            V_H=V_H, D_handle=D, meta=meta or {}, eigvals=eigvals, eigvecs=eigvecs
        )
