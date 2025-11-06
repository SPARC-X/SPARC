from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any, Dict

@dataclass(frozen=True)
class DensitySnapshot:
    """Immutable description of fixed electron density and needed context."""
    Z: int
    r_quad: np.ndarray
    w_quad: np.ndarray
    rho: np.ndarray
    grad_rho: Optional[np.ndarray]
    tau: Optional[np.ndarray]
    V_H: Optional[np.ndarray]
    D_handle: Any
    meta: Dict[str, Any]
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path, Z=self.Z, r=self.r_quad, w=self.w_quad, rho=self.rho,
            grad_rho=self.grad_rho, tau=self.tau, V_H=self.V_H, meta=self.meta,
            eigvals=self.eigvals, eigvecs=self.eigvecs
        )
