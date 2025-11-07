from __future__ import annotations
import numpy as np
from typing import Tuple, Any

class OEPBuilder:
    """Build OEP exchange/correlation potentials from eigenstates. Placeholder."""
    def __init__(self, lamda: float = 1.0):
        self.lamda = float(lamda)

    def build(self, eigvals: np.ndarray, eigvecs: np.ndarray, meta: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Vx_OEP, Vc_OEP) on quadrature grid."""
        raise NotImplementedError("OEP builder not implemented yet.")
