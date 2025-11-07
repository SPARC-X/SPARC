from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

class PostProcessor:
    """Produce uniform r-grid orbitals and info dict. Placeholder."""
    def __init__(self, mesh_spacing: float = 0.1):
        self.mesh_spacing = float(mesh_spacing)

    def make_outputs(self, R: float, eigvecs_FE: np.ndarray, occ_nl: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return (r_array, orbitals, n_l_orbitals, info_dict)."""
        # TODO: interpolate from FE grid to uniform grid
        raise NotImplementedError
