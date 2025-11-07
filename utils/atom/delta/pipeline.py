from __future__ import annotations
from typing import Tuple, Dict, Any

class DeltaLearningPipeline:
    """High-level Δ-learning pipeline at fixed density snapshots. Placeholder."""
    def __init__(self, solver, xc_evaluator):
        self.solver = solver
        self.xc_eval = xc_evaluator

    def run_once(self, xc_ref: str, xc_other: str, include_tau: bool = False) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Return (snapshot, ref_v, other_v, delta)."""
        raise NotImplementedError
