"""
Convergence criteria and checkers for SCF iterations
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass

# Convergence Checker Error Messages
LOOP_TYPE_ERROR_MESSAGE = \
    "parameter loop_type must be 'Inner' or 'Outer', get {} instead"

@dataclass
class ConvergenceHistory:
    """Track convergence history"""
    iterations: list[int]
    residuals: list[float]
    converged: bool = False
    
    def add(self, iteration: int, residual: float):
        """Add convergence data point"""
        self.iterations.append(iteration)
        self.residuals.append(residual)


class ConvergenceChecker:
    """
    Check SCF convergence based on density residual
    
    Convergence criterion:
        ||ρ_out - ρ_in|| / ||ρ_out|| < tolerance
    
    Requires consecutive satisfaction to avoid false convergence
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        n_consecutive: int = 1,
        norm_type: int = 2,
        loop_type: str = "Inner"
    ):
        """
        Parameters
        ----------
        tolerance : float
            Convergence threshold
        n_consecutive : int
            Number of consecutive iterations that must satisfy criterion
        norm_type : int
            Norm type for residual calculation (1, 2, or np.inf)
        loop_type : str
            Type of loop for display purposes ("Inner" or "Outer")
        """
        assert loop_type in ["Inner", "Outer"], \
            LOOP_TYPE_ERROR_MESSAGE.format(loop_type)
            
        self.tolerance     = tolerance
        self.n_consecutive = n_consecutive
        self.norm_type     = norm_type
        self.loop_type     = loop_type
        
        self._consecutive_count = 0
        self._history = ConvergenceHistory(iterations=[], residuals=[])
    
    
    def check(
        self,
        rho_in: np.ndarray,
        rho_out: np.ndarray,
        iteration: int,
        print_status: bool = False,
        prefix: str = ""
        ) -> tuple[bool, float]:
        """
        Check if SCF has converged
        
        Parameters
        ----------
        rho_in : np.ndarray
            Input density for this iteration
        rho_out : np.ndarray
            Output density from this iteration
        iteration : int
            Current iteration number
        print_status : bool, optional
            If True, print convergence status for this iteration
            Default: False
        prefix : str, optional
            Prefix for printed output (e.g., "  " for indentation)
            Default: ""
        
        Returns
        -------
        converged : bool
            Whether SCF has converged
        residual : float
            Convergence residual for this iteration
        """
        # Compute relative residual
        residual = self._compute_residual(rho_in, rho_out)
        
        # Update history
        self._history.add(iteration, residual)
        
        # Check if residual is below tolerance
        if residual < self.tolerance:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        
        # Converged if satisfied for n_consecutive iterations
        converged = (self._consecutive_count >= self.n_consecutive)
        
        if converged:
            self._history.converged = True
        
        # Print status if requested
        if print_status:
            self.print_status(iteration, residual, prefix)
        
        return converged, residual
    
    
    def _compute_residual(self, rho_in: np.ndarray, rho_out: np.ndarray) -> float:
        """
        Compute relative residual: ||ρ_out - ρ_in|| / ||ρ_out||
        """
        diff = rho_out - rho_in        
        norm_diff = np.linalg.norm(diff, ord=self.norm_type)
        norm_out = np.linalg.norm(rho_out, ord=self.norm_type)
        
        if norm_out < 1e-14:
            return np.inf
        
        return norm_diff / norm_out
    
    
    def reset(self):
        """Reset convergence state"""
        self._consecutive_count = 0
        self._history = ConvergenceHistory(iterations=[], residuals=[])
    
    
    @property
    def history(self) -> ConvergenceHistory:
        """Get convergence history"""
        return self._history
    
    
    def print_header(self, prefix: str = ""):
        """Print table header for convergence status"""
        print(f"{prefix}\t {'Iter':^4}  {'Residual':^14}  {'Target':^14}  {'Converged':^8}")
        print(f"{prefix}{'-'*60}")
    
    
    def print_status(self, iteration: int, residual: float, prefix: str = ""):
        """Print convergence status in table format"""
        status = "Yes" if residual < self.tolerance else "No"
        if self.loop_type == "Outer":
            print(f"{prefix}\t Density residual of outer iteration: {residual:.6e}")
            print()
        else:
            # print(f"[Inner] Iter {iteration:3d}: residual = {residual:.8e} {status}")
            print(f"{prefix}\t {iteration:^4d}  {residual:^14.6e}  {self.tolerance:^14.6e}  {status:^8}")
    
    
    def print_footer(self, converged: bool, n_iterations: int, prefix: str = ""):
        """Print table footer with final status"""
        status_msg = "converged" if converged else "not converged"
        if self.loop_type == "Outer":
            print(f"[Outer] Converged after {n_iterations} iteration(s)")
        else:
            print(f"{prefix}\t SCF Iteration {status_msg} after {n_iterations} iteration(s)")
            print()

