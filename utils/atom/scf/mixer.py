from __future__ import annotations
import numpy as np
from typing import Optional

class Mixer:
    """
    Density mixing for SCF convergence.
    
    Supports:
    - Simple linear mixing: rho_new = rho_out (pulay_mixing=0)
    - Pulay mixing: DIIS-like method using history of residuals
    - Alternating linear mixing: odd/even steps with different coefficients
    
    Parameters
    ----------
    tol : float
        Convergence tolerance (not used in mixing itself)
    alpha_lin : tuple of float
        Linear mixing parameters (alpha21, alpha22) for odd/even steps
    alpha_pulay : float
        Pulay mixing parameter (alpha11 in original code)
    history : int
        Number of previous iterations to keep for Pulay mixing
    frequency : int
        Apply Pulay mixing every 'frequency' iterations
    pulay_mixing : int
        0: simple linear (rho_in = rho_out)
        1: use Pulay + alternating linear mixing
    un_zero_rho : float
        Minimum density value to avoid negative densities
    """
    
    def __init__(
        self, 
        tol: float, 
        alpha_lin=(0.5, 0.5), 
        alpha_pulay=0.55, 
        history=7, 
        frequency=2,
        pulay_mixing=1,
        un_zero_rho=1e-12
    ):
        self.tol = float(tol)
        self.alpha21 = float(alpha_lin[0])  # Odd step mixing
        self.alpha22 = float(alpha_lin[1])  # Even step mixing
        self.alpha11 = float(alpha_pulay)    # Pulay mixing parameter
        self.history = int(history)
        self.frequency = int(frequency)
        self.pulay_mixing = int(pulay_mixing)
        self.un_zero_rho = float(un_zero_rho)
        
        # History storage: (N_q, history+1)
        self.rho_in_store: Optional[np.ndarray] = None
        self.rho_out_store: Optional[np.ndarray] = None
        self.iteration_count = 0
        self.n_points = None
    
    
    def reset(self) -> None:
        """
        Reset the mixer state.
        
        Clears all history buffers and resets iteration counter.
        Called at the start of each SCF calculation.
        """
        self.rho_in_store = None
        self.rho_out_store = None
        self.iteration_count = 0
        self.n_points = None
    
    
    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """
        Mix input and output densities to produce next input density.
        
        Implements the exact logic from the original code, including:
        - History management with rolling window
        - Pulay mixing every 'frequency' iterations
        - Alternating linear mixing (odd/even steps)
        - Negative density correction
        
        Parameters
        ----------
        rho_in : np.ndarray
            Input density for current iteration, shape (n_points,)
        rho_out : np.ndarray
            Output density from KS solve, shape (n_points,)
        
        Returns
        -------
        rho_next : np.ndarray
            Mixed density for next iteration
        
        Notes
        -----
        This method stores rho_out internally and returns the next rho_in.
        The caller should update the SCF loop accordingly.
        """
        # Initialize history storage on first call
        if self.rho_in_store is None:
            self.n_points = rho_in.shape[0]
            self.rho_in_store = np.zeros((self.n_points, self.history + 1))
            self.rho_out_store = np.zeros((self.n_points, self.history + 1))
            self.rho_in_store[:, 0] = rho_in
            self.iteration_count = 0
        
        runs = self.iteration_count
        
        # Store rho_out in appropriate column
        if runs < self.history:
            self.rho_out_store[:, runs] = rho_out
        else:
            self.rho_out_store[:, -1] = rho_out
        
        # Decide on next rho_in based on mixing strategy
        if self.pulay_mixing == 0:
            # Simple mixing: rho_in = rho_out
            if runs < self.history:
                rho_next = self.rho_out_store[:, runs].copy()
            else:
                rho_next = self.rho_out_store[:, -1].copy()
        
        else:
            # Pulay + alternating linear mixing
            rho_next = self._hybrid_mix(runs)
        
        # Store next rho_in for next iteration
        if runs + 1 < self.history:
            self.rho_in_store[:, runs + 1] = rho_next
        else:
            # Roll history and store in last column
            if runs + 1 > self.history:
                self.rho_in_store = np.roll(self.rho_in_store, -1, axis=1)
                self.rho_out_store = np.roll(self.rho_out_store, -1, axis=1)
            self.rho_in_store[:, -1] = rho_next
        
        self.iteration_count += 1
        return rho_next
    
    
    def _hybrid_mix(self, runs: int) -> np.ndarray:
        """
        Hybrid mixing: Pulay every 'frequency' iterations, linear otherwise.
        
        Parameters
        ----------
        runs : int
            Current iteration number (before increment in reference code)
        
        Returns
        -------
        rho_next : np.ndarray
            Mixed density
        
        Notes
        -----
        Reference code increments 'runs' before mixing, so we add 1 to match:
        - runs=0 (here) -> runs=1 (ref): linear (1%2=1)
        - runs=1 (here) -> runs=2 (ref): Pulay (2%2=0)
        - runs=2 (here) -> runs=3 (ref): linear (3%2=1)
        - runs=3 (here) -> runs=4 (ref): Pulay (4%2=0)
        """
        use_pulay = ((runs + 1) % self.frequency == 0) and (runs > 0)
        
        if runs < self.history:
            # Early phase: direct indexing
            if use_pulay:
                rho_next = self._pulay_mix_early(runs)
            else:
                rho_next = self._linear_mix_early(runs)
        else:
            # Stable phase: use rolling window (last column)
            if use_pulay:
                rho_next = self._pulay_mix_stable()
            else:
                rho_next = self._linear_mix_stable()
        
        # Correct negative densities
        negative_mask = rho_next <= 0
        rho_next[negative_mask] = self.un_zero_rho
        
        return rho_next
    
    
    def _linear_mix_early(self, runs: int) -> np.ndarray:
        """Linear mixing for early phase (runs < history)"""
        # Alternating coefficients for odd/even steps
        # Reference code increments before mixing, so:
        # runs=0 (here) -> runs=1 (ref, odd) -> use alpha21
        # runs=1 (here) -> runs=2 (ref, even) -> use alpha22
        if (runs + 1) % 2 != 0:  # equivalent to: runs % 2 == 0
            alpha = self.alpha21
        else:
            alpha = self.alpha22
        
        rho_in_prev = self.rho_in_store[:, runs]
        rho_out_prev = self.rho_out_store[:, runs]
        
        return rho_in_prev + alpha * (rho_out_prev - rho_in_prev)
    
    
    def _linear_mix_stable(self) -> np.ndarray:
        """Linear mixing for stable phase (runs >= history)"""
        runs = self.iteration_count
        
        # Alternating coefficients (matching reference after increment)
        if (runs + 1) % 2 != 0:  # equivalent to: runs % 2 == 0
            alpha = self.alpha21
        else:
            alpha = self.alpha22
        
        rho_in_prev = self.rho_in_store[:, -2]
        rho_out_prev = self.rho_out_store[:, -2]
        
        return rho_in_prev + alpha * (rho_out_prev - rho_in_prev)
    
    
    def _pulay_mix_early(self, runs: int) -> np.ndarray:
        """Pulay mixing for early phase (runs < history)"""
        # Compute residuals using differences
        rho_in_residual = np.diff(self.rho_in_store[:, :runs+1], axis=1)
        rho_out_residual = np.diff(self.rho_out_store[:, :runs+1], axis=1)
        rho_out_minus_in_residual = rho_out_residual - rho_in_residual
        
        # Pulay projection matrix formula
        # P = alpha11*I - (Δρ_in + alpha11*ΔF) @ (ΔF^T @ ΔF)^(-1) @ ΔF^T
        N_q = self.n_points
        alpha11 = self.alpha11
        
        # ΔF^T @ ΔF
        A = rho_out_minus_in_residual.T @ rho_out_minus_in_residual
        A_inv = np.linalg.inv(A)
        
        # Projection matrix
        term1 = rho_in_residual + alpha11 * rho_out_minus_in_residual
        term2 = term1 @ A_inv @ rho_out_minus_in_residual.T
        P = alpha11 * np.eye(N_q) - term2
        
        # Apply to residual
        rho_in_prev = self.rho_in_store[:, runs]
        rho_out_prev = self.rho_out_store[:, runs]
        residual = rho_out_prev - rho_in_prev
        
        return rho_in_prev + P @ residual
    
    
    def _pulay_mix_stable(self) -> np.ndarray:
        """Pulay mixing for stable phase (runs >= history)"""
        # Compute residuals using differences (all history columns)
        rho_in_residual = np.diff(self.rho_in_store, axis=1)
        rho_out_residual = np.diff(self.rho_out_store, axis=1)
        rho_out_minus_in_residual = rho_out_residual - rho_in_residual
        
        # Pulay projection
        N_q = self.n_points
        alpha11 = self.alpha11
        
        A = rho_out_minus_in_residual.T @ rho_out_minus_in_residual
        A_inv = np.linalg.inv(A)
        
        term1 = rho_in_residual + alpha11 * rho_out_minus_in_residual
        term2 = term1 @ A_inv @ rho_out_minus_in_residual.T
        P = alpha11 * np.eye(N_q) - term2
        
        # Apply to residual
        rho_in_prev = self.rho_in_store[:, -2]
        rho_out_prev = self.rho_out_store[:, -2]
        residual = rho_out_prev - rho_in_prev
        
        return rho_in_prev + P @ residual
