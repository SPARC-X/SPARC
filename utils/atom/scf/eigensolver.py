from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
from typing import Optional


class EigenSolver:
    """
    Simplified eigenvalue solver matching reference implementation.
    
    For SCAN/RSCAN/R2SCAN: uses LinearOperator + eigsh
    For others: uses eigh with subset_by_index
    """
    
    def __init__(self, xc_functional: Optional[str] = None):
        """
        Initialize eigenvalue solver.
        
        Parameters
        ----------
        xc_functional : str, optional
            XC functional name (e.g., 'SCAN', 'RSCAN', 'R2SCAN', 'GGA_PBE')
        """
        self.xc_functional = xc_functional
    
    def solve_lowest(self, H: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve for k lowest eigenvalues and eigenvectors.
        
        Parameters
        ----------
        H : np.ndarray
            Hamiltonian matrix (symmetric/Hermitian)
        k : int
            Number of lowest eigenvalues to compute
        
        Returns
        -------
        eigenvalues : np.ndarray
            k lowest eigenvalues (sorted in ascending order)
        eigenvectors : np.ndarray
            Corresponding eigenvectors, shape (n, k)
        """
        n = H.shape[0]
        
        # Use LinearOperator + eigsh for SCAN/R2SCAN
        if self.xc_functional in ['SCAN', 'R2SCAN']:
            def matvec(x):
                return H @ x
            
            H_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
            k_solve = n - 1
            
            eigvals, eigvecs = eigsh(H_op, k=k_solve, which='SA', tol=1e-14)
            
            # Sort and return only k eigenvalues
            sort_idx = np.argsort(eigvals)
            return eigvals[sort_idx][:k], eigvecs[:, sort_idx][:, :k]
        
        # Use eigh with subset_by_index for others (reference: subset_by_index=[0, k-1])
        else:
            eigvals, eigvecs = eigh(
                H,
                subset_by_index=[0, k-1],
                check_finite=False,
                driver='evr'
            )
            return eigvals, eigvecs
