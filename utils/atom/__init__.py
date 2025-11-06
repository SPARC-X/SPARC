"""
Atomic DFT Solver Package

This package provides a comprehensive implementation of Atomic Density Functional Theory
solver using finite element method.

Main class:
    AtomicDFTSolver: Main solver class for atomic DFT calculations

Example:
    >>> from delta.atomic_dft import AtomicDFTSolver
    >>> solver = AtomicDFTSolver(atomic_number=13, xc_functional="GGA_PBE")
    >>> results = solver.solve()
"""

__version__ = "0.1.0"

# Main solver class
from .solver import AtomicDFTSolver

# Make AtomicDFTSolver easily accessible
__all__ = ["AtomicDFTSolver"]
