"""
Poisson solver for 1D and spherically symmetric 3D problems

Mathematical transformation for spherical symmetry:
    3D: ∇²V = -4πρ  →  Let u = rV  →  1D: d²u/dr² = -4πrρ
"""

from __future__ import annotations
import scipy
import numpy as np
from typing import Optional
from ..mesh.operators import RadialOperatorsBuilder
from ..mesh.builder import Mesh1D



OPS_BUILDER_TYPE_ERROR_MESSAGE = \
    "ops_builder must be an instance of RadialOperatorsBuilder, but got {} instead"
Z_VALENCE_TYPE_ERROR_MESSAGE = \
    "parameter z_valence must be a float, get type {} instead"
RHS_VECTOR_TYPE_ERROR_MESSAGE = \
    "parameter rhs_vector must be a numpy array, get type {} instead"
RHS_VECTOR_NDIM_ERROR_MESSAGE = \
    "parameter rhs_vector must be a 1D array, get type {} instead"
RHS_VECTOR_SHAPE_ERROR_MESSAGE = \
    "parameter rhs_vector shape {} must match Laplacian matrix size {}"

RHO_TYPE_ERROR_MESSAGE = \
    "parameter rho must be a numpy array, get type {} instead"
RHO_NDIM_ERROR_MESSAGE = \
    "parameter rho must be a 1D array, get type {} instead"
RHO_SHAPE_ERROR_MESSAGE = \
    "parameter rho shape {} must match quadrature nodes size {}"



class PoissonSolver:
    """
    Poisson equation solver supporting:
    1. Generic 1D equation: d²u/dx² = f(x)
    2. Spherically symmetric 3D: ∇²V = -4πρ(r)
    
    Parameters
    ----------
    ops_builder : RadialOperatorsBuilder
        Provides Laplacian matrix and grid information
    """
    
    def __init__(
        self, 
        ops_builder: RadialOperatorsBuilder, 
        z_valence: float,
        ):
        # type check for required parameters
        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_TYPE_ERROR_MESSAGE.format(type(ops_builder))
        try:
            z_valence = float(z_valence)
        except:
            raise ValueError(Z_VALENCE_TYPE_ERROR_MESSAGE.format(type(z_valence)))

        self.ops_builder = ops_builder
        self.z_valence = z_valence
        
        
        # laplacian matrix and set boundary conditions
        _laplacian = ops_builder.laplacian
        _laplacian[0,:] = 0
        _laplacian[-1,:] = 0
        _laplacian[:,0] = 0
        _laplacian[0,0] = -1
        _laplacian[-1,-1] = -1
        self.laplacian_with_boundary_conditions = _laplacian

        # store some other useful information
        self.number_of_finite_elements : int = ops_builder.number_of_finite_elements
        self.quadrature_nodes   : np.ndarray = ops_builder.quadrature_nodes
        self.quadrature_weights : np.ndarray = ops_builder.quadrature_weights



    
    
    def solve_1d(self, rhs_vector: np.ndarray) -> np.ndarray:
        """
        Solve 1D Poisson equation: d²u/dx² = f(x)
        
        This method solves the linear system L @ u = rhs_vector using the 
        pre-configured Laplacian matrix with Dirichlet boundary conditions.
        
        The Laplacian matrix L has been modified during initialization:
            - L[0, :] = [1, 0, 0, ...]      → First equation: u[0] = rhs[0]
            - L[-1, :] = [0, 0, ..., 0, 1]  → Last equation: u[-1] = rhs[-1]
            - Interior rows maintain original finite element discretization
        
        Therefore, the caller must set boundary values at the endpoints of rhs_vector:
            rhs_vector[0] = left boundary value
            rhs_vector[-1] = right boundary value
        
        Parameters
        ----------
        rhs_vector : np.ndarray
            Right-hand side vector of the linear system, containing:
            - rhs[0]: Left boundary condition value (enforced as u[0])
            - rhs[1:-1]: Interior source terms f(x_i)
            - rhs[-1]: Right boundary condition value (enforced as u[-1])
            Shape: (n_nodes,) must match Laplacian matrix dimension
        
        Returns
        -------
        solution : np.ndarray
            Solution vector u satisfying the boundary conditions
            Shape: (n_nodes,)
        
        Notes
        -----
        This method directly solves the linear system without coordinate 
        transformations or interpolation. Boundary condition enforcement 
        is handled through matrix modification (done in __init__).
        """
        # Type and shape validation
        assert isinstance(rhs_vector, np.ndarray), \
            RHS_VECTOR_TYPE_ERROR_MESSAGE.format(type(rhs_vector))
        assert rhs_vector.ndim == 1, \
            RHS_VECTOR_NDIM_ERROR_MESSAGE.format(rhs_vector.ndim)
        assert rhs_vector.shape[0] == self.laplacian_with_boundary_conditions.shape[0], \
            RHS_VECTOR_SHAPE_ERROR_MESSAGE.format(rhs_vector.shape[0], self.laplacian.shape[0])

        # Solve linear system: L @ u = rhs
        solution = scipy.linalg.solve(self.laplacian_with_boundary_conditions, rhs_vector[:, np.newaxis])[:, 0]

        # np.savetxt("solution.txt", solution.reshape(-1))
        # np.savetxt("rhs_vector.txt", rhs_vector.reshape(-1))
        # np.savetxt("laplacian_with_boundary_conditions.txt", self.laplacian_with_boundary_conditions.reshape(-1))
        # raise RuntimeError("Stop here")
        
        return solution
    
    
    def solve_hartree(self, rho: np.ndarray) -> np.ndarray:
        """
        Solve 3D spherically symmetric Poisson equation for Hartree potential
        
        Equation: ∇²V = -4πρ(r)
        
        Transform: u = rV, then d²u/dr² = -4πrρ(r)
        
        Boundary conditions:
        - u(0) = 0  (V finite at origin)
        - u(∞) = Q  (Q = total charge for V → Q/r at infinity)
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density at quadrature nodes
            Shape: (N_quad,)
        Z : float, optional
            Nuclear charge (for computing total charge Q)
            If provided: Q = Z - ∫ρd³r
            If None: Q = ∫ρd³r (for testing/special cases)
        
        Returns
        -------
        v_hartree : np.ndarray
            Hartree potential at quadrature nodes
            Shape: (N_quad,)
        """
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_ERROR_MESSAGE.format(type(rho))
        assert rho.ndim == 1, \
            RHO_NDIM_ERROR_MESSAGE.format(rho.ndim)
        assert rho.shape[0] == self.quadrature_nodes.shape[0], \
            RHO_SHAPE_ERROR_MESSAGE.format(rho.shape[0], self.quadrature_nodes.shape[0])


        # prepare the rhs_vector for the 1D Poisson equation
        rhs_vector = self.ops_builder.assemble_poisson_rhs_vector(rho, self.z_valence)

        # Set the boundary values
        rhs_vector[0] = 0.0
        rhs_vector[-1] = - self.z_valence

        # solve the 1D Poisson equation at dense physical nodes
        r_times_v_hartree_at_dense_physical_nodes = self.solve_1d(rhs_vector)  # r * v_hartree

        # Convert to quadrature nodes
        r_times_v_hartree_at_dense_physical_nodes_reshaped = Mesh1D.fe_flat_to_block2d(
            flat = r_times_v_hartree_at_dense_physical_nodes, 
            n_elem = self.number_of_finite_elements, 
            endpoints_shared = True
        )

        r_times_v_hartree_at_quadrature_nodes = np.einsum(
            "emi,ei->em",
            self.ops_builder.lagrange_basis,                     # (n_elem, n_quad, n_phys)
            r_times_v_hartree_at_dense_physical_nodes_reshaped,  # (n_elem, n_phys)
            optimize=True
        ).reshape(-1,)

        v_hartree = r_times_v_hartree_at_quadrature_nodes / self.quadrature_nodes
        
        return v_hartree
    
    
    def compute_hartree_energy(self, rho: np.ndarray, V_H: np.ndarray) -> float:
        """
        Compute Hartree energy: E_H = (1/2) ∫ ρ(r) V_H(r) 4πr² dr
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density
        V_H : np.ndarray
            Hartree potential
        
        Returns
        -------
        E_H : float
            Hartree energy
        """
        integrand = rho * V_H * 4.0 * np.pi * self.r_quad**2
        E_H = 0.5 * np.sum(integrand * self.w_quad)
        raise NotImplementedError("This function is not tested")
