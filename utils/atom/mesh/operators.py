from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .builder import LagrangeShapeFunctions, Mesh1D
from dataclasses import dataclass



# Error messages
NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR = \
    "parameter number_of_finite_elements must be greater than 0, get {} instead"
PHYSICAL_NODES_NOT_1D_ARRAY_ERROR = \
    "parameter physical_nodes must be a 1D array"
QUADRATURE_NODES_NOT_1D_ARRAY_ERROR = \
    "parameter quadrature_nodes must be a 1D array"
QUADRATURE_WEIGHTS_NOT_1D_ARRAY_ERROR = \
    "parameter quadrature_weights must be a 1D array"
QUADRATURE_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR = \
    "parameter quadrature_nodes and quadrature_weights must have the same length"
Z_NUCLEAR_NOT_FLOAT_ERROR = \
    "parameter z_nuclear must be a float, get {} instead"
ALL_ELECTRON_FLAG_NOT_PROVIDED_ERROR = \
    "parameter all_electron_flag must be provided"
V_LOCAL_COMPONENT_PSP_NOT_NP_NDARRAY_ERROR = \
    "parameter v_local_component_psp must be a numpy array, get {} instead"
V_LOCAL_COMPONENT_PSP_NOT_THE_SAME_SIZE_AS_QUADRATURE_NODES_ERROR = \
    "parameter v_local_component_psp must have the same size as quadrature_nodes, get {} and {} instead"
POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODES_ERROR = \
    "parameter potential_values shape {} does not match quadrature_nodes shape {}"
POTENTIAL_VALUES_DO_NOT_MATCH_NUMBER_OF_FINITE_ELEMENTS_ERROR = \
    "parameter potential_values shape {} does not match number_of_finite_elements shape {}"
POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODE_NUMBER_ERROR = \
    "parameter potential_values shape {} does not match quadrature_node_number shape {}"
POTENTIAL_VALUES_NDIM_ERROR = \
    "parameter potential_values must be a 1D or 2D array, get dimension {} instead"

RHO_TYPE_ERROR_MESSAGE = \
    "parameter rho must be a numpy array, get type {} instead"
RHO_NDIM_ERROR_MESSAGE = \
    "parameter rho must be a 1D array, get dimension {} instead"
RHO_SHAPE_ERROR_MESSAGE = \
    "parameter rho shape {} does not match quadrature_node_number shape {}"

DE_XC_DTAU_SHAPE_ERROR_MESSAGE = \
    "parameter de_xc_dtau shape {} does not match quadrature_node_number shape {}"
DE_XC_DTAU_NDIM_ERROR = \
    "parameter de_xc_dtau must be 1D or 2D array, got {}D instead"

GIVEN_GRID_NOT_MONOTONICALLY_INCREASING_ERROR = \
    "The given grid must be monotonically increasing"
GIVEN_GRID_NOT_WITHIN_PHYSICAL_NODES_ERROR = \
    "The given grid must be within the physical nodes"
ORBITAL_NDIM_ERROR_MESSAGE = \
    "parameter orbital must be a 1D array, get dimension {} instead"
ORBITAL_SHAPE_ERROR_MESSAGE = \
    "parameter orbital shape {} does not match number_of_finite_elements * physical_node_number shape {}"


# Warning messages
V_LOCAL_COMPONENT_PSP_NOT_USED_IN_ALL_ELECTRON_CALCULATIONS_WARNING = \
    "WARNING: v_local_component_psp is not used in all-electron calculations"
Z_NUCLEAR_NOT_USED_IN_NON_ALL_ELECTRON_CALCULATIONS_WARNING = \
    "WARNING: z_nuclear is not used in non-all-electron calculations"
NUMBER_OF_FINITE_ELEMENTS_NOT_USED_WHEN_USING_GRID_DATA_WARNING = \
    "WARNING: number_of_finite_elements is not used when using GridData"
PHYSICAL_NODES_NOT_USED_WHEN_USING_GRID_DATA_WARNING = \
    "WARNING: physical_nodes is not used when using GridData"
QUADRATURE_NODES_NOT_USED_WHEN_USING_GRID_DATA_WARNING = \
    "WARNING: quadrature_nodes is not used when using GridData"
QUADRATURE_WEIGHTS_NOT_USED_WHEN_USING_GRID_DATA_WARNING = \
    "WARNING: quadrature_weights is not used when using GridData"



@dataclass(frozen=True)
class GridData:
    """
    Immutable grid information needed for XC calculations.
    
    Attributes
    ----------
    r_quad : np.ndarray
        Radial quadrature points
    w_quad : np.ndarray
        Quadrature weights
    derivative_matrix : np.ndarray
        Matrix for computing derivatives at quadrature points
        Shape: (n_elem, n_quad, n_quad)
    n_elem : int
        Number of finite elements
    n_quad_per_elem : int
        Number of quadrature points per element
    """
    number_of_finite_elements: int
    physical_nodes           : np.ndarray
    quadrature_nodes         : np.ndarray
    quadrature_weights       : np.ndarray


    def __post_init__(self):
        # check if the input parameters are valid
        assert self.number_of_finite_elements > 0, \
            NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR.format(self.number_of_finite_elements)
        assert self.physical_nodes.ndim == 1, \
            PHYSICAL_NODES_NOT_1D_ARRAY_ERROR
        assert self.quadrature_nodes.ndim == 1, \
            QUADRATURE_NODES_NOT_1D_ARRAY_ERROR
        assert self.quadrature_weights.ndim == 1, \
            QUADRATURE_WEIGHTS_NOT_1D_ARRAY_ERROR
        assert self.quadrature_nodes.shape[0] == self.quadrature_weights.shape[0], \
            QUADRATURE_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR
            



class RadialOperatorsBuilder:
    """
    Assemble radial FE operators and interpolation matrices.

    Can be initialized either from individual parameters or from GridData.
    """

    def __init__(self, 
        number_of_finite_elements : Optional[int] = None, 
        physical_nodes            : Optional[np.ndarray] = None, 
        quadrature_nodes          : Optional[np.ndarray] = None,
        quadrature_weights        : Optional[np.ndarray] = None,
        grid_data                 : Optional[GridData] = None,
        verbose                   : bool = False,
        ):
        """
        Initialize radial operators builder.
        
        Two initialization modes:
        
        Mode 1 - From individual parameters:
            number_of_finite_elements, physical_nodes, quadrature_nodes, quadrature_weights
        
        Mode 2 - From GridData (preferred):
            grid_data (contains all the above)
        
        Parameters
        ----------
        grid_data : GridData, optional
            Grid data object (preferred way)
        number_of_finite_elements : int, optional
            Number of finite elements (legacy way)
        physical_nodes : np.ndarray, optional
            Physical FE nodes (legacy way)
        quadrature_nodes : np.ndarray, optional
            Quadrature points (legacy way)
        quadrature_weights : np.ndarray, optional
            Quadrature weights (legacy way)
        """
        # Extract parameters from GridData or use individual parameters
        if grid_data is not None:
            # Mode 2: Use GridData (preferred)
            self.number_of_finite_elements = grid_data.number_of_finite_elements
            self.physical_nodes = grid_data.physical_nodes
            self.quadrature_nodes = grid_data.quadrature_nodes
            self.quadrature_weights = grid_data.quadrature_weights

            # if other parameters are provided, they will be ignored, and a warning will be printed
            if number_of_finite_elements is not None:
                print(NUMBER_OF_FINITE_ELEMENTS_NOT_USED_WHEN_USING_GRID_DATA_WARNING)
            if physical_nodes is not None:
                print(PHYSICAL_NODES_NOT_USED_WHEN_USING_GRID_DATA_WARNING)
            if quadrature_nodes is not None:
                print(QUADRATURE_NODES_NOT_USED_WHEN_USING_GRID_DATA_WARNING)
            if quadrature_weights is not None:
                print(QUADRATURE_WEIGHTS_NOT_USED_WHEN_USING_GRID_DATA_WARNING)
        else:
            # Mode 1: Use individual parameters (legacy)
            assert number_of_finite_elements is not None, \
                "number_of_finite_elements required"
            assert physical_nodes is not None, \
                "physical_nodes required"
            assert quadrature_nodes is not None, \
                "quadrature_nodes required"
            assert quadrature_weights is not None, \
                "quadrature_weights required"
            
            # Validation
            assert physical_nodes.ndim == 1, \
                PHYSICAL_NODES_NOT_1D_ARRAY_ERROR
            assert quadrature_nodes.ndim == 1, \
                QUADRATURE_NODES_NOT_1D_ARRAY_ERROR
            assert quadrature_weights.ndim == 1, \
                QUADRATURE_WEIGHTS_NOT_1D_ARRAY_ERROR
            assert quadrature_nodes.shape[0] == quadrature_weights.shape[0], \
                QUADRATURE_NODES_AND_WEIGHTS_NOT_THE_SAME_LENGTH_ERROR

            self.number_of_finite_elements = number_of_finite_elements
            self.physical_nodes = physical_nodes
            self.quadrature_nodes = quadrature_nodes
            self.quadrature_weights = quadrature_weights



        # Store verbose flag
        self.verbose = verbose
        
        # Reshape to element-wise structure
        self._reshape_grid_data()
        
        # Compute Lagrange basis functions
        self._compute_basis_functions()
        
        # Print summary if verbose
        if self.verbose:
            self._print_initialization_summary()
    
    
    def _reshape_grid_data(self):
        """Reshape 1D arrays to (n_elem, n_points) structure."""
        self.physical_nodes_reshaped = Mesh1D.fe_flat_to_block2d(
            self.physical_nodes, 
            self.number_of_finite_elements, 
            endpoints_shared=True
        )
        self.quadrature_nodes_reshaped = Mesh1D.fe_flat_to_block2d(
            self.quadrature_nodes, 
            self.number_of_finite_elements, 
            endpoints_shared=False
        )
        self.quadrature_weights_reshaped = Mesh1D.fe_flat_to_block2d(
            self.quadrature_weights, 
            self.number_of_finite_elements, 
            endpoints_shared=False
        )
        
        self.physical_node_number = self.physical_nodes_reshaped.shape[1]
        self.quadrature_node_number = self.quadrature_nodes_reshaped.shape[1]
    
    
    def _compute_basis_functions(self):
        """Compute Lagrange basis functions and derivatives."""
        self.lagrange_basis, self.lagrange_basis_derivatives = \
            LagrangeShapeFunctions.lagrange_basis_and_derivatives(
                x_node=self.physical_nodes_reshaped,
                x_eval=self.quadrature_nodes_reshaped
            )
        # Shape: (n_elem, n_quad, n_basis)
    
    
    def _print_initialization_summary(self):
        """Print initialization summary."""
        print("=" * 60)
        print("\t\t RadialOperatorsBuilder")
        print("=" * 60)
        print(f"\t Number of elements            : {self.number_of_finite_elements}")
        print(f"\t Physical nodes per element    : {self.physical_nodes_reshaped.shape[1]}")
        print(f"\t Quadrature points per element : {self.quadrature_node_number}")
        print(f"\t Total DOFs                    : {self.number_of_finite_elements * self.physical_node_number + 1}")
        print(f"\t Total quadrature points       : {len(self.quadrature_nodes)}")
        print(f"\t Lagrange basis shape          : {self.lagrange_basis.shape}")
        print(f"\t Lagrange derivatives shape    : {self.lagrange_basis_derivatives.shape}")
        print()
    
    
    @classmethod
    def from_grid_data(cls, grid_data: GridData, verbose: bool = False) -> 'RadialOperatorsBuilder':
        """
        Create RadialOperatorsBuilder from GridData.
        
        Convenience factory method for cleaner code.
        
        Parameters
        ----------
        grid_data : GridData
            Grid data object
        verbose : bool, optional
            If True, print initialization summary
            Default: False
        
        Returns
        -------
        RadialOperatorsBuilder
            Initialized operators builder
        
        Example
        -------
        >>> grid_data = GridData(...)
        >>> ops = RadialOperatorsBuilder.from_grid_data(grid_data, verbose=True)
        """
        return cls(grid_data=grid_data, verbose=verbose)


    def get_H_kinetic(self) -> np.ndarray:
        """
        Kinetic energy matrix
        """
        # check if the kinetic energy matrix has been computed
        if hasattr(self, "_H_kinetic"):
            return self._H_kinetic
        
        # compute the kinetic energy matrix
        H_kinetic = - 0.5 * self.get_laplacian()
        
        # store the kinetic energy matrix
        self._H_kinetic = H_kinetic

        return H_kinetic


    def get_laplacian(self) -> np.ndarray:
        """
        Laplacian matrix
        """
        # check if the Laplacian matrix has been computed
        if hasattr(self, "_laplacian"):
            return self._laplacian
        
        # compute the Laplacian matrix
        laplacian_local = - np.einsum("emi,emk,em->eik", 
            self.lagrange_basis_derivatives, 
            self.lagrange_basis_derivatives, 
            self.quadrature_weights_reshaped, 
            optimize=True)

        # Assemble local matrices into global matrix
        laplacian = self._assemble_local_to_global_matrix(laplacian_local)
        
        # store the Laplacian matrix
        self._laplacian = laplacian
        
        return laplacian


    def build_potential_matrix(self, potential_values: np.ndarray) -> np.ndarray:
        """
        Construct Hamiltonian matrix from potential energy values at quadrature points.
        
        This is a general-purpose matrix builder that can be used for any potential:
        - Nuclear Coulomb potential: V(r) = -Z/r
        - Local pseudopotential: V_loc(r)
        - Hartree potential: V_H(r)
        - Exchange-correlation potential: V_xc(r)
        
        Theory
        ------
        Given potential V(r) sampled at quadrature points, compute:
            H_V[i,j] = ∫ φ_i(r) V(r) φ_j(r) dr
        
        Using finite element basis φ and Gauss quadrature.
        
        Parameters
        ----------
        potential_values : np.ndarray
            Potential energy values at quadrature points.
            Can be 1D (flat, length = n_elem * n_quad) or 
            2D (reshaped, shape = (n_elem, n_quad))
        
        Returns
        -------
        H_potential : np.ndarray
            Assembled global Hamiltonian matrix contribution from this potential.
            Shape: (n_global_dofs, n_global_dofs)
        
        Examples
        --------
        >>> # Nuclear Coulomb potential
        >>> V_nuc = -Z / ops.quadrature_nodes
        >>> H_nuc = ops.build_potential_matrix(V_nuc)
        
        >>> # XC potential
        >>> V_xc = compute_xc_potential(rho)
        >>> H_xc = ops.build_potential_matrix(V_xc)
        """
        # Reshape potential if needed
        if potential_values.ndim == 1:
            assert potential_values.size == self.quadrature_nodes_reshaped.size, \
                POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODES_ERROR.format(potential_values.size, self.quadrature_nodes_reshaped.size)
            potential_reshaped = potential_values.reshape(self.quadrature_nodes_reshaped.shape)
        elif potential_values.ndim == 2:
            assert potential_values.shape[0] == self.number_of_finite_elements, \
                POTENTIAL_VALUES_DO_NOT_MATCH_NUMBER_OF_FINITE_ELEMENTS_ERROR.format(potential_values.shape[0], self.number_of_finite_elements)
            assert potential_values.shape[1] == self.quadrature_node_number, \
                POTENTIAL_VALUES_DO_NOT_MATCH_QUADRATURE_NODE_NUMBER_ERROR.format(potential_values.shape[1], self.quadrature_node_number)
            potential_reshaped = potential_values
        else:
            raise ValueError(POTENTIAL_VALUES_NDIM_ERROR.format(potential_values.ndim))
        
        # Compute local element matrices: H^e[i,k] = ∫ φ_i V φ_k dr
        H_local = np.einsum("emi,emk,em,em->eik", 
            self.lagrange_basis,       # φ_i at quadrature points
            self.lagrange_basis,       # φ_k at quadrature points
            self.quadrature_weights_reshaped,  # quadrature weights
            potential_reshaped,        # V(r) at quadrature points
            optimize=True)
        
        # Assemble local matrices into global matrix
        H_potential = self._assemble_local_to_global_matrix(H_local)
        
        return H_potential


    def build_metagga_kinetic_density_matrix(self, de_xc_dtau: np.ndarray) -> np.ndarray:
        """
        Construct Hamiltonian matrix from meta-GGA kinetic density term at quadrature points.
        
        This implements the meta-GGA contribution to the Hamiltonian:
            H_metagga[i,j] = ∫ φ'_i * (0.5 * de_xc_dtau) * φ'_j dr
                          + ∫ φ_i * (0.5 * w/r * d(de_xc_dtau)/dr) * φ_j dr
        
        Theory
        ------
        For meta-GGA functionals, the kinetic energy density τ contributes additional
        terms to the Hamiltonian. The implementation follows:
            V3 = de_xc_dtau
            V3grad = d(V3)/dr (computed via derivative matrix)
            
            Term 1: ∫ φ'_i * (0.5 * V3 * w) * φ'_j dr
            Term 2: ∫ φ_i * (0.5 * w/r * V3grad) * φ_j dr
        
        Parameters
        ----------
        de_xc_dtau : np.ndarray
            Derivative of XC energy density w.r.t. kinetic energy density τ.
            Can be 1D (flat, length = n_elem * n_quad) or 
            2D (reshaped, shape = (n_elem, n_quad))
        
        Returns
        -------
        H_metagga : np.ndarray
            Assembled global Hamiltonian matrix contribution from meta-GGA kinetic density term.
            Shape: (n_global_dofs, n_global_dofs)
        """
        # Reshape de_xc_dtau if needed
        if de_xc_dtau.ndim == 1:
            assert de_xc_dtau.size == self.quadrature_nodes_reshaped.size, \
                DE_XC_DTAU_SHAPE_ERROR_MESSAGE.format(de_xc_dtau.size, self.quadrature_nodes_reshaped.size)
            de_xc_dtau_reshaped = de_xc_dtau.reshape(self.quadrature_nodes_reshaped.shape)
        elif de_xc_dtau.ndim == 2:
            assert de_xc_dtau.shape[0] == self.number_of_finite_elements, \
                DE_XC_DTAU_SHAPE_ERROR_MESSAGE.format(de_xc_dtau.shape[0], self.number_of_finite_elements)
            assert de_xc_dtau.shape[1] == self.quadrature_node_number, \
                DE_XC_DTAU_SHAPE_ERROR_MESSAGE.format(de_xc_dtau.shape[1], self.quadrature_node_number)
            de_xc_dtau_reshaped = de_xc_dtau
        else:
            raise ValueError(DE_XC_DTAU_NDIM_ERROR.format(de_xc_dtau.ndim))
        
        # Get derivative matrix for computing d(de_xc_dtau)/dr
        D = self.get_derivative_matrix()  # Shape: (n_elem, n_quad, n_quad)
        
        # Compute gradient: v3grad = D @ de_xc_dtau (element-wise)
        # For each element: v3grad[e, i] = sum_k D[e, i, k] * de_xc_dtau[e, k]
        de_xc_dtau_grad = np.einsum("eik,ek->ei", D, de_xc_dtau_reshaped)
        
        # Term 1: ∫ φ'_i * (0.5 * de_xc_dtau * w) * φ'_j dr
        # Using derivative basis functions
        H_term1_local = np.einsum("emi,emk,em,em->eik",
            self.lagrange_basis_derivatives,  # φ'_i
            self.lagrange_basis_derivatives,  # φ'_j
            self.quadrature_weights_reshaped,  # w
            0.5 * de_xc_dtau_reshaped,  # 0.5 * de_xc_dtau
            optimize=True)
        
        # Term 2: ∫ φ_i * (0.5 * w/r * d(de_xc_dtau)/dr) * φ_j dr
        # Using regular basis functions
        # Note: 0.5 * w / r * v3grad
        weight_over_r = 0.5 * self.quadrature_weights_reshaped / self.quadrature_nodes_reshaped
        H_term2_local = np.einsum("emi,emk,em,em->eik",
            self.lagrange_basis,  # φ_i
            self.lagrange_basis,  # φ_j
            weight_over_r,  # 0.5 * w / r
            de_xc_dtau_grad,  # d(de_xc_dtau)/dr
            optimize=True)
        
        # Combine both terms
        H_metagga_local = H_term1_local + H_term2_local
        
        # Assemble local matrices into global matrix
        H_metagga = self._assemble_local_to_global_matrix(H_metagga_local)
        
        return H_metagga




    def get_nuclear_coulomb_potential(self, z_nuclear: float) -> np.ndarray:
        """
        Compute nuclear Coulomb potential: V_nuc(r) = -Z/r
        
        Parameters
        ----------
        z_nuclear : float
            Nuclear charge (atomic number for all-electron, or effective charge)
        
        Returns
        -------
        V_nuclear : np.ndarray
            Nuclear Coulomb potential at quadrature points.
            Shape: (n_elem * n_quad,) flat array
        
        Notes
        -----
        For all-electron calculations, z_nuclear is the atomic number Z.
        The negative sign accounts for attractive electron-nucleus interaction.
        """
        try:
            z_nuclear = float(z_nuclear)
        except:
            raise ValueError(Z_NUCLEAR_NOT_FLOAT_ERROR.format(type(z_nuclear)))
        
        # V_nuc(r) = -Z/r at all quadrature points
        V_nuclear = -z_nuclear / self.quadrature_nodes
        
        return V_nuclear


    def get_H_r_inv_sq(self) -> np.ndarray:
        """
        Inverse square of the radial distance matrix
        """
        # check if the inverse square of the radial distance matrix has been computed
        if hasattr(self, "_H_r_inv_sq"):
            return self._H_r_inv_sq
    
        # compute the inverse square of the radial distance matrix
        H_r_inv_sq_local = np.einsum("emi,emk,em,em->eik", 
            self.lagrange_basis, 
            self.lagrange_basis, 
            self.quadrature_weights_reshaped, 
            1.0 / self.quadrature_nodes_reshaped**2, 
            optimize=True)
        
        # Assemble local matrices into global matrix
        H_r_inv_sq = self._assemble_local_to_global_matrix(H_r_inv_sq_local)
        
        # store the inverse square of the radial distance matrix
        self._H_r_inv_sq = H_r_inv_sq
        
        return H_r_inv_sq


    def get_S(self) -> np.ndarray:
        """
        Overlap matrix S
        """
        # check if the overlap matrix has been computed
        if hasattr(self, "_S"):
            return self._S
        
        # compute the overlap matrix
        S_local = np.einsum("emi,emk,em->eik", 
            self.lagrange_basis, 
            self.lagrange_basis, 
            self.quadrature_weights_reshaped, 
            optimize=True)

        # Assemble local matrices into global matrix
        S = self._assemble_local_to_global_matrix(S_local)
        
        # store the overlap matrix
        self._S = S
        
        return S


    def get_S_inv_sqrt(self) -> np.ndarray:
        """
        Inverse square root of the overlap matrix: S^(-1/2)
        Computed via eigendecomposition: S^(-1/2) = V * Λ^(-1/2) * V^T
        """
        if hasattr(self, "_S_inv_sqrt"):
            return self._S_inv_sqrt
        
        S = self.get_S()
        eigvals, eigvecs = np.linalg.eigh(S, UPLO='L')
        S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        S_inv_sqrt = 0.5 * (S_inv_sqrt + S_inv_sqrt.T)  # Symmetrize
        
        self._S_inv_sqrt = S_inv_sqrt
        return S_inv_sqrt


    def get_derivative_matrix(self) -> np.ndarray:
        """
        Differentiation matrix for computing derivatives at quadrature points.
        
        This matrix D enables direct computation of derivatives from function values:
            f'(x_quad) = D @ f(x_quad)
        
        Theory:
        -------
        For a function f(x) = sum_j c_j φ_j(x) in the FE space,
        we have f' = (dL/dx) @ L⁺ @ f, where:
        - L[i,j] = φ_j(x_quad_i) : basis functions at quadrature points
        - L⁺ : pseudoinverse of L (maps function values -> coefficients)
        - dL/dx[i,j] = φ'_j(x_quad_i) : basis derivatives at quadrature points
        
        Returns
        -------
        np.ndarray
            Shape: (n_elements, n_quad_points, n_quad_points)
            D[e, i, k] maps f(x_k) → f'(x_i) within element e
        """
        if hasattr(self, "_derivative_matrix"):
            return self._derivative_matrix
        
        # Compute pseudoinverse of Lagrange basis: L⁺ maps values → coefficients
        # Shape: (n_elem, n_basis, n_quad) where n_basis = physical_node_number
        n_elem = self.number_of_finite_elements
        n_basis = self.physical_node_number
        n_quad = self.quadrature_node_number
        
        basis_pseudoinverse = np.zeros((n_elem, n_basis, n_quad))
        for elem_idx in range(n_elem):
            # self.lagrange_basis[elem_idx] has shape (n_quad, n_basis)
            basis_pseudoinverse[elem_idx] = np.linalg.pinv(self.lagrange_basis[elem_idx])
        
        # Differentiation matrix: D = (dL/dx) @ L⁺
        # Maps function values at quadrature points to derivative values
        derivative_matrix = np.matmul(self.lagrange_basis_derivatives, basis_pseudoinverse)
        
        self._derivative_matrix = derivative_matrix
        return derivative_matrix


    def get_global_interpolation_matrix(self) -> np.ndarray:
        """
        Global interpolation matrix for evaluating functions at quadrature points.
        
        This matrix maps global nodal coefficients to function values at all quadrature points:
            f(x_quad) = interpolation_matrix @ f_nodal
        
        Theory
        ------
        For a function f(x) represented by nodal values on the global FE grid,
        this matrix evaluates f at all quadrature points across all elements.
        
        Unlike _assemble_local_to_global_matrix (which accumulates contributions),
        this constructs a rectangular interpolation matrix by:
        1. Placing local Lagrange basis matrices in block-diagonal form
        2. Merging columns corresponding to shared boundary nodes
        3. Removing duplicate columns
        
        Returns
        -------
        np.ndarray
            Shape: (n_elements * n_quad_points, n_global_nodes)
            where n_global_nodes = n_elements * polynomial_order + 1
            
            Example: n_elem=17, p=31, n_quad=95 → shape (1615, 528)
        
        Usage
        -----
        Essential for Hartree-Fock exchange (HF, PBE0) where orbital overlaps
        must be computed at quadrature points:
            psi_i(x_quad) = interp_matrix @ psi_i_nodal
        
        Not needed for LDA, GGA, or meta-GGA functionals.
        """
        if hasattr(self, "_global_interpolation_matrix"):
            return self._global_interpolation_matrix
        
        n_elem = self.number_of_finite_elements
        n_quad = self.quadrature_node_number
        n_local = self.physical_node_number
        n_global = n_elem * (self.physical_node_number - 1) + 1
        
        # Method: Build block-diagonal then remove duplicate columns
        # Alternative could use fancy indexing, but this is clearer
        
        # Reshape lagrange_basis from (n_elem, n_quad, n_local) to 2D blocks
        # Result shape: (n_elem * n_quad, n_elem * n_local)
        interp_matrix = np.zeros((n_elem * n_quad, n_elem * n_local))
        
        for elem_idx in range(n_elem):
            row_slice = slice(elem_idx * n_quad, (elem_idx + 1) * n_quad)
            col_slice = slice(elem_idx * n_local, (elem_idx + 1) * n_local)
            interp_matrix[row_slice, col_slice] = self.lagrange_basis[elem_idx]
        
        # Shared boundary nodes: right edge of each element (except last)
        # These correspond to duplicate columns that must be merged
        shared_cols = np.array([n_local * (i + 1) - 1 for i in range(n_elem - 1)])
        
        # Merge: add right neighbor column to left neighbor column
        interp_matrix[:, shared_cols] += interp_matrix[:, shared_cols + 1]
        
        # Remove duplicate columns
        interp_matrix = np.delete(interp_matrix, shared_cols + 1, axis=1)
        
        assert interp_matrix.shape == (n_elem * n_quad, n_global), \
            f"Shape mismatch: expected {(n_elem * n_quad, n_global)}, got {interp_matrix.shape}"
        
        self._global_interpolation_matrix = interp_matrix
        return interp_matrix


    def assemble_poisson_rhs_vector(self, rho: np.ndarray, z_nuclear: float) -> np.ndarray:
        """
        Assemble the right-hand side vector for the Poisson equation.
        
        Computes the RHS vector for solving d²u/dr² = -4πrρ(r) where u = rV.
        The weak form involves integrating -4πrρ(r) against test functions.
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density at quadrature points
            Shape: (n_elements * n_quad_points,)
        z_nuclear : float
            Nuclear charge (used for boundary condition)
        
        Returns
        -------
        rhs_vector : np.ndarray
            Assembled global RHS vector
            Shape: (n_global_dofs,)
            Note: Caller must set boundary values:
                rhs_vector[0] = left_bc
                rhs_vector[-1] = right_bc
        
        Notes
        -----
        - The local RHS contribution for each element is:
                RHS_local[i] = ∫ φ_i(r) * (-4πrρ(r)) dr
            These are assembled into a global vector, with contributions from
            adjacent elements added at shared boundary nodes.
        """
        # Type and shape validation
        assert isinstance(rho, np.ndarray), \
            RHO_TYPE_ERROR_MESSAGE.format(type(rho))
        assert rho.ndim == 1, \
            RHO_NDIM_ERROR_MESSAGE.format(rho.ndim)
        assert rho.shape[0] == self.number_of_finite_elements * self.quadrature_node_number, \
            RHO_SHAPE_ERROR_MESSAGE.format(rho.shape[0], self.number_of_finite_elements * self.quadrature_node_number)
        assert isinstance(z_nuclear, float), \
            Z_NUCLEAR_NOT_FLOAT_ERROR.format(type(z_nuclear))

        # Reshape density to element-wise structure
        rho_reshaped = rho.reshape(self.number_of_finite_elements, self.quadrature_node_number)
        
        # Compute source term at quadrature points: -4πrρ
        r_rho = self.quadrature_nodes_reshaped * rho_reshaped
        source = - 4.0 * np.pi * r_rho

        
        # Compute local RHS: ∫ φ_i(r) * source(r) dr
        rhs_vector_local = np.einsum(
            "emi,em,em->ei",
            self.lagrange_basis,              # φ_i at quadrature points
            self.quadrature_weights_reshaped, # quadrature weights
            source,                           # -4πrρ at quadrature points
            optimize=True
        )  # Shape: (n_elem, n_physical_nodes)

        # Assemble local vectors into global vector
        # Shared boundary nodes will have contributions from adjacent elements added
        rhs_vector = self._assemble_local_to_global_vector(rhs_vector_local)

        return rhs_vector


    def _assemble_local_to_global_vector(self, local_vector: np.ndarray) -> np.ndarray:
        """
        Assemble local element vectors into a global vector.
        
        This method takes local element-wise vectors (shape: [n_elements, n_dofs_per_elem])
        and assembles them into a single global vector by adding overlapping contributions 
        at shared nodes.
        
        Parameters
        ----------
        local_vector : np.ndarray
            Local element vectors with shape (n_elements, n_local_dofs)
            where n_local_dofs = polynomial_order + 1 = physical_node_number
            
        Returns
        -------
        np.ndarray
            Assembled global vector with shape (n_global_dofs,)
            where n_global_dofs = n_elements * polynomial_order + 1
            
        Notes
        -----
        The assembly process accounts for shared endpoints between adjacent elements,
        accumulating contributions from all elements that share a degree of freedom.
        
        For example, with 3 elements and polynomial order 2:
            Element 0: [a0, a1, a2]  → global indices [0, 1, 2]
            Element 1: [b0, b1, b2]  → global indices [2, 3, 4]  (b0 adds to global[2])
            Element 2: [c0, c1, c2]  → global indices [4, 5, 6]  (c0 adds to global[4])
        
        Global vector: [a0, a1, a2+b0, b1, b2+c0, c1, c2]
        """
        # Initialize global vector
        global_size = self.number_of_finite_elements * (self.physical_node_number - 1) + 1
        global_vector = np.zeros(global_size)
        
        # Get assembly indices (mapping from local to global DOFs)
        indices_global = self._build_assembly_indices_vector()
        
        # Assemble: add local contributions to global vector
        # np.add.at handles accumulation at repeated indices automatically
        np.add.at(global_vector, indices_global, local_vector.reshape(-1))
        
        return global_vector


    def _assemble_local_to_global_matrix(self, local_matrix: np.ndarray) -> np.ndarray:
        """
        Assemble local element matrices into a global matrix.
        
        This method takes local element-wise matrices (shape: [n_elements, n_dofs_per_elem, n_dofs_per_elem])
        and assembles them into a single global matrix by adding overlapping contributions at shared nodes.
        
        Parameters
        ----------
        local_matrix : np.ndarray
            Local element matrices with shape (n_elements, n_local_dofs, n_local_dofs)
            where n_local_dofs = polynomial_order + 1
            
        Returns
        -------
        np.ndarray
            Assembled global matrix with shape (n_global_dofs, n_global_dofs)
            where n_global_dofs = n_elements * polynomial_order + 1
            
        Notes
        -----
        The assembly process accounts for shared endpoints between adjacent elements,
        accumulating contributions from all elements that share a degree of freedom.
        """
        # Initialize global matrix
        global_size = self.number_of_finite_elements * (self.physical_node_number - 1) + 1
        global_matrix = np.zeros((global_size, global_size))
        
        # Get assembly indices (mapping from local to global DOFs)
        rows_global, cols_global = self._build_assembly_indices()
        
        # Assemble: add local contributions to global matrix
        np.add.at(global_matrix, (rows_global, cols_global), local_matrix.reshape(-1))
        
        return global_matrix
    
    
    def _build_assembly_indices_vector(self):
        """
        Return local→global indices for assembling element vectors into
        a global vector (with shared endpoints).
        
        This is similar to _build_assembly_indices but for vectors (1D).

        Primary FE space:
            local dofs per element       : N_grid = physical_node_number
            global stride (overlap by 1) : stride = N_grid - 1
            global dofs                  : N_elem * stride + 1
        
        Returns
        -------
        indices_global : np.ndarray
            1D array of global indices, shape (n_elements * n_local_dofs,)
            Maps local_vector.reshape(-1) to positions in global_vector
        
        Example
        -------
        For 3 elements with 3 nodes each (polynomial order 2):
            Element 0: local indices [0, 1, 2] → global indices [0, 1, 2]
            Element 1: local indices [0, 1, 2] → global indices [2, 3, 4]
            Element 2: local indices [0, 1, 2] → global indices [4, 5, 6]
        
        Returns: [0, 1, 2, 2, 3, 4, 4, 5, 6]
                  ^^^^^^^  ^^^^^^^  ^^^^^^^
                  elem 0   elem 1   elem 2
        
        Note the repeated indices (2, 2) and (4, 4) at element boundaries.
        """
        N_elem = self.number_of_finite_elements  # e for element
        N_grid = self.physical_node_number       # g for grid

        # compute the assembly indices
        stride = N_grid - 1
        elem_array = np.arange(N_elem)           # [0, 1, 2, ...]
        grid_array = np.arange(N_grid)           # [0, 1, 2, ..., N_grid-1]
        
        # Global index for each (element, local_node) pair
        # I_eg[elem, local_idx] = global_idx
        I_eg = elem_array[:, None] * stride + grid_array[None, :]   # (N_elem, N_grid)
        
        # Flatten to 1D array
        indices_global = I_eg.reshape(-1)
        
        return indices_global
    
    
    def _build_assembly_indices(self):
        """
        Return clean local→global row/col indices for assembling element blocks into
        a global matrix (with shared endpoints).

        Primary FE space:
            local dofs per element       : N_g    = physical_node_number - 1
            global stride (overlap by 1) : stride = N_g - 1
            global dofs                  : N_e * stride + 1
        """
        # ----- primary FE space -----
        N_elem = self.number_of_finite_elements # e for element
        N_grid = self.physical_node_number      # g for grid

        # compute the assembly indices
        stride = N_grid - 1
        elem_array = np.arange(N_elem)
        grid_array = np.arange(N_grid)
        I_eg = elem_array[:, None]*stride + grid_array[None, :]   # (N_elem, N_grid)

        rows_primary = np.repeat(I_eg[:, :, None], N_grid, axis=2).reshape(-1)
        cols_primary = np.repeat(I_eg[:, None, :], N_grid, axis=1).reshape(-1)

        return rows_primary, cols_primary



    @property
    def H_kinetic(self) -> np.ndarray:
        return self.get_H_kinetic()

    @property
    def H_r_inv_sq(self) -> np.ndarray:
        return self.get_H_r_inv_sq()

    @property
    def S(self) -> np.ndarray:
        return self.get_S()

    @property
    def S_inv_sqrt(self) -> np.ndarray:
        return self.get_S_inv_sqrt()

    @property
    def laplacian(self) -> np.ndarray:
        return self.get_laplacian()

    @property
    def derivative_matrix(self) -> np.ndarray:
        return self.get_derivative_matrix()

    @property
    def global_interpolation_matrix(self) -> np.ndarray:
        """
        Property accessor for global interpolation matrix.
        Used for HF/PBE0 calculations.
        """
        return self.get_global_interpolation_matrix()


    def evaluate_single_orbital_on_given_grid(
        self,
        given_grid: np.ndarray,
        orbital   : np.ndarray, 
        ) -> np.ndarray:
        """
        Evaluate a single orbital on a given grid using Lagrange interpolation.
        
        This function takes an orbital represented by its values at quadrature
        points and evaluates it at arbitrary grid points using finite element
        Lagrange basis functions. For each grid point, the function:
        1. Identifies which finite element contains the point
        2. Converts quadrature point values to nodal coefficients (using pseudoinverse)
        3. Evaluates the Lagrange basis functions at that point
        4. Computes the orbital value as a linear combination of basis functions
        
        Parameters
        ----------
        given_grid : np.ndarray, shape (n_points,)
            Grid points where the orbital should be evaluated.
            Must be monotonically increasing and within the physical domain.
        orbital : np.ndarray, shape (n_elem * n_quad,)
            Orbital values at quadrature points (global quadrature points).
            The orbital is stored as: ψ[r_quad] where r_quad are quadrature nodes.
        
        Returns
        -------
        orbital_values : np.ndarray, shape (n_points,)
            Orbital values evaluated at the given grid points.
            orbital_values[i] = ψ(given_grid[i])
        
        Implementation Logic
        --------------------
        1. **Input validation**:
           - Grid must be monotonically increasing
           - Grid must be within physical domain bounds
           - Orbital must be 1D array with shape (n_elem * n_quad,)
        
        2. **Reshape orbital to element structure**:
           - Global orbital: (n_elem * n_quad,) → Element-wise: (n_elem, n_quad)
           - Each row contains the orbital values at quadrature points for one element
        
        3. **For each element, compute pseudoinverse of Lagrange basis**:
           - lagrange_basis[elem_idx] has shape (n_quad, n_basis)
           - basis_pseudoinverse[elem_idx] maps quadrature values → nodal coefficients
        
        4. **For each grid point**:
           a. Find which element contains the point (using element boundaries)
           b. Convert quadrature values to nodal coefficients: c_nodal = L⁺ @ ψ_quad
           c. Evaluate Lagrange basis functions at that point
           d. Compute orbital value: ψ(x) = Σᵢ c_nodal_i * Lᵢ(x)
        
        5. **Handle boundary nodes**:
           - Boundary nodes are shared between adjacent elements
           - Use the element that naturally contains the point
           - For the last element, include the right boundary point
        
        Notes
        -----
        - The orbital input is at quadrature points, not physical nodes.
        - The interpolation uses Lagrange basis functions defined at physical nodes.
        - The conversion from quadrature points to nodal coefficients uses the
          pseudoinverse of the Lagrange basis matrix.
        - Points outside the domain are not allowed (will raise assertion error).
        
        Example
        -------
        >>> # Given orbital values at quadrature points
        >>> orbital = np.array([...])  # shape: (n_elem * n_quad,)
        >>> # Evaluate on a uniform grid
        >>> uniform_grid = np.linspace(0, domain_size, 1000)
        >>> orbital_values = ops_builder.evaluate_single_orbital_on_given_grid(
        ...     given_grid=uniform_grid,
        ...     orbital=orbital
        ... )
        >>> # orbital_values.shape = (1000,)
        """
        # Validate input: grid must be monotonically increasing
        assert np.all(np.diff(given_grid) > 0.0), \
            GIVEN_GRID_NOT_MONOTONICALLY_INCREASING_ERROR

        # Validate input: grid must be within physical domain
        assert np.all(given_grid >= self.physical_nodes[0]) and \
               np.all(given_grid <= self.physical_nodes[-1]), \
            GIVEN_GRID_NOT_WITHIN_PHYSICAL_NODES_ERROR

        # Validate input: orbital must be 1D array with correct shape
        assert orbital.ndim == 1, \
            ORBITAL_NDIM_ERROR_MESSAGE.format(orbital.ndim)
        assert orbital.shape[0] == self.number_of_finite_elements * self.quadrature_node_number, \
            ORBITAL_SHAPE_ERROR_MESSAGE.format(
                orbital.shape[0], 
                self.number_of_finite_elements * self.quadrature_node_number
            )

        n_elem = self.number_of_finite_elements
        n_quad = self.quadrature_node_number
        n_basis = self.physical_node_number
        n_points = len(given_grid)
        
        # Reshape orbital from global quadrature points to element-wise structure
        # Shape: (n_elem * n_quad,) -> (n_elem, n_quad)
        # Each row contains the orbital values at quadrature points for one element
        orbital_reshaped = orbital.reshape(n_elem, n_quad)
        
        # Compute pseudoinverse of Lagrange basis for each element
        # This maps quadrature point values to nodal coefficients
        # basis_pseudoinverse[elem_idx] has shape (n_basis, n_quad)
        basis_pseudoinverse = np.zeros((n_elem, n_basis, n_quad))
        for elem_idx in range(n_elem):
            # self.lagrange_basis[elem_idx] has shape (n_quad, n_basis)
            # Pseudoinverse maps: ψ_quad → c_nodal
            basis_pseudoinverse[elem_idx] = np.linalg.pinv(self.lagrange_basis[elem_idx])
        
        # Get element boundaries from physical nodes
        boundaries = np.zeros(n_elem + 1)
        boundaries[0] = self.physical_nodes_reshaped[0, 0]
        for elem_idx in range(n_elem):
            boundaries[elem_idx + 1] = self.physical_nodes_reshaped[elem_idx, -1]
        
        # Initialize output array
        orbital_values = np.zeros(n_points)
        
        # For each grid point, find its element and evaluate the orbital
        for point_idx, x_point in enumerate(given_grid):
            # Find which element contains this point
            # For the last element, include the right boundary
            elem_idx = None
            for e_idx in range(n_elem):
                left = boundaries[e_idx]
                right = boundaries[e_idx + 1]
                if e_idx == n_elem - 1:
                    # Last element: include right boundary
                    if left <= x_point <= right:
                        elem_idx = e_idx
                        break
                else:
                    # Other elements: exclude right boundary (handled by next element)
                    if left <= x_point < right:
                        elem_idx = e_idx
                        break
            
            if elem_idx is None:
                # This should not happen if assertions passed, but handle gracefully
                raise ValueError(f"Point {x_point} not found in any element")
            
            # Get orbital values at quadrature points for this element
            orbital_quad_elem = orbital_reshaped[elem_idx, :]  # Shape: (n_quad,)
            
            # Convert quadrature point values to nodal coefficients
            # c_nodal = L⁺ @ ψ_quad, where L⁺ is the pseudoinverse
            orbital_coef_nodal = basis_pseudoinverse[elem_idx] @ orbital_quad_elem  # Shape: (n_basis,)
            
            # Get physical nodes for this element (as row vector for LagrangeShapeFunctions)
            nodes_elem = self.physical_nodes_reshaped[elem_idx:elem_idx+1, :]  # Shape: (1, n_basis)
            
            # Evaluate Lagrange basis functions at this point
            # x_eval should be (1, 1) for single point
            basis_elem, _ = LagrangeShapeFunctions.lagrange_basis_and_derivatives(
                x_node=nodes_elem,           # (1, n_basis)
                x_eval=np.array([[x_point]])  # (1, 1)
            )
            # basis_elem shape: (1, 1, n_basis) -> (n_basis,)
            basis_values = basis_elem[0, 0, :]
            
            # Compute orbital value: ψ(x) = Σᵢ c_nodal_i * Lᵢ(x)
            orbital_values[point_idx] = np.dot(orbital_coef_nodal, basis_values)

        return orbital_values

