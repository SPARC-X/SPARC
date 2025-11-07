"""
Density calculator for Kohn-Sham DFT
Computes electron density from orbitals and occupation numbers
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass

from ..mesh.operators import GridData
from ..utils.occupation_states import OccupationInfo

# Constant values
DEFAULT_DENSITY_VALUE = 1e-12

# Warning messages
RHO_INTEGRAL_TOO_SMALL_WARNING = \
    "Warning: Density integral almost zero, using default values {}"

# Error messages
GRID_DATA_TYPE_ERROR_MESSAGE = \
    "parameter grid_data must be an instance of GridData, get type {} instead"
DERIVATIVE_MATRIX_TYPE_ERROR_MESSAGE = \
    "parameter derivative_matrix must be an instance of np.ndarray, get type {} instead"
DERIVATIVE_MATRIX_SHAPE_ERROR_MESSAGE = \
    "parameter derivative_matrix's shape {shape} must match grid_data shape ({n_elem}, {n_quad}, {n_quad})"
DERIVATIVE_MATRIX_NDIM_ERROR_MESSAGE = \
    "parameter derivative_matrix must be a 3D array, get dimension {} instead"
OCCUPATION_INFO_TYPE_ERROR_MESSAGE = \
    "parameter occupation_info must be an instance of OccupationInfo, get type {} instead"
VERBOSE_TYPE_ERROR_MESSAGE = \
    "parameter verbose must be an instance of bool, get type {} instead"

RHO_NLCC_SHAPE_ERROR_MESSAGE = \
    "parameter rho_nlcc must have the same number of quadrature points as orbitals, get shape {} instead"


@dataclass
class DensityData:
    """Container for density and related quantities"""
    rho      : np.ndarray                   # electron density ρ(r)
    grad_rho : Optional[np.ndarray] = None  # |∇ρ(r)| for GGA
    tau      : Optional[np.ndarray] = None  # kinetic energy density for meta-GGA
    
    def __post_init__(self):
        """Validate shapes"""
        n_points = self.rho.shape[0]
        if self.grad_rho is not None:
            assert self.grad_rho.shape[0] == n_points
        if self.tau is not None:
            assert self.tau.shape[0] == n_points


class DensityCalculator:
    """
    Computes electron density from Kohn-Sham orbitals
    
    For spherically symmetric atoms, the density is:
        ρ(r) = (1/4π) Σ_i f_i |ψ_i(r)/r|²
    
    where f_i is the occupation number and ψ_i(r) is the radial wavefunction.
    """
    
    def __init__(
        self, 
        grid_data         : GridData, 
        occupation_info   : OccupationInfo, 
        derivative_matrix : np.ndarray, 
        verbose: bool = False):
        """
        Parameters
        ----------
        grid_data : GridData
            Grid information (quadrature points, weights)
        occupation_info : OccupationInfo
            Occupation numbers and quantum numbers
        derivative_matrix : np.ndarray
            Differentiation matrix
        verbose : bool
            If True, print normalization information during density calculation
        """

        self.quadrature_nodes   = grid_data.quadrature_nodes
        self.quadrature_weights = grid_data.quadrature_weights
        self.occupation_info    = occupation_info
        self.derivative_matrix  = derivative_matrix
        self.verbose            = verbose
        self.n_electrons        = np.sum(occupation_info.occupations)

        assert isinstance(grid_data, GridData), \
            GRID_DATA_TYPE_ERROR_MESSAGE.format(type(grid_data))
        assert isinstance(self.occupation_info, OccupationInfo), \
            OCCUPATION_INFO_TYPE_ERROR_MESSAGE.format(type(self.occupation_info))
        assert isinstance(self.derivative_matrix, np.ndarray), \
            DERIVATIVE_MATRIX_TYPE_ERROR_MESSAGE.format(type(self.derivative_matrix))
        assert isinstance(self.verbose, bool), \
            VERBOSE_TYPE_ERROR_MESSAGE.format(type(self.verbose))
        assert self._check_derivative_matrix_shape(self.derivative_matrix, grid_data)
    

    @staticmethod
    def _check_derivative_matrix_shape(derivative_matrix: np.ndarray, grid_data: GridData) -> bool:
        assert derivative_matrix.ndim == 3, \
            DERIVATIVE_MATRIX_NDIM_ERROR_MESSAGE.format(derivative_matrix.ndim)
        _n_elem = grid_data.number_of_finite_elements
        _n_quad = grid_data.quadrature_nodes.shape[0] // grid_data.number_of_finite_elements
        assert derivative_matrix.shape == (_n_elem, _n_quad, _n_quad), \
            DERIVATIVE_MATRIX_SHAPE_ERROR_MESSAGE.format(
                shape = derivative_matrix.shape, 
                n_elem = _n_elem, 
                n_quad = _n_quad)
        return True



    def compute_density(
        self, 
        orbitals: np.ndarray,
        normalize: bool = True
        ) -> np.ndarray:
        """
        Compute electron density from orbitals
        
        Parameters
        ----------
        orbitals : np.ndarray
            Radial wavefunctions, shape (n_states, n_quad_points)
            These are R_nl(r), not R_nl(r)/r
        normalize : bool
            If True, normalize density to correct number of electrons
        
        Returns
        -------
        rho : np.ndarray
            Electron density at quadrature points, shape (n_quad_points,)
        """
        
        # Convert radial wavefunction R(r) to true wavefunction ψ(r) = R(r)/r
        wavefunction = orbitals / self.quadrature_nodes[:, np.newaxis]

        
        # Density: ρ(r) = (1/4π) Σ f_i |ψ_i|²
        occupations = self.occupation_info.occupations[np.newaxis, :]


        rho = (1.0 / (4 * np.pi)) * np.sum(
            occupations * wavefunction**2,
            axis=1
        )
        
        # Normalize to correct electron count
        if normalize:
            rho = self.normalize_density(rho)

        return rho
    
    
    def compute_density_gradient(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute density gradient magnitude for GGA functionals in spherical coordinates.
        
        For spherical systems with radial symmetry: ∇ρ = dρ/dr · r̂
        So |∇ρ| = |dρ/dr|
        
        The radial derivative is computed using the chain rule:
            d(ρ·r)/dr = ρ + r·dρ/dr
        Rearranging:
            dρ/dr = [d(ρ·r)/dr - ρ] / r
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density at quadrature points, shape (N_quad,)
        
        Returns
        -------
        grad_rho : np.ndarray
            |∇ρ(r)| = |dρ/dr| at quadrature points, shape (N_quad,)
        
        Notes
        -----
        Matches reference implementation:
            grad_rho = (D @ ((ρ·r).reshape(N_e, q, 1)) - ρ) / r
        
        This is used by GGA functionals (PBE, etc.) to compute σ = |∇ρ|².
        """
        # Get dimensions
        n_elem = self.derivative_matrix.shape[0]  # number of elements
        n_quad = self.derivative_matrix.shape[1]    # quadrature points per element
        
        # Compute ρ·r and reshape for element-wise differentiation
        rho_times_r = rho * self.quadrature_nodes
        rho_times_r_reshaped = rho_times_r.reshape(n_elem, n_quad, 1)
        
        # Apply derivative matrix: D @ (ρ·r)
        d_rho_r_dr = np.matmul(self.derivative_matrix, rho_times_r_reshaped).reshape(-1)
        
        # Compute dρ/dr = [d(ρ·r)/dr - ρ] / r
        grad_rho = (d_rho_r_dr - rho) / self.quadrature_nodes
        
        return grad_rho
    
    
    def compute_kinetic_energy_density(self, orbitals: np.ndarray) -> np.ndarray:
        """
        Compute kinetic energy density τ for meta-GGA functionals.
        
        τ(r) = (1/2) Σ_i f_i [|∂(ψ_i)/∂r|² + l(l+1)|ψ_i/r|²] / (4π)
        
        Parameters
        ----------
        orbitals : np.ndarray
            Radial wavefunctions R_nl(r), shape (n_grid, n_states)
        
        Returns
        -------
        tau : np.ndarray
            Kinetic energy density at quadrature points, shape (n_grid,)
        """
        # Get dimensions
        n_grid, n_states = orbitals.shape         # orbitals: (n_quad_points, n_states)
        n_elem = self.derivative_matrix.shape[0]  # Number of finite elements
        n_quad = self.derivative_matrix.shape[1]  # Quadrature points per element
        
        # Convert R(r) to ψ(r) = R(r)/r
        orbitals_by_r = orbitals / self.quadrature_nodes[:, np.newaxis]
        
        # Compute ∂R/∂r using finite element derivative matrix
        orbitals_reshaped = orbitals.reshape(n_elem, n_quad, n_states)
        diff_orbitals = np.matmul(self.derivative_matrix, orbitals_reshaped).reshape(n_grid, n_states)
        
        # Compute ∂(ψ)/∂r = [∂R/∂r - R/r] / r
        diff_orbitals_by_r = (diff_orbitals - orbitals_by_r) / self.quadrature_nodes[:, np.newaxis]
        
        # Get occupation numbers and angular momentum quantum numbers
        occupations = self.occupation_info.occupations # Shape: (n_states,)
        l_values    = self.occupation_info.l_values    # Shape: (n_states,)
        
        # Term 1: Radial derivative contribution |∂(ψ)/∂r|²
        term1 = 0.5 * occupations[np.newaxis, :] * diff_orbitals_by_r**2
        
        # Term 2: Angular momentum contribution l(l+1)|ψ/r|²
        l_times_l_plus_1 = l_values * (l_values + 1)
        term2 = 0.5 * occupations[np.newaxis, :] * l_times_l_plus_1[np.newaxis, :] * \
                (orbitals_by_r / self.quadrature_nodes[:, np.newaxis])**2
        
        # Sum over orbitals and divide by 4π
        tau = np.sum(term1 + term2, axis=-1) / (4 * np.pi)
        
        return tau
    
    
    def create_density_data_from_rho(
        self,
        rho: np.ndarray,
        compute_gradient: bool = False
        ) -> DensityData:
        """
        Create DensityData from electron density ρ
        
        This method constructs DensityData from an existing density array.
        Note: tau (kinetic energy density) cannot be computed from ρ alone,
        so it is always set to None when using this method.
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density, shape (n_quad_points,)
        compute_gradient : bool, optional
            If True, compute |∇ρ| for GGA functionals
            If False, grad_rho is set to None
            Default: False
        
        Returns
        -------
        density_data : DensityData
            Container with ρ, optionally |∇ρ|, and tau=None
        
        Examples
        --------
        >>> # LDA: only need density
        >>> density_data = calc.create_density_data_from_rho(rho, compute_gradient=False)
        >>> 
        >>> # GGA: need density and gradient
        >>> density_data = calc.create_density_data_from_rho(rho, compute_gradient=True)
        """
        # Compute gradient if requested
        grad_rho = None
        if compute_gradient:
            grad_rho = self.compute_density_gradient(rho)
        
        # tau cannot be computed from rho alone, always None
        tau = None
        
        return DensityData(
            rho      = rho, 
            grad_rho = grad_rho, 
            tau      = tau
        )
    
    
    def create_density_data_from_mixed(
        self,
        rho_mixed        : np.ndarray,
        orbitals         : Optional[np.ndarray] = None,
        compute_gradient : bool = False,
        compute_tau      : bool = False,
        rho_nlcc         : Optional[np.ndarray] = None
        ) -> DensityData:
        """
        Create DensityData from mixed density and orbitals.
        
        This method is specifically designed for SCF iterations where:
        - rho is taken from input rho_mixed (NOT recomputed from orbitals)
        - grad_rho is computed from the input rho_mixed + rho_nlcc
        - tau is computed from input orbitals (only if compute_tau=True)
        
        Key difference from create_density_data_from_orbitals:
        The density rho is NOT recomputed from orbitals, but taken directly
        from the input rho_mixed. This is because in SCF mixing, the mixed
        density may not correspond exactly to any orbital-based density.
        
        Use this method AFTER mixing in SCF loops.
        
        Parameters
        ----------
        rho_mixed : np.ndarray
            Mixed valence density from mixer.mix(), shape (n_quad_points,)
            This is the density to be used in the next iteration
        orbitals : np.ndarray, optional
            Current orbitals, shape (n_states, n_quad_points)
            Only required if compute_tau=True
            Default: None
        compute_gradient : bool, optional
            If True, compute |∇ρ| for GGA functionals
            Default: False
        compute_tau : bool, optional
            If True, compute kinetic energy density τ for meta-GGA functionals
            Requires orbitals to be provided
            Default: False
        rho_nlcc : np.ndarray, optional
            Non-linear core correction density from pseudopotential
            If provided, will be added to rho_mixed: ρ_total = ρ_mixed + ρ_nlcc
            Default: None (no core correction)
        
        Returns
        -------
        density_data : DensityData
            Container with ρ, optionally |∇ρ| and τ
            Note: rho = rho_mixed + rho_nlcc (not recomputed from orbitals)
        
        Raises
        ------
        ValueError
            If compute_tau=True but orbitals is None
        
        Examples
        --------
        >>> # LDA/GGA: no tau needed
        >>> rho_mixed = mixer.mix(rho, rho_new)
        >>> density_data = calc.create_density_data_from_mixed(
        ...     rho_mixed=rho_mixed,
        ...     compute_gradient=True,
        ...     compute_tau=False
        ...     # orbitals not needed
        ... )
        >>> 
        >>> # meta-GGA: tau needed
        >>> density_data = calc.create_density_data_from_mixed(
        ...     rho_mixed=rho_mixed,
        ...     orbitals=orbitals,     # Required for tau
        ...     compute_gradient=True,
        ...     compute_tau=True,
        ...     rho_nlcc=rho_nlcc
        ... )
        
        Notes
        -----
        Key difference from create_density_data_from_orbitals:
        - rho is taken from input rho_mixed (NOT recomputed from orbitals)
        - rho comes from mixing: rho_mixed = mixer.mix(rho_in, rho_out)
        - grad_rho is computed from the mixed density (rho_mixed + rho_nlcc)
        - tau is computed from input orbitals (only if requested)
        
        Why "mixed"? Because the density rho is not computed from orbitals
        but taken directly from the input rho_mixed. In SCF mixing, the
        mixed density may not correspond exactly to any orbital-based density.
        """
        # Validate: if tau is requested, orbitals must be provided
        if compute_tau and orbitals is None:
            raise ValueError(
                "orbitals must be provided when compute_tau=True. "
                "Cannot compute kinetic energy density without orbitals."
            )
        
        # Total density for XC (mixed + NLCC)
        rho_total = rho_mixed + rho_nlcc if rho_nlcc is not None else rho_mixed
        
        # Compute gradient from mixed total density
        grad_rho = None
        if compute_gradient:
            grad_rho = self.compute_density_gradient(rho_total)
        
        # Compute tau from orbitals (only if requested)
        tau = None
        if compute_tau:
            tau = self.compute_kinetic_energy_density(orbitals)
        
        return DensityData(
            rho      = rho_total,
            grad_rho = grad_rho,
            tau      = tau
        )
    
    
    def create_density_data_from_orbitals(
        self,
        orbitals         : np.ndarray,
        compute_gradient : bool = False,
        compute_tau      : bool = False,
        normalize        : bool = True,
        rho_nlcc         : Optional[np.ndarray] = None
        ) -> DensityData:
        """
        Create DensityData from Kohn-Sham orbitals
        
        This method computes density from orbitals and optionally computes
        gradient and kinetic energy density for advanced functionals.
        
        For pseudopotential calculations, the NLCC (non-linear core correction)
        density can be added to the valence density for XC functional evaluation.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Radial wavefunctions, shape (n_states, n_quad_points)
        compute_gradient : bool, optional
            If True, compute |∇ρ| for GGA functionals
            If False, grad_rho is set to None
            Default: False
        compute_tau : bool, optional
            If True, compute kinetic energy density τ for meta-GGA functionals
            If False, tau is set to None
            Default: False
        normalize : bool, optional
            If True, normalize density to match total electron count
            Default: True
        rho_nlcc : np.ndarray, optional
            Non-linear core correction density from pseudopotential
            If provided, will be added to valence density: ρ_total = ρ_valence + ρ_nlcc
            This is used for XC functional evaluation in pseudopotential calculations
            Default: None (no core correction)
        
        Returns
        -------
        density_data : DensityData
            Container with ρ, optionally |∇ρ| and τ
            Note: If rho_nlcc is provided, ρ includes the core correction
        
        Examples
        --------
        >>> # LDA: only need density
        >>> density_data = calc.create_density_data_from_orbitals(
        ...     orbitals, 
        ...     compute_gradient=False, 
        ...     compute_tau=False
        ... )
        >>> 
        >>> # GGA: need density and gradient
        >>> density_data = calc.create_density_data_from_orbitals(
        ...     orbitals, 
        ...     compute_gradient=True, 
        ...     compute_tau=False
        ... )
        >>> 
        >>> # meta-GGA: need density, gradient, and tau
        >>> density_data = calc.create_density_data_from_orbitals(
        ...     orbitals, 
        ...     compute_gradient=True, 
        ...     compute_tau=True
        ... )
        >>> 
        >>> # Pseudopotential with NLCC
        >>> density_data = calc.create_density_data_from_orbitals(
        ...     orbitals,
        ...     compute_gradient=True,
        ...     rho_nlcc=pseudo.get_rho_core_correction(r_nodes)
        ... )
        
        Notes
        -----
        For pseudopotential calculations:
        - Valence density ρ_v is computed from orbitals
        - NLCC density ρ_nlcc represents frozen core electrons
        - Total density ρ_total = ρ_v + ρ_nlcc is used for XC evaluation
        - This improves XC energy accuracy compared to valence-only
        """

        if rho_nlcc is not None:
            assert rho_nlcc.shape[0] == orbitals.shape[0], \
                RHO_NLCC_SHAPE_ERROR_MESSAGE.format(rho_nlcc.shape)

        # Compute valence density from orbitals
        rho_valence = self.compute_density(orbitals, normalize=normalize)
        
        # Add NLCC if provided (for pseudopotential calculations)
        if rho_nlcc is not None:
            rho = rho_valence + rho_nlcc
        else:
            rho = rho_valence
        
        # Compute gradient if requested (uses total density including NLCC)
        grad_rho = None
        if compute_gradient:
            grad_rho = self.compute_density_gradient(rho)
        
        # Compute tau if requested (only from valence orbitals)
        tau = None
        if compute_tau:
            tau = self.compute_kinetic_energy_density(orbitals)
        
        return DensityData(
            rho      = rho, 
            grad_rho = grad_rho, 
            tau      = tau
        )
    

    
    def normalize_density(self, rho: np.ndarray) -> np.ndarray:
        """
        Normalize density to integrate to correct number of electrons
        
        Ensures that ∫ 4π r² ρ(r) dr = N_electrons by rescaling the density.
        
        Parameters
        ----------
        density : np.ndarray
            Unnormalized electron density at quadrature points
        
        Returns
        -------
        normalized_density : np.ndarray
            Density rescaled to integrate to the correct electron count
        """
        # Compute the integrated electron count from current density
        # ∫ 4π r² ρ(r) dr using quadrature: Σ 4π r² ρ(r) w
        integrated_electron_count = np.sum(
            4 * np.pi * self.quadrature_nodes**2 * rho * self.quadrature_weights
        )
        
        # Handle edge case: if density is almost zero
        if integrated_electron_count < DEFAULT_DENSITY_VALUE:
            print(RHO_INTEGRAL_TOO_SMALL_WARNING.format(DEFAULT_DENSITY_VALUE))
            return np.ones_like(rho) * DEFAULT_DENSITY_VALUE
        
        # Rescale density to match target electron count
        scaling_factor = self.n_electrons / integrated_electron_count
        normalized_rho = rho * scaling_factor
        
        return normalized_rho
    
    
    def check_normalization(self, rho: np.ndarray) -> float:
        """
        Check how many electrons the density integrates to
        
        Returns the integrated electron count
        """
        return np.sum(4 * np.pi * self.quadrature_nodes**2 * rho * self.quadrature_weights)
    
    @staticmethod
    def add_nlcc_to_density_data(
        density_data_valence: 'DensityData',
        rho_nlcc: Optional[np.ndarray],
        quadrature_nodes: Optional[np.ndarray] = None,
        derivative_matrix: Optional[np.ndarray] = None
        ) -> 'DensityData':
        """
        Add NLCC (non-linear core correction) to valence density data.
        
        This method properly handles the NLCC correction for pseudopotential calculations.
        When rho_nlcc is added to valence density:
        - rho_total = rho_valence + rho_nlcc
        - grad_rho must be recomputed from rho_total (not reused from valence)
        - tau remains unchanged (only depends on valence orbitals)
        
        Parameters
        ----------
        density_data_valence : DensityData
            Valence electron density data (from KS orbitals)
        rho_nlcc : np.ndarray, optional
            Non-linear core correction density
            If None, returns the input density_data_valence unchanged
        quadrature_nodes : np.ndarray, optional
            Quadrature node positions, shape (n_quad_points,)
            Required if density_data_valence.grad_rho is not None
        derivative_matrix : np.ndarray, optional
            Derivative matrix for computing grad_rho
            Required if density_data_valence.grad_rho is not None
            Shape: (n_elem, n_quad, n_quad)
        
        Returns
        -------
        density_data_total : DensityData
            Total density data including NLCC correction
            - rho = rho_valence + rho_nlcc
            - grad_rho = recomputed from total density (if grad_rho was provided)
            - tau = same as input (unchanged)
        
        Raises
        ------
        ValueError
            If rho_nlcc is provided and grad_rho is not None, but quadrature_nodes
            or derivative_matrix are missing
        
        Notes
        -----
        This static method follows the same logic as DensityCalculator.compute_density_gradient
        for consistency with the rest of the codebase.
        
        Examples
        --------
        >>> # LDA: no gradient needed
        >>> density_total = DensityCalculator.add_nlcc_to_density_data(
        ...     density_data_valence,
        ...     rho_nlcc=rho_nlcc
        ... )
        >>> 
        >>> # GGA: gradient needs to be recomputed
        >>> density_total = DensityCalculator.add_nlcc_to_density_data(
        ...     density_data_valence,
        ...     rho_nlcc=rho_nlcc,
        ...     quadrature_nodes=r_nodes,
        ...     derivative_matrix=D
        ... )
        """
        # If no NLCC provided, return input unchanged
        if rho_nlcc is None:
            return density_data_valence
        
        # Add NLCC to valence density
        rho_total = density_data_valence.rho + rho_nlcc
        
        # Handle gradient: if it was provided, it needs to be recomputed
        grad_rho = None
        if density_data_valence.grad_rho is not None:
            # Need quadrature_nodes and derivative_matrix to recompute gradient
            if quadrature_nodes is None or derivative_matrix is None:
                raise ValueError(
                    "quadrature_nodes and derivative_matrix must be provided "
                    "when density_data_valence.grad_rho is not None, "
                    "since grad_rho needs to be recomputed from the total density."
                )
            
            # Recompute gradient using same logic as compute_density_gradient
            N_e = derivative_matrix.shape[0]  # number of elements
            q = derivative_matrix.shape[1]    # quadrature points per element
            
            # Compute ρ·r and reshape for element-wise differentiation
            rho_times_r = rho_total * quadrature_nodes
            rho_times_r_reshaped = rho_times_r.reshape(N_e, q, 1)
            
            # Apply derivative matrix: D @ (ρ·r)
            d_rho_r_dr = np.matmul(derivative_matrix, rho_times_r_reshaped).reshape(-1)
            
            # Compute dρ/dr = [d(ρ·r)/dr - ρ] / r
            grad_rho = (d_rho_r_dr - rho_total) / quadrature_nodes
        
        # tau remains unchanged (only depends on valence orbitals)
        tau = density_data_valence.tau
        
        return DensityData(
            rho=rho_total,
            grad_rho=grad_rho,
            tau=tau
        )

