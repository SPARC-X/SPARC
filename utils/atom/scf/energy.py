"""
Energy calculator for Kohn-Sham DFT
Computes total energy and all components from converged SCF solution
"""

from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..mesh.operators import GridData
from ..utils.occupation_states import OccupationInfo
from ..scf.poisson import PoissonSolver
from ..xc.evaluator import XCEvaluator, XCPotentialData
from ..xc.hybrid import HartreeFockExchange
from ..mesh.operators import RadialOperatorsBuilder
from ..pseudo.local import LocalPseudopotential
from ..pseudo.non_local import NonLocalPseudopotential
from .density import DensityCalculator


if TYPE_CHECKING:
    from .density import DensityData


# Error messages
DERIVATIVE_MATRIX_SHAPE_ERROR_MESSAGE = \
    "parameter derivative_matrix's shape {shape} must match grid_data shape ({n_elem}, {n_quad}, {n_quad})"
Z_NUCLEAR_NOT_PROVIDED_ERROR_MESSAGE = \
    "parameter z_nuclear must be provided, get {z_nuclear} instead"
V_LOCAL_PSP_NOT_PROVIDED_ERROR_MESSAGE = \
    "parameter v_local_psp must be provided, get {v_local_psp} instead"
INTEGRAND_NDIM_ERROR_MESSAGE = \
    "Integrand should be one-dimensional, but got {ndim} dimensions"
MIXING_PARAMETER_NOT_A_FLOAT_ERROR = \
    "parameter mixing_parameter must be a float, get type {} instead"
MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter mixing_parameter must be in [0, 1], got {} instead"


@dataclass
class EnergyComponents:
    """Container for all energy components"""
    # Kinetic energy
    kinetic_radial  : float  # T_radial = (1/2) ∫ |dR/dr|² dr
    kinetic_angular : float  # T_angular = (1/2) Σ l(l+1) ∫ |R/r|² dr
    
    # Potential energies
    external        : float  # E_ext = ∫ ρ V_ext dr (nuclear or local pseudopotential)
    hartree         : float  # E_H = (1/2) ∫ ρ V_H dr
    exchange        : float  # E_x (from XC functional)
    correlation     : float  # E_c (from XC functional)
    
    # Optional: advanced functionals and corrections
    nonlocal_psp    : float = 0.0  # E_nl (non-local pseudopotential)
    hf_exchange     : float = 0.0  # E_HF (Hartree-Fock exchange)
    rpa_correlation : float = 0.0  # E_RPA (RPA correlation)
    
    @property
    def total_kinetic(self) -> float:
        """Total kinetic energy"""
        return self.kinetic_radial + self.kinetic_angular
    
    @property
    def total_potential(self) -> float:
        """Total potential energy"""
        return (self.external + self.nonlocal_psp + self.hartree + 
                self.exchange + self.correlation + 
                self.hf_exchange + self.rpa_correlation)
    
    @property
    def total(self) -> float:
        """Total energy"""
        return self.total_kinetic + self.total_potential
    
    def print_info(self, title: str = "Energy Components"):
        """Print formatted energy information"""
        # Title
        print(f"{'='*60}")
        print(f"\t\t {title}")
        print(f"{'='*60}")

        # Kinetic energy
        print(f"\t Kinetic (radial)     : {self.kinetic_radial:16.8f} Ha")
        print(f"\t Kinetic (angular)    : {self.kinetic_angular:16.8f} Ha")
        print(f"\t Total Kinetic        : {self.total_kinetic:16.8f} Ha")
        print(f"\t {'-'*42}")

        # External potential energy
        print(f"\t External potential   : {self.external:16.8f} Ha")

        print(f"\t Hartree              : {self.hartree:16.8f} Ha")
        print(f"\t Exchange             : {self.exchange:16.8f} Ha")
        print(f"\t Correlation          : {self.correlation:16.8f} Ha")
        if abs(self.nonlocal_psp) > 1e-12:
            print(f"\t Nonlocal PSP         : {self.nonlocal_psp:16.8f} Ha")
        if abs(self.hf_exchange) > 1e-12:
            print(f"\t HF Exchange          : {self.hf_exchange:16.8f} Ha")
        if abs(self.rpa_correlation) > 1e-12:
            print(f"\t RPA Correlation      : {self.rpa_correlation:16.8f} Ha")
        print(f"\t Total Potential      : {self.total_potential:16.8f} Ha")
        print(f"\t {'-'*42}")

        # Total energy
        print(f"\t TOTAL ENERGY         : {self.total:16.8f} Ha")
        print()


class EnergyCalculator:
    """
    Computes total energy from converged Kohn-Sham solution
    
    Total energy:
        E = T_s + E_ext + E_H + E_xc + E_HF + E_RPA
    
    where:
        T_s = kinetic energy of non-interacting electrons
        E_ext = external potential energy (nuclear or pseudopotential)
        E_H = Hartree energy (classical electron-electron repulsion)
        E_xc = exchange-correlation energy
        E_HF = Hartree-Fock exchange (optional)
        E_RPA = RPA correlation (optional)
    """
    
    def __init__(
        self,
        grid_data         : GridData,
        occupation_info   : OccupationInfo,
        ops_builder       : RadialOperatorsBuilder,
        poisson_solver    : PoissonSolver,
        pseudo            : LocalPseudopotential,
        xc_calculator     : Optional[XCEvaluator]         = None,
        hf_calculator     : Optional[HartreeFockExchange] = None,
        derivative_matrix : Optional[np.ndarray]          = None,
    ):
        """
        Parameters
        ----------
        grid_data : GridData
            Grid information (quadrature points, weights) from standard grid
        occupation_info : OccupationInfo
            Occupation numbers and quantum numbers
        ops_builder : RadialOperatorsBuilder
            Operator builder from standard grid
            Note: ops_builder should be from standard grid for general use
        poisson_solver : PoissonSolver
            Solver for Hartree potential
        pseudo : LocalPseudopotential
            Pseudopotential object containing all-electron or pseudopotential information
        xc_calculator : XCEvaluator or None
            XC functional evaluator for energy densities
        hf_calculator : HartreeFockExchange, optional
            HF exchange calculator for hybrid functionals
            Should be the same instance as used in SCFDriver
        derivative_matrix : np.ndarray, optional
            Derivative matrix for kinetic energy computation
            Note: If None, uses ops_builder.derivative_matrix.
                  If provided, should be from dense grid for more accurate results.
        """
        self.quadrature_nodes   = grid_data.quadrature_nodes
        self.quadrature_weights = grid_data.quadrature_weights
        self.occupation_info    = occupation_info
        self.ops_builder        = ops_builder
        self.poisson_solver     = poisson_solver
        self.pseudo             = pseudo
        self.xc_calculator      = xc_calculator
        self.hf_calculator      = hf_calculator
        
        # Use provided derivative_matrix or fall back to ops_builder's matrix
        if derivative_matrix is not None:
            self.derivative_matrix = derivative_matrix
        else:
            self.derivative_matrix = ops_builder.derivative_matrix
        
        # External potential - determined by pseudo.all_electron
        self.all_electron = pseudo.all_electron
        if self.all_electron:
            self.v_ext = -pseudo.z_nuclear / self.quadrature_nodes
        else:
            self.v_ext = pseudo.get_v_local_component_psp(self.quadrature_nodes)
        
        # NLCC (Non-linear Core Correction) density
        if self.all_electron:
            # For all-electron calculations, rho_nlcc = 0
            self.rho_nlcc = np.zeros_like(self.quadrature_nodes)
        else:
            # For pseudopotential calculations, XC functionals should use: rho_total = rho_valence + rho_nlcc
            self.rho_nlcc = pseudo.get_rho_core_correction(self.quadrature_nodes)
        
        # Non-local pseudopotential calculator
        if not self.all_electron:
            self.nonlocal_calculator = NonLocalPseudopotential(
                pseudo=self.pseudo,
                ops_builder=self.ops_builder
            )
        else:
            self.nonlocal_calculator = None
    
    
    def compute_energy(
        self,
        orbitals: np.ndarray,
        density_data: 'DensityData',
        mixing_parameter: Optional[float] = None,
        ) -> EnergyComponents:
        """
        Compute total energy from converged SCF solution
        
        Parameters
        ----------
        orbitals : np.ndarray
            Converged Kohn-Sham orbitals (radial wavefunctions R_nl(r))
            Shape: (n_states, n_quad_points)
        density_data : DensityData
            Electron density and related quantities (rho, grad_rho, tau)
            This density data should not include NLCC
        
        Returns
        -------
        energy : EnergyComponents
            All energy components
        
        Notes
        -----
        This is a simplified interface that handles all internal details.
        Matches reference code energy formula:
            E = E_Ts1 + E_Ts2 + E_ext + E_nonlocal + E_H + E_HF + E_RPA + E_x + E_c
        """
        if mixing_parameter is not None:
            assert isinstance(mixing_parameter, float), \
                "mixing_parameter must be a float, get type {} instead".format(type(mixing_parameter))
            assert 0.0 <= mixing_parameter <= 1.0, \
                "mixing_parameter must be in [0, 1], got {}".format(mixing_parameter)

        # 1. Kinetic energy
        T_radial, T_angular = self._compute_kinetic_energy(orbitals, self.derivative_matrix)
        
        # 2. External potential energy (local part)
        E_ext = self._compute_external_energy(density_data.rho)
        
        # 3. Hartree energy (solve internally)
        v_hartree = self.poisson_solver.solve_hartree(density_data.rho)
        E_hartree = self._compute_hartree_energy(density_data.rho, v_hartree)
        
        # 4. XC energy (compute internally using density data with NLCC correction)
        E_x, E_c = self._compute_xc_energy(density_data)
        
        # 5. Non-local pseudopotential energy
        E_nonlocal_psp = self._compute_nonlocal_psp_energy(orbitals)
        
        # 6. Hartree-Fock exchange energy (if HF calculator is available)
        E_hf_exchange = self._compute_hf_exchange_energy(orbitals)
        
        # 7. Advanced energy terms (placeholders for future implementation)
        E_rpa_correlation = 0.0

        if mixing_parameter is not None:
            E_x *= (1.0 - mixing_parameter)
            E_hf_exchange *= mixing_parameter
        
        return EnergyComponents(
            kinetic_radial=T_radial,
            kinetic_angular=T_angular,
            external=E_ext,
            hartree=E_hartree,
            exchange=E_x,
            correlation=E_c,
            nonlocal_psp=E_nonlocal_psp,
            hf_exchange=E_hf_exchange,
            rpa_correlation=E_rpa_correlation
        )
    
    
    def _compute_kinetic_energy(
        self,
        orbitals: np.ndarray,
        derivative_matrix: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute kinetic energy: T = T_radial + T_angular
        
        Given orbitals are R_nl(r), the radial wavefunction.
        Orbitals are stored as column vectors: shape = (n_grid, n_orbitals)
        
        T_radial = (1/2) Σ_i f_i ∫ |dR_i/dr|² dr
        T_angular = (1/2) Σ_i f_i l_i(l_i+1) ∫ |R_i/r|² dr
        
        Returns
        -------
        T_radial, T_angular : float, float
        """
        occupations = self.occupation_info.occupations
        l_values    = self.occupation_info.l_values

        # Save the shape information
        n_grid_tol, n_orbitals = orbitals.shape
        n_elem = derivative_matrix.shape[0] # Number of finite elements
        n_quad = derivative_matrix.shape[1] # Quadrature points per element
        assert n_grid_tol == n_elem * n_quad, \
            DERIVATIVE_MATRIX_SHAPE_ERROR_MESSAGE.format(shape = derivative_matrix.shape, n_elem = n_elem, n_quad = n_quad)

        # Reshape the orbitals to (n_elem, n_quad, n_orbitals)
        orbitals_reshaped = orbitals.reshape(n_elem, n_quad, n_orbitals)

        # Radial kinetic energy : T_radial
        # Formula: T_radial = 0.5 * sum(occ * (dR/dr)^2 * w)
        dR_dr = np.einsum('ejk,ekn->ejn', derivative_matrix, orbitals_reshaped).reshape(n_grid_tol, n_orbitals)
        T_radial = 0.5 * np.sum(occupations[np.newaxis, :] * (dR_dr ** 2) * self.quadrature_weights[:, np.newaxis])
        
        # Angular kinetic energy: T_angular
        # Formula: T_angular = sum(occ * l(l+1)/2 * (R/r)^2 * w)
        orb_div_r = orbitals / self.quadrature_nodes[:, np.newaxis]
        l_factor = l_values * (l_values + 1) / 2.0
        T_angular = np.sum(occupations[np.newaxis, :] * l_factor[np.newaxis, :] * (orb_div_r**2) * self.quadrature_weights[:, np.newaxis])
        
        return T_radial, T_angular
    
    
    def _compute_external_energy(self, rho: np.ndarray) -> float:
        """
        External potential energy: E_ext = ∫ ρ(r) v_ext(r) d³r
        """
        integrand = rho * self.v_ext
        return self._integrate(integrand)
    
    
    def _compute_hartree_energy(self, rho: np.ndarray, v_hartree: np.ndarray) -> float:
        """
        Hartree energy: E_H = (1/2) ∫ ρ(r) V_H(r) d³r
        
        The factor 1/2 avoids double-counting
        """
        integrand = rho * v_hartree
        return 0.5 * self._integrate(integrand)
    
    
    def _compute_xc_energy(self, density_data: 'DensityData') -> tuple[float, float]:
        """
        Compute exchange-correlation energy with NLCC correction.
        
        This method handles the proper inclusion of NLCC density for pseudopotential
        calculations. The valence density is updated to include core correction before
        computing XC energy.
        
        Parameters
        ----------
        density_data : DensityData
            Valence electron density data (does not include NLCC)
        
        Returns
        -------
        E_x, E_c : float, float
            Exchange and correlation energy components
        """
        if self.xc_calculator is None:
            return 0.0, 0.0
        
        # Get total density including NLCC for pseudopotential calculations
        density_data_total = self.get_total_density_data_for_xc(density_data)
        
        # Compute XC potential using total density
        xc_potential_data = self.xc_calculator.compute_xc(density_data_total)
        
        # Integrate energy densities
        E_x = self._integrate(density_data_total.rho * xc_potential_data.e_x)
        E_c = self._integrate(density_data_total.rho * xc_potential_data.e_c)
        
        return E_x, E_c
    
    
    def _integrate(self, integrand: np.ndarray) -> float:
        """
        Integrate in spherical coordinates: ∫ f(r) 4π r² dr, the integrand should be one-dimensional
        """
        assert integrand.ndim == 1, \
            INTEGRAND_NDIM_ERROR_MESSAGE.format(ndim = integrand.ndim)
        return np.sum(4 * np.pi * self.quadrature_nodes**2 * integrand * self.quadrature_weights)
    
    
    def get_total_density_data_for_xc(self, density_data: 'DensityData') -> 'DensityData':
        """
        Get total density data for exchange-correlation calculations.
        
        For pseudopotential calculations, XC functionals should use:
            rho_total = rho_valence + rho_nlcc
        
        For all-electron calculations:
            rho_total = rho_valence
        
        Note: grad_rho will be recomputed from the total density if it was provided,
              since grad_rho must be consistent with the total density.
        
        Parameters
        ----------
        density_data : DensityData
            Valence electron density data from KS orbitals
        
        Returns
        -------
        density_data_total : DensityData
            Total density data including NLCC correction
        """
        # Use static method to add NLCC correction
        # This properly handles recomputing grad_rho if needed
        density_data_total = DensityCalculator.add_nlcc_to_density_data(
            density_data_valence=density_data,
            rho_nlcc=self.rho_nlcc if not self.all_electron else None,
            quadrature_nodes=self.quadrature_nodes,
            derivative_matrix=self.derivative_matrix
        )
        
        return density_data_total
    
    
    def compute_xc_potential(self, density_data: 'DensityData'):
        """
        Compute exchange-correlation potential for given density data.
        
        This method handles NLCC correction for pseudopotential calculations.
        The input density_data contains valence density only, and NLCC is added
        for XC calculations.
        
        Parameters
        ----------
        density_data : DensityData
            Valence electron density data (does not include NLCC)
        
        Returns
        -------
        xc_potential_data : XCPotentialData
            Exchange-correlation potential and energy density data
        """
        if self.xc_calculator is None:
            # Return zero XC potential data when no XC calculator is available
            n_grid = len(self.quadrature_nodes)
            return XCPotentialData(
                v_x=np.zeros(n_grid),
                v_c=np.zeros(n_grid),
                e_x=np.zeros(n_grid),
                e_c=np.zeros(n_grid)
            )
        
        # Get total density data (including NLCC for pseudopotential calculations)
        density_data_total = self.get_total_density_data_for_xc(density_data)
        
        # Compute XC potential using total density
        xc_potential_data = self.xc_calculator.compute_xc(density_data_total)
        
        return xc_potential_data


    def _compute_nonlocal_psp_energy(self, orbitals: np.ndarray) -> float:
        """
        Compute non-local pseudopotential energy contribution.
        
        This implements the Kleinman-Bylander form:
            E_nl = Σ_l Σ_j γ_{lj} ⟨φ|χ_{lj}⟩⟨χ_{lj}|φ⟩
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions)
            Shape: (n_grid, n_orbitals)
        
        Returns
        -------
        float
            Non-local pseudopotential energy contribution
        """
        # Return 0.0 for all-electron calculations (no pseudopotential)
        if self.all_electron or self.nonlocal_calculator is None:
            return 0.0
        
        # Delegate computation to nonlocal calculator
        return self.nonlocal_calculator.compute_nonlocal_energy(
            orbitals=orbitals,
            occupations=self.occupation_info.occupations,
            l_values=self.occupation_info.l_values,
            unique_l_values=self.occupation_info.unique_l_values
        )


    def _compute_hf_exchange_energy(
        self,
        orbitals: np.ndarray
        ) -> float:
        """
        Compute Hartree-Fock exchange energy.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        float
            HF exchange energy (0.0 if no HF calculator available)
        """
        if self.hf_calculator is None:
            return 0.0
        
        # Delegate computation to HF calculator
        return self.hf_calculator.compute_exchange_energy(orbitals)


