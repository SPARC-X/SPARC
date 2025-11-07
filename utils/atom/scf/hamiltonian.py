"""
Hamiltonian builder for Kohn-Sham DFT
Responsible for constructing the total Hamiltonian matrix from various potential components
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass

from ..mesh.operators import RadialOperatorsBuilder
from ..pseudo.local import LocalPseudopotential
from ..utils.occupation_states import OccupationInfo


# Error messages
OPS_BUILDER_MUST_BE_A_RADIAL_OPERATORS_BUILDER_ERROR = \
    "parameter ops_builder must be a RadialOperatorsBuilder, get type {} instead"
LOCAL_PSEUDOPOTENTIAL_MUST_BE_A_LOCAL_PSEUDOPOTENTIAL_ERROR = \
    "parameter pseudo must be a LocalPseudopotential, get type {} instead"
OCCUPATION_INFO_MUST_BE_AN_OCCUPATION_INFO_ERROR = \
    "parameter occupation_info must be an OccupationInfo, get type {} instead"
ALL_ELECTRON_MUST_BE_A_BOOLEAN_ERROR = \
    "parameter all_electron must be a boolean, get type {} instead"

EIGENVECTORS_MUST_BE_A_NUMPY_ARRAY_ERROR = \
    "parameter eigenvectors must be a numpy array, get type {} instead"
EIGENVECTORS_MUST_BE_A_2D_ARRAY_ERROR = \
    "parameter eigenvectors must be a 2D array, get dimension {} instead"
HARTREE_FOCK_EXCHANGE_MATRIX_FOR_L_CHANNEL_NOT_AVAILABLE_ERROR = \
    "Hartree-Fock exchange matrix for l={l} is not available, please set the HF exchange matrices first"

MIXING_PARAMETER_NOT_A_FLOAT_ERROR = \
    "parameter mixing_parameter must be a float, get type {} instead"
MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter mixing_parameter must be in [0, 1], got {} instead"

DE_XC_DTAU_NOT_AVAILABLE_ERROR = \
    "parameter de_xc_dtau is not available for l={l}, please set the de_xc_dtau first"


# Warning messages
HARTREE_FOCK_EXCHANGE_MATRIX_FOR_L_CHANNEL_NOT_AVAILABLE_WARNING = \
    "WARNING: Hartree-Fock exchange matrix for l={l} is not available, please set the HF exchange matrices first"
HF_EXCHANGE_MATRIX_NOT_AVAILABLE_WARNING = \
    "WARNING: Hartree-Fock exchange matrix is not available, please set the HF exchange matrices first"




class SwitchesFlags:
    """
    Internal helper class for HamiltonianBuilder to manage functional switches.
    Determines which Hamiltonian components to include based on the XC functional.
    
    Attributes:
        use_hf_exchange (bool): Whether to use Hartree-Fock exchange
        use_oep_exchange (bool): Whether to use Optimized Effective Potential (OEP) exchange
        use_oep_correlation (bool): Whether to use OEP correlation
        use_rpa_correlation (bool): Whether to use Random Phase Approximation (RPA) correlation
        use_metagga (bool): Whether to use meta-GGA functional (requires de_xc_dtau in Hamiltonian)
        OEP (bool): Whether to use OEP methods
        inside_OEP (bool): Whether currently inside OEP iteration
        uscrpa (bool): Whether to use RPA (Random Phase Approximation) - "use RPA" flag
    """
    def __init__(self, xc_functional: str, hybrid_mixing_parameter: Optional[float] = None):
        """
        Initialize switches based on XC functional and hybrid mixing parameter.
        
        Parameters
        ----------
        xc_functional : str
            Name of XC functional
        hybrid_mixing_parameter : float, optional
            Mixing parameter for hybrid functionals (0-1). Required only for hybrid functionals.
            - For PBE0: should be 0.25
            - For HF: should be 1.0
        """
        # Type checking
        assert isinstance(xc_functional, str), \
            "parameter xc_functional must be a string, get type {} instead".format(type(xc_functional))
        if hybrid_mixing_parameter is not None:
            assert isinstance(hybrid_mixing_parameter, (float, int)), \
                "parameter hybrid_mixing_parameter must be a float, get type {} instead".format(type(hybrid_mixing_parameter))
            hybrid_mixing_parameter = float(hybrid_mixing_parameter)
        
        self.xc_functional = xc_functional
        self.hybrid_mixing_parameter = hybrid_mixing_parameter
        
        # Initialize all flags to False
        self.OEP = False
        self.inside_OEP = False
        self.uscrpa = False
        self.use_hf_exchange = False
        self.use_oep_exchange = False
        self.use_oep_correlation = False
        self.use_rpa_correlation = False
        self.use_metagga = False
        
        # Set flags based on functional
        if xc_functional == 'RPA':
            self.OEP = True
            self.uscrpa = True
            self.use_oep_exchange = True
            self.use_oep_correlation = True
            self.use_rpa_correlation = True
        elif xc_functional == 'OEPx':
            self.OEP = True
            self.uscrpa = True
            self.use_oep_exchange = True
            self.use_oep_correlation = True
        elif xc_functional == 'HF':
            self.use_hf_exchange = True
            # Check if hybrid_mixing_parameter is provided for hybrid functionals
            if hybrid_mixing_parameter is None:
                print("WARNING: hybrid_mixing_parameter not provided for HF functional, using default value 1.0")
                self.hybrid_mixing_parameter = 1.0
            elif not np.isclose(hybrid_mixing_parameter, 1.0):
                print("WARNING: hybrid_mixing_parameter for HF should be 1.0, got {}".format(hybrid_mixing_parameter))
        elif xc_functional == 'PBE0':
            self.use_hf_exchange = True
            # Check if hybrid_mixing_parameter is provided for hybrid functionals
            if hybrid_mixing_parameter is None:
                print("WARNING: hybrid_mixing_parameter not provided for PBE0 functional, using default value 0.25")
                self.hybrid_mixing_parameter = 0.25
            elif not np.isclose(hybrid_mixing_parameter, 0.25):
                print("WARNING: hybrid_mixing_parameter for PBE0 should be 0.25, got {}".format(hybrid_mixing_parameter))
        
        # Set meta-GGA flag for functionals that require de_xc_dtau
        if xc_functional in ['SCAN', 'RSCAN', 'R2SCAN']:
            self.use_metagga = True
        
        # LDA/GGA functionals: no special flags (default False)
    
    def print_info(self):
        """
        Print functional switch information summary.
        """
        print("=" * 60)
        print("\t\t FUNCTIONAL SWITCHES")
        print("=" * 60)
        print(f"\t XC Functional              : {self.xc_functional}")
        if self.hybrid_mixing_parameter is not None:
            print(f"\t Hybrid Mixing Parameter    : {self.hybrid_mixing_parameter}")
        else:
            print(f"\t Hybrid Mixing Parameter    : None (not applicable)")
        print()
        print("\t FUNCTIONAL FLAGS:")
        print(f"\t use_hf_exchange            : {self.use_hf_exchange}")
        print(f"\t use_oep_exchange           : {self.use_oep_exchange}")
        print(f"\t use_oep_correlation        : {self.use_oep_correlation}")
        print(f"\t use_rpa_correlation        : {self.use_rpa_correlation}")
        print(f"\t use_metagga                : {self.use_metagga}")
        print(f"\t OEP                        : {self.OEP}")
        print(f"\t inside_OEP                 : {self.inside_OEP}")
        print(f"\t use_RPA (uscrpa)           : {self.uscrpa}")
        print()


class HamiltonianBuilder:
    """
    Constructs Kohn-Sham Hamiltonian matrices for each angular momentum channel
    
    H_l = T + V_ext + V_H + V_xc + l(l+1)/(2r²) + V_nl(l) + V_HF(l)
    
    where:
    - T: kinetic energy (radial derivative)
    - V_ext: external potential (nuclear or pseudopotential)
    - V_H: Hartree potential (electron-electron repulsion)
    - V_xc: exchange-correlation potential
    - l(l+1)/(2r²): angular momentum centrifugal term
    - V_nl(l): non-local pseudopotential (l-dependent)
    - V_HF(l): Hartree-Fock exchange (l-dependent, optional)
    
    This class is self-contained: it computes all fixed matrices internally
    from the operators builder and pseudopotential information.
    """
    
    def __init__(
        self,
        ops_builder     : RadialOperatorsBuilder,
        pseudo          : LocalPseudopotential,
        occupation_info : OccupationInfo,
        all_electron    : bool = True
        ):
        """
        Parameters
        ----------
        ops_builder : RadialOperatorsBuilder
            Operator builder for constructing matrices
        pseudo : LocalPseudopotential
            Pseudopotential information (z_nuclear, z_valence, projectors, etc.)
        occupation_info : OccupationInfo
            Occupation information (needed for l channels in non-local PSP)
        all_electron : bool
            If True, use nuclear Coulomb potential; if False, use pseudopotential
        """
        self.ops_builder = ops_builder
        self.pseudo = pseudo
        self.occupation_info = occupation_info
        self.all_electron = all_electron

        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_MUST_BE_A_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder))
        assert isinstance(pseudo, LocalPseudopotential), \
            LOCAL_PSEUDOPOTENTIAL_MUST_BE_A_LOCAL_PSEUDOPOTENTIAL_ERROR.format(type(pseudo))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_MUST_BE_AN_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        assert isinstance(all_electron, bool), \
            ALL_ELECTRON_MUST_BE_A_BOOLEAN_ERROR.format(type(all_electron))
        
        # Pre-compute fixed matrices
        self._compute_fixed_matrices()
        
        # Optional: Hartree-Fock exchange matrices (set by outer loop)
        self.H_hf_exchange_dict: Dict[int, np.ndarray] = {}
    
    
    def _compute_fixed_matrices(self):
        """
        Pre-compute all fixed Hamiltonian matrix components
        
        These matrices are computed once and reused in all SCF iterations:
        - H_kinetic: kinetic energy operator
        - H_ext: external potential (nuclear or local pseudopotential)
        - H_r_inv_sq: 1/r² operator for angular momentum term
        - H_nonlocal: non-local pseudopotential matrices (if applicable)
        """
        # 1. Kinetic energy matrix
        self.H_kinetic = self.ops_builder.get_H_kinetic()
        
        # 2. External potential matrix
        if self.all_electron:
            # All-electron: nuclear Coulomb potential V = -Z/r
            V_external = self.ops_builder.get_nuclear_coulomb_potential(
                self.pseudo.z_nuclear
            )
        else:
            # Pseudopotential: local component
            V_external = self.pseudo.get_v_local_component_psp(
                self.ops_builder.quadrature_nodes
            )
        
        self.H_ext = self.ops_builder.build_potential_matrix(V_external)
        
        # 3. Angular momentum operator: 1/r²
        self.H_r_inv_sq = self.ops_builder.get_H_r_inv_sq()
        
        # 4. Non-local pseudopotential matrices (if using pseudopotential)
        if not self.all_electron:
            from ..pseudo.non_local import NonLocalPseudopotential
            
            nonlocal_calculator = NonLocalPseudopotential(
                pseudo=self.pseudo,
                ops_builder=self.ops_builder
            )
            
            self.H_nonlocal = nonlocal_calculator.compute_all_nonlocal_matrices(
                l_channels=self.occupation_info.unique_l_values
            )
        else:
            self.H_nonlocal = {}
    
    
    def build_for_l_channel(
        self,
        l: int,
        v_hartree: np.ndarray,
        v_x: np.ndarray,
        v_c: np.ndarray,
        switches: SwitchesFlags,
        v_x_oep: Optional[np.ndarray] = None,
        v_c_oep: Optional[np.ndarray] = None,
        de_xc_dtau: Optional[np.ndarray] = None,
        symmetrize: bool = False
        ) -> np.ndarray:
        """
        Build total Hamiltonian for angular momentum channel l
        
        Parameters
        ----------
        l : int
            Angular momentum quantum number
        v_hartree : np.ndarray
            Hartree potential V_H(r) at quadrature points
        v_x : np.ndarray
            Exchange potential V_x(r) at quadrature points
        v_c : np.ndarray
            Correlation potential V_c(r) at quadrature points
        switches : SwitchesFlags
            Functional switches determining which components to include
        v_x_oep : np.ndarray, optional
            OEP exchange potential (if used)
        v_c_oep : np.ndarray, optional
            OEP correlation potential (if used)
        de_xc_dtau : np.ndarray, optional
            Derivative ∂ε_xc/∂τ of XC energy density w.r.t. kinetic energy density (for meta-GGA)
        symmetrize : bool, optional
            Whether to apply symmetrization using S^(-1/2) (default: False)
            Transforms: H → S^(-1/2) @ H @ S^(-1/2), then H → (H+H^T)/2
        
        Returns
        -------
        H_total : np.ndarray
            Total Hamiltonian matrix for this l channel
        
        Notes
        -----
        Symmetrization uses the overlap matrix S to transform the generalized
        eigenvalue problem (H·ψ = ε·S·ψ) into a standard eigenvalue problem.
        For meta-GGA functionals, de_xc_dtau is used to add additional terms to the Hamiltonian.
        """
        # Determine whether to include HF exchange based on switches
        include_hf_exchange = switches.use_hf_exchange
        if include_hf_exchange and len(self.H_hf_exchange_dict) == 0:
            print(HF_EXCHANGE_MATRIX_NOT_AVAILABLE_WARNING)
            include_hf_exchange = False
        
        # Determine whether to mix the exchange functionals of HF and GGA_PBE
        hybrid_mixing_parameter = 0.0
        if switches.use_hf_exchange:
            assert isinstance(switches.hybrid_mixing_parameter, float), \
                MIXING_PARAMETER_NOT_A_FLOAT_ERROR.format(type(switches.hybrid_mixing_parameter))
            assert 0.0 <= switches.hybrid_mixing_parameter <= 1.0, \
                MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR.format(switches.hybrid_mixing_parameter)
            hybrid_mixing_parameter = switches.hybrid_mixing_parameter
        
        # Determine whether to include de_xc_dtau based on switches
        if switches.use_metagga:
            assert de_xc_dtau is not None, \
                DE_XC_DTAU_NOT_AVAILABLE_ERROR.format(l)
        
        if switches.use_oep_exchange:
            v_x_oep = v_x_oep if v_x_oep is not None else np.zeros_like(v_x)
        
        if switches.use_oep_correlation:
            v_c_oep = v_c_oep if v_c_oep is not None else np.zeros_like(v_c)
        
        # Start with fixed terms
        H = self.H_kinetic + self.H_ext
        
        # Add Hartree potential
        H_hartree = self.ops_builder.build_potential_matrix(v_hartree)
        H += H_hartree
        
        # Add XC potential (may include OEP contributions)
        v_xc_total = (1.0 - hybrid_mixing_parameter) * v_x
        v_xc_total += v_c
        if v_x_oep is not None:
            v_xc_total += v_x_oep
        if v_c_oep is not None:
            v_xc_total += v_c_oep
        
        H_xc = self.ops_builder.build_potential_matrix(v_xc_total)
        H += H_xc


        # np.savetxt("H_part.txt", self.H_kinetic + self.H_ext)
        # np.savetxt("H_hartree.txt", H_hartree)
        # np.savetxt("H_xc.txt", H_xc)
        # np.savetxt("v_hartree.txt", v_hartree)
        # np.savetxt("v_x.txt", v_x)
        # np.savetxt("v_c.txt", v_c)
        # raise RuntimeError("Stop here")

        # Add meta-GGA kinetic density term (radial component)
        if switches.use_metagga:
            H_metagga_tau = self.ops_builder.build_metagga_kinetic_density_matrix(de_xc_dtau)
            H += H_metagga_tau

        # Add angular momentum term: l(l+1)/(2r²)
        if l > 0:
            angular_term = self.H_r_inv_sq
            if switches.use_metagga:
                # Meta-GGA angular term: ∫ φ_i * (w * de_xc_dtau / r²) * φ_j dr
                potential_angular = de_xc_dtau / self.ops_builder.quadrature_nodes**2
                angular_term += self.ops_builder.build_potential_matrix(potential_angular)

            H += (l * (l + 1) / 2) * angular_term
        
        # Add non-local pseudopotential (if present for this l)
        if l in self.H_nonlocal:
            H += self.H_nonlocal[l]
        
        # Add Hartree-Fock exchange (if requested and available)
        if include_hf_exchange:
            assert l in self.H_hf_exchange_dict, \
                HARTREE_FOCK_EXCHANGE_MATRIX_FOR_L_CHANNEL_NOT_AVAILABLE_ERROR.format(l)
            H += self.H_hf_exchange_dict[l] * hybrid_mixing_parameter
        
        # Apply symmetrization transformation (if requested)
        if symmetrize:
            # Get S^(-1/2) from ops_builder
            S_inv_sqrt = self.ops_builder.get_S_inv_sqrt()

            # Transform: H → S^(-1/2) @ H @ S^(-1/2)
            H = S_inv_sqrt @ H @ S_inv_sqrt
            
            # Enforce symmetry: H → (H + H^T) / 2
            H = 0.5 * (H + H.T)
        
        return H
    
    
    def set_hf_exchange_matrices(self, H_hf_by_l: Dict[int, np.ndarray]):
        """
        Set Hartree-Fock exchange matrices for each l channel
        
        Called by outer SCF loop after computing HF exchange from orbitals
        """
        self.H_hf_exchange_dict = H_hf_by_l


    def interpolate_eigenvectors_to_quadrature(
        self, 
        eigenvectors: np.ndarray,
        symmetrize: bool = False,
        pad_width: int = 0,
        ) -> np.ndarray:
        """
        Interpolate eigenvectors to quadrature points using the global interpolation matrix
        
        Parameters
        ----------
        eigenvectors : np.ndarray
            Eigenvectors at physical grid points (shape: (n_physical_nodes, n_states))
        symmetrize : bool, optional
            Whether to apply symmetrization using S^(-1/2) (default: False)
            Transforms: H → S^(-1/2) @ H @ S^(-1/2), then H → (H+H^T)/2
        pad_width : int, optional
            Number of points to pad on each side of the eigenvectors (default: 1)
        """

        assert isinstance(eigenvectors, np.ndarray), \
            EIGENVECTORS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(eigenvectors))
        assert eigenvectors.ndim == 2, \
            EIGENVECTORS_MUST_BE_A_2D_ARRAY_ERROR.format(eigenvectors.ndim)
        
        if symmetrize:
            S_inv_sqrt = self.ops_builder.get_S_inv_sqrt()
            if pad_width > 0:
                S_inv_sqrt = S_inv_sqrt[pad_width:-pad_width,pad_width:-pad_width]
            eigenvectors = S_inv_sqrt @ eigenvectors

        # print("eigenvectors.shape:", eigenvectors.shape)
        # np.savetxt("eigenvectors.txt", eigenvectors)
        # raise RuntimeError("Stop here")

        if pad_width > 0:
            eigenvectors = np.pad(eigenvectors,((pad_width,pad_width),(0,0)))

        global_interpolation_matrix = self.ops_builder.get_global_interpolation_matrix()
        eigenvectors_quadrature = global_interpolation_matrix @ eigenvectors

        return eigenvectors_quadrature
        


        


