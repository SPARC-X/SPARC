from __future__ import annotations


import numpy as np
from typing import Any, Tuple, List, Dict

from .hybrid import CoulombCouplingCalculator
from ..mesh.builder import Quadrature1D
from ..mesh.operators import RadialOperatorsBuilder
from ..utils.occupation_states import OccupationInfo

from contextlib import nullcontext

try:
    # Optional dependency: used to limit BLAS/OpenMP threads during parallel sections
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore

# Error messages
OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter `ops_builder` must be a `RadialOperatorsBuilder` instance, get type `{}` instead"
OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR = \
    "Parameter `occupation_info` must be a `OccupationInfo` instance, get type `{}` instead"
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "Parameter `frequency_quadrature_point_number` must be an integer, get type {} instead"
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "Parameter `frequency_quadrature_point_number` must be greater than 0, get {} instead"
ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR = \
    "Parameter `angular_momentum_cutoff` must be an integer, get type {} instead"
ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR = \
    "Parameter `angular_momentum_cutoff` must be non-negative, get {} instead"
FREQUENCY_NOT_FLOAT_ERROR = \
    "Parameter `frequency` must be a float or a scaler numpy array, get type {} instead"

ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR_MESSAGE = \
    "Parameter `angular_momentum_cutoff` must be not None, get None instead"
OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR = \
    "Occupied l terms are not consistent with the occupation information, please check your inputs, got {} instead of {}"
PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR = \
    "Parent class `RPACorrelation` is not initialized, please initialize it first"

L_OCC_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_occ_max` must be an integer, get type {} instead"
L_UNOCC_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_unocc_max` must be an integer, get type {} instead"
L_COUPLE_MAX_NOT_INTEGER_ERROR = \
    "Parameter `l_couple_max` must be an integer, get type {} instead"
ENABLE_PARALLELIZATION_NOT_BOOL_ERROR = \
    "Parameter `enable_parallelization` must be a bool, get type {} instead"



class RPACorrelation:
    """
    Compute RPA correlation energy from eigenstates.
    """
    def __init__(
        self, 
        ops_builder                       : 'RadialOperatorsBuilder',
        occupation_info                   : 'OccupationInfo',
        frequency_quadrature_point_number : int,
        angular_momentum_cutoff           : int
        ):
        """
        Parameters
        ----------
        ops_builder                       : instance of RadialOperatorsBuilder
            RadialOperatorsBuilder instance
        occupation_info                   : instance of OccupationInfo
            Occupation information
        frequency_quadrature_point_number : int
            Number of frequency quadrature points
        angular_momentum_cutoff           : int
            Maximum angular momentum quantum number to include
        """
        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        assert isinstance(frequency_quadrature_point_number, int), \
            FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(frequency_quadrature_point_number))
        assert frequency_quadrature_point_number > 0, \
            FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(frequency_quadrature_point_number)
        assert isinstance(angular_momentum_cutoff, int), \
            ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR.format(type(angular_momentum_cutoff))
        assert angular_momentum_cutoff >= 0, \
            ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR.format(angular_momentum_cutoff)


        # Extract quadrature data from ops_builder
        self.n_quad             = len(ops_builder.quadrature_nodes)
        self.quadrature_nodes   = ops_builder.quadrature_nodes
        self.quadrature_weights = ops_builder.quadrature_weights

        # initialize the frequency grid and weights
        self.frequency_quadrature_point_number = frequency_quadrature_point_number
        self.frequency_grid, self.frequency_weights = \
            self._initialize_frequency_grid_and_weights(frequency_quadrature_point_number)

        # occupation information
        self.occupations  : np.ndarray = occupation_info.occupations
        self.occ_l_values : np.ndarray = occupation_info.l_values
        self.occ_n_values : np.ndarray = occupation_info.n_values

        # angular momentum cutoff
        self.angular_momentum_cutoff = angular_momentum_cutoff


    @staticmethod
    def _initialize_frequency_grid_and_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize frequency grid and weights for RPA correlation energy calculations.

        Parameters
        ----------
        n : int
            Number of frequency quadrature points

        Reference:
            https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.134.016402/scrpa4_SM.pdf

        Returns
        -------
        frequency_grid : np.ndarray
            Frequency grid
        frequency_weights : np.ndarray
            Frequency weights
        """
        frequency_scale = 2.5

        # Get Gauss-Legendre nodes on reference interval [-1, 1]
        reference_nodes, reference_weights = Quadrature1D.gauss_legendre(n)
        
        # Transform to semi-infinite interval [0, ∞)
        nodes   = frequency_scale * (1 + reference_nodes) / (1 - reference_nodes)
        weights = reference_weights * 2 * frequency_scale / (1 - reference_nodes)**2

        return nodes, weights


    @staticmethod
    def _compute_rpa_correlation_driving_term_for_single_frequency(
        frequency               : float,
        angular_momentum_cutoff : int,
        occupation_info         : OccupationInfo,
        full_eigen_energies     : np.ndarray, 
        full_orbitals           : np.ndarray, 
        full_l_terms            : np.ndarray,
        wigner_symbols_squared  : np.ndarray,
        radial_kernels_dict     : Dict[int, np.ndarray],
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RPA correlation driving term for at given frequency.
        """
        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        
        # get occupation information
        occupations  = occupation_info.occupations
        occ_l_values = occupation_info.l_values


        # get the number of occupied and unoccupied orbitals
        occ_orbitals_num   = len(occ_l_values)
        total_orbitals_num = len(full_eigen_energies)

        # get the number of quadrature and n_interior points
        n_quad     = radial_kernels_dict[0].shape[0]
        n_interior = len(np.argwhere(full_l_terms == 0)[:, 0])

        # get occupied and unoccupied orbitals and energies
        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]     # shape: (n_grid, total_orbitals_num)
        occ_energies   = full_eigen_energies[:occ_orbitals_num]  # shape: (total_orbitals_num,)
        occ_l_terms    = full_l_terms[:occ_orbitals_num]         # shape: (total_orbitals_num,)
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]     # shape: (n_grid, unocc_orbitals_num)
        unocc_energies = full_eigen_energies[occ_orbitals_num:]  # shape: (unocc_orbitals_num,)
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]         # shape: (total_orbitals_num,)

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)

        ### ================================================ ###
        ###  Part 1: Compute the RPA correlation prefactors  ###
        ### ================================================ ###

        # Angular degeneracy factors f_p * (2l_q + 1)
        #   shape: (occ_num, unocc_num)
        deg_factors = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]

        # Energy differences Δε_{pq} = ε_p - ε_q
        #   shape: (occ_num, unocc_num)
        delta_eps = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]

        # Filter out zero entries (valid (p,q) pairs)
        occ_idx, unocc_idx = np.argwhere((deg_factors != 0) & (delta_eps != 0)).T
        deg_factors_valid = deg_factors[occ_idx, unocc_idx]
        delta_eps_valid   = delta_eps[occ_idx, unocc_idx]
        
        # Compute Lorentzian frequency response: Δε / (Δε² + ω²)
        #   shape: (n_valid_pairs)
        lorentzian_factors = delta_eps_valid / (delta_eps_valid ** 2 + frequency ** 2)

        # Compute frequency derivative factors for Q2c: (Δε² - ω²) / (Δε² + ω²)²
        #   This arises from ∂χ₀,L/∂(iω), used in the Q2c driving term
        #   shape: (n_valid_pairs)
        frequency_derivative_factors = (delta_eps_valid ** 2 - frequency ** 2) / (delta_eps_valid ** 2 + frequency ** 2) ** 2

        # Combine angular and frequency factors (without Wigner 3j symbol yet)
        #   shape: (n_valid_pairs)
        prefactors_q1c = deg_factors_valid * lorentzian_factors
        prefactors_q2c = deg_factors_valid * frequency_derivative_factors

        #   shape: (occ_num, unocc_num)
        prefactors_self_energy = deg_factors * delta_eps / (delta_eps ** 2 + frequency ** 2)


        ### ================================================== ###
        ###  Part 2: Compute the nonzero Wigner symbols        ###
        ### ================================================== ###

        l_couple_min = np.min(np.abs(occ_l_terms[occ_idx] - unocc_l_terms[unocc_idx])).astype(np.int32)
        l_couple_max = np.max(occ_l_terms[occ_idx] + unocc_l_terms[unocc_idx]).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)


        # Use advanced indexing with broadcasting - one operation instead of triple loop
        # occ_l_terms[i] gives angular momentum of orbital i
        # unocc_l_terms[j] gives angular momentum of orbital j

        wigner_symbols_squared_all = wigner_symbols_squared[
            occ_l_terms   [:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            unocc_l_terms [np.newaxis, :, np.newaxis],  # shape: (1, unocc_orbitals_num, 1)
            l_couple_range[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        wigner_symbols_squared_valid = wigner_symbols_squared_all[occ_idx, unocc_idx, :]

        # Compute only the nonzero Wigner symbols for each l_couple channel
        active_l_couple_idx_list           : List[int]        = []
        active_wigner_symbols_indices_list : List[np.ndarray] = []

        for l_couple_idx, l_couple in enumerate(l_couple_range):
            non_zero_wigner_symbols_indices = np.argwhere(wigner_symbols_squared_valid[:, l_couple_idx] != 0)[:, 0]

            # Collect active l_couple and corresponding nonzero Wigner symbols indices
            if len(non_zero_wigner_symbols_indices) != 0:
                active_l_couple_idx_list.append(l_couple_idx)
                active_wigner_symbols_indices_list.append(non_zero_wigner_symbols_indices)


        ### ================================================== ###
        ###  Part 3: Compute orbital products                  ###
        ### ================================================== ###

        # orbital outer product: φ_p(r) ⊗ φ_q(r) for all (p,q) pairs
        #   shape: (n_grid, occ_num, unocc_num)
        orbital_product_outer = np.einsum('li,lj->ijl',
            occ_orbitals,
            unocc_orbitals,
            optimize = True,
        )

        # Orbital squared difference: φ_p²(r) - φ_q²(r) for valid (p,q) pairs
        #   shape: (n_grid, n_valid_pairs)
        orbital_squared_diff = occ_orbitals[:, occ_idx] ** 2 - unocc_orbitals[:, unocc_idx] ** 2
        
        # Orbital pair product: Φ_{pq}(r) = φ_p(r)φ_q(r) for valid (p,q) pairs
        #   shape: (n_grid, n_valid_pairs)
        orbital_pair_product = occ_orbitals[:, occ_idx] * unocc_orbitals[:, unocc_idx]


        # initialize the full self-energy potential 
        #   shape: (total_orbitals_num, n_quad)
        full_self_energy_potential = np.zeros((total_orbitals_num, n_quad))


        ### ================================================== ###
        ###  Part 4: Compute RPA correlation driving term      ###
        ### ================================================== ###

        # initialize the q1c and q2c terms
        #   shape: (n_quad,)
        full_q1c_term = np.zeros(n_quad)
        full_q2c_term = np.zeros(n_quad)


        # Compute RPA correlation driving term for each l_couple channel
        for active_l_couple_idx, active_wigner_symbols_indices in zip(active_l_couple_idx_list, active_wigner_symbols_indices_list):


            active_l_couple = l_couple_range[active_l_couple_idx]

            # Get radial kernel
            #   shape: (n_quad, n_quad)
            radial_kernel = radial_kernels_dict[active_l_couple]

            # Compute rpa_response_kernel
            #   shape: (n_quad, n_quad)
            #   Reference: https://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
            rpa_response_kernel = np.matmul(
                radial_kernel, 
                2 * np.einsum(
                    'ij,ik->kj',
                    np.einsum('ji,i->ij',
                        orbital_pair_product[:, active_wigner_symbols_indices],
                        prefactors_q1c[active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    np.einsum('i,ki->ik',
                        wigner_symbols_squared_valid[active_wigner_symbols_indices, active_l_couple],
                        orbital_pair_product[:, active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    optimize=True,
                ),
            )

            # Compute dyson solved response
            dyson_solved_response = np.linalg.solve(np.eye(n_quad) - rpa_response_kernel, radial_kernel) - radial_kernel

            # Compute self-energy potential term
            #   shape: (total_orbitals_num, n_quad)
            _self_energy_potential = np.zeros((total_orbitals_num, n_quad))

            # Compute self-energy potential term for occupied states
            for occ_l_index in range(occ_orbitals_num):
                # Get nonzero unoccupied l indices for this occupied l
                nonzero_unocc_indices = np.argwhere(wigner_symbols_squared_all[occ_l_index, :, active_l_couple] != 0)[:, 0]
                _prefactor = prefactors_self_energy[occ_l_index, nonzero_unocc_indices] * wigner_symbols_squared_all[occ_l_index, nonzero_unocc_indices, active_l_couple]
                
                _self_energy_potential[occ_l_index, :] = \
                    np.einsum('i,ki,il,lk->k',
                        _prefactor,
                        unocc_orbitals[:, nonzero_unocc_indices],
                        orbital_product_outer[occ_l_index, nonzero_unocc_indices, :],
                        dyson_solved_response,
                        optimize=True,
                    ) * (2 * active_l_couple + 1)


            # Compute self-energy potential term for unoccupied states
            self_energy_coupling_matrix = prefactors_self_energy * wigner_symbols_squared_all[:, :, active_l_couple]
            coupled_unocc_indices = np.argwhere(~np.all(self_energy_coupling_matrix == 0, axis=0))[:, 0]
            
            _self_energy_potential[occ_orbitals_num:, :][coupled_unocc_indices, :] = \
                np.einsum('ji,kj,jik->ik',
                    self_energy_coupling_matrix[:, coupled_unocc_indices],
                    occ_orbitals,
                    np.einsum('jil,lk->jik',
                        orbital_product_outer[:, coupled_unocc_indices, :],
                        dyson_solved_response,
                        optimize=True, 
                    ),
                    optimize=True,
                ) * (2 * active_l_couple + 1)

            # Compute q2c term
            _q2c_term = np.einsum('ki, i, i->k',
                orbital_squared_diff[:, active_wigner_symbols_indices],
                prefactors_q2c[active_wigner_symbols_indices] * wigner_symbols_squared_valid[active_wigner_symbols_indices, active_l_couple],
                np.einsum('li,pi,pl->i',
                    orbital_pair_product[:, active_wigner_symbols_indices],
                    orbital_pair_product[:, active_wigner_symbols_indices],
                    dyson_solved_response,
                    optimize=True,
                ),
                optimize=True,
            ) * (2 * active_l_couple + 1)

            # Update the full q2c term and self-energy potential
            full_q2c_term += _q2c_term
            full_self_energy_potential += _self_energy_potential
        
        

        # Compute q1c term
        for l_value in range(angular_momentum_cutoff + 1):

            # Get total orbitals in this l_value channel
            l_indices = np.argwhere(full_l_terms == l_value)[:, 0]
            total_orbitals_in_l_channel = full_orbitals[:, l_indices]
            self_energy_in_l_channel    = full_self_energy_potential[l_indices, :]
            eigenvalues_in_l_channel    = full_eigen_energies[l_indices]
            
            # Compute difference of eigenvalues in this l_value channel
            diff_eigenvalues = eigenvalues_in_l_channel.reshape(-1, 1) - eigenvalues_in_l_channel.reshape(1, -1)
            one_over_diff_eigenvalues = 1 / (diff_eigenvalues + np.eye(n_interior))      # avoid division by zero
            one_over_diff_eigenvalues[np.arange(n_interior), np.arange(n_interior)] = 0  # set the diagonal to zero


            # Compute q1c term for this l_value channel
            q1c_term_in_l_channel = \
                np.einsum('ki,ik->k',
                    total_orbitals_in_l_channel,
                    np.einsum('ij,kj,ij->ik',
                        one_over_diff_eigenvalues,
                        total_orbitals_in_l_channel,
                        np.einsum('ix,xj->ij',
                            self_energy_in_l_channel,
                            total_orbitals_in_l_channel,
                            optimize=True,
                        ),
                        optimize=True,
                    ),
                    optimize=True,
                )

            full_q1c_term -= q1c_term_in_l_channel

        assert full_q1c_term.shape == (n_quad,)
        assert full_q2c_term.shape == (n_quad,)


        return full_q1c_term, full_q2c_term



    @staticmethod
    def _compute_rpa_wigner_symbols_squared(
        l_occ_max    : int,
        l_unocc_max  : int,
        ) -> np.ndarray:
        """
        Compute RPA Wigner symbols squared array.

        Parameters
        ----------
        l_occ_max : int
            Maximum angular momentum quantum number for occupied orbitals
        l_unocc_max : int
            Maximum angular momentum quantum number for unoccupied orbitals

        Returns
        -------
        wigner_symbols_squared : np.ndarray
            Wigner symbols squared array
            shape: (l_occ_max + 1, l_unocc_max + 1, l_couple_max + 1), where l_couple_max = l_occ_max + l_unocc_max
        """
        try:
            l_occ_max = int(l_occ_max)
        except ValueError:
            raise ValueError(L_OCC_MAX_NOT_INTEGER_ERROR.format(type(l_occ_max)))
        try:
            l_unocc_max = int(l_unocc_max)
        except ValueError:
            raise ValueError(L_UNOCC_MAX_NOT_INTEGER_ERROR.format(type(l_unocc_max)))
        assert l_occ_max >= 0 and l_unocc_max >= 0, \
            "All angular momentum quantum numbers must be non-negative"

        # Compute the maximum angular momentum quantum number for the coupled system
        l_couple_max = l_occ_max + l_unocc_max

        # Initialize Wigner symbols squared array
        wigner_symbols_squared = np.zeros((l_occ_max + 1, l_unocc_max + 1, l_couple_max + 1))

        # Compute Wigner symbols squared for all (l_occ, l_unocc, l_couple) combinations
        for l_occ in range(l_occ_max + 1):
            for l_unocc in range(l_unocc_max + 1):
                for l_couple in range(l_couple_max + 1):
                    wigner_symbols_squared[l_occ, l_unocc, l_couple] = \
                        CoulombCouplingCalculator.wigner_3j_000(l_occ, l_unocc, l_couple) ** 2

        return wigner_symbols_squared

        
    @staticmethod
    def _compute_correlation_energy_for_single_frequency(
        frequency               : float,
        occupation_info         : OccupationInfo,
        full_eigen_energies     : np.ndarray, 
        full_orbitals           : np.ndarray, 
        full_l_terms            : np.ndarray,
        wigner_symbols_squared  : np.ndarray,
        radial_kernels_dict     : Dict[int, np.ndarray],
        ) -> float:
        """
        Compute RPA correlation driving term for at given frequency.
        """
        try:
            frequency = float(frequency)
        except ValueError:
            raise ValueError(FREQUENCY_NOT_FLOAT_ERROR.format(type(frequency)))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        
        # get occupation information
        occupations  = occupation_info.occupations
        occ_l_values = occupation_info.l_values


        # get the number of occupied and unoccupied orbitals
        occ_orbitals_num   = len(occ_l_values)

        # get the number of quadrature and n_interior points
        n_quad     = radial_kernels_dict[0].shape[0]

        # get occupied and unoccupied orbitals and energies
        occ_orbitals   = full_orbitals[:, :occ_orbitals_num]     # shape: (n_grid, total_orbitals_num)
        occ_energies   = full_eigen_energies[:occ_orbitals_num]  # shape: (total_orbitals_num,)
        occ_l_terms    = full_l_terms[:occ_orbitals_num]         # shape: (total_orbitals_num,)
        unocc_orbitals = full_orbitals[:, occ_orbitals_num:]     # shape: (n_grid, unocc_orbitals_num)
        unocc_energies = full_eigen_energies[occ_orbitals_num:]  # shape: (unocc_orbitals_num,)
        unocc_l_terms  = full_l_terms[occ_orbitals_num:]         # shape: (total_orbitals_num,)

        assert np.all(occ_l_terms == occ_l_values), \
            OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR.format(occ_l_terms, occ_l_values)


        ### ================================================ ###
        ###  Part 1: Compute the RPA correlation prefactors  ###
        ### ================================================ ###

        # Angular degeneracy factors f_p * (2l_q + 1)
        #   shape: (occ_num, unocc_num)
        deg_factors = occupations[:, np.newaxis] * (2 * unocc_l_terms + 1)[np.newaxis, :]

        # Energy differences Δε_{pq} = ε_p - ε_q
        #   shape: (occ_num, unocc_num)
        delta_eps = occ_energies[:, np.newaxis] - unocc_energies[np.newaxis, :]

        # Filter out zero entries (valid (p,q) pairs)
        occ_idx, unocc_idx = np.argwhere((deg_factors != 0) & (delta_eps != 0)).T
        deg_factors_valid = deg_factors[occ_idx, unocc_idx]
        delta_eps_valid   = delta_eps[occ_idx, unocc_idx]
        
        # Compute Lorentzian frequency response: Δε / (Δε² + ω²)
        #   shape: (n_valid_pairs)
        lorentzian_factors = delta_eps_valid / (delta_eps_valid ** 2 + frequency ** 2)

        # Combine angular and frequency factors (without Wigner 3j symbol yet)
        #   shape: (n_valid_pairs)
        prefactors_q1c = deg_factors_valid * lorentzian_factors

        ### ================================================== ###
        ###  Part 2: Compute the nonzero Wigner symbols        ###
        ### ================================================== ###

        l_couple_min = np.min(np.abs(occ_l_terms[occ_idx] - unocc_l_terms[unocc_idx])).astype(np.int32)
        l_couple_max = np.max(occ_l_terms[occ_idx] + unocc_l_terms[unocc_idx]).astype(np.int32)
        l_couple_range = np.arange(l_couple_min, l_couple_max + 1)


        # Use advanced indexing with broadcasting - one operation instead of triple loop
        # occ_l_terms[i] gives angular momentum of orbital i
        # unocc_l_terms[j] gives angular momentum of orbital j

        wigner_symbols_squared_all = wigner_symbols_squared[
            occ_l_terms   [:, np.newaxis, np.newaxis],  # shape: (occ_orbitals_num, 1, 1)
            unocc_l_terms [np.newaxis, :, np.newaxis],  # shape: (1, unocc_orbitals_num, 1)
            l_couple_range[np.newaxis, np.newaxis, :],  # shape: (1, 1, l_couple_num)
        ]
        wigner_symbols_squared_valid = wigner_symbols_squared_all[occ_idx, unocc_idx, :]

        # Compute only the nonzero Wigner symbols for each l_couple channel
        active_l_couple_idx_list           : List[int]        = []
        active_wigner_symbols_indices_list : List[np.ndarray] = []

        for l_couple_idx, l_couple in enumerate(l_couple_range):
            non_zero_wigner_symbols_indices = np.argwhere(wigner_symbols_squared_valid[:, l_couple_idx] != 0)[:, 0]

            # Collect active l_couple and corresponding nonzero Wigner symbols indices
            if len(non_zero_wigner_symbols_indices) != 0:
                active_l_couple_idx_list.append(l_couple_idx)
                active_wigner_symbols_indices_list.append(non_zero_wigner_symbols_indices)


        ### ================================================== ###
        ###  Part 3: Compute the RPA correlation energy        ###
        ### ================================================== ###


        # Orbital pair product: Φ_{pq}(r) = φ_p(r)φ_q(r) for valid (p,q) pairs
        #   shape: (n_grid, n_valid_pairs)
        orbital_pair_product = occ_orbitals[:, occ_idx] * unocc_orbitals[:, unocc_idx]

        # initialize the q1c and q2c terms
        #   shape: (n_quad,)
        full_correlation_energy_at_single_frequency = 0.0

        # Compute RPA correlation driving term for each l_couple channel
        for active_l_couple_idx, active_wigner_symbols_indices in zip(active_l_couple_idx_list, active_wigner_symbols_indices_list):

            active_l_couple = l_couple_range[active_l_couple_idx]

            # Get radial kernel
            #   shape: (n_quad, n_quad)
            radial_kernel = radial_kernels_dict[active_l_couple]

            # Compute rpa_response_kernel
            #   shape: (n_quad, n_quad)
            #   Reference: https://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
            rpa_response_kernel = np.matmul(
                radial_kernel, 
                2 * np.einsum(
                    'ij,ik->kj',
                    np.einsum('ji,i->ij',
                        orbital_pair_product[:, active_wigner_symbols_indices],
                        prefactors_q1c[active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    np.einsum('i,ki->ik',
                        wigner_symbols_squared_valid[active_wigner_symbols_indices, active_l_couple],
                        orbital_pair_product[:, active_wigner_symbols_indices],
                        optimize=True,
                    ),
                    optimize=True,
                ),
            )

            # Compute correlation energy for this l_couple channel
            full_correlation_energy_at_single_frequency += \
                (2 * active_l_couple + 1) * (np.log(np.linalg.det(np.eye(n_quad) - rpa_response_kernel)) + np.trace(rpa_response_kernel))

        return full_correlation_energy_at_single_frequency




    def compute_correlation_energy(
        self, 
        full_eigen_energies : np.ndarray, 
        full_orbitals       : np.ndarray, 
        full_l_terms        : np.ndarray,
        enable_parallelization: bool = False,
        ) -> float:
        """
        Compute RPA correlation energy from eigenstates.
        """
        assert hasattr(self, 'frequency_grid') and hasattr(self, 'frequency_weights'), \
            PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR

        assert isinstance(enable_parallelization, bool), \
            ENABLE_PARALLELIZATION_NOT_BOOL_ERROR.format(type(enable_parallelization))



        self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        l_occ_max    = np.max(self.occ_l_values)
        l_unocc_max  = np.max(full_l_terms)
        l_couple_max = l_occ_max + l_unocc_max

        # Compute RPA Wigner symbols squared array
        wigner_symbols_squared = self._compute_rpa_wigner_symbols_squared(
            l_occ_max    = np.max(self.occ_l_values),
            l_unocc_max  = np.max(full_l_terms),
        )

        # Compute RPA radial kernels dictionary
        radial_kernels_dict = {}
        for l_couple in range(l_couple_max + 1):
            radial_kernels_dict[l_couple] = CoulombCouplingCalculator.radial_kernel(
                l         = l_couple,
                r_nodes   = self.quadrature_nodes,
                r_weights = self.quadrature_weights,
            )

        # Compute RPA correlation energy at each frequency and sum them up
        correlation_energy = 0.0

        if not enable_parallelization:
            for frequency, frequency_weight in zip(self.frequency_grid, self.frequency_weights):
                correlation_energy_at_single_frequency = self._compute_correlation_energy_for_single_frequency(
                    frequency               = frequency,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,
                )
                
                correlation_energy += correlation_energy_at_single_frequency * frequency_weight
        else:
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor

            def _single_frequency_task(args):
                idx, (frequency, frequency_weight) = args
                correlation_energy_at_single_frequency = self._compute_correlation_energy_for_single_frequency(
                    frequency               = frequency,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,   
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,   
                )
                return (
                    idx,
                    correlation_energy_at_single_frequency * frequency_weight,
                )

            n_workers = min(max(1, mp.cpu_count()), len(self.frequency_grid))

            # limit BLAS/OpenMP threads during parallel sections
            if threadpool_limits is not None:
                blas_ctx = threadpool_limits(limits=1)
            else:
                blas_ctx = nullcontext()


            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(
                    _single_frequency_task,
                    enumerate(zip(self.frequency_grid, self.frequency_weights))
                )
                for _, correlation_energy_single_weighted in results:
                    correlation_energy += correlation_energy_single_weighted


        return correlation_energy / (2 * np.pi)



