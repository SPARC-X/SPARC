from __future__ import annotations

import scipy
from scipy.linalg import LinAlgWarning
import numpy as np

import warnings
from typing import Tuple, List, Optional

from .hybrid import CoulombCouplingCalculator, HartreeFockExchange
from .rpa import RPACorrelation
from ..utils.occupation_states import OccupationInfo
from ..mesh.operators import RadialOperatorsBuilder


from contextlib import nullcontext

try:
    # Optional dependency: used to limit BLAS/OpenMP threads during parallel sections
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # type: ignore



# Error messages
USE_RPA_CORRELATION_NOT_BOOL_ERROR = \
    "Parameter use_rpa_correlation must be a bool, get type {} instead"
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_ERROR = \
    "Parameter frequency_quadrature_point_number must be not None, get None instead"
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "Parameter frequency_quadrature_point_number must be an integer, get type {} instead"
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "Parameter frequency_quadrature_point_number must be greater than 0, get {} instead"
ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR = \
    "Parameter angular_momentum_cutoff must be not None, get None instead"
ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR = \
    "Parameter angular_momentum_cutoff must be an integer, get type {} instead"
ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR = \
    "Parameter angular_momentum_cutoff must be non-negative, get {} instead"
ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR_MESSAGE = \
    "Parameter angular_momentum_cutoff must be not None, get None instead"
OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter ops_builder must be a RadialOperatorsBuilder instance, get type {} instead"
OPS_BUILDER_OEP_NOT_RADIAL_OPERATORS_BUILDER_ERROR = \
    "Parameter ops_builder_oep must be a RadialOperatorsBuilder instance, get type {} instead"
OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_NODES_ERROR = \
    "Parameter ops_builder.quadrature_nodes must be equal to ops_builder_oep.quadrature_nodes, please check the grid data and the operators builder"
OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_WEIGHTS_ERROR = \
    "Parameter ops_builder.quadrature_weights must be equal to ops_builder_oep.quadrature_weights, please check the grid data and the operators builder"
OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR = \
    "Parameter occupation_info must be a OccupationInfo instance, get type {} instead"
OCCUPATION_INFO_L_TERMS_NOT_CONSISTENT_WITH_OCCUPATION_INFO_ERROR = \
    "Occupied l terms are not consistent with the occupation information, please check your inputs, got {} instead of {}"
FULL_EIGEN_ENERGIES_NOT_NUMPY_ARRAY_ERROR = \
    "Parameter full_eigen_energies must be a numpy array, get type {} instead"
FULL_ORBITALS_NOT_NUMPY_ARRAY_ERROR = \
    "Parameter full_orbitals must be a numpy array, get type {} instead"
FULL_L_TERMS_NOT_NUMPY_ARRAY_ERROR = \
    "Parameter full_l_terms must be a numpy array, get type {} instead"
FULL_EIGEN_ENERGIES_NOT_1D_ARRAY_ERROR = \
    "Parameter full_eigen_energies must be a 1D array, got ndim={}"
FULL_ORBITALS_NOT_2D_ARRAY_ERROR = \
    "Parameter full_orbitals must be a 2D array, got ndim={}"
FULL_L_TERMS_NOT_1D_ARRAY_ERROR = \
    "Parameter full_l_terms must be a 1D array, got ndim={}"
FULL_EIGEN_ENERGIES_AND_ORBITALS_SHAPE_ERROR = \
    "Parameter full_eigen_energies.shape[0] must equal full_orbitals.shape[1], got {} and {} instead"
FULL_EIGEN_ENERGIES_AND_L_TERMS_SHAPE_ERROR = \
    "Parameter full_eigen_energies.shape[0] must equal full_l_terms.shape[0], got {} and {} instead"
FULL_ORBITALS_AND_L_TERMS_SHAPE_ERROR = \
    "Parameter full_orbitals.shape[1] must equal full_l_terms.shape[0], got {} and {} instead"
FULL_L_TERMS_NON_NEGATIVE_ERROR = \
    "Parameter full_l_terms must be non-negative, got {} instead of all non-negative values"
INVALID_EX_TAG_ERROR = \
    "Parameter `ex_tag` must be 'exchange' or 'correlation', got {} instead"
PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR = \
    "Parent class RPACorrelation is not initialized, check your initialization at OEPCalculator class"
ENABLE_PARALLELIZATION_NOT_BOOL_ERROR = \
    "Parameter `parallelize` must be a bool, get type {} instead"

# WARNING Messages
FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_WHEN_RPA_CORRELATION_IS_NOT_USED_WARNING = \
    "WARNING: parameter `frequency_quadrature_point_number` is not None when RPA correlation is not used, so it will be ignored"
ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_WHEN_RPA_CORRELATION_IS_NOT_USED_WARNING = \
    "WARNING: parameter `angular_momentum_cutoff` is not None when RPA correlation is not used, so it will be ignored"


class OEPCalculator(HartreeFockExchange, RPACorrelation):

    """Prepare and build OEP exchange/correlation potentials from eigenstates."""

    def __init__(
        self,
        ops_builder         : RadialOperatorsBuilder,
        ops_builder_oep     : RadialOperatorsBuilder,
        occupation_info     : OccupationInfo,
        use_rpa_correlation : bool,
        frequency_quadrature_point_number : Optional[int] = None,  # parameters for RPA correlation potential
        angular_momentum_cutoff           : Optional[int] = None,  # parameters for RPA functional only
        ):

        """
        Parameters
        ----------
        ops_builder : RadialOperatorsBuilder
            Operators builder for the standard grid
        ops_builder_oep : RadialOperatorsBuilder
            Operators builder for the OEP grid
        occupation_info : OccupationInfo
            Occupation information
        use_rpa_correlation : bool
            Whether to use RPA correlation potential
            If True, use RPA correlation potential, otherwise return zero correlation potential
        """
        assert isinstance(ops_builder, RadialOperatorsBuilder), \
            OPS_BUILDER_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder))
        assert isinstance(ops_builder_oep, RadialOperatorsBuilder), \
            OPS_BUILDER_OEP_NOT_RADIAL_OPERATORS_BUILDER_ERROR.format(type(ops_builder_oep))
        assert isinstance(occupation_info, OccupationInfo), \
            OCCUPATION_INFO_NOT_OCCUPATION_INFO_ERROR.format(type(occupation_info))
        assert isinstance(use_rpa_correlation, bool), \
            USE_RPA_CORRELATION_NOT_BOOL_ERROR.format(type(use_rpa_correlation))

        # check if the two ops_builders are consistent at quadrature nodes
        self._check_ops_builder_consistency_at_quadrature_nodes(ops_builder, ops_builder_oep)
    
        # initialize the Hartree-Fock exchange class
        HartreeFockExchange.__init__(
            self,
            ops_builder     = ops_builder, 
            occupation_info = occupation_info,
            )

        self.ops_builder_oep = ops_builder_oep
        self.physical_nodes  = ops_builder.physical_nodes

        # Some dimension information
        self.n_quad     : int = len(self.quadrature_nodes)
        self.n_interior : int = len(self.physical_nodes) - 2

        # Occupation information
        self.occupations  : np.ndarray = self.occupation_info.occupations
        self.occ_l_values : np.ndarray = self.occupation_info.l_values
        self.occ_n_values : np.ndarray = self.occupation_info.n_values

        # Ill_conditioned warning
        self.ill_conditioned_warning_caught_times_for_exchange : int = 0
        self.ill_conditioned_warning_caught_times_for_correlation : int = 0
        self.rcond_list_for_exchange : List[float] = []
        self.rcond_list_for_correlation : List[float] = []

        # Parameters for RPA correlation potential
        self.use_rpa_correlation = use_rpa_correlation

        if use_rpa_correlation:
            # check frequency quadrature point number
            assert frequency_quadrature_point_number is not None, \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_ERROR.format(frequency_quadrature_point_number)
            assert isinstance(frequency_quadrature_point_number, int), \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(frequency_quadrature_point_number))
            assert frequency_quadrature_point_number > 0, \
                FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(frequency_quadrature_point_number)
            
            # check angular momentum cutoff
            assert angular_momentum_cutoff is not None, \
                ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR.format(angular_momentum_cutoff)
            assert isinstance(angular_momentum_cutoff, int), \
                ANGULAR_MOMENTUM_CUTOFF_NOT_INTEGER_ERROR.format(type(angular_momentum_cutoff))
            assert angular_momentum_cutoff >= 0, \
                ANGULAR_MOMENTUM_CUTOFF_NEGATIVE_ERROR.format(angular_momentum_cutoff)

            # initialize the RPA correlation class
            RPACorrelation.__init__(
                self,
                ops_builder                       = ops_builder,
                occupation_info                   = occupation_info,
                frequency_quadrature_point_number = frequency_quadrature_point_number,
                angular_momentum_cutoff           = angular_momentum_cutoff,
                )
        else:
            if frequency_quadrature_point_number is not None:
                print(FREQUENCY_QUADRATURE_POINT_NUMBER_NOT_NONE_WHEN_RPA_CORRELATION_IS_NOT_USED_WARNING)
            if angular_momentum_cutoff is not None:
                print(ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_WHEN_RPA_CORRELATION_IS_NOT_USED_WARNING)


    def _check_ops_builder_consistency_at_quadrature_nodes(
        self,
        ops_builder    : RadialOperatorsBuilder,
        ops_builder_oep: RadialOperatorsBuilder
        ) -> None:
        assert np.allclose(ops_builder.quadrature_nodes, ops_builder_oep.quadrature_nodes), \
            OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_NODES_ERROR
        assert np.allclose(ops_builder.quadrature_weights, ops_builder_oep.quadrature_weights), \
            OPS_BUILDERS_NOT_CONSISTENT_AT_QUADRATURE_WEIGHTS_ERROR
        
        # skip checking for now, will implement other consistency checks later (if needed)
        pass 


    def reset(self, ):
        """
        Reset the OEP calculator
        """
        # reset the warning caught time
        self.ill_conditioned_warning_caught_times_for_exchange = 0
        self.ill_conditioned_warning_caught_times_for_correlation = 0
        # Reset the rcond list for exchange and correlation
        self.rcond_list_for_exchange.clear()
        self.rcond_list_for_correlation.clear()


    def compute_oep_potentials(
        self, 
        full_eigen_energies    : np.ndarray, 
        full_orbitals          : np.ndarray, 
        full_l_terms           : np.ndarray,
        enable_parallelization : Optional[bool] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OEP potentials from full orbitals and eigenvalues.

        Parameters
        ----------
        full_eigen_energies : np.ndarray
            Full eigenvalues of the system, shape (n_total_orbitals,)
        full_orbitals : np.ndarray
            Full orbitals of the system, shape (n_grid, n_total_orbitals)
        full_l_terms : np.ndarray
            Specify the l value of each orbital, shape (n_total_orbitals,)
        enable_parallelization : bool
            Flag for parallelization of RPA calculations

        Returns
        -------
        v_x_oep : np.ndarray
            OEP exchange potential, shape (n_grid,)
        v_c_oep : np.ndarray
            OEP correlation potential, shape (n_grid,)
        """

        # Type check for required fields
        self._validate_full_spectrum_inputs(full_eigen_energies, full_orbitals, full_l_terms)

        # Get occupation information
        occ_orbitals = full_orbitals[:, :len(self.occ_l_values)]

        # get the global interpolation matrix
        global_interpolation_matrix = self.ops_builder_oep.global_interpolation_matrix


        ### =========================================== ###
        ###  Part 1: Compute OEP exchange potential     ###
        ### =========================================== ###

        # Compute exact exchange potentials
        exact_exchange_potentials = self.compute_exchange_potentials(occ_orbitals)

        # Compute OEP exchange kernel and the exchange driving term
        chi_0_kernel, exchange_driving_term = \
            self._compute_oep_kernel_and_exchange_driving_term(
                full_eigen_energies = full_eigen_energies,
                full_orbitals       = full_orbitals,
                full_l_terms        = full_l_terms,
                exchange_potentials = exact_exchange_potentials
            )


        # Convert chi_0_kernel to sparser grid, 
        #   Note: this matrix is shared while computing the RPA correlation potential
        chi_0_kernel_sparser_grid = \
            self.convert_chi_0_kernel_to_sparser_grid(
                chi_0_kernel                = chi_0_kernel,
                quadrature_weights          = self.quadrature_weights,
                global_interpolation_matrix = global_interpolation_matrix,
            )


        # Convert exchange_driving_term to sparser grid
        exchange_driving_term_sparser_grid = \
            self.convert_driving_term_to_sparser_grid(
                driving_term                = exchange_driving_term,
                quadrature_weights          = self.quadrature_weights,
                global_interpolation_matrix = global_interpolation_matrix,
            )

        # solve for the OEP coefficient
        oep_coefficient = self.solve_oep_coefficients(
            chi_0_kernel = chi_0_kernel_sparser_grid,
            driving_term = exchange_driving_term_sparser_grid,
            ex_tag       = 'exchange'
        )

        # compute the OEP exchange potential
        v_x_oep = global_interpolation_matrix @ oep_coefficient


        # zero point shift the OEP exchange potential
        energy_correction = self._compute_zero_point_shift_correction_term(
            quadrature_weights = self.quadrature_weights,
            orbital_homo       = occ_orbitals[:, -1],
            exx_potential_homo = exact_exchange_potentials[-1, :],
            oep_potential      = v_x_oep,
        )

        # Add the zero point shift correction term to the OEP exchange potential
        v_x_oep += energy_correction


        ### =========================================== ###
        ###  Part 2: Compute OEP correlation potential  ###
        ### =========================================== ###

        # compute RPA correlation potential, if needed
        if not self.use_rpa_correlation:
            v_c_oep = np.zeros_like(v_x_oep)
        else:
            # Compute RPA correlation driving term
            rpa_correlation_driving_term = self._compute_rpa_correlation_driving_term(
                full_eigen_energies    = full_eigen_energies,
                full_orbitals          = full_orbitals,
                full_l_terms           = full_l_terms,
                enable_parallelization = enable_parallelization,
            )
            
            # Convert RPA correlation driving term to sparser grid
            rpa_correlation_driving_term_sparser_grid = \
                self.convert_driving_term_to_sparser_grid(
                    driving_term                = rpa_correlation_driving_term,
                    quadrature_weights          = self.quadrature_weights,
                    global_interpolation_matrix = global_interpolation_matrix
                )

            # Solve for the RPA correlation coefficient
            rpa_correlation_coefficient = self.solve_oep_coefficients(
                chi_0_kernel = chi_0_kernel_sparser_grid,
                driving_term = rpa_correlation_driving_term_sparser_grid,
                ex_tag       = 'correlation'
            )

            # shift the RPA correlation coefficient by the HOMO coefficient
            rpa_correlation_coefficient -= rpa_correlation_coefficient[-1]

            # Compute the RPA correlation potential
            v_c_oep = global_interpolation_matrix @ rpa_correlation_coefficient

        return v_x_oep, v_c_oep


    def solve_oep_coefficients(
        self,
        chi_0_kernel : np.ndarray,
        driving_term : np.ndarray,
        ex_tag       : str,
        ) -> np.ndarray:
        """
        Solve the OEP coefficients for exchange or correlation potential
        
        Parameters
        ----------
        chi_0_kernel : np.ndarray
            Chi_0 kernel, shape (n_quad, n_quad)
        driving_term : np.ndarray
            Driving term, shape (n_quad,)
        ex_tag : str
            Tag for the potential, must be 'exchange' or 'correlation'
            This parameter is used to record the warning caught times and rcond_list for exchange or 
            correlation potential while solving the OEP coefficients.
        
        Returns
        -------
        oep_coefficient : np.ndarray
            OEP coefficients, shape (n_quad,)
        """
        assert ex_tag in ['exchange', 'correlation'], \
            INVALID_EX_TAG_ERROR.format(ex_tag)
        
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LinAlgWarning)
            oep_coefficient = scipy.linalg.solve(chi_0_kernel, driving_term)

        if caught:
            cond = np.linalg.cond(chi_0_kernel)
            rcond = 1.0 / cond if cond != 0 else np.inf
            if ex_tag == 'exchange':
                self.ill_conditioned_warning_caught_times_for_exchange += 1
                self.rcond_list_for_exchange.append(rcond)
            elif ex_tag == 'correlation':
                self.ill_conditioned_warning_caught_times_for_correlation += 1
                self.rcond_list_for_correlation.append(rcond)
        
        return oep_coefficient


    @staticmethod
    def _validate_full_spectrum_inputs(
        full_eigen_energies : np.ndarray, 
        full_orbitals       : np.ndarray, 
        full_l_terms        : np.ndarray
        ) -> None:
        """
        Validate the inputs of full spectrum
        """
        assert isinstance(full_eigen_energies, np.ndarray), \
            FULL_EIGEN_ENERGIES_NOT_NUMPY_ARRAY_ERROR.format(type(full_eigen_energies))
        assert isinstance(full_orbitals, np.ndarray), \
            FULL_ORBITALS_NOT_NUMPY_ARRAY_ERROR.format(type(full_orbitals))
        assert isinstance(full_l_terms, np.ndarray), \
            FULL_L_TERMS_NOT_NUMPY_ARRAY_ERROR.format(type(full_l_terms))
        assert full_eigen_energies.ndim == 1, \
            FULL_EIGEN_ENERGIES_NOT_1D_ARRAY_ERROR.format(full_eigen_energies.ndim)
        assert full_orbitals.ndim == 2, \
            FULL_ORBITALS_NOT_2D_ARRAY_ERROR.format(full_orbitals.ndim)
        assert full_l_terms.ndim == 1, \
            FULL_L_TERMS_NOT_1D_ARRAY_ERROR.format(full_l_terms.ndim)
        assert full_eigen_energies.shape[0] == full_orbitals.shape[1], \
            FULL_EIGEN_ENERGIES_AND_ORBITALS_SHAPE_ERROR.format(full_eigen_energies.shape[0], full_orbitals.shape[1])
        assert full_eigen_energies.shape[0] == full_l_terms.shape[0], \
            FULL_EIGEN_ENERGIES_AND_L_TERMS_SHAPE_ERROR.format(full_eigen_energies.shape[0], full_l_terms.shape[0])
        assert full_orbitals.shape[1] == full_l_terms.shape[0], \
            FULL_ORBITALS_AND_L_TERMS_SHAPE_ERROR.format(full_orbitals.shape[1], full_l_terms.shape[0])
        assert np.all(full_l_terms >= 0), \
            FULL_L_TERMS_NON_NEGATIVE_ERROR.format(full_l_terms)


    @staticmethod
    def _compute_zero_point_shift_correction_term(
        quadrature_weights : np.ndarray,
        orbital_homo       : np.ndarray,
        exx_potential_homo : np.ndarray,
        oep_potential      : np.ndarray,
        ) -> float:
        """
        Compute the zero point shift correction term for the OEP exchange potential
        """
        homo_oep_projection_energy = np.sum(oep_potential * (orbital_homo ** 2) * quadrature_weights)
        homo_exact_exchange_energy = np.sum(exx_potential_homo * orbital_homo)
        energy_correction = -homo_oep_projection_energy + homo_exact_exchange_energy
        return energy_correction


    @staticmethod
    def convert_chi_0_kernel_to_sparser_grid(
        chi_0_kernel                : np.ndarray,
        quadrature_weights          : np.ndarray,
        global_interpolation_matrix : np.ndarray,
        ) -> np.ndarray:
        """
        Convert chi_0_kernel and driving_term to sparser grid
        """
        # convert chi_0_kernel and exchange_driving_term to sparser grid
        chi_0_kernel_sparser_grid = np.einsum('i,ij,il,lk,l->jk',
            quadrature_weights,
            global_interpolation_matrix,
            chi_0_kernel,
            global_interpolation_matrix,
            quadrature_weights,
            optimize=True
        )

        return chi_0_kernel_sparser_grid


    @staticmethod
    def convert_driving_term_to_sparser_grid(
        driving_term                : np.ndarray,
        quadrature_weights          : np.ndarray,
        global_interpolation_matrix : np.ndarray,
        ) -> np.ndarray:
        """
        Convert driving term to sparser grid
        """
        driving_term_sparser_grid = np.einsum('i, ij, i->j',
            quadrature_weights,
            global_interpolation_matrix,
            driving_term,
            optimize=True
        )
        return driving_term_sparser_grid


    def _compute_oep_kernel_and_exchange_driving_term(
        self,
        full_eigen_energies : np.ndarray, 
        full_orbitals       : np.ndarray, 
        full_l_terms        : np.ndarray,
        exchange_potentials : np.ndarray
        ) -> np.ndarray:
        """
        Compute OEP exchange kernel and the exchange driving term
        """
        # get occupied orbitals
        occ_orbitals = full_orbitals[:, :len(self.occ_l_values)]

        # get l channel indices for all orbitals
        l_max = np.max(self.occ_l_values)
        l_channel_orbital_indices = np.zeros((l_max + 1, self.n_interior), dtype=np.int32)
        for l in range(l_max + 1):
            l_channel_orbital_indices[l, :] = np.argwhere(full_l_terms == l)[:,0]

        # compute chi_0_kernel and the exchange driving term
        chi_0_kernel          = np.zeros((self.n_quad, self.n_quad))
        exchange_driving_term = np.zeros(self.n_quad)

        for idx in range(len(self.occ_l_values)):
            # get l and n index
            l_value = self.occ_l_values[idx]
            n_value = self.occ_n_values[idx] - l_value - 1
            l_occ_num = len(np.argwhere(self.occ_l_values == l_value)[:,0])

            # get all orbitals with indices of the same l value
            unocc_orbitals_in_l_channel = full_orbitals[:, l_channel_orbital_indices[l_value, :]][:, l_occ_num:]

            # get the difference of eigenvalues
            l_channel_eigenvalues = full_eigen_energies[l_channel_orbital_indices[l_value, :]]
            diff_eigenvalues = l_channel_eigenvalues.reshape(-1, 1) - l_channel_eigenvalues.reshape(1, -1)
            one_over_diff_eigenvalues = 1 / (diff_eigenvalues + np.eye(self.n_interior))           # avoid division by zero
            one_over_diff_eigenvalues[np.arange(self.n_interior), np.arange(self.n_interior)] = 0  # set the diagonal to zero
            
            # get the green function block
            _exchange_green_block = np.einsum('ji,ki,i->jk',
                unocc_orbitals_in_l_channel,
                unocc_orbitals_in_l_channel,
                one_over_diff_eigenvalues[n_value, l_occ_num:],
                optimize=True
            )

            # get the orbital and corresponding exchange potential inside this for loop
            orbital            = occ_orbitals[:, idx]
            exchange_potential = exchange_potentials[idx]

            # update chi_0_kernel
            chi_0_kernel += 4 * np.einsum('k,kj,j->kj',
                orbital,
                _exchange_green_block,
                orbital,
                optimize=True
            ) * (2 * l_value + 1) 

            # update exchange driving term
            exchange_driving_term += 4 * np.einsum('k,kl,l->k',
                orbital,
                _exchange_green_block,
                exchange_potential,
                optimize=True
            ) * (2 * l_value + 1)

        return chi_0_kernel, exchange_driving_term



    def _compute_rpa_correlation_driving_term(
        self,
        full_eigen_energies    : np.ndarray,
        full_orbitals          : np.ndarray,
        full_l_terms           : np.ndarray,
        enable_parallelization : bool = False,
        ) -> np.ndarray:
        """
        Compute RPA correlation driving term from full spectrum in parallel.
        """
        assert hasattr(self, 'frequency_grid') and hasattr(self, 'frequency_weights'), \
            PARENT_CLASS_RPACORRELATION_NOT_INITIALIZED_ERROR
        assert self.angular_momentum_cutoff is not None, \
            ANGULAR_MOMENTUM_CUTOFF_NOT_NONE_ERROR_MESSAGE
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

        q1c_term = np.zeros(self.n_quad)
        q2c_term = np.zeros(self.n_quad)

        # if enable_parallelization is False, compute RPA correlation driving term at each frequency and sum them up

        if not enable_parallelization:
            # Compute RPA correlation driving term at each frequency and sum them up
            for frequency, frequency_weight in zip(self.frequency_grid, self.frequency_weights):
                q1c_term_single, q2c_term_single = self._compute_rpa_correlation_driving_term_for_single_frequency(
                    frequency               = frequency,
                    angular_momentum_cutoff = self.angular_momentum_cutoff,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,
                )
                
                q1c_term += 4 * q1c_term_single * frequency_weight  # 4 comes from spin degeneracy
                q2c_term += 2 * q2c_term_single * frequency_weight  # 2 comes from spin degeneracy
        else:
            import multiprocessing as mp
            from concurrent.futures import ThreadPoolExecutor


            def _single_frequency_task(args):
                idx, (frequency, frequency_weight) = args
                q1c_term_single, q2c_term_single = self._compute_rpa_correlation_driving_term_for_single_frequency(
                    frequency               = frequency,
                    angular_momentum_cutoff = self.angular_momentum_cutoff,
                    occupation_info         = self.occupation_info,
                    full_eigen_energies     = full_eigen_energies,
                    full_orbitals           = full_orbitals,
                    full_l_terms            = full_l_terms,
                    wigner_symbols_squared  = wigner_symbols_squared,
                    radial_kernels_dict     = radial_kernels_dict,
                )
                return (
                    idx,
                    4 * q1c_term_single * frequency_weight,
                    2 * q2c_term_single * frequency_weight,
                )

            n_workers = min(max(1, mp.cpu_count()), len(self.frequency_grid))

            # limit BLAS/OpenMP threads during parallel sections
            if threadpool_limits is not None:
                blas_ctx = threadpool_limits(limits=1)
            else:
                blas_ctx = nullcontext()

            # run the parallel sections
            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(
                    _single_frequency_task,
                    enumerate(zip(self.frequency_grid, self.frequency_weights))
                )
                for _, q1c_single_weighted, q2c_single_weighted in results:
                    q1c_term += q1c_single_weighted
                    q2c_term += q2c_single_weighted


        assert q1c_term.shape == (self.n_quad,)
        assert q2c_term.shape == (self.n_quad,)

        total_driving_term = (q1c_term + q2c_term) / (2 * np.pi)

        return total_driving_term

