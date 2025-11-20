"""
Hartree-Fock Exchange Calculation

Implements exact (Hartree-Fock) exchange for hybrid functionals.
This is used for orbital-dependent functionals like PBE0, B3LYP, etc.

Reference implementation: datagen/tools/HF_EX.py
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.occupation_states import OccupationInfo
    from ..mesh.operators import RadialOperatorsBuilder

# Error messages
FACTORIAL_N_MUST_BE_NON_NEGATIVE_INTEGER_ERROR = \
    "parameter n must be a non-negative integer, get {} instead"

L_VALUES_MUST_BE_INTEGERS_ERROR = \
    "parameter l_values in class OccupationInfo must be integers, get type {} instead"

ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR = \
    "parameter orbitals must be a numpy array, get type {} instead"
ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR = \
    "parameter orbitals must be a 2D numpy array, get dimension {} instead"
ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR = \
    "parameter orbitals must have n_grid rows, get {} rows instead of {} rows"
ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR = \
    "parameter orbitals must have n_orbitals columns, get {} columns instead of {} columns"

EXCHANGE_POTENTIAL_OUTPUT_SHAPE_ERROR = \
    "exchange potential must have shape (n_orbitals, n_grid), get shape {} instead"

def factorial(n: int) -> int:
    """
    Compute factorial n! = n * (n-1) * ... * 2 * 1
    
    For n = 0, returns 1.
    
    Uses lookup table for common values to avoid repeated computation.
    """
    assert n >= 0 and isinstance(n, int), \
        FACTORIAL_N_MUST_BE_NON_NEGATIVE_INTEGER_ERROR.format(n)

    if n == 0: return 1
    elif n == 1: return 1
    elif n == 2: return 2
    elif n == 3: return 6
    elif n == 4: return 24
    elif n == 5: return 120
    elif n == 6: return 720
    elif n == 7: return 5040
    elif n == 8: return 40320
    else:
        # Use iterative approach to avoid recursion depth issues
        result = 40320
        for i in range(9, n + 1):
            result *= i
        return result



class CoulombCouplingCalculator:

    @staticmethod
    def radial_kernel(l: int, r_nodes: np.ndarray, r_weights: np.ndarray) -> np.ndarray:
        """
        Compute kernel K^(l) with entries:
            K_ij^(l) = [ r_<^l / r_>^(l+1) ] * (w_i w_j) / (2l + 1),
        where r_< = min(r_i, r_j), r_> = max(r_i, r_j).

        This term represents the radial part of the spherical harmonic expansion of the Coulomb interaction.
        """
        r_min = np.minimum(r_nodes, r_nodes.reshape(-1, 1))
        r_max = np.maximum(r_nodes, r_nodes.reshape(-1, 1))
        
        return ((r_min / r_max)**l / r_max) * (r_weights * r_weights.reshape(-1, 1)) / (2*l + 1)


    @staticmethod
    def wigner_3j_000(l1: int, l2: int, L: int) -> float:
        """
        Wigner 3j symbol (l1 l2 L; 0 0 0) with built-in selection rules.
        """
        assert isinstance(l1, int), "l1 must be an integer, get type {} instead".format(type(l1))
        assert isinstance(l2, int), "l2 must be an integer, get type {} instead".format(type(l2))
        assert isinstance(L, int) , "L must be an integer, get type {} instead".format(type(L))
        J = l1 + l2 + L
        # parity: l1 + l2 + L must be even
        if (J & 1) == 1:
            return 0.0
        # triangle inequalities
        if l1 < abs(l2 - L) or l1 > l2 + L:
            return 0.0
        if l2 < abs(l1 - L) or l2 > l1 + L:
            return 0.0
        if L  < abs(l1 - l2) or L  > l1 + l2:
            return 0.0

        g = J // 2
        W = (-1)**g
        W *= np.sqrt(
            factorial(J - 2*l1) * factorial(J - 2*l2) * factorial(J - 2*L)
            / factorial(J + 1)
        )
        W *= factorial(g) / (factorial(g - l1) * factorial(g - l2) * factorial(g - L))
        return float(W)



class HartreeFockExchange:
    """
    Hartree-Fock Exchange Calculator
    
    Computes exact (Hartree-Fock) exchange for hybrid functionals.
    This is used for orbital-dependent functionals like PBE0, B3LYP, etc.
    """
    
    def __init__(
        self,
        ops_builder    : 'RadialOperatorsBuilder',
        occupation_info: 'OccupationInfo'
        ):
        """
        Initialize HF exchange calculator.
        
        Parameters
        ----------
        ops_builder
            RadialOperatorsBuilder instance containing quadrature data
        occupation_info : OccupationInfo
            Occupation information containing l_values and occupations
        """
        self.ops_builder = ops_builder
        self.occupation_info = occupation_info
        
        # Extract quadrature data from ops_builder
        self.quadrature_nodes   = ops_builder.quadrature_nodes
        self.quadrature_weights = ops_builder.quadrature_weights
        self.n_grid = len(self.quadrature_nodes)
        
        # Extract occupation data
        self.l_values    = occupation_info.l_values
        self.occupations = occupation_info.occupations
        self.n_orbitals  = len(self.l_values)

        assert self.l_values.dtype == int, \
            L_VALUES_MUST_BE_INTEGERS_ERROR.format(self.l_values.dtype)



    def _compute_exchange_matrix(
        self, 
        l_value  : int,
        orbitals : np.ndarray) -> np.ndarray:

        # interpolation_matrix: (n_quad, n_physical)
        interpolation_matrix = self.ops_builder.global_interpolation_matrix
        
        # Determine l's range
        l_min = np.min(np.abs(l_value - self.l_values))
        l_max = np.max(l_value + self.l_values)
        l_coupling = np.arange(int(l_min), int(l_max) + 1)

        # Compute exchange matrix on physical grid
        n_physical = interpolation_matrix.shape[1]  # (n_quad, n_physical)
        H_hf_exchange_matrix = np.zeros((n_physical, n_physical), dtype=float)


        for l_prime in l_coupling:
            # Angular part: compute vectorized alpha without in-place modification
            w3j_values = np.array([
                CoulombCouplingCalculator.wigner_3j_000(int(l_value), int(lj), int(l_prime)) for lj in self.l_values
            ], dtype=float)
            
            # Radial part: compute the radial coupling kernel K^(L)
            radial_kernel = CoulombCouplingCalculator.radial_kernel(
                int(l_prime), self.quadrature_nodes, self.quadrature_weights
            )

            # Compute exchange matrix contribution
            H_hf_exchange_matrix_l_contribution_at_quadrature_nodes = np.einsum(
                'jn,ln,n->jl',
                orbitals,                                     # (n_grid, n_orbitals)
                orbitals,                                     # (n_grid, n_orbitals)
                (2 * self.l_values + 1) * (w3j_values ** 2),  # (n_orbitals, )
                optimize=True,
            ) * radial_kernel

            H_hf_exchange_matrix += (2 * l_prime + 1) * \
                np.einsum(
                    'ij,lk,il->jk',
                    interpolation_matrix,                                    # (n_quad, n_physical)
                    interpolation_matrix,                                    # (n_quad, n_physical)
                    H_hf_exchange_matrix_l_contribution_at_quadrature_nodes, # (n_quad, n_quad)
                    optimize=True,
                )
        
        # Be careful with the sign change of the exchange matrix here.
        return - H_hf_exchange_matrix


    def compute_exchange_potentials(
        self,
        orbitals):
        """
        Compute Hartree-Fock exchange potentials for all angular momentum channels.

        This function is useful for the OEP calculation. Here, the input orbitals should only contain the occupied orbitals.

        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)

        Returns
        -------
        np.ndarray
            Hartree-Fock exchange potential for all angular momentum channels
            Shape: (len(l_values), n_grid)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(self.n_grid, orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(self.n_orbitals, orbitals.shape[1])

        # Compute HF exchange matrices for all l channels
        l_coupling = np.arange(0, 2 * np.max(self.l_values) + 1)

        # Compute exchange potential for each l channel
        exchange_potential_l_contribution_list : List[np.ndarray] = []
        for l_prime in l_coupling:
            _wigner_term = np.array([
                CoulombCouplingCalculator.wigner_3j_000(int(l1), int(l2), int(l_prime))**2 for l1 in self.l_values for l2 in self.l_values
            ], dtype=float).reshape(len(self.l_values), len(self.l_values))
            
            _exchange_potential_l_contribution = -0.5 * np.einsum(
                'ki,ji,ikj->kj',
                _wigner_term * self.occupations,
                orbitals,
                np.einsum('li,lk,jl->ikj',
                    orbitals,
                    orbitals,
                    CoulombCouplingCalculator.radial_kernel(l_prime, self.quadrature_nodes, self.quadrature_weights) * (2 * l_prime + 1),
                ),
                optimize=True,
            )

            exchange_potential_l_contribution_list.append(_exchange_potential_l_contribution)

        exchange_potential = np.sum(exchange_potential_l_contribution_list, axis=0)
        assert exchange_potential.shape == (self.n_orbitals, self.n_grid), \
            EXCHANGE_POTENTIAL_OUTPUT_SHAPE_ERROR.format(exchange_potential.shape, self.n_orbitals, self.n_grid)
        
        return exchange_potential


    def compute_exchange_matrices_dict(
        self,
        orbitals: np.ndarray
        ) -> Dict[int, np.ndarray]:
        """
        Compute Hartree-Fock exchange matrices for all l channels.
        
        This method calculates HF exchange matrices for each angular momentum
        channel separately and returns them as a dictionary.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping l values to HF exchange matrices
            Keys are unique l values from occupation_info
            Values are HF exchange matrices of shape (n_physical, n_physical)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        
        # Compute HF exchange matrices for all l channels
        H_hf_exchange_matrices_dict : Dict[int, np.ndarray] = {}
        for l_value in self.occupation_info.unique_l_values:
            H_hf_exchange_matrix = self._compute_exchange_matrix(l_value, orbitals)
            H_hf_exchange_matrices_dict[l_value] = H_hf_exchange_matrix
        
        return H_hf_exchange_matrices_dict


    def compute_exchange_energy(
        self,
        orbitals: np.ndarray
        ) -> float:
        """
        Compute Hartree-Fock exchange energy.
        
        This method calculates the total HF exchange energy using the same
        logic as the reference implementation but adapted to our data structures.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        float
            Total Hartree-Fock exchange energy (scalar)
        """
        # Check Type and shape
        assert isinstance(orbitals, np.ndarray), \
            ORBITALS_MUST_BE_A_NUMPY_ARRAY_ERROR.format(type(orbitals))
        assert orbitals.ndim == 2, \
            ORBITALS_MUST_BE_A_2D_NUMPY_ARRAY_ERROR.format(orbitals.ndim)
        assert orbitals.shape[0] == self.n_grid, \
            ORBITALS_MUST_HAVE_N_GRID_ROWS_ERROR.format(orbitals.shape[0])
        assert orbitals.shape[1] == self.n_orbitals, \
            ORBITALS_MUST_HAVE_N_ORBITALS_COLUMNS_ERROR.format(orbitals.shape[1])
        
        # Extract occupation data
        l_values    = self.l_values    # Angular momentum quantum numbers
        occupations = self.occupations # Occupation numbers
        
        # Initialize total exchange energy
        E_HF = 0.0
        
        # Loop over all possible l values for coupling
        max_l = np.max(l_values)
        for l_coupling in range(0, 2 * max_l + 1):
            
            # Create Wigner 3j symbol matrix for this l coupling
            wigner_matrix = np.zeros((len(l_values), len(l_values)))
            for i1 in range(len(l_values)):
                for i2 in range(len(l_values)):
                    wigner_matrix[i1, i2] = CoulombCouplingCalculator.wigner_3j_000(int(l_values[i1]), int(l_values[i2]), int(l_coupling))**2
            
            # Create occupation matrix
            occ_matrix = occupations * occupations.reshape(-1, 1)
            
            # Compute radial kernel for this l coupling
            r_kernel = CoulombCouplingCalculator.radial_kernel(l_coupling, self.quadrature_nodes, self.quadrature_weights)
            
            # Compute exchange energy contribution for this l coupling
            # This is the complex einsum from the reference code:
            # 'ij,il,ik,jk,jl,kl->'
            # where:
            # - ij: occupation * wigner matrix
            # - il,ik: orbitals (first two indices)
            # - jk,jl: orbitals (second two indices) 
            # - kl: radial kernel
            exchange_contribution = -0.25 * (2 * l_coupling + 1) * np.einsum(
                'ij,li,ki,kj,lj,kl->',
                occ_matrix * wigner_matrix,
                orbitals,  # il
                orbitals,  # ik  
                orbitals,  # jk
                orbitals,  # jl
                r_kernel,  # kl
                optimize=True
            )
            
            E_HF += exchange_contribution
        
        return E_HF


