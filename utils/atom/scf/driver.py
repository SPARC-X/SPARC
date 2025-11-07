from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field

from .hamiltonian import HamiltonianBuilder, SwitchesFlags
from .density import DensityCalculator, DensityData
from .poisson import PoissonSolver
from .convergence import ConvergenceChecker
from .eigensolver import EigenSolver
from .mixer import Mixer
from ..utils.occupation_states import OccupationInfo
from ..xc.evaluator import XCEvaluator, create_xc_evaluator, XCPotentialData
from ..xc.functional_requirements import get_functional_requirements, FunctionalRequirements
from ..xc.hybrid import HartreeFockExchange


# SCF Settings Error Messages
INNER_MAX_ITER_TYPE_ERROR_MESSAGE = \
    "parameter inner_max_iter must be an integer, get type {} instead"
RHO_TOL_TYPE_ERROR_MESSAGE = \
    "parameter rho_tol must be a float, get type {} instead"
N_CONSECUTIVE_TYPE_ERROR_MESSAGE = \
    "parameter n_consecutive must be an integer, get type {} instead"
OUTER_MAX_ITER_TYPE_ERROR_MESSAGE = \
    "parameter outer_max_iter must be an integer, get type {} instead"
OUTER_RHO_TOL_TYPE_ERROR_MESSAGE = \
    "parameter outer_rho_tol must be a float, get type {} instead"
PRINT_DEBUG_TYPE_ERROR_MESSAGE = \
    "parameter print_debug must be a boolean, get type {} instead"

# SCF Result Error Messages
EIGENVALUES_TYPE_ERROR_MESSAGE = \
    "parameter eigenvalues must be a numpy array, get type {} instead"
EIGENVECTORS_TYPE_ERROR_MESSAGE = \
    "parameter eigenvectors must be a numpy array, get type {} instead"
RHO_TYPE_ERROR_MESSAGE = \
    "parameter rho must be a numpy array, get type {} instead"
CONVERGED_TYPE_ERROR_MESSAGE = \
    "parameter converged must be a boolean, get type {} instead"
ITERATIONS_TYPE_ERROR_MESSAGE = \
    "parameter iterations must be an integer, get type {} instead"
RESIDUAL_TYPE_ERROR_MESSAGE = \
    "parameter residual must be a float, get type {} instead"
TOTAL_ENERGY_TYPE_ERROR_MESSAGE = \
    "parameter total_energy must be a float, get type {} instead"
ENERGY_COMPONENTS_TYPE_ERROR_MESSAGE = \
    "parameter energy_components must be a dictionary, get type {} instead"
OUTER_ITERATIONS_TYPE_ERROR_MESSAGE = \
    "parameter outer_iterations must be an integer, get type {} instead"
OUTER_CONVERGED_TYPE_ERROR_MESSAGE = \
    "parameter outer_converged must be a boolean, get type {} instead"


# SCF Driver Error Messages
HAMILTONIAN_BUILDER_TYPE_ERROR_MESSAGE = \
    "parameter hamiltonian_builder must be a HamiltonianBuilder, get type {} instead"
DENSITY_CALCULATOR_TYPE_ERROR_MESSAGE = \
    "parameter density_calculator must be a DensityCalculator, get type {} instead"
POISSON_SOLVER_TYPE_ERROR_MESSAGE = \
    "parameter poisson_solver must be a PoissonSolver, get type {} instead"
EIGENSOLVER_TYPE_ERROR_MESSAGE = \
    "parameter eigensolver must be a EigenSolver, get type {} instead"
MIXER_TYPE_ERROR_MESSAGE = \
    "parameter mixer must be a Mixer, get type {} instead"
OCCUPATION_INFO_TYPE_ERROR_MESSAGE = \
    "parameter occupation_info must be a OccupationInfo, get type {} instead"


RHO_INITIAL_TYPE_ERROR_MESSAGE = \
    "parameter rho_initial must be a numpy array, get type {} instead"
SETTINGS_TYPE_ERROR_MESSAGE = \
    "parameter settings must be a SCFSettings or a dictionary, get type {} instead"
H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR_MESSAGE = \
    "provided parameter H_hf_exchange_dict_by_l must be a dictionary, get type {} instead"
ORBITALS_INITIAL_TYPE_ERROR_MESSAGE = \
    "parameter orbitals_initial must be a numpy array, get type {} instead"

# SCF Driver Warning Messages
INNER_SCF_DID_NOT_CONVERGE_WARNING = \
    "WARNING: Inner SCF did not converge after {max_iter} iterations"
HF_CALCULATOR_NOT_AVAILABLE_WARNING = \
    "WARNING: Hartree-Fock calculator is not available, please initialize the HF calculator first"


@dataclass
class SCFSettings:
    """
    Configuration settings for SCF calculation
    
    Attributes
    ----------
    inner_max_iter : int
        Maximum number of inner SCF iterations
    rho_tol : float
        Convergence tolerance for density residual
    n_consecutive : int
        Number of consecutive converged iterations required
    outer_max_iter : int
        Maximum number of outer SCF iterations (for HF/OEP/RPA)
    outer_rho_tol : float
        Convergence tolerance for outer loop density residual
    print_debug : bool
        Whether to print debug information during SCF
    """
    
    # Inner loop settings
    inner_max_iter: int = 200
    rho_tol: float = 1e-6
    n_consecutive: int = 1
    
    # Outer loop settings
    outer_max_iter: int = 1
    outer_rho_tol: float = 1e-5
    
    # Output settings
    print_debug: bool = False


    def __post_init__(self):
        # type check (allow int, float and numpy types)
        assert isinstance(self.inner_max_iter, (int, np.integer)), \
            INNER_MAX_ITER_TYPE_ERROR_MESSAGE.format(type(self.inner_max_iter))
        assert isinstance(self.rho_tol, (int, float, np.floating)), \
            RHO_TOL_TYPE_ERROR_MESSAGE.format(type(self.rho_tol))
        assert isinstance(self.n_consecutive, (int, np.integer)), \
            N_CONSECUTIVE_TYPE_ERROR_MESSAGE.format(type(self.n_consecutive))
        assert isinstance(self.outer_max_iter, (int, np.integer)), \
            OUTER_MAX_ITER_TYPE_ERROR_MESSAGE.format(type(self.outer_max_iter))
        assert isinstance(self.outer_rho_tol, (int, float, np.floating)), \
            OUTER_RHO_TOL_TYPE_ERROR_MESSAGE.format(type(self.outer_rho_tol))
        assert isinstance(self.print_debug, (bool, np.bool_)), \
            PRINT_DEBUG_TYPE_ERROR_MESSAGE.format(type(self.print_debug))

    
    @classmethod
    def from_dict(cls, settings_dict: dict) -> SCFSettings:
        """
        Create SCFSettings from dictionary
        
        Parameters
        ----------
        settings_dict : dict
            Dictionary containing settings
            
        Returns
        -------
        SCFSettings
            Settings object
        """
        return cls(
            inner_max_iter=settings_dict.get('inner_max_iter', 200),
            rho_tol=settings_dict.get('rho_tol', 1e-6),
            n_consecutive=settings_dict.get('n_consecutive', 1),
            outer_max_iter=settings_dict.get('outer_max_iter', 1),
            outer_rho_tol=settings_dict.get('outer_rho_tol', 1e-5),
            print_debug=settings_dict.get('print_debug', False)
        )
    

    def to_dict(self) -> dict:
        """
        Convert to dictionary format
        
        Returns
        -------
        dict
            Dictionary representation of settings
        """
        return {
            'inner_max_iter': self.inner_max_iter,
            'rho_tol': self.rho_tol,
            'n_consecutive': self.n_consecutive,
            'outer_max_iter': self.outer_max_iter,
            'outer_rho_tol': self.outer_rho_tol,
            'print_debug': self.print_debug
        }


@dataclass
class SCFResult:
    """
    Results from SCF calculation
    
    Attributes
    ----------
    eigen_energies : np.ndarray
        Kohn-Sham eigenvalues (orbital energies) for all states, shape (n_states,)
    orbitals : np.ndarray
        Converged Kohn-Sham orbitals (radial wavefunctions R_nl(r))
        Shape: (n_states, n_quad_points)
    density_data : DensityData
        Converged electron density and related quantities (rho, grad_rho, tau)
    converged : bool
        Whether inner SCF loop converged
    iterations : int
        Number of inner SCF iterations performed
    rho_residual : float
        Final density residual of inner loop (L2 norm)
    outer_iterations : int, optional
        Number of outer SCF iterations (for HF/OEP/RPA methods)
    outer_converged : bool, optional
        Whether outer SCF loop converged
    total_energy : float, optional
        Total energy of the system
    energy_components : dict, optional
        Breakdown of energy components (kinetic, Hartree, XC, etc.)
    """
    
    # Core results
    eigen_energies: np.ndarray
    orbitals: np.ndarray
    density_data: DensityData
    
    # Inner loop convergence info
    converged: bool
    iterations: int
    rho_residual: float
    
    # Outer loop info (optional)
    outer_iterations  : Optional[int]   = None
    outer_converged   : Optional[bool]  = None
    
    # Energy info (optional)
    total_energy      : Optional[float] = None
    energy_components : Optional[dict]  = field(default=None)
    
    def __post_init__(self):
        # type check for required fields
        assert isinstance(self.eigen_energies, np.ndarray), \
            "eigen_energies must be a numpy array, got {}".format(type(self.eigen_energies))
        assert isinstance(self.orbitals, np.ndarray), \
            "orbitals must be a numpy array, got {}".format(type(self.orbitals))
        assert isinstance(self.density_data, DensityData), \
            "density_data must be a DensityData instance, got {}".format(type(self.density_data))
        assert isinstance(self.converged, bool), \
            CONVERGED_TYPE_ERROR_MESSAGE.format(type(self.converged))
        assert isinstance(self.iterations, int), \
            ITERATIONS_TYPE_ERROR_MESSAGE.format(type(self.iterations))
        assert isinstance(self.rho_residual, (float, np.floating)), \
            "rho_residual must be a float, got {}".format(type(self.rho_residual))
        
        # type check for optional fields (only if not None)
        if self.outer_iterations is not None:
            assert isinstance(self.outer_iterations, int), \
                OUTER_ITERATIONS_TYPE_ERROR_MESSAGE.format(type(self.outer_iterations))
        if self.outer_converged is not None:
            assert isinstance(self.outer_converged, bool), \
                OUTER_CONVERGED_TYPE_ERROR_MESSAGE.format(type(self.outer_converged))
        if self.total_energy is not None:
            assert isinstance(self.total_energy, (float, np.floating)), \
                TOTAL_ENERGY_TYPE_ERROR_MESSAGE.format(type(self.total_energy))
        if self.energy_components is not None:
            assert isinstance(self.energy_components, dict), \
                ENERGY_COMPONENTS_TYPE_ERROR_MESSAGE.format(type(self.energy_components))


    @classmethod
    def from_dict(cls, result_dict: dict) -> SCFResult:
        """
        Create SCFResult from dictionary
        
        Parameters
        ----------
        result_dict : dict
            Dictionary containing results
            
        Returns
        -------
        SCFResult
            Result object
        """
        return cls(
            eigenvalues=result_dict['eigenvalues'],
            eigenvectors=result_dict['eigenvectors'],
            rho=result_dict['rho'],
            converged=result_dict['converged'],
            iterations=result_dict['iterations'],
            residual=result_dict['residual'],
            outer_iterations=result_dict.get('outer_iterations'),
            outer_converged=result_dict.get('outer_converged'),
            total_energy=result_dict.get('total_energy'),
            energy_components=result_dict.get('energy_components')
        )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format
        
        Returns
        -------
        dict
            Dictionary representation of results
        """
        result = {
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'rho': self.rho,
            'converged': self.converged,
            'iterations': self.iterations,
            'residual': self.residual
        }
        
        # Add optional fields if present
        if self.outer_iterations is not None:
            result['outer_iterations'] = self.outer_iterations
        if self.outer_converged is not None:
            result['outer_converged'] = self.outer_converged
        if self.total_energy is not None:
            result['total_energy'] = self.total_energy
        if self.energy_components is not None:
            result['energy_components'] = self.energy_components
            
        return result
    
    def summary(self) -> str:
        """
        Get a summary string of the SCF results
        
        Returns
        -------
        str
            Summary of results
        """
        lines = [
            "=" * 60,
            "SCF Results Summary",
            "=" * 60,
            f"Inner SCF converged: {self.converged}",
            f"Inner iterations: {self.iterations}",
            f"Final residual: {self.residual:.6e}",
        ]
        
        if self.outer_iterations is not None:
            lines.extend([
                f"Outer SCF converged: {self.outer_converged}",
                f"Outer iterations: {self.outer_iterations}",
            ])
        
        if self.total_energy is not None:
            lines.append(f"Total energy: {self.total_energy:.10f} Ha")
        
        if self.energy_components is not None:
            lines.append("\nEnergy components:")
            for key, value in self.energy_components.items():
                lines.append(f"  {key}: {value:.10f} Ha")
        
        lines.extend([
            f"Number of states: {len(self.eigenvalues)}",
            f"Lowest eigenvalue: {self.eigenvalues[0]:.6f} Ha",
            "=" * 60
        ])
        
        return "\n".join(lines)



class SCFDriver:
    """
    Self-consistent field driver for Kohn-Sham DFT
    
    Manages both inner and outer SCF loops:
    - Inner loop: standard KS self-consistency (rho → V → H → solve → rho')
    - Outer loop: for methods requiring orbital-dependent potentials (HF, OEP, RPA)
    """
    
    def __init__(
        self,
        hamiltonian_builder : HamiltonianBuilder,
        density_calculator  : DensityCalculator,
        poisson_solver      : PoissonSolver,
        eigensolver         : EigenSolver,
        mixer               : Mixer,
        occupation_info     : OccupationInfo,
        xc_functional       : str,
        hybrid_mixing_parameter : Optional[float] = None
        ):
        """
        Parameters
        ----------
        hamiltonian_builder : HamiltonianBuilder
            Constructs Hamiltonian matrices for each angular momentum channel
        density_calculator : DensityCalculator
            Computes electron density from Kohn-Sham orbitals
        poisson_solver : PoissonSolver
            Solves Poisson equation for Hartree potential
        eigensolver : EigenSolver
            Solves eigenvalue problems (H ψ = ε S ψ)
        mixer : Mixer
            Density mixing strategy for SCF convergence (linear, Pulay, etc.)
        occupation_info : OccupationInfo
            Occupation numbers and quantum numbers for atomic states
        xc_functional : str
            Name of XC functional (e.g., 'LDA_PZ', 'GGA_PBE', 'SCAN')
            Used to determine what density-related quantities to compute
            Also used to initialize the XC calculator internally
        hybrid_mixing_parameter : float, optional
            Mixing parameter for hybrid functionals (HF exchange fraction)
            Required only for hybrid functionals (PBE0, HF)
            - For PBE0: typically 0.25
            - For HF: 1.0
            For non-hybrid functionals, this parameter is ignored
            This parameter is designed to be autodiff-compatible for delta learning
        """
        self.hamiltonian_builder = hamiltonian_builder
        self.density_calculator  = density_calculator
        self.poisson_solver      = poisson_solver
        self.eigensolver         = eigensolver
        self.mixer               = mixer
        self.occupation_info     = occupation_info
        self.xc_functional       = xc_functional
        self.hybrid_mixing_parameter = hybrid_mixing_parameter
        
        # Create SwitchesFlags instance (handles validation internally)
        self.switches = SwitchesFlags(xc_functional, hybrid_mixing_parameter)
        
        # Get functional requirements (what to compute for this functional)
        self.xc_requirements : FunctionalRequirements = get_functional_requirements(xc_functional)
        
        # Initialize XC calculator internally based on functional
        self.xc_calculator : Optional[XCEvaluator] = self._initialize_xc_calculator(
            derivative_matrix=density_calculator.derivative_matrix,
            r_quad=density_calculator.quadrature_nodes
        )
        
        # Initialize HF exchange calculator for hybrid functionals
        self.hf_calculator : Optional[HartreeFockExchange] = self._initialize_hf_calculator()
        
        # Extract NLCC density from pseudopotential (if using pseudopotential)
        self.rho_nlcc : np.ndarray = self._initialize_nlcc_density()
        
        # Initialize convergence checkers (will be configured in run method)
        self.inner_convergence_checker : Optional[ConvergenceChecker] = None
        self.outer_convergence_checker : Optional[ConvergenceChecker] = None

        # type check for required parameters
        assert isinstance(self.hamiltonian_builder, HamiltonianBuilder), \
            HAMILTONIAN_BUILDER_TYPE_ERROR_MESSAGE.format(type(self.hamiltonian_builder))
        assert isinstance(self.density_calculator, DensityCalculator), \
            DENSITY_CALCULATOR_TYPE_ERROR_MESSAGE.format(type(self.density_calculator))
        assert isinstance(self.poisson_solver, PoissonSolver), \
            POISSON_SOLVER_TYPE_ERROR_MESSAGE.format(type(self.poisson_solver))
        assert isinstance(self.eigensolver, EigenSolver), \
            EIGENSOLVER_TYPE_ERROR_MESSAGE.format(type(self.eigensolver))
        assert isinstance(self.mixer, Mixer), \
            MIXER_TYPE_ERROR_MESSAGE.format(type(self.mixer))
        assert isinstance(self.occupation_info, OccupationInfo), \
            OCCUPATION_INFO_TYPE_ERROR_MESSAGE.format(type(self.occupation_info))
    
    
    def _initialize_xc_calculator(
        self, 
        derivative_matrix: np.ndarray,
        r_quad: np.ndarray
        ) -> Optional[XCEvaluator]:
        """
        Initialize XC calculator based on the functional name.
        
        For functional 'None' (pure kinetic energy), no XC calculator is needed.
        For all other functionals, create the appropriate evaluator instance.
        
        Parameters
        ----------
        derivative_matrix : np.ndarray
            Finite element derivative matrix (from DensityCalculator)
            Required for GGA and meta-GGA to transform gradients to spherical form
        r_quad : np.ndarray
            Radial quadrature nodes (coordinates)
            Required for spherical coordinate transformations
        
        Returns
        -------
        xc_calculator : XCEvaluator or None
            Specific XC functional evaluator (e.g., LDA_PZ, GGA_PBE).
            Returns None if xc_functional is 'None'.
        
        Raises
        ------
        ValueError
            If the specified functional is not implemented
        """
        if self.xc_functional in ['None', 'HF']:
            return None
        else:
            return create_xc_evaluator(
                functional_name=self.xc_functional,
                derivative_matrix=derivative_matrix,
                r_quad=r_quad
            )
    
    
    def _initialize_hf_calculator(self) -> Optional[HartreeFockExchange]:
        """
        Initialize Hartree-Fock exchange calculator for hybrid functionals.
        
        Returns
        -------
        Optional[HartreeFockExchange]
            HF exchange calculator if functional requires it, None otherwise
        """
        # Only create HF calculator for hybrid functionals
        if not self.switches.use_hf_exchange:
            return None
        
        # Create HF exchange calculator with ops_builder and occupation_info
        hf_calculator = HartreeFockExchange(
            ops_builder=self.hamiltonian_builder.ops_builder,
            occupation_info=self.occupation_info
        )
        
        return hf_calculator
    
    
    def _initialize_nlcc_density(self) -> np.ndarray:
        """
        Initialize non-linear core correction (NLCC) density from pseudopotential.
        
        NLCC is used to improve the accuracy of exchange-correlation energy
        in pseudopotential calculations by including core electron effects.
        
        Returns
        -------
        rho_nlcc : np.ndarray
            Core correction density at quadrature points.
            Returns zeros for all-electron calculations.
        """
        # Get quadrature nodes from hamiltonian builder
        quadrature_nodes = self.hamiltonian_builder.ops_builder.quadrature_nodes
        
        # Check if using pseudopotential or all-electron
        if self.hamiltonian_builder.all_electron:
            # No NLCC for all-electron calculations
            return np.zeros_like(quadrature_nodes)
        else:
            # Extract NLCC density from pseudopotential
            pseudo = self.hamiltonian_builder.pseudo
            rho_nlcc = pseudo.get_rho_core_correction(quadrature_nodes)
            return rho_nlcc
    
    
    def get_total_density_for_xc(self, rho_valence: np.ndarray) -> np.ndarray:
        """
        Get total density for exchange-correlation calculations.
        
        For pseudopotential calculations, XC functionals should use:
            rho_total = rho_valence + rho_nlcc
        
        For all-electron calculations:
            rho_total = rho_valence
        
        Parameters
        ----------
        rho_valence : np.ndarray
            Valence electron density from KS orbitals
        
        Returns
        -------
        rho_total : np.ndarray
            Total density including NLCC correction
        """
        return rho_valence + self.rho_nlcc
    
    
    def _get_zero_hf_exchange_matrices_dict(self) -> Dict[int, np.ndarray]:
        """
        Create zero HF exchange matrices dictionary for all l channels.
        
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping l values to zero matrices
            Keys are unique l values from occupation_info
            Values are zero matrices of shape (n_physical_nodes, n_physical_nodes)
        """
        zero_matrices_dict = {}
        
        # Get matrix size from kinetic energy matrix
        H_kinetic = self.hamiltonian_builder.H_kinetic
        matrix_size = H_kinetic.shape[0]
        
        # Create zero matrices for all unique l values
        for l in self.occupation_info.unique_l_values:
            zero_matrices_dict[l] = np.zeros((matrix_size, matrix_size))
        
        return zero_matrices_dict
    

    def _compute_hf_exchange_matrices_dict(self, orbitals: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute Hartree-Fock exchange matrices for all l channels.
        
        This method delegates the calculation to the hf_calculator.
        
        Parameters
        ----------
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions) at quadrature points
            Shape: (n_grid, n_orbitals)
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping l values to HF exchange matrices
        """
        if self.hf_calculator is None:
            # Return zero matrices for all l channels
            print(HF_CALCULATOR_NOT_AVAILABLE_WARNING)
            return self._get_zero_hf_exchange_matrices_dict()
        
        # Delegate to hf_calculator
        return self.hf_calculator.compute_exchange_matrices_dict(orbitals)
    


    def run(
        self,
        rho_initial : np.ndarray,
        settings    : Union[SCFSettings, Dict[str, Any]],
        orbitals_initial : Optional[np.ndarray] = None
        ) -> SCFResult:
        """
        Run SCF calculation
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density guess
        settings : SCFSettings or dict
            SCF settings (max iterations, tolerances, etc.)
            Can be a SCFSettings object or a dictionary
        
        Returns
        -------
        result : SCFResult
            SCF solution including eigenvalues, eigenvectors, density, energy
        """
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR_MESSAGE.format(type(rho_initial))
        
        # Convert dict to SCFSettings if needed
        if isinstance(settings, dict):
            settings = SCFSettings.from_dict(settings)
        elif not isinstance(settings, SCFSettings):
            raise TypeError(SETTINGS_TYPE_ERROR_MESSAGE.format(type(settings)))

        # Determine if outer loop is needed
        needs_outer_loop = (settings.outer_max_iter > 1)

        # Configure convergence checkers
        self.inner_convergence_checker = ConvergenceChecker(
            tolerance     = settings.rho_tol,
            n_consecutive = settings.n_consecutive,
            loop_type     = "Inner"
        )
        
        # Initialize outer convergence checker if outer loop is needed
        if needs_outer_loop:
            self.outer_convergence_checker = ConvergenceChecker(
                tolerance     = settings.outer_rho_tol,
                n_consecutive = settings.n_consecutive,  # Outer loop typically converges in 1 iteration
                loop_type     = "Outer"
            )
        
        if hasattr(settings, 'print_debug') and settings.print_debug:
            print("="*60)
            print("\t\t Self-Consistent Field")
            print("="*60)

        if needs_outer_loop:
            return self._outer_loop(rho_initial, settings, orbitals_initial)
        else:
            return self._inner_loop(rho_initial, settings, orbitals_initial)

    
    def _reorder_eigenstates_by_occupation(
        self, 
        eigenvalues_all : List[np.ndarray], 
        eigenvectors_all: List[np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reorder eigenvalues and eigenvectors from l-channel lists to match occupation order.
        
        When solving each l-channel separately, results are grouped by l:
            [all l=0 states] [all l=1 states] [all l=2 states] ...
        
        But occupation list may have interleaved l values:
            [1s(l=0), 2s(l=0), 2p(l=1), 3s(l=0), 3p(l=1)]
        
        This function reorders the results to match the occupation list order.
        
        Parameters
        ----------
        eigenvalues_all : List[np.ndarray]
            List of eigenvalue arrays, one per unique l value
            eigenvalues_all[i] has shape (n_states_for_l[i],)
        eigenvectors_all : List[np.ndarray]
            List of eigenvector arrays, one per unique l value
            eigenvectors_all[i] has shape (n_grid_points, n_states_for_l[i])
        
        Returns
        -------
        eigenvalues : np.ndarray
            Reordered eigenvalues in occupation list order, shape (n_total_states,)
        eigenvectors : np.ndarray
            Reordered eigenvectors in occupation list order, shape (n_grid_points, n_total_states)
        
        Examples
        --------
        For Al (Z=13): occupation = [1s(l=0), 2s(l=0), 2p(l=1), 3s(l=0), 3p(l=1)]
        - Input: eigenvalues_all = [eigvals_l0, eigvals_l1] with l0=[1s,2s,3s], l1=[2p,3p]
        - Output: eigenvalues = [1s, 2s, 2p, 3s, 3p] (correctly interleaved)
        """
        # Preallocate output arrays
        n_total_states = len(self.occupation_info.occupations)
        n_grid_points = eigenvectors_all[0].shape[0]
        eigenvalues = np.zeros(n_total_states)
        eigenvectors = np.zeros((n_grid_points, n_total_states))
        
        # Fill arrays according to occupation list order
        for i_l, l in enumerate(self.occupation_info.unique_l_values):
            # Find all indices in occupation list where l matches
            sort_index = np.where(self.occupation_info.l_values == l)[0]
            n_states = len(sort_index)
            
            # Place eigenstates at correct positions
            eigenvalues[sort_index] = eigenvalues_all[i_l][:n_states]
            eigenvectors[:, sort_index] = eigenvectors_all[i_l][:, :n_states]
        
        return eigenvalues, eigenvectors




    def _inner_loop(
        self,
        rho_initial             : np.ndarray,
        settings                : SCFSettings,
        orbitals_initial        : Optional[np.ndarray]            = None,
        H_hf_exchange_dict_by_l : Optional[Dict[int, np.ndarray]] = None,
        ) -> SCFResult:
        """
        Inner SCF loop: standard Kohn-Sham self-consistency
        
        Fixed: external potential, HF exchange (if any)
        Iterate: rho → V_H, V_xc → H → solve → orbitals → rho'
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density guess
        settings : SCFSettings
            SCF settings
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess for debugging
            Shape: (n_grid, n_orbitals)
            If provided, will be used as initial orbitals instead of solving eigenvalue problem
        H_hf_exchange_dict_by_l : dict, optional
            Hartree-Fock exchange matrices dictionary (from outer loop)
        
        Returns
        -------
        result : SCFResult
            Converged SCF state
        """

        # type check for required fields
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR_MESSAGE.format(type(rho_initial))
        assert isinstance(settings, SCFSettings), \
            SETTINGS_TYPE_ERROR_MESSAGE.format(type(settings))
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, np.ndarray), \
                ORBITALS_INITIAL_TYPE_ERROR_MESSAGE.format(type(orbitals_initial))
        if H_hf_exchange_dict_by_l is not None:
            assert isinstance(H_hf_exchange_dict_by_l, dict), \
                H_HF_EXCHANGE_DICT_BY_L_TYPE_ERROR_MESSAGE.format(type(H_hf_exchange_dict_by_l))
        
        # initialize variables
        max_iter    = settings.inner_max_iter
        print_debug = settings.print_debug

        # rho = rho_initial.copy()

        rho = self.density_calculator.normalize_density(rho_initial.copy())

        density_data = self.density_calculator.create_density_data_from_mixed(
            rho_mixed        = rho,
            orbitals         = orbitals_initial,
            compute_gradient = self.xc_requirements.needs_gradient,
            compute_tau      = self.xc_requirements.needs_tau and (orbitals_initial is not None),
            rho_nlcc         = self.rho_nlcc
        )

        # Reset mixer and convergence checker
        self.mixer.reset()
        self.inner_convergence_checker.reset()

        # Set HF exchange if provided
        if H_hf_exchange_dict_by_l is not None:
            self.hamiltonian_builder.set_hf_exchange_matrices(H_hf_exchange_dict_by_l)
        
        # Print convergence table header
        if print_debug:
            self.inner_convergence_checker.print_header(prefix="")

        # Main inner SCF loop
        for iteration in range(max_iter):
            
            # ===== Step 1: Compute potentials =====
            # Hartree potential
            v_hartree = self.poisson_solver.solve_hartree(rho)

            # XC potential
            # For pseudopotentials: use rho_total = rho_valence + rho_nlcc
            # For all-electron: rho_nlcc is zero, so rho_total = rho_valence
            v_x = np.zeros_like(rho)  # Default: no exchange potential
            v_c = np.zeros_like(rho)  # Default: no correlation potential

            if self.xc_calculator is not None:
                # Compute XC using new interface: DensityData → XCPotentialData
                xc_potential_data = self.xc_calculator.compute_xc(density_data)
                v_x = xc_potential_data.v_x
                v_c = xc_potential_data.v_c
                de_xc_dtau = xc_potential_data.de_xc_dtau
            
            # ===== Step 2: Build and solve for each l channel =====
            eigenvalues_all  = []
            eigenvectors_all = []
            
            for l in self.occupation_info.unique_l_values:
                # Build Hamiltonian for this l
                H_l = self.hamiltonian_builder.build_for_l_channel(
                    l          = l,
                    v_hartree  = v_hartree,
                    v_x        = v_x,
                    v_c        = v_c,
                    switches   = self.switches,
                    de_xc_dtau = de_xc_dtau,
                    symmetrize = False
                )
                                
                S_inv_sqrt = self.hamiltonian_builder.ops_builder.get_S_inv_sqrt()
                H_l = S_inv_sqrt[1:-1,1:-1] @ H_l[1:-1,1:-1] @ S_inv_sqrt[1:-1,1:-1]
                H_l = 0.5 * (H_l + H_l.T)

                # Number of states to solve for this l
                n_states = self.occupation_info.n_states_for_l(l)
                
                # Solve eigenvalue problem
                # eigvals, eigvecs = self.eigensolver.solve_lowest(H_l[1:-1,1:-1], n_states)
                eigvals, eigvecs = self.eigensolver.solve_lowest(H_l, n_states)
                
                eigenvalues_all.append(eigvals)
                eigenvectors_all.append(eigvecs)
            
            # Reorder eigenstates to match occupation list order
            eigenvalues, eigenvectors = self._reorder_eigenstates_by_occupation(
                eigenvalues_all, eigenvectors_all
            )
            
            # Interpolate eigenvectors to quadrature points, also symmetrize the eigenvectors
            orbitals = self.hamiltonian_builder.interpolate_eigenvectors_to_quadrature(
                eigenvectors = eigenvectors,
                symmetrize   = True,
                pad_width    = 1,
            )
            
            # ===== Step 3: Compute new density =====
            # Compute new density from orbitals
            rho_new = self.density_calculator.compute_density(orbitals, normalize=True)
            
            # ===== Step 4: Check convergence =====
            converged, residual = self.inner_convergence_checker.check(
                rho, rho_new, iteration + 1, 
                print_status=print_debug, 
                prefix=""
            )
            
            if converged:
                break
            
            # ===== Step 5: Mix densities and update density_data =====
            rho = self.mixer.mix(rho, rho_new)
            
            # Update density_data for next iteration using mixed density
            density_data = self.density_calculator.create_density_data_from_mixed(
                rho_mixed        = rho,
                orbitals         = orbitals,
                compute_gradient = self.xc_requirements.needs_gradient,
                compute_tau      = self.xc_requirements.needs_tau,
                rho_nlcc         = self.rho_nlcc
            )
        
        # Print convergence footer
        if print_debug:
            self.inner_convergence_checker.print_footer(converged, iteration + 1, prefix="")
        
        if not converged:
            print(INNER_SCF_DID_NOT_CONVERGE_WARNING.format(max_iter=max_iter))

        
        # Create final density_data from converged orbitals, do not include NLCC
        final_density_data : DensityData = self.density_calculator.create_density_data_from_orbitals(
            orbitals         = orbitals,
            compute_gradient = self.xc_requirements.needs_gradient,
            compute_tau      = self.xc_requirements.needs_tau,
            rho_nlcc         = None
        )
        
        # Return final state
        return SCFResult(
            eigen_energies = eigenvalues,
            orbitals       = orbitals,
            density_data   = final_density_data,
            converged      = converged,
            iterations     = iteration + 1,
            rho_residual   = residual
        )


    def _outer_loop(
        self,
        rho_initial      : np.ndarray,
        settings         : SCFSettings,
        orbitals_initial : Optional[np.ndarray] = None
        ) -> SCFResult:
        """
        Outer SCF loop: for orbital-dependent functionals
        
        Used for:
        - Hartree-Fock exchange (requires orbitals)
        - OEP methods (requires full spectrum)
        - RPA correlation (requires response functions)
        
        Parameters
        ----------
        rho_initial : np.ndarray
            Initial density
        settings : SCFSettings
            SCF settings
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess for debugging
            Shape: (n_grid, n_orbitals)
            If provided, will be used as initial orbitals instead of solving eigenvalue problem
        Returns
        -------
        result : SCFResult
            Converged SCF state from outer loop
        """
        # type check for required fields
        assert isinstance(rho_initial, np.ndarray), \
            RHO_INITIAL_TYPE_ERROR_MESSAGE.format(type(rho_initial))
        assert isinstance(settings, SCFSettings), \
            SETTINGS_TYPE_ERROR_MESSAGE.format(type(settings))
        if orbitals_initial is not None:
            assert isinstance(orbitals_initial, np.ndarray), \
                ORBITALS_INITIAL_TYPE_ERROR_MESSAGE.format(type(orbitals_initial))

        # initialize variables
        max_outer_iter = settings.outer_max_iter
        print_debug    = settings.print_debug
        
        rho = rho_initial.copy()
        orbitals = orbitals_initial

        # Compute HF exchange matrices from initial orbitals if provided
        if orbitals_initial is not None:
            H_hf_exchange_dict_by_l = self._compute_hf_exchange_matrices_dict(orbitals_initial)
        else:
            H_hf_exchange_dict_by_l = self._get_zero_hf_exchange_matrices_dict()
        
            
        # Reset outer convergence checker
        self.outer_convergence_checker.reset()
        
        for outer_iter in range(max_outer_iter):
            
            if print_debug:
                print(f"Outer iteration {outer_iter + 1}")
            
            # Run inner SCF with fixed HF exchange
            inner_result : SCFResult = self._inner_loop(
                rho_initial             = rho,
                settings                = settings,
                H_hf_exchange_dict_by_l = H_hf_exchange_dict_by_l
            )
            
            # update rho and orbitals
            rho_new  = inner_result.density_data.rho
            orbitals = inner_result.orbitals

            # update HF exchange dictionary
            H_hf_exchange_dict_by_l = self._compute_hf_exchange_matrices_dict(orbitals) 

            
            # Check outer loop convergence
            outer_converged, outer_residual = self.outer_convergence_checker.check(
                rho, rho_new, outer_iter + 1,
                print_status=print_debug
            )
            
            if outer_converged:
                break
            
            # Update for next outer iteration
            rho = rho_new
        

        # Update outer loop info in result
        outer_iterations = outer_iter + 1
        outer_converged = (outer_iter < max_outer_iter - 1)
        
        # Print outer loop footer if debug enabled
        if print_debug:
            self.outer_convergence_checker.print_footer(outer_converged, outer_iterations)
        
        return SCFResult(
            eigen_energies = inner_result.eigen_energies,
            orbitals       = inner_result.orbitals,
            density_data   = inner_result.density_data,
            converged      = outer_converged,
            iterations     = outer_iterations,
            rho_residual   = outer_residual,
        )
        
