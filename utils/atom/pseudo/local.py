from __future__ import annotations
import numpy as np
import os
from typing import Dict, Any, Optional

from scipy.interpolate import interp1d, make_interp_spline
from .read_pseudopotential import read_pseudopotential_file

__all__ = ["LocalPseudopotential"]

PSEUDOPOTENTIAL_PATH_NOT_PROVIDED_ERROR = \
    "Pseudopotential path must be provided"
PSEUDOPOTENTIAL_FILENAME_NOT_PROVIDED_ERROR = \
    "Pseudopotential filename must be provided"
PSEUDOPOTENTIAL_PATH_DOES_NOT_EXIST_ERROR = \
    "Pseudopotential path {} does not exist"
PSEUDOPOTENTIAL_FILENAME_DOES_NOT_EXIST_ERROR = \
    "Pseudopotential filename {} does not exist"
R_NODES_MUST_BE_MONOTONICALLY_INCREASING_AND_NON_NEGATIVE_ERROR = \
    "r_nodes must be monotonically increasing and non-negative"

class LocalPseudopotential:
    """
    Local pseudopotential reader and interpolator.
    
    This class handles loading and interpolation of local pseudopotential data
    from psp8 format files. It provides methods to evaluate the local potential
    on arbitrary radial grids.
    """

    def __init__(self, 
        atomic_number: int,
        path: Optional[str] = None, 
        filename: Optional[str] = None):
        """
        Initialize local pseudopotential.
        
        Parameters
        ----------
        atomic_number : int
            Atomic number of the element
        path : str, optional
            Path to pseudopotential files directory
        filename : str, optional
            Pseudopotential filename (e.g., 'H.psp8')
        """

        # Initialize attributes
        if path is not None and filename is not None:
            self.all_electron = False
            self.load(path, filename)
            self.u_local = 1.0
            self.u_non_local = 1.0
            self.load_psp = True
        else:
            self.all_electron = True
            self.z_valence = atomic_number
            self.z_nuclear = atomic_number
            self.u_local = 0.0
            self.u_non_local = 0.0
            self.load_psp = False


    @staticmethod
    def thomas_fermi_density_guess(z_valence: float, r_nodes: np.ndarray) -> np.ndarray:
        """
        Thomas-Fermi approximation for all-electron density guess
        Based on screening model with empirical fitting parameters
        """
        # Compute scaled radial coordinate for Thomas-Fermi model
        # x = r * k_TF where k_TF is the Thomas-Fermi wave vector
        thomas_fermi_prefactor = (128 * z_valence / (9 * np.pi**2))**(1/3)
        r_scaled = r_nodes * thomas_fermi_prefactor
        
        # Empirical fitting parameters for effective charge screening
        # From quantum Monte Carlo or DFT fitting
        screening_alpha = 0.7280642371   # Primary screening parameter
        screening_beta = -0.5430794693   # Secondary screening parameter  
        screening_gamma = 0.3612163121   # Decay parameter
        
        # Compute effective nuclear charge with screening corrections
        # Z_eff accounts for electron-electron screening effects
        sqrt_r_scaled = np.sqrt(r_scaled)
        screening_correction = (
            1 + screening_alpha * sqrt_r_scaled 
            + screening_beta * r_scaled * np.exp(-screening_gamma * sqrt_r_scaled)
        )
        z_effective = z_valence * screening_correction**2 * np.exp(-2 * screening_alpha * sqrt_r_scaled)
        
        # Compute initial potential: V(r) = -Z_eff / r
        v_coulomb_screened = -z_effective / r_nodes
        
        # Thomas-Fermi density: rho(r) = (1/(3π²)) * [2|V(r)|]^(3/2)
        # This comes from the semiclassical approximation
        thomas_fermi_constant = 1 / (3 * np.pi**2)
        rho_thomas_fermi = thomas_fermi_constant * ((-2 * v_coulomb_screened)**(3/2))
        
        return rho_thomas_fermi


    def get_rho_guess(self, r_nodes: np.ndarray) -> np.ndarray:
        """
        Get the initial density guess for the SCF.

        Parameters
        ----------
        r_nodes : np.ndarray
            Radial nodes where to evaluate the density
            
        Returns
        -------
        np.ndarray
            Density guess at the nodes
        """
        # check if r_nodes is monotonically increasing and non-negative
        if not np.all(np.diff(r_nodes) > 0.0) or not np.all(r_nodes > 0.0):
            raise ValueError(R_NODES_MUST_BE_MONOTONICALLY_INCREASING_AND_NON_NEGATIVE_ERROR)
        
        if self.all_electron:
            return self.thomas_fermi_density_guess(self.z_valence, r_nodes)
        else:
            # Interpolate the density guess from pseudopotential data
            # For r < r_cutoff: use cubic spline interpolation
            # For r >= r_cutoff: use exponential extrapolation based on tail behavior
            rho_grid_cutoff = np.max(self.r_grid_density)
            
            # Get values at last two grid points for exponential extrapolation
            rho_last = self.density_initial_guess[-1]
            rho_second_last = self.density_initial_guess[-2]
            r_last = self.r_grid_density[-1]
            r_second_last = self.r_grid_density[-2]
            
            # Fit exponential tail: rho(r) = prefactor * exp(decay_rate * r)
            # Using last two points to determine decay rate and prefactor
            decay_rate = np.log(rho_last / rho_second_last) / (r_last - r_second_last)
            prefactor = rho_last / np.exp(decay_rate * r_last)
            
            # Split nodes into inner and outer regions
            indices_below_cutoff = np.where(r_nodes < rho_grid_cutoff)
            indices_above_cutoff = np.where(r_nodes >= rho_grid_cutoff)
            r_below_cutoff = r_nodes[indices_below_cutoff[0]]
            
            # Inner region: cubic spline interpolation
            rho_interpolator = make_interp_spline(self.r_grid_density, self.density_initial_guess, k=3)
            rho_interpolated = np.zeros_like(r_nodes)
            rho_interpolated[indices_below_cutoff[0]] = rho_interpolator(r_below_cutoff)
            
            # Outer region: exponential extrapolation
            rho_extrapolated = np.zeros_like(r_nodes)
            rho_extrapolated[indices_above_cutoff[0]] = prefactor * np.exp(decay_rate * r_nodes[indices_above_cutoff[0]])
            
            # Combine interpolated and extrapolated regions
            rho_guess = rho_interpolated + rho_extrapolated
            return rho_guess
        

    def get_rho_core_correction(self, r_nodes: np.ndarray) -> np.ndarray:
        """
        Get the nonlinear core correction (NLCC) density for the SCF, evaluated on the r_nodes.
        """
        # check if r_nodes is monotonically increasing and non-negative
        if not np.all(np.diff(r_nodes) > 0.0) or not np.all(r_nodes > 0.0):
            raise ValueError(R_NODES_MUST_BE_MONOTONICALLY_INCREASING_AND_NON_NEGATIVE_ERROR)
        
        if self.all_electron:
            # NLCC is not used in all-electron calculations
            return np.zeros_like(r_nodes)
        else:
            # Interpolate the nonlinear core correction (NLCC) density
            # For r < r_cutoff: use cubic spline interpolation
            # For r >= r_cutoff: density is zero (core correction vanishes)
            
            nlcc_grid_cutoff = np.max(self.r_grid_nlcc)
            
            # Find nodes within the NLCC grid range
            indices_below_cutoff = np.where(r_nodes < nlcc_grid_cutoff)
            r_below_cutoff = r_nodes[indices_below_cutoff[0]]
            
            # Initialize NLCC density array (zero outside cutoff)
            rho_nlcc = np.zeros_like(r_nodes)
            
            # Interpolate NLCC density using cubic spline
            rho_nlcc_interpolator = make_interp_spline(
                self.r_grid_nlcc, 
                self.density_nlcc, 
                k=3
            )
            rho_nlcc[indices_below_cutoff[0]] = rho_nlcc_interpolator(r_below_cutoff).reshape(-1)
            
            return rho_nlcc
 



    def get_v_local_component_psp(self, r_nodes: np.ndarray) -> np.ndarray:
        """
        Get the local potential component of the pseudopotential, evaluated on the r_nodes.
        
        For r < r_cutoff: interpolates from the pseudopotential data
        For r >= r_cutoff: uses the Coulomb tail -Z/r
        
        Parameters
        ----------
        r_nodes : np.ndarray
            Radial grid points where to evaluate the local potential
            
        Returns
        -------
        np.ndarray
            Local potential values at the given radial nodes
        """
        # Validate input: r_nodes must be monotonically increasing and non-negative
        if not np.all(np.diff(r_nodes) > 0.0) or not np.all(r_nodes > 0.0):
            raise ValueError(R_NODES_MUST_BE_MONOTONICALLY_INCREASING_AND_NON_NEGATIVE_ERROR)

        if self.all_electron:
            # Local potential component of the pseudopotential is not used in all-electron calculations
            return np.zeros_like(r_nodes)
        else:
            # Interpolate the local potential component of the pseudopotential
            r_cutoff = self.r_grid_local[-1]
            
            # Split nodes into inner region (r < r_cutoff) and outer region (r >= r_cutoff)
            indices_below_cutoff = np.where(r_nodes < r_cutoff)
            indices_above_cutoff = np.where(r_nodes >= r_cutoff)
            r_below_cutoff = r_nodes[indices_below_cutoff[0]]
            
            # Initialize local potential array
            v_local = np.zeros_like(r_nodes)
            
            # Outer region: use Coulomb tail -Z/r
            v_local[indices_above_cutoff[0]] = -self.z_valence
            
            # Inner region: interpolate from pseudopotential grid
            # Multiply by r to get r*V(r) for interpolation
            r_times_v_local = self.r_grid_local * self.v_local_values
            v_local_interpolator = interp1d(
                self.r_grid_local, 
                r_times_v_local, 
                kind='cubic', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            v_local[indices_below_cutoff[0]] = v_local_interpolator(r_below_cutoff)
            
            # Divide by r to get V(r) = (r*V(r))/r
            return v_local / r_nodes


    def load(self, path: str, filename: str) -> Dict[str, Any]:
        """
        Load pseudopotential data from file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing loaded pseudopotential information
        """
        assert path is not None, \
            PSEUDOPOTENTIAL_PATH_NOT_PROVIDED_ERROR
        assert filename is not None, \
            PSEUDOPOTENTIAL_FILENAME_NOT_PROVIDED_ERROR
        assert os.path.exists(path), \
            PSEUDOPOTENTIAL_PATH_DOES_NOT_EXIST_ERROR.format(path)
        assert os.path.exists(os.path.join(path, filename)), \
            PSEUDOPOTENTIAL_FILENAME_DOES_NOT_EXIST_ERROR.format(filename)
        
        # Read pseudopotential file
        psp_data = read_pseudopotential_file(
            psp_dir_path  = path,
            psp_file_name = filename,
            print_debug   = False
        )
        
        # Unpack pseudopotential data (keep original variable names for reference)
        (Z, Zatom, XC, Vloc, r_grid_vloc, rc, Pot, lmax, lloc, nproj,
         r_grid_rho, rho_isolated_guess, rho_tilde, r_grid_rho_tilde,
         pspsoc, Potso, rc_max_list) = psp_data
        
        # Store atomic charge information
        self.z_valence          = Z                  # Z: Valence charge (for pseudopotential Coulomb tail)
        self.z_nuclear          = Zatom              # Zatom: True nuclear charge of the atom
        self.xc_functional      = XC                 # XC: Exchange-correlation functional identifier

        # Store local pseudopotential data
        self.v_local_values     = Vloc               # Vloc: Local pseudopotential values on radial grid
        self.r_grid_local       = r_grid_vloc        # r_grid_vloc: Radial grid points for local potential
        self.r_core_cutoff      = rc                 # rc: Core radius cutoff for pseudopotential

        # Store non-local pseudopotential data
        self.nonlocal_projectors = Pot               # Pot: Non-local projector functions for each angular momentum l
        self.n_projectors_per_l = nproj              # nproj: Number of projectors for each l channel

        # Store density-related data
        self.r_grid_density     = r_grid_rho         # r_grid_rho: Radial grid for density guess
        self.density_initial_guess = rho_isolated_guess # rho_isolated_guess: Initial atomic density for SCF guess
        
        # Store nonlinear core correction (NLCC) data
        self.density_nlcc       = rho_tilde          # rho_tilde: Core electron density for NLCC
        self.r_grid_nlcc        = r_grid_rho_tilde   # r_grid_rho_tilde: Radial grid for NLCC density
        
        # Store spin-orbit coupling data
        self.has_spin_orbit     = pspsoc             # pspsoc: Spin-orbit coupling flag (0=no, 1=yes)
        self.spin_orbit_projectors = Potso           # Potso: Spin-orbit coupling projector functions
        
        # Store cutoff radii information
        self.r_cutoff_max_per_l = rc_max_list        # rc_max_list: Maximum cutoff radius for each l channel
        

    def print_info(self):
        """
        Print pseudopotential information summary.
        """
        print("=" * 60)
        print("\t\t PSEUDOPOTENTIAL INFORMATION")
        print("=" * 60)



        # Basic atomic information
        print(f"  Valence Charge             (z_valence)             : {self.z_valence}")
        print(f"  Nuclear Charge             (z_nuclear)             : {self.z_nuclear}")
        print()

        if self.load_psp:
            # Local potential information
            print("LOCAL POTENTIAL:")
            print(f"  Core Cutoff Radius         (r_core_cutoff)         : {self.r_core_cutoff:.6f} Bohr")
            print(f"  Grid Points                (r_grid_local)          : {len(self.r_grid_local)}")
            print(f"  Grid Range                 (r_grid_local)          : [{self.r_grid_local[0]:.6f}, {self.r_grid_local[-1]:.6f}] Bohr")
            print(f"  Potential Range            (v_local_values)        : [{np.min(self.v_local_values):.6f}, {np.max(self.v_local_values):.6f}] Hartree")
            print()
        
        # Non-local potential information
        if hasattr(self, 'nonlocal_projectors') and len(self.nonlocal_projectors) > 0:
            print("NON-LOCAL POTENTIAL:")
            print(f"  Maximum Angular Momentum   (l_max)                 : {len(self.nonlocal_projectors) - 1}")
            print(f"  Projectors per l           (n_projectors_per_l)    : {self.n_projectors_per_l}")
            print(f"  Spin-Orbit Coupling        (has_spin_orbit)        : {'Yes' if self.has_spin_orbit else 'No'}")
            print()
            
            # Projector information for each l
            for l in range(len(self.nonlocal_projectors)):
                if l < len(self.n_projectors_per_l):
                    n_proj_l = self.n_projectors_per_l[l]
                    print(f"  l = {l}: {n_proj_l} projector(s)")
                    if l < len(self.r_cutoff_max_per_l):
                        print(f"    Max cutoff radius: {self.r_cutoff_max_per_l[l]:.6f} Bohr")
            print()
        
        # Density information
        if hasattr(self, 'r_grid_density') and len(self.r_grid_density) > 0:
            print("DENSITY INFORMATION:")
            print(f"  Density Grid Points        (r_grid_density)        : {len(self.r_grid_density)}")
            print(f"  Density Grid Range         (r_grid_density)        : [{self.r_grid_density[0]:.6f}, {self.r_grid_density[-1]:.6f}] Bohr")
            if hasattr(self, 'density_initial_guess') and len(self.density_initial_guess) > 0:
                print(f"  Initial Density Range      (density_initial_guess) : [{np.min(self.density_initial_guess):.6f}, {np.max(self.density_initial_guess):.6f}]")
            if hasattr(self, 'density_nlcc') and len(self.density_nlcc) > 0:
                print(f"  NLCC Density Range         (density_nlcc)          : [{np.min(self.density_nlcc):.6f}, {np.max(self.density_nlcc):.6f}]")
        
            print()


    def evaluate_on(self, r_quad: np.ndarray) -> np.ndarray:
        """
        Evaluate local potential on quadrature grid.
        
        Parameters
        ----------
        r_quad : np.ndarray
            Radial quadrature points where to evaluate the potential
            
        Returns
        -------
        np.ndarray
            Local potential values at quadrature points
        """
        raise NotImplementedError("check again")

        if len(self.v_local_values) == 0:
            raise ValueError("Pseudopotential data not loaded. Call load() first.")
        
        # Handle edge cases
        if len(r_quad) == 0:
            return np.array([])
        
        # Ensure r_quad is within the pseudopotential grid range
        r_min = np.min(self.r_grid_local)
        r_max = np.max(self.r_grid_local)
        
        # Clamp r_quad to valid range
        r_quad_clamped = np.clip(r_quad, r_min, r_max)
        
        # Interpolate local potential
        try:
            # Use cubic spline interpolation for smooth results
            interp_func = interp1d(
                self.r_grid_local, 
                self.v_local_values, 
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            V_local_interp = interp_func(r_quad_clamped)
            
        except ValueError:
            # Fallback to linear interpolation if cubic fails
            interp_func = interp1d(
                self.r_grid_local, 
                self.v_local_values, 
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            V_local_interp = interp_func(r_quad_clamped)
        
        return V_local_interp
