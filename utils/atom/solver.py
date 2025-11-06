"""
Atomic Density Functional Theory (DFT) Solver

    This module provides a comprehensive implementation of Atomic DFT solver using
    finite element method for solving the Kohn-Sham equations for atomic systems.

    The solver supports:
    - Multiple exchange-correlation functionals (LDA, GGA, Meta-GGA, Hybrid, etc.)
    - Both all-electron and pseudopotential calculations
    - Self-consistent field (SCF) iterations with convergence control
    - High-order finite element discretization with Legendre-Gauss-Lobatto nodes
    - Various mesh types (exponential, polynomial, uniform)

    @file    solver.py
    @brief   Atomic DFT Solver using finite element method
    @authors Shubhang Trivedi <strivedi44@gatech.edu>
             Qihao Cheng <qcheng61@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2025 Material Physics & Mechanics Group, Georgia Tech.
"""


from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix the relative import issue when running as a script
try:
    __package__
except NameError:
    __package__ = None

if __package__ is None:
    # Set the package name, so the relative import can work
    __package__ = 'atom'
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import numpy as np
from typing import Optional, Dict, Any, Tuple

# Mesh & operators
from .mesh.builder import Quadrature1D, Mesh1D, RPAFrequencyGrid
from .mesh.operators import GridData, RadialOperatorsBuilder

# Typing imports
from .pseudo.local import LocalPseudopotential
from .pseudo.non_local import NonLocalPseudopotential
from .utils.occupation_states import OccupationInfo
from .scf.energy import EnergyComponents
from .scf.driver import SCFResult
from .xc.evaluator import XCPotentialData
from .xc.functional_requirements import get_functional_requirements

# SCF stack
from .scf import (
    HamiltonianBuilder,
    DensityCalculator,
    PoissonSolver,
    SCFDriver,
    EnergyCalculator,
    EigenSolver,
    Mixer,
    ConvergenceChecker
)

# XC and snapshot
from .xc.evaluator import XCEvaluator
from .post.builder import SnapshotBuilder



# Valid XC Functional
VALID_XC_FUNCTIONAL_LIST = ['GGA_PBE', 'RPA', 'OEPx', 'LDA_PZ', 'LDA_PW', 'SCAN', 'RSCAN', 'R2SCAN', 'PBE0', 'HF']

# Valid Mesh Type
VALID_MESH_TYPE_LIST = ['exponential', 'polynomial', 'uniform']



# Type Check Error Messages
ATOMIC_NUMBER_NOT_INTEGER_ERROR = \
    "parameter atomic_number must be an integer, get {} instead"
DOMAIN_SIZE_NOT_FLOAT_ERROR = \
    "parameter domain_size must be a float, get {} instead"
NUMBER_OF_FINITE_ELEMENTS_NOT_INTEGER_ERROR = \
    "parameter number_of_finite_elements must be an integer, get {} instead"
POLYNOMIAL_ORDER_NOT_INTEGER_ERROR = \
    "parameter polynomial_order must be an integer, get {} instead"
QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "parameter quadrature_point_number must be an integer, get {} instead"
XC_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter xc_functional must be a string, get {} instead"
MESH_TYPE_NOT_STRING_ERROR = \
    "parameter mesh_type must be a string, get {} instead"
MESH_CONCENTRATION_NOT_FLOAT_ERROR = \
    "parameter mesh_concentration must be a float, get {} instead"
SCF_TOLERANCE_NOT_FLOAT_ERROR = \
    "parameter scf_tolerance must be a float, get {} instead"
ALL_ELECTRON_FLAG_NOT_BOOL_ERROR = \
    "parameter all_electron_flag must be a boolean, get {} instead"
ENABLE_DOMAIN_SIZE_TEST_NOT_BOOL_ERROR = \
    "parameter enable_domain_size_test must be a boolean, get {} instead"
PSP_DIR_PATH_NOT_STRING_ERROR = \
    "parameter psp_dir_path must be a string, get {} instead"
PSP_FILE_NAME_NOT_STRING_ERROR = \
    "parameter psp_file_name must be a string, get {} instead"
FREQUENCY_INTEGRATION_POINT_NUMBER_NOT_INTEGER_ERROR = \
    "parameter frequency_integration_point_number must be an integer, get {} instead"
EIGENTOLERANCE_X_NOT_FLOAT_ERROR = \
    "parameter eigentolerance_x must be a float, get {} instead"
EIGENTOLERANCE_C_NOT_FLOAT_ERROR = \
    "parameter eigentolerance_c must be a float, get {} instead"
L_MAX_QUANTUM_NUMBER_NOT_INTEGER_ERROR = \
    "parameter l_max_quantum_number must be an integer, get {} instead"
SMOOTHENING_CUTOFF_FREQUENCY_NOT_FLOAT_ERROR = \
    "parameter smoothening_cutoff_frequency must be a float, get {} instead"
DOUBLE_HYBRID_FLAG_NOT_BOOL_ERROR = \
    "parameter double_hybrid_flag must be a boolean, get {} instead"
HYBRID_MIXING_PARAMETER_NOT_FLOAT_ERROR = \
    "parameter hybrid_mixing_parameter must be a float, get {} instead"
MESH_SPACING_NOT_FLOAT_ERROR = \
    "parameter mesh_spacing must be a float, get {} instead"
PRINT_DEBUG_NOT_BOOL_ERROR = \
    "parameter print_debug must be a boolean, get {} instead"

# Value Check Error Messages
ATOMIC_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter atomic_number must be greater than 0, get {} instead"
ATOMIC_NUMBER_LARGER_THAN_119_ERROR = \
    "parameter atomic_number must be smaller than 119, get {} instead"
DOMAIN_SIZE_NOT_GREATER_THAN_0_ERROR = \
    "parameter domain_size must be greater than 0, get {} instead"
NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR = \
    "parameter number_of_finite_elements must be greater than 0, get {} instead"
POLYNOMIAL_ORDER_NOT_GREATER_THAN_0_ERROR = \
    "parameter polynomial_order must be greater than 0, get {} instead"

QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter quadrature_point_number must be greater than 0, get {} instead"
FREQUENCY_INTEGRATION_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR = \
    "parameter frequency_integration_point_number must be greater than 0, get {} instead"
L_MAX_QUANTUM_NUMBER_NEGATIVE_ERROR = \
    "parameter l_max_quantum_number must be non-negative, get {} instead"
XC_FUNCTIONAL_TYPE_ERROR_MESSAGE = \
    "parameter xc_functional must be a string, get type {} instead"
XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR = \
    "parameter xc_functional must be in {}, get {} instead"
MESH_TYPE_NOT_IN_VALID_LIST_ERROR = \
    "parameter mesh_type must be in {}, get {} instead"
MESH_CONCENTRATION_NOT_GREATER_THAN_0_ERROR = \
    "parameter mesh_concentration must be greater than 0, get {} instead"
SCF_TOLERANCE_NOT_GREATER_THAN_0_ERROR = \
    "parameter scf_tolerance must be greater than 0, get {} instead"
PSP_DIR_PATH_NOT_EXISTS_ERROR = \
    "parameter default psp directory path {} does not exist, please provide a valid psp directory path"
PSP_FILE_NAME_NOT_EXISTS_ERROR = \
    "parameter psp file name `{}` does not exist in the psp file path `{}`, please provide a valid psp file name"
EIGENTOLERANCE_X_NOT_GREATER_THAN_0_ERROR = \
    "parameter eigentolerance_x must be greater than 0, get {} instead"
EIGENTOLERANCE_C_NOT_GREATER_THAN_0_ERROR = \
    "parameter eigentolerance_c must be greater than 0, get {} instead"
SMOOTHENING_CUTOFF_FREQUENCY_NEGATIVE_ERROR = \
    "parameter smoothening_cutoff_frequency must be non-negative, get {} instead"
HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter hybrid_mixing_parameter must be in [0, 1], get {} instead"
HYBRID_MIXING_PARAMETER_NOT_ONE_FOR_NON_HYBRID_FUNCTIONAL_ERROR = \
    "parameter hybrid_mixing_parameter must be 1.0 for non-hybrid functional, get {} instead"
DENSITY_MIXING_PARAMETER_NOT_FLOAT_ERROR = \
    "parameter density_mixing_parameter must be a float, get {} instead"
DENSITY_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR = \
    "parameter density_mixing_parameter must be in [0, 1], get {} instead"
MESH_SPACING_NOT_GREATER_THAN_0_ERROR = \
    "parameter mesh_spacing must be greater than 0, get {} instead"

# WARNING Messages
MESH_CONCENTRATION_NOT_NONE_FOR_UNIFORM_MESH_TYPE_WARNING = \
    "WARNING: parameter mesh_concentration is not None for uniform mesh type, so it will be ignored"
PSP_DIR_PATH_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING = \
    "WARNING: parameter psp_dir_path is not None for all-electron calculation, so it will be ignored"
PSP_FILE_NAME_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING = \
    "WARNING: parameter psp_file_name is not None for all-electron calculation, so it will be ignored"
FREQUENCY_INTEGRATION_POINT_NUMBER_NOT_NONE_FOR_OEPX_AND_NONE_XC_FUNCTIONAL_WARNING = \
    "WARNING: parameter frequency_integration_point_number is not None for XC functional `{}`, so it will be ignored"
EIGENTOLERANCE_X_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING = \
    "WARNING: parameter eigentolerance_x is not None for XC functional `{}`, so it will be ignored"
EIGENTOLERANCE_C_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING = \
    "WARNING: parameter eigentolerance_c is not None for XC functional `{}`, so it will be ignored"
SMOOTHENING_CUTOFF_FREQUENCY_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING = \
    "WARNING: parameter smoothening_cutoff_frequency is not None for XC functional `{}`, so it will be ignored"
NO_HYBRID_MIXING_PARAMETER_PROVIDED_FOR_HYBRID_FUNCTIONAL_WARNING = \
    "WARNING: hybrid_mixing_parameter not provided for {} functional, using default value {}"
HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_WARNING = \
    "WARNING: hybrid_mixing_parameter for {} should be in [0, 1], got {}"
HYBRID_MIXING_PARAMETER_NOT_ONE_FOR_NON_HYBRID_FUNCTIONAL_WARNING = \
    "WARNING: hybrid_mixing_parameter for {} must be 1.0 for non-hybrid functional, got {}"
HYBRID_MIXING_PARAMETER_NOT_FLOAT_WARNING = \
    "WARNING: hybrid_mixing_parameter for {} must be a float, got {}"
HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_WARNING = \
    "WARNING: hybrid_mixing_parameter for {} must be in [0, 1], got {}"
HYBRID_MIXING_PARAMETER_NOT_ONE_FOR_HF_ERROR = \
    "WARNING: hybrid_mixing_parameter for {} must be 1.0 for HF functional, got {}"
WARM_START_NOT_CONVERGED_WARNING = \
    "WARNING: warm start calculation for {} did not converge, using intermediate result"

class AtomicDFTSolver:
    """
    Atomic Density Functional Theory (DFT) Solver using finite element method.
    
    This class provides a comprehensive interface for solving the Kohn-Sham equations
    for atomic systems using various exchange-correlation functionals and computational
    parameters. It supports both all-electron and pseudopotential calculations.
    """

    # Basic physical parameters
    atomic_number                      : int   # Atomic number of the element to calculate (e.g., 13 for Aluminum)
    domain_size                        : float # Radial computational domain size in atomic units (typically 10-30)
    number_of_finite_elements          : int   # Number of finite elements in the computational domain
    polynomial_order                   : int   # Polynomial order of basis functions within each finite element
    quadrature_point_number            : int   # Number of quadrature points for numerical integration (recommended: 3-4x polynomial_order)
    
    # Exchange-correlation functional parameters
    xc_functional                      : str   # XC functional type: 'GGA_PBE', 'RPA', 'OEPx', 'LDA_PZ', 'LDA_PW', 'SCAN', 'RSCAN', 'R2SCAN'
    mesh_type                          : str   # Mesh distribution type: 'exponential' (higher density near nucleus), 'polynomial', or 'uniform'
    mesh_concentration                 : float # Mesh concentration parameter (controls point density distribution)
    
    # Self-consistent field (SCF) convergence parameters
    scf_tolerance                      : float # SCF convergence tolerance (typically 1e-8)
    all_electron_flag                  : bool  # True for all-electron calculation, False for pseudopotential calculation
    enable_domain_size_test            : bool  # Flag for domain size convergence testing
    
    # Pseudopotential parameters
    psp_dir_path                       : str   # Path to pseudopotential files directory (required when all_electron_flag=False)
    psp_file_name                      : str   # Name of the pseudopotential file (required when all_electron_flag=False)
    
    # Advanced functional parameters (for OEPx, RPA, etc.)
    frequency_integration_point_number : int   # Number of frequency integration points for RPA calculations
    eigentolerance_x                   : float # Eigenvalue convergence tolerance for exchange term
    eigentolerance_c                   : float # Eigenvalue convergence tolerance for correlation term
    l_max_quantum_number               : int   # Maximum angular momentum quantum number to include
    smoothening_cutoff_frequency       : float # Smoothing cutoff frequency for numerical stability
    double_hybrid_flag                 : bool  # Flag for double-hybrid functional methods
    hybrid_mixing_parameter            : float # Mixing parameter for hybrid/double-hybrid functionals (e.g., 0.25 for PBE0)
    
    # Grid and computational parameters
    mesh_spacing                       : float # Minimum mesh spacing (should match output file spacing)
    density_mixing_parameter           : float # Density mixing parameter for SCF convergence (alpha in linear mixing)
    print_debug                        : bool  # Flag for printing debug information during calculation



    def __init__(self, 
        atomic_number                      : int,  # Only atomic_number is required, all other parameters have default values
        domain_size                        : Optional[float] = None,   # 20.0 by default
        number_of_finite_elements          : Optional[int]   = None,   # 17 by default
        polynomial_order                   : Optional[int]   = None,   # 31 by default
        quadrature_point_number            : Optional[int]   = None,   # 95 by default
        xc_functional                      : Optional[str]   = None,   # 'GGA_PBE' by default
        mesh_type                          : Optional[str]   = None,   # 'exponential' by default
        mesh_concentration                 : Optional[float] = None,   # 61.0 by default
        scf_tolerance                      : Optional[float] = None,   # 1e-8 by default
        all_electron_flag                  : Optional[bool]  = None,   # False by default
        enable_domain_size_test            : Optional[bool]  = None,   # False by default
        psp_dir_path                       : Optional[str]   = None,   # ../psps by default
        psp_file_name                      : Optional[str]   = None,   # {atomic_number}.psp8 by default
        frequency_integration_point_number : Optional[int]   = None,   # if xc_functional is 'RPA', 1200 by default
        eigentolerance_x                   : Optional[float] = None,   # for RPA, 1e-9 by default; for OEPx, 1e-11 by default, otherwise not needed
        eigentolerance_c                   : Optional[float] = None,   # for RPA, 1e-9 by default; for OEPx, 1e-09 by default, otherwise not needed
        l_max_quantum_number               : Optional[int]   = None,   # for RPA, 4 by default; for OEPx, 8 by default, otherwise 0 by default
        smoothening_cutoff_frequency       : Optional[float] = None,   # for xc other than RPA, not needed; otherwise 0.0 for most atoms
        double_hybrid_flag                 : Optional[bool]  = None,   # False by default
        hybrid_mixing_parameter            : Optional[float] = None,   # 1.0 by default (0.25 for PBE0, variable for RPA)
        mesh_spacing                       : Optional[float] = None,   # 0.1 by default
        density_mixing_parameter           : Optional[float] = None,   # 0.5 by default (alpha in linear mixing)
        print_debug                        : Optional[bool]  = None,   # False by default
        ):   

        """
        Initialize the AtomicDFTSolver with computational parameters.
        
        Args:
            atomic_number                      (int)  : Atomic number of the element (e.g., 13 for Aluminum)
            domain_size                        (float): Radial domain size in atomic units (typically 10-30)
            number_of_finite_elements          (int)  : Number of finite elements in the domain
            polynomial_order                   (int)  : Polynomial order of basis functions (typically 20-40)
            quadrature_point_number            (int)  : Quadrature points for integration (3-4x polynomial_order)
            xc_functional                      (str)  : Exchange-correlation functional ('GGA_PBE', 'RPA', 'OEPx', etc.)
            mesh_type                          (str)  : Mesh type ('exponential', 'polynomial', 'uniform')
            mesh_concentration                 (float): Mesh concentration parameter (controls point density)
            scf_tolerance                      (float): SCF convergence tolerance (typically 1e-8)
            all_electron_flag                  (bool) : True for all-electron, False for pseudopotential
            enable_domain_size_test            (bool) : Enable domain size convergence testing
            psp_dir_path                       (str)  : Path to pseudopotential directory (required if all_electron_flag=False)
            psp_file_name                      (str)  : Name of pseudopotential file (required if all_electron_flag=False)
            frequency_integration_point_number (int)  : Frequency points for RPA calculations
            eigentolerance_x                   (float): Exchange eigenvalue convergence tolerance
            eigentolerance_c                   (float): Correlation eigenvalue convergence tolerance
            l_max_quantum_number               (int)  : Maximum angular momentum quantum number
            smoothening_cutoff_frequency       (float): Smoothing cutoff for numerical stability
            double_hybrid_flag                 (bool) : Enable double-hybrid functional methods
            hybrid_mixing_parameter            (float): Mixing parameter for hybrid functionals (e.g., 0.25 for PBE0)
            mesh_spacing                       (float): Minimum mesh spacing (should match output file)
            density_mixing_parameter           (float): Density mixing parameter for SCF (alpha in linear mixing)
            print_debug                        (bool) : Enable debug output
            
        Raises:
            ValueError: If any parameter has incorrect type
            
        Note:
            The solver uses finite element method with Legendre-Gauss-Lobatto nodes
            for high-order accuracy in solving the Kohn-Sham equations.
        """

        # Initialize the class attributes
        self.atomic_number                      = atomic_number
        self.domain_size                        = domain_size
        self.number_of_finite_elements          = number_of_finite_elements
        self.polynomial_order                   = polynomial_order
        self.quadrature_point_number            = quadrature_point_number
        self.xc_functional                      = xc_functional
        self.mesh_type                          = mesh_type
        self.mesh_concentration                 = mesh_concentration
        self.scf_tolerance                      = scf_tolerance
        self.all_electron_flag                  = all_electron_flag
        self.enable_domain_size_test            = enable_domain_size_test
        self.psp_dir_path                       = psp_dir_path
        self.psp_file_name                      = psp_file_name
        self.frequency_integration_point_number = frequency_integration_point_number
        self.eigentolerance_x                   = eigentolerance_x
        self.eigentolerance_c                   = eigentolerance_c
        self.l_max_quantum_number               = l_max_quantum_number
        self.smoothening_cutoff_frequency       = smoothening_cutoff_frequency
        self.double_hybrid_flag                 = double_hybrid_flag
        self.hybrid_mixing_parameter            = hybrid_mixing_parameter
        self.mesh_spacing                       = mesh_spacing
        self.density_mixing_parameter           = density_mixing_parameter
        self.print_debug                        = print_debug


        # set the default parameters, if not provided
        self.set_and_check_initial_parameters()

        # initialize the psuedopotential data
        self.pseudo = LocalPseudopotential(
            atomic_number = self.atomic_number, 
            path          = self.psp_dir_path, 
            filename      = self.psp_file_name)            

        # initialize the occupation information
        self.occupation_info = OccupationInfo(
            z_nuclear         = int(self.pseudo.z_nuclear), 
            z_valence         = int(self.pseudo.z_valence),
            all_electron_flag = self.all_electron_flag)
            

        # XC evaluator for delta-learning usage (to be initialized when needed)
        self.xc_evaluator = None  # TODO: Initialize when delta-learning is implemented

        # Grid data and operators (initialized in __init__)
        self.grid_data_standard   : Optional[GridData] = None
        self.grid_data_dense      : Optional[GridData] = None
        self.ops_builder_standard : Optional[RadialOperatorsBuilder] = None
        self.ops_builder_dense    : Optional[RadialOperatorsBuilder] = None

        # SCF components (initialized in __init__)
        self.hamiltonian_builder : Optional[HamiltonianBuilder] = None
        self.density_calculator  : Optional[DensityCalculator]  = None
        self.poisson_solver      : Optional[PoissonSolver]      = None
        self.energy_calculator   : Optional[EnergyCalculator]   = None
        self.scf_driver          : Optional[SCFDriver]          = None

        # Initialize grids and operators
        self.grid_data_standard, self.grid_data_dense = self._initialize_grids()
        self.ops_builder_standard = RadialOperatorsBuilder.from_grid_data(
            self.grid_data_standard, verbose=self.print_debug
        )
        self.ops_builder_dense = RadialOperatorsBuilder.from_grid_data(
            self.grid_data_dense, verbose=self.print_debug
        )
        
        # Initialize SCF components
        self._initialize_scf_components(
            ops_builder_standard = self.ops_builder_standard,
            grid_data_standard   = self.grid_data_standard,
            ops_builder_dense    = self.ops_builder_dense,
        )

        if self.print_debug:
            self.print_input_parameters()
            self.pseudo.print_info()
            self.occupation_info.print_info()


    def set_and_check_initial_parameters(self):
        """
        set and check the default parameters, if not provided
        """
        # atomic number
        assert isinstance(self.atomic_number, int), \
            ATOMIC_NUMBER_NOT_INTEGER_ERROR.format(type(self.atomic_number))
        assert self.atomic_number > 0, \
            ATOMIC_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.atomic_number)
        assert self.atomic_number < 119, \
            ATOMIC_NUMBER_LARGER_THAN_119_ERROR.format(self.atomic_number)

        # domain size
        if self.domain_size is None:
            self.domain_size = 20.0
        try:
            self.domain_size = float(self.domain_size)
        except:
            raise ValueError(DOMAIN_SIZE_NOT_FLOAT_ERROR.format(type(self.domain_size)))
        assert isinstance(self.domain_size, float), \
            DOMAIN_SIZE_NOT_FLOAT_ERROR.format(type(self.domain_size))
        assert self.domain_size > 0, \
            DOMAIN_SIZE_NOT_GREATER_THAN_0_ERROR.format(self.domain_size)

        # number of finite elements
        if self.number_of_finite_elements is None:
            self.number_of_finite_elements = 17
        assert isinstance(self.number_of_finite_elements, int), \
            NUMBER_OF_FINITE_ELEMENTS_NOT_INTEGER_ERROR.format(type(self.number_of_finite_elements))
        assert self.number_of_finite_elements > 0, \
            NUMBER_OF_FINITE_ELEMENTS_NOT_GREATER_THAN_0_ERROR.format(self.number_of_finite_elements)

        # polynomial order
        if self.polynomial_order is None:
            self.polynomial_order = 31
        assert isinstance(self.polynomial_order, int), \
            POLYNOMIAL_ORDER_NOT_INTEGER_ERROR.format(type(self.polynomial_order))
        assert self.polynomial_order > 0, \
            POLYNOMIAL_ORDER_NOT_GREATER_THAN_0_ERROR.format(self.polynomial_order)

        # grid points integration quadrature
        if self.quadrature_point_number is None:
            self.quadrature_point_number = 95
        assert isinstance(self.quadrature_point_number, int), \
            QUADRATURE_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(self.quadrature_point_number))
        assert self.quadrature_point_number > 0, \
            QUADRATURE_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.quadrature_point_number)

        # xc functional
        if self.xc_functional is None:
            self.xc_functional = 'GGA_PBE'
        assert isinstance(self.xc_functional, str), \
            XC_FUNCTIONAL_NOT_STRING_ERROR.format(type(self.xc_functional))
        assert self.xc_functional in VALID_XC_FUNCTIONAL_LIST, \
            XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, self.xc_functional)

        # mesh type
        if self.mesh_type is None:
            self.mesh_type = 'exponential'
        assert isinstance(self.mesh_type, str), \
            MESH_TYPE_NOT_STRING_ERROR.format(type(self.mesh_type))
        assert self.mesh_type in ['exponential', 'polynomial', 'uniform'], \
            MESH_TYPE_NOT_IN_VALID_LIST_ERROR.format(VALID_MESH_TYPE_LIST, self.mesh_type)

        # mesh concentration
        if self.mesh_concentration is None: # default value
            if self.mesh_type == 'exponential':
                self.mesh_concentration = 61.0
            elif self.mesh_type == 'polynomial':
                self.mesh_concentration = 3.0
            elif self.mesh_type == 'uniform':
                self.mesh_concentration = None
        if self.mesh_type in ['exponential', 'polynomial']: # type check
            try:
                self.mesh_concentration = float(self.mesh_concentration)
            except:
                raise ValueError(MESH_CONCENTRATION_NOT_FLOAT_ERROR.format(type(self.mesh_concentration)))
            assert isinstance(self.mesh_concentration, float), \
                MESH_CONCENTRATION_NOT_FLOAT_ERROR.format(type(self.mesh_concentration))
            assert self.mesh_concentration > 0., \
                MESH_CONCENTRATION_NOT_GREATER_THAN_0_ERROR.format(self.mesh_concentration)
        elif self.mesh_type == 'uniform':
            if self.mesh_concentration is not None:
                print(MESH_CONCENTRATION_NOT_NONE_FOR_UNIFORM_MESH_TYPE_WARNING)
                self.mesh_concentration = None
        else:
            raise ValueError("This error should never be raised")
            
        # scf tolerance
        if self.scf_tolerance is None:
            # For most functionals, the default tolerance is 1e-8
            self.scf_tolerance = 1e-8
            if self.xc_functional in ['SCAN']:
                # SCAN functional suffers from convergence issues, so we use a higher tolerance
                self.scf_tolerance = 1e-4
        try:
            self.scf_tolerance = float(self.scf_tolerance)
        except:
            raise ValueError(SCF_TOLERANCE_NOT_FLOAT_ERROR.format(type(self.scf_tolerance)))
        assert isinstance(self.scf_tolerance, float), \
            SCF_TOLERANCE_NOT_FLOAT_ERROR.format(type(self.scf_tolerance))
        assert self.scf_tolerance > 0., \
            SCF_TOLERANCE_NOT_GREATER_THAN_0_ERROR.format(self.scf_tolerance)

        # all electron flag
        if self.all_electron_flag is None:
            self.all_electron_flag = False
        if self.all_electron_flag in [0, 1]:
            self.all_electron_flag = False if self.all_electron_flag == 0 else True
        assert isinstance(self.all_electron_flag, bool), \
            ALL_ELECTRON_FLAG_NOT_BOOL_ERROR.format(type(self.all_electron_flag))

        # enable domain size test
        if self.enable_domain_size_test is None:
            self.enable_domain_size_test = False
        if self.enable_domain_size_test in [0, 1]:
            self.enable_domain_size_test = False if self.enable_domain_size_test == 0 else True
        assert isinstance(self.enable_domain_size_test, bool), \
            ENABLE_DOMAIN_SIZE_TEST_NOT_BOOL_ERROR.format(type(self.enable_domain_size_test))

        # psp directory path
        if self.all_electron_flag == False:
            if self.psp_dir_path is None:
                self.psp_dir_path = os.path.join(os.path.dirname(__file__), "..", "psps")
            if not os.path.exists(self.psp_dir_path):
                # if the psp directory path is not absolute path, make it absolute path
                self.psp_dir_path = os.path.join(os.path.dirname(__file__), "..", self.psp_dir_path) 
            assert isinstance(self.psp_dir_path, str), \
                PSP_DIR_PATH_NOT_STRING_ERROR.format(type(self.psp_dir_path))
            assert os.path.exists(self.psp_dir_path), \
                PSP_DIR_PATH_NOT_EXISTS_ERROR.format(self.psp_dir_path)

        elif self.all_electron_flag == True:
            if self.psp_dir_path is not None:
                print(PSP_DIR_PATH_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING)
                self.psp_dir_path = None
        else:
            raise ValueError("This error should never be raised")

        # psp file name
        if self.all_electron_flag == False:
            if self.psp_file_name is None:
                # default value
                if self.atomic_number < 10:
                    self.psp_file_name = "0" + str(self.atomic_number) + ".psp8"
                else:
                    self.psp_file_name = str(self.atomic_number) + ".psp8"
            assert isinstance(self.psp_file_name, str), \
                PSP_FILE_NAME_NOT_STRING_ERROR.format(type(self.psp_file_name))
            assert os.path.exists(os.path.join(self.psp_dir_path, self.psp_file_name)), \
                PSP_FILE_NAME_NOT_EXISTS_ERROR.format(self.psp_file_name, self.psp_dir_path)
        elif self.all_electron_flag == True:
            if self.psp_file_name is not None:
                print(PSP_FILE_NAME_NOT_NONE_FOR_ALL_ELECTRON_CALCULATION_WARNING)
                self.psp_file_name = None
        else:
            raise ValueError("This error should never be raised")

        # frequency integration point number
        if self.xc_functional in ['RPA', ]:
            if self.frequency_integration_point_number is None:
                self.frequency_integration_point_number = 1200
            assert isinstance(self.frequency_integration_point_number, int), \
                FREQUENCY_INTEGRATION_POINT_NUMBER_NOT_INTEGER_ERROR.format(type(self.frequency_integration_point_number))
            assert self.frequency_integration_point_number > 0, \
                FREQUENCY_INTEGRATION_POINT_NUMBER_NOT_GREATER_THAN_0_ERROR.format(self.frequency_integration_point_number)
        else:
            if self.frequency_integration_point_number is not None:
                print(FREQUENCY_INTEGRATION_POINT_NUMBER_NOT_NONE_FOR_OEPX_AND_NONE_XC_FUNCTIONAL_WARNING.format(self.xc_functional))
                self.frequency_integration_point_number = None

        # eigentolerance x
        if self.xc_functional in ['OEPx', 'RPA']:  # Default value: 1e-11 for OEPx, 1e-9 for RPA
            if self.eigentolerance_x is None:
                self.eigentolerance_x = 1e-11 if self.xc_functional == 'OEPx' else 1e-9
            assert isinstance(self.eigentolerance_x, float), \
                EIGENTOLERANCE_X_NOT_FLOAT_ERROR.format(type(self.eigentolerance_x))
            assert self.eigentolerance_x > 0., \
                EIGENTOLERANCE_X_NOT_GREATER_THAN_0_ERROR.format(self.eigentolerance_x)
        else:
            if self.eigentolerance_x is not None:
                print(EIGENTOLERANCE_X_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING.format(self.xc_functional))
                self.eigentolerance_x = None
        
        # eigentolerance c
        if self.xc_functional in ['OEPx', 'RPA']:  # Default value: 1e-9 for OEPx, 1e-9 for RPA
            if self.eigentolerance_c is None:
                self.eigentolerance_c = 1e-9
            assert isinstance(self.eigentolerance_c, float), \
                EIGENTOLERANCE_C_NOT_FLOAT_ERROR.format(type(self.eigentolerance_c))
            assert self.eigentolerance_c > 0., \
                EIGENTOLERANCE_C_NOT_GREATER_THAN_0_ERROR.format(self.eigentolerance_c)
        else:
            if self.eigentolerance_c is not None:
                print(EIGENTOLERANCE_C_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING.format(self.xc_functional))
                self.eigentolerance_c = None

        # l_max_quantum_number
        if self.l_max_quantum_number is None:
            if self.xc_functional == 'OEPx':
                self.l_max_quantum_number = 8
            elif self.xc_functional == 'RPA':
                self.l_max_quantum_number = 4
            else:
                self.l_max_quantum_number = 0
        assert isinstance(self.l_max_quantum_number, int), \
            L_MAX_QUANTUM_NUMBER_NOT_INTEGER_ERROR.format(type(self.l_max_quantum_number))
        assert self.l_max_quantum_number >= 0., \
            L_MAX_QUANTUM_NUMBER_NEGATIVE_ERROR.format(self.l_max_quantum_number)

        # smoothening cutoff frequency
        if self.xc_functional in ['RPA', ]:
            if self.smoothening_cutoff_frequency is None:
                self.smoothening_cutoff_frequency = 60.0
                if self.atomic_number == 2:
                    self.smoothening_cutoff_frequency = 60.0
                elif self.atomic_number == 4:
                    self.smoothening_cutoff_frequency = 60.0
                elif self.atomic_number == 10:
                    self.smoothening_cutoff_frequency = 60.0
                elif self.atomic_number == 12:
                    self.smoothening_cutoff_frequency = 60.0
                elif self.atomic_number == 18:
                    self.smoothening_cutoff_frequency = 100.0
            assert isinstance(self.smoothening_cutoff_frequency, float), \
                SMOOTHENING_CUTOFF_FREQUENCY_NOT_FLOAT_ERROR.format(type(self.smoothening_cutoff_frequency))
            assert self.smoothening_cutoff_frequency >= 0., \
                SMOOTHENING_CUTOFF_FREQUENCY_NEGATIVE_ERROR.format(self.smoothening_cutoff_frequency)
        else:
            if self.smoothening_cutoff_frequency is not None:
                print(SMOOTHENING_CUTOFF_FREQUENCY_NOT_NONE_FOR_XC_FUNCTIONAL_OTHER_THAN_RPA_WARNING.format(self.xc_functional))
                self.smoothening_cutoff_frequency = None

        # double hybrid flag
        if self.double_hybrid_flag is None:
            self.double_hybrid_flag = False
        if self.double_hybrid_flag in [0, 1]:
            self.double_hybrid_flag = False if self.double_hybrid_flag == 0 else True
        assert isinstance(self.double_hybrid_flag, bool), \
            DOUBLE_HYBRID_FLAG_NOT_BOOL_ERROR.format(type(self.double_hybrid_flag))

        # hybrid mixing parameter
        # Only validate for hybrid functionals (PBE0, HF)
        if self.xc_functional in ['PBE0', 'HF']:
            if self.hybrid_mixing_parameter is None:
                # Use default values based on functional
                if self.xc_functional == 'PBE0':
                    self.hybrid_mixing_parameter = 0.25
                    # print(NO_HYBRID_MIXING_PARAMETER_PROVIDED_FOR_HYBRID_FUNCTIONAL_WARNING.format(self.xc_functional, 0.25))
                elif self.xc_functional == 'HF':
                    self.hybrid_mixing_parameter = 1.0

            # If the hybrid mixing parameter is provided, check the type and value
            assert isinstance(self.hybrid_mixing_parameter, (float, int)), \
                HYBRID_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.hybrid_mixing_parameter))
            assert 0.0 <= self.hybrid_mixing_parameter <= 1.0, \
                HYBRID_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR.format(self.hybrid_mixing_parameter)
            if self.xc_functional == "HF":
                assert self.hybrid_mixing_parameter == 1.0, \
                    HYBRID_MIXING_PARAMETER_NOT_ONE_FOR_HF_ERROR.format(self.hybrid_mixing_parameter)
        else:
            # For non-hybrid functionals, hybrid_mixing_parameter is not used
            # Set it to None to avoid confusion
            self.hybrid_mixing_parameter = None
        
        # density mixing parameter
        if self.density_mixing_parameter is None:
            self.density_mixing_parameter = 0.5
        try:
            self.density_mixing_parameter = float(self.density_mixing_parameter)
        except:
            raise ValueError(DENSITY_MIXING_PARAMETER_NOT_FLOAT_ERROR.format(type(self.density_mixing_parameter)))
        assert 0.0 < self.density_mixing_parameter <= 1.0, \
            DENSITY_MIXING_PARAMETER_NOT_IN_ZERO_ONE_ERROR.format(self.density_mixing_parameter)

        # mesh spacing
        if self.mesh_spacing is None:
            self.mesh_spacing = 0.1
        assert isinstance(self.mesh_spacing, float), \
            MESH_SPACING_NOT_FLOAT_ERROR.format(type(self.mesh_spacing))
        assert self.mesh_spacing > 0., \
            MESH_SPACING_NOT_GREATER_THAN_0_ERROR.format(self.mesh_spacing)


        # print debug
        if self.print_debug is None:
            self.print_debug = False
        if self.print_debug in [0, 1]:
            self.print_debug = False if self.print_debug == 0 else True
        assert isinstance(self.print_debug, bool), \
            PRINT_DEBUG_NOT_BOOL_ERROR.format(type(self.print_debug))


    def print_input_parameters(self):

        # Display relative path for psp_dir_path
        if self.psp_dir_path is not None:
            try:
                # Try to get relative path from current working directory
                psp_path_display = os.path.relpath(self.psp_dir_path)
            except ValueError:
                # If relative path fails (e.g., different drives on Windows), use absolute path
                psp_path_display = self.psp_dir_path
        else:
            psp_path_display = self.psp_dir_path

        # print the input parameters
        print("=" * 60)
        print("\t\t INPUT PARAMETERS")
        print("=" * 60)

        print("\t atomic_number                      : {}".format(self.atomic_number))
        print("\t domain_size                        : {}".format(self.domain_size))
        print("\t number_of_finite_elements          : {}".format(self.number_of_finite_elements))
        print("\t polynomial_order                   : {}".format(self.polynomial_order))
        print("\t quadrature_point_number            : {}".format(self.quadrature_point_number))
        print("\t xc_functional                      : {}".format(self.xc_functional))
        print("\t mesh_type                          : {}".format(self.mesh_type))
        print("\t mesh_concentration                 : {}".format(self.mesh_concentration))
        print("\t scf_tolerance                      : {}".format(self.scf_tolerance))
        print("\t all_electron_flag                  : {}".format(self.all_electron_flag))
        print("\t enable_domain_size_test            : {}".format(self.enable_domain_size_test))
        print("\t psp_dir_path                       : {}".format(psp_path_display))
        print("\t psp_file_name                      : {}".format(self.psp_file_name))
        print("\t frequency_integration_point_number : {}".format(self.frequency_integration_point_number))
        print("\t eigentolerance_x                   : {}".format(self.eigentolerance_x))
        print("\t eigentolerance_c                   : {}".format(self.eigentolerance_c))
        print("\t l_max_quantum_number               : {}".format(self.l_max_quantum_number))
        print("\t smoothening_cutoff_frequency       : {}".format(self.smoothening_cutoff_frequency))
        print("\t double_hybrid_flag                 : {}".format(self.double_hybrid_flag))
        print("\t hybrid_mixing_parameter            : {}".format(self.hybrid_mixing_parameter))
        print("\t mesh_spacing                       : {}".format(self.mesh_spacing))
        print("\t density_mixing_parameter           : {}".format(self.density_mixing_parameter))
        print()


    def _initialize_grids(self) -> Tuple[GridData, GridData]:
        """
        Initialize finite element grids and quadrature.
        
        Generates two grid configurations:
        - Standard grid: for most operators and wavefunctions
        - Dense grid: refined mesh for Hartree potential solver (double density)
        
        Returns
        -------
        grid_data_standard : GridData
            Standard grid data for operators and wavefunctions
        grid_data_dense : GridData
            Dense grid data for Hartree solver
        """
        # Generate Lobatto interpolation nodes on reference interval [-1, 1]
        interp_nodes_ref, _ = Quadrature1D.lobatto(self.polynomial_order)
        
        # Generate mesh boundaries
        mesh1d = Mesh1D(
            domain_radius = self.domain_size / 2.0,
            finite_elements_num = self.number_of_finite_elements,
            mesh_type           = self.mesh_type,
            clustering_param    = self.mesh_concentration,
            exp_shift           = getattr(self, 'exp_shift', None)
        )
        boundaries_nodes, _ = mesh1d.generate_mesh_nodes_and_width()
        
        # Generate standard FE nodes
        global_nodes = Mesh1D.generate_fe_nodes(
            boundaries_nodes = boundaries_nodes,
            interp_nodes = interp_nodes_ref
        )

        # Generate refined FE nodes (for Hartree potential solver)
        refined_interp_nodes_ref = Mesh1D.refine_interpolation_nodes(interp_nodes_ref)
        refined_global_nodes = Mesh1D.generate_fe_nodes(
            boundaries_nodes = boundaries_nodes,
            interp_nodes = refined_interp_nodes_ref
        )
        
        # Generate Gauss-Legendre quadrature nodes and weights
        quadrature_nodes_ref, quadrature_weights_ref = Quadrature1D.gauss_legendre(
            self.quadrature_point_number
        )
        
        # Map quadrature to physical elements
        quadrature_nodes, quadrature_weights = Mesh1D.map_quadrature_to_physical_elements(
            boundaries_nodes = boundaries_nodes,
            interp_nodes     = quadrature_nodes_ref,
            interp_weights   = quadrature_weights_ref,
            flatten          = True
        )
        
        # Create grid data objects
        grid_data_standard = GridData(
            number_of_finite_elements = self.number_of_finite_elements,
            physical_nodes            = global_nodes,
            quadrature_nodes          = quadrature_nodes,
            quadrature_weights        = quadrature_weights
        )
        
        grid_data_dense = GridData(
            number_of_finite_elements = self.number_of_finite_elements,
            physical_nodes            = refined_global_nodes,
            quadrature_nodes          = quadrature_nodes,
            quadrature_weights        = quadrature_weights
        )
        
        return grid_data_standard, grid_data_dense


    def _initialize_scf_components(
        self, 
        ops_builder_standard: RadialOperatorsBuilder,
        grid_data_standard: GridData,
        ops_builder_dense: RadialOperatorsBuilder,
        ) -> None:
        """
        Initialize all SCF components.
        
        This method creates and configures all the modular SCF components:
        - HamiltonianBuilder : constructs Hamiltonian matrices (uses standard grid)
        - DensityCalculator  : computes density from orbitals (uses standard grid)
        - PoissonSolver      : solves for Hartree potential (uses dense grid)
        - EnergyCalculator   : computes total energy (uses standard grid)
        - SCFDriver          : manages SCF iterations
        
        Note: Only PoissonSolver uses the dense grid for accurate Hartree potential.
              All other components use the standard grid.
        """

        # Hamiltonian builder (uses standard grid)
        self.hamiltonian_builder = HamiltonianBuilder(
            ops_builder     = ops_builder_standard,
            pseudo          = self.pseudo,
            occupation_info = self.occupation_info,
            all_electron    = self.all_electron_flag
        )
        
        # Density calculator (uses standard grid, but the derivative matrix is from the dense grid)
        self.density_calculator = DensityCalculator(
            grid_data         = grid_data_standard,
            occupation_info   = self.occupation_info,
            derivative_matrix = ops_builder_dense.derivative_matrix
        )
        
        # Poisson solver for Hartree potential (uses dense grid for accuracy)
        self.poisson_solver = PoissonSolver(
            ops_builder = ops_builder_dense,
            z_valence   = float(self.occupation_info.z_valence)
        )
        
        # SCF Driver (create first to get xc_calculator)
        eigensolver = EigenSolver(xc_functional=self.xc_functional)
        mixer = Mixer(
            tol         = self.scf_tolerance, 
            alpha_lin   = (self.density_mixing_parameter, self.density_mixing_parameter), 
            alpha_pulay = 0.55, 
            history     = 7, 
            frequency   = 2
        )
        
        self.scf_driver = SCFDriver(
            hamiltonian_builder = self.hamiltonian_builder,
            density_calculator  = self.density_calculator,
            poisson_solver      = self.poisson_solver,
            eigensolver         = eigensolver,
            mixer               = mixer,
            occupation_info     = self.occupation_info,
            xc_functional       = self.xc_functional,
            hybrid_mixing_parameter = self.hybrid_mixing_parameter
        )
        
        # Get XC calculator and HF calculator from scf_driver
        xc_calculator = self.scf_driver.xc_calculator if hasattr(self.scf_driver, 'xc_calculator') else None
        hf_calculator = self.scf_driver.hf_calculator if hasattr(self.scf_driver, 'hf_calculator') else None
        
        # Energy calculator (uses standard grid data and ops_builder, but dense derivative matrix)
        self.energy_calculator = EnergyCalculator(
            grid_data          = grid_data_standard,
            occupation_info    = self.occupation_info,
            ops_builder        = ops_builder_standard,
            poisson_solver     = self.poisson_solver,
            pseudo             = self.pseudo,
            xc_calculator      = xc_calculator,
            hf_calculator      = hf_calculator,  # Pass HF calculator from SCFDriver
            derivative_matrix  = ops_builder_dense.derivative_matrix  # Use dense grid derivative for accuracy
        )
        

    def _get_scf_settings(self, xc_functional: str) -> Dict[str, Any]:
        assert isinstance(xc_functional, str), \
            XC_FUNCTIONAL_TYPE_ERROR_MESSAGE.format(type(xc_functional))
        assert xc_functional in VALID_XC_FUNCTIONAL_LIST, \
            XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, xc_functional)
        
        """Get SCF settings based on XC functional."""
        settings = {
            'inner_max_iter' : 500,
            'outer_max_iter' : 1,  # Default: no outer loop for LDA/GGA
            'rho_tol'        : self.scf_tolerance,
            'outer_rho_tol'  : self.scf_tolerance,
            'n_consecutive'  : 1,
            'print_debug'    : self.print_debug
        }
        
        # For functionals requiring outer loop (HF, OEP, RPA)
        if xc_functional in ['HF', 'PBE0', 'OEPx', 'RPA']:
            settings['outer_max_iter'] = 50
        
        return settings


    def _evaluate_basis_on_uniform_grid(
        self, 
        ops_builder_standard: RadialOperatorsBuilder,
        orbitals: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate all orbitals on a uniform evaluation grid.
        
        This function generates a uniform grid spanning the domain and evaluates
        each orbital state on that grid using Lagrange interpolation. The result
        is useful for:
        - Visualization and analysis (uniform spacing for plotting)
        - Output formatting (matching reference data formats)
        - Further post-processing (interpolation to different grids)
        
        Parameters
        ----------
        ops_builder_standard : RadialOperatorsBuilder
            Operators builder containing mesh information and interpolation methods.
            Used to evaluate orbitals on the given grid using finite element basis functions.
        orbitals : np.ndarray
            Orbital coefficients at physical nodes, shape (n_physical_nodes, n_states).
            Each column represents one orbital state (eigenvector).
        
        Returns
        -------
        orbitals_on_given_grid : np.ndarray
            Orbital values evaluated on the uniform grid, shape (n_grid_points, n_states).
            Each column contains the values of one orbital state on the uniform grid.
        
        Notes
        -----
        - The uniform grid is generated with spacing `self.mesh_spacing` over `[0, domain_size]`.
        - Each orbital is evaluated independently using `evaluate_single_orbital_on_given_grid`.
        - The evaluation uses Lagrange polynomial interpolation within each finite element.
        
        Example
        -------
        >>> uniform_grid_values = solver._evaluate_basis_on_uniform_grid(
        ...     ops_builder_standard=ops_builder,
        ...     orbitals=eigenvectors  # shape: (n_nodes, n_states)
        ... )
        >>> # uniform_grid_values.shape = (n_grid_points, n_states)
        """
        # Generate uniform evaluation grid with specified spacing
        uniform_eval_grid = np.linspace(
            start=0.0, 
            stop=self.domain_size, 
            num=int(self.domain_size / self.mesh_spacing) + 1, 
            endpoint=True
        )

        # Evaluate each orbital state on the uniform grid
        n_states     = orbitals.shape[1]
        n_grid_given = len(uniform_eval_grid)
        orbitals_on_given_grid = np.zeros((n_grid_given, n_states))

        for state_index in range(n_states):
            # Evaluate single orbital on the uniform grid using Lagrange interpolation
            orbitals_on_given_grid[:, state_index] = ops_builder_standard.evaluate_single_orbital_on_given_grid(
                given_grid = uniform_eval_grid,
                orbital = orbitals[:, state_index]
            )
        return uniform_eval_grid, orbitals_on_given_grid


    def _get_initial_density_and_orbitals_with_warm_start(
        self, 
        xc_functional    : str, 
        rho_initial      : np.ndarray, 
        orbitals_initial : Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warm start calculation using specified XC functional to obtain initial density and orbitals.
        
        This function is used for initializing meta-GGA functionals (e.g., SCAN) by running
        a simpler functional (e.g., GGA_PBE) first to obtain better initial guesses, which
        improves convergence.
        
        Parameters
        ----------
        xc_functional : str
            XC functional name for warm start (e.g., "GGA_PBE" or "RSCAN")
        rho_initial : np.ndarray
            Initial density guess to start the warm calculation
        orbitals_initial : np.ndarray, optional
            Initial orbitals guess (if available), None otherwise
        
        Returns
        -------
        rho_initial : np.ndarray
            Initial density from warm start calculation
        orbitals_initial : np.ndarray
            Initial orbitals from warm start calculation
        
        Notes
        -----
        - Creates a temporary SCFDriver based on existing components
        - Uses relaxed convergence criteria to accelerate warm start
        - Warm start calculation does not require full convergence, only a reasonable initial guess
        """
        # Create temporary eigensolver with specified functional type
        eigensolver_warm = EigenSolver(xc_functional=xc_functional)


        # Create temporary SCFDriver with specified xc_functional
        # Reuse existing hamiltonian_builder, density_calculator, poisson_solver in the main SCFDriver
        scf_driver_warm = SCFDriver(
            hamiltonian_builder = self.scf_driver.hamiltonian_builder,
            density_calculator  = self.scf_driver.density_calculator,
            poisson_solver      = self.scf_driver.poisson_solver,
            eigensolver         = eigensolver_warm,
            mixer               = self.scf_driver.mixer,
            occupation_info     = self.scf_driver.occupation_info,
            xc_functional       = xc_functional,  # Use specified functional
            hybrid_mixing_parameter = self.scf_driver.hybrid_mixing_parameter
        )
        
        
        # Run warm start SCF calculation
        if self.print_debug:
            print(f"[Warm Start] Running {xc_functional} pre-calculation for initial guess")
        
        scf_result_warm = scf_driver_warm.run(
            rho_initial      = rho_initial,
            settings         = self._get_scf_settings(xc_functional),
            orbitals_initial = orbitals_initial
        )
        
        if self.print_debug:
            if not scf_result_warm.converged:
                print(WARM_START_NOT_CONVERGED_WARNING.format(xc_functional))
        
        # Extract results: density and orbitals
        rho_final      = scf_result_warm.density_data.rho
        orbitals_final = scf_result_warm.orbitals
        
        return rho_final, orbitals_final


    def forward(self, orbitals) -> Dict[str, Any]:
        """
        Forward pass of the atomic DFT solver.
        
        This method performs a single forward pass without SCF iteration:
        - Takes rho and orbitals as input
        - Computes XC potential and energy
        - Returns results in the same format as solve()
        
        Parameters
        ----------
        rho : np.ndarray
            Electron density, shape (n_quad_points,)
        orbitals : np.ndarray
            Kohn-Sham orbitals (radial wavefunctions R_nl(r))
            Shape: (n_states, n_quad_points)
        
        Returns
        -------
        final_result : Dict[str, Any]
            Dictionary containing:
            - eigen_energies: None (not computed in forward pass)
            - orbitals: input orbitals
            - rho: valence density computed from orbitals
            - density_data: DensityData computed from orbitals (without NLCC)
            - rho_nlcc: non-linear core correction density
            - energy: total energy
            - energy_components: EnergyComponents object
            - grid_data: GridData for standard grid
            - occupation_info: OccupationInfo
            - xc_potential: XCPotentialData
        """
        # Phase 1: Get XC functional requirements
        # Note: Grids and SCF components are already initialized in __init__
        xc_requirements = get_functional_requirements(self.xc_functional)
        
        # Phase 2: Calculate rho_nlcc (non-linear core correction for pseudopotentials)
        rho_nlcc = self.pseudo.get_rho_core_correction(self.grid_data_standard.quadrature_nodes)
        
        # Phase 3: Create density_data from orbitals (with NLCC for XC potential calculation)
        # Note: For XC potential calculation, we need density_data with NLCC
        # For energy calculation, we use density_data without NLCC
        density_data_with_nlcc = self.density_calculator.create_density_data_from_orbitals(
            orbitals         = orbitals,
            compute_gradient = xc_requirements.needs_gradient,
            compute_tau      = xc_requirements.needs_tau,
            normalize        = True,
            rho_nlcc         = rho_nlcc
        )
        
        # Phase 4: Compute XC potential data (using density_data with NLCC)
        xc_potential : XCPotentialData = self.energy_calculator.compute_xc_potential(density_data_with_nlcc)
        
        # Phase 5: Create density_data without NLCC for energy calculation
        # Energy calculation uses valence density only (without NLCC)
        density_data_valence = self.density_calculator.create_density_data_from_orbitals(
            orbitals         = orbitals,
            compute_gradient = xc_requirements.needs_gradient,
            compute_tau      = xc_requirements.needs_tau,
            normalize        = True,
            rho_nlcc         = None  # No NLCC for energy calculation
        )
        
        # Phase 6: Compute final energy (using valence density only)
        energy_components : EnergyComponents = self.energy_calculator.compute_energy(
            orbitals         = orbitals,
            density_data     = density_data_valence,
            mixing_parameter = self.hybrid_mixing_parameter
        )

        # Phase 7: Evaluate basis functions on uniform grid (optional, for future use)
        # TODO: Implement this when needed
        # Note: _evaluate_basis_on_uniform_grid expects SCFResult, which we don't have in forward pass
        # For now, we'll skip this step as it's not essential for the forward pass
        # If needed in the future, we can create a minimal SCFResult-like object or modify the method
        uniform_grid, orbitals_on_uniform_grid = self._evaluate_basis_on_uniform_grid(
            ops_builder_standard = self.ops_builder_standard,
            orbitals             = orbitals
        )

        # Phase 10: Pack and return results
        final_result = {
            'eigen_energies'    : None,  # Not computed in forward pass
            'orbitals'          : orbitals,
            'rho'               : density_data_valence.rho,  # Valence density
            'density_data'      : density_data_valence,      # Density data without NLCC
            'rho_nlcc'          : rho_nlcc,
            'energy'            : energy_components.total,
            'energy_components' : energy_components,
            'converged'         : True,  # Forward pass doesn't iterate, so always "converged"
            'iterations'        : 0,     # No iterations in forward pass
            'rho_residual'      : 0.0,   # No residual in forward pass
            'grid_data'         : self.grid_data_standard,
            'occupation_info'   : self.occupation_info,
            'xc_potential'      : xc_potential,
        }
        
        return final_result        


    def solve(self) -> Dict[str, Any]:
        """
        Solve the Kohn-Sham equations using modular SCF architecture.
        
        Clean workflow:
        1. Initialize grids and operators
        2. Initialize SCF components
        3. Get initial density guess
        4. Run SCF iteration
        5. Compute final energy
        6. Return results

        """        
        # Phase 1: Initial density guess
        # Note: Grids and SCF components are already initialized in __init__
        rho_initial = self.pseudo.get_rho_guess(self.grid_data_standard.quadrature_nodes)
        rho_nlcc    = self.pseudo.get_rho_core_correction(self.grid_data_standard.quadrature_nodes)
        orbitals_initial = None

        # Warm start calculation for relatively expensive meta-GGA functionals
        if self.xc_functional in ['SCAN', 'RSCAN', 'R2SCAN']:
            rho_initial, orbitals_initial = self._get_initial_density_and_orbitals_with_warm_start(
                xc_functional    = "GGA_PBE", 
                rho_initial      = rho_initial, 
                orbitals_initial = orbitals_initial)

        # Phase 2: Run SCF
        scf_result : SCFResult = self.scf_driver.run(
            rho_initial      = rho_initial,
            settings         = self._get_scf_settings(self.xc_functional),
            orbitals_initial = orbitals_initial
        )

        # Phase 3: Compute final xc potential data
        xc_potential : XCPotentialData = self.energy_calculator.compute_xc_potential(scf_result.density_data)
        
        # Phase 4: Compute final energy
        energy_components : EnergyComponents = self.energy_calculator.compute_energy(
            orbitals         = scf_result.orbitals,
            density_data     = scf_result.density_data,
            mixing_parameter = self.hybrid_mixing_parameter
        )

        # Phase 5: Evaluate basis functions on uniform grid
        uniform_grid, orbitals_on_uniform_grid = self._evaluate_basis_on_uniform_grid(
            ops_builder_standard = self.ops_builder_standard,
            orbitals             = scf_result.orbitals
        )


        # Phase 6: Pack and return results
        final_result = {
            'eigen_energies'    : scf_result.eigen_energies,
            'orbitals'          : scf_result.orbitals,
            'rho'               : scf_result.density_data.rho,  # Interpolate over psi and calculate rho at that site
            'density_data'      : scf_result.density_data,      # 
            'rho_nlcc'          : rho_nlcc,
            'energy'            : energy_components.total,
            'energy_components' : energy_components,
            'converged'         : scf_result.converged,
            'iterations'        : scf_result.iterations,
            'rho_residual'      : scf_result.rho_residual,
            'grid_data'         : self.grid_data_standard,
            'occupation_info'   : self.occupation_info,
            'xc_potential'      : xc_potential,
            'uniform_grid'      : uniform_grid,
            'orbitals_on_uniform_grid' : orbitals_on_uniform_grid,
        }


        # source : rho, grad_rho, lap_rho, v_hartree...
        # target : vx, vc

        # Make a note. 
        # 1. Generalize to partial occupation
        # 2. Net charge
        # 3. inverse density -> Vp problem

        # Gauge for energy density (double integral -> different gauges)
        # Non-locality


        if self.print_debug:
            energy_components.print_info(title = f"Total Energy ({self.xc_functional})")
            print("="*60)
            print("\t\t Calculation Complete")
            print("="*60)
            print()
        
        return final_result



if __name__ == "__main__":
    atomic_dft_solver = AtomicDFTSolver(
        atomic_number     = 13, 
        print_debug       = True, 
        xc_functional     = "GGA_PBE",
        all_electron_flag = True,
    )

    results = atomic_dft_solver.solve()
    rho = results['rho']
    orbitals = results['orbitals']
    print(rho.shape)
    print(orbitals.shape)
