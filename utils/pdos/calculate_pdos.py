
"""
PDOS Calculator for SPARC Output Files

    This module provides functionality to calculate Projected Density of States (PDOS) 
    from SPARC (Spatial Partitioning of Atomic orbitals for Rapid Computation) output files.

    The calculator supports:
    - Full system PDOS calculation for all atoms and orbitals
    - Selective atom PDOS calculation for specific atoms
    - Automatic atomic orbital generation or custom UPF file support
    - K-point parallelization for large systems
    - Multiple exchange-correlation functionals support

    @file    calculate_pdos.py
    @brief   PDOS Calculator for SPARC output files
    @authors Qihao Cheng <qcheng61@gatech.edu>
             Shashikant Kumar <shashikant@gatech.edu>
             Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>

    Copyright (c) 2025 Material Physics & Mechanics Group, Georgia Tech.
"""


import os
import sys
import re
import shutil
import time
import argparse
import logging
import yaml
import traceback
import math
from pathlib import Path
from functools import partial

from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import fractional_matrix_power

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


# Regex patterns
SCIENTIFIC_NOTATION_PATTERN = r'[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?'

# XC Functional valid list
# Valid XC Functional
VALID_XC_FUNCTIONAL_LIST = [
    'LDA_PZ' , # LDA Perdew-Zunger
    'LDA_PW' , # LDA Perdew-Wang
    'GGA_PBE', # GGA Perdew-Burke-Ernzerhof
    'SCAN'   , # SCAN functional, meta-GGA
    'RSCAN'  , # RSCAN functional, meta-GGA
    'R2SCAN' , # R2SCAN functional, meta-GGA
    'HF'     , # Hartree-Fock
    'PBE0'   , # PBE0 Perdew-Burke-Ernzerhof, hybrid functional
    'EXX'    , # Exact Exchange, using OEP method
    'RPA'    , # Random Phase Approximation, with exact exchange
]

# Type check errors
NATOM_TYPE_NOT_INTEGER_ERROR = \
    "parameter natom_types must be an integer, get {} instead"
OUT_DIRNAME_NOT_STRING_ERROR = \
    "parameter out_dirname must be a string, get {} instead"
FNAME_NOT_STRING_ERROR = \
    "parameter fname must be a string, get {} instead"
SAVE_FNAME_NOT_STRING_ERROR = \
    "parameter save_fname must be a string, get {} instead"
MU_PDOS_NOT_FLOAT_ERROR = \
    "parameter mu_PDOS must be a float, get {} instead"
N_PDOS_NOT_INTEGER_ERROR = \
    "parameter N_PDOS must be an integer, get {} instead"
MIN_E_PLOT_NOT_FLOAT_ERROR = \
    "parameter min_E_plot must be a float, get {} instead"
MAX_E_PLOT_NOT_FLOAT_ERROR = \
    "parameter max_E_plot must be a float, get {} instead"
R_CUT_MAX_NOT_FLOAT_OR_LIST_ERROR = \
    "parameter r_cut_max must be a float or a list, get {} instead"
R_CUT_MAX_NUM_NOT_EQUAL_TO_NATOM_TYPES_ERROR = \
    "number of r_cut_max must be equal to the number of atom types, get {} instead"
ATOMIC_WAVE_FUNCTION_TOL_NOT_FLOAT_ERROR = \
    "parameter atomic_wave_function_tol must be a float, get {} instead"
ATOMIC_WAVE_FUNCTION_TOL_NOT_POSITIVE_ERROR = \
    "parameter atomic_wave_function_tol must be positive, get {} instead"
ORTHOGONALIZE_ATOMIC_ORBITALS_NOT_BOOL_ERROR = \
    "parameter orthogonalize_atomic_orbitals must be a bool, get {} instead"
RELAX_FLAG_NOT_INTEGER_ERROR = \
    "parameter relax_flag must be an integer, get {} instead"
RELAX_FLAG_NOT_VALID_ERROR = \
    "parameter relax_flag must be 1, 2, or 3, get {} instead"
K_POINT_PARALLELIZATION_NOT_BOOL_ERROR = \
    "parameter k_point_parallelization must be a bool, get {} instead"
K_POINT_FRACTIONAL_INPUT_NOT_NP_ARRAY_ERROR = \
    "parameter k_point_fractional must be a numpy array, get {} instead"

# Boundary condition errors
BCX_NOT_D_OR_P_ERROR = \
    "parameter bcx must be either 'D' or 'P', get {} instead"
BCY_NOT_D_OR_P_ERROR = \
    "parameter bcy must be either 'D' or 'P', get {} instead"
BCZ_NOT_D_OR_P_ERROR = \
    "parameter bcz must be either 'D' or 'P', get {} instead"


ATOM_TYPE_NOT_FOUND_ERROR = \
    "parameter atom_type {} not found in the atom_species_list"
ATOM_TYPE_INDEX_NOT_INTEGER_ERROR = \
    "parameter atom_type_index must be an integer, get {} instead"
ATOM_TYPE_NOT_INTEGER_OR_STRING_ERROR = \
    "parameter atom type must be an integer or a string, get {} instead"
ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_NOT_INTEGER_ERROR = \
    "parameter atom_index_for_specified_atom_type must be an integer, get {} instead"
ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_NOT_FOUND_ERROR = \
    "parameter atom index {} for specified atom type {} not found"
ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_OUT_OF_BOUND_ERROR = \
    "parameter atom index {} for specified atom type {} is out of bound"
ATOM_TYPE_AND_ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_LENGTH_MISMATCH_ERROR = \
    "The length of atom_type and atom_index_for_specified_atom_type must be the same, get {} and {} instead"
ATOM_COUNT_LIST_NOT_LIST_ERROR = \
    "parameter atom_count_list must be a list, get type {} instead"

# Atomic wave function generator check errors
ATOMIC_WAVE_FUNCTION_GENERATOR_NOT_CALLABLE_ERROR = \
    "parameter atomic_wave_function_generator must be a callable, get {} instead"
ATOMIC_WAVE_FUNCTION_GENERATOR_NOT_VALID_ERROR = \
    "parameter atomic_wave_function_generator is not valid, got error {}"
ATOMIC_WAVE_FUNCTION_GENERATOR_NO_UPF_FILES_PROVIDED_WARNING = \
    "WARNING: No UPF files provided, please provide the function to generate the atomic wave functions as instructed in the documentation."
ATOMIC_WAVE_FUNCTION_GENERATOR_NO_UPF_FILES_PROVIDED_ERROR = \
    "No UPF files provided, please provide the function to generate the atomic wave functions as instructed in the documentation."

# Projection data from txt path check errors
LOAD_PROJECTION_DATA_FROM_TXT_PATH_NOT_STRING_ERROR = \
    "parameter load_projection_data_from_txt_path must be a string, get {} instead"
LOAD_PROJECTION_DATA_FROM_TXT_PATH_NOT_EXIST_ERROR = \
    "Data projections file not found at {}"

# File existence check errors
FNAME_NOT_EXIST_ERROR = \
    "file {} does not exist"
OUTPUT_FNAME_NOT_EXIST_ERROR = \
    "output file {} does not exist"
EIGEN_FNAME_NOT_EXIST_ERROR = \
    "eigen file {} does not exist"
IS_RELAXATION_NOT_BOOL_ERROR = \
    "parameter is_relaxation must be a bool, get type {} instead"
STATIC_FNAME_NOT_EXIST_ERROR = \
    "static file {} does not exist"
GEOPT_FNAME_NOT_EXIST_ERROR = \
    "geopt file {} does not exist"
ION_FNAME_NOT_EXIST_ERROR = \
    "ion file {} does not exist"
ORBITAL_FNAME_NOT_EXIST_ERROR = \
    "orbital file {} does not exist"

# Output file check errors
OUTPUT_FNAME_NOT_STRING_ERROR = \
    "output filename must be a string, get {} instead"
INVALID_LATTICE_VECTOR_FORMAT_ERROR = \
    "Invalid lattice vector format in line: {}"
LATVEC_SCALE_OR_CELL_NOT_FOUND_ERROR = \
    "LATVEC_SCALE or CELL not found in the output file, cannot determine the lattice type"
NUMBER_OF_ATOM_TYPES_NOT_EQUAL_TO_OUTPUT_FILE_ERROR = \
    "number of atom types must be equal to the number of atom types in the output file, get {} instead"
NUMBER_OF_ATOM_COUNTS_NOT_EQUAL_TO_OUTPUT_FILE_ERROR = \
    "number of atom counts must be equal to the number of atom types in the output file, get {} instead"
ATOM_TYPE_NOT_EQUAL_TO_OUTPUT_FILE_ERROR = \
    "atom type must be equal to the index of the atom type in the output file, get {} instead"

XC_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter xc_functional must be a string, get {} instead"
XC_FUNCTIONAL_NOT_VALID_ERROR = \
    "parameter xc_functional must be a valid XC functional, get {} instead"


# UPF file check errors
UPF_FNAME_LIST_NOT_LIST_ERROR = \
    "parameter upf_fname_list must be a list, get {} instead"
UPF_FNAME_NUM_NOT_EQUAL_TO_NATOM_TYPES_ERROR = \
    "number of upf file names must be equal to the number of atom types, get {} instead"
UPF_FNAME_NOT_ENDING_WITH_UPF_ERROR = \
    "The upf file's name must end with .upf, get {} instead"
UPF_FNAME_NOT_STRING_ERROR = \
    "The upf file's name must be a string, get {} instead"
UPF_FNAME_NOT_EXIST_ERROR = \
    "The upf file {} does not exist"

# Static / geopt file check errors
STATIC_FNAME_NOT_STRING_ERROR = \
    "The static filename must be a string, get {} instead"
GEOPT_FNAME_NOT_STRING_ERROR = \
    "The geopt filename must be a string, get {} instead"
GEOPT_FNAME_NOT_ENDING_WITH_GEOPT_ERROR = \
    "The geopt filename must end with .geopt, get {} instead"
ION_FNAME_NOT_STRING_ERROR = \
    "The ion filename must be a string, get {} instead"

# Eigen file check errors
EIGEN_FNAME_NOT_STRING_ERROR = \
    "The eigen filename must be a string, get {} instead"
NUMBER_OF_KPOINTS_NOT_EQUAL_TO_EIGEN_FILE_ERROR = \
    "Expected {} weight indices, got {}"
KPOINT_NUMBER_NOT_EQUAL_TO_EIGEN_FILE_ERROR = \
    "Expected k-point number {}, got {}"
KPOINT_WEIGHTS_NOT_FOUND_ERROR = \
    "No k-point weights found for k-point {}"
EIGEN_FILE_NO_EIGENVALUES_AND_OCCUPATIONS_FOUND_ERROR = \
    "No eigenvalues and occupations found for k-point {} in line {}"

# Orbital file check errors
ORBITAL_FNAME_NOT_STRING_ERROR = \
    "The psi filename must be a string, get {} instead"

# Atom orbitals check errors
INVALID_ATOM_INPUT_ERROR = \
    "parameter atom_type_index must be an integer, get {} instead"
INDEX_MASK_DICT_INPUT_NOT_DICT_ERROR = \
    "parameter index_mask_dict must be a dict, get {} instead"
INDEX_MASK_AND_EFFECTIVE_GRID_POINT_POSITIONS_DICT_NOT_FOUND_ERROR = \
    "The (0,0,0) shift index is not in the index_mask_and_effective_grid_point_positions_dict"
ATOM_INDEX_MASK_AND_EFFECTIVE_GRID_POINT_POSITIONS_DICT_EMPTY_ERROR = \
    "The index_mask_and_effective_grid_point_positions_dict is empty, please call atom_wise_compute_index_mask_and_effective_grid_point_positions_dict() first."
PHI_ORBITAL_INPUT_NOT_PHI_ORBITAL_ERROR = \
    "parameter phi_orbital must be a PDOSCalculator.PhiOrbital, get {} instead"
PHI_ORBITALS_NOT_EQUAL_TO_EXPECTED_NUM_ERROR = \
    "Expected {} phi orbitals, got {} instead"


# Overlap matrix check errors
DV_NOT_FLOAT_ERROR = \
    "parameter dV must be a float, get {} instead"
ATOMS_INPUT_NOT_LIST_ERROR = \
    "parameter atoms must be a list, get {} instead"
ATOM_INPUT_NOT_ATOM_ERROR = \
    "Elements in the atoms list must be PDOSCalculator.Atom, get type {} instead"
ATOM_LIST_NOT_LIST_ERROR = \
    "parameter atom_list must be a list, get {} instead"
TWO_ATOMS_NOT_SAME_ERROR = \
    "The two atoms are not the same, the id for the two atoms are {} and {}"
TWO_ORBITALS_NOT_SAME_ERROR = \
    "The two orbitals are not the same, the id for the two orbitals are {} and {}"


# PDOS calculation check errors
INDEX_MASK_NOT_FOUND_ERROR = \
    "The index_mask is not found for the shifted unit cell ({}, {}, {})"
LATTICE_VECTOR_NOT_NORMALIZED_ERROR = \
    "Lattice vector direction {} is not normalized, norm = {}"


# Some typings
RCUT_MAX_TYPE = \
    Union[List[float], float]
ShiftIndexVectorType = \
    Tuple[int, int, int]
FractionalKPointVectorType = \
    Tuple[float, float, float] # (kx_frac, ky_frac, kz_frac)
ShiftIndexListType = \
    Union[List[int], List[List[int]]] # Shift index of each atom, depends on the type of self.r_cut_max, if the r_cut_max is a list, then the shift index is a list of list of integers, otherwise it is a list of integers
PhiValueDictType = \
    Dict[ShiftIndexVectorType, np.ndarray[float]] # (x_index_shift, y_index_shift, z_index_shift) -> phi_value
NormalizationFactorDictType = \
    Dict[FractionalKPointVectorType, float] # (kx_frac, ky_frac, kz_frac) -> normalization factor
TotalOrbitalsInUnitCellDictType = \
    Dict[FractionalKPointVectorType, np.ndarray[np.complex128]] # (kx_frac, ky_frac, kz_frac) -> total orbitals in the unit cell, size = (tot_grid_pt, )
IndexMaskDictType = \
    Dict[ShiftIndexVectorType, Tuple[np.ndarray[bool], np.ndarray[float]]] # (x_index_shift, y_index_shift, z_index_shift) -> (index_mask, effective_grid_point_positions_array)
AtomicWaveFunctionGeneratorType = \
    Optional[Callable[[int], Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], Dict[str, Any]]]] # atomic_number -> (r_array, orbitals, n_l_orbitals, info_dict)

# Warning messages
GEOPT_FNAME_PROVIDED_BUT_IS_RELAXATION_FALSE_WARNING = \
    "WARNING: geopt_fname is provided, but is_relaxation is False, so geopt_fname will be ignored"
STATIC_FNAME_PROVIDED_BUT_IS_RELAXATION_TRUE_WARNING = \
    "WARNING: static_fname is provided, but is_relaxation is True, so static_fname will be ignored"
LATTICE_VECTOR_NOT_NORMALIZED_WARNING = \
    "WARNING: Lattice vector direction {} is not normalized, norm = {}"
LATVEC_SECTION_NOT_FOUND_WARNING = \
    "Warning: Parameter 'LATVEC' section not found in output file, trying to determine the lattice type from LATVEC_SCALE, perceed as the lattice is orthogonal"
NOT_ORTHOGONALIZE_ATOMIC_ORBITALS_OVERLAP_MATRIX_NOT_PRINTED_WARNING = \
    "WARNING: Since we are not to orthogonalize the atomic orbitals, the overlap matrix is not printed"
XC_FUNCTIONAL_NOT_VALID_WARNING = \
    "WARNING: Currently, only the following XC functionals are supported: \n\t\t{}. \n\t Degrade to functional 'GGA_PBE' to generate the atomic orbitals for now.".format(', '.join(VALID_XC_FUNCTIONAL_LIST))

# PDOS output format
pdos_output_header_msg = \
"""
:Description: 
:Desc_PDOS: Projected density of states for each orbital. Unit=states/eV
:Desc_FERMI_LEVEL: Fermi level energy. Unit=eV
:Desc_BROADENING: Gaussian broadening parameter for DOS calculation. Unit=eV
:Desc_GRID_POINTS: Number of energy grid points for PDOS calculation
:Desc_MIN_E: Minimum energy for PDOS calculation. Unit=eV
:Desc_MAX_E: Maximum energy for PDOS calculation. Unit=eV
:Desc_BAND_NUM: Number of bands included in the calculation
:Desc_KPT_NUM: Number of k-points used in the calculation
:Desc_ATOM_TYPE_NUM: Number of atom types in the system
:Desc_TOT_ORBITALS: Total number of atomic orbitals
:Desc_PROJ_ORBITALS: Total number of projected orbitals
:Desc_VOLUME: Lattice unit cell volume. Unit=Bohr^3
:Desc_CALCULATION_TIME: Total calculation time. Unit=seconds
:Desc_M_INDEX_SUMMED_OVER: Whether the m index is summed over 
:Desc_ORTHOGONALIZE_ATOMIC_ORBITALS: Whether the atomic orbitals are orthogonalized
:Desc_ATOM_INDEX_FOR_PDOS: The index of the atom for which the PDOS is calculated
:Desc_PDOS_INFO: The information of the PDOS
    :Desc_ATOM_TYPE: The type of the atom
    :Desc_ATOMIC_NUMBER: The atomic number of the atom
    :Desc_ATOM_INDEX: The index of the atom
    :Desc_ATOM_POSITION_CARTESIAN: The position of the atom in Cartesian coordinates
    :Desc_ATOM_POSITION_FRACTIONAL: The fractional position of the atom
    :Desc_PDOS_UNIT: The unit of the PDOS value
    :Desc_HEADER_FORMAT: The format of the header


:BASIC_INFO:
:FERMI_LEVEL: {fermi_level:.3f}
:BROADENING: {broadening:.3f}
:GRID_POINTS: {grid_points}
:MIN_E: {min_energy:.3f}
:MAX_E: {max_energy:.3f}
:BAND_NUM: {bands}
:KPT_NUM: {kpoints}
:ATOM_TYPE_NUM: {atom_type_num}
:TOT_ORBITALS: {orbitals}
:PROJ_ORBITALS: {proj_orbitals}
:VOLUME: {volume:.3f}
:CALCULATION_TIME: {calculation_time:.3f}
:M_INDEX_SUMMED_OVER: {m_index_summed_over}
:ORTHOGONALIZE_ATOMIC_ORBITALS: {orthogonalize_atomic_orbitals}
:ATOM_INDEX_FOR_PDOS: {atom_index_for_pdos}


"""

pdos_info_msg = \
"""
:PDOS_INFO:
    :ATOM_TYPE: {atom_type}
    :ATOMIC_NUMBER: {atomic_number}
    :ATOM_INDEX: {atom_index}
    :ATOM_POSITION_CARTESIAN: {atom_position_cartesian}
    :ATOM_POSITION_FRACTIONAL: {atom_position_fractional}
    :PDOS_UNIT: {pdos_unit}
    :HEADER_FORMAT: {header_format}
"""




def gauss_distribution(x, s):
    return np.exp(-x**2 / (s ** 2)) / (s * np.sqrt(np.pi))


def _compute_pdos_dos(
    eig_eV,                   # shape (band, kpt), in eV
    DATA_projections,         # list/array length K, each element (band, M) complex coefficients P_{n,k,mu}
    kpt_wts_store,            # shape (kpt,)
    mu_eV,                    # broadening parameter, in eV
    N_PDOS,                   # number of energy grid points
    min_E_plot, 
    max_E_plot, 
    
    spin_deg=2.0,             # spin degeneracy factor
    kpt_mesh_shape=None       # if need to keep the MATLAB's / (kpt1*kpt2*kpt3), pass a (kpt1,kpt2,kpt3)
    ):
    """
    Return:
      E: (N,) eV
      PDOS_plot: (N, M) projected DOS with atomic orbitals as columns
      DOS: (N,) total DOS
    """
    bands, K = eig_eV.shape
    # assemble P as (K, band, M)
    # print("DATA_projections shape = ", DATA_projections.shape)
    # print("mean of DATA_projections = ", np.mean(DATA_projections))

    P_list = [np.asarray(Pk) for Pk in DATA_projections]  # each (band, M)
    M = P_list[0].shape[1]
    P = np.stack(P_list, axis=0)                          # (K, band, M)

    # energy grid
    if (min_E_plot is None) and (max_E_plot is None):
        Emin = np.min(eig_eV) - 5.0 * mu_eV
        Emax = np.max(eig_eV) + 5.0 * mu_eV
    else:
        Emin = min_E_plot
        Emax = max_E_plot

    E = np.linspace(Emin, Emax, N_PDOS)

    # accumulate PDOS
    PDOS_plot = np.zeros((N_PDOS, M), dtype=float)
    DOS = np.zeros((N_PDOS,), dtype=float)

    for i in range(K):
        lam = eig_eV[:, i]                     # (band,)
        w   = float(kpt_wts_store[i])
        G = gauss_distribution(E[:, None] - lam[None, :], mu_eV)  # (N, band)

        # PDOS contribution: G (N×band) × |P|^2 (band×M)  → (N×M)
        W = np.abs(P[i])**2                    # (band, M)
        PDOS_plot += w * (G @ W)

        # DOS contribution: each band has a Gaussian, weight w
        DOS += w * np.sum(G, axis=1)

    # normalization
    total_w = np.sum(kpt_wts_store)
    PDOS_plot *= (spin_deg / total_w)
    DOS       *= (spin_deg / total_w)

    return E, PDOS_plot, DOS


def _find_last_line_index(keyword, lines: List[str]):
    for i in reversed(range(len(lines))):
        if keyword in lines[i]:
            return i
    return -1


def read_psi(filename, endian="<", verbose=True):
    """
    Read the binary wave function file (.psi) in the same layout as the MATLAB version.

    Parameters
    ----------
    filename : str
        file path
    endian : str
        byte order, default is little endian("<"); if the file is written by a big endian machine, use ">"
    verbose : bool
        whether to print the information consistent with the MATLAB version

    Returns
    -------
    psi : np.ndarray
        array of shape (Nd * Nspinor_eig, nband, nkpt),
        it is arranged as (spinor-blocked, band, kpt)
    header : dict
        header information (Nx, Ny, Nz, Nd, dx, dy, dz, dV, Nspinor_eig, isGamma, nspin, nkpt, nband)
    per_band_meta : list[dict]
        metadata for each (kpt, band): spin_index, kpt_index, kpt_vec(3,), band_indx
        the index order is outer loop of kpt, inner loop of band
    """
    def _read(f, dtype, count=1):
        # read count elements from the file, return a numpy array
        dt = np.dtype(dtype).newbyteorder(endian)
        nbytes = dt.itemsize * count
        buf = f.read(nbytes)
        if len(buf) != nbytes:
            raise EOFError("Unexpected end of file while reading binary data.")
        return np.frombuffer(buf, dtype=dt, count=count)

    with open(filename, "rb") as f:
        Nx   = int(_read(f, np.int32)[0])
        Ny   = int(_read(f, np.int32)[0])
        Nz   = int(_read(f, np.int32)[0])
        Nd   = int(_read(f, np.int32)[0])

        dx   = float(_read(f, np.float64)[0])
        dy   = float(_read(f, np.float64)[0])
        dz   = float(_read(f, np.float64)[0])
        dV   = float(_read(f, np.float64)[0])

        Nspinor_eig = int(_read(f, np.int32)[0])
        isGamma     = int(_read(f, np.int32)[0])
        nspin       = int(_read(f, np.int32)[0])
        nkpt        = int(_read(f, np.int32)[0])
        nband       = int(_read(f, np.int32)[0])

        assert Nd == Nx * Ny * Nz, "Nd != Nx*Ny*Nz"

        if verbose:
            print(f" Nx {Nx}, Ny {Ny}, Nz {Nz}\n"
                  f" dx {dx},dy {dy}, dz {dz}\n"
                  f" dV {dV}, isgamma {isGamma}, Nspinor_eig {Nspinor_eig}\n"
                  f" nspin {nspin}, nkpt {nkpt}, nband {nband}")

        psi = np.zeros((Nd * Nspinor_eig, nband, nkpt), dtype=np.complex128 if not isGamma else np.float64)
        per_band_meta = []

        for kpt in range(nkpt):
            for band in range(nband):
                spin_index = int(_read(f, np.int32)[0])
                kpt_index  = int(_read(f, np.int32)[0])
                kpt_vec    = _read(f, np.float64, 3).astype(float)
                band_indx  = int(_read(f, np.int32)[0])

                if verbose:
                    print(f"extracting spin_indx {spin_index}, kpt_indx {kpt_index}, "
                          f"kpt_vec {kpt_vec[0]:.6f} {kpt_vec[1]:.6f} {kpt_vec[2]:.6f},band_indx {band_indx}")

                per_band_meta.append({
                    "spin_index": spin_index,
                    "kpt_index": kpt_index,
                    "kpt_vec": kpt_vec,
                    "band_indx": band_indx,
                })

                # fill psi
                for spinor in range(Nspinor_eig):
                    start = spinor * Nd
                    end   = (spinor + 1) * Nd

                    if isGamma:
                        # real number: fread([1 Nd],'double')' -> column vector in MATLAB
                        arr = _read(f, np.float64, Nd)
                        # directly put the column vector into psi
                        psi[start:end, band, kpt] = arr
                    else:
                        # complex number: fread([2 Nd],'double') -> 2×Nd(column-major) in MATLAB, then complex(row1, row2).'
                        raw = _read(f, np.float64, 2 * Nd)
                        # restore to (2, Nd) column-major shape (simulate MATLAB column-major)
                        raw = np.reshape(raw, (2, Nd), order='F')
                        arr = raw[0, :] + 1j * raw[1, :]
                        psi[start:end, band, kpt] = arr

    header = dict(
        Nx=Nx, Ny=Ny, Nz=Nz, Nd=Nd,
        dx=dx, dy=dy, dz=dz, dV=dV,
        Nspinor_eig=Nspinor_eig, isGamma=isGamma,
        nspin=nspin, nkpt=nkpt, nband=nband
    )
    return psi, header, per_band_meta



def get_psp_fname_list(output_fname : str):
    assert isinstance(output_fname, str), OUTPUT_FNAME_NOT_STRING_ERROR.format(type(output_fname))
    assert os.path.exists(output_fname), OUTPUT_FNAME_NOT_EXIST_ERROR.format(output_fname)

    output_dir = Path(output_fname).parent
    
    with open(output_fname, 'r') as f:
        lines = f.read().splitlines()

    psp_fname_list = []
    for line in lines:
        if "Pseudopotential" in line.split():
            psp_fname_list.append(os.path.join(output_dir, line.split()[-1]))

    return psp_fname_list


def render_progress(completed: int, total: int, start_ts: float, bar_len: int = 28) -> str:
    """Render a single progress bar"""
    pct = completed / total if total else 0.0
    filled = int(pct * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.time() - start_ts
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else math.inf
    eta_str = f"{eta:>6.1f}s" if math.isfinite(eta) else "  inf s"
    return f"[{bar}] {completed}/{total} ({pct:>6.2%}) | elapsed {elapsed:>6.1f}s | eta {eta_str}"



def spherical_harmonics(X, Y, Z, l, m):
    """
    RealSphericalHarmonics calculates real spherical harmonics

    Parameters
    ----------
    X : np.ndarray[float]
        The x coordinates of the positions 
    Y : np.ndarray[float]
        The y coordinates of the positions 
    Z : np.ndarray[float]
        The z coordinates of the positions 
    l : int
        The azimuthal quantum number
    m : int
        The magnetic quantum number

    Returns
    -------
    Ylm : np.ndarray[float]
        The real spherical harmonics at the given positions (X,Y,Z)

    """
    assert isinstance(l, int), "l must be an integer"
    assert isinstance(m, int), "m must be an integer"
    assert l >= 0, "l must be greater than or equal to 0"
    assert -l <= m <= l, "m must be between -l and l"

    if l > 2:
        raise ValueError("Only <l> less than or equal to 2 supported.")

    r = np.sqrt(X**2 + Y**2 + Z**2)

    if l == 0:
        # l=0
        C00 = 0.282094791773878    # 0.5*sqrt(1/pi)
        Ylm = C00 * np.ones_like(X)

    elif l == 1:
        # l=1
        C1m1 = 0.488602511902920   # sqrt(3/(4*pi))
        C10  = 0.488602511902920   # sqrt(3/(4*pi))
        C1p1 = 0.488602511902920   # sqrt(3/(4*pi))
        if m == -1:
            Ylm = C1m1 * (Y / r)
        elif m == 0:
            Ylm = C10  * (Z / r)
        elif m == 1:
            Ylm = C1p1 * (X / r)

    elif l == 2:
        # l=2
        C2m2 = 1.092548430592079   # 0.5*sqrt(15/pi)
        C2m1 = 1.092548430592079   # 0.5*sqrt(15/pi)
        C20  = 0.315391565252520   # 0.25*sqrt(5/pi)
        C2p1 = 1.092548430592079   # 0.5*sqrt(15/pi)
        C2p2 = 0.546274215296040   # 0.25*sqrt(15/pi)

        r2 = r * r
        if m == -2:
            Ylm = C2m2 * (X * Y) / r2
        elif m == -1:
            Ylm = C2m1 * (Y * Z) / r2
        elif m == 0:
            Ylm = C20 * (-X**2 - Y**2 + 2*Z**2) / r2
        elif m == 1:
            Ylm = C2p1 * (Z * X) / r2
        elif m == 2:
            Ylm = C2p2 * (X**2 - Y**2) / r2

    elif l == 3:
        C3m3 =  0.590043589926644;  # 0.25*sqrt(35/(2*pi))  
        C3m2 = 2.890611442640554;   # 0.5*sqrt(105/(pi))
        C3m1 = 0.457045799464466;   # 0.25*sqrt(21/(2*pi))
        C30 =  0.373176332590115;   # 0.25*sqrt(7/pi)
        C3p1 =  0.457045799464466;  # 0.25*sqrt(21/(2*pi))
        C3p2 = 1.445305721320277;   # 0.25*sqrt(105/(pi))
        C3p3 = 0.590043589926644;   # 0.25*sqrt(35/(2*pi))
        r3 = r**3
        if m == -3:
            Ylm = C3m3*(3*X**2 - Y**2)*Y/r3
        elif m == -2:
            Ylm = C3m2*(X*Y*Z)/r3
        elif m == -1:
            Ylm = C3m1*Y*(4*Z**2 - X**2 - Y**2)/r3
        elif m == 0:
            Ylm = C30*Z*(2*Z**2-3*X**2-3*Y**2)/r3
        elif m == 1:
            Ylm = C3p1*X*(4*Z**2 - X**2 - Y**2)/r3
        elif m == 2:
            Ylm = C3p2*Z*(X**2 - Y**2)/r3
        elif m == 3:
            Ylm = C3p3*X*(X**2-3*Y**2)/r3
    
    elif l == 4:
        r4 = r**4
        if m == -4:
            Ylm = (3.0/4.0)*np.sqrt(35.0/np.pi)*(X*Y*(X**2 - Y**2))/r4
        elif m == -3:
            Ylm = (3.0/4.0)*np.sqrt(35.0/(2.0*np.pi))*((3.0*X**2 - Y**2)*Y*Z)/r4
        elif m == -2:
            Ylm = (3.0/4.0)*np.sqrt(5.0/np.pi)*(X*Y*(7.0*Z**2 - r**2))/r4
        elif m == -1:
            Ylm = (3.0/4.0)*np.sqrt(5.0/(2.0*np.pi))*(Y*Z*(7.0*Z**2 - 3.0*r**2))/r4
        elif m == 0:
            Ylm = (3.0/16.0)*np.sqrt(1.0/np.pi)*(35.0*Z**4 - 30.0*Z**2*r**2 + 3.0*r**4)/r4
        elif m == 1:
            Ylm = (3.0/4.0)*np.sqrt(5.0/(2.0*np.pi))*(X*Z*(7.0*Z**2 - 3.0*r**2))/r4
        elif m == 2:
            Ylm = (3.0/8.0)*np.sqrt(5.0/np.pi)*((X**2 - Y**2)*(7.0*Z**2 - r**2))/r4
        elif m == 3:
            Ylm = (3.0/4.0)*np.sqrt(35.0/(2.0*np.pi))*((X**2 - 3.0*Y**2)*X*Z)/r4
        elif m == 4:
            Ylm = (3.0/16.0)*np.sqrt(35.0/np.pi)*((X**2*(X**2 - 3.0*Y**2) - Y**2*(3.0*X**2 - Y**2)))/r4

    elif l == 5:
        r5 = r**5
        p  = np.sqrt(X**2 + Y**2)
        p4 = p**4
        if m == -5:
            Ylm = (3.0*np.sqrt(2.0*77.0/np.pi)/32.0)*(8.0*X**4*Y - 4.0*X**2*Y**3 + 4.0*Y**5 - 3.0*Y*p4)/r5
        elif m == -4:
            Ylm = (3.0/16.0)*np.sqrt(385.0/np.pi)*(4.0*X**3*Y - 4.0*X*Y**3)*Z/r5
        elif m == -3:
            Ylm = (np.sqrt(2.0*385.0/np.pi)/32.0)*((3.0*Y*p*p - 4.0*Y**3)*(9.0*Z**2 - r**2))/r5
        elif m == -2:
            Ylm = (1.0/8.0)*np.sqrt(1155.0/np.pi)*(2.0*X*Y*(3.0*Z**3 - Z*r**2))/r5
        elif m == -1:
            Ylm = (1.0/16.0)*np.sqrt(165.0/np.pi)*Y*(21.0*Z**4 - 14.0*r**2*Z**2 + r**4)/r5
        elif m == 0:
            Ylm = (1.0/16.0)*np.sqrt(11.0/np.pi)*(63.0*Z**5 - 70.0*Z**3*r**2 + 15.0*Z*r**4)/r5
        elif m == 1:
            Ylm = (1.0/16.0)*np.sqrt(165.0/np.pi)*X*(21.0*Z**4 - 14.0*r**2*Z**2 + r**4)/r5
        elif m == 2:
            Ylm = (1.0/8.0)*np.sqrt(1155.0/np.pi)*((X**2 - Y**2)*(3.0*Z**3 - r**2*Z))/r5
        elif m == 3:
            Ylm = (np.sqrt(2.0*385.0/np.pi)/32.0)*((4.0*X**3 - 3.0*p*p*X)*(9.0*Z**2 - r**2))/r5
        elif m == 4:
            Ylm = (3.0/16.0)*np.sqrt(385.0/np.pi)*(4.0*(X**4 + Y**4) - 3.0*p4)*Z/r5
        elif m == 5:
            Ylm = (3.0*np.sqrt(2.0)/32.0)*np.sqrt(77.0/np.pi)*(4.0*X**5 + 8.0*X*Y**4 - 4.0*X**3*Y**2 - 3.0*X*p4)/r5

    elif l == 6:
        r6 = r**6
        p  = np.sqrt(X**2 + Y**2)
        p2 = p*p
        p4 = p2*p2
        if m == -6:
            Ylm = (np.sqrt(2.0*3003.0/np.pi)/64.0)*(12.0*X**5*Y + 12.0*X*Y**5 - 8.0*X**3*Y**3 - 6.0*X*Y*p4)/r6
        elif m == -5:
            Ylm = (3.0/32.0)*np.sqrt(2.0*1001.0/np.pi)*(8.0*X**4*Y - 4.0*X**2*Y**3 + 4.0*Y**5 - 3.0*Y*p4)*Z/r6
        elif m == -4:
            Ylm = (3.0/32.0)*np.sqrt(91.0/np.pi)*(4.0*X**3*Y - 4.0*X*Y**3)*(11.0*Z**2 - r**2)/r6
        elif m == -3:
            Ylm = (np.sqrt(2.0*1365.0/np.pi)/32.0)*(-4.0*Y**3 + 3.0*Y*p2)*(11.0*Z**3 - 3.0*Z*r**2)/r6
        elif m == -2:
            Ylm = (np.sqrt(2.0*1365.0/np.pi)/64.0)*(2.0*X*Y)*(33.0*Z**4 - 18.0*Z**2*r**2 + r**4)/r6
        elif m == -1:
            Ylm = (np.sqrt(273.0/np.pi)/16.0)*Y*(33.0*Z**5 - 30.0*Z**3*r**2 + 5.0*Z*r**4)/r6
        elif m == 0:
            Ylm = (np.sqrt(13.0/np.pi)/32.0)*(231.0*Z**6 - 315.0*Z**4*r**2 + 105.0*Z**2*r**4 - 5.0*r**6)/r6
        elif m == 1:
            Ylm = (np.sqrt(273.0/np.pi)/16.0)*X*(33.0*Z**5 - 30.0*Z**3*r**2 + 5.0*Z*r**4)/r6
        elif m == 2:
            Ylm = (np.sqrt(2.0*1365.0/np.pi)/64.0)*(X**2 - Y**2)*(33.0*Z**4 - 18.0*Z**2*r**2 + r**4)/r6
        elif m == 3:
            Ylm = (np.sqrt(2.0*1365.0/np.pi)/32.0)*(4.0*X**3 - 3.0*X*p2)*(11.0*Z**3 - 3.0*Z*r**2)/r6
        elif m == 4:
            Ylm = (3.0/32.0)*np.sqrt(91.0/np.pi)*(4.0*X**4 + 4.0*Y**4 - 3.0*p4)*(11.0*Z**2 - r**2)/r6
        elif m == 5:
            Ylm = (3.0/32.0)*np.sqrt(2.0*1001.0/np.pi)*(4.0*X**5 + 8.0*X*Y**4 - 4.0*X**3*Y**2 - 3.0*X*p4)*Z/r6
        elif m == 6:
            Ylm = (np.sqrt(2.0*3003.0/np.pi)/64.0)*(4.0*X**6 - 4.0*Y**6 + 12.0*X**2*Y**4 - 12.0*X**4*Y**2 + 3.0*Y**2*p4 - 3.0*X**2*p4)/r6

    else:
        # Shouldn't reach here due to the l>6 check
        raise ValueError("Only <l> less than or equal to 6 supported.")

    if l > 0:
        Ylm[r < 1e-9] = 0.

    return Ylm


def atomic_number_to_name(atomic_number):
    if atomic_number == 1: return "H"
    elif atomic_number == 2: return "He"
    elif atomic_number == 3: return "Li"
    elif atomic_number == 4: return "Be"
    elif atomic_number == 5: return "B"
    elif atomic_number == 6: return "C"
    elif atomic_number == 7: return "N"
    elif atomic_number == 8: return "O"
    elif atomic_number == 9: return "F"
    elif atomic_number == 10: return "Ne"
    elif atomic_number == 11: return "Na"
    elif atomic_number == 12: return "Mg"
    elif atomic_number == 13: return "Al"
    elif atomic_number == 14: return "Si"
    elif atomic_number == 15: return "P"
    elif atomic_number == 16: return "S"
    elif atomic_number == 17: return "Cl"
    elif atomic_number == 18: return "Ar"
    elif atomic_number == 19: return "K"
    elif atomic_number == 20: return "Ca"
    elif atomic_number == 21: return "Sc"
    elif atomic_number == 22: return "Ti"
    elif atomic_number == 23: return "V"
    elif atomic_number == 24: return "Cr"
    elif atomic_number == 25: return "Mn"
    elif atomic_number == 26: return "Fe"
    elif atomic_number == 27: return "Co"
    elif atomic_number == 28: return "Ni"
    elif atomic_number == 29: return "Cu"
    elif atomic_number == 30: return "Zn"
    elif atomic_number == 31: return "Ga"
    elif atomic_number == 32: return "Ge"
    elif atomic_number == 33: return "As"
    elif atomic_number == 34: return "Se"
    elif atomic_number == 35: return "Br"
    elif atomic_number == 36: return "Kr"
    elif atomic_number == 37: return "Rb"
    elif atomic_number == 38: return "Sr"
    elif atomic_number == 39: return "Y"
    elif atomic_number == 40: return "Zr"
    elif atomic_number == 41: return "Nb"
    elif atomic_number == 42: return "Mo"
    elif atomic_number == 43: return "Tc"
    elif atomic_number == 44: return "Ru"
    elif atomic_number == 45: return "Rh"
    elif atomic_number == 46: return "Pd"
    elif atomic_number == 47: return "Ag"
    elif atomic_number == 48: return "Cd"
    elif atomic_number == 49: return "In"
    elif atomic_number == 50: return "Sn"
    elif atomic_number == 51: return "Sb"
    elif atomic_number == 52: return "Te"
    elif atomic_number == 53: return "I"
    elif atomic_number == 54: return "Xe"
    elif atomic_number == 55: return "Cs"
    elif atomic_number == 56: return "Ba"
    elif atomic_number == 57: return "La"
    elif atomic_number == 58: return "Ce"
    elif atomic_number == 59: return "Pr"
    elif atomic_number == 60: return "Nd"
    elif atomic_number == 61: return "Pm"
    elif atomic_number == 62: return "Sm"
    elif atomic_number == 63: return "Eu"
    elif atomic_number == 64: return "Gd"
    elif atomic_number == 65: return "Tb"
    elif atomic_number == 66: return "Dy"
    elif atomic_number == 67: return "Ho"
    elif atomic_number == 68: return "Er"
    elif atomic_number == 69: return "Tm"
    elif atomic_number == 70: return "Yb"
    elif atomic_number == 71: return "Lu"
    elif atomic_number == 72: return "Hf"
    elif atomic_number == 73: return "Ta"
    elif atomic_number == 74: return "W"
    elif atomic_number == 75: return "Re"
    elif atomic_number == 76: return "Os"
    elif atomic_number == 77: return "Ir"
    elif atomic_number == 78: return "Pt"
    elif atomic_number == 79: return "Au"
    elif atomic_number == 80: return "Hg"
    elif atomic_number == 81: return "Tl"
    elif atomic_number == 82: return "Pb"
    elif atomic_number == 83: return "Bi"
    elif atomic_number == 84: return "Po"
    elif atomic_number == 85: return "At"
    elif atomic_number == 86: return "Rn"
    elif atomic_number == 87: return "Fr"
    elif atomic_number == 88: return "Ra"
    elif atomic_number == 89: return "Ac"
    elif atomic_number == 90: return "Th"
    elif atomic_number == 91: return "Pa"
    elif atomic_number == 92: return "U"
    elif atomic_number == 93: return "Np"
    else:
        raise ValueError(f"Atomic number {atomic_number} is not supported")


def name_to_atomic_number(name: str) -> int:
    if name == "H": return "01"
    elif name == "He": return "02"
    elif name == "Li": return "03"
    elif name == "Be": return "04"
    elif name == "B": return "05"
    elif name == "C": return "06"
    elif name == "N": return "07"
    elif name == "O": return "08"
    elif name == "F": return "09"
    elif name == "Ne": return "10"
    elif name == "Na": return "11"
    elif name == "Mg": return "12"
    elif name == "Al": return "13"
    elif name == "Si": return "14"
    elif name == "P": return "15"
    elif name == "S": return "16"
    elif name == "Cl": return "17"
    elif name == "Ar": return "18"
    elif name == "K": return "19"
    elif name == "Ca": return "20"      
    elif name == "Sc": return "21"
    elif name == "Ti": return "22"
    elif name == "V": return "23"
    elif name == "Cr": return "24"
    elif name == "Mn": return "25"
    elif name == "Fe": return "26"
    elif name == "Co": return "27"
    elif name == "Ni": return "28"
    elif name == "Cu": return "29"
    elif name == "Zn": return "30"
    elif name == "Ga": return "31"
    elif name == "Ge": return "32"
    elif name == "As": return "33"    
    elif name == "Se": return "34"
    elif name == "Br": return "35"
    elif name == "Kr": return "36"
    elif name == "Rb": return "37"
    elif name == "Sr": return "38"
    elif name == "Y": return "39"
    elif name == "Zr": return "40"
    elif name == "Nb": return "41"
    elif name == "Mo": return "42"
    elif name == "Tc": return "43"
    elif name == "Ru": return "44"
    elif name == "Rh": return "45"
    elif name == "Pd": return "46"
    elif name == "Ag": return "47"
    elif name == "Cd": return "48"
    elif name == "In": return "49"
    elif name == "Sn": return "50"
    elif name == "Sb": return "51"
    elif name == "Te": return "52"
    elif name == "I": return "53"
    elif name == "Xe": return "54"
    elif name == "Cs": return "55"
    elif name == "Ba": return "56"
    elif name == "La": return "57"
    elif name == "Ce": return "58"
    elif name == "Pr": return "59"
    elif name == "Nd": return "60"
    elif name == "Pm": return "61"
    elif name == "Sm": return "62"
    elif name == "Eu": return "63"
    elif name == "Gd": return "64"
    elif name == "Tb": return "65"  
    elif name == "Dy": return "66"
    elif name == "Ho": return "67"
    elif name == "Er": return "68"
    elif name == "Tm": return "69"
    elif name == "Yb": return "70"
    elif name == "Lu": return "71"
    elif name == "Hf": return "72"
    elif name == "Ta": return "73"
    elif name == "W": return "74"
    elif name == "Re": return "75"
    elif name == "Os": return "76"
    elif name == "Ir": return "77"  
    elif name == "Pt": return "78"
    elif name == "Au": return "79"
    elif name == "Hg": return "80"
    elif name == "Tl": return "81"
    elif name == "Pb": return "82"
    elif name == "Bi": return "83"
    elif name == "Po": return "84"
    elif name == "At": return "85"
    elif name == "Rn": return "86"
    elif name == "Fr": return "87"
    elif name == "Ra": return "88"
    elif name == "Ac": return "89"
    elif name == "Th": return "90"
    elif name == "Pa": return "91"
    elif name == "U": return "92"
    elif name == "Np": return "93"
    else:
        raise ValueError(f"Atomic number {name} is not supported")  


def number_to_spdf(number: int) -> str:
    assert number in [0, 1, 2, 3, 4, 5], "number must be in [0, 1, 2, 3, 4, 5]"
    return {
        0: "s",
        1: "p",
        2: "d",
        3: "f",
        4: "g",
        5: "h",
    }[number]


def get_default_generator_for_atomic_wave_function(xc_functional, psp_file_path):
    """
    Get default generator for atomic wave functions using AtomicDFTSolver.
    
    This function creates a generator that uses AtomicDFTSolver to compute
    atomic wave functions for a given atomic number.
    
    Args:
        XC_functional: Exchange-correlation functional name
        psp_dir_path: Path to pseudopotential directory
    """
    xc_functional = "123"
    if xc_functional not in VALID_XC_FUNCTIONAL_LIST:
        print(XC_FUNCTIONAL_NOT_VALID_WARNING)
        xc_functional = 'GGA_PBE'

    assert os.path.exists(psp_file_path), "psp_file_path {} does not exist".format(psp_file_path)

    # Add parent directory to path in order to import AtomicDFTSolver
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from atom.solver import AtomicDFTSolver
    except ImportError:
        raise ImportError("AtomicDFTSolver class is not found in the path. Make sure you have place the atom folder in the path {}".format(parent_dir))

    psp_dir_path  = os.path.dirname(psp_file_path)
    psp_file_name = os.path.basename(psp_file_path)

    def generator(atomic_number: int):
        print(f"\t Generating atomic wave function for atomic number {atomic_number}")
        atomic_dft_solver = AtomicDFTSolver(
            atomic_number = atomic_number,
            xc_functional = xc_functional,
            psp_dir_path  = psp_dir_path,
            psp_file_name = psp_file_name,
            all_electron_flag = False,
            print_debug       = False,
        )

        results = atomic_dft_solver.solve()

        grid            = results['uniform_grid']
        orbitals        = results['orbitals_on_uniform_grid']
        occupation_info = results['occupation_info']
        info_dict       = results

        n_l_orbitals = np.zeros((2, occupation_info.n_states))
        n_l_orbitals[:, 0] = occupation_info.occ_n
        n_l_orbitals[:, 1] = occupation_info.occ_l
        return grid, orbitals, n_l_orbitals, info_dict

    return generator



_G = {}

def _shm_from_ndarray(arr: np.ndarray):
    """Create shared memory from ndarray; return (meta, owner_shm)."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr  # copy once
    meta = {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
    return meta, shm  # remember to close+unlink at the end


def _ndarray_from_shm(meta):
    """Attach to shared memory and return (shm, ndarray view)."""

    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=meta["name"])
    view = np.ndarray(meta["shape"], dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    return shm, view



def _init_worker_for_single_atom(psi_meta, I_indices, kpts_store, latvec, latvec_inv,
                                  is_orthogonal_lattice, Lx, Ly, Lz, atom_packed, atoms,
                                  dV, mutiply_weights, start_idx, end_idx, kpt_num):
    """Initializer for single atom projection worker"""
    psi_shm, psi_view = _ndarray_from_shm(psi_meta)
    _G["psi_shm"] = psi_shm
    _G["psi_total"] = psi_view
    _G["I_indices"] = I_indices
    _G["kpts_store"] = kpts_store
    _G["latvec"] = latvec
    _G["latvec_inv"] = latvec_inv
    _G["is_orthogonal_lattice"] = is_orthogonal_lattice
    _G["Lx"], _G["Ly"], _G["Lz"] = Lx, Ly, Lz
    _G["atoms_packed"] = atom_packed  # packed single atom data
    _G["atoms"] = atoms  # full atoms list for calculate_orbital_weights
    _G["dV"] = dV
    _G["mutiply_weights"] = mutiply_weights
    _G["start_idx"] = start_idx
    _G["end_idx"] = end_idx
    _G["kpt_num"] = kpt_num


# ===== Unified worker for complete k-point processing (overlap + projection) =====
def _process_single_kpoint_complete_worker(kpt_index: int):
    """
    Unified worker function that processes a complete k-point:
    1. Compute overlap matrix or orbital weights
    2. Compute S^(-1/2) or w^(-1/2)
    3. Compute projection
    4. Apply orthogonalization/weights
    Returns the final P_single_kpt_array
    
    This is the parallel version of the inner k-point loop in sequential version.
    """
    import numpy as np
    from scipy.linalg import fractional_matrix_power
    
    psi_total = _G["psi_total"]
    I_indices = _G["I_indices"]
    kpts_store = _G["kpts_store"]
    atoms = _G["atoms"]
    dV = _G["dV"]
    orthogonalize = _G["orthogonalize_atomic_orbitals"]
    mutiply_weights = _G["mutiply_weights"]
    out_dirname = _G.get("out_dirname", None)
    print_overlap = _G.get("print_overlap_matrix", False)
    tot_orbital_num = _G.get("tot_orbital_num", 0)
    kpt_num = _G.get("kpt_num", 0)
    
    # Get k-point fractional coordinates
    kx_frac = kpts_store[kpt_index, 0]
    ky_frac = kpts_store[kpt_index, 1]
    kz_frac = kpts_store[kpt_index, 2]
    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac])
    
    # Get the wavefunction
    psi_kpt = psi_total[:, :, kpt_index]
    perm = I_indices[:, kpt_index].astype(int)
    psi_kpt = psi_kpt[:, perm]  # (tot_grid_pt, band_num)
    band_num = psi_kpt.shape[1]
    
    # ===== Step 1: Compute overlap matrix or orbital weights (reuse classmethod) =====
    if orthogonalize:
        save_fname = None
        if print_overlap and out_dirname is not None:
            save_fname = out_dirname + f"/overlap_matrix_kpt_{kpt_index}.txt"
        
        _overlap_matrix = PDOSCalculator.calculate_overlap_matrix(
            atoms=atoms,
            dV=dV,
            k_point_fractional=k_point_fractional,
            save_fname=save_fname,
        )
        _S_inv_sqrt = fractional_matrix_power(_overlap_matrix, -0.5)
    else:
        _orbital_weights = PDOSCalculator.calculate_orbital_weights(
            atoms=atoms,
            dV=dV,
            k_point_fractional=k_point_fractional,
        )
        _w_inv_sqrt = _orbital_weights ** (-0.5)
    
    # ===== Step 2: Compute projection =====
    P_single_kpt_array = np.zeros((band_num, tot_orbital_num), dtype=np.complex128)
    
    orbital_index = 0
    for atom in atoms:
        for orbital in atom.phi_orbitals_list:
            # Construct total orbital in unit cell with Bloch phases (use classmethod)
            _phi_orbital_unit_cell = PDOSCalculator.construct_total_orbitals_in_unit_cell(
                phi_orbital=orbital,
                k_point_fractional=k_point_fractional,
                index_mask_dict=atom.index_mask_and_effective_grid_point_positions_dict
            )
            
            # Compute projection
            projected_value = np.sum(_phi_orbital_unit_cell[:, np.newaxis] * psi_kpt, axis=0)
            
            # Compute normalization factor (use classmethod)
            normalization_factor = PDOSCalculator.compute_normalization_factor(
                dV=dV,
                phi_orbital=orbital,
                k_point_fractional=k_point_fractional,
                index_mask_dict=atom.index_mask_and_effective_grid_point_positions_dict
            )
            
            P_single_kpt_array[:, orbital_index] = projected_value * normalization_factor
            orbital_index += 1
    
    # ===== Step 3: Apply orthogonalization/weights =====
    if mutiply_weights:
        if orthogonalize:
            P_single_kpt_array = np.einsum("jk,kl->jl", P_single_kpt_array, _S_inv_sqrt)
        else:
            P_single_kpt_array = np.einsum("jk,k->jk", P_single_kpt_array, _w_inv_sqrt)
    
    print(f"\t kpt_index = {kpt_index + 1:>3d} calculation done!! ({kpt_num} k-points in total)")
    return kpt_index, P_single_kpt_array


# ===== Initializer for unified worker =====
def _init_unified_worker(psi_meta, I_indices, kpts_store, atoms, dV,
                        orthogonalize_atomic_orbitals, mutiply_weights,
                        out_dirname, print_overlap_matrix, tot_orbital_num, kpt_num):
    """Initialize worker for complete k-point processing."""
    psi_shm, psi_view = _ndarray_from_shm(psi_meta)
    _G["psi_shm"] = psi_shm
    _G["psi_total"] = psi_view
    _G["I_indices"] = I_indices
    _G["kpts_store"] = kpts_store
    _G["atoms"] = atoms
    _G["dV"] = dV
    _G["orthogonalize_atomic_orbitals"] = orthogonalize_atomic_orbitals
    _G["mutiply_weights"] = mutiply_weights
    _G["out_dirname"] = out_dirname
    _G["print_overlap_matrix"] = print_overlap_matrix
    _G["tot_orbital_num"] = tot_orbital_num
    _G["kpt_num"] = kpt_num



def _process_single_kpoint_for_single_atom_worker(kpt_index: int):
    import numpy as np
    psi_total = _G["psi_total"]
    I_indices = _G["I_indices"]
    kpts_store = _G["kpts_store"]
    latvec = _G["latvec"]
    latvec_inv = _G["latvec_inv"]
    is_orth = _G["is_orthogonal_lattice"]
    Lx, Ly, Lz = _G["Lx"], _G["Ly"], _G["Lz"]
    atom = _G["atoms_packed"]
    atoms_full = _G["atoms"]  # Need full atoms list for calculate_orbital_weights
    kpt_num = _G.get("kpt_num", 0)
    dV = _G.get("dV", 1.0)
    mutiply_weights = _G.get("mutiply_weights", True)
    start_idx = _G.get("start_idx", 0)
    end_idx = _G.get("end_idx", 0)

    # Get k-point fractional coordinates
    kx_frac = kpts_store[kpt_index, 0]
    ky_frac = kpts_store[kpt_index, 1]
    kz_frac = kpts_store[kpt_index, 2]
    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac])

    # ---- get the psi of the k-point (view, no copy) and reorder the bands ----
    psi_kpt = psi_total[:, :, kpt_index]
    perm = I_indices[:, kpt_index].astype(int)
    psi_kpt = psi_kpt[:, perm]   # (Nd, band_num)
    band_num = psi_kpt.shape[1]

    # Compute the orbital weights for this k-point (reuse classmethod)
    _orbital_weights = PDOSCalculator.calculate_orbital_weights(
        atoms=atoms_full,
        dV=dV,
        k_point_fractional=k_point_fractional,
    )
    _w_inv_sqrt = _orbital_weights ** (-0.5)

    # ---- result: shape (band_num, atom.num_orbitals) ----
    result = np.zeros((band_num, atom["num_orbitals"]), dtype=np.complex128)

    orbital_index = 0
    for orb in atom["orbitals"]:
        # Reconstruct phi_value_dict and index_mask_dict from entries
        phi_value_dict = {}
        index_mask_dict = {}
        for e in orb["entries"]:
            ijk = tuple(e["ijk"])
            phi_value_dict[ijk] = e["phi"]
            index_mask_dict[ijk] = (e["mask"], None)
        
        # Create temporary PhiOrbital object to use classmethod
        temp_orbital = PDOSCalculator.PhiOrbital(n=0, l=0, m=0)
        temp_orbital.phi_value_dict = phi_value_dict
        
        # Construct total orbital in unit cell with Bloch phases (use classmethod)
        _phi_orbital_unit_cell = PDOSCalculator.construct_total_orbitals_in_unit_cell(
            phi_orbital=temp_orbital,
            k_point_fractional=k_point_fractional,
            index_mask_dict=index_mask_dict
        )
        
        # Compute projection
        projected_value = np.sum(_phi_orbital_unit_cell[:, np.newaxis] * psi_kpt, axis=0)
        
        # Compute normalization factor for this k-point (use classmethod)
        normalization_factor = PDOSCalculator.compute_normalization_factor(
            dV=dV,
            phi_orbital=temp_orbital,
            k_point_fractional=k_point_fractional,
            index_mask_dict=index_mask_dict
        )
        
        result[:, orbital_index] = projected_value * normalization_factor
        orbital_index += 1

    # Apply weights to this k-point (same as sequential version)
    if mutiply_weights:
        result = np.einsum("jk,k->jk", result, _w_inv_sqrt[start_idx:end_idx])

    print(f"\t kpt_index = {kpt_index + 1:>3d} calculation done!! ({kpt_num} k-points in total)")
    return kpt_index, result


class PDOSCalculator:
    """
    PDOS Calculator Class - Python version of MATLAB PDOS code for SPARC output files
    """
    # Super parameters
    k_point_parallel_threshold : int = 4 # if the number of k-points is greater than this threshold, use parallel calculation
    
    # File / Directory names - upon input
    natom_types    : int       # number of atom types
    upf_fname_list : List[str] # list of upf file names
    output_fname   : str       # SPARC's output file name
    eigen_fname    : str       # eigen file name
    is_relaxation  : bool      # whether the result is from relaxation
    static_fname   : Optional[str] # static file name
    geopt_fname    : Optional[str] # geopt file name
    psi_fname      : str       # orbital file name, usually the .psi file output by SPARC
    out_dirname    : str       # PDOS output directory name


    # Parameters for PDOS calculation at initial stage
    r_cut_max                      : RCUT_MAX_TYPE                   # maximum cutoff radius for the atomic wave functions
    atomic_wave_function_tol       : float                           # tolerance for the atomic wave functions
    orthogonalize_atomic_orbitals  : bool                            # whether to orthogonalize the atomic orbitals
    atomic_wave_function_generator : AtomicWaveFunctionGeneratorType # function to generate the atomic wave functions
    print_overlap_matrix           : bool                            # whether to print the overlap matrix


    # Parameters for PDOS calculation at running stage
    mu_PDOS : float    # in eV, Gaussian width for DOS calculation
    N_PDOS  : int      # number of points in PDOS
    min_E_plot : float # min energy value (eV) for which PDOS is calculated
    max_E_plot : float # max energy value (eV) for which PDOS is calculated
    load_projection_data_from_txt_path : Optional[str] # path to the projection data file
    print_projection_data : bool # whether to print the projection data
    sum_over_m_index      : bool # whether to sum over the m index


    # Parameters from SPARC's output file (.out)
    ## The following parameters are used for both orthogonal and non-orthogonal lattices
    bcx : str      # boundary condition in the x-direction, P for periodic, D for Dirichlet
    bcy : str      # boundary condition in the y-direction, P for periodic, D for Dirichlet
    bcz : str      # boundary condition in the z-direction, P for periodic, D for Dirichlet
    band_num : int # number of bands
    kpt_num : int  # number of symmetry adapted k-points
    fermi : float  # Fermi level
    
    xc_functional : str # exchange-correlation functional
    psp_fname_list : List[str] # list of psp file names

    tot_atoms_num         : int        # total number of atoms
    atom_count_list       : List[int]  # number of atoms of each type
    is_orthogonal_lattice : bool       # whether the lattice is orthogonal
    relax_flag            : Optional[int] = None  # relaxation flag


    kpt1 : int    # number of k-points in the x-direction
    kpt2 : int    # number of k-points in the y-direction
    kpt3 : int    # number of k-points in the z-direction
    Nx : int      # number of grid points in the x-direction
    Ny : int      # number of grid points in the y-direction
    Nz : int      # number of grid points in the z-direction
    Lx : float    # lattice constant in the x-direction (in Bohr)
    Ly : float    # lattice constant in the y-direction (in Bohr)
    Lz : float    # lattice constant in the z-direction (in Bohr)

    dx : float    # grid spacing in the x-direction (in Bohr)
    dy : float    # grid spacing in the y-direction (in Bohr)
    dz : float    # grid spacing in the z-direction (in Bohr)

    x_lattice_vector : np.ndarray[float] # x-lattice vector
    y_lattice_vector : np.ndarray[float] # y-lattice vector
    z_lattice_vector : np.ndarray[float] # z-lattice vector

    ## If is_orthogonal_lattice is False, then the lattice is not orthogonal, and the following extra parameters are initialized later
    # Non-orthogonal lattice parameters
    latvec       : np.ndarray[float]      # 3x3 lattice vectors matrix (in Bohr)
    latvec_inv   : np.ndarray[float]      # inverse of lattice vectors matrix
    latvec_det   : float                  # determinant of lattice vectors matrix
    cell_volume  : float                  # unit cell volume (in Bohr^3)
    jacobian_det : float                  # Jacobian determinant for coordinate transformation
    metric_tensor     : np.ndarray[float] # metric tensor g_ij = a_i · a_j
    metric_tensor_inv : np.ndarray[float] # inverse of metric tensor


    # Parameters from UPF files (.upf)
    @dataclass
    class PseudoWaveFunction:
        n: int = 0     # principal quantum number
        l: int = 0     # angular momentum quantum number
        no_m: int = 0  # number of m quantum states
        chi  : List[float] = field(default_factory=list) # chi(r)
        r    : List[float] = field(default_factory=list) # r-grid
        phi_r: List[float] = field(default_factory=list) # phi(r)

    @dataclass
    class AtomicWaveFunction:
        atom_type_index: int              # atom type index
        atom_species: str                 # atom species
        upf_fname: str                    # upf file name
        r_cut: int                        # cutoff radius, no greater than 10.0 Bohr
        num_psdwfn: int                   # number of pseudo-wavefunctions
        num_orbitals: int                 # total number of orbitals
        psdwfn_list: List['PDOSCalculator.PseudoWaveFunction'] = field(default_factory=list)

    atomic_wave_function_list : List[AtomicWaveFunction] # list of atom wave functions


    # Parameters from SPARC's static file (.static) or geopt file (.geopt)
    @dataclass
    class Atom:
        atom_type_index: int              # atom type index
        atom_posn_frac: np.ndarray[float] # fractional coordinates of the atom
        atom_posn_cart: np.ndarray[float] # cartesian coordinates of the atom

        # Other parameters related to the grid-wise phi orbitals, will be updated when .run() is called.
        # For index_mask_and_r_norm_dict, the key is a tuple of 3 integers, which specifies the index shift of the unit cell in the x, y, z directions. 
        # (x_index_shift, y_index_shift, z_index_shift)
        # - For example, (0, 0, 0) means the grid points are in the unit cell with index (0, 0, 0), and index_mask specifies the grid points where the orbital takes non-zero values.
        # - Since the cutoff radius is fixed for each atom, the index_mask is the same for all atoms, even if the orbitals are different.
        # - r_norm_array is the distance between the atom and the grid points, in Bohr
        # index_mask : np.ndarray[bool]
        # - specifies the grid points where the orbital takes non-zero values.
        # - Use self.cartesian_gridwise_coord[index_mask] to get the cartesian coordinates of the grid points
        # r_norm_array : np.ndarray[float]
        # - the distance between the atom and the grid points, in Bohr
        index_mask_and_effective_grid_point_positions_dict: IndexMaskDictType = field(default_factory=dict) # (x_index_shift, y_index_shift, z_index_shift) -> (index_mask, effective_grid_point_positions_array)
        
        r_cut : float = 0.0 # cutoff radius, in Bohr
        num_psdwfn : int = 0 # number of pseudo wave functions
        num_orbitals : int = 0 # total number of orbitals
        phi_orbitals_list: List['PDOSCalculator.PhiOrbital'] = field(default_factory=list) # list of phi orbitals, len(phi_orbitals_list) = num_orbitals

    atoms : List[Atom]  # list of atoms


    # Data class for the phi orbitals, stored in the Atom.phi_orbitals_list
    @dataclass
    class PhiOrbital:
        n: int   # principal quantum number (optional, for bookkeeping)
        l: int   # angular momentum
        m: int   # magnetic quantum number

        # If the atom will not overlap with itself, then the normalization factor is 1.0, otherwise it should be calculated by considering different images inside the unit cell
        normalization_factor_dict : NormalizationFactorDictType = field(default_factory=dict)  # (x_index_shift, y_index_shift, z_index_shift) -> normalization factor for the PhiOrbital, this factor is needed when you place the atom inside the unit cell

        # total_orbitals_in_unit_cell : np.ndarray[np.complex128] # total orbitals in the unit cell, size = (tot_grid_pt, )
        total_orbitals_in_unit_cell_dict : TotalOrbitalsInUnitCellDictType = field(default_factory=dict) # (x_index_shift, y_index_shift, z_index_shift) -> total orbitals in the unit cell, size = (tot_grid_pt, )

        # Phi(r) values at the effective grid points, corresponding to the index of the grid points in Atom.index_mask_dict
        phi_value_dict : PhiValueDictType = field(default_factory=dict)  # (x_index_shift, y_index_shift, z_index_shift) -> phi_value



    # Parameters from SPARC's eigen file (.eigen)
    eign           : np.ndarray[float] # (totkpt, band_num) energies sorted ascending within each k
    occ            : np.ndarray[float] # (totkpt, band_num) occupations sorted according to eign
    I_indices      : np.ndarray[int]   # (band_num, totkpt) argsort indices for each k
    kpts_store     : np.ndarray[float] # (totkpt, 3) k-vectors
    kpt_wts_store  : np.ndarray[float] # (totkpt,)  weights * (kpt1*kpt2*kpt3)


    # Parameters from the SPARC's orbital file (.psi)
    psi_total : np.ndarray[float]         # (tot_grid_pt, band_num, totkpt)
    header    : Dict[str, Any]            # header information, some already parsed from the SPARC's output file (.out), while contains some other information like "isGamma", "Nspinor_eig", "nspin", etc.
    per_band_meta : List[Dict[str, Any]]  # Information for the bands in each k-point


    # Parameters related to the cartesian coordinates of the grid points, and its relations to the cut-off radius
    x_coord : np.ndarray[float] # x-coordinate of the unit cell
    y_coord : np.ndarray[float] # y-coordinate of the unit cell
    z_coord : np.ndarray[float] # z-coordinate of the unit cell
    x_shift_index_list : ShiftIndexListType # All x-shift indices that should be considered
    y_shift_index_list : ShiftIndexListType # All y-shift indices that should be considered
    z_shift_index_list : ShiftIndexListType # All z-shift indices that should be considered
    tot_grid_pt : int # total number of grid points in the unit cell
    cartesian_gridwise_coord : np.ndarray[float] # cartesian coordinates of all the grid points in the unit cell

    # Parameters for the PDOS calculation
    tot_orbital_num : int # total number of orbitals
    overlap_matrix : Optional[np.ndarray[float]] # overlap matrix of different orbitals
    S_inv_sqrt : Optional[np.ndarray[float]] # inverse square root of the overlap matrix
    data_projections : np.ndarray[float] # coefficients of the projection of the wavefunction onto the atomic orbital basis


    # Initialization flags
    out_file_loaded_flag        : bool = False # whether the SPARC's output file (.out) is loaded
    atomic_orbitals_loaded_flag : bool = False # whether the atomic orbitals are loaded, usually from the UPF files (.upf), but can be from other sources
    eigen_file_loaded_flag      : bool = False # whether the eigen file is loaded
    static_file_loaded_flag     : bool = False # whether the static file is loaded
    psi_file_loaded_flag        : bool = False # whether the psi file is loaded


    def __init__(self, 
                 upf_fname_list : Optional[List[str]], # list of upf file names
                 output_fname   : str,                 # SPARC's output file name
                 eigen_fname    : str,                 # eigen file name
                 static_fname   : Optional[str],       # static file name
                 psi_fname      : str,                 # orbital file name
                 out_dirname    : str,                 # PDOS output directory name
                 r_cut_max      : Union[List[float], float]     = 15.0,  # maximum cutoff radius for the atomic wave functions
                 atomic_wave_function_tol      : float          = 1e-5,   # tolerance for the atomic wave functions
                 orthogonalize_atomic_orbitals : bool           = False,  # whether to orthogonalize the atomic orbitals
                 is_relaxation                 : bool           = False,  # The result is from relaxation
                 geopt_fname                   : Optional[str]  = None,   # geopt file name
                 k_point_parallelization       : Optional[bool] = None,  # whether to use k-point parallelization
                 ):
        """
        1. Initialize the PDOSCalculator class
        """
        time1 = time.time()

        # output_fname check
        assert isinstance(output_fname, str), OUTPUT_FNAME_NOT_STRING_ERROR.format(type(output_fname))
        self.check_fname_existence(output_fname)
        self.natom_types, self.atom_species_list = self.parse_atom_species_from_sparc_out_file(output_fname = output_fname)

        # upf_fname_list check
        self.check_upf_fname_list_or_get_default_atomic_wave_function_generators(
            upf_fname_list = upf_fname_list, 
            natom_types = self.natom_types)

        # natom_types check
        assert isinstance(self.natom_types, int), NATOM_TYPE_NOT_INTEGER_ERROR.format(type(self.natom_types))


        # eigen_fname check
        assert isinstance(eigen_fname, str), EIGEN_FNAME_NOT_STRING_ERROR.format(type(eigen_fname))
        self.check_fname_existence(eigen_fname)

    
        # static_fname / geopt_fname check
        assert isinstance(is_relaxation, bool), IS_RELAXATION_NOT_BOOL_ERROR.format(type(is_relaxation))
        if not is_relaxation:
            assert isinstance(static_fname, str), STATIC_FNAME_NOT_STRING_ERROR.format(type(static_fname))
            self.check_fname_existence(static_fname)
            if geopt_fname is not None:
                print(GEOPT_FNAME_PROVIDED_BUT_IS_RELAXATION_FALSE_WARNING)
        else:
            assert isinstance(geopt_fname, str), GEOPT_FNAME_NOT_STRING_ERROR.format(type(geopt_fname))
            self.check_fname_existence(geopt_fname)
            if static_fname is not None:
                print(STATIC_FNAME_PROVIDED_BUT_IS_RELAXATION_TRUE_WARNING)

        # psi_fname check
        assert isinstance(psi_fname, str), ORBITAL_FNAME_NOT_STRING_ERROR.format(type(psi_fname))
        self.check_fname_existence(psi_fname)   

        # out_dirname check
        assert isinstance(out_dirname, str), OUT_DIRNAME_NOT_STRING_ERROR.format(type(out_dirname))
        if os.path.exists(out_dirname):
            shutil.rmtree(out_dirname)
        os.makedirs(out_dirname)

        # r_cut_max and atomic_wave_function_tol check
        assert isinstance(r_cut_max, (float, list)), R_CUT_MAX_NOT_FLOAT_OR_LIST_ERROR.format(type(r_cut_max))
        if isinstance(r_cut_max, list):
            assert len(r_cut_max) == self.natom_types, R_CUT_MAX_NUM_NOT_EQUAL_TO_NATOM_TYPES_ERROR.format(len(r_cut_max))
        assert isinstance(atomic_wave_function_tol, float), ATOMIC_WAVE_FUNCTION_TOL_NOT_FLOAT_ERROR.format(type(atomic_wave_function_tol))
        assert atomic_wave_function_tol > 0.0, ATOMIC_WAVE_FUNCTION_TOL_NOT_POSITIVE_ERROR.format(atomic_wave_function_tol)

        # orthogonalize_atomic_orbitals check
        assert isinstance(orthogonalize_atomic_orbitals, bool), ORTHOGONALIZE_ATOMIC_ORBITALS_NOT_BOOL_ERROR.format(type(orthogonalize_atomic_orbitals))


        # k_point_parallelization check
        if k_point_parallelization is not None:
            assert isinstance(k_point_parallelization, bool), K_POINT_PARALLELIZATION_NOT_BOOL_ERROR.format(type(k_point_parallelization))
            if k_point_parallelization:
                self.k_point_parallel_threshold = 1
            else:
                self.k_point_parallel_threshold = 1e9


        # set input parameters - file names
        self.output_fname   : str                 = output_fname
        self.upf_fname_list : Optional[List[str]] = upf_fname_list
        self.eigen_fname    : str                 = eigen_fname
        self.static_fname   : Optional[str]       = static_fname
        self.geopt_fname    : Optional[str]       = geopt_fname
        self.psi_fname      : str                 = psi_fname
        self.out_dirname    : str                 = out_dirname
        self.is_relaxation  : bool                = is_relaxation

        # set input parameters - PDOS calculation parameters
        self.r_cut_max      : Union[List[float], float] = r_cut_max
        self.atomic_wave_function_tol      : float = atomic_wave_function_tol
        self.orthogonalize_atomic_orbitals : bool  = orthogonalize_atomic_orbitals


        """
        2. Read in the SPARC's output file as program's input
        """
        time2 = time.time()


        self.eV2Ha = 1/27.21140795
        self.Ha2eV = 27.21140795

        # read in the SPARC's output file (.out)
        self.read_sparc_out_file_parameters(fname = output_fname)

        
        # read in the upf files (.upf), if provided
        self.atomic_wave_function_list : List[PDOSCalculator.AtomicWaveFunction] = []
        if self.upf_fname_list is not None:
            for index, upf_fname in enumerate(self.upf_fname_list):
                if upf_fname == "Default":
                    atomic_number = int(name_to_atomic_number(self.atom_species_list[index]))
                    atomic_wave_function_generator = get_default_generator_for_atomic_wave_function(xc_functional = self.xc_functional, psp_file_path = self.psp_fname_list[index])
                    atomic_wave_function = self.read_atomic_wave_function_from_atomic_wave_function_generator(
                        atomic_wave_function_generator = atomic_wave_function_generator, 
                        atom_type_index = index, 
                        atomic_number   = atomic_number, 
                        atom_species    = self.atom_species_list[index])
                else:
                    atomic_wave_function = self.read_atomic_wave_function_from_upf_file(fname = upf_fname, atom_type_index = index, atom_species = self.atom_species_list[index])
                self.atomic_wave_function_list.append(atomic_wave_function)
        else:
            for index, atom_species in enumerate(self.atom_species_list):
                atomic_number = int(name_to_atomic_number(atom_species))
                atomic_wave_function_generator = get_default_generator_for_atomic_wave_function(xc_functional = self.xc_functional, psp_file_path = self.psp_fname_list[index])
                atomic_wave_function = self.read_atomic_wave_function_from_atomic_wave_function_generator(
                    atomic_wave_function_generator = atomic_wave_function_generator, 
                    atom_type_index = index, 
                    atomic_number   = atomic_number, 
                    atom_species    = atom_species)
                self.atomic_wave_function_list.append(atomic_wave_function)



        # read in the SPARC's static file (.static)
        if not self.is_relaxation:
            self.read_sparc_static_file_parameters(fname = static_fname, atom_count_list = self.atom_count_list)
        else:
            self.read_sparc_geopt_file_parameters(fname = geopt_fname, atom_count_list = self.atom_count_list)



        # read in the SPARC's eigen file (.eigen)
        self.read_sparc_eigen_file_parameters(fname = eigen_fname)


        # # read in the SPARC's orbital file (.psi)
        self.read_sparc_orbital_file_parameters(fname = psi_fname)



        """
        3. Get cartesian coordinates and other information of the unit cell
        """
        time3 = time.time()
        self.get_cartesian_coordinates_for_the_unit_cell()


        """
        4. Update the atom's orbital information
        """
        time4 = time.time()

        for atom in self.atoms:
            self.atom_wise_compute_index_mask_and_effective_grid_point_positions_dict(atom = atom)
            self.atom_wise_compute_grid_wise_phi_orbitals(atom = atom)
            # self.atom_wise_compute_normalization_factor_in_unit_cell(atom = atom)
        self.tot_orbital_num = self.get_total_number_of_orbitals(atoms = self.atoms)


        time_final = time.time()

        self.time_init = time1
        self.step1_time = time2 - time1
        self.step2_time = time3 - time2
        self.step3_time = time4 - time3
        self.step4_time = time_final - time4



    def run(self, 
            mu_PDOS : float = 0.2721140795,  # in eV, Gaussian width for DOS calculation
            N_PDOS  : int   = 1000,          # number of points in PDOS
            min_E_plot : Optional[float] = None,       # min energy value (eV) for which PDOS is calculated
            max_E_plot : Optional[float] = None,       # max energy value (eV) for which PDOS is calculated
            load_projection_data_from_txt_path : Optional[str] = None,
            print_projection_data = False,
            sum_over_m_index = True,
            print_overlap_matrix = False,
            ):
        
        # set the min_E_plot and max_E_plot
        if min_E_plot is None:
            min_E_plot = self.eign.min() * self.Ha2eV - 5.0 * mu_PDOS
        if max_E_plot is None:
            max_E_plot = self.eign.max() * self.Ha2eV + 5.0 * mu_PDOS

        # type check
        assert isinstance(mu_PDOS, float), MU_PDOS_NOT_FLOAT_ERROR.format(type(mu_PDOS))
        assert isinstance(N_PDOS, int), N_PDOS_NOT_INTEGER_ERROR.format(type(N_PDOS))
        assert isinstance(min_E_plot, float), MIN_E_PLOT_NOT_FLOAT_ERROR.format(type(min_E_plot)) # in eV
        assert isinstance(max_E_plot, float), MAX_E_PLOT_NOT_FLOAT_ERROR.format(type(max_E_plot)) # in eV
        if load_projection_data_from_txt_path is not None:
            assert isinstance(load_projection_data_from_txt_path, str), LOAD_PROJECTION_DATA_FROM_TXT_PATH_NOT_STRING_ERROR.format(type(load_projection_data_from_txt_path))
            assert os.path.exists(load_projection_data_from_txt_path), LOAD_PROJECTION_DATA_FROM_TXT_PATH_NOT_EXIST_ERROR.format(load_projection_data_from_txt_path)

        self.mu_PDOS        : float     = mu_PDOS
        self.N_PDOS         : int       = N_PDOS
        self.min_E_plot     : float     = min_E_plot
        self.max_E_plot     : float     = max_E_plot

        """
        5. Project wavefuntions onto the atomic orbital basis.
        """
        time5 = time.time()
        if load_projection_data_from_txt_path is not None:
            # projection_path = self.out_dirname + "/data_projections.txt"
            assert os.path.exists(load_projection_data_from_txt_path), "Data projections file not found at {}".format(load_projection_data_from_txt_path)
            data_projections, kpts_red = self.load_projections_from_txt(path = load_projection_data_from_txt_path)
            print("\t data_projections.shape = ", data_projections.shape)
        else:
            data_projections = self.project_wavefunction_onto_atomic_orbital_basis_and_return_the_corresponding_coefficients(
                orthogonalize_atomic_orbitals = self.orthogonalize_atomic_orbitals,
                mutiply_weights = True,
                save_fname = self.out_dirname + "/data_projections.txt" if print_projection_data else None,
                print_overlap_matrix = print_overlap_matrix,
                )
                

        # self.calculate_pdos(overlap_matrix = self.overlap_matrix)
        """
        6. Compute Projected Density of States (PDOS), and output the PDOS and DOS to the output directory
        """
        time6 = time.time()

        E, PDOS_plot, DOS = self.compute_pdos_dos(data_projections = data_projections)

        # initialize the pdos_header
        dos_header = "Energy(eV)   DOS"

        # sum over the m index
        if sum_over_m_index:
            PDOS_plot = self._sum_over_m_index(PDOS_plot = PDOS_plot)
            column_header_format = "(nl)"
        else:
            column_header_format = "(nl,m)"


        print("Saving the PDOS and DOS to the output directory")

        np.savetxt(self.out_dirname + "/DOS.txt",
                np.column_stack([E, DOS]),
                header = dos_header, comments='', fmt="%.10f")

        with open(self.out_dirname + "/PDOS.txt", "w") as f:
            f.write(pdos_output_header_msg.format(
                fermi_level = self.fermi,
                broadening = self.mu_PDOS,
                grid_points = self.N_PDOS,
                min_energy = self.min_E_plot,
                max_energy = self.max_E_plot,
                bands = self.band_num,
                kpoints = self.kpt_num,
                atom_type_num = self.natom_types,
                orbitals = self.tot_orbital_num,
                proj_orbitals = len(PDOS_plot[0]),
                volume = self.cell_volume,
                calculation_time = time6 - self.time_init,
                m_index_summed_over = sum_over_m_index,
                atom_index_for_pdos = "All",
                column_header_format = column_header_format,
                orthogonalize_atomic_orbitals = self.orthogonalize_atomic_orbitals,))

        for atom_index, atom in enumerate(self.atoms):
            start_idx, end_idx = self._get_atom_orbital_indices(target_atom = atom, sum_over_m_index = sum_over_m_index)
            pdos_header, pdos_info_str = self._get_pdos_header_for_single_atom(
                sum_over_m_index = sum_over_m_index, 
                atom_of_interest = atom,
                )

            with open(self.out_dirname + "/PDOS.txt", "a") as f:
                atom_name = self.atom_species_list[atom.atom_type_index]
                f.write(pdos_info_msg.format(
                    atom_type = atom_name,
                    atomic_number = name_to_atomic_number(atom_name),
                    atom_index = atom_index,
                    atom_position_cartesian = atom.atom_posn_cart,
                    atom_position_fractional = atom.atom_posn_frac,
                    pdos_unit = "states/eV",
                    header_format = column_header_format,))
                np.savetxt(f,
                    np.column_stack([E, PDOS_plot[:, start_idx:end_idx]]),
                    header = pdos_header, 
                    comments='', 
                    fmt='\t%.10f',)

        time_final = time.time()

        # print("DOS.shape = ", DOS.shape)
        # print("DOS.sum() = ", DOS.sum())
        # print("DOS.sum() * (E[1] - E[0]) = ", DOS.sum() * (E[1] - E[0]))


        self.step5_time = time6 - time5
        self.step6_time = time_final - time6

        self.print_time_taken(load_projection_data_from_txt_path = load_projection_data_from_txt_path)

        return E, PDOS_plot, DOS


    def _sum_over_m_index(self, PDOS_plot : np.ndarray):
        print("summing over the m index")

        atom_nl_index_lists : List[List[int]] = []
        total_number_of_nl_orbitals = 0
        for atom_index, atom in enumerate(self.atoms):
            atom_type = self.atom_species_list[atom.atom_type_index]
            nl_index_list : List[int] = []

            for orbital in atom.phi_orbitals_list:
                if (orbital.n, orbital.l) not in nl_index_list:
                    nl_index_list.append((orbital.n, orbital.l))
            atom_nl_index_lists.append(nl_index_list)
            total_number_of_nl_orbitals += len(nl_index_list)


        # initialize the PDOS_plot_summed
        atoms_nl_combination_number_list = [len(atom_nl_index_list) for atom_nl_index_list in atom_nl_index_lists]
        PDOS_plot_summed = np.zeros((PDOS_plot.shape[0], sum(atoms_nl_combination_number_list)))
        
        # sum over the m index
        orbital_index, summed_orbital_index = 0, 0
        for atom_index, atom_nl_index_list in enumerate(atom_nl_index_lists):
            for (n, l) in atom_nl_index_list:
                number_of_m = 2 * l + 1
                PDOS_plot_summed[:, summed_orbital_index] = np.sum(PDOS_plot[:, orbital_index:orbital_index + number_of_m], axis=1)
                
                summed_orbital_index += 1
                orbital_index += number_of_m
        
        assert summed_orbital_index == PDOS_plot_summed.shape[1], "summed_orbital_index != PDOS_plot_summed.shape[1]"
        return PDOS_plot_summed



    def print_time_taken(self, load_projection_data_from_txt_path : Optional[str] = None):
        print("[Step 1] Time taken to initialize the PDOSCalculator class   : {:7.4f} secs".format(self.step1_time))
        print("[Step 2] Time taken to read in the SPARC's output file       : {:7.4f} secs".format(self.step2_time))
        print("[Step 3] Time taken to get cartesian coordinates and others  : {:7.4f} secs".format(self.step3_time))
        print("[Step 4] Time taken to update the atom's orbital information : {:7.4f} secs".format(self.step4_time))
        # if self.orthogonalize_atomic_orbitals:
        #     print("[Step 5] Time taken to calculate the overlap matrix S        : {:7.4f} secs".format(self.step5_time))
        #     print("[Step 6] Time taken to compute S^(-1/2)                      : {:7.4f} secs".format(self.step6_time))
        # else:
        #     print("[Step 5] Time taken to calculate orbital weights w           : {:7.4f} secs".format(self.step5_time))
        #     print("[Step 6] Time taken to calculate w^(-1/2)                    : {:7.4f} secs".format(self.step6_time))
        
        if load_projection_data_from_txt_path is not None:
            print("[Step 5] Time taken to load the projection data from txt     : {:7.4f} secs".format(self.step5_time))
        else:
            print("[Step 5] Time taken to project wfn onto the atomic orbital   : {:7.4f} secs, {:.4f} secs per k-point".format(self.step5_time, self.step5_time / self.kpt_num))
        
        print("[Step 6] Time taken to compute and save the PDOS and DOS     : {:7.4f} secs".format(self.step6_time))

        total_time = self.step1_time + self.step2_time + self.step3_time + self.step4_time + self.step5_time + self.step6_time
        print("Total time taken: {:7.4f} secs".format(total_time))
    

    def run_single_atom(self, 
                        atom_type : Union[int, str, List[int], List[str]],
                        atom_index_for_specified_atom_type : Union[int, List[int]],
                        mu_PDOS : float = 0.2721140795,  # in eV, Gaussian width for DOS calculation
                        N_PDOS  : int   = 1000,          # number of points in PDOS
                        min_E_plot : Optional[float] = None,       # min energy value (eV) for which PDOS is calculated
                        max_E_plot : Optional[float] = None,       # max energy value (eV) for which PDOS is calculated
                        load_projection_data_from_txt_path : Optional[str] = None,
                        print_projection_data : bool = False,
                        sum_over_m_index : bool = True,
                        print_overlap_matrix : bool = False,
                        ):
        """
        Run the PDOSCalculator class for a single atom
        """

        if min_E_plot is None:
            min_E_plot = self.eign.min() * self.Ha2eV - 5.0 * mu_PDOS
        if max_E_plot is None:
            max_E_plot = self.eign.max() * self.Ha2eV + 5.0 * mu_PDOS


        # Type check
        assert isinstance(mu_PDOS, float), MU_PDOS_NOT_FLOAT_ERROR.format(type(mu_PDOS))
        assert isinstance(N_PDOS, int), N_PDOS_NOT_INTEGER_ERROR.format(type(N_PDOS))
        assert isinstance(min_E_plot, float), MIN_E_PLOT_NOT_FLOAT_ERROR.format(type(min_E_plot)) # in eV
        assert isinstance(max_E_plot, float), MAX_E_PLOT_NOT_FLOAT_ERROR.format(type(max_E_plot)) # in eV

        self.mu_PDOS        : float     = mu_PDOS    
        self.N_PDOS         : int       = N_PDOS     
        self.min_E_plot     : float     = min_E_plot 
        self.max_E_plot     : float     = max_E_plot 

        # initialize the pdos_atom_type_list and pdos_atom_index_list
        self._initialize_pdos_atom_type_and_index_list(
            atom_type = atom_type, 
            atom_index_for_specified_atom_type = atom_index_for_specified_atom_type)
        assert hasattr(self, "pdos_atom_type_list"), "pdos_atom_type_list is not initialized"
        assert hasattr(self, "pdos_atom_index_list"), "pdos_atom_index_list is not initialized"

        """
        5. Project wavefuntions onto the atomic orbital basis.
        """
        time5 = time.time()
        P_single_atom_array_list : List[np.ndarray] = []
        if load_projection_data_from_txt_path is not None:
            # projection_path = self.out_dirname + "/data_projections.txt"
            assert os.path.exists(load_projection_data_from_txt_path), "Data projections file not found at {}".format(load_projection_data_from_txt_path)
            data_projections, kpts_red = self.load_projections_from_txt(path = load_projection_data_from_txt_path)

            print("data_projections.shape = ", data_projections.shape)
            for pdos_atom_index in self.pdos_atom_index_list:
                atom_of_interest = self.atoms[pdos_atom_index]
                start_idx, end_idx = self._get_atom_orbital_indices(target_atom = atom_of_interest)

                P_single_atom_array = data_projections[:, :, start_idx:end_idx]
                P_single_atom_array_list.append(P_single_atom_array)
        else:
            P_single_atom_array_list = self._project_single_atom_wavefunction_onto_atomic_orbital_basis(
                pdos_atom_index_list = self.pdos_atom_index_list,
                orthogonalize_atomic_orbitals = self.orthogonalize_atomic_orbitals,
                save_fname = self.out_dirname + "/data_projections.txt" if print_projection_data else None,
                print_overlap_matrix = print_overlap_matrix,
                )
        self.P_single_atom_array_list = P_single_atom_array_list

        

        """
        6. Compute Projected Density of States (PDOS)   
        """
        time6 = time.time()

        # Computation
        E_list : List[np.ndarray] = []
        DOS_list : List[np.ndarray] = []
        PDOS_plot_list : List[np.ndarray] = []
        for idx, pdos_atom_index in enumerate(self.pdos_atom_index_list):
            atom_of_interest = self.atoms[pdos_atom_index]
            E, PDOS_plot, DOS = self.compute_pdos_dos(data_projections = self.P_single_atom_array_list[idx])
            E_list.append(E)
            DOS_list.append(DOS)
            PDOS_plot_list.append(PDOS_plot)

        
        # Output
        print("Saving the PDOS and DOS to the output directory")

        with open(self.out_dirname + "/PDOS.txt", "w") as f:
            f.write(pdos_output_header_msg.format(
                fermi_level = self.fermi,
                broadening = self.mu_PDOS,
                grid_points = self.N_PDOS,
                min_energy = self.min_E_plot,
                max_energy = self.max_E_plot,
                bands = self.band_num,
                kpoints = self.kpt_num,
                atom_type_num = self.natom_types,
                orbitals = self.tot_orbital_num,
                proj_orbitals = sum(len(PDOS_plot_list[idx][0]) for idx in range(len(self.pdos_atom_index_list))),
                volume = self.cell_volume,
                calculation_time = time6 - self.time_init,
                atom_index_for_pdos = self.pdos_atom_index_list,
                m_index_summed_over = sum_over_m_index,
                orthogonalize_atomic_orbitals = self.orthogonalize_atomic_orbitals,))

        for idx, pdos_atom_index in enumerate(self.pdos_atom_index_list):
            atom = self.atoms[pdos_atom_index]
            E = E_list[idx]
            PDOS_plot = PDOS_plot_list[idx]

            # initialize the pdos_header
            pdos_header, pdos_info_str = self._get_pdos_header_for_single_atom(sum_over_m_index = sum_over_m_index, atom_of_interest = atom_of_interest)

            # sum over the m index
            if sum_over_m_index:
                PDOS_plot = self._sum_over_m_index_for_single_atom(PDOS_plot = PDOS_plot, atom = atom_of_interest)
                column_header_format = "(nl)"
            else:
                column_header_format = "(nl,m)"

            start_idx, end_idx = self._get_atom_orbital_indices(
                target_atom = atom, 
                sum_over_m_index = sum_over_m_index)
            pdos_header, pdos_info_str = self._get_pdos_header_for_single_atom(
                sum_over_m_index = sum_over_m_index, 
                atom_of_interest = atom,)

            with open(self.out_dirname + "/PDOS.txt", "a") as f:
                atom_name = self.atom_species_list[atom.atom_type_index]
                f.write(pdos_info_msg.format(
                    atom_type = atom_name,
                    atomic_number = name_to_atomic_number(atom_name),
                    atom_index = pdos_atom_index,
                    atom_position_cartesian = atom.atom_posn_cart,
                    atom_position_fractional = atom.atom_posn_frac,
                    pdos_unit = "states/eV",
                    header_format = column_header_format,))
                np.savetxt(f,
                    np.column_stack([E, PDOS_plot]),
                    header = pdos_header, 
                    comments='', 
                    fmt='\t%.10f',)


        time_final = time.time()

        self.step5_time = time6 - time5
        self.step6_time = time_final - time6
        self.print_time_taken()

        return E, PDOS_plot, DOS


    def _initialize_pdos_atom_type_and_index_list(
        self, atom_type : Union[int, str, List[int], List[str]], 
        atom_index_for_specified_atom_type : Union[int, List[int]]) -> Tuple[List[str], List[int]]:

        # type check for atom_type and atom_index_for_specified_atom_type
        if isinstance(atom_type, Union[int, str]):
            assert isinstance(atom_index_for_specified_atom_type, int), ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_NOT_INTEGER_ERROR.format(type(atom_index_for_specified_atom_type))
            atom_type_list = [atom_type,]
            atom_index_for_specified_atom_type_list = [atom_index_for_specified_atom_type,]
        elif isinstance(atom_type, list):
            assert len(atom_type) == len(atom_index_for_specified_atom_type), ATOM_TYPE_AND_ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_LENGTH_MISMATCH_ERROR.format(len(atom_type), len(atom_index_for_specified_atom_type))
            for i in range(len(atom_type)):
                assert isinstance(atom_type[i], Union[int, str]), ATOM_TYPE_NOT_INTEGER_OR_STRING_ERROR.format(type(atom_type[i]))
                assert isinstance(atom_index_for_specified_atom_type[i], int), ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_NOT_INTEGER_ERROR.format(type(atom_index_for_specified_atom_type[i]))
            atom_type_list = atom_type
            atom_index_for_specified_atom_type_list = atom_index_for_specified_atom_type
        else:
            raise ValueError("atom_type must be an integer or a string or a list of integers or a list of strings")
        

        # initialize the pdos_atom_type_list and pdos_atom_index_list
        self.pdos_atom_type_list = []
        self.pdos_atom_index_list = []
        for idx, _atom_type in enumerate(atom_type_list):
            _atom_index_for_specified_atom_type = atom_index_for_specified_atom_type_list[idx]
            if isinstance(_atom_type, int):
                _atom_type = atomic_number_to_name(_atom_type)

            # further check the atom_type and atom_index_for_specified_atom_type
            assert _atom_type in  self.atom_species_list, ATOM_TYPE_NOT_FOUND_ERROR.format(_atom_type)
            _atom_type_index = self.atom_species_list.index(_atom_type)
            _atom_num_for_specified_atom_type = self.atom_count_list[_atom_type_index]
            assert _atom_index_for_specified_atom_type < _atom_num_for_specified_atom_type, \
                ATOM_INDEX_FOR_SPECIFIED_ATOM_TYPE_OUT_OF_BOUND_ERROR.format(_atom_type, _atom_index_for_specified_atom_type)
            
            _index_in_total_atom_list = int(np.sum(self.atom_count_list[:_atom_type_index]) + _atom_index_for_specified_atom_type)
            self.pdos_atom_type_list.append(_atom_type)
            self.pdos_atom_index_list.append(_index_in_total_atom_list)

        return self.pdos_atom_type_list, self.pdos_atom_index_list


    def _get_pdos_header_for_single_atom(self, sum_over_m_index : bool, atom_of_interest : Atom):
        # initialize the pdos_header
        header_parts = ["\tEnergy(eV)  "]
        if not sum_over_m_index:
            atom_type = self.atom_species_list[atom_of_interest.atom_type_index]
            for orbital in atom_of_interest.phi_orbitals_list:
                desc = f"({orbital.n}{number_to_spdf(orbital.l)}, m={orbital.m})"
                header_parts.append(desc)
            # Join with proper spacing
            pdos_header = " \t".join([f"{part:<12}" for part in header_parts])
            pdos_info_str = \
                "Number of Projected DOS = " + str(atom_of_interest.num_orbitals) + "\n" + \
                "Orbitals are labeled as AtomType(n,l,m)" + "\n"
        else:
            atom_type = self.atom_species_list[atom_of_interest.atom_type_index]
            nl_index_list : List[int] = []

            for orbital in atom_of_interest.phi_orbitals_list:
                if (orbital.n, orbital.l) not in nl_index_list:
                    nl_index_list.append((orbital.n, orbital.l))

            for (n, l) in nl_index_list:
                desc = f"({n}{number_to_spdf(l)})"
                header_parts.append(desc)

            # Join with proper spacing
            pdos_header = " \t".join([f"{part:<12}" for part in header_parts])
            pdos_info_str = \
                "The m index is summed over" + "\n" + \
                "Number of Projected DOS = " + str(len(nl_index_list)) + "\n" + \
                "Orbitals are labeled as AtomType(n,l)" + "\n"
        return pdos_header, pdos_info_str



    def _sum_over_m_index_for_single_atom(self, PDOS_plot : np.ndarray, atom : Atom):
            print("summing over the m index")

            # get the nl index list
            nl_index_list : List[int] = []

            for orbital in atom.phi_orbitals_list:
                if (orbital.n, orbital.l) not in nl_index_list:
                    nl_index_list.append((orbital.n, orbital.l))


            # initialize the PDOS_plot_summed
            PDOS_plot_summed = np.zeros((PDOS_plot.shape[0], len(nl_index_list)))
            
            # sum over the m index
            orbital_index, summed_orbital_index = 0, 0
            for (n, l) in nl_index_list:
                number_of_m = 2 * l + 1
                PDOS_plot_summed[:, summed_orbital_index] = np.sum(PDOS_plot[:, orbital_index:orbital_index + number_of_m], axis=1)
                    
                summed_orbital_index += 1
                orbital_index += number_of_m
            
            assert orbital_index == atom.num_orbitals, "orbital_index != atom.num_orbitals"
            return PDOS_plot_summed



    def run_single_orbital(self, 
                           atom_type : Union[int, str],     # atomic number or atom type name
                           orbital_index : int,
                           mu_PDOS : float = 0.2721140795,  # in eV, Gaussian width for DOS calculation
                           N_PDOS  : int   = 1000,          # number of points in PDOS
                           min_E_plot : float = -20.,       # min energy value (eV) for which PDOS is calculated
                           max_E_plot : float =  20.,       # max energy value (eV) for which PDOS is calculated
                           print_projection_data = False,
                           ):    
        raise NotImplementedError("Not to run the single orbital is not implemented yet")


    def _interp1d_spline_or_linear(self, x_grid, y_grid, x_query):
        """
        Try cubic spline (SciPy). If SciPy not available, fall back to numpy.interp (linear).
        - x_grid, y_grid: 1D arrays (ascending x_grid)
        - x_query: 1D array of query points (will be clipped into [x_grid[0], x_grid[-1]])
        Returns: y(x_query) as 1D array.
        """
        xg = np.asarray(x_grid, dtype=float)
        yg = np.asarray(y_grid, dtype=float)
        xq = np.asarray(x_query, dtype=float)

        # clip to the grid range (MATLAB uses rcut + r_grid to cover, usually not out of bounds, but here is more safe)
        xq = np.clip(xq, xg[0], xg[-1])

        try:
            # use SciPy's cubic spline, equivalent to MATLAB's 'spline' style
            from scipy import CubicSpline
            cs = CubicSpline(xg, yg, extrapolate=False)
            yq = cs(xq)
            # out of range will give nan (extrapolate=False), but we have clipped, so no problem
            # still do a safe treatment
            yq = np.where(np.isfinite(yq), yq, 0.0)
            return yq
        except Exception:
            # if SciPy is not available or fails, use linear interpolation
            return np.interp(xq, xg, yg)


    def get_cartesian_coordinates_for_the_unit_cell(self):
        """
        Updated parameters
            x_coord : np.ndarray[float] # x-coordinate of the unit cell
            y_coord : np.ndarray[float] # y-coordinate of the unit cell
            z_coord : np.ndarray[float] # z-coordinate of the unit cell
            x_shift_index_list : ShiftIndexListType # All x-shift indices that should be considered
            y_shift_index_list : ShiftIndexListType # All y-shift indices that should be considered
            z_shift_index_list : ShiftIndexListType # All z-shift indices that should be considered
            tot_grid_pt               : int # total number of grid points in the unit cell
            cartesian_gridwise_coord  : np.ndarray[float] # cartesian coordinates of the grid points for the input of the psi file
        """
        if self.is_orthogonal_lattice:
            # Orthogonal lattice implementation
            self.get_cartesian_coordinates_for_orthogonal_lattice()
        else:
            # Non-orthogonal lattice implementation
            self.get_cartesian_coordinates_for_non_orthogonal_lattice()


    def get_cartesian_coordinates_for_orthogonal_lattice(self):
        # get the cartesian coordinates for the unit cell
        x_coord = np.linspace(0, self.Lx, self.Nx + 1)[:-1]
        y_coord = np.linspace(0, self.Ly, self.Ny + 1)[:-1]
        z_coord = np.linspace(0, self.Lz, self.Nz + 1)[:-1]

        # get the cartesian coordinates for the unit cell
        X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord, indexing='ij')
        cartesian_gridwise_coord = np.stack([X.ravel(order='F'), 
                                             Y.ravel(order='F'), 
                                             Z.ravel(order='F')], axis=-1)


        # get the x, y, z shift index for each type of atom
        x_shift_index_list, y_shift_index_list, z_shift_index_list = [], [], []
        if isinstance(self.r_cut_max, float):
            rcut = self.r_cut_max
            x_idx_max = max(0, int(np.floor(rcut / self.Lx)) + 1)
            y_idx_max = max(0, int(np.floor(rcut / self.Ly)) + 1)
            z_idx_max = max(0, int(np.floor(rcut / self.Lz)) + 1)
            x_shift_index_list = list(range(-x_idx_max, x_idx_max + 1))
            y_shift_index_list = list(range(-y_idx_max, y_idx_max + 1))
            z_shift_index_list = list(range(-z_idx_max, z_idx_max + 1))
            if self.bcx == "D":
                x_shift_index_list = [0]
            if self.bcy == "D":
                y_shift_index_list = [0]
            if self.bcz == "D":
                z_shift_index_list = [0]

        elif isinstance(self.r_cut_max, list):
            for rcut in self.r_cut_max:
                x_idx_max = max(0, int(np.floor(rcut / self.Lx)) + 1)
                y_idx_max = max(0, int(np.floor(rcut / self.Ly)) + 1)
                z_idx_max = max(0, int(np.floor(rcut / self.Lz)) + 1)
                # x-direction
                if self.bcx == "D": 
                    x_shift_index_list.append([0])
                elif self.bcx == "P": 
                    x_shift_index_list.append(list(range(-x_idx_max, x_idx_max + 1)))
                else: 
                    raise ValueError(BCX_NOT_D_OR_P_ERROR.format(self.bcx))

                # y-direction
                if self.bcy == "D":
                    y_shift_index_list.append([0])
                elif self.bcy == "P": 
                    y_shift_index_list.append(list(range(-y_idx_max, y_idx_max + 1)))
                else: 
                    raise ValueError(BCY_NOT_D_OR_P_ERROR.format(self.bcy))

                # z-direction
                if self.bcz == "D":
                    z_shift_index_list.append([0])
                elif self.bcz == "P": 
                    z_shift_index_list.append(list(range(-z_idx_max, z_idx_max + 1)))
                else: 
                    raise ValueError(BCZ_NOT_D_OR_P_ERROR.format(self.bcz))

        else:
            raise ValueError(R_CUT_MAX_NOT_FLOAT_OR_LIST_ERROR.format(type(self.r_cut_max)))

        # store the parsed parameters in the class
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord
        self.x_shift_index_list = x_shift_index_list
        self.y_shift_index_list = y_shift_index_list
        self.z_shift_index_list = z_shift_index_list
        self.tot_grid_pt = self.Nx * self.Ny * self.Nz
        self.cartesian_gridwise_coord = cartesian_gridwise_coord


    def get_cartesian_coordinates_for_non_orthogonal_lattice(self):
        """
        Get cartesian coordinates for non-orthogonal lattice
        """
        # Create fractional coordinates grid (0 to 1 in each direction)
        x_frac = np.linspace(0, 1, self.Nx, endpoint=False)
        y_frac = np.linspace(0, 1, self.Ny, endpoint=False)
        z_frac = np.linspace(0, 1, self.Nz, endpoint=False)
        
        # Create 3D grid in fractional coordinates
        X_frac, Y_frac, Z_frac = np.meshgrid(x_frac, y_frac, z_frac, indexing='ij')
        
        # Convert to cartesian coordinates using lattice vectors
        # r_cart = A * r_frac where A is the lattice vectors matrix
        cartesian_gridwise_coord = np.zeros((self.Nx * self.Ny * self.Nz, 3))
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    idx = i + j * self.Nx + k * self.Nx * self.Ny
                    frac_coord = np.array([X_frac[i, j, k], Y_frac[i, j, k], Z_frac[i, j, k]])
                    # Convert fractional to cartesian: r_cart = A * r_frac
                    # Since self.latvec is now column vectors, we use A^T * r_frac
                    cart_coord = np.dot(self.latvec, frac_coord)
                    cartesian_gridwise_coord[idx, :] = cart_coord

        
        # Calculate shift indices for non-orthogonal lattice using proper cutoff analysis
        x_shift_index_list, y_shift_index_list, z_shift_index_list = [], [], []
        
        if isinstance(self.r_cut_max, float):
            rcut = self.r_cut_max
            shift_indices = self.calculate_shift_indices_for_cutoff(rcut)
            x_shift_index_list = shift_indices[0]
            y_shift_index_list = shift_indices[1]
            z_shift_index_list = shift_indices[2]
            if self.bcx == "D":
                x_shift_index_list = [0]
            if self.bcy == "D":
                y_shift_index_list = [0]
            if self.bcz == "D":
                z_shift_index_list = [0]
            
        elif isinstance(self.r_cut_max, list):
            for rcut in self.r_cut_max:
                shift_indices = self.calculate_shift_indices_for_cutoff(rcut)
                # x-direction
                if self.bcx == "D":
                    x_shift_index_list.append([0])
                elif self.bcx == "P":
                    x_shift_index_list.append(shift_indices[0])
                else:
                    raise ValueError(BCX_NOT_D_OR_P_ERROR.format(self.bcx))

                # y-direction
                if self.bcy == "D":
                    y_shift_index_list.append([0])
                elif self.bcy == "P":
                    y_shift_index_list.append(shift_indices[1])
                else:
                    raise ValueError(BCY_NOT_D_OR_P_ERROR.format(self.bcy))

                # z-direction
                if self.bcz == "D":
                    z_shift_index_list.append([0])
                elif self.bcz == "P":
                    z_shift_index_list.append(shift_indices[2])
                else:
                    raise ValueError(BCZ_NOT_D_OR_P_ERROR.format(self.bcz))

        
        # Store parameters
        self.x_coord = x_frac
        self.y_coord = y_frac
        self.z_coord = z_frac
        self.x_shift_index_list = x_shift_index_list
        self.y_shift_index_list = y_shift_index_list
        self.z_shift_index_list = z_shift_index_list
        self.tot_grid_pt = self.Nx * self.Ny * self.Nz
        self.cartesian_gridwise_coord = cartesian_gridwise_coord


    def calculate_shift_indices_for_cutoff(self, rcut):
        """
        Calculate shift indices that properly cover the cutoff sphere in non-orthogonal lattice
        
        For non-orthogonal lattice, the cutoff sphere in cartesian space becomes 
        an ellipsoid in fractional space. We need to find all integer shifts that 
        could contain points within the cutoff radius.
        
        Args:
            rcut: cutoff radius in Bohr
            
        Returns:
            tuple: (x_shifts, y_shifts, z_shifts) lists of shift indices
        """
        # Method: Analytical approach using metric tensor
        # The cutoff condition in cartesian space: |r| <= rcut
        # In fractional space: r_frac^T * G * r_frac <= rcut^2
        # where G is the metric tensor G_ij = a_i · a_j
        
        # Calculate the metric tensor G = A^T * A where A is the lattice vector matrix
        G = np.dot(self.latvec.T, self.latvec)
        
        # For a conservative approach, we find the maximum extent in each direction
        # by considering the ellipsoid equation: r_frac^T * G * r_frac <= rcut^2
        
        # We can find the maximum extent in each direction by setting other coordinates to zero
        # and solving for the maximum value of each coordinate
        
        max_shifts = []
        
        for i in range(3):
            # Set other coordinates to zero and find maximum extent in direction i
            # The ellipsoid equation becomes: G[i,i] * x_i^2 <= rcut^2
            # So: x_i <= rcut / sqrt(G[i,i])
            
            max_extent = rcut / np.sqrt(G[i, i])
            max_shift = int(np.ceil(max_extent)) + 1  # Add buffer
            
            max_shifts.append(max_shift)
        
        # Create shift lists
        x_shifts = list(range(-max_shifts[0], max_shifts[0] + 1))
        y_shifts = list(range(-max_shifts[1], max_shifts[1] + 1))
        z_shifts = list(range(-max_shifts[2], max_shifts[2] + 1))
        
        return x_shifts, y_shifts, z_shifts


    def atom_wise_compute_index_mask_and_effective_grid_point_positions_dict(self, atom : 'PDOSCalculator.Atom'):
        """
        Updated parameters
            atom.r_cut : float 
                cutoff radius, in Bohr
            atom.index_mask_and_effective_grid_point_positions_dict : IndexMaskDictType
                (x_index_shift, y_index_shift, z_index_shift) -> (index_mask, effective_grid_point_positions_array)
        """

        assert isinstance(atom, self.Atom), INVALID_ATOM_INPUT_ERROR.format(type(atom))

        # Prepare the parameters
        atom_type_index = atom.atom_type_index
        atom_posn_cart = atom.atom_posn_cart
        rcut = self.atomic_wave_function_list[atom_type_index].r_cut
        if isinstance(self.r_cut_max, float):
            x_shift_index_range, y_shift_index_range, z_shift_index_range = self.x_shift_index_list, self.y_shift_index_list, self.z_shift_index_list
        elif isinstance(self.r_cut_max, list):
            x_shift_index_range, y_shift_index_range, z_shift_index_range = self.x_shift_index_list[atom_type_index], self.y_shift_index_list[atom_type_index], self.z_shift_index_list[atom_type_index]
        else:
            raise ValueError(R_CUT_MAX_NOT_FLOAT_OR_LIST_ERROR.format(type(self.r_cut_max)))

        # Compute the atom-centered cartesian coordinates of the unit cell
        atom_centered_cart_coord_of_unit_cell = self.cartesian_gridwise_coord - atom_posn_cart

        # Update the parameters
        atom.r_cut = rcut
        atom.index_mask_and_effective_grid_point_positions_dict.clear()
        
        for x_shift_index in x_shift_index_range:
            for y_shift_index in y_shift_index_range:
                for z_shift_index in z_shift_index_range:
                    if self.is_orthogonal_lattice:
                        # Orthogonal lattice: direct vector addition
                        shifted_vector = x_shift_index * self.x_lattice_vector + y_shift_index * self.y_lattice_vector + z_shift_index * self.z_lattice_vector
                    else:
                        # Non-orthogonal lattice: use lattice vectors matrix
                        # For shift in fractional coordinates, convert to cartesian
                        shifted_vector = x_shift_index * self.latvec[:, 0] + y_shift_index * self.latvec[:, 1] + z_shift_index * self.latvec[:, 2]
                    
                    atom_centered_cart_coord_of_shifted_unit_cell = atom_centered_cart_coord_of_unit_cell + shifted_vector

                    # Compute the distance between the atom and the shifted unit cell
                    r_norm_array = np.linalg.norm(atom_centered_cart_coord_of_shifted_unit_cell, axis = -1)
                    
                    # Compute the index mask for the shifted unit cell, if not all False, then update the index_mask_and_r_norm_dict
                    index_mask = r_norm_array <= rcut

                    # Get the effective grid point position array
                    effective_grid_point_position_array = atom_centered_cart_coord_of_shifted_unit_cell[index_mask]

                    if not np.all(index_mask == False):
                        atom.index_mask_and_effective_grid_point_positions_dict[(x_shift_index, y_shift_index, z_shift_index)] = (index_mask, effective_grid_point_position_array)
        
        assert (0, 0, 0) in atom.index_mask_and_effective_grid_point_positions_dict, INDEX_MASK_AND_EFFECTIVE_GRID_POINT_POSITIONS_DICT_NOT_FOUND_ERROR


    def atom_wise_compute_grid_wise_phi_orbitals(self, atom : 'PDOSCalculator.Atom'):
        """
        Compute the grid-wise Phi_{nlm}(r) orbitals for the atom
            Phi_{nlm}(r) = phi_{nl}(r) * Y_{lm}(r)
        
        Updated parameters
            atom.num_orbitals      : int                               # total number of orbitals
            atom.phi_orbitals_list : List['PDOSCalculator.PhiOrbital'] # list of Phi_{nlm}(r) orbitals
        """
        assert isinstance(atom, self.Atom), INVALID_ATOM_INPUT_ERROR.format(type(atom))
        assert len(atom.index_mask_and_effective_grid_point_positions_dict) > 0, ATOM_INDEX_MASK_AND_EFFECTIVE_GRID_POINT_POSITIONS_DICT_EMPTY_ERROR


        # Get some related paramters
        atom_type_index : int                                     = atom.atom_type_index                            # atom type index
        atomic_wfn_info : PDOSCalculator.AtomicWaveFunction       = self.atomic_wave_function_list[atom_type_index] # atomic wave function information
        num_psdwfn      : int                                     = atomic_wfn_info.num_psdwfn                      # number of pseudo wave functions
        psdwfn_list     : List[PDOSCalculator.PseudoWaveFunction] = atomic_wfn_info.psdwfn_list                     # pseudo wave function list



        # Some preparation
        atom.phi_orbitals_list.clear()
        phi_r_dict : PhiValueDictType = {}


        # First, loop over all pseudo wave functions, corresponding to quantum number n and l
        for psdwfn_index in range(num_psdwfn): 
            psdwfn_info : PDOSCalculator.PseudoWaveFunction = psdwfn_list[psdwfn_index]
            n    : int = psdwfn_info.n     # principal quantum number
            l    : int = psdwfn_info.l     # angular momentum quantum number
            no_m : int = psdwfn_info.no_m  # number of magnetic quantum number m


            # Compute the phi_{nl}(r) array for all shift index vectors
            phi_r_dict.clear()
            for shift_index_vector in atom.index_mask_and_effective_grid_point_positions_dict.keys():
                _index_mask, effective_grid_point_positions_array = atom.index_mask_and_effective_grid_point_positions_dict[shift_index_vector]
                r_norm_array = np.linalg.norm(effective_grid_point_positions_array, axis = -1)
                phi_r_array = self._interp1d_spline_or_linear(
                    x_grid = psdwfn_info.r, 
                    y_grid = psdwfn_info.phi_r, 
                    x_query = r_norm_array)
                phi_r_dict[shift_index_vector] = phi_r_array

            # Then, loop over all magnetic quantum number m
            for m_index in range(no_m):
                m = m_index - l

                # initialize the PhiOrbital object
                phi_orbital = PDOSCalculator.PhiOrbital(n = n, l = l, m = m) #, normalization_factor = 1.0)
                phi_orbital.phi_value_dict.clear()

                # Finally, loop over all shift index vectors, corresponding to the shifted unit cell
                for shift_index_vector in atom.index_mask_and_effective_grid_point_positions_dict.keys():
                    _index_mask, effective_grid_point_positions_array = atom.index_mask_and_effective_grid_point_positions_dict[shift_index_vector]

                    # Get phi_{nl}(r) array
                    phi_r_array = phi_r_dict[shift_index_vector]

                    # Calculate Y_{lm}(r) array
                    Ylm_array = spherical_harmonics(X = effective_grid_point_positions_array[:, 0],
                                                    Y = effective_grid_point_positions_array[:, 1],
                                                    Z = effective_grid_point_positions_array[:, 2],
                                                    l = l, m = m)
                    
                    # Calculate Phi_{nlm}(r) array
                    Phi_nlm_r_array = phi_r_array * Ylm_array
                    
                    # Store the Phi_{nlm}(r) array
                    phi_orbital.phi_value_dict[shift_index_vector] = Phi_nlm_r_array
                    
                # Append the PhiOrbital object to the list
                atom.phi_orbitals_list.append(phi_orbital)
            
            # End of loop over all magnetic quantum number m
        
        assert len(atom.phi_orbitals_list) == atomic_wfn_info.num_orbitals, PHI_ORBITALS_NOT_EQUAL_TO_EXPECTED_NUM_ERROR.format(len(atom.phi_orbitals_list), atomic_wfn_info.num_orbitals)
        atom.num_psdwfn = num_psdwfn
        atom.num_orbitals = len(atom.phi_orbitals_list)



    @classmethod
    def get_total_number_of_orbitals(cls, atoms : List[Atom]):
        """
        Get the total number of atoms
        """
        assert isinstance(atoms, list), "atoms must be a list, get {} instead".format(type(atoms))
        num_orbitals = 0
        for atom in atoms:
            assert isinstance(atom, PDOSCalculator.Atom), "atom must be a PDOSCalculator.Atom, get {} instead".format(type(atom))
            num_orbitals += atom.num_orbitals
        
        return num_orbitals


    @classmethod
    def calculate_overlap_matrix(cls, 
        atoms : List['PDOSCalculator.Atom'], 
        dV : float, 
        k_point_fractional : np.ndarray[float],
        save_fname : Optional[str] = None,
        ) -> np.ndarray[float]:
        """
        Compute the overlap matrix, size = (num_orbitals, num_orbitals)
        """
        # print("Calculating the overlap matrix")

        # Type check
        assert isinstance(k_point_fractional, np.ndarray), K_POINT_FRACTIONAL_INPUT_NOT_NP_ARRAY_ERROR.format(type(k_point_fractional))
        assert isinstance(atoms, list), ATOMS_INPUT_NOT_LIST_ERROR.format(type(atoms))
        for atom in atoms:
            assert isinstance(atom, PDOSCalculator.Atom), ATOM_INPUT_NOT_ATOM_ERROR.format(type(atom))
        assert isinstance(dV, float), DV_NOT_FLOAT_ERROR.format(type(dV))
        if save_fname is not None:
            assert isinstance(save_fname, str), SAVE_FNAME_NOT_STRING_ERROR.format(type(save_fname))
        
        # Get the total number of orbitals
        num_atoms = len(atoms)
        num_orbitals = cls.get_total_number_of_orbitals(atoms)
        overlap_matrix = np.zeros((num_orbitals, num_orbitals), dtype = np.complex128)

        # Loop over different atoms
        row_index = 0
        for atom_i_index, atom_i in enumerate(atoms):
            num_orbitals_i = atom_i.num_orbitals

            # initialize the column index as the row index, because the overlap matrix is symmetric
            col_index = row_index 

            # Computing the overlap matrix between the same atom's orbitals
            block_overlap_matrix = cls._calculate_overlap_matrix_between_two_atoms(
                atom1 = atom_i, 
                atom2 = atom_i, 
                is_same_atom = True,
                k_point_fractional = k_point_fractional,
                dV = dV,
                )

            # Update the overlap matrix
            overlap_matrix[row_index:row_index + num_orbitals_i, col_index:col_index + num_orbitals_i] = block_overlap_matrix
            
            # Update the column index
            col_index += num_orbitals_i

            # Computing the overlap matrix between different atoms' orbitals
            for atom_j_index in range(atom_i_index + 1, num_atoms):
                atom_j = atoms[atom_j_index]
                num_orbitals_j = atom_j.num_orbitals

                # Computing the overlap matrix between different atoms' orbitals
                block_overlap_matrix = cls._calculate_overlap_matrix_between_two_atoms(
                    atom1 = atom_i, 
                    atom2 = atom_j, 
                    is_same_atom = False,
                    k_point_fractional = k_point_fractional,
                    dV = dV,
                    )

                # Update the overlap matrix
                overlap_matrix[row_index:row_index + num_orbitals_i, col_index:col_index + num_orbitals_j] = block_overlap_matrix
                overlap_matrix[col_index:col_index + num_orbitals_j, row_index:row_index + num_orbitals_i] = block_overlap_matrix.conj().T

                # Update the column index
                col_index += num_orbitals_j
            
            # Update the row index
            row_index += num_orbitals_i

        # Multiply the overlap matrix by the volume element dV
        # overlap_matrix *= dV

        # Save the overlap matrix
        if save_fname is not None:
            np.savetxt(save_fname, overlap_matrix, fmt = "%10.6f")

        # check if the overlap matrix is Hermitian
        assert np.allclose(overlap_matrix, overlap_matrix.conj().T), "The overlap matrix is not Hermitian"

        # Clear the total orbitals in the unit cell dictionary for all atoms
        # cls.clear_total_orbitals_in_unit_cell_dict(atoms)

        return overlap_matrix


    @staticmethod
    def clear_total_orbitals_in_unit_cell_dict(atoms : List['PDOSCalculator.Atom']):
        """
        Clear the total orbitals in the unit cell dictionary for all atoms
        """
        assert isinstance(atoms, list), ATOMS_INPUT_NOT_LIST_ERROR.format(type(atoms))
        for atom in atoms:
            assert isinstance(atom, PDOSCalculator.Atom), ATOM_INPUT_NOT_ATOM_ERROR.format(type(atom))
            for phi_orbital in atom.phi_orbitals_list:
                phi_orbital.total_orbitals_in_unit_cell_dict.clear()

    
    @classmethod
    def calculate_orbital_weights(cls, 
        atoms : List['PDOSCalculator.Atom'], 
        dV : float,
        k_point_fractional : np.ndarray[float],
        ) -> np.ndarray[float]:
        """
        Compute the orbital weights, size = (num_orbitals, )
        """
        # print("Calculating the orbital weights")
        
        # Type check
        assert isinstance(atoms, list), ATOMS_INPUT_NOT_LIST_ERROR.format(type(atoms))
        for atom in atoms:
            assert isinstance(atom, PDOSCalculator.Atom), ATOM_INPUT_NOT_ATOM_ERROR.format(type(atom))
        assert isinstance(dV, float), DV_NOT_FLOAT_ERROR.format(type(dV))
        
        # Get the total number of orbitals
        num_atoms = len(atoms)
        num_orbitals = cls.get_total_number_of_orbitals(atoms)
        orbital_weights = np.zeros((num_orbitals, ),)

        # Loop over different atoms
        index = 0
        for atom_index, atom in enumerate(atoms):
            for idx_1, phi_orbital_1 in enumerate(atom.phi_orbitals_list):
                weight_value = cls._calculate_overlap_between_two_orbitals(
                    phi_orbital_1 = phi_orbital_1, 
                    phi_orbital_2 = phi_orbital_1, 
                    k_point_fractional = k_point_fractional,
                    is_same_orbital = True,
                    index_mask_dict_1 = atom.index_mask_and_effective_grid_point_positions_dict,
                    dV = dV,
                    )
                assert np.isclose(weight_value.imag, 0.), "The weight value of the same orbital should always be real, got {} instead".format(weight_value)
                orbital_weights[index + idx_1] = weight_value.real
            index += atom.num_orbitals
        assert index == num_orbitals, "The number of orbitals is not equal to the expected number, got {} instead".format(index)

        # Multiply the orbital's weights by the volume element dV
        # orbital_weights *= dV
        # print("orbital_weights = ", orbital_weights)
        return orbital_weights



    @classmethod
    def _calculate_overlap_matrix_between_two_atoms(cls, 
        atom1 : 'PDOSCalculator.Atom', 
        atom2 : 'PDOSCalculator.Atom', 
        k_point_fractional : np.ndarray[float],
        is_same_atom : bool,
        dV : float,
        ) -> np.ndarray[float]:
        """
        Compute the overlap matrix between two atoms, size = (atom1.num_orbitals, atom2.num_orbitals)
            - Parameter dV is not considered here, because it is a constant for all atoms, and will be multiplied later
            - If same_atom is True, then the overlap matrix is computed between the same atom, otherwise it is computed between two different atoms
        """
        num_orbitals_1 = atom1.num_orbitals
        num_orbitals_2 = atom2.num_orbitals
        overlap_matrix = np.zeros((num_orbitals_1, num_orbitals_2), dtype = np.complex128)

        # If the two atoms are the same, then the overlap matrix is computed between the same atom's orbitals, and should be symmetric
        if is_same_atom:
            assert id(atom1) == id(atom2), TWO_ATOMS_NOT_SAME_ERROR.format(id(atom1), id(atom2))
            num_orbitals_1 = num_orbitals_2 = atom1.num_orbitals

            for idx_1, phi_orbital_1 in enumerate(atom1.phi_orbitals_list):
                # Diagonal elements
                overlap_matrix[idx_1, idx_1] = cls._calculate_overlap_between_two_orbitals(
                    phi_orbital_1 = phi_orbital_1, 
                    phi_orbital_2 = phi_orbital_1, 
                    k_point_fractional = k_point_fractional,
                    is_same_orbital = True, 
                    index_mask_dict_1 = atom1.index_mask_and_effective_grid_point_positions_dict,
                    dV = dV,
                    )

                # Off-diagonal elements: always zero, so commented out
                for idx_2 in range(idx_1 + 1, num_orbitals_1):
                    phi_orbital_2 = atom1.phi_orbitals_list[idx_2]
                    overlap_value = cls._calculate_overlap_between_two_orbitals(phi_orbital_1 = phi_orbital_1, 
                                                                                phi_orbital_2 = phi_orbital_2, 
                                                                                k_point_fractional = k_point_fractional,
                                                                                is_same_orbital = False, 
                                                                                index_mask_dict_1 = atom1.index_mask_and_effective_grid_point_positions_dict, 
                                                                                index_mask_dict_2 = atom1.index_mask_and_effective_grid_point_positions_dict,
                                                                                dV = dV,
                                                                                )
                    if not np.isclose(overlap_value, 0.):
                        pass
                        # print(f"WARNING: The overlap matrix for the same atom's orbitals should always be diagonal, but got non-zero value {overlap_value} for the index ({idx_1}, {idx_2}).")
                    overlap_matrix[idx_1, idx_2] = overlap_value
                    overlap_matrix[idx_2, idx_1] = overlap_value.conj()
                    # assert np.isclose(overlap_value, 0.), "The overlap matrix for the same atom's orbitals should always be diagonal, but got non-zero value {} for the index ({}, {}).".format(overlap_value, idx_1, idx_2)

        # If the two atoms are different, then the overlap matrix is computed between two different atoms' orbitals
        else:
            # If the distance between the two atoms is greater than the sum of the cutoff radii, then the overlap matrix is zero.
            if np.linalg.norm(atom1.atom_posn_cart - atom2.atom_posn_cart) > atom1.r_cut + atom2.r_cut:
                return overlap_matrix
            
            for idx_1, phi_orbital_1 in enumerate(atom1.phi_orbitals_list):
                for idx_2, phi_orbital_2 in enumerate(atom2.phi_orbitals_list):
                    overlap_value = cls._calculate_overlap_between_two_orbitals(phi_orbital_1 = phi_orbital_1, 
                                                                                phi_orbital_2 = phi_orbital_2, 
                                                                                k_point_fractional = k_point_fractional,
                                                                                is_same_orbital = False, 
                                                                                index_mask_dict_1 = atom1.index_mask_and_effective_grid_point_positions_dict, 
                                                                                index_mask_dict_2 = atom2.index_mask_and_effective_grid_point_positions_dict,
                                                                                dV = dV,
                                                                                )
                    overlap_matrix[idx_1, idx_2] = overlap_value
        
        return overlap_matrix


    @classmethod
    def _calculate_overlap_between_two_orbitals(cls, 
                                                phi_orbital_1 : 'PDOSCalculator.PhiOrbital', 
                                                phi_orbital_2 : 'PDOSCalculator.PhiOrbital', 
                                                k_point_fractional : np.ndarray[float],
                                                is_same_orbital : bool,
                                                dV : float,
                                                index_mask_dict_1 : Optional[IndexMaskDictType] = None,
                                                index_mask_dict_2 : Optional[IndexMaskDictType] = None,
                                                ) -> np.ndarray[np.complex128]:
        """
        Compute the overlap between two orbitals, size = (num_orbitals_1, num_orbitals_2)
            - If same_orbital is True, then the overlap is computed between the same orbital, and the index_mask_dict_1 and index_mask_dict_2 is not needed
        """
        # assert isinstance(phi_orbital_1, PDOSCalculator.PhiOrbital), "phi_orbital_1 must be a PDOSCalculator.PhiOrbital, get {} instead".format(type(phi_orbital_1))
        # assert isinstance(phi_orbital_2, PDOSCalculator.PhiOrbital), "phi_orbital_2 must be a PDOSCalculator.PhiOrbital, get {} instead".format(type(phi_orbital_2))

        overlap_value = 0.0
        if is_same_orbital:
            assert id(phi_orbital_1) == id(phi_orbital_2), TWO_ORBITALS_NOT_SAME_ERROR.format(id(phi_orbital_1), id(phi_orbital_2))
            # If two orbitals are the same, then the overlap is computed between the same orbital, so just the square of all the phi_value_dict values
            # _phi_orbital_unit_cell = cls.construct_total_orbitals_in_unit_cell(phi_orbital = phi_orbital_1, k_point_fractional = k_point_fractional, index_mask_dict = index_mask_dict_1)
            # overlap_value = (phi_orbital_1.normalization_factor ** 2) * np.sum(_phi_orbital_unit_cell.conj() * _phi_orbital_unit_cell)
            overlap_value = np.array(1.0, dtype = np.complex128)
        else:
            # If two orbitals are different, then the overlap is computed between two different orbitals, so just the product of the phi_value_dict values
            _phi_orbital_unit_cell_1 = cls.construct_total_orbitals_in_unit_cell(phi_orbital = phi_orbital_1, k_point_fractional = k_point_fractional, index_mask_dict = index_mask_dict_1)
            _phi_orbital_unit_cell_2 = cls.construct_total_orbitals_in_unit_cell(phi_orbital = phi_orbital_2, k_point_fractional = k_point_fractional, index_mask_dict = index_mask_dict_2)

            _normalization_factor_1 = cls.compute_normalization_factor(dV = dV, phi_orbital = phi_orbital_1, k_point_fractional = k_point_fractional, index_mask_dict = index_mask_dict_1)
            _normalization_factor_2 = cls.compute_normalization_factor(dV = dV, phi_orbital = phi_orbital_2, k_point_fractional = k_point_fractional, index_mask_dict = index_mask_dict_2)
            overlap_value = (_normalization_factor_1 * _normalization_factor_2) * np.sum(_phi_orbital_unit_cell_1.conj() * _phi_orbital_unit_cell_2) * dV
        
        assert overlap_value.shape == (), "The overlap value must be a scalar, got {} instead".format(overlap_value.shape)

        return overlap_value


    @classmethod
    def compute_normalization_factor(cls,
        dV : float,
        phi_orbital : 'PDOSCalculator.PhiOrbital', 
        k_point_fractional : np.ndarray[float],
        index_mask_dict : 'PDOSCalculator.IndexMaskDictType') -> np.ndarray[float]:
        """
        Compute the normalization factor for a given phi orbital
        """
        assert isinstance(dV, float), DV_NOT_FLOAT_ERROR.format(type(dV))
        assert isinstance(phi_orbital, PDOSCalculator.PhiOrbital), PHI_ORBITAL_INPUT_NOT_PHI_ORBITAL_ERROR.format(type(phi_orbital))
        assert isinstance(k_point_fractional, np.ndarray), K_POINT_FRACTIONAL_INPUT_NOT_NP_ARRAY_ERROR.format(type(k_point_fractional))
        assert isinstance(index_mask_dict, dict), INDEX_MASK_DICT_INPUT_NOT_DICT_ERROR.format(type(index_mask_dict))
        
        # Convert k_point_fractional to tuple for use as dictionary key (numpy arrays are unhashable)
        k_point_key = tuple(k_point_fractional)

        # If the normalization factor is already computed, then return it
        if k_point_key in phi_orbital.normalization_factor_dict:
            return phi_orbital.normalization_factor_dict[k_point_key]
        
        # If the normalization factor is not computed, then compute it
        _phi_orbital_unit_cell = cls.construct_total_orbitals_in_unit_cell(
            phi_orbital = phi_orbital, 
            k_point_fractional = k_point_fractional, 
            index_mask_dict = index_mask_dict)
        integral_value = np.sum(_phi_orbital_unit_cell.real**2 + _phi_orbital_unit_cell.imag**2) * dV
        normalization_factor = 1 / np.sqrt(integral_value)

        # Store in cache using tuple key
        phi_orbital.normalization_factor_dict[k_point_key] = normalization_factor

        return normalization_factor


    @staticmethod
    def construct_total_orbitals_in_unit_cell(
        phi_orbital : 'PDOSCalculator.PhiOrbital', 
        k_point_fractional : np.ndarray[float],
        index_mask_dict : 'PDOSCalculator.IndexMaskDictType') -> np.ndarray[np.complex128]:
        
        assert isinstance(phi_orbital, PDOSCalculator.PhiOrbital), PHI_ORBITAL_INPUT_NOT_PHI_ORBITAL_ERROR.format(type(phi_orbital))
        assert isinstance(k_point_fractional, np.ndarray), K_POINT_FRACTIONAL_INPUT_NOT_NP_ARRAY_ERROR.format(type(k_point_fractional))
        assert isinstance(index_mask_dict, dict), INDEX_MASK_DICT_INPUT_NOT_DICT_ERROR.format(type(index_mask_dict))

        # Convert k_point_fractional to tuple for use as dictionary key (numpy arrays are unhashable)
        k_point_key = tuple(k_point_fractional)

        # If the total orbitals in the unit cell is already computed, then return it
        if k_point_key in phi_orbital.total_orbitals_in_unit_cell_dict:
            return phi_orbital.total_orbitals_in_unit_cell_dict[k_point_key]

        # If the total orbitals in the unit cell is not computed, then compute it
        _phi_orbital_unit_cell = np.zeros_like(index_mask_dict[(0,0,0)][0], dtype=np.complex128)

        for shift_index_vector, phi_value in phi_orbital.phi_value_dict.items():
            _index_mask, _ = index_mask_dict[shift_index_vector]
            bloch_phase = np.exp(1j * 2 * np.pi * np.dot(k_point_fractional, shift_index_vector))
            _phi_orbital_unit_cell[_index_mask] += phi_value * bloch_phase

        # Store in cache using tuple key
        phi_orbital.total_orbitals_in_unit_cell_dict[k_point_key] = _phi_orbital_unit_cell

        return _phi_orbital_unit_cell


    @staticmethod
    def compute_normalization_factor_in_unit_cell_with_certain_k_point(
        dV : float,
        k_point_fractional : np.ndarray[float], 
        phi_orbital : 'PDOSCalculator.PhiOrbital',
        index_mask_dict : 'PDOSCalculator.IndexMaskDictType') -> np.ndarray[float]:
        """
        Compute the normalization factor in the unit cell with a certain k-point
        """
        raise NotImplementedError("This function is deprecated, please use compute_normalization_factor() instead")
        assert isinstance(dV, float), DV_NOT_FLOAT_ERROR.format(type(dV))
        assert isinstance(k_point_fractional, np.ndarray), K_POINT_FRACTIONAL_INPUT_NOT_NP_ARRAY_ERROR.format(type(k_point_fractional))
        assert isinstance(phi_orbital, PDOSCalculator.PhiOrbital), PHI_ORBITAL_INPUT_NOT_PHI_ORBITAL_ERROR.format(type(phi_orbital))
        assert isinstance(index_mask_dict, dict), INDEX_MASK_DICT_INPUT_NOT_DICT_ERROR.format(type(index_mask_dict))
       
        _phi_orbital_unit_cell = np.zeros_like(index_mask_dict[(0,0,0)][0], dtype=np.complex128)

        for shift_index_vector, phi_value in phi_orbital.phi_value_dict.items():
            _index_mask, _ = index_mask_dict[shift_index_vector]
            bloch_phase = np.exp(1j * 2 * np.pi * np.dot(k_point_fractional, shift_index_vector))
            _phi_orbital_unit_cell[_index_mask] += phi_value * bloch_phase

        integral_value = np.sum(_phi_orbital_unit_cell.real**2 + _phi_orbital_unit_cell.imag**2) * dV
        normalization_factor = 1 / np.sqrt(integral_value)

        return normalization_factor



    def project_wavefunction_onto_atomic_orbital_basis_and_return_the_corresponding_coefficients(self, 
        orthogonalize_atomic_orbitals : bool = True, 
        mutiply_weights               : bool = True,
        save_fname                    : Optional[str] = None,
        print_overlap_matrix          : bool          = False,  # print overlap matrix 
        ) -> np.ndarray[float]:
        r"""
        Project the wavefunction onto the atomic orbital basis, and return the corresponding coefficients
        \begin{equation}
            P_{n \mathbf{k}, \mu}=\int \psi_{n \mathbf{k}}^*(\mathbf{r}) \Phi_\mu^{\text {orth }}(\mathbf{r}) d^3 \mathbf{r}
        \end{equation}

        return:
            P_all_array : np.ndarray[float], shape = (kpt_num, band_num, tot_orbital_num)
        """
        print("Projecting the wavefunction onto the atomic orbital basis, and return the corresponding coefficients")

        # Sequential version
        if self.kpt_num <= self.k_point_parallel_threshold:
            print("\t Using sequential version")
            P_all_array = self._project_wavefunction_onto_atomic_orbital_basis_sequential(
                orthogonalize_atomic_orbitals = orthogonalize_atomic_orbitals, 
                print_overlap_matrix = print_overlap_matrix,
                mutiply_weights = mutiply_weights)
        else:
            print("\t Using parallel version")
            P_all_array = self._project_wavefunction_onto_atomic_orbital_basis_parallel(
                orthogonalize_atomic_orbitals = orthogonalize_atomic_orbitals, 
                print_overlap_matrix = print_overlap_matrix,
                mutiply_weights = mutiply_weights)

        # ===== save: k-point + combination coefficients P_{n,k,mu} =====
        if save_fname is not None:
            print(f"\t Saving to {save_fname} ...")
            A = P_all_array   # (K, B, M) complex
            K, B, M = A.shape
            k_red = self.kpts_store[:K, :]           # (K,3) reduced k

            import os
            os.makedirs(os.path.dirname(save_fname), exist_ok=True)
            with open(save_fname, "w") as f:
                f.write("# columns: kpt  band  mu  kx_red  ky_red  kz_red  Re(P_nkmu)  Im(P_nkmu)\n")
                for k in range(K):
                    # 1-based indices like MATLAB
                    kpt_col  = np.full((B*M, 1), k+1, dtype=int)
                    band_col = np.repeat(np.arange(1, B+1), M).reshape(-1, 1)
                    mu_col   = np.tile(np.arange(1, M+1), B).reshape(-1, 1)
                    kred_col = np.tile(k_red[k], (B*M, 1))  # (B*M, 3)
                    Pk = A[k]                               # (B, M)
                    Re = Pk.real.reshape(-1, 1)
                    Im = Pk.imag.reshape(-1, 1)
                    rows = np.hstack([kpt_col, band_col, mu_col, kred_col, Re, Im])
                    np.savetxt(f, rows,
                               fmt=["%d","%d","%d","%.8f","%.8f","%.8f","%.15e","%.15e"])

        return P_all_array



    def project_wavefunction_onto_single_atom_atomic_orbital_basis_and_return_the_corresponding_coefficients(self, 
        atom_index: int, 
        mutiply_weights: bool = True, 
        save_fname: Optional[str] = None) -> np.ndarray[float]:
        """
        Project the wavefunction onto the atomic orbital basis of a single atom, and return the corresponding coefficients
        This function is called only if we are not to orthogonalize the atomic orbitals
        """
        atom = self.atoms[atom_index]
        assert isinstance(atom, PDOSCalculator.Atom), "atom must be a PDOSCalculator.Atom, get type {} instead".format(type(atom))
        assert isinstance(mutiply_weights, bool), "mutiply_weights must be a bool, get type {} instead".format(type(mutiply_weights))

        atom_name = self.atom_species_list[atom.atom_type_index]
        print("Projecting the wavefunction onto the atomic orbital basis of a single atom")
        print("\tatom_type = {}, atomic_number = {}, atom_index = {}, num_orbitals = {}".format(atom_name, name_to_atomic_number(atom_name), atom_index, atom.num_orbitals))
        if self.kpt_num <= self.k_point_parallel_threshold:
            print("\t Using sequential version")
            P_single_atom_array = self._project_wavefunction_onto_single_atom_atomic_orbital_basis_sequential(atom = atom, mutiply_weights = mutiply_weights)
        else:
            print("\t Using parallel version")
            P_single_atom_array = self._project_wavefunction_onto_single_atom_atomic_orbital_basis_parallel(atom = atom, mutiply_weights = mutiply_weights)


        # ===== save: k-point + combination coefficients P_{n,k,mu} =====
        if save_fname is not None:
            print("WARNING: Not to orthogonalize the atomic orbitals is not implemented yet, so not saving the projection data")


        # raise NotImplementedError("Not to orthogonalize the atomic orbitals is not implemented yet")
        return P_single_atom_array


    def _project_wavefunction_onto_atomic_orbital_basis_sequential(self, 
        orthogonalize_atomic_orbitals : bool = True, 
        print_overlap_matrix          : bool = False,
        mutiply_weights               : bool = True
        ) -> np.ndarray[float]:
        """
        Project the wavefunction onto the atomic orbital basis, and return the corresponding coefficients
        """

        # Initialize the P_all_array
        P_all_array = np.zeros((self.kpt_num, self.band_num, self.tot_orbital_num), dtype = np.complex128)
        if orthogonalize_atomic_orbitals:
            self.overlap_matrix = np.zeros((self.kpt_num, self.tot_orbital_num, self.tot_orbital_num), dtype = np.complex128)
            self.orbital_weights = None
        else:
            self.overlap_matrix = None
            self.orbital_weights = np.zeros((self.kpt_num, self.tot_orbital_num,), dtype = np.complex128)
        
        # Loop over all k-points
        for kpt_index in range(self.kpt_num):

            # Initialize the P_single_kpt_array
            P_single_kpt_array = np.zeros((self.band_num, self.tot_orbital_num), dtype = np.complex128)

            # Get the k-vector (note the scaling of box length)
            kx_frac = self.kpts_store[kpt_index, 0]
            ky_frac = self.kpts_store[kpt_index, 1]
            kz_frac = self.kpts_store[kpt_index, 2]

            # Get the wavefunction of the k-point and rearangeig by columns
            psi_kpt = self.psi_total[:, :, kpt_index]
            perm = self.I_indices[:, kpt_index].astype(int)
            psi_kpt = psi_kpt[:, perm]  # (tot_grid_pt, band_num)

            # Compute the overlap matrix
            if orthogonalize_atomic_orbitals:
                _overlap_matrix = self.calculate_overlap_matrix(
                    atoms = self.atoms,
                    dV = self.dV,
                    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                    save_fname = self.out_dirname + "/overlap_matrix_kpt_{}.txt".format(kpt_index) if print_overlap_matrix else None,
                    )
                self.overlap_matrix[kpt_index, :, :] = _overlap_matrix
            else:
                _orbital_weights = self.calculate_orbital_weights(
                    atoms = self.atoms,
                    dV = self.dV,
                    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                    )
                self.orbital_weights[kpt_index, :] = _orbital_weights

            # S_inv_sqrt.shape = (total_orbitals_num, total_orbitals_num)
            # In principle, you should still compute Phi^{orth} as Phi * S_inv_sqrt, but the stacked Phi array is not 
            # readily available. So, we prospone the calculation of such mutiplication after the projection is completed, namely
            #        \bra \psi_{nk} | \Phi_{\mu}^{orth} \ket  =  \bra \psi_{nk} | \Phi_{\nu} \ket S^{-1/2}_{\mu\nu}

            if self.overlap_matrix is not None:
                _S_inv_sqrt = fractional_matrix_power(_overlap_matrix, -0.5)
            else:
                _w_inv_sqrt = _orbital_weights ** (-0.5)

            # Compute the overlap with the atomic orbitals
            orbital_index = 0
            for atom in self.atoms:
                
                for orbital in atom.phi_orbitals_list:
                    
                    # compute the overlap with the atomic orbitals
                    _phi_orbital_unit_cell = self.construct_total_orbitals_in_unit_cell(
                        phi_orbital = orbital, 
                        k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                        index_mask_dict = atom.index_mask_and_effective_grid_point_positions_dict,
                        )
                    projected_value = np.sum(_phi_orbital_unit_cell[:, np.newaxis] * psi_kpt, axis = 0)


                    # Compute the normalization factor
                    normalization_factor = self.compute_normalization_factor(
                        dV = self.dV,
                        k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                        phi_orbital = orbital,
                        index_mask_dict = atom.index_mask_and_effective_grid_point_positions_dict,
                        )

                    # Update the P_all_array
                    P_single_kpt_array[:, orbital_index] = projected_value * normalization_factor

                    # Update the orbital index
                    orbital_index += 1
            
            # Orthogonalize the P_single_kpt_array
            if mutiply_weights:
                if orthogonalize_atomic_orbitals:
                    P_single_kpt_array = np.einsum("jk,kl->jl", P_single_kpt_array, _S_inv_sqrt)
                else:
                    P_single_kpt_array = np.einsum("jk,k->jk", P_single_kpt_array, _w_inv_sqrt)

            # Update the P_all_array
            P_all_array[kpt_index, :, :] = P_single_kpt_array
            
            print(f"\r\t {kpt_index + 1} out of {self.kpt_num} k-points done!!", end='', flush=True)

        print()  # New line after progress updates
        # Multiply by the discrete volume element dV
        P_all_array *= self.dV


        return P_all_array


    def _project_wavefunction_onto_single_atom_atomic_orbital_basis_sequential(self, 
        atom: 'PDOSCalculator.Atom', 
        mutiply_weights: bool = True) -> np.ndarray[float]:
        """
        Project the wavefunction onto the atomic orbital basis, and return the corresponding coefficients
        When this function is called, we are not to orthogonalize the atomic orbitals
        """
        
        # Initialize the P_single_atom_array
        P_single_atom_array = np.zeros((self.kpt_num, self.band_num, atom.num_orbitals), dtype = np.complex128)

        # Loop over all k-points
        for kpt_index in range(self.kpt_num):

            # Initialize the P_single_kpt_single_atom_array
            P_single_kpt_single_atom_array = np.zeros((self.band_num, atom.num_orbitals), dtype = np.complex128)

            # Get the k-vector (note the scaling of box length)
            kx_frac = self.kpts_store[kpt_index, 0]
            ky_frac = self.kpts_store[kpt_index, 1]
            kz_frac = self.kpts_store[kpt_index, 2]

            # Get the wavefunction of the k-point and rearangeig by columns
            psi_kpt = self.psi_total[:, :, kpt_index]
            perm = self.I_indices[:, kpt_index].astype(int)
            psi_kpt = psi_kpt[:, perm]  # (tot_grid_pt, band_num)

            # Compute the orbital weights
            _orbital_weights = self.calculate_orbital_weights(
                    atoms = self.atoms,
                    dV = self.dV,
                    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                    )
            _w_inv_sqrt = _orbital_weights ** (-0.5)

            # Compute the overlap with the atomic orbitals
            for orbital_index, orbital in enumerate(atom.phi_orbitals_list):
                
                # Loop over all the shifted unit cells and compute the overlap
                _phi_orbital_unit_cell = self.construct_total_orbitals_in_unit_cell(
                    phi_orbital = orbital, 
                    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                    index_mask_dict = atom.index_mask_and_effective_grid_point_positions_dict,
                    )
                projected_value = np.sum(_phi_orbital_unit_cell[:, np.newaxis] * psi_kpt, axis = 0)
                
                # Compute the normalization factor
                normalization_factor = self.compute_normalization_factor(
                    dV = self.dV,
                    k_point_fractional = np.array([kx_frac, ky_frac, kz_frac]),
                    phi_orbital = orbital,
                    index_mask_dict = atom.index_mask_and_effective_grid_point_positions_dict,
                    )

                # Update the P_all_array
                P_single_kpt_single_atom_array[:, orbital_index] = projected_value * normalization_factor
            
            # Orthogonalize the P_all_array
            start_idx, end_idx = self._get_atom_orbital_indices(target_atom = atom)
            if mutiply_weights:
                P_single_kpt_single_atom_array = np.einsum("jk,k->jk", P_single_kpt_single_atom_array, _w_inv_sqrt[start_idx : end_idx])

            # Update the P_single_atom_array
            P_single_atom_array[kpt_index, :, :] = P_single_kpt_single_atom_array

            print(f"\r\t {kpt_index + 1} out of {self.kpt_num} k-points done!!", end='', flush=True)

        print()  # New line after progress updates
        # Multiply by the discrete volume element dV
        P_single_atom_array *= self.dV

        return P_single_atom_array


    def _get_atom_orbital_indices(self, target_atom: 'PDOSCalculator.Atom', sum_over_m_index : bool = False) -> tuple[int, int]:
        """
        Get the starting and ending orbital indices for a specific atom in the global orbital numbering.
        Returns (start_index, end_index) where end_index is exclusive.
        """
        assert isinstance(target_atom, PDOSCalculator.Atom), "target_atom must be a PDOSCalculator.Atom, get type {} instead".format(type(target_atom))
        assert isinstance(sum_over_m_index, bool), "sum_over_m_index must be a bool, get type {} instead".format(type(sum_over_m_index))
        
        start_index = 0
        if not sum_over_m_index:
            for atom in self.atoms:
                if id(atom) == id(target_atom):
                    return start_index, start_index + atom.num_orbitals
                start_index += atom.num_orbitals
            raise ValueError(f"Target atom not found in atoms list")
        else:
            for atom in self.atoms:
                if id(atom) == id(target_atom):
                    return start_index, start_index + atom.num_psdwfn
                start_index += atom.num_orbitals
            raise ValueError(f"Target atom not found in atoms list")


    def _project_single_atom_wavefunction_onto_atomic_orbital_basis(self, 
        pdos_atom_index_list : List[int], 
        orthogonalize_atomic_orbitals : bool = True, 
        save_fname : Optional[str] = None,
        print_overlap_matrix : bool = False,
        ) -> np.ndarray[float]:

        """
        Project the wavefunction onto the atomic orbital basis, and return the corresponding coefficients
        """
        P_single_atom_array_list = []

        if orthogonalize_atomic_orbitals:
            # initialize the P_all_array
            P_all_array = self.project_wavefunction_onto_atomic_orbital_basis_and_return_the_corresponding_coefficients(
                orthogonalize_atomic_orbitals = True,
                mutiply_weights = True,
                save_fname = save_fname, 
                print_overlap_matrix = print_overlap_matrix,
                )
            for pdos_atom_index in pdos_atom_index_list:
                # get the atom
                atom = self.atoms[pdos_atom_index]
                start_idx, end_idx = self._get_atom_orbital_indices(target_atom = atom)
                _P_single_atom_array = P_all_array[:, :, start_idx : end_idx]
                P_single_atom_array_list.append(_P_single_atom_array)
        else:
            # not to orthogonalize the atomic orbitals, thus all the calculation can be accelerated
            for pdos_atom_index in pdos_atom_index_list:
                if print_overlap_matrix:
                    print(NOT_ORTHOGONALIZE_ATOMIC_ORBITALS_OVERLAP_MATRIX_NOT_PRINTED_WARNING)

                _P_single_atom_array = self.project_wavefunction_onto_single_atom_atomic_orbital_basis_and_return_the_corresponding_coefficients(
                    atom_index = pdos_atom_index, 
                    mutiply_weights = True, 
                    save_fname = save_fname)
                P_single_atom_array_list.append(_P_single_atom_array)

        return P_single_atom_array_list


    def _project_wavefunction_onto_atomic_orbital_basis_parallel(self, 
        orthogonalize_atomic_orbitals: bool = True, 
        print_overlap_matrix: bool = False,
        mutiply_weights: bool = True,
        ) -> np.ndarray:
        """
        Parallel projection: unified worker that processes complete k-point calculation.
        Same logic as sequential version, but parallelized over k-points.
        """
        import multiprocessing as mp
        import time

        # Initialize overlap matrix or orbital weights storage (same as sequential version)
        if orthogonalize_atomic_orbitals:
            self.overlap_matrix = np.zeros((self.kpt_num, self.tot_orbital_num, self.tot_orbital_num), dtype=np.complex128)
            self.orbital_weights = None
        else:
            self.overlap_matrix = None
            self.orbital_weights = np.zeros((self.kpt_num, self.tot_orbital_num,), dtype=np.complex128)

        # Put psi_total into shared memory
        time_shm_start = time.time()
        psi_meta, psi_owner = _shm_from_ndarray(self.psi_total)
        time_shm_end = time.time()
        print(f"\t Shared memory warm-up time: {time_shm_end - time_shm_start:.4f} seconds ({self.psi_total.nbytes / 1024 / 1024:.2f} MB)")

        # Result array
        P_all_array = np.zeros((self.kpt_num, self.band_num, self.tot_orbital_num), dtype=np.complex128)

        # Setup initializer arguments for unified worker
        initargs = (
            psi_meta,
            self.I_indices,
            self.kpts_store,
            self.atoms,
            self.dV,
            orthogonalize_atomic_orbitals,
            mutiply_weights,
            self.out_dirname,
            print_overlap_matrix,
            self.tot_orbital_num,
            self.kpt_num,
        )

        # Parallel processing: each worker processes one complete k-point
        with mp.Pool(processes=mp.cpu_count(),
                    initializer=_init_unified_worker,
                    initargs=initargs) as pool:
            chunksize = max(1, self.kpt_num // (4 * mp.cpu_count()) or 1)
            for kpt_index, result in pool.imap_unordered(
                _process_single_kpoint_complete_worker, range(self.kpt_num), chunksize=chunksize):
                P_all_array[kpt_index, :, :] = result

        # Multiply by dV (same as sequential version)
        P_all_array *= self.dV

        # Release shared memory
        psi_owner.close()
        psi_owner.unlink()
        
        return P_all_array


    def _project_wavefunction_onto_single_atom_atomic_orbital_basis_parallel(
        self, atom: 'PDOSCalculator.Atom', mutiply_weights: bool = True) -> np.ndarray[float]:
        """
        Project the wavefunction onto the atomic orbital basis, and return the corresponding coefficients.
        Same logic as sequential version.
        """
        import multiprocessing as mp
        import time
        
        # Get atom orbital indices for weights
        start_idx, end_idx = self._get_atom_orbital_indices(target_atom=atom)
            
        # --- pack the single atom into a pure numpy structure for serialization ---
        atom_packed = {
            "num_orbitals": atom.num_orbitals,
            "orbitals": []
        }
        for orbital in atom.phi_orbitals_list:
            entries = []
            for (i, j, k), phi_value in orbital.phi_value_dict.items():
                mask, _ = atom.index_mask_and_effective_grid_point_positions_dict.get((i, j, k), (None, None))
                assert mask is not None, INDEX_MASK_NOT_FOUND_ERROR.format(i, j, k)
                entries.append({
                    "ijk": (int(i), int(j), int(k)),
                    "phi": np.asarray(phi_value),
                    "mask": np.asarray(mask)
                })
            atom_packed["orbitals"].append({"entries": entries})

        # --- 1) put psi_total into shared memory (only copy once) ---
        time_shm_start = time.time()
        psi_meta, psi_owner = _shm_from_ndarray(self.psi_total)
        time_shm_end = time.time()
        print(f"\t Shared memory warm-up time: {time_shm_end - time_shm_start:.4f} seconds ({self.psi_total.nbytes / 1024 / 1024:.2f} MB)")

        try:
            # --- 2) pre-allocate the result ---
            P_single_atom_array = np.zeros((self.kpt_num, self.band_num, atom.num_orbitals), dtype=np.complex128)

            # --- 3) start the pool: use the _init_worker_for_single_atom ---
            initargs = (
                psi_meta,
                self.I_indices,
                self.kpts_store,
                getattr(self, "latvec", None),
                getattr(self, "latvec_inv", None),
                self.is_orthogonal_lattice,
                self.Lx, self.Ly, self.Lz,
                atom_packed,                # packed atom data
                self.atoms,                 # full atoms list for calculate_orbital_weights
                self.dV,
                mutiply_weights,
                start_idx,
                end_idx,
                self.kpt_num,
            )

            with mp.Pool(processes=mp.cpu_count(),
                        initializer=_init_worker_for_single_atom,
                        initargs=initargs) as pool:
                chunksize = max(1, self.kpt_num // (4*mp.cpu_count()) or 1)
                for kpt_index, result in pool.imap_unordered(
                    _process_single_kpoint_for_single_atom_worker,
                    range(self.kpt_num),
                    chunksize=chunksize
                ):
                    P_single_atom_array[kpt_index, :, :] = result

            # --- 4) post-processing: multiply by dV (same as sequential version) ---
            P_single_atom_array *= self.dV

        finally:
            # --- 5) clean up the shared memory ---
            psi_owner.close()
            psi_owner.unlink()
        
        return P_single_atom_array



    @classmethod
    def load_projections_from_txt(cls, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read the combination coefficients file (text) output by the save routine
        Column order: kpt band mu kx_red ky_red kz_red Re(P) Im(P)

        Returns
        -------
        P : np.ndarray
            complex array of shape (K, B, M), corresponding to P_{n,k,mu}
        kpts_red : np.ndarray
            shape (K, 3) reduced k-point coordinates
        """
        # read in (automatically skip the header starting with #; support comma or blank separation)
        arr = np.genfromtxt(path, comments="#", delimiter=None)
        if arr.ndim == 1:  # only one row
            arr = arr[None, :]

        if arr.shape[1] < 8:
            raise ValueError("Input file must have at least 8 columns: "
                             "kpt band mu kx_red ky_red kz_red Re Im")

        kpt_col  = arr[:, 0].astype(int)
        band_col = arr[:, 1].astype(int)
        mu_col   = arr[:, 2].astype(int)
        kx_red   = arr[:, 3]
        ky_red   = arr[:, 4]
        kz_red   = arr[:, 5]
        Re       = arr[:, 6]
        Im       = arr[:, 7]

        K = int(kpt_col.max())
        B = int(band_col.max())
        M = int(mu_col.max())

        # Basic consistency check
        expected = K * B * M
        if arr.shape[0] != expected:
            raise ValueError(f"Row count mismatch: got {arr.shape[0]}, expected {expected} (= K*B*M).")

        # Construct P(k,b,m)
        P = np.empty((K, B, M), dtype=np.complex128)
        P[kpt_col - 1, band_col - 1, mu_col - 1] = Re + 1j * Im

        # Extract the reduced coordinates of each kpt (each kpt should be consistent, take the first one)
        kpts_red = np.zeros((K, 3), dtype=float)
        for kk in range(1, K + 1):
            mask = (kpt_col == kk)
            if not np.any(mask):
                raise ValueError(f"No rows found for kpt={kk}")
            kpts_red[kk - 1, 0] = kx_red[mask][0]
            kpts_red[kk - 1, 1] = ky_red[mask][0]
            kpts_red[kk - 1, 2] = kz_red[mask][0]

        return P, kpts_red



    def compute_pdos_dos(self, data_projections : np.ndarray):
        E, PDOS_plot, DOS = _compute_pdos_dos(
            eig_eV = self.eign * self.Ha2eV,
            DATA_projections = data_projections,
            kpt_wts_store = self.kpt_wts_store,
            mu_eV = self.mu_PDOS,
            N_PDOS = self.N_PDOS,
            min_E_plot = self.min_E_plot,
            max_E_plot = self.max_E_plot,
            kpt_mesh_shape = (self.kpt1, self.kpt2, self.kpt3),
        )
        return E, PDOS_plot, DOS


    def __print_parameters(self):

        
        # Print initialization parameters
        self.print_initialization_parameters()

        # Print SPARC's output file parameters (.out)
        self.print_sparc_out_file_parameters()

        # Print SPARC's static file parameters (.static)
        self.print_sparc_static_file_parameters()

        # Print SPARC's eigen file parameters (.eigen)
        self.print_sparc_eigen_file_parameters()

        # Print SPARC's orbital file parameters (.psi)
        self.print_sparc_orbital_file_parameters()

        # Print UPF files parameters (.upf)
        for i in range(self.natom_types):
            self.print_atomic_wave_function(self.atomic_wave_function_list[i])

        # Print the index mask, effective grid point positions and phi_orbitals_list information for all atoms
        self.print_atoms_updated_parameters()
                     


    def print_parameters(self, save_fname : Optional[str] = None):
        """
        Some visualization, if necessary
            Although all printing/plotting actions are nested in this function, they can be called separately
        """
        if save_fname is not None:
            assert isinstance(save_fname, str), SAVE_FNAME_NOT_STRING_ERROR.format(type(save_fname))

            # Create a log file
            log_file_path = os.path.join(self.out_dirname, save_fname)
            log_file = open(log_file_path, 'w', encoding='utf-8')

            # Redirect the standard output
            class TeeOutput:
                def __init__(self, file):
                    self.file = file
                    self.stdout = sys.stdout
                
                def write(self, text):
                    self.stdout.write(text)
                    self.file.write(text)
                    self.file.flush()
                
                def flush(self):
                    self.stdout.flush()
                    self.file.flush()

            sys.stdout = TeeOutput(log_file)

        # Print the parameters
        self.__print_parameters()

        # Close the log file
        if save_fname is not None:
            log_file.close()
            sys.stdout = sys.__stdout__

        

    # @print_time("Time taken to read the SPARC's output file (function={}) : {:.4f} seconds \n")
    def read_sparc_out_file_parameters(self, fname : str):
        assert isinstance(fname, str), FNAME_NOT_STRING_ERROR.format(type(fname))
        print("Reading the SPARC's output file")
        print("\t fname = ", fname)

        # read the SPARC's output file
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
        
        # KPOINT_GRID
        kpt_values = re.findall(r'\d+', lines[_find_last_line_index("KPOINT_GRID", lines)])
        self.kpt1, self.kpt2, self.kpt3 = map(int, kpt_values)

        # FD_GRID
        fd_values = re.findall(r'\d+', lines[_find_last_line_index("FD_GRID", lines)])
        self.Nx, self.Ny, self.Nz = map(int, fd_values)

        # XC_FUNCTIONAL
        self.xc_functional = lines[_find_last_line_index("EXCHANGE_CORRELATION", lines)].split()[-1]

        # BC
        parts = lines[_find_last_line_index("BC", lines)].split()
        self.bcx, self.bcy, self.bcz = parts[1], parts[2], parts[3]

        # NSTATES
        self.band_num = int(re.findall(r'\d+', lines[_find_last_line_index("NSTATES", lines)])[0])

        # Number of symmetry adapted k-points
        self.kpt_num = int(re.findall(r'\d+', lines[_find_last_line_index("Number of symmetry adapted k-points", lines)])[0])

        # Fermi level
        fermi_str = re.findall(r'[+-]?[0-9]+\.[0-9]+E[+-]?[0-9]+', lines[_find_last_line_index("Fermi level", lines)])[0]
        self.fermi = float(fermi_str) * self.Ha2eV

        # get the psp file names
        self.psp_fname_list = get_psp_fname_list(output_fname = fname)

        # update the number of grid points according to the boundary conditions
        if self.bcx == "D": self.Nx = self.Nx + 1
        if self.bcy == "D": self.Ny = self.Ny + 1
        if self.bcz == "D": self.Nz = self.Nz + 1

        # parse the atom types and species
        self.atom_count_list = self.parse_atom_types_from_sparc_out_file(fname = fname)
        self.tot_atoms_num = sum(self.atom_count_list)
        self.out_file_loaded_flag = True

        if not self.is_relaxation:        
            # Lattice setup will be handled by check_lattice_type() method
            self.check_and_setup_lattice_parameters(lines = lines)
        else:
            # Read in the relaxation flag for furthur usage
            self.relax_flag = int(re.findall(r'\d+', lines[_find_last_line_index("RELAX_FLAG", lines)])[0])
            print("\t self.relax_flag = ", self.relax_flag)

        # check if all the parameters are read in correctly
        self.check_sparc_out_file_parameters()


    def check_and_setup_lattice_parameters(self, lines : List[str]):
        """
        Check and setup the lattice type based on the input file
        """
        # Check if LATVEC is present in the output file
        with open(self.output_fname, 'r') as f:
            content = f.read()
            
        self.is_orthogonal_lattice = self.check_orthogonal_lattice(lines = lines)

        if self.is_orthogonal_lattice:
            self.setup_orthogonal_lattice(lines = lines)
        else:
            self.setup_non_orthogonal_lattice(lines = lines)


    def check_orthogonal_lattice(self, lines : List[str]) -> bool:
        """
        Check if the lattice is orthogonal
        return:
            True  : orthogonal lattice
            False : non-orthogonal lattice
        """
        # try to directly determine the lattice type from lattice vectors, if proviced

        # Find LATVEC section (these are normalized direction vectors)
        latvec_start = -1
        for i, line in enumerate(lines):
            if 'LATVEC:' in line:
                latvec_start = i + 1
                break
        
        if latvec_start == -1:
            print(LATVEC_SECTION_NOT_FOUND_WARNING)
            return True
        else:

            # Read three lattice vector directions (normalized)
            latvec_lines = lines[latvec_start:latvec_start + 3]
            latvec_directions = np.zeros((3, 3))
            
            for i, line in enumerate(latvec_lines):
                values = [float(x) for x in line.strip().split()]
                if len(values) != 3:
                    raise ValueError(INVALID_LATTICE_VECTOR_FORMAT_ERROR.format(line))
                
                if not np.isclose(np.linalg.norm(values), 1.0, atol = 1e-6):
                    values = np.array(values) / np.linalg.norm(values)
                assert np.isclose(np.linalg.norm(values), 1.0, atol = 1e-6), LATTICE_VECTOR_NOT_NORMALIZED_ERROR.format(i+1, np.linalg.norm(values))
                
                latvec_directions[i, :] = values

            return np.allclose(latvec_directions, np.eye(3), atol = 1e-6)



    def setup_orthogonal_lattice(self, lines : List[str]):
        """
        Setup parameters for orthogonal lattice
        """

        # For orthogonal lattice, the lattice vectors are the same as the lattice constants
        self.Lx, self.Ly, self.Lz = self.get_lattice_scale_from_output_file(lines = lines)

        # calculate the lattice vectors
        self.x_lattice_vector = np.array([self.Lx, 0, 0])
        self.y_lattice_vector = np.array([0, self.Ly, 0])
        self.z_lattice_vector = np.array([0, 0, self.Lz])

        # calculate the grid spacing
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        # calculate the volume of the unit grid cell
        self.dV = self.dx * self.dy * self.dz

        # Cell volume
        self.cell_volume = self.Lx * self.Ly * self.Lz


    def setup_non_orthogonal_lattice(self, lines : List[str]):
        """
        Setup parameters for non-orthogonal lattice
        """
        # Read lattice vectors from output file
        self.read_lattice_vectors_and_cell_scale_from_output_for_non_orthogonal_lattice(lines = lines)
        
        # Calculate derived quantities
        self.calculate_non_orthogonal_parameters()

        # Check the lattice parameters
        self.check_lattice_parameters()


    @staticmethod
    def get_lattice_scale_from_output_file(lines : List[str]) -> Tuple[float, float, float]:
        """
        Get lattice scale from output file
        """
        # For non-orthogonal lattice, read CELL parameters (these are the actual lengths)
        try:
            cell_line = lines[_find_last_line_index("CELL", lines)]
            cell_values = re.findall(SCIENTIFIC_NOTATION_PATTERN, cell_line) # Match both regular floats and scientific notation (e.g., 7.1968E+00)
            assert len(cell_values) == 3
            return map(float, cell_values)
        except:
            pass
        
        try:
            latvec_line = lines[_find_last_line_index("LATVEC_SCALE", lines)]
            latvec_values = re.findall(SCIENTIFIC_NOTATION_PATTERN, latvec_line) # Match both regular floats and scientific notation (e.g., 7.1968E+00)
            assert len(latvec_values) == 3
            return map(float, latvec_values)
        except:
            raise ValueError(LATVEC_SCALE_OR_CELL_NOT_FOUND_ERROR)
            
    def read_lattice_vectors_and_cell_scale_from_output_for_non_orthogonal_lattice(self, lines : List[str]):
        """
        Read lattice vectors and cell scale from SPARC output file
        Initialized paramters:
            self.Lx, self.Ly, self.Lz : float
            self.latvec : np.ndarray[float]
        """
        # For non-orthogonal lattice, read CELL parameters (these are the actual lengths)
        self.Lx, self.Ly, self.Lz = self.get_lattice_scale_from_output_file(lines = lines)

        # Find LATVEC section (these are normalized direction vectors)
        latvec_start = -1
        for i, line in enumerate(lines):
            if 'Lattice vectors (Bohr)' in line:
                latvec_start = i + 1
                break
        
        if latvec_start == -1:
            raise ValueError("Cell section not found in output file")
        
        # Read three lattice vector directions (normalized)
        latvec_row = np.zeros((3, 3))
        latvec_lines = lines[latvec_start:latvec_start + 3]

        for i, line in enumerate(latvec_lines):
            values = [float(x) for x in line.strip().split()]
            if len(values) != 3:
                raise ValueError(INVALID_LATTICE_VECTOR_FORMAT_ERROR.format(line))
            
            if not np.isclose(np.linalg.norm(values), 1.0, atol = 1e-6):
                print(LATTICE_VECTOR_NOT_NORMALIZED_WARNING.format(i+1, np.linalg.norm(values)))
                # values = np.array(values) / np.linalg.norm(values)
            # assert np.isclose(np.linalg.norm(values), 1.0, atol = 1e-6), LATTICE_VECTOR_NOT_NORMALIZED_ERROR.format(i+1, np.linalg.norm(values))
            
            latvec_row[i, :] = values

        # Store as column vectors (transpose) for compatibility with our code
        self.latvec = latvec_row.T
        
        print("\t Cell lengths: Lx={:.6f}, Ly={:.6f}, Lz={:.6f} Bohr".format(self.Lx, self.Ly, self.Lz))
        print("\t Actual lattice vectors (Bohr):")
        print("\t  a1 = ({:.6f}, {:.6f}, {:.6f})".format(self.latvec[0, 0], self.latvec[1, 0], self.latvec[2, 0]))
        print("\t  a2 = ({:.6f}, {:.6f}, {:.6f})".format(self.latvec[0, 1], self.latvec[1, 1], self.latvec[2, 1]))
        print("\t  a3 = ({:.6f}, {:.6f}, {:.6f})".format(self.latvec[0, 2], self.latvec[1, 2], self.latvec[2, 2]))


    def calculate_non_orthogonal_parameters(self):
        """
        Calculate derived parameters for non-orthogonal lattice
        """

        # Calculate determinant and inverse
        self.latvec_det = np.linalg.det(self.latvec)
        if abs(self.latvec_det) < 1e-12:
            raise ValueError("Lattice vectors are linearly dependent")
        self.latvec_inv = np.linalg.inv(self.latvec)

        # Cell volume is the determinant of the lattice vectors matrix
        self.cell_volume = abs(self.latvec_det)
        
        # Calculate metric tensor g_ij = a_i · a_j
        self.metric_tensor = np.dot(self.latvec, self.latvec.T)
        self.metric_tensor_inv = np.linalg.inv(self.metric_tensor)
        
        # Jacobian determinant for coordinate transformation
        self.jacobian_det = abs(self.latvec_det)
        
        # Set lattice vectors for compatibility
        self.x_lattice_vector = self.latvec[:, 0]
        self.y_lattice_vector = self.latvec[:, 1]
        self.z_lattice_vector = self.latvec[:, 2]

        
        # calculate the grid spacing
        # For non-orthogonal lattice, we need to calculate grid spacing based on actual lattice vectors
        # The grid spacing should be calculated from the actual lattice vector lengths
        # We use the CELL values (which are the actual lengths) divided by grid points
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        # Calculate volume element
        # For non-orthogonal lattice, the volume element is the cell volume divided by total grid points
        self.dV = self.cell_volume / (self.Nx * self.Ny * self.Nz)
        
        print(f"Cell volume: {self.cell_volume:.6f} Bohr³")
        print(f"Volume element dV: {self.dV:.6e} Bohr³")


    def parse_atom_types_from_sparc_out_file(self, fname: str) -> List[int]:
        """
        Parse atomic species and counts from output file.
        Example match: 'Number of atoms of type 1 : 2'
        return: 
            atom_count_list : List[int] , number of atoms of each type
        """
        atom_type_list = []
        atom_count_list = []

        with open(fname, 'r') as f:
            lines = f.readlines()

        # pattern: Number of atoms of type 1 : 2
        pattern = re.compile(r'Number of atoms of type\s+(\d+)\s*:\s*(\d+)')

        for line in lines:
            match = pattern.search(line)
            if match:
                atom_type_list.append(int(match.group(1)))
                atom_count_list.append(int(match.group(2)))


        # check if the number of atoms is correct
        assert len(atom_type_list) == self.natom_types, NUMBER_OF_ATOM_TYPES_NOT_EQUAL_TO_OUTPUT_FILE_ERROR.format(len(atom_type_list))
        assert len(atom_count_list) == self.natom_types, NUMBER_OF_ATOM_COUNTS_NOT_EQUAL_TO_OUTPUT_FILE_ERROR.format(len(atom_count_list))
        for i in range(self.natom_types):
            assert atom_type_list[i] == i+1, ATOM_TYPE_NOT_EQUAL_TO_OUTPUT_FILE_ERROR.format(atom_type_list[i])

        return atom_count_list


    @classmethod
    def parse_atom_species_from_sparc_out_file(cls, output_fname: str) -> Tuple[int, List[str]]:
        """
        Parse atomic species names from output file.
        Example match: 'Atom type 1  (valence electrons)   :  Al 3'
        return: 
            natom_types       : int       , number of atom types
            atom_species_list : List[str] , names of atomic species
        """
        atom_species_list = []

        with open(output_fname, 'r') as f:
            lines = f.readlines()

        # pattern: Atom type 1  (valence electrons)   :  Al 3
        pattern = re.compile(r'Atom type\s+(\d+)\s*\([^)]*\)\s*:\s*(\w+)')

        for line in lines:
            match = pattern.search(line)
            if match:
                atom_type_num = int(match.group(1))
                atom_species = match.group(2)
                atom_species_list.append(atom_species)

        # Sort by atom type number to ensure correct order
        atom_species_list = [x for _, x in sorted(zip(range(1, len(atom_species_list) + 1), atom_species_list))]
        natom_types = len(atom_species_list)

        # check if the number of atom species is correct
        assert natom_types != 0, f"Number of atom species ({len(atom_species_list)}) is 0"

        return natom_types, atom_species_list


    def read_atomic_wave_function_from_upf_file(self, fname: str, atom_type_index: int, atom_species: str) -> 'PDOSCalculator.AtomicWaveFunction':
        """
        Parse the atomic wave function for one atom type from a UPF file (.upf)
        """
        assert isinstance(atom_type_index, int), f"atom_index must be int, got {type(atom_type_index)}"

        # reading the upf file for each type of element
        with open(fname, 'r') as f:
            lines = f.read().splitlines()

        # ---- Find all PP_CHI entries ----
        chi_indices = [i for i, line in enumerate(lines) if 'PP_CHI' in line]
        num_psdwfn = len(chi_indices) // 2

        AtomicWF = PDOSCalculator.AtomicWaveFunction(
            atom_type_index=atom_type_index, 
            upf_fname=fname,
            atom_species=atom_species,
            r_cut=self.r_cut_max[atom_type_index] if isinstance(self.r_cut_max, list) else self.r_cut_max,
            num_psdwfn=num_psdwfn,
            num_orbitals=0,
        )
        AtomicWF.psdwfn_list.clear()

        r_cut_orbitals = []

        for k in range(num_psdwfn):
            psdwfn = PDOSCalculator.PseudoWaveFunction()

            # extract l
            l_line = lines[chi_indices[2 * k] + 8]
            psdwfn.l = int(re.findall(r'\d+', l_line)[0])
            psdwfn.no_m = 2 * psdwfn.l + 1
            AtomicWF.num_orbitals += psdwfn.no_m

            # extract n
            n_line = lines[chi_indices[2 * k] + 7]
            psdwfn.n = int(re.findall(r'\d+', n_line)[0])

            # extract chi(r)
            chi_start = chi_indices[2 * k] + 9
            chi_end = chi_indices[2 * k + 1] - 1
            chi = []
            for line in lines[chi_start:chi_end+1]:
                numbers = re.findall(r'[+-]?[0-9]+\.[0-9]+E[+-]?[0-9]+', line)
                chi.extend(map(float, numbers))
            chi = np.array(chi)
            psdwfn.chi = chi

            # extract r-grid from PP_R section
            r_indices = [i for i, line in enumerate(lines) if 'PP_R' in line]
            if len(r_indices) < 2:
                raise ValueError("PP_R section not found or malformed.")
            r_lines = lines[r_indices[0]+1 : r_indices[1]]
            r = []
            for line in r_lines:
                numbers = re.findall(r'\d+\.\d+', line)
                r.extend(map(float, numbers))
            psdwfn.r = r

            # construct phi_r
            phi_r = np.zeros_like(chi)
            phi_r[1:] = chi[1:] / r[1:]  # phi_r(r) = chi(r) / r, for r > 0
            phi_r[0] = phi_r[1]          # phi_r(0) = phi_r(1), for r = 0
            psdwfn.phi_r = phi_r 

            # update r_cut variable in AtomicWF according to the tolerance and r_cut_max
            r_cut_max_index = len(psdwfn.r) - 1
            for i in range(len(psdwfn.r)):
                integral = np.trapezoid(np.array(psdwfn.chi[:i+1]) ** 2, psdwfn.r[:i+1])
                if abs(integral - 1.0) < self.atomic_wave_function_tol:
                    r_cut_max_index = i
                    break
            r_cut_orbitals.append(psdwfn.r[r_cut_max_index])
            # print(f"\t atomic wave function {atom_species} r_cut = {AtomicWF.r_cut}")

            # append psdwfn to AtomicWF
            AtomicWF.psdwfn_list.append(psdwfn)
            # print(f"\t atomic wave function {atom_species}, n={psdwfn.n}, l={psdwfn.l} r_cut = {psdwfn.r[r_cut_max_index]}")


            # # check if r_cut_max is acceptable       
            # # update r_cut variable in AtomicWF according to the tolerance and r_cut_max
            # for i in range(len(psdwfn.r)):
            #     if psdwfn.r[i] > AtomicWF.r_cut:
            #         r_cut_max_index = i
            #         break
            # integral_test = np.trapezoid(np.array(psdwfn.chi[:r_cut_max_index+1]) ** 2, psdwfn.r[:r_cut_max_index+1])
            # print(f"\t atomic wave function {atom_species}, n={psdwfn.n}, l={psdwfn.l}: integral_chi²(r) in [0, 10] = {integral_test}")


        # update the r_cut variable in AtomicWF according to the tolerance and r_cut_max
        AtomicWF.r_cut = min(AtomicWF.r_cut, max(r_cut_orbitals))

        # print the r_cut value
        # print(f"\t atomic wave function {atom_species} r_cut = {AtomicWF.r_cut}")

        return AtomicWF


    def read_atomic_wave_function_from_atomic_wave_function_generator(self, 
        atomic_wave_function_generator: AtomicWaveFunctionGeneratorType, 
        atom_type_index: int, 
        atomic_number: int, 
        atom_species: str) -> 'PDOSCalculator.AtomicWaveFunction':
        """
        Read the atomic wave function from the atomic wave function generator
        """
        assert callable(atomic_wave_function_generator), \
            ATOMIC_WAVE_FUNCTION_GENERATOR_NOT_CALLABLE_ERROR.format(type(atomic_wave_function_generator))
        r_array, orbitals, n_l_orbitals, info_dict = atomic_wave_function_generator(atomic_number = atomic_number)
        
        # Validate input shapes
        assert r_array.ndim == 1, f"r_array should be 1D, got shape {r_array.shape}"
        assert orbitals.ndim == 2, f"orbitals should be 2D, got shape {orbitals.shape}"
        assert n_l_orbitals.ndim == 2, f"n_l_orbitals should be 2D, got shape {n_l_orbitals.shape}"
        assert n_l_orbitals.shape[1] == 2, f"n_l_orbitals second dimension should be 2, got {n_l_orbitals.shape[1]}"
        assert orbitals.shape[1] == n_l_orbitals.shape[0], f"Number of orbitals ({orbitals.shape[1]}) should match n_l_orbitals first dimension ({n_l_orbitals.shape[0]})"
        
        AtomicWF = PDOSCalculator.AtomicWaveFunction(
            atom_type_index = atom_type_index,
            atom_species = atom_species,
            upf_fname = None,
            r_cut = self.r_cut_max[atom_type_index] if isinstance(self.r_cut_max, list) else self.r_cut_max,  # Use the full r_array as r_cut for all angular directions
            num_psdwfn = orbitals.shape[1],  # Number of orbitals
            num_orbitals = 0,  # n_l_orbitals array with shape (#orbitals, 2)
        )
        
        # Clear existing psdwfn_list and populate with orbital data
        AtomicWF.psdwfn_list.clear()
        
        # Create PseudoWaveFunction objects for each orbital
        r_cut_orbitals = []
        for k in range(orbitals.shape[1]):
            psdwfn = PDOSCalculator.PseudoWaveFunction()
            
            # Extract n and l from n_l_orbitals
            n_val = int(n_l_orbitals[k, 0])  # First column is n
            l_val = int(n_l_orbitals[k, 1])  # Second column is l
            
            psdwfn.n = n_val
            psdwfn.l = l_val
            psdwfn.no_m = 2 * l_val + 1  # Number of m values for this l
            AtomicWF.num_orbitals += psdwfn.no_m
            
            # compute the phi_r value
            chi, r = orbitals[:, k], r_array
            phi_r = np.zeros_like(chi)
            phi_r[1:] = chi[1:] / r[1:]  # phi_r(r) = chi(r) / r, for r > 0
            phi_r[0] = phi_r[1]          # phi_r(0) = phi_r(1), for r = 0

            # Extract the orbital data for this specific orbital
            psdwfn.chi   = chi     # Wave function values for this orbital
            psdwfn.r     = r_array # Radial grid points
            psdwfn.phi_r = phi_r   # Same as phi for atomic orbitals
            
            # update r_cut variable in AtomicWF according to the tolerance and r_cut_max
            r_cut_max_index = len(psdwfn.r) - 1
            for i in range(len(psdwfn.r)):
                integral = np.trapezoid(np.array(psdwfn.chi[:i+1]) ** 2, psdwfn.r[:i+1])
                if abs(integral - 1.0) < self.atomic_wave_function_tol:
                    r_cut_max_index = i
                    break
            r_cut_orbitals.append(psdwfn.r[r_cut_max_index])

            # Append to AtomicWF
            AtomicWF.psdwfn_list.append(psdwfn)

        # update the r_cut variable in AtomicWF according to the tolerance and r_cut_max
        AtomicWF.r_cut = min(AtomicWF.r_cut, max(r_cut_orbitals))

        return AtomicWF


    def atom_wise_compute_normalization_factor_in_unit_cell(self, atom: 'PDOSCalculator.Atom'):
        """
        Normalize the atomic wave functions
        """
        assert isinstance(atom, PDOSCalculator.Atom), ATOM_INPUT_NOT_ATOM_ERROR.format(type(atom))
        assert len(atom.phi_orbitals_list) == atom.num_orbitals, PHI_ORBITALS_NOT_EQUAL_TO_EXPECTED_NUM_ERROR.format(atom.num_orbitals, len(atom.phi_orbitals_list))

        for phi_orbital in atom.phi_orbitals_list:

            # initialize the grid-wise phi orbitals to zero in the unit cell

            _phi_orbital_unit_cell = self.construct_total_orbitals_in_unit_cell(
                phi_orbital = phi_orbital, 
                k_point_fractional = np.array([0, 0, 0]), 
                index_mask_dict = atom.index_mask_and_effective_grid_point_positions_dict)

            # normalize the grid-wise phi orbitals
            integral_of_phi_orbital_unit_cell = np.sum(_phi_orbital_unit_cell ** 2) * self.dV
            phi_orbital.normalization_factor = 1 / np.sqrt(integral_of_phi_orbital_unit_cell)

    
    # @print_time("Time taken to read the SPARC's static file (function={}) : {:.4f} seconds \n")
    def read_sparc_static_file_parameters(self, fname: str, atom_count_list: List[int]):
        """
        Read atomic positions (fractional and cartesian) from SPARC .static file.
        """
        assert isinstance(fname, str), STATIC_FNAME_NOT_STRING_ERROR.format(type(fname))
        assert isinstance(atom_count_list, list), ATOM_COUNT_LIST_NOT_LIST_ERROR.format(type(atom_count_list))

        print("Reading the SPARC's static file")
        print("\t fname = ", fname)
        print("\t number of atoms = ", sum(atom_count_list))

        with open(fname, 'r') as f:
            lines = f.read().splitlines()

        atom_position_indices = [i for i, line in enumerate(lines) if "Fractional coordinates" in line]
        assert len(atom_position_indices) == len(atom_count_list), \
            f"Expected {len(atom_count_list)} atomic blocks, but found {len(atom_position_indices)}"

        atoms = []
        for atom_type_index, natom in enumerate(atom_count_list):
            start_index = atom_position_indices[atom_type_index]
            for j in range(1, natom + 1):
                line = lines[start_index + j]
                frac_coords = [float(x) for x in re.findall(r'\d+\.\d+', line)]
                assert len(frac_coords) == 3, f"Expected 3 fractional coordinates, got {frac_coords}"
                if self.is_orthogonal_lattice:
                    atom = PDOSCalculator.Atom(
                        atom_type_index = atom_type_index,
                        atom_posn_frac = np.array(frac_coords, dtype = float),
                        atom_posn_cart = np.array([self.Lx * frac_coords[0], self.Ly * frac_coords[1], self.Lz * frac_coords[2]], dtype = float)
                    )
                else:
                    atom = PDOSCalculator.Atom(
                        atom_type_index = atom_type_index,
                        atom_posn_frac = np.array(frac_coords, dtype = float),
                        atom_posn_cart = np.array(self.latvec @ frac_coords, dtype = float)
                    )
                atoms.append(atom)

        # store the parsed atoms in the class
        self.atoms = atoms


    def read_sparc_ion_file_parameters(self, fname: str, atom_count_list: List[int]):
        """
        Read atomic positions (fractional and cartesian) from SPARC .ion file.
        """
        assert isinstance(fname, str), ION_FNAME_NOT_STRING_ERROR.format(type(fname))
        assert isinstance(atom_count_list, list), ATOM_COUNT_LIST_NOT_LIST_ERROR.format(type(atom_count_list))

        raise NotImplementedError("read_sparc_ion_file_parameters is not implemented")
        


    def read_sparc_geopt_file_parameters(self, fname: str, atom_count_list: List[int],):
        assert isinstance(fname, str), GEOPT_FNAME_NOT_STRING_ERROR.format(type(fname))
        assert isinstance(atom_count_list, list), ATOM_COUNT_LIST_NOT_LIST_ERROR.format(type(atom_count_list))
        assert isinstance(self.relax_flag, int), RELAX_FLAG_NOT_INTEGER_ERROR.format(type(self.relax_flag))
        assert self.relax_flag in [1, 2, 3], RELAX_FLAG_NOT_VALID_ERROR.format(self.relax_flag)
        
        print("Reading the SPARC's geopt file")
        print("\t fname = ", fname)
        print("\t number of atoms = ", sum(atom_count_list))
        
        with open(fname, 'r') as f:
            geopt_lines = f.read().splitlines()

        # read in the lattice parameters from the .geopt file
        self.is_orthogonal_lattice = False
        self.setup_non_orthogonal_lattice(lines = geopt_lines)
        
        # read in the atomic positions from the .geopt or .ion file
        if self.relax_flag == 1 or self.relax_flag == 3:
            _last_R_index = int(_find_last_line_index("R(Bohr)", geopt_lines))
            assert _last_R_index != -1, "R(Bohr) not found in the .geopt file"

            use_frac_coords = False
            atom_cart_position_indices = [_last_R_index + i for i in range(len(atom_count_list))]
            atom_position_lines = geopt_lines # lines containing the cartesian coordinates of the atoms

        elif self.relax_flag == 2:
            # This is when the fractional coordinates are fixed during the relaxation, but the cell size is not fixed
            # secretly ask for the .ion file, and read the parameters from the .ion file
            assert fname.endswith(".geopt"), GEOPT_FNAME_NOT_ENDING_WITH_GEOPT_ERROR.format(fname)
            ion_fname = fname.replace(".geopt", ".ion")
            assert os.path.exists(ion_fname), ION_FNAME_NOT_EXIST_ERROR.format(ion_fname)

            # read in the atomic positions from the .ion file
            with open(ion_fname, 'r') as f:
                atom_position_lines = f.read().splitlines()

            # check if the .ion file uses fractional coordinates or cartesian coordinates
            atom_frac_position_indices = [i for i, line in enumerate(atom_position_lines) if "COORD_FRAC" in line]
            if len(atom_frac_position_indices) == 0:
                atom_cart_position_indices = [i for i, line in enumerate(atom_position_lines) if "COORD" in line]
                assert len(atom_cart_position_indices) == len(atom_count_list), \
                    f"Expected {len(atom_count_list)} atomic blocks, but found {len(atom_cart_position_indices)}"
                use_frac_coords = False
            else:
                assert len(atom_frac_position_indices) == len(atom_count_list), \
                    f"Expected {len(atom_count_list)} atomic blocks, but found {len(atom_frac_position_indices)}"
                use_frac_coords = True
        else:
            raise ValueError(f"relax_flag must be 1, 2, or 3, got {self.relax_flag}") # This error should never be raised


        atoms = []

        # loop over each atom type
        for atom_type_index, natom in enumerate(atom_count_list):
            start_index = atom_frac_position_indices[atom_type_index] if use_frac_coords else atom_cart_position_indices[atom_type_index]

            # loop over each atom in the current atom type
            for j in range(1, natom + 1):
                line = atom_position_lines[start_index + j]

                if use_frac_coords:
                    _frac_coords = [float(x) for x in re.findall(SCIENTIFIC_NOTATION_PATTERN, line)]
                    assert len(_frac_coords) == 3, f"Expected 3 fractional coordinates, got {_frac_coords}"
                    atom = PDOSCalculator.Atom(
                        atom_type_index = atom_type_index,
                        atom_posn_frac = np.array(_frac_coords, dtype = float),
                        atom_posn_cart = np.array(self.latvec @ _frac_coords, dtype = float)
                    )
                else:
                    _cart_coords = [float(x) for x in re.findall(SCIENTIFIC_NOTATION_PATTERN, line)]
                    assert len(_cart_coords) == 3, f"Expected 3 cartesian coordinates, got {_cart_coords}"
                    atom = PDOSCalculator.Atom(
                        atom_type_index = atom_type_index,
                        atom_posn_frac = np.array(self.latvec_inv @ _cart_coords, dtype = float),
                        atom_posn_cart = np.array(_cart_coords, dtype = float)
                    )

                atoms.append(atom)

        # store the parsed atoms in the class
        self.atoms = atoms

    

    # @print_time("Time taken to read the SPARC's eigen file (function={}) : {:.4f} seconds \n")
    def read_sparc_eigen_file_parameters(self, fname : str):
        """
        Read in the SPARC's eigen file (.eigen)
            eign           : (totkpt, band) energies sorted ascending within each k
            I_indices      : (band, totkpt) argsort indices for each k
            kpts_store     : (totkpt, 3) k-vectors
            kpt_wts_store  : (totkpt,)  weights * (kpt1*kpt2*kpt3)

        """
        assert isinstance(fname, str), EIGEN_FNAME_NOT_STRING_ERROR.format(type(fname))


        print("Reading the SPARC's eigen file")
        print("\t fname = ", fname)
        print("\t number of k-points = ", self.kpt_num)
        print("\t number of bands = ", self.band_num)


        kpts_store = []
        kpt_wts_store = []

        eign_array = np.zeros((self.band_num, self.kpt_num))        # (band, kpt), original order
        eign_sorted_array = np.zeros((self.band_num, self.kpt_num)) # (band, kpt), sorted order
        occ_array = np.zeros((self.band_num, self.kpt_num))         # (band, kpt), original order
        occ_sorted_array = np.zeros((self.band_num, self.kpt_num))  # (band, kpt), sorted order
        I_indices_array = np.zeros((self.band_num, self.kpt_num))   # (band, kpt), indexing


        kvec_re = re.compile(r"kred\s*#(\d+)\s*=\s*\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)")
        wgt_re  = re.compile(r"=\s*(-?\d+\.\d+)")
        eign_re = re.compile(r"\s*(\d+)\s+([-+]?\d+\.\d+E[-+]?\d+)\s+([-+]?\d+\.\d+)")


        with open(fname, 'r') as f:
            lines = f.read().splitlines()
        
        # Find the line indices for the k-point weights
        kpt_vector_indices = [i for i, line in enumerate(lines) if "kred" in line]
        assert len(kpt_vector_indices) == self.kpt_num, NUMBER_OF_KPOINTS_NOT_EQUAL_TO_EIGEN_FILE_ERROR.format(self.kpt_num, len(kpt_vector_indices))


        for kpt_index in range(self.kpt_num):

            # extract k-point vector
            kvec_line = lines[kpt_vector_indices[kpt_index]]
            kvec_match = re.search(kvec_re, kvec_line)
            if kvec_match:
                kvec_number = int(kvec_match.group(1))
                kvec_x = float(kvec_match.group(2))
                kvec_y = float(kvec_match.group(3))
                kvec_z = float(kvec_match.group(4))
                assert kvec_number == kpt_index + 1, KPOINT_NUMBER_NOT_EQUAL_TO_EIGEN_FILE_ERROR.format(kpt_index + 1, kvec_number)
            kpts_store.append([kvec_x, kvec_y, kvec_z])

            # extract k-point weights
            wgt_line = lines[kpt_vector_indices[kpt_index] + 1]
            wgt_match = re.search(wgt_re, wgt_line)
            if wgt_match:
                weight = float(wgt_match.group(1))
                kpt_wts_store.append(weight)
            else:
                raise ValueError(KPOINT_WEIGHTS_NOT_FOUND_ERROR.format(kpt_index + 1))

            # extract eigenvalues and occ
            eig_start = kpt_vector_indices[kpt_index] + 3
            eig_end = kpt_vector_indices[kpt_index] + 3 + self.band_num
            eig_temp = [] 
            occ_temp = []
            for line in lines[eig_start:eig_end]:
                eign_match = re.search(eign_re, line)
                if eign_match:
                    eig_temp.append(float(eign_match.group(2))) # Ha
                    occ_temp.append(float(eign_match.group(3))) # may be 2.0 or fractional
                else:
                    raise ValueError(EIGEN_FILE_NO_EIGENVALUES_AND_OCCUPATIONS_FOUND_ERROR.format(kpt_index + 1, line))

            # sorting & indexing
            indexing = np.argsort(eig_temp, axis=0)

            eign_array[:, kpt_index] = np.asarray(eig_temp, dtype=float)
            eign_sorted_array[:, kpt_index] = eign_array[indexing, kpt_index]
            occ_array[:, kpt_index] = np.asarray(occ_temp, dtype=float)
            occ_sorted_array[:, kpt_index] = occ_array[indexing, kpt_index]
            I_indices_array[:,  kpt_index] = indexing
        

        # store the parsed parameters in the class
        self.I_indices = I_indices_array # indexing
        self.eign = eign_sorted_array    # already sorted
        self.occ  = occ_sorted_array     # already sorted

        self.kpts_store = np.asarray(kpts_store, dtype=float)
        self.kpt_wts_store = np.asarray(kpt_wts_store, dtype=float)



    # @print_time("Time taken to read the SPARC's orbital file (function={}) : {:.4f} seconds \n")
    def read_sparc_orbital_file_parameters(self, fname : str):
        """
        Read in the SPARC's orbital file (.psi)
        """
        assert isinstance(fname, str), "fname must be a string, get {} instead".format(type(fname))
        print("Reading the SPARC's orbital file")

        psi_total, header, per_band_meta = read_psi(fname, verbose=False)

        self.check_valid_psi_header(header = header)

        print("\t fname = ", fname)
        print("\t psi_total.shape = ", psi_total.shape)

        self.psi_total = psi_total
        self.header = header
        self.per_band_meta = per_band_meta


        # psi_kpt = psi_total[:, :, 0]
        # for band_index in range(4):
        #     psi_per_band = psi_kpt[:, band_index]
        #     norm_with_header_dV = np.sum(psi_per_band.real**2 + psi_per_band.imag**2).real * self.header['dV']
        #     norm_with_correct_dV = np.sum(psi_per_band.real**2 + psi_per_band.imag**2).real * self.dV
        #     print("\t band[{}] squared sum (header dV): {:.6f}, (correct dV): {:.6f}".format(
        #         band_index, norm_with_header_dV, norm_with_correct_dV))
        # print("self.dV = ", self.dV)
        # print("header['dV'] = ", self.header['dV'])
        # print("Ratio header['dV']/self.dV = {:.6f}".format(self.header['dV'] / self.dV))
        # print("header = \n", self.header)
        # print("self.per_band_meta = \n", self.per_band_meta)



    @classmethod
    def check_fname_existence(cls, fname : str):
        """
        Check if the file exists
        """
        assert isinstance(fname, str), FNAME_NOT_STRING_ERROR.format(type(fname))
        assert os.path.exists(fname), FNAME_NOT_EXIST_ERROR.format(fname)


    @classmethod
    def check_valid_atomic_wave_function_generator(cls, atomic_wave_function_generator : AtomicWaveFunctionGeneratorType):
        assert callable(atomic_wave_function_generator), ATOMIC_WAVE_FUNCTION_GENERATOR_NOT_CALLABLE_ERROR.format(type(atomic_wave_function_generator))
        r_array, orbitals, n_l_orbitals, info_dict = None, None, None, None

        try:
            r_array, orbitals, n_l_orbitals, info_dict = atomic_wave_function_generator(Atomic_number = 12)
        except Exception as e:
            raise ValueError(ATOMIC_WAVE_FUNCTION_GENERATOR_NOT_VALID_ERROR.format(e))

        assert isinstance(r_array, np.ndarray), "r_array must be a numpy array, get {} instead".format(type(r_array))
        assert isinstance(orbitals, np.ndarray), "orbitals must be a numpy array, get {} instead".format(type(orbitals))
        assert isinstance(n_l_orbitals, np.ndarray), "n_l_orbitals must be a numpy array, get {} instead".format(type(n_l_orbitals))
        assert isinstance(info_dict, dict), "info_dict must be a dictionary, get {} instead".format(type(info_dict))
        assert len(r_array) == len(orbitals), "The number of r_array points must be equal to the number of orbitals"
        
        print("r_array.shape = ", r_array.shape)
        print("orbitals.shape = ", orbitals.shape)
        print("n_l_orbitals.shape = ", n_l_orbitals.shape)
        print("n_l_orbitals = ", n_l_orbitals)
        # print("info_dict = ", info_dict)
        # raise ValueError("test")
        pass




    @classmethod
    def check_upf_fname_list_or_get_default_atomic_wave_function_generators(cls, 
        upf_fname_list : Optional[List[str]], 
        natom_types    : int):

        if upf_fname_list is not None:
            assert isinstance(upf_fname_list, list), UPF_FNAME_LIST_NOT_LIST_ERROR.format(type(upf_fname_list))
            assert len(upf_fname_list) == natom_types, UPF_FNAME_NUM_NOT_EQUAL_TO_NATOM_TYPES_ERROR.format(len(upf_fname_list))

            for upf_fname in upf_fname_list:
                if upf_fname == "Default":
                    continue
                else:
                    assert isinstance(upf_fname, str), UPF_FNAME_NOT_STRING_ERROR.format(type(upf_fname))
                    assert upf_fname.endswith(".upf"), UPF_FNAME_NOT_ENDING_WITH_UPF_ERROR.format(upf_fname)
                    cls.check_fname_existence(upf_fname)
                    # TODO : check the order of the upf_fname_list, it should be the same as the order of the atoms in the unit cell
        else:
            raise ValueError(ATOMIC_WAVE_FUNCTION_GENERATOR_NO_UPF_FILES_PROVIDED_ERROR)


    def check_sparc_out_file_parameters(self):
        """
        Check if the output parameters are correctly read in
        """
        # type check
        assert isinstance(self.kpt1, int), "kpt1 must be an integer, get {} instead".format(type(self.kpt1))
        assert isinstance(self.kpt2, int), "kpt2 must be an integer, get {} instead".format(type(self.kpt2))
        assert isinstance(self.kpt3, int), "kpt3 must be an integer, get {} instead".format(type(self.kpt3))
        assert isinstance(self.Nx, int), "Nx must be an integer, get {} instead".format(type(self.Nx))
        assert isinstance(self.Ny, int), "Ny must be an integer, get {} instead".format(type(self.Ny))
        assert isinstance(self.Nz, int), "Nz must be an integer, get {} instead".format(type(self.Nz))
        assert isinstance(self.bcx, str), "bcx must be a string, get {} instead".format(type(self.bcx))
        assert isinstance(self.bcy, str), "bcy must be a string, get {} instead".format(type(self.bcy))
        assert isinstance(self.bcz, str), "bcz must be a string, get {} instead".format(type(self.bcz))
        assert isinstance(self.band_num, int), "band must be an integer, get {} instead".format(type(self.band_num))
        assert isinstance(self.kpt_num, int), "kpt must be an integer, get {} instead".format(type(self.kpt_num))
        assert isinstance(self.fermi, float), "fermi must be a float, get {} instead".format(type(self.fermi))
        assert isinstance(self.psp_fname_list, list), "psp_fname_list must be a list, get {} instead".format(type(self.psp_fname_list))

        # value check
        assert self.kpt1 > 0, "kpt1 must be a positive integer, get {} instead".format(self.kpt1)
        assert self.kpt2 > 0, "kpt2 must be a positive integer, get {} instead".format(self.kpt2)
        assert self.kpt3 > 0, "kpt3 must be a positive integer, get {} instead".format(self.kpt3)
        assert self.Nx > 0, "Nx must be a positive integer, get {} instead".format(self.Nx)
        assert self.Ny > 0, "Ny must be a positive integer, get {} instead".format(self.Ny)
        assert self.Nz > 0, "Nz must be a positive integer, get {} instead".format(self.Nz)
        assert self.bcx in ["P", "D"], "bcx must be either P or D, get {} instead".format(self.bcx)
        assert self.bcy in ["P", "D"], "bcy must be either P or D, get {} instead".format(self.bcy)
        assert self.bcz in ["P", "D"], "bcz must be either P or D, get {} instead".format(self.bcz)
        assert self.band_num > 0, "band must be a positive integer, get {} instead".format(self.band_num)
        for atom_index, psp_fname in enumerate(self.psp_fname_list):
            assert isinstance(psp_fname, str), "psp_fname must be a string, get {} instead".format(type(psp_fname))
            if self.upf_fname_list is not None:
                upf_fname = self.upf_fname_list[atom_index]
                if upf_fname == "Default":
                    self.check_fname_existence(psp_fname)
                else:
                    self.check_fname_existence(upf_fname)
            else:
                self.check_fname_existence(psp_fname)

        if not self.is_relaxation:
            self.check_lattice_parameters()
        else:
            assert isinstance(self.relax_flag, int), RELAX_FLAG_NOT_INTEGER_ERROR.format(type(self.relax_flag))
            assert self.relax_flag in [1, 2, 3], RELAX_FLAG_NOT_VALID_ERROR.format(self.relax_flag)



    def check_lattice_parameters(self):
        assert isinstance(self.Lx, float), "Lx must be a float, get {} instead".format(type(self.Lx))
        assert isinstance(self.Ly, float), "Ly must be a float, get {} instead".format(type(self.Ly))
        assert isinstance(self.Lz, float), "Lz must be a float, get {} instead".format(type(self.Lz))

        assert self.Lx > 0, "Lx must be a positive float, get {} instead".format(self.Lx)
        assert self.Ly > 0, "Ly must be a positive float, get {} instead".format(self.Ly)
        assert self.Lz > 0, "Lz must be a positive float, get {} instead".format(self.Lz)


    def check_valid_psi_header(self, header : dict):
        try:
            assert isinstance(header, dict), "header must be a dictionary, get {} instead".format(type(header))
            # grid number check
            assert header["Nx"] == self.Nx, "Nx = {} != header['Nx'] = {}".format(self.Nx, header["Nx"])
            assert header["Ny"] == self.Ny, "Ny = {} != header['Ny'] = {}".format(self.Ny, header["Ny"])
            assert header["Nz"] == self.Nz, "Nz = {} != header['Nz'] = {}".format(self.Nz, header["Nz"])
            assert header["Nd"] == self.Nx * self.Ny * self.Nz, "Nd = {} != header['Nd'] = {}".format(self.Nx * self.Ny * self.Nz, header["Nd"])

            # grid spacing check
            assert np.isclose(header["dx"], self.dx, atol=3e-2), "dx = {} != header['dx'] = {}".format(self.dx, header["dx"])
            assert np.isclose(header["dy"], self.dy, atol=3e-2), "dy = {} != header['dy'] = {}".format(self.dy, header["dy"])
            assert np.isclose(header["dz"], self.dz, atol=3e-2), "dz = {} != header['dz'] = {}".format(self.dz, header["dz"])
            assert np.isclose(header["dV"], self.dV, atol=3e-3), "dV = {} != header['dV'] = {}".format(self.dV, header["dV"])

            # nband and nkpt check
            assert header["nband"] == self.band_num, "nband = {} != header['nband'] = {}".format(self.band_num, header["nband"])
            assert header["nkpt"] == self.kpt_num, "nkpt = {} != header['nkpt'] = {}".format(self.kpt_num, header["nkpt"])
        except Exception as e:
            print("\t Warning: Error in checking valid psi header: {}".format(e))


    def print_initialization_parameters(self):
        """
        Print the input parameters
        """
        print("File / Directory parameters:")
        print("\t natom_types: {}".format(self.natom_types))
        print("\t upf_fname_list: {}".format(self.upf_fname_list))
        print("\t output_fname: {}".format(self.output_fname))
        print("\t eigen_fname: {}".format(self.eigen_fname))
        print("\t static_fname: {}".format(self.static_fname))
        print("\t psi_fname: {}".format(self.psi_fname))
        print("\t out_dirname: {}".format(self.out_dirname))

        print("PDOS calculation parameters:")
        print("\t mu_PDOS: {:.4f} eV".format(self.mu_PDOS))
        print("\t N_PDOS: {}".format(self.N_PDOS))
        print("\t min_E_plot: {:.2f} eV".format(self.min_E_plot))
        print("\t max_E_plot: {:.2f} eV".format(self.max_E_plot))
        print("\t r_cut_max: {:.2f} Bohr".format(self.r_cut_max))
        print("\n")


    def print_sparc_out_file_parameters(self):
        """
        Print the parameters read in from the SPARC's output file (.out)
        """
        # print the results
        print("SPARC's output file parameters:")
        print("\t kpt1: {}".format(self.kpt1))
        print("\t kpt2: {}".format(self.kpt2))
        print("\t kpt3: {}".format(self.kpt3))
        print("\t Nx: {}".format(self.Nx))
        print("\t Ny: {}".format(self.Ny))
        print("\t Nz: {}".format(self.Nz))
        print("\t Lx: {:.4f} Bohr".format(self.Lx))
        print("\t Ly: {:.4f} Bohr".format(self.Ly))
        print("\t Lz: {:.4f} Bohr".format(self.Lz))
        print("\t bcx: {}".format(self.bcx))
        print("\t bcy: {}".format(self.bcy))
        print("\t bcz: {}".format(self.bcz))
        print("\t band_num: {}".format(self.band_num))
        print("\t kpt_num: {}".format(self.kpt_num))
        print("\t fermi: {:.6f} eV".format(self.fermi))
        print("\t dx: {:.4f} Bohr".format(self.dx))
        print("\t dy: {:.4f} Bohr".format(self.dy))
        print("\t dz: {:.4f} Bohr".format(self.dz))

        print("\t tot_atoms: {}".format(self.tot_atoms_num))
        print("\t atom_count_list: {}".format(self.atom_count_list))
        print("\t atom_species_list: {}".format(self.atom_species_list))

        if not self.is_orthogonal_lattice and not self.is_relaxation:
            self.print_non_orthogonal_lattice_parameters()

        print("\n")
    

    def print_non_orthogonal_lattice_parameters(self):
        print("Non-orthogonal lattice parameters:")
        print("\t latvec:")
        print("\t\t a1 = {}".format(self.latvec[:, 0]))
        print("\t\t a2 = {}".format(self.latvec[:, 1]))
        print("\t\t a3 = {}".format(self.latvec[:, 2]))

        print("\t latvec_matrix: \n{}".format(self.latvec))
        print("\t latvec_inv_matrix: \n{}".format(self.latvec_inv))
        print("\t cell_volume: {:.6f} Bohr³".format(self.cell_volume))
        print("\t dV: {:.6f} Bohr³".format(self.dV))



    def print_atomic_wave_function(self, atomic_wave_function : 'PDOSCalculator.AtomicWaveFunction'):
        """
        Print the atomic wave function from the UPF file (.upf)
        """
        assert isinstance(atomic_wave_function, PDOSCalculator.AtomicWaveFunction), "atomic_wave_function must be a PDOSCalculator.AtomicWaveFunction, get {} instead".format(type(atomic_wave_function))

        print("atomic_wavefunction from file [{}]".format(atomic_wave_function.upf_fname))
        print("\t atom_type_index: {}".format(atomic_wave_function.atom_type_index))
        print("\t r_cut: {}".format(atomic_wave_function.r_cut))
        print("\t # (pseudo-wave functions): {}".format(atomic_wave_function.num_psdwfn))
        print("\t # (orbitals): {}".format(atomic_wave_function.num_orbitals))
        for psdwfn in atomic_wave_function.psdwfn_list:
            print("\t pseudo-wave function [n = {}, l = {}, #(m) = {}]".format(psdwfn.n, psdwfn.l, psdwfn.no_m))
            print("\t\t length of chi(r): {}".format(len(psdwfn.chi)))
            print("\t\t length of r-grid: {}".format(len(psdwfn.r)))
            print("\t\t length of phi(r): {}".format(len(psdwfn.phi_r)))
        print("\n")


    def plot_atomic_wave_function(self, atomic_wave_function : 'PDOSCalculator.AtomicWaveFunction', save_fname : Optional[str] = None):
        """
        Plot the atomic wave function
        """
        assert isinstance(atomic_wave_function, PDOSCalculator.AtomicWaveFunction), "atomic_wave_function must be a PDOSCalculator.AtomicWaveFunction, get {} instead".format(type(atomic_wave_function))

        atom_type = atomic_wave_function.atom_species
        if save_fname is None:
            save_fname = os.path.join(self.out_dirname, atom_type + '.png')


        import matplotlib.pyplot as plt

        # plot the atomic wave function
        plt.figure(figsize=(10, 5))
        for psdwfn in atomic_wave_function.psdwfn_list:
            plt.plot(psdwfn.r, psdwfn.phi_r, label = 'n = {}, l = {}'.format(psdwfn.n, psdwfn.l))

        # plot the cutoff radius
        plt.axvline(x = atomic_wave_function.r_cut, color = 'red', linestyle = '--', label = 'cutoff radius')

        plt.xlabel('r (Bohr)')
        plt.ylabel('phi(r)')
        plt.title('Atomic Wave Function for atom {}'.format(atom_type))

        plt.legend(loc="upper right")
        plt.grid(linestyle='--')
        plt.savefig(save_fname)


    def print_sparc_static_file_parameters(self):
        """
        Print the parameters from the SPARC's static file (.static)
        """
        print("SPARC's static file parameters:")
        print("\t fname = ", self.static_fname)
        print("\t natoms = ", len(self.atoms))
        print("\t natom_types = ", self.natom_types)
        print("\t atom_species_list = ", self.atom_species_list)
        if not self.is_orthogonal_lattice:
            print("\t atoms' positions in cartesian coordinates:")
            for atom_index, atom in enumerate(self.atoms):
                print("\t\t atom{}.atom_posn_cart = {}".format(atom_index, atom.atom_posn_cart))
        print("\n")


    def print_sparc_eigen_file_parameters(self):
        """
        Print the parameters from the SPARC's eigen file (.eigen)
        """
        print("SPARC's eigen file parameters")
        print("\t fname = ", self.eigen_fname)
        print("\t number of bands = ", self.band_num)
        print("\t number of k-points = ", self.kpt_num)
        print("\t sum of weighted occupations = ", np.sum(self.occ* self.kpt_wts_store))
        
        print("\t eign.shape = ", self.eign.shape)
        print("\t occ.shape = ", self.occ.shape)
        print("\t I_indices.shape = ", self.I_indices.shape)
        
        for i in range(self.kpt_num):
            print("\t k-point {}".format(i))
            print("\t\t sum(occ) = ", sum(self.occ[:, i]))
            print("\t\t k-vector = ", self.kpts_store[i, :])
            print("\t\t k-point weight = ", self.kpt_wts_store[i])
        print("\n")


    def print_sparc_orbital_file_parameters(self, print_band_meta_in_details : bool = False):
        """
        Print the parameters from the SPARC's orbital file (.psi)
        """
        print("SPARC's orbital file parameters:")
        print("\t fname = ", self.orbital_fname)
        print("\t header['nband'] = ", self.header['nband'])
        print("\t header['nkpt'] = ", self.header['nkpt'])
        print("\t per_band_meta's length = ", len(self.per_band_meta))
        print("\t psi_total.shape = ", self.psi_total.shape)
        
        msg = "\t\t band_meta : 'spin_index' = {}, kpt_index = {}, band_index = {}, kpt_vec = ({:.2f}, {:.2f}, {:.2f})"
        if print_band_meta_in_details:
            for band_meta in self.per_band_meta:
                band_meta_msg = msg.format(
                    band_meta['spin_index'], band_meta['kpt_index'], band_meta['band_indx'],
                    self.kpts_store[band_meta['kpt_index'], 0], self.kpts_store[band_meta['kpt_index'], 1], self.kpts_store[band_meta['kpt_index'], 2]
                    )
                print(band_meta_msg)
        print("\n")


    def print_atoms_updated_parameters(self):
        """
        Print the index mask, effective grid point positions and phi_orbitals_list information for all atoms
        """
        print("--------------------------------------------------------------------------------------------")
        print("  Printing the updated parameters of all atoms  ")
        print("--------------------------------------------------------------------------------------------")
        print("tot_orbital_num = ", self.tot_orbital_num)

        for atom_index, atom in enumerate(self.atoms):
            print("atom_index = {}".format(atom_index))
            print("\t atom_type_index = {}".format(atom.atom_type_index))
            print("\t atom_posn_cart = {}".format(atom.atom_posn_cart))
            print("\t rcut = {} Bohr".format(atom.r_cut))
            print("\t len(atom.index_mask_and_effective_grid_point_positions_dict) = {}".format(len(atom.index_mask_and_effective_grid_point_positions_dict)))
            print("\t len(atom.phi_orbitals_list) = {}".format(len(atom.phi_orbitals_list)))
            for phi_orbital in atom.phi_orbitals_list:
                print("\t\t phi_orbital.n = {}, phi_orbital.l = {}, phi_orbital.m = {}".format(phi_orbital.n, phi_orbital.l, phi_orbital.m))
                print("\t\t len(phi_orbital.phi_value_dict) = {}".format(len(phi_orbital.phi_value_dict)))
                for shift_index_vector, phi_value_array in phi_orbital.phi_value_dict.items():
                    print("\t\t\t shift_index_vector = {}, len(phi_value_array) = {}".format(shift_index_vector, len(phi_value_array)))

        print("--------------------------------------------------------------------------------------------")




def dict2namespace(config):
    """Convert dictionary to namespace object recursively."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config(config_path=None, doc_path=None):
    """Parse command line arguments and configuration file."""
    parser = argparse.ArgumentParser(description="PDOS Calculator for SPARC output files")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--path", type=str, help="Path to SPARC output directory (simple mode)")

    args = parser.parse_args()

    if config_path is not None:
        args.config = config_path


    
    
    # args.log documents the actual path for saving the running related data
    # args.log = os.path.join(args.run, args.doc)
    
    # Parse config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        new_config = dict2namespace(config)
    else:
        # Simple mode with just path
        if not args.path:
            raise ValueError("Either --config or --path must be provided")
        
        # Create default config for simple mode
        config = {
            'sparc_path': args.path,
            'output_dir': None,  # Will be set to parent directory of sparc_path
            'gaussian_width': 0.1,
            'max_E_plot': None,
            'min_E_plot': None,
            'N_PDOS': 1000,
            'print_single_atom': False,
            'atom_type': None,
            'atom_index_for_specified_atom_type': 0
        }
        new_config = dict2namespace(config)
    
    # Create log directory if it doesn't exist
    # os.makedirs(args.log, exist_ok=True)
    
    # Setup logging
    # level = getattr(logging, args.verbose.upper(), None)
    # if not isinstance(level, int):
    #     raise ValueError('level {} not supported'.format(args.verbose))

    # Remove all the existing handler
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    handler1 = logging.StreamHandler()  # Output to the terminal
    # handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))  # output to the stdout.txt file
    formatter = logging.Formatter('%(levelname)s-%(filename)s-%(asctime)s-%(message)s')
    handler1.setFormatter(formatter)
    # handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    # logger.addHandler(handler2)
    # logger.setLevel(level)

    return args, new_config


def setup_file_paths(config):
    """Setup file paths based on configuration."""
    sparc_path = config.sparc_path
    
    # Extract directory and base name
    # Normalize path separators for the current OS
    sparc_path = os.path.normpath(sparc_path)
    if '/' in sparc_path or '\\' in sparc_path:
        sparc_dir = os.path.dirname(sparc_path)
        base_name = os.path.basename(sparc_path)
    else:
        sparc_dir = '.'
        base_name = sparc_path
    
    # Set output directory
    if config.output_dir is None:
        # Use a subdirectory within the SPARC directory for PDOS output
        output_dir = os.path.join(sparc_dir, "PDOS_output")
    else:
        output_dir = config.output_dir
    
    # Construct file paths
    output_fname = os.path.join(sparc_dir, f"{base_name}.out")
    static_fname = os.path.join(sparc_dir, f"{base_name}.static")
    geopt_fname = os.path.join(sparc_dir, f"{base_name}.geopt")
    eigen_fname = os.path.join(sparc_dir, f"{base_name}.eigen")
    psi_fname = os.path.join(sparc_dir, f"{base_name}.psi")

    
    # Check if required files exist
    required_files = [output_fname, eigen_fname, psi_fname]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if os.path.exists(static_fname):
        is_relaxation = False
    elif os.path.exists(geopt_fname):
        is_relaxation = True
    else:
        missing_files.append(static_fname)
        missing_files.append(geopt_fname)
        raise FileNotFoundError(f"Both static and geopt files not found: {static_fname} or {geopt_fname}")

    if missing_files:
        raise FileNotFoundError(f"Required files not found: {missing_files}")
    
    # Check if UPF file exists, if not, we'll use default generator
    upf_fname_list = get_upf_fname_list(output_fname = output_fname)
    
    return {
        'is_relaxation': is_relaxation,
        'output_fname': output_fname,
        'static_fname': static_fname if not is_relaxation else None,
        'geopt_fname': geopt_fname if is_relaxation else None,
        'eigen_fname': eigen_fname,
        'psi_fname': psi_fname,
        'upf_fname_list': upf_fname_list,
        'output_dir': output_dir,
        'natom_types': len(upf_fname_list) if upf_fname_list else 1
    }


def get_upf_fname_list(output_fname : str):
    assert isinstance(output_fname, str), OUTPUT_FNAME_NOT_STRING_ERROR.format(type(output_fname))
    assert os.path.exists(output_fname), OUTPUT_FNAME_NOT_EXIST_ERROR.format(output_fname)
    output_dir = Path(output_fname).parent
    natom_types, atom_species_list = PDOSCalculator.parse_atom_species_from_sparc_out_file(output_fname = output_fname)
    upf_fname_list_temp = [os.path.join(output_dir, f"{atom_species_list[i]}.upf") for i in range(natom_types)]
    upf_fname_list = []
    for upf_fname_temp in upf_fname_list_temp:
        if os.path.exists(upf_fname_temp):
            upf_fname_list.append(upf_fname_temp)
        else:
            upf_fname_list.append("Default")
    
    return upf_fname_list





def main():
    """Main function to run PDOS calculation."""
    try:
        # Parse arguments and configuration
        args, config = parse_args_and_config()
        
        # Present all the information in the terminal/stdout.txt file
        # logging.info("Writing log file to {}".format(args.log))
        # logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Config=")
        logging.info(">" * 80)
        logging.info(config)
        logging.info("<" * 80)
        
        # Setup file paths
        file_paths_config = setup_file_paths(config)

        # Run calculation
        if not hasattr(config, 'full_pdos_calculation'):
            config.full_pdos_calculation = True
        if not hasattr(config, 'sum_over_m_index'):
            config.sum_over_m_index = False
        if not hasattr(config, 'gaussian_width'):
            config.gaussian_width = 0.2721140795
        if not hasattr(config, 'N_PDOS'):
            config.N_PDOS = 1000
        if not hasattr(config, 'min_E_plot'):
            config.min_E_plot = None
        if not hasattr(config, 'max_E_plot'):
            config.max_E_plot = None
        if not hasattr(config, 'print_projection_data'):
            config.print_projection_data = False
        if not hasattr(config, 'recompute_pdos'):
            config.recompute_pdos = False
        if not hasattr(config, 'projection_datapath'):
            config.projection_datapath = None
        if not hasattr(config, 'orthogonalize_atomic_orbitals'):
            config.orthogonalize_atomic_orbitals = False
        if not hasattr(config, 'k_point_parallelization'):
            config.k_point_parallelization = None

        assert isinstance(config.orthogonalize_atomic_orbitals, bool), \
            ORTHOGONALIZE_ATOMIC_ORBITALS_NOT_BOOL_ERROR.format(type(config.orthogonalize_atomic_orbitals))
        # print("is_relaxation = ", file_paths_config['is_relaxation'])
        
        # Create PDOS calculator
        pdos_calculator = PDOSCalculator(
            upf_fname_list = file_paths_config['upf_fname_list'],
            output_fname = file_paths_config['output_fname'],
            eigen_fname = file_paths_config['eigen_fname'],
            static_fname = file_paths_config['static_fname'],
            geopt_fname = file_paths_config['geopt_fname'],
            psi_fname = file_paths_config['psi_fname'],
            out_dirname = file_paths_config['output_dir'],
            is_relaxation = file_paths_config['is_relaxation'],
            orthogonalize_atomic_orbitals = config.orthogonalize_atomic_orbitals,
            k_point_parallelization = config.k_point_parallelization,
        )
        

        if not config.full_pdos_calculation:
            if config.atom_type is None:
                raise ValueError("atom_type must be specified when print_single_atom is True")
            if config.atom_index_for_specified_atom_type is None:
                raise ValueError("atom_index_for_specified_atom_type must be specified when print_single_atom is True")
            
            pdos_calculator.run_single_atom(
                atom_type                          = config.atom_type,
                atom_index_for_specified_atom_type = config.atom_index_for_specified_atom_type,
                mu_PDOS                            = config.gaussian_width,
                N_PDOS                             = config.N_PDOS,
                min_E_plot                         = config.min_E_plot,
                max_E_plot                         = config.max_E_plot,
                load_projection_data_from_txt_path = config.projection_datapath if not config.recompute_pdos else None,
                print_projection_data              = config.print_projection_data,
                sum_over_m_index                   = config.sum_over_m_index
            )
        else:
            pdos_calculator.run(
                mu_PDOS = config.gaussian_width,
                N_PDOS  = config.N_PDOS,
                min_E_plot = config.min_E_plot,
                max_E_plot = config.max_E_plot,
                print_projection_data = config.print_projection_data,
                sum_over_m_index = config.sum_over_m_index,
                load_projection_data_from_txt_path = config.projection_datapath if not config.recompute_pdos else None,
            )
        
        logging.info("PDOS calculation completed successfully!")

    except Exception as e:
        logging.error(f"Error during PDOS calculation: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)




if __name__ == "__main__":
    main()

