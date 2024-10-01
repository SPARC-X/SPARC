/***
 * @file    exactExchangeInitialization.h
 * @brief   This file contains the function declarations for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGEINITIALIZATION_H
#define EXACTEXCHANGEINITIALIZATION_H

#include "isddft.h"


/**
 * @brief   Initialization of all variables for exact exchange.
 */
void init_exx(SPARC_OBJ *pSPARC);

/**
 * @brief   Allocate singularity removal constants
 */
void allocate_singularity_removal_const(SPARC_OBJ *pSPARC);

/**
 * @brief   singularity removal calculation
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 *          Auxiliary function - Method by Gygi and Baldereschi
 *          DOI: 10.1103/PhysRevB.34.4405
 */
void singularity_remooval_const(SPARC_OBJ *pSPARC, double G2, double *sr_const, 
                    double *sr_const_stress, double *sr_const_stress2, double *sr_const_press);

/**
 * @brief   Compute eigenvalues of Laplacian applied with singularity removal methods
 */
void compute_pois_kron_cons(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 */
void compute_pois_fft_const(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 * 
 *          Note: using Finite difference approximation of G2
 */
void compute_pois_fft_const_FD(SPARC_OBJ *pSPARC);

/**
 * @brief   Create communicators between k-point topology and dmcomm.
 * 
 * Notes:   Occupations are only correct within all dmcomm processes.
 *          When not using ACE method, this communicator might be required. 
 */
void create_kpttopo_dmcomm_inter(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute the coefficient for G = 0 term in auxiliary function method
 * 
 *          Note: Using QE's method. Not the original one by Gygi 
 */
void auxiliary_constant(SPARC_OBJ *pSPARC);

/**
 * @brief   Estimation of Ecut by (pi/h)^2/2
 */
double ecut_estimate(double hx, double hy, double hz);

/**
 * @brief   Find out the unique Bloch vector shifts (k-q)
 */
void find_k_shift(SPARC_OBJ *pSPARC);

/**
 * @brief   Find out the positive shift exp(i*(k-q)*r) and negative shift exp(-i*(k-q)*r)
 *          for each unique Bloch shifts
 */
void kshift_phasefactor(SPARC_OBJ *pSPARC);

/**
 * @brief   Find out the k-point for hartree-fock exact exchange in local process
 */
void find_local_kpthf(SPARC_OBJ *pSPARC);

/**
 * @brief   Estimate memory requirement for hybrid calculation
 */
double estimate_memory_exx(const SPARC_OBJ *pSPARC);

#endif // EXACTEXCHANGEINITIALIZATION_H 