/**
 * @file    spdft.c
 * @brief   This file contains functions for the spectral-partitioned DFT method.
 *
 * @authors Abhiraj Sharma <sharma20@llnl.gov>
 *          John E. Pask <pask1@llnl.gov>
 *
 * Copyright (c) 2022 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef SPDFT_H
#define SPDFT_H 

#include "isddft.h"


// create plane wavevectors & allocate memory
void spDFT_initialization(SPARC_OBJ *pSPARC, double eshift);

// Calculate rescaled planewaves during NPT type simulations
void spDFT_planewaves(SPARC_OBJ *pSPARC);

// Calculate kinetic and nonlocal contribution of plane waves
void spDFT_eigen_kinetic_nonlocal(SPARC_OBJ *pSPARC);


// Calculate kinetic contribution of planewaves
void spDFT_kinetic(SPARC_OBJ *pSPARC, int kpt);


// Calculate nonlocal contribution of planewaves
void spDFT_nonlocal_energy_stress(SPARC_OBJ *pSPARC, int kpt);
void spDFT_nonlocal_energy_pressure(SPARC_OBJ *pSPARC, int kpt, int ncol, int *indices_G, int *indices, int *COUNT_idn);


// Calculate electron density due to plane waves
void spDFT_electrondensity_homo(SPARC_OBJ *pSPARC, double *rho);


// Calculate total eigenvalue contribution of plane waves
void spDFT_eigen_total(SPARC_OBJ *pSPARC);


// Calculate minimum and maximum eigenvalues among plane waves
void spDFT_eigen_minmax(SPARC_OBJ *pSPARC, double *eigmin, double *eigmax);


double sigmoid(double x, double shift1, double shift2, double smearing);


// Occupation constrint for Fermi level calulation 
double occ_constraint_spDFT(SPARC_OBJ *pSPARC, double lambda_f);


// Calculation of electronic occupations
void spDFT_occupation(SPARC_OBJ *pSPARC, double lambda_f);


// functions needed for electronic entropy calculation in spDFT
double partialoccupation_sp(double x, double x0, double y0);
double entropy_sp_limx0(double x, double x0, double y0);
double entropy_FD(double x);
double generate_grid(double x1, double x2, int nnodes, double *nodes);
int find_xref_entropyref(double occ, double *x, double *entropy_x);
double numeric_intg_trapz_uniform(int nnodes, double shift, double h, double *f);
double numeric_intg_trapz_nonuniform(int nnodes, double h, double occ, double x_ref, double entropy_ref, double *dentropy);

// Electronic entropy calculation in spDFT
double spDFT_entropy(SPARC_OBJ *pSPARC, int kpt, int spn);

// Stress contribution from homogenous gas in spDFT
void spDFT_stress(SPARC_OBJ *pSPARC);

// Pressure contribution from homogenous gas in spDFT
void spDFT_pressure_homo(SPARC_OBJ *pSPARC);

// Check the minimum pw occupation
void spDFT_occcheck(SPARC_OBJ *pSPARC, int rank);

// Print the splitting energy in .spDFT file
void spDFT_write(SPARC_OBJ *pSPARC, int rank);

// Read and scatter data from .spDFT file
void spDFT_read(SPARC_OBJ *pSPARC);

// Free variables at the end of each scf cycle
void spDFT_free_scfvar(SPARC_OBJ *pSPARC);


// Free variables at the end of the simulation
void spDFT_finalization(SPARC_OBJ *pSPARC); 

#endif // SPDFT_H

