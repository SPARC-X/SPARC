/**
 * @file    ofdft.h
 * @brief   This file contains the function declarations for Orbital Free DFT.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef OFDFT_H
#define OFDFT_H 

#include "isddft.h"

/**
 * @brief   Initialize OFDFT variables
 */
void init_OFDFT(SPARC_OBJ *pSPARC);

/**
 * @brief   OFDFT NLCG algorithm to solve for electron density
 */
void OFDFT_NLCG_TETER(SPARC_OBJ *pSPARC);

/**
 * @brief   Set up sub-communicators for OFDFT.
 * 
 * Note:    Part of the codes are copied and modified from the code in parallelization.c file
 */
void Setup_Comms_OFDFT(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Hamiltonian times a vector in a matrix-free way.
 *          
 *          The Hamiltonian includes the TFW kinetic functional. 
 *          TODO: add WGC kinetic functional. 
 */
void HamiltonianVecRoutines_OFDFT(
        SPARC_OBJ *pSPARC, int DMnd, int *DMVertices,
        double *u, double *Hu, MPI_Comm comm);

/**
 * @brief   Brents algorithm to find a minimal point within an interval
 * 
 *          argmin_s Energy(u + s * d)
 *          return the s and s_iter is the number of iterations to get s. 
 *          TODO: make it more general. Input option for interval, input constraint function, etc. 
 */
double Brent_Emin(SPARC_OBJ *pSPARC, double tol, double *u, double *d, int *s_iter);

/**
 * @brief   Compute the energy with sqrt(rho) = u + s * d
 *          
 *          return Energy per atom in Ha/atom
 */
double energy_constraint(SPARC_OBJ *pSPARC, double *u, double *d, double s, int DMnd, MPI_Comm comm);

/**
 * @brief   Compute the energy with sqrt(rho) = u
 */
void ofdftTotalEnergy(SPARC_OBJ *pSPARC, double *u);

/**
 * @brief   Free all allocated memory for OFDFT
 */
void Free_OFDFT(SPARC_OBJ *pSPARC);

/**
 * @brief   Free all allocated memory within each relax step
 */
void Free_NLCGvar_OFDFT(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute normalized vector with a direction 
 * 
 *          nu = c * normalize (u + s * d)
 */
void extend_normalized_vector(
        double *u, double *d, double s, double *nu, 
        int len, double c, MPI_Comm comm);

#endif // OFDFT_H
