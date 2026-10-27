/**
 * @file    sq3.h
 * @brief   This file contains the function declarations for SQ3 method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQ3_H
#define SQ3_H 

#include "isddft.h"

/**
 * @brief   main function of SQ3 method
 */
void SQ3(SPARC_OBJ *pSPARC, int spn_i);

/**
 * @brief   Initialze communicators for SQ3 and allocate memory space.
 */
void init_SQ3(SPARC_OBJ *pSPARC);

#ifdef USE_DP_SUBEIG
/**
 * @brief   Distribute projected Hamiltonian and Density matrix
 */
void DP_Dist2SQ3(SPARC_OBJ *pSPARC);
#else
/**
 * @brief   Distribute projected Hamiltonian and Density matrix
 */
void Dist2SQ3(SPARC_OBJ *pSPARC);
#endif

/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the dense projected Hamiltonian (Hp)
 *
 * @param A            (INPUT)  Dense matrix A (Dense projected Hamiltonian Hp)
 * @param lmin         (OUTPUT) Minimum eigenvalue of A
 * @param lmax         (OUTPUT) Maximum eigenvalue of A
 * @param descA        (INPUT)  Descriptor of A
 * @param mV           (INPUT)  Row blocking factor.
 * @param nV           (INPUT)  Column blocking factor.
 * @param comm         (INPUT)  Communicator of SQ grid.
 * @param tol          (INPUT)  Tolerance for convergence.
 * @param maxit        (INPUT)  Maximum number of iterations allowed.
 */
void Lanczos_dense(const double *A, double *lmin, double *lmax, const int *descA, 
    const int mV, const int nV, MPI_Comm comm, double tol, int maxit);

/**
 * @brief   Initialize Chebyshev components
 */
void init_CHEBCOMP(CHEBCOMP *cc, const int sq_npl, const int m, const int n);

/**
 * @brief   Compute Chebyshev matrix vector components
 *
 * @param pSPARC       (INPUT)  Dense matrix A (Dense projected Hamiltonian Hp).
 * @param cc           (OUTPUT) Pointer to Chebyshev components.
 * @param sq_npl       (INPUT)  Degree of polynomial of Chebyshev components.
 * @param A            (INPUT)  Dense matrix A (Dense projected Hamiltonian Hp).
 * @param a            (INPUT)  Lower bound for Chebyshev filtering.
 * @param b            (INPUT)  Higher bound for Chebyshev filtering.
 */
void Chebyshev_matvec_comp(SPARC_OBJ *pSPARC, CHEBCOMP *cc, 
    const int sq_npl, const double *A, const double a, const double b);

/**
 * @brief   Chebyshev coefficients
 */
void ChebyshevCoeff(double *Ci, const int npl, double (*fun)(double, double, double, int), 
    const double a, const double b, const double beta, const double lambda_f, const int type);

/**
 * @brief   return occupation provided lambda_f when using SQ3 method
 */
double occ_constraint_SQ3(SPARC_OBJ *pSPARC, double lambda_f);

/**
 * @brief   Compute Density matrix 
 */
void SubDensMat(SPARC_OBJ *pSPARC, double *Ds, const double lambda_f, CHEBCOMP *cc);

/**
 * @brief   Compute rho given density matrix
 */
void update_rho_sq3(SPARC_OBJ *pSPARC, double *rho);

void Subspace_Rotation_sq3(SPARC_OBJ *pSPARC);

/**
 * @brief   Entropy using different smearing function
 */
double Eent_func(const double beta, const double lambda, const double lambda_f, const int type);

/**
 * @brief   Uband function 
 */
double Uband_func(const double beta, const double lambda, const double lambda_f, const int type);

/**
 * @brief   Band energy using SQ3 method (Density matrix)
 */
double Calculate_Eband_SQ3(SPARC_OBJ *pSPARC, CHEBCOMP *cc);
/**
 * @brief   Calculate electronic Entropy using SQ3 method (Density matrix)
 */
double Calculate_electronicEntropy_SQ3(SPARC_OBJ *pSPARC, CHEBCOMP *cc);

/**
 * @brief   Free memory space of Chebyshev components
 */
void free_ChemComp(SPARC_OBJ *pSPARC);

/**
 * @brief   Free memory space and communicators for SQ3.
 */
void free_SQ3(SPARC_OBJ *pSPARC);

#endif // EIGENSOLVER_H 
