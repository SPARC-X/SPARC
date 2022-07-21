/**
 * @file    sq.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQ_H
#define SQ_H 

#include "isddft.h"

/**
 * @brief   Compute nodal Hamiltonian times a vector
 * 
 * @param vec   The 3-d vector multiplied by nodal hamiltonian. Vector size is the nodal Rcut domain
 * @param i     The corresponding x - coordinates of the node w.r.t. PR domain
 * @param j     The corresponding y - coordinates of the node w.r.t. PR domain
 * @param k     The corresponding z - coordinates of the node w.r.t. PR domain
 * @param Hv    The 3-d vector stores the result of Hx
 */
void HsubTimesVec(SPARC_OBJ *pSPARC, double ***vec, int i, int j, int k, double ***Hv);

/**
 * @brief   Gauss Quadrature method
 * 
 * @param scf_iter  The scf iteration counter
 */
void GaussQuadrature(SPARC_OBJ *pSPARC, int scf_iter);

/**
 * @brief   Tridiagonal eigenvalue solver for Gauss method
 * 
 * @param diag          Diagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param subdiag       Subdiagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param nd            FD node index in current process domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void TridiagEigenSolve_gauss(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int nd, double *lambda_min, double *lambda_max);

/**
 * @brief   Lanczos algorithm for Gauss method
 * 
 * @param vkm1  Initial guess for Lanczos algorithm
 * @param i     The corresponding x - coordinates of the node w.r.t. PR domain
 * @param j     The corresponding y - coordinates of the node w.r.t. PR domain
 * @param k     The corresponding z - coordinates of the node w.r.t. PR domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 * @param nd            FD node index in current process domain
 */
void LanczosAlgorithm_gauss(SPARC_OBJ *pSPARC, double ***vkm1, int i, int j, int k, double *lambda_min, double *lambda_max, int nd);

/**
 * @brief   Compute chebyshev coefficients Ci (length npl_c+1) to fit the function "func"
 * 
 * NOTE:    THIS FUNCTION CURRENTLY USES DISCRETE ORTHOGONALITY PROPERTY OF CHEBYSHEV POLYNOMIALS WHICH ONLY GIVE "approximate" 
 *          COEFFICIENTS. SO IF THE FUNCTION IS NOT SMOOTH ENOUGH THE COEFFICIENTS WILL NOT BE ACCURATE ENOUGH. 
 *          SO THIS FUNCTION HAS TO BE REPLACED WITH ITS CONTINUOUS COUNTER PART WHICH EVALUATES OSCILLATORY INTEGRALS AND USES FFT.
 */
void ChebyshevCoeff_SQ(int npl_c, double *Ci, double *d, double (*fun)(double, double, double, int), 
                const double beta, const double lambda_f, const int type);

/**
 * @brief   Lanczos algorithm computing only minimal eigenvales
 * 
 * TODO:    There are 2 different Lanczos algorithm with confusing names. Change them or combine them
 */
void LanczosAlgorithm_new(SPARC_OBJ *pSPARC, int i, int j, int k, double *lambda_min, int nd, int choice);

/**
 * @brief   Tridiagonal solver for eigenvalues. Part of Lanczos algorithm.
 */
void TridiagEigenSolve_new(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int n, double *lambda_min, int choice, int nd);

#endif // SQ_H