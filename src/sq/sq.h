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


/**
 * @brief   Initialize SQ variables and create necessary communicators
 */
void init_SQ(SPARC_OBJ *pSPARC);

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
 * @brief   Free all allocated memory spaces and communicators 
 */
void Free_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Free SCF variables for SQ methods.
 */
void Free_scfvar_SQ(SPARC_OBJ *pSPARC);

#endif // SQ_H