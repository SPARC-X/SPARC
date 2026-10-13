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
 * @brief   Gauss Quadrature method
 * 
 * @param scf_iter  The scf iteration counter
 */
// void GaussQuadrature(SPARC_OBJ *pSPARC, int SCFCount, int spn_i);
void GaussQuadrature(SPARC_OBJ *pSPARC, int spn_i);

/**
 * @brief   Tridiagonal eigenvalue solver for Gauss method (dstedc on the workspace of init_SQ): the eigenvalues
 *          are the nodes of point nd, the squared first components of the eigenvectors its weights; the
 *          eigenvectors go to pSQ->w (w_all[nd] at high temperature); in the force step (pSQ->forceFlag) also
 *          f(T) e1 into pSQ->eig_F.
 * 
 * @param diag          Diagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param subdiag       Subdiagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param nd            FD node index in current process domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void TridiagEigenSolve_gauss(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int nd, int spn_i, double *lambda_min, double *lambda_max);

/**
 * @brief   Lanczos algorithm for Gauss method on grid point nd (start vector: the unit vector at the point;
 *          the point's Veff and nonlocal projectors are gathered first): the fused sweep (static
 *          LanczosAlgorithm_fused in sq.c), or the classical steps for hybrid functionals
 * 
 * @param nd            FD node index in current process domain
 * @param spn_i         local spin index
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void LanczosAlgorithm_gauss(SPARC_OBJ *pSPARC, int nd, int spn_i, double *lambda_min, double *lambda_max);


/**
 * @brief   Compute nodal Hamiltonian times a vector (Veff: pSQ->Veff_box, nonlocal projectors: pSQ->nlocProj_pt,
 *          both gathered for the point and spin of the current Lanczos run by LanczosAlgorithm_gauss)
 * 
 * @param x     The vector multiplied by nodal hamiltonian. Vector size is the nodal Rcut domain
 * @param nd    The index of current node in the process domain
 * @param Hx    The vector stores the result of Hx
 */
void HsubTimesVec(SPARC_OBJ *pSPARC, const double *x, const int nd, double *Hx);

/**
 * @brief Calculate (a * Laplacian + diag(Veff)) times x on the Rcut box in Kronecker sum form with the dense 1D
 *        Laplacians (pSQ->Lap_1D_*) and the point's Veff (pSQ->Veff_box): with MKL per z-slab with JIT dgemm
 *        kernels (Veff, x and y stage) and the z stage in row chunks, otherwise dgemm along x and per slab along
 *        y and z
 * 
 * @param pSPARC    SPARC object pointer
 * @param x         vector
 * @param a         scaling of Laplacian
 * @param Hx        (a * Laplacian + diag(Veff))x 
 */
void Lap_vec_mult_orth_SQ_kron(SPARC_OBJ *pSPARC, const double *x, const double a, double *Hx);

/**
 * @brief   Destroy the JIT dgemm kernels of the Laplacian, the fused Lanczos sweep and the nonlocal pass (no-op without MKL)
 */
void SQ_jit_free(SPARC_OBJ *pSPARC);

#endif // SQ_H
