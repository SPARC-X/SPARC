/**
 * @file    eigenSolver.h
 * @brief   This file contains the function declarations for eigen-solvers.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H 

#include "isddft.h"
#if USE_PCE
#include <libpce_structs.h>
#include "hamstruct.h"
#endif

/*
 @ brief: Main function of Chebyshev filtering 
*/

#if USE_PCE
void eigSolve_CheFSI(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error,
                     Hybrid_Decomp *hd, Chebyshev_Info *cheb, Eig_Info *Eigvals,
                     Our_Hamiltonian_Struct *ham_struct, 
                     Psi_Info *Psi1, Psi_Info *Psi2, Psi_Info *Psi3,
                     MPI_Comm kptcomm, MPI_Comm dmcomm, MPI_Comm blacscomm);
#else
void eigSolve_CheFSI(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error);
#endif

/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Hamiltonian corresponding to the given k-point.
 *
 * @param eigmin       (OUTPUT) Minimum eigenvalue of the Hamiltonian H_k.
 * @param eigmax       (OUTPUT) Maximum eigenvalue of the Hamiltonian H_k.
 * @param x0           (INPUT)  Initial guess vector.
 * @param TOL_min      (INPUT)  Tolerance for minimum eigenvalue.
 * @param TOL_max      (INPUT)  Tolerance for maximum eigenvalue.
 * @param MAXIT        (INPUT)  Maximum number of iterations allowed.
 * @param kpt          (INPUT)  k-th k-point assigned to the kptcomm group.
 * @param req_veff_loc (INPUT)  MPI_Request handle for checking availability of Veff_loc.
 */
void Lanczos(const SPARC_OBJ *pSPARC, int *DMVertices, double *Veff_loc, 
             ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
             double *eigmin, double *eigmax, double *x0, double TOL_min, double TOL_max, 
             int MAXIT, int k, int spn_i, MPI_Comm comm, MPI_Request *req_veff_loc);


/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Lap.  
 *
 * @param eigmin       (OUTPUT) Minimum eigenvalue of the Hamiltonian H_k.
 * @param eigmax       (OUTPUT) Maximum eigenvalue of the Hamiltonian H_k.
 * @param x0           (INPUT)  Initial guess vector.
 * @param TOL_min      (INPUT)  Tolerance for minimum eigenvalue.
 * @param TOL_max      (INPUT)  Tolerance for maximum eigenvalue.
 * @param MAXIT        (INPUT)  Maximum number of iterations allowed.
 */
void Lanczos_laplacian(
    const SPARC_OBJ *pSPARC, const int *DMVertices, double *eigmin, 
    double *eigmax, double *x0, const double TOL_min, const double TOL_max, 
    const int MAXIT, int k, int spn_i, MPI_Comm comm
);



/**
 * @brief   Apply Chebyshev-filtered subspace iteration steps.
 */
#if USE_PCE
void CheFSI(SPARC_OBJ *pSPARC, double lambda_cutoff, double *x0, int count, int k, int spn_i,
            Hybrid_Decomp *hd, Chebyshev_Info *cheb, Eig_Info *Eigvals,
            Our_Hamiltonian_Struct *ham_struct, 
            Psi_Info *Psi1, Psi_Info *Psi2, Psi_Info *Psi3,
            MPI_Comm kptcomm, MPI_Comm dmcomm, MPI_Comm blacscomm);
#else
void CheFSI(SPARC_OBJ *pSPARC, double lambda_cutoff, double *x0, int count, int k, int spn_i);
#endif;

/**
 * @brief   Find Chebyshev filtering bounds and cutoff constants.
 */
void Chebyshevfilter_constants(SPARC_OBJ *pSPARC, double *x0, double *lambda_cutoff, double *eigmin, double *eigmax, int count, int k, int spn_i);

/**
 * @brief   Perform Chebyshev filtering.
 */
void ChebyshevFiltering(SPARC_OBJ *pSPARC, int *DMVertices, double *X, double *Y, int ncol, 
                        int m, double a, double b, double a0, int k, int spn_i, MPI_Comm comm, double *time_info);


#ifdef USE_DP_SUBEIG
/**
 * @brief   Initialize domain parallelization data structures for calculating projected Hamiltonian,  
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI().
 */
void init_DP_CheFSI(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix with domain parallelization
 *          data partitioning. 
 *
 *          Hp = Y' * H * Y,  Mp = Y' * Y. The tall-skinny column-matrices Y and HY 
 *          are partitioned by rows only in this function. MPI_Alltoallv is used to 
 *          convert the data layout from band parallelization to domain parallelization
 *          in each original domain parallelization part (blacscomm). Then we need 2 
 *          MPI_Reduce to get the final Hp and Mp on rank 0 of each kpt_comm.
 */
void DP_Project_Hamiltonian(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, double *Hp, double *Mp, int spn_i);

/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x using domain parallelization
 *          data partitioning. 
 *
 *          Rank 0 of each kpt_comm will solve this generalized eigenproblem locally and 
 *          then broadcast the obtained eigenvectors to all other processes in this kpt_comm.
 */
void DP_Solve_Generalized_EigenProblem(SPARC_OBJ *pSPARC, int spn_i);

/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals, using domain parallelization
 *          data partitioning. 
 *
 *          This is just to perform a matrix-matrix multiplication: PsiQ = YPsi* Q, here 
 *          Psi == Y, Q == eigvec. Note that Y and YQ are in domain parallelization data 
 *          layout. We use MPI_Alltoallv to convert the obtained YQ back to the band + domain 
 *          parallelization format in SPARC and copy the transformed YQ to Psi_rot. 
 */
void DP_Subspace_Rotation(SPARC_OBJ *pSPARC, double *Psi_rot);

/**
 * @brief   Free domain parallelization data structures for calculating projected Hamiltonian, 
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI().
 */
void free_DP_CheFSI(SPARC_OBJ *pSPARC);
#endif  // End of "#ifdef USE_GTMATRIX"

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix.
 *
 *          Hp = X' * H * X, 
 *          M = X' * X.
 */
void Project_Hamiltonian(SPARC_OBJ *pSPARC, int *DMVertices, double *X, double *Hp, double *M, int k, int spn_i, MPI_Comm comm);


/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x.
 *
 *          Note: Hp = Psi' * H * Psi, Mp = Psi' * Psi. Also note that both Hp and Mp are distributed block cyclically.
 */
void Solve_Generalized_EigenProblem(SPARC_OBJ *pSPARC, int k, int spn_i);


/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals.
 *
 *          This is just to perform a matrix-matrix multiplication: Psi = Psi * Q.
 *          Note that Psi, Q and PsiQ are distributed block cyclically, Psi_rot is
 *          the band + domain parallelization format of PsiQ.
 */
void Subspace_Rotation(SPARC_OBJ *pSPARC, double *Psi, double *Q, double *PsiQ, double *Psi_rot, int k, int spn_i);


/**
 * @brief   Calculate Chebyshev polynomial degree based on effective mesh size.
 *
 *          The concept of effective mesh size used here since the Chebyshev polynomial
 *          degree is directly related to the spectral width of the Hamiltonian, which
 *          can be approximated by the maximum eigenvalue the -0.5*Laplacian, which is
 *          preportional to h_eff^-2. We use a cubic polynomial p(x) to fit four data
 *          points (h, npl) = {(0.1,50), (0.2,35), (0.4,20), (0.7,14)}. If h > 0.7, npl
 *          is fixed to be 14.
 *
 * @param h   Effective mesh size.
 */
int Mesh2ChebDegree(double h);


#endif // EIGENSOLVER_H 
