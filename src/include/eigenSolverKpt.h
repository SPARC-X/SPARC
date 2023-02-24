/**
 * @file    eigenSolverKpt.h
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

#ifndef EIGENSOLVERKPT_H
#define EIGENSOLVERKPT_H 

#include "isddft.h"

/*
 @ brief: Main function of Chebyshev filtering 
*/

void eigSolve_CheFSI_kpt(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error);

/**
 * @brief   Apply Chebyshev-filtered subspace iteration steps.
 */
void CheFSI_kpt(SPARC_OBJ *pSPARC, double lambda_cutoff, double _Complex *x0, int count, int kpt, int spn_i);

/**
 * @brief   Find Chebyshev filtering bounds and cutoff constants.
 */
void Chebyshevfilter_constants_kpt(
    SPARC_OBJ *pSPARC, double _Complex *x0, double *lambda_cutoff, double *eigmin, 
    double *eigmax, int count, int kpt, int spn_i
);

/**
 * @brief   Perform Chebyshev filtering.
 */
void ChebyshevFiltering_kpt(
    SPARC_OBJ *pSPARC, int *DMVertices, double _Complex *X, double _Complex *Y, int ncol, 
    int m, double a, double b, double a0, int kpt, int spn_i, MPI_Comm comm, 
    double *time_info
);

#ifdef USE_DP_SUBEIG
/**
 * @brief   Initialize domain parallelization data structures for calculating projected Hamiltonian,  
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI_kpt().
 */
void init_DP_CheFSI_kpt(SPARC_OBJ *pSPARC);

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
void DP_Project_Hamiltonian_kpt(SPARC_OBJ *pSPARC, int *DMVertices, double _Complex *Y, double _Complex *Hp, double _Complex *Mp, int spn_i, int kpt);

/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x using domain parallelization
 *          data partitioning. 
 *
 *          Rank 0 of each kpt_comm will solve this generalized eigenproblem locally and 
 *          then broadcast the obtained eigenvectors to all other processes in this kpt_comm.
 */
void DP_Solve_Generalized_EigenProblem_kpt(SPARC_OBJ *pSPARC, int kpt, int spn_i);

/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals, using domain parallelization
 *          data partitioning. 
 *
 *          This is just to perform a matrix-matrix multiplication: PsiQ = YPsi* Q, here 
 *          Psi == Y, Q == eigvec. Note that Y and YQ are in domain parallelization data 
 *          layout. We use MPI_Alltoallv to convert the obtained YQ back to the band + domain 
 *          parallelization format in SPARC and copy the transformed YQ to Psi_rot. 
 */
void DP_Subspace_Rotation_kpt(SPARC_OBJ *pSPARC, double _Complex *Psi_rot);

/**
 * @brief   Free domain parallelization data structures for calculating projected Hamiltonian, 
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI_kpt().
 */
void free_DP_CheFSI_kpt(SPARC_OBJ *pSPARC);
#endif  // End of "#ifdef USE_GTMATRIX"

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix.
 *
 *          Hp = Y' * H * Y, 
 *          Mp = Y' * Y.
 */
void Project_Hamiltonian_kpt(SPARC_OBJ *pSPARC, int *DMVertices, double _Complex *Y, 
                         double _Complex *Hp, double _Complex *Mp, int kpt, int spn_i, MPI_Comm comm);


/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x.
 *
 *          Note: Hp = Psi' * H * Psi, Mp = Psi' * Psi. Also note that both Hp and 
 *                Mp are distributed block cyclically.
 *          
 *          TODO: At some point it is better to use ELPA (https://elpa.mpcdf.mpg.de/) 
 *                for solving subspace eigenvalue problem, which can provide up to 
 *                3x speedup.
 */
void Solve_Generalized_EigenProblem_kpt(SPARC_OBJ *pSPARC, int kpt, int spn_i);


/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals.
 *
 *          This is just to perform a matrix-matrix multiplication: Psi = Psi * Q.
 *          Note that Psi, Q and PsiQ are distributed block cyclically, Psi_rot is
 *          the band + domain parallelization format of PsiQ.
 */
void Subspace_Rotation_kpt(SPARC_OBJ *pSPARC, double _Complex *Psi, double _Complex  *Q, double _Complex *PsiQ, double _Complex *Psi_rot, int kpt, int spn_i);


/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Hamiltonian corresponding to the given k-point.  
 */
void Lanczos_kpt(const SPARC_OBJ *pSPARC, int *DMVertices, double *Veff_loc,
             ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
             double *eigmin, double *eigmax, double _Complex *x0, double TOL_min, double TOL_max, 
             int MAXIT, int kpt, int spn_i, MPI_Comm comm, MPI_Request *req_veff_loc);
             

/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Lap.  
 */
void Lanczos_laplacian_kpt(
    const SPARC_OBJ *pSPARC, const int *DMVertices, double *eigmin, 
    double *eigmax, double _Complex *x0, const double TOL_min, const double TOL_max, 
    const int MAXIT, int kpt, int spn_i, MPI_Comm comm
);            

#endif // EIGENSOLVERKPT_H 
