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

/*
 @ brief: Main function of Chebyshev filtering 
*/

void eigSolve_CheFSI(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error);

/*
 * @brief Set up initial guess for Lanczos.
 *        The eigvecs of Laplacian can be used as good initial guess for the Hamiltonian. 
 *        On the other hand, one can also use random vectors as initial guess.
 *
 * @param x0         Output vector.
 * @param gridsizes  Global grid sizes, [Nx,Ny,Nz].
 * @param DMVert     Local domain vertices owned by the current process.
 * @param RandFlag   Flag that specifies whether random vectors are used.
 * @param comm       The communicator where the vector is distributed.
 */
void init_guess_Lanczos(
    double *x0, double cellsizes[3], int gridsizes[3], double meshes[3], 
    int DMVert[6], int RandFlag, MPI_Comm comm
);

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
void CheFSI(SPARC_OBJ *pSPARC, double lambda_cutoff, double *x0, int count, int k, int spn_i);


/**
 * @brief   Find Chebyshev filtering bounds and cutoff constants.
 */
void Chebyshevfilter_constants(SPARC_OBJ *pSPARC, double *x0, double *lambda_cutoff, double *eigmin, double *eigmax, int count, int k, int spn_i);

/**
 * @brief   Perform Chebyshev filtering.
 */
void ChebyshevFiltering(SPARC_OBJ *pSPARC, int *DMVertices, 
        double *X, int ldi, double *Y, int ldo, int ncol, 
        int m, double a, double b, double a0, int k, int spn_i, MPI_Comm comm, 
        double *time_info);


/* ============================================================================= 
   For solving the standard subspace eigenproblem instead of the generalized one 
   ============================================================================= */
void Solve_standard_EigenProblem(SPARC_OBJ *pSPARC, int k, int spn_i);

/* ============================================================================= 
   ============================================================================= */

#ifdef USE_DP_SUBEIG
struct DP_CheFSI_s
{
    int      nproc_row;         // Number of processes in process row, == comm size of pSPARC->blacscomm
    int      nproc_kpt;         // Number of processes in kpt_comm 
    int      rank_row;          // Rank of this process in process row, == rank in pSPARC->blacscomm
    int      rank_kpt;          // Rank of this process in kpt_comm;
    int      Ns_bp;             // Number of bands this process has in the original band parallelization (BP), 
                                // == number of local states (bands) in SPARC == pSPARC->{band_end_indx-band_start_indx} + 1
    int      Ns_dp;             // Number of bands this process has in the converted domain parallelization (DP),
                                // == number of total states (bands) in SPARC == pSPARC->Nstates
    int      Nd_bp;             // Number of FD points this process has in the original band parallelization (BP), == pSPARC->Nd_d_dmcomm
    int      Ndsp_bp;           // Leading dimension of this process
    int      Nd_dp;             // Number of FD points this process has after converted to domain parallelization (DP)
    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int      desc_Hp_local[9];  // descriptor for Hp_local on each ictxt_blacs_topo
    int      desc_Mp_local[9];  // descriptor for Mp_local on each ictxt_blacs_topo
    int      desc_eig_vecs[9];  // descriptor for eig_vecs on each ictxt_blacs_topo
    #endif
    int      *Ns_bp_displs;     // Size nproc_row+1, the pSPARC->band_start_indx on each process in pSPARC->blacscomm
    int      *Nd_dp_displs;     // Size nproc_row+1, displacements of FD points for each process in DP
    int      *bp2dp_sendcnts;   // BP to DP send counts
    int      *bp2dp_sdispls;    // BP to DP displacements
    int      *dp2bp_sendcnts;   // DP to BP send counts
    int      *dp2bp_sdispls;    // DP to BP send displacements
    double   *Y_packbuf;        // Y pack buffer
    double   *HY_packbuf;       // HY pack buffer
    double   *Y_dp;             // Y block in DP
    double   *HY_dp;            // HY block in DP
    double   *Mp_local;         // Local Mp result
    double   *Hp_local;         // Local Hp result
    double   *eig_vecs;         // Eigen vectors from solving generalized eigenproblem
    MPI_Comm kpt_comm;          // MPI communicator that contains all active processes in pSPARC->kptcomm
};
typedef struct DP_CheFSI_s* DP_CheFSI_t;

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
void DP_Project_Hamiltonian(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, int ldi, double *HY, int ldo, double *Hp, double *Mp, int spn_i);

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix with domain parallelization
 *          data partitioning for standard eigenvalue problem.
 */
void DP_Project_Hamiltonian_std(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, int ldi, double *HY, int ldo, int spn_i);

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
void Project_Hamiltonian(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, int ldi, double *HY, int ldo,
                         double *Hp, double *Mp, int k, int spn_i, MPI_Comm comm);
                        


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

/**
 * @brief   Orthogonalization of dense matrix A by Choleskey factorization
 * 
 * @param A            (INPUT)  Distributed dense matrix A.
 * @param descA        (INPUT)  Descriptor of A.
 * @param z            (INPUT/OUTPUT) INPUT: z=A'*A, OUTPUT: A'*A=z'*z, z is upper triangular matrix.
 * @param descz        (INPUT)  Descriptor of Z.
 * @param m            (INPUT)  Row blocking factor.
 * @param n            (INPUT)  Column blocking factor.
 */
void Chol_orth(double *A, const int *descA, double *z, const int *descz, const int *m, const int *n);

#endif // EIGENSOLVER_H 
