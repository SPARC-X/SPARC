/***
 * @file    sq3.c
 * @brief   This file contains the functions for SQ3 method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/** ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif

#include "sq3.h"
#include "eigenSolver.h"
#include "tools.h" 
#include "linearSolver.h" // Lanczos
#include "lapVecRoutines.h"
#include "hamiltonianVecRoutines.h"
#include "occupation.h"
#include "isddft.h"
#include "parallelization.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
int CheFSI_use_EVA = -1;
#endif

#define TEMP_TOL (1e-14)

/**
 * @brief   main function of SQ3 method
 */
void SQ3(SPARC_OBJ *pSPARC, int spn_i)
{
    int rank;
    double t1, t2, t3, lmin, lmax, send[2];
    CHEBCOMP *cc = pSPARC->ChebComp+spn_i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // copy to SQ and cmc grid
    t1 = MPI_Wtime();
    #ifdef USE_DP_SUBEIG
    DP_Dist2SQ3(pSPARC);
    #else
    Dist2SQ3(pSPARC);
    #endif
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("Distributing and broadcasting Hp took %.3f ms\n",(t2-t1)*1e3); 
    #endif

    // TODO: gather and broadcast within each blacscomm
    t1 = MPI_Wtime();
    if (pSPARC->ictxt_SQ3 > -1)
        Lanczos_dense(pSPARC->Hp_SQ3, &lmin, &lmax, pSPARC->desc_Hp_SQ3, 
            pSPARC->nr_Hp_SQ3, pSPARC->nc_Hp_SQ3, pSPARC->SQ3comm, 1e-2, 1000);
    t2 = MPI_Wtime();
    
    send[0] = lmin; send[1] = lmax;
    MPI_Bcast(send, 2, MPI_DOUBLE, 0, pSPARC->kptcomm);
    lmin = send[0]; lmax = send[1];

    pSPARC->lambda_sorted[pSPARC->Nstates * spn_i] = lmin;
    pSPARC->lambda_sorted[pSPARC->Nstates * (spn_i + 1) - 1] = lmax;

    lmin = lmin - 0.2 - 0.1 * fabs(lmin);
    lmax = lmax + 0.2 + 0.1 * fabs(lmax);

    cc->eigmin = lmin;
    cc->eigmax = lmax;
    t3 = MPI_Wtime();
    
    t1 = MPI_Wtime();
    Chebyshev_matvec_comp(pSPARC, cc, cc->sq3_npl, pSPARC->Hp_cmc, cc->eigmin, cc->eigmax);
    t2 = MPI_Wtime();
    if (pSPARC->Dscomm != MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE, cc->tr_Ti, cc->sq3_npl+1, MPI_DOUBLE, MPI_SUM, pSPARC->Dscomm);
    t3 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) {
        printf("Time for Chebyshev matrix-vector component: %.3f ms, Allreduce trace of each component: %.3fms\n", (t2-t1)*1e3, (t3-t2)*1e3);
        printf("Total time for calculating Chebyshev matrix-vector component: %.3f ms\n", (t3-t1)*1e3);
    }
    #endif
    (void)t1; (void)t2; (void)t3; // timers are only printed in DEBUG mode
}


/**
 * @brief   Initialze communicators for SQ3 and allocate memory space.
 */
void init_SQ3(SPARC_OBJ *pSPARC)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, size_kptcomm, nproc_kptcomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    size_kptcomm = nproc_kptcomm;
    if (pSPARC->kptcomm_index != -1){
        // initialize kptcomm grid
        pSPARC->bhandle_kptcomm = Csys2blacs_handle(pSPARC->kptcomm);
        pSPARC->ictxt_kptcomm = pSPARC->bhandle_kptcomm;

        Cblacs_gridinit( &pSPARC->ictxt_kptcomm, "Row", 1, size_kptcomm);
    } else {
        pSPARC->ictxt_kptcomm = -1;
    }

    int nprow, npcol, myrow, mycol, info, ZERO = 0, llda;

    // set up a square grid for the matrix operation on Hp
    int mb_SQ, nb_SQ;
    // using the largest square grid for Hp_SQ
    nprow = npcol = (int)sqrt((double)size_kptcomm);
    #ifdef DEBUG
    if (!rank) printf("SQ grid size: %d x %d\n", nprow, npcol);
    #endif
    mb_SQ = nb_SQ = pSPARC->Nstates / nprow;

    if (pSPARC->kptcomm_index != -1){
        // initialize SQ grid
        pSPARC->bhandle_SQ3 = Csys2blacs_handle(pSPARC->kptcomm);
        pSPARC->ictxt_SQ3 = pSPARC->bhandle_SQ3;
        
        Cblacs_gridinit( &pSPARC->ictxt_SQ3, "Row", nprow, npcol);
        Cblacs_gridinfo( pSPARC->ictxt_SQ3, &nprow, &npcol, &myrow, &mycol );

        // Construct SQ_comm including only processors within SQ_comm
        MPI_Group kptgroup, SQ_group;
        MPI_Comm_group(pSPARC->kptcomm, &kptgroup);
        int *incl_ranks = (int*) calloc(nprow*npcol, sizeof(int));
        for (int i = 0; i < nprow*npcol; i++)
            incl_ranks[i] = i;

        MPI_Group_incl(kptgroup, nprow*npcol, incl_ranks, &SQ_group);
        free(incl_ranks);

        MPI_Comm_create(pSPARC->kptcomm, SQ_group, &pSPARC->SQ3comm);
        MPI_Group_free(&kptgroup);
        MPI_Group_free(&SQ_group);

        // calculating the local size for projected Hamiltonian 
        if (pSPARC->ictxt_SQ3 > -1){
            pSPARC->nr_Hp_SQ3 = numroc_( &pSPARC->Nstates, &mb_SQ, &myrow, &ZERO, &nprow );
            pSPARC->nc_Hp_SQ3 = numroc_( &pSPARC->Nstates, &nb_SQ, &mycol, &ZERO, &npcol );
            llda = max(1,pSPARC->nr_Hp_SQ3);
            descinit_(pSPARC->desc_Hp_SQ3, &pSPARC->Nstates, &pSPARC->Nstates, 
                    &mb_SQ, &nb_SQ,  &ZERO, &ZERO, &pSPARC->ictxt_SQ3, &llda, &info);
        } else {
            pSPARC->nr_Hp_SQ3 = pSPARC->nc_Hp_SQ3 = 0;
            for (int i = 0; i < 9; i++){
                pSPARC->desc_Hp_SQ3[i] = -1;
            }
        }
    } else {
        // Suggest to set default values -1 rather than 0
        pSPARC->ictxt_SQ3 = -1;
        pSPARC->nr_Hp_SQ3 = pSPARC->nc_Hp_SQ3 = 0;
        for (int i = 0; i < 9; i++){
            pSPARC->desc_Hp_SQ3[i] = -1;
        }
        pSPARC->SQ3comm = MPI_COMM_NULL;
    }

    // Allocating memory space for Hp
    pSPARC->Hp_SQ3 = (double*) calloc(pSPARC->nr_Hp_SQ3 * pSPARC->nc_Hp_SQ3, sizeof(double));

    // set up a context with single processor to receive entire density matrix Ds
    int mb_cmc, nb_cmc;
    nprow = npcol = 1;
    mb_cmc = nb_cmc = pSPARC->Nstates;

    if (pSPARC->kptcomm_index != -1){
        pSPARC->bhandle_cmc = Csys2blacs_handle(pSPARC->kptcomm);
        pSPARC->ictxt_cmc = pSPARC->bhandle_cmc;

        Cblacs_gridinit( &pSPARC->ictxt_cmc, "Row", nprow, npcol);
        Cblacs_gridinfo( pSPARC->ictxt_cmc, &nprow, &npcol, &myrow, &mycol );

        if (pSPARC->ictxt_cmc > -1){
            llda = max(1, pSPARC->Nstates);
            descinit_(pSPARC->desc_Hp_cmc, &pSPARC->Nstates, &pSPARC->Nstates, 
                    &mb_cmc, &nb_cmc,  &ZERO, &ZERO, &pSPARC->ictxt_cmc, &llda, &info);   
        } else {
            for (int i = 0; i < 9; i++){
                pSPARC->desc_Hp_cmc[i] = -1;
            }
        }
    } else {
        pSPARC->ictxt_cmc = -1;
        for (int i = 0; i < 9; i++){
            pSPARC->desc_Hp_cmc[i] = -1;
        }
    }

    // Construct Dscomm only including processors using in calculating density matrix Ds
    pSPARC->size_Dscomm = min(nproc_kptcomm, pSPARC->Nstates);
    #ifdef DEBUG
    if (!rank) printf("Using %d processors to calculate Ds\n", pSPARC->size_Dscomm);
    #endif
    int color = (rank_kptcomm < pSPARC->size_Dscomm) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(pSPARC->kptcomm, color, rank_kptcomm, &pSPARC->Dscomm);
    // Using cyclic distribution 
    pSPARC->cmc_cols = pSPARC->Nstates / size_kptcomm + ((rank_kptcomm < pSPARC->Nstates % size_kptcomm) ? 1 : 0);

    if (rank_kptcomm < pSPARC->size_Dscomm) {
        #ifndef USE_DP_SUBEIG
        pSPARC->Hp_cmc = (double*) calloc(pSPARC->Nstates * pSPARC->Nstates, sizeof(double));
        #endif
        pSPARC->Ds_cmc = (double*) calloc(pSPARC->Nstates * pSPARC->cmc_cols, sizeof(double));
    }

    // Create cyclic distribution grid for Ds
    int mb_ds, nb_ds;
    nprow = 1;
    npcol = pSPARC->size_Dscomm;
    mb_ds = pSPARC->Nstates;
    nb_ds = 1;

    if (pSPARC->Dscomm != MPI_COMM_NULL){
        pSPARC->bhandle_Ds = Csys2blacs_handle(pSPARC->Dscomm);
        pSPARC->ictxt_Ds = pSPARC->bhandle_Ds;

        Cblacs_gridinit( &pSPARC->ictxt_Ds, "Row", nprow, npcol);
        Cblacs_gridinfo( pSPARC->ictxt_Ds, &nprow, &npcol, &myrow, &mycol );

        llda = max(1, pSPARC->Nstates);
        descinit_(pSPARC->desc_Ds, &pSPARC->Nstates, &pSPARC->Nstates, 
                &mb_ds, &nb_ds,  &ZERO, &ZERO, &pSPARC->ictxt_Ds, &llda, &info);   
    } else {
        pSPARC->ictxt_Ds = -1;
        for (int i = 0; i < 9; i++){
            pSPARC->desc_Ds[i] = -1;
        }
    }

    // Ensure only the first blacscomm involved in communication
    int dmrank = -1;
    if (pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Comm_rank(pSPARC->dmcomm, &dmrank);
    }
    if (dmrank != 0) {
        for (int i = 0; i < 9; i++) 
            pSPARC->desc_HMp_BLCYC_1blacs[i] = -1;
    } else {
        for (int i = 0; i < 9; i++) 
            pSPARC->desc_HMp_BLCYC_1blacs[i] = pSPARC->desc_Hp_BLCYC[i];
    }

#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    pSPARC->ChebComp = (CHEBCOMP*) calloc(pSPARC->Nspin_spincomm, sizeof(CHEBCOMP));
    for (int i = 0; i < pSPARC->Nspin_spincomm; i++){
        init_CHEBCOMP(pSPARC->ChebComp+i, pSPARC->sq3_npl, pSPARC->Nstates, pSPARC->cmc_cols);
    }
}


#ifdef USE_DP_SUBEIG
/**
 * @brief   Distribute projected Hamiltonian and Density matrix
 */
void DP_Dist2SQ3(SPARC_OBJ *pSPARC)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank_kptcomm, ONE = 1;
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

    if (DP_CheFSI != NULL)
        pSPARC->Hp_cmc = DP_CheFSI->Hp_local;

    pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp_cmc, &ONE, &ONE, 
              pSPARC->desc_Hp_cmc, pSPARC->Hp_SQ3, &ONE, &ONE, pSPARC->desc_Hp_SQ3, 
              &pSPARC->ictxt_kptcomm);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}

#else
/**
 * @brief   Distribute projected Hamiltonian and Density matrix
 */
void Dist2SQ3(SPARC_OBJ *pSPARC)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int ONE  = 1;
    pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp, &ONE, &ONE, pSPARC->desc_HMp_BLCYC_1blacs,
         pSPARC->Hp_SQ3, &ONE, &ONE, pSPARC->desc_Hp_SQ3, &pSPARC->ictxt_kptcomm);
    pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp, &ONE, &ONE, pSPARC->desc_HMp_BLCYC_1blacs,
         pSPARC->Hp_cmc, &ONE, &ONE, pSPARC->desc_Hp_cmc, &pSPARC->ictxt_kptcomm);
    
    if (pSPARC->Dscomm != MPI_COMM_NULL)
        MPI_Bcast(pSPARC->Hp_cmc, pSPARC->Nstates*pSPARC->Nstates, MPI_DOUBLE, 0, pSPARC->Dscomm);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)    
}
#endif 

/***
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the dense projected Hamiltonian (Hp)
 */
void Lanczos_dense(const double *A, double *lmin, double *lmax, const int *descA, 
    const int mV, const int nV, MPI_Comm comm, double tol, int maxit)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, i, k;
    int descv[9], ONE = 1;
    double *vkm1, *vk, *vkp1, *a, *b, *L, *M, *d, *e, D[2];
    double alpha = 1.0, beta = 0.0, norm, scalar, DL, DM, eig[2];

    MPI_Comm_rank(comm, &rank);
    srand(rank+1);

    for (i = 0; i < 9; i++){
        descv[i] = descA[i];
    }
    descv[3] = 1;
    int m = descA[2];
    vkm1 = (double*) malloc(mV*nV*sizeof(double));
    vk   = (double*) malloc(mV*nV*sizeof(double));
    vkp1 = (double*) malloc(mV*nV*sizeof(double));
    a    = (double*) malloc(maxit*sizeof(double));
    b    = (double*) malloc(maxit*sizeof(double));
    d    = (double*) malloc(maxit*sizeof(double));
    e    = (double*) malloc(maxit*sizeof(double));
    L    = (double*) malloc(maxit*sizeof(double));
    M    = (double*) malloc(maxit*sizeof(double));

    for (i = 0; i < mV*nV; i++){
        vkm1[i] = ((double)rand())/RAND_MAX;
    }
    pdnrm2_(&m, &norm, vkm1, &ONE, &ONE, descv, &ONE);
    scalar = 1.0/ norm;
    pdscal_(&m, &scalar, vkm1, &ONE, &ONE, descv, &ONE);
    pdgemv_("N", &m, &m, &alpha, A, &ONE, &ONE, descA, vkm1, &ONE, &ONE, 
            descv, &ONE, &beta, vk, &ONE, &ONE, descv, &ONE);   // vk = A*vkm1
    pddot_(&m, a, vkm1, &ONE, &ONE, descv, &ONE, vk, &ONE, &ONE, descv, &ONE);  // a(1) = vkm1'*vk  
    scalar = -1.0 * a[0];
    pdaxpy_(&m, &scalar, vkm1, &ONE, &ONE, descv, &ONE, vk, &ONE, &ONE, descv, &ONE);   // vk = vk - a(1)*vkm1 
    pdnrm2_(&m, b, vk, &ONE, &ONE, descv, &ONE);    // b(1) = norm(vk)
    scalar = 1.0 / b[0];
    pdscal_(&m, &scalar, vk, &ONE, &ONE, descv, &ONE);    // vk = vk/b(1)

    k = 1;
    DL = DM = 1.0;
    L[0] = M[0] = 0.0;
    while ((DL > tol || DM > tol) && k <= maxit) {
        pdgemv_("N", &m, &m, &alpha, A, &ONE, &ONE, descA, vk, &ONE, &ONE, 
                descv, &ONE, &beta, vkp1, &ONE, &ONE, descv, &ONE);   // vkp1 = A*vk 
        pddot_(&m, a+k, vk, &ONE, &ONE, descv, &ONE, vkp1, &ONE, &ONE, descv, &ONE);    // a(k+1) = transpose(vk)*vkp1  
        scalar = -1.0 * a[k];
        pdaxpy_(&m, &scalar, vk, &ONE, &ONE, descv, &ONE, vkp1, &ONE, &ONE, descv, &ONE);   //  vkp1 = vkp1 - a(k+1)*vk
        scalar = -1.0 * b[k-1];
        pdaxpy_(&m, &scalar, vkm1, &ONE, &ONE, descv, &ONE, vkp1, &ONE, &ONE, descv, &ONE);   //  vkp1 = vkp1 - b(k)*vkm1
        pdcopy_(&m, vk, &ONE, &ONE, descv, &ONE, vkm1, &ONE, &ONE, descv, &ONE);    // vkm1 = vk ;     
        pdnrm2_(&m, b+k, vkp1, &ONE, &ONE, descv, &ONE);  // b(k+1) = norm(vkp1)
        memset(vk, 0, sizeof(double)*mV*nV);
        scalar = 1.0/b[k];
        pdaxpy_(&m, &scalar, vkp1, &ONE, &ONE, descv, &ONE, vk, &ONE, &ONE, descv, &ONE); // vk = vkp1/b(k+1)
        for (i = 0; i < k+1; i++){
            d[i] = a[i];
            e[i] = b[i];
        }
        if (!LAPACKE_dsterf(k+1, d, e)) {
            M[k] = d[0];
            L[k] = d[k];
        } else {
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }
        D[0] = fabs(M[k] - M[k-1]);
        D[1] = fabs(L[k] - L[k-1]);        
        MPI_Allreduce(MPI_IN_PLACE, D, 2, MPI_DOUBLE, MPI_MAX, comm);        
        DL = D[0];
        DM = D[1];
        k++;
    }

    if(k > maxit){
        if(rank==0) printf("Lanczos achieves max iteration and doesn't converge\n");
        exit(255);
    }
    eig[0] = M[k-1];
    eig[1] = L[k-1];    
    MPI_Bcast(eig, 2, MPI_DOUBLE, 0, comm);
    *lmin = eig[0];
    *lmax = eig[1];
    
    free(vkm1);
    free(vk);
    free(vkp1);
    free(a);
    free(b);
    free(d);
    free(e);
    free(L);
    free(M);

#endif //(#if defined(USE_MKL) || defined(USE_SCALAPACK))    
}

/**
 * @brief   Initialize Chebyshev components
 */
void init_CHEBCOMP(CHEBCOMP *cc, const int sq3_npl, const int m, const int n)
{
    int i;

    cc->sq3_npl = sq3_npl;
    cc->Ti = (double **) calloc(sq3_npl+1, sizeof(double*));
    for (i = 0; i <= sq3_npl; i++){
        cc->Ti[i] = (double *) calloc(m*n, sizeof(double));
    }
    cc->tr_Ti = (double *) calloc(sq3_npl+1, sizeof(double));  
}

/**
 * @brief   Compute Chebyshev matrix vector components
 */
void Chebyshev_matvec_comp(SPARC_OBJ *pSPARC, CHEBCOMP *cc, const int sq3_npl, const double *A, const double a, const double b)
{
#define Y(i,j) Y[(i) + (j) * m]
#define A(i,j) A[(i) + (j) * m]
#define Xnew(i,j) Xnew[(i) + (j) * m]
#define Ynew(i,j) Ynew[(i) + (j) * m]

    if (pSPARC->Dscomm == MPI_COMM_NULL) return;

    int i, j, ii, len, m, n, rank, shift, count_dgemm;
    double a0, e, c, sigma, sigma1, sigma2, gamma;
    double alpha;
    double *Y, *Ynew, *Xnew;
    double t0, t1, tdgemm;

    MPI_Comm_rank(pSPARC->Dscomm, &rank);
    
    m = pSPARC->Nstates;
    n = pSPARC->cmc_cols;
    len = m * n;
    shift = pSPARC->size_Dscomm;
    
    a0 = a;
    e = (b - a)/2;
    c = (b + a)/2;
    sigma = e/(c - a0); 
    sigma1 = sigma;
    gamma = 2/sigma1;
    count_dgemm = 0;
    tdgemm = 0.0;

    for (i = 0; i < sq3_npl+1; i++)
        cc->tr_Ti[i] = 0.0;

    for (i = 0; i < n; i++)
        cc->Ti[0][i * m + rank + i * shift] = 1.0;

    cc->tr_Ti[0] = pSPARC->cmc_cols;

    if (sq3_npl < 1) return;

    Y    = (double *) calloc(len, sizeof(double));
    Ynew = (double *) calloc(len, sizeof(double));
    Xnew = (double *) calloc(len, sizeof(double));

    alpha = sigma1/e;
    for (j = 0; j < n; j++)
        for (i = 0; i < m; i++)
            Y(i,j) = alpha * A(i, rank + j *shift);
    
    alpha *= (-1*c);
    for (j = 0; j < n; j++)
        Y(rank + j * shift, j) += alpha;

    for (i = 0; i < len; i++)
        cc->Ti[1][i] = Y[i];

    for (j = 0; j < n; j++){
        Xnew(rank + j * shift, j) = 1.0;
        cc->tr_Ti[1] +=  Y(rank + j * shift, j);
    }

    ii = 2;
    while (ii <= sq3_npl) {
        sigma2 = 1/(gamma - sigma);
        memset(Ynew, 0, len*sizeof(double));

        alpha = 2*sigma2/e;
        t0 = MPI_Wtime();   
        if (n > 1){
            // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m, alpha, A, m, Y, m, 0, Ynew, m); // Ynew = (2*sigma1/e)*A*Y
            cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, m, n, alpha, A, m, Y, m, 0, Ynew, m);
        } else {
            // cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, alpha, A, m, Y, 1, 0.0, Ynew, 1); // Ynew = (2*sigma1/e)*A*Y
            cblas_dsymv(CblasColMajor, CblasUpper, m, alpha, A, m, Y, 1, 0.0, Ynew, 1);
        }

        t1 = MPI_Wtime();
        tdgemm += (t1 - t0);
        count_dgemm ++;
        alpha *= (-1*c);
        for (i = 0; i < len; i++)
            Ynew[i] += alpha * Y[i];

        alpha = -1*sigma*sigma2;
        for (i = 0; i < len; i++){
            Ynew[i] += alpha * Xnew[i];
            cc->Ti[ii][i] = Ynew[i];
        }
        for (j = 0; j < n; j++)
            cc->tr_Ti[ii] +=  Ynew(rank + j * shift, j);

        for (i = 0; i < len; i++){
            Xnew[i] = Y[i];
            Y[i] = Ynew[i];
        }
        sigma = sigma2;
        ii++;
    }

    free(Y);
    free(Ynew);
    free(Xnew);
#undef Y
#undef A
#undef Xnew
#undef Ynew

#ifdef DEBUG
    if(!rank) printf("Matrix size: %d x %d, vector size: %d x %d.\ncalling dgemm %d times took %.3fms, average time: %.3fms\n",
                                m, m, m, n, count_dgemm, tdgemm*1e3, tdgemm/count_dgemm*1e3);
#endif  
}

/**
 * @brief   Chebyshev coefficients
 */
void ChebyshevCoeff(double *Ci, const int npl, double (*fun)(double, double, double, int), 
    const double a, const double b, const double beta, const double lambda_f, const int type)
{
    int i, j, rank;
    double c, e, fac1, fac2, sum;
    double *y, *d;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    c = (b+a)/2;
    e = (b-a)/2;
    y = (double*) calloc(npl+1, sizeof(double));
    d = (double*) calloc(npl+1, sizeof(double));
    fac1 = M_PI/(npl+1);
    for (i = 0; i < npl+1; i++){
        y[i] = cos(fac1 * (i + 0.5));
        d[i] = fun(beta, y[i]*e+c, lambda_f, type);
    }
    fac2 = 2.0/(npl+1);
    for (i = 0; i < npl+1; i++){
        sum = 0.0;
        for (j = 0; j < npl+1; j++){
            sum += d[j] * cos(fac1 * i * (j + 0.5));
        }
        Ci[i] = fac2 * sum;
    }
    Ci[0] = Ci[0]/2.0;

    free(y);
    free(d);
}

/**
 * @brief   return occupation provided lambda_f when using SQ3 method
 */
double occ_constraint_SQ3(SPARC_OBJ *pSPARC, double lambda_f)
{
    if (pSPARC->kptcomm_index < 0) return 0.0;
    // TODO: confirm whether to use number of electrons or NegCharge
    double g_cmc = 0.0, Ne = pSPARC->NegCharge, *Ci; 
    Ci = (double *) calloc(pSPARC->sq3_npl+1, sizeof(double));
    CHEBCOMP *cc_cmc;

    if (pSPARC->isGammaPoint) { // for gamma-point systems
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            cc_cmc = pSPARC->ChebComp + spn_i;
            ChebyshevCoeff(Ci, pSPARC->sq3_npl, smearing_function, cc_cmc->eigmin, cc_cmc->eigmax, pSPARC->Beta, lambda_f, pSPARC->elec_T_type);

            for (int i = 0; i < pSPARC->sq3_npl+1; i++){
                g_cmc += (2.0/pSPARC->Nspin) * cc_cmc->tr_Ti[i] * Ci[i];
            }
        }
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find g
            MPI_Allreduce(MPI_IN_PLACE, &g_cmc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }
    } else { // for k-points
        exit(-1);
    }

    free(Ci);
    return g_cmc + Ne;
    // return g - pSPARC->Nelectron; // this will work even when Ns = Nelectron/2
}

/**
 * @brief   Compute Density matrix 
 */
void SubDensMat(SPARC_OBJ *pSPARC, double *Ds, const double lambda_f, CHEBCOMP *cc)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int i, j, npl, len, rank, ONE = 1, spn_i, nproc_dmcomm;
    double *Ci, t0, t1, t2;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->dmcomm != MPI_COMM_NULL)
        MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    t0 = MPI_Wtime();
    npl = cc->sq3_npl;
    len = pSPARC->Nstates * pSPARC->cmc_cols;

    Ci     = (double *) calloc(npl+1, sizeof(double)); 
    memset(Ds, 0, sizeof(double)*len);
    
    // for loop over spin
    for (spn_i = 0; spn_i< pSPARC->Nspin_spincomm; spn_i++){
        ChebyshevCoeff(Ci, npl, smearing_function, cc[spn_i].eigmin, cc[spn_i].eigmax, pSPARC->Beta, lambda_f, pSPARC->elec_T_type);

        for (i = 0; i <= npl; i++)
            for (j = 0; j < len; j++)
                Ds[j] += Ci[i] * cc->Ti[i][j]; 
    }

    if (pSPARC->npspin != 1) { 
        MPI_Allreduce(MPI_IN_PLACE, &Ds, len, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    free(Ci);
    t1 = MPI_Wtime();

    #ifdef USE_DP_SUBEIG
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    double *Ds_local = NULL;
    if (DP_CheFSI != NULL)
        Ds_local = DP_CheFSI->eig_vecs;

    pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, Ds, &ONE, &ONE, pSPARC->desc_Ds,
         Ds_local, &ONE, &ONE, pSPARC->desc_Hp_cmc, &pSPARC->ictxt_kptcomm);
    
    if (DP_CheFSI != NULL){
        int Ns_dp = DP_CheFSI->Ns_dp;
        MPI_Bcast(DP_CheFSI->eig_vecs, Ns_dp * Ns_dp, MPI_DOUBLE, 0, DP_CheFSI->kpt_comm);
    }
    #else
    pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, Ds, &ONE, &ONE, pSPARC->desc_Ds,
         pSPARC->Mp, &ONE, &ONE, pSPARC->desc_HMp_BLCYC_1blacs, &pSPARC->ictxt_kptcomm);
    #endif

    if (nproc_dmcomm > 1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(pSPARC->Mp, pSPARC->nr_Mp_BLCYC*pSPARC->nc_Mp_BLCYC, MPI_DOUBLE, 0, pSPARC->dmcomm);
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank) {
        printf("Calculating Density matrix: %.3fms\n", (t1-t0)*1e3);
        printf("Redistribute Density matrix: %.3fms\n", (t2-t1)*1e3);
    }
    #endif
    (void)t0; (void)t1; (void)t2; // timers are only printed in DEBUG mode
#endif //(#if defined(USE_MKL) || defined(USE_SCALAPACK))
}

/**
 * @brief   Compute rho given density matrix
 */
void update_rho_sq3(SPARC_OBJ *pSPARC, double *rho)
{
    int Nd = pSPARC->Nd_d_dmcomm;
    int Ns = pSPARC->Nstates;
    int nstart = pSPARC->band_start_indx;
    int nend = pSPARC->band_end_indx;	

#if defined(USE_MKL) || defined(USE_SCALAPACK)

    int i, n, ONE = 1, count, rank=0;
    double alpha = 1.0, beta = 0.0;
    double t0=0, t1=0, t2=0, t3=0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    t0 = MPI_Wtime();

#ifdef USE_DP_SUBEIG
    DP_Subspace_Rotation(pSPARC, pSPARC->Yorb); 
    t2 = MPI_Wtime();

#else

    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {            
        pSPARC->Xorb_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));
        pSPARC->Yorb_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));        
        assert(pSPARC->Xorb_BLCYC != NULL && pSPARC->Yorb_BLCYC != NULL);

        pdgemr2d_(&Nd, &Ns, pSPARC->Xorb, &ONE, &ONE, pSPARC->desc_orbitals,
            pSPARC->Xorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, &pSPARC->ictxt_blacs); 
    } else {
        pSPARC->Xorb_BLCYC = pSPARC->Xorb;
        pSPARC->Yorb_BLCYC = pSPARC->Yorb;
    }

    t2 = MPI_Wtime();
    // Yorb = Xorb * Ds
    pdgemm_("N", "N", &Nd, &Ns, &Ns, &alpha, pSPARC->Xorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, 
        pSPARC->Mp, &ONE, &ONE, pSPARC->desc_Mp_BLCYC, &beta, pSPARC->Yorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC);
    t3 = MPI_Wtime();

    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {        
        // update Yorb    
        pdgemr2d_(&Nd, &Ns, pSPARC->Yorb_BLCYC, &ONE, &ONE, 
            pSPARC->desc_orb_BLCYC, pSPARC->Yorb, &ONE, &ONE, 
            pSPARC->desc_orbitals, &pSPARC->ictxt_blacs);
        free(pSPARC->Xorb_BLCYC); pSPARC->Xorb_BLCYC = NULL;
        free(pSPARC->Yorb_BLCYC); pSPARC->Yorb_BLCYC = NULL;
    }

#endif // USE_DP_SUBEIG

    count = 0;
    for (n = nstart; n <= nend; n++) {
        for (i = 0; i < Nd; i++, count++) {
            rho[i] += 2 * pSPARC->Yorb[count] * pSPARC->Xorb[count];
        }
    }
    t1 = MPI_Wtime();
    
    
#ifdef DEBUG
    if(!rank) {
        #ifndef USE_DP_SUBEIG
        printf("copying orbitals psi into BLCYC format: %.3f ms\n", (t2-t0)*1e3);
        printf("psi * Ds by pdgemm used: %.3f ms\n", (t3-t2)*1e3);
        printf("copying result back into block format and computing rho: %.3f ms\n", (t1-t3)*1e3);
        #else
        printf("psi * Ds by DP used: %.3f ms\n", (t2-t0)*1e3);
        printf("Updating rho used: %3f ms\n", (t1-t2)*1e3);
        #endif
        printf("Total time for updating rho and psi: %.3f ms\n", (t1-t0)*1e3);

    }
#endif //DEBUG
(void)t0;
(void)t1;
(void)t2;
(void)t3;
(void)beta;
(void)alpha;
(void)ONE;
(void)Ns;
(void)rank;
#endif //(#if defined(USE_MKL) || defined(USE_SCALAPACK))  
}


/**
 * @brief   Entropy using different smearing function
 */
double Eent_func(const double beta, const double lambda, const double lambda_f, const int type)
{
    double occ, v;

    occ = smearing_function(beta, lambda, lambda_f, type);
    if (type == 0){
        if (fabs(occ)<0.01*TEMP_TOL || fabs(occ-1.0)<0.01*TEMP_TOL)
            v = 0;
        else
            v = (occ*log(occ) + (1-occ)*log(1-occ));
    } else {
        printf("Entropy not implemented for Gaussian smearing\n");
        exit(-1);
    }

    return v;
}

/**
 * @brief   Uband function 
 */
double Uband_func(const double beta, const double lambda, const double lambda_f, const int type)
{
    return (lambda * smearing_function(beta, lambda, lambda_f, type));
}

/**
 * @brief   Band energy using SQ3 method (Density matrix)
 */
double Calculate_Eband_SQ3(SPARC_OBJ *pSPARC, CHEBCOMP *cc)
{
    int sq3_npl, spn_i, i;
    double Eband, *Ci;

    sq3_npl = cc->sq3_npl;
    Eband = 0.0;
    Ci = (double*) calloc(sq3_npl+1, sizeof(double));
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++){
        ChebyshevCoeff(Ci, sq3_npl, Uband_func, cc[spn_i].eigmin, cc[spn_i].eigmax, pSPARC->Beta, pSPARC->Efermi, pSPARC->elec_T_type);
        for (i = 0; i < sq3_npl+1; i++)
            Eband += (2.0/pSPARC->Nspin) * Ci[i] * cc[spn_i].tr_Ti[i];
    }

    free(Ci);
    return Eband;
}

/**
 * @brief   Calculate electronic Entropy using SQ3 method (Density matrix)
 */
double Calculate_electronicEntropy_SQ3(SPARC_OBJ *pSPARC, CHEBCOMP *cc)
{
    int sq3_npl, spn_i, i;
    double Eent, *Ci;

    sq3_npl = cc->sq3_npl;
    Eent = 0.0;
    Ci = (double*) calloc(sq3_npl+1, sizeof(double));
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++){
        ChebyshevCoeff(Ci, sq3_npl, Eent_func, cc[spn_i].eigmin, cc[spn_i].eigmax, pSPARC->Beta, pSPARC->Efermi, pSPARC->elec_T_type);
        for (i = 0; i < sq3_npl+1; i++)
            Eent += Ci[i] * cc[spn_i].tr_Ti[i];
    }
    Eent *= (2.0/pSPARC->Nspin) / pSPARC->Beta;

    free(Ci);
    return Eent;
}

/**
 * @brief   Free memory space of Chebyshev components
 */
void free_ChemComp(SPARC_OBJ *pSPARC)
{
    int i, j, npl = pSPARC->sq3_npl;

    for (i = 0; i < pSPARC->Nspin_spincomm; i++){
        for (j = 0; j <= npl; j++){
            free(pSPARC->ChebComp[i].Ti[j]);
        }
        free(pSPARC->ChebComp[i].Ti);
        free(pSPARC->ChebComp[i].tr_Ti);
    }
    free(pSPARC->ChebComp);
}


/**
 * @brief   Free memory space and communicators for SQ3.
 */
void free_SQ3(SPARC_OBJ *pSPARC)
{
    free_ChemComp(pSPARC);
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->kptcomm_index != -1){
        Cfree_blacs_system_handle(pSPARC->bhandle_kptcomm);
        if (pSPARC->ictxt_kptcomm > -1)
            Cblacs_gridexit(pSPARC->ictxt_kptcomm);
        
        Cfree_blacs_system_handle(pSPARC->bhandle_SQ3);
        if (pSPARC->ictxt_SQ3 > -1)
            Cblacs_gridexit(pSPARC->ictxt_SQ3);
        
        Cfree_blacs_system_handle(pSPARC->bhandle_cmc);
        if (pSPARC->ictxt_cmc > -1)
            Cblacs_gridexit(pSPARC->ictxt_cmc);

        Cfree_blacs_system_handle(pSPARC->bhandle_Ds);
        if (pSPARC->ictxt_Ds > -1)
            Cblacs_gridexit(pSPARC->ictxt_Ds);
    }
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    free(pSPARC->Hp_SQ3);
    if (pSPARC->SQ3comm != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->SQ3comm);   

    if (pSPARC->Dscomm != MPI_COMM_NULL){
        MPI_Comm_free(&pSPARC->Dscomm);  
        #ifndef USE_DP_SUBEIG
        free(pSPARC->Hp_cmc);
        #endif
        free(pSPARC->Ds_cmc);
    }
}
