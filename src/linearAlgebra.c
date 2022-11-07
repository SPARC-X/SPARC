#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <complex.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
	#define MKL_Complex16 double complex
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/* ScaLAPACK routines */
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

#include "linearAlgebra.h"
#include "parallelization.h"
#include "tools.h"

#define TEMP_TOL 1e-12
#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))

/**
 * @brief Call the pdsyevx_ routine with an automatic workspace setup.
 *
 *        The original pdsyevx_ routine asks uses to provide the size of the 
 *        workspace. This routinae calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsyevx_ routine to
 *        do the calculation.
 */
void automem_pdsyevx_ ( 
	char *jobz, char *range, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
	double *vl, double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef DEBUG
    double t1, t2;
#endif

	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork, *iwork, liwork, *icluster;
	double *work, *gap;
	lwork = liwork = -1;
	work  = (double *)malloc(100 * sizeof(double));
	gap   = (double *)malloc(nprow * npcol * sizeof(double));
	iwork = (int *)malloc(100 * sizeof(int));
	icluster = (int *)malloc(2 * nprow * npcol * sizeof(int));    

	//** first do a workspace query **//
#ifdef DEBUG    
    t1 = MPI_Wtime();
#endif
	pdsyevx_(jobz, range, uplo, n, a, ia, ja, desca, 
			vl, vu, il, iu, abstol, m, nz, w, 
			orfac, z, iz, jz, descz, work, &lwork, iwork, 
			&liwork, ifail, icluster, gap, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %d, work(1) = %f, time for "
                        "workspace query: %.3f ms\n", 
                        grank, work[0], (t2 - t1)*1e3);
#endif

	int NNP, NN, NP0, MQ0, NB, N = *n;
	lwork = (int) fabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );
	NNP = max(max(N,4), nprow * npcol+1);

	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
	if (fabs(*orfac) < TEMP_TOL) lwork += 200000;   // TODO: Additioal 1.5Mb, to be optimized
	if (fabs(*orfac) > TEMP_TOL) lwork += max(N*N, min(10*lwork,2000000)); // for safety
	work = realloc(work, lwork * sizeof(double));

	liwork = iwork[0];
	liwork = max(liwork, 6 * NNP);
	// liwork += max(N*N, min(20*liwork, 200000)); // for safety
	iwork = realloc(iwork, liwork * sizeof(int));

	// call the routine again to perform the calculation
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsyevx_(jobz, range, uplo, n, a, ia, ja, desca, 
			vl, vu, il, iu, abstol, m, nz, w, 
			orfac, z, iz, jz, descz, work, &lwork, iwork, 
			&liwork, ifail, icluster, gap, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!grank) {
    printf("rank = %d, info = %d, time for solving standard "
            "eigenproblem in %d x %d process grid: %.3f ms\n", 
            grank, *info, nprow, npcol, (t2 - t1)*1e3);
    printf("rank = %d, after calling pdsyevx, Nstates = %d\n", grank, *n);
}
#endif

	free(work);
	free(gap);
	free(iwork);
	free(icluster);
#endif // (USE_MKL or USE_SCALAPACK)	
}


/**
 * @brief Call the pdsyevx_ routine with an automatic workspace setup.
 *
 *        The original pdsyevx_ routine asks uses to provide the size of the 
 *        workspace. This routinae calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsyevx_ routine to
 *        do the calculation.
 */
void automem_pdsyev_ ( 
	char *jobz, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
    double *w, double *z, int *iz, int *jz, int *descz, int *info)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef DEBUG
    double t1, t2;
#endif
	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork, liwork;
	double *work;
	lwork = liwork = -1;
	work  = (double *)malloc(100 * sizeof(double));

	//** first do a workspace query **//
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsyev_(jobz, uplo, n, a, ia, ja, desca, 
			w, z, iz, jz, descz, work, &lwork, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %d, work(1) = %f, time for "
                        "workspace query: %.3f ms\n", 
                        grank, work[0], (t2 - t1)*1e3);
#endif

	int NN, NP0, MQ0, NB, N = *n;
	lwork = (int) fabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );

	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
    // TODO: increase lwork 
    /**
     * The ScaLAPACK routine for estimating memory might not work well for 
     * the case where only few processes contain the whole matrix and most 
     * processes do not. i.e., where the data are concentrated.
     */
    //lwork += max(N*N, min(10*lwork,2000000));
    lwork += 10*N; // for safety
	work = realloc(work, lwork * sizeof(double));

    /** ScaLAPACK might fail when the the matrix is distributed only on 
     *  few processes and most process contains empty local matrices. 
     *  consider using a subgroup of the processors to solve the 
     *  eigenvalue problem and then re-distribute.
     */
	// call the routine again to perform the calculation
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsyev_(jobz, uplo, n, a, ia, ja, desca, 
			w, z, iz, jz, descz, work, &lwork, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!grank) {
    printf("rank = %d, info = %d, time for solving standard "
            "eigenproblem in %d x %d process grid: %.3f ms\n", 
            grank, *info, nprow, npcol, (t2 - t1)*1e3);
    printf("rank = %d, after calling pdsyev, Nstates = %d\n", grank, *n);
}
#endif

	free(work);
#endif // (USE_MKL or USE_SCALAPACK)	
}


/**
 * @brief Call the pdsygvx_ routine with an automatic workspace setup.
 *
 *        The original pdsygvx_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsygvx_ routine to
 *        do the calculation.
 */
void automem_pdsygvx_ ( 
	int *ibtype, char *jobz, char *range, char *uplo, int *n, double *a, int *ia,
	int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef DEBUG
    double t1, t2;
#endif

	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork, *iwork, liwork, *icluster;
	double *work, *gap;
	lwork = liwork = -1;
	work  = (double *)malloc(100 * sizeof(double));
	gap   = (double *)malloc(nprow * npcol * sizeof(double));
	iwork = (int *)malloc(100 * sizeof(int));
	icluster = (int *)malloc(2 * nprow * npcol * sizeof(int));    

	//** first do a workspace query **//
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, 
		 desca, b, ib, jb, descb, vl, 
		 vu, il, iu, abstol, m, nz, w, 
		 orfac, z, iz, jz, descz, 
		 work, &lwork, iwork, &liwork, 
		 ifail, icluster, gap, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %d, work(1) = %f, time for "
                        "workspace query: %.3f ms\n", 
                        grank, work[0], (t2 - t1)*1e3);
#endif

	// Warning: pdsygvx requires the block sizes in both row and col 
	//          dimension to be the same!

	int NNP, NN, NP0, MQ0, NB, N = *n;
	lwork = (int) fabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );
	NNP = max(max(N,4), nprow * npcol+1);

	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
	if (fabs(*orfac) < TEMP_TOL) lwork += 200000;   // TODO: Additioal 1.5Mb, to be optimized
	if (fabs(*orfac) > TEMP_TOL) lwork += max(N*N, min(10*lwork,2000000)); // for safety
	work = realloc(work, lwork * sizeof(double));

	liwork = iwork[0];
	liwork = max(liwork, 6 * NNP);
	// liwork += max(N*N, min(20*liwork, 200000)); // for safety
	iwork = realloc(iwork, liwork * sizeof(int));

	// call the routine again to perform the calculation
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsygvx_(ibtype, jobz, range, uplo, n, a, ia, ja, 
		 desca, b, ib, jb, descb, vl, 
		 vu, il, iu, abstol, m, nz, w, 
		 orfac, z, iz, jz, descz, 
		 work, &lwork, iwork, &liwork, 
		 ifail, icluster, gap, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!grank) {
    printf("rank = %d, info = %d, time for solving generalized "
            "eigenproblem in %d x %d process grid: %.3f ms\n", 
            grank, *info, nprow, npcol, (t2 - t1)*1e3);
    printf("rank = %d, after calling pdsygvx, Nstates = %d\n", grank, *n);
}
#endif

	free(work);
	free(gap);
	free(iwork);
	free(icluster);
#endif // (USE_MKL or USE_SCALAPACK)	
}


/**
 * @brief Call the pdsygvx_ routine with an automatic workspace setup.
 *
 *        The original pdsygvx_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsygvx_ routine to
 *        do the calculation.
 */
void automem_pzhegvx_ ( 
	int *ibtype, char *jobz, char *range, char *uplo, int *n, double complex *a, int *ia,
	int *ja, int *desca, double complex *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double complex *z, int *iz, int *jz, int *descz, int *ifail, int *info)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef DEBUG
    double t1, t2;
#endif
	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork, lrwork, *iwork, liwork, *icluster;
	double complex *work;
	double *rwork, *gap;

	lwork = lrwork = liwork = -1;
	work  = (double complex *)malloc(100 * sizeof(double complex));
	rwork  = (double *)malloc(100 * sizeof(double));
	gap   = (double *)malloc(nprow * npcol * sizeof(double));
	iwork = (int *)malloc(100 * sizeof(int));
	icluster = (int *)malloc(2 * nprow * npcol * sizeof(int));

	//** first do a workspace query **//
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pzhegvx_(ibtype, jobz, range, uplo, n, a, ia, ja, 
		 desca, b, ib, jb, descb, vl, 
		 vu, il, iu, abstol, m, nz, w, 
		 orfac, z, iz, jz, descz, 
		 work, &lwork, rwork, &lrwork, iwork, 
		 &liwork, ifail, icluster, gap, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %d, work(1) = %f+i%f, time for "
                        "workspace query: %.3f ms\n", 
                        grank, creal(work[0]), cimag(work[0]), (t2 - t1)*1e3);
#endif
	// Warning: pdsygvx requires the block sizes in both row and col 
	//          dimension to be the same!

	int NB, NN, NP0, MQ0, NNP, N = *n;
	lwork = (int) cabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );
	lwork = max(lwork, N + (NP0 + MQ0 + NB) * NB);
	// TODO: increase lwork 
	/**
	 * The ScaLAPACK routine for estimating memory might not work well for 
	 * the case where only few processes contain the whole matrix and most 
	 * processes do not. i.e., where the data are concentrated.
	 */
	//lwork += min(10*lwork,2000000); // TODO: for safety, to be optimized
	if (fabs(*orfac) < TEMP_TOL) lwork += 200000;   // TODO: Additioal 1.5Mb, to be optimized
	if (fabs(*orfac) > TEMP_TOL) lwork += max(N*N, min(10*lwork,2000000));
	work = realloc(work, lwork * sizeof(double complex));
	
	lrwork = (int) fabs(rwork[0]);
	lrwork = max(lrwork, 4 * N + max(5 * NN, NP0 * MQ0) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
	lrwork += max(N*N, min(10*lrwork,2000000));                  
	rwork = realloc(rwork, lrwork * sizeof(double));
	
	liwork = iwork[0];
	NNP = max(max(N,4), nprow * npcol+1);
	liwork = max(liwork, 6 * NNP);

	//liwork += min(20*liwork, 200000); // TODO: for safety, to be optimized
	//liwork += max(N*N, min(20*liwork, 200000));
	iwork = realloc(iwork, liwork * sizeof(int));

	/** ScaLAPACK might fail when the the matrix is distributed only on 
	 *  few processes and most process contains empty local matrices. 
	 *  consider using a subgroup of the processors to solve the 
	 *  eigenvalue problem and then re-distribute.
	 */

	// call the routine again to perform the calculation
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pzhegvx_(ibtype, jobz, range, uplo, n, a, ia, ja, 
		 desca, b, ib, jb, descb, vl, 
		 vu, il, iu, abstol, m, nz, w, 
		 orfac, z, iz, jz, descz, 
		 work, &lwork, rwork, &lrwork, iwork, 
		 &liwork, ifail, icluster, gap, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!grank) {
    printf("rank = %d, info = %d, time for solving standard "
            "eigenproblem in %d x %d process grid: %.3f ms\n", 
            grank, *info, nprow, npcol, (t2 - t1)*1e3);
    printf("rank = %d, after calling pzhegvx, Nstates = %d\n", grank, *n);
}
#endif

	free(work);
	free(rwork);
	free(gap);
	free(iwork);
	free(icluster);
#endif // (USE_MKL or USE_SCALAPACK)	
}


/**
 * @brief Call the pdsyevd_ routine with an automatic workspace setup.
 *
 *        The original pdsyevd_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsyevd_ routine to
 *        do the calculation.
 */
void automem_pdsyevd_ ( 
	char *jobz, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
	double *w, double *z, int *iz, int *jz, int *descz, int *info)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef DEBUG
    double t1, t2;
#endif

	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork, *iwork, liwork;
	double *work;
	lwork = liwork = -1;
	work  = (double *)malloc(100 * sizeof(double));
	iwork = (int *)malloc(100 * sizeof(int)); 

	//** first do a workspace query **//
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, 
		work, &lwork, iwork, &liwork, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %d, work(1) = %f, time for "
                        "workspace query: %.3f ms\n", 
                        grank, work[0], (t2 - t1)*1e3);
#endif

	int NNP, NN, NP0, MQ0, NB, N = *n;
	lwork = (int) fabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );
	NNP = max(max(N,4), nprow * npcol+1);

	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
	//lwork += max(N*N, min(10*lwork,2000000)); // for safety
	lwork += 10*N; // for safety
	work = realloc(work, lwork * sizeof(double));

	liwork = iwork[0];
	liwork = max(liwork, 6 * NNP);
	//liwork += max(N*N, min(20*liwork, 200000)); // for safety
	iwork = realloc(iwork, liwork * sizeof(int));

	// call the routine again to perform the calculation
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
	pdsyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, 
		work, &lwork, iwork, &liwork, info);
#ifdef DEBUG
    t2 = MPI_Wtime();
if(!grank) {
    printf("rank = %d, info = %d, time for solving standard "
            "eigenproblem in %d x %d process grid: %.3f ms\n", 
            grank, *info, nprow, npcol, (t2 - t1)*1e3);
    printf("rank = %d, after calling pdsyevd, Nstates = %d\n", grank, *n);
}
#endif

	free(work);
	free(iwork);
#endif // (USE_MKL or USE_SCALAPACK)
}


/**
 * @brief  Solve real generalized eigenvalue problem in a subgrid 
 *         commmunicator with a better grid dimension.
 */
void pdsygvx_subcomm_ (
    int *ibtype, char *jobz, char *range, char *uplo, int *n, double *a, int *ia,
	int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info,
    MPI_Comm comm, int *dims, int blksz)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    if (dims[0] * dims[1] > nproc) {
        if (!rank) printf("ERROR: number of processes in the subgrid (%d, %d) is larger than "
                          "       total number of processors in the provided communicator.\n", dims[0], dims[1]);
        exit(EXIT_FAILURE);
    }
    #ifdef DEBUGSUBGRID
    if (rank == 0) printf("pdsyevx_subcomm_: process grid = (%d, %d)\n", dims[0], dims[1]);
    #endif

    // generate a (subset) process grid within comm
    int bhandle = Csys2blacs_handle(comm); // create a context out of rowcomm
    int ictxt = bhandle, N = *n;
    // create new context with dimensions: dims[0] x dims[1]
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

    // create a global context corresponding to comm
    int bhandle_global = Csys2blacs_handle(comm); // create a context out of rowcomm
    int ictxt_old = bhandle_global;
    // create new context with dimensions: nproc x 1
    Cblacs_gridinit(&ictxt_old, "Row", nproc, 1);

    if (ictxt >= 0) {
        int nprow, npcol, myrow, mycol;
        Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
        // nproc_grid = nprow * npcol;

        // define new BLCYC distribution of A, B and Z
        int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info2,
            descA_BLCYC[9], descB_BLCYC[9], descZ_BLCYC[9];

        mb = nb = blksz;
        m_loc = numroc_(&N, &mb, &myrow, &ZERO, &nprow);
        n_loc = numroc_(&N, &nb, &mycol, &ZERO, &npcol);
        llda = max(1, m_loc);
        descinit_(descA_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        descinit_(descB_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        descinit_(descZ_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        
        double *A_BLCYC  = (double *)calloc(m_loc*n_loc,sizeof(double));
        double *B_BLCYC  = (double *)calloc(m_loc*n_loc,sizeof(double));
        double *Z_BLCYC  = (double *)calloc(m_loc*n_loc,sizeof(double));
        assert(A_BLCYC != NULL && B_BLCYC != NULL && Z_BLCYC != NULL);

        #ifdef DEBUGSUBGRID
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        // convert A from original distribution to BLCYC in the new context
        pdgemr2d_(&N, &N, a, ia, ja, desca, A_BLCYC, &ONE, &ONE,
            descA_BLCYC, &ictxt_old);
        // convert B from original distribution to BLCYC in the new context
        pdgemr2d_(&N, &N, b, ib, jb, descb, B_BLCYC, &ONE, &ONE,
            descB_BLCYC, &ictxt_old);
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pdsygvx_subcomm: A,B -> A_BLCYC,B_BLCYC: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        // if abstol is not provided in advance, use the most orthogonal setting
        if (*abstol < 0) *abstol = pdlamch_(&ictxt, "U");

        automem_pdsygvx_(ibtype, jobz, range, uplo, n, A_BLCYC, &ONE, &ONE, descA_BLCYC, 
			 B_BLCYC, &ONE, &ONE, descB_BLCYC, vl, vu, il, iu, abstol, 
			 m, nz, w, orfac, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, ifail, info);
        
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pdsygvx_subcomm: AZ=ZD: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        // convert Z_BLCYC to given format
        pdgemr2d_(&N, &N, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, z, iz, jz, descz, &ictxt_old);
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pdsygvx_subcomm: Z_BLCYC -> Z: %.3f ms\n", (t2-t1)*1e3);
        #endif

        free(A_BLCYC);
        free(B_BLCYC);
        free(Z_BLCYC);
        Cblacs_gridexit(ictxt);
    } else {
        int i, ONE = 1, descA_BLCYC[9], descB_BLCYC[9], descZ_BLCYC[9];
        double *A_BLCYC, *B_BLCYC, *Z_BLCYC;
        A_BLCYC = B_BLCYC = Z_BLCYC = NULL;
        for (i = 0; i < 9; i++)
            descA_BLCYC[i] = descB_BLCYC[i] = descZ_BLCYC[i] = -1;

        pdgemr2d_(&N, &N, a, ia, ja, desca, A_BLCYC, &ONE, &ONE, descA_BLCYC, &ictxt_old);
        pdgemr2d_(&N, &N, b, ib, jb, descb, B_BLCYC, &ONE, &ONE, descB_BLCYC, &ictxt_old);
        pdgemr2d_(&N, &N, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, z, iz, jz, descz, &ictxt_old);
        *info = 0;
    }
    if (dims[0] * dims[1] < nproc)
        Bcast_eigenvalues(w, N, ictxt, dims[0]*dims[1], comm);
    Cblacs_gridexit(ictxt_old);
#endif // (USE_MKL or USE_SCALAPACK)
}




/**
 * @brief  Solve real standard eigenvalue problem in a subgrid 
 *         commmunicator with a better grid dimension.
 */
void pdsyevx_subcomm_ (
    char *jobz, char *range, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
	double *vl, double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info,
    MPI_Comm comm, int *dims, int blksz)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    if (dims[0] * dims[1] > nproc) {
        if (!rank) printf("ERROR: number of processes in the subgrid (%d, %d) is larger than "
                          "       total number of processors in the provided communicator.\n", dims[0], dims[1]);
        exit(EXIT_FAILURE);
    }
    #ifdef DEBUGSUBGRID
    if (rank == 0) printf("pdsyevx_subcomm_: process grid = (%d, %d)\n", dims[0], dims[1]);
    #endif

    // generate a (subset) process grid within comm
    int bhandle = Csys2blacs_handle(comm); // create a context out of rowcomm
    int ictxt = bhandle, N = *n;
    // create new context with dimensions: dims[0] x dims[1]
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

    // limitations: only correct when a, b, z are in the same context
    int ictxt_old = desca[1];

    if (ictxt >= 0) {
        int nprow, npcol, myrow, mycol;
        Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
        // nproc_grid = nprow * npcol;

        // define new BLCYC distribution of A, B and Z
        int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info2,
            descA_BLCYC[9], descZ_BLCYC[9];

        mb = nb = blksz;
        m_loc = numroc_(&N, &mb, &myrow, &ZERO, &nprow);
        n_loc = numroc_(&N, &nb, &mycol, &ZERO, &npcol);
        llda = max(1, m_loc);
        descinit_(descA_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        descinit_(descZ_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        
        double *A_BLCYC  = (double *)calloc(m_loc*n_loc,sizeof(double));
        double *Z_BLCYC  = (double *)calloc(m_loc*n_loc,sizeof(double));
        assert(A_BLCYC != NULL && Z_BLCYC != NULL);

        #ifdef DEBUGSUBGRID
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        // convert A from original distribution to BLCYC in the new context
        pdgemr2d_(&N, &N, a, ia, ja, desca, A_BLCYC, &ONE, &ONE,
            descA_BLCYC, &ictxt_old);
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pdsyevx_subcomm_: A -> A_BLCYC: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        automem_pdsyevx_ ( 
            jobz, range, uplo, n, A_BLCYC, &ONE, &ONE, descA_BLCYC, 
	        vl, vu, il, iu, abstol, m, nz, w, orfac, Z_BLCYC, 
            &ONE, &ONE, descZ_BLCYC, ifail, info);

        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pdsyevx_subcomm_: AZ=ZD: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        // convert Z_BLCYC to given format
        pdgemr2d_(&N, &N, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, z, iz, jz, descz, &ictxt_old);
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pdsygvx_subcomm: Z_BLCYC -> Z: %.3f ms\n", (t2-t1)*1e3);
        #endif

        free(A_BLCYC);
        free(Z_BLCYC);
        Cblacs_gridexit(ictxt);
    } else {
        int i, ONE = 1, descA_BLCYC[9], descZ_BLCYC[9];
        double *A_BLCYC, *Z_BLCYC;
        A_BLCYC = Z_BLCYC = NULL;
        for (i = 0; i < 9; i++)
            descA_BLCYC[i] = descZ_BLCYC[i] = -1;

        pdgemr2d_(&N, &N, a, ia, ja, desca, A_BLCYC, &ONE, &ONE, descA_BLCYC, &ictxt_old);
        pdgemr2d_(&N, &N, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, z, iz, jz, descz, &ictxt_old);
    }

    if (dims[0] * dims[1] < nproc)
        Bcast_eigenvalues(w, N, ictxt, dims[0]*dims[1], comm);
#endif // (USE_MKL or USE_SCALAPACK)
}



/**
 * @brief  Solve complex generalized eigenvalue problem in a subgrid 
 *         commmunicator with a better grid dimension.
 */
void pzhegvx_subcomm_ (
    int *ibtype, char *jobz, char *range, char *uplo, int *n, double complex *a, int *ia,
	int *ja, int *desca, double complex *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double complex *z, int *iz, int *jz, int *descz, int *ifail, int *info,
    MPI_Comm comm, int *dims, int blksz)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, nproc, grank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);

    if (dims[0] * dims[1] > nproc) {
        if (!rank) printf("ERROR: number of processes in the subgrid (%d, %d) is larger than "
                          "       total number of processors in the provided communicator.\n", dims[0], dims[1]);
        exit(EXIT_FAILURE);
    }
    #ifdef DEBUGSUBGRID
    if (rank == 0) printf("pzhegvx_subcomm_: process grid = (%d, %d)\n", dims[0], dims[1]);
    #endif

    // generate a (subset) process grid within comm
    int bhandle = Csys2blacs_handle(comm); // create a context out of rowcomm
    int ictxt = bhandle, N = *n;
    // create new context with dimensions: dims[0] x dims[1]
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

    // limitations: only correct when a, b, z are in the same context
    int ictxt_old = desca[1];

    if (ictxt >= 0) {
        int nprow, npcol, myrow, mycol;
        Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
        // nproc_grid = nprow * npcol;

        // define new BLCYC distribution of A, B and Z
        int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info2,
            descA_BLCYC[9], descB_BLCYC[9], descZ_BLCYC[9];

        mb = nb = blksz;
        m_loc = numroc_(&N, &mb, &myrow, &ZERO, &nprow);
        n_loc = numroc_(&N, &nb, &mycol, &ZERO, &npcol);
        llda = max(1, m_loc);
        descinit_(descA_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        descinit_(descB_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        descinit_(descZ_BLCYC, &N, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info2);
        assert(info2 == 0);
        
        double complex *A_BLCYC  = (double complex *)calloc(m_loc*n_loc,sizeof(double complex));
        double complex *B_BLCYC  = (double complex *)calloc(m_loc*n_loc,sizeof(double complex));
        double complex *Z_BLCYC  = (double complex *)calloc(m_loc*n_loc,sizeof(double complex));
        assert(A_BLCYC != NULL && B_BLCYC != NULL && Z_BLCYC != NULL);

        #ifdef DEBUGSUBGRID
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        // convert A from original distribution to BLCYC in the new context
        pzgemr2d_(&N, &N, a, ia, ja, desca, A_BLCYC, &ONE, &ONE,
            descA_BLCYC, &ictxt_old);

        // convert B from original distribution to BLCYC in the new context
        pzgemr2d_(&N, &N, b, ib, jb, descb, B_BLCYC, &ONE, &ONE,
            descB_BLCYC, &ictxt_old);
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pzhegvx_subcomm_: A,B -> A_BLCYC,B_BLCYC: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        automem_pzhegvx_(ibtype, jobz, range, uplo, n, A_BLCYC, &ONE, &ONE, descA_BLCYC, 
			 B_BLCYC, &ONE, &ONE, descB_BLCYC, vl, vu, il, iu, abstol, 
			 m, nz, w, orfac, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, ifail, info);
        
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pzhegvx_subcomm_: AZ=ZD: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        // convert Z_BLCYC to given format
        pzgemr2d_(&N, &N, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, z, iz, jz, descz, &ictxt_old);
        #ifdef DEBUGSUBGRID
        t2 = MPI_Wtime();
        if (!rank) printf("pzhegvx_subcomm_: Z_BLCYC -> Z: %.3f ms\n", (t2-t1)*1e3);
        #endif

        free(A_BLCYC);
        free(B_BLCYC);
        free(Z_BLCYC);
        Cblacs_gridexit(ictxt);
    } else {
        int i, ONE = 1, descA_BLCYC[9], descB_BLCYC[9], descZ_BLCYC[9];
        double complex *A_BLCYC, *B_BLCYC, *Z_BLCYC;
        A_BLCYC = B_BLCYC = Z_BLCYC = NULL;
        for (i = 0; i < 9; i++)
            descA_BLCYC[i] = descB_BLCYC[i] = descZ_BLCYC[i] = -1;

        pzgemr2d_(&N, &N, a, ia, ja, desca, A_BLCYC, &ONE, &ONE, descA_BLCYC, &ictxt_old);
        pzgemr2d_(&N, &N, b, ib, jb, descb, B_BLCYC, &ONE, &ONE, descB_BLCYC, &ictxt_old);
        pzgemr2d_(&N, &N, Z_BLCYC, &ONE, &ONE, descZ_BLCYC, z, iz, jz, descz, &ictxt_old);
    }

    if (dims[0] * dims[1] < nproc)
        Bcast_eigenvalues(w, N, ictxt, dims[0]*dims[1], comm);
#endif // (USE_MKL or USE_SCALAPACK)
}


/**
 * @brief   Broadcast eigenvalues using inter communicator
 */
void Bcast_eigenvalues(double *w, int N, int ictxt_subcomm, int np_subcomm, MPI_Comm comm) 
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    int color = (ictxt_subcomm >= 0);

    /* Build intra-communicator for local sub-group */
    MPI_Comm subcomm, intercomm;
    MPI_Comm_split(comm, color, rank, &subcomm);

    // now create an inter-comm between kptcomm_topo and kptcomm_topo_excl
    // assuming first np_subcomm processors are assigned to subgrid communicator
    if (color == 1) {
        MPI_Intercomm_create(subcomm, 0, comm, np_subcomm, 111, &intercomm);
    } else {
        MPI_Intercomm_create(subcomm, 0, comm, 0, 111, &intercomm);
    }
    MPI_Comm_free(&subcomm);

    if (rank == 0) {
        // the root process will broadcast the values
        MPI_Bcast(w, N, MPI_DOUBLE, MPI_ROOT, intercomm);
    } else if (rank < np_subcomm) {
        // the non-root processes in subgrid comm do nothing
        MPI_Bcast(w, N, MPI_DOUBLE, MPI_PROC_NULL, intercomm);
    } else {
        // all processes not in subgrid comm receive eigenvalues
        MPI_Bcast(w, N, MPI_DOUBLE, 0, intercomm);
    }
    MPI_Comm_free(&intercomm);
}


/**
 * @brief  Perform matrix-matrix product A'*B in a sub-comm with a better grid
 *         dimension.
 *
 *         This routine first creates a sub-process grid within the given
 *         communicator. And then redistribute the matrices into block-cyclic
 *         format and then perform the matrix-matrix product in the new process
 *         grid:
 *                        C = alpha * op(A) * op(B) + beta * C.
 *         The result C is distributed based on the given descriptor for C.
 *
 * @param transa Specify the form of op(A):
 *              if transa = "T": op(A) = A^T; if transa = "N": op(A) = A.
 * @param transb Specify the form of op(B):
 *              if transb = "T": op(B) = B^T; if transb = "N": op(B) = B.
 * @param M     The (global) number of rows in the resulting matrix C.
 * @param N     The (global) number of columns in the resulting matrix C.
 * @param K     The (global) number of rows in the matrix B.
 * @param alpha Scalar alpha.
 * @param A     The local pieces of the distributed matrix A (original paral.).
 * @param descA The descriptor for matrix A.
 * @param B     The local pieces of the distributed matrix B (original paral.).
 * @param descB The descriptor for matrix B.
 * @param beta  Scalar beta.
 * @param C     The local pieces of the distributed matrix C (original paral.).
 * @param descC The descriptor for matrix C.
 * @param comm  The communicator where this operation is performed.
 * @param max_nproc The max allowed number of processes in the subcomm to be used.
 */
void pdgemm_subcomm(
    char *transa, char *transb, int M, int N, int K, double alpha, const double *A,
    int *descA, const double *B, int *descB, double beta, double *C, int *descC,
    MPI_Comm comm, int max_nproc)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // generate a (subset) process grid within comm
    // extern int Csys2blacs_handle(MPI_Comm comm); // TODO: move this to scalapack.h
    int bhandle = Csys2blacs_handle(comm); // create a context out of rowcomm
    int ictxt = bhandle;

    // find global matrix sizes for A and B
    int M_A, N_A, M_B, N_B;
    M_A = M; N_A = K;
    M_B = K; N_B = N;

    if (strcmpi(transa, "T") == 0) {
        int temp = M_A; M_A = N_A; N_A = temp; // swap M_A and N_A
    }
    if (strcmpi(transb, "T") == 0) {
        int temp = M_B; M_B = N_B; N_B = temp; // swap M_B and N_B
    }

    // decide on the new process grid dimensions
    int nproc, gridsizes[2] = {M,K}, dims[2], ierr;
    MPI_Comm_size(comm, &nproc);
    // for square matrices of size < 20000, it doesn't scale well beyond 64 proc
    ierr = 1;
    int ishift = 8;
    while (ierr && ishift) {
        SPARC_Dims_create(min(nproc,max_nproc), 2, gridsizes, 1<<ishift, dims, &ierr); // best min size 256
    	ishift--;
    }
    if (ierr) dims[0] = dims[1] = 1;
    // TODO: swap dim[0] and dim[1] value, since SPARC_Dims_create tends to give
    // TODO: larger dim for dim[1] on a tie situation
    #ifdef DEBUG_PDGEMM_SUBCOMM
    if (rank == 0) printf("pdgemm_subcomm: process grid = (%d, %d)\n", dims[0], dims[1]);
    #endif

    // create new context with dimensions: dims[0] x dims[1]
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

    int ictxt_rowcomm = descA[1];
    if (ictxt >= 0) {
        int nprow, npcol, myrow, mycol;
        Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
        // nproc_grid = nprow * npcol;

        // define new BLCYC distribution of A and B
        int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info,
            descA_BLCYC[9], descB_BLCYC[9];

        // new descriptor for A in block-cyclic format
        mb = max(1, M_A/nprow); // block size in 1st dim
        nb = max(1, N_A/npcol); // block size in 2nd dim
        m_loc = numroc_(&M_A, &mb, &myrow, &ZERO, &nprow);
        n_loc = numroc_(&N_A, &nb, &mycol, &ZERO, &npcol);
        llda = max(1, m_loc);
        descinit_(descA_BLCYC, &M_A, &N_A, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info);
        assert(info == 0);
        double *A_BLCYC  = (double *)malloc(m_loc*n_loc*sizeof(double));
        assert(A_BLCYC != NULL);

        // new descriptor for B in block-cyclic format
        mb = max(1, M_B/nprow); // block size in 1st dim
        nb = max(1, N_B/npcol); // block size in 2nd dim
        m_loc = numroc_(&M_B, &mb, &myrow, &ZERO, &nprow);
        n_loc = numroc_(&N_B, &nb, &mycol, &ZERO, &npcol);
        llda = max(1, m_loc);
        descinit_(descB_BLCYC, &M_B, &N_B, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info);
        assert(info == 0);
        double *B_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
        assert(B_BLCYC != NULL);

#ifdef PRINT_MAT
        if (rank == 0) {
            printf("pdgemm_subcomm (before): local A in rank %d\n", rank);
            int ncol = 5; // hard-coded
            for (int i = 0; i < min(K,10); i++) {
                for (int j = 0; j < min(ncol,12); j++) {
                    printf("%15.8e ", A[j*K+i]);
                }
                printf("\n");
            }
        }
#endif

        #ifdef DEBUG_PDGEMM_SUBCOMM
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        // convert A from original distribution to BLCYC in the new context
        pdgemr2d_(&M_A, &N_A, A, &ONE, &ONE, descA, A_BLCYC, &ONE, &ONE,
            descA_BLCYC, &ictxt_rowcomm);
        #ifdef DEBUG_PDGEMM_SUBCOMM
        t2 = MPI_Wtime();
        if (!rank) printf("pdgemm_subcomm: A -> A_BLCYC: %.3f ms\n", (t2-t1)*1e3);
        #endif

        // if A = B, then skip the transfer for B
        if (A != B) {
            #ifdef DEBUG_PDGEMM_SUBCOMM
            t1 = MPI_Wtime();
            #endif
            // convert B from original distribution to BLCYC in the new context
            pdgemr2d_(&M_B, &N_B, B, &ONE, &ONE, descB, B_BLCYC, &ONE, &ONE,
                descB_BLCYC, &ictxt_rowcomm);            
            #ifdef DEBUG_PDGEMM_SUBCOMM
            t2 = MPI_Wtime();
            if (!rank) printf("pdgemm_subcomm: B -> B_BLCYC: %.3f ms\n", (t2-t1)*1e3);
            #endif
        }

        // find mass matrix
        // define BLCYC distribution of Mt
        int descC_BLCYC[9];
        mb = nb = max(1, M/max(nprow,npcol)); // block size mb must equal nb
        m_loc = numroc_(&M, &mb, &myrow, &ZERO, &nprow);
        n_loc = numroc_(&N, &nb, &mycol, &ZERO, &npcol);
        llda = max(1, m_loc);
        descinit_(descC_BLCYC, &M, &N, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info);
        assert(info == 0);

        double *C_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
        assert(C_BLCYC != NULL);

        // if beta != 0.0, we need to copy the values from C to C_BLCYC
        if (fabs(beta) > TEMP_TOL) {
            #ifdef DEBUG_PDGEMM_SUBCOMM
            t1 = MPI_Wtime();
            #endif
            pdgemr2d_(&M, &N, C, &ONE, &ONE, descC, C_BLCYC, &ONE, &ONE, descC_BLCYC, &ictxt_rowcomm);
            #ifdef DEBUG_PDGEMM_SUBCOMM
            t2 = MPI_Wtime();
            if (!rank) printf("pdgemm_subcomm: C -> C_BLCYC: %.3f ms\n", (t2-t1)*1e3);
            #endif
        }

        #ifdef DEBUG_PDGEMM_SUBCOMM
        t1 = MPI_Wtime();
        #endif
        // double alpha = 1.0, beta = 0.0;
        // C = A' * B
        if (A == B && strcmpi(transa,"T") == 0 && strcmpi(transb,"N") == 0) { // C = A'*A
            // pdsyrk_("U", "T", &M, &K, &alpha, A_BLCYC, &ONE, &ONE, descA_BLCYC,
            //     &beta, C_BLCYC, &ONE, &ONE, descC_BLCYC);
            pdgemm_(transa, transb, &M, &N, &K, &alpha, A_BLCYC, &ONE, &ONE, descA_BLCYC,
                A_BLCYC, &ONE, &ONE, descA_BLCYC, &beta, C_BLCYC, &ONE, &ONE, descC_BLCYC);
        } else { // C = A'*B
            pdgemm_(transa, transb, &M, &N, &K, &alpha, A_BLCYC, &ONE, &ONE, descA_BLCYC,
                B_BLCYC, &ONE, &ONE, descB_BLCYC, &beta, C_BLCYC, &ONE, &ONE, descC_BLCYC);
        }
        #ifdef DEBUG_PDGEMM_SUBCOMM
        t2 = MPI_Wtime();
        if (!rank) printf("pdgemm_subcomm: C = A' * B: %.3f ms\n", (t2-t1)*1e3);
        t1 = MPI_Wtime();
        #endif

        // convert C_BLCYC to given format
        pdgemr2d_(&M, &N, C_BLCYC, &ONE, &ONE, descC_BLCYC, C, &ONE, &ONE, descC, &ictxt_rowcomm);
        #ifdef DEBUG_PDGEMM_SUBCOMM
        t2 = MPI_Wtime();
        if (!rank) printf("pdgemm_subcomm: C_BLCYC -> C: %.3f ms\n", (t2-t1)*1e3);
        #endif

#ifdef PRINT_MAT
        if (rank == 0) {
            printf("C: local C in rank %d\n", rank);
            int ncol = 2;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < min(ncol,12); j++) {
                    printf("%15.8e ", C[j*M+i]);
                }
                printf("\n");
            }
        }
#endif
        free(A_BLCYC);
        free(B_BLCYC);
        free(C_BLCYC);
        Cblacs_gridexit(ictxt);
    } else {
        int ONE = 1, descA_BLCYC[9], descB_BLCYC[9], descC_BLCYC[9];
        double *A_BLCYC, *B_BLCYC, *C_BLCYC;
        A_BLCYC = B_BLCYC = C_BLCYC = NULL;
        for (int i = 0; i < 9; i++)
            descA_BLCYC[i] = descB_BLCYC[i] = descC_BLCYC[i] = -1;

        pdgemr2d_(&M_A, &N_A, A, &ONE, &ONE, descA, A_BLCYC, &ONE, &ONE, descA_BLCYC, &ictxt_rowcomm);

        if (A != B)
            pdgemr2d_(&M_B, &N_B, B, &ONE, &ONE, descB, B_BLCYC, &ONE, &ONE, descB_BLCYC, &ictxt_rowcomm);
        if (fabs(beta) > TEMP_TOL) {
            pdgemr2d_(&M, &N, C, &ONE, &ONE, descC, C_BLCYC, &ONE, &ONE, descC_BLCYC, &ictxt_rowcomm);
        }
        pdgemr2d_(&M, &N, C_BLCYC, &ONE, &ONE, descC_BLCYC, C, &ONE, &ONE, descC, &ictxt_rowcomm);
    }
#endif // (USE_MKL or USE_SCALAPACK)
}
