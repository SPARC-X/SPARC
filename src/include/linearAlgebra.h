#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H



/**
 * @brief Call the pdsyevx_ routine with an automatic workspace setup.
 *
 *        The original pdsyevx_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsyevx_ routine to
 *        do the calculation.
 */
void automem_pdsyevx_ ( 
	char *jobz, char *range, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
	double *vl, double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info);

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
    double *w, double *z, int *iz, int *jz, int *descz, int *info);

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
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info);

/**
 * @brief Call the pzhegvx_ routine with an automatic workspace setup.
 *
 *        The original pzhegvx_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pzhegvx_ routine to
 *        do the calculation.
 */
void automem_pzhegvx_ ( 
	int *ibtype, char *jobz, char *range, char *uplo, int *n, double complex *a, int *ia,
	int *ja, int *desca, double complex *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double complex *z, int *iz, int *jz, int *descz, int *ifail, int *info);


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
	double *w, double *z, int *iz, int *jz, int *descz, int *info);

/**
 * @brief  Solve real generalized eigenvalue problem in a subgrid 
 *         commmunicator with a better grid dimension.
 */
void pdsygvx_subcomm_ (
    int *ibtype, char *jobz, char *range, char *uplo, int *n, double *a, int *ia,
	int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info,
    MPI_Comm comm, int *dims, int blksz);


/**
 * @brief  Solve real standard eigenvalue problem in a subgrid 
 *         commmunicator with a better grid dimension.
 */
void pdsyevx_subcomm_ (
    char *jobz, char *range, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
	double *vl, double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info,
    MPI_Comm comm, int *dims, int blksz);



/**
 * @brief  Solve complex generalized eigenvalue problem in a subgrid 
 *         commmunicator with a better grid dimension.
 */
void pzhegvx_subcomm_ (
    int *ibtype, char *jobz, char *range, char *uplo, int *n, double complex *a, int *ia,
	int *ja, int *desca, double complex *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double complex *z, int *iz, int *jz, int *descz, int *ifail, int *info,
    MPI_Comm comm, int *dims, int blksz);


/**
 * @brief   Broadcast eigenvalues using inter communicator
 */
void Bcast_eigenvalues(double *w, int N, int ictxt_subcomm, int np_subcomm, MPI_Comm comm);


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
    MPI_Comm comm, int max_nproc);

#endif // LINEARALGEBRA_H