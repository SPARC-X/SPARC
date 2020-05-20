#ifndef __JACOBI_DSYEV_H__
#define __JACOBI_DSYEV_H__

// Parallel Jacobi method for standard eigenproblem 
// Input parameters:
//   n         : Size of matrix A
//   A         : Symmetric matrix to be decomposed, row-major, 
//               size >= ldA * n, will be overwritten when exit
//   ldA       : Leading dimension of A, size >= n
//   ldV       : Leading dimension of V, size >= n
//   max_sweep : Maximum number of Jacobi sweeps
//   rel_tol   : Tolerance of relative fro-norm ||A - diag(D)||_fro / ||A||_fro
//   nthread   : Number of threads to use in this function
// Output parameters:
//   V : Eigenvectors, each row is an eigenvector, size >= ldV * n
//   D : Eigenvalues, size >= n
void Jacobi_dsyev(
    const int n, double *A, const int ldA,
    double *V, const int ldV, double *D, 
    int max_sweep, double rel_tol, const int nthread
);

#endif
