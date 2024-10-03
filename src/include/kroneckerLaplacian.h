/**
 * @file    kroneckerLaplacian.h
 * @brief   This file contains function declarations for Laplacians using kronecker product
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef KRONECKERLAPLACIAN_H
#define KRONECKERLAPLACIAN_H

#ifdef USE_MKL
    #include <mkl.h>
#endif

/**
 * @brief   initialize matrices and eigenvalues for Laplacian using kronecker product
 */
void init_kron_Lap(SPARC_OBJ *pSPARC, int Nx, int Ny, int Nz, 
    int BCx, int BCy, int BCz, double k1, double k2, double k3, int isGammaPoint, KRON_LAP *kron_lap);

/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with Dirichlet BC
 */
void Lap_1D_D_EigenDecomp(int N, int FDn, double *FDweights_D2, double *V, double *lambda);

/**
 * @brief   Create 1D dense Laplacian matrix with Dirichlet BC
 */
void Lap_1D_Dirichlet(int FDn, double *FDweights_D2, int N, double *Lap_1D_D);

/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with periodic BC
 */
void Lap_1D_P_EigenDecomp(int N, int FDn, double *FDweights_D2, double *V, double *lambda);

/**
 * @brief   Create 1D dense Laplacian matrix with periodic BC
 */
void Lap_1D_Periodic(int FDn, double *FDweights_D2, int N, double *Lap_1D_P);

/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with bloch periodic BC
 */
void Lap_1D_P_EigenDecomp_complex(int N, int FDn, double *FDweights_D2, double _Complex *V, double *lambda, double _Complex phase_fac);

/**
 * @brief   Create 1D dense Laplacian matrix with bloch periodic BC
 */
void Lap_1D_Periodic_complex(int FDn, double *FDweights_D2, int N, double _Complex *Lap_1D_P, double _Complex phase_fac);

/**
 * @brief   Calculate eigenvalues for 3D Laplacian using eigenvalues for 1D Laplacian
 */
void eigval_Lap_3D(int Nx, double *lambda_x, int Ny, double *lambda_y, int Nz, double *lambda_z, double *lambda);

/**
 * @brief   Free KRON_LAP structure
 */
void free_kron_Lap(KRON_LAP * kron_lap);

/**
 * @brief   apply inverse of Laplacian to a vector by kronecker product 
 */
void Lap_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, double *diag, double *out);

/**
 * @brief   apply Laplacian or inverse of Laplacian to a vector by kronecker product 
 * 
 * Note:    diag could be either eigenvalue of Laplacian or inverse of eigenvalue of Laplacian
 */
void Lap_Kron_complex(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, double *diag, double _Complex *out);

#ifdef USE_MKL
void Lap_Kron_batch(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, double *diag, double *out);

void Lap_Kron_complex_batch(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, double *diag, double _Complex *out);

void Lap_Kron_multicol_batch(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, int ncol, double *diag, double *out);

void Lap_Kron_multicol_complex_batch(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, int ncol, double *diag, double _Complex *out);

void cblas_dgemm_batch_wrapper(const CBLAS_LAYOUT Layout, 
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    const MKL_INT m, const MKL_INT n, const MKL_INT k, 
    const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, 
    const double beta, double *c, const MKL_INT ldc, const int aFixed, const int bFixed, const MKL_INT batch);

void cblas_zgemm_batch_wrapper(const CBLAS_LAYOUT Layout, 
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    const MKL_INT m, const MKL_INT n, const MKL_INT k, 
    const double _Complex *alpha, const double _Complex *a, const MKL_INT lda, const double _Complex *b, const MKL_INT ldb, 
    const double _Complex *beta, double _Complex *c, const MKL_INT ldc, const int aFixed, const int bFixed, const MKL_INT batch);

#define LAP_KRON Lap_Kron_batch
#define LAP_KRON_COMPLEX Lap_Kron_complex_batch
#else

#define LAP_KRON Lap_Kron
#define LAP_KRON_COMPLEX Lap_Kron_complex
#endif 

#endif 