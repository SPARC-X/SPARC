#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <complex.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
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

#include "isddft.h"
#include "kroneckerLaplacian.h"

#define TEMP_TOL 1e-12
#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))


/**
 * @brief   initialize matrices and eigenvalues for Laplacian using kronecker product
 */
void init_kron_Lap(SPARC_OBJ *pSPARC, int Nx, int Ny, int Nz, 
    int BCx, int BCy, int BCz, double k1, double k2, double k3, int isGammaPoint, KRON_LAP *kron_lap)
{
    kron_lap->Nx = Nx;
    kron_lap->Ny = Ny;
    kron_lap->Nz = Nz;
    kron_lap->isGammaPoint = isGammaPoint;

    int Nd = kron_lap->Nd = Nx * Ny * Nz;
    int FDn = pSPARC->order / 2;
    if (kron_lap->isGammaPoint) {
        kron_lap->Vx = (double *) malloc(Nx*Nx * sizeof(double));
        kron_lap->Vy = (double *) malloc(Ny*Ny * sizeof(double));
        kron_lap->Vz = (double *) malloc(Nz*Nz * sizeof(double));
        assert(kron_lap->Vx != NULL && kron_lap->Vy != NULL && kron_lap->Vz != NULL);
    } else {
        kron_lap->Vx_kpt = (double _Complex *) malloc(Nx*Nx * sizeof(double _Complex));
        kron_lap->Vy_kpt = (double _Complex *) malloc(Ny*Ny * sizeof(double _Complex));
        kron_lap->Vz_kpt = (double _Complex *) malloc(Nz*Nz * sizeof(double _Complex));
        kron_lap->VyH_kpt = (double _Complex *) malloc(Ny*Ny * sizeof(double _Complex));
        kron_lap->VzH_kpt = (double _Complex *) malloc(Nz*Nz * sizeof(double _Complex));
        assert(kron_lap->Vx_kpt != NULL && kron_lap->Vy_kpt != NULL && kron_lap->Vz_kpt != NULL
                                        && kron_lap->VyH_kpt != NULL && kron_lap->VzH_kpt != NULL);
    }
    double *lambda_x = (double *) malloc(Nx * sizeof(double));
    double *lambda_y = (double *) malloc(Ny * sizeof(double));
    double *lambda_z = (double *) malloc(Nz * sizeof(double));
    kron_lap->eig = (double *) calloc(sizeof(double),  Nd);
    kron_lap->inv_eig = (double *) calloc(sizeof(double),  Nd);
    assert(lambda_x != NULL && lambda_y != NULL && lambda_z != NULL && kron_lap->eig != NULL && kron_lap->inv_eig != NULL);

    if (kron_lap->isGammaPoint) {
        if (BCx == 1)
            Lap_1D_D_EigenDecomp(Nx, FDn, pSPARC->D2_stencil_coeffs_x, kron_lap->Vx, lambda_x);
        else 
            Lap_1D_P_EigenDecomp(Nx, FDn, pSPARC->D2_stencil_coeffs_x, kron_lap->Vx, lambda_x);
    } else {            
        double _Complex phase_fac = cos(k1*pSPARC->range_x) + I * sin(k1*pSPARC->range_x);
        Lap_1D_P_EigenDecomp_complex(Nx, FDn, pSPARC->D2_stencil_coeffs_x, kron_lap->Vx_kpt, lambda_x, phase_fac);
    }

    if (kron_lap->isGammaPoint) {
        if (BCy == 1)
            Lap_1D_D_EigenDecomp(Ny, FDn, pSPARC->D2_stencil_coeffs_y, kron_lap->Vy, lambda_y);
        else 
            Lap_1D_P_EigenDecomp(Ny, FDn, pSPARC->D2_stencil_coeffs_y, kron_lap->Vy, lambda_y);
    } else {
        double _Complex phase_fac = cos(k2*pSPARC->range_y) + I * sin(k2*pSPARC->range_y);
        Lap_1D_P_EigenDecomp_complex(Ny, FDn, pSPARC->D2_stencil_coeffs_y, kron_lap->Vy_kpt, lambda_y, phase_fac);
        for (int i = 0; i < Ny*Ny; i++) kron_lap->VyH_kpt[i] = conj(kron_lap->Vy_kpt[i]);
    }

    if (kron_lap->isGammaPoint) {
        if (BCz == 1)
            Lap_1D_D_EigenDecomp(Nz, FDn, pSPARC->D2_stencil_coeffs_z, kron_lap->Vz, lambda_z);
        else 
            Lap_1D_P_EigenDecomp(Nz, FDn, pSPARC->D2_stencil_coeffs_z, kron_lap->Vz, lambda_z);
    } else {
        double _Complex phase_fac = cos(k3*pSPARC->range_z) + I * sin(k3*pSPARC->range_z);
        Lap_1D_P_EigenDecomp_complex(Nz, FDn, pSPARC->D2_stencil_coeffs_z, kron_lap->Vz_kpt, lambda_z, phase_fac);
        for (int i = 0; i < Nz*Nz; i++) kron_lap->VzH_kpt[i] = conj(kron_lap->Vz_kpt[i]);
    }
    
    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, kron_lap->eig);

    for (int i = 0; i < Nd; i++) {
        if (fabs(kron_lap->eig[i]) > 1e-8)
            kron_lap->inv_eig[i] = 1./kron_lap->eig[i];
        else
            kron_lap->inv_eig[i] = 0;
    }

    free(lambda_x); free(lambda_y); free(lambda_z);
}


/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with Dirichlet BC
 */
void Lap_1D_D_EigenDecomp(int N, int FDn, double *FDweights_D2, double *V, double *lambda)
{
    double *Lap_1D_D = V;
    memset(Lap_1D_D, 0, N*N*sizeof(double));
    Lap_1D_Dirichlet(FDn, FDweights_D2, N, Lap_1D_D);

    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 
        'V', 'U', N, V, N, lambda);
    assert(info == 0);
}


/**
 * @brief   Create 1D dense Laplacian matrix with Dirichlet BC
 */
void Lap_1D_Dirichlet(int FDn, double *FDweights_D2, int N, double *Lap_1D_D)
{
#define  Lap_1D_D(i,j) Lap_1D_D[(j)*N + (i)]

    for (int i = 0; i < N; i++) {
        for (int j = max(0, i-FDn); j <= min(N-1,i+FDn); j++) {
            int shift = abs(j - i);            
            Lap_1D_D(i,j) = FDweights_D2[shift];
        }
    }

#undef Lap_1D_D
}


/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with periodic BC
 */
void Lap_1D_P_EigenDecomp(int N, int FDn, double *FDweights_D2, double *V, double *lambda)
{
    double *Lap_1D_P = V;
    memset(Lap_1D_P, 0, N*N*sizeof(double));
    Lap_1D_Periodic(FDn, FDweights_D2, N, Lap_1D_P);

    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 
        'V', 'U', N, V, N, lambda);
    assert(info == 0);
}


/**
 * @brief   Create 1D dense Laplacian matrix with periodic BC
 */
void Lap_1D_Periodic(int FDn, double *FDweights_D2, int N, double *Lap_1D_P)
{
#define  Lap_1D_P(i,j) Lap_1D_P[(j)*N + (i)]

    for (int i = 0; i < N; i++) {
        for (int j = i-FDn; j <= i+FDn; j++) {
            int shift = abs(j - i);
            int j_ = (j + N) % N;
            Lap_1D_P(i,j_) += FDweights_D2[shift];
        }
    }

#undef Lap_1D_P
}


/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with bloch periodic BC
 */
void Lap_1D_P_EigenDecomp_complex(int N, int FDn, double *FDweights_D2, double _Complex *V, double *lambda, double _Complex phase_fac)
{
    double _Complex *Lap_1D_P = V;
    memset(Lap_1D_P, 0, N*N*sizeof(double _Complex));
    Lap_1D_Periodic_complex(FDn, FDweights_D2, N, Lap_1D_P, phase_fac);

    int info = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', N, V, N, lambda);
    assert(info == 0);
}


/**
 * @brief   Create 1D dense Laplacian matrix with bloch periodic BC
 */
void Lap_1D_Periodic_complex(int FDn, double *FDweights_D2, int N, double _Complex *Lap_1D_P, double _Complex phase_fac)
{
#define  Lap_1D_P(i,j) Lap_1D_P[(j)*N + (i)]

    for (int i = 0; i < N; i++) {
        for (int j = i-FDn; j <= i+FDn; j++) {
            int shift = abs(j - i);
            int j_ = (j + N) % N;
            if (j < 0) {
                Lap_1D_P(i,j_) += FDweights_D2[shift] * conj(phase_fac);
            } else if (j >= N) {
                Lap_1D_P(i,j_) += FDweights_D2[shift] * phase_fac;
            } else {
                Lap_1D_P(i,j_) += FDweights_D2[shift];
            }
        }
    }

#undef Lap_1D_P
}


/**
 * @brief   Calculate eigenvalues for 3D Laplacian using eigenvalues for 1D Laplacian
 */
void eigval_Lap_3D(int Nx, double *lambda_x, int Ny, double *lambda_y, int Nz, double *lambda_z, double *lambda)
{
    // I3 x I2 x Lx
    int count = 0;
    for (int k = 0; k < Ny*Nz; k++) {
        for (int j = 0; j < Nx; j++) {
            lambda[count ++] = lambda_x[j];
        }
    }

    // I3 x Ly x I1
    count = 0; 
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                lambda[count ++] += lambda_y[j];
            }
        }
    }

    // Lz x I2 x I1
    count = 0; 
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny * Nx; j++) {
            lambda[count ++] += lambda_z[k];
        }
    }
}


/**
 * @brief   Free KRON_LAP structure
 */
void free_kron_Lap(KRON_LAP * kron_lap)
{
    if (kron_lap->isGammaPoint) {
        free(kron_lap->Vx);
        free(kron_lap->Vy);
        free(kron_lap->Vz);
    } else {
        free(kron_lap->Vx_kpt);
        free(kron_lap->Vy_kpt);
        free(kron_lap->Vz_kpt);
        free(kron_lap->VyH_kpt);
        free(kron_lap->VzH_kpt);
    }
    free(kron_lap->eig);
    free(kron_lap->inv_eig);
}


/**
 * @brief   apply Laplacian or inverse of Laplacian to a vector by kronecker product 
 * 
 * Note:    diag could be either eigenvalue of Laplacian or inverse of eigenvalue of Laplacian
 */
void Lap_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, double *diag, double *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *vecTVy = (double *) malloc(sizeof(double) * NxNy);
    double *VxtvecTVy = out;
    double *P = (double *) malloc(sizeof(double) * Nd);
    double *PTVyt = vecTVy;
    double *VxPTVyt = P;

    // P = diag .* (Vz' x Vy' x Vx') * vec
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, vec + k*NxNy, Nx, Vy, Ny, 0.0, vecTVy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, vecTVy, Nx, 0.0, VxtvecTVy + k*NxNy, Nx);
    }    
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtvecTVy, NxNy, Vz, Nz, 0.0, P, NxNy);
    
    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }
    
    // out = diag .* (Vz x Vy x Vx) * P    
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P + k*NxNy, Nx, Vy, Ny, 0.0, PTVyt, Nx);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt, Nx, 0.0, VxPTVyt + k*NxNy, Nx);
    }
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, out, NxNy);

    free(vecTVy);
    free(P);
}



/**
 * @brief   apply Laplacian or inverse of Laplacian to a vector by kronecker product 
 * 
 * Note:    diag could be either eigenvalue of Laplacian or inverse of eigenvalue of Laplacian
 */
void Lap_Kron_complex(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, double *diag, double _Complex *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double _Complex *vecTVy = (double _Complex*) malloc(sizeof(double _Complex) * NxNy);
    double _Complex *VxtvecTVy = out;
    double _Complex *P = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *PTVyt = vecTVy;
    double _Complex *VxPTVyt = P;
    double _Complex aplha = 1, beta = 0;

    // P = Lambda .* (VzH x VyH x VxH) * vec
    for (int k = 0; k < Nz; k++) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    &aplha, vec + k*NxNy, Nx, VyH, Ny, &beta, vecTVy, Nx);

        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &aplha, Vx, Nx, vecTVy, Nx, &beta, VxtvecTVy + k*NxNy, Nx);
    }

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    &aplha, VxtvecTVy, NxNy, VzH, Nz, &beta, P, NxNy);

    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }

    // out = (Vz x Vy x Vx) * P    
    for (int k = 0; k < Nz; k++) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    &aplha, P + k*NxNy, Nx, Vy, Ny, &beta, PTVyt, Nx);

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &aplha, Vx, Nx, PTVyt, Nx, &beta, VxPTVyt + k*NxNy, Nx);
    }

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    &aplha, VxPTVyt, NxNy, Vz, Nz, &beta, out, NxNy);

    free(vecTVy);
    free(P);
}

#ifdef USE_MKL
void Lap_Kron_batch(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, double *diag, double *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *extra, *vecTVy, *VxtvecTVy, *P, *PTVyt, *VxPTVyt;
    extra = (double *) malloc(sizeof(double) * Nd);
    vecTVy = P = VxPTVyt = extra;
    VxtvecTVy = PTVyt = out;

    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, vec, Nx, Vy, Ny, 0.0, vecTVy, Nx, 0, 1, Nz);
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, vecTVy, Nx, 0.0, VxtvecTVy, Nx, 1, 0, Nz);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtvecTVy, NxNy, Vz, Nz, 0.0, P, NxNy);

    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }

    // out = (Vz x Vy x Vx) * P
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P, Nx, Vy, Ny, 0.0, PTVyt, Nx, 0, 1, Nz);
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt, Nx, 0.0, VxPTVyt, Nx, 1, 0, Nz);
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, out, NxNy);

    free(extra);
}


void Lap_Kron_complex_batch(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, double *diag, double _Complex *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double _Complex *extra = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *vecTVy, *VxtvecTVy, *P, *PTVyt, *VxPTVyt;
    vecTVy = P = VxPTVyt = extra;
    VxtvecTVy = PTVyt = out;
    double _Complex alpha = 1, beta = 0;

    // P = Lambda .* (VzH x VyH x VxH) * vec
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    &alpha, vec, Nx, VyH, Ny, &beta, vecTVy, Nx, 0, 1, Nz);
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasConjTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &alpha, Vx, Nx, vecTVy, Nx, &beta, VxtvecTVy, Nx, 1, 0, Nz);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    &alpha, VxtvecTVy, NxNy, VzH, Nz, &beta, P, NxNy);

    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }

    // out = (Vz x Vy x Vx) * P
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    &alpha, P, Nx, Vy, Ny, &beta, PTVyt, Nx, 0, 1, Nz);
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &alpha, Vx, Nx, PTVyt, Nx, &beta, VxPTVyt, Nx, 1, 0, Nz);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    &alpha, VxPTVyt, NxNy, Vz, Nz, &beta, out, NxNy);

    free(extra);
}


// Multi-column function is not useful on CPU, no speedup at all. Try them if you want.  
void Lap_Kron_multicol_batch(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, int ncol, double *diag, double *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    int len = Nd * ncol;
    double *extra, *VxTvec, *VxtvecTVy, *P, *PTVyt, *VxPTVyt;
    extra = (double *) malloc(sizeof(double) * len);
    VxTvec = P = VxPTVyt = extra;
    VxtvecTVy = PTVyt = out;

    // P = Lambda .* (Vz' x Vy' x Vx') * vec
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                1.0, Vx, Nx, vec, Nx, 0.0, VxTvec, Nx, 1, 0, Nz*ncol);

    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                1.0, VxTvec, Nx, Vy, Ny, 0.0, VxtvecTVy, Nx, 0, 1, Nz*ncol);

    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                1.0, VxtvecTVy, NxNy, Vz, Nz, 0.0, P, NxNy, 0, 1, ncol);
    
    // apply diagonal term
    for (int n = 0; n < ncol; n++) {
        for (int i = 0; i < Nd; i++) {
            P[i+n*Nd] *= diag[i];
        }
    }

    // out = (Vz x Vy x Vx) * P
    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                1.0, Vx, Nx, P, Nx, 0.0, PTVyt, Nx, 1, 0, Nz*ncol);

    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                1.0, PTVyt, Nx, Vy, Ny, 0.0, VxPTVyt, Nx, 0, 1, Nz*ncol);

    cblas_dgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, out, NxNy, 0, 1, ncol);

    free(extra);
}


void Lap_Kron_multicol_complex_batch(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, int ncol, double *diag, double _Complex *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double _Complex *extra = (double _Complex*) malloc(sizeof(double _Complex) * Nd * ncol);
    double _Complex *vecTVy, *VxtvecTVy, *P, *PTVyt, *VxPTVyt;
    vecTVy = P = VxPTVyt = extra;
    VxtvecTVy = PTVyt = out;
    double _Complex alpha = 1, beta = 0;

    // P = Lambda .* (VzH x VyH x VxH) * vec
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    &alpha, vec, Nx, VyH, Ny, &beta, vecTVy, Nx, 0, 1, Nz*ncol);
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasConjTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &alpha, Vx, Nx, vecTVy, Nx, &beta, VxtvecTVy, Nx, 1, 0, Nz*ncol);

    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    &alpha, VxtvecTVy, NxNy, VzH, Nz, &beta, P, NxNy, 0, 1, ncol);

    for (int n = 0; n < ncol; n++) {
        for (int i = 0; i < Nd; i++) {
            P[i+n*Nd] *= diag[i];
        }
    }

    // out = (Vz x Vy x Vx) * P
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    &alpha, P, Nx, Vy, Ny, &beta, PTVyt, Nx, 0, 1, Nz*ncol);
    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &alpha, Vx, Nx, PTVyt, Nx, &beta, VxPTVyt, Nx, 1, 0, Nz*ncol);

    cblas_zgemm_batch_wrapper(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    &alpha, VxPTVyt, NxNy, Vz, Nz, &beta, out, NxNy, 0, 1, ncol);

    free(extra);
}


void cblas_dgemm_batch_wrapper(const CBLAS_LAYOUT Layout, 
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    const MKL_INT m, const MKL_INT n, const MKL_INT k, 
    const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, 
    const double beta, double *c, const MKL_INT ldc, const int aFixed, const int bFixed, const MKL_INT batch)
{
    double** a_array = (double**) malloc(sizeof(double *)*batch);
    double** b_array = (double**) malloc(sizeof(double *)*batch);
    double** c_array = (double**) malloc(sizeof(double *)*batch);
    int size_a = transa == CblasNoTrans ? lda*k : lda*m;
    int size_b = transb == CblasNoTrans ? ldb*n : ldb*k;
    int size_c = ldc*n;
    for (int i = 0; i < batch; i++) {
        a_array[i] = (double*) a + (aFixed ? 0 : i*size_a);
        b_array[i] = (double*) b + (bFixed ? 0 : i*size_b);
        c_array[i] = (double*) c + i*size_c;
    }

    // Passing pointers
    cblas_dgemm_batch(Layout, &transa, &transb, 
                      &m, &n, &k, 
                      &alpha, (const double**)a_array, &lda, 
                      (const double**)b_array, &ldb, 
                      &beta, c_array, &ldc, 
                      1, &batch);
    
    free(a_array);
    free(b_array);
    free(c_array);
}


void cblas_zgemm_batch_wrapper(const CBLAS_LAYOUT Layout, 
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
    const MKL_INT m, const MKL_INT n, const MKL_INT k, 
    const double _Complex *alpha, const double _Complex *a, const MKL_INT lda, const double _Complex *b, const MKL_INT ldb, 
    const double _Complex *beta, double _Complex *c, const MKL_INT ldc, const int aFixed, const int bFixed, const MKL_INT batch)
{
    double _Complex ** a_array = (double _Complex **) malloc(sizeof(double _Complex *)*batch);
    double _Complex ** b_array = (double _Complex **) malloc(sizeof(double _Complex *)*batch);
    double _Complex ** c_array = (double _Complex **) malloc(sizeof(double _Complex *)*batch);
    int size_a = transa == CblasNoTrans ? lda*k : lda*m;
    int size_b = transb == CblasNoTrans ? ldb*n : ldb*k;
    int size_c = ldc*n;
    for (int i = 0; i < batch; i++) {
        a_array[i] = (double _Complex*) a + (aFixed ? 0 : i*size_a);
        b_array[i] = (double _Complex*) b + (bFixed ? 0 : i*size_b);
        c_array[i] = (double _Complex*) c + i*size_c;
    }

    // Passing the arrays as pointers
    cblas_zgemm_batch(Layout, &transa, &transb, 
                      &m, &n, &k, 
                      alpha, (const void**)a_array, &lda, 
                      (const void**)b_array, &ldb, 
                      beta, (void**)c_array, &ldc, 
                      1, &batch);
    

    free(a_array);
    free(b_array);
    free(c_array);
}
#endif