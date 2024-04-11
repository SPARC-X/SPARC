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
	#define MKL_Complex16 double _Complex
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
    int BCx, int BCy, int BCz, KRON_LAP *kron_lap)
{
    kron_lap->Nx = Nx;
    kron_lap->Ny = Ny;
    kron_lap->Nz = Nz;
    int Nd = kron_lap->Nd = Nx * Ny * Nz;
    int FDn = pSPARC->order / 2;
    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;
    kron_lap->Vx = (double *) malloc(Nx*Nx * sizeof(double));
    kron_lap->Vy = (double *) malloc(Ny*Ny * sizeof(double));
    kron_lap->Vz = (double *) malloc(Nz*Nz * sizeof(double));
    double *lambda_x = (double *) malloc(Nx * sizeof(double));
    double *lambda_y = (double *) malloc(Ny * sizeof(double));
    double *lambda_z = (double *) malloc(Nz * sizeof(double));
    kron_lap->eig = (double *) calloc(sizeof(double),  Nd);
    kron_lap->inv_eig = (double *) calloc(sizeof(double),  Nd);
    assert(kron_lap->Vx != NULL && kron_lap->Vy != NULL && kron_lap->Vz != NULL &&
           lambda_x != NULL && lambda_y != NULL && lambda_z != NULL && kron_lap->eig != NULL && kron_lap->inv_eig != NULL);

    if (BCx == 1) {
        Lap_1D_D_EigenDecomp(Nx, FDn, dx, pSPARC->D2_stencil_coeffs_x, kron_lap->Vx, lambda_x);
    } else {
        Lap_1D_P_EigenDecomp(Nx, FDn, dx, pSPARC->D2_stencil_coeffs_x, kron_lap->Vx, lambda_x);
    }
        
    if (BCy == 1) {
        Lap_1D_D_EigenDecomp(Ny, FDn, dy, pSPARC->D2_stencil_coeffs_y, kron_lap->Vy, lambda_y);
    } else {
        Lap_1D_P_EigenDecomp(Ny, FDn, dy, pSPARC->D2_stencil_coeffs_y, kron_lap->Vy, lambda_y);
    }
    
    if (BCz == 1) {
        Lap_1D_D_EigenDecomp(Nz, FDn, dz, pSPARC->D2_stencil_coeffs_z, kron_lap->Vz, lambda_z);
    } else {
        Lap_1D_P_EigenDecomp(Nz, FDn, dz, pSPARC->D2_stencil_coeffs_z, kron_lap->Vz, lambda_z);
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
void Lap_1D_D_EigenDecomp(int N, int FDn, 
    double mesh, double *FDweights_D2, double *V, double *lambda)
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
void Lap_1D_P_EigenDecomp(int N, int FDn, 
    double mesh, double *FDweights_D2, double *V, double *lambda)
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
            Lap_1D_P(i,j_) = FDweights_D2[shift];
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
    free(kron_lap->Vx);
    free(kron_lap->Vy);
    free(kron_lap->Vz);
    free(kron_lap->eig);
    free(kron_lap->inv_eig);
}


/**
 * @brief   apply Laplacian or inverse of Laplacian to a vector by kronecker product 
 * 
 * Note:    diag could be either eigenvalue of Laplacian or inverse of eigenvalue of Laplacian
 */
void Lap_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *diag, double *invLapx)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *rhsTVy = (double *) malloc(sizeof(double) * NxNy);
    double *VxtrhsTVy = invLapx;
    double *P = (double *) malloc(sizeof(double) * Nd);
    double *PTVyt = rhsTVy;
    double *VxPTVyt = P;

    // P = inv(Lambda) .* (Vz' x Vy' x Vx') * rhs
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, rhs + k*NxNy, Nx, Vy, Ny, 0.0, rhsTVy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, rhsTVy, Nx, 0.0, VxtrhsTVy + k*NxNy, Nx);
    }    
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtrhsTVy, NxNy, Vz, Nz, 0.0, P, NxNy);
    
    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }
    
    // invLapx = inv(Lambda) .* (Vz x Vy x Vx) * P    
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P + k*NxNy, Nx, Vy, Ny, 0.0, PTVyt, Nx);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt, Nx, 0.0, VxPTVyt + k*NxNy, Nx);
    }
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, invLapx, NxNy);

    free(rhsTVy);
    free(P);
}



