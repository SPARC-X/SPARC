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

/**
 * @brief   initialize matrices and eigenvalues for Laplacian using kronecker product
 */
void init_kron_Lap(SPARC_OBJ *pSPARC, int Nx, int Ny, int Nz, 
    int BCx, int BCy, int BCz, KRON_LAP *kron_lap);

/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with Dirichlet BC
 */
void Lap_1D_D_EigenDecomp(int N, int FDn, 
    double mesh, double *FDweights_D2, double *V, double *lambda);

/**
 * @brief   Create 1D dense Laplacian matrix with Dirichlet BC
 */
void Lap_1D_Dirichlet(int FDn, double *FDweights_D2, int N, double *Lap_1D_D);

/**
 * @brief   Eigen decomposition of 1D Laplacian matrix with periodic BC
 */
void Lap_1D_P_EigenDecomp(int N, int FDn, 
    double mesh, double *FDweights_D2, double *V, double *lambda);

/**
 * @brief   Create 1D dense Laplacian matrix with periodic BC
 */
void Lap_1D_Periodic(int FDn, double *FDweights_D2, int N, double *Lap_1D_P);

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
                 double *rhs, double *diag, double *invLapx);
#endif 