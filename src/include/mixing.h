/**
 * @file    mixing.h
 * @brief   This file contains the function declarations for mixing.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
 

#ifndef MIXING_H
#define MIXING_H 

#include "isddft.h"


/**
 * @brief   Perform mixing and preconditioner.
 *
 *          Note that this is only done in "phi-domain".
 */
void Mixing(SPARC_OBJ *pSPARC, int iter_count);


/**
 * @brief   Periodic Pulay mixing.
 *
 *          Note that this is only done in "phi-domain".
 */
void Mixing_periodic_pulay(SPARC_OBJ *pSPARC, int iter_count);


/**
 * @brief   Anderson extrapolation update.
 *
 *          x_{k+1} = (x_k - X * Gamma) + beta * P * (f_k - F * Gamma),
 *          where P is the preconditioner, and Gamma = inv(F^T * F) * F^T * f.
 *          Expanding above equation gives: 
 *          x_{k+1} = x_k + beta * P * f - (X + beta * P * F) * inv(F^T * F) * F^T * f          
 */
void  AndersonExtrapolation(
        const int N, const int m, double *x_kp1, const double *x_k, 
        const double *f_k, const double *X, const double *F, 
        const double beta, MPI_Comm comm
);

void  AndersonExtrapolation_complex(
        const int N, const int m, double _Complex *x_kp1, const double _Complex *x_k, 
        const double _Complex *f_k, const double _Complex *X, const double _Complex *F, 
        const double beta, MPI_Comm comm
);


/**
 * @brief   Anderson extrapolation weighted average vectors.
 *
 *          Find x_wavg := x_k - X * Gamma and f_wavg := (f_k - F * Gamma),
 *          where Gamma = inv(F^T * F) * F^T * f.
 */
void  AndersonExtrapWtdAvg(
        const int N, const int m, const int Nspden, const int opt, 
        const double *x_k, const double *f_k, 
        const double *X, const double *F, double *x_wavg, double *f_wavg, 
        MPI_Comm comm
);


/**
 * @brief   Anderson extrapolation coefficiens.
 *
 *          Gamma = inv(F^T * F) * F^T * f.         
 */
void AndersonExtrapCoeff(
    const int N, const int m, const int Nspden, const int opt, 
    const double *f, const double *F, double* Gamma, MPI_Comm comm
);


void compute_FtF(const double *F, int m, int N, int nspden, int opt, double *FtF);

void compute_Ftf(const double *F, const double *f, int m, int N, int nspden, int opt, double *Ftf);

double dotprod_nc(const double *vec1, const double *vec2, int N, int opt);

/**
 * @brief   Perform Kerker preconditioner.
 *
 *          Apply Kerker preconditioner in real space. For given 
 *          function f, this function returns 
 *          Pf := a * (L - lambda_TF^2)^-1 * (L - idemac*lambda_TF^2)f, 
 *          where L is the discrete Laplacian operator, c is the 
 *          inverse of diemac (dielectric macroscopic constant).
 *          When c is 0, it's the original Kerker preconditioner.
 *          The result is written in Pf.
 */
void Kerker_precond(
    SPARC_OBJ *pSPARC, double *f, const double a, 
    const double lambda_TF, const double idiemac, const double tol, 
    const int DMnd, const int *DMVertices, double *Pf, MPI_Comm comm
);


/**
 * @brief   Perform Resta preconditioner.
 *
 *          Apply Resta preconditioner in real space. For given 
 *          function f, this function returns 
 *          Pf := sum_i a(i) * (L - lambda_sqr(i))^-1 * Lf + k*f, 
 *          where L is the discrete Laplacian operator. The result 
 *          is written in Pf.
 */
void Resta_precond(
    SPARC_OBJ *pSPARC, double *f, const double tol, const int DMnd, 
    const int *DMVertices, double *Pf, MPI_Comm comm
);


/**
 * @brief   Perform truncated Kerker preconditioner.
 *
 *          Apply truncated Kerker preconditioner in real space. For given 
 *          function f, this function returns 
 *          Pf := sum_i a(i) * (L - lambda_sqr(i))^-1 * Lf + k*f, 
 *          where L is the discrete Laplacian operator. The result 
 *          is written in Pf.
 */
void TruncatedKerker_precond(
    SPARC_OBJ *pSPARC, double *f, const double tol, const int DMnd, 
    const int *DMVertices, double *Pf, MPI_Comm comm
);


/** 
 * @brief   RSFIT_PRECOND applies real-space preconditioner with any rational fit 
 *          coefficients.
 *
 *          RSFIT_PRECOND effectively applies sum_i (a_i*(Lap - k_TF2_i)^-1 * Lap + k*I) 
 *          to f by solving the linear systems a_i*(Lap - k_TF2_i) s = Lap*f and 
 *          summing the sols. To apply any preconditioner, simply perform a
 *          rational curve fit to the preconditioner in fourier space and provide
 *          the fit coeffs a(i), lambda_TF(i) and a constant k.
 */
void RSfit_Precond(
    SPARC_OBJ *pSPARC, double *f, const int npow, 
    const double _Complex *a, 
    const double _Complex *lambda_sqr, 
    const double k,  // k should always be real
    const double tol, const int DMnd, const int *DMVertices, 
    double *Pf, MPI_Comm comm
);


#endif // MIXING_H 

