/**
 * @file    cyclix_lapVec.c
 * @brief   This file contains the functions for performing laplacian matrix times vector
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */


#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "cyclix_lapVec.h"
#include "isddft.h"




/*
 @ brief: function to perform 4 component stencil operation for systems with only cyclic symmetry
*/
void stencil_4comp_cyclix(
    const SPARC_OBJ *pSPARC,
    const double *X,        const int radius,
    const int stride_y_X1,  const int stride_y_X,
    const int stride_z_X1,  const int stride_z_X,
    const int x_X1_spos,    const int x_X1_epos,
    const int y_X1_spos,    const int y_X1_epos,
    const int z_X1_spos,    const int z_X1_epos,
    const int x_X_spos,     const int y_X_spos,
    const int z_X_spos,     const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double w2_diag,   const double b,
    const double xin,       const double a,
    const double *v0,       double *X1
)
{
    int i, j, k, jj, kk, r;
    double dx = pSPARC->delta_x;
    double c0 = pSPARC->D2_stencil_coeffs_y[0] * a;

    for (k = z_X1_spos, kk = z_X_spos; k < z_X1_epos; k++, kk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        for (j = y_X1_spos, jj = y_X_spos; j < y_X1_epos; j++, jj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            #pragma omp simd
            for (i = x_X1_spos; i < x_X1_epos; i++)
            {
                int ishift_X1 = jshift_X1 + i;
                int ishift_X = jshift_X + i - x_X1_spos + x_X_spos;
                double x = xin + i * dx;
                double coef_yy = 1.0/(x*x);
                double coef_0 = w2_diag + coef_yy * c0;
                double res = coef_0 * X [ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 4 * r + 1;
                    double res_xx = (X[ishift_X - r]             + X[ishift_X + r])             * (stencil_coefs[r_fac]);
                    double res_x  = (X[ishift_X + r]             - X[ishift_X - r])             * (stencil_coefs[r_fac+1]/x);
                    double res_yy = (X[ishift_X - stride_y_r]    + X[ishift_X + stride_y_r])    * (stencil_coefs[r_fac+2] * coef_yy);
                    double res_zz = (X[ishift_X - stride_z_r]    + X[ishift_X + stride_z_r])    *  stencil_coefs[r_fac+3];
                    res += res_xx + res_x + res_yy + res_zz;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}



/*
 @ brief: function to perform 5 component stencil operation for systems with cyclix symmetries
*/
void stencil_5comp_cyclix(
    const SPARC_OBJ *pSPARC,
    const double *X,        const double *DX,
    const int radius,       const int stride_DX,
    const int stride_y_X1,
    const int stride_y_X,   const int stride_y_DX,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX,  const int x_X1_spos,
    const int x_X1_epos,    const int y_X1_spos,
    const int y_X1_epos,    const int z_X1_spos,
    const int z_X1_epos,    const int x_X_spos,
    const int y_X_spos,     const int z_X_spos,
    const int x_DX_spos,    const int y_DX_spos,
    const int z_DX_spos,    const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double w2_diag,   const double b,
    const double xin,       const double a,
    const double *v0,       double *X1
)
{
    int i, j, k, jj, kk, jjj, kkk, r;
    double dx = pSPARC->delta_x;
    double tw2 = pSPARC->twist * pSPARC->twist;
    double c0 = pSPARC->D2_stencil_coeffs_y[0] * a;

    for (k = z_X1_spos, kk = z_X_spos, kkk = z_DX_spos; k < z_X1_epos; k++, kk++, kkk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        int kshift_DX = kkk * stride_z_DX;
        for (j = y_X1_spos, jj = y_X_spos, jjj = y_DX_spos; j < y_X1_epos; j++, jj++, jjj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            int jshift_DX = kshift_DX + jjj * stride_y_DX;
            const int shift_X1_X = x_X_spos - x_X1_spos;
            const int shift_X1_DX = x_DX_spos - x_X1_spos;
            #pragma omp simd
            for (i = x_X1_spos; i < x_X1_epos; i++)
            {
                int ishift_X1 = jshift_X1 + i;
                int ishift_X = jshift_X + i + shift_X1_X;
                int ishift_DX = jshift_DX + i + shift_X1_DX;
                double x = xin + i * dx;
                double coef_yy = tw2 + 1.0/(x*x);
                double coef_0 = w2_diag + coef_yy * c0;
                double res = coef_0 * X [ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX_r = r * stride_DX;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 5 * r;
                    double res_xx = (X[ishift_X - r]             + X[ishift_X + r])             * (stencil_coefs[r_fac]);
                    double res_x  = (X[ishift_X + r]             - X[ishift_X - r])             * (stencil_coefs[r_fac+1]/x);
                    double res_yy = (X[ishift_X - stride_y_r]    + X[ishift_X + stride_y_r])    * (stencil_coefs[r_fac+2] * coef_yy);
                    double res_zz = (X[ishift_X - stride_z_r]    + X[ishift_X + stride_z_r])    *  stencil_coefs[r_fac+3];
                    double res_yz = (DX[ishift_DX + stride_DX_r] - DX[ishift_DX - stride_DX_r]) *  stencil_coefs[r_fac+4];
                    res += res_xx + res_x + res_yy + res_zz + res_yz;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}



/*
 @ brief: function to perform 4 component stencil operation for systems with only cyclic symmetry
*/
void stencil_4comp_kpt_cyclix(
    const SPARC_OBJ *pSPARC,   const double _Complex *X,
    const int *DMVertices,     const int radius,
    const int stride_y_X1,     const int stride_y_X,
    const int stride_z_X1,     const int stride_z_X,
    const int x_X1_spos,       const int x_X1_epos,
    const int y_X1_spos,       const int y_X1_epos,
    const int z_X1_spos,       const int z_X1_epos,
    const int x_X_spos,        const int y_X_spos,
    const int z_X_spos,        const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double coef_0,       const double b,
    const double *v0,          double _Complex *X1,
    const double a
)
{
    int i, j, k, jj, kk, r;

    double tw2 = pSPARC->twist * pSPARC->twist;
    double c0 = pSPARC->D2_stencil_coeffs_y[0] * a;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;
 
    for (k = z_X1_spos, kk = z_X_spos; k < z_X1_epos; k++, kk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        for (j = y_X1_spos, jj = y_X_spos; j < y_X1_epos; j++, jj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            #ifdef ENABLE_SIMD_COMPLEX
            #pragma omp simd
            #endif 
            for (i = x_X1_spos; i < x_X1_epos; i++)
            {
                int ishift_X1 = jshift_X1 + i;
                int ishift_X  = jshift_X + i - x_X1_spos + x_X_spos;
                double x = xin + i * pSPARC->delta_x;
                double coef_yy = tw2 + 1.0/(x*x);
                double coef_00 = coef_0 + coef_yy * c0;
                double _Complex res = coef_00 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 4 * r + 1;
                    double _Complex res_xx = (X[ishift_X - r] + X[ishift_X + r]) * (stencil_coefs[r_fac]);
                    double _Complex res_x  = (X[ishift_X + r] - X[ishift_X - r]) * (stencil_coefs[r_fac+1]/x);
                    double _Complex res_yy = (X[ishift_X - stride_y_r] + X[ishift_X + stride_y_r]) * (stencil_coefs[r_fac+2] * coef_yy);
                    double _Complex res_zz = (X[ishift_X - stride_z_r] + X[ishift_X + stride_z_r]) * stencil_coefs[r_fac+3];
                    res += res_xx + res_x + res_yy + res_zz;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}



/*
 @ brief: function to perform 5 component stencil operation for systems with cyclix symmetries
*/
void stencil_5comp_kpt_cyclix(
    const SPARC_OBJ *pSPARC,   const double _Complex *X,
    const double _Complex *DX,
    const int *DMVertices,     const int radius,
    const int stride_DX,       const int stride_y_X1,
    const int stride_y_X,      const int stride_y_DX,
    const int stride_z_X1,     const int stride_z_X,
    const int stride_z_DX,     const int x_X1_spos,
    const int x_X1_epos,       const int y_X1_spos,
    const int y_X1_epos,       const int z_X1_spos,
    const int z_X1_epos,       const int x_X_spos,
    const int y_X_spos,        const int z_X_spos,
    const int x_DX_spos,       const int y_DX_spos,
    const int z_DX_spos,       const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double coef_0,       const double b,
    const double *v0,          double _Complex *X1,
    const double a
)
{
    int i, j, k, jj, kk, jjj, kkk, r;

    double tw2 = pSPARC->twist * pSPARC->twist;
    double c0 = pSPARC->D2_stencil_coeffs_y[0] * a;
    double xin = pSPARC->xin + DMVertices[0] * pSPARC->delta_x;

    for (k = z_X1_spos, kk = z_X_spos, kkk = z_DX_spos; k < z_X1_epos; k++, kk++, kkk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        int kshift_DX = kkk * stride_z_DX;
        for (j = y_X1_spos, jj = y_X_spos, jjj = y_DX_spos; j < y_X1_epos; j++, jj++, jjj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            int jshift_DX = kshift_DX + jjj * stride_y_DX;
            #ifdef ENABLE_SIMD_COMPLEX
            #pragma omp simd
            #endif
            for (i = x_X1_spos; i < x_X1_epos; i++)
            {
                int ishift_X1 = jshift_X1 + i;
                int ishift_X = jshift_X + i - x_X1_spos + x_X_spos;
                int ishift_DX = jshift_DX + i - x_X1_spos + x_DX_spos;
                double x = xin + i * pSPARC->delta_x;
                double coef_yy = tw2 + 1.0/(x*x);
                double coef_00 = coef_0 + coef_yy * c0;
                double _Complex res = coef_00 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX_r = r * stride_DX;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 5 * r;
                    double _Complex res_xx = (X[ishift_X - r] + X[ishift_X + r]) * (stencil_coefs[r_fac]);
                    double _Complex res_x  = (X[ishift_X + r] - X[ishift_X - r]) * (stencil_coefs[r_fac+1]/x);
                    double _Complex res_yy = (X[ishift_X - stride_y_r] + X[ishift_X + stride_y_r]) * (stencil_coefs[r_fac+2] * coef_yy) ;
                    double _Complex res_zz = (X[ishift_X - stride_z_r] + X[ishift_X + stride_z_r]) * stencil_coefs[r_fac+3];
                    double _Complex res_yz = (DX[ishift_DX + stride_DX_r] - DX[ishift_DX - stride_DX_r]) * stencil_coefs[r_fac+4];
                    res += res_xx + res_x + res_yy + res_zz + res_yz;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}
