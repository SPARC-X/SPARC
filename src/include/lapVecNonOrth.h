/**
 * @file    lapVecnonorth.h
 * @brief   This file contains functions declarations for performing 
 *          the non orthogonal laplacian-vector multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef NONORTHLAP_H
#define NONORTHLAP_H

#include "isddft.h"


/**
 * @brief   Calculate (a * Lap + c * I) times vectors.
 */
void Lap_vec_mult_nonorth(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double a, const double c, const double *x, 
    double *y, MPI_Comm comm,  MPI_Comm comm2, const int *dims
);


/*
 @ brief: function to provide start and end indices of the buffer for communication
*/

void snd_rcv_buffer(
        const int nproc, const int *dims, const int *periods, const int FDn,
        const int DMnx, const int DMny, const int DMnz, int *istart,
        int *iend, int *jstart, int *jend, int *kstart,
        int *kend, int *istart_in, int *iend_in, int *jstart_in, 
        int *jend_in, int *kstart_in, int *kend_in, int *isnonzero
); 




/**
 * @brief   Calculate (a * Lap + b * diag(v) + c * I) times vectors.
 *
 *          Warning:
 *            Although the function is intended for multiple vectors,
 *          it turns out to be slower than applying it to one vector
 *          at a time. So we will be calling this function ncol times
 *          if ncol is greater than 1.
 */
void Lap_plus_diag_vec_mult_nonorth(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
    const int ncol, const double a, const double b, const double c, 
    const double *v, const double *x, double *y, MPI_Comm comm,  MPI_Comm comm2,
    const int *dims
);



/*
 @ brief: function to store the laplacian stencil compactly
*/

void Lap_stencil_coef_compact(const SPARC_OBJ *pSPARC, const double FDn, double *Lap_stencil, const double a);


/*
 @ brief: function to calculate two derivatives together
*/

void Calc_DX1_DX2(
    const double *X,       double *DX,
    const int radius,      const int stride_X_1,
    const int stride_X_2,
    const int stride_y_X,  const int stride_y_DX, 
    const int stride_z_X,  const int stride_z_DX,
    const int x_DX_spos,   const int x_DX_epos, 
    const int y_DX_spos,   const int y_DX_epos,
    const int z_DX_spos,   const int z_DX_epos,
    const int x_X_spos,
    const int y_X_spos,    const int z_X_spos,
    const double *stencil_coefs1,
    const double *stencil_coefs2
);





void stencil_4comp(
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
    const double coef_0,    const double b,   
    const double *v0,       double *X1
);


/*
 @ brief: function to perform 5 component stencil operation
*/

void stencil_5comp(
    const double *X,        const double *DX1,
    const double *DX2,      const int radius,
    const int stride_DX1,   const int stride_DX2,
    const int stride_y_X1,  const int stride_y_X,
    const int stride_y_DX1, const int stride_y_DX2,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX1, const int stride_z_DX2,
    const int x_X1_spos,    const int x_X1_epos,
    const int y_X1_spos,    const int y_X1_epos,
    const int z_X1_spos,    const int z_X1_epos,
    const int x_X_spos,     const int y_X_spos,
    const int z_X_spos,     const int x_DX1_spos,
    const int y_DX1_spos,   const int z_DX1_spos,
    const int x_DX2_spos,   const int y_DX2_spos,
    const int z_DX2_spos,
    const double *stencil_coefs,
    const double coef_0,    const double b,
    const double *v0,       double *X1
);

/*
 @ brief: function to perform 4 component stencil operation y = Ax + beta*y
*/

void stencil_4comp_extd(
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
    const double coef_0,    const double b,   
    const double *v0,       const double beta,
    double *X1
);


/*
 @ brief: function to perform 5 component stencil operation y = Ax + beta*y
*/

void stencil_5comp_extd(
    const double *X,        const double *DX1,
    const double *DX2,      const int radius,
    const int stride_DX1,   const int stride_DX2,
    const int stride_y_X1,  const int stride_y_X,
    const int stride_y_DX1, const int stride_y_DX2,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX1, const int stride_z_DX2,
    const int x_X1_spos,    const int x_X1_epos,
    const int y_X1_spos,    const int y_X1_epos,
    const int z_X1_spos,    const int z_X1_epos,
    const int x_X_spos,     const int y_X_spos,
    const int z_X_spos,     const int x_DX1_spos,
    const int y_DX1_spos,   const int z_DX1_spos,
    const int x_DX2_spos,   const int y_DX2_spos,
    const int z_DX2_spos,
    const double *stencil_coefs,
    const double coef_0,    const double b,
    const double *v0,       const double beta,
    double *X1
);

#endif // NONORTHLAP_H


