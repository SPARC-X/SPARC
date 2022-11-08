/**
 * @file    lapVecRoutines.h
 * @brief   This file contains function declarations for performing 
 *          the laplacian-vector multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef LAPVECROUTINES_H
#define LAPVECROUTINES_H

#include "isddft.h"


/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 *
 *          General routine, for both orthogonal and non-orthogonal.
 */
void Lap_vec_mult(
	const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
	const int ncol, const double c, double *x, double *Lapx, MPI_Comm comm
);


/**
 * @brief   Calculate the residual of the poisson equation: r = b - A * x, where A = -(Lap + c).
 *
 *          The vector x is assumed to be stored domain-wisely among the processors. The
 *          structure pSPARC contains the description of the distribution info of x, and
 *          in this case the info of Laplacian operator such as boundary conditions, 
 *          finite-difference order and coefficients etc.
 */
void poisson_residual(
    SPARC_OBJ *pSPARC, int N, double c, double *x, double *b, 
    double *r, MPI_Comm comm, double *time_info
);


/**
 * @brief   Calculate (a * Lap + c * I) times vectors.
 *
 *          This is only for orthogonal systems.
 *
 */
void Lap_vec_mult_orth(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double a, const double c, const double *x, 
    double *y, MPI_Comm comm, const int *dims
);


/**
 * @brief   Calculate (a * Lap + b * diag(v) + c * I) times vectors.
 *
 *          This is only for orthogonal systems.
 *
 *          Warning:
 *            Although the function is intended for multiple vectors,
 *          it turns out to be slower than applying it to one vector
 *          at a time. So we will be calling this function ncol times
 *          if ncol is greater than 1.
 */
void Lap_plus_diag_vec_mult_orth(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
    const int ncol, const double a, const double b, const double c, 
    const double *v, const double *x, double *y, MPI_Comm comm,
    const int *dims
);


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0) + c * I) * x.
 *          For the input & output domain, z/x index is the slowest/fastest running index
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element
 * @param a                : Scaling factor of Lap
 * @param b                : Scaling factor of v0
 * @param c                : Shift constant
 * @param v0               : Values of the diagonal matrix
 * @param beta             : Scaling factor of y
 * @param y (OUT)          : Output domain with original boundary
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *
 * @modified by Qimen Xu <qimenxu@gatech.edu>, Mar 2019, Georgia Tech
 *
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
void stencil_3axis_thread_v2(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
);



/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0) + c * I) * x + beta * y
 *          For the input & output domain, z/x index is the slowest/fastest running index
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element
 * @param a                : Scaling factor of Lap
 * @param b                : Scaling factor of v0
 * @param c                : Shift constant
 * @param v0               : Values of the diagonal matrix
 * @param beta             : Scaling factor of y
 * @param y (OUT)          : Output domain with original boundary
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *
 * @modified by Qimen Xu <qimenxu@gatech.edu>, Mar 2019, Georgia Tech
 *
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
void stencil_3axis_thread_v1(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    const double beta, 
    double *y
);



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

#endif // LAPVECROUTINES_H


