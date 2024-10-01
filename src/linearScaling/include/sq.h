/**
 * @file    sq.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQ_H
#define SQ_H 

#include "isddft.h"


/**
 * @brief   Gauss Quadrature method
 * 
 * @param scf_iter  The scf iteration counter
 */
void GaussQuadrature(SPARC_OBJ *pSPARC, int SCFCount, int spn_i);

/**
 * @brief   Tridiagonal eigenvalue solver for Gauss method
 * 
 * @param diag          Diagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param subdiag       Subdiagonal elements of tridiagonal matrix. Length = # of FD nodes in nodal Rcut domain 
 * @param nd            FD node index in current process domain
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 */
void TridiagEigenSolve_gauss(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int nd, int spn_i, double *lambda_min, double *lambda_max);

/**
 * @brief   Lanczos algorithm for Gauss method
 * 
 * @param vkm1  Initial guess for Lanczos algorithm
 * @param lambda_min    minimal eigenvale
 * @param lambda_max    maximal eigenvale
 * @param nd            FD node index in current process domain
 */
void LanczosAlgorithm_gauss(SPARC_OBJ *pSPARC, double *vkm1, double *lambda_min, double *lambda_max, int nd, int spn_i);


/**
 * @brief   Compute nodal Hamiltonian times a vector
 * 
 * @param x     The vector multiplied by nodal hamiltonian. Vector size is the nodal Rcut domain
 * @param nd    The index of current node w.r.t. PR domain
 * @param Hx    The vector stores the result of Hx
 */
void HsubTimesVec(SPARC_OBJ *pSPARC, const double *x, const int nd, double *Hx);

/**
 * @brief Calculate (a * Laplacian + diag(Veff)) times x 
 * 
 * @param pSPARC    SPARC object pointer
 * @param x         vector
 * @param a         scaling of Laplacian
 * @param Veff      effective potential 
 * @param nd        number of grid points
 * @param Hx        (a * Laplacian + diag(Veff))x 
 */
void Lap_vec_mult_orth_SQ(
    SPARC_OBJ *pSPARC, const double *x, const double a, const double *Veff, int nd, double *Hx);


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0)) * x.
 *          Modified from stencil_3axis_thread_v2 function
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
 * @param b                : Scaling factor of v0
 * @param v0               : Values of the diagonal matrix
 * @param stride_y_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param stride_z_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param istart           : start index in x-direction of v0 
 * @param jstart           : start index in y-direction of v0 
 * @param kstart           : start index in z-direction of v0 
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_sq(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b, const double *v0, 
    const int stride_y_v, const int stride_z_v, 
    const int istart, const int jstart, const int kstart, 
    double *y
);


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0)) * x.
 *          Modified from stencil_3axis_thread_v2 function
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
 * @param b                : Scaling factor of v0
 * @param v0               : Values of the diagonal matrix
 * @param stride_y_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param stride_z_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param istart           : start index in x-direction of v0 
 * @param jstart           : start index in y-direction of v0 
 * @param kstart           : start index in z-direction of v0 
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_radius6_sq(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b, const double *v0, 
    const int stride_y_v, const int stride_z_v, 
    const int istart, const int jstart, const int kstart, 
    double *y
);


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0)) * x.
 *          Modified from stencil_3axis_thread_v2 function
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
 * @param b                : Scaling factor of v0
 * @param v0               : Values of the diagonal matrix
 * @param stride_y_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param stride_z_v       : Distance between v0(i, j, k) and v0(i, j+1, k)
 * @param istart           : start index in x-direction of v0 
 * @param jstart           : start index in y-direction of v0 
 * @param kstart           : start index in z-direction of v0 
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_variable_radius_sq(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b, const double *v0, 
    const int stride_y_v, const int stride_z_v, 
    const int istart, const int jstart, const int kstart, 
    double *y
);

#endif // SQ_H