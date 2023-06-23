/**
 * @file    cyclix_lapVec.h
 * @brief   This file contains the functions for performing laplacian matrix times vector
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#ifndef CYCLIX_LAPVEC_H
#define CYCLIX_LAPVEC_H 

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
);



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
);




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
);



/*
 @ brief: function to perform 5 component stencil operation for cyclix
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
);

#endif // CYCLIX_LAPVEC_H
