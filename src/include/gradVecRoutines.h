/**
 * @file    gradVecRoutines.h
 * @brief   This file contains function declarations for performing gradient-
 *          vector multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
 
#ifndef GRADVECROUTINES_H
#define GRADVECROUTINES_H

#include "isddft.h"


/**
 * @brief   Calculate (Gradient + c * I) times a bunch of vectors in the given direction in a matrix-free way.
 *          
 *          This function simply calls the Gradient_vec_mult multiple times. 
 *          For some reason it is more efficient than calling it ones and do  
 *          the multiplication together. TODO: think of a more efficient way!
 */
void Gradient_vectors_dir(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                          const int ncol, const double c, const double *x, const int ldi,
                          double *Dx, const int ldo, const int dir, MPI_Comm comm);



/**
 * @brief   Calculate (Gradient + c * I) times a bunch of vectors in the given direction in a matrix-free way.
 *
 * @param dir   Direction of derivatives to take: 0 -- x-dir, 1 -- y-dir, 2 -- z-dir
 */
void Gradient_vec_dir(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                      const int ncol, const double c, const double *x, const int ldi,
                      double *Dx,  const int ldo, const int dir, MPI_Comm comm, const int* dims);



void Calc_DX(
    const double *X,       double *DX,
    const int radius,      const int stride_X,
    const int stride_y_X,  const int stride_y_DX, 
    const int stride_z_X,  const int stride_z_DX,
    const int x_DX_spos,   const int x_DX_epos, 
    const int y_DX_spos,   const int y_DX_epos,
    const int z_DX_spos,   const int z_DX_epos,
    const int x_X_spos,    const int y_X_spos,
    const int z_X_spos,    const double *stencil_coefs,
    const double c
);


#endif // GRADVECROUTINES_H


