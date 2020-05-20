/**
 * @file    EVA_Lap_MV_orth.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Laplacian matrix with multiple vectors
 *          For orthogonal cells
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#ifndef __EVA_LAP_MV_ORTH_H__
#define __EVA_LAP_MV_ORTH_H__

#include <mpi.h>

#include "isddft.h"
#include "EVA_buff.h"

// ********** For orthogonal cell ********** 

// Initialize parameters used in EVA_Lap_MV_orth
// Input parameters:
//    pSPARC     : SPARC object
//    DMVertices : Number of grid points on each axis
//    ncol       : Number of column vectors
//    a          : Value for scaling the Laplacian
//    b          : Value for scaling the v matrix
//    c          : Value for scaling the I matrix
//    comm       : MPI communicator for halo exchange
//    nproc      : Number of processes in comm
// Output parameters:
//    EVA_buff   : Updated halo exchange parameters in EVA_buff
void EVA_init_Lap_MV_orth_params(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, const double a,
    const double b, const double c, MPI_Comm comm, const int nproc, EVA_buff_t EVA_buff
);


// Copy inner part of x to x_ex
// Input parameters:
//    EVA_buff   : EVA_buff with updated halo exchange parameters
//    ncol       : Number of column vectors
//    x          : Input vectors (current values of the original domain grid points)
// Output parameters:
//    EVA_buff   : Updated EVA_buff->x_ex
void EVA_Lap_MV_orth_copy_inner_x(EVA_buff_t EVA_buff, const int ncol, const double *x);


// Perform halo exchange and copy the received halo values to the extended 
// domain when there are more than 1 MPI processes in the communicator
// Input parameters:
//    EVA_buff   : EVA_buff with updated halo exchange parameters
//    ncol       : Number of column vectors
//    x          : Input vectors (current values of the original domain grid points)
//    comm       : MPI communicator for halo exchange
// Output parameters:
//    EVA_buff   : Updated EVA_buff->x_ex
void EVA_Lap_MV_orth_NPK_Halo(EVA_buff_t EVA_buff, const int ncol, const double *x, MPI_Comm comm);


// Set up halo values in the extended domain when there is 1 MPI process in the communicator
// Input parameters:
//    EVA_buff   : EVA_buff with updated halo exchange parameters
//    ncol       : Number of column vectors
//    x          : Input vectors (current values of the original domain grid points)
// Output parameters:
//    EVA_buff   : Updated EVA_buff->x_ex
void EVA_Lap_MV_orth_NP1_Halo(EVA_buff_t EVA_buff, const int ncol, const double *x);

// Do the computation of (a * Lap + b * diag(v) + c * I) * x
// Input parameters:
//    EVA_buff   : EVA_buff with updated halo exchange parameters
//    ncol       : Number of column vectors
//    v          : Values of the diagonal matrix, should have the same size as x
// Output parameters:
//    Hx         : (a * Lap + b * diag(v) + c * I) * x
void EVA_Lap_MV_orth_stencil_kernel(EVA_buff_t EVA_buff, const int ncol, const double *v, double *Hx);


// Calculate (a * Lap + b * diag(v) + c * I) * x in a matrix-free way
// Input parameters:
//    EVA_buff   : EVA_buff with updated halo exchange parameters
//    ncol       : Number of column vectors
//    x          : Input vectors (current values of the original domain grid points)
//    v          : Values of the diagonal matrix, should have the same size as x
//    comm       : MPI communicator for halo exchange
//    nproc      : Number of MPI processes in comm
// Output parameters:
//    Hx         : (a * Lap + b * diag(v) + c * I) * x
void EVA_Lap_MV_orth(
    EVA_buff_t EVA_buff, const int ncol, const double *x, const double *v, 
    MPI_Comm comm, const int nproc, double *Hx
);

#endif
