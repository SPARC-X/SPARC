/**
 * @file    EVA_Lap_MV_nonorth.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Laplacian matrix with multiple vectors
 *          For nonorthogonal cells
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#ifndef __EVA_LAP_MV_NONORTH_H__
#define __EVA_LAP_MV_NONORTH_H__

#include <mpi.h>

#include "isddft.h"
#include "EVA_buff.h"

// ********** For nonorthogonal cell ********** 

// Initialize parameters used in EVA_Lap_MV_nonorth
// Input parameters:
//    pSPARC     : SPARC object
//    DMVertices : Number of grid points on each axis
//    ncol       : Number of column vectors
//    a          : Value for scaling the Laplacian
//    b          : Value for scaling the v matrix
//    c          : Value for scaling the I matrix
//    comm       : MPI communicator of process grid
//    nproc      : Number of processes in comm
// Output parameters:
//    EVA_buff   : Updated halo exchange parameters in EVA_buff
void EVA_init_Lap_MV_nonorth_params(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, 
    const double a, const double b, const double c, MPI_Comm comm,
    const int nproc, EVA_buff_t EVA_buff
);

// Calculate (a * Lap + b * diag(v) + c * I) * x in a matrix-free way
// Input parameters:
//    EVA_buff   : EVA_buff with updated halo exchange parameters
//    ncol       : Number of column vectors
//    x          : Input vectors (current values of the original domain grid points)
//    v          : Values of the diagonal matrix, should have the same size as x
//    comm       : MPI communicator for halo exchange (not the one used in EVA_init_Lap_MV_nonorth_params)
//    nproc      : Number of MPI processes in comm
// Output parameters:
//    Hx         : (a * Lap + b * diag(v) + c * I) * x
void EVA_Lap_MV_nonorth(
    EVA_buff_t EVA_buff, const int ncol, const double *x, const double *v, 
    MPI_Comm comm, const int nproc, double *Hx
);

#endif
