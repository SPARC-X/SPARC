/**
 * @file    EVA_CheFSI_CUDA.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for Chebyshev Filtering Subspace Iteration 
 *          with CUDA acceleration
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
 
#ifndef __EVA_CHEFSI_CUDA_H__
#define __EVA_CHEFSI_CUDA_H__

#include <mpi.h>

#include "isddft.h"
#include "EVA_buff.h"

// Perform Chebyshev filtering using CUDA
// Input parameters:
//    pSPARC     : SPARC object
//    DMVertices : Number of grid points on each axis
//    ncol       : Number of column vectors
//    m          : Degree of Chebyshev polynomial 
//    a, b       : Eigenvalues in [a, b] will be damped, b is the estimated largest eigenvalue
//    a0         : Estimated smallest eigenvalue
//    comm       : MPI communicator for halo exchange
//    X          : Vectors to be filtered
// Output parameters:
//    Y          : Filtered vectors
void EVA_Chebyshev_Filtering_CUDA(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, 
    const int m, const double a, const double b, const double a0, 
    MPI_Comm comm, double *X, double *Y
);

#endif
