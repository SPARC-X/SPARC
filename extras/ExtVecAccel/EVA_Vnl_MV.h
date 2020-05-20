/**
 * @file    EVA_Vnl_MV.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Vnl operator with multiple vectors
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#ifndef __EVA_VNL_MV_H__
#define __EVA_VNL_MV_H__

#include <mpi.h>

#include "isddft.h"
#include "EVA_buff.h"

// Initialize EVA Vnl SpMV buffer
// Input parameters:
//    pSPARC   : SPARC object
//    ANI      : ATOM_NLOC_INFLUENCE_OBJ used in Vnl operator
//    nlocProj : NLOC_PROJ_OBJ used in Vnl operator
//    ncol     : Number of column vectors
// Output parameters:
//    EVA_buff : Initialized EVA Vnl SpMV buffer in EVA_buff
void EVA_Vnl_SpMV_init(
    EVA_buff_t EVA_buff, const SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ *ANI, 
    NLOC_PROJ_OBJ *nlocProj, const int ncol
);

// Compute Vnl operator times vectors in a SpMV way
// Input parameters:
//    EVA_buff : Initialized EVA Vnl SpMV buffer in EVA_buff
//    ncol     : Number of column vectors
//    x        : Input vectors (current values of the original domain grid points)
//    comm     : MPI communicator for halo exchange
//    nproc    : Number of MPI processes in comm
// Output parameters:
//    Hx       : Initialized EVA Vnl SpMV buffer in EVA_buff
void EVA_Vnl_SpMV(
    EVA_buff_t EVA_buff, const int ncol, const double *x, 
    MPI_Comm comm, const int nproc, double *Hx
);

// Reset buffers used in EVA_Vnl_SpMV 
// Input & output parameter:
//    EVA_buff : Reset EVA Vnl SpMV buffer in EVA_buff
void EVA_Vnl_SpMV_reset(EVA_buff_t EVA_buff);

#endif
