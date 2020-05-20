/**
 * @file    EVA_Hamil_MV.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Hamiltonian matrix with multiple vectors
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
 
#ifndef __EVA_HAMIL_MV_H__
#define __EVA_HAMIL_MV_H__

#include "isddft.h"
#include "EVA_buff.h"

// Calculate (Hamiltonian + c * I) * x in a SpMV way, x is a bunch of vectors.
// This function is called with the same ANI and nlocProj; for different ANI
// and nlocProj, call EVA_Hamil_MV_Vnl_SpMV_reset() first.
// Input parameters:
//    pSPARC     : SPARC object
//    DMnd       : Total number of grid points in the original domain
//    DMVertices : Number of grid points on each axis
//    ncol       : Number of column vectors
//    c          : Value for scaling the I matrix
//    Veff_loc   : Diagonal of Veff matrix
//    ANI        : Atom nloc influence object 
//    nlocProj   : Nloc proj object
//    comm       : MPI communicator for halo exchange
//    x          : Input vectors (current values of the original domain grid points)
// Output parameters:
//    Hx         : (H + c * I) * x
void EVA_Hamil_MatVec(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, const double *Veff_loc, 
    ATOM_NLOC_INFLUENCE_OBJ *ANI, NLOC_PROJ_OBJ *nlocProj,
    double *x, double *Hx, MPI_Comm comm
);

// Add timing information from the original functions
void EVA_buff_timer_add(double cpyx_t, double pack_t, double comm_t, double unpk_t, double krnl_t, double Vnl_t);

// Add number of RHS vectors from the original functions
void EVA_buff_rhs_add(int Lap_MV_rhs, int Vnl_MV_rhs);

#endif
