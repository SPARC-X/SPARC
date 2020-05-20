/**
 * @file    EVA_Hamil_MV.c
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Hamiltonian matrix with multiple vectors
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>

#ifdef USE_EVA_MKL_MODULE
#include <mkl.h>
#include <mkl_spblas.h>
#endif

#include "isddft.h"
#include "nlocVecRoutines.h"

#include "EVA_buff.h"
#include "EVA_Lap_MV_orth.h"
#include "EVA_Lap_MV_nonorth.h"
#include "EVA_Vnl_MV.h"
#include "EVA_Hamil_MV.h"

extern EVA_buff_t EVA_buff;

// Calculate (Hamiltonian + c * I) * x in a SpMV way, x is a bunch of vectors.
// This function is called with the same ANI and nlocProj; for different ANI
// and nlocProj, call EVA_Hamil_MV_Vnl_SpMV_reset() first.
void EVA_Hamil_MatVec(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, const double *Veff_loc, 
    ATOM_NLOC_INFLUENCE_OBJ *ANI, NLOC_PROJ_OBJ *nlocProj,
    double *x, double *Hx, MPI_Comm comm
)
{
    int nproc;
    MPI_Comm_size(comm, &nproc);
    
    if (EVA_buff == NULL) EVA_buff_init(pSPARC->order, pSPARC->cell_typ);
    
    int DMnx = 1 - DMVertices[0] + DMVertices[1];
    int DMny = 1 - DMVertices[2] + DMVertices[3];
    int DMnz = 1 - DMVertices[4] + DMVertices[5];
    int _DMnd = DMnx * DMny * DMnz;
    if (_DMnd != DMnd)
    {
        printf(RED"FATAL ERROR : DMnd != DMnx * DMny * DMnz in Hamiltonian_vectors_mult() !!!\n");
        assert(EVA_buff->DMnd == DMnd);
    }
    
    int Lap_MV_ncol = 1;
    MPI_Comm nonorth_halo_comm = pSPARC->comm_dist_graph_psi;
    
    if (pSPARC->cell_typ == 0)
    {
        EVA_init_Lap_MV_orth_params   (pSPARC, DMVertices, Lap_MV_ncol, -0.5, 1.0, c, comm, nproc, EVA_buff);
    } else {
        EVA_init_Lap_MV_nonorth_params(pSPARC, DMVertices, Lap_MV_ncol, -0.5, 1.0, c, comm, nproc, EVA_buff);
    }
    
    if (EVA_buff->Vnl_SpMV_init == 0)
    {
        EVA_buff->Vnl_SpMV_init = 1;
        EVA_Vnl_SpMV_init(EVA_buff, pSPARC, ANI, nlocProj, ncol);
    }
    
    if (pSPARC->cell_typ == 0)
    {
        for (int icol = 0; icol < ncol; icol++)
        {
            double *x_i  = x  + icol * DMnd;
            double *Hx_i = Hx + icol * DMnd;
            EVA_Lap_MV_orth(EVA_buff, Lap_MV_ncol, x_i, Veff_loc, comm, nproc, Hx_i);
        }
    } else {
        for (int icol = 0; icol < ncol; icol++)
        {
            double *x_i  = x  + icol * DMnd;
            double *Hx_i = Hx + icol * DMnd;
            EVA_Lap_MV_nonorth(EVA_buff, Lap_MV_ncol, x_i, Veff_loc, nonorth_halo_comm, nproc, Hx_i);
        }
    }
    
    // If we are using only 1 thread / process, use the original Vnl_vec_mult for better performance
    if (EVA_buff->nthreads > 1) 
    {
        EVA_Vnl_SpMV(EVA_buff, ncol, x, comm, nproc, Hx);
    } else {
        double st = MPI_Wtime();
        Vnl_vec_mult(pSPARC, DMnd, ANI, nlocProj, ncol, x, Hx, comm);
        double et = MPI_Wtime();
        EVA_buff->Vnl_MV_t   += et - st;
        EVA_buff->Vnl_MV_rhs += ncol;
    }
}

// Add timing information from the original functions
void EVA_buff_timer_add(double cpyx_t, double pack_t, double comm_t, double unpk_t, double krnl_t, double Vnl_t)
{
    EVA_buff->Lap_MV_cpyx_t += cpyx_t;
    EVA_buff->Lap_MV_pack_t += pack_t;
    EVA_buff->Lap_MV_comm_t += comm_t;
    EVA_buff->Lap_MV_unpk_t += unpk_t;
    EVA_buff->Lap_MV_krnl_t += krnl_t;
    EVA_buff->Vnl_MV_t      += Vnl_t;
}

// Add number of RHS vectors from the original functions
void EVA_buff_rhs_add(int Lap_MV_rhs, int Vnl_MV_rhs)
{
    EVA_buff->Lap_MV_rhs += Lap_MV_rhs;
    EVA_buff->Vnl_MV_rhs += Vnl_MV_rhs;
}
