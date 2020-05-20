/**
 * @file    EVA_CheFSI_CUDA.cu
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for Chebyshev Filtering Subspace Iteration 
 *          with CUDA acceleration
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
#include <math.h>

#include "isddft.h"
#include "EVA_buff.h"
#include "EVA_Lap_MV_orth.h"
#include "EVA_Lap_MV_nonorth.h"
#include "EVA_Vnl_MV.h"

#include "EVA_CheFSI_CUDA_drivers.h"

extern EVA_buff_t EVA_buff;

// Perform Chebyshev filtering
void EVA_Chebyshev_Filtering_CUDA(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, 
    const int m, const double a, const double b, const double a0, 
    MPI_Comm comm, double *X, double *Y
)
{
    const double e = 0.5 * (b - a);
    const double c = 0.5 * (b + a);
    const double a_coef = -0.5;
    const double b_coef = 1.0;
    const double c_coef = -c;

    if (EVA_buff->Vnl_SpMV_init == 0)
    {
        EVA_buff->Vnl_SpMV_init = 1;
        if (EVA_buff->cell_type == 0)
        {
            EVA_init_Lap_MV_orth_params   (pSPARC, DMVertices, ncol, a_coef, b_coef, c_coef, comm, 1, EVA_buff);
        } else {
            EVA_init_Lap_MV_nonorth_params(pSPARC, DMVertices, ncol, a_coef, b_coef, c_coef, comm, 1, EVA_buff);
        }
        EVA_Vnl_SpMV_init(EVA_buff, pSPARC, pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol);
    } else {
        double w2_diag;
        w2_diag  = pSPARC->D2_stencil_coeffs_x[0]; 
        w2_diag += pSPARC->D2_stencil_coeffs_y[0];
        w2_diag += pSPARC->D2_stencil_coeffs_z[0];
        w2_diag  = a_coef * w2_diag + c_coef;
        
        EVA_buff->w2_diag = w2_diag;
        EVA_buff->b = b_coef;       // Don't know why we always need this
    }
    
    if (EVA_buff->CheFSI_CUDA_init == 0)
    {
        EVA_buff->CheFSI_CUDA_init = 1;
        
        CSRPlusMatrix_t CSRP_x2a = EVA_buff->CSRP_x2a;
        CSRPlusMatrix_t CSRP_a2x = EVA_buff->CSRP_a2x;
        int nnz    = CSRP_x2a->row_ptr[CSRP_x2a->nrows];
        
        // Get the number of MPI processes on the same node
        char *host_name = (char*) malloc(sizeof(char) * 128);
        int  shm_nproc, shm_rank, len;
        MPI_Comm shm_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, EVA_buff->my_global_rank, MPI_INFO_NULL, &shm_comm);
        MPI_Comm_size(shm_comm, &shm_nproc);
        MPI_Comm_rank(shm_comm, &shm_rank);
        MPI_Comm_free(&shm_comm);
        MPI_Get_processor_name(host_name, &len);
        
        EVA_CheFSI_CUDA_init_device_buffer(
            EVA_buff->my_global_rank, EVA_buff->FDn, DMVertices, 
            EVA_buff->alpha_length, ncol, nnz, EVA_buff->cell_type, EVA_buff->alpha_scale,
            CSRP_x2a->row_ptr, CSRP_x2a->col, CSRP_x2a->val,
            CSRP_a2x->row_ptr, CSRP_a2x->col, CSRP_a2x->val,
            EVA_buff->Lap_stencils, EVA_buff->Lap_wt_nonorth, EVA_buff->Grad_stencils,
            EVA_buff->ijk_se_displs, EVA_buff->ijk_se_displs_nonorth, EVA_buff->cart_array + 3,
            shm_nproc, shm_rank, host_name
        );
        
        free(host_name);
    }
    
    double *v = pSPARC->Veff_loc_dmcomm;
    EVA_CheFSI_CUDA_ChebvsheyFiltering(
        m, a, b, a0, EVA_buff->w2_diag, 
        EVA_buff->b, X, v, Y
    );
}
