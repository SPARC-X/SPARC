/**
 * @file    EVA_buff.c
 * @brief   External Vectorization and Acceleration (EVA) module 
 *          EVA_buff structure for storing intermediate variables in EVA module
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

#include "isddft.h"
#include "EVA_buff.h"
#include "EVA_CheFSI_CUDA_drivers.h"

// EVA_buff is shared among EVA_* files
EVA_buff_t EVA_buff = NULL;

// Initialize EVA_buff
void EVA_buff_init(const int order, const int cell_type)
{
    if (EVA_buff != NULL) return;
    
    // Create EVA_buff object
    EVA_buff = (EVA_buff_t) malloc(sizeof(struct EVA_buff_));
    assert(EVA_buff != NULL);
    
    EVA_buff->nthreads = omp_get_max_threads();
    EVA_buff->order    = order;
    EVA_buff->FDn      = order / 2;
    
    // Allocate space for arrays
    int FDn1 = EVA_buff->FDn + 1;
    EVA_buff->cart_array            = (int*)    malloc(sizeof(int) * 9);
    EVA_buff->ijk_se_displs         = (int*)    malloc(sizeof(int) * 6 * 16);
    EVA_buff->ijk_se_displs_nonorth = (int*)    malloc(sizeof(int) * 26 * 17);
    EVA_buff->Lap_stencils          = (double*) malloc(sizeof(double) * 3 * FDn1);
    EVA_buff->Lap_wt_nonorth        = (double*) malloc(sizeof(double) * 5 * FDn1);
    EVA_buff->Grad_stencils         = (double*) malloc(sizeof(double) * 8 * FDn1);
    assert(EVA_buff->cart_array            != NULL);
    assert(EVA_buff->ijk_se_displs         != NULL);
    assert(EVA_buff->ijk_se_displs_nonorth != NULL);
    assert(EVA_buff->Lap_stencils          != NULL);
    assert(EVA_buff->Lap_wt_nonorth        != NULL);
    assert(EVA_buff->Grad_stencils         != NULL);
    
    EVA_buff->x_ex               = NULL;
    EVA_buff->x_in               = NULL;
    EVA_buff->x_out              = NULL;
    EVA_buff->Dx1                = NULL;
    EVA_buff->Dx2                = NULL;
    EVA_buff->alpha              = NULL;
    EVA_buff->alpha_scale        = NULL;
    EVA_buff->Ynew               = NULL;
    EVA_buff->x_ex_msize         = 0;
    EVA_buff->halo_msize         = 0;
    EVA_buff->Dx1_msize          = 0;
    EVA_buff->Dx2_msize          = 0;
    EVA_buff->alpha_msize        = 0;
    EVA_buff->Ynew_size          = 0;
    EVA_buff->Vnl_SpMV_init      = 0;
    
    EVA_buff->CSRP_x2a           = NULL;
    EVA_buff->CSRP_a2x           = NULL;
    EVA_buff->Vnl_Hx_tmp         = NULL;
    EVA_buff->Vnl_Hx_tmp_msize   = 0;
    
    EVA_buff->mkl_x2a_sp_mat_p   = NULL;
    EVA_buff->mkl_a2x_sp_mat_p   = NULL;
    EVA_buff->mkl_x2a_mat_desc_p = NULL;
    EVA_buff->mkl_a2x_mat_desc_p = NULL;
    
    EVA_buff->CheFSI_CUDA_init   = 0;
    
    // Reset statis info
    EVA_buff->Lap_MV_cpyx_t      = 0.0;
    EVA_buff->Lap_MV_pack_t      = 0.0;
    EVA_buff->Lap_MV_comm_t      = 0.0;
    EVA_buff->Lap_MV_unpk_t      = 0.0;
    EVA_buff->Lap_MV_krnl_t      = 0.0;
    EVA_buff->Vnl_MV_t           = 0.0;
    EVA_buff->Lap_MV_rhs         = 0;
    EVA_buff->Vnl_MV_rhs         = 0;
    
    int use_EVA = 0;
    char *use_EVA_p = getenv("EXT_VEC_ACCEL");
    if (use_EVA_p != NULL) use_EVA = atoi(use_EVA_p);
    else use_EVA = 0;
    if (use_EVA != 1) use_EVA = 0;
    //if (cell_type != 0) use_EVA = 0;
    EVA_buff->use_EVA = use_EVA;
    EVA_buff->cell_type = cell_type;
    
    int use_EVA_CUDA = 0;
    #ifdef USE_EVA_CUDA_MODULE
    char *use_EVA_CUDA_p = getenv("EXT_VEC_ACCEL_CUDA");
    if (use_EVA_CUDA_p != NULL) use_EVA_CUDA = atoi(use_EVA_CUDA_p);
    else use_EVA_CUDA = 0;
    if (use_EVA_CUDA != 1) use_EVA_CUDA = 0;
    #endif
    EVA_buff->use_EVA_CUDA = use_EVA_CUDA && use_EVA;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &EVA_buff->my_global_rank);
    if (EVA_buff->my_global_rank == 0) 
    {
        printf("External Vectorization and Acceleration (EVA) module initialized, status:\n");
        
        printf("    * EVA module functions: ");
        if (EVA_buff->use_EVA == 1) printf("active\n"); 
        else printf("inactive, report timing results collected from original SPARC functions\n");
        
        printf("    * EVA module CUDA kernels: ");
        #ifdef USE_EVA_CUDA_MODULE
        printf("compiled, ");
        if (EVA_buff->use_EVA_CUDA == 1) printf("active\n");
        else printf("inactive\n");
        #else
        printf("not compiled, inactive\n");
        #endif
    }
}


// Free EVA_buff and print statistic info
void EVA_buff_finalize()
{
    // Rank 0 print statistic info
    if (EVA_buff->my_global_rank == 0)
    {
        printf("\n");
        printf("========== External Vectorization and Acceleration (EVA) module statistic ==========\n");
        printf("===================== EVA module last modified: April 25, 2019 =====================\n");
        printf("Laplacian operator RHS vectors = %d\n", EVA_buff->Lap_MV_rhs);
        printf("    * Copying x --> x_ex        = %.4lf s\n", EVA_buff->Lap_MV_cpyx_t);
        printf("    * Packing send halo data    = %.4lf s\n", EVA_buff->Lap_MV_pack_t);
        printf("    * MPI_Wait for halo comm    = %.4lf s\n", EVA_buff->Lap_MV_comm_t);
        printf("    * Unpacking recv halo data  = %.4lf s\n", EVA_buff->Lap_MV_unpk_t);
        printf("    * Laplacian stencil kernel  = %.4lf s\n", EVA_buff->Lap_MV_krnl_t);
        printf("Vnl operator RHS vectors = %d\n",      EVA_buff->Vnl_MV_rhs);
        printf("    * Vnl  kernel               = %.4lf s\n", EVA_buff->Vnl_MV_t);
        printf("====================================================================================\n");
        printf("\n");
    }
    
    // Free array buffers
    free(EVA_buff->cart_array);
    free(EVA_buff->ijk_se_displs);
    free(EVA_buff->ijk_se_displs_nonorth);
    free(EVA_buff->Lap_stencils);
    free(EVA_buff->Lap_wt_nonorth);
    free(EVA_buff->Grad_stencils);
    free(EVA_buff->x_ex);
    free(EVA_buff->x_in);
    free(EVA_buff->x_out);
    free(EVA_buff->Dx1);
    free(EVA_buff->Dx2);
    free(EVA_buff->alpha);
    free(EVA_buff->alpha_scale);
    free(EVA_buff->Ynew);

    free(EVA_buff->mkl_x2a_sp_mat_p);
    free(EVA_buff->mkl_a2x_sp_mat_p);
    free(EVA_buff->mkl_x2a_mat_desc_p);
    free(EVA_buff->mkl_a2x_mat_desc_p);
    
    CSRP_free(EVA_buff->CSRP_x2a);
    CSRP_free(EVA_buff->CSRP_a2x);
    free(EVA_buff->CSRP_x2a);
    free(EVA_buff->CSRP_a2x);
    
    #ifdef USE_EVA_CUDA_MODULE
    if (EVA_buff->use_EVA_CUDA == 1) EVA_CheFSI_CUDA_free_device_buffer();
    #endif
    
    // Free EVA_buff object
    free(EVA_buff);
    EVA_buff = NULL;
}
