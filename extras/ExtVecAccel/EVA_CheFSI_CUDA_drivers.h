/**
 * @file    EVA_CheFSI_CUDA_driver.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          CUDA driver functions
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
 
#ifndef __EVA_CHEFSI_CUDA_DRIVERS_H__
#define __EVA_CHEFSI_CUDA_DRIVERS_H__

void EVA_CheFSI_CUDA_init_device_buffer(
    const int my_rank, const int FDn, const int *DMVertices, const int alen, 
    const int ncol, const int nnz, const int cell_type, const double *alpha_scale, 
    const int    *x2a_row_ptr,   const int    *x2a_col_idx,           const double *x2a_val, 
    const int    *a2x_row_ptr,   const int    *a2x_col_idx,           const double *a2x_val,
    const double *Lap_stencils,  const double *Lap_stencils_nonorth,  const double *Grad_stencils, 
    const int    *ijk_se_displs, const int    *ijk_se_displs_nonorth, const int    *periods,
    const int    shm_nproc,      const int    shm_rank,               const char   *host_name
);

void EVA_CheFSI_CUDA_free_device_buffer();

void EVA_CheFSI_CUDA_ChebvsheyFiltering(
    const int m, const double a, const double b, const double a0, 
    const double coef_0, const double b_coef, 
    const double *X, const double *v, double *Y
);

#endif
