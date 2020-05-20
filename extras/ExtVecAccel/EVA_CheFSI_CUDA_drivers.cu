/**
 * @file    EVA_CheFSI_CUDA_driver.cu
 * @brief   External Vectorization and Acceleration (EVA) module
 *          CUDA driver functions
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

// Note: This file should be compiled by NVCC and should not use other
//       functions that is not compiled by NVCC from the EVA module. 
//       However, functions in this file will be called by other functions
//       that are not compiled by NVCC from the EVA module.
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "CUDA_Utils.h"

extern "C" {
#include "EVA_CheFSI_CUDA_drivers.h"
}

#include "EVA_CheFSI_CUDA_buff.h"
#include "EVA_CheFSI_CUDA_kernels.cuh"

EVA_CUDA_buff_t EVA_CUDA_buff = NULL;

static double get_wtime_sec()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

extern "C"
void EVA_CheFSI_CUDA_init_device_buffer(
    const int my_rank, const int FDn, const int *DMVertices, const int alen, 
    const int ncol, const int nnz, const int cell_type, const double *alpha_scale, 
    const int    *x2a_row_ptr,   const int    *x2a_col_idx,           const double *x2a_val, 
    const int    *a2x_row_ptr,   const int    *a2x_col_idx,           const double *a2x_val,
    const double *Lap_stencils,  const double *Lap_stencils_nonorth,  const double *Grad_stencils, 
    const int    *ijk_se_displs, const int    *ijk_se_displs_nonorth, const int    *periods,
    const int    shm_nproc,      const int    shm_rank,               const char   *host_name
)
{
    double st = get_wtime_sec();
    
    if (EVA_CUDA_buff == NULL)
    {
        EVA_CUDA_buff = (EVA_CUDA_buff_t) malloc(sizeof(struct EVA_CUDA_buff_));
    } else {
        if (my_rank == 0) printf("[WARNING] EVA_CUDA_buff has been allocated and initialized!\n");
    }
    EVA_CUDA_buff->my_rank = my_rank;
    
    EVA_CUDA_buff->cell_type = cell_type;
    
    size_t GPU_memsize = 0.0;
    
    // 0. Set GPU device used by this process
    int ndevice, mydevice, nproc_dev;
    cudaCheck( cudaGetDeviceCount(&ndevice) );
    mydevice  = shm_rank % ndevice;
    nproc_dev = (shm_nproc + ndevice - 1) / ndevice;
    cudaCheck( cudaSetDevice(mydevice) );
    if ((shm_nproc % ndevice) && (ndevice > 1))
    {
        printf("[WARNING] Host %s: number of MPI process %d is not a multiple of number of CUDA device %d\n", host_name, shm_nproc, ndevice);
        printf("[WARNING] Host %s: local MPI process rank %d will use CUDA device %d\n", host_name, shm_rank, mydevice);
    }
    
    // Calculate required GPU memory size first, then allocate later
    
    // 1. Copy parameters
    int nx   = 1 - DMVertices[0] + DMVertices[1];
    int ny   = 1 - DMVertices[2] + DMVertices[3];
    int nz   = 1 - DMVertices[4] + DMVertices[5];
    int nxyz = nx * ny * nz;
    EVA_CUDA_buff->CheFSI_FDn  = FDn;
    EVA_CUDA_buff->CheFSI_nx   = nx;
    EVA_CUDA_buff->CheFSI_ny   = ny;
    EVA_CUDA_buff->CheFSI_nz   = nz;
    EVA_CUDA_buff->CheFSI_nxyz = nxyz;
    EVA_CUDA_buff->CheFSI_alen = alen;
    EVA_CUDA_buff->CheFSI_ncol = ncol;
    EVA_CUDA_buff->CheFSI_nnz  = nnz;
    EVA_CUDA_buff->CheFSI_npts = nxyz * ncol;
    
    // 2. Laplacian stencil coefficients 
    int FDn1 = FDn + 1;
    const double *Lap_x_orth = Lap_stencils;
    const double *Lap_y_orth = Lap_x_orth + FDn1;
    const double *Lap_z_orth = Lap_y_orth + FDn1;
    GPU_memsize += 3 * FDn1 * sizeof(double);
    
    // 3. Arrays for Laplacian operator & Chebyshev Filtering
    int nxyz_ex = (nx + 2 * FDn) * (ny + 2 * FDn) * (nz + 2 * FDn);
    size_t nxyz_memsize      = sizeof(double) * nxyz;
    size_t npts_memsize      = sizeof(double) * nxyz * ncol;
    size_t npts_ex_memsize   = sizeof(double) * nxyz_ex * ncol;
    size_t ijk_se_memsize    = sizeof(int)    * 6 * 12;
    size_t ijk_se_no_memsize = sizeof(int)    * 26 * 17;
    size_t periods_memsize   = sizeof(int)    * 3;
    GPU_memsize += nxyz_memsize + npts_ex_memsize + 3 * npts_memsize;
    
    // Allocate Dx1 and Dx2
    int nx_ex = nx + 2 * FDn;
    int ny_ex = ny + 2 * FDn;
    int nz_ex = nz + 2 * FDn;
    int DMnxny = nx * ny;
    int DMnxexny = nx_ex * ny;
    int DMnxnyex = nx * ny_ex;
    int DMnd_xex = DMnxexny * nz;
    int DMnd_yex = DMnxnyex * nz;
    int DMnd_zex = DMnxny * nz_ex;
    int Dx1_len  = 0;
    switch (cell_type)
    {
        case 11: Dx1_len = DMnd_xex; break;  // df/dy  
        case 12: Dx1_len = DMnd_xex; break;  // df/dz
        case 13: Dx1_len = DMnd_yex; break;  // df/dz
        case 14: Dx1_len = DMnd_xex; break;  // 2*T_12*df/dy + 2*T_13*df/dz
        case 15: Dx1_len = DMnd_zex; break;  // 2*T_13*dV/dx + 2*T_23*dV/dy
        case 16: Dx1_len = DMnd_yex; break;  // 2*T_12*dV/dx + 2*T_23*dV/dz
        case 17: Dx1_len = DMnd_xex; break;  // 2*T_12*df/dy + 2*T_13*df/dz
    }
    size_t Dx1_memsize = sizeof(double) * ncol * Dx1_len;
    size_t Dx2_memsize = sizeof(double) * ncol * DMnd_yex;
    GPU_memsize += Dx1_memsize + Dx2_memsize;
    
    // 4. Arrays for Vnl_SpMV on GPU
    int alen1 = alen + 1;
    int nxyz1 = nxyz + 1;
    size_t alen1_int = alen1 * sizeof(int);
    size_t nxyz1_int = nxyz1 * sizeof(int);
    size_t nnz_int   = nnz   * sizeof(int);
    size_t alen_dbl  = alen  * sizeof(double);
    size_t nnz_dbl   = nnz   * sizeof(double);
    GPU_memsize += alen1_int + 2 * (nnz_int + nnz_dbl) + nxyz1_int + alen_dbl * (ncol + 1);
    
    // 5. Allocate and initialize CUDA arrays and handles
    size_t free_msize, total_msize;
    cudaCheck( cudaMemGetInfo(&free_msize, &total_msize));
    if (GPU_memsize * nproc_dev > free_msize)
    {
        double proc_mem_MB  = (double) GPU_memsize / 1048576.0;
        double total_mem_MB = (double) nproc_dev * proc_mem_MB;
        double free_mem_MB  = (double) free_msize  / 1048576.0;
        printf(
            "[FATAL] Host %s: requires %.3lf * %d = %.3lf MB GPU memory but only %.3lf MB available, please reduce the PPN and run again!\n", 
            host_name, proc_mem_MB, nproc_dev, total_mem_MB, free_mem_MB
        );
        exit(EXIT_FAILURE);
    }
    
    cudaCheck( cudaMemcpyToSymbol(cu_Lap_x_orth, Lap_x_orth, sizeof(double) * FDn1) );
    cudaCheck( cudaMemcpyToSymbol(cu_Lap_y_orth, Lap_y_orth, sizeof(double) * FDn1) );
    cudaCheck( cudaMemcpyToSymbol(cu_Lap_z_orth, Lap_z_orth, sizeof(double) * FDn1) );
    cudaCheck( cudaMemcpyToSymbol(cu_Lap_wt_nonorth,  Lap_stencils_nonorth, sizeof(double) * FDn1 * 5) );
    cudaCheck( cudaMemcpyToSymbol(cu_Grad_wt_nonorth, Grad_stencils,        sizeof(double) * FDn1 * 8) );
    
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_v,    nxyz_memsize)    ); 
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_X_ex, npts_ex_memsize) ); 
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_X,    npts_memsize)    ); 
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_Y,    npts_memsize)    ); 
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_Ynew, npts_memsize)    ); 
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_Dx1,  Dx1_memsize)     ); 
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_Dx2,  Dx2_memsize)     ); 
    EVA_CUDA_buff->ijk_se_displs = (int*) malloc(ijk_se_memsize);
    EVA_CUDA_buff->periods       = (int*) malloc(periods_memsize);
    EVA_CUDA_buff->ijk_se_displs_nonorth = (int*) malloc(ijk_se_no_memsize);
    memcpy(EVA_CUDA_buff->ijk_se_displs, ijk_se_displs, ijk_se_memsize);
    memcpy(EVA_CUDA_buff->periods,       periods,       periods_memsize);
    memcpy(EVA_CUDA_buff->ijk_se_displs_nonorth, ijk_se_displs_nonorth, ijk_se_no_memsize);
    
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_x2a_row_ptr, alen1_int) );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_x2a_col_idx, nnz_int)   );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_x2a_val,     nnz_dbl)   );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_a2x_row_ptr, nxyz1_int) );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_a2x_col_idx, nnz_int)   );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_a2x_val,     nnz_dbl)   );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_alpha_scale, alen_dbl)  );
    cudaCheck( cudaMalloc(&EVA_CUDA_buff->cu_alpha,       alen_dbl * ncol) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_x2a_row_ptr, x2a_row_ptr, alen1_int, cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_x2a_col_idx, x2a_col_idx, nnz_int,   cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_x2a_val,     x2a_val,     nnz_dbl,   cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_a2x_row_ptr, a2x_row_ptr, nxyz1_int, cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_a2x_col_idx, a2x_col_idx, nnz_int,   cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_a2x_val,     a2x_val,     nnz_dbl,   cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(EVA_CUDA_buff->cu_alpha_scale, alpha_scale, alen_dbl,  cudaMemcpyHostToDevice) );
    
    cudaCheck( cudaStreamCreate(&EVA_CUDA_buff->cu_stream) );
    cudaSparseCheck( cusparseCreate(&EVA_CUDA_buff->cusparse_handle) );
    cudaSparseCheck( cusparseCreateMatDescr(&EVA_CUDA_buff->cusparse_mat_x2a) );
    cudaSparseCheck( cusparseCreateMatDescr(&EVA_CUDA_buff->cusparse_mat_a2x) );
    cudaSparseCheck( cusparseSetStream(EVA_CUDA_buff->cusparse_handle, EVA_CUDA_buff->cu_stream) );
    cudaSparseCheck( cusparseSetMatType(EVA_CUDA_buff->cusparse_mat_x2a, CUSPARSE_MATRIX_TYPE_GENERAL)  );
    cudaSparseCheck( cusparseSetMatType(EVA_CUDA_buff->cusparse_mat_a2x, CUSPARSE_MATRIX_TYPE_GENERAL)  );
    cudaSparseCheck( cusparseSetMatIndexBase(EVA_CUDA_buff->cusparse_mat_x2a, CUSPARSE_INDEX_BASE_ZERO) );
    cudaSparseCheck( cusparseSetMatIndexBase(EVA_CUDA_buff->cusparse_mat_a2x, CUSPARSE_INDEX_BASE_ZERO) );
    
    // 6. Initialize timers
    for (int i = 0; i < 6; i++)
    {
        cudaCheck( cudaEventCreate(&EVA_CUDA_buff->cu_events[i]) );
        EVA_CUDA_buff->cu_timers[i] = 0;
    }
    
    EVA_CUDA_buff->mem_move_size = 0;
    EVA_CUDA_buff->numHamil = 0;
    
    double proc_mem_MB = (double) GPU_memsize / 1048576.0;
    double init_t = get_wtime_sec() - st;
    if (my_rank == 0) 
    {
        printf("EVA CUDA module initialized, GPU memory usage = %.2lf (MB), ", proc_mem_MB);
        printf("used time = %.3lf (s) \n", init_t);
    }
}

extern "C"
void EVA_CheFSI_CUDA_free_device_buffer()
{
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_v)    ); 
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_X_ex) ); 
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_X)    ); 
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_Y)    ); 
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_Ynew) ); 
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_Dx1)  ); 
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_Dx2)  ); 
    
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_x2a_row_ptr) );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_x2a_col_idx) );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_x2a_val)     );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_a2x_row_ptr) );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_a2x_col_idx) );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_a2x_val)     );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_alpha_scale) );
    cudaCheck( cudaFree(EVA_CUDA_buff->cu_alpha)       );
    
    cudaSparseCheck( cusparseDestroy(EVA_CUDA_buff->cusparse_handle)          );
    cudaSparseCheck( cusparseDestroyMatDescr(EVA_CUDA_buff->cusparse_mat_x2a) );
    cudaSparseCheck( cusparseDestroyMatDescr(EVA_CUDA_buff->cusparse_mat_a2x) );
    
    float *cu_timers = EVA_CUDA_buff->cu_timers;
    double mem_move_MB = (double) EVA_CUDA_buff->mem_move_size / 1048576.0;
    if (EVA_CUDA_buff->my_rank == 0)
    {
        printf("\n");
        printf("==================== EVA CUDA module statistic ====================\n");
        printf("========== EVA CUDA module last modified: April 25, 2019 ==========\n");
        printf("Host <---> GPU data transfer: %.2lf MB, %.2f (ms)\n", mem_move_MB, cu_timers[0]);
        printf("CheFSI Hamiltonian MatVec on GPU: %d * %d RHS columns\n", EVA_CUDA_buff->numHamil, EVA_CUDA_buff->CheFSI_ncol);
        printf("    * Lap operator halo setup     = %.2f (ms)\n", cu_timers[1]);
        printf("    * Lap operator stencil kernel = %.2f (ms)\n", cu_timers[2]);
        printf("    * Vnl operator SpMV kernel    = %.2f (ms)\n", cu_timers[3]);
        printf("    * Chebyshev Filtering update  = %.2f (ms)\n", cu_timers[4]);
        printf("===================================================================\n");
        printf("\n");
    }
    
    free(EVA_CUDA_buff);
    EVA_CUDA_buff = NULL;
}

static void EVA_CheFSI_CUDA_Lap_MV_orth_halo(double *cu_X, double *cu_X_ex)
{
    int nx   = EVA_CUDA_buff->CheFSI_nx;
    int ny   = EVA_CUDA_buff->CheFSI_ny;
    int nz   = EVA_CUDA_buff->CheFSI_nz;
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int ncol = EVA_CUDA_buff->CheFSI_ncol;

    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;
    
    int *istart    = EVA_CUDA_buff->ijk_se_displs + 6 * 0;
    int *iend      = EVA_CUDA_buff->ijk_se_displs + 6 * 1;
    int *jstart    = EVA_CUDA_buff->ijk_se_displs + 6 * 2;
    int *jend      = EVA_CUDA_buff->ijk_se_displs + 6 * 3;
    int *kstart    = EVA_CUDA_buff->ijk_se_displs + 6 * 4;
    int *kend      = EVA_CUDA_buff->ijk_se_displs + 6 * 5;
    int *istart_in = EVA_CUDA_buff->ijk_se_displs + 6 * 6;
    int *iend_in   = EVA_CUDA_buff->ijk_se_displs + 6 * 7;
    int *jstart_in = EVA_CUDA_buff->ijk_se_displs + 6 * 8;
    int *jend_in   = EVA_CUDA_buff->ijk_se_displs + 6 * 9;
    int *kstart_in = EVA_CUDA_buff->ijk_se_displs + 6 * 10;
    int *kend_in   = EVA_CUDA_buff->ijk_se_displs + 6 * 11;
    int *periods   = EVA_CUDA_buff->periods;

    dim3 dim_grid, dim_block;
    
    // 1. Copy X to X_ex
    dim_block.x = 16;
    dim_block.y = 16;
    dim_block.z = 4;
    dim_grid.x  = (unsigned int) ceil((double) nx / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) ny / (double) dim_block.y);
    dim_grid.z  = (unsigned int) ceil((double) nz / (double) dim_block.z);
    Lap_MV_orth_copy_x_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(nx, ny, nz, FDn, cu_X, ncol, cu_X_ex);
    cudaCheckAfterCall();
    
    // 2. Handle the halo part of X_ex
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) 
    {
        // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
        int nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)); // * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
        
        if (periods[nbr_i / 2])
        {
            int i_spos  = istart[nbrcount];
            int j_spos  = jstart[nbrcount];
            int k_spos  = kstart[nbrcount];
            int i_len   = iend[nbrcount] - i_spos;
            int j_len   = jend[nbrcount] - j_spos;
            int k_len   = kend[nbrcount] - k_spos;   
            int ip_spos = istart_in[nbr_i];
            int jp_spos = jstart_in[nbr_i];
            int kp_spos = kstart_in[nbr_i];
            
            dim_grid.x  = (unsigned int) ceil((double) i_len / (double) dim_block.x);
            dim_grid.y  = (unsigned int) ceil((double) j_len / (double) dim_block.y);
            dim_grid.z  = (unsigned int) ceil((double) k_len / (double) dim_block.z);
            
            Lap_MV_orth_period_BC_kernel<<<dim_grid, dim_block, 0, cu_stream>>> (
                nx, ny, nz, FDn, 
                i_spos, i_len, ip_spos,
                j_spos, j_len, jp_spos,
                k_spos, k_len, kp_spos,
                cu_X, ncol, cu_X_ex
            );
            cudaCheckAfterCall();
        } else {
            int ip_spos = istart_in[nbr_i];
            int jp_spos = jstart_in[nbr_i];
            int kp_spos = kstart_in[nbr_i];
            int ip_len  = iend_in[nbr_i] - ip_spos;
            int jp_len  = jend_in[nbr_i] - jp_spos;
            int kp_len  = kend_in[nbr_i] - kp_spos;
            
            dim_grid.x  = (unsigned int) ceil((double) ip_len / (double) dim_block.x);
            dim_grid.y  = (unsigned int) ceil((double) jp_len / (double) dim_block.y);
            dim_grid.z  = (unsigned int) ceil((double) kp_len / (double) dim_block.z);
            
            // Not tested yet...
            Lap_MV_orth_zero_BC_kernel<<<dim_grid, dim_block, 0, cu_stream>>> (
                nx, ny, nz, FDn, 
                ip_len, ip_spos,
                jp_len, jp_spos,
                kp_len, kp_spos,
                cu_X, ncol, cu_X_ex
            );
            cudaCheckAfterCall();
        }
    }
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_halo(double *cu_X, double *cu_X_ex)
{
    int nx   = EVA_CUDA_buff->CheFSI_nx;
    int ny   = EVA_CUDA_buff->CheFSI_ny;
    int nz   = EVA_CUDA_buff->CheFSI_nz;
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int ncol = EVA_CUDA_buff->CheFSI_ncol;

    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;
    
    int *istart    = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 0;
    int *iend      = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 1;
    int *jstart    = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 2;
    int *jend      = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 3;
    int *kstart    = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 4;
    int *kend      = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 5;
    int *istart_in = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 6;
    int *jstart_in = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 8;
    int *kstart_in = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 10;
    int *isnonzero = EVA_CUDA_buff->ijk_se_displs_nonorth + 26 * 12;

    dim3 dim_grid, dim_block;
    
    // 1. Copy X to X_ex
    dim_block.x = 16;
    dim_block.y = 16;
    dim_block.z = 4;
    dim_grid.x  = (unsigned int) ceil((double) nx / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) ny / (double) dim_block.y);
    dim_grid.z  = (unsigned int) ceil((double) nz / (double) dim_block.z);
    Lap_MV_orth_copy_x_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(nx, ny, nz, FDn, cu_X, ncol, cu_X_ex);
    cudaCheckAfterCall();
    
    // 2. Handle the halo part of X_ex
    for (int nbr_i = 0; nbr_i < 26; nbr_i++) 
    {
        if (!isnonzero[nbr_i]) continue;
        int i_spos  = istart[nbr_i];
        int j_spos  = jstart[nbr_i];
        int k_spos  = kstart[nbr_i];
        int i_len   = iend[nbr_i] - i_spos;
        int j_len   = jend[nbr_i] - j_spos;
        int k_len   = kend[nbr_i] - k_spos;   
        int ip_spos = istart_in[nbr_i];
        int jp_spos = jstart_in[nbr_i];
        int kp_spos = kstart_in[nbr_i];
        dim_grid.x  = (unsigned int) ceil((double) i_len / (double) dim_block.x);
        dim_grid.y  = (unsigned int) ceil((double) j_len / (double) dim_block.y);
        dim_grid.z  = (unsigned int) ceil((double) k_len / (double) dim_block.z);
        
        Lap_MV_orth_period_BC_kernel<<<dim_grid, dim_block, 0, cu_stream>>> (
            nx, ny, nz, FDn, 
            i_spos, i_len, ip_spos,
            j_spos, j_len, jp_spos,
            k_spos, k_len, kp_spos,
            cu_X, ncol, cu_X_ex
        );
        cudaCheckAfterCall();
    }
}

static void EVA_CheFSI_CUDA_Lap_MV_orth_kernel(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X)
{
    int nx = EVA_CUDA_buff->CheFSI_nx;
    int ny = EVA_CUDA_buff->CheFSI_ny;
    int nz = EVA_CUDA_buff->CheFSI_nz;
    
    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;

    dim3 dim_block, dim_grid;
    dim_block.x = X_BLK_SIZE;
    dim_block.y = Y_BLK_SIZE;
    dim_grid.x  = (unsigned int) ceil((double) nx / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) ny / (double) dim_block.y);
    dim_grid.z  = EVA_CUDA_buff->CheFSI_ncol;
    Lap_orth_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(nx, ny, nz, coef_0, b, cu_X_ex, cu_v, cu_X);
    cudaCheckAfterCall();
}

static void EVA_CheFSI_CUDA_DX_r6_kernel(
    const int DMnx_in,     const int DMny_in,     const int DMnz_in,
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int x0_stride_x, const int x1_stride_y, const int DMnd_Dx, 
    const int Grad_stencil_offset, double *cu_X_ex, double coef_0, double *cu_Dx
)
{
    int nx  = EVA_CUDA_buff->CheFSI_nx;
    int ny  = EVA_CUDA_buff->CheFSI_ny;
    int nz  = EVA_CUDA_buff->CheFSI_nz;
    int FDn = EVA_CUDA_buff->CheFSI_FDn;
    int DMnx_ex     = nx + 2 * FDn;
    int DMnxexny    = DMnx_ex * ny;
    int stride_z_ex = DMnx_ex * (ny+2*FDn);
    int DMnd_ex     = stride_z_ex * (nz+2*FDn);

    const int x0_stride_y = DMnx_ex;
    const int x0_stride_z = stride_z_ex;
    const int x1_stride_z = DMnxexny;
    
    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;

    dim3 dim_block, dim_grid;
    dim_block.x = X_BLK_SIZE;
    dim_block.y = Y_BLK_SIZE;
    dim_grid.x  = (unsigned int) ceil((double) DMnx_in / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) DMny_in / (double) dim_block.y);
    dim_grid.z  = EVA_CUDA_buff->CheFSI_ncol;
    
    if (x0_stride_x == x0_stride_y)
    {
        DY_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            DMnx_in, DMny_in, DMnz_in, 
            x0_stride_x, x0_stride_y, x1_stride_y, x0_stride_z, x1_stride_z,
            ix_offset, iy_offset, iz_offset, 
            DMnd_Dx, DMnd_ex, Grad_stencil_offset,
            coef_0, cu_X_ex, cu_Dx
        );
    }
    
    if (x0_stride_x == x0_stride_z)
    {
        DZ_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            DMnx_in, DMny_in, DMnz_in, 
            x0_stride_x, x0_stride_y, x1_stride_y, x0_stride_z, x1_stride_z,
            ix_offset, iy_offset, iz_offset, 
            DMnd_Dx, DMnd_ex, Grad_stencil_offset,
            coef_0, cu_X_ex, cu_Dx
        );
    }
    
    cudaCheckAfterCall();
}

static void EVA_CheFSI_CUDA_DX1_DX2_r6_kernel(
    const int DMnd_Dx,     
    const int DMnx_in,     const int DMny_in,     const int DMnz_in,
    const int x0_stride_0, const int x0_stride_1, const int x0_stride_y, 
    const int x1_stride_y, const int x0_stride_z, const int x1_stride_z, 
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int Grad_stencil_offset0, const int Grad_stencil_offset1, 
    double *cu_X_ex, double *cu_Dx
)
{
    int nx  = EVA_CUDA_buff->CheFSI_nx;
    int ny  = EVA_CUDA_buff->CheFSI_ny;
    int nz  = EVA_CUDA_buff->CheFSI_nz;
    int FDn = EVA_CUDA_buff->CheFSI_FDn;
    int DMnx_ex = nx + 2 * FDn;
    int DMny_ex = ny + 2 * FDn;
    int DMnz_ex = nz + 2 * FDn;
    int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;

    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;

    dim3 dim_block, dim_grid;
    dim_block.x = X_BLK_SIZE;
    dim_block.y = Y_BLK_SIZE;
    dim_grid.x  = (unsigned int) ceil((double) DMnx_in / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) DMny_in / (double) dim_block.y);
    dim_grid.z  = EVA_CUDA_buff->CheFSI_ncol;
    
    if (x0_stride_0 == 1 && x0_stride_1 == DMnx_ex)
    {
        DX_DY_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            DMnx_in, DMny_in, DMnz_in, 
            x0_stride_0, x0_stride_1, x0_stride_y, 
            x1_stride_y, x0_stride_z, x1_stride_z,
            ix_offset, iy_offset, iz_offset, 
            DMnd_Dx, DMnd_ex, 
            Grad_stencil_offset0, Grad_stencil_offset1,
            cu_X_ex, cu_Dx
        );
    }
    
    if (x0_stride_0 == 1 && x0_stride_1 == DMnx_ex * DMny_ex)
    {
        DX_DZ_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            DMnx_in, DMny_in, DMnz_in, 
            x0_stride_0, x0_stride_1, x0_stride_y, 
            x1_stride_y, x0_stride_z, x1_stride_z,
            ix_offset, iy_offset, iz_offset, 
            DMnd_Dx, DMnd_ex, 
            Grad_stencil_offset0, Grad_stencil_offset1,
            cu_X_ex, cu_Dx
        );
    } 
    
    if (x0_stride_0 == DMnx_ex && x0_stride_1 == DMnx_ex * DMny_ex)
    {
        DY_DZ_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            DMnx_in, DMny_in, DMnz_in, 
            x0_stride_0, x0_stride_1, x0_stride_y, 
            x1_stride_y, x0_stride_z, x1_stride_z,
            ix_offset, iy_offset, iz_offset, 
            DMnd_Dx, DMnd_ex, 
            Grad_stencil_offset0, Grad_stencil_offset1,
            cu_X_ex, cu_Dx
        );
    }
    
    cudaCheckAfterCall();
}

static void EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
    const int DMnd_Dx,     const int stride_Dx,   const int x1_stride_y, 
    const int x0_stride_y, const int Dx_stride_y, const int Dx_stride_z,
    const int Dx_x_offset, const int Dx_y_offset, const int Dx_z_offset, 
    const double coef_0,   const double b,        double *cu_X_ex, 
    double *cu_v,          double *cu_Dx,         double *cu_X
)
{
    int nx  = EVA_CUDA_buff->CheFSI_nx;
    int ny  = EVA_CUDA_buff->CheFSI_ny;
    int nz  = EVA_CUDA_buff->CheFSI_nz;
    int FDn = EVA_CUDA_buff->CheFSI_FDn;
    int DMnd = EVA_CUDA_buff->CheFSI_nxyz;
    int DMnx_ex = nx + 2 * FDn;
    int stride_z = nx * ny;
    int stride_z_ex = DMnx_ex * (ny + 2 * FDn);
    int DMnd_ex = stride_z_ex * (nz + 2 * FDn);

    const int x1_stride_z = stride_z;
    const int x0_stride_z = stride_z_ex;
    
    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;

    dim3 dim_block, dim_grid;
    dim_block.x = X_BLK_SIZE;
    dim_block.y = Y_BLK_SIZE;
    dim_grid.x  = (unsigned int) ceil((double) nx / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) ny / (double) dim_block.y);
    dim_grid.z  = EVA_CUDA_buff->CheFSI_ncol;
    
    if (stride_Dx == 1)
    {
        Lap_nonorth_DX_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            nx, ny, nz, cu_X_ex, cu_Dx, stride_Dx,
            DMnd, DMnd_ex, DMnd_Dx, 
            x1_stride_y, x0_stride_y, Dx_stride_y,
            x1_stride_z, x0_stride_z, Dx_stride_z,
            FDn, FDn, FDn, Dx_x_offset, Dx_y_offset, Dx_z_offset,
            coef_0, b, cu_v, cu_X
        );
    }
    
    if (stride_Dx == nx)
    {
        Lap_nonorth_DY_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            nx, ny, nz, cu_X_ex, cu_Dx, stride_Dx,
            DMnd, DMnd_ex, DMnd_Dx, 
            x1_stride_y, x0_stride_y, Dx_stride_y,
            x1_stride_z, x0_stride_z, Dx_stride_z,
            FDn, FDn, FDn, Dx_x_offset, Dx_y_offset, Dx_z_offset,
            coef_0, b, cu_v, cu_X
        );
    }
    
    if (stride_Dx == nx * ny)
    {
        Lap_nonorth_DZ_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
            nx, ny, nz, cu_X_ex, cu_Dx, stride_Dx,
            DMnd, DMnd_ex, DMnd_Dx, 
            x1_stride_y, x0_stride_y, Dx_stride_y,
            x1_stride_z, x0_stride_z, Dx_stride_z,
            FDn, FDn, FDn, Dx_x_offset, Dx_y_offset, Dx_z_offset,
            coef_0, b, cu_v, cu_X
        );
    }
    
    cudaCheckAfterCall();
}

static void EVA_CheFSI_CUDA_Lap_nonorth_DX_DY_r6_kernel(
    const double coef_0, const double b, double *cu_X_ex, double *cu_v,  
    double *cu_Dx1, double *cu_Dx2, double *cu_X
)
{
    int nx  = EVA_CUDA_buff->CheFSI_nx;
    int ny  = EVA_CUDA_buff->CheFSI_ny;
    int nz  = EVA_CUDA_buff->CheFSI_nz;

    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;

    dim3 dim_block, dim_grid;
    dim_block.x = X_BLK_SIZE;
    dim_block.y = Y_BLK_SIZE;
    dim_grid.x  = (unsigned int) ceil((double) nx / (double) dim_block.x);
    dim_grid.y  = (unsigned int) ceil((double) ny / (double) dim_block.y);
    dim_grid.z  = EVA_CUDA_buff->CheFSI_ncol;
    
    Lap_nonorth_DX_DY_r6_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(
        nx, ny, nz, 
        cu_X_ex, cu_Dx1, cu_Dx2,
        coef_0, b, cu_v, cu_X
    );
    
    cudaCheckAfterCall();
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct11(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMnxexny = DMnx_ex * DMny; 
    int DMnd_xex = DMnxexny * DMnz;
    int D1_stencil_y_offset = (FDn + 1) * 0;

    EVA_CheFSI_CUDA_DX_r6_kernel(
        DMnx_ex, DMny, DMnz, 
        0, FDn, FDn,
        DMnx_ex, DMnx_ex, DMnd_xex, 
        D1_stencil_y_offset, cu_X_ex, 0.0, cu_Dx
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
        DMnd_xex, 1, DMnx, DMnx_ex, DMnx_ex, DMnxexny,
        FDn, 0, 0, coef_0, _b, cu_X_ex, _cu_v, cu_Dx, cu_X
    );
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct12(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMny_ex = DMny + 2 * FDn;
    int DMnxexny = DMnx_ex * DMny; 
    int DMnd_xex = DMnxexny * DMnz;
    int stride_z_ex = DMnx_ex * DMny_ex;
    int D1_stencil_z_offset = (FDn + 1) * 1;

    EVA_CheFSI_CUDA_DX_r6_kernel(
        DMnx_ex, DMny, DMnz, 
        0, FDn, FDn,
        stride_z_ex, DMnx_ex, DMnd_xex, 
        D1_stencil_z_offset, cu_X_ex, 0.0, cu_Dx
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
        DMnd_xex, 1, DMnx, DMnx_ex, DMnx_ex, DMnxexny,
        FDn, 0, 0, coef_0, _b, cu_X_ex, _cu_v, cu_Dx, cu_X
    );
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct13(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMny_ex = DMny + 2 * FDn;    
    int DMnxnyex = DMnx * DMny_ex;
    int DMnd_yex = DMnxnyex * DMnz;
    int stride_z_ex = DMnx_ex * DMny_ex;
    int D1_stencil_z_offset = (FDn + 1) * 1;
    
    EVA_CheFSI_CUDA_DX_r6_kernel(
        DMnx, DMny_ex, DMnz, 
        FDn, 0, FDn, 
        stride_z_ex, DMnx, DMnd_yex, 
        D1_stencil_z_offset, cu_X_ex, 0.0, cu_Dx
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
        DMnd_yex, DMnx, DMnx, DMnx_ex, DMnx, DMnxnyex,
        0, FDn, 0, coef_0, _b, cu_X_ex, _cu_v, cu_Dx, cu_X
    );
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct14(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int FDn1 = FDn + 1;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMny_ex = DMny + 2 * FDn;
    int DMnxexny = DMnx_ex * DMny;
    int DMnd_xex = DMnxexny * DMnz;
    int stride_z_ex = DMnx_ex * DMny_ex;
    int D1_stencil_xy_offset = FDn1 * 2;
    int D1_stencil_xz_offset = FDn1 * 3;

    EVA_CheFSI_CUDA_DX1_DX2_r6_kernel(
        DMnd_xex, DMnx_ex, DMny, DMnz,
        DMnx_ex, stride_z_ex, DMnx_ex, 
        DMnx_ex, stride_z_ex, DMnxexny,
        0, FDn, FDn, 
        D1_stencil_xy_offset, D1_stencil_xz_offset,
        cu_X_ex, cu_Dx
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
        DMnd_xex, 1, DMnx, DMnx_ex, DMnx_ex, DMnxexny,
        FDn, 0, 0, coef_0, _b, cu_X_ex, _cu_v, cu_Dx, cu_X
    );
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct15(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int FDn1 = FDn + 1;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnxny = DMnx * DMny;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMny_ex = DMny + 2 * FDn;
    int DMnz_ex = DMnz + 2 * FDn;
    int DMnd_zex = DMnxny * DMnz_ex;
    int stride_z_ex = DMnx_ex * DMny_ex;
    int D1_stencil_zx_offset = FDn1 * 6;
    int D1_stencil_zy_offset = FDn1 * 7; 
    
    EVA_CheFSI_CUDA_DX1_DX2_r6_kernel(
        DMnd_zex, DMnx, DMny, DMnz_ex, 
        1, DMnx_ex, DMnx_ex, 
        DMnx, stride_z_ex, DMnxny,
        FDn, FDn, 0, 
        D1_stencil_zx_offset, D1_stencil_zy_offset,
        cu_X_ex, cu_Dx
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
        DMnd_zex, DMnxny, DMnx, DMnx_ex, DMnx, DMnxny,
        0, 0, FDn, coef_0, _b, cu_X_ex, _cu_v, cu_Dx, cu_X
    );
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct16(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int FDn1 = FDn + 1;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMny_ex = DMny + 2 * FDn;
    int DMnxnyex = DMnx * DMny_ex;
    int DMnd_yex = DMnxnyex * DMnz;
    int stride_z_ex = DMnx_ex * DMny_ex;
    int D1_stencil_yx_offset = FDn1 * 4;
    int D1_stencil_yz_offset = FDn1 * 5;
    
    EVA_CheFSI_CUDA_DX1_DX2_r6_kernel(
        DMnd_yex, DMnx, DMny_ex, DMnz, 
        1, stride_z_ex, DMnx_ex, 
        DMnx, stride_z_ex, DMnxnyex,
        FDn, 0, FDn, 
        D1_stencil_yx_offset, D1_stencil_yz_offset,
        cu_X_ex, cu_Dx
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_r6_kernel(
        DMnd_yex, DMnx, DMnx, DMnx_ex, DMnx, DMnxnyex,
        0, FDn, 0, coef_0, _b, cu_X_ex, _cu_v, cu_Dx, cu_X
    );
}

static void EVA_CheFSI_CUDA_Lap_MV_nonorth_ct17(double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X, double *cu_Dx1, double *cu_Dx2)
{
    double *_cu_v = cu_v, _b = b;
    if (fabs(_b) < 1e-14) { _cu_v = cu_X; _b = 0.0; } 
    int FDn  = EVA_CUDA_buff->CheFSI_FDn;
    int FDn1 = FDn + 1;
    int DMnx = EVA_CUDA_buff->CheFSI_nx;
    int DMny = EVA_CUDA_buff->CheFSI_ny;
    int DMnz = EVA_CUDA_buff->CheFSI_nz;
    int DMnx_ex = DMnx + 2 * FDn;
    int DMny_ex = DMny + 2 * FDn;
    int DMnxexny = DMnx_ex * DMny;
    int DMnxnyex = DMnx * DMny_ex;
    int DMnd_xex = DMnxexny * DMnz;
    int DMnd_yex = DMnxnyex * DMnz;
    int stride_z_ex = DMnx_ex * DMny_ex;
    int D1_stencil_z_offset  = FDn1 * 1;
    int D1_stencil_xy_offset = FDn1 * 2;
    int D1_stencil_xz_offset = FDn1 * 3;
    
    EVA_CheFSI_CUDA_DX1_DX2_r6_kernel(
        DMnd_xex, DMnx_ex, DMny, DMnz,
        DMnx_ex, stride_z_ex, DMnx_ex, 
        DMnx_ex, stride_z_ex, DMnxexny,
        0, FDn, FDn, 
        D1_stencil_xy_offset, D1_stencil_xz_offset,
        cu_X_ex, cu_Dx1
    );
    
    EVA_CheFSI_CUDA_DX_r6_kernel(
        DMnx, DMny_ex, DMnz, 
        FDn, 0, FDn, 
        stride_z_ex, DMnx, DMnd_yex, 
        D1_stencil_z_offset, cu_X_ex, 0.0, cu_Dx2
    );
    
    EVA_CheFSI_CUDA_Lap_nonorth_DX_DY_r6_kernel(coef_0, _b, cu_X_ex, _cu_v, cu_Dx1, cu_Dx2, cu_X);
}

static void EVA_CheFSI_CUDA_Vnl_SpMV(double *cu_Y, double *cu_Ynew)
{
    int nxyz = EVA_CUDA_buff->CheFSI_nxyz;
    int alen = EVA_CUDA_buff->CheFSI_alen;
    int ncol = EVA_CUDA_buff->CheFSI_ncol;
    int nnz  = EVA_CUDA_buff->CheFSI_nnz;
    double d_zero = 0.0, d_one = 1.0;

    int    *cu_x2a_row_ptr = EVA_CUDA_buff->cu_x2a_row_ptr;
    int    *cu_x2a_col_idx = EVA_CUDA_buff->cu_x2a_col_idx;
    double *cu_x2a_val     = EVA_CUDA_buff->cu_x2a_val;
    int    *cu_a2x_row_ptr = EVA_CUDA_buff->cu_a2x_row_ptr;
    int    *cu_a2x_col_idx = EVA_CUDA_buff->cu_a2x_col_idx;
    double *cu_a2x_val     = EVA_CUDA_buff->cu_a2x_val;
    double *cu_alpha       = EVA_CUDA_buff->cu_alpha;
    double *cu_alpha_scale = EVA_CUDA_buff->cu_alpha_scale;
    
    cudaStream_t       cu_stream        = EVA_CUDA_buff->cu_stream;
    cusparseHandle_t   cusparse_handle  = EVA_CUDA_buff->cusparse_handle;
    cusparseMatDescr_t cusparse_mat_x2a = EVA_CUDA_buff->cusparse_mat_x2a;
    cusparseMatDescr_t cusparse_mat_a2x = EVA_CUDA_buff->cusparse_mat_a2x;
    
    // 1. CSR SpMV: alpha = x2alpha * Y
    cudaSparseCheck(cusparseDcsrmm(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alen, ncol, nxyz, nnz, &d_one, cusparse_mat_x2a, 
        cu_x2a_val, cu_x2a_row_ptr, cu_x2a_col_idx, 
        cu_Y, nxyz, &d_zero, cu_alpha, alen
    ));
    
    // 2. Scale alpha
    dim3 dim_block, dim_grid;
    dim_grid.y  = ncol;
    dim_grid.x  = (alen + 31) / 32;
    dim_block.x = 32;
    Vnl_SpMV_scale_alpha_kernel<<<dim_grid, dim_block, 0, cu_stream>>>(alen, cu_alpha_scale, cu_alpha);
    cudaCheckAfterCall();
    
    // 3. CSR SpMV: Ynew += alpha2x * alpha
    cudaSparseCheck(cusparseDcsrmm(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        nxyz, ncol, alen, nnz, &d_one, cusparse_mat_a2x, 
        cu_a2x_val, cu_a2x_row_ptr, cu_a2x_col_idx, 
        cu_alpha, alen, &d_one, cu_Ynew, nxyz
    ));
}

static void EVA_CheFSI_CUDA_update_Ynew_by_X(double vscale1, double vscale2, double *cu_X, double *cu_Ynew)
{
    int npts = EVA_CUDA_buff->CheFSI_npts;
    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;
    
    update_Ynew_by_X_kernel<<<(npts + 31) / 32, 32, 0, cu_stream>>>(npts, vscale1, vscale2, cu_X, cu_Ynew);
    cudaCheckAfterCall();
}

static void EVA_CheFSI_CUDA_Lap_MV_halo(const int cell_type, double *cu_X, double *cu_X_ex)
{
    if (cell_type == 0) EVA_CheFSI_CUDA_Lap_MV_orth_halo   (cu_X, cu_X_ex);
    else                EVA_CheFSI_CUDA_Lap_MV_nonorth_halo(cu_X, cu_X_ex);
}

static void EVA_CheFSI_CUDA_Lap_MV_kernel(const int cell_type, double *cu_X_ex, double *cu_v, double coef_0, double b, double *cu_X)
{
    double *cu_Dx1 = EVA_CUDA_buff->cu_Dx1;
    double *cu_Dx2 = EVA_CUDA_buff->cu_Dx2;
    switch (cell_type)
    {
        case  0: EVA_CheFSI_CUDA_Lap_MV_orth_kernel (cu_X_ex, cu_v, coef_0, b, cu_X); break;
        case 11: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct11(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1); break;
        case 12: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct12(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1); break;
        case 13: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct13(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1); break;
        case 14: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct14(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1); break;
        case 15: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct15(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1); break;
        case 16: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct16(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1); break;
        case 17: EVA_CheFSI_CUDA_Lap_MV_nonorth_ct17(cu_X_ex, cu_v, coef_0, b, cu_X, cu_Dx1, cu_Dx2); break;
    }
}


extern "C"
void EVA_CheFSI_CUDA_ChebvsheyFiltering(
    const int m, const double a, const double b, const double a0, 
    const double coef_0, const double b_coef, 
    const double *X, const double *v, double *Y
)
{
    double e = 0.5 * (b - a);
    double c = 0.5 * (b + a);
    double sigma  = e / (a0 - c);
    double gamma  = 2.0 / sigma;
    double vscal1 = sigma / e;
    double vscal2 = 0.0;
    double sigma2 = 0.0;
    
    double *cu_X    = EVA_CUDA_buff->cu_X;
    double *cu_X_ex = EVA_CUDA_buff->cu_X_ex;
    double *cu_Y    = EVA_CUDA_buff->cu_Y;
    double *cu_Ynew = EVA_CUDA_buff->cu_Ynew;
    double *cu_v    = EVA_CUDA_buff->cu_v;
    double *cu_tmp;
    
    float time_ms;
    float *cu_timers = EVA_CUDA_buff->cu_timers;
    cudaEvent_t *cu_events = EVA_CUDA_buff->cu_events;
    
    int ct = EVA_CUDA_buff->cell_type;
    
    cudaStream_t cu_stream = EVA_CUDA_buff->cu_stream;
    size_t npts_memsize = sizeof(double) * EVA_CUDA_buff->CheFSI_npts;
    size_t nxyz_memsize = sizeof(double) * EVA_CUDA_buff->CheFSI_nxyz;
    cudaCheck( cudaEventRecord(cu_events[0], cu_stream) );
    cudaCheck( cudaMemcpyAsync(cu_X, X, npts_memsize, cudaMemcpyHostToDevice, cu_stream) );
    cudaCheck( cudaMemcpyAsync(cu_v, v, nxyz_memsize, cudaMemcpyHostToDevice, cu_stream) );
    EVA_CUDA_buff->mem_move_size += npts_memsize + nxyz_memsize;
    
    cudaCheck( cudaEventRecord(cu_events[1], cu_stream) );
    EVA_CheFSI_CUDA_Lap_MV_halo(ct, cu_X, cu_X_ex);
    cudaCheck( cudaEventRecord(cu_events[2], cu_stream) );
    EVA_CheFSI_CUDA_Lap_MV_kernel(ct, cu_X_ex, cu_v, coef_0, b_coef, cu_Y);
    cudaCheck( cudaEventRecord(cu_events[3], cu_stream) );
    EVA_CheFSI_CUDA_Vnl_SpMV(cu_X, cu_Y);

    cudaCheck( cudaEventRecord(cu_events[4], cu_stream) );
    EVA_CheFSI_CUDA_update_Ynew_by_X(vscal1, vscal2, cu_X, cu_Y);
    cudaCheck( cudaEventRecord(cu_events[5], cu_stream) );
    
    cudaCheck( cudaStreamSynchronize(cu_stream) );
    for (int i = 0; i < 5; i++)
    {
        cudaEventElapsedTime(&time_ms, cu_events[i], cu_events[i + 1]);
        cu_timers[i] += time_ms;
    }

    for (int j = 1; j < m; j++)
    {
        // 1. Ynew = (H - c*I) * Y
        cudaCheck( cudaEventRecord(cu_events[1], cu_stream) );
        EVA_CheFSI_CUDA_Lap_MV_halo(ct, cu_Y, cu_X_ex);
        cudaCheck( cudaEventRecord(cu_events[2], cu_stream) );
        EVA_CheFSI_CUDA_Lap_MV_kernel(ct, cu_X_ex, cu_v, coef_0, b_coef, cu_Ynew);
        cudaCheck( cudaEventRecord(cu_events[3], cu_stream) );
        EVA_CheFSI_CUDA_Vnl_SpMV(cu_Y, cu_Ynew);
        
        // 2. Ynew = (2*sigma2/e) * Ynew - (sigma*sigma2) * X
        sigma2 = 1.0 / (gamma - sigma);
        vscal1 = 2.0 * sigma2 / e; 
        vscal2 = sigma * sigma2;
        cudaCheck( cudaEventRecord(cu_events[4], cu_stream) );
        EVA_CheFSI_CUDA_update_Ynew_by_X(vscal1, vscal2, cu_X, cu_Ynew);
        cudaCheck( cudaEventRecord(cu_events[5], cu_stream) );
        
        // 3. Update X and Y: X = Y, Y = Ynew; swap pointers to reduce data movement
        sigma   = sigma2;
        cu_tmp  = cu_X; 
        cu_X    = cu_Y; 
        cu_Y    = cu_Ynew; 
        cu_Ynew = cu_tmp;
        
        cudaCheck( cudaStreamSynchronize(cu_stream) );
        for (int i = 1; i < 5; i++)
        {
            cudaEventElapsedTime(&time_ms, cu_events[i], cu_events[i + 1]);
            cu_timers[i] += time_ms;
        }
    }
    
    EVA_CUDA_buff->numHamil += m;
    
    cudaCheck( cudaEventRecord(cu_events[0], cu_stream) );
    cudaCheck( cudaMemcpyAsync(Y, cu_Y, npts_memsize, cudaMemcpyDeviceToHost, cu_stream) );
    EVA_CUDA_buff->mem_move_size += npts_memsize;
    cudaCheck( cudaEventRecord(cu_events[1], cu_stream) );

    cudaCheck( cudaStreamSynchronize(cu_stream) );
    cudaEventElapsedTime(&time_ms, cu_events[0], cu_events[1]);
    cu_timers[0] += time_ms;
}

