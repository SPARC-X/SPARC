/**
 * @file    EVA_CheFSI_CUDA_buff.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          CUDA buffers
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#ifndef __EVA_CHEFSI_CUDA_BUFF_H__
#define __EVA_CHEFSI_CUDA_BUFF_H__

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

struct EVA_CUDA_buff_
{
    int my_rank, cell_type;
    
    // Parameters
    int    CheFSI_FDn;   // Finite difference radius 
    int    CheFSI_nx;    // Number of FD grid points on x-direction
    int    CheFSI_ny;    // Number of FD grid points on y-direction
    int    CheFSI_nz;    // Number of FD grid points on z-direction
    int    CheFSI_nxyz;  // Number of FD grid points in the original domain
    int    CheFSI_alen;  // Length of the alpha array in Vnl_SpMV
    int    CheFSI_ncol;  // Number of columns (domains) this process has
    int    CheFSI_nnz;   // Number of non-zeros in x2a and a2x matrices in Vnl_SpMV
    int    CheFSI_npts;  // == CheFSI_ncol * CheFSI_nxyz
    
    // Arrays for Laplacian operator
    int    *ijk_se_displs;         // Boundary indices and displacements for orthogonal cell
    int    *ijk_se_displs_nonorth; // Boundary indices and displacements for nonorthogonal cell
    int    *periods;               // If boundary conditions are periodic, for orthogonal cell
    double *cu_v;                  // Diagonal matrix v
    double *cu_X_ex;               // Extended domain for Laplacian operator
    double *cu_Dx1, *cu_Dx2;       // Derivatives used in nonorthogonal cells

    size_t Dx1_msize, Dx2_msize, npts_msize, nxyz_msize, npts_ex_msize;
    
    // Arrays for Chebyshev Filtering
    double *cu_X, *cu_Y;     // Input and output arrays of Chebyshev Filtering
    double *cu_Ynew;         // Temporary array used in Chebyshev Filtering
    
    // Arrays for Vnl_SpMV
    int    *cu_x2a_row_ptr;  // x2alpha CSR matrix non-zero row displacements
    int    *cu_x2a_col_idx;  // x2alpha CSR matrix column indices
    double *cu_x2a_val;      // x2alpha CSR matrix non-zero values
    int    *cu_a2x_row_ptr;  // alpha2x CSR matrix non-zero row displacements
    int    *cu_a2x_col_idx;  // alpha2x CSR matrix column indices
    double *cu_a2x_val;      // alpha2x CSR matrix non-zero values
    double *cu_alpha;        // The alpha array in Vnl_SpMV
    double *cu_alpha_scale;  // Coefficients for scaling alpha after x2alpha
    
    // CUDA handles
    cudaStream_t       cu_stream;
    cusparseHandle_t   cusparse_handle;
    cusparseMatDescr_t cusparse_mat_x2a;
    cusparseMatDescr_t cusparse_mat_a2x;
    
    // Timing
    cudaEvent_t  cu_events[6];
    float        cu_timers[6];  // [D2H & H2D data movement, LapMV Halo, LapMV kernel, Vnl kernel, update Ynew]
    size_t       mem_move_size; 
    int          numHamil;
};

typedef struct EVA_CUDA_buff_* EVA_CUDA_buff_t;

#endif
