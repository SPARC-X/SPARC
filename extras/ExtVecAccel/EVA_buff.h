/**
 * @file    EVA_buff.h
 * @brief   External Vectorization and Acceleration (EVA) module 
 *          EVA_buff structure for storing intermediate variables in EVA module
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
 
#ifndef __EVA_BUFF_H__
#define __EVA_BUFF_H__

#include "CSRPlus.h"

struct EVA_buff_
{
    int use_EVA;                       // If EVA functions will be used 
    int my_global_rank;                // This process's rank in MPI_COMM_WORLD
    int nthreads;                      // Number of threads to use
    
    // Variables used in EVA_Lap_MV_orth() & EVA_Lap_MV_nonorth()
    int    order, FDn;                 // Finite difference order and stencil length
    int    DMnx, DMny, DMnz;           // Number of grid points on each axis in the original domain
    int    DMnx_ex, DMny_ex, DMnz_ex;  // Number of grid points on each axis in the extended domain
    int    DMnd, DMnd_ex;              // Total number of grid points in the original and extended domains
    int    halo_size;                  // Current size of x_ex and x_{in, out}
    int    cell_type;                  // Cell type
    int    *cart_array;                // For MPI cart
    int    *ijk_se_displs;             // Boundary indices and displacements for EVA_Lap_MV_orth()
    int    *ijk_se_displs_nonorth;     // Boundary indices and displacements for EVA_Lap_MV_nonorth()
    double w2_diag, b;                 // Scaling factors for I and v
    double *x_ex;                      // Values of the extended domain
    double *x_in, *x_out;              // Boundary values for exchange
    double *Dx1, *Dx2;                 // Derivatives used in EVA_Lap_MV_nonorth()
    double *Grad_stencils;             // Gradient finite difference coefficients for EVA_Lap_MV_nonorth()
    double *Lap_stencils;              // Laplacian finite difference coefficients for EVA_Lap_MV_orth()
    double *Lap_wt_nonorth;            // Laplacian finite difference coefficients for EVA_Lap_MV_nonorth()
    size_t x_ex_msize, halo_msize;     // Current memory size of x_ex and x_{in, out}
    size_t Dx1_msize, Dx2_msize;       // Current memory size of Dx1 and Dx2
    
    // Buffers for EVA_Hamil_MatVec()
    int    Vnl_SpMV_init;              // If the buffers below has been initialized
    int    alpha_length;               // Length of alpha array
    double *alpha;                     // The alpha array
    double *alpha_scale;               // Coefficients for scaling alpha after x2alpha
    double *Vnl_Hx_tmp;                // Temporary array used in Vnl 
    size_t alpha_msize;                // Memory size of alpha array
    size_t Vnl_Hx_tmp_msize;           // Memory size of Vnl_Hx_tmp
    void   *mkl_x2a_sp_mat_p;          // x --> alpha MKL sparse BLAS matrix object pointer
    void   *mkl_a2x_sp_mat_p;          // alpha --> x MKL sparse BLAS matrix object pointer
    void   *mkl_x2a_mat_desc_p;        // x2alpha MKL sparse BLAS matrix property description pointer
    void   *mkl_a2x_mat_desc_p;        // alpha2x MKL sparse BLAS matrix property description pointer
    CSRPlusMatrix_t CSRP_x2a;          // x --> alpha CSR matrix
    CSRPlusMatrix_t CSRP_a2x;          // alpha --> x CSR matrix
    
    // Buffers for EVA_Chebyshev_Filtering()
    double *Ynew;                      // Temporary array used in Chebyshev Filtering 
    size_t Ynew_size;                  // Memory size of Ynew
    int    use_EVA_CUDA;               // If we need to use CUDA kernels for Chebyshev Filtering 
    int    CheFSI_CUDA_init;           // If CUDA kernels for Chebyshev Filtering initialized 
    
    // Statistic information
    int    Lap_MV_rhs;                 // Total right-hand side vectors in Laplacian matvec
    int    Vnl_MV_rhs;                 // Total right-hand side vectors in Vnl operator matvec
    double Lap_MV_cpyx_t;              // Total wall time for copying x to x_ex
    double Lap_MV_pack_t;              // Total wall time for packing halo data that is to be sent
    double Lap_MV_comm_t;              // Total wall time for waiting halo exchange to be finished
    double Lap_MV_unpk_t;              // Total wall time for copying received halo data
    double Lap_MV_krnl_t;              // Total wall time for Laplacian matvec 
    double Vnl_MV_t;                   // Total wall time for Vnl operator matvec 
};

typedef struct EVA_buff_* EVA_buff_t;

// Initialize EVA_buff
// Input parameters:
//   order     : Finite difference order, / 2 = stencil length
//   cell_type : Cell type, 0 == orthogonal
void EVA_buff_init(const int order, const int cell_type);

// Free EVA_buff and print statistic information
void EVA_buff_finalize();

#endif
