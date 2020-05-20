/**
 * @file    CSRPlus.h
 * @brief   CSRPlus matrix header file 
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2017-2019 Georgia Institute of Technology
 */

#ifndef __CSRPLUS_H__
#define __CSRPLUS_H__

struct CSRPlusMatrix_
{
    // Standard CSR arrays and parameters
    int    nrows, ncols, nnz;
    int    *row_ptr;
    int    *col;
    double *val;
    
    // CSRPlus task partitioning information
    int    nblocks;     // Total number of nnz blocks
    int    X_ncol;      // Number of X matrix column
    int    *nnz_spos;   // First nnz of a block
    int    *nnz_epos;   // Last  nnz (included) of a block
    int    *first_row;  // First row of a block
    int    *last_row;   // Last  row of a block
    int    *fr_intact;  // If the first row of a block is intact
    int    *lr_intact;  // If the last  row of a block is intact
    double *fr_res;     // Partial result of the first row 
    double *lr_res;     // Partial result of the last  row
};

typedef struct CSRPlusMatrix_  CSRPlusMatrix;
typedef struct CSRPlusMatrix_* CSRPlusMatrix_t;

#ifdef __cplusplus
extern "C" {
#endif

// =============== Helper Functions ===============

// Initialize a CSRPlus matrix using a COO matrix
// Note: This function assumes that the input COO matrix is not sorted
// Input:
//   nrows, ncols  : Number of rows and columns
//   nnz           : Number of non-zeros
//   row, col, val : Indices and values of non-zero elements
//   CSRP          : Pointer to CSRPlus matrix structure pointer
// Output:
//   CSRP         : Pointer to initialized CSRPlus matrix structure pointer
void CSRP_init_with_COO_matrix(
    const int nrows, const int ncols, const int nnz, const int *row,
    const int *col, const double *val, CSRPlusMatrix_t *CSRP
);

// Free a CSRPlus matrix structure
// Input:
//   CSRP : CSRPlus matrix structure pointer
void CSRP_free(CSRPlusMatrix_t CSRP);

// Partition a CSR matrix into multiple blocks with the same nnz
// Input:
//   nblocks : Number of blocks to be partitioned
//   CSRP    : CSRPlus matrix structure pointer
// Output:
//   CSRP    : CSRPlus matrix structure pointer with partitioning information
void CSRP_partition(const int nblocks, CSRPlusMatrix_t CSRP);

// Use first-touch policy to optimize the storage of CSR arrays in a CSRPlus matrix
// Input:
//   CSRP      : CSRPlus matrix structure pointer
// Output:
//   CSRP      : CSRPlus matrix structure pointer with NUMA optimized storage
void CSRP_optimize_NUMA(CSRPlusMatrix_t CSRP);


// =============== Computation Kernels ===============

// Perform OpenMP parallelized CSR SpMV with a CSRPlus matrix
// Input:
//   CSRP : CSRPlus matrix structure pointer
//   x    : Input vector
// Output:
//   y    : Output vector
void CSRP_SpMV(CSRPlusMatrix_t CSRP, const double *x, double *y);

// Perform OpenMP parallelized CSR SpMM with a CSRPlus matrix and column-major dense matrix
// Input:
//   CSRP : CSRPlus matrix structure pointer
//   X    : Input matrix, stored in column-major style
//   ldX  : Leading dimension of X
//   ldY  : Leading dimension of Y
// Output:
//   y    : Output matrix, stored in row-major style
void CSRP_SpMM_CM(
    CSRPlusMatrix_t CSRP, const double *X, const int ldX, 
    const int ncol, double *Y, const int ldY
);

// Using CSRP_SpMV() to perform CSRP_SpMM_CM() computation, for testing
void CSRP_SpMV_nvec(
    CSRPlusMatrix_t CSRP, const double *X, const int ldX, 
    const int ncol, double *Y, const int ldY
);

#ifdef __cplusplus
}
#endif

#endif
