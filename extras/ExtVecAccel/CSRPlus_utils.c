/**
 * @file    CSRPlus_utils.c
 * @brief   CSRPlus matrix helper functions
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2017-2019 Georgia Institute of Technology
 */

#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <omp.h>

#include "CSRPlus.h"

static int cmp_pair(int M1, int N1, int M2, int N2)
{
    if (M1 == M2) return (N1 < N2);
    else return (M1 < M2);
}

static void qsortCOO2CSR(int *row, int *col, double *val, int l, int r)
{
    int i = l, j = r, row_tmp, col_tmp;
    int mid_row = row[(l + r) / 2];
    int mid_col = col[(l + r) / 2];
    double val_tmp;
    while (i <= j)
    {
        while (cmp_pair(row[i], col[i], mid_row, mid_col)) i++;
        while (cmp_pair(mid_row, mid_col, row[j], col[j])) j--;
        if (i <= j)
        {
            row_tmp = row[i]; row[i] = row[j]; row[j] = row_tmp;
            col_tmp = col[i]; col[i] = col[j]; col[j] = col_tmp;
            val_tmp = val[i]; val[i] = val[j]; val[j] = val_tmp;
            
            i++;  j--;
        }
    }
    if (i < r) qsortCOO2CSR(row, col, val, i, r);
    if (j > l) qsortCOO2CSR(row, col, val, l, j);
}

static void compressIndices(int *idx, int *idx_ptr, int nindex, int nelem)
{
    int curr_pos = 0, end_pos;
    idx_ptr[0] = 0;
    for (int index = 0; index < nindex; index++)
    {
        for (end_pos = curr_pos; end_pos < nelem; end_pos++)
            if (idx[end_pos] > index) break;
        idx_ptr[index + 1] = end_pos;
        curr_pos = end_pos;
    }
    idx_ptr[nindex] = nelem; 
}

static void partitionBlocks(const int nelem, const int nblocks, int *displs)
{
    int bs0 = nelem / nblocks;
    int bs1 = bs0;
    int remainder = nelem % nblocks;
    if (remainder > 0) bs1++;
    displs[0] = 0;
    for (int i = 0; i < remainder; i++)
        displs[i + 1] = displs[i] + bs1;
    for (int i = remainder; i < nblocks; i++)
        displs[i + 1] = displs[i] + bs0;
}

static int lower_bound(const int *a, int n, int x) 
{
    int l = 0, h = n;
    while (l < h) 
    {
        int mid = l + (h - l) / 2;
        if (x <= a[mid]) h = mid;
        else l = mid + 1;
    }
    return l;
}

// Initialize a CSRPlus matrix using a COO matrix
void CSRP_init_with_COO_matrix(
    const int nrows, const int ncols, const int nnz, const int *row,
    const int *col, const double *val, CSRPlusMatrix_t *CSRP
)
{
    CSRPlusMatrix_t _CSRP = (CSRPlusMatrix_t) malloc(sizeof(CSRPlusMatrix));
    
    _CSRP->nrows   = nrows;
    _CSRP->ncols   = ncols;
    _CSRP->nnz     = nnz;
    _CSRP->row_ptr = (int*)    malloc(sizeof(int)    * (nrows + 1));
    _CSRP->col     = (int*)    malloc(sizeof(int)    * nnz);
    _CSRP->val     = (double*) malloc(sizeof(double) * nnz);
    int *_row      = (int*)    malloc(sizeof(int)    * nnz);
    assert(_CSRP->row_ptr != NULL);
    assert(_CSRP->col     != NULL);
    assert(_CSRP->val     != NULL);
    assert(_row           != NULL);
    
    _CSRP->nnz_spos  = NULL;
    _CSRP->nnz_epos  = NULL;
    _CSRP->first_row = NULL;
    _CSRP->last_row  = NULL;
    _CSRP->fr_intact = NULL;
    _CSRP->lr_intact = NULL;
    _CSRP->fr_res    = NULL;
    _CSRP->lr_res    = NULL;
    
    #pragma omp parallel for
    for (int i = 0; i < nnz; i++)
    {
        _row[i]       = row[i];
        _CSRP->col[i] = col[i];
        _CSRP->val[i] = val[i];
    }
    
    qsortCOO2CSR(_row, _CSRP->col, _CSRP->val, 0, nnz - 1);
    compressIndices(_row, _CSRP->row_ptr, nrows, nnz);
    
    free(_row);
    *CSRP = _CSRP;
}

// Free a CSRPlus matrix structure
void CSRP_free(CSRPlusMatrix_t CSRP)
{
    if (CSRP == NULL) return;
    free(CSRP->row_ptr  );
    free(CSRP->col      );
    free(CSRP->val      );
    free(CSRP->nnz_spos );
    free(CSRP->nnz_epos );
    free(CSRP->first_row);
    free(CSRP->last_row );
    free(CSRP->fr_intact);
    free(CSRP->lr_intact);
    free(CSRP->fr_res   );
    free(CSRP->lr_res   );
}

// Partition a CSR matrix into multiple blocks with the same nnz
void CSRP_partition(const int nblocks, CSRPlusMatrix_t CSRP)
{
    CSRP->X_ncol    = 1;
    CSRP->nblocks   = nblocks;
    CSRP->nnz_spos  = (int*)    malloc(sizeof(int)    * nblocks);
    CSRP->nnz_epos  = (int*)    malloc(sizeof(int)    * nblocks);
    CSRP->first_row = (int*)    malloc(sizeof(int)    * nblocks);
    CSRP->last_row  = (int*)    malloc(sizeof(int)    * nblocks);
    CSRP->fr_intact = (int*)    malloc(sizeof(int)    * nblocks);
    CSRP->lr_intact = (int*)    malloc(sizeof(int)    * nblocks);
    CSRP->fr_res    = (double*) malloc(sizeof(double) * nblocks);
    CSRP->lr_res    = (double*) malloc(sizeof(double) * nblocks);
    assert(CSRP->nnz_spos  != NULL);
    assert(CSRP->nnz_epos  != NULL);
    assert(CSRP->first_row != NULL);
    assert(CSRP->last_row  != NULL);
    assert(CSRP->fr_intact != NULL);
    assert(CSRP->lr_intact != NULL);
    assert(CSRP->fr_res    != NULL);
    assert(CSRP->lr_res    != NULL);
    
    int nnz   = CSRP->nnz;
    int nrows = CSRP->nrows;
    int *row_ptr = CSRP->row_ptr;
    
    int *nnz_displs = (int *) malloc((nblocks + 1) * sizeof(int));
    partitionBlocks(nnz, nblocks, nnz_displs);
    
    for (int iblock = 0; iblock < nblocks; iblock++)
    {
        int block_nnz_spos = nnz_displs[iblock];
        int block_nnz_epos = nnz_displs[iblock + 1] - 1;
        int spos_in_row = lower_bound(row_ptr, nrows + 1, block_nnz_spos);
        int epos_in_row = lower_bound(row_ptr, nrows + 1, block_nnz_epos);
        if (row_ptr[spos_in_row] > block_nnz_spos) spos_in_row--;
        if (row_ptr[epos_in_row] > block_nnz_epos) epos_in_row--;
        
        // Note: It is possible that the last nnz is the first nnz in a row,
        // and there are some empty rows between the last row and previous non-empty row
        while (row_ptr[epos_in_row] == row_ptr[epos_in_row + 1]) epos_in_row++;  
        
        CSRP->nnz_spos[iblock]  = block_nnz_spos;
        CSRP->nnz_epos[iblock]  = block_nnz_epos;
        CSRP->first_row[iblock] = spos_in_row;
        CSRP->last_row[iblock]  = epos_in_row;
        
        if ((epos_in_row - spos_in_row) >= 1)
        {
            int fr_intact = (block_nnz_spos == row_ptr[spos_in_row]);
            int lr_intact = (block_nnz_epos == row_ptr[epos_in_row + 1] - 1);
            
            CSRP->fr_intact[iblock] = fr_intact;
            CSRP->lr_intact[iblock] = lr_intact;
        } else {
            // Mark that this thread only handles a segment of a row 
            CSRP->fr_intact[iblock] = 0;
            CSRP->lr_intact[iblock] = -1;
        }
    }
    
    CSRP->last_row[nblocks - 1] = nrows - 1;
    CSRP->nnz_epos[nblocks - 1] = row_ptr[nrows] - 1;
    
    free(nnz_displs);
}

// Use first-touch policy to optimize the storage of CSR arrays in a CSRPlus matrix
void CSRP_optimize_NUMA(CSRPlusMatrix_t CSRP)
{
    int nnz = CSRP->nnz;
    int nrows = CSRP->nrows;
    int nblocks = CSRP->nblocks;
    
    int    *row_ptr = (int*)    malloc(sizeof(int)    * (nrows + 1));
    int    *col     = (int*)    malloc(sizeof(int)    * nnz);
    double *val     = (double*) malloc(sizeof(double) * nnz);
    assert(row_ptr != NULL);
    assert(col     != NULL);
    assert(val     != NULL);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nrows + 1; i++)
            row_ptr[i] = CSRP->row_ptr[i];
        
        #pragma omp for schedule(static)
        for (int iblock = 0; iblock < nblocks; iblock++)
        {
            int nnz_spos = CSRP->nnz_spos[iblock];
            int nnz_epos = CSRP->nnz_epos[iblock];
            for (int i = nnz_spos; i <= nnz_epos; i++)
            {
                col[i] = CSRP->col[i];
                val[i] = CSRP->val[i];
            }
        }
    }
    
    free(CSRP->row_ptr);
    free(CSRP->col);
    free(CSRP->val);
    
    CSRP->row_ptr = row_ptr;
    CSRP->col     = col;
    CSRP->val     = val;
}

