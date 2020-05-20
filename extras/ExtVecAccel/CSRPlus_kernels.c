/**
 * @file    CSRPlus_kernels.c
 * @brief   CSRPlus matrix SpMV / SpMM kernels
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

// =============== SpMV kernels ===============

static double CSR_SpMV_row_seg(
    const int seg_len, const int *__restrict col, 
    const double *__restrict val, const double *__restrict x
)
{
    register double res = 0.0;
    #pragma omp simd
    for (int idx = 0; idx < seg_len; idx++)
        res += val[idx] * x[col[idx]];
    return res;
}

static void CSR_SpMV_row_block(
    const int srow, const int erow,
    const int *row_ptr, const int *col, const double *val, 
    const double *__restrict x, double *__restrict y
)
{
    for (int irow = srow; irow < erow; irow++)
    {
        register double res = 0.0;
        #pragma omp simd
        for (int idx = row_ptr[irow]; idx < row_ptr[irow + 1]; idx++)
            res += val[idx] * x[col[idx]];
        y[irow] = res;
    }
}

static void CSRP_SpMV_block(CSRPlusMatrix_t CSRP, const int iblock, const double *x, double *y)
{
    int    *row_ptr = CSRP->row_ptr;
    int    *col     = CSRP->col;
    double *val     = CSRP->val;
    
    if (CSRP->first_row[iblock] == CSRP->last_row[iblock])
    {
        // This thread handles a segment on 1 row
        
        int nnz_spos = CSRP->nnz_spos[iblock];
        int nnz_epos = CSRP->nnz_epos[iblock];
        int seg_len  = nnz_epos - nnz_spos + 1;
        
        CSRP->fr_res[iblock] = CSR_SpMV_row_seg(seg_len, col + nnz_spos, val + nnz_spos, x);
        CSRP->lr_res[iblock] = 0.0;
        y[CSRP->first_row[iblock]] = 0.0;
    } else {
        // This thread handles segments on multiple rows
        
        int first_intact_row = CSRP->first_row[iblock];
        int last_intact_row  = CSRP->last_row[iblock];
        
        if (CSRP->fr_intact[iblock] == 0)
        {
            int nnz_spos = CSRP->nnz_spos[iblock];
            int nnz_epos = row_ptr[first_intact_row + 1];
            int seg_len  = nnz_epos - nnz_spos;
            
            CSRP->fr_res[iblock] = CSR_SpMV_row_seg(seg_len, col + nnz_spos, val + nnz_spos, x);
            y[first_intact_row] = 0.0;
            first_intact_row++;
        }
        
        if (CSRP->lr_intact[iblock] == 0)
        {
            int nnz_spos = row_ptr[last_intact_row];
            int nnz_epos = CSRP->nnz_epos[iblock];
            int seg_len  = nnz_epos - nnz_spos + 1;
            
            CSRP->lr_res[iblock] = CSR_SpMV_row_seg(seg_len, col + nnz_spos, val + nnz_spos, x);
            y[last_intact_row] = 0.0;
            last_intact_row--;
        }
        
        CSR_SpMV_row_block(
            first_intact_row, last_intact_row + 1,
            row_ptr, col, val, x, y
        );
    }
}

// Perform OpenMP parallelized CSR SpMV with a CSRPlus matrix
void CSRP_SpMV(CSRPlusMatrix_t CSRP, const double *x, double *y)
{
    int nblocks = CSRP->nblocks;
    
    #pragma omp parallel for schedule(static)
    for (int iblock = 0; iblock < nblocks; iblock++)
        CSRP_SpMV_block(CSRP, iblock, x, y);
    
    for (int iblock = 0; iblock < nblocks; iblock++)
    {
        if (CSRP->fr_intact[iblock] == 0)
        {
            int first_row = CSRP->first_row[iblock];
            y[first_row] += CSRP->fr_res[iblock];
        }
        
        if (CSRP->lr_intact[iblock] == 0)
        {
            int last_row = CSRP->last_row[iblock];
            y[last_row] += CSRP->lr_res[iblock];
        }
    }
}

// =============== SpMM kernels ===============

static void CSR_SpMM_RM_row_seg(
    const int seg_len, const int ncol, const int *__restrict col, const double *__restrict val, 
    const double *__restrict X, const int ldX, double *__restrict Y, const int ldY
)
{
    int icol = 0;
    for (icol = 0; icol < ncol - 8; icol += 8)
    {
        double res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int idx = 0; idx < seg_len; idx++)
        {
            register double val_idx = val[idx];
            const double *X_ptr = X + col[idx] + icol * ldX;
            #pragma omp simd
            for (int i = 0; i < 8; i++)
                res[i] += val_idx * X_ptr[i * ldX];
        }
        for (int i = 0; i < 8; i++) Y[(icol + i) * ldY] = res[i];
    }
    
    int ncol0 = ncol - icol;
    double res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int idx = 0; idx < seg_len; idx++)
    {
        register double val_idx = val[idx];
        const double *X_ptr = X + col[idx] + icol * ldX;
        for (int i = 0; i < ncol0; i++)
            res[i] += val_idx * X_ptr[i * ldX];
    }
    for (int i = 0; i < ncol0; i++) Y[(icol + i) * ldY] = res[i];
}

static void CSR_SpMM_RM_row_block(
    const int srow, const int erow, const int ncol,
    const int *row_ptr, const int *col, const double *val, 
    const double *__restrict X, const int ldX, 
    double *__restrict Y, const int ldY
)
{
    for (int irow = srow; irow < erow; irow++)
    {
        int icol = 0;
        for (icol = 0; icol < ncol - 8; icol += 8)
        {
            double res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int idx = row_ptr[irow]; idx < row_ptr[irow + 1]; idx++)
            {
                register double val_idx = val[idx];
                const double *X_ptr = X + col[idx] + icol * ldX;
                #pragma omp simd
                for (int i = 0; i < 8; i++)
                    res[i] += val_idx * X_ptr[i * ldX];
            }
            for (int i = 0; i < 8; i++) Y[(icol + i) * ldY + irow] = res[i];
        }
        
        int ncol0 = ncol - icol;
        double res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int idx = row_ptr[irow]; idx < row_ptr[irow + 1]; idx++)
        {
            register double val_idx = val[idx];
            const double *X_ptr = X + col[idx] + icol * ldX;
            for (int i = 0; i < ncol0; i++)
                res[i] += val_idx * X_ptr[i * ldX];
        }
        for (int i = 0; i < ncol0; i++) Y[(icol + i) * ldY + irow] = res[i];
    }
}

static void CSRP_SpMM_CM_block(
    CSRPlusMatrix_t CSRP, const int iblock, const int ncol, 
    const double *X, const int ldX, double *Y, const int ldY)
{
    int    nblocks  = CSRP->nblocks;
    int    *row_ptr = CSRP->row_ptr;
    int    *col     = CSRP->col;
    double *val     = CSRP->val;
    
    if (CSRP->first_row[iblock] == CSRP->last_row[iblock])
    {
        // This thread handles a segment on 1 row
        
        int nnz_spos  = CSRP->nnz_spos[iblock];
        int nnz_epos  = CSRP->nnz_epos[iblock];
        int seg_len   = nnz_epos - nnz_spos + 1;
        int first_row = CSRP->first_row[iblock];
        
        CSR_SpMM_RM_row_seg(
            seg_len, ncol, col + nnz_spos, val + nnz_spos,
            X, ldX, CSRP->fr_res + iblock, nblocks
        );
        
        for (int icol = 0; icol < ncol; icol++)
        {
            CSRP->lr_res[icol * nblocks + iblock] = 0.0;
            Y[icol * ldY + first_row] = 0.0;
        }
    } else {
        // This thread handles segments on multiple rows
        
        int first_intact_row = CSRP->first_row[iblock];
        int last_intact_row  = CSRP->last_row[iblock];
        
        if (CSRP->fr_intact[iblock] == 0)
        {
            int nnz_spos = CSRP->nnz_spos[iblock];
            int nnz_epos = row_ptr[first_intact_row + 1];
            int seg_len  = nnz_epos - nnz_spos;
            
            CSR_SpMM_RM_row_seg(
                seg_len, ncol, col + nnz_spos, val + nnz_spos,
                X, ldX, CSRP->fr_res + iblock, nblocks
            );
            
            for (int icol = 0; icol < ncol; icol++)
                Y[icol * ldY + first_intact_row] = 0.0;

            first_intact_row++;
        }
        
        if (CSRP->lr_intact[iblock] == 0)
        {
            int nnz_spos = row_ptr[last_intact_row];
            int nnz_epos = CSRP->nnz_epos[iblock];
            int seg_len  = nnz_epos - nnz_spos + 1;
            
            CSR_SpMM_RM_row_seg(
                seg_len, ncol, col + nnz_spos, val + nnz_spos,
                X, ldX, CSRP->lr_res + iblock, nblocks
            );
            
            for (int icol = 0; icol < ncol; icol++)
                Y[icol * ldY + last_intact_row] = 0.0;

            last_intact_row--;
        }
        
        CSR_SpMM_RM_row_block(
            first_intact_row, last_intact_row + 1, ncol, 
            row_ptr, col, val, X, ldX, Y, ldY
        );
    }
}

void CSRP_SpMM_CM(
    CSRPlusMatrix_t CSRP, const double *X, const int ldX, 
    const int ncol, double *Y, const int ldY
)
{
    assert(ldX >= CSRP->ncols);
    assert(ldY >= CSRP->nrows);
    
    int nblocks = CSRP->nblocks;
    if (ncol > CSRP->X_ncol)
    {
        free(CSRP->fr_res);
        free(CSRP->lr_res);
        CSRP->X_ncol = ncol;
        CSRP->fr_res = (double*) malloc(sizeof(double) * CSRP->X_ncol * nblocks);
        CSRP->lr_res = (double*) malloc(sizeof(double) * CSRP->X_ncol * nblocks);
        assert(CSRP->fr_res != NULL);
        assert(CSRP->lr_res != NULL);
    }
    
    #pragma omp parallel for schedule(static)
    for (int iblock = 0; iblock < nblocks; iblock++)
        CSRP_SpMM_CM_block(CSRP, iblock, ncol, X, ldX, Y, ldY);
    
    for (int icol = 0; icol < ncol; icol++)
    {
        double *Y_i = Y + icol * ldY;
        double *fr_res_i = CSRP->fr_res + icol * nblocks;
        double *lr_res_i = CSRP->lr_res + icol * nblocks;
        for (int iblock = 0; iblock < nblocks; iblock++)
        {
            if (CSRP->fr_intact[iblock] == 0)
            {
                int first_row = CSRP->first_row[iblock];
                Y_i[first_row] += fr_res_i[iblock];
            }
            
            if (CSRP->lr_intact[iblock] == 0)
            {
                int last_row = CSRP->last_row[iblock];
                Y_i[last_row] += lr_res_i[iblock];
            }
        }
    }
}

void CSRP_SpMV_nvec(
    CSRPlusMatrix_t CSRP, const double *X, const int ldX, 
    const int ncol, double *Y, const int ldY
)
{
    assert(ldX >= CSRP->ncols);
    assert(ldY >= CSRP->nrows);
    
    for (int icol = 0; icol < ncol; icol++)
        CSRP_SpMV(CSRP, X + icol * ldX, Y + icol * ldY);
}

