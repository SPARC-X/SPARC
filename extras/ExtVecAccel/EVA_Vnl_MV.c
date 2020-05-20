/**
 * @file    EVA_Vnl_MV.c
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Vnl operator with multiple vectors
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

#ifdef USE_EVA_MKL_MODULE
#include <mkl.h>
#include <mkl_spblas.h>
#endif

#include "isddft.h"
#include "EVA_buff.h"
#include "EVA_Vnl_MV.h"

// Initialize EVA Vnl SpMV buffer
void EVA_Vnl_SpMV_init(
    EVA_buff_t EVA_buff, const SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ *ANI, 
    NLOC_PROJ_OBJ *nlocProj, const int ncol
)
{
    int alpha_length = pSPARC->IP_displ[pSPARC->n_atom];
    int DMnd = EVA_buff->DMnd;
    EVA_buff->alpha_length = alpha_length;
    
    size_t alpha_msize = sizeof(double) * ncol * pSPARC->IP_displ[pSPARC->n_atom];
    if (alpha_msize > EVA_buff->alpha_msize)
    {
        EVA_buff->alpha_msize = alpha_msize;
        free(EVA_buff->alpha);
        EVA_buff->alpha = (double*) malloc(EVA_buff->alpha_msize);
        assert(EVA_buff->alpha != NULL);
    }
    
    // 1. Get the number of non-zeros in x2alpha and alpha2x
    int Chi_nnz = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) 
    {
        if (nlocProj[ityp].nproj == 0) continue; // this is typical for hydrogen
        for (int iat = 0; iat < ANI[ityp].n_atom; iat++) 
        {
            int ndc = ANI[ityp].ndc[iat]; 
            Chi_nnz += nlocProj[ityp].nproj * ndc;
        }
    }    
    
    // 2. Allocate the CSR arrays for x2alpha and alpha2x
    int    *idx_tmp     = (int*)    malloc(sizeof(int)    * Chi_nnz);
    int    *x2a_row_ptr = (int*)    malloc(sizeof(int)    * (alpha_length + 1));
    int    *x2a_col     = (int*)    malloc(sizeof(int)    * Chi_nnz);
    double *x2a_val     = (double*) malloc(sizeof(double) * Chi_nnz);
    int    *a2x_row_ptr = (int*)    malloc(sizeof(int)    * (DMnd + 1));
    int    *a2x_col     = (int*)    malloc(sizeof(int)    * Chi_nnz);
    double *a2x_val     = (double*) malloc(sizeof(double) * Chi_nnz);
    assert(x2a_row_ptr != NULL);
    assert(x2a_col     != NULL);
    assert(x2a_val     != NULL);
    assert(a2x_row_ptr != NULL);
    assert(a2x_col     != NULL);
    assert(a2x_val     != NULL);
    
    // 3. Generate x2alpha COO format matrix
    int *x2a_row = idx_tmp;
    int Chi_elem = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) 
    {
        if (nlocProj[ityp].nproj == 0) continue; // this is typical for hydrogen
        for (int iat = 0; iat < ANI[ityp].n_atom; iat++) 
        {
            int ndc = ANI[ityp].ndc[iat]; 
            int atom_index = ANI[ityp].atom_index[iat];
            int *grid_pos_ptr = ANI[ityp].grid_pos[iat];
            
            // Notice: Chi is column-major & used in transposed, so it could be viewed as
            // a row-major matrix with ndc columns (LDA = ndc) and nlocProj[ityp].nproj rows
            double *Chi = nlocProj[ityp].Chi[iat];
            for (int iproj = 0; iproj < nlocProj[ityp].nproj; iproj++)
            {
                int iproj_row_idx = pSPARC->IP_displ[atom_index] + iproj;
                double *Chi_row = Chi + iproj * ndc;
                
                for (int i = 0; i < ndc; i++)
                {
                    x2a_row[Chi_elem] = iproj_row_idx;
                    x2a_col[Chi_elem] = grid_pos_ptr[i];
                    x2a_val[Chi_elem] = pSPARC->dV * Chi_row[i];
                    Chi_elem++;
                }
            }
        }
    }
    
    // 4. Convert x2alpha COO format to CSR format
    CSRP_init_with_COO_matrix(alpha_length, DMnd, Chi_elem, x2a_row, x2a_col, x2a_val, &EVA_buff->CSRP_x2a);
    
    // 5. Generate alpha2x COO format matrix
    int *a2x_row = idx_tmp;
    Chi_elem = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) 
    {
        if (! nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (int iat = 0; iat < ANI[ityp].n_atom; iat++) 
        {
            int ndc = ANI[ityp].ndc[iat]; 
            int atom_index = ANI[ityp].atom_index[iat];
            int *grid_pos_ptr = ANI[ityp].grid_pos[iat];
            
            // Notice: Chi is column-major, it has ndc rows * nlocProj[ityp].nproj columns
            double *Chi = nlocProj[ityp].Chi[iat];
            for (int idc = 0; idc < ndc; idc++)
            {
                double *Chi_row = Chi + idc;
                int alpha_spos = pSPARC->IP_displ[atom_index];
                
                for (int i = 0; i < nlocProj[ityp].nproj; i++)
                {
                    a2x_row[Chi_elem] = grid_pos_ptr[idc];
                    a2x_col[Chi_elem] = alpha_spos + i;
                    a2x_val[Chi_elem] = Chi_row[i * ndc];
                    Chi_elem++;
                }
            }
        }
    }
    
    // 6. Convert alpha2x COO format to CSR format
    CSRP_init_with_COO_matrix(DMnd, alpha_length, Chi_elem, a2x_row, a2x_col, a2x_val, &EVA_buff->CSRP_a2x);
    
    free(idx_tmp);
    free(x2a_row_ptr);
    free(x2a_col);
    free(x2a_val);
    free(a2x_row_ptr);
    free(a2x_col);
    free(a2x_val);
    
    // 7. Initialize CSRPlus matrix
    int nthreads = omp_get_max_threads();
    CSRP_partition(nthreads, EVA_buff->CSRP_x2a);
    CSRP_partition(nthreads, EVA_buff->CSRP_a2x);
    CSRP_optimize_NUMA(EVA_buff->CSRP_x2a);
    CSRP_optimize_NUMA(EVA_buff->CSRP_a2x);

    // 8. Scale alpha
    EVA_buff->alpha_scale = (double*) malloc(sizeof(double) * alpha_length);
    assert(EVA_buff->alpha_scale != NULL);
    #pragma omp simd
    for (int i = 0; i < alpha_length; i++)
        EVA_buff->alpha_scale[i] = 1.0;
    int count = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) 
    {
        int lmax = pSPARC->psd[ityp].lmax;
        for (int iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) 
        {
            //for (int n = 0; n < ncol; n++) 
            //{
                int ldispl = 0;
                for (int l = 0; l <= lmax; l++) 
                {
                    // skip the local l
                    if (l == pSPARC->localPsd[ityp]) 
                    {
                        ldispl += pSPARC->psd[ityp].ppl[l];
                        continue;
                    }
                    for (int np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) 
                    {
                        for (int m = -l; m <= l; m++)
                            EVA_buff->alpha_scale[count++] = pSPARC->psd[ityp].Gamma[ldispl+np];
                    }
                    ldispl += pSPARC->psd[ityp].ppl[l];
                }
            //}
        }
    }
    
    #ifdef USE_EVA_MKL_MODULE
    // 9. Initialize MKL sparse BLAS handle
    EVA_buff->mkl_x2a_sp_mat_p   = malloc(sizeof(sparse_matrix_t));
    EVA_buff->mkl_a2x_sp_mat_p   = malloc(sizeof(sparse_matrix_t));
    EVA_buff->mkl_x2a_mat_desc_p = malloc(sizeof(struct matrix_descr));
    EVA_buff->mkl_a2x_mat_desc_p = malloc(sizeof(struct matrix_descr));
    
    sparse_matrix_t *mkl_x2a_sp_mat_p = (sparse_matrix_t*) EVA_buff->mkl_x2a_sp_mat_p;
    sparse_matrix_t *mkl_a2x_sp_mat_p = (sparse_matrix_t*) EVA_buff->mkl_a2x_sp_mat_p;
    struct matrix_descr *mkl_x2a_mat_desc_p = (struct matrix_descr*) EVA_buff->mkl_x2a_mat_desc_p;
    struct matrix_descr *mkl_a2x_mat_desc_p = (struct matrix_descr*) EVA_buff->mkl_a2x_mat_desc_p;
    
    mkl_x2a_mat_desc_p->type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_x2a_mat_desc_p->diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(
        mkl_x2a_sp_mat_p, SPARSE_INDEX_BASE_ZERO, alpha_length, DMnd, 
        EVA_buff->CSRP_x2a->row_ptr, EVA_buff->CSRP_x2a->row_ptr + 1, 
        EVA_buff->CSRP_x2a->col, EVA_buff->CSRP_x2a->val
    );
    if (ncol > 1)
    {
        mkl_sparse_set_mm_hint(
            *mkl_x2a_sp_mat_p, SPARSE_OPERATION_NON_TRANSPOSE,
            *mkl_x2a_mat_desc_p, SPARSE_LAYOUT_COLUMN_MAJOR, ncol, 200
        );
    } else {
        mkl_sparse_set_mv_hint(
            *mkl_x2a_sp_mat_p, SPARSE_OPERATION_NON_TRANSPOSE,
            *mkl_x2a_mat_desc_p, 200
        );
    }
    mkl_sparse_optimize(*mkl_x2a_sp_mat_p);
    
    mkl_a2x_mat_desc_p->type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_a2x_mat_desc_p->diag = SPARSE_DIAG_NON_UNIT;
    mkl_sparse_d_create_csr(
        EVA_buff->mkl_a2x_sp_mat_p, SPARSE_INDEX_BASE_ZERO, DMnd, alpha_length,
        EVA_buff->CSRP_a2x->row_ptr, EVA_buff->CSRP_a2x->row_ptr + 1, 
        EVA_buff->CSRP_a2x->col, EVA_buff->CSRP_a2x->val
    );
    if (ncol > 1)
    {
        mkl_sparse_set_mm_hint(
            *mkl_a2x_sp_mat_p, SPARSE_OPERATION_NON_TRANSPOSE,
            *mkl_a2x_mat_desc_p, SPARSE_LAYOUT_COLUMN_MAJOR, ncol, 200
        );
    } else {
        mkl_sparse_set_mv_hint(
            *mkl_a2x_sp_mat_p, SPARSE_OPERATION_NON_TRANSPOSE,
            *mkl_a2x_mat_desc_p, 200
        );
    }
    mkl_sparse_optimize(*mkl_a2x_sp_mat_p);
    #endif
}

// Compute Vnl operator times vectors in a SpMV way
void EVA_Vnl_SpMV(
    EVA_buff_t EVA_buff, const int ncol, const double *x, 
    MPI_Comm comm, const int nproc, double *Hx
)
{
    int DMnd = EVA_buff->DMnd;
    int alpha_length = EVA_buff->alpha_length;
    
    double st = MPI_Wtime();
    
    size_t alpha_msize = sizeof(double) * ncol * alpha_length;
    if (alpha_msize > EVA_buff->alpha_msize)
    {
        EVA_buff->alpha_msize = alpha_msize;
        free(EVA_buff->alpha);
        EVA_buff->alpha = (double*) malloc(EVA_buff->alpha_msize);
        assert(EVA_buff->alpha != NULL);
    }
    
    size_t Hx_size = sizeof(double) * ncol * DMnd;
    if (Hx_size > EVA_buff->Vnl_Hx_tmp_msize)
    {
        EVA_buff->Vnl_Hx_tmp_msize = Hx_size;
        free(EVA_buff->Vnl_Hx_tmp);
        EVA_buff->Vnl_Hx_tmp = (double*) malloc(EVA_buff->Vnl_Hx_tmp_msize);
        assert(EVA_buff->Vnl_Hx_tmp != NULL);
    }
    
    double *alpha  = EVA_buff->alpha;
    double *Hx_tmp = EVA_buff->Vnl_Hx_tmp;
    memset(alpha, 0, sizeof(double) * alpha_length * ncol);
    
    #ifdef USE_EVA_MKL_MODULE
    sparse_matrix_t *mkl_x2a_sp_mat_p = (sparse_matrix_t*) EVA_buff->mkl_x2a_sp_mat_p;
    sparse_matrix_t *mkl_a2x_sp_mat_p = (sparse_matrix_t*) EVA_buff->mkl_a2x_sp_mat_p;
    struct matrix_descr *mkl_x2a_mat_desc_p = (struct matrix_descr*) EVA_buff->mkl_x2a_mat_desc_p;
    struct matrix_descr *mkl_a2x_mat_desc_p = (struct matrix_descr*) EVA_buff->mkl_a2x_mat_desc_p;
    #endif
    
    // (1) Find inner product
    // CSR SpMV: alpha = x2alpha * x
    #ifdef USE_EVA_MKL_MODULE
    if (ncol > 1)
    {
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *mkl_x2a_sp_mat_p,
            *mkl_x2a_mat_desc_p, SPARSE_LAYOUT_COLUMN_MAJOR, x, ncol,
            DMnd, 0.0, alpha, alpha_length
        );
    } else {
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *mkl_x2a_sp_mat_p,
            *mkl_x2a_mat_desc_p, x, 0.0, alpha
        );
    }
    #else
    if (ncol > 1) CSRP_SpMV_nvec(EVA_buff->CSRP_x2a, x, DMnd, ncol, alpha, alpha_length);
    else          CSRP_SpMV(EVA_buff->CSRP_x2a, x, alpha);
    #endif
    
    // (2) Sum alpha over all processes over domain comm 
    // Notice: currently ncol == 1
    if (nproc > 1) MPI_Allreduce(MPI_IN_PLACE, alpha, alpha_length * ncol, MPI_DOUBLE, MPI_SUM, comm);
    
    // (3) Go over all atoms and multiply gamma_Jl to the inner product
    for (int icol = 0; icol < ncol; icol++)
    {
        double *alpha_i = alpha + alpha_length * icol;
        #pragma omp simd
        for (int i = 0; i < alpha_length; i++)
            alpha_i[i] *= EVA_buff->alpha_scale[i];
    }
    
    // (4) Multiply the inner product and the nonlocal projector
    // CSR SpMV: Hx += alpha2x * alpha
    #ifdef USE_EVA_MKL_MODULE
    if (ncol > 1)
    {
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *mkl_a2x_sp_mat_p,
            *mkl_a2x_mat_desc_p, SPARSE_LAYOUT_COLUMN_MAJOR, alpha, ncol,
            alpha_length, 1.0, Hx, DMnd
        );
    } else {
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE, 1.0, *mkl_a2x_sp_mat_p,
            *mkl_a2x_mat_desc_p, alpha, 1.0, Hx
        );
    }
    #else
    if (ncol > 1) CSRP_SpMM_CM(EVA_buff->CSRP_a2x, alpha, alpha_length, ncol, Hx_tmp, DMnd);
    else          CSRP_SpMV(EVA_buff->CSRP_a2x, alpha, Hx_tmp);
    #pragma omp parallel for simd
    for (int i = 0; i < ncol * DMnd; i++)
        Hx[i] += Hx_tmp[i];
    #endif

    double et = MPI_Wtime();
    EVA_buff->Vnl_MV_t   += et - st;
    EVA_buff->Vnl_MV_rhs += ncol;
}

// Reset buffers used in EVA_Vnl_SpMV 
void EVA_Vnl_SpMV_reset(EVA_buff_t EVA_buff)
{
    EVA_buff->Vnl_SpMV_init = 0;
    
    free(EVA_buff->mkl_x2a_sp_mat_p);
    free(EVA_buff->mkl_a2x_sp_mat_p);
    free(EVA_buff->mkl_x2a_mat_desc_p);
    free(EVA_buff->mkl_a2x_mat_desc_p);
    EVA_buff->mkl_x2a_sp_mat_p   = NULL;
    EVA_buff->mkl_a2x_sp_mat_p   = NULL;
    EVA_buff->mkl_x2a_mat_desc_p = NULL;
    EVA_buff->mkl_a2x_mat_desc_p = NULL;
    
    CSRP_free(EVA_buff->CSRP_x2a);
    CSRP_free(EVA_buff->CSRP_a2x);
    free(EVA_buff->CSRP_x2a);
    free(EVA_buff->CSRP_a2x);
    EVA_buff->CSRP_x2a = NULL;
    EVA_buff->CSRP_a2x = NULL;
}
