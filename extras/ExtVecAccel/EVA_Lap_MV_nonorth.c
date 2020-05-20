/**
 * @file    EVA_Lap_MV_nonorth.c
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Laplacian matrix with multiple vectors
 *          For nonorthogonal cells
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
#include "lapVecNonOrth.h"
#include "gradVecRoutines.h"

#include "EVA_buff.h"
#include "EVA_Lap_MV_common.h"
#include "EVA_Lap_MV_nonorth.h"

// Initialize parameters used in EVA_Lap_MV_nonorth
void EVA_init_Lap_MV_nonorth_params(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, 
    const double a, const double b, const double c, MPI_Comm comm, 
    const int nproc, EVA_buff_t EVA_buff
)
{
    if (pSPARC->cell_typ == 0)
    {
        printf(RED"FATAL: EVA_init_Lap_MV_nonorth_params is for non-orthogonal cell!\n");
        assert(pSPARC->cell_typ != 0);
    }
    
    int *dims      = &EVA_buff->cart_array[0];
    int *periods   = &EVA_buff->cart_array[3];
    int *my_coords = &EVA_buff->cart_array[6];
    MPI_Cart_get(comm, 3, dims, periods, my_coords);
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    
    int FDn    = EVA_buff->FDn;
    int DMnx   = 1 - DMVertices[0] + DMVertices[1];
    int DMny   = 1 - DMVertices[2] + DMVertices[3];
    int DMnz   = 1 - DMVertices[4] + DMVertices[5];
    int DMnd   = DMnx * DMny * DMnz;
    int DMnxny = DMnx * DMny;
    
    int DMnx_ex   = DMnx + pSPARC->order;
    int DMny_ex   = DMny + pSPARC->order;
    int DMnz_ex   = DMnz + pSPARC->order;
    int DMnd_ex   = DMnx_ex * DMny_ex * DMnz_ex;
    
    double w2_diag;
    w2_diag  = pSPARC->D2_stencil_coeffs_x[0]; 
    w2_diag += pSPARC->D2_stencil_coeffs_y[0];
    w2_diag += pSPARC->D2_stencil_coeffs_z[0];
    w2_diag  = w2_diag * a + c;
    
    EVA_buff->DMnx = DMnx;
    EVA_buff->DMny = DMny;
    EVA_buff->DMnz = DMnz;
    EVA_buff->DMnd = DMnd;
    
    EVA_buff->DMnx_ex = DMnx_ex;
    EVA_buff->DMny_ex = DMny_ex;
    EVA_buff->DMnz_ex = DMnz_ex;
    EVA_buff->DMnd_ex = DMnd_ex;
    
    EVA_buff->w2_diag = w2_diag;
    EVA_buff->b = b;
    
    double *Lap_wt_nonorth = EVA_buff->Lap_wt_nonorth;
    double *Lap_stencil_nonorth = Lap_wt_nonorth + 5;
    // This function is in lapVecnonorth.c
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil_nonorth, a);
    
    // Check if we need to reallocate x_ex
    size_t x_ex_msize = sizeof(double) * ncol * DMnd_ex;
    if (x_ex_msize > EVA_buff->x_ex_msize)
    {
        EVA_buff->x_ex_msize = x_ex_msize;
        free(EVA_buff->x_ex);
        EVA_buff->x_ex = (double*) malloc(EVA_buff->x_ex_msize);
        assert(EVA_buff->x_ex != NULL);
    }
    
    int *istart     = EVA_buff->ijk_se_displs_nonorth + 26 * 0;
    int *iend       = EVA_buff->ijk_se_displs_nonorth + 26 * 1;
    int *jstart     = EVA_buff->ijk_se_displs_nonorth + 26 * 2;
    int *jend       = EVA_buff->ijk_se_displs_nonorth + 26 * 3;
    int *kstart     = EVA_buff->ijk_se_displs_nonorth + 26 * 4;
    int *kend       = EVA_buff->ijk_se_displs_nonorth + 26 * 5;
    int *istart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 6;
    int *iend_in    = EVA_buff->ijk_se_displs_nonorth + 26 * 7;
    int *jstart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 8;
    int *jend_in    = EVA_buff->ijk_se_displs_nonorth + 26 * 9;
    int *kstart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 10;
    int *kend_in    = EVA_buff->ijk_se_displs_nonorth + 26 * 11;
    int *isnonzero  = EVA_buff->ijk_se_displs_nonorth + 26 * 12;

    // This function is in lapVecnonorth.c
    snd_rcv_buffer(
        nproc, dims, periods, FDn, DMnx, DMny, DMnz, 
        istart, iend, jstart, jend, kstart, kend, 
        istart_in, iend_in, jstart_in, jend_in, kstart_in, kend_in, isnonzero
    );
    
    // For multiple MPI processes: compute offsets for neighbor alltoallv
    if (nproc > 1)
    {
        int *sendcounts = EVA_buff->ijk_se_displs_nonorth + 26 * 13;
        int *sdispls    = EVA_buff->ijk_se_displs_nonorth + 26 * 14;
        int *recvcounts = EVA_buff->ijk_se_displs_nonorth + 26 * 15;
        int *rdispls    = EVA_buff->ijk_se_displs_nonorth + 26 * 16;
        
        sendcounts[0]  = sendcounts[2]  = sendcounts[6]  = sendcounts[8]  = ncol * FDn  * FDn  * FDn;
        sendcounts[17] = sendcounts[19] = sendcounts[23] = sendcounts[25] = ncol * FDn  * FDn  * FDn;
        sendcounts[1]  = sendcounts[7]  = sendcounts[18] = sendcounts[24] = ncol * DMnx * FDn  * FDn;
        sendcounts[3]  = sendcounts[5]  = sendcounts[20] = sendcounts[22] = ncol * FDn  * DMny * FDn;
        sendcounts[9]  = sendcounts[11] = sendcounts[14] = sendcounts[16] = ncol * FDn  * FDn  * DMnz;
        sendcounts[4]  = sendcounts[21] = ncol * DMnxny * FDn;
        sendcounts[10] = sendcounts[15] = ncol * DMnx   * FDn  * DMnz;
        sendcounts[12] = sendcounts[13] = ncol * FDn    * DMny * DMnz;
        
        for (int i = 0; i < 26; i++) recvcounts[i] = sendcounts[i];
        
        rdispls[0] = sdispls[0] = 0;
        for (int i = 1; i < 26; i++) rdispls[i] = sdispls[i] = sdispls[i-1] + sendcounts[i-1]; 
        
        // Check if we need to reallocate x_{in, out}
        int halo_size = ncol * 2 * FDn * (4 * FDn * FDn + 2 * FDn * (DMnx + DMny + DMnz) + (DMnxny + DMnx * DMnz + DMny * DMnz));
        size_t halo_msize = sizeof(double) * (size_t) (halo_size);
        if (halo_msize > EVA_buff->halo_msize)
        {
            EVA_buff->halo_size  = halo_size;
            EVA_buff->halo_msize = halo_msize;
            free(EVA_buff->x_in);
            free(EVA_buff->x_out);
            EVA_buff->x_in  = (double*) malloc(EVA_buff->halo_msize);
            EVA_buff->x_out = (double*) malloc(EVA_buff->halo_msize);
            assert(EVA_buff->x_in != NULL && EVA_buff->x_out != NULL);
        }
    }
    
    // Allocate Dx1 and Dx2
    int DMnxexny  = DMnx_ex * DMny;
    int DMnxnyex  = DMnx * DMny_ex;
    int DMnd_xex  = DMnxexny * DMnz;
    int DMnd_yex  = DMnxnyex * DMnz;
    int DMnd_zex  = DMnxny * DMnz_ex;
    int Dx1_len   = 0;
    switch (EVA_buff->cell_type)
    {
        // Hua: comments here are copied from the original file
        // but I double some of them are wrong
        case 11: Dx1_len = DMnd_xex; break;  // df/dy  
        case 12: Dx1_len = DMnd_xex; break;  // df/dz
        case 13: Dx1_len = DMnd_yex; break;  // df/dz
        case 14: Dx1_len = DMnd_xex; break;  // 2*T_12*df/dy + 2*T_13*df/dz
        case 15: Dx1_len = DMnd_zex; break;  // 2*T_13*dV/dx + 2*T_23*dV/dy
        case 16: Dx1_len = DMnd_yex; break;  // 2*T_12*dV/dx + 2*T_23*dV/dz
        case 17: Dx1_len = DMnd_xex; break;  // 2*T_12*df/dy + 2*T_13*df/dz
    }
    size_t Dx1_msize = sizeof(double) * ncol * Dx1_len;
    size_t Dx2_msize = sizeof(double) * ncol * DMnd_yex;
    if (Dx1_msize > EVA_buff->Dx1_msize)
    {
        EVA_buff->Dx1_msize = Dx1_msize;
        free(EVA_buff->Dx1);
        EVA_buff->Dx1 = (double*) malloc(Dx1_msize);
        assert(EVA_buff->Dx1 != NULL);
    }
    if (Dx2_msize > EVA_buff->Dx2_msize)
    {
        EVA_buff->Dx2_msize = Dx2_msize;
        free(EVA_buff->Dx2);
        EVA_buff->Dx2 = (double*) malloc(Dx2_msize);
        assert(EVA_buff->Dx2 != NULL);
    }
    
    // Copy D1 stencils
    int FDn1 = FDn + 1;
    double *D1_stencil_y  = EVA_buff->Grad_stencils + FDn1 * 0;
    double *D1_stencil_z  = EVA_buff->Grad_stencils + FDn1 * 1;
    double *D1_stencil_xy = EVA_buff->Grad_stencils + FDn1 * 2;
    double *D1_stencil_xz = EVA_buff->Grad_stencils + FDn1 * 3;
    double *D1_stencil_yx = EVA_buff->Grad_stencils + FDn1 * 4;
    double *D1_stencil_yz = EVA_buff->Grad_stencils + FDn1 * 5;
    double *D1_stencil_zx = EVA_buff->Grad_stencils + FDn1 * 6;
    double *D1_stencil_zy = EVA_buff->Grad_stencils + FDn1 * 7;
    for (int i = 0; i < FDn1; i++)
    {
        D1_stencil_y[i]  = pSPARC->D1_stencil_coeffs_y[i];
        D1_stencil_z[i]  = pSPARC->D1_stencil_coeffs_z[i];
        D1_stencil_xy[i] = pSPARC->D1_stencil_coeffs_xy[i];
        D1_stencil_xz[i] = pSPARC->D1_stencil_coeffs_xz[i];
        D1_stencil_yx[i] = pSPARC->D1_stencil_coeffs_yx[i];
        D1_stencil_yz[i] = pSPARC->D1_stencil_coeffs_yz[i];
        D1_stencil_zx[i] = pSPARC->D1_stencil_coeffs_zx[i];
        D1_stencil_zy[i] = pSPARC->D1_stencil_coeffs_zy[i];
    }
}

// Set up halo values in the extended domain when there is 1 MPI process in the communicator
void EVA_Lap_MV_nonorth_NP1_Halo(EVA_buff_t EVA_buff, const int ncol, const double *x)
{   
    double *x_ex = EVA_buff->x_ex;
    double st, et;
    
    int *istart     = EVA_buff->ijk_se_displs_nonorth + 26 * 0;
    int *iend       = EVA_buff->ijk_se_displs_nonorth + 26 * 1;
    int *jstart     = EVA_buff->ijk_se_displs_nonorth + 26 * 2;
    int *jend       = EVA_buff->ijk_se_displs_nonorth + 26 * 3;
    int *kstart     = EVA_buff->ijk_se_displs_nonorth + 26 * 4;
    int *kend       = EVA_buff->ijk_se_displs_nonorth + 26 * 5;
    int *istart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 6;
    int *jstart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 8;
    int *kstart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 10;
    int *isnonzero  = EVA_buff->ijk_se_displs_nonorth + 26 * 12;
    
    st = MPI_Wtime();
    EVA_Lap_MV_copy_inner_x(EVA_buff, ncol, x);
    et = MPI_Wtime();
    EVA_buff->Lap_MV_cpyx_t += et - st;
    
    st = MPI_Wtime();
    // Copy the extended part from x into x_ex
    for (int nbr_i = 0; nbr_i < 26; nbr_i++) 
    {
        if (!isnonzero[nbr_i]) continue;
        EVA_Lap_MV_copy_periodic_halo(
            EVA_buff, x, ncol, kstart[nbr_i], jstart[nbr_i], istart[nbr_i],
            kend[nbr_i], jend[nbr_i], iend[nbr_i], 
            kstart_in[nbr_i], jstart_in[nbr_i], istart_in[nbr_i]
        );
    }
    et = MPI_Wtime();
    EVA_buff->Lap_MV_unpk_t += et - st;
}

// Perform halo exchange and copy the received halo values to the extended 
// domain when there are more than 1 MPI processes in the communicator
void EVA_Lap_MV_nonorth_NPK_Halo(EVA_buff_t EVA_buff, const int ncol, const double *x, MPI_Comm comm)
{ 
    int *istart     = EVA_buff->ijk_se_displs_nonorth + 26 * 0;
    int *iend       = EVA_buff->ijk_se_displs_nonorth + 26 * 1;
    int *jstart     = EVA_buff->ijk_se_displs_nonorth + 26 * 2;
    int *jend       = EVA_buff->ijk_se_displs_nonorth + 26 * 3;
    int *kstart     = EVA_buff->ijk_se_displs_nonorth + 26 * 4;
    int *kend       = EVA_buff->ijk_se_displs_nonorth + 26 * 5;
    int *istart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 6;
    int *iend_in    = EVA_buff->ijk_se_displs_nonorth + 26 * 7;
    int *jstart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 8;
    int *jend_in    = EVA_buff->ijk_se_displs_nonorth + 26 * 9;
    int *kstart_in  = EVA_buff->ijk_se_displs_nonorth + 26 * 10;
    int *kend_in    = EVA_buff->ijk_se_displs_nonorth + 26 * 11;
    int *sendcounts = EVA_buff->ijk_se_displs_nonorth + 26 * 13;
    int *sdispls    = EVA_buff->ijk_se_displs_nonorth + 26 * 14;
    int *recvcounts = EVA_buff->ijk_se_displs_nonorth + 26 * 15;
    int *rdispls    = EVA_buff->ijk_se_displs_nonorth + 26 * 16;

    double *x_in  = EVA_buff->x_in;
    double *x_out = EVA_buff->x_out;
    double *x_ex  = EVA_buff->x_ex;
    double st, et;
    
    // Necessary, DO NOT REMOVE
    #pragma omp parallel for 
    for (int i = 0; i < EVA_buff->halo_size; i++) x_in[i] = 0.0;

    st = MPI_Wtime();
    int count = 0;
    for (int nbr_i = 0; nbr_i < 26; nbr_i++) 
    {
        EVA_Lap_MV_pack_halo(
            EVA_buff, x, ncol, kstart[nbr_i], jstart[nbr_i], istart[nbr_i],
            kend[nbr_i], jend[nbr_i], iend[nbr_i], &count
        );
    }
    et = MPI_Wtime();
    EVA_buff->Lap_MV_pack_t += et - st;
    
    // Halo exchange
    MPI_Request request;
    MPI_Ineighbor_alltoallv(
        x_out, sendcounts, sdispls, MPI_DOUBLE, 
        x_in,  recvcounts, rdispls, MPI_DOUBLE, 
        comm, &request
    );

    // Overlap copy inner_x with halo exchange
    st = MPI_Wtime();
    EVA_Lap_MV_copy_inner_x(EVA_buff, ncol, x);
    et = MPI_Wtime();
    EVA_buff->Lap_MV_cpyx_t += et - st;
    
    // Wait halo exchange finish
    st = MPI_Wtime();
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    et = MPI_Wtime();
    EVA_buff->Lap_MV_comm_t += et - st;

    // Copy receive buffer into extended domain
    st = MPI_Wtime();
    count = 0;
    for (int nbr_i = 0; nbr_i < 26; nbr_i++) 
    {
        EVA_Lap_MV_unpack_halo(
            EVA_buff, ncol, kstart_in[nbr_i], jstart_in[nbr_i], istart_in[nbr_i],
            kend_in[nbr_i], jend_in[nbr_i], iend_in[nbr_i], &count
        );
    }
    et = MPI_Wtime();
    EVA_buff->Lap_MV_unpk_t += et - st;
}

void DX_stencil_thread(
    const double *x0, double *x1, const int radius, const int x0_stride_x,
    const int x0_stride_y, const int x1_stride_y, const int x0_stride_z, const int x1_stride_z,
    const int x1_x_spos, const int x1_y_spos, const int x1_z_spos,
    const int x1_x_epos, const int x1_y_epos, const int x1_z_epos,
    const int ix_offset, const int iy_offset, const int iz_offset, 
    const double *stencil_coefs, const double coef_0
)
{
    for (int z = x1_z_spos; z < x1_z_epos; z++)
    {
        int iz = z + iz_offset;
        for (int y = x1_y_spos; y < x1_y_epos; y++)
        {
            int iy = y + iy_offset;
            int idx_x0_zy = iz * x0_stride_z + iy * x0_stride_y;
            int idx_x1_zy =  z * x1_stride_z +  y * x1_stride_y;
            
            #pragma omp simd
            for (int x = x1_x_spos; x < x1_x_epos; x++)
            {
                int idx_x0 = idx_x0_zy + x + ix_offset;
                double temp = x0[idx_x0] * coef_0;
                for (int r = 1; r <= radius; r++)
                {
                    int x0_stride_x_r = r * x0_stride_x;
                    temp += (x0[idx_x0 + x0_stride_x_r] - x0[idx_x0 - x0_stride_x_r]) * stencil_coefs[r];
                }
                x1[idx_x1_zy + x] = temp;
            }
        }
    }
}

void DX1_DX2_stencil_thread(
    const double *x0, double *x1, const int radius, const int x0_stride_0, const int x0_stride_1,
    const int x0_stride_y, const int x1_stride_y, const int x0_stride_z, const int x1_stride_z,
    const int x1_x_spos, const int x1_y_spos, const int x1_z_spos,
    const int x1_x_epos, const int x1_y_epos, const int x1_z_epos,
    const int ix_offset, const int iy_offset, const int iz_offset, 
    const double *stencil_coefs0,  const double *stencil_coefs1
)
{
    for (int z = x1_z_spos; z < x1_z_epos; z++)
    {
        int iz = z + iz_offset;
        for (int y = x1_y_spos; y < x1_y_epos; y++)
        {
            int iy = y + iy_offset;
            int idx_x0_zy = iz * x0_stride_z + iy * x0_stride_y;
            int idx_x1_zy =  z * x1_stride_z +  y * x1_stride_y;
            
            #pragma omp simd
            for (int x = x1_x_spos; x < x1_x_epos; x++)
            {
                int idx_x0 = idx_x0_zy + x + ix_offset;
                double temp1 = 0.0, temp2 = 0.0;
                for (int r = 1; r <= radius; r++)
                {
                    int x0_stride_0_r = r * x0_stride_0;
                    int x0_stride_1_r = r * x0_stride_1;
                    temp1 += (x0[idx_x0 + x0_stride_0_r] - x0[idx_x0 - x0_stride_0_r]) * stencil_coefs0[r];
                    temp2 += (x0[idx_x0 + x0_stride_1_r] - x0[idx_x0 - x0_stride_1_r]) * stencil_coefs1[r];
                }
                x1[idx_x1_zy + x] = temp1 + temp2;
            }
        }
    }
}

void Lap_DX_stencil_thread(
    const double *x0, const double *Dx, const int radius, const int stride_Dx, 
    const int x1_stride_y, const int x0_stride_y, const int Dx_stride_y, 
    const int x1_stride_z, const int x0_stride_z, const int Dx_stride_z, 
    const int x1_x_spos, const int x1_y_spos, const int x1_z_spos, 
    const int x1_x_epos, const int x1_y_epos, const int x1_z_epos,
    const int x0_x_offset, const int x0_y_offset, const int x0_z_offset,
    const int Dx_x_offset, const int Dx_y_offset, const int Dx_z_offset,
    const double *stencil_coefs, const double coef_0, const double b,
    const double *v0, double *x1
)
{
    const int z_len = x1_z_epos - x1_z_spos;
    const int y_len = x1_y_epos - x1_y_spos;
    const int x_len = x1_x_epos - x1_x_spos;
    for (int z = 0; z < z_len; z++)
    {
        int x1_z = (x1_z_spos + z) * x1_stride_z;
        int x0_z = (x1_z_spos + z + x0_z_offset) * x0_stride_z;
        int Dx_z = (x1_z_spos + z + Dx_z_offset) * Dx_stride_z;
        for (int y = 0; y < y_len; y++)
        {
            int x1_zy = x1_z + (x1_y_spos + y) * x1_stride_y;
            int x0_zy = x0_z + (x1_y_spos + y + x0_y_offset) * x0_stride_y;
            int Dx_zy = Dx_z + (x1_y_spos + y + Dx_y_offset) * Dx_stride_y;
            #pragma omp simd
            for (int x = 0; x < x_len; x++)
            {
                int x1_zyx = x1_zy + (x1_x_spos + x);
                int x0_zyx = x0_zy + (x1_x_spos + x + x0_x_offset);
                int Dx_zyx = Dx_zy + (x1_x_spos + x + Dx_x_offset);
                double res = coef_0 * x0[x0_zyx];
                for (int r = 1; r <= radius; r++)
                {
                    int r_fac = 4 * r + 1;
                    int stride_y_r  = r * x0_stride_y;
                    int stride_z_r  = r * x0_stride_z;
                    int stride_Dx_r = r * stride_Dx;
                    res += (x0[x0_zyx - r]           + x0[x0_zyx + r])           * stencil_coefs[r_fac];
                    res += (x0[x0_zyx - stride_y_r]  + x0[x0_zyx + stride_y_r])  * stencil_coefs[r_fac+1];
                    res += (x0[x0_zyx - stride_z_r]  + x0[x0_zyx + stride_z_r])  * stencil_coefs[r_fac+2];
                    res += (Dx[Dx_zyx + stride_Dx_r] - Dx[Dx_zyx - stride_Dx_r]) * stencil_coefs[r_fac+3];
                }
                x1[x1_zyx] = res + b * (v0[x1_zyx] * x0[x0_zyx]);
            }
        }
    }
}

void Lap_DX1_DX2_stencil_thread(
    const double *x0, const double *Dx1, const double *Dx2, 
    const int radius, const int stride_Dx1, const int stride_Dx2,
    const int x1_stride_y, const int x0_stride_y, const int Dx1_stride_y, const int Dx2_stride_y,
    const int x1_stride_z, const int x0_stride_z, const int Dx1_stride_z, const int Dx2_stride_z,
    const int x1_x_spos,  const int x1_y_spos,  const int x1_z_spos,
    const int x1_x_epos,  const int x1_y_epos,  const int x1_z_epos,
    const int x0_x_offset,  const int x0_y_offset,  const int x0_z_offset,
    const int Dx1_x_offset, const int Dx1_y_offset, const int Dx1_z_offset,
    const int Dx2_x_offset, const int Dx2_y_offset, const int Dx2_z_offset,
    const double *stencil_coefs, const double coef_0, const double b,
    const double *v0, double *x1
)
{
    const int z_len = x1_z_epos - x1_z_spos;
    const int y_len = x1_y_epos - x1_y_spos;
    const int x_len = x1_x_epos - x1_x_spos;
    for (int z = 0; z < z_len; z++)
    {
        int x1_z  = (x1_z_spos + z) * x1_stride_z;
        int x0_z  = (x1_z_spos + z +  x0_z_offset) * x0_stride_z;
        int Dx1_z = (x1_z_spos + z + Dx1_z_offset) * Dx1_stride_z;
        int Dx2_z = (x1_z_spos + z + Dx2_z_offset) * Dx2_stride_z;
        for (int y = 0; y < y_len; y++)
        {
            int x1_zy  = x1_z  + (x1_y_spos + y) * x1_stride_y;
            int x0_zy  = x0_z  + (x1_y_spos + y +  x0_y_offset) * x0_stride_y;
            int Dx1_zy = Dx1_z + (x1_y_spos + y + Dx1_y_offset) * Dx1_stride_y;
            int Dx2_zy = Dx2_z + (x1_y_spos + y + Dx2_y_offset) * Dx2_stride_y;
            #pragma omp simd
            for (int x = 0; x < x_len; x++)
            {
                int x1_zyx  = x1_zy  + (x1_x_spos + x);
                int x0_zyx  = x0_zy  + (x1_x_spos + x +  x0_x_offset);
                int Dx1_zyx = Dx1_zy + (x1_x_spos + x + Dx1_x_offset);
                int Dx2_zyx = Dx2_zy + (x1_x_spos + x + Dx2_x_offset);
                double res = coef_0 * x0[x0_zyx];
                for (int r = 1; r <= radius; r++)
                {
                    int r_fac = 5 * r;
                    int stride_y_r = r * x0_stride_y;
                    int stride_z_r = r * x0_stride_z;
                    int stride_Dx1_r = r * stride_Dx1;
                    int stride_Dx2_r = r * stride_Dx2;
                    res += (x0[x0_zyx - r]              + x0[x0_zyx + r])              * stencil_coefs[r_fac];
                    res += (x0[x0_zyx - stride_y_r]     + x0[x0_zyx + stride_y_r])     * stencil_coefs[r_fac+1];
                    res += (x0[x0_zyx - stride_z_r]     + x0[x0_zyx + stride_z_r])     * stencil_coefs[r_fac+2];
                    res += (Dx1[Dx1_zyx + stride_Dx1_r] - Dx1[Dx1_zyx - stride_Dx1_r]) * stencil_coefs[r_fac+3];
                    res += (Dx2[Dx2_zyx + stride_Dx2_r] - Dx2[Dx2_zyx - stride_Dx2_r]) * stencil_coefs[r_fac+4];
                }
                x1[x1_zyx] = res + b * (v0[x1_zyx] * x0[x0_zyx]);
            }
        }
    }
}

void EVA_DX_stencil_omp(
    EVA_buff_t EVA_buff,   const int ncol, 
    const int DMnx_in,     const int DMny_in,     const int DMnz_in,
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const int x0_stride_x, const int x1_stride_y, const int DMnd_Dx, 
    double *Dx_stencil, double *Dx
)
{
    int FDn  = EVA_buff->FDn;
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    int DMnxexny = EVA_buff->DMnx_ex * EVA_buff->DMny;
    int stride_z_ex = EVA_buff->DMnx_ex * EVA_buff->DMny_ex;
    
    int nblk_x, nblk_y, nblk_z, nblk_xy, nblk_xyz;
    int x_blk_size, y_blk_size, z_blk_size;
    
    const int x0_stride_y = DMnx_ex;
    const int x0_stride_z = stride_z_ex;
    const int x1_stride_z = DMnxexny;
    
    EVA_Lap_MV_stencil_omp_dd(
        EVA_buff->nthreads, DMnx_in, DMny_in, DMnz_in,
        &nblk_x, &nblk_y, &nblk_z, &x_blk_size, &y_blk_size, &z_blk_size
    );
    nblk_xy  = nblk_x  * nblk_y;
    nblk_xyz = nblk_xy * nblk_z;
    #pragma omp parallel for schedule(static)
    for (int i_domain = 0; i_domain < nblk_xyz * ncol; i_domain++)
    {
        int icol = i_domain / nblk_xyz;
        int iblk = i_domain % nblk_xyz;
        double *x_ex_icol = EVA_buff->x_ex + icol * DMnd_ex;
        double *Dx_icol = Dx + icol * DMnd_Dx;
        int x_spos, y_spos, z_spos, x_epos, y_epos, z_epos;
        EVA_Lap_MV_unpack_dd_idx(
            iblk, nblk_x, nblk_xy, DMnx_in, DMny_in, DMnz_in,
            x_blk_size, y_blk_size, z_blk_size,
            &x_spos, &y_spos, &z_spos, &x_epos, &y_epos, &z_epos
        );
        DX_stencil_thread(
            x_ex_icol, Dx_icol, FDn, x0_stride_x, 
            x0_stride_y, x1_stride_y, x0_stride_z, x1_stride_z,
            x_spos, y_spos, z_spos, x_epos, y_epos, z_epos, 
            ix_offset, iy_offset, iz_offset, Dx_stencil, 0.0
        );
    }
}
    
void EVA_DX1_DX2_stencil_omp(
    EVA_buff_t EVA_buff,   const int ncol,        const int DMnd_Dx, 
    const int DMnx_in,     const int DMny_in,     const int DMnz_in,
    const int x0_stride_0, const int x0_stride_1, const int x0_stride_y, 
    const int x1_stride_y, const int x0_stride_z, const int x1_stride_z,
    const int ix_offset,   const int iy_offset,   const int iz_offset, 
    const double *stencil_coefs0,  const double *stencil_coefs1
)
{
    int FDn  = EVA_buff->FDn;
    int DMnd_ex = EVA_buff->DMnd_ex;
    
    int nblk_x, nblk_y, nblk_z, nblk_xy, nblk_xyz;
    int x_blk_size, y_blk_size, z_blk_size;
    
    EVA_Lap_MV_stencil_omp_dd(
        EVA_buff->nthreads, DMnx_in, DMny_in, DMnz_in,
        &nblk_x, &nblk_y, &nblk_z, &x_blk_size, &y_blk_size, &z_blk_size
    );
    nblk_xy  = nblk_x  * nblk_y;
    nblk_xyz = nblk_xy * nblk_z;
    #pragma omp parallel for schedule(static)
    for (int i_domain = 0; i_domain < nblk_xyz * ncol; i_domain++)
    {
        int icol = i_domain / nblk_xyz;
        int iblk = i_domain % nblk_xyz;
        double *x_ex_icol = EVA_buff->x_ex + icol * DMnd_ex;
        double *Dx1_icol = EVA_buff->Dx1 + icol * DMnd_Dx;
        int x_spos, y_spos, z_spos, x_epos, y_epos, z_epos;
        EVA_Lap_MV_unpack_dd_idx(
            iblk, nblk_x, nblk_xy, DMnx_in, DMny_in, DMnz_in,
            x_blk_size, y_blk_size, z_blk_size,
            &x_spos, &y_spos, &z_spos, &x_epos, &y_epos, &z_epos
        );
        DX1_DX2_stencil_thread(
            x_ex_icol, Dx1_icol, FDn, x0_stride_0, x0_stride_1, 
            x0_stride_y, x1_stride_y, x0_stride_z, x1_stride_z, 
            x_spos, y_spos, z_spos, x_epos, y_epos, z_epos, 
            ix_offset, iy_offset, iz_offset, stencil_coefs0, stencil_coefs1
        );
    }
}

void EVA_Lap_nonorth_DX_omp(
    EVA_buff_t EVA_buff, const int ncol, 
    const int DMnd_Dx,     const int stride_Dx,   const int x1_stride_y, 
    const int x0_stride_y, const int Dx_stride_y, const int Dx_stride_z, 
    const int Dx_x_offset, const int Dx_y_offset, const int Dx_z_offset, 
    const double _b, const double *_v, double *y
)
{
    int FDn  = EVA_buff->FDn;
    int DMnx = EVA_buff->DMnx;
    int DMny = EVA_buff->DMny;
    int DMnz = EVA_buff->DMnz;
    int DMnd = EVA_buff->DMnd;
    int DMnd_ex = EVA_buff->DMnd_ex;
    int stride_z = DMnx * DMny;
    int stride_z_ex = EVA_buff->DMnx_ex * EVA_buff->DMny_ex;
    
    int nblk_x, nblk_y, nblk_z, nblk_xy, nblk_xyz;
    int x_blk_size, y_blk_size, z_blk_size;

    const int x1_stride_z = stride_z;
    const int x0_stride_z = stride_z_ex;
    
    EVA_Lap_MV_stencil_omp_dd(
        EVA_buff->nthreads, DMnx, DMny, DMnz, 
        &nblk_x, &nblk_y, &nblk_z, &x_blk_size, &y_blk_size, &z_blk_size
    );
    nblk_xy  = nblk_x  * nblk_y;
    nblk_xyz = nblk_xy * nblk_z;
    #pragma omp parallel for schedule(static)
    for (int i_domain = 0; i_domain < nblk_xyz * ncol; i_domain++)
    {
        int icol = i_domain / nblk_xyz;
        int iblk = i_domain % nblk_xyz;
        double *x_ex_icol = EVA_buff->x_ex + icol * DMnd_ex;
        double *Dx1_icol = EVA_buff->Dx1 + icol * DMnd_Dx;
        double *y_icol = y + icol * DMnd;
        int x_spos, y_spos, z_spos, x_epos, y_epos, z_epos;
        EVA_Lap_MV_unpack_dd_idx(
            iblk, nblk_x, nblk_xy, DMnx, DMny, DMnz, 
            x_blk_size, y_blk_size, z_blk_size,
            &x_spos, &y_spos, &z_spos, &x_epos, &y_epos, &z_epos
        );
        Lap_DX_stencil_thread(
            x_ex_icol, Dx1_icol, FDn, stride_Dx, 
            x1_stride_y, x0_stride_y, Dx_stride_y, x1_stride_z, x0_stride_z, Dx_stride_z,
            x_spos, y_spos, z_spos, x_epos, y_epos, z_epos, FDn, FDn, FDn, 
            Dx_x_offset, Dx_y_offset, Dx_z_offset,
            EVA_buff->Lap_wt_nonorth, EVA_buff->w2_diag, _b, _v, y_icol
        );
    }
}


void EVA_Lap_nonorth_DX1_DX2_omp(
    EVA_buff_t EVA_buff, const int ncol, 
    const int stride_Dx1,   const int stride_Dx2,   const int DMnd_Dx1,     const int DMnd_Dx2,   
    const int x1_stride_y,  const int x0_stride_y,  const int Dx1_stride_y, const int Dx2_stride_y,
    const int x1_stride_z,  const int x0_stride_z,  const int Dx1_stride_z, const int Dx2_stride_z,
    const int x0_x_offset,  const int x0_y_offset,  const int x0_z_offset,
    const int Dx1_x_offset, const int Dx1_y_offset, const int Dx1_z_offset,
    const int Dx2_x_offset, const int Dx2_y_offset, const int Dx2_z_offset,
    const double _b, const double *_v, double *y
)
{
    int FDn  = EVA_buff->FDn;
    int DMnx = EVA_buff->DMnx;
    int DMny = EVA_buff->DMny;
    int DMnz = EVA_buff->DMnz;
    int DMnd = EVA_buff->DMnd;
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    int DMny_ex = EVA_buff->DMny_ex;
    int stride_z = DMnx * DMny;
    int stride_z_ex = DMnx_ex * DMny_ex;
    
    int nblk_x, nblk_y, nblk_z, nblk_xy, nblk_xyz;
    int x_blk_size, y_blk_size, z_blk_size;
    
    EVA_Lap_MV_stencil_omp_dd(
        EVA_buff->nthreads, DMnx, DMny, DMnz,
        &nblk_x, &nblk_y, &nblk_z, &x_blk_size, &y_blk_size, &z_blk_size
    );
    nblk_xy  = nblk_x  * nblk_y;
    nblk_xyz = nblk_xy * nblk_z;
    #pragma omp parallel for schedule(static)
    for (int i_domain = 0; i_domain < nblk_xyz * ncol; i_domain++)
    {
        int icol = i_domain / nblk_xyz;
        int iblk = i_domain % nblk_xyz;
        double *x_ex_icol = EVA_buff->x_ex + icol * DMnd_ex;
        double *Dx1_icol = EVA_buff->Dx1 + icol * DMnd_Dx1;
        double *Dx2_icol = EVA_buff->Dx2 + icol * DMnd_Dx2;
        double *y_icol = y + icol * DMnd;
        int x_spos, y_spos, z_spos, x_epos, y_epos, z_epos;
        EVA_Lap_MV_unpack_dd_idx(
            iblk, nblk_x, nblk_xy, DMnx, DMny, DMnz,
            x_blk_size, y_blk_size, z_blk_size,
            &x_spos, &y_spos, &z_spos, &x_epos, &y_epos, &z_epos
        );
        Lap_DX1_DX2_stencil_thread(
            x_ex_icol, Dx1_icol, Dx2_icol, 
            FDn, stride_Dx1, stride_Dx2, 
            x1_stride_y, x0_stride_y, Dx1_stride_y, Dx2_stride_y,
            x1_stride_z, x0_stride_z, Dx1_stride_z, Dx2_stride_z,
            x_spos, y_spos, z_spos, x_epos, y_epos, z_epos,
            x0_x_offset, x0_y_offset, x0_z_offset,
            Dx1_x_offset, Dx1_y_offset, Dx1_z_offset,
            Dx2_x_offset, Dx2_y_offset, Dx2_z_offset,
            EVA_buff->Lap_wt_nonorth, EVA_buff->w2_diag, _b, _v, y_icol
        );
    }
}


#define LAP_MV_NONORTH_PARAMS \
    EVA_buff_t EVA_buff, const int ncol, const double *x, const double *v, double *y

#define LAP_MV_NONORTH_LOAD_PARAMS \
    const double *_v = v;               \
    double _b = EVA_buff->b;            \
    if (fabs(_b) < 1e-14 || v == NULL)  \
    {                                   \
        _v = x;                         \
        _b = 0.0;                       \
    }                                   \
    int FDn  = EVA_buff->FDn;           \
    int FDn1 = FDn + 1;                 \
    int DMnx = EVA_buff->DMnx;          \
    int DMny = EVA_buff->DMny;          \
    int DMnz = EVA_buff->DMnz;          \
    int DMnd = EVA_buff->DMnd;          \
    int DMnxny  = DMnx * DMny;          \
    int DMnx_ex = EVA_buff->DMnx_ex;    \
    int DMny_ex = EVA_buff->DMny_ex;    \
    int DMnz_ex = EVA_buff->DMnz_ex;    \
    int DMnd_ex = EVA_buff->DMnd_ex;    \
    int stride_z    = DMnx * DMny;          \
    int stride_z_ex = DMnx_ex * DMny_ex;    \
    int DMnxexny    = DMnx_ex * DMny;       \
    int DMnd_xex    = DMnxexny * DMnz;      \
    int DMnxnyex    = DMnx * DMny_ex;       \
    int DMnd_yex    = DMnxnyex * DMnz;      \
    int DMnd_zex    = DMnxny * DMnz_ex;     \
    double w2_diag  = EVA_buff->w2_diag;        \
    double *Lap_wt  = EVA_buff->Lap_wt_nonorth; \
    double *x_ex    = EVA_buff->x_ex;           \
    double *Dx1     = EVA_buff->Dx1;            \
    double *Dx2     = EVA_buff->Dx2;            \
    double *D1_stencil_y  = EVA_buff->Grad_stencils + FDn1 * 0; \
    double *D1_stencil_z  = EVA_buff->Grad_stencils + FDn1 * 1; \
    double *D1_stencil_xy = EVA_buff->Grad_stencils + FDn1 * 2; \
    double *D1_stencil_xz = EVA_buff->Grad_stencils + FDn1 * 3; \
    double *D1_stencil_yx = EVA_buff->Grad_stencils + FDn1 * 4; \
    double *D1_stencil_yz = EVA_buff->Grad_stencils + FDn1 * 5; \
    double *D1_stencil_zx = EVA_buff->Grad_stencils + FDn1 * 6; \
    double *D1_stencil_zy = EVA_buff->Grad_stencils + FDn1 * 7; \
    int nblk_x, nblk_y, nblk_z, nblk_xy, nblk_xyz; \
    int x_blk_size, y_blk_size, z_blk_size;

void EVA_Lap_MV_nonorth_ct11(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;
    
    double st = MPI_Wtime();

    EVA_DX_stencil_omp(
        EVA_buff, ncol, DMnx_ex, DMny, DMnz, 0, FDn, FDn,
        DMnx_ex, DMnx_ex, DMnd_xex, D1_stencil_y, Dx1
    );
    
    EVA_Lap_nonorth_DX_omp(
        EVA_buff, ncol, 
        DMnd_xex, 1, DMnx, DMnx_ex, DMnx_ex, DMnxexny,
        FDn, 0, 0, _b, _v, y
    );
    
    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth_ct12(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;
    
    double st = MPI_Wtime();

    EVA_DX_stencil_omp(
        EVA_buff, ncol, DMnx_ex, DMny, DMnz, 0, FDn, FDn,
        stride_z_ex, DMnx_ex, DMnd_xex, D1_stencil_z, Dx1
    );

    EVA_Lap_nonorth_DX_omp(
        EVA_buff, ncol, 
        DMnd_xex, 1, DMnx, DMnx_ex, DMnx_ex, DMnxexny,
        FDn, 0, 0, _b, _v, y
    );
    
    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth_ct13(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;
    
    double st = MPI_Wtime();
    
    EVA_DX_stencil_omp(
        EVA_buff, ncol, DMnx, DMny_ex, DMnz, FDn, 0, FDn, 
        stride_z_ex, DMnx, DMnd_yex, D1_stencil_z, Dx1
    );
    
    EVA_Lap_nonorth_DX_omp(
        EVA_buff, ncol, 
        DMnd_yex, DMnx, DMnx, DMnx_ex, DMnx, DMnxnyex,
        0, FDn, 0, _b, _v, y
    );

    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth_ct14(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;
    
    double st = MPI_Wtime();

    EVA_DX1_DX2_stencil_omp(
        EVA_buff, ncol, DMnd_xex, DMnx_ex, DMny, DMnz,
        DMnx_ex, stride_z_ex, DMnx_ex, DMnx_ex, stride_z_ex, DMnxexny,
        0, FDn, FDn, D1_stencil_xy, D1_stencil_xz
    );

    EVA_Lap_nonorth_DX_omp(
        EVA_buff, ncol, 
        DMnd_xex, 1, DMnx, DMnx_ex, DMnx_ex, DMnxexny,
        FDn, 0, 0, _b, _v, y
    );

    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth_ct15(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;
    
    double st = MPI_Wtime();

    EVA_DX1_DX2_stencil_omp(
        EVA_buff, ncol, DMnd_zex, DMnx, DMny, DMnz_ex, 
        1, DMnx_ex, DMnx_ex, DMnx, stride_z_ex, DMnxny,
        FDn, FDn, 0, D1_stencil_zx, D1_stencil_zy
    );

    EVA_Lap_nonorth_DX_omp(
        EVA_buff, ncol, 
        DMnd_zex, DMnxny, DMnx, DMnx_ex, DMnx, DMnxny,
        0, 0, FDn, _b, _v, y
    );
    
    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth_ct16(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;
    
    double st = MPI_Wtime();
    
    EVA_DX1_DX2_stencil_omp(
        EVA_buff, ncol, DMnd_yex, DMnx, DMny_ex, DMnz, 
        1, stride_z_ex, DMnx_ex, DMnx, stride_z_ex, DMnxnyex,
        FDn, 0, FDn, D1_stencil_yx, D1_stencil_yz
    );
    
    EVA_Lap_nonorth_DX_omp(
        EVA_buff, ncol, 
        DMnd_yex, DMnx, DMnx, DMnx_ex, DMnx, DMnxnyex,
        0, FDn, 0, _b, _v, y
    );

    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth_ct17(LAP_MV_NONORTH_PARAMS)
{
    LAP_MV_NONORTH_LOAD_PARAMS;

    double st = MPI_Wtime();
    
    // Calc DX1DX2
    EVA_DX1_DX2_stencil_omp(
        EVA_buff, ncol, DMnd_xex, DMnx_ex, DMny, DMnz,
        DMnx_ex, stride_z_ex, DMnx_ex, DMnx_ex, stride_z_ex, DMnxexny,
       0, FDn, FDn, D1_stencil_xy, D1_stencil_xz
    );
    
    // Calc DX
    EVA_DX_stencil_omp(
        EVA_buff, ncol, DMnx, DMny_ex, DMnz, FDn, 0, FDn, 
        stride_z_ex, DMnx, DMnd_yex, D1_stencil_z, Dx2
    );
    
    // Calc y
    EVA_Lap_nonorth_DX1_DX2_omp(
        EVA_buff, ncol, 
        1, DMnx, DMnd_xex, DMnd_yex,
        DMnx, DMnx_ex, DMnx_ex, DMnx, 
        stride_z, stride_z_ex, DMnxexny, DMnxnyex, 
        FDn, FDn, FDn, FDn, 0, 0, 0, FDn, 0, 
        _b, _v, y
        
    );

    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

void EVA_Lap_MV_nonorth(
    EVA_buff_t EVA_buff, const int ncol, const double *x, const double *v, 
    MPI_Comm comm, const int nproc, double *Hx
)
{
    // Halo exchange and pack data to halo
    if (nproc > 1) EVA_Lap_MV_nonorth_NPK_Halo(EVA_buff, ncol, x, comm);
    else           EVA_Lap_MV_nonorth_NP1_Halo(EVA_buff, ncol, x);
    
    switch (EVA_buff->cell_type)
    {
        case 11: EVA_Lap_MV_nonorth_ct11(EVA_buff, ncol, x, v, Hx); break;
        case 12: EVA_Lap_MV_nonorth_ct12(EVA_buff, ncol, x, v, Hx); break;
        case 13: EVA_Lap_MV_nonorth_ct13(EVA_buff, ncol, x, v, Hx); break;
        case 14: EVA_Lap_MV_nonorth_ct14(EVA_buff, ncol, x, v, Hx); break;
        case 15: EVA_Lap_MV_nonorth_ct15(EVA_buff, ncol, x, v, Hx); break;
        case 16: EVA_Lap_MV_nonorth_ct16(EVA_buff, ncol, x, v, Hx); break;
        case 17: EVA_Lap_MV_nonorth_ct17(EVA_buff, ncol, x, v, Hx); break;
    }

    EVA_buff->Lap_MV_rhs += ncol;
}
