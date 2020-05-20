/**
 * @file    EVA_Lap_MV_orth.c
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Functions for multiplying Laplacian matrix with multiple vectors
 *          For orthogonal cells
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
#include "EVA_buff.h"
#include "EVA_Lap_MV_common.h"
#include "EVA_Lap_MV_orth.h"

// Initialize parameters used in EVA_Lap_MV_orth
void EVA_init_Lap_MV_orth_params(
    const SPARC_OBJ *pSPARC, const int *DMVertices, const int ncol, const double a, 
    const double b, const double c, MPI_Comm comm, const int nproc, EVA_buff_t EVA_buff
)
{
    if (pSPARC->cell_typ != 0)
    {
        printf(RED"FATAL: EVA_init_Lap_MV_orth_params is for orthogonal cell!\n");
        assert(pSPARC->cell_typ == 0);
    }
    
    int *dims      = &EVA_buff->cart_array[0];
    int *periods   = &EVA_buff->cart_array[3];
    int *my_coords = &EVA_buff->cart_array[6];
    if (nproc > 1)
    {
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    } else {
        periods[0] = 1 - pSPARC->BCx;
        periods[1] = 1 - pSPARC->BCy;
        periods[2] = 1 - pSPARC->BCz;
    }
    
    int FDn = EVA_buff->FDn;
    
    int DMnx = 1 - DMVertices[0] + DMVertices[1];
    int DMny = 1 - DMVertices[2] + DMVertices[3];
    int DMnz = 1 - DMVertices[4] + DMVertices[5];
    int DMnd = DMnx * DMny * DMnz;
    
    int DMnx_ex = DMnx + pSPARC->order;
    int DMny_ex = DMny + pSPARC->order;
    int DMnz_ex = DMnz + pSPARC->order;
    int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;
    
    double w2_diag;
    w2_diag  = pSPARC->D2_stencil_coeffs_x[0]; 
    w2_diag += pSPARC->D2_stencil_coeffs_y[0];
    w2_diag += pSPARC->D2_stencil_coeffs_z[0];
    w2_diag  = a * w2_diag + c;
    
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
    
    double *Lap_stencil_x = EVA_buff->Lap_stencils + 0 * (FDn + 1);
    double *Lap_stencil_y = EVA_buff->Lap_stencils + 1 * (FDn + 1);
    double *Lap_stencil_z = EVA_buff->Lap_stencils + 2 * (FDn + 1);
    for (int p = 0; p < FDn + 1; p++)
    {
        Lap_stencil_x[p] = pSPARC->D2_stencil_coeffs_x[p] * a;
        Lap_stencil_y[p] = pSPARC->D2_stencil_coeffs_y[p] * a;
        Lap_stencil_z[p] = pSPARC->D2_stencil_coeffs_z[p] * a;
    }
    
    // Check if we need to reallocate x_ex
    size_t x_ex_msize = sizeof(double) * ncol * DMnd_ex;
    if (x_ex_msize > EVA_buff->x_ex_msize)
    {
        EVA_buff->x_ex_msize = x_ex_msize;
        free(EVA_buff->x_ex);
        EVA_buff->x_ex = (double*) malloc(EVA_buff->x_ex_msize);
        assert(EVA_buff->x_ex != NULL);
    }
    
    // For multiple MPI processes: set up send buffer based on the ordering of the neighbors
    // For single MPI process: set up copy indices for sending edge nodes in x
    int *istart = EVA_buff->ijk_se_displs + 6 * 0;
    int *iend   = EVA_buff->ijk_se_displs + 6 * 1;
    int *jstart = EVA_buff->ijk_se_displs + 6 * 2;
    int *jend   = EVA_buff->ijk_se_displs + 6 * 3;
    int *kstart = EVA_buff->ijk_se_displs + 6 * 4;
    int *kend   = EVA_buff->ijk_se_displs + 6 * 5;
    istart[0] = 0;        iend[0] = FDn;  jstart[0] = 0;        jend[0] = DMny; kstart[0] = 0;        kend[0] = DMnz;
    istart[1] = DMnx-FDn; iend[1] = DMnx; jstart[1] = 0;        jend[1] = DMny; kstart[1] = 0;        kend[1] = DMnz; 
    istart[2] = 0;        iend[2] = DMnx; jstart[2] = 0;        jend[2] = FDn;  kstart[2] = 0;        kend[2] = DMnz;
    istart[3] = 0;        iend[3] = DMnx; jstart[3] = DMny-FDn; jend[3] = DMny; kstart[3] = 0;        kend[3] = DMnz;
    istart[4] = 0;        iend[4] = DMnx; jstart[4] = 0;        jend[4] = DMny; kstart[4] = 0;        kend[4] = FDn;
    istart[5] = 0;        iend[5] = DMnx; jstart[5] = 0;        jend[5] = DMny; kstart[5] = DMnz-FDn; kend[5] = DMnz;
    
    // For multiple MPI processes: set up start and end indices for copy receive buffer
    // For single MPI process: set up start and end indices for copying edge nodes in x_ex
    int *istart_in = EVA_buff->ijk_se_displs + 6 * 6;
    int *iend_in   = EVA_buff->ijk_se_displs + 6 * 7;
    int *jstart_in = EVA_buff->ijk_se_displs + 6 * 8;
    int *jend_in   = EVA_buff->ijk_se_displs + 6 * 9;
    int *kstart_in = EVA_buff->ijk_se_displs + 6 * 10;
    int *kend_in   = EVA_buff->ijk_se_displs + 6 * 11;
    istart_in[0] = 0;        iend_in[0] = FDn;        jstart_in[0] = FDn;      jend_in[0] = DMny+FDn;   kstart_in[0] = FDn;      kend_in[0] = DMnz+FDn;
    istart_in[1] = DMnx+FDn; iend_in[1] = DMnx+2*FDn; jstart_in[1] = FDn;      jend_in[1] = DMny+FDn;   kstart_in[1] = FDn;      kend_in[1] = DMnz+FDn; 
    istart_in[2] = FDn;      iend_in[2] = DMnx+FDn;   jstart_in[2] = 0;        jend_in[2] = FDn;        kstart_in[2] = FDn;      kend_in[2] = DMnz+FDn;
    istart_in[3] = FDn;      iend_in[3] = DMnx+FDn;   jstart_in[3] = DMny+FDn; jend_in[3] = DMny+2*FDn; kstart_in[3] = FDn;      kend_in[3] = DMnz+FDn;
    istart_in[4] = FDn;      iend_in[4] = DMnx+FDn;   jstart_in[4] = FDn;      jend_in[4] = DMny+FDn;   kstart_in[4] = 0;        kend_in[4] = FDn;
    istart_in[5] = FDn;      iend_in[5] = DMnx+FDn;   jstart_in[5] = FDn;      jend_in[5] = DMny+FDn;   kstart_in[5] = DMnz+FDn; kend_in[5] = DMnz+2*FDn;
    
    // For multiple MPI processes: compute offsets for neighbor alltoallv
    if (nproc > 1)
    {
        int *sendcounts = EVA_buff->ijk_se_displs + 6 * 12;
        int *sdispls    = EVA_buff->ijk_se_displs + 6 * 13;
        int *recvcounts = EVA_buff->ijk_se_displs + 6 * 14;
        int *rdispls    = EVA_buff->ijk_se_displs + 6 * 15;
        sendcounts[0] = sendcounts[1] = recvcounts[0] = recvcounts[1] = ncol * FDn * (DMny * DMnz);
        sendcounts[2] = sendcounts[3] = recvcounts[2] = recvcounts[3] = ncol * FDn * (DMnx * DMnz);
        sendcounts[4] = sendcounts[5] = recvcounts[4] = recvcounts[5] = ncol * FDn * (DMnx * DMny);
        
        rdispls[0] = sdispls[0] = 0;
        rdispls[1] = sdispls[1] = sdispls[0] + sendcounts[0];
        rdispls[2] = sdispls[2] = sdispls[1] + sendcounts[1];
        rdispls[3] = sdispls[3] = sdispls[2] + sendcounts[2];
        rdispls[4] = sdispls[4] = sdispls[3] + sendcounts[3];
        rdispls[5] = sdispls[5] = sdispls[4] + sendcounts[4];
        
        // Check if we need to reallocate x_{in, out}
        int halo_size = (DMnx * DMny + DMny * DMnz + DMnx * DMnz) * ncol * EVA_buff->order;
        size_t halo_msize = sizeof(double) * (size_t) halo_size;
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
}

// Perform halo exchange and copy the received halo values to the extended 
// domain when there are more than 1 MPI processes in the communicator
void EVA_Lap_MV_orth_NPK_Halo(EVA_buff_t EVA_buff, const int ncol, const double *x, MPI_Comm comm)
{
    int DMnx = EVA_buff->DMnx;
    int DMny = EVA_buff->DMny;
    int DMnz = EVA_buff->DMnz;
    int DMnd = EVA_buff->DMnd;
    
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMny_ex = EVA_buff->DMny_ex;
    int DMnz_ex = EVA_buff->DMnz_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    
    int *dims    = &EVA_buff->cart_array[0];
    int *periods = &EVA_buff->cart_array[3];
    
    int *istart = EVA_buff->ijk_se_displs + 6 * 0;
    int *iend   = EVA_buff->ijk_se_displs + 6 * 1;
    int *jstart = EVA_buff->ijk_se_displs + 6 * 2;
    int *jend   = EVA_buff->ijk_se_displs + 6 * 3;
    int *kstart = EVA_buff->ijk_se_displs + 6 * 4;
    int *kend   = EVA_buff->ijk_se_displs + 6 * 5;
    
    int *istart_in = EVA_buff->ijk_se_displs + 6 * 6;
    int *iend_in   = EVA_buff->ijk_se_displs + 6 * 7;
    int *jstart_in = EVA_buff->ijk_se_displs + 6 * 8;
    int *jend_in   = EVA_buff->ijk_se_displs + 6 * 9;
    int *kstart_in = EVA_buff->ijk_se_displs + 6 * 10;
    int *kend_in   = EVA_buff->ijk_se_displs + 6 * 11;
    
    int *sendcounts = EVA_buff->ijk_se_displs + 6 * 12;
    int *sdispls    = EVA_buff->ijk_se_displs + 6 * 13;
    int *recvcounts = EVA_buff->ijk_se_displs + 6 * 14;
    int *rdispls    = EVA_buff->ijk_se_displs + 6 * 15;
    
    double *x_ex  = EVA_buff->x_ex;
    double *x_in  = EVA_buff->x_in;
    double *x_out = EVA_buff->x_out;
    
    double st, et;
    
    // Copy halo values to send buffer
    st = MPI_Wtime();
    
    // Necessary, DO NOT REMOVE
    #pragma omp parallel for 
    for (int i = 0; i < EVA_buff->halo_size; i++) x_in[i] = 0.0;
    
    int count = 0;
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) 
    {
        // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
        int nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)) * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
        EVA_Lap_MV_pack_halo(
            EVA_buff, x, ncol, kstart[nbrcount], jstart[nbrcount], istart[nbrcount],
            kend[nbrcount], jend[nbrcount], iend[nbrcount], &count
        );
    }
    et = MPI_Wtime();
    EVA_buff->Lap_MV_pack_t += et - st;
    
    // Overlap halo exchange with copy inner part of x 
    MPI_Request request;
    MPI_Ineighbor_alltoallv(
        x_out, sendcounts, sdispls, MPI_DOUBLE, 
        x_in,  recvcounts, rdispls, MPI_DOUBLE, 
        comm, &request
    ); 
    
    st = MPI_Wtime();
    EVA_Lap_MV_copy_inner_x(EVA_buff, ncol, x);
    et = MPI_Wtime();
    EVA_buff->Lap_MV_cpyx_t += et - st;
    
    st = MPI_Wtime();
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    et = MPI_Wtime();
    EVA_buff->Lap_MV_comm_t += et - st;
    
    // Copy received buffer to extended domain
    st = MPI_Wtime();
    count = 0;
    for (int nbrcount = 0; nbrcount < 6; nbrcount++) 
    {
        EVA_Lap_MV_unpack_halo(
            EVA_buff, ncol, kstart_in[nbrcount], jstart_in[nbrcount], istart_in[nbrcount],
            kend_in[nbrcount], jend_in[nbrcount], iend_in[nbrcount], &count
        );
    }
    et = MPI_Wtime();
    EVA_buff->Lap_MV_unpk_t += et - st;
}

// Set up halo values in the extended domain when there is 1 MPI process in the communicator
void EVA_Lap_MV_orth_NP1_Halo(EVA_buff_t EVA_buff, const int ncol, const double *x)
{
    int DMnx = EVA_buff->DMnx;
    int DMnd = EVA_buff->DMnd;
    int DMnxny = DMnx * EVA_buff->DMny;
    
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    int DMnxny_ex = DMnx_ex * EVA_buff->DMny_ex;
    
    int *periods = &EVA_buff->cart_array[3];
    
    int *istart = EVA_buff->ijk_se_displs + 6 * 0;
    int *iend   = EVA_buff->ijk_se_displs + 6 * 1;
    int *jstart = EVA_buff->ijk_se_displs + 6 * 2;
    int *jend   = EVA_buff->ijk_se_displs + 6 * 3;
    int *kstart = EVA_buff->ijk_se_displs + 6 * 4;
    int *kend   = EVA_buff->ijk_se_displs + 6 * 5;
    
    int *istart_in = EVA_buff->ijk_se_displs + 6 * 6;
    int *iend_in   = EVA_buff->ijk_se_displs + 6 * 7;
    int *jstart_in = EVA_buff->ijk_se_displs + 6 * 8;
    int *jend_in   = EVA_buff->ijk_se_displs + 6 * 9;
    int *kstart_in = EVA_buff->ijk_se_displs + 6 * 10;
    int *kend_in   = EVA_buff->ijk_se_displs + 6 * 11;
    
    double *x_ex  = EVA_buff->x_ex;
    
    double st, et;
    
    st = MPI_Wtime();
    EVA_Lap_MV_copy_inner_x(EVA_buff, ncol, x);
    et = MPI_Wtime();
    EVA_buff->Lap_MV_cpyx_t += et - st;
    
    // Copy the extended part from x into x_ex
    st = MPI_Wtime();
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) 
    {
        // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
        int nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)); // * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
        int nelem = iend[nbrcount] - istart[nbrcount];
        
        if (periods[nbr_i / 2])
        {
            EVA_Lap_MV_copy_periodic_halo(
                EVA_buff, x, ncol, kstart[nbrcount], jstart[nbrcount], istart[nbrcount],
                kend[nbrcount], jend[nbrcount], iend[nbrcount], 
                kstart_in[nbr_i], jstart_in[nbr_i], istart_in[nbr_i]
            );
        } else {
            int kp_spos = kstart_in[nbr_i];
            int kp_epos = kend_in[nbr_i];
            int jp_spos = jstart_in[nbr_i];
            int jp_epos = jend_in[nbr_i];
            #pragma omp parallel for collapse(3)
            for (int n = 0; n < ncol; n++) 
            {
                for (int kp = kp_spos; kp < kp_epos; kp++) 
                {
                    for (int jp = jp_spos; jp < jp_epos; jp++) 
                    {
                        double *x_ex_ptr = x_ex + n * DMnd_ex + kp * DMnxny_ex + jp * DMnx_ex + istart_in[nbr_i];
                        memset(x_ex_ptr, 0, sizeof(double) * nelem);
                    }
                }
            }
        }
    }
    et = MPI_Wtime();
    EVA_buff->Lap_MV_unpk_t += et - st;
}

// Kernel for calculating (a * Lap + b * diag(v) + c * I) * x
// For the input & output domain, z/x index is the slowest/fastest running index
// Input parameters:
//    x0               : Input domain with extended boundary 
//    radius           : Radius of the stencil (radius * 2 = stencil order)
//    stride_y         : Distance between x1(x, y, z) and x1(x, y+1, z)
//    stride_y_ex      : Distance between x0(x, y, z) and x0(x, y+1, z)
//    stride_z         : Distance between x1(x, y, z) and x1(x, y, z+1)
//    stride_z_ex      : Distance between x0(x, y, z) and x0(x, y, z+1)
//    [x_spos, x_epos) : X index range that will be computed in this kernel
//    [y_spos, y_epos) : Y index range that will be computed in this kernel
//    [z_spos, z_epos) : Z index range that will be computed in this kernel
//    stencil_x_coefs  : Stencil coefficients for the x-axis points, length radius+1
//    stencil_y_coefs  : Stencil coefficients for the y-axis points, length radius+1 
//    stencil_z_coefs  : Stencil coefficients for the z-axis points, length radius+1
//    coef_0           : Stencil coefficient for the center element
//    b                : Scaling factor of v
//    v                : Values of the diagonal matrix
// Output parameters:
//    x1               : Output domain with original boundary
void stencil_3axis_thread(
    const double *x0,   const int radius, 
    const int stride_y, const int stride_y_ex, 
    const int stride_z, const int stride_z_ex,
    const int x_spos,   const int x_epos, 
    const int y_spos,   const int y_epos,
    const int z_spos,   const int z_epos,
    const double *stencil_x_coefs,
    const double *stencil_y_coefs,
    const double *stencil_z_coefs,
    const double coef_0, const double b,
    const double *v, double *x1
)
{
    const double *_v = v;
    if (b == 0.0) _v = x1;
    
    for (int z = z_spos; z < z_epos; z++)
    {
        int iz = z + radius;
        for (int y = y_spos; y < y_epos; y++)
        {
            int iy = y + radius;
            int offset = z * stride_z + y * stride_y;
            int offset_ex = iz * stride_z_ex + iy * stride_y_ex + radius;
            
            #pragma omp simd
            for (int x = x_spos; x < x_epos; x++)
            {
                int idx = offset + x;
                int idx_ex = offset_ex + x;
                double res = coef_0 * x0[idx_ex];
                for (int r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = stencil_x_coefs[r] * (x0[idx_ex + r]          + x0[idx_ex - r]);
                    double res_y = stencil_y_coefs[r] * (x0[idx_ex + stride_y_r] + x0[idx_ex - stride_y_r]);
                    double res_z = stencil_z_coefs[r] * (x0[idx_ex + stride_z_r] + x0[idx_ex - stride_z_r]);
                    res += res_x + res_y + res_z;
                }
                x1[idx] = res + b * (_v[idx] * x0[idx_ex]);
            }
        }
    }
}

// Calculate (a * Lap + b * diag(v) + c * I) * x
void EVA_Lap_MV_orth_stencil_kernel(EVA_buff_t EVA_buff, const int ncol, const double *v, double *Hx)
{
    int radius = EVA_buff->FDn;
    
    int DMnx = EVA_buff->DMnx;
    int DMny = EVA_buff->DMny;
    int DMnz = EVA_buff->DMnz;
    int DMnd = EVA_buff->DMnd;
    
    int DMnx_ex = EVA_buff->DMnx_ex;
    int DMny_ex = EVA_buff->DMny_ex;
    int DMnz_ex = EVA_buff->DMnz_ex;
    int DMnd_ex = EVA_buff->DMnd_ex;
    
    int stride_y = DMnx;
    int stride_z = DMnx * DMny;
    int stride_y_ex = DMnx_ex;
    int stride_z_ex = DMnx_ex * DMny_ex;
    
    double b     = EVA_buff->b;
    double coef0 = EVA_buff->w2_diag;
    double *x_ex = EVA_buff->x_ex;
    double *stencil_x_coefs = EVA_buff->Lap_stencils + 0 * (radius + 1);
    double *stencil_y_coefs = EVA_buff->Lap_stencils + 1 * (radius + 1);
    double *stencil_z_coefs = EVA_buff->Lap_stencils + 2 * (radius + 1);
    
    int nblk_x, nblk_y, nblk_z;
    int x_blk_size, y_blk_size, z_blk_size;
    EVA_Lap_MV_stencil_omp_dd(
        EVA_buff->nthreads, DMnx, DMny, DMnz,
        &nblk_x, &nblk_y, &nblk_z, &x_blk_size, &y_blk_size, &z_blk_size
    );
    int nblk_xy  = nblk_x  * nblk_y;
    int nblk_xyz = nblk_xy * nblk_z;
    
    double st = MPI_Wtime();
    #pragma omp parallel for schedule(static)
    for (int i_domain = 0; i_domain < nblk_xyz * ncol; i_domain++)
    {
        int icol = i_domain / nblk_xyz;
        int iblk = i_domain % nblk_xyz;
        
        double *x0_icol = x_ex + icol * DMnd_ex;
        double *x1_icol = Hx   + icol * DMnd;
        
        int x_spos, y_spos, z_spos, x_epos, y_epos, z_epos;
        EVA_Lap_MV_unpack_dd_idx(
            iblk, nblk_x, nblk_xy, DMnx, DMny, DMnz,
            x_blk_size, y_blk_size, z_blk_size,
            &x_spos, &y_spos, &z_spos, &x_epos, &y_epos, &z_epos
        );

        stencil_3axis_thread(
            x0_icol, radius, stride_y, stride_y_ex, stride_z, stride_z_ex,
            x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, 
            stencil_x_coefs, stencil_y_coefs, stencil_z_coefs,
            coef0, b, v, x1_icol
        );
    }
    double et = MPI_Wtime();
    EVA_buff->Lap_MV_krnl_t += et - st;
}

// Calculate (a * Lap + b * diag(v) + c * I) * x in a matrix-free way
void EVA_Lap_MV_orth(
    EVA_buff_t EVA_buff, const int ncol, const double *x, const double *v, 
    MPI_Comm comm, const int nproc, double *Hx
)
{
    if (nproc > 1) EVA_Lap_MV_orth_NPK_Halo(EVA_buff, ncol, x, comm);
    else           EVA_Lap_MV_orth_NP1_Halo(EVA_buff, ncol, x);

    EVA_Lap_MV_orth_stencil_kernel(EVA_buff, ncol, v, Hx);
    
    EVA_buff->Lap_MV_rhs += ncol;
}
