/**
 * @file    gradVecRoutinesKpt.c
 * @brief   This file contains functions for performing gradient-vector 
 *          multiply routines with a Bloch factor.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 
#include <assert.h>

#include "gradVecRoutinesKpt.h"
#include "tools.h"
#include "isddft.h"



/**
 * @brief   Calculate (Gradient + c * I) times a bunch of vectors in the given direction in a matrix-free way.
 *          
 *          This function simply calls the Gradient_vec_mult multiple times. 
 *          For some reason it is more efficient than calling it ones and do  
 *          the multiplication together. TODO: think of a more efficient way!
 */
void Gradient_vectors_dir_kpt(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                              const int ncol, const double c, const double complex *x, 
                              double complex *Dx, const int dir, const double kpt_vec, MPI_Comm comm)
{
    int nproc;
    MPI_Comm_size(comm, &nproc);

    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else 
        dims[0] = dims[1] = dims[2] = 1;
   
    for (int i = 0; i < ncol; i++)
        Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, x+i*(unsigned)DMnd, Dx+i*(unsigned)DMnd, dir, kpt_vec, comm, dims);  
}



/**
 * @brief   Calculate (Gradient + c * I) times a bunch of vectors in the given direction in a matrix-free way.
 *
 * @param dir   Direction of derivatives to take: 0 -- x-dir, 1 -- y-dir, 2 -- z-dir
 */
void Gradient_vec_dir_kpt(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                      const int ncol, const double c, const double complex *x,
                      double complex *Dx, const int dir, const double kpt_vec, MPI_Comm comm, const int* dims)
{
    int nproc = dims[0] * dims[1] * dims[2];
    double cellsizes[3];
    cellsizes[0] = pSPARC->range_x;
    cellsizes[1] = pSPARC->range_y;
    cellsizes[2] = pSPARC->range_z;
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int periods[3];
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    
    int FDn = pSPARC->order / 2;
    
    int isDir[3], exDir[3];
    isDir[0] = (int)(dir == 0); isDir[1] = (int)(dir == 1); isDir[2] = (int)(dir == 2);
    exDir[0] = isDir[0] * FDn; exDir[1] = isDir[1] * FDn; exDir[2] = isDir[2] * FDn;
    
    // The user has to make sure DMnd = DMnx * DMny * DMnz
    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
    int DMnxny = DMnx * DMny;
    int DMnx_ex = DMnx + pSPARC->order * isDir[0];
    int DMny_ex = DMny + pSPARC->order * isDir[1];
    int DMnz_ex = DMnz + pSPARC->order * isDir[2];
    int DMnxny_ex = DMnx_ex * DMny_ex;
    int DMnd_ex = DMnxny_ex * DMnz_ex;
    int DMnx_in = DMnx - FDn;
    int DMny_in = DMny - FDn;
    int DMnz_in = DMnz - FDn;
    
    double w1_diag = c;
    
    // set up send buffer based on the ordering of the neighbors
    int istart[6] = {0,    DMnx_in, 0,    0,        0,    0}, 
          iend[6] = {FDn,  DMnx,     DMnx, DMnx,     DMnx, DMnx}, 
        jstart[6] = {0,    0,        0,    DMny_in, 0,    0},  
          jend[6] = {DMny, DMny,     FDn,  DMny,     DMny, DMny}, 
        kstart[6] = {0,    0,        0,    0,        0,    DMnz_in}, 
          kend[6] = {DMnz, DMnz,     DMnz, DMnz,     FDn,  DMnz};
    
    int count, n, k, j, i, nshift, kshift, jshift;
    int nbrcount, nbr_i;
    
    MPI_Request request;
    double complex *x_in, *x_out;   
    x_in = NULL;
    x_out = NULL;

    if (nproc > 1) {
        int nd_in = ncol * pSPARC->order * (isDir[0] * DMny * DMnz + DMnx * isDir[1] * DMnz + DMnxny * isDir[2]);
        int nd_out = nd_in;
        x_in  = (double complex *)calloc( nd_in, sizeof(double complex));
        x_out = (double complex *)malloc( nd_out * sizeof(double complex)); // no need to init x_out

        int sendcounts[6], sdispls[6], recvcounts[6], rdispls[6];
        // set up parameters for MPI_Ineighbor_alltoallv
        // TODO: do this in Initialization to save computation time!
        sendcounts[0] = sendcounts[1] = recvcounts[0] = recvcounts[1] = ncol * FDn * (DMny * DMnz * isDir[0]);
        sendcounts[2] = sendcounts[3] = recvcounts[2] = recvcounts[3] = ncol * FDn * (DMnx * DMnz * isDir[1]);
        sendcounts[4] = sendcounts[5] = recvcounts[4] = recvcounts[5] = ncol * FDn * (DMnxny * isDir[2]);
    
        rdispls[0] = sdispls[0] = 0;
        rdispls[1] = sdispls[1] = sdispls[0] + sendcounts[0];
        rdispls[2] = sdispls[2] = sdispls[1] + sendcounts[1];
        rdispls[3] = sdispls[3] = sdispls[2] + sendcounts[2];
        rdispls[4] = sdispls[4] = sdispls[3] + sendcounts[3];
        rdispls[5] = sdispls[5] = sdispls[4] + sendcounts[4];

        count = 0;
        for (nbr_i = dir*2; nbr_i < dir*2+2; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
            nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)) * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
            for (n = 0; n < ncol; n++) {
                nshift = n * DMnd;
                for (k = kstart[nbrcount]; k < kend[nbrcount]; k++) {
                    kshift = nshift + k * DMnxny;
                    for (j = jstart[nbrcount]; j < jend[nbrcount]; j++) {
                        jshift = kshift + j * DMnx;
                        for (i = istart[nbrcount]; i < iend[nbrcount]; i++) {
                            x_out[count++] = x[jshift+i];
                        }
                    }
                }
            }
        }    

        // first transfer info. to/from neighbor processors
        MPI_Ineighbor_alltoallv(x_out, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, 
                                x_in, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, 
                                comm, &request); // non-blocking
    }

    // while the non-blocking communication is undergoing, compute Dx which only requires values from local memory
    double complex *x_ex = (double complex *)malloc(ncol * DMnd_ex * sizeof(double complex));
    assert(x_ex != NULL);

    double *D1_stencil_coeffs_dirs[3];
    D1_stencil_coeffs_dirs[0] = pSPARC->D1_stencil_coeffs_x;
    D1_stencil_coeffs_dirs[1] = pSPARC->D1_stencil_coeffs_y;
    D1_stencil_coeffs_dirs[2] = pSPARC->D1_stencil_coeffs_z;

    int pshift_ex_dirs[3];
    pshift_ex_dirs[0] = 1;
    pshift_ex_dirs[1] = DMnx_ex;
    pshift_ex_dirs[2] = DMnxny_ex;
    int pshift_ex = pshift_ex_dirs[dir];

    // note: here kpt_vec should have been different for different directions! But the other dirs won't be used
    double complex phase_fac_l_x = cos(kpt_vec * cellsizes[0]) - sin(kpt_vec * cellsizes[0]) * I;
    double complex phase_fac_l_y = cos(kpt_vec * cellsizes[1]) - sin(kpt_vec * cellsizes[1]) * I;
    double complex phase_fac_l_z = cos(kpt_vec * cellsizes[2]) - sin(kpt_vec * cellsizes[2]) * I;
    double complex phase_fac_r_x = conj(phase_fac_l_x);
    double complex phase_fac_r_y = conj(phase_fac_l_y);
    double complex phase_fac_r_z = conj(phase_fac_l_z);
    double complex phase_factors[6]; // xl, xr, yl, yr, zl, zr
    phase_factors[0] = phase_fac_l_x;
    phase_factors[1] = phase_fac_r_x;
    phase_factors[2] = phase_fac_l_y;
    phase_factors[3] = phase_fac_r_y;
    phase_factors[4] = phase_fac_l_z;
    phase_factors[5] = phase_fac_r_z;

    int DMnz_exDir = DMnz+exDir[2];
    int DMny_exDir = DMny+exDir[1];
    int DMnx_exDir = DMnx+exDir[0];
    count = 0;
    for (n = 0; n < ncol; n++){
        nshift = n * DMnd_ex;
        for (k = exDir[2]; k < DMnz_exDir; k++){
            kshift = nshift + k * DMnxny_ex;
            for (j = exDir[1]; j < DMny_exDir; j++){
                jshift = kshift + j * DMnx_ex;
                for (i = exDir[0]; i < DMnx_exDir; i++){
                    x_ex[jshift+i] = x[count++]; // this saves index calculation time
                }
            }
        }
    }

    if (nproc > 1) {
        // set up start and end indices for copy receive buffer
        istart[0] = 0;             iend[0] = exDir[0];        
        jstart[0] = exDir[1];      jend[0] = DMny+exDir[1];   
        kstart[0] = exDir[2];      kend[0] = DMnz+exDir[2];
        istart[1] = DMnx+exDir[0]; iend[1] = DMnx+2*exDir[0]; 
        jstart[1] = exDir[1];      jend[1] = DMny+exDir[1];   
        kstart[1] = exDir[2];      kend[1] = DMnz+exDir[2]; 
        istart[2] = exDir[0];      iend[2] = DMnx+exDir[0];   
        jstart[2] = 0;             jend[2] = exDir[1];        
        kstart[2] = exDir[2];      kend[2] = DMnz+exDir[2];
        istart[3] = exDir[0];      iend[3] = DMnx+exDir[0];   
        jstart[3] = DMny+exDir[1]; jend[3] = DMny+2*exDir[1]; 
        kstart[3] = exDir[2];      kend[3] = DMnz+exDir[2];
        istart[4] = exDir[0];      iend[4] = DMnx+exDir[0];   
        jstart[4] = exDir[1];      jend[4] = DMny+exDir[1];   
        kstart[4] = 0;             kend[4] = exDir[2];
        istart[5] = exDir[0];      iend[5] = DMnx+exDir[0];   
        jstart[5] = exDir[1];      jend[5] = DMny+exDir[1];   
        kstart[5] = DMnz+exDir[2]; kend[5] = DMnz+2*exDir[2];

        // make sure receive buffer is ready
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // copy receive buffer into extended domain
        count = 0;
        for (nbrcount = dir*2; nbrcount < dir*2+2; nbrcount++) {
            const int k_s = kstart[nbrcount];
            const int k_e = kend[nbrcount];
            const int j_s = jstart[nbrcount];
            const int j_e = jend[nbrcount];
            const int i_s = istart[nbrcount];
            const int i_e = iend[nbrcount];
            int is_block_out = is_grid_outside(
                i_s, j_s, k_s, -exDir[0], -exDir[1], -exDir[2], DMVertices, gridsizes);
            double complex phase_factor = is_block_out ? phase_factors[nbrcount] : 1.0;
            for (n = 0; n < ncol; n++) {
                nshift = n * DMnd_ex;
                for (k = k_s; k < k_e; k++) {
                    kshift = nshift + k * DMnxny_ex;
                    for (j = j_s; j < j_e; j++) {
                        jshift = kshift + j * DMnx_ex;
                        for (i = i_s; i < i_e; i++) {
                            x_ex[jshift+i] = x_in[count++] * phase_factor;
                        }
                    }
                }
            }
        }
        free(x_out);
        free(x_in);
    } else {
        int istart_in[6], iend_in[6], jstart_in[6], jend_in[6], kstart_in[6], kend_in[6];
        istart_in[0] = 0;             iend_in[0] = exDir[0];        
        jstart_in[0] = exDir[1];      jend_in[0] = DMny+exDir[1];   
        kstart_in[0] = exDir[2];      kend_in[0] = DMnz+exDir[2];
        istart_in[1] = DMnx+exDir[0]; iend_in[1] = DMnx+2*exDir[0]; 
        jstart_in[1] = exDir[1];      jend_in[1] = DMny+exDir[1];   
        kstart_in[1] = exDir[2];      kend_in[1] = DMnz+exDir[2]; 
        istart_in[2] = exDir[0];      iend_in[2] = DMnx+exDir[0];   
        jstart_in[2] = 0;             jend_in[2] = exDir[1];        
        kstart_in[2] = exDir[2];      kend_in[2] = DMnz+exDir[2];
        istart_in[3] = exDir[0];      iend_in[3] = DMnx+exDir[0];   
        jstart_in[3] = DMny+exDir[1]; jend_in[3] = DMny+2*exDir[1]; 
        kstart_in[3] = exDir[2];      kend_in[3] = DMnz+exDir[2];
        istart_in[4] = exDir[0];      iend_in[4] = DMnx+exDir[0];   
        jstart_in[4] = exDir[1];      jend_in[4] = DMny+exDir[1];   
        kstart_in[4] = 0;             kend_in[4] = exDir[2];
        istart_in[5] = exDir[0];      iend_in[5] = DMnx+exDir[0];   
        jstart_in[5] = exDir[1];      jend_in[5] = DMny+exDir[1];   
        kstart_in[5] = DMnz+exDir[2]; kend_in[5] = DMnz+2*exDir[2];

        int nshift1, kshift1, jshift1, kp, jp, ip;
        // copy the extended part from x into x_ex
        for (nbr_i = dir * 2; nbr_i < dir * 2 + 2; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
            nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)); // * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
            const int kp_s = kstart_in[nbr_i];
            const int kp_e = kend_in  [nbr_i];
            const int jp_s = jstart_in[nbr_i];
            const int jp_e = jend_in  [nbr_i];
            const int ip_s = istart_in[nbr_i];
            const int ip_e = iend_in  [nbr_i];
            if (periods[nbr_i / 2]) {
                const int k_s = kstart[nbrcount];
                const int k_e = kend  [nbrcount];
                const int j_s = jstart[nbrcount];
                const int j_e = jend  [nbrcount];
                const int i_s = istart[nbrcount];
                const int i_e = iend  [nbrcount];
                // assuming the whole block is either all in or all out
                int is_block_out = is_grid_outside(
                    ip_s, jp_s, kp_s, -exDir[0], -exDir[1], -exDir[2], DMVertices, gridsizes);
                double complex phase_factor = is_block_out ? phase_factors[nbr_i] : 1.0;
                for (n = 0; n < ncol; n++) {
                    nshift = n * DMnd_ex; nshift1 = n * DMnd;
                    for (k = k_s, kp = kp_s; k < k_e; k++, kp++) {
                        kshift = nshift + kp * DMnxny_ex; kshift1 = nshift1 + k * DMnxny;
                        for (j = j_s, jp = jp_s; j < j_e; j++, jp++) {
                            jshift = kshift + jp * DMnx_ex; jshift1 = kshift1 + j * DMnx;
                            for (i = i_s, ip = ip_s; i < i_e; i++, ip++) {
                                x_ex[jshift+ip] = x[jshift1+i] * phase_factor;
                            }
                        }
                    }
                }
            } else {
                for (n = 0; n < ncol; n++){
                    nshift = n * DMnd_ex;
                    for (kp = kp_s; kp < kp_e; kp++){
                        kshift = nshift + kp * DMnxny_ex;
                        for (jp = jp_s; jp < jp_e; jp++){
                            jshift = kshift + jp * DMnx_ex;
                            for (ip = ip_s; ip < ip_e; ip++){
                                x_ex[jshift+ip] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // calculate dx
    for (n = 0; n < ncol; n++) {
        Calc_DX_kpt(x_ex+n*DMnd_ex, Dx+n*DMnd, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                0, DMnx, 0, DMny, 0, DMnz, exDir[0], exDir[1], exDir[2], D1_stencil_coeffs_dirs[dir], w1_diag);
    }

    free(x_ex);
}



/*  
 * @brief: function to calculate derivative
 */
void Calc_DX_kpt(
    const double complex *X, double complex *DX,
    const int radius,      const int stride_X,
    const int stride_y_X,  const int stride_y_DX,
    const int stride_z_X,  const int stride_z_DX,
    const int x_DX_spos,   const int x_DX_epos,
    const int y_DX_spos,   const int y_DX_epos,
    const int z_DX_spos,   const int z_DX_epos,
    const int x_X_spos,    const int y_X_spos,
    const int z_X_spos,    const double *stencil_coefs,
    const double c
)
{
    int i, j, k, jj, kk, r;

    for (k = z_DX_spos, kk = z_X_spos; k < z_DX_epos; k++, kk++)
    {
        int kshift_DX = k * stride_z_DX;
        int kshift_X = kk * stride_z_X;
        for (j = y_DX_spos, jj = y_X_spos; j < y_DX_epos; j++, jj++)
        {
            int jshift_DX = kshift_DX + j * stride_y_DX;
            int jshift_X = kshift_X + jj * stride_y_X;
            const int niters = x_DX_epos - x_DX_spos;
#ifdef ENABLE_SIMD_COMPLEX
#pragma omp simd
#endif
            for (i = 0; i < niters; i++)
            {
                int ishift_DX = jshift_DX + i + x_DX_spos;
                int ishift_X = jshift_X + i + x_X_spos;
                double complex res = X[ishift_X] * c;
                for (r = 1; r <= radius; r++)
                {
                    int stride_X_r = r * stride_X;
                    res += (X[ishift_X + stride_X_r] - X[ishift_X - stride_X_r]) * stencil_coefs[r];
                }
                DX[ishift_DX] = res;
            }
        }
    }
}
