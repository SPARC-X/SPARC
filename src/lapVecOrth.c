/**
 * @file    lapVecOrth.c
 * @brief   This file contains functions for performing the orthogonal
 *          laplacian-vector multiply routines.
 *
 * @author  Qimen Xu <qimenxu@gatech.edu>
 *          Phanisri Pradeep Pratapa <ppratapa@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2019 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h> 

#include "lapVecOrth.h"
#include "isddft.h"

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
#endif

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

/**
 * @brief   Calculate (a * Lap + c * I) times vectors.
 *
 *          This is only for orthogonal discretization.
 *
 */
void Lap_vec_mult_orth(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
        const int ncol, const double a, const double c, const double *x, 
        double *y, MPI_Comm comm, const int *dims
) 
{   
    unsigned i;
    // Call the function for (a*Lap+b*v+c)x with b = 0 and v = NULL
    for (i = 0; i < ncol; i++) {
        Lap_plus_diag_vec_mult_orth(
            pSPARC, DMnd, DMVertices, 1, a, 0.0, c, NULL, 
            x+i*(unsigned)DMnd, y+i*(unsigned)DMnd, comm, dims
        );
    }
}



/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0) + c * I) * x.
 *          For the input & output domain, z/x index is the slowest/fastest running index
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element
 * @param a                : Scaling factor of Lap
 * @param b                : Scaling factor of v0
 * @param c                : Shift constant
 * @param v0               : Values of the diagonal matrix
 * @param beta             : Scaling factor of y
 * @param y (OUT)          : Output domain with original boundary
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *
 * @modified by Qimen Xu <qimenxu@gatech.edu>, Mar 2019, Georgia Tech
 *
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
void stencil_3axis_thread_variable_radius(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]); 
            }
        }
    }
}


void stencil_3axis_thread_radius6(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= 6; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]); 
            }
        }
    }
}


void stencil_3axis_thread_radius4(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= 4; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]); 
            }
        }
    }
}


void stencil_3axis_thread_radius8(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= 8; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]); 
            }
        }
    }
}


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0) + c * I) * x.
 *          For the input & output domain, z/x index is the slowest/fastest running index
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element
 * @param a                : Scaling factor of Lap
 * @param b                : Scaling factor of v0
 * @param c                : Shift constant
 * @param v0               : Values of the diagonal matrix
 * @param beta             : Scaling factor of y
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_v2(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    switch (radius)
    {
        case 4:
            stencil_3axis_thread_radius4(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, y
            );
            return;
            break;

        case 6:
            stencil_3axis_thread_radius6(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, y
            );
            return;
            break;

        case 8:
            stencil_3axis_thread_radius8(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, y
            );
            return;
            break;

        default:
            stencil_3axis_thread_variable_radius(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, y
            );
            return;
            break;
    }
}



/**
 * @brief   Calculate (a * Lap + b * diag(v) + c * I) times vectors.
 *
 *          This is only for orthogonal discretization.
 *
 *          Warning:
 *            Although the function is intended for multiple vectors,
 *          it turns out to be slower than applying it to one vector
 *          at a time. So we will be calling this function ncol times
 *          if ncol is greater than 1.
 */
void Lap_plus_diag_vec_mult_orth(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
        const int ncol, const double a, const double b, const double c, 
        const double *v, const double *x, double *y, MPI_Comm comm,
        const int *dims
) 
{
#define INDEX(n,i,j,k) ((n)*DMnd+(k)*DMnxny+(j)*DMnx+(i))
#define INDEX_EX(n,i,j,k) ((n)*DMnd_ex+(k)*DMnxny_ex+(j)*DMnx_ex+(i))
#define X(n,i,j,k) x[(n)*DMnd+(k)*DMnxny+(j)*DMnx+(i)]
#define x_ex(n,i,j,k) x_ex[(n)*DMnd_ex+(k)*DMnxny_ex+(j)*DMnx_ex+(i)]

    #ifdef USE_EVA_MODULE
    double pack_t = 0.0, cpyx_t = 0.0, krnl_t = 0.0, unpk_t = 0.0, comm_t = 0.0;
    double st, et;
    #endif

    const double *_v = v; double _b = b; 
    if (fabs(b) < 1e-14 || v == NULL) _v = x, _b = 0.0;
    
    int nproc = dims[0] * dims[1] * dims[2];
    int periods[3];
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    
    int FDn = pSPARC->order / 2;
    
    // The user has to make sure DMnd = DMnx * DMny * DMnz
    int DMnx = 1 - DMVertices[0] + DMVertices[1];
    int DMny = 1 - DMVertices[2] + DMVertices[3];
    int DMnz = 1 - DMVertices[4] + DMVertices[5];
    int DMnxny = DMnx * DMny;
    
    int DMnx_ex = DMnx + pSPARC->order;
    int DMny_ex = DMny + pSPARC->order;
    int DMnz_ex = DMnz + pSPARC->order;
    int DMnxny_ex = DMnx_ex * DMny_ex;
    int DMnd_ex = DMnxny_ex * DMnz_ex;
    
    int DMnx_in  = DMnx - FDn;
    int DMny_in  = DMny - FDn;
    int DMnz_in  = DMnz - FDn;
    int DMnx_out = DMnx + FDn;
    int DMny_out = DMny + FDn;
    int DMnz_out = DMnz + FDn;
    
    // int Nx = pSPARC->Nx;
    // int Ny = pSPARC->Ny;
    
    #ifdef USE_EVA_MODULE
    st = MPI_Wtime();
    #endif
    
    // integrate a into coefficients weights
    double *Lap_weights = (double *)malloc(3*(FDn+1)*sizeof(double)); 
    double *Lap_stencil = Lap_weights;
    int p;
    for (p = 0; p < FDn + 1; p++)
    {
        (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
        (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
        (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
    }
    
    double w2_diag;
    w2_diag  = Lap_weights[0];
    w2_diag += Lap_weights[1];
    w2_diag += Lap_weights[2];
    w2_diag += c; // shift the diagonal by c

    // set up send buffer based on the ordering of the neighbors
    int istart[6] = {0,    DMnx_in,  0,    0,        0,    0}, 
          iend[6] = {FDn,  DMnx,     DMnx, DMnx,     DMnx, DMnx}, 
        jstart[6] = {0,    0,        0,    DMny_in,  0,    0},  
          jend[6] = {DMny, DMny,     FDn,  DMny,     DMny, DMny}, 
        kstart[6] = {0,    0,        0,    0,        0,    DMnz_in}, 
          kend[6] = {DMnz, DMnz,     DMnz, DMnz,     FDn,  DMnz};
    
    int nbrcount;
    MPI_Request request;
    double *x_in, *x_out;
    x_in = NULL; x_out = NULL;
    if (nproc > 1) { // pack info and init Halo exchange
        // TODO: we have to take BC into account here!
        int nd_in = ncol * pSPARC->order * (DMnx*DMny + DMny*DMnz + DMnx*DMnz);
        int nd_out = nd_in;
        
        // Notice here we init x_in to 0
        x_in  = (double *)calloc( nd_in, sizeof(double)); 
        x_out = (double *)malloc( nd_out * sizeof(double)); // no need to init x_out
        assert(x_in != NULL && x_out != NULL);

        int nbr_i, n, k, j, i, count = 0;
        for (nbr_i = 0; nbr_i < 6; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
            nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)) * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
            const int k_s = kstart[nbrcount];
            const int k_e = kend  [nbrcount];
            const int j_s = jstart[nbrcount];
            const int j_e = jend  [nbrcount];
            const int i_s = istart[nbrcount];
            const int i_e = iend  [nbrcount];
            for (n = 0; n < ncol; n++) {
                for (k = k_s; k < k_e; k++) {
                    for (j = j_s; j < j_e; j++) {
                        for (i = i_s; i < i_e; i++) {
                            x_out[count++] = X(n,i,j,k);
                        }
                    }
                }
            }
        }
        
        #ifdef USE_EVA_MODULE
        et = MPI_Wtime();
        pack_t = et - st;
        #endif
        
        int sendcounts[6], sdispls[6], recvcounts[6], rdispls[6];
        // set up parameters for MPI_Ineighbor_alltoallv
        // TODO: do this in Initialization to save computation time!
        sendcounts[0] = sendcounts[1] = recvcounts[0] = recvcounts[1] = ncol * FDn * (DMny * DMnz);
        sendcounts[2] = sendcounts[3] = recvcounts[2] = recvcounts[3] = ncol * FDn * (DMnx * DMnz);
        sendcounts[4] = sendcounts[5] = recvcounts[4] = recvcounts[5] = ncol * FDn * (DMnx * DMny);
        
        rdispls[0] = sdispls[0] = 0;
        rdispls[1] = sdispls[1] = sdispls[0] + sendcounts[0];
        rdispls[2] = sdispls[2] = sdispls[1] + sendcounts[1];
        rdispls[3] = sdispls[3] = sdispls[2] + sendcounts[2];
        rdispls[4] = sdispls[4] = sdispls[3] + sendcounts[3];
        rdispls[5] = sdispls[5] = sdispls[4] + sendcounts[4];
        
        // first transfer info. to/from neighbor processors
        //MPI_Request request;
        MPI_Ineighbor_alltoallv(x_out, sendcounts, sdispls, MPI_DOUBLE, 
                                 x_in, recvcounts, rdispls, MPI_DOUBLE, 
                                 comm, &request); // non-blocking
    }
    
    // overlap some work with communication
    #ifdef USE_EVA_MODULE
    st = MPI_Wtime();
    #endif
    
    int *pshifty    = (int *)malloc( (FDn+1) * sizeof(int));
    int *pshiftz    = (int *)malloc( (FDn+1) * sizeof(int));
    int *pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    int *pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    double *x_ex = (double *)malloc(ncol * DMnd_ex * sizeof(double));
    pshifty[0] = pshiftz[0] = pshifty_ex[0] = pshiftz_ex[0] = 0;
    for (p = 1; p <= FDn; p++) {
        // for x
        pshifty[p] = p * DMnx;
        pshiftz[p] = pshifty[p] * DMny;
        // for x_ex
        pshifty_ex[p] = p * DMnx_ex;
        pshiftz_ex[p] = pshifty_ex[p] * DMny_ex;
    }
    
    // copy x into extended x_ex
    int n, kp, jp, ip;
    int count = 0;
    for (n = 0; n < ncol; n++) {
        for (kp = FDn; kp < DMnz_out; kp++) {
            for (jp = FDn; jp < DMny_out; jp++) {
                for (ip = FDn; ip < DMnx_out; ip++) {
                    x_ex(n,ip,jp,kp) = x[count++]; 
                }
            }
        }
    } 
    
    #ifdef USE_EVA_MODULE
    et = MPI_Wtime();
    cpyx_t = et - st;
    
    st = MPI_Wtime();
    #endif

    int overlap_flag = 0; 
    overlap_flag = (int) (nproc > 1 && DMnx > pSPARC->order 
                          && DMny > pSPARC->order && DMnz > pSPARC->order);
    
    // TODO: REMOVE AFTER CHECK
    //overlap_flag = 0;
                          
    int i, j, k;
    // while the non-blocking communication is undergoing, compute 
    // inner part of y which only requires values from local memory
    if (overlap_flag) {
        // find Lx(FDn:DMnx_in, FDn:DMny_in, FDn:DMnz_in)
       for (n = 0; n < ncol; n++) {
           stencil_3axis_thread_v2(
               x+n*DMnd, FDn, pshifty[1], pshifty[1], pshiftz[1], pshiftz[1], 
               FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, 
               Lap_weights, w2_diag, _b, _v, y+n*DMnd
           );
       }
    }
    
    // set up start and end indices for copying edge nodes in x_ex
    int istart_in[6] = {0,       DMnx_out, FDn,     FDn,      FDn,      FDn}; 
    int   iend_in[6] = {FDn,     DMnx_ex,  DMnx_out,DMnx_out, DMnx_out, DMnx_out};
    int jstart_in[6] = {FDn,     FDn,      0,       DMny_out, FDn,      FDn};
    int   jend_in[6] = {DMny_out,DMny_out, FDn,     DMny_ex,  DMny_out, DMny_out};
    int kstart_in[6] = {FDn,     FDn,      FDn,     FDn,      0,        DMnz_out}; 
    int   kend_in[6] = {DMnz_out,DMnz_out, DMnz_out,DMnz_out, FDn,      DMnz_ex};
    
    if (nproc > 1) { // unpack info and copy into x_ex
        // make sure receive buffer is ready
        #ifdef USE_EVA_MODULE
        st = MPI_Wtime();
        #endif
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        #ifdef USE_EVA_MODULE
        et = MPI_Wtime();
        comm_t = et - st;
        #endif

        // copy receive buffer into extended domain
        #ifdef USE_EVA_MODULE
        st = MPI_Wtime();
        #endif
        count = 0;
        for (nbrcount = 0; nbrcount < 6; nbrcount++) {
            const int k_s = kstart_in[nbrcount];
            const int k_e = kend_in  [nbrcount];
            const int j_s = jstart_in[nbrcount];
            const int j_e = jend_in  [nbrcount];
            const int i_s = istart_in[nbrcount];
            const int i_e = iend_in  [nbrcount];
            for (n = 0; n < ncol; n++) {
                for (k = k_s; k < k_e; k++) {
                    for (j = j_s; j < j_e; j++) {
                        for (i = i_s; i < i_e; i++) {
                            x_ex(n,i,j,k) = x_in[count++];
                        }
                    }
                }
            }
        }
        
        free(x_out);
        free(x_in);
        
        #ifdef USE_EVA_MODULE
        et = MPI_Wtime();
        unpk_t = et - st;
        #endif    
    
    } else { // copy the extended part directly from x into x_ex
        int nbr_i;
        for (nbr_i = 0; nbr_i < 6; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send 
            // buffer for left and right neighbors
            nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)); 
            // * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
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
                for (n = 0; n < ncol; n++) {
                    for (k = k_s, kp = kp_s; k < k_e; k++, kp++) {
                        for (j = j_s, jp = jp_s; j < j_e; j++, jp++) {
                            for (i = i_s, ip = ip_s; i < i_e; i++, ip++) {
                                x_ex(n,ip,jp,kp) = X(n,i,j,k);
                            }
                        }
                    }
                }
            } else {
                for (n = 0; n < ncol; n++) {
                    for (kp = kp_s; kp < kp_e; kp++) {
                        for (jp = jp_s; jp < jp_e; jp++) {
                            for (ip = ip_s; ip < ip_e; ip++) {
                                x_ex(n,ip,jp,kp) = 0.0;
                            }
                        }
                    }
                }
            }
            //bc = periods[nbr_i / 2];
            //for (n = 0; n < ncol; n++) {
            //    for (k = kstart[nbrcount], kp = kstart_in[nbr_i]; k < kend[nbrcount]; k++, kp++) {
            //        for (j = jstart[nbrcount], jp = jstart_in[nbr_i]; j < jend[nbrcount]; j++, jp++) {
            //            for (i = istart[nbrcount], ip = istart_in[nbr_i]; i < iend[nbrcount]; i++, ip++) {
            //                x_ex(n,ip,jp,kp) = X(n,i,j,k) * bc;
            //            }
            //        }
            //    }
            //}
        }
    }
    
    #ifdef USE_EVA_MODULE
    st = MPI_Wtime();
    #endif
    
    //int ind_ex;
    // calculate Lx
    if (overlap_flag) {
        for (n = 0; n < ncol; n++) {
            // Lx(0:DMnx, 0:DMny, 0:FDn)
            stencil_3axis_thread_v2(
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
            // Lx(0:DMnx, 0:DMny, DMnz-FDn:DMnz)
            stencil_3axis_thread_v2(
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                0, DMnx, 0, DMny, DMnz-FDn, DMnz, FDn, FDn, DMnz, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
        }
        
        for (n = 0; n < ncol; n++) { 
            // Lx(0:DMnx, 0:FDn, FDn:DMnz-FDn)
            stencil_3axis_thread_v2( 
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                0, DMnx, 0, FDn, FDn, DMnz-FDn, FDn, FDn, FDn+FDn, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
            // Lx(0:DMnx, DMny-FDn:DMny, FDn:DMnz-FDn)
            stencil_3axis_thread_v2(
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                0, DMnx, DMny-FDn, DMny, FDn, DMnz-FDn, FDn, DMny, FDn+FDn, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
        }

        for (n = 0; n < ncol; n++) {
            // Lx(0:FDn, FDn:DMny-FDn, FDn:DMnz-FDn)
            stencil_3axis_thread_v2(
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                0, FDn, FDn, DMny-FDn, FDn, DMnz-FDn, FDn, FDn+FDn, FDn+FDn, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
            // Lx(DMnx-FDn:DMnx, FDn:DMny-FDn, FDn:DMnz-FDn)
            stencil_3axis_thread_v2(
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                DMnx-FDn, DMnx, FDn, DMny-FDn, FDn, DMnz-FDn, DMnx, FDn+FDn, FDn+FDn, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
        }
    } else {
        for (n = 0; n < ncol; n++) {
            stencil_3axis_thread_v2(
                x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
                0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 
                Lap_weights, w2_diag, _b, _v, y+n*DMnd
            );
        }
    } 

    free(x_ex);
    free(pshifty);
    free(pshiftz);
    free(pshifty_ex);
    free(pshiftz_ex);
    
    free(Lap_weights);
    
    #ifdef USE_EVA_MODULE
    et = MPI_Wtime();
    krnl_t += et - st;
    
    EVA_buff_timer_add(cpyx_t, pack_t, comm_t, unpk_t, krnl_t, 0.0);
    EVA_buff_rhs_add(ncol, 0);
    #endif

#undef INDEX
#undef INDEX_EX
#undef X
#undef x_ex
}


