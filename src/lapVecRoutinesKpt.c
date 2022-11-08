/**
 * @file    lapVecRoutinesKpt.c
 * @brief   This file contains functions for performing the laplacian-vector
 *          multiply routines with a Bloch factor.
 *
 * @authors Abhiraj Sharma <asharma424@gatech.edu>
 *          Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "lapVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "tools.h"
#include "isddft.h"

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
#endif

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))



/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 */
void Lap_vec_mult_kpt(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, double complex *x, double complex *Lapx, int kpt, MPI_Comm comm
) 
{
    int dims[3], periods[3], my_coords[3];
    MPI_Cart_get(comm, 3, dims, periods, my_coords);
    
    if(pSPARC->cell_typ == 0) {
        Lap_vec_mult_orth_kpt(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, dims, kpt);
    } else {
        MPI_Comm comm2;
        comm2 = pSPARC->kptcomm_topo_dist_graph;
        // TODO: make the second communicator general rather than only for phi
        //Lap_vec_mult_nonorth(pSPARC, DMnd, DMVertices, ncol, c, x, Lapx, comm, comm2);
        Lap_vec_mult_nonorth_kpt(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, comm2, dims, kpt);       
    }
}



/**
 * @brief   Calculate (a * Lap + c * I) times vectors.
 *
 *          This is only for orthogonal discretization.
 *
 */
void Lap_vec_mult_orth_kpt(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
        const int ncol, const double a, const double c, const double complex *x, 
        double complex *y, MPI_Comm comm, const int *dims, const int kpt
) 
{   
    unsigned i;
    // Call the function for (a*Lap+b*v+c)x with b = 0 and v = NULL
    for (i = 0; i < ncol; i++) {
        Lap_plus_diag_vec_mult_orth_kpt(
            pSPARC, DMnd, DMVertices, 1, a, 0.0, c, NULL, 
            x+i*(unsigned)DMnd, y+i*(unsigned)DMnd, comm, dims, kpt
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
 * @param kpt              : Bloch wavevector
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *
 * @modified by Abhiraj Sharma <asharma424@gatech.edu>, April 2019, Georgia Tech
 * @modified by Qimen Xu <qimenxu@gatech.edu>, Jul 2022, Georgia Tech
 *
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
void stencil_3axis_thread_complex_v2(
    const double complex *x0, const int radius,
    const int stride_y,  const int stride_y_ex,
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos,
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,
    const double *v0,    double complex *y
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
            #ifdef ENABLE_SIMD_COMPLEX
            #pragma omp simd
            #endif
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double complex res = coef_0 * x0[idx_ex];
                for (r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double complex res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double complex res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double complex res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]);
            }
        }
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
void Lap_plus_diag_vec_mult_orth_kpt(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
        const int ncol, const double a, const double b, const double c, 
        const double *v, const double complex *x, double complex *y, MPI_Comm comm,
        const int *dims, int kpt
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
    if (fabs(b) < 1e-14 || v == NULL) _v = (double * )x, _b = 0.0;
    
    int nproc = dims[0] * dims[1] * dims[2];
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
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
    double complex *x_in, *x_out;
    x_in = NULL; x_out = NULL;
    if (nproc > 1) { // pack info and init Halo exchange
        // TODO: we have to take BC into account here!
        int nd_in = ncol * pSPARC->order * (DMnx*DMny + DMny*DMnz + DMnx*DMnz);
        int nd_out = nd_in;
        
        // Notice here we init x_in to 0
        x_in  = (double complex *)calloc( nd_in, sizeof(double complex)); 
        x_out = (double complex *)malloc( nd_out * sizeof(double complex)); // no need to init x_out
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
        MPI_Ineighbor_alltoallv(x_out, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, 
                                x_in, recvcounts, rdispls, MPI_DOUBLE_COMPLEX,
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
    double complex *x_ex = (double complex *)malloc(ncol * DMnd_ex * sizeof(double complex));
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
                          
    int i, j, k;
    
    // set up start and end indices for copying edge nodes in x_ex
    int istart_in[6] = {0,       DMnx_out, FDn,     FDn,      FDn,      FDn}; 
    int   iend_in[6] = {FDn,     DMnx_ex,  DMnx_out,DMnx_out, DMnx_out, DMnx_out};
    int jstart_in[6] = {FDn,     FDn,      0,       DMny_out, FDn,      FDn};
    int   jend_in[6] = {DMny_out,DMny_out, FDn,     DMny_ex,  DMny_out, DMny_out};
    int kstart_in[6] = {FDn,     FDn,      FDn,     FDn,      0,        DMnz_out}; 
    int   kend_in[6] = {DMnz_out,DMnz_out, DMnz_out,DMnz_out, FDn,      DMnz_ex};

    double complex phase_fac_l_x = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) - sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;
    double complex phase_fac_l_y = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) - sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
    double complex phase_fac_l_z = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) - sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
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
            // apply phase factor if the domain goes outside the global domain
            const double complex phase_factor = phase_factors[nbrcount];
            for (n = 0; n < ncol; n++) {
                for (k = k_s; k < k_e; k++) {
                    for (j = j_s; j < j_e; j++) {
                        for (i = i_s; i < i_e; i++) {
                            // can check just once, if we assume the entire block is either out or in
                            if (is_grid_outside(i, j, k, -FDn, -FDn, -FDn, DMVertices, gridsizes))
                                x_ex(n,i,j,k) = x_in[count++] * phase_factor;
                            else
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
                // apply phase factor if the domain goes outside the global domain
                const double complex phase_factor = phase_factors[nbr_i];
                for (n = 0; n < ncol; n++) {
                    for (k = k_s, kp = kp_s; k < k_e; k++, kp++) {
                        for (j = j_s, jp = jp_s; j < j_e; j++, jp++) {
                            for (i = i_s, ip = ip_s; i < i_e; i++, ip++) {
                                // can check just once, if we assume the entire block is either out or in
                                if (is_grid_outside(ip, jp, kp, -FDn, -FDn, -FDn, DMVertices, gridsizes))
                                    x_ex(n,ip,jp,kp) = X(n,i,j,k) * phase_factor;
                                else
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
    for (n = 0; n < ncol; n++) {
        stencil_3axis_thread_complex_v2(
            x_ex+n*DMnd_ex, FDn, pshifty[1], pshifty_ex[1], pshiftz[1], pshiftz_ex[1], 
            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 
            Lap_weights, w2_diag, _b, _v, y+n*DMnd
        );
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



/**
 * @brief   Calculate (a * Lap + c * I) times vectors.
 */
void Lap_vec_mult_nonorth_kpt(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
        const int ncol, const double a, const double c, const double complex *x,
        double complex *y, MPI_Comm comm,  MPI_Comm comm2, const int *dims, const int kpt
)
{
    unsigned i;
    // Call the function for (a*Lap+b*v+c)x with b = 0 and v = NULL
    for (i = 0; i < ncol; i++) {
        Lap_plus_diag_vec_mult_nonorth_kpt(
            pSPARC, DMnd, DMVertices, 1, a, 0.0, c, NULL,
            x+i*(unsigned)DMnd, y+i*(unsigned)DMnd, comm, comm2, dims, kpt
        );
    }
}



/**
 * @brief Finds the phase factor in the Bloch's theorem for a given kpoint vector
 * and lattice vector. Bloch's theorem states:
 *        \psi(x + L) = \psi * e^{i k L},
 * where e^{i k L} = cos(k L) + i * sin(k L) is the phase factor calculated here,
 * k is the kpoint vector, L is the translation vector.
 * 
 * @param kpt_vec K-point vector.
 * @param trans_vec Translation vector.
 * @return double 
 */
double complex calculate_phase_factor(double kpt_vec[3], double trans_vec[3])
{
    double theta = kpt_vec[0] * trans_vec[0]
                 + kpt_vec[1] * trans_vec[1]
                 + kpt_vec[2] * trans_vec[2];
    return cos(theta) + sin(theta) * I;
}



/**
 * @brief   Calculate (a * Lap + b * diag(v) + c * I) times vectors.
 *          Warning:
 *            Although the function is intended for multiple vectors,
 *          it turns out to be slower than applying it to one vector
 *          at a time. So we will be calling this function ncol times
 *          if ncol is greater than 1.
 */
void Lap_plus_diag_vec_mult_nonorth_kpt(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
        const int ncol, const double a, const double b, const double c,
        const double *v, const double complex *x, double complex *y, MPI_Comm comm,  MPI_Comm comm2,
        const int *dims, const int kpt
)
{
    #ifdef USE_EVA_MODULE
    double pack_t = 0.0, cpyx_t = 0.0, krnl_t = 0.0, unpk_t = 0.0, comm_t = 0.0;
    double st, et;
    #endif

    const double *_v = v; double _b = b;
    if (fabs(b) < 1e-14 || v == NULL) _v = (double * )x, _b = 0.0;

    int nproc = dims[0] * dims[1] * dims[2];
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int periods[3];
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    double kpt_vec[3];
    kpt_vec[0] = pSPARC->k1_loc[kpt];
    kpt_vec[1] = pSPARC->k2_loc[kpt];
    kpt_vec[2] = pSPARC->k3_loc[kpt];
    double trans_vec[3] = {0.,0.,0.};

    int order = pSPARC->order;
    int FDn = order / 2;

    // The user has to make sure DMnd = DMnx * DMny * DMnz
    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
    int DMnxny = DMnx * DMny;

    int DMnx_ex = DMnx + order;
    int DMny_ex = DMny + order;
    int DMnz_ex = DMnz + order;
    int DMnxny_ex = DMnx_ex * DMny_ex;
    int DMnd_ex = DMnxny_ex * DMnz_ex;

    int DMnx_out = DMnx + FDn;
    int DMny_out = DMny + FDn;
    int DMnz_out = DMnz + FDn;

    #ifdef USE_EVA_MODULE
    st = MPI_Wtime();
    #endif

    // Laplacian coefficients
    double *Lap_wt, w2_diag;
    w2_diag  = (pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0] + pSPARC->D2_stencil_coeffs_z[0]) * a;
    w2_diag += c; // shift the diagonal by c
    Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
    double *Lap_stencil = Lap_wt+5;
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, a);

    int nbrcount, n, i, j, k, nshift, kshift, jshift, ind, count;
    int nshift1, kshift1, jshift1;
    int kp, jp, ip;

    MPI_Request request;
    double complex *x_in, *x_out;
    x_in = NULL; x_out = NULL;
    // set up send-receive buffer based on the ordering of the neighbors
    int istart[26], iend[26], jstart[26], jend[26], kstart[26], kend[26];
    int istart_in[26], iend_in[26], jstart_in[26], jend_in[26], kstart_in[26], kend_in[26], isnonzero[26];

    snd_rcv_buffer(nproc, dims, periods, FDn, DMnx, DMny, DMnz, istart, iend, jstart, jend, kstart, kend, istart_in, iend_in, jstart_in, jend_in, kstart_in, kend_in, isnonzero);

    double complex phase_factors[26];
    count = 0; // count for neighbor index
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                if (i == 0 && j == 0 && k == 0) continue; // skip center block
                trans_vec[0] = i * Lx;
                trans_vec[1] = j * Ly;
                trans_vec[2] = k * Lz;
                phase_factors[count] = calculate_phase_factor(kpt_vec, trans_vec);
                count++;
            }
        }
    }

    if (nproc > 1) { // pack info and init Halo exchange

        int sendcounts[26], sdispls[26], recvcounts[26], rdispls[26];
        // set up parameters for MPI_Ineighbor_alltoallv
        // TODO: do this in Initialization to save computation time!
        sendcounts[0] = sendcounts[2] = sendcounts[6] = sendcounts[8] = sendcounts[17] = sendcounts[19] = sendcounts[23] = sendcounts[25] = ncol * FDn * FDn * FDn;
        sendcounts[1] = sendcounts[7] = sendcounts[18] = sendcounts[24] = ncol * DMnx * FDn * FDn;
        sendcounts[3] = sendcounts[5] = sendcounts[20] = sendcounts[22] = ncol * FDn * DMny * FDn;
        sendcounts[4] = sendcounts[21] = ncol * DMnxny * FDn;
        sendcounts[9] = sendcounts[11] = sendcounts[14] = sendcounts[16] = ncol * FDn * FDn * DMnz;
        sendcounts[10] = sendcounts[15] = ncol * DMnx * FDn * DMnz;
        sendcounts[12] = sendcounts[13] = ncol * FDn * DMny * DMnz;

        for(i = 0; i < 26; i++){
            recvcounts[i] = sendcounts[i];
        }

        rdispls[0] = sdispls[0] = 0;
        for(i = 1; i < 26;i++){
            rdispls[i] = sdispls[i] = sdispls[i-1] + sendcounts[i-1];
        }

        int nd_in = ncol * 2 * FDn* (4 * FDn * FDn + 2 * FDn * (DMnx + DMny + DMnz) + (DMnxny + DMnx * DMnz + DMny * DMnz) );
        int nd_out = nd_in;
        x_in  = (double complex *)calloc( nd_in, sizeof(double complex)); // TODO: init to 0
        x_out = (double complex *)malloc( nd_out * sizeof(double complex)); // no need to init x_out
        assert(x_in != NULL && x_out != NULL);

        int nbr_i;
        count = 0;
        for (nbr_i = 0; nbr_i < 26; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
            nbrcount = nbr_i;
            // TODO: Start loop over n here
            for (n = 0; n < ncol; n++) {
                nshift = n * DMnd;
                for (k = kstart[nbrcount]; k < kend[nbrcount]; k++) {
                    kshift = nshift + k * DMnxny;
                    for (j = jstart[nbrcount]; j < jend[nbrcount]; j++) {
                        jshift = kshift + j * DMnx;
                        for (i = istart[nbrcount]; i < iend[nbrcount]; i++) {
                            ind = jshift + i;
                            x_out[count++] = x[ind];
                        }
                    }
                }
            }
        }

        #ifdef USE_EVA_MODULE
        et = MPI_Wtime();
        pack_t = et - st;
        #endif

        // first transfer info. to/from neighbor processors
        //MPI_Request request;
        MPI_Ineighbor_alltoallv(x_out, sendcounts, sdispls, MPI_DOUBLE_COMPLEX,
                                x_in, recvcounts, rdispls, MPI_DOUBLE_COMPLEX,
                                comm2, &request); // non-blocking
    }

    // overlap some work with communication
    #ifdef USE_EVA_MODULE
    st = MPI_Wtime();
    #endif

    int pshifty = DMnx;
    int pshiftz = pshifty * DMny;
    int pshifty_ex = DMnx_ex;
    int pshiftz_ex = pshifty_ex * DMny_ex;

    double complex *x_ex = (double complex *)calloc(ncol * DMnd_ex, sizeof(double complex));
    assert(x_ex != NULL);
    // copy x into extended x_ex
    count = 0;
    for (n = 0; n < ncol; n++) {
        nshift = n * DMnd_ex;
        for (kp = FDn; kp < DMnz_out; kp++) {
            kshift = nshift + kp * DMnxny_ex;
            for (jp = FDn; jp < DMny_out; jp++) {
                jshift = kshift + jp * DMnx_ex;
                for (ip = FDn; ip < DMnx_out; ip++) {
                    ind = jshift + ip;
                    x_ex[ind] = x[count++];
                }
            }
        }
    }

    #ifdef USE_EVA_MODULE
    et = MPI_Wtime();
    cpyx_t = et - st;

    st = MPI_Wtime();
    #endif

    int DMnxexny = DMnx_ex * DMny;
    int DMnd_xex = DMnxexny * DMnz;
    int DMnxnyex = DMnx * DMny_ex;
    int DMnd_yex = DMnxnyex * DMnz;
    int DMnd_zex = DMnxny * DMnz_ex;
    double complex *Dx1, *Dx2;
    Dx1 = NULL; Dx2 = NULL;
    if(pSPARC->cell_typ == 11){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // df/dy
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 12){
        Dx1 = (double complex*) malloc(ncol * DMnd_xex * sizeof(double complex) ); // df/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 13){
        Dx1 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // df/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 14){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // 2*T_12*df/dy + 2*T_13*df/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 15){
        Dx1 = (double complex *) malloc(ncol * DMnd_zex * sizeof(double complex) ); // 2*T_13*dV/dx + 2*T_23*dV/dy
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 16){
        Dx1 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // 2*T_12*dV/dx + 2*T_23*dV/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 17){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // 2*T_12*df/dy + 2*T_13*df/dz
        Dx2 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // df/dz
        assert(Dx1 != NULL && Dx2 != NULL);
    }

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

        // copy receive buffer into extended domain
        count = 0;
        for (nbrcount = 0; nbrcount < 26; nbrcount++) {
            int block_region = grid_outside_region(
                istart_in[nbrcount], jstart_in[nbrcount], kstart_in[nbrcount],
                -FDn, -FDn, -FDn, DMVertices, gridsizes
            );
            double complex phase_factor = block_region >= 0 ? phase_factors[block_region] : 1.0;
            for (n = 0; n < ncol; n++) {
                nshift = n * DMnd_ex;
                for (k = kstart_in[nbrcount]; k < kend_in[nbrcount]; k++) {
                    kshift = nshift + k * DMnxny_ex;
                    for (j = jstart_in[nbrcount]; j < jend_in[nbrcount]; j++) {
                        jshift = kshift + j * DMnx_ex;
                        for (i = istart_in[nbrcount]; i < iend_in[nbrcount]; i++) {
                            ind = jshift + i;
                            x_ex[ind] = x_in[count++] * phase_factor;
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
    } else {
        int nbr_i, ind1;

        // copy the extended part from x into x_ex
        for (nbr_i = 0; nbr_i < 26; nbr_i++) {
            if(isnonzero[nbr_i]){
                const double complex phase_factor = phase_factors[nbr_i];
                for (n = 0; n < ncol; n++) {
                    nshift = n * DMnd; nshift1 = n * DMnd_ex;
                    for (k = kstart[nbr_i], kp = kstart_in[nbr_i]; k < kend[nbr_i]; k++, kp++) {
                        kshift = nshift + k * DMnxny; kshift1 = nshift1 + kp * DMnxny_ex;
                        for (j = jstart[nbr_i], jp = jstart_in[nbr_i]; j < jend[nbr_i]; j++, jp++) {
                            jshift = kshift + j * DMnx; jshift1 = kshift1 + jp * DMnx_ex;
                            for (i = istart[nbr_i], ip = istart_in[nbr_i]; i < iend[nbr_i]; i++, ip++) {
                                ind = jshift + i;
                                ind1 = jshift1 + ip;
                                if (is_grid_outside(ip, jp, kp, -FDn, -FDn, -FDn, DMVertices, gridsizes))
                                    x_ex[ind1] = x[ind] * phase_factor;
                                else
                                    x_ex[ind1] = x[ind];
                            }
                        }
                    }
                }
            }
        }
    }

    #ifdef USE_EVA_MODULE
    st = MPI_Wtime();
    #endif

    if(pSPARC->cell_typ == 11){
        // calculate Lx
        for (n = 0; n < ncol; n++) {
            Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 12){
        // calculate Lx
        for (n = 0; n < ncol; n++) {
            Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 13){
        // calculate Lx
        for (n = 0; n < ncol; n++) {
            Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                    0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 14){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 15){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                            0, DMnx, 0, DMny, 0, DMnz_ex, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 16){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 17){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                                0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);

            Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_5comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
        free(Dx2); Dx2 = NULL;
    }

    free(x_ex);
    free(Lap_wt);

    #ifdef USE_EVA_MODULE
    et = MPI_Wtime();
    krnl_t += et - st;

    EVA_buff_timer_add(cpyx_t, pack_t, comm_t, unpk_t, krnl_t, 0.0);
    EVA_buff_rhs_add(ncol, 0);
    #endif
}



/*
 * @brief: function to calculate two derivatives together
 */
void Calc_DX1_DX2_kpt(
    const double complex *X, double complex *DX,
    const int radius,
    const int stride_X_1,    const int stride_X_2,
    const int stride_y_X,    const int stride_y_DX,
    const int stride_z_X,    const int stride_z_DX,
    const int x_DX_spos,     const int x_DX_epos,
    const int y_DX_spos,     const int y_DX_epos,
    const int z_DX_spos,     const int z_DX_epos,
    const int x_X_spos,      const int y_X_spos,
    const int z_X_spos,
    const double *stencil_coefs1,
    const double *stencil_coefs2)
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
                double complex temp1 = 0.0;
                double complex temp2 = 0.0;
                for (r = 1; r <= radius; r++)
                {
                    int stride_X_1_r = r * stride_X_1;
                    int stride_X_2_r = r * stride_X_2;
                    temp1 += (X[ishift_X + stride_X_1_r] - X[ishift_X - stride_X_1_r]) * stencil_coefs1[r];
                    temp2 += (X[ishift_X + stride_X_2_r] - X[ishift_X - stride_X_2_r]) * stencil_coefs2[r];
                }
                DX[ishift_DX] = temp1 + temp2;
            }
        }
    }
}



/*
 * @brief: function to perform 4 component stencil operation
 */
void stencil_4comp_kpt(
    const double complex *X,   const double complex *DX,
    const int radius,
    const int stride_DX,       const int stride_y_X1,
    const int stride_y_X,      const int stride_y_DX,
    const int stride_z_X1,     const int stride_z_X,
    const int stride_z_DX,     const int x_X1_spos,
    const int x_X1_epos,       const int y_X1_spos,
    const int y_X1_epos,       const int z_X1_spos,
    const int z_X1_epos,       const int x_X_spos,
    const int y_X_spos,        const int z_X_spos,
    const int x_DX_spos,       const int y_DX_spos,
    const int z_DX_spos,       const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double coef_0,       const double b,
    const double *v0,          double complex *X1)
{
    int i, j, k, jj, kk, jjj, kkk, r;

    for (k = z_X1_spos, kk = z_X_spos, kkk = z_DX_spos; k < z_X1_epos; k++, kk++, kkk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        int kshift_DX = kkk * stride_z_DX;
        for (j = y_X1_spos, jj = y_X_spos, jjj = y_DX_spos; j < y_X1_epos; j++, jj++, jjj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            int jshift_DX = kshift_DX + jjj * stride_y_DX;
            const int niters = x_X1_epos - x_X1_spos;
#ifdef ENABLE_SIMD_COMPLEX            
#pragma omp simd
#endif
            for (i = 0; i < niters; i++)
            {
                int ishift_X1    = jshift_X1 + i + x_X1_spos;
                int ishift_X     = jshift_X  + i + x_X_spos;
                int ishift_DX    = jshift_DX + i + x_DX_spos;
                double complex res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX_r = r * stride_DX;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 4 * r + 1;
                    double complex res_x = (X[ishift_X - r]             + X[ishift_X + r]            ) * stencil_coefs[r_fac];
                    double complex res_y = (X[ishift_X - stride_y_r]    + X[ishift_X + stride_y_r]   ) * stencil_coefs[r_fac+1];
                    double complex res_z = (X[ishift_X - stride_z_r]    + X[ishift_X + stride_z_r]   ) * stencil_coefs[r_fac+2];
                    double complex res_m = (DX[ishift_DX + stride_DX_r] - DX[ishift_DX - stride_DX_r]) * stencil_coefs[r_fac+3];
                    res += res_x + res_y + res_z + res_m;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}



/*
 @ brief: function to perform 5 component stencil operation
*/
void stencil_5comp_kpt(
    const double complex *X,   const double complex *DX1,
    const double complex *DX2, const int radius,
    const int stride_DX1,      const int stride_DX2,
    const int stride_y_X1,     const int stride_y_X,
    const int stride_y_DX1,    const int stride_y_DX2,
    const int stride_z_X1,     const int stride_z_X,
    const int stride_z_DX1,    const int stride_z_DX2,
    const int x_X1_spos,       const int x_X1_epos,
    const int y_X1_spos,       const int y_X1_epos,
    const int z_X1_spos,       const int z_X1_epos,
    const int x_X_spos,        const int y_X_spos,
    const int z_X_spos,        const int x_DX1_spos,
    const int y_DX1_spos,      const int z_DX1_spos,
    const int x_DX2_spos,      const int y_DX2_spos,
    const int z_DX2_spos,      const double *stencil_coefs,
    const double coef_0,       const double b,
    const double *v0,          double complex *X1)
{
    int i, j, k, jj, kk, jjj, kkk, jjjj, kkkk, r;
    for (k = z_X1_spos, kk = z_X_spos, kkk = z_DX1_spos, kkkk = z_DX2_spos; k < z_X1_epos; k++, kk++, kkk++, kkkk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        int kshift_DX1 = kkk * stride_z_DX1;
        int kshift_DX2 = kkkk * stride_z_DX2;
        for (j = y_X1_spos, jj = y_X_spos, jjj = y_DX1_spos, jjjj = y_DX2_spos; j < y_X1_epos; j++, jj++, jjj++, jjjj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            int jshift_DX1 = kshift_DX1 + jjj * stride_y_DX1;
            int jshift_DX2 = kshift_DX2 + jjjj * stride_y_DX2;
            const int niters = x_X1_epos - x_X1_spos;
#ifdef ENABLE_SIMD_COMPLEX            
#pragma omp simd
#endif            
            for (i = 0; i < niters; i++)
            {
                int ishift_X1  = jshift_X1  + i + x_X1_spos;
                int ishift_X   = jshift_X   + i + x_X_spos;
                int ishift_DX1 = jshift_DX1 + i + x_DX1_spos;
                int ishift_DX2 = jshift_DX2 + i + x_DX2_spos;
                double complex res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX1_r = r * stride_DX1;
                    int stride_DX2_r = r * stride_DX2;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 5 * r;
                    double complex res_x  = (X[ishift_X - r]                + X[ishift_X + r]               ) * stencil_coefs[r_fac];
                    double complex res_y  = (X[ishift_X - stride_y_r]       + X[ishift_X + stride_y_r]      ) * stencil_coefs[r_fac+1];
                    double complex res_z  = (X[ishift_X - stride_z_r]       + X[ishift_X + stride_z_r]      ) * stencil_coefs[r_fac+2];
                    double complex res_m1 = (DX1[ishift_DX1 + stride_DX1_r] - DX1[ishift_DX1 - stride_DX1_r]) * stencil_coefs[r_fac+3];
                    double complex res_m2 = (DX2[ishift_DX2 + stride_DX2_r] - DX2[ishift_DX2 - stride_DX2_r]) * stencil_coefs[r_fac+4];
                    res += res_x + res_y + res_z + res_m1 + res_m2;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}
