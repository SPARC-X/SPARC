/**
 * @file    lapVecRoutines.c
 * @brief   This file contains functions for performing laplacian-vector 
 *          multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h> 

#include "lapVecRoutines.h"
#include "gradVecRoutines.h"
#include "isddft.h"

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
#endif

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))


/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 */
void Lap_vec_mult(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, double *x, double *Lapx, MPI_Comm comm
) 
{
    int dims[3], periods[3], my_coords[3];
    MPI_Cart_get(comm, 3, dims, periods, my_coords);
    
    if(pSPARC->cell_typ == 0) {
        Lap_vec_mult_orth(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, dims);
    } else {
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph; // pSPARC->comm_dist_graph_phi
        else    
            comm2 = pSPARC->comm_dist_graph_phi;
        
        //comm2 = pSPARC->comm_dist_graph_phi;
        // TODO: make the second communicator general rather than only for phi
        Lap_vec_mult_nonorth(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, comm2, dims);       
    }
}


/**
 * @brief   Calculate the residual of the poisson equation: r = b - (-(Lap+c) * x).
 *
 *          The vector x is assumed to be stored domain-wisely among the processors. The
 *          structure pSPARC contains the description of the distribution info of x, and
 *          in this case the info of Laplacian operator such as boundary conditions, 
 *          finite-difference order and coefficients etc.
 */
void poisson_residual(SPARC_OBJ *pSPARC, int N, double c, double *x, double *b, double *r, MPI_Comm comm, double *time_info) 
{
    int i;
    double t1 = MPI_Wtime();
    Lap_vec_mult(pSPARC, N, pSPARC->DMVertices, 1, c, x, r, comm);
    double t2 = MPI_Wtime();
    *time_info = t2 - t1;
    
    // Calculate residual once Lx is obtained
    for (i = 0; i < N; i++) r[i] += b[i];
}



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
        case 6:
            stencil_3axis_thread_radius6(
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
                          
    int i, j, k;
    
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
    for (n = 0; n < ncol; n++) {
        stencil_3axis_thread_v2(
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
void Lap_vec_mult_nonorth(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
        const int ncol, const double a, const double c, const double *x, 
        double *y, MPI_Comm comm,  MPI_Comm comm2, const int *dims
) 
{   
    unsigned i;
    // Call the function for (a*Lap+b*v+c)x with b = 0 and v = NULL
    for (i = 0; i < ncol; i++) {
        Lap_plus_diag_vec_mult_nonorth(
            pSPARC, DMnd, DMVertices, 1, a, 0.0, c, NULL, 
            x+i*(unsigned)DMnd, y+i*(unsigned)DMnd, comm, comm2, dims
        );
    }
}

/**
 * @brief: function to provide start and end indices of the buffer for communication
 */
void snd_rcv_buffer(
        const int nproc, const int *dims, const int *periods, const int FDn,
        const int DMnx, const int DMny, const int DMnz, int *istart,
        int *iend, int *jstart, int *jend, int *kstart,
        int *kend, int *istart_in, int *iend_in, int *jstart_in, 
        int *jend_in, int *kstart_in, int *kend_in, int *isnonzero
)        
{
    int DMnx_in  = DMnx - FDn;
    int DMny_in  = DMny - FDn;
    int DMnz_in  = DMnz - FDn;
    int DMnx_out = DMnx + FDn;
    int DMny_out = DMny + FDn;
    int DMnz_out = DMnz + FDn;
        
    int DMnx_ex = DMnx_out + FDn;
    int DMny_ex = DMny_out + FDn;
    int DMnz_ex = DMnz_out + FDn;   
    
    if(nproc > 1){
        if(dims[0] > 2 || periods[0] == 0){
            istart[0] = 0;          iend[0] = FDn;    
            istart[1] = 0;          iend[1] = DMnx;   
            istart[2] = DMnx_in;    iend[2] = DMnx;   
            istart[3] = 0;          iend[3] = FDn;    
            istart[4] = 0;          iend[4] = DMnx;   
            istart[5] = DMnx_in;    iend[5] = DMnx;   
            istart[6] = 0;          iend[6] = FDn;    
            istart[7] = 0;          iend[7] = DMnx;   
            istart[8] = DMnx_in;    iend[8] = DMnx;   

            istart[9] = 0;          iend[9] = FDn;    
            istart[10] = 0;         iend[10] = DMnx;  
            istart[11] = DMnx_in;   iend[11] = DMnx;  
            istart[12] = 0;         iend[12] = FDn;   
            istart[13] = DMnx_in;   iend[13] = DMnx;  
            istart[14] = 0;         iend[14] = FDn;   
            istart[15] = 0;         iend[15] = DMnx;  
            istart[16] = DMnx_in;   iend[16] = DMnx;  

            istart[17] = 0;          iend[17] = FDn;  
            istart[18] = 0;          iend[18] = DMnx; 
            istart[19] = DMnx_in;    iend[19] = DMnx; 
            istart[20] = 0;          iend[20] = FDn;  
            istart[21] = 0;          iend[21] = DMnx; 
            istart[22] = DMnx_in;    iend[22] = DMnx; 
            istart[23] = 0;          iend[23] = FDn;  
            istart[24] = 0;          iend[24] = DMnx; 
            istart[25] = DMnx_in;    iend[25] = DMnx; 
        } else{
            istart[0] = DMnx_in;    iend[0] = DMnx;     
            istart[1] = 0;          iend[1] = DMnx;     
            istart[2] = 0;          iend[2] = FDn;      
            istart[3] = DMnx_in;    iend[3] = DMnx;     
            istart[4] = 0;          iend[4] = DMnx;     
            istart[5] = 0;          iend[5] = FDn;      
            istart[6] = DMnx_in;    iend[6] = DMnx;     
            istart[7] = 0;          iend[7] = DMnx;     
            istart[8] = 0;          iend[8] = FDn;      
            
            istart[9] = DMnx_in;   iend[9] = DMnx;     
           istart[10] = 0;         iend[10] = DMnx;    
           istart[11] = 0;         iend[11] = FDn;     
           istart[12] = DMnx_in;   iend[12] = DMnx;    
           istart[13] = 0;         iend[13] = FDn;     
           istart[14] = DMnx_in;   iend[14] = DMnx;    
           istart[15] = 0;         iend[15] = DMnx;    
           istart[16] = 0;         iend[16] = FDn;     
           
           istart[17] = DMnx_in;   iend[17] = DMnx;    
           istart[18] = 0;         iend[18] = DMnx;    
           istart[19] = 0;         iend[19] = FDn;     
           istart[20] = DMnx_in;   iend[20] = DMnx;    
           istart[21] = 0;         iend[21] = DMnx;    
           istart[22] = 0;         iend[22] = FDn;     
           istart[23] = DMnx_in;   iend[23] = DMnx;    
           istart[24] = 0;         iend[24] = DMnx;    
           istart[25] = 0;         iend[25] = FDn;     
        }
        
        if(dims[1] > 2 || periods[1] == 0){
            jstart[0] = 0;          jend[0] = FDn;
            jstart[1] = 0;          jend[1] = FDn;
            jstart[2] = 0;          jend[2] = FDn;
            jstart[3] = 0;          jend[3] = DMny;
            jstart[4] = 0;          jend[4] = DMny;
            jstart[5] = 0;          jend[5] = DMny;
            jstart[6] = DMny_in;    jend[6] = DMny;
            jstart[7] = DMny_in;    jend[7] = DMny;
            jstart[8] = DMny_in;    jend[8] = DMny;
            
            jstart[9] = 0;          jend[9] = FDn;
            jstart[10] = 0;         jend[10] = FDn;
            jstart[11] = 0;         jend[11] = FDn;
            jstart[12] = 0;         jend[12] = DMny;
            jstart[13] = 0;         jend[13] = DMny;
            jstart[14] = DMny_in;   jend[14] = DMny;
            jstart[15] = DMny_in;   jend[15] = DMny;
            jstart[16] = DMny_in;   jend[16] = DMny;
            
            jstart[17] = 0;          jend[17] = FDn;
            jstart[18] = 0;          jend[18] = FDn;
            jstart[19] = 0;          jend[19] = FDn;
            jstart[20] = 0;          jend[20] = DMny;
            jstart[21] = 0;          jend[21] = DMny;
            jstart[22] = 0;          jend[22] = DMny;
            jstart[23] = DMny_in;    jend[23] = DMny;
            jstart[24] = DMny_in;    jend[24] = DMny;
            jstart[25] = DMny_in;    jend[25] = DMny;
        } else{	
            jstart[0] = DMny_in;    jend[0] = DMny; 	
            jstart[1] = DMny_in;    jend[1] = DMny; 
            jstart[2] = DMny_in;    jend[2] = DMny; 
            jstart[3] = 0;          jend[3] = DMny; 
            jstart[4] = 0;          jend[4] = DMny; 
            jstart[5] = 0;          jend[5] = DMny; 
            jstart[6] = 0;          jend[6] = FDn;  
            jstart[7] = 0;          jend[7] = FDn;  
            jstart[8] = 0;          jend[8] = FDn;  
                                                 
            jstart[9] = DMny_in;    jend[9] = DMny; 
            jstart[10] = DMny_in;   jend[10] = DMny; 
            jstart[11] = DMny_in;   jend[11] = DMny; 
            jstart[12] = 0;         jend[12] = DMny; 
            jstart[13] = 0;         jend[13] = DMny; 
            jstart[14] = 0;         jend[14] = FDn;  
            jstart[15] = 0;         jend[15] = FDn;  
            jstart[16] = 0;         jend[16] = FDn;  
                                                 
            jstart[17] = DMny_in;   jend[17] = DMny; 
            jstart[18] = DMny_in;   jend[18] = DMny; 
            jstart[19] = DMny_in;   jend[19] = DMny; 
            jstart[20] = 0;         jend[20] = DMny; 
            jstart[21] = 0;         jend[21] = DMny; 
            jstart[22] = 0;         jend[22] = DMny; 
            jstart[23] = 0;         jend[23] = FDn;  
            jstart[24] = 0;         jend[24] = FDn;  
            jstart[25] = 0;         jend[25] = FDn;  

        }

        if(dims[2] > 2 || periods[2] == 0){
            kstart[0] = 0;   kend[0] = FDn;   
            kstart[1] = 0;   kend[1] = FDn;   
            kstart[2] = 0;   kend[2] = FDn;   
            kstart[3] = 0;   kend[3] = FDn;   
            kstart[4] = 0;   kend[4] = FDn;   
            kstart[5] = 0;   kend[5] = FDn;   
            kstart[6] = 0;   kend[6] = FDn;   
            kstart[7] = 0;   kend[7] = FDn;   
            kstart[8] = 0;   kend[8] = FDn;   
                                                     
            kstart[9] = 0;       kend[9] = DMnz;   
            kstart[10] = 0;      kend[10] = DMnz;  
            kstart[11] = 0;      kend[11] = DMnz;  
            kstart[12] = 0;      kend[12] = DMnz;  
            kstart[13] = 0;      kend[13] = DMnz;  
            kstart[14] = 0;      kend[14] = DMnz;  
            kstart[15] = 0;      kend[15] = DMnz;  
            kstart[16] = 0;      kend[16] = DMnz;  
                                                     
            kstart[17] = DMnz_in;   kend[17] = DMnz; 
            kstart[18] = DMnz_in;   kend[18] = DMnz; 
            kstart[19] = DMnz_in;   kend[19] = DMnz; 
            kstart[20] = DMnz_in;   kend[20] = DMnz; 
            kstart[21] = DMnz_in;   kend[21] = DMnz; 
            kstart[22] = DMnz_in;   kend[22] = DMnz; 
            kstart[23] = DMnz_in;   kend[23] = DMnz; 
            kstart[24] = DMnz_in;   kend[24] = DMnz; 
            kstart[25] = DMnz_in;   kend[25] = DMnz;
        } else{
            kstart[0] = DMnz_in;   kend[0] = DMnz;
            kstart[1] = DMnz_in;   kend[1] = DMnz;
            kstart[2] = DMnz_in;   kend[2] = DMnz;
            kstart[3] = DMnz_in;   kend[3] = DMnz;
            kstart[4] = DMnz_in;   kend[4] = DMnz;
            kstart[5] = DMnz_in;   kend[5] = DMnz;
            kstart[6] = DMnz_in;   kend[6] = DMnz;
            kstart[7] = DMnz_in;   kend[7] = DMnz;
            kstart[8] = DMnz_in;   kend[8] = DMnz;
                                                 
            kstart[9] = 0;          kend[9] = DMnz;
            kstart[10] = 0;         kend[10] = DMnz;
            kstart[11] = 0;         kend[11] = DMnz;
            kstart[12] = 0;         kend[12] = DMnz;
            kstart[13] = 0;         kend[13] = DMnz;
            kstart[14] = 0;         kend[14] = DMnz;
            kstart[15] = 0;         kend[15] = DMnz;
            kstart[16] = 0;         kend[16] = DMnz;
                                                 
            kstart[17] = 0;         kend[17] = FDn; 
            kstart[18] = 0;         kend[18] = FDn; 
            kstart[19] = 0;         kend[19] = FDn; 
            kstart[20] = 0;         kend[20] = FDn; 
            kstart[21] = 0;         kend[21] = FDn; 
            kstart[22] = 0;         kend[22] = FDn; 
            kstart[23] = 0;         kend[23] = FDn; 
            kstart[24] = 0;         kend[24] = FDn; 
            kstart[25] = 0;         kend[25] = FDn; 
        }
  
  
        istart_in[0] = 0;          iend_in[0] = FDn;       jstart_in[0] = 0;         jend_in[0] = FDn;       kstart_in[0] = 0;          kend_in[0] = FDn; 
        istart_in[1] = FDn;        iend_in[1] = DMnx_out;  jstart_in[1] = 0;         jend_in[1] = FDn;       kstart_in[1] = 0;          kend_in[1] = FDn;
        istart_in[2] = DMnx_out;   iend_in[2] = DMnx_ex;   jstart_in[2] = 0;         jend_in[2] = FDn;       kstart_in[2] = 0;          kend_in[2] = FDn;
        istart_in[3] = 0;          iend_in[3] = FDn;       jstart_in[3] = FDn;       jend_in[3] = DMny_out;  kstart_in[3] = 0;          kend_in[3] = FDn;
        istart_in[4] = FDn;        iend_in[4] = DMnx_out;  jstart_in[4] = FDn;       jend_in[4] = DMny_out;  kstart_in[4] = 0;          kend_in[4] = FDn;
        istart_in[5] = DMnx_out;   iend_in[5] = DMnx_ex;   jstart_in[5] = FDn;       jend_in[5] = DMny_out;  kstart_in[5] = 0;          kend_in[5] = FDn;
        istart_in[6] = 0;          iend_in[6] = FDn;       jstart_in[6] = DMny_out;  jend_in[6] = DMny_ex;   kstart_in[6] = 0;          kend_in[6] = FDn;
        istart_in[7] = FDn;        iend_in[7] = DMnx_out;  jstart_in[7] = DMny_out;  jend_in[7] = DMny_ex;   kstart_in[7] = 0;          kend_in[7] = FDn;
        istart_in[8] = DMnx_out;   iend_in[8] = DMnx_ex;   jstart_in[8] = DMny_out;  jend_in[8] = DMny_ex;   kstart_in[8] = 0;          kend_in[8] = FDn;
        
        istart_in[9] = 0;          iend_in[9] = FDn;       jstart_in[9] = 0;         jend_in[9] = FDn;       kstart_in[9] = FDn;        kend_in[9] = DMnz_out; 
        istart_in[10] = FDn;       iend_in[10] = DMnx_out; jstart_in[10] = 0;        jend_in[10] = FDn;      kstart_in[10] = FDn;       kend_in[10] = DMnz_out;
        istart_in[11] = DMnx_out;  iend_in[11] = DMnx_ex;  jstart_in[11] = 0;        jend_in[11] = FDn;      kstart_in[11] = FDn;       kend_in[11] = DMnz_out;
        istart_in[12] = 0;         iend_in[12] = FDn;      jstart_in[12] = FDn;      jend_in[12] = DMny_out; kstart_in[12] = FDn;       kend_in[12] = DMnz_out;
        istart_in[13] = DMnx_out;  iend_in[13] = DMnx_ex;  jstart_in[13] = FDn;      jend_in[13] = DMny_out; kstart_in[13] = FDn;       kend_in[13] = DMnz_out;
        istart_in[14] = 0;         iend_in[14] = FDn;      jstart_in[14] = DMny_out; jend_in[14] = DMny_ex;  kstart_in[14] = FDn;       kend_in[14] = DMnz_out;
        istart_in[15] = FDn;       iend_in[15] = DMnx_out; jstart_in[15] = DMny_out; jend_in[15] = DMny_ex;  kstart_in[15] = FDn;       kend_in[15] = DMnz_out;
        istart_in[16] = DMnx_out;  iend_in[16] = DMnx_ex;  jstart_in[16] = DMny_out; jend_in[16] = DMny_ex;  kstart_in[16] = FDn;       kend_in[16] = DMnz_out;
        
        istart_in[17] = 0;         iend_in[17] = FDn;      jstart_in[17] = 0;        jend_in[17] = FDn;      kstart_in[17] = DMnz_out;  kend_in[17] = DMnz_ex; 
        istart_in[18] = FDn;       iend_in[18] = DMnx_out; jstart_in[18] = 0;        jend_in[18] = FDn;      kstart_in[18] = DMnz_out;  kend_in[18] = DMnz_ex;
        istart_in[19] = DMnx_out;  iend_in[19] = DMnx_ex;  jstart_in[19] = 0;        jend_in[19] = FDn;      kstart_in[19] = DMnz_out;  kend_in[19] = DMnz_ex;
        istart_in[20] = 0;         iend_in[20] = FDn;      jstart_in[20] = FDn;      jend_in[20] = DMny_out; kstart_in[20] = DMnz_out;  kend_in[20] = DMnz_ex;
        istart_in[21] = FDn;       iend_in[21] = DMnx_out; jstart_in[21] = FDn;      jend_in[21] = DMny_out; kstart_in[21] = DMnz_out;  kend_in[21] = DMnz_ex;
        istart_in[22] = DMnx_out;  iend_in[22] = DMnx_ex;  jstart_in[22] = FDn;      jend_in[22] = DMny_out; kstart_in[22] = DMnz_out;  kend_in[22] = DMnz_ex;
        istart_in[23] = 0;         iend_in[23] = FDn;      jstart_in[23] = DMny_out; jend_in[23] = DMny_ex;  kstart_in[23] = DMnz_out;  kend_in[23] = DMnz_ex;
        istart_in[24] = FDn;       iend_in[24] = DMnx_out; jstart_in[24] = DMny_out; jend_in[24] = DMny_ex;  kstart_in[24] = DMnz_out;  kend_in[24] = DMnz_ex;
        istart_in[25] = DMnx_out;  iend_in[25] = DMnx_ex;  jstart_in[25] = DMny_out; jend_in[25] = DMny_ex;  kstart_in[25] = DMnz_out;  kend_in[25] = DMnz_ex;
    } else{
        istart[0] = DMnx_in;    iend[0] = DMnx;   jstart[0] = DMny_in;    jend[0] = DMny;   kstart[0] = DMnz_in;   kend[0] = DMnz;   isnonzero[0] = (periods[0] && periods[1] && periods[2]);
        istart[1] = 0;          iend[1] = DMnx;   jstart[1] = DMny_in;    jend[1] = DMny;   kstart[1] = DMnz_in;   kend[1] = DMnz;   isnonzero[1] = (periods[1] && periods[2]);
        istart[2] = 0;          iend[2] = FDn;    jstart[2] = DMny_in;    jend[2] = DMny;   kstart[2] = DMnz_in;   kend[2] = DMnz;   isnonzero[2] = (periods[0] && periods[1] && periods[2]);
        istart[3] = DMnx_in;    iend[3] = DMnx;   jstart[3] = 0;          jend[3] = DMny;   kstart[3] = DMnz_in;   kend[3] = DMnz;   isnonzero[3] = (periods[0] && periods[2]);
        istart[4] = 0;          iend[4] = DMnx;   jstart[4] = 0;          jend[4] = DMny;   kstart[4] = DMnz_in;   kend[4] = DMnz;   isnonzero[4] = (periods[2]);
        istart[5] = 0;          iend[5] = FDn;    jstart[5] = 0;          jend[5] = DMny;   kstart[5] = DMnz_in;   kend[5] = DMnz;   isnonzero[5] = (periods[0] && periods[2]);
        istart[6] = DMnx_in;    iend[6] = DMnx;   jstart[6] = 0;          jend[6] = FDn;    kstart[6] = DMnz_in;   kend[6] = DMnz;   isnonzero[6] = (periods[0] && periods[1] && periods[2]);
        istart[7] = 0;          iend[7] = DMnx;   jstart[7] = 0;          jend[7] = FDn;    kstart[7] = DMnz_in;   kend[7] = DMnz;   isnonzero[7] = (periods[1] && periods[2]);
        istart[8] = 0;          iend[8] = FDn;    jstart[8] = 0;          jend[8] = FDn;    kstart[8] = DMnz_in;   kend[8] = DMnz;   isnonzero[8] = (periods[0] && periods[1] && periods[2]);
    
        istart[9] = DMnx_in;    iend[9] = DMnx;   jstart[9] = DMny_in;    jend[9] = DMny;   kstart[9] = 0;          kend[9] = DMnz;   isnonzero[9] = (periods[0] && periods[1]);
        istart[10] = 0;         iend[10] = DMnx;  jstart[10] = DMny_in;   jend[10] = DMny;  kstart[10] = 0;         kend[10] = DMnz;  isnonzero[10] = (periods[1]);
        istart[11] = 0;         iend[11] = FDn;   jstart[11] = DMny_in;   jend[11] = DMny;  kstart[11] = 0;         kend[11] = DMnz;  isnonzero[11] = (periods[0] && periods[1]);
        istart[12] = DMnx_in;   iend[12] = DMnx;  jstart[12] = 0;         jend[12] = DMny;  kstart[12] = 0;         kend[12] = DMnz;  isnonzero[12] = (periods[0]);
        istart[13] = 0;         iend[13] = FDn;   jstart[13] = 0;         jend[13] = DMny;  kstart[13] = 0;         kend[13] = DMnz;  isnonzero[13] = (periods[0]);
        istart[14] = DMnx_in;   iend[14] = DMnx;  jstart[14] = 0;         jend[14] = FDn;   kstart[14] = 0;         kend[14] = DMnz;  isnonzero[14] = (periods[0] && periods[1]);
        istart[15] = 0;         iend[15] = DMnx;  jstart[15] = 0;         jend[15] = FDn;   kstart[15] = 0;         kend[15] = DMnz;  isnonzero[15] = (periods[1]);
        istart[16] = 0;         iend[16] = FDn;   jstart[16] = 0;         jend[16] = FDn;   kstart[16] = 0;         kend[16] = DMnz;  isnonzero[16] = (periods[0] && periods[1]);

        istart[17] = DMnx_in;   iend[17] = DMnx;  jstart[17] = DMny_in;   jend[17] = DMny;  kstart[17] = 0;         kend[17] = FDn;   isnonzero[17] = (periods[0] && periods[1] && periods[2]);
        istart[18] = 0;         iend[18] = DMnx;  jstart[18] = DMny_in;   jend[18] = DMny;  kstart[18] = 0;         kend[18] = FDn;   isnonzero[18] = (periods[1] && periods[2]);
        istart[19] = 0;         iend[19] = FDn;   jstart[19] = DMny_in;   jend[19] = DMny;  kstart[19] = 0;         kend[19] = FDn;   isnonzero[19] = (periods[0] && periods[1] && periods[2]);
        istart[20] = DMnx_in;   iend[20] = DMnx;  jstart[20] = 0;         jend[20] = DMny;  kstart[20] = 0;         kend[20] = FDn;   isnonzero[20] = (periods[0] && periods[2]);
        istart[21] = 0;         iend[21] = DMnx;  jstart[21] = 0;         jend[21] = DMny;  kstart[21] = 0;         kend[21] = FDn;   isnonzero[21] = (periods[2]);
        istart[22] = 0;         iend[22] = FDn;   jstart[22] = 0;         jend[22] = DMny;  kstart[22] = 0;         kend[22] = FDn;   isnonzero[22] = (periods[0] && periods[2]);
        istart[23] = DMnx_in;   iend[23] = DMnx;  jstart[23] = 0;         jend[23] = FDn;   kstart[23] = 0;         kend[23] = FDn;   isnonzero[23] = (periods[0] && periods[1] && periods[2]);
        istart[24] = 0;         iend[24] = DMnx;  jstart[24] = 0;         jend[24] = FDn;   kstart[24] = 0;         kend[24] = FDn;   isnonzero[24] = (periods[1] && periods[2]);
        istart[25] = 0;         iend[25] = FDn;   jstart[25] = 0;         jend[25] = FDn;   kstart[25] = 0;         kend[25] = FDn;   isnonzero[25] = (periods[0] && periods[1] && periods[2]);

 
        istart_in[0] = 0;          iend_in[0] = FDn;       jstart_in[0] = 0;         jend_in[0] = FDn;       kstart_in[0] = 0;          kend_in[0] = FDn; 
        istart_in[1] = FDn;        iend_in[1] = DMnx_out;  jstart_in[1] = 0;         jend_in[1] = FDn;       kstart_in[1] = 0;          kend_in[1] = FDn;
        istart_in[2] = DMnx_out;   iend_in[2] = DMnx_ex;   jstart_in[2] = 0;         jend_in[2] = FDn;       kstart_in[2] = 0;          kend_in[2] = FDn;
        istart_in[3] = 0;          iend_in[3] = FDn;       jstart_in[3] = FDn;       jend_in[3] = DMny_out;  kstart_in[3] = 0;          kend_in[3] = FDn;
        istart_in[4] = FDn;        iend_in[4] = DMnx_out;  jstart_in[4] = FDn;       jend_in[4] = DMny_out;  kstart_in[4] = 0;          kend_in[4] = FDn;
        istart_in[5] = DMnx_out;   iend_in[5] = DMnx_ex;   jstart_in[5] = FDn;       jend_in[5] = DMny_out;  kstart_in[5] = 0;          kend_in[5] = FDn;
        istart_in[6] = 0;          iend_in[6] = FDn;       jstart_in[6] = DMny_out;  jend_in[6] = DMny_ex;   kstart_in[6] = 0;          kend_in[6] = FDn;
        istart_in[7] = FDn;        iend_in[7] = DMnx_out;  jstart_in[7] = DMny_out;  jend_in[7] = DMny_ex;   kstart_in[7] = 0;          kend_in[7] = FDn;
        istart_in[8] = DMnx_out;   iend_in[8] = DMnx_ex;   jstart_in[8] = DMny_out;  jend_in[8] = DMny_ex;   kstart_in[8] = 0;          kend_in[8] = FDn;
        
        istart_in[9] = 0;          iend_in[9] = FDn;       jstart_in[9] = 0;         jend_in[9] = FDn;       kstart_in[9] = FDn;        kend_in[9] = DMnz_out; 
        istart_in[10] = FDn;       iend_in[10] = DMnx_out; jstart_in[10] = 0;        jend_in[10] = FDn;      kstart_in[10] = FDn;       kend_in[10] = DMnz_out;
        istart_in[11] = DMnx_out;  iend_in[11] = DMnx_ex;  jstart_in[11] = 0;        jend_in[11] = FDn;      kstart_in[11] = FDn;       kend_in[11] = DMnz_out;
        istart_in[12] = 0;         iend_in[12] = FDn;      jstart_in[12] = FDn;      jend_in[12] = DMny_out; kstart_in[12] = FDn;       kend_in[12] = DMnz_out;
        istart_in[13] = DMnx_out;  iend_in[13] = DMnx_ex;  jstart_in[13] = FDn;      jend_in[13] = DMny_out; kstart_in[13] = FDn;       kend_in[13] = DMnz_out;
        istart_in[14] = 0;         iend_in[14] = FDn;      jstart_in[14] = DMny_out; jend_in[14] = DMny_ex;  kstart_in[14] = FDn;       kend_in[14] = DMnz_out;
        istart_in[15] = FDn;       iend_in[15] = DMnx_out; jstart_in[15] = DMny_out; jend_in[15] = DMny_ex;  kstart_in[15] = FDn;       kend_in[15] = DMnz_out;
        istart_in[16] = DMnx_out;  iend_in[16] = DMnx_ex;  jstart_in[16] = DMny_out; jend_in[16] = DMny_ex;  kstart_in[16] = FDn;       kend_in[16] = DMnz_out;
        
        istart_in[17] = 0;         iend_in[17] = FDn;      jstart_in[17] = 0;        jend_in[17] = FDn;      kstart_in[17] = DMnz_out;  kend_in[17] = DMnz_ex; 
        istart_in[18] = FDn;       iend_in[18] = DMnx_out; jstart_in[18] = 0;        jend_in[18] = FDn;      kstart_in[18] = DMnz_out;  kend_in[18] = DMnz_ex;
        istart_in[19] = DMnx_out;  iend_in[19] = DMnx_ex;  jstart_in[19] = 0;        jend_in[19] = FDn;      kstart_in[19] = DMnz_out;  kend_in[19] = DMnz_ex;
        istart_in[20] = 0;         iend_in[20] = FDn;      jstart_in[20] = FDn;      jend_in[20] = DMny_out; kstart_in[20] = DMnz_out;  kend_in[20] = DMnz_ex;
        istart_in[21] = FDn;       iend_in[21] = DMnx_out; jstart_in[21] = FDn;      jend_in[21] = DMny_out; kstart_in[21] = DMnz_out;  kend_in[21] = DMnz_ex;
        istart_in[22] = DMnx_out;  iend_in[22] = DMnx_ex;  jstart_in[22] = FDn;      jend_in[22] = DMny_out; kstart_in[22] = DMnz_out;  kend_in[22] = DMnz_ex;
        istart_in[23] = 0;         iend_in[23] = FDn;      jstart_in[23] = DMny_out; jend_in[23] = DMny_ex;  kstart_in[23] = DMnz_out;  kend_in[23] = DMnz_ex;
        istart_in[24] = FDn;       iend_in[24] = DMnx_out; jstart_in[24] = DMny_out; jend_in[24] = DMny_ex;  kstart_in[24] = DMnz_out;  kend_in[24] = DMnz_ex;
        istart_in[25] = DMnx_out;  iend_in[25] = DMnx_ex;  jstart_in[25] = DMny_out; jend_in[25] = DMny_ex;  kstart_in[25] = DMnz_out;  kend_in[25] = DMnz_ex;
    }
}




/**
 * @brief   Calculate (a * Lap + b * diag(v) + c * I) times vectors.
 *          Warning:
 *            Although the function is intended for multiple vectors,
 *          it turns out to be slower than applying it to one vector
 *          at a time. So we will be calling this function ncol times
 *          if ncol is greater than 1.
 */
void Lap_plus_diag_vec_mult_nonorth(
        const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
        const int ncol, const double a, const double b, const double c, 
        const double *v, const double *x, double *y, MPI_Comm comm,  MPI_Comm comm2,
        const int *dims
) 
{
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
    double *x_in, *x_out;
    x_in = NULL; x_out = NULL;
    // set up send-receive buffer based on the ordering of the neighbors
    int istart[26], iend[26], jstart[26], jend[26], kstart[26], kend[26];
    int istart_in[26], iend_in[26], jstart_in[26], jend_in[26], kstart_in[26], kend_in[26], isnonzero[26];
    
    snd_rcv_buffer(nproc, dims, periods, FDn, DMnx, DMny, DMnz, istart, iend, jstart, jend, kstart, kend, istart_in, iend_in, jstart_in, jend_in, kstart_in, kend_in, isnonzero);

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
        x_in  = (double *)calloc( nd_in, sizeof(double)); // TODO: init to 0
        x_out = (double *)malloc( nd_out * sizeof(double)); // no need to init x_out        
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
        MPI_Ineighbor_alltoallv(x_out, sendcounts, sdispls, MPI_DOUBLE, 
                                x_in, recvcounts, rdispls, MPI_DOUBLE, 
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
    
    double *x_ex = (double *)calloc(ncol * DMnd_ex, sizeof(double));
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
    double *Dx1, *Dx2;
    Dx1 = NULL; Dx2 = NULL;
    if(pSPARC->cell_typ == 11){
        Dx1 = (double *) malloc(ncol * DMnd_xex * sizeof(double) ); // df/dy
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 12){
        Dx1 = (double *) malloc(ncol * DMnd_xex * sizeof(double) ); // df/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 13){
        Dx1 = (double *) malloc(ncol * DMnd_yex * sizeof(double) ); // df/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 14){
        Dx1 = (double *) malloc(ncol * DMnd_xex * sizeof(double) ); // 2*T_12*df/dy + 2*T_13*df/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 15){
        Dx1 = (double *) malloc(ncol * DMnd_zex * sizeof(double) ); // 2*T_13*dV/dx + 2*T_23*dV/dy
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 16){
        Dx1 = (double *) malloc(ncol * DMnd_yex * sizeof(double) ); // 2*T_12*dV/dx + 2*T_23*dV/dz
        assert(Dx1 != NULL);
    } else if(pSPARC->cell_typ == 17){
        Dx1 = (double *) malloc(ncol * DMnd_xex * sizeof(double) ); // 2*T_12*df/dy + 2*T_13*df/dz
        Dx2 = (double *) malloc(ncol * DMnd_yex * sizeof(double) ); // df/dz
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
            for (n = 0; n < ncol; n++) {
                nshift = n * DMnd_ex;
                for (k = kstart_in[nbrcount]; k < kend_in[nbrcount]; k++) {
                    kshift = nshift + k * DMnxny_ex;
                    for (j = jstart_in[nbrcount]; j < jend_in[nbrcount]; j++) {
                        jshift = kshift + j * DMnx_ex;
                        for (i = istart_in[nbrcount]; i < iend_in[nbrcount]; i++) {
                            ind = jshift + i;
                            x_ex[ind] = x_in[count++];
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
                for (n = 0; n < ncol; n++) {
                    nshift = n * DMnd; nshift1 = n * DMnd_ex;
                    for (k = kstart[nbr_i], kp = kstart_in[nbr_i]; k < kend[nbr_i]; k++, kp++) {
                        kshift = nshift + k * DMnxny; kshift1 = nshift1 + kp * DMnxny_ex;
                        for (j = jstart[nbr_i], jp = jstart_in[nbr_i]; j < jend[nbr_i]; j++, jp++) {
                            jshift = kshift + j * DMnx; jshift1 = kshift1 + jp * DMnx_ex;
                            for (i = istart[nbr_i], ip = istart_in[nbr_i]; i < iend[nbr_i]; i++, ip++) {
                                ind = jshift + i;
                                ind1 = jshift1 + ip;
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
            Calc_DX(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 12){
        // calculate Lx
        for (n = 0; n < ncol; n++) {
            Calc_DX(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 13){
        // calculate Lx
        for (n = 0; n < ncol; n++) {
            Calc_DX(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                    0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 14){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 15){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                            0, DMnx, 0, DMny, 0, DMnz_ex, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 16){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz);
        }

        for (n = 0; n < ncol; n++) {
            stencil_4comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 17){
        for (n = 0; n < ncol; n++) {
            Calc_DX1_DX2(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                                0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);

            Calc_DX(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
        }

        for (n = 0; n < ncol; n++) {
            stencil_5comp(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
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
 * @brief: function to store the laplacian stencil compactly
 */
void Lap_stencil_coef_compact(const SPARC_OBJ *pSPARC, const double FDn, double *Lap_stencil, const double a)
{
    int p;
    
    if(pSPARC->cell_typ == 0){
        for (p = 0; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
        } 
    } else if(pSPARC->cell_typ == 11){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_xy[p] * a;
        }    
    } else if(pSPARC->cell_typ == 12){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_xz[p] * a;
        }
    } else if(pSPARC->cell_typ == 13){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_yz[p] * a;
        }
    } else if(pSPARC->cell_typ == 14){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D1_stencil_coeffs_x[p] * a;
        }
    } else if(pSPARC->cell_typ == 15){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D1_stencil_coeffs_z[p] * a;
        }
    } else if(pSPARC->cell_typ == 16){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D1_stencil_coeffs_y[p] * a;
        }
    } else if(pSPARC->cell_typ == 17){
        for (p = 1; p <= FDn; p++)
        {
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_y[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_z[p] * a;
            (*Lap_stencil++) = pSPARC->D1_stencil_coeffs_x[p] * a;
            (*Lap_stencil++) = pSPARC->D2_stencil_coeffs_yz[p] * a; 
        }
    }
}


/*
 * @brief: function to calculate two derivatives together
 */
void Calc_DX1_DX2(
    const double *X,       double *DX,
    const int radius,      const int stride_X_1,
    const int stride_X_2,
    const int stride_y_X,  const int stride_y_DX,
    const int stride_z_X,  const int stride_z_DX,
    const int x_DX_spos,   const int x_DX_epos,
    const int y_DX_spos,   const int y_DX_epos,
    const int z_DX_spos,   const int z_DX_epos,
    const int x_X_spos,
    const int y_X_spos,    const int z_X_spos,
    const double *stencil_coefs1,
    const double *stencil_coefs2
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
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_DX = jshift_DX + i + x_DX_spos;
                int ishift_X  = jshift_X  + i + x_X_spos;
                double temp1 = 0.0;
                double temp2 = 0.0;
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
void stencil_4comp(
    const double *X,        const double *DX,
    const int radius,       const int stride_DX, 
    const int stride_y_X1,
    const int stride_y_X,   const int stride_y_DX,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX,  const int x_X1_spos,
    const int x_X1_epos,    const int y_X1_spos,
    const int y_X1_epos,    const int z_X1_spos,
    const int z_X1_epos,    const int x_X_spos,
    const int y_X_spos,     const int z_X_spos,
    const int x_DX_spos,    const int y_DX_spos,
    const int z_DX_spos,    const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double coef_0,    const double b,   
    const double *v0,       double *X1
)
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
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_X1    = jshift_X1 + i + x_X1_spos;
                int ishift_X     = jshift_X  + i + x_X_spos;
                int ishift_DX    = jshift_DX + i + x_DX_spos;
                double res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX_r = r * stride_DX;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 4 * r + 1;
                    double res_x = (X[ishift_X - r]             + X[ishift_X + r])             * stencil_coefs[r_fac];
                    double res_y = (X[ishift_X - stride_y_r]    + X[ishift_X + stride_y_r])    * stencil_coefs[r_fac+1];
                    double res_z = (X[ishift_X - stride_z_r]    + X[ishift_X + stride_z_r])    * stencil_coefs[r_fac+2];
                    double res_m = (DX[ishift_DX + stride_DX_r] - DX[ishift_DX - stride_DX_r]) * stencil_coefs[r_fac+3];
                    res += res_x + res_y + res_z + res_m;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}


/*
 * @brief: function to perform 5 component stencil operation
 */
void stencil_5comp(
    const double *X,        const double *DX1,
    const double *DX2,      const int radius,
    const int stride_DX1,   const int stride_DX2,
    const int stride_y_X1,  const int stride_y_X,
    const int stride_y_DX1, const int stride_y_DX2,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX1, const int stride_z_DX2,
    const int x_X1_spos,    const int x_X1_epos,
    const int y_X1_spos,    const int y_X1_epos,
    const int z_X1_spos,    const int z_X1_epos,
    const int x_X_spos,     const int y_X_spos,
    const int z_X_spos,     const int x_DX1_spos,
    const int y_DX1_spos,   const int z_DX1_spos,
    const int x_DX2_spos,   const int y_DX2_spos,
    const int z_DX2_spos,
    const double *stencil_coefs,
    const double coef_0,    const double b,
    const double *v0,        double *X1
)
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
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_X1  = jshift_X1  + i + x_X1_spos;
                int ishift_X   = jshift_X   + i + x_X_spos;
                int ishift_DX1 = jshift_DX1 + i + x_DX1_spos;
                int ishift_DX2 = jshift_DX2 + i + x_DX2_spos;
                double res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX1_r = r * stride_DX1;
                    int stride_DX2_r = r * stride_DX2;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 5 * r;
                    double res_x = (X[ishift_X - r]                 + X[ishift_X + r])                * stencil_coefs[r_fac];
                    double res_y = (X[ishift_X - stride_y_r]        + X[ishift_X + stride_y_r])       * stencil_coefs[r_fac+1];
                    double res_z = (X[ishift_X - stride_z_r]        + X[ishift_X + stride_z_r])       * stencil_coefs[r_fac+2];
                    double res_m1 = (DX1[ishift_DX1 + stride_DX1_r] - DX1[ishift_DX1 - stride_DX1_r]) * stencil_coefs[r_fac+3];
                    double res_m2 = (DX2[ishift_DX2 + stride_DX2_r] - DX2[ishift_DX2 - stride_DX2_r]) * stencil_coefs[r_fac+4];
                    res += res_x + res_y + res_z + res_m1 + res_m2;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
            }
        }
    }
}

/*
 * @brief: function to perform 4 component stencil operation y = Ax + beta*y
 */
void stencil_4comp_extd(
    const double *X,        const double *DX,
    const int radius,       const int stride_DX, 
    const int stride_y_X1,
    const int stride_y_X,   const int stride_y_DX,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX,  const int x_X1_spos,
    const int x_X1_epos,    const int y_X1_spos,
    const int y_X1_epos,    const int z_X1_spos,
    const int z_X1_epos,    const int x_X_spos,
    const int y_X_spos,     const int z_X_spos,
    const int x_DX_spos,    const int y_DX_spos,
    const int z_DX_spos,    const double *stencil_coefs, // ordered [x0 y0 z0 Dx0 x1 y1 y2 ... x_radius y_radius z_radius Dx_radius]
    const double coef_0,    const double b,   
    const double *v0,       const double beta,
    double *X1
)
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
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_X1 = jshift_X1 + i + x_X1_spos;
                int ishift_X  = jshift_X  + i + x_X_spos;
                int ishift_DX = jshift_DX + i + x_DX_spos;
                double res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX_r = r * stride_DX;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 4 * r + 1;
                    double res_x = (X[ishift_X - r]             + X[ishift_X + r])             * stencil_coefs[r_fac];
                    double res_y = (X[ishift_X - stride_y_r]    + X[ishift_X + stride_y_r])    * stencil_coefs[r_fac+1];
                    double res_z = (X[ishift_X - stride_z_r]    + X[ishift_X + stride_z_r])    * stencil_coefs[r_fac+2];
                    double res_m = (DX[ishift_DX + stride_DX_r] - DX[ishift_DX - stride_DX_r]) * stencil_coefs[r_fac+3];
                    res += res_x + res_y + res_z + res_m;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X])) + beta * X1[ishift_X1];
            }
        }
    }
}


/*
 * @brief: function to perform 5 component stencil operation y = Ax + beta*y
 */
void stencil_5comp_extd(
    const double *X,        const double *DX1,
    const double *DX2,      const int radius,
    const int stride_DX1,   const int stride_DX2,
    const int stride_y_X1,  const int stride_y_X,
    const int stride_y_DX1, const int stride_y_DX2,
    const int stride_z_X1,  const int stride_z_X,
    const int stride_z_DX1, const int stride_z_DX2,
    const int x_X1_spos,    const int x_X1_epos,
    const int y_X1_spos,    const int y_X1_epos,
    const int z_X1_spos,    const int z_X1_epos,
    const int x_X_spos,     const int y_X_spos,
    const int z_X_spos,     const int x_DX1_spos,
    const int y_DX1_spos,   const int z_DX1_spos,
    const int x_DX2_spos,   const int y_DX2_spos,
    const int z_DX2_spos,
    const double *stencil_coefs,
    const double coef_0,    const double b,
    const double *v0,       const double beta,
    double *X1
)
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
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_X1    = jshift_X1   + i + x_X1_spos;
                int ishift_X     = jshift_X    + i + x_X_spos;
                int ishift_DX1    = jshift_DX1 + i + x_DX1_spos;
                int ishift_DX2    = jshift_DX2 + i + x_DX2_spos;
                double res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX1_r = r * stride_DX1;
                    int stride_DX2_r = r * stride_DX2;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 5 * r;
                    double res_x = (X[ishift_X - r]                 + X[ishift_X + r])                * stencil_coefs[r_fac];
                    double res_y = (X[ishift_X - stride_y_r]        + X[ishift_X + stride_y_r])       * stencil_coefs[r_fac+1];
                    double res_z = (X[ishift_X - stride_z_r]        + X[ishift_X + stride_z_r])       * stencil_coefs[r_fac+2];
                    double res_m1 = (DX1[ishift_DX1 + stride_DX1_r] - DX1[ishift_DX1 - stride_DX1_r]) * stencil_coefs[r_fac+3];
                    double res_m2 = (DX2[ishift_DX2 + stride_DX2_r] - DX2[ishift_DX2 - stride_DX2_r]) * stencil_coefs[r_fac+4];
                    res += res_x + res_y + res_z + res_m1 + res_m2;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X])) + beta * X1[ishift_X1];
            }
        }
    }
}
