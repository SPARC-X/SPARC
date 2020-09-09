/**
 * @file    lapVecNonOrthKpt.c
 * @brief   This file contains functions for performing the non orthogonal
 *          laplacian-vector multiply routines with a Bloch factor.
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

#include "lapVecNonOrthKpt.h"
#include "lapVecNonOrth.h"
#include "gradVecRoutinesKpt.h"
#include "isddft.h"

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
#endif


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
    int periods[3];
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;

    int FDn = pSPARC->order / 2;

    // The user has to make sure DMnd = DMnx * DMny * DMnz
    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
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

    int overlap_flag = (int) (nproc > 1 && DMnx > pSPARC->order
                          && DMny > pSPARC->order && DMnz > pSPARC->order);
    int DMnxexny = DMnx_ex * DMny;
    int DMnd_xex = DMnxexny * DMnz;
    int DMnxnyex = DMnx * DMny_ex;
    int DMnd_yex = DMnxnyex * DMnz;
    int DMnd_zex = DMnxny * DMnz_ex;
    double complex *Dx1, *Dx2, phase_fac_m1, phase_fac_m2, phase_fac_m3;
    Dx1 = NULL; Dx2 = NULL;
    phase_fac_m1 = phase_fac_m2 = phase_fac_m3 = 0.0;
    if(pSPARC->cell_typ == 11){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // df/dy
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) + sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
        phase_fac_m2 = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) + sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x+n*DMnd, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty, pshifty, DMnx_ex, pshiftz, DMnxexny,
                            FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, FDn, DMny_in, pSPARC->Ny);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty, DMnx_ex, pshiftz, pshiftz, DMnxexny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, pSPARC->order, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, FDn, DMnx_in, pSPARC->Nx);

            }
        }
    } else if(pSPARC->cell_typ == 12){
        Dx1 = (double complex*) malloc(ncol * DMnd_xex * sizeof(double complex) ); // df/dz
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) + sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
        phase_fac_m2 = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) + sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x+n*DMnd, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz, pshifty, DMnx_ex, pshiftz, DMnxexny,
                            FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty, DMnx_ex, pshiftz, pshiftz, DMnxexny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, pSPARC->order, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, FDn, DMnx_in, pSPARC->Nx);
            }
        }
    } else if(pSPARC->cell_typ == 13){
        Dx1 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // df/dz
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) + sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
        phase_fac_m2 = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) + sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x+n*DMnd, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz, pshifty, DMnx, pshiftz, DMnxnyex,
                            FDn, DMnx_in, FDn, DMny_out, FDn, DMnz_in, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty, DMnx, pshiftz, pshiftz, DMnxnyex,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, FDn, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  0, pSPARC->order, 0, phase_fac_m2, FDn, DMny_in, pSPARC->Ny);
            }
        }
    } else if(pSPARC->cell_typ == 14){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // 2*T_12*df/dy + 2*T_13*df/dz
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) + sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
        phase_fac_m2 = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) + sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
        phase_fac_m3 = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) + sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty, pshiftz, pshifty, DMnx_ex, pshiftz, DMnxexny,
                                 FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                                 0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMny_in, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty, DMnx_ex, pshiftz, pshiftz, DMnxexny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, pSPARC->order, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m3, FDn, DMnx_in, pSPARC->Nx);
            }
        }
    } else if(pSPARC->cell_typ == 15){
        Dx1 = (double complex *) malloc(ncol * DMnd_zex * sizeof(double complex) ); // 2*T_13*dV/dx + 2*T_23*dV/dy
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) + sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;
        phase_fac_m2 = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) + sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
        phase_fac_m3 = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) + sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty, pshifty, DMnx, pshiftz, DMnxny,
                                 FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_out, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                                 pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, FDn, DMnx_in, FDn, DMny_in, pSPARC->Nx, pSPARC->Ny);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty, DMnx, pshiftz, pshiftz, DMnxny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, FDn, FDn, pSPARC->order, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  0, 0, pSPARC->order, phase_fac_m3, FDn, DMnz_in, pSPARC->Nz);
            }
        }
    } else if(pSPARC->cell_typ == 16){
        Dx1 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // 2*T_12*dV/dx + 2*T_23*dV/dz
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) + sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;
        phase_fac_m2 = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) + sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
        phase_fac_m3 = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) + sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz, pshifty, DMnx, pshiftz, DMnxnyex,
                                 FDn, DMnx_in, FDn, DMny_out, FDn, DMnz_in, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                                 pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMnx_in, FDn, DMnz_in, pSPARC->Nx, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty, DMnx, pshiftz, pshiftz, DMnxnyex,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, FDn, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  0, pSPARC->order, 0, phase_fac_m3, FDn, DMny_in, pSPARC->Ny);
            }
        }
    } else if(pSPARC->cell_typ == 17){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // 2*T_12*df/dy + 2*T_13*df/dz
        if(Dx1 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        Dx2 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // df/dz
        if(Dx2 == NULL){
            printf("\nMemory allocation failed in Laplacian vector multiplication!\n");
            exit(EXIT_FAILURE);
        }

        phase_fac_m1 = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) + sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
        phase_fac_m2 = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) + sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
        phase_fac_m3 = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) + sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty, pshiftz, pshifty, DMnx_ex, pshiftz, DMnxexny,
                                 FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                                 0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMny_in, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);

                Calc_DX_kpt(x+n*DMnd, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz, pshifty, DMnx, pshiftz, DMnxnyex,
                            FDn, DMnx_in, FDn, DMny_out, FDn, DMnz_in, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m2, FDn, DMnz_in, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(pSPARC, x+n*DMnd, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty, DMnx_ex, DMnx,
                                  pshiftz, pshiftz, DMnxexny, DMnxnyex,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, pSPARC->order, FDn, FDn, FDn, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, FDn, DMnx_in, FDn, DMny_in, pSPARC->Nx, pSPARC->Ny);
            }
        }
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
        if (overlap_flag) {
            //  df/dy
            // (0:DMnx_ex, 0:DMny, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex,DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, FDn, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, 0, DMny, pSPARC->Ny);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, DMnz_in, DMnz, 0, FDn, DMnz, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, 0, DMny, pSPARC->Ny);
            }

            // (0:DMnx_ex, [0:FDn,DMny_in:DMny], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, FDn, FDn, DMnz_in, 0, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, 0, FDn, pSPARC->Ny);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, DMny_in, DMny, FDn, DMnz_in, 0, DMny, pSPARC->order, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, DMny_in, DMny, pSPARC->Ny);
            }

            // ([0:FDn,DMnx_out:DMnx_ex], FDn:DMny_in, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, FDn, FDn, DMny_in, FDn, DMnz_in, 0, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, FDn, DMny_in, pSPARC->Ny);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            DMnx_out, DMnx_ex, FDn, DMny_in, FDn, DMnz_in, DMnx_out, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, FDn, DMny_in, pSPARC->Ny);
            }

            // Laplacian
            // first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, FDn, 0, DMnz_in, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, FDn, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, FDn, DMny_in, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, FDn, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, FDn, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, DMnx_in, DMnx, pSPARC->Nx);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0,
                            0, pSPARC->order, 0, phase_fac_m1, 0, DMny, pSPARC->Ny);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);
            }
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 12){
        // calculate Lx
        if (overlap_flag) {
            //  df/dz
            // (0:DMnx_ex, 0:DMny, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex,DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, FDn, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, 0, FDn, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, DMnz_in, DMnz, 0, FDn, DMnz, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, DMnz_in, DMnz, pSPARC->Nz);
            }

            // (0:DMnx_ex, [0:FDn,DMny_in:DMny], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, FDn, FDn, DMnz_in, 0, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, DMny_in, DMny, FDn, DMnz_in, 0, DMny, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);
            }

            // ([0:FDn,DMnx_out:DMnx_ex], FDn:DMny_in, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, FDn, FDn, DMny_in, FDn, DMnz_in, 0, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            DMnx_out, DMnx_ex, FDn, DMny_in, FDn, DMnz_in, DMnx_out, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);
            }

            // Laplacian
            // first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, FDn, 0, DMnz_in, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, FDn, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, FDn, DMny_in, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, FDn, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, FDn, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, DMnx_in, DMnx, pSPARC->Nx);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, 0, DMnz, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, phase_fac_m2, 0, DMnx, pSPARC->Nx);
            }
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 13){
        // calculate Lx
        if (overlap_flag) {
            //  df/dz
            // (0:DMnx, 0:DMny_ex, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny_ex, 0, FDn, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, 0, FDn, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny_ex, DMnz_in, DMnz, FDn, 0, DMnz, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m1, DMnz_in, DMnz, pSPARC->Nz);
            }

            // (0:DMnx, [0:FDn,DMny_out:DMny_ex], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, FDn, FDn, DMnz_in, FDn, 0, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, DMny_out, DMny_ex, FDn, DMnz_in, FDn, DMny_out, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);
            }

            // ([0:FDn,DMnx_in:DMnx], FDn:DMny_out, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, FDn, FDn, DMny_out, FDn, DMnz_in, FDn, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        DMnx_in, DMnx, FDn, DMny_out, FDn, DMnz_in, DMnx, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m1, FDn, DMnz_in, pSPARC->Nz);
            }

            // Laplacian
            // first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, 0, DMny, pSPARC->Ny);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, 0, FDn, DMnz_in, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, 0, DMny, pSPARC->Ny);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, 0, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, 0, FDn, pSPARC->Ny);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, 0, DMny, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, DMny_in, DMny, pSPARC->Ny);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, 0, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, FDn, DMny_in, pSPARC->Ny);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx_in, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, FDn, DMny_in, pSPARC->Ny);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m1, 0, DMnz, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m2, 0, DMny, pSPARC->Ny);
            }
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 14){
        if (overlap_flag) {
            // 2*T_12*df/dy + 2*T_13*df/dz
            // (0:DMnx_ex, 0:DMny, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, DMny, 0, FDn, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMny, 0, FDn, pSPARC->Ny, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, DMny, DMnz_in, DMnz, 0, FDn, DMnz, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMny, DMnz_in, DMnz, pSPARC->Ny, pSPARC->Nz);
            }

            // (0:DMnx_ex, [0:FDn,DMny_in:DMny], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, FDn, FDn, DMnz_in, 0, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, FDn, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, DMny_in, DMny, FDn, DMnz_in, 0, DMny, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, DMny_in, DMny, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
            }

            // ([0:FDn,DMnx_out:DMnx_ex], FDn:DMny_in, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, FDn, FDn, DMny_in, FDn, DMnz_in, 0, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMny_in, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             DMnx_out, DMnx_ex, FDn, DMny_in, FDn, DMnz_in, DMnx_out, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMny_in, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
            }

            // Laplacian
            // first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, 0, DMnx, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, FDn, 0, DMnz_in, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, 0, DMnx, pSPARC->Nx);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, FDn, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, 0, DMnx, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, FDn, DMny_in, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, 0, DMnx, pSPARC->Nx);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, FDn, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, 0, FDn, pSPARC->Nx);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, DMnx_in, DMnx, pSPARC->Nx);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMny, 0, DMnz, pSPARC->Ny, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices, DMVertices[0], FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              pSPARC->order, 0, 0, phase_fac_m3, 0, DMnx, pSPARC->Nx);
            }
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 15){
        if (overlap_flag) {
            // 2*T_13*dV/dx + 2*T_23*dV/dy
            // (0:DMnx, 0:DMny, [0:FDn,DMnz_out:DMnz_ex])
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, DMnx, 0, DMny, 0, FDn, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, 0, DMnx, 0, DMny, pSPARC->Nx, pSPARC->Ny);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, DMnx, 0, DMny, DMnz_out, DMnz_ex, FDn, FDn, DMnz_out, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, 0, DMnx, 0, DMny, pSPARC->Nx, pSPARC->Ny);
            }

            // (0:DMnx, [0:FDn,DMny_in:DMny], FDn:DMnz_out)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, DMnx, 0, FDn, FDn, DMnz_out, FDn, FDn, FDn, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, 0, DMnx, 0, FDn, pSPARC->Nx, pSPARC->Ny);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, DMnx, DMny_in, DMny, FDn, DMnz_out, FDn, DMny, FDn, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, 0, DMnx, DMny_in, DMny, pSPARC->Nx, pSPARC->Ny);
            }

            // ([0:FDn,DMnx_in:DMnx], FDn:DMny_in, FDn:DMnz_out)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, FDn, FDn, DMny_in, FDn, DMnz_out, FDn, pSPARC->order, FDn, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, 0, FDn, FDn, DMny_in, pSPARC->Nx, pSPARC->Ny);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_out, DMnx, pSPARC->order, FDn, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, DMnx_in, DMnx, FDn, DMny_in, pSPARC->Nx, pSPARC->Ny);
            }

            // Laplacian
            // first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, 0, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, 0, FDn, pSPARC->Nz);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, 0, 0, DMnz, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, DMnz_in, DMnz, pSPARC->Nz);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, 0, 0, pSPARC->order, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, FDn, DMnz_in, pSPARC->Nz);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, 0, DMny_in, pSPARC->order, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, FDn, DMnz_in, pSPARC->Nz);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, 0, FDn, pSPARC->order, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, FDn, DMnz_in, pSPARC->Nz);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx_in, FDn, pSPARC->order, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, FDn, DMnz_in, pSPARC->Nz);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices[0], DMVertices[2], FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, DMnx, 0, DMny, 0, DMnz_ex, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy,
                             pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m1, phase_fac_m2, 0, DMnx, 0, DMny, pSPARC->Nx, pSPARC->Ny);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, DMVertices, DMVertices[4], FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, 0, pSPARC->order, phase_fac_m3, 0, DMnz, pSPARC->Nz);
            }
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 16){
        if (overlap_flag) {
            // 2*T_12*dV/dx + 2*T_23*dV/dz
            //(0:DMnx, 0:DMny_ex, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, DMnx, 0, DMny_ex, 0, FDn, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMnx, 0, FDn, pSPARC->Nx, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, DMnx, 0, DMny_ex, DMnz_in, DMnz, FDn, 0, DMnz, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMnx, DMnz_in, DMnz, pSPARC->Nx, pSPARC->Nz);
            }

            // (0:DMnx, [0:FDn,DMny_out:DMny_ex], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, DMnx, 0, FDn, FDn, DMnz_in, FDn, 0, pSPARC->order, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMnx, FDn, DMnz_in, pSPARC->Nx, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, DMnx, DMny_out, DMny_ex, FDn, DMnz_in, FDn, DMny_out, pSPARC->order, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMnx, FDn, DMnz_in, pSPARC->Nx, pSPARC->Nz);
            }

            // ([0:FDn,DMnx_in:DMnx], FDn:DMny_out, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, FDn, FDn, DMny_out, FDn, DMnz_in, FDn, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, FDn, FDn, DMnz_in, pSPARC->Nx, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             DMnx_in, DMnx, FDn, DMny_out, FDn, DMnz_in, DMnx, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, DMnx_in, DMnx, FDn, DMnz_in, pSPARC->Nx, pSPARC->Nz);
            }

            // Laplacian
            //first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, 0, DMny, pSPARC->Ny);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, 0, FDn, DMnz_in, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, 0, DMny, pSPARC->Ny);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, 0, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, 0, FDn, pSPARC->Ny);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, 0, DMny, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, DMny_in, DMny, pSPARC->Ny);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, 0, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, FDn, DMny_in, pSPARC->Ny);

                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx_in, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, FDn, DMny_in, pSPARC->Ny);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices[0], DMVertices[4], FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz,
                             pSPARC->order, 0, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMnx, 0, DMnz, pSPARC->Nx, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, DMVertices, DMVertices[2], FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                              0, pSPARC->order, 0, phase_fac_m3, 0, DMny, pSPARC->Ny);
            }
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 17){
        if (overlap_flag) {
            // 2*T_12*df/dy + 2*T_13*df/dz
            // (0:DMnx_ex, 0:DMny, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, DMny, 0, FDn, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMny, 0, FDn, pSPARC->Ny, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, DMny, DMnz_in, DMnz, 0, FDn, DMnz, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMny, DMnz_in, DMnz, pSPARC->Ny, pSPARC->Nz);
            }

            // (0:DMnx_ex, [0:FDn,DMny_in:DMny], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, FDn, FDn, DMnz_in, 0, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, FDn, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, DMny_in, DMny, FDn, DMnz_in, 0, DMny, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, DMny_in, DMny, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
            }

            // ([0:FDn,DMnx_out:DMnx_ex], FDn:DMny_in, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, FDn, FDn, DMny_in, FDn, DMnz_in, 0, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMny_in, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             DMnx_out, DMnx_ex, FDn, DMny_in, FDn, DMnz_in, DMnx_out, pSPARC->order, pSPARC->order, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                             0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, FDn, DMny_in, FDn, DMnz_in, pSPARC->Ny, pSPARC->Nz);
            }

            //  df/dz
            // (0:DMnx, 0:DMny_ex, [0:FDn,DMnz_in:DMnz])
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, DMny_ex, 0, FDn, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m2, 0, FDn, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, DMny_ex, DMnz_in, DMnz, FDn, 0, DMnz, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m2, DMnz_in, DMnz, pSPARC->Nz);
            }

            // (0:DMnx, [0:FDn,DMny_out:DMny_ex], FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, FDn, FDn, DMnz_in, FDn, 0, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m2, FDn, DMnz_in, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, DMny_out, DMny_ex, FDn, DMnz_in, FDn, DMny_out, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m2, FDn, DMnz_in, pSPARC->Nz);
            }

            // ([0:FDn,DMnx_in:DMnx], FDn:DMny_out, FDn:DMnz_in)
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, FDn, FDn, DMny_out, FDn, DMnz_in, FDn, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m2, FDn, DMnz_in, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        DMnx_in, DMnx, FDn, DMny_out, FDn, DMnz_in, DMnx, FDn, pSPARC->order, pSPARC->D1_stencil_coeffs_z, 0.0,
                        0, 0, pSPARC->order, phase_fac_m2, FDn, DMnz_in, pSPARC->Nz);
            }

            // Laplacian
            // first calculate Lx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])
            for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  0, DMnx, 0, DMny, 0, FDn, FDn, FDn, FDn, FDn, 0, 0, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, 0, DMnx, 0, DMny, pSPARC->Nx, pSPARC->Ny);

                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  0, DMnx, 0, DMny, DMnz_in, DMnz, FDn, FDn, DMnz, FDn, 0, DMnz_in, 0, FDn, DMnz_in, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, 0, DMnx, 0, DMny, pSPARC->Nx, pSPARC->Ny);
            }

            // then calculate Lx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  0, DMnx, 0, FDn, FDn, DMnz_in, FDn, FDn, pSPARC->order, FDn, 0, FDn, 0, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, 0, DMnx, 0, FDn, pSPARC->Nx, pSPARC->Ny);

                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  0, DMnx, DMny_in, DMny, FDn, DMnz_in, FDn, DMny, pSPARC->order, FDn, DMny_in, FDn, 0, DMny, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, 0, DMnx, DMny_in, DMny, pSPARC->Nx, pSPARC->Ny);
            }

            // finally calculate Lx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
            for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  0, FDn, FDn, DMny_in, FDn, DMnz_in, FDn, pSPARC->order, pSPARC->order, FDn, FDn, FDn, 0, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, 0, FDn, FDn, DMny_in, pSPARC->Nx, pSPARC->Ny);

                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx, pSPARC->order, pSPARC->order, DMnx, FDn, FDn, DMnx_in, pSPARC->order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, DMnx_in, DMnx, FDn, DMny_in, pSPARC->Nx, pSPARC->Ny);
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, DMVertices[2], DMVertices[4], FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                                 0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz,
                                 0, pSPARC->order, 0, 0, 0, pSPARC->order, phase_fac_m1, phase_fac_m2, 0, DMny, 0, DMnz, pSPARC->Ny, pSPARC->Nz);

                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, DMVertices[4], FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                            0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0,
                            0, 0, pSPARC->order, phase_fac_m2, 0, DMnz, pSPARC->Nz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(pSPARC, x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, DMVertices, DMVertices[0], DMVertices[2], FDn, 1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx,
                                  pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                                  0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd, kpt,
                                  pSPARC->order, 0, 0, 0, pSPARC->order, 0, phase_fac_m3, phase_fac_m1, 0, DMnx, 0, DMny, pSPARC->Nx, pSPARC->Ny);
            }
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
    const int m1_DMVertex,   const int m2_DMVertex,
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
    const double *stencil_coefs2,
    const int shift1_m1,     const int shift2_m1,
    const int shift3_m1,     const int shift1_m2,
    const int shift2_m2,     const int shift3_m2,
    const double complex phase_fac_r_m1,
    const double complex phase_fac_r_m2,
    const int m1_spos,       const int m1_epos,
    const int m2_spos,       const int m2_epos,
    const int l_m1,          const int l_m2
)
{
    int i, j, k, ii, jj, kk, r;

    int sz_m1 = 2 * (m1_epos - m1_spos) * radius;
    int sz_m2 = 2 * (m2_epos - m2_spos) * radius;
    double complex phase_fac_l_m1 = conj(phase_fac_r_m1);
    double complex phase_fac_l_m2 = conj(phase_fac_r_m2);
    double complex *Phase_fac = (double complex *) malloc((sz_m1 + sz_m2) * sizeof(double complex));
    int count, global_i, init1_m1, init2_m1, init3_m1, init1_m2, init2_m2, init3_m2, count_m1, count_m2;

    count = 0;
    for(i = m1_spos; i < m1_epos; i++){
        global_i = i + m1_DMVertex;
        for (r = 1; r <= radius; r++){
            if((global_i + r) >= l_m1)
                Phase_fac[count] = phase_fac_r_m1;
            else
                Phase_fac[count] = 1.0;
            count++;
            
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_m1;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    for(i = m2_spos; i < m2_epos; i++){
        global_i = i + m2_DMVertex;
        for (r = 1; r <= radius; r++){
         if((global_i + r) >= l_m2)
                Phase_fac[count] = phase_fac_r_m2;
            else
                Phase_fac[count] = 1.0;
            count++;
            
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_m2;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }
    int shift = sz_m1;
    init3_m1 = init3_m2 = 0;
    for (k = z_DX_spos, kk = z_X_spos; k < z_DX_epos; k++, kk++)
    {
        int kshift_DX = k * stride_z_DX;
        int kshift_X = kk * stride_z_X;
        init2_m1 = init2_m2 = 0;
        for (j = y_DX_spos, jj = y_X_spos; j < y_DX_epos; j++, jj++)
        {
            int jshift_DX = kshift_DX + j * stride_y_DX;
            int jshift_X = kshift_X + jj * stride_y_X;
            init1_m1 = init1_m2 = 0;
            const int niters = x_DX_epos - x_DX_spos;
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_DX = jshift_DX + i + x_DX_spos;
                int ishift_X = jshift_X + i + x_X_spos;
                double complex temp1 = 0.0;
                double complex temp2 = 0.0;
                count_m1 = init1_m1 + init2_m1 + init3_m1;
                count_m2 = init1_m2 + init2_m2 + init3_m2;
                for (r = 1; r <= radius; r++)
                {
                    int stride_X_1_r = r * stride_X_1;
                    int stride_X_2_r = r * stride_X_2;
                    temp1 += (X[ishift_X + stride_X_1_r] * Phase_fac[count_m1]         - X[ishift_X - stride_X_1_r] * Phase_fac[count_m1 + 1])         * stencil_coefs1[r];
                    temp2 += (X[ishift_X + stride_X_2_r] * Phase_fac[count_m2 + shift] - X[ishift_X - stride_X_2_r] * Phase_fac[count_m2 + shift + 1]) * stencil_coefs2[r];
                    count_m1 += 2; count_m2 += 2;
                }
                DX[ishift_DX] = temp1 + temp2;
                init1_m1 += shift1_m1;
                init1_m2 += shift1_m2;
            }
            init2_m1 += shift2_m1;
            init2_m2 += shift2_m2;
        }
        init3_m1 += shift3_m1;
        init3_m2 += shift3_m2;
    }

    free(Phase_fac);
}




/*
 * @brief: function to perform 4 component stencil operation
 */
void stencil_4comp_kpt(
    const SPARC_OBJ *pSPARC,   const double complex *X,
    const double complex *DX,
    const int *DMVertices,     const int m_DMVertex,
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
    const double *v0,          double complex *X1,
    const int kpt,             const int shift1,
    const int shift2,          const int shift3,
    const double complex phase_fac_r_m,
    const int m_spos,          const int m_epos,
    const int l_m
)
{
    int i, j, k, ii, jj, kk, iii, jjj, kkk, r;

    double complex phase_fac_l_x = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) - sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;
    double complex phase_fac_l_y = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) - sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
    double complex phase_fac_l_z = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) - sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
    double complex phase_fac_r_x = conj(phase_fac_l_x);
    double complex phase_fac_r_y = conj(phase_fac_l_y);
    double complex phase_fac_r_z = conj(phase_fac_l_z);
    double complex phase_fac_l_m = conj(phase_fac_r_m);

    int global_i, global_j, global_k;
    int nel_x = (x_X1_epos - x_X1_spos);
    int nel_y = (y_X1_epos - y_X1_spos);
    int nel_z = (z_X1_epos - z_X1_spos);
    int sz_x = 2 * nel_x * radius;
    int sz_y = 2 * nel_y * radius;
    int sz_z = 2 * nel_z * radius;
    int sz_m = 2 * (m_epos - m_spos) * radius;
    double complex *Phase_fac = (double complex *) malloc((sz_x + sz_y + sz_z + sz_m) * sizeof(double complex));

    int count;
    
    count = 0;
    for(i = x_X1_spos; i < x_X1_epos; i++){
        global_i = i + DMVertices[0];
        for (r = 1; r <= radius; r++){
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_x;
            else
                Phase_fac[count] = 1.0;
            count++;

            if((global_i + r) >= pSPARC->Nx)
                Phase_fac[count] = phase_fac_r_x;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    //count = 0;
    for(j = y_X1_spos; j < y_X1_epos; j++){
        global_j = j + DMVertices[2];
        for (r = 1; r <= radius; r++){
            if((global_j - r) < 0)
                Phase_fac[count] = phase_fac_l_y;
            else
                Phase_fac[count] = 1.0;
            count++;

            if((global_j + r) >= pSPARC->Ny)
                Phase_fac[count] = phase_fac_r_y;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    //count = 0;
    for(k = z_X1_spos; k < z_X1_epos; k++){
        global_k = k + DMVertices[4];
        for (r = 1; r <= radius; r++){
            if((global_k - r) < 0)
                Phase_fac[count] = phase_fac_l_z;
            else
                Phase_fac[count] = 1.0;
            count++;

            if((global_k + r) >= pSPARC->Nz)
                Phase_fac[count] = phase_fac_r_z;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    for(i = m_spos; i < m_epos; i++){
        global_i = i + m_DMVertex;
        for (r = 1; r <= radius; r++){
            if((global_i + r) >= l_m)
                Phase_fac[count] = phase_fac_r_m;
            else
                Phase_fac[count] = 1.0;
            count++;
        
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_m;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    int count2, count3, init1, init2, init3;
    int countx, county, countz, countm;
    int shift_y = sz_x;
    int shift_z = shift_y + sz_y;
    int shift_m = shift_z + sz_z;
    count3 = init3 = 0;
    for (k = z_X1_spos, kk = z_X_spos, kkk = z_DX_spos; k < z_X1_epos; k++, kk++, kkk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        int kshift_DX = kkk * stride_z_DX;
        count2 = init2 = 0;
        for (j = y_X1_spos, jj = y_X_spos, jjj = y_DX_spos; j < y_X1_epos; j++, jj++, jjj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            int jshift_DX = kshift_DX + jjj * stride_y_DX;
            countx = init1 = 0;
            const int niters = x_X1_epos - x_X1_spos;
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_X1    = jshift_X1 + i + x_X1_spos;
                int ishift_X     = jshift_X  + i + x_X_spos;
                int ishift_DX    = jshift_DX + i + x_DX_spos;
                county = count2;
                countz = count3;
                countm = init1 + init2 + init3;
                double complex res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX_r = r * stride_DX;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 4 * r + 1;
                    double complex res_x = (X[ishift_X - r]             * Phase_fac[countx]           + X[ishift_X + r]             * Phase_fac[countx + 1])           * stencil_coefs[r_fac];
                    double complex res_y = (X[ishift_X - stride_y_r]    * Phase_fac[county + shift_y] + X[ishift_X + stride_y_r]    * Phase_fac[county + shift_y + 1]) * stencil_coefs[r_fac+1];
                    double complex res_z = (X[ishift_X - stride_z_r]    * Phase_fac[countz + shift_z] + X[ishift_X + stride_z_r]    * Phase_fac[countz + shift_z + 1]) * stencil_coefs[r_fac+2];
                    double complex res_m = (DX[ishift_DX + stride_DX_r] * Phase_fac[countm + shift_m] - DX[ishift_DX - stride_DX_r] * Phase_fac[countm + shift_m + 1]) * stencil_coefs[r_fac+3];
                    res += res_x + res_y + res_z + res_m;
                    countx += 2; county += 2; countz += 2; countm += 2;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
                init1 += shift1;
            }
            count2 += 2*radius;
            init2 += shift2;
        }
        count3 += 2*radius;
        init3 += shift3;
    }

    free(Phase_fac);
}


/*
 @ brief: function to perform 5 component stencil operation
*/

void stencil_5comp_kpt(
    const SPARC_OBJ *pSPARC,   const double complex *X,
    const double complex *DX1, const double complex *DX2,
    const int *DMVertices,     const int m1_DMVertex,
    const int m2_DMVertex,     const int radius,
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
    const int z_DX2_spos,
    const double *stencil_coefs,
    const double coef_0,       const double b,
    const double *v0,          double complex *X1,
    const int kpt,             const int shift1_m1,
    const int shift2_m1,       const int shift3_m1,
    const int shift1_m2,       const int shift2_m2,
    const int shift3_m2,
    const double complex phase_fac_r_m1,
    const double complex phase_fac_r_m2,
    const int m1_spos,         const int m1_epos,
    const int m2_spos,         const int m2_epos,
    const int l_m1,            const int l_m2

)
{
    int i, j, k, ii, jj, kk, iii, jjj, kkk, iiii, jjjj, kkkk, r;

    double complex phase_fac_l_x = cos(pSPARC->k1_loc[kpt] * pSPARC->range_x) - sin(pSPARC->k1_loc[kpt] * pSPARC->range_x) * I;
    double complex phase_fac_l_y = cos(pSPARC->k2_loc[kpt] * pSPARC->range_y) - sin(pSPARC->k2_loc[kpt] * pSPARC->range_y) * I;
    double complex phase_fac_l_z = cos(pSPARC->k3_loc[kpt] * pSPARC->range_z) - sin(pSPARC->k3_loc[kpt] * pSPARC->range_z) * I;
    double complex phase_fac_r_x = conj(phase_fac_l_x);
    double complex phase_fac_r_y = conj(phase_fac_l_y);
    double complex phase_fac_r_z = conj(phase_fac_l_z);
    double complex phase_fac_l_m1 = conj(phase_fac_r_m1);
    double complex phase_fac_l_m2 = conj(phase_fac_r_m2);

    int global_i, global_j, global_k;
    int nel_x = (x_X1_epos - x_X1_spos);
    int nel_y = (y_X1_epos - y_X1_spos);
    int nel_z = (z_X1_epos - z_X1_spos);
    int sz_x = 2 * nel_x * radius;
    int sz_y = 2 * nel_y * radius;
    int sz_z = 2 * nel_z * radius;
    int sz_m1 = 2 * (m1_epos - m1_spos) * radius;
    int sz_m2 = 2 * (m2_epos - m2_spos) * radius;
    double complex *Phase_fac = (double complex *) malloc((sz_x + sz_y + sz_z + sz_m1 + sz_m2) * sizeof(double complex));

    int count;
    
    count = 0;
    for(i = x_X1_spos; i < x_X1_epos; i++){
        global_i = i + DMVertices[0];
        for (r = 1; r <= radius; r++){
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_x;
            else
                Phase_fac[count] = 1.0;
            count++;

            if((global_i + r) >= pSPARC->Nx)
                Phase_fac[count] = phase_fac_r_x;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    //count = 0;
    for(j = y_X1_spos; j < y_X1_epos; j++){
        global_j = j + DMVertices[2];
        for (r = 1; r <= radius; r++){
            if((global_j - r) < 0)
                Phase_fac[count] = phase_fac_l_y;
            else
                Phase_fac[count] = 1.0;
            count++;

            if((global_j + r) >= pSPARC->Ny)
                Phase_fac[count] = phase_fac_r_y;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    //count = 0;
    for(k = z_X1_spos; k < z_X1_epos; k++){
        global_k = k + DMVertices[4];
        for (r = 1; r <= radius; r++){
            if((global_k - r) < 0)
                Phase_fac[count] = phase_fac_l_z;
            else
                Phase_fac[count] = 1.0;
            count++;

            if((global_k + r) >= pSPARC->Nz)
                Phase_fac[count] = phase_fac_r_z;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    for(i = m1_spos; i < m1_epos; i++){
        global_i = i + m1_DMVertex;
        for (r = 1; r <= radius; r++){
            if((global_i + r) >= l_m1)
                Phase_fac[count] = phase_fac_r_m1;
            else
                Phase_fac[count] = 1.0;
            count++;
            
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_m1;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    for(i = m2_spos; i < m2_epos; i++){
        global_i = i + m2_DMVertex;
        for (r = 1; r <= radius; r++){
            if((global_i + r) >= l_m2)
                Phase_fac[count] = phase_fac_r_m2;
            else
                Phase_fac[count] = 1.0;
            count++;
            
            if((global_i - r) < 0)
                Phase_fac[count] = phase_fac_l_m2;
            else
                Phase_fac[count] = 1.0;
            count++;
        }
    }

    int count2, count3, init1_m1, init2_m1, init3_m1, init1_m2, init2_m2, init3_m2;
    int countx, county, countz, countm1, countm2;
    int shift_y = sz_x;
    int shift_z = shift_y + sz_y;
    int shift_m1 = shift_z + sz_z;
    int shift_m2 = shift_m1 + sz_m1;

    count3 = init3_m1 = init3_m2 = 0;
    for (k = z_X1_spos, kk = z_X_spos, kkk = z_DX1_spos, kkkk = z_DX2_spos; k < z_X1_epos; k++, kk++, kkk++, kkkk++)
    {
        int kshift_X1 = k * stride_z_X1;
        int kshift_X  = kk * stride_z_X;
        int kshift_DX1 = kkk * stride_z_DX1;
        int kshift_DX2 = kkkk * stride_z_DX2;
        count2 = init2_m1 = init2_m2 = 0;
        for (j = y_X1_spos, jj = y_X_spos, jjj = y_DX1_spos, jjjj = y_DX2_spos; j < y_X1_epos; j++, jj++, jjj++, jjjj++)
        {
            int jshift_X1 = kshift_X1 + j * stride_y_X1;
            int jshift_X  = kshift_X + jj * stride_y_X;
            int jshift_DX1 = kshift_DX1 + jjj * stride_y_DX1;
            int jshift_DX2 = kshift_DX2 + jjjj * stride_y_DX2;
            countx = init1_m1 = init1_m2 = 0;
            const int niters = x_X1_epos - x_X1_spos;
            #pragma omp simd
            for (i = 0; i < niters; i++)
            {
                int ishift_X1  = jshift_X1  + i + x_X1_spos;
                int ishift_X   = jshift_X   + i + x_X_spos;
                int ishift_DX1 = jshift_DX1 + i + x_DX1_spos;
                int ishift_DX2 = jshift_DX2 + i + x_DX2_spos;
                county = count2;
                countz = count3;
                countm1 = init1_m1 + init2_m1 + init3_m1;
                countm2 = init1_m2 + init2_m2 + init3_m2;
                double complex res = coef_0 * X[ishift_X];
                for (r = 1; r <= radius; r++)
                {
                    int stride_DX1_r = r * stride_DX1;
                    int stride_DX2_r = r * stride_DX2;
                    int stride_y_r = r * stride_y_X;
                    int stride_z_r = r * stride_z_X;
                    int r_fac = 5 * r;
                    double complex res_x  = (X[ishift_X - r]                * Phase_fac[countx]             + X[ishift_X + r]               * Phase_fac[countx + 1])             * stencil_coefs[r_fac];
                    double complex res_y  = (X[ishift_X - stride_y_r]       * Phase_fac[county + shift_y]   + X[ishift_X + stride_y_r]      * Phase_fac[county + shift_y + 1])   * stencil_coefs[r_fac+1];
                    double complex res_z  = (X[ishift_X - stride_z_r]       * Phase_fac[countz + shift_z]   + X[ishift_X + stride_z_r]      * Phase_fac[countz + shift_z + 1])   * stencil_coefs[r_fac+2];
                    double complex res_m1 = (DX1[ishift_DX1 + stride_DX1_r] * Phase_fac[countm1 + shift_m1] - DX1[ishift_DX1 - stride_DX1_r] * Phase_fac[countm1 + shift_m1 + 1]) * stencil_coefs[r_fac+3];
                    double complex res_m2 = (DX2[ishift_DX2 + stride_DX2_r] * Phase_fac[countm2 + shift_m2] - DX2[ishift_DX2 - stride_DX2_r] * Phase_fac[countm2 + shift_m2 + 1]) * stencil_coefs[r_fac+4];
                    res += res_x + res_y + res_z + res_m1 + res_m2;
                    countx += 2; county += 2; countz += 2; countm1 += 2; countm2 += 2;
                }
                X1[ishift_X1] = (res + b * (v0[ishift_X1] * X[ishift_X]));
                init1_m1 += shift1_m1;
                init1_m2 += shift1_m2;
            }
            count2 += 2*radius;
            init2_m1 += shift2_m1;
            init2_m2 += shift2_m2;
        }
        count3 += 2*radius;
        init3_m1 += shift3_m1;
        init3_m2 += shift3_m2;
    }
    free(Phase_fac);
}
