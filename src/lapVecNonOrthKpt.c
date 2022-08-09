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
#include "tools.h"
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

    int overlap_flag = (int) (nproc > 1 && DMnx > order
                          && DMny > order && DMnz > order);

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

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, pshifty, pshifty, DMnx_ex, pshiftz, DMnxexny,
                            FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty, DMnx_ex, pshiftz, pshiftz, DMnxexny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, order, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);

            }
        }
    } else if(pSPARC->cell_typ == 12){
        Dx1 = (double complex*) malloc(ncol * DMnd_xex * sizeof(double complex) ); // df/dz
        assert(Dx1 != NULL);

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, pshiftz, pshifty, DMnx_ex, pshiftz, DMnxexny,
                            FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty, DMnx_ex, pshiftz, pshiftz, DMnxexny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, order, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
    } else if(pSPARC->cell_typ == 13){
        Dx1 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // df/dz
        assert(Dx1 != NULL);

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x+n*DMnd, Dx1+n*DMnd_yex, FDn, pshiftz, pshifty, DMnx, pshiftz, DMnxnyex,
                            FDn, DMnx_in, FDn, DMny_out, FDn, DMnz_in, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x+n*DMnd, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty, DMnx, pshiftz, pshiftz, DMnxnyex,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, FDn, order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
    } else if(pSPARC->cell_typ == 14){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // 2*T_12*df/dy + 2*T_13*df/dz
        assert(Dx1 != NULL);

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, pshifty, pshiftz, pshifty, DMnx_ex, pshiftz, DMnxexny,
                                 FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty, DMnx_ex, pshiftz, pshiftz, DMnxexny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, order, FDn, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
    } else if(pSPARC->cell_typ == 15){
        Dx1 = (double complex *) malloc(ncol * DMnd_zex * sizeof(double complex) ); // 2*T_13*dV/dx + 2*T_23*dV/dy
        assert(Dx1 != NULL);

        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_zex, FDn, 1, pshifty, pshifty, DMnx, pshiftz, DMnxny,
                                 FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_out, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x+n*DMnd, Dx1+n*DMnd_zex, FDn, DMnxny, pshifty, pshifty, DMnx, pshiftz, pshiftz, DMnxny,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, FDn, FDn, order, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
    } else if(pSPARC->cell_typ == 16){
        Dx1 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // 2*T_12*dV/dx + 2*T_23*dV/dz
        assert(Dx1 != NULL);
        
        if(overlap_flag){
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_yex, FDn, 1, pshiftz, pshifty, DMnx, pshiftz, DMnxnyex,
                                 FDn, DMnx_in, FDn, DMny_out, FDn, DMnz_in, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x+n*DMnd, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty, DMnx, pshiftz, pshiftz, DMnxnyex,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, FDn, order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
    } else if(pSPARC->cell_typ == 17){
        Dx1 = (double complex *) malloc(ncol * DMnd_xex * sizeof(double complex) ); // 2*T_12*df/dy + 2*T_13*df/dz
        Dx2 = (double complex *) malloc(ncol * DMnd_yex * sizeof(double complex) ); // df/dz
        assert(Dx1 != NULL && Dx2 != NULL);
        if (overlap_flag) {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x+n*DMnd, Dx1+n*DMnd_xex, FDn, pshifty, pshiftz, pshifty, DMnx_ex, pshiftz, DMnxexny,
                                 FDn, DMnx_out, FDn, DMny_in, FDn, DMnz_in, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);

                Calc_DX_kpt(x+n*DMnd, Dx2+n*DMnd_yex, FDn, pshiftz, pshifty, DMnx, pshiftz, DMnxnyex,
                            FDn, DMnx_in, FDn, DMny_out, FDn, DMnz_in, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(x+n*DMnd, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, FDn, 1, DMnx, pshifty, pshifty, DMnx_ex, DMnx,
                                  pshiftz, pshiftz, DMnxexny, DMnxnyex,
                                  FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn, order, FDn, FDn, FDn, order, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
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

    // positions of sub-blocks when overlap_flag is on
    const int x_Y_spos[6] = {0,	0,	0,	0,	0,	DMnx_in};
    const int x_Y_epos[6] = {DMnx,	DMnx,	DMnx,	DMnx,	FDn,	DMnx};
    const int y_Y_spos[6] = {0,	0,	0,	DMny_in,	FDn,	FDn};
    const int y_Y_epos[6] = {DMny,	DMny,	FDn,	DMny,	DMny_in,	DMny_in};
    const int z_Y_spos[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
    const int z_Y_epos[6] = {FDn,	DMnz,	DMnz_in,	DMnz_in,	DMnz_in,	DMnz_in};
    const int x_x_ex_spos_lap[6] = {FDn,	FDn,	FDn,	FDn,	FDn,	DMnx};
    const int y_x_ex_spos_lap[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
    const int z_x_ex_spos_lap[6] = {FDn,	DMnz,	order,	order,	order,	order};

    if(pSPARC->cell_typ == 11){
        // calculate Lx
        if (overlap_flag) {
            // df/dy
            const int x_Dx1_spos[6] = {0,	0,	0,	0,	0,	DMnx_out};
            const int x_Dx1_epos[6] = {DMnx_ex,	DMnx_ex,	DMnx_ex,	DMnx_ex,	FDn,	DMnx_ex};
            const int y_Dx1_spos[6] = {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int y_Dx1_epos[6] = {DMny,	DMny,	FDn,	DMny,	DMny_in,	DMny_in};
            const int z_Dx1_spos[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            const int z_Dx1_epos[6] = {FDn,	DMnz,	DMnz_in,	DMnz_in,	DMnz_in,	DMnz_in};
            const int x_x_ex_spos[6] = {0,	0,	0,	0,	0,	DMnx_out};
            const int y_x_ex_spos[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_x_ex_spos[6] = {FDn,	DMnz,	order,	order,	order,	order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        x_Dx1_spos[blk], x_Dx1_epos[blk], y_Dx1_spos[blk], y_Dx1_epos[blk], z_Dx1_spos[blk], z_Dx1_epos[blk],
                        x_x_ex_spos[blk], y_x_ex_spos[blk], z_x_ex_spos[blk],
                        pSPARC->D1_stencil_coeffs_y, 0.0);
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {FDn,	FDn,	FDn,	FDn,	    FDn,	DMnx};
            const int y_Dx1_spos_lap[6] = {0,	0,	    0,	    DMny_in,	FDn,	FDn};
            const int z_Dx1_spos_lap[6] = {0,	DMnz_in,FDn,    FDn,	    FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                        x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                        x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk], x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                        Lap_wt, w2_diag, _b, _v, y+n*DMnd
                    );
                }
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 12){
        // calculate Lx
        if (overlap_flag) {
            //  df/dz
            const int x_Dx1_spos[6] = {0,	0,	0,	0,	0,	DMnx_out};
            const int x_Dx1_epos[6] = {DMnx_ex,	DMnx_ex,	DMnx_ex,	DMnx_ex,	FDn,	DMnx_ex};
            const int y_Dx1_spos[6] = {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int y_Dx1_epos[6] = {DMny,	DMny,	FDn,	DMny,	DMny_in,	DMny_in};
            const int z_Dx1_spos[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            const int z_Dx1_epos[6] = {FDn,	DMnz,	DMnz_in,	DMnz_in,	DMnz_in,	DMnz_in};
            const int x_x_ex_spos[6] = {0,	0,	0,	0,	0,	DMnx_out};
            const int y_x_ex_spos[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_x_ex_spos[6] = {FDn,	DMnz,	order,	order,	order,	order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        x_Dx1_spos[blk], x_Dx1_epos[blk], y_Dx1_spos[blk], y_Dx1_epos[blk], z_Dx1_spos[blk], z_Dx1_epos[blk],
                        x_x_ex_spos[blk], y_x_ex_spos[blk], z_x_ex_spos[blk],
                        pSPARC->D1_stencil_coeffs_z, 0.0);
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {FDn,	FDn,	FDn,	FDn,	    FDn,	DMnx};
            const int y_Dx1_spos_lap[6] = {0,	0,	    0,	    DMny_in,	FDn,	FDn};
            const int z_Dx1_spos_lap[6] = {0,	DMnz_in,FDn,    FDn,	    FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                        x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                        x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk], x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                        Lap_wt, w2_diag, _b, _v, y+n*DMnd
                    );
                }
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                            0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                                  0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 13){
        // calculate Lx
        if (overlap_flag) {
            // df/dz
            const int x_Dx1_spos[6] =  {0,	0,	0,	0,	0,	DMnx_in};
            const int x_Dx1_epos[6] =  {DMnx,	DMnx,	DMnx,	DMnx,	FDn,	DMnx};
            const int y_Dx1_spos[6] =  {0,	0,	0,	DMny_out,	FDn,	FDn};
            const int y_Dx1_epos[6] =  {DMny_ex,	DMny_ex,	FDn,	DMny_ex,	DMny_out,	DMny_out};
            const int z_Dx1_spos[6] =  {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            const int z_Dx1_epos[6] =  {FDn,	DMnz,	DMnz_in,	DMnz_in,	DMnz_in,	DMnz_in};
            const int x_x_ex_spos[6] = {FDn,	FDn,	FDn,	FDn,	FDn,	DMnx};
            const int y_x_ex_spos[6] = {0,	0,	0,	DMny_out,	FDn,	FDn};
            const int z_x_ex_spos[6] = {FDn,	DMnz,	order,	order,	order,	order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        x_Dx1_spos[blk], x_Dx1_epos[blk], y_Dx1_spos[blk], y_Dx1_epos[blk], z_Dx1_spos[blk], z_Dx1_epos[blk],
                        x_x_ex_spos[blk], y_x_ex_spos[blk], z_x_ex_spos[blk],
                        pSPARC->D1_stencil_coeffs_z, 0.0);
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {0,	0,	0,	0,	0,	DMnx_in};
            const int y_Dx1_spos_lap[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_Dx1_spos_lap[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                        x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                        x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk], x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                        Lap_wt, w2_diag, _b, _v, y+n*DMnd
                    );
                }
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }

        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 14){
        if (overlap_flag) {
            // 2*T_12*df/dy + 2*T_13*df/dz
            const int x_Dx1_spos_dydz[6] =  {0,	0,	0,	0,	0,	DMnx_out};
            const int x_Dx1_epos_dydz[6] =  {DMnx_ex,	DMnx_ex,	DMnx_ex,	DMnx_ex,	FDn,	DMnx_ex};
            const int y_Dx1_spos_dydz[6] =  {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int y_Dx1_epos_dydz[6] =  {DMny,	DMny,	FDn,	DMny,	DMny_in,	DMny_in};
            const int z_Dx1_spos_dydz[6] =  {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            const int z_Dx1_epos_dydz[6] =  {FDn,	DMnz,	DMnz_in,	DMnz_in,	DMnz_in,	DMnz_in};
            const int x_x_ex_spos_dydz[6] = {0,	0,	0,	0,	0,	DMnx_out};
            const int y_x_ex_spos_dydz[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_x_ex_spos_dydz[6] = {FDn,	DMnz,	order,	order,	order,	order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        x_Dx1_spos_dydz[blk], x_Dx1_epos_dydz[blk], y_Dx1_spos_dydz[blk], y_Dx1_epos_dydz[blk], z_Dx1_spos_dydz[blk], z_Dx1_epos_dydz[blk],
                        x_x_ex_spos_dydz[blk], y_x_ex_spos_dydz[blk], z_x_ex_spos_dydz[blk],
                        pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz
                    );
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {FDn,	FDn,	FDn,	FDn,	FDn,	DMnx};
            const int y_Dx1_spos_lap[6] = {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int z_Dx1_spos_lap[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                        x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                        x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk], x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                        Lap_wt, w2_diag, _b, _v, y+n*DMnd
                    );
                }
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                             0, DMnx_ex, 0, DMny, 0, DMnz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, 1, pshifty, pshifty_ex, DMnx_ex, pshiftz, pshiftz_ex, DMnxexny,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, FDn, 0, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 15){
        if (overlap_flag) {
            // 2*T_13*dV/dx + 2*T_23*dV/dy
            const int x_Dx1_spos_dxdy[6] =  {0,	0,	0,	0,	0,	DMnx_in};
            const int x_Dx1_epos_dxdy[6] =  {DMnx,	DMnx,	DMnx,	DMnx,	FDn,	DMnx};
            const int y_Dx1_spos_dxdy[6] =  {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int y_Dx1_epos_dxdy[6] =  {DMny,	DMny,	FDn,	DMny,	DMny_in,	DMny_in};
            const int z_Dx1_spos_dxdy[6] =  {0,	DMnz_out,	FDn,	FDn,	FDn,	FDn};
            const int z_Dx1_epos_dxdy[6] =  {FDn,	DMnz_ex,	DMnz_out,	DMnz_out,	DMnz_out,	DMnz_out};
            const int x_x_ex_spos_dxdy[6] = {FDn,	FDn,	FDn,	FDn,	FDn,	DMnx};
            const int y_x_ex_spos_dxdy[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_x_ex_spos_dxdy[6] = {0,	DMnz_out,	FDn,	FDn,	FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                        x_Dx1_spos_dxdy[blk], x_Dx1_epos_dxdy[blk], y_Dx1_spos_dxdy[blk], y_Dx1_epos_dxdy[blk], z_Dx1_spos_dxdy[blk], z_Dx1_epos_dxdy[blk],
                        x_x_ex_spos_dxdy[blk], y_x_ex_spos_dxdy[blk], z_x_ex_spos_dxdy[blk],
                        pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy
                    );
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {0,	0,	0,	0,	0,	DMnx_in};
            const int y_Dx1_spos_lap[6] = {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int z_Dx1_spos_lap[6] = {FDn,	DMnz,	order,	order,	order,	order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                        x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                        x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk], x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                        Lap_wt, w2_diag, _b, _v, y+n*DMnd
                    );
                }
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, 1, pshifty_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxny,
                             0, DMnx, 0, DMny, 0, DMnz_ex, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_zex, FDn, DMnxny, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxny,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, 0, FDn, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 16){
        if (overlap_flag) {
            // 2*T_12*dV/dx + 2*T_23*dV/dz
            const int x_Dx1_spos_dxdz[6] =  {0,	0,	0,	0,	0,	DMnx_in};
            const int x_Dx1_epos_dxdz[6] =  {DMnx,	 DMnx,	 DMnx,	 DMnx,	 FDn,	 DMnx};
            const int y_Dx1_spos_dxdz[6] =  {0,	0,	0,	 DMny_out,	 FDn,	 FDn};
            const int y_Dx1_epos_dxdz[6] =  {DMny_ex,	 DMny_ex,	 FDn,	 DMny_ex,	 DMny_out,	 DMny_out};
            const int z_Dx1_spos_dxdz[6] =  {0,	 DMnz_in,	 FDn,	 FDn,	 FDn,	 FDn};
            const int z_Dx1_epos_dxdz[6] =  {FDn,	 DMnz,	 DMnz_in,	 DMnz_in,	 DMnz_in,	 DMnz_in};
            const int x_x_ex_spos_dxdz[6] = {FDn,	 FDn,	 FDn,	 FDn,	 FDn,	 DMnx};
            const int y_x_ex_spos_dxdz[6] = {0,	0,	0,	 DMny_out,	 FDn,	 FDn};
            const int z_x_ex_spos_dxdz[6] = {FDn,	 DMnz,	 order,	 order,	 order,	 order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        x_Dx1_spos_dxdz[blk], x_Dx1_epos_dxdz[blk], y_Dx1_spos_dxdz[blk], y_Dx1_epos_dxdz[blk], z_Dx1_spos_dxdz[blk], z_Dx1_epos_dxdz[blk],
                        x_x_ex_spos_dxdz[blk], y_x_ex_spos_dxdz[blk], z_x_ex_spos_dxdz[blk],
                        pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz
                    );
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {0,	0,	0,	0,	0,	DMnx_in};
            const int y_Dx1_spos_lap[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_Dx1_spos_lap[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                        x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                        x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk], x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                        Lap_wt, w2_diag, _b, _v, y+n*DMnd
                    );
                }
            }
        } else {
            for (n = 0; n < ncol; n++) {
                Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, 1, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                             0, DMnx, 0, DMny_ex, 0, DMnz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz);
            }

            for (n = 0; n < ncol; n++) {
                stencil_4comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_yex, FDn, DMnx, pshifty, pshifty_ex, DMnx, pshiftz, pshiftz_ex, DMnxnyex,
                              0, DMnx, 0, DMny, 0, DMnz, FDn, FDn, FDn, 0, FDn, 0, Lap_wt, w2_diag, _b, _v, y+n*DMnd);
            }
        }
        free(Dx1); Dx1 = NULL;
    } else if(pSPARC->cell_typ == 17){
        if (overlap_flag) {
            // 2*T_12*df/dy + 2*T_13*df/dz
            const int x_Dx1_spos_dydz[6] =  {0,	0,	0,	0,	0,	DMnx_out};
            const int x_Dx1_epos_dydz[6] =  {DMnx_ex,	 DMnx_ex,	 DMnx_ex,	 DMnx_ex,	 FDn,	 DMnx_ex};
            const int y_Dx1_spos_dydz[6] =  {0,	0,	0,	 DMny_in,	 FDn,	 FDn};
            const int y_Dx1_epos_dydz[6] =  {DMny,	 DMny,	 FDn,	 DMny,	 DMny_in,	 DMny_in};
            const int z_Dx1_spos_dydz[6] =  {0,	 DMnz_in,	 FDn,	 FDn,	 FDn,	 FDn};
            const int z_Dx1_epos_dydz[6] =  {FDn,	 DMnz,	 DMnz_in,	 DMnz_in,	 DMnz_in,	 DMnz_in};
            const int x_x_ex_spos_dydz[6] = {0,	0,	0,	0,	0,	 DMnx_out};
            const int y_x_ex_spos_dydz[6] = {FDn,	 FDn,	 FDn,	 DMny,	 order,	 order};
            const int z_x_ex_spos_dydz[6] = {FDn,	 DMnz,	 order,	 order,	 order,	 order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX1_DX2_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, FDn, pshifty_ex, pshiftz_ex, pshifty_ex, DMnx_ex, pshiftz_ex, DMnxexny,
                        x_Dx1_spos_dydz[blk], x_Dx1_epos_dydz[blk], y_Dx1_spos_dydz[blk], y_Dx1_epos_dydz[blk], z_Dx1_spos_dydz[blk], z_Dx1_epos_dydz[blk],
                        x_x_ex_spos_dydz[blk], y_x_ex_spos_dydz[blk], z_x_ex_spos_dydz[blk],
                        pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz
                    );
                }
            }

            //  df/dz
            const int x_Dx1_spos[6] =  {0,	0,	0,	0,	0,	DMnx_in};
            const int x_Dx1_epos[6] =  {DMnx,	DMnx,	DMnx,	DMnx,	FDn,	DMnx};
            const int y_Dx1_spos[6] =  {0,	0,	0,	DMny_out,	FDn,	FDn};
            const int y_Dx1_epos[6] =  {DMny_ex,	DMny_ex,	FDn,	DMny_ex,	DMny_out,	DMny_out};
            const int z_Dx1_spos[6] =  {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            const int z_Dx1_epos[6] =  {FDn,	DMnz,	DMnz_in,	DMnz_in,	DMnz_in,	DMnz_in};
            const int x_x_ex_spos[6] = {FDn,	FDn,	FDn,	FDn,	FDn,	DMnx};
            const int y_x_ex_spos[6] = {0,	0,	0,	DMny_out,	FDn,	FDn};
            const int z_x_ex_spos[6] = {FDn,	DMnz,	order,	order,	order,	order};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                    Calc_DX_kpt(x_ex+n*DMnd_ex, Dx2+n*DMnd_yex, FDn, pshiftz_ex, pshifty_ex, DMnx, pshiftz_ex, DMnxnyex,
                        x_Dx1_spos[blk], x_Dx1_epos[blk], y_Dx1_spos[blk], y_Dx1_epos[blk], z_Dx1_spos[blk], z_Dx1_epos[blk],
                        x_x_ex_spos[blk], y_x_ex_spos[blk], z_x_ex_spos[blk],
                        pSPARC->D1_stencil_coeffs_z, 0.0);
                }
            }

            // Laplacian
            const int x_Dx1_spos_lap[6] = {FDn,	FDn,	FDn,	FDn,	FDn,	DMnx};
            const int y_Dx1_spos_lap[6] = {0,	0,	0,	DMny_in,	FDn,	FDn};
            const int z_Dx1_spos_lap[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            const int x_Dx2_spos_lap[6] = {0,	0,	0,	0,	0,	DMnx_in};
            const int y_Dx2_spos_lap[6] = {FDn,	FDn,	FDn,	DMny,	order,	order};
            const int z_Dx2_spos_lap[6] = {0,	DMnz_in,	FDn,	FDn,	FDn,	FDn};
            for (int blk = 0; blk < 6; blk++) {
                for (n = 0; n < ncol; n++) {
                stencil_5comp_kpt(x_ex+n*DMnd_ex, Dx1+n*DMnd_xex, Dx2+n*DMnd_yex, FDn,
                    1, DMnx, pshifty, pshifty_ex, DMnx_ex, DMnx, pshiftz, pshiftz_ex, DMnxexny, DMnxnyex,
                    x_Y_spos[blk], x_Y_epos[blk], y_Y_spos[blk], y_Y_epos[blk], z_Y_spos[blk], z_Y_epos[blk],
                    x_x_ex_spos_lap[blk], y_x_ex_spos_lap[blk], z_x_ex_spos_lap[blk],
                    x_Dx1_spos_lap[blk], y_Dx1_spos_lap[blk], z_Dx1_spos_lap[blk],
                    x_Dx2_spos_lap[blk], y_Dx2_spos_lap[blk], z_Dx2_spos_lap[blk],
                    Lap_wt, w2_diag, _b, _v, y+n*DMnd);
                }
            }
        } else {
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
