/***
 * @file    exactExchangeEnergyDensity.c
 * @brief   This file contains the functions for Exact Exchange energy density.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <complex.h>

#include "exactExchange.h"
#include "exactExchangeKpt.h"
#include "exactExchangeEnergyDensity.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Compute exact exchange energy density
 */
void computeExactExchangeEnergyDensity(SPARC_OBJ *pSPARC, double *Exxrho)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, Nband, DMnd, Ns, spn_i, Nk, kpt, count;
    int n, nstart, nend, sg, size_s, size_k, spinDMnd;
    double *X, *Vexx, g_nk;
    double _Complex *X_kpt, *Vexx_kpt;
    MPI_Comm comm;

    #ifdef DEBUG
    double t1, t2;
    #endif

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    spinDMnd = (pSPARC->spin_typ == 0) ? DMnd : 2*DMnd;
    comm = pSPARC->dmcomm;
    memset(Exxrho, 0, sizeof(double) * DMnd * (2*pSPARC->Nspin-1));

    if (pSPARC->isGammaPoint == 1) {
        size_s = DMnd * Nband;
        Vexx = (double *) calloc(sizeof(double), DMnd * Nband);
        assert(Vexx != NULL);
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            sg  = pSPARC->spin_start_indx + spn_i;
            X = pSPARC->Xorb + spn_i*size_s;

            memset(Vexx, 0, sizeof(double)*size_s);
            exact_exchange_potential(pSPARC, X, Nband, DMnd, Vexx, spn_i, comm);

            count = 0;
            for (n = nstart; n <= nend; n++) {
                g_nk = pSPARC->occ[n+spn_i*Ns];
                for (i = 0; i < DMnd; i++, count++) {
                    // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                    // only total in case of non-spin-polarized calculation
                    // different from electron density 
                    Exxrho[sg*DMnd + i] += g_nk * X[count] * Vexx[count];
                }
            }
        }
        free(Vexx);
    } else {
        size_k = DMnd * Nband;
        size_s = size_k * Nk;
        Vexx_kpt = (double _Complex *) calloc(sizeof(double _Complex), DMnd * Nband);
        assert(Vexx_kpt != NULL);
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            sg  = pSPARC->spin_start_indx + spn_i;
            for (kpt = 0; kpt < Nk; kpt++) {
                X_kpt = pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s;

                memset(Vexx_kpt, 0, sizeof(double _Complex) * DMnd * Nband);
                exact_exchange_potential_kpt(pSPARC, X_kpt, Nband, DMnd, Vexx_kpt, spn_i, kpt, comm);
                
                count = 0;
                for (n = nstart; n <= nend; n++) {
                    g_nk = (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * pSPARC->occ[spn_i*Nk*Ns+kpt*Ns+n];
                    for (i = 0; i < DMnd; i++, count++) {
                        // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                        // only total in case of non-spin-polarized calculation
                        // different from electron density 
                        Exxrho[sg*DMnd + i] += g_nk * creal(conj(X_kpt[count]) * Vexx_kpt[count]);
                    }
                }
            }
        }
        free(Vexx_kpt);
    }


    // sum over spin comm group
    if(pSPARC->npspin > 1) {
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(Exxrho, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, --- Calculate kinetic energy density: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all k-point groups
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(Exxrho, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, --- Calculate exact exchange energy density: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0) {
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(Exxrho, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, --- Calculate exact exchange energy density: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    double vscal = 1.0 / pSPARC->dV * pSPARC->exx_frac;
    if (pSPARC->spin_typ == 0) {
        for (i = 0; i < DMnd; i++) {
            Exxrho[i] *= vscal;
        }
    } else {
        vscal *= 0.5;       // spin factor
        for (i = 0; i < 2*DMnd; i++) {
            Exxrho[i] *= vscal;
        }
        // Total Kinetic energy density 
        for (i = 0; i < DMnd; i++) {
            Exxrho[i+2*DMnd] = Exxrho[i] + Exxrho[i+DMnd];
        }
    }
#ifdef DEBUG
    double Exx = 0.0;
    for (i = 0; i < DMnd; i++) {
        Exx += Exxrho[i + pSPARC->spin_typ*2*DMnd];
    }
    if (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) {
        MPI_Allreduce(MPI_IN_PLACE, &Exx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    Exx *= pSPARC->dV;
    if (!rank) printf("\nExact exchange energy from energy density: %f\n"
                        "Exact exchange energy calculated directly: %f\n", Exx, pSPARC->exx_frac*pSPARC->Eexx);
#endif
}