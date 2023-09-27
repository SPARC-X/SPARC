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

    int i, Nband, DMnd, Ns, Nk, kpt, count;
    int n, size_k, ncol;
    double *X, *Vexx, g_nk;
    double _Complex *X_kpt, *Vexx_kpt;
    MPI_Comm comm;

    #ifdef DEBUG
    double t1, t2;
    #endif

    DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Nband = pSPARC->Nband_bandcomm;
    size_k = DMndsp * Nband;
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    ncol = (pSPARC->spin_typ == 0) ? DMnd : 2*DMnd;
    comm = pSPARC->dmcomm;
    memset(Exxrho, 0, sizeof(double) * DMnd * (2*pSPARC->Nspin-1));
    double *Exxrho_ = (pSPARC->spin_typ > 0) ? Exxrho+DMnd : Exxrho;

    if (pSPARC->isGammaPoint == 1) {
        Vexx = (double *) malloc(sizeof(double) * DMnd * Nband);
        assert(Vexx != NULL);
        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            int sg  = pSPARC->spinor_start_indx + spinor;
            X = pSPARC->Xorb + spinor*DMnd;
            double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ+spinor*Ns) : pSPARC->occ;

            memset(Vexx, 0, sizeof(double)* DMnd * Nband);
            exact_exchange_potential(pSPARC, X, DMndsp, Nband, DMnd, Vexx, DMnd, spinor, comm);

            count = 0;
            for (n = 0; n < Nband; n++) {
                g_nk = occ[n+pSPARC->band_start_indx];
                for (i = 0; i < DMnd; i++, count++) {
                    // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                    // only total in case of non-spin-polarized calculation
                    // different from electron density 
                    Exxrho_[sg*DMnd + i] += g_nk * X[i+n*DMndsp] * Vexx[count];
                }
            }
        }
        free(Vexx);
    } else {                
        Vexx_kpt = (double _Complex *) malloc(sizeof(double _Complex) * DMnd * Nband);
        assert(Vexx_kpt != NULL);
        for (kpt = 0; kpt < Nk; kpt++) {
            for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                int sg  = pSPARC->spin_start_indx + spinor;
                X_kpt = pSPARC->Xorb_kpt + kpt*size_k + spinor*DMnd;
                double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ+spinor*Ns*Nk) : pSPARC->occ;

                memset(Vexx_kpt, 0, sizeof(double _Complex) * DMnd * Nband);
                exact_exchange_potential_kpt(pSPARC, X_kpt, DMndsp, Nband, DMnd, Vexx_kpt, DMnd, spinor, kpt, comm);
                
                count = 0;
                for (n = 0; n < Nband; n++) {
                    g_nk = (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * occ[kpt*Ns+n+pSPARC->band_start_indx];
                    for (i = 0; i < DMnd; i++, count++) {
                        // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                        // only total in case of non-spin-polarized calculation
                        // different from electron density 
                        Exxrho_[sg*DMnd + i] += g_nk * creal(conj(X_kpt[i+n*DMndsp]) * Vexx_kpt[count]);
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
        MPI_Allreduce(MPI_IN_PLACE, Exxrho_, ncol, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);

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
        MPI_Allreduce(MPI_IN_PLACE, Exxrho_, ncol, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        
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
        MPI_Allreduce(MPI_IN_PLACE, Exxrho_, ncol, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, --- Calculate exact exchange energy density: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    double vscal = 1.0 / pSPARC->dV;
    if (pSPARC->spin_typ == 0) {
        for (i = 0; i < DMnd; i++) {
            Exxrho[i] *= vscal;
        }
    } else {
        vscal *= 0.5;
        for (i = 0; i < 2*DMnd; i++) {
            Exxrho[i+DMnd] *= vscal;
        }
        // Total exx energy density 
        for (i = 0; i < DMnd; i++) {
            Exxrho[i] = Exxrho[i+DMnd] + Exxrho[i+2*DMnd];
        }
    }
#ifdef DEBUG
    double Exx = 0.0;
    for (i = 0; i < DMnd; i++) {
        Exx += Exxrho[i];
    }
    if (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) {
        MPI_Allreduce(MPI_IN_PLACE, &Exx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    Exx *= pSPARC->dV;
    if (!rank) printf("\nExact exchange energy from energy density: %f\n"
                        "Exact exchange energy calculated directly: %f\n", Exx, pSPARC->Eexx);
#endif
}