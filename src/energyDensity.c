/**
 * @file    energyDensity.c
 * @brief   This file contains the functions for calculating energy densities
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "isddft.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "energyDensity.h"
#include "tools.h"

/**
 * @brief   compute kinetic energy density \tau
 */
void compute_Kinetic_Density_Tau(SPARC_OBJ *pSPARC, double *Krho)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, n, DMnd, Ns, Nk, nstart, nend, spn_i, sg, kpt;
    int size_s, size_k, count, spinDMnd, Nband;
    // double *LapX, *X, g_nk, t1, t2, *Dx, *Dy, *Dz, *Krho;
    double *LapX, *X, g_nk, t1, t2, *Dx, *Dy, *Dz;
    double _Complex *LapX_kpt, *X_kpt, *Dx_kpt, *Dy_kpt, *Dz_kpt;

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    spinDMnd = (pSPARC->spin_typ == 0) ? DMnd : 2*DMnd;

    if (pSPARC->isGammaPoint == 1) {
        size_s = DMnd * Nband;
        Dx = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dx != NULL);
        Dy = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dy != NULL);
        Dz = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dz != NULL);
        int lapcT[6];
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            sg  = pSPARC->spin_start_indx + spn_i;
            X = pSPARC->Xorb + spn_i*size_s;
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, X, Dx, 0, pSPARC->dmcomm);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, X, Dy, 1, pSPARC->dmcomm);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, X, Dz, 2, pSPARC->dmcomm);
            count = 0;
            for (n = nstart; n <= nend; n++) {
                g_nk = pSPARC->occ[n+spn_i*Ns];
                for (i = 0; i < DMnd; i++, count++) {
                    // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                    // only total in case of non-spin-polarized calculation
                    // different from electron density 
                    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
                        double LapD[3] = {0.0, 0.0, 0.0};
                        double Dxyz[3] = {Dx[count], Dy[count], Dz[count]};
                        matrixTimesVec_3d(pSPARC->lapcT, Dxyz, LapD);
                        Krho[sg*DMnd + i] += g_nk* (Dxyz[0] * LapD[0] + Dxyz[1] * LapD[1] + Dxyz[2] * LapD[2]);
                    } else {
                        Krho[sg*DMnd + i] += g_nk * (Dx[count] * Dx[count] + Dy[count] * Dy[count] + Dz[count] * Dz[count]);
                    }
                }
            }
        }

        free(Dx);
        free(Dy);
        free(Dz);
    } else {
        size_k = DMnd * Nband;
        size_s = size_k * Nk;
        Dx_kpt = (double _Complex *) calloc(DMnd * Nband, sizeof(double _Complex));
        assert(Dx_kpt != NULL);
        Dy_kpt = (double _Complex *) calloc(DMnd * Nband, sizeof(double _Complex));
        assert(Dy_kpt != NULL);
        Dz_kpt = (double _Complex *) calloc(DMnd * Nband, sizeof(double _Complex));
        assert(Dz_kpt != NULL);
        int lapcT[6];
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            sg  = pSPARC->spin_start_indx + spn_i;
            for (kpt = 0; kpt < Nk; kpt++) {
                X_kpt = pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s;
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, X_kpt, Dx_kpt, 0, pSPARC->k1_loc[kpt], pSPARC->dmcomm);
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, X_kpt, Dy_kpt, 1, pSPARC->k2_loc[kpt], pSPARC->dmcomm);
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, X_kpt, Dz_kpt, 2, pSPARC->k3_loc[kpt], pSPARC->dmcomm);
                
                count = 0;
                for (n = nstart; n <= nend; n++) {
                    g_nk = (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * pSPARC->occ[spn_i*Nk*Ns+kpt*Ns+n];
                    for (i = 0; i < DMnd; i++, count++) {
                        // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                        // only total in case of non-spin-polarized calculation
                        // different from electron density 
                        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
                            double _Complex LapD[3] = {0};
                            double _Complex Dxyz[3] = {Dx_kpt[count], Dy_kpt[count], Dz_kpt[count]};
                            matrixTimesVec_3d_complex(pSPARC->lapcT, Dxyz, LapD);
                            Krho[sg*DMnd + i] += g_nk* creal(conj(Dxyz[0]) * LapD[0] + conj(Dxyz[1]) * LapD[1] + conj(Dxyz[2]) * LapD[2]);
                        } else {
                            Krho[sg*DMnd + i] += g_nk * creal(conj(Dx_kpt[count]) * Dx_kpt[count] 
                                                        + conj(Dy_kpt[count]) * Dy_kpt[count] 
                                                        + conj(Dz_kpt[count]) * Dz_kpt[count]);
                        }
                    }
                }
            }
        }
        free(Dx_kpt);
        free(Dy_kpt);
        free(Dz_kpt);
    }

    // sum over spin comm group
    if(pSPARC->npspin > 1) {
        t1 = MPI_Wtime();
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(Krho, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("rank = %d, --- compute_Kinetic_Density_Tau: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all k-point groups
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {    
        t1 = MPI_Wtime();
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(Krho, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("rank = %d, --- compute_Kinetic_Density_Tau: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0) {
        t1 = MPI_Wtime();
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(Krho, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("rank = %d, --- compute_Kinetic_Density_Tau: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }


    double vscal = 1.0 / pSPARC->dV;
    if (pSPARC->spin_typ == 0) {
        for (i = 0; i < DMnd; i++) {
            Krho[i] *= vscal;
        }
    } else {
        vscal *= 0.5;       // spin factor
        for (i = 0; i < 2*DMnd; i++) {
            Krho[i] *= vscal;
        }
        // Total Kinetic energy density 
        for (i = 0; i < DMnd; i++) {
            Krho[i+2*DMnd] = Krho[i] + Krho[i+DMnd];
        }
    }

#ifdef DEBUG
    double KE = 0.0;
    for (i = 0; i < DMnd; i++) {
        if (pSPARC->spin_typ == 0) {
            KE += Krho[i];
        }
        else {
            KE += Krho[i + 2*DMnd];
        }
    }
    if (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) {
        MPI_Allreduce(MPI_IN_PLACE, &KE, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    KE *= pSPARC->dV;
    if (!rank) printf("Kinetic Energy computed by gradient: %f\n", KE);
#endif
}
