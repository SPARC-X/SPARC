/**
 * @file    MGGAexchangeCorrelation.c
 * @brief   This file contains the functions computing kinetic density tau and 
 * transferring tau and Vxc from phi domain to psi domain.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "mGGAtauTransferTauVxc.h"

#include "isddft.h"
#include "tools.h"
#include "parallelization.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "electronicGroundState.h"
#include "exchangeCorrelation.h"



/**
 * @brief   compute kinetic energy density \tau
 */
void compute_Kinetic_Density_Tau(SPARC_OBJ *pSPARC, double *Krho)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, n, DMnd, DMndsp, Ns, Nk, nstart, nend, spinor, Nspinor, sg, kpt;
    int size_k, count, Nband, spinDMnd;
    double g_nk, *Dx, *Dy, *Dz;
    double _Complex *Dx_kpt, *Dy_kpt, *Dz_kpt;
#ifdef DEBUG
    double t1, t2;
#endif
    
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor =  pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    size_k = DMndsp * Nband;
    spinDMnd = pSPARC->Nspin * DMnd;

    if (pSPARC->isGammaPoint == 1) {
        Dx = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dx != NULL);
        Dy = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dy != NULL);
        Dz = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dz != NULL);
        for (spinor = 0; spinor < Nspinor; spinor++) {
            sg  = pSPARC->spinor_start_indx + spinor;
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, pSPARC->Xorb + spinor*DMnd, DMndsp, Dx, DMnd, 0, pSPARC->dmcomm);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, pSPARC->Xorb + spinor*DMnd, DMndsp, Dy, DMnd, 1, pSPARC->dmcomm);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, pSPARC->Xorb + spinor*DMnd, DMndsp, Dz, DMnd, 2, pSPARC->dmcomm);
            count = 0;
            for (n = nstart; n <= nend; n++) {
                double *occ = pSPARC->occ;
                if (pSPARC->spin_typ == 1) occ += spinor * Ns;
                g_nk = occ[n];
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
        Dx_kpt = (double _Complex *) calloc(DMnd * Nband, sizeof(double _Complex));
        assert(Dx_kpt != NULL);
        Dy_kpt = (double _Complex *) calloc(DMnd * Nband, sizeof(double _Complex));
        assert(Dy_kpt != NULL);
        Dz_kpt = (double _Complex *) calloc(DMnd * Nband, sizeof(double _Complex));
        assert(Dz_kpt != NULL);
        
        for (kpt = 0; kpt < Nk; kpt++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                sg  = pSPARC->spinor_start_indx + spinor;
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, 
                    pSPARC->Xorb_kpt + kpt*size_k + spinor*DMnd, DMndsp, Dx_kpt, DMnd, 0, &pSPARC->k1_loc[kpt], pSPARC->dmcomm);
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, 
                    pSPARC->Xorb_kpt + kpt*size_k + spinor*DMnd, DMndsp, Dy_kpt, DMnd, 1, &pSPARC->k2_loc[kpt], pSPARC->dmcomm);
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, Nband, 0.0, 
                    pSPARC->Xorb_kpt + kpt*size_k + spinor*DMnd, DMndsp, Dz_kpt, DMnd, 2, &pSPARC->k3_loc[kpt], pSPARC->dmcomm);
                
                count = 0;
                for (n = nstart; n <= nend; n++) {
                    double *occ = pSPARC->occ + kpt*Ns;
                    if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                    g_nk = (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * occ[n];
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
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif        
        MPI_Allreduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, --- compute_Kinetic_Density_Tau: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all k-point groups
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {    
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        MPI_Allreduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, --- compute_Kinetic_Density_Tau: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0) {
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        
        MPI_Allreduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);        

    #ifdef DEBUG
        t2 = MPI_Wtime();
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



/**
 * @brief   compute the kinetic energy density tau and transfer it to phi domain for computing Vxc of metaGGA
 *          
 */
void compute_Kinetic_Density_Tau_Transfer_phi(SPARC_OBJ *pSPARC) {
    double *Krho = (double *) calloc(pSPARC->Nd_d_dmcomm * (2*pSPARC->Nspin-1), sizeof(double));
    assert(Krho != NULL);
    int i;
    if (!(pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL)) {
        compute_Kinetic_Density_Tau(pSPARC, Krho);
        if (pSPARC->Nspin == 1) // unpolarized
            TransferDensity(pSPARC, Krho, pSPARC->KineticTauPhiDomain); // D2D from dmcomm to dmcomm_phi
        else { // polarized
            for (i = 0; i < pSPARC->Nspin; i++)
                TransferDensity(pSPARC, Krho + i * pSPARC->Nd_d_dmcomm, pSPARC->KineticTauPhiDomain + pSPARC->Nd_d*(i+1)); // D2D from dmcomm to dmcomm_phi
            for (i = 0; i < pSPARC->Nd_d; i++) 
                pSPARC->KineticTauPhiDomain[i] = pSPARC->KineticTauPhiDomain[i+pSPARC->Nd_d] + pSPARC->KineticTauPhiDomain[i+2*pSPARC->Nd_d]; // tau total = tau up + tau dn
        }
    }
    else {
        if (pSPARC->dmcomm_phi != MPI_COMM_NULL) { // but in dmcomm_phi, they are receivers of the transferring
            if (pSPARC->Nspin == 1) // unpolarized
                TransferDensity(pSPARC, Krho, pSPARC->KineticTauPhiDomain); // D2D from dmcomm to dmcomm_phi
            else { // polarized
                for (i = 0; i < pSPARC->Nspin; i++)
                    TransferDensity(pSPARC, Krho + i * pSPARC->Nd_d_dmcomm, pSPARC->KineticTauPhiDomain + pSPARC->Nd_d*(i+1)); // D2D from dmcomm to dmcomm_phi
                for (i = 0; i < pSPARC->Nd_d; i++) pSPARC->KineticTauPhiDomain[i] = pSPARC->KineticTauPhiDomain[i+pSPARC->Nd_d] + pSPARC->KineticTauPhiDomain[i+2*pSPARC->Nd_d]; // tau total = tau up + tau dn
            }
        } 
    }
    free(Krho);
}


/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from phi-domain to psi-domain.   
 */
void Transfer_vxcMGGA3_phi_psi(SPARC_OBJ *pSPARC, double *vxcMGGA3_phi_domain, double *vxcMGGA3_psi_domain) 
{
#ifdef DEBUG
    double t1, t2;
#endif
    
    int rank, spin;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
        if (rank == 0) printf("Transmitting vxcMGGA3 from phi-domain to psi-domain (LOCAL) ...\n");
    #endif    
    
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx_phi; sdims[1] = pSPARC->npNdy_phi; sdims[2] = pSPARC->npNdz_phi;
    rdims[0] = pSPARC->npNdx; rdims[1] = pSPARC->npNdy; rdims[2] = pSPARC->npNdz;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    for (spin = 0; spin < pSPARC->Nspin; spin++) {
        D2D(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, vxcMGGA3_phi_domain + spin*pSPARC->Nd_d, 
            pSPARC->DMVertices_dmcomm, vxcMGGA3_psi_domain + spin*pSPARC->Nd_d_dmcomm, pSPARC->dmcomm_phi, sdims, 
            (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, 
            rdims, MPI_COMM_WORLD, sizeof(double));
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: D2D took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();
    #endif
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first kptcomms in each spincomm
    if (pSPARC->npspin > 1 && pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index == 0) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm * pSPARC->Nspin, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
    }
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: bcast btw/ spincomms of 1st kptcomm took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();
    #endif
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first bandcomms in each kptcomm
    if (pSPARC->spincomm_index >= 0 && pSPARC->npkpt > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->bandcomm_index == 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm * pSPARC->Nspin, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
    }
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: bcast btw/ kptcomms of 1st bandcomm took %.3f ms\n",(t2-t1)*1e3);
    #endif

    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast below on Quartz cluster, this Barrier fixed the issue (why?)
    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    
    // Bcast phi from first bandcomm to all other bandcomms
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm * pSPARC->Nspin, MPI_DOUBLE, 0, pSPARC->blacscomm);    
    }
    // pSPARC->req_veff_loc = MPI_REQUEST_NULL; // it seems that it is unnecessary to use the variable in vxcMGGA3?
    
    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast above on Quartz cluster, this Barrier fixed the issue (why?)
    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: mpi_bcast (count = %d) to all bandcomms took %.3f ms\n",pSPARC->Nd_d_dmcomm,(t2-t1)*1e3);
    #endif
}

