/**
 * @file    MGGAexchangeCorrelation.c
 * @brief   This file contains the functions used by metaGGA functionals.
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

#include "MGGAscan.h"
#include "MGGASscan.h"
#include "MGGAexchangeCorrelation.h"

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

    int i, n, DMnd, Ns, Nk, nstart, nend, spn_i, sg, kpt;
    int size_s, size_k, count, spinDMnd, Nband;
    // double *LapX, *X, g_nk, t1, t2, *Dx, *Dy, *Dz, *Krho;
    double *X, g_nk, *Dx, *Dy, *Dz;
    double _Complex *X_kpt, *Dx_kpt, *Dy_kpt, *Dz_kpt;
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

    if (pSPARC->isGammaPoint == 1) {
        size_s = DMnd * Nband;
        Dx = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dx != NULL);
        Dy = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dy != NULL);
        Dz = (double *) calloc(DMnd * Nband, sizeof(double));
        assert(Dz != NULL);
        // int lapcT[6];
        // lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        // lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
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
        // int lapcT[6];
        // lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        // lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
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
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(Krho, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        
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
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(Krho, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
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
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(Krho, Krho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
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
 * @brief   the main function in the file, compute needed xc energy density and its potentials, then transfer them to needed communicators
 *          
 * @param rho               electron density vector
 */
void Calculate_transfer_Vxc_MGGA(SPARC_OBJ *pSPARC,  double *rho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->countPotentialCalculate != 0) {
        compute_Kinetic_Density_Tau_Transfer_phi(pSPARC);
    }
    Calculate_Vxc_MGGA(pSPARC, rho);
    // printf("rank %d, Nspin %d, spincomm_index %d, reached the beginning of transfer_Vxc\n", rank, pSPARC->Nspin, pSPARC->spincomm_index);
    if (pSPARC->countPotentialCalculate != 0) { // not the first SCF step
        Transfer_vxcMGGA3_phi_psi(pSPARC, pSPARC->vxcMGGA3, pSPARC->vxcMGGA3_loc_dmcomm); // only transfer the potential they are going to use
        // if (!(pSPARC->spin_typ == 0 && pSPARC->is_phi_eq_kpt_topo)) { // the conditional judge whether this computation is necessary to do that. 
        // This function is moved to file eigenSolver.c and eigenSolverKpt.c
    }
    // printf("rank %d reached the end of transfer_Vxc\n", rank);
    pSPARC->countPotentialCalculate++;
}

/**
 * @brief   compute epsilon and XCPotential; vxcMGGA3 of metaGGA functional
 *          
 * @param rho               electron density vector
 */
void Calculate_Vxc_MGGA(SPARC_OBJ *pSPARC,  double *rho) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    if (pSPARC->countPotentialCalculate == 0) {
        // Initialize constants    
        XCCST_OBJ xc_cst;
        xc_constants_init(&xc_cst, pSPARC);
        if (pSPARC->Nspin == 1)
            Calculate_Vxc_GGA_PBE(pSPARC, &xc_cst, rho);
        else
            Calculate_Vxc_GSGA_PBE(pSPARC, &xc_cst, rho);
        // printf("finished first SCF PBE!\n");
        return;
    }

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    int DMnd;
    DMnd = pSPARC->Nd_d;

    int i;
    double *Drho_x, *Drho_y, *Drho_z, *normDrho, *lapcT;
    Drho_x = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(Drho_x != NULL);
    Drho_y = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(Drho_y != NULL);
    Drho_z = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(Drho_z != NULL);
    normDrho = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(normDrho != NULL);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2*pSPARC->Nspin-1, 0.0, rho, Drho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2*pSPARC->Nspin-1, 0.0, rho, Drho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2*pSPARC->Nspin-1, 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        lapcT = (double *) malloc(6 * sizeof(double));
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(i = 0; i < DMnd * (2*pSPARC->Nspin-1); i++){
            normDrho[i] = sqrt(Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                       Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i])); 
        }
        free(lapcT);
    } else {
        for(i = 0; i < (2*pSPARC->Nspin-1) * DMnd; i++){
            normDrho[i] = sqrt(Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i]);
        }
    }

    if (strcmpi(pSPARC->XC, "SCAN") == 0) {
        if (pSPARC->Nspin == 1)
            SCAN_EnergyDens_Potential(pSPARC, rho, normDrho, pSPARC->KineticTauPhiDomain, pSPARC->e_xc, pSPARC->vxcMGGA1, pSPARC->vxcMGGA2, pSPARC->vxcMGGA3);
        else
            SSCAN_EnergyDens_Potential(pSPARC, rho, normDrho, pSPARC->KineticTauPhiDomain, pSPARC->e_xc, pSPARC->vxcMGGA1, pSPARC->vxcMGGA2, pSPARC->vxcMGGA3);
    }

    for (i = 0; i < (2*pSPARC->Nspin-1) * DMnd; i++) {
        pSPARC->Dxcdgrho[i] = pSPARC->vxcMGGA2[i] / normDrho[i]; // to unify with the variable in GGA
    } 

    double *DDrho_x, *DDrho_y, *DDrho_z;
    DDrho_x = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(DDrho_x != NULL);
    DDrho_y = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(DDrho_y != NULL);
    DDrho_z = (double *) malloc((2*pSPARC->Nspin-1) * DMnd * sizeof(double));
    assert(DDrho_z != NULL);
    double temp1, temp2, temp3;
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < (2*pSPARC->Nspin-1) * DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * pSPARC->Dxcdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * pSPARC->Dxcdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * pSPARC->Dxcdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
        for(i = 0; i < (2*pSPARC->Nspin-1) * DMnd; i++){
            Drho_x[i] *= pSPARC->Dxcdgrho[i]; // Now the vector is (d(n\epsilon)/d(|grad n|)) * dn/dx / |grad n|
            Drho_y[i] *= pSPARC->Dxcdgrho[i]; // Now the vector is (d(n\epsilon)/d(|grad n|)) * dn/dy / |grad n|
            Drho_z[i] *= pSPARC->Dxcdgrho[i]; // Now the vector is (d(n\epsilon)/d(|grad n|)) * dn/dz / |grad n|
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2*pSPARC->Nspin-1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2*pSPARC->Nspin-1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2*pSPARC->Nspin-1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);

    if (pSPARC->Nspin == 1) {
        for (i = 0; i < DMnd; i++) {
            // epsilon has been computed in function SCAN_EnergyDens_Potential
            pSPARC->XCPotential[i] = pSPARC->vxcMGGA1[i] - DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
            // pSPARC->vxcMGGA3[i] has been computed in function SCAN_EnergyDens_Potential
        }
    }
    else {
        for (i = 0; i < DMnd; i++) {
            pSPARC->XCPotential[i] = pSPARC->vxcMGGA1[i] - DDrho_x[i] - DDrho_y[i] - DDrho_z[i] - DDrho_x[DMnd + i] - DDrho_y[DMnd + i] - DDrho_z[DMnd + i];
            pSPARC->XCPotential[DMnd + i] = pSPARC->vxcMGGA1[DMnd + i] - DDrho_x[i] - DDrho_y[i] - DDrho_z[i] - DDrho_x[2*DMnd + i] - DDrho_y[2*DMnd + i] - DDrho_z[2*DMnd + i];
        }
    }
    
    free(Drho_x); free(Drho_y); free(Drho_z); free(normDrho);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);

    #ifdef DEBUG
    t2 = MPI_Wtime();
        if (rank == 0) printf("end of Calculating Vxc_MGGA, took %.3f ms\n", (t2 - t1)*1000);
    #endif
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
            rdims, MPI_COMM_WORLD);
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

/**
 * @brief   the function to compute the exchange-correlation energy of metaGGA functional
 */
void Calculate_Exc_MGGA(SPARC_OBJ *pSPARC,  double *rho) {
    if (pSPARC->countPotentialCalculate == 1) {
        if (pSPARC->Nspin == 1) {
            Calculate_Exc_GGA_PBE(pSPARC, rho);
            return;
        }
        else {
            Calculate_Exc_GSGA_PBE(pSPARC, rho);
            return;
        }
    }
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int i;
    double Exc = 0.0;
    for (i = 0; i < pSPARC->Nd_d; i++) {
        //if(electronDens[i] != 0)
        Exc += rho[i] * pSPARC->e_xc[i]; 
    }
    
    Exc *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
}