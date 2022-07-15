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
#include "lapVecRoutines.h"
#include "energyDensity.h"



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
    if (pSPARC->countSCF != 0) {
        compute_Kinetic_Density_Tau_Transfer_phi(pSPARC);
    }
    Calculate_Vxc_MGGA(pSPARC, rho);
    // printf("rank %d, Nspin %d, spincomm_index %d, reached the beginning of transfer_Vxc\n", rank, pSPARC->Nspin, pSPARC->spincomm_index);
    if (pSPARC->countSCF != 0) { // not the first SCF step
        Transfer_vxcMGGA3_phi_psi(pSPARC, pSPARC->vxcMGGA3, pSPARC->vxcMGGA3_loc_dmcomm); // only transfer the potential they are going to use
        // if (!(pSPARC->spin_typ == 0 && pSPARC->is_phi_eq_kpt_topo)) { // the conditional judge whether this computation is necessary to do that. 
        // This function is moved to file eigenSolver.c and eigenSolverKpt.c
    }
    // printf("rank %d reached the end of transfer_Vxc\n", rank);
    pSPARC->countSCF++;
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
    if (pSPARC->countSCF == 0) {
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

    double t1, t2;
    t1 = MPI_Wtime();

    int DMnx, DMny, DMnz, DMnd;
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

    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("end of Calculating Vxc_MGGA, took %.3f ms\n", (t2 - t1)*1000);
    #endif
}


/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from phi-domain to psi-domain.   
 */
void Transfer_vxcMGGA3_phi_psi(SPARC_OBJ *pSPARC, double *vxcMGGA3_phi_domain, double *vxcMGGA3_psi_domain) 
{
    double t1, t2;
    
    int rank, spin;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
        if (rank == 0) printf("Transmitting vxcMGGA3 from phi-domain to psi-domain (LOCAL) ...\n");
    #endif    
    
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx_phi; sdims[1] = pSPARC->npNdy_phi; sdims[2] = pSPARC->npNdz_phi;
    rdims[0] = pSPARC->npNdx; rdims[1] = pSPARC->npNdy; rdims[2] = pSPARC->npNdz;

    t1 = MPI_Wtime();
    for (spin = 0; spin < pSPARC->Nspin; spin++) {
        D2D(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, vxcMGGA3_phi_domain + spin*pSPARC->Nd_d, 
            pSPARC->DMVertices_dmcomm, vxcMGGA3_psi_domain + spin*pSPARC->Nd_d_dmcomm, pSPARC->dmcomm_phi, sdims, 
            (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, 
            rdims, MPI_COMM_WORLD);
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: D2D took %.3f ms\n",(t2-t1)*1e3);
    #endif
    
    t1 = MPI_Wtime();
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first kptcomms in each spincomm
    if (pSPARC->npspin > 1 && pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index == 0) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm * pSPARC->Nspin, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: bcast btw/ spincomms of 1st kptcomm took %.3f ms\n",(t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first bandcomms in each kptcomm
    if (pSPARC->spincomm_index >= 0 && pSPARC->npkpt > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->bandcomm_index == 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm * pSPARC->Nspin, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: bcast btw/ kptcomms of 1st bandcomm took %.3f ms\n",(t2-t1)*1e3);
    #endif

    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast below on Quartz cluster, this Barrier fixed the issue (why?)
    t1 = MPI_Wtime();
    
    // Bcast phi from first bandcomm to all other bandcomms
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm * pSPARC->Nspin, MPI_DOUBLE, 0, pSPARC->blacscomm);    
    }
    // pSPARC->req_veff_loc = MPI_REQUEST_NULL; // it seems that it is unnecessary to use the variable in vxcMGGA3?
    
    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast above on Quartz cluster, this Barrier fixed the issue (why?)
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: mpi_bcast (count = %d) to all bandcomms took %.3f ms\n",pSPARC->Nd_d_dmcomm,(t2-t1)*1e3);
    #endif
    
}

/**
 * @brief   the function to compute the exchange-correlation energy of metaGGA functional
 */
void Calculate_Exc_MGGA(SPARC_OBJ *pSPARC,  double *rho) {
    if (pSPARC->countSCF == 1) {
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