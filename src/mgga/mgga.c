/**
 * @file    scan.c
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

#include "scan.h"
#include "mgga.h"

#include "isddft.h"
#include "tools.h"
#include "parallelization.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "electronicGroundState.h"
#include "exchangeCorrelation.h"
#include "lapVecRoutines.h"


void initialize_MGGA(SPARC_OBJ *pSPARC) { // allocate space to variables
    int DMnx, DMny, DMnz, DMnd;
    
    DMnd = pSPARC->Nd_d;
    pSPARC->KineticTauPhiDomain = (double *) calloc(DMnd * (2*pSPARC->Nspin-1), sizeof(double)); // different from pSPARC->KineticRho, which is in dmcomm
    assert(pSPARC->KineticTauPhiDomain != NULL);
    
    pSPARC->vxcMGGA1 = (double*)malloc(sizeof(double)*DMnd); // d(n\epsilon)/dn in dmcomm_phi
    assert(pSPARC->vxcMGGA1 != NULL);
    pSPARC->vxcMGGA2 = (double*)malloc(sizeof(double)*DMnd); // d(n\epsilon)/d|grad n| in dmcomm_phi
    assert(pSPARC->vxcMGGA2 != NULL);
    pSPARC->vxcMGGA3 = (double*)calloc(DMnd, sizeof(double)); // d(n\epsilon)/d\tau in dmcomm_phi
    assert(pSPARC->vxcMGGA3 != NULL);
    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) { // d(n\epsilon)/d\tau in dmcomm
        pSPARC->vxcMGGA3_loc_dmcomm = (double *)calloc( pSPARC->Nd_d_dmcomm * pSPARC->Nspin, sizeof(double) );
        assert(pSPARC->vxcMGGA3_loc_dmcomm != NULL);
    }
    pSPARC->vxcMGGA3_loc_kptcomm = (double *)calloc( pSPARC->Nd_d_kptcomm, sizeof(double) ); // d(n\epsilon)/d\tau in kptcomm
    assert(pSPARC->vxcMGGA3_loc_kptcomm != NULL);

    pSPARC->countSCF = 0;
}

/**
 * @brief   compute the kinetic energy density tau and transfer it to phi domain for computing Vxc of metaGGA
 *          
 */
void compute_Kinetic_Density_Tau_Transfer_phi(SPARC_OBJ *pSPARC) {
    double *Krho;
    if (!(pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL)) {
        Krho = (double *) calloc(pSPARC->Nd_d_dmcomm * (2*pSPARC->Nspin-1), sizeof(double));
        assert(Krho != NULL);
        compute_Kinetic_Density_Tau(pSPARC, Krho);
        TransferDensity(pSPARC, Krho, pSPARC->KineticTauPhiDomain); // D2D from dmcomm to dmcomm_phi
        free(Krho);
    }
    else {
        if (pSPARC->dmcomm_phi != MPI_COMM_NULL) TransferDensity(pSPARC, Krho, pSPARC->KineticTauPhiDomain); // D2D from dmcomm to dmcomm_phi
    }
}

/**
 * @brief   the main function in the file, compute needed xc energy density and its potentials, then transfer them to needed communicators
 *          
 * @param rho               electron density vector
 */
void Calculate_transfer_Vxc_MGGA(SPARC_OBJ *pSPARC,  double *rho) {
    if (pSPARC->countSCF != 0) {
        compute_Kinetic_Density_Tau_Transfer_phi(pSPARC);
    }
    Calculate_Vxc_MGGA(pSPARC, rho);
    // printf("rank %d reached the beginning of transfer_Vxc\n", rank);
    if (pSPARC->countSCF != 0) { // not the first SCF step
        Transfer_vxcMGGA3_phi_psi(pSPARC, pSPARC->vxcMGGA3, pSPARC->vxcMGGA3_loc_dmcomm);
        if (!(pSPARC->spin_typ == 0 && pSPARC->is_phi_eq_kpt_topo)) { // the conditional judge whether this computation is necessary to do that
            if (pSPARC->kptcomm != MPI_COMM_NULL) // the conditional judge whether this processor can join the transfer
                Transfer_vxcMGGA3_psi_kptTopo(pSPARC, pSPARC->vxcMGGA3_loc_dmcomm, pSPARC->vxcMGGA3_loc_kptcomm);
        }
    }
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
        
        Calculate_Vxc_GGA_PBE(pSPARC, &xc_cst, rho);
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

    if (pSPARC->spin_typ == 0) { // spin unpolarized
        int i;
        double *Drho_x, *Drho_y, *Drho_z, *normDrho, *lapcT;
        Drho_x = (double *) malloc(DMnd * sizeof(double));
        assert(Drho_x != NULL);
        Drho_y = (double *) malloc(DMnd * sizeof(double));
        assert(Drho_y != NULL);
        Drho_z = (double *) malloc(DMnd * sizeof(double));
        assert(Drho_z != NULL);
        normDrho = (double *) malloc(DMnd * sizeof(double));
        assert(normDrho != NULL);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_x, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_y, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);
        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
            lapcT = (double *) malloc(6 * sizeof(double));
            lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
            lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
            for(i = 0; i < DMnd; i++){
                normDrho[i] = sqrt(Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                           Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i])); 
            }
            free(lapcT);
        } else {
            for(i = 0; i < DMnd; i++){
                normDrho[i] = sqrt(Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i]);
            }
        }
        
        if (strcmpi(pSPARC->XC, "SCAN") == 0) {
            SCAN_EnergyDens_Potential(pSPARC, rho, normDrho, pSPARC->KineticTauPhiDomain, pSPARC->e_xc, pSPARC->vxcMGGA1, pSPARC->vxcMGGA2, pSPARC->vxcMGGA3);
        }
        
        // double *vxcMGGA2 = pSPARC->vxcMGGA2;

        for (i = 0; i < DMnd; i++) {
            pSPARC->Dxcdgrho[i] = pSPARC->vxcMGGA2[i] / normDrho[i]; // to unify with the variable in GGA
        } 

        double *DDrho_x, *DDrho_y, *DDrho_z;
        DDrho_x = (double *) malloc(DMnd * sizeof(double));
        assert(DDrho_x != NULL);
        DDrho_y = (double *) malloc(DMnd * sizeof(double));
        assert(DDrho_y != NULL);
        DDrho_z = (double *) malloc(DMnd * sizeof(double));
        assert(DDrho_z != NULL);
        double temp1, temp2, temp3;
        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
            for(i = 0; i < DMnd; i++){
                temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * pSPARC->Dxcdgrho[i];
                temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * pSPARC->Dxcdgrho[i];
                temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * pSPARC->Dxcdgrho[i];
                Drho_x[i] = temp1;
                Drho_y[i] = temp2;
                Drho_z[i] = temp3;
            }
        } else {
            for(i = 0; i < DMnd; i++){
                Drho_x[i] *= pSPARC->Dxcdgrho[i]; // Now the vector is (d(n\epsilon)/d(|grad n|)) * dn/dx / |grad n|
                Drho_y[i] *= pSPARC->Dxcdgrho[i]; // Now the vector is (d(n\epsilon)/d(|grad n|)) * dn/dy / |grad n|
                Drho_z[i] *= pSPARC->Dxcdgrho[i]; // Now the vector is (d(n\epsilon)/d(|grad n|)) * dn/dz / |grad n|
            }
        }
        
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);

        for (i = 0; i < DMnd; i++) {
            // epsilon has been computed in function SCAN_EnergyDens_Potential
            pSPARC->XCPotential[i] = pSPARC->vxcMGGA1[i] - DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
            // pSPARC->vxcMGGA3[i] has been computed in function SCAN_EnergyDens_Potential
        }
        
        free(Drho_x); free(Drho_y); free(Drho_z); free(normDrho);
        free(DDrho_x); free(DDrho_y); free(DDrho_z);
    }
    else { // spin polarized
        if (rank == 0)
            printf(RED "ERROR: SCAN functional for polarized spin is not available yet.\n" RESET);
        exit(EXIT_FAILURE); 
    }
    
    // pSPARC->countSCF++;
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("end of Calculating Vxc_MGGA, took %.3f ms\n", (t2 - t1)*1000);
    #endif
}

/**
 * @brief   the function to compute the exchange-correlation energy of metaGGA functional
 */
void Calculate_Exc_MGGA(SPARC_OBJ *pSPARC,  double *rho) {
    if (pSPARC->countSCF == 1) {
        Calculate_Exc_GGA_PBE(pSPARC, rho);
        return;
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


/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from phi-domain to psi-domain.   
 */
void Transfer_vxcMGGA3_phi_psi(SPARC_OBJ *pSPARC, double *vxcMGGA3_phi_domain, double *vxcMGGA3_psi_domain) 
{
    double t1, t2;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
        if (rank == 0) printf("Transmitting vxcMGGA3 from phi-domain to psi-domain (LOCAL) ...\n");
    #endif    
    //void DD2DD(SPARC_OBJ *pSPARC, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert, double *rdata, 
    //       MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims)
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx_phi; sdims[1] = pSPARC->npNdy_phi; sdims[2] = pSPARC->npNdz_phi;
    rdims[0] = pSPARC->npNdx; rdims[1] = pSPARC->npNdy; rdims[2] = pSPARC->npNdz;

    t1 = MPI_Wtime();
    D2D(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, vxcMGGA3_phi_domain, 
        pSPARC->DMVertices_dmcomm, vxcMGGA3_psi_domain, pSPARC->dmcomm_phi, sdims, 
        (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, 
        rdims, MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: D2D took %.3f ms\n",(t2-t1)*1e3);
    #endif
    
    t1 = MPI_Wtime();
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first kptcomms in each spincomm
    if (pSPARC->npspin > 1 && pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index == 0) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: bcast btw/ spincomms of 1st kptcomm took %.3f ms\n",(t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first bandcomms in each kptcomm
    if (pSPARC->spincomm_index >= 0 && pSPARC->npkpt > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->bandcomm_index == 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: bcast btw/ kptcomms of 1st bandcomm took %.3f ms\n",(t2-t1)*1e3);
    #endif

    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast below on Quartz cluster, this Barrier fixed the issue (why?)
    t1 = MPI_Wtime();
    
    // Bcast phi from first bandcomm to all other bandcomms
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(vxcMGGA3_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->blacscomm);    
    }
    // pSPARC->req_veff_loc = MPI_REQUEST_NULL; // it seems that it is unnecessary to use the variable in vxcMGGA3?
    
    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast above on Quartz cluster, this Barrier fixed the issue (why?)
    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_phi_psi: mpi_bcast (count = %d) to all bandcomms took %.3f ms\n",pSPARC->Nd_d_dmcomm,(t2-t1)*1e3);
    #endif
    
}

/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from psi-domain to k-point topology.   
 */
void Transfer_vxcMGGA3_psi_kptTopo(SPARC_OBJ *pSPARC, double *vxcMGGA3_psi_domain, double *vxcMGGA3_kpt_topo) 
{
    double t1, t2;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Transmitting vxcMGGA3 from psi-domain (LOCAL) to k-point topology ...\n");
#endif    

    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx; sdims[1] = pSPARC->npNdy; sdims[2] = pSPARC->npNdz; 
    rdims[0] = pSPARC->npNdx_kptcomm; rdims[1] = pSPARC->npNdy_kptcomm; rdims[2] = pSPARC->npNdz_kptcomm;
    // int sg  = pSPARC->spin_start_indx + spn_i; // currently there is no spin polarization

    t1 = MPI_Wtime();

    D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, pSPARC->DMVertices_dmcomm, vxcMGGA3_psi_domain, 
        pSPARC->DMVertices_kptcomm, pSPARC->vxcMGGA3_loc_kptcomm, pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
        sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm);

    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("---Transfer_vxcMGGA3_psi_kptTopo: D2D from psi domain to kpt topology took %.3f ms\n",(t2-t1)*1e3);
    #endif
    
}

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult
 */
void compute_mGGA_term_hamil(const SPARC_OBJ *pSPARC, double *x, int ncol, int colLength, int *DMVertices, double *vxcMGGA3_dm, double *mGGAterm, int spin, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    t1 = MPI_Wtime();

    double *Dx_x = (double *) calloc(colLength, sizeof(double));
    assert(Dx_x != NULL);
    double *Dx_y = (double *) calloc(colLength, sizeof(double));
    assert(Dx_y != NULL);
    double *Dx_z = (double *) calloc(colLength, sizeof(double));
    assert(Dx_z != NULL);
    double *Dvxc3Dx_x = (double *) calloc(colLength, sizeof(double));
    assert(Dvxc3Dx_x != NULL);
    double *Dvxc3Dx_y = (double *) calloc(colLength, sizeof(double));
    assert(Dvxc3Dx_y != NULL);
    double *Dvxc3Dx_z = (double *) calloc(colLength, sizeof(double));
    assert(Dvxc3Dx_z != NULL);

    int i, j;
    for (i = 0; i < ncol; i++) {
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, &(x[i*(unsigned)colLength]), Dx_x, 0, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, &(x[i*(unsigned)colLength]), Dx_y, 1, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, &(x[i*(unsigned)colLength]), Dx_z, 2, comm);

        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){ // transform for unorthogonal cell
            double DxAfter[3], DxBefore[3];
            double *lapcT = (double *)pSPARC->lapcT;

            DxAfter[0] = 0.0; DxAfter[1] = 0.0; DxAfter[2] = 0.0;
            for (j = 0; j < colLength; j++) {
                DxBefore[0] = Dx_x[j]; DxBefore[1] = Dx_y[j]; DxBefore[2] = Dx_z[j];

                DxAfter[0] = DxBefore[0] * pSPARC->lapcT[0] + DxBefore[1] * pSPARC->lapcT[1] + DxBefore[2] * pSPARC->lapcT[2];
                DxAfter[1] = DxBefore[0] * pSPARC->lapcT[3] + DxBefore[1] * pSPARC->lapcT[4] + DxBefore[2] * pSPARC->lapcT[5];
                DxAfter[2] = DxBefore[0] * pSPARC->lapcT[6] + DxBefore[1] * pSPARC->lapcT[7] + DxBefore[2] * pSPARC->lapcT[8];

                Dx_x[j] = DxAfter[0]; 
                Dx_y[j] = DxAfter[1]; 
                Dx_z[j] = DxAfter[2]; 
            }
        }

        for (j = 0; j < colLength; j++) {
            Dx_x[j] *= vxcMGGA3_dm[j]; 
            Dx_y[j] *= vxcMGGA3_dm[j]; 
            Dx_z[j] *= vxcMGGA3_dm[j]; // Now the vectors are Vxc3*gradX
        }
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, Dx_x, Dvxc3Dx_x, 0, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, Dx_y, Dvxc3Dx_y, 1, comm);
        Gradient_vectors_dir(pSPARC, colLength, DMVertices, 1, 0.0, Dx_z, Dvxc3Dx_z, 2, comm);
        
        for (j = 0; j < colLength; j++) {
            mGGAterm[j + i*colLength] = Dvxc3Dx_x[j] + Dvxc3Dx_y[j] + Dvxc3Dx_z[j];
        }
    }

    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("end of Calculating mGGA term in Hamiltonian, took %.3f ms\n", (t2 - t1)*1000);
    #endif
    free(Dx_x); free(Dx_y); free(Dx_z);
    free(Dvxc3Dx_x); free(Dvxc3Dx_y); free(Dvxc3Dx_z);
}

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult_kpt
 */
void compute_mGGA_term_hamil_kpt(const SPARC_OBJ *pSPARC, double _Complex *x, int ncol, int colLength, int *DMVertices, double *vxcMGGA3_dm, double _Complex *mGGAterm, int spin, int kpt, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    t1 = MPI_Wtime();
    
    int size_k = colLength * ncol;
    double _Complex *Dx_x_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dx_x_kpt != NULL);
    double _Complex *Dx_y_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dx_y_kpt != NULL);
    double _Complex *Dx_z_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dx_z_kpt != NULL);
    double _Complex *Dvxc3Dx_x_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dvxc3Dx_x_kpt != NULL);
    double _Complex *Dvxc3Dx_y_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dvxc3Dx_y_kpt != NULL);
    double _Complex *Dvxc3Dx_z_kpt = (double _Complex *) calloc(size_k, sizeof(double _Complex));
    assert(Dvxc3Dx_z_kpt != NULL);

    int j; // seems that in computations having k-point, there is no need to loop over bands

    double _Complex *X_kpt = x;
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, X_kpt, Dx_x_kpt, 0, pSPARC->k1_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, X_kpt, Dx_y_kpt, 1, pSPARC->k2_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, X_kpt, Dx_z_kpt, 2, pSPARC->k3_loc[kpt], comm);
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){ // transform for unorthogonal cell
        double _Complex DxAfter[3] = {0.0};
        for (j = 0; j < size_k; j++) {
            double _Complex DxBefore[3] = {Dx_x_kpt[j], Dx_y_kpt[j], Dx_z_kpt[j]};
            double *lapcT = (double *)pSPARC->lapcT;
            // matrixTimesVec_3d_complex(lapcT, DxBefore, DxAfter);
            DxAfter[0] = DxBefore[0] * pSPARC->lapcT[0] + DxBefore[1] * pSPARC->lapcT[1] + DxBefore[2] * pSPARC->lapcT[2];
            DxAfter[1] = DxBefore[0] * pSPARC->lapcT[3] + DxBefore[1] * pSPARC->lapcT[4] + DxBefore[2] * pSPARC->lapcT[5];
            DxAfter[2] = DxBefore[0] * pSPARC->lapcT[6] + DxBefore[1] * pSPARC->lapcT[7] + DxBefore[2] * pSPARC->lapcT[8];
            Dx_x_kpt[j] = DxAfter[0]; 
            Dx_y_kpt[j] = DxAfter[1]; 
            Dx_z_kpt[j] = DxAfter[2]; 
        }
    }

    for (j = 0; j < size_k; j++) {
        Dx_x_kpt[j] *= vxcMGGA3_dm[j%colLength]; 
        Dx_y_kpt[j] *= vxcMGGA3_dm[j%colLength]; 
        Dx_z_kpt[j] *= vxcMGGA3_dm[j%colLength]; // Now the vectors are Vxc3*gradX
    }
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, Dx_x_kpt, Dvxc3Dx_x_kpt, 0, pSPARC->k1_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, Dx_y_kpt, Dvxc3Dx_y_kpt, 1, pSPARC->k2_loc[kpt], comm);
    Gradient_vectors_dir_kpt(pSPARC, colLength, DMVertices, ncol, 0.0, Dx_z_kpt, Dvxc3Dx_z_kpt, 2, pSPARC->k3_loc[kpt], comm);
    
    for (j = 0; j < size_k; j++) {
        mGGAterm[j] = Dvxc3Dx_x_kpt[j] + Dvxc3Dx_y_kpt[j] + Dvxc3Dx_z_kpt[j];
    }
    // if ((pSPARC->countSCF == 5) && (kpt == 1) && (comm == pSPARC->dmcomm)) {
    //     FILE *compute_mGGA_term_hamil_kpt = fopen("X_mGGA_term_hamil_kpt.txt","w");
    //     fprintf(compute_mGGA_term_hamil_kpt, "SCF %d, ncol %d, colLength %d, spin %d, kpt %d\n", 
    //     pSPARC->countSCF, ncol, colLength, spin, kpt);
    //     int index;
    //     fprintf(compute_mGGA_term_hamil_kpt, "SCF 5, vxcMGGA3_dm is listed below\n");
    //     for (index = 0; index < colLength; index++) {
    //         fprintf(compute_mGGA_term_hamil_kpt, "%10.9E\n", vxcMGGA3_dm[index]);
    //     }
        
    //     fprintf(compute_mGGA_term_hamil_kpt, "SCF 5, kpt 1, 2nd column [colLength + index] of x is listed below\n");
    //     for (index = 0; index < colLength; index++) {
    //         fprintf(compute_mGGA_term_hamil_kpt, "%10.9E %10.9E\n", creal(x[colLength + index]), cimag(x[colLength + index]));
    //     }
        
    //     fprintf(compute_mGGA_term_hamil_kpt, "2nd column [colLength + index] of mGGAterm is listed below\n");
    //     for (index = 0; index < colLength; index++) {
    //         fprintf(compute_mGGA_term_hamil_kpt, "%10.9E %10.9E\n", creal(mGGAterm[colLength + index]), cimag(mGGAterm[colLength + index]));
    //     }
    //     fclose(compute_mGGA_term_hamil_kpt);
    // }

    t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("end of Calculating mGGA term in Hamiltonian, took %.3f ms\n", (t2 - t1)*1000);
    #endif
    
    free(Dx_x_kpt); free(Dx_y_kpt); free(Dx_z_kpt);
    free(Dvxc3Dx_x_kpt); free(Dvxc3Dx_y_kpt); free(Dvxc3Dx_z_kpt);
}


/**
 * @brief   compute the metaGGA psi stress term, which is 
 * \int_{Omega}(\rho\frac{\partial \epsilon_{xc}}{\partial \tau})[\sum_{n=1}^N_s g_n(2\nabla_{\alpha}\psi_n * \nabla_{\beta}\psi_n)] dx
 * called by Calculate_electronic_stress in stress.c and Calculate_electronic_pressure in pressure.c
 * Attention: it is NOT the full contrubution of metaGGA on stress!
 */
void Calculate_XC_stress_mGGA_psi_term(SPARC_OBJ *pSPARC) {
    // currently there is no spin-polarized
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return; // mimic Calculate_nonlocal_kinetic_stress_linear in stress.c ???
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nspin, ncol, Ns, DMnd, DMnx, DMny, size_s, dim, count, spn_i, n, i, j;
    double *dpsi_ptr, *dpsi_ptr2, dpsi1_dpsi1, g_nk, dpsi1_dpsi2, dpsi1_dpsi3, dpsi2_dpsi3, dpsi2_dpsi2, dpsi3_dpsi3;
    nspin = pSPARC->Nspin_spincomm; // number of spin in my spin communicator
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates; // total number of bands
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    size_s = ncol * DMnd;
    int len_psi = DMnd * ncol * nspin;
    double *stress_mGGA_psi, *stress_mGGA_psi_cartesian, *dpsi_full; // the metaGGA term directly related to all wave functions psi
    stress_mGGA_psi = (double*) calloc(6, sizeof(double));
    stress_mGGA_psi_cartesian = (double*) calloc(6, sizeof(double));
    dpsi_full = (double *)malloc( len_psi * sizeof(double) );
    if (dpsi_full == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    double *XorY = pSPARC->Xorb;
    double *YorZ = pSPARC->Yorb;

    double *vxcMGGA3_loc = pSPARC->vxcMGGA3_loc_dmcomm; // local rho*d\epsilon_{xc} / d\tau array
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        count = 0;
        for (dim = 0; dim < 3; dim++) {
            // find dPsi in direction dim
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, XorY+spn_i*size_s, YorZ+spn_i*size_s, dim, pSPARC->dmcomm);
            // Kinetic stress
            if(dim == 0){
                // component (1,1)
                // find d(Psi*Ds)
                double *dpsi_1p = YorZ; 
                
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = YorZ + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_1p + spn_i * size_s + n * DMnd; // dpsi_1p
                    dpsi1_dpsi1 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi1 += vxcMGGA3_loc[i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_mGGA_psi[0] += dpsi1_dpsi1 * g_nk; // component (1,1)
                }

                // component (1,2)
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, dpsi_full+spn_i*size_s, 1, pSPARC->dmcomm);
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = YorZ + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_full + spn_i * size_s + n * DMnd; // dpsi_2
                    dpsi1_dpsi2 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi2 += vxcMGGA3_loc[i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_mGGA_psi[1] += dpsi1_dpsi2 * g_nk; // component (1,2)
                }
                

                // component (1,3)
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, dpsi_full+spn_i*size_s, 2, pSPARC->dmcomm);
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = YorZ + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_full + spn_i * size_s + n * DMnd; // dpsi_3
                    dpsi1_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi3 += vxcMGGA3_loc[i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_mGGA_psi[2] += dpsi1_dpsi3 * g_nk; // component (1,3)
                }
            } else if(dim == 1){
                // component (2,3)
                double *dpsi_3p; 
                
                dpsi_3p = dpsi_full;
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = YorZ + spn_i * size_s + n * DMnd; // dpsi_2
                    dpsi_ptr2 = dpsi_3p + spn_i * size_s + n * DMnd; // dpsi_3
                    dpsi2_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi2_dpsi3 += vxcMGGA3_loc[i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_mGGA_psi[4] += dpsi2_dpsi3 * g_nk; // component (2,3)
                }

                // component (2,2)
                double *dpsi_2p = YorZ; 
                
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = YorZ + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_2p + spn_i * size_s + n * DMnd; // dpsi_1p
                    dpsi2_dpsi2 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi2_dpsi2 += vxcMGGA3_loc[i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_mGGA_psi[3] += dpsi2_dpsi2 * g_nk; // component (2,2)
                }
            } else if (dim == 2) {
                // component (3,3)
                double *dpsi_3p = YorZ; 
                
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = YorZ + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_3p + spn_i * size_s + n * DMnd; // dpsi_1p
                    dpsi3_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi3_dpsi3 += vxcMGGA3_loc[i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_mGGA_psi[5] += dpsi3_dpsi3 * g_nk; // component (3,3)
                }
            }
        }    
        count++;
    }

    stress_mGGA_psi[0] *= -(2.0/pSPARC->Nspin); // component (1,1)
    stress_mGGA_psi[1] *= -(2.0/pSPARC->Nspin); // component (1,2)
    stress_mGGA_psi[2] *= -(2.0/pSPARC->Nspin); // component (1,3)
    stress_mGGA_psi[3] *= -(2.0/pSPARC->Nspin); // component (2,2)
    stress_mGGA_psi[4] *= -(2.0/pSPARC->Nspin); // component (2,3)
    stress_mGGA_psi[5] *= -(2.0/pSPARC->Nspin); // component (3,3)

    free(dpsi_full);
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if(pSPARC->cell_typ == 0){
        for(i = 0; i < 6; i++)
            stress_mGGA_psi_cartesian[i] = stress_mGGA_psi[i];
    } else {
        double *c_g, *c_c;
        c_g = (double*) malloc(9*sizeof(double));
        c_c = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
                c_c[3*i+j] = pSPARC->LatUVec[3*j+i];
            }
        }

        stress_mGGA_psi_cartesian[0] = c_g[0] * (c_g[0] * stress_mGGA_psi[0] + c_g[1] * stress_mGGA_psi[1] + c_g[2] * stress_mGGA_psi[2]) +
                              c_g[1] * (c_g[0] * stress_mGGA_psi[1] + c_g[1] * stress_mGGA_psi[3] + c_g[2] * stress_mGGA_psi[4]) +
                              c_g[2] * (c_g[0] * stress_mGGA_psi[2] + c_g[1] * stress_mGGA_psi[4] + c_g[2] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[1] = c_g[0] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                              c_g[1] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                              c_g[2] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[2] = c_g[0] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                              c_g[1] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                              c_g[2] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[3] = c_g[3] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                              c_g[4] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                              c_g[5] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[4] = c_g[3] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                              c_g[4] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                              c_g[5] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[5] = c_g[6] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                              c_g[7] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                              c_g[8] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);                                                                                   
  
        free(c_g); free(c_c);
    }


    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(stress_mGGA_psi_cartesian, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }
    
    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(stress_mGGA_psi_cartesian, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }

    if (pSPARC->Calc_stress == 1) {
        if (!rank) {
            // Define measure of unit cell
            double cell_measure = pSPARC->Jacbdet;
            if(pSPARC->BCx == 0)
                cell_measure *= pSPARC->range_x;
            if(pSPARC->BCy == 0)
                cell_measure *= pSPARC->range_y;
            if(pSPARC->BCz == 0)
                cell_measure *= pSPARC->range_z;
            printf("\npSPARC->dV=%12.9E, cell_measure=%12.9E\n", pSPARC->dV, cell_measure);
            for(int i = 0; i < 6; i++) {
                stress_mGGA_psi_cartesian[i] /= cell_measure;
            }
            #ifdef DEBUG
            printf("\nmetaGGA psi stress term (which is not the total contribution of metaGGA on stress!) (GPa):\n");
            printf("%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n", 
                    stress_mGGA_psi_cartesian[0]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[3]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[5]*CONST_HA_BOHR3_GPA);
            #endif
            // add metaGGA psi stress term to exchange-correlation stress
            pSPARC->stress_xc[0] += stress_mGGA_psi_cartesian[0]; // (1,1)
            pSPARC->stress_xc[1] += stress_mGGA_psi_cartesian[1]; // (1,2)
            pSPARC->stress_xc[2] += stress_mGGA_psi_cartesian[2]; // (1,3)
            pSPARC->stress_xc[3] += stress_mGGA_psi_cartesian[3]; // (2,2)
            pSPARC->stress_xc[4] += stress_mGGA_psi_cartesian[4]; // (2,3)
            pSPARC->stress_xc[5] += stress_mGGA_psi_cartesian[5]; // (3,3)
        }
    } 

    if (pSPARC->Calc_pres == 1) {
        if(!rank) {
            pSPARC->pres_xc += (stress_mGGA_psi_cartesian[0] + stress_mGGA_psi_cartesian[3] + stress_mGGA_psi_cartesian[5]);
        }
    }
    free(stress_mGGA_psi);
    free(stress_mGGA_psi_cartesian);
}

/**
 * @brief   compute the metaGGA psi stress term, which is 
 * \int_{Omega}(\rho\frac{\partial \epsilon_{xc}}{\partial \tau})[\sum_{n=1}^N_s g_n(2\nabla_{\alpha}\psi_n * \nabla_{\beta}\psi_n)] dx
 * called by Calculate_electronic_stress in stress.c
 * Attention: it is NOT the full contrubution of metaGGA on stress!
 */
void Calculate_XC_stress_mGGA_psi_term_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, j, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, dim, count, l, m, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;

    double complex *psi_ptr, *dpsi_ptr, *dpsi_ptr2, *dpsi_full;
    double *temp_k, *stress_mGGA_psi, *stress_mGGA_psi_cartesian, g_nk, kptwt;
    double dpsi1_dpsi1, dpsi1_dpsi2, dpsi1_dpsi3, dpsi2_dpsi2, dpsi2_dpsi3, dpsi3_dpsi3;

    double energy_nl = 0.0;
    temp_k = (double*) malloc(6 * sizeof(double));
    stress_mGGA_psi = (double*) calloc(6, sizeof(double));
    stress_mGGA_psi_cartesian = (double*) calloc(6, sizeof(double));

    // dpsi_full = (double complex *)malloc( size_s * nspin * sizeof(double complex) );
    dpsi_full = (double complex *)malloc( size_k * sizeof(double complex) );
    if (dpsi_full == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    double *vxcMGGA3_loc = pSPARC->vxcMGGA3_loc_dmcomm; // local rho*d\epsilon_{xc} / d\tau array
    
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, kpt_vec;
    double complex bloch_fac, a, b;


    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            for (dim = 0; dim < 3; dim++) {
                // find dPsi in direction dim
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, dim, kpt_vec, pSPARC->dmcomm);

                // Kinetic stress
                if(dim == 0){
                    kpt_vec = k2;
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, dpsi_full, 1, kpt_vec, pSPARC->dmcomm);
                    //ts = MPI_Wtime();
                    temp_k[0] = temp_k[1] = temp_k[3] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_1
                        dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_2
                        dpsi1_dpsi1 = dpsi1_dpsi2 = dpsi2_dpsi2 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi1_dpsi1 += vxcMGGA3_loc[i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr + i)));
                            dpsi1_dpsi2 += vxcMGGA3_loc[i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i)));
                            dpsi2_dpsi2 += vxcMGGA3_loc[i] * (creal(*(dpsi_ptr2 + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr2 + i)) * cimag(*(dpsi_ptr2 + i)));
                        }
                        g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                        temp_k[0] += dpsi1_dpsi1 * g_nk;
                        temp_k[1] += dpsi1_dpsi2 * g_nk;
                        temp_k[3] += dpsi2_dpsi2 * g_nk;
                    }
                    stress_mGGA_psi[0] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[0];
                    stress_mGGA_psi[1] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[1];
                    stress_mGGA_psi[3] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[3];

                    kpt_vec = k3;
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, dpsi_full, 2, kpt_vec, pSPARC->dmcomm);
                    temp_k[2] = temp_k[5] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_1
                        dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_3
                        dpsi1_dpsi3 = dpsi3_dpsi3 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi1_dpsi3 += vxcMGGA3_loc[i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i)));
                            dpsi3_dpsi3 += vxcMGGA3_loc[i] * (creal(*(dpsi_ptr2 + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr2 + i)) * cimag(*(dpsi_ptr2 + i)));
                        }
                        g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                        temp_k[2] += dpsi1_dpsi3 * g_nk;
                        temp_k[5] += dpsi3_dpsi3 * g_nk;
                    }
                    stress_mGGA_psi[2] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[2];
                    stress_mGGA_psi[5] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[5];
                } else if(dim == 1){
                    temp_k[4] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_2
                        dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_3
                        dpsi2_dpsi3 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi2_dpsi3 += vxcMGGA3_loc[i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i)));
                        }
                        temp_k[4] += dpsi2_dpsi3 * pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                    }
                    stress_mGGA_psi[4] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[4];
                }
            }
            count++;
        }    
    }
    
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if(pSPARC->cell_typ == 0){
        for(i = 0; i < 6; i++)
            stress_mGGA_psi_cartesian[i] = stress_mGGA_psi[i];
    } else {
        double *c_g, *c_c;
        c_g = (double*) malloc(9*sizeof(double));
        c_c = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
                c_c[3*i+j] = pSPARC->LatUVec[3*j+i];
            }
        }
        stress_mGGA_psi_cartesian[0] = c_g[0] * (c_g[0] * stress_mGGA_psi[0] + c_g[1] * stress_mGGA_psi[1] + c_g[2] * stress_mGGA_psi[2]) +
                                       c_g[1] * (c_g[0] * stress_mGGA_psi[1] + c_g[1] * stress_mGGA_psi[3] + c_g[2] * stress_mGGA_psi[4]) +
                                       c_g[2] * (c_g[0] * stress_mGGA_psi[2] + c_g[1] * stress_mGGA_psi[4] + c_g[2] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[1] = c_g[0] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                                       c_g[1] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                                       c_g[2] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[2] = c_g[0] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                                       c_g[1] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                                       c_g[2] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[3] = c_g[3] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                                       c_g[4] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                                       c_g[5] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[4] = c_g[3] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                                       c_g[4] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                                       c_g[5] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[5] = c_g[6] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                                       c_g[7] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                                       c_g[8] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);                                                                                   
  
        free(c_g); free(c_c);
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(stress_mGGA_psi_cartesian, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }
    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }
    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(stress_mGGA_psi_cartesian, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }

    if (pSPARC->Calc_stress == 1) {
        if (!rank) {
            // Define measure of unit cell
            double cell_measure = pSPARC->Jacbdet;
            if(pSPARC->BCx == 0)
                cell_measure *= pSPARC->range_x;
            if(pSPARC->BCy == 0)
                cell_measure *= pSPARC->range_y;
            if(pSPARC->BCz == 0)
                cell_measure *= pSPARC->range_z;
            printf("\npSPARC->dV=%12.9E, cell_measure=%12.9E\n", pSPARC->dV, cell_measure);
            for(int i = 0; i < 6; i++) {
                stress_mGGA_psi_cartesian[i] /= cell_measure;
            }
            #ifdef DEBUG
            printf("\nmetaGGA psi stress term (which is not the total contribution of metaGGA on stress!) (GPa):\n");
            printf("%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n", 
                    stress_mGGA_psi_cartesian[0]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[3]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[5]*CONST_HA_BOHR3_GPA);
            #endif
            // add metaGGA psi stress term to exchange-correlation stress
            pSPARC->stress_xc[0] += stress_mGGA_psi_cartesian[0]; // (1,1)
            pSPARC->stress_xc[1] += stress_mGGA_psi_cartesian[1]; // (1,2)
            pSPARC->stress_xc[2] += stress_mGGA_psi_cartesian[2]; // (1,3)
            pSPARC->stress_xc[3] += stress_mGGA_psi_cartesian[3]; // (2,2)
            pSPARC->stress_xc[4] += stress_mGGA_psi_cartesian[4]; // (2,3)
            pSPARC->stress_xc[5] += stress_mGGA_psi_cartesian[5]; // (3,3)
        }
    } 

    if (pSPARC->Calc_pres == 1) {
        if(!rank) {
            pSPARC->pres_xc += (stress_mGGA_psi_cartesian[0] + stress_mGGA_psi_cartesian[3] + stress_mGGA_psi_cartesian[5]);
        }
    }
    free(temp_k);
    
    free(stress_mGGA_psi);
    free(stress_mGGA_psi_cartesian);
}


/**
 * @brief   free space allocated to MGGA variables
 */
void free_MGGA(SPARC_OBJ *pSPARC) { // free space allocated to MGGA variables
    free(pSPARC->KineticTauPhiDomain);
    free(pSPARC->vxcMGGA1);
    free(pSPARC->vxcMGGA2);
    free(pSPARC->vxcMGGA3);
    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) { // d(n\epsilon)/d\tau in dmcomm
        free(pSPARC->vxcMGGA3_loc_dmcomm);
    }
    free(pSPARC->vxcMGGA3_loc_kptcomm);
}