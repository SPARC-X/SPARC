/**
 * @file    MGGAinitialization.c
 * @brief   This file contains the function initializing variables needed by MGGA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "isddft.h"
#include "mGGAinitialization.h"

/**
 * @brief   allocate space to MGGA variables
 */
void initialize_MGGA(SPARC_OBJ *pSPARC) {
    int DMnd;
    
    DMnd = pSPARC->Nd_d;
    pSPARC->KineticTauPhiDomain = (double *) calloc(DMnd * (2*pSPARC->Nspin-1), sizeof(double)); // different from pSPARC->KineticRho, which is in dmcomm
    assert(pSPARC->KineticTauPhiDomain != NULL);
    
    pSPARC->vxcMGGA1 = (double*)malloc(sizeof(double)*DMnd * pSPARC->Nspin); // d(n\epsilon)/dn in dmcomm_phi
    assert(pSPARC->vxcMGGA1 != NULL);
    pSPARC->vxcMGGA2 = (double*)malloc(sizeof(double)*DMnd * (2*pSPARC->Nspin-1)); // d(n\epsilon)/d|grad n| in dmcomm_phi
    assert(pSPARC->vxcMGGA2 != NULL);
    pSPARC->vxcMGGA3 = (double*)calloc(DMnd * pSPARC->Nspin, sizeof(double)); // d(n\epsilon)/d\tau in dmcomm_phi
    assert(pSPARC->vxcMGGA3 != NULL);
    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) { // d(n\epsilon)/d\tau in dmcomm
        pSPARC->vxcMGGA3_loc_dmcomm = (double *)calloc( pSPARC->Nd_d_dmcomm * pSPARC->Nspin, sizeof(double) ); // spin polarization, every processor in dmcomm has potential of both spin up and spin dn
        assert(pSPARC->vxcMGGA3_loc_dmcomm != NULL);
    }
    pSPARC->vxcMGGA3_loc_kptcomm = (double *)calloc( pSPARC->Nd_d_kptcomm, sizeof(double) ); // d(n\epsilon)/d\tau in kptcomm 
    // space can only contain potential of one spin. TODO: move the transferring function to eigensolver.c. 
    // If the processor in dmcomm contains both spin up and dn, potential up and dn need to be transferred separately. Reference: electronicGroundState.c, 574
    assert(pSPARC->vxcMGGA3_loc_kptcomm != NULL);

    pSPARC->countPotentialCalculate = 0;
}