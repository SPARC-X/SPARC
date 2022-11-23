/**
 * @file    MGGAfinalization.c
 * @brief   This file contains the function initializing variables needed by MGGA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "isddft.h"
#include "mGGAfinalization.h"

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