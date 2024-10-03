/***
 * @file    exactExchangeFinalization.c
 * @brief   This file contains the functions for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "exactExchangeFinalization.h"
#include "exactExchangeInitialization.h"
#include "electrostatics.h"
#include "parallelization.h"
#include "sqFinalization.h"
#include "kroneckerLaplacian.h"

/**
 * @brief   Memory free of all variables for exact exchange.
 */
void free_exx(SPARC_OBJ *pSPARC) {
    if (pSPARC->sqHighTFlag == 1) {
        free_exx_SQ(pSPARC);
        return;
    }

    free(pSPARC->k1_hf);
    free(pSPARC->k2_hf);
    free(pSPARC->k3_hf);
    free(pSPARC->kpthf_ind);
    free(pSPARC->kpthf_ind_red);
    free(pSPARC->kpthfred2kpthf);
    free(pSPARC->kpthf_pn);
    free(pSPARC->kpts_hf_red_list);
    free(pSPARC->k1_shift);
    free(pSPARC->k2_shift);
    free(pSPARC->k3_shift);
    free(pSPARC->Kptshift_map);
    
    free(pSPARC->kpthf_flag_kptcomm);
    free(pSPARC->Nkpts_hf_list);
    free(pSPARC->kpthf_start_indx_list);    
    if (pSPARC->ExxAcc == 0) {
        if (pSPARC->isGammaPoint == 1) {
            free(pSPARC->psi_outer);
            free(pSPARC->psi_outer_kptcomm_topo);
            free(pSPARC->occ_outer);
        } else {
            free(pSPARC->psi_outer_kpt);
            free(pSPARC->psi_outer_kptcomm_topo_kpt);
            free(pSPARC->occ_outer);
        }
    } else {
        if (pSPARC->isGammaPoint == 1) {
            if (pSPARC->spincomm_index >= 0) {
                free(pSPARC->Xi);
                free(pSPARC->Xi_kptcomm_topo);
            }
        } else {
            free(pSPARC->occ_outer);
            if (pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index >= 0) {
                free(pSPARC->Xi_kpt);
                free(pSPARC->Xi_kptcomm_topo_kpt);
            }
        }
    }
    
    if (pSPARC->dmcomm != MPI_COMM_NULL || pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        free_singularity_removal_const(pSPARC);
        if (pSPARC->ExxMethod == 1) {
            for (int i = 0; i < pSPARC->Nkpts_shift; i++) {
                free_kron_Lap(pSPARC->kron_lap_exx[i]);
                free(pSPARC->kron_lap_exx[i]);
            }
            free(pSPARC->kron_lap_exx);
            if (pSPARC->BC == 1) {
                free_multipole_expansion(pSPARC->MpExp_exx, MPI_COMM_SELF);
                free(pSPARC->MpExp_exx);
            }
        }
    }

    if (pSPARC->Nkpts_shift > 1) {
        free(pSPARC->neg_phase);
        free(pSPARC->pos_phase);
    }

    if (pSPARC->kpttopo_dmcomm_inter != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->kpttopo_dmcomm_inter);
}


void free_singularity_removal_const(SPARC_OBJ *pSPARC)
{
    if (pSPARC->ExxMethod == 1 && pSPARC->BC == 1) return;

    free(pSPARC->pois_const);
    if (pSPARC->Calc_stress == 1) {
        free(pSPARC->pois_const_stress);
        if (pSPARC->ExxDivFlag == 0)
            free(pSPARC->pois_const_stress2);
    } else if (pSPARC->Calc_pres == 1) {
        if (pSPARC->ExxDivFlag != 0)
            free(pSPARC->pois_const_press);
    }
}