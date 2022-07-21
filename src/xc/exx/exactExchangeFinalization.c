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


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Memory free of all variables for exact exchange.
 */
void free_exx(SPARC_OBJ *pSPARC) {
    int blacs_size, kpt_bridge_size;
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);
    
    free(pSPARC->kpthf_flag_kptcomm);
    free(pSPARC->Nkpts_hf_list);
    free(pSPARC->kpthf_start_indx_list);
    if (pSPARC->ACEFlag == 0) {
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

    if (pSPARC->EXXMeth_Flag == 0) {
        if (pSPARC->dmcomm != MPI_COMM_NULL || pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            free(pSPARC->pois_FFT_const);
            if (pSPARC->Calc_stress == 1) {
                free(pSPARC->pois_FFT_const_stress);
                if (pSPARC->EXXDiv_Flag == 0)
                    free(pSPARC->pois_FFT_const_stress2);
            } else if (pSPARC->Calc_pres == 1) {
                if (pSPARC->EXXDiv_Flag != 0)
                    free(pSPARC->pois_FFT_const_press);
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

