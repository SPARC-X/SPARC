/***
 * @file    sqInitialization.c
 * @brief   This file contains the functions for SQ method.
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
#include <time.h>

#include "sqInitialization.h"
#include "sqParallelization.h"

/**
 * @brief   Initialize SQ variables and create necessary communicators
 */
void init_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    SQIND *SqInd = pSPARC->pSQ->SqInd;
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int i, j, k, rank, size, FDn, DMnx, DMny, DMnz, DMnd;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);

    ////////////////////////////////////////////////////////////////////////////////////////////////

    pSQ->SqInd = (SQIND *) malloc(sizeof(SQIND));
    DMnx = pSQ->Nx_d_SQ;
    DMny = pSQ->Ny_d_SQ;
    DMnz = pSQ->Nz_d_SQ;
    DMnd = pSQ->Nd_d_SQ;

    // start to get variables
    pSQ->nloc[0] = ceil(pSPARC->SQ_rcut / pSPARC->delta_x);
    pSQ->nloc[1] = ceil(pSPARC->SQ_rcut / pSPARC->delta_y);
    pSQ->nloc[2] = ceil(pSPARC->SQ_rcut / pSPARC->delta_z);
    FDn = pSPARC->order / 2;
    if (pSQ->nloc[0] < FDn || pSQ->nloc[1] < FDn || pSQ->nloc[2] < FDn) {
        if(!rank) 
            printf(RED "ERROR: Rcut/delta is smaller than FD_ORDER/2.\n"
                "Please use finer mesh or larger Rcut!\n" RESET);
        exit(EXIT_FAILURE);
    }

    // Number of FD nodes in local Rcut domain and PR domain 
    pSQ->Nd_loc = (1+2*pSQ->nloc[0]) * (1+2*pSQ->nloc[1]) * (1+2*pSQ->nloc[2]);
    pSQ->Nd_PR  = (DMnz+2*pSQ->nloc[2]) 
                * (DMny+2*pSQ->nloc[1]) * (DMnx+2*pSQ->nloc[0]);
    
    pSQ->Veff_PR = (double ***) calloc(sizeof(double**), DMnz+2*pSQ->nloc[2]);
	assert(pSQ->Veff_PR != NULL);
	for (k = 0; k < DMnz+2*pSQ->nloc[2]; k++) {
        pSQ->Veff_PR[k] = (double **) calloc(sizeof(double*), DMny+2*pSQ->nloc[1]);
        assert(pSQ->Veff_PR[k] != NULL);
        for (j = 0; j < DMny+2*pSQ->nloc[1]; j++) {
            pSQ->Veff_PR[k][j] = (double *) calloc(sizeof(double), DMnx+2*pSQ->nloc[0]);
            assert(pSQ->Veff_PR[k][j] != NULL);
        }
    }

    // Veff in local dmcomm_SQ
    pSQ->Veff_loc_SQ = (double *) calloc(sizeof(double), pSQ->Nd_d_SQ);

    // Get coordinates for each process in kptcomm_topo
    MPI_Cart_coords(pSQ->dmcomm_SQ, rank, 3, pSQ->coords);

    // Get information of process configuration
    pSQ->rem[0] = pSPARC->Nx % pSPARC->npNdx_SQ;
    pSQ->rem[1] = pSPARC->Ny % pSPARC->npNdy_SQ;
    pSQ->rem[2] = pSPARC->Nz % pSPARC->npNdz_SQ;

    int DMnxnynz[3] = {DMnx, DMny, DMnz};
    int npNd[3] = {pSPARC->npNdx_SQ, pSPARC->npNdy_SQ, pSPARC->npNdz_SQ};
    // Creating communication topology for PR domain
    Comm_topologies_sq(pSPARC, DMnxnynz, npNd, pSQ->dmcomm_SQ, &pSQ->SQ_dist_graph_comm);

    pSQ->mineig = (double *) calloc(sizeof(double), pSQ->Nd_d_SQ);
    pSQ->maxeig = (double *) calloc(sizeof(double), pSQ->Nd_d_SQ);
    pSQ->lambda_max = (double *) calloc(sizeof(double), pSQ->Nd_d_SQ);
    pSQ->chi = (double *) calloc(sizeof(double), pSQ->Nd_d_SQ);
    pSQ->zee = (double *) calloc(sizeof(double), pSQ->Nd_d_SQ);

    pSQ->gnd = (double **) calloc(sizeof(double *), pSQ->Nd_d_SQ);
    pSQ->gwt = (double **) calloc(sizeof(double *), pSQ->Nd_d_SQ);
    pSQ->Ci = (double **) calloc(sizeof(double *), pSQ->Nd_d_SQ);
    for (i = 0; i < pSQ->Nd_d_SQ; i++) {
        pSQ->gnd[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
        pSQ->gwt[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
        pSQ->Ci[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_c + 1);
    }
    
    if (pSPARC->SQ_typ_dm == 2) { 
        if (pSPARC->SQ_gauss_mem == 1) {                    // save vectors for all FD nodes
            pSQ->lanczos_vec_all = (double **) calloc(sizeof(double *), pSQ->Nd_d_SQ);
            assert(pSQ->lanczos_vec_all != NULL);
            for (i = 0; i < pSQ->Nd_d_SQ; i++) {
                pSQ->lanczos_vec_all[i] = (double *) calloc(sizeof(double), pSQ->Nd_loc * pSPARC->SQ_npl_g);
                assert(pSQ->lanczos_vec_all[i] != NULL);
            }

            pSQ->w_all = (double **) calloc(sizeof(double *), pSQ->Nd_d_SQ);
            assert(pSQ->w_all != NULL);
            for (i = 0; i < pSQ->Nd_d_SQ; i++) {
                pSQ->w_all[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g * pSPARC->SQ_npl_g);
                assert(pSQ->w_all[i] != NULL);
            }
        } else {
            pSQ->lanczos_vec = (double *) calloc(sizeof(double), pSQ->Nd_loc * pSPARC->SQ_npl_g);
            assert(pSQ->lanczos_vec != NULL);

            pSQ->w = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g * pSPARC->SQ_npl_g);
            assert(pSQ->w != NULL);
        }
    }

    if (pSPARC->SQ_typ_dm == 1) {
        pSQ->vec = (double ****) calloc(sizeof(double ***), pSQ->Nd_d_SQ);
        for (i = 0; i < pSQ->Nd_d_SQ; i++) {
            pSQ->vec[i] = (double ***) calloc(sizeof(double **), 2*pSQ->nloc[2]+1);
            for (k = 0; k < 2*pSQ->nloc[2]+1; k++) {
                pSQ->vec[i][k] = (double **) calloc(sizeof(double *), 2*pSQ->nloc[1]+1);
                for (j = 0; j < 2*pSQ->nloc[1]+1; j++) {
                    pSQ->vec[i][k][j] = (double *) calloc(sizeof(double), 2*pSQ->nloc[0]+1);
                }
            }
        }
    }

    pSQ->DMVertices_PR[0] = pSQ->DMVertices_SQ[0] - pSQ->nloc[0];
    pSQ->DMVertices_PR[1] = pSQ->DMVertices_SQ[1] + pSQ->nloc[0];
    pSQ->DMVertices_PR[2] = pSQ->DMVertices_SQ[2] - pSQ->nloc[1];
    pSQ->DMVertices_PR[3] = pSQ->DMVertices_SQ[3] + pSQ->nloc[1];
    pSQ->DMVertices_PR[4] = pSQ->DMVertices_SQ[4] - pSQ->nloc[2];
    pSQ->DMVertices_PR[5] = pSQ->DMVertices_SQ[5] + pSQ->nloc[2];

#ifdef DEBUG
    if (!rank)
    printf("rank %d, DMVertices_PR [%d, %d, %d, %d, %d, %d], nloc [%d, %d, %d]\n", rank,
            pSQ->DMVertices_PR[0], pSQ->DMVertices_PR[1], pSQ->DMVertices_PR[2], 
            pSQ->DMVertices_PR[3], pSQ->DMVertices_PR[4], pSQ->DMVertices_PR[5],
            pSQ->nloc[0], pSQ->nloc[1], pSQ->nloc[2]);
#endif
}
