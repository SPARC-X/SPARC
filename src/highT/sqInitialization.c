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
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int rank, FDn;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);
    
    pSQ->SqInd = (SQIND *) malloc(sizeof(SQIND));
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
    pSQ->Nx_loc = 1+2*pSQ->nloc[0];
    pSQ->Ny_loc = 1+2*pSQ->nloc[1];
    pSQ->Nz_loc = 1+2*pSQ->nloc[2];
    pSQ->Nd_loc = pSQ->Nx_loc * pSQ->Ny_loc * pSQ->Nz_loc;

    pSQ->DMnx_PR = pSQ->DMnx_SQ + 2*pSQ->nloc[0];
    pSQ->DMny_PR = pSQ->DMny_SQ + 2*pSQ->nloc[1];
    pSQ->DMnz_PR = pSQ->DMnz_SQ + 2*pSQ->nloc[2];
    pSQ->DMnd_PR  = pSQ->DMnx_PR * pSQ->DMny_PR * pSQ->DMnz_PR;

    // Number of FD nods in local extend domain
    pSQ->Nx_ex = pSQ->Nx_loc + pSPARC->order;
    pSQ->Ny_ex = pSQ->Ny_loc + pSPARC->order;
    pSQ->Nz_ex = pSQ->Nz_loc + pSPARC->order;
    pSQ->Nd_ex = pSQ->Nx_ex * pSQ->Ny_ex * pSQ->Nz_ex;

    // allocate variables 
    pSQ->Veff_PR = (double *) calloc(sizeof(double), pSQ->DMnd_PR);
    pSQ->Veff_loc_SQ = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);
    pSQ->x_ex = (double *) malloc(sizeof(double)*pSQ->Nd_ex);
    memset(pSQ->x_ex, 0, sizeof(double)*pSQ->Nd_ex);
    
    // Get coordinates for each process in kptcomm_topo
    MPI_Cart_coords(pSQ->dmcomm_SQ, rank, 3, pSQ->coords);

    // Get information of process configuration
    pSQ->rem[0] = pSPARC->Nx % pSPARC->npNdx_SQ;
    pSQ->rem[1] = pSPARC->Ny % pSPARC->npNdy_SQ;
    pSQ->rem[2] = pSPARC->Nz % pSPARC->npNdz_SQ;

    int DMnxnynz[3] = {pSQ->DMnx_SQ, pSQ->DMny_SQ, pSQ->DMnz_SQ};
    int npNd[3] = {pSPARC->npNdx_SQ, pSPARC->npNdy_SQ, pSPARC->npNdz_SQ};
    // Creating communication topology for PR domain
    Comm_topologies_sq(pSPARC, DMnxnynz, npNd, pSQ->dmcomm_SQ, &pSQ->SQ_dist_graph_comm);

    pSQ->mineig = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);
    pSQ->maxeig = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);
    pSQ->lambda_max = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);

    pSQ->gnd = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
    pSQ->gwt = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
    for (int i = 0; i < pSQ->DMnd_SQ; i++) {
        pSQ->gnd[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
        pSQ->gwt[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
    }
    
    if (pSPARC->SQ_gauss_mem == 1) {                    // save vectors for all FD nodes
        pSQ->lanczos_vec_all = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
        assert(pSQ->lanczos_vec_all != NULL);
        for (int i = 0; i < pSQ->DMnd_SQ; i++) {
            pSQ->lanczos_vec_all[i] = (double *) calloc(sizeof(double), pSQ->Nd_loc * pSPARC->SQ_npl_g);
            assert(pSQ->lanczos_vec_all[i] != NULL);
        }

        pSQ->w_all = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
        assert(pSQ->w_all != NULL);
        for (int i = 0; i < pSQ->DMnd_SQ; i++) {
            pSQ->w_all[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g * pSPARC->SQ_npl_g);
            assert(pSQ->w_all[i] != NULL);
        }
    } else {
        pSQ->lanczos_vec = (double *) calloc(sizeof(double), pSQ->Nd_loc * pSPARC->SQ_npl_g);
        assert(pSQ->lanczos_vec != NULL);

        pSQ->w = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g * pSPARC->SQ_npl_g);
        assert(pSQ->w != NULL);
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
    pSQ->forceFlag = 0;
}
