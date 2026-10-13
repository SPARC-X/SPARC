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

/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "sqInitialization.h"
#include "parallelization.h"
#include "electrostatics.h"
#include "sqHighTExactExchange.h"
#include "sqFinalization.h"
#include "kroneckerLaplacian.h"
#include "tools.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/**
 * @brief   Initialize SQ variables and create necessary communicators
 */
void init_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int rank, FDn;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);

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
    pSQ->Veff_PR = (double *) calloc(sizeof(double), (size_t)pSQ->DMnd_PR * pSPARC->Nspin_spincomm);   // one slice per spin
    pSQ->Veff_loc_SQ = (double *) calloc(sizeof(double), pSQ->DMnd_SQ*pSPARC->Nspden);
    pSQ->x_ex = (double *) malloc(sizeof(double)*pSQ->Nd_ex);
    memset(pSQ->x_ex, 0, sizeof(double)*pSQ->Nd_ex);
    init_SQ_lanczos(pSPARC);

    // initialize for d2dext communication
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int dims[3] = {pSPARC->npNdx_SQ, pSPARC->npNdy_SQ, pSPARC->npNdz_SQ};
    pSQ->d2dext_dmcomm_sq = (D2Dext_OBJ *) malloc(sizeof(D2Dext_OBJ));
    pSQ->d2dext_dmcomm_sq_ext = (D2Dext_OBJ *) malloc(sizeof(D2Dext_OBJ));
    Set_D2Dext_Target(pSQ->d2dext_dmcomm_sq, pSQ->d2dext_dmcomm_sq_ext, 
        pSQ->DMnx_SQ, pSQ->DMny_SQ, pSQ->DMnz_SQ, pSQ->nloc[0], pSQ->nloc[1], pSQ->nloc[2],
        gridsizes, dims, pSQ->dmcomm_SQ, &pSQ->dmcomm_d2dext_sq);

    pSQ->mineig = (double *) calloc(sizeof(double), pSQ->DMnd_SQ * pSPARC->Nspin_spincomm);
    pSQ->maxeig = (double *) calloc(sizeof(double), pSQ->DMnd_SQ * pSPARC->Nspin_spincomm);
    pSQ->lambda_max = (double *) calloc(sizeof(double), pSQ->DMnd_SQ * pSPARC->Nspin_spincomm);

    pSQ->gnd = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ * pSPARC->Nspin_spincomm);
    pSQ->gwt = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ * pSPARC->Nspin_spincomm);
    for (int i = 0; i < pSQ->DMnd_SQ * pSPARC->Nspin_spincomm; i++) {
        pSQ->gnd[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
        pSQ->gwt[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
    }
    
    pSQ->lanczos_vec = (double *) calloc((size_t)pSQ->Nd_loc * (size_t)pSPARC->SQ_npl_g, sizeof(double));
    assert(pSQ->lanczos_vec != NULL);

    pSQ->w = (double *) calloc(pSPARC->SQ_npl_g * pSPARC->SQ_npl_g, sizeof(double));
    assert(pSQ->w != NULL);

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


/**
 * @brief   Allocate what the Lanczos runs of the SQ method use (init_SQ and init_SQ_HighT): the point's Veff on its
 *          Rcut box, the last H v of a run, the 1D Dirichlet Laplacians of the box (Kronecker form of the
 *          Laplacian), the workspace of the tridiagonal eigensolver (dstedc), the JIT kernel tables (MKL,
 *          created on first use) and the buffers of the current point's nonlocal projectors (grow on use)
 */
void init_SQ_lanczos(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int FDn = pSPARC->order / 2;
    size_t N = pSPARC->SQ_npl_g;

    pSQ->Veff_box = (double *) malloc((size_t)pSQ->Nd_loc * sizeof(double));
    pSQ->lanczos_tmp = (double *) malloc((size_t)pSQ->Nd_loc * sizeof(double));
    assert(pSQ->Veff_box != NULL && pSQ->lanczos_tmp != NULL);

    pSQ->Lap_1D_x = (double *) calloc((size_t)pSQ->Nx_loc * pSQ->Nx_loc, sizeof(double));
    pSQ->Lap_1D_y = (double *) calloc((size_t)pSQ->Ny_loc * pSQ->Ny_loc, sizeof(double));
    pSQ->Lap_1D_z = (double *) calloc((size_t)pSQ->Nz_loc * pSQ->Nz_loc, sizeof(double));
    assert(pSQ->Lap_1D_x != NULL && pSQ->Lap_1D_y != NULL && pSQ->Lap_1D_z != NULL);
    Lap_1D_Dirichlet(FDn, pSPARC->D2_stencil_coeffs_x, pSQ->Nx_loc, pSQ->Lap_1D_x);
    Lap_1D_Dirichlet(FDn, pSPARC->D2_stencil_coeffs_y, pSQ->Ny_loc, pSQ->Lap_1D_y);
    Lap_1D_Dirichlet(FDn, pSPARC->D2_stencil_coeffs_z, pSQ->Nz_loc, pSQ->Lap_1D_z);

    pSQ->eig_lwork = 1 + 4*(long)N + (long)N*N;          // dstedc 'I'
    pSQ->eig_liwork = 3 + 5*(long)N;
    pSQ->eig_d = (double *) malloc(N * sizeof(double));
    pSQ->eig_e = (double *) malloc(N * sizeof(double));
    pSQ->eig_work = (double *) malloc((size_t)pSQ->eig_lwork * sizeof(double));
    pSQ->eig_iwork = (int *) malloc((size_t)pSQ->eig_liwork * sizeof(int));
    pSQ->eig_C = (double *) malloc(N * sizeof(double));
    pSQ->eig_F = (double *) malloc(N * sizeof(double));
    assert(pSQ->eig_d != NULL && pSQ->eig_e != NULL && pSQ->eig_work != NULL && pSQ->eig_iwork != NULL
        && pSQ->eig_C != NULL && pSQ->eig_F != NULL);

    pSQ->jit_state = pSQ->jits_state = 0;
    pSQ->jitnl = NULL; pSQ->jitnl_n = pSQ->jitnl_cap = 0;

    pSQ->nlocProj_pt = (NLOC_PROJ_OBJ *) calloc(pSPARC->Ntypes, sizeof(NLOC_PROJ_OBJ));
    pSQ->Atom_Influence_nloc_pt = (ATOM_NLOC_INFLUENCE_OBJ *) calloc(pSPARC->Ntypes, sizeof(ATOM_NLOC_INFLUENCE_OBJ));
    pSQ->nlpt_cap = (int *) calloc(pSPARC->Ntypes, sizeof(int));
    assert(pSQ->nlocProj_pt != NULL && pSQ->Atom_Influence_nloc_pt != NULL && pSQ->nlpt_cap != NULL);
    pSQ->nlpt_pos = NULL; pSQ->nlpt_chi = NULL;
    pSQ->nlpt_pos_cap = pSQ->nlpt_chi_cap = 0;
}


double memory_estimation_SQ(const SPARC_OBJ *pSPARC)
{
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    SQ_OBJ *pSQ = pSPARC->pSQ;
    // TODO: add accurate memory estimation
    double mem_PR = (double) sizeof(double) * pSQ->DMnd_PR * pSPARC->Nspin_spincomm * nproc;
    double mem_phi = (double) sizeof(double) * pSPARC->Nd_d * (6+pSPARC->SQ_npl_g*2+1) * pSPARC->Nspin_spincomm * nproc;
    double mem_chi = (double) sizeof(double) * pSPARC->Nd_d * pSQ->Nd_loc * 0.3 * pSPARC->Nspin_spincomm * nproc; 
    
    double mem_Rcut = (double) sizeof(double) * pSQ->Nd_loc * pSPARC->SQ_npl_g  * pSPARC->Nspin_spincomm * nproc;

    double memory_usage = 0.0;
    memory_usage = mem_PR + mem_Rcut + mem_phi + mem_chi;

#ifdef DEBUG
    if (rank == 0) {
        char mem_str[32];
        printf("----------------------\n");
        printf("Estimated memory usage\n");
        formatBytes(memory_usage, 32, mem_str);
        printf("Total: %s\n", mem_str);
        formatBytes(mem_PR, 32, mem_str);
        printf("Vectors in P.R. domain : %s\n", mem_str);
        formatBytes(mem_Rcut, 32, mem_str);
        printf("Vectors in Rcut domain : %s\n", mem_str);
        formatBytes(mem_phi, 32, mem_str);
        printf("Vectors in phi domain : %s\n", mem_str);
        formatBytes(mem_chi, 32, mem_str);
        printf("All saved nonlocal projectors: %s\n", mem_str);
        printf("----------------------------------------------\n");
        formatBytes(memory_usage/nproc,32,mem_str);
        printf("Estimated memory usage per processor: %s\n",mem_str);
    }
#endif

    return memory_usage;   
}