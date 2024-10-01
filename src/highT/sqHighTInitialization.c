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

#include "sqHighTInitialization.h"
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
void init_SQ_HighT(SPARC_OBJ *pSPARC) {
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
    pSQ->Veff_PR = (double *) calloc(sizeof(double), pSQ->DMnd_PR);
    pSQ->Veff_loc_SQ = (double *) calloc(sizeof(double), pSQ->DMnd_SQ*pSPARC->Nspden);
    pSQ->x_ex = (double *) malloc(sizeof(double)*pSQ->Nd_ex);
    memset(pSQ->x_ex, 0, sizeof(double)*pSQ->Nd_ex);
    
    // initialize for d2dext communication
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int dims[3] = {pSPARC->npNdx_SQ, pSPARC->npNdy_SQ, pSPARC->npNdz_SQ};
    pSQ->d2dext_dmcomm_sq = (D2Dext_OBJ *) malloc(sizeof(D2Dext_OBJ));
    pSQ->d2dext_dmcomm_sq_ext = (D2Dext_OBJ *) malloc(sizeof(D2Dext_OBJ));
    Set_D2Dext_Target(pSQ->d2dext_dmcomm_sq, pSQ->d2dext_dmcomm_sq_ext, 
        pSQ->DMnx_SQ, pSQ->DMny_SQ, pSQ->DMnz_SQ, pSQ->nloc[0], pSQ->nloc[1], pSQ->nloc[2],
        gridsizes, dims, pSQ->dmcomm_SQ, &pSQ->dmcomm_d2dext_sq);

    pSQ->mineig = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);
    pSQ->maxeig = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);
    pSQ->lambda_max = (double *) calloc(sizeof(double), pSQ->DMnd_SQ);

    pSQ->gnd = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
    pSQ->gwt = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
    for (int i = 0; i < pSQ->DMnd_SQ; i++) {
        pSQ->gnd[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
        pSQ->gwt[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g);
    }
        
    pSQ->lanczos_vec_all = (double **) calloc(sizeof(double *), (size_t)pSQ->DMnd_SQ);
    assert(pSQ->lanczos_vec_all != NULL);
    for (int i = 0; i < pSQ->DMnd_SQ; i++) {
        pSQ->lanczos_vec_all[i] = (double *) calloc(sizeof(double), (size_t)pSQ->Nd_loc * (size_t)pSPARC->SQ_npl_g);
        assert(pSQ->lanczos_vec_all[i] != NULL);
    }

    pSQ->w_all = (double **) calloc(sizeof(double *), pSQ->DMnd_SQ);
    assert(pSQ->w_all != NULL);
    for (int i = 0; i < pSQ->DMnd_SQ; i++) {
        pSQ->w_all[i] = (double *) calloc(sizeof(double), pSPARC->SQ_npl_g * pSPARC->SQ_npl_g);
        assert(pSQ->w_all[i] != NULL);
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

    if (pSPARC->usefock) {
        pSQ->Dn = (double **) malloc(pSQ->DMnd_SQ*sizeof(double *));
        assert(pSQ->Dn != NULL);
        for (int i = 0; i < pSQ->DMnd_SQ; i++) {
            pSQ->Dn[i] = (double *) malloc(pSQ->Nd_loc*sizeof(double));
            assert(pSQ->Dn[i] != NULL);
        }
        pSQ->Dn_exx = (double **) malloc(pSQ->DMnd_PR*sizeof(double *));
        assert(pSQ->Dn_exx != NULL);
        for (int i = 0; i < pSQ->DMnd_PR; i++) {
            pSQ->Dn_exx[i] = (double *) malloc(pSQ->Nd_loc*sizeof(double));
            assert(pSQ->Dn_exx[i] != NULL);
        }
        init_exx_SQ_highT(pSPARC);
    }
}



void init_exx_SQ_highT(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);

    int Nx = pSQ->Nx_loc;
    int Ny = pSQ->Ny_loc;
    int Nz = pSQ->Nz_loc;

    pSQ->kron_lap = (KRON_LAP *) malloc(sizeof(KRON_LAP));    
    init_kron_Lap(pSPARC, Nx, Ny, Nz, 1, 1, 1, 0 ,0, 0, 1, pSQ->kron_lap);
    
    int DMVertices[6] = {0, Nx-1, 0, Ny-1, 0, Nz-1};   
    pSQ->MpExp_exx = (MPEXP_OBJ *) malloc(sizeof(MPEXP_OBJ));    
    init_multipole_expansion(pSPARC, pSQ->MpExp_exx, Nx, Ny, Nz, Nx, Ny, Nz, DMVertices, MPI_COMM_SELF);

    if (pSPARC->ExxAcc == 1) {
        pSQ->basis = (double *) malloc(pSQ->Nd_loc*pSQ->Nd_loc*sizeof(double));
        assert(pSQ->basis != NULL);
        double t1 = MPI_Wtime();
        calculate_basis(pSPARC);
        double t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank) printf("SQ-hybrid calculating basis takes %.3f ms\n", (t2-t1)*1e3);
    #endif
        
        int nExxPot = (pSPARC->SQ_highT_hybrid_gauss_mem == 1) ? pSQ->DMnd_SQ : 1;
        pSQ->exxPot = (double **) malloc(sizeof(double *)*nExxPot);
        assert(pSQ->exxPot != NULL);
        for (int i = 0; i < nExxPot; i++) {
            pSQ->exxPot[i] = (double *) malloc(sizeof(double)*pSQ->Nd_loc*pSQ->Nd_loc);
            assert(pSQ->exxPot[i] != NULL);
        }
    }

    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        pSQ->erfcR = (double *) malloc(pSQ->Nd_loc*pSQ->Nd_loc*sizeof(double));
        assert(pSQ->erfcR != NULL);
        double t1 = MPI_Wtime();
        calculate_erfcR(pSPARC);
        double t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank) printf("SQ-hybrid calculating erfcR takes %.3f ms\n", (t2-t1)*1e3);
    #endif
    }
}


double memory_estimation_SQ_highT(const SPARC_OBJ *pSPARC)
{
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    SQ_OBJ *pSQ = pSPARC->pSQ;
    // TODO: add accurate memory estimation
    double mem_PR = (double) sizeof(double) * pSQ->DMnd_PR * nproc;
    double mem_phi = (double) sizeof(double) * pSPARC->Nd_d * (6+pSPARC->SQ_npl_g*2+1) * nproc;
    double mem_chi = (double) sizeof(double) * pSPARC->Nd_d * pSQ->Nd_loc * 0.3 * nproc; 
    
    double mem_Rcut = 0;
    mem_Rcut += (double) sizeof(double) * pSPARC->Nd_d * pSQ->Nd_loc * pSPARC->SQ_npl_g * nproc;
    mem_phi += (double) sizeof(double) * pSPARC->Nd_d * pSPARC->SQ_npl_g * pSPARC->SQ_npl_g * nproc;

    double mem_exx = 0;
    if (pSPARC->usefock) {
        mem_exx += (double) sizeof(double) * pSPARC->Nd * pSQ->Nd_loc; // Dn
        mem_exx += (double) sizeof(double) * pSQ->DMnd_PR * nproc * pSQ->Nd_loc; // Dn_exx
        if (pSPARC->ExxAcc) {
            mem_exx += (double) sizeof(double) * pSQ->Nd_loc * pSQ->Nd_loc * nproc; // basis
            if (pSPARC->SQ_highT_hybrid_gauss_mem)
                mem_exx += (double) sizeof(double) * pSPARC->Nd * pSQ->Nd_loc * pSQ->Nd_loc; // exxPot
            else 
                mem_exx += (double) sizeof(double) * nproc * pSQ->Nd_loc * pSQ->Nd_loc; // exxPot
            if (strcmpi(pSPARC->XC,"HSE") == 0) {
                mem_exx += (double) sizeof(double) * nproc * pSQ->Nd_loc * pSQ->Nd_loc; // erfcR
            }
        }
    }

    double memory_usage = 0.0;
    memory_usage = mem_PR + mem_Rcut + mem_phi + mem_chi + mem_exx;

    if (!rank) {
        printf("WARNING: If test failed due to memory limit, please switch to low-T SQ by setting HIGHT_FLAG: 0\n");
    }

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
        formatBytes(mem_exx, 32, mem_str);
        printf("Hybrid vectors: %s\n", mem_str);
        printf("----------------------------------------------\n");
        formatBytes(memory_usage/nproc,32,mem_str);
        printf("Estimated memory usage per processor: %s\n",mem_str);
    }
#endif

    return memory_usage;   
}