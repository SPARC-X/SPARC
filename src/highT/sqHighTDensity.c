/***
 * @file    sqHighTDensity.c
 * @brief   This file contains the functions for force calculation using SQ method.
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
#include <float.h>
#include <assert.h>
#include <mpi.h>
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "sqHighT.h"
#include "sqHighTDensity.h"
#include "sqHighTExactExchange.h"
#include "sqParallelization.h"
#include "sqDensity.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ_highT(SPARC_OBJ *pSPARC, int SCFcount) 
{
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    // communicate to get full values for PR domain        
    TransferVeff_sq2sqext(pSPARC, pSQ->Veff_loc_SQ, pSQ->Veff_PR);

    // Gauss Quadrature for electron density    
    GaussQuadrature_highT(pSPARC, SCFcount);

    // find Fermi energy (Efermi)
    find_Efermi_SQ(pSPARC, SCFcount);

    // Calculate Electron Density
    Calculate_elecDens_Gauss(pSPARC);
}

/**
 * @brief   Compute all columns of density matrix using Gauss Quadrature
 */
void calculate_density_matrix_SQ_highT(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int DMnd = pSQ->DMnd_SQ;
    int *nloc = pSQ->nloc;
    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc*Ny_loc;
    int center = nloc[0] + nloc[1]*Nx_loc + nloc[2]*NxNy_loc;
    int flag_exxPot = (pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) 
                   && (pSPARC->ExxAcc == 1) && (pSPARC->SQ_highT_hybrid_gauss_mem == 0);

    for (int nd = 0; nd < DMnd; nd++) {
        // Already saved
        Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, pSQ->Dn[nd], pSQ->lanczos_vec_all[nd], pSQ->w_all[nd], pSQ->gnd[nd]);
    }
}