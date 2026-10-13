/***
 * @file    sqHighT.c
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
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "sq.h"
#include "sqHighT.h"
#include "sqHighTExactExchange.h"

#ifdef SPARCX_ACCEL
#include "accel.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)

/**
 * SQ TODO list:
 * 1. Newtown Raphson algorithm for fermi level
 * 2. Energy and forces correction
 * 3. Always storage of nolocal projectors
 * 4. Always use spherical Rcut region
 * 5. Function for cell replication
 */


/**
 * @brief   Gauss Quadrature method
 * 
 * @param scf_iter  The scf iteration counter
 */
void GaussQuadrature_highT(SPARC_OBJ *pSPARC) {
    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    int nd, rank;
    int DMnx, DMny, DMnz, DMnd;
    double lambda_min, lambda_max;
#ifdef DEBUG
    double time1, time2;
#endif
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    DMnx = pSQ->DMnx_SQ;
    DMny = pSQ->DMny_SQ;
    DMnz = pSQ->DMnz_SQ;
    DMnd = pSQ->DMnd_SQ;

    (void)DMnx;
    (void)DMny;
    (void)DMnz;

    MPI_Comm_rank(pSQ->dmcomm_SQ, & rank);
    int flag_exxPot = (pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) 
                   && (pSPARC->ExxAcc == 1) && (pSPARC->SQ_highT_hybrid_gauss_mem == 0);
    //////////////////////////////////////////////////////////////////////

#ifdef DEBUG
    time2 = MPI_Wtime();
#endif

    #ifdef SPARCX_ACCEL
	if (pSPARC->useACCEL == 1 && pSPARC->cell_typ == 0 && pSPARC->usefock <=1)
	{
		ACCEL_SQ_LanczosAlgorithm_gauss(pSPARC, DMnx, DMny, DMnz, &lambda_min, &lambda_max);
	}
	else
	#endif // SPARCX_ACCEL
	{
    	for (nd = 0; nd < DMnd; nd ++) {
            // calculate exx potential if not saved
            if (flag_exxPot == 1) {
                double t1 = MPI_Wtime();
                compute_exx_potential_node_SQ(pSPARC, nd, pSQ->exxPot[0]);
                double t2 = MPI_Wtime();
                pSPARC->ACEtime += (t2 - t1);
            }

            LanczosAlgorithm_gauss(pSPARC, nd, 0, &lambda_min, &lambda_max);
        	pSQ->mineig[nd]  = lambda_min;
        	pSQ->maxeig[nd]  = lambda_max;
    	}
	}

    // This barrier fixed the severe slowdown of MPI communication on hive
    MPI_Barrier(pSQ->dmcomm_SQ);    

#ifdef DEBUG
    time1 = MPI_Wtime();
    if(!rank) printf("Rank %d finished Lanczos taking %.3f ms\n",rank, (time1-time2)*1e3); 
#endif
}
