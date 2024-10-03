/***
 * @file    sqProperties.c
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

#include "sq.h"
#include "sqDensity.h"
#include "sqEnergy.h"
#include "occupation.h"
#include "tools.h"
#include "sqParallelization.h"
#include "electronDensity.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ(SPARC_OBJ *pSPARC, int SCFcount) {
    SQ_OBJ  *pSQ = pSPARC->pSQ;
    int DMnd = pSQ->DMnd_SQ;

    // Gauss Quadrature for electron density
    for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        // communicate to get full values for PR domain        
        int spn_g = spn_i + pSPARC->spin_start_indx;
        TransferVeff_sq2sqext(pSPARC, pSQ->Veff_loc_SQ + spn_g * DMnd, pSQ->Veff_PR);

        // perform Gauss Quadrature 
        GaussQuadrature(pSPARC, SCFcount, spn_i);
    }

    // find Fermi energy (Efermi)
    find_Efermi_SQ(pSPARC, SCFcount);

    // Calculate Electron Density
    Calculate_elecDens_Gauss(pSPARC);

    if (pSPARC->spin_typ == 1) {
        Calculate_Magz(pSPARC, pSPARC->Nd_d, pSPARC->mag, pSPARC->electronDens+pSPARC->Nd_d, pSPARC->electronDens+2*pSPARC->Nd_d); // magz
    }
}


// TODO: ADD Newton Raphson method in SQDFT here. 
void find_Efermi_SQ(SPARC_OBJ *pSPARC, int SCFcount)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    
    SQ_OBJ  *pSQ = pSPARC->pSQ;
    int DMnd = pSQ->DMnd_SQ;

#ifdef DEBUG
    double t1 = MPI_Wtime();
#endif

    double lambda_min = DBL_MAX;
    double lambda_max = DBL_MIN;

    // For the first SCF without Efermi, communicate across procs to find global  
    // lambda_min and lambda_max to give as input guesses for Brent's algorithm
    if (SCFcount == 0) {
        for (int nd = 0; nd < DMnd * pSPARC->Nspin_spincomm; nd++) {
            lambda_min = min(lambda_min, pSQ->mineig[nd]);
            lambda_max = max(lambda_max, pSQ->maxeig[nd]);
        }
        MPI_Allreduce(MPI_IN_PLACE, &lambda_max, 1, MPI_DOUBLE, MPI_MAX, pSQ->dmcomm_SQ);
        MPI_Allreduce(MPI_IN_PLACE, &lambda_min, 1, MPI_DOUBLE, MPI_MIN, pSQ->dmcomm_SQ);

        if (pSPARC->npspin > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &lambda_max, 1, MPI_DOUBLE, MPI_MAX, pSPARC->spin_bridge_comm);
            MPI_Allreduce(MPI_IN_PLACE, &lambda_min, 1, MPI_DOUBLE, MPI_MIN, pSPARC->spin_bridge_comm);
        }        
    }

    double x1 = (SCFcount > 0) ? pSPARC->Efermi - 2.0 : min(lambda_min - 1, -6.907755278982137*(1/pSPARC->Beta));
    double x2 = (SCFcount > 0) ? pSPARC->Efermi + 2.0 : lambda_max + 1.0;
    pSPARC->Efermi = Calculate_FermiLevel(pSPARC, x1, x2, 1e-12, 100, occ_constraint_SQ_gauss); 

#ifdef DEBUG
    double t2 = MPI_Wtime();
    if(!rank) printf("Rank %d Waiting and finding fermi level %.6f takes %.3f ms\n",rank, pSPARC->Efermi, (t2-t1)*1e3); 
#endif

}


void Calculate_elecDens_Gauss(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    SQ_OBJ  *pSQ = pSPARC->pSQ;
    int DMnd = pSQ->DMnd_SQ;

#ifdef DEBUG
    double t1 = MPI_Wtime();
#endif

    double *rho = (double *) calloc(DMnd * pSPARC->Nspdentd, sizeof(double));
    double *rho_ = rho + (pSPARC->Nspin-1) * DMnd; // Nspin here is only 1 and 2

    for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        int spn_g = spn_i + pSPARC->spin_start_indx;
        for (int i = 0; i < DMnd; i++) {
            double rho_pc = 0.0;
            for (int j = 0; j < pSPARC->SQ_npl_g; j++) {
                rho_pc += pSQ->gwt[i + spn_i * DMnd][j] * smearing_function(
                            pSPARC->Beta, pSQ->gnd[i + spn_i * DMnd][j], pSPARC->Efermi, pSPARC->elec_T_type);
            }
            rho_[i + spn_g*DMnd] = pSPARC->occfac * rho_pc / pSPARC->dV;
        }
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, rho, pSPARC->Nspdentd*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

    if (pSPARC->Nspin > 1) {
        for (int i = 0; i < DMnd; i++) {
            rho[i] = rho[DMnd+i] + rho[DMnd*2+i];
        }
    }

#ifdef DEBUG
    double t2 = MPI_Wtime();
    if (!rank) printf("rank %d, calculating electron density takes %.3f ms\n",rank, (t2-t1)*1e3); 
#endif

    for (int n = 0; n < pSPARC->Nspdentd; n++) {
        TransferDensity_sq2phi(pSPARC, rho+n*DMnd, pSPARC->electronDens + n*pSPARC->Nd_d);
    }

#ifdef DEBUG
    t1 = MPI_Wtime();
    if (!rank) printf("rank %d, transferring electron denisty into phi domain takes %.3f ms\n",rank, (t1-t2)*1e3); 
#endif
    free(rho);
}



/**
 * @brief   Compute column of density matrix using Gauss Quadrature
 */
void Gauss_density_matrix_col(SPARC_OBJ *pSPARC, int Nd, int npl, double *DMcol, double *V, double *w, double *D) 
{
    double *wte1 = (double *) calloc(sizeof(double), npl);
    double *gdwte1 = (double *) calloc(sizeof(double), npl);
    double *wgdwte1 = (double *) calloc(sizeof(double), npl);

    for (int j = 0; j < npl; j++) {
        wte1[j] = w[j * npl];
    }

    for (int j = 0; j < npl; j++) {
        gdwte1[j] = wte1[j] * smearing_function(
                    pSPARC->Beta, D[j], pSPARC->Efermi, pSPARC->elec_T_type);
    }

    cblas_dgemv (CblasColMajor, CblasNoTrans, npl, npl, 1.0, 
                    w, npl, gdwte1, 1, 0.0, wgdwte1, 1);

    cblas_dgemv (CblasColMajor, CblasNoTrans, Nd, npl, 1.0/pSPARC->dV, 
                    V, Nd, wgdwte1, 1, 0.0, DMcol, 1);
        
    free(wte1);
    free(gdwte1);
    free(wgdwte1);
}