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
#include "sqEnergy.h"
#include "sqDensity.h"
#include "occupation.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)

/**
 * @brief   Calculate Band energy with SQ method
 */
double Calculate_Eband_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    double Eband = 0;
    int DMnd = pSQ->DMnd_SQ;

    Eband = Calculate_Eband_SQ_Gauss(pSPARC, DMnd, pSQ->dmcomm_SQ);
    return Eband;
}

/**
 * @brief   Calculate electronic entropy with SQ method
 */
double Calculate_electronicEntropy_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    double Entropy = 0;
    int DMnd = pSQ->DMnd_SQ;

    Entropy = Calculate_electronicEntropy_SQ_Gauss(pSPARC, DMnd, pSQ->dmcomm_SQ);
    return Entropy;
}

/**
 * @brief   Calculate band energy with Gauss quadrature in SQ method
 */
double Calculate_Eband_SQ_Gauss(SPARC_OBJ *pSPARC, int DMnd, MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) return 0.0;
    int i, j, size, rank;
    double Eband, ebs;
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    Eband = 0.0;
    for (i = 0; i < DMnd; i++) {
        ebs = 0.0;
        for (j = 0; j < pSPARC->SQ_npl_g; j++) {
            ebs = ebs + UbandFunc(pSQ->gnd[i][j], pSPARC->Efermi, pSPARC->Beta, pSPARC->elec_T_type) * pSQ->gwt[i][j];
        }
        Eband += 2 * ebs;
    }
    if (size > 1)
        MPI_Allreduce(MPI_IN_PLACE, &Eband, 1, MPI_DOUBLE, MPI_SUM, comm);

    return Eband;
}

/**
 * @brief   Calculate electronic entropy with Gauss quadrature in SQ method
 */
double Calculate_electronicEntropy_SQ_Gauss(SPARC_OBJ *pSPARC, int DMnd, MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) return 0.0;
    int i, j, size, rank;
    double Entropy, ent;
    SQ_OBJ  *pSQ = pSPARC->pSQ;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Entropy = 0.0;
    for (i = 0; i < DMnd; i++) {
        // entropy energy
        ent = 0.0;
        for (j = 0; j < pSPARC->SQ_npl_g; j++) {
            ent = ent + EentFunc(pSQ->gnd[i][j], pSPARC->Efermi, pSPARC->Beta, pSPARC->elec_T_type) * pSQ->gwt[i][j];
        }
        Entropy += 2 * ent/pSPARC->Beta;
    }
    if (size > 1)
        MPI_Allreduce(MPI_IN_PLACE, &Entropy, 1, MPI_DOUBLE, MPI_SUM, comm);

    return Entropy;
}

/**
 * @brief   Occupation constraints with Gauss quadrature
 */
double occ_constraint_SQ_gauss(SPARC_OBJ *pSPARC, double lambda_f) {
    int k, j, rank;
    double g, fval, g_global;
    SQ_OBJ* pSQ  = pSPARC->pSQ;
    MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);

    g = 0.0;
    for (k = 0; k < pSQ->DMnd_SQ; k++) {
        for (j = 0; j < pSPARC->SQ_npl_g; j++) {
            g += 2 * pSQ->gwt[k][j] * smearing_function(
                pSPARC->Beta, pSQ->gnd[k][j], lambda_f, pSPARC->elec_T_type);
        }
    }

    MPI_Allreduce(&g, &g_global, 1, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
    g = g_global;

    // fval = (g - (-pSQ->Ncharge))/(-1*pSQ->Ncharge);
    fval = (g - pSPARC->PosCharge)/pSPARC->PosCharge;
    return fval;
}

/**
 * @brief   Band energy function
 * 
 * TODO:    Can be removed and simply replace the code.
 */
double UbandFunc(double t, double lambda_f, double bet, int type) {
    double v;
    // v = t * (1 / (1 + exp(bet * (t - lambda_f))));
    v = t * smearing_function(bet, t, lambda_f, type);
    return v;
}

/**
 * @brief   Entropy function
 */
double EentFunc(double t, double lambda_f, double bet, int type) {
    double v, focc;
    // focc = (1 / (1 + exp(bet * (t - lambda_f))));
    focc = smearing_function(bet, t, lambda_f, type);

    if (fabs(focc) < 0.01 * TEMP_TOL || fabs(focc - 1.0) < 0.01 * TEMP_TOL) {
        v = 0.0;
    } else {
        v = (focc * log(focc) + (1 - focc) * log(1 - focc));
    }
    return v;
}



/**
 * @brief   Compute repulsive energy correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_SQ(SPARC_OBJ *pSPARC) {
    return;
}

