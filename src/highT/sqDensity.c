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
#include "sqDensity.h"
#include "occupation.h"
#include "tools.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ(SPARC_OBJ *pSPARC, int SCFcount) {
    // Gauss Quadrature for electron density
    GaussQuadrature(pSPARC, SCFcount);
}


/**
 * @brief   Compute column of density matrix using Gauss Quadrature
 */
void Gauss_density_matrix_col(SPARC_OBJ *pSPARC, int Nd, int npl, double *DMcol, double *V, double *w, double *D) 
{
#define DMcol(i,j,k) DMcol[(i)+(j)*DMnx_locex+(k)*DMnxny_locex]
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int j, *nloc, FDn;
    double *wte1, *gdwte1, *wgdwte1, *DMcol_vec;

    DMcol_vec = (double *) malloc(sizeof(double) * pSQ->Nd_loc);    

    nloc = pSQ->nloc;
    FDn = pSPARC->order / 2;
    
    int DMnx_locex, DMny_locex, DMnxny_locex;
    DMnx_locex = 2*nloc[0]+1+2*FDn;
    DMny_locex = 2*nloc[1]+1+2*FDn;
    DMnxny_locex = DMnx_locex * DMny_locex;

    wte1 = (double *) calloc(sizeof(double), npl);
    gdwte1 = (double *) calloc(sizeof(double), npl);
    wgdwte1 = (double *) calloc(sizeof(double), npl);

    for (j = 0; j < npl; j++) {
        wte1[j] = w[j * npl];
    }

    for (j = 0; j < npl; j++) {
        gdwte1[j] = wte1[j] * smearing_function(
                    pSPARC->Beta, D[j], pSPARC->Efermi, pSPARC->elec_T_type);
    }

    cblas_dgemv (CblasColMajor, CblasNoTrans, npl, npl, 1.0, 
                    w, npl, gdwte1, 1, 0.0, wgdwte1, 1);

    cblas_dgemv (CblasColMajor, CblasNoTrans, Nd, npl, 1.0/pSPARC->dV, 
                    V, Nd, wgdwte1, 1, 0.0, DMcol_vec, 1);

    int Nx_loc, Ny_loc, Nz_loc, NxNy_loc;
    Nx_loc = pSQ->Nx_loc;
    Ny_loc = pSQ->Ny_loc;
    Nz_loc = pSQ->Nz_loc;
    NxNy_loc = Nx_loc * Ny_loc;
    restrict_to_subgrid(DMcol_vec, DMcol, DMnx_locex, Nx_loc, DMnxny_locex, NxNy_loc, 
        FDn, Nx_loc+FDn-1, FDn, Ny_loc+FDn-1, FDn, Nz_loc+FDn-1, 0, 0, 0);    
    
    free(DMcol_vec);
    free(wte1);
    free(gdwte1);
    free(wgdwte1);
#undef DMcol
}
