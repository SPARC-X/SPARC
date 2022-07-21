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
// #include "sqtool.h"
// #include "sqNlocVecRoutines.h"
#include "occupation.h"
// #include "stress.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ(SPARC_OBJ *pSPARC, int SCFcount) {
    // TODO: Add Clenshaw Curtis method
    if (pSPARC->SQ_typ == 2)
        GaussQuadrature(pSPARC, SCFcount);
}


/**
 * @brief   Compute column of density matrix using Gauss Quadrature
 */
void Gauss_density_matrix_col(SPARC_OBJ *pSPARC, int Nd, int npl, double ***DMcol, double *V, double *w, double *D) 
{
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int i, j, *nloc, FDn;
    double *wte1, *gdwte1, *wgdwte1;
    
    double *DMcol_vec;
    DMcol_vec = (double *) calloc(sizeof(double), pSQ->Nd_loc);
    memset(DMcol_vec, 0, sizeof(double)*pSQ->Nd_loc);

    nloc = pSQ->nloc;
    FDn = pSPARC->order / 2;
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

    cblas_dgemv (CblasColMajor, CblasNoTrans, Nd, npl, 1.0, 
                    V, Nd, wgdwte1, 1, 0.0, DMcol_vec, 1);

    int idx;
    for (idx = 0; idx < Nd; idx++) {
        DMcol_vec[idx] *= 1.0/pSPARC->dV;
    }

    int count, ii, jj, kk;
    count = 0;
    for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
        for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
            for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] = DMcol_vec[count];
                count ++;
            }
        }
    }
    free(DMcol_vec);
    free(wte1);
    free(gdwte1);
    free(wgdwte1);
}


/**
 * @brief   Compute column of density matrix using Clenshaw curtis Quadrature
 */
void Clenshaw_curtis_density_matrix_col(SPARC_OBJ *pSPARC, double ***DMcol, int i, int j, int k, int nd) 
{
    int ii, jj, kk, *nloc, FDn, nq;
    double ***Hv, ***t0, ***t1, ***t2, *di;

    SQ_OBJ *pSQ = pSPARC->pSQ;
    nloc = pSQ->nloc;
    FDn = pSPARC->order / 2;

    // Compute quadrature coefficients
    double bet0, bet, lambda_fp;
    bet0 = pSPARC->Beta;
    di = (double *) calloc(sizeof(double), pSPARC->SQ_npl_c+1);
    bet = bet0 * pSQ->zee[nd];
    lambda_fp = (pSPARC->Efermi - pSQ->chi[nd]) / pSQ->zee[nd];
    ChebyshevCoeff_SQ(pSPARC->SQ_npl_c, pSQ->Ci[nd], di, smearing_function, bet, lambda_fp, pSPARC->elec_T_type);
    free(di);
    
    // Start to find density matrix
    Hv = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	t0 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	t1 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	t2 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);

	for(kk = 0; kk < 2*nloc[2]+1; kk++) {
		Hv[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		t0[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		t1[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		t2[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);

		for(jj = 0; jj < 2*nloc[1]+1; jj++) {
			Hv[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			t0[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			t1[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			t2[kk][jj] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
		}
	}

    /// For each FD node, loop over quadrature order to find Chebyshev expansion components, Use the HsubTimesVec function this.
    for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
        for(jj = -nloc[1];jj <= nloc[1]; jj++) {
            for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]=0.0;
                if(ii==0 && jj==0 && kk==0)
                    t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]=1.0;
            }
        }
    }

    HsubTimesVec(pSPARC,t0,i,j,k,Hv); // Hv=Hsub*t0. Here t0 is the vector and i,j,k are the indices of node in proc domain to which the Hsub corresponds to and the indices are w.r.t proc+Rcut domain

    for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
        for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
            for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]=(Hv[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] - pSQ->chi[nd]*t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]) / pSQ->zee[nd];

                DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] = 0.0;
            }
        }
    }
    
    // loop over quadrature order
    for(nq = 0; nq <= pSPARC->SQ_npl_c; nq++) {
        if(nq == 0) {
            for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
                for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
                    for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                        DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] += 
                            (double)(t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]*pSQ->Ci[nd][nq]/pSPARC->dV);
                    }
                }
            }
        } else if(nq == 1) {
            for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
                for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
                    for(ii = -nloc[0];ii <= nloc[0]; ii++) {
                        DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] += 
                            (double)(t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]*pSQ->Ci[nd][nq]/pSPARC->dV);
                    }
                }
            }

        } else {
            HsubTimesVec(pSPARC, t1, i, j, k, Hv); // Hv=Hsub*t1

            for(kk = -nloc[2]; kk <= nloc[2]; kk++) {
                for(jj = -nloc[1]; jj <= nloc[1]; jj++) {
                    for(ii = -nloc[0]; ii <= nloc[0]; ii++) {
                        t2[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] = 
                            (2*(Hv[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] - pSQ->chi[nd]*t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]]) / pSQ->zee[nd]) - t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]];

                        DMcol[kk+nloc[2]+FDn][jj+nloc[1]+FDn][ii+nloc[0]+FDn] += (double)(t2[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] * pSQ->Ci[nd][nq]/pSPARC->dV);

                        t0[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] = t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]];
                        t1[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]] = t2[kk+nloc[2]][jj+nloc[1]][ii+nloc[0]];
                    }
                }
            }
        }
    }
    
    for(kk = 0; kk < 2*nloc[2]+1; kk++) {
        for(jj = 0; jj < 2*nloc[1]+1; jj++) {
            free(Hv[kk][jj]);
            free(t0[kk][jj]);
            free(t1[kk][jj]);
            free(t2[kk][jj]);
        }
        free(Hv[kk]);
        free(t0[kk]);
        free(t1[kk]);
        free(t2[kk]);
    }
    free(Hv);
    free(t0);
    free(t1);
    free(t2);
}