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
/** ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif

#include "isddft.h"
#include "sqProperties.h"
#include "sq.h"
#include "sqtool.h"
#include "sqNlocVecRoutines.h"
#include "occupation.h"
#include "stress.h"


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
 * @brief   Calculate Band energy with SQ method
 */
double Calculate_Eband_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    double Eband;
    int DMnd = pSQ->Nd_d_SQ;

    // TODO: add Clenshaw Curtis method
    if (pSPARC->SQ_typ == 2) {
        Eband = Calculate_Eband_SQ_Gauss(pSPARC, DMnd, pSQ->dmcomm_SQ);
    }
    return Eband;
}

/**
 * @brief   Calculate electronic entropy with SQ method
 */
double Calculate_electronicEntropy_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    double Entropy;
    int DMnd = pSQ->Nd_d_SQ;

    // TODO: add Clenshaw Curtis method
    if (pSPARC->SQ_typ == 2) {
        Entropy = Calculate_electronicEntropy_SQ_Gauss(pSPARC, DMnd, pSQ->dmcomm_SQ);
    }
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
    for (k = 0; k < pSQ->Nd_d_SQ; k++) {
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
 * @brief   Calculate nonlocal forces using SQ method
 */
void Calculate_nonlocal_forces_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int rank, size, i, j, k, ii, jj, kk, a, atom_count, JJ, JJ_typ, count;
    int DMnx, DMny, DMnz, FDn, *nloc;
    double ***t0, ***DMcol, ***gradx_DM, ***grady_DM, ***gradz_DM, ***psipsi, lambda_min, lambda_max, *di, delta_lmin, delta_lmax;
    double time1, time2, time_lan;

	MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);
    MPI_Comm_size(pSQ->dmcomm_SQ, &size);
    nloc = pSQ->nloc;
    int dims[3] = {2*nloc[0]+1, 2*nloc[1]+1, 2*nloc[2]+1};

#ifdef DEBUG
    if(!rank) printf("Start computing non-local component of the forces... ");
#endif

    time1 = MPI_Wtime();
    DMnx = pSQ->Nx_d_SQ;
    DMny = pSQ->Ny_d_SQ;
    DMnz = pSQ->Nz_d_SQ;
    FDn = pSPARC->order / 2;

    double energy_nl;
    double s_k[9], s_nl[9];

	if(pSPARC->Calc_stress == 1) {
        energy_nl = 0.0;
		for(i = 0; i < 9; i++) {
            s_k[i] = 0.0;
            s_nl[i] = 0.0;
		}
	} else if(pSPARC->Calc_pres == 1){ 
        pSPARC->pres_nl = 0.0;
        energy_nl = 0.0;
    }

#define s_k(i,j) s_k[3*i+j]
#define s_nl(i,j) s_nl[3*i+j]

    double *force_nloc;
    // force_nloc is in order nonlocal_force_x, nonlocal_force_y, nonlocal_force_z
    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));

    gradx_DM = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	grady_DM = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	gradz_DM = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	psipsi = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
	for(k = 0; k < 2*nloc[2]+1; k++) {
        gradx_DM[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		grady_DM[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		gradz_DM[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		psipsi[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
		for(j = 0; j < 2*nloc[1]+1; j++) {
            gradx_DM[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			grady_DM[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			gradz_DM[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
			psipsi[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
		}
	}

    DMcol = (double ***) calloc(sizeof(double **), 2*nloc[2]+1+2*FDn);
	for(k = 0; k < 2*nloc[2]+1 + 2*FDn; k++) {
        DMcol[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1 + 2*FDn);
		for(j = 0; j < 2*nloc[1]+1 + 2*FDn; j++) {
            DMcol[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1 + 2*FDn);
		}
	}

    double *grad_DM_vec;
    grad_DM_vec = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    // only need t0 for initial gauss guess when vectors are not saved for Gauss quadrature
    if (pSPARC->SQ_gauss_mem == 0 && pSPARC->SQ_typ_dm == 2) {
        t0 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
        for(k = 0; k < 2*nloc[2]+1; k++) {
            t0[k] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
            for(j = 0; j < 2*nloc[1]+1; j++) {
                t0[k][j] = (double *) calloc(sizeof(double), 2*nloc[0]+1);
            }
        }
    }

	// loop over finite difference nodes in the processor domain
	int nd;
	if(pSPARC->SQ_typ_dm == 1) {
	    if(pSPARC->SQ_EigshiftFlag == 0) {
	        if(pSPARC->MDCount == 0) {
                nd = 0;
		        for (k = nloc[2]; k < DMnz + nloc[2]; k++) {
			        for (j = nloc[1]; j < DMny + nloc[1]; j++) {
				        for (i = nloc[0]; i < DMnx + nloc[0]; i++) {

					        for (kk = -nloc[2]; kk <= nloc[2]; kk++) {
						        for (jj = -nloc[1]; jj <= nloc[1]; jj++) {
							        for (ii = -nloc[0]; ii <= nloc[0]; ii++) {
								        pSQ->vec[nd][kk + nloc[2]][jj + nloc[1]][ii + nloc[0]] = ((double)(rand()) / (double)(RAND_MAX));
							        }
						        }
					        }

					        LanczosAlgorithm_new(pSPARC, i, j, k, &lambda_min, nd, 1);

					        pSQ->chi[nd] = (pSQ->lambda_max[nd] + lambda_min) / 2;
					        pSQ->zee[nd] = (pSQ->lambda_max[nd] - lambda_min) / 2;

					        nd = nd + 1;
				        }
				    }
		        }
		    } else {
                nd = 0;
		        for (k = nloc[2]; k < DMnz + nloc[2]; k++) {
			        for (j = nloc[1]; j < DMny + nloc[1]; j++) {
				        for (i = nloc[0]; i < DMnx + nloc[0]; i++) {
					        LanczosAlgorithm_new(pSPARC, i, j, k, &lambda_min, nd, 0);

					        pSQ->chi[nd] = (pSQ->lambda_max[nd] + lambda_min) / 2;
					        pSQ->zee[nd] = (pSQ->lambda_max[nd] - lambda_min) / 2;

					        nd = nd + 1;
				        }
				    }
		        }
		    }
		} else {
            nd = 0;
		    for (k = nloc[2]; k < DMnz + nloc[2]; k++) {
			    for (j = nloc[1]; j < DMny + nloc[1]; j++) {
				    for (i = nloc[0]; i < DMnx + nloc[0]; i++) {
					    delta_lmin = (pSPARC->SQ_eigshift*0.01) * (pSPARC->Efermi - pSQ->mineig[nd]);//eigshift is in percent
					    delta_lmax = (pSPARC->SQ_eigshift*0.01) * (pSQ->maxeig[nd] - pSPARC->Efermi);

					    lambda_min = pSQ->mineig[nd] - delta_lmin;
					    lambda_max = pSQ->maxeig[nd] + delta_lmax;

					    pSQ->chi[nd] = (lambda_max + lambda_min) / 2;
					    pSQ->zee[nd] = (lambda_max - lambda_min) / 2;
                        // if(!rank) printf("nd %d, chi %.4f, zee %.4f\n", nd, pSQ->chi[nd], pSQ->zee[nd]);
					    nd = nd+1;
				    }
			    }
		    }
		}

        // This barrier fixed the severe slowdown of MPI communication on hive
        MPI_Barrier(pSQ->dmcomm_SQ);
#ifdef DEBUG
    time_lan = MPI_Wtime();
    if (pSPARC->SQ_EigshiftFlag == 0) {
        if(!rank) 
            printf("Lanczos and Chebyshev Coeff in non-local forces takes %.3f ms\n", (time_lan - time1)*1e3);
    } else {
        if(!rank) 
            printf("Eigenshift and Chebyshev Coeff in non-local forces takes %.3f ms\n", (time_lan - time1)*1e3);
    }
#endif
	} 

    nd = 0;
	for(k = nloc[2]; k <= DMnz + nloc[2]-1; k++) {
		for(j = nloc[1]; j <= DMny + nloc[1]-1; j++) {
			for(i = nloc[0]; i <= DMnx+nloc[0]-1; i++) {
                
                if (pSPARC->SQ_typ_dm == 1) {
                    // Density matrix by Clenshaw-Curtis quadrature
                    Clenshaw_curtis_density_matrix_col(pSPARC, DMcol, i, j, k, nd);
                } else {
                    // Density matrix by Gauss quadrature
                    if (pSPARC->SQ_gauss_mem == 0) {
                        for (kk = -nloc[2]; kk <= nloc[2]; kk++) {
                            for (jj = -nloc[1]; jj <= nloc[1]; jj++) {
                                for (ii = -nloc[0]; ii <= nloc[0]; ii++) {
                                    t0[kk + nloc[2]][jj + nloc[1]][ii + nloc[0]] = 0.0;
                                    if (ii == 0 && jj == 0 && kk == 0)
                                        t0[kk + nloc[2]][jj + nloc[1]][ii + nloc[0]] = 1.0;
                                }
                            }
                        }
                        LanczosAlgorithm_gauss(pSPARC, t0, i, j, k, &lambda_min, &lambda_max, nd);
                        Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, DMcol, pSQ->lanczos_vec, pSQ->w, pSQ->gnd[nd]);
                    } else {
                        // Already saved
                        Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, DMcol, pSQ->lanczos_vec_all[nd], pSQ->w_all[nd], pSQ->gnd[nd]);
                    }
                }

				// find gradient of DM cols
				for(kk = FDn; kk < 2*nloc[2]+1 + FDn; kk++) {
					for(jj = FDn; jj < 2*nloc[1]+1 + FDn; jj++) {
						for(ii = FDn; ii < 2*nloc[0]+1 + FDn; ii++) {
							gradx_DM[kk-FDn][jj-FDn][ii-FDn] = 0.0;
							grady_DM[kk-FDn][jj-FDn][ii-FDn] = 0.0;
							gradz_DM[kk-FDn][jj-FDn][ii-FDn] = 0.0;
						}
					}
				}

				for(kk = FDn; kk < 2*nloc[2]+1 + FDn; kk++) {
					for(jj = FDn; jj < 2*nloc[1]+1 + FDn; jj++) {
						for(ii = FDn; ii < 2*nloc[0]+1 + FDn; ii++) {
							// x-direction
							for(a = 1; a <= FDn; a++) {
								gradx_DM[kk-FDn][jj-FDn][ii-FDn] += (-DMcol[kk][jj][ii-a] + DMcol[kk][jj][ii+a]) * pSPARC->D1_stencil_coeffs_x[a];
							}
							// y-direction
							for(a = 1; a <= FDn; a++) {
								grady_DM[kk-FDn][jj-FDn][ii-FDn] += (-DMcol[kk][jj-a][ii] + DMcol[kk][jj+a][ii]) * pSPARC->D1_stencil_coeffs_y[a];
							}
							// z-direction
							for(a = 1; a <= FDn; a++) {
								gradz_DM[kk-FDn][jj-FDn][ii-FDn] += (-DMcol[kk-a][jj][ii] + DMcol[kk+a][jj][ii]) * pSPARC->D1_stencil_coeffs_z[a];
							}
							if(pSPARC->Calc_pres == 1 || pSPARC->Calc_stress == 1) {
								psipsi[kk-FDn][jj-FDn][ii-FDn] = DMcol[kk][jj][ii];
							}
						}
					}
				}

                if(pSPARC->Calc_stress == 1) {
                    // x-direction
                    s_k(0,0) += 2*DMcol[nloc[2] + FDn][nloc[1] + FDn][nloc[0] + FDn] * pSPARC->D2_stencil_coeffs_x[0] * pSPARC->dV;

                    for(a = 1; a <= FDn; a++) {
                        s_k(0,0) += 2*(DMcol[nloc[2] + FDn][nloc[1] + FDn][nloc[0] + FDn - a] + DMcol[nloc[2] + FDn][nloc[1] + FDn][nloc[0] + FDn+a])*pSPARC->D2_stencil_coeffs_x[a]*pSPARC->dV;
                    }

                    for(a = 1; a <= FDn; a++) {
                        s_k(0,1) += 2*(-grady_DM[nloc[2]][nloc[1]][nloc[0] - a] + grady_DM[nloc[2]][nloc[1]][nloc[0] + a])*pSPARC->D1_stencil_coeffs_x[a]*pSPARC->dV;
                    }

                    for(a = 1; a <= FDn; a++){
                        s_k(0,2) += 2*(-gradz_DM[nloc[2]][nloc[1]][nloc[0] - a] + gradz_DM[nloc[2]][nloc[1]][nloc[0] + a])*pSPARC->D1_stencil_coeffs_x[a]*pSPARC->dV;
                    }

                    // y-direction
                    for(a = 1; a <= FDn; a++) {
                        s_k(1,0) += 2*(-gradx_DM[nloc[2]][nloc[1] - a][nloc[0]] + gradx_DM[nloc[2]][nloc[1] + a][nloc[0]])*pSPARC->D1_stencil_coeffs_y[a]*pSPARC->dV;
                    }

                    s_k(1,1) += 2*DMcol[nloc[2] + FDn][nloc[1] + FDn][nloc[0] + FDn]*pSPARC->D2_stencil_coeffs_y[0]*pSPARC->dV;

                    for(a = 1; a <= FDn; a++) {
                        s_k(1,1) += 2*(DMcol[nloc[2] + FDn][nloc[1] + FDn - a][nloc[0] + FDn] + DMcol[nloc[2] + FDn][nloc[1] + FDn + a][nloc[0] + FDn])*pSPARC->D2_stencil_coeffs_y[a]*pSPARC->dV;
                    }

                    for(a = 1; a <= FDn; a++) {
                        s_k(1,2) += 2*(-gradz_DM[nloc[2]][nloc[1] - a][nloc[0]] + gradz_DM[nloc[2]][nloc[1] + a][nloc[0]])*pSPARC->D1_stencil_coeffs_y[a]*pSPARC->dV;
                    }

                    // z-direction
                    for(a = 1; a <= FDn; a++) {
                        s_k(2,0) += 2*(-gradx_DM[nloc[2] - a][nloc[1]][nloc[0]] + gradx_DM[nloc[2] + a][nloc[1]][nloc[0]])*pSPARC->D1_stencil_coeffs_z[a]*pSPARC->dV;
                    }

                    for(a = 1; a <= FDn; a++) {
                        s_k(2,1) += 2*(-grady_DM[nloc[2] - a][nloc[1]][nloc[0]] + grady_DM[nloc[2] + a][nloc[1]][nloc[0]])*pSPARC->D1_stencil_coeffs_z[a]*pSPARC->dV;
                    }
                    s_k(2,2) += 2*DMcol[nloc[2] + FDn][nloc[1] + FDn][nloc[0] + FDn]*pSPARC->D2_stencil_coeffs_z[0]*pSPARC->dV;
                    for(a = 1; a <= FDn; a++) {
                        s_k(2,2) += 2*(DMcol[nloc[2] + FDn - a][nloc[1] + FDn][nloc[0] + FDn] + DMcol[nloc[2] + FDn + a][nloc[1] + FDn][nloc[0] + FDn])*pSPARC->D2_stencil_coeffs_z[a]*pSPARC->dV;
                    }
			    }
                
                if(pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1) {
                    Vec3Dto1D(psipsi, grad_DM_vec, dims);
                    double *temp;
                    temp = (double *) calloc(sizeof(double), pSPARC->n_atom);
                    Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], grad_DM_vec, temp);
                    for (ii = 0; ii < pSPARC->n_atom; ii++) 
                        energy_nl += temp[ii];
                    free(temp);
                }

                Vec3Dto1D(gradx_DM, grad_DM_vec, dims);
                Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], grad_DM_vec, force_nloc);

                if(pSPARC->Calc_stress == 1) {
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 1, grad_DM_vec, &s_nl(0,0));
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 2, grad_DM_vec, &s_nl(0,1));
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 3, grad_DM_vec, &s_nl(0,2));    
                } else if (pSPARC->Calc_pres == 1) {
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 1, grad_DM_vec, &pSPARC->pres_nl);
                }

                Vec3Dto1D(grady_DM, grad_DM_vec, dims);
                Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], grad_DM_vec, force_nloc + pSPARC->n_atom);
                
                if(pSPARC->Calc_stress == 1) {
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 1, grad_DM_vec, &s_nl(1,0));
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 2, grad_DM_vec, &s_nl(1,1));
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 3, grad_DM_vec, &s_nl(1,2));    
                } else if (pSPARC->Calc_pres == 1) {
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 2, grad_DM_vec, &pSPARC->pres_nl);
                }
                
                Vec3Dto1D(gradz_DM, grad_DM_vec, dims);
                Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], grad_DM_vec, force_nloc + 2 * pSPARC->n_atom);
                
                if(pSPARC->Calc_stress == 1) {
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 1, grad_DM_vec, &s_nl(2,0));
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 2, grad_DM_vec, &s_nl(2,1));
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 3, grad_DM_vec, &s_nl(2,2));    
                } else if (pSPARC->Calc_pres == 1) {
                    Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, i-nloc[0], j-nloc[1], k-nloc[2], pSPARC->Atom_Influence_nloc_SQ[nd], 
                                        pSPARC->nlocProj_SQ[nd], 3, grad_DM_vec, &pSPARC->pres_nl);
                }
                
                nd = nd+1;
            }
        }
    } // end loop over FD nodes in proc domain

    for (i = 0; i < 3 * pSPARC->n_atom; i++) {
        force_nloc[i] *= (-4 * pSPARC->dV);
    }

    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, force_nloc, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
    }
    time2 = MPI_Wtime();

#ifdef DEBUG
    if(!rank) 
        printf("Computing non-local component of the forces takes %.3f ms\n", (time2 - time1)*1e3);

    if (!rank) {
        printf("force_nloc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", force_nloc[i], force_nloc[i+pSPARC->n_atom], force_nloc[i+pSPARC->n_atom*2]);
        }
    }    
    if (!rank) {
        printf("force_loc = \n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            printf("%18.14f %18.14f %18.14f\n", pSPARC->forces[i*3], pSPARC->forces[i*3+1], pSPARC->forces[i*3+2]);
        }
    }
#endif

    // Add nonlocal forces into correct position.
    count = 0;
    for (j = 0; j < 3; j++) {
        for (i = 0; i < pSPARC->n_atom; i++) {
            pSPARC->forces[j + 3*i] += force_nloc[count++];
        }
    }
    
    double vol;
    vol = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;

    if (pSPARC->Calc_pres + pSPARC->Calc_stress > 0) {
        pSPARC->pres_nl *= (-4 * pSPARC->dV);
        energy_nl *= (-2 * pSPARC->dV);
        pSPARC->pres_nl += energy_nl;
    }

    if (pSPARC->Calc_stress == 1) {
        if (size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &energy_nl, 1, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
            MPI_Allreduce(MPI_IN_PLACE, s_k, 9, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
            MPI_Allreduce(MPI_IN_PLACE, s_nl, 9, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
        }
        for (i = 0; i < 9; i++) {
            s_k[i] /= vol;
            s_nl[i] *= (-4 * pSPARC->dV);
            s_nl[i] /= vol;
        }
        energy_nl /= vol;

        symmetrize_stress_SQ(pSPARC->stress_k, s_k);
        symmetrize_stress_SQ(pSPARC->stress_nl, s_nl);
        pSPARC->stress_nl[0] += energy_nl;
        pSPARC->stress_nl[3] += energy_nl;
        pSPARC->stress_nl[5] += energy_nl;
    } else if (pSPARC->Calc_pres == 1) {
        if (size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &pSPARC->pres_nl, 1, MPI_DOUBLE, MPI_SUM, pSQ->dmcomm_SQ);
        }
    }

#undef s_k
#undef s_nl

    free(grad_DM_vec);
    free(force_nloc);

    for(k = 0; k < 2*nloc[2]+1; k++) {
        for(j = 0; j < 2*nloc[1]+1; j++) {
            free(gradx_DM[k][j]);
            free(grady_DM[k][j]);
            free(gradz_DM[k][j]);
            free(psipsi[k][j]);
        }
        free(gradx_DM[k]);
        free(grady_DM[k]);
        free(gradz_DM[k]);
        free(psipsi[k]);
    }
    free(gradx_DM);
    free(grady_DM);
    free(gradz_DM);
    free(psipsi);


    for(k = 0; k < 2*nloc[2]+1 + 2*FDn; k++) {
        for(j=0;j<2*nloc[1]+1 + 2*FDn;j++) {
            free(DMcol[k][j]);
        }
        free(DMcol[k]);
    }
    free(DMcol);

    if (pSPARC->SQ_gauss_mem == 0 && pSPARC->SQ_typ_dm == 2) {
        for(k = 0; k < 2*nloc[2]+1; k++) {
            for(j = 0; j < 2*nloc[1]+1; j++) {
                free(t0[k][j]);
            }
            free(t0[k]);
        }
        free(t0);
    }
}

 
/**
 * @brief   Compute chebyshev coefficients Ci (length npl_c+1) to fit the function "func"
 * 
 * NOTE:    THIS FUNCTION CURRENTLY USES DISCRETE ORTHOGONALITY PROPERTY OF CHEBYSHEV POLYNOMIALS WHICH ONLY GIVE "approximate" 
 *          COEFFICIENTS. SO IF THE FUNCTION IS NOT SMOOTH ENOUGH THE COEFFICIENTS WILL NOT BE ACCURATE ENOUGH. 
 *          SO THIS FUNCTION HAS TO BE REPLACED WITH ITS CONTINUOUS COUNTER PART WHICH EVALUATES OSCILLATORY INTEGRALS AND USES FFT.
 */
void ChebyshevCoeff_SQ(int npl_c, double *Ci, double *d, double (*fun)(double, double, double, int), 
                const double beta, const double lambda_f, const int type) {
    int k, j;
    double y, fac1, fac2, sum;

    fac1 = M_PI/(npl_c + 1);
    for (k = 0; k < npl_c + 1; k++) {
        y = cos(fac1 * (k + 0.5));
        // d[k] = (*func)(y, lambda_fp, bet);
        d[k] = fun(beta, y, lambda_f, type);
    }

    fac2 = 2.0 / (npl_c + 1);
    for (j = 0; j < npl_c + 1; j++) {
        sum = 0.0;
        for (k = 0; k < npl_c + 1; k++) {
            // sum = sum + d[k] * cos((M_PI * (j - 1 + 1)) * ((double)((k - 0.5 + 1) / (npl_c + 1))));
            sum = sum + d[k] * cos(fac1 * j * (k + 0.5));
        }
        Ci[j] = fac2 * sum;
    }
    Ci[0] = Ci[0] / 2.0;
}

/**
 * @brief   Lanczos algorithm computing only minimal eigenvales
 * 
 * TODO:    There are 2 different Lanczos algorithm with confusing names. Change them or combine them
 */
void LanczosAlgorithm_new(SPARC_OBJ *pSPARC, int i, int j, int k, double *lambda_min, int nd, int choice) {
    int rank, ii, jj, kk, ll, p, max_iter = 500;
    int *nloc, flag = 0, count;
    double ***vk, ***vkp1, val, *aa, *bb, **lanc_V, lambda_min_temp;
    double lmin_prev = 0.0, dl = 1.0, dm=1.0 ,lmax_prev=0.0;
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    SQ_OBJ *pSQ = pSPARC->pSQ;
    nloc = pSQ->nloc;


    aa = (double *) calloc(sizeof(double), pSQ->Nd_loc);
    bb = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    vk = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    vkp1 = (double ***) calloc(sizeof(double **), 2*nloc[2]+1);
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        vk[kk] = (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        vkp1[kk] =  (double **) calloc(sizeof(double *), 2*nloc[1]+1);
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            vk[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
            vkp1[kk][jj] = (double *) calloc(sizeof(double ), 2*nloc[0]+1);
        }
    }
    
    lanc_V = (double **) calloc(sizeof(double *), max_iter+1);
    for (p = 0; p < max_iter + 1; p++)
        lanc_V[p] = (double *) calloc(sizeof(double), pSQ->Nd_loc);

    Vector2Norm_local(pSQ->vec[nd], nloc, &val);

    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                pSQ->vec[nd][kk][jj][ii] = pSQ->vec[nd][kk][jj][ii] / val;
            }
        }
    }

    HsubTimesVec(pSPARC, pSQ->vec[nd], i, j, k, vk); 
    VectorDotProduct_local(pSQ->vec[nd], vk, nloc, &val);

    aa[0] = val;
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] - aa[0] * pSQ->vec[nd][kk][jj][ii];
            }
        }
    }
    Vector2Norm_local(vk, nloc, &val);
    bb[0] = val;
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                vk[kk][jj][ii] = vk[kk][jj][ii] / bb[0];
                lanc_V[flag][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] = vk[kk][jj][ii];
            }
        }
    }

    flag += 1;
    count = 0;
    while ((count < max_iter && dl > pSPARC->TOL_LANCZOS) || (choice == 1 && dm > pSPARC->TOL_LANCZOS)) {
        HsubTimesVec(pSPARC, vk, i, j, k, vkp1); // vkp1=Hsub*vk
        VectorDotProduct_local(vk, vkp1, nloc, &val); // val=vk'*vkp1
        aa[count + 1] = val;
        for (kk = 0; kk <= 2 * nloc[2]; kk++) {
            for (jj = 0; jj <= 2 * nloc[1]; jj++) {
                for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                    vkp1[kk][jj][ii] = vkp1[kk][jj][ii] - aa[count + 1] * vk[kk][jj][ii] - bb[count] * pSQ->vec[nd][kk][jj][ii];
                }
            }
        }
        Vector2Norm_local(vkp1, nloc, &val);
        bb[count + 1] = val;

        for (kk = 0; kk <= 2 * nloc[2]; kk++) {
            for (jj = 0; jj <= 2 * nloc[1]; jj++) {
                for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                    pSQ->vec[nd][kk][jj][ii] = vk[kk][jj][ii];
                    vk[kk][jj][ii] = vkp1[kk][jj][ii] / bb[count + 1];
                    lanc_V[flag][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] = vk[kk][jj][ii];
                }
            }
        }
        flag += 1;
        // Eigendecompose the tridiagonal matrix
        TridiagEigenSolve_new(pSPARC, aa, bb, count + 2, lambda_min, choice, nd);
        
        dl = fabs(( * lambda_min) - lmin_prev);
        dm = fabs((pSQ->lambda_max[nd])-lmax_prev);
        lmin_prev = * lambda_min;
        lmax_prev = pSQ->lambda_max[nd];
        count = count + 1;
    }
    lambda_min_temp = *lambda_min; *lambda_min -= pSPARC->TOL_LANCZOS;
    if (choice == 1) {
        pSQ->lambda_max[nd] += pSPARC->TOL_LANCZOS;
        // free(pSQ->low_eig_vec);
    }
    // if(choice == 0)
    //     free(pSQ->low_eig_vec);
    
    // eigenvector corresponding to lowest eigenvalue of the tridiagonal matrix of the current node
    pSQ->low_eig_vec = (double *) calloc(sizeof(double), count + 1);
    TridiagEigenSolve_new(pSPARC, aa, bb, count + 1, &lambda_min_temp, 2, nd);
    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++)
                pSQ->vec[nd][kk][jj][ii] = 0.0;
        }
    }

    for (kk = 0; kk <= 2 * nloc[2]; kk++) {
        for (jj = 0; jj <= 2 * nloc[1]; jj++) {
            for (ii = 0; ii <= 2 * nloc[0]; ii++) {
                for (ll = 0; ll < count + 1; ll++)
                    pSQ->vec[nd][kk][jj][ii] += lanc_V[ll][kk * (2 * nloc[0] + 1) * (2 * nloc[1] + 1) + jj * (2 * nloc[0] + 1) + ii] * pSQ->low_eig_vec[ll];
            }
        }
    }

    free(pSQ->low_eig_vec);

#ifdef DEBUG
    if (!rank && !nd) {
        printf("\nrank %d, nd %d, Lanczos took %d iterations. \n", rank, nd, count);
    }
#endif

    if (count == max_iter)
        printf("WARNING: Lanczos exceeded max_iter. count=%d, dl=%f \n", count, dl);
    
    for (kk = 0; kk < 2 * nloc[2] + 1; kk++) {
        for (jj = 0; jj < 2 * nloc[1] + 1; jj++) {
            free(vk[kk][jj]);
            free(vkp1[kk][jj]);
        }
        free(vk[kk]);
        free(vkp1[kk]);
    }
    free(vk);
    free(vkp1);
    free(aa);
    free(bb);
    for (p = 0; p < max_iter + 1; p++)
        free(lanc_V[p]);
    free(lanc_V);
}

/**
 * @brief   Tridiagonal solver for eigenvalues. Part of Lanczos algorithm.
 */
void TridiagEigenSolve_new(SPARC_OBJ *pSPARC, double *diag, double *subdiag, 
                            int n, double *lambda_min, int choice, int nd) {

    int i, j, k, m, l, iter, index;
    double s, r, p, g, f, dd, c, b;
    // d has diagonal and e has subdiagonal
    double *d, *e, **z; 
    SQ_OBJ* pSQ  = pSPARC->pSQ;

    d = (double *) calloc(sizeof(double), n);
    e = (double *) calloc(sizeof(double), n);
    z = (double **) calloc(sizeof(double *), n);

    for (i = 0; i < n; i++)
        z[i] = (double *) calloc(sizeof(double), n);

    if (choice == 2) {
        for (i = 0; i < n; i++) {
            z[i][i] = 1.0;
        }
    }

    //create copy of diag and subdiag in d and e
    for (i = 0; i < n; i++) {
        d[i] = diag[i];
        e[i] = subdiag[i];
    }

    // e has the subdiagonal elements 
    // ignore last element(n-1) of e, make it zero
    e[n - 1] = 0.0;

    for (l = 0; l <= n - 1; l++) {
        iter = 0;
        do {
            for (m = l; m <= n - 2; m++) {
                dd = fabs(d[m]) + fabs(d[m + 1]);
                if ((double)(fabs(e[m]) + dd) == dd) break;
            }
            if (m != l) {
                if (iter++ == 200) {
                    printf("Too many iterations in Tridiagonal solver\n");
                    exit(1);
                }
                g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                r = sqrt(g * g + 1.0); // pythag
                g = d[m] - d[l] + e[l] / (g + SIGN(r, g)); 
                s = c = 1.0;
                p = 0.0;

                for (i = m - 1; i >= l; i--) {
                    f = s * e[i];
                    b = c * e[i];
                    e[i + 1] = (r = sqrt(g * g + f * f));
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    if (choice == 2) {
                        for (k = 0; k < n; k++) {
                            f = z[k][i + 1];
                            z[k][i + 1] = s * z[k][i] + c * f;
                            z[k][i] = c * z[k][i] - s * f;
                        }
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    // go over the array d to find the smallest and largest eigenvalue
    *lambda_min = d[0];
    if (choice == 1)
        pSQ->lambda_max[nd] = d[0];
    index = 0;
    for (i = 1; i < n; i++) {
        if (choice == 1) {
            if (d[i] > pSQ->lambda_max[nd]) {
                pSQ->lambda_max[nd] = d[i];
            }
        }
        if (d[i] < *lambda_min) { 
            *lambda_min = d[i];
            index = i;
        }
    }

    if (choice == 2) {
        for (i = 0; i < n; i++)
            pSQ->low_eig_vec[i] = z[i][index];
    }

    free(d);
    free(e);
    for (i = 0; i < n; i++)
        free(z[i]);
    free(z);

    return;
}


/**
 * @brief   Calculate nonlocal term of pressure in SQ method
 */
void Calculate_nonlocal_pressure_SQ(SPARC_OBJ *pSPARC) {
    // Nonlocal pressure term are calculated in SQ force calculation.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (!rank){ 
        printf("Pressure contribution from nonlocal pseudopotential: %.15f Ha\n",pSPARC->pres_nl);
    }    
#endif
    return;
}

/**
 * @brief   Calculate nonlocal and kinetic term of stress in SQ method
 */
void Calculate_nonlocal_kinetic_stress_SQ(SPARC_OBJ *pSPARC) {
    // Nonlocal and kinetic stress are calculated in SQ force calculation.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG    
    if (!rank){
        printf("\nNon-local contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_nl, NULL);
        printf("\nKinetic contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_k, NULL);  
    } 
#endif
    return;
}

/**
 * @brief   Compute repulsive energy correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_SQ(SPARC_OBJ *pSPARC) {
    return;
}

/**
 * @brief   Compute repulsive force correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_forces_SQ(SPARC_OBJ *pSPARC) {
    return;
}


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