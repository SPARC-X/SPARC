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

#include "isddft.h"
#include "sqDensity.h"
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
 * @brief   Calculate nonlocal forces using SQ method
 */
void Calculate_nonlocal_forces_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int rank, size, i, j, k, ii, jj, kk, a, count;
    int DMnx, DMny, DMnz, FDn, *nloc;
    double ***t0 = NULL, ***DMcol, ***gradx_DM, ***grady_DM, ***gradz_DM, ***psipsi, lambda_min, lambda_max, delta_lmin, delta_lmax;

	MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);
    MPI_Comm_size(pSQ->dmcomm_SQ, &size);
    nloc = pSQ->nloc;
    int dims[3] = {2*nloc[0]+1, 2*nloc[1]+1, 2*nloc[2]+1};

#ifdef DEBUG
    if(!rank) printf("Start computing non-local component of the forces... ");
    double time1, time2, time_lan;
    time1 = MPI_Wtime();
#endif

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

#ifdef DEBUG
    time2 = MPI_Wtime();
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
 * @brief   Compute repulsive force correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_forces_SQ(SPARC_OBJ *pSPARC) {
    return;
}

