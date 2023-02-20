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

#include "sqProperties.h"
#include "isddft.h"
#include "sqDensity.h"
#include "sq.h"
#include "sqtool.h"
#include "sqNlocVecRoutines.h"
#include "occupation.h"
#include "stress.h"
#include "gradVecRoutines.h"
#include "tools.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate nonlocal forces using SQ method
 */
void Calculate_nonlocal_forces_SQ(SPARC_OBJ *pSPARC) {
#define s_k(i,j) s_k[3*i+j]
#define s_nl(i,j) s_nl[3*i+j]
    SQ_OBJ *pSQ = pSPARC->pSQ; 
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;

    int rank, size, i, count;
    int DMnd, FDn, *nloc;
    double *t0 = NULL, lambda_min, lambda_max;

	MPI_Comm_rank(pSQ->dmcomm_SQ, &rank);
    MPI_Comm_size(pSQ->dmcomm_SQ, &size);
    nloc = pSQ->nloc;
    int DMnx_loc = 1+2*nloc[0];
    int DMny_loc = 1+2*nloc[1];
    int DMnxny_loc = DMnx_loc * DMny_loc;

#ifdef DEBUG
    if(!rank) printf("Start computing non-local component of the forces... ");
    double time1, time2;
    time1 = MPI_Wtime();
#endif

    DMnd = pSQ->DMnd_SQ;
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

    double *force_nloc;
    // force_nloc is in order nonlocal_force_x, nonlocal_force_y, nonlocal_force_z
    force_nloc = (double *)calloc(3 * pSPARC->n_atom, sizeof(double));

    int Nx_loc = pSQ->Nx_loc;
    int Ny_loc = pSQ->Ny_loc;
    int Nz_loc = pSQ->Nz_loc;
    int Nd_loc = pSQ->Nd_loc;
    int NxNy_loc = Nx_loc * Ny_loc;

    int Nx_ex = pSQ->Nx_ex;
    int Ny_ex = pSQ->Ny_ex;
    int NxNy_ex = Nx_ex * Ny_ex;

    double *gradx_DM = (double *) calloc(sizeof(double), Nd_loc);
    double *grady_DM = (double *) calloc(sizeof(double), Nd_loc);
    double *gradz_DM = (double *) calloc(sizeof(double), Nd_loc);
    double *DMcol_loc = (double *) calloc(sizeof(double), Nd_loc);

    double *DMcol = pSQ->x_ex;
    // only need t0 for initial gauss guess when vectors are not saved for Gauss quadrature
    if (pSPARC->SQ_gauss_mem == 0) {
        t0 = (double *) calloc(sizeof(double), Nd_loc);
    }

	// loop over finite difference nodes in the processor domain
    for (int nd = 0; nd < DMnd; nd++) {
        // Density matrix by Gauss quadrature
        if (pSPARC->SQ_gauss_mem == 0) {
            memset(t0, 0, sizeof(double) * Nd_loc);
            int center = nloc[0] + nloc[1]*DMnx_loc + nloc[2]*DMnxny_loc;
            t0[center] = 1;

            LanczosAlgorithm_gauss(pSPARC, t0, &lambda_min, &lambda_max, nd);
            Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, DMcol, pSQ->lanczos_vec, pSQ->w, pSQ->gnd[nd]);
        } else {
            // Already saved
            Gauss_density_matrix_col(pSPARC, pSQ->Nd_loc, pSPARC->SQ_npl_g, DMcol, pSQ->lanczos_vec_all[nd], pSQ->w_all[nd], pSQ->gnd[nd]);
        }

        Calc_DX( DMcol, gradx_DM, FDn, 1, Nx_ex, Nx_loc, NxNy_ex, NxNy_loc, 
                0, Nx_loc, 0, Ny_loc, 0, Nz_loc, FDn, FDn, FDn, pSPARC->D1_stencil_coeffs_x, 0);
        
        Calc_DX( DMcol, grady_DM, FDn, Nx_ex, Nx_ex, Nx_loc, NxNy_ex, NxNy_loc, 
                0, Nx_loc, 0, Ny_loc, 0, Nz_loc, FDn, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0);
        
        Calc_DX( DMcol, gradz_DM, FDn, NxNy_ex, Nx_ex, Nx_loc, NxNy_ex, NxNy_loc, 
                0, Nx_loc, 0, Ny_loc, 0, Nz_loc, FDn, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0);
        
        if(pSPARC->Calc_pres == 1 || pSPARC->Calc_stress == 1) {
            restrict_to_subgrid(DMcol, DMcol_loc, Nx_loc, Nx_ex, NxNy_loc, NxNy_ex, 
                    0, Nx_loc-1, 0, Ny_loc-1, 0, Nz_loc-1, FDn, FDn, FDn);
        }

        if(pSPARC->Calc_stress == 1) {
            int center = nloc[0] + nloc[1]*DMnx_loc + nloc[2]*DMnxny_loc;
            // int center_ex = (nloc[0]+FDn) + (nloc[1]+FDn)*DMnx_locex + (nloc[2]+FDn)*DMnxny_locex;
            // D2 way converge much slower than D1*D1 way for diagonal term

            // x-direction
            // s_k(0,0) += 2* D2_1point(DMcol, center_ex, pSPARC->D2_stencil_coeffs_x, FDn, 1) * pSPARC->dV;
            s_k(0,0) += 2* D1_1point(gradx_DM, center, pSPARC->D1_stencil_coeffs_x, FDn, 1) * pSPARC->dV;
            s_k(0,1) += 2* D1_1point(grady_DM, center, pSPARC->D1_stencil_coeffs_x, FDn, 1) * pSPARC->dV;
            s_k(0,2) += 2* D1_1point(gradz_DM, center, pSPARC->D1_stencil_coeffs_x, FDn, 1) * pSPARC->dV;

            // y-direction
            s_k(1,0) += 2* D1_1point(gradx_DM, center, pSPARC->D1_stencil_coeffs_y, FDn, DMnx_loc) * pSPARC->dV;
            // s_k(1,1) += 2* D2_1point(DMcol, center_ex, pSPARC->D2_stencil_coeffs_y, FDn, DMnx_locex) * pSPARC->dV;
            s_k(1,1) += 2* D1_1point(grady_DM, center, pSPARC->D1_stencil_coeffs_y, FDn, DMnx_loc) * pSPARC->dV;
            s_k(1,2) += 2* D1_1point(gradz_DM, center, pSPARC->D1_stencil_coeffs_y, FDn, DMnx_loc) * pSPARC->dV;

            // z-direction
            s_k(2,0) += 2* D1_1point(gradx_DM, center, pSPARC->D1_stencil_coeffs_z, FDn, DMnxny_loc) * pSPARC->dV;
            s_k(2,1) += 2* D1_1point(grady_DM, center, pSPARC->D1_stencil_coeffs_z, FDn, DMnxny_loc) * pSPARC->dV;
            // s_k(2,2) += 2* D2_1point(DMcol, center_ex, pSPARC->D2_stencil_coeffs_z, FDn, DMnxny_locex) * pSPARC->dV;
            s_k(2,2) += 2* D1_1point(gradz_DM, center, pSPARC->D1_stencil_coeffs_z, FDn, DMnxny_loc) * pSPARC->dV;
        }
        
        if(pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1) {
            double *temp;
            temp = (double *) calloc(sizeof(double), pSPARC->n_atom);
            Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                        pSPARC->nlocProj_SQ[nd], DMcol_loc, temp);
            for (i = 0; i < pSPARC->n_atom; i++) 
                energy_nl += temp[i];
            free(temp);
        }

        Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                        pSPARC->nlocProj_SQ[nd], gradx_DM, force_nloc);

        if(pSPARC->Calc_stress == 1) {
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 1, gradx_DM, &s_nl(0,0));
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 2, gradx_DM, &s_nl(0,1));
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 3, gradx_DM, &s_nl(0,2));    
        } else if (pSPARC->Calc_pres == 1) {
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 1, gradx_DM, &pSPARC->pres_nl);
        }

        Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                        pSPARC->nlocProj_SQ[nd], grady_DM, force_nloc + pSPARC->n_atom);
        
        if(pSPARC->Calc_stress == 1) {
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 1, grady_DM, &s_nl(1,0));
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 2, grady_DM, &s_nl(1,1));
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 3, grady_DM, &s_nl(1,2));    
        } else if (pSPARC->Calc_pres == 1) {
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 2, grady_DM, &pSPARC->pres_nl);
        }
        
        Vnl_vec_mult_J_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                        pSPARC->nlocProj_SQ[nd], gradz_DM, force_nloc + 2 * pSPARC->n_atom);
        
        if(pSPARC->Calc_stress == 1) {
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 1, gradz_DM, &s_nl(2,0));
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 2, gradz_DM, &s_nl(2,1));
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 3, gradz_DM, &s_nl(2,2));    
        } else if (pSPARC->Calc_pres == 1) {
            Vnl_vec_mult_dir_SQ(pSPARC, pSQ->Nd_loc, nd, pSPARC->Atom_Influence_nloc_SQ[nd], 
                                pSPARC->nlocProj_SQ[nd], 3, gradz_DM, &pSPARC->pres_nl);
        }
    } 

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
    for (int j = 0; j < 3; j++) {
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

    free(force_nloc);
    free(gradx_DM);
    free(grady_DM);
    free(gradz_DM);
    free(DMcol_loc);
    if (pSPARC->SQ_gauss_mem == 0) {
        free(t0);
    }
#undef s_k
#undef s_nl
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

/**
 * @brief   Calculate second derivative of 1 point
 */
double D2_1point(double *vec, int N, double *D2_stencil_coeffs, int FDn, int stride)
{
    double D2 = vec[N] * D2_stencil_coeffs[0];
    for(int r = 1; r <= FDn; r++) {
        int rstride = r * stride;
        D2 += (vec[N + rstride] + vec[N - rstride]) * D2_stencil_coeffs[r];
    }
    return D2;
}

/**
 * @brief   Calculate first derivative of 1 point
 */
double D1_1point(double *vec, int N, double *D1_stencil_coeffs, int FDn, int stride)
{
    double D1 = 0;
    for(int r = 1; r <= FDn; r++) {
        int rstride = r * stride;
        D1 += (vec[N + rstride] - vec[N - rstride]) * D1_stencil_coeffs[r];
    }
    return D1;
}
