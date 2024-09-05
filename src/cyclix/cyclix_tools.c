/**
 * @file    cyclix_tools.c
 * @brief   This file contains the functions for performing transformations related to cyclix geometry
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>

/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

// this is for checking existence of files
#include "cyclix_tools.h"
#include "isddft.h"


void init_cyclix(SPARC_OBJ *pSPARC)
{
    pSPARC->lambda_sorted = (double *)calloc(pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof(double));
    pSPARC->occ_sorted = (double *)calloc(pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof(double));
    
    int FDn = pSPARC->order / 2; 
    pSPARC->D2_stencil_coeffs_yz = (double *)malloc((FDn + 1) * sizeof(double));
    if ( pSPARC->D2_stencil_coeffs_yz == NULL) {
        printf("\nmemory cannot be allocated\n");
        exit(EXIT_FAILURE);
    }

    LapStencil_cyclix(pSPARC);

    // Create integration weights for cyclix
    pSPARC->Intgwt_kpttopo = (double *)malloc(pSPARC->Nd_d_kptcomm * sizeof(double));
    Integration_weights_cyclix(pSPARC, pSPARC->Intgwt_kpttopo, pSPARC->DMVertices_kptcomm[0], pSPARC->Nx_d_kptcomm, pSPARC->Ny_d_kptcomm, pSPARC->Nz_d_kptcomm);    

    // Create integration weights for cyclix
    pSPARC->Intgwt_psi = (double *)malloc(pSPARC->Nd_d_dmcomm * sizeof(double));
    Integration_weights_cyclix(pSPARC, pSPARC->Intgwt_psi, pSPARC->DMVertices_dmcomm[0], pSPARC->Nx_d_dmcomm, pSPARC->Ny_d_dmcomm, pSPARC->Nz_d_dmcomm);

    // Create integration weights for cyclix
    pSPARC->Intgwt_phi = (double *) malloc(pSPARC->Nd_d * sizeof(double));
    Integration_weights_cyclix(pSPARC, pSPARC->Intgwt_phi, pSPARC->DMVertices[0], pSPARC->Nx_d, pSPARC->Ny_d, pSPARC->Nz_d);
    
    if (pSPARC->bandcomm_index == 0) {
        if (pSPARC->isGammaPoint){
            pSPARC->vl = (double *) calloc(pSPARC->Nstates * pSPARC->Nstates, sizeof(double));
            pSPARC->vr = (double *) calloc(pSPARC->Nstates * pSPARC->Nstates, sizeof(double));
            pSPARC->lambda_temp1 = (double *)calloc(pSPARC->Nstates, sizeof(double));
            pSPARC->lambda_temp2 = (double *)calloc(pSPARC->Nstates, sizeof(double));
            pSPARC->lambda_temp3 = (double *)calloc(pSPARC->Nstates, sizeof(double));
        } else{         
            pSPARC->vl_kpt = (double _Complex *) calloc(pSPARC->Nstates * pSPARC->Nstates, sizeof(double _Complex));
            pSPARC->vr_kpt = (double _Complex *) calloc(pSPARC->Nstates * pSPARC->Nstates, sizeof(double _Complex));
            pSPARC->lambda_temp1_kpt = (double _Complex *)calloc(pSPARC->Nstates, sizeof(double _Complex));
            pSPARC->lambda_temp2_kpt = (double _Complex *)calloc(pSPARC->Nstates, sizeof(double _Complex));
        }    
    }
}

void free_cyclix(SPARC_OBJ *pSPARC)
{
    free (pSPARC->lambda_sorted);
    free (pSPARC->occ_sorted);
    free(pSPARC->D2_stencil_coeffs_yz); 
    free(pSPARC->Intgwt_kpttopo);    
    free(pSPARC->Intgwt_psi);
    free(pSPARC->Intgwt_phi);    
    
    if (pSPARC->bandcomm_index == 0) {
        if (pSPARC->isGammaPoint) {
            free(pSPARC->lambda_temp1);
            free(pSPARC->lambda_temp2);
            free(pSPARC->lambda_temp3);
            free(pSPARC->vl);
            free(pSPARC->vr);
        } else {
            free(pSPARC->lambda_temp1_kpt);
            free(pSPARC->lambda_temp2_kpt);
            free(pSPARC->vl_kpt);
            free(pSPARC->vr_kpt);
        }
    }    
}

/*
@ brief: function to calculate rotational matrices in cyclix symmetry
*/

void RotMat_cyclix(SPARC_OBJ *pSPARC, double ty, double tz) {
    double theta1 = ty * pSPARC->range_y;
    double theta2 = tz * pSPARC->twist * pSPARC->range_z;

    pSPARC->RotM_cyclix[0] = cos(theta1+theta2); pSPARC->RotM_cyclix[1] = -sin(theta1+theta2); pSPARC->RotM_cyclix[2] = 0;
    pSPARC->RotM_cyclix[3] = sin(theta1+theta2); pSPARC->RotM_cyclix[4] = cos(theta1+theta2);  pSPARC->RotM_cyclix[5] = 0;
    pSPARC->RotM_cyclix[6] = 0;                  pSPARC->RotM_cyclix[7] = 0;                   pSPARC->RotM_cyclix[8] = 1;
}


/*
@ brief: function to determine cell typ corresponding to the given cyclix symmetry
*/
void CellTyp_cyclix(SPARC_OBJ *pSPARC) {
    if (pSPARC->BC == 5) {
        // Cyclic+periodic
        pSPARC->BCx = 1; pSPARC->BCy = 0; pSPARC->BCz = 0;
        pSPARC->cell_typ = 21;
    } else if (pSPARC->BC == 6) {
        // Helical
        pSPARC->BCx = 1; pSPARC->BCy = 1; pSPARC->BCz = 0;
        pSPARC->cell_typ = 22;
    } else if (pSPARC->BC == 7) {
        // Cyclic+Helical
        pSPARC->BCx = 1; pSPARC->BCy = 0; pSPARC->BCz = 0;
        pSPARC->cell_typ = 23;
    } else {
        exit(EXIT_FAILURE);
    }
}

/*
@ brief: function to determine cell parameters like vacuum, inner and outer radii
*/

void CellParm_cyclix(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int count;
    // Create useful cell parameters for cyclix
    pSPARC->xmin_at = 10000000;
    pSPARC->xmax_at = 0.0;
    for(count = 0; count < pSPARC->n_atom; count++) {
        if(pSPARC->atom_pos[3*count] < pSPARC->xmin_at)
            pSPARC->xmin_at = pSPARC->atom_pos[3*count];
        if(pSPARC->atom_pos[3*count] > pSPARC->xmax_at)
            pSPARC->xmax_at = pSPARC->atom_pos[3*count];
    }

    pSPARC->xvac = (pSPARC->range_x - (pSPARC->xmax_at - pSPARC->xmin_at))/2.0;
    pSPARC->xin = pSPARC->xmin_at - pSPARC->xvac;
    pSPARC->xout = pSPARC->xmax_at + pSPARC->xvac;
#ifdef DEBUG
    if(!rank) {
        printf("\n\nCELL_TYP: %d\n\n",pSPARC->cell_typ);
        printf("cyclix radial direction parameters:\n");
        printf("Vacuum %f, Inner radius %f, Outer radius %f\n", pSPARC->xvac, pSPARC->xin, pSPARC->xout);
    }    
#endif
}




/*
@ brief: function to store laplacian coefficients
*/

void LapStencil_cyclix(SPARC_OBJ *pSPARC) {
    double dy_inv, dx2_inv, dy2_inv, dz2_inv;
    int p;
    int FDn = pSPARC->order / 2; 
    dy_inv = 1.0 / pSPARC->delta_y;
    dx2_inv = 1.0 / (pSPARC->delta_x * pSPARC->delta_x);
    dy2_inv = 1.0 / (pSPARC->delta_y * pSPARC->delta_y);
    dz2_inv = 1.0 / (pSPARC->delta_z * pSPARC->delta_z);

    for (p = 0; p < FDn + 1; p++) {
        pSPARC->D2_stencil_coeffs_x[p] = pSPARC->FDweights_D2[p] * dx2_inv;
        pSPARC->D2_stencil_coeffs_y[p] = pSPARC->FDweights_D2[p] * dy2_inv;
        pSPARC->D2_stencil_coeffs_z[p] = pSPARC->FDweights_D2[p] * dz2_inv;
        pSPARC->D2_stencil_coeffs_yz[p] = -2.0 * pSPARC->twist * pSPARC->FDweights_D1[p] * dy_inv;
    }
}


/*
@ brief: function to convert non-cartesian to cartesian coordinates
*/

void nonCart2Cart_coord_cyclix(const SPARC_OBJ *pSPARC, double *x, double *y, double *z) {
    double x1, x2, x3;
    x1 = (*x) * cos((*y) + pSPARC->twist * (*z));
    x2 = (*x) * sin((*y) + pSPARC->twist * (*z));
    x3 = *z;
    *x = x1; *y = x2; *z = x3;
}  



/*
@ brief: function to convert cartesian to non-cartesian coordinates
*/

void Cart2nonCart_coord_cyclix(const SPARC_OBJ *pSPARC, double *x, double *y, double *z) {
    double x1, x2, x3;
    x1 = sqrt((*x) * (*x) + (*y) * (*y));
    x2 = atan2(*y,*x) - pSPARC->twist * (*z);
    if (*y < 0.0)
        x2 += 2*M_PI;
    x3 = (*z);
    *x = x1; *y = x2; *z = x3;
}


/*
@brief: function to calculate integration weights
*/

void Integration_weights_cyclix(SPARC_OBJ *pSPARC, double *Intg_wt, int ipos_x, int Nx, int Ny, int Nz) {
    int i, j, k;
    int count = 0;
    for(k = 0; k < Nz; k++) {
        for(j = 0; j < Ny; j++) {
            for(i = 0; i < Nx; i++) {
                if(ipos_x + i == 0)
                    Intg_wt[count] = (0.5 * (pSPARC->xin + (ipos_x + i) * pSPARC->delta_x) + 0.125 * pSPARC->delta_x) * pSPARC->dV;
                else if(ipos_x + i == pSPARC->Nx-1)
                    Intg_wt[count] = (0.5 * (pSPARC->xin + (ipos_x + i) * pSPARC->delta_x) - 0.125 * pSPARC->delta_x) * pSPARC->dV;
                else
                    Intg_wt[count] = (pSPARC->xin + (ipos_x + i) * pSPARC->delta_x) * pSPARC->dV;

                count++;
            }
        }
    }
}

/*
@brief: function to calculate distance between two points
*/

void CalculateDistance_cyclix(SPARC_OBJ *pSPARC, double x, double y, double z, double xref, double yref, double zref, double *d) {
    nonCart2Cart_coord_cyclix(pSPARC, &x, &y, &z);
    nonCart2Cart_coord_cyclix(pSPARC, &xref, &yref, &zref);
    *d = sqrt(pow((x-xref),2.0) + pow((y-yref),2.0) + pow((z-zref),2.0));
}


/*
@ brief: function to precondition the laplacian for Poisson equation
*/

void Jacobi_preconditioner_cyclix(SPARC_OBJ *pSPARC, int N, double c, double *r, double *f, MPI_Comm comm) {
    // Warning: Assumed phi_domain communicator calls it
    int i, j, k, count;
    int DMnx = pSPARC->Nx_d;
    int DMny = pSPARC->Ny_d;
    int DMnz = pSPARC->Nz_d;

    double *m_inv = (double *)malloc( N * sizeof(double) );
    double xin = pSPARC->xin + pSPARC->DMVertices[0] * pSPARC->delta_x; 
    double c0 = pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_z[0] + c;
    double tw2 = pSPARC->twist*pSPARC->twist;
    double x;

    count = 0;
    for(k = 0; k < DMnz; k++){
        for(j = 0; j < DMny; j++){
            for(i = 0; i < DMnx; i++){
                x = xin + i * pSPARC->delta_x;
                m_inv[count] = c0 + (tw2 + 1.0/(x*x)) * pSPARC->D2_stencil_coeffs_y[0];
                if (fabs(m_inv[count]) < 1e-14) {
                    m_inv[count] = 1.0;
                }
                m_inv[count] = -1.0/m_inv[count];
                count++;
            }
        }
    }
    
    for (i = 0; i < N; i++)
        f[i] = m_inv[i] * r[i];

    free(m_inv);  
}


/*
@ brief: function to normalize the eigenvectors for cyclix system
*/
void NormalizeEigfunc_cyclix(SPARC_OBJ *pSPARC, int spn_i) {
    if (pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

    int i, n, indx;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;

    double *intg_psi = (double *) calloc(pSPARC->Nband_bandcomm, sizeof(double));
    
    for(n = 0; n < pSPARC->Nband_bandcomm; n++){
        indx = spn_i*DMnd + n*DMndsp;
        for(i = 0; i < DMnd; i++) {
            intg_psi[n] += (pSPARC->Xorb[indx+i] * pSPARC->Xorb[indx+i])  * pSPARC->Intgwt_psi[i];
        }
    }    
    
    if(pSPARC->npNd > 0)
        MPI_Allreduce(MPI_IN_PLACE, intg_psi, pSPARC->Nband_bandcomm, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    
    for(n = 0; n < pSPARC->Nband_bandcomm; n++){
        indx = spn_i*DMnd + n*DMndsp;
        for(i = 0; i < DMnd; i++) {
            pSPARC->Xorb[indx+i] /= sqrt(intg_psi[n]);
        }
    }

    free(intg_psi);
}


/*
@ brief: function to normalize the eigenvectors for cyclix system
*/
void NormalizeEigfunc_kpt_cyclix(SPARC_OBJ *pSPARC, int spn_i, int kpt) {
    if (pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

    int i, n, indx, size_k, DMnd, spinor, DMndsp;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    size_k = DMndsp * pSPARC->Nband_bandcomm;

    double *intg_psi = (double *) calloc(pSPARC->Nband_bandcomm, sizeof(double));

    for(n = 0; n < pSPARC->Nband_bandcomm; n++){
        for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
            indx = kpt*size_k + n*DMndsp + (spinor+spn_i)*DMnd;
            for(i = 0; i < DMnd; i++) {
                intg_psi[n] += (pow(creal(pSPARC->Xorb_kpt[indx+i]), 2.0) + pow(cimag(pSPARC->Xorb_kpt[indx+i]), 2.0))  * pSPARC->Intgwt_psi[i];
            }
        }
    }

    if(pSPARC->npNd > 0)
        MPI_Allreduce(MPI_IN_PLACE, intg_psi, pSPARC->Nband_bandcomm, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);

    for(n = 0; n < pSPARC->Nband_bandcomm; n++){
        for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
            indx = kpt*size_k + n*DMndsp + (spinor+spn_i)*DMnd;
            for(i = 0; i < DMnd; i++) {
                pSPARC->Xorb_kpt[indx+i] /= sqrt(intg_psi[n]);
            }
        }
    }

    free(intg_psi);
}

/*
@ brief: generalized eigenvalue problem solver for cyclix
*/
int generalized_eigenvalue_problem_cyclix(SPARC_OBJ *pSPARC, double *Hp_local, double *Mp_local, double *eig_val)
{
    int info = LAPACKE_dggev(LAPACK_COL_MAJOR,'N','V',pSPARC->Nstates, Hp_local,
            pSPARC->Nstates, Mp_local, pSPARC->Nstates,
            pSPARC->lambda_temp1, pSPARC->lambda_temp2, pSPARC->lambda_temp3,
            pSPARC->vl, pSPARC->Nstates, pSPARC->vr, pSPARC->Nstates);    
    
    for(int n = 0; n < pSPARC->Nstates; n++){        
        // Warning if lambda_temp3 is almost zero                        
        assert(fabs(pSPARC->lambda_temp3[n]) > 1e-15);
        
        eig_val[n] = pSPARC->lambda_temp1[n]/pSPARC->lambda_temp3[n];
        for(int m = 0; m < pSPARC->Nstates; m++){
            Hp_local[n*pSPARC->Nstates+m] = pSPARC->vr[n*pSPARC->Nstates+m];
        }
    }
   return info; 
}

/*
@ brief: generalized eigenvalue problem solver for cyclix complex case
*/
int generalized_eigenvalue_problem_cyclix_kpt(SPARC_OBJ *pSPARC, double _Complex *Hp_local, double _Complex *Mp_local, double *eig_val)
{
    int info = LAPACKE_zggev(LAPACK_COL_MAJOR,'N','V',pSPARC->Nstates, Hp_local,
                pSPARC->Nstates, Mp_local,pSPARC->Nstates,
                pSPARC->lambda_temp1_kpt, pSPARC->lambda_temp2_kpt,
                pSPARC->vl_kpt, pSPARC->Nstates, pSPARC->vr_kpt, pSPARC->Nstates);
    
    for(int n = 0; n < pSPARC->Nstates; n++){        
        // Warning if lambda_temp2_kpt is almost zero
        assert(fabs(creal(pSPARC->lambda_temp2_kpt[n])) > 1e-15);
        
        eig_val[n] = creal(pSPARC->lambda_temp1_kpt[n])/creal(pSPARC->lambda_temp2_kpt[n]);
        for(int m = 0; m < pSPARC->Nstates; m++){
            Hp_local[n*pSPARC->Nstates+m] = pSPARC->vr_kpt[n*pSPARC->Nstates+m];
        }
    }
    return info;
}