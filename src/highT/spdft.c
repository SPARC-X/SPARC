/**
 * @file    spdft.c
 * @brief   This file contains functions for the spectral-partitioned DFT method.
 *
 * @authors Abhiraj Sharma <sharma20@llnl.gov>
 *          John E. Pask <pask1@llnl.gov>
 *
 * Copyright (c) 2022 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
	#define MKL_Complex16 double _Complex
	#include <mkl.h>
#else
	#include <cblas.h>
	#include <lapacke.h>
#endif
/* ScaLAPACK routines */
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

#include "tools.h"
#include "isddft.h"
#include "initialization.h"
#include "spdft.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))
#define TEMP_TOL 1e-15





// create plane wavevectors & allocate memory
void spDFT_initialization(SPARC_OBJ *pSPARC, double eshift) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return;
	
	double occmin = 1.0;
	double scale_x = (2*M_PI/pSPARC->range_x);
	double scale_y = (2*M_PI/pSPARC->range_y);
	double scale_z = (2*M_PI/pSPARC->range_z);
	int Nx = 0;
	int Ny = 0;
	int Nz = 0;
	double kin;
	while(occmin > pSPARC->spDFT_tol_occ) {
		Nx++; Ny++; Nz++;
		double Gx = Nx * scale_x;
		double Gy = Ny * scale_y;
		double Gz = Nz * scale_z;
		kin = 0.5*(pSPARC->lapcT[0]*Gx*Gx + 2.0*pSPARC->lapcT[1]*Gx*Gy +
                   2.0*pSPARC->lapcT[2]*Gx*Gz + pSPARC->lapcT[4]*Gy*Gy +
                   2.0*pSPARC->lapcT[5]*Gy*Gz + pSPARC->lapcT[8]*Gz*Gz);
		occmin = 1.0 / (1.0 + exp((kin-eshift)*pSPARC->Beta));
	}

	pSPARC->spDFT_NG = (2*Nx+1)*(2*Ny+1)*(2*Nz+1);
	pSPARC->spDFT_GVec = (double *) malloc(3*pSPARC->spDFT_NG * sizeof(double));
	int count = 0;
	for (int k = 0; k < (2*Nz+1); k++){
		for (int j = 0; j < (2*Ny+1); j++){
			for (int i = 0; i < (2*Nx+1); i++){
				pSPARC->spDFT_GVec[count] = scale_x * (i-Nx); pSPARC->spDFT_GVec[count+1] = scale_y * (j-Ny); pSPARC->spDFT_GVec[count+2] = scale_z * (k-Nz);
				count = count+3;
			}
		}
	}

	pSPARC->spDFT_Gocc1 = (double *) malloc(pSPARC->spDFT_NG*pSPARC->Nkpts_kptcomm*pSPARC->Nspin_spincomm * sizeof(double));
	pSPARC->spDFT_Gocc2 = (double *) malloc(pSPARC->spDFT_NG*pSPARC->Nkpts_kptcomm*pSPARC->Nspin_spincomm * sizeof(double));
	pSPARC->spDFT_Geigen = (double *) malloc(pSPARC->spDFT_NG*pSPARC->Nkpts_kptcomm*pSPARC->Nspin_spincomm * sizeof(double));
	pSPARC->spDFT_Geigen_kin_nl = (double *) malloc(pSPARC->spDFT_NG*pSPARC->Nkpts_kptcomm * sizeof(double));
	if (pSPARC->Calc_stress == 1)
		pSPARC->spDFT_Gstress_kin_nl = (double *) malloc(6*pSPARC->spDFT_NG*pSPARC->Nkpts_kptcomm * sizeof(double));
	else
		pSPARC->spDFT_Gpres_nl = (double *) malloc(pSPARC->spDFT_NG*pSPARC->Nkpts_kptcomm * sizeof(double));
	pSPARC->spDFT_Ec = (double *) malloc(pSPARC->Nkpts_kptcomm*pSPARC->Nspin_spincomm * sizeof(double));

	if(pSPARC->spDFT_isesplit_const){
		pSPARC->spDFT_esplit = (double *) malloc(pSPARC->Nkpts_kptcomm*pSPARC->Nspin_spincomm * sizeof(double));
		pSPARC->PrintspDFTFlag = 0;
		spDFT_read(pSPARC);
		int nproc_kptcomm;
		MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
		if (nproc_kptcomm > 1) 
	    	MPI_Bcast(pSPARC->spDFT_esplit, pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, MPI_DOUBLE, 0, pSPARC->kptcomm);
	}		
}






// Calculate rescaled planewaves during NPT type simulations
void spDFT_planewaves(SPARC_OBJ *pSPARC) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return;
	for (int i = 0; i < 3*pSPARC->spDFT_NG; i++){
		pSPARC->spDFT_GVec[i] /= pSPARC->scale;
	}

}





// Calculate kinetic and nonlocal contribution of plane waves
void spDFT_eigen_kinetic_nonlocal(SPARC_OBJ *pSPARC) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int NG = pSPARC->spDFT_NG;
	int *indices = (int *) malloc(NG * sizeof(int));
	double *temp = (double *) malloc(NG * sizeof(double));
	for (int kpt = 0; kpt < Nkpt; kpt++) {
		spDFT_kinetic(pSPARC, kpt);
		
		if(pSPARC->Calc_stress == 1) {
			spDFT_nonlocal_energy_stress(pSPARC, kpt);
		} else{
			Sort(pSPARC->spDFT_Geigen_kin_nl+kpt*NG, NG, pSPARC->spDFT_Geigen_kin_nl+kpt*NG, indices); // sort pw kinetic energies
			double *Geigen = pSPARC->spDFT_Geigen_kin_nl+kpt*NG;
			int count_unq = 0, count_idn;
			for (int i = 0; i < NG; i += count_idn){
				count_unq++;
				count_idn = 0;
				for (int j = i; j < NG; j++){
					if (fabs(Geigen[i] - Geigen[j]) > 1e-12){
						break;
					} else{
						count_idn++;
					}
				}
			}
		
			int *indices_unq = (int *) malloc(count_unq *sizeof(int));
			int *indices_unq_G = (int *) malloc(count_unq *sizeof(int));
			int *COUNT_idn = (int *) malloc(count_unq *sizeof(int));

			count_unq = 0;
			for (int i = 0; i < NG; i += count_idn){
				indices_unq_G[count_unq] = indices[i];
				indices_unq[count_unq] = i;
				count_idn = 0;
				for (int j = i; j < NG; j++){
					if (fabs(Geigen[i] - Geigen[j]) > 1e-12){
						break;
					} else{
						count_idn++;
					}
				}
				COUNT_idn[count_unq++] = count_idn;
			}
			spDFT_nonlocal_energy_pressure(pSPARC, kpt, count_unq, indices_unq_G, indices_unq, COUNT_idn);
			free(indices_unq);
			free(indices_unq_G);
			free(COUNT_idn);
		}

		// Sort pw kinetic+nonlocal energies in ascending order
		Sort(pSPARC->spDFT_Geigen_kin_nl+kpt*NG, NG, pSPARC->spDFT_Geigen_kin_nl+kpt*NG, indices);

		if(pSPARC->Calc_stress == 1) {
			for (int j = 0; j < 6; j++){
				int shift_stress = kpt*NG*6 + j;
				for (int i = 0; i < NG; i++){
					temp[i] = pSPARC->spDFT_Gstress_kin_nl[shift_stress+6*i];
				}
				for (int i = 0; i < NG; i++){
					pSPARC->spDFT_Gstress_kin_nl[shift_stress+6*i] = temp[indices[i]];
				}
			}
		} else {
			for (int i = 0; i < NG; i++){
				temp[i] = pSPARC->spDFT_Gpres_nl[kpt*NG+i];
			}
			for (int i = 0; i < NG; i++){
				pSPARC->spDFT_Gpres_nl[kpt*NG+i] = temp[indices[i]];
			}
		}

		
	}
	
	free (indices);
	free(temp);
}






// Calculate kinetic contribution of planewaves
void spDFT_kinetic(SPARC_OBJ *pSPARC, int kpt) {
	int shift = kpt*pSPARC->spDFT_NG;
	double k1 = pSPARC->k1_loc[kpt];
	double k2 = pSPARC->k2_loc[kpt];
	double k3 = pSPARC->k3_loc[kpt];
	for (int i = 0; i < pSPARC->spDFT_NG; i++){
		double KGx = pSPARC->spDFT_GVec[3*i] + k1;
		double KGy = pSPARC->spDFT_GVec[3*i+1] + k2;
		double KGz = pSPARC->spDFT_GVec[3*i+2] + k3;
		pSPARC->spDFT_Geigen_kin_nl[shift+i] = 0.5*(pSPARC->lapcT[0]*KGx*KGx + 2.0*pSPARC->lapcT[1]*KGx*KGy +
                       							  	2.0*pSPARC->lapcT[2]*KGx*KGz + pSPARC->lapcT[4]*KGy*KGy +
                       							  	2.0*pSPARC->lapcT[5]*KGy*KGz + pSPARC->lapcT[8]*KGz*KGz);
		if(pSPARC->Calc_stress == 1) {
			double KG1 = pSPARC->gradT[0]*KGx + pSPARC->gradT[3]*KGy + pSPARC->gradT[6]*KGz;
			double KG2 = pSPARC->gradT[1]*KGx + pSPARC->gradT[4]*KGy + pSPARC->gradT[7]*KGz;
			double KG3 = pSPARC->gradT[2]*KGx + pSPARC->gradT[5]*KGy + pSPARC->gradT[8]*KGz;
			int shift_stress = 6*(shift+i);
			pSPARC->spDFT_Gstress_kin_nl[shift_stress] = KG1*KG1;
			pSPARC->spDFT_Gstress_kin_nl[shift_stress+1] = KG1*KG2;
			pSPARC->spDFT_Gstress_kin_nl[shift_stress+2] = KG1*KG3;
			pSPARC->spDFT_Gstress_kin_nl[shift_stress+3] = KG2*KG2;
			pSPARC->spDFT_Gstress_kin_nl[shift_stress+4] = KG2*KG3;
			pSPARC->spDFT_Gstress_kin_nl[shift_stress+5] = KG3*KG3;
		}
	}
	/*double eig_cos1, eig_cos2, eig_cos3, eig_sin1, eig_sin2, eig_sin3, eig_disc;
	int FDn = pSPARC->order / 2;
	double dx = pSPARC->delta_x;
	double dy = pSPARC->delta_y;
	double dz = pSPARC->delta_z;
	for (int i = 0; i < pSPARC->spDFT_NG; i++){
		double KGx = pSPARC->spDFT_GVec[3*i] + k1;
		double KGy = pSPARC->spDFT_GVec[3*i+1] + k2;
		double KGz = pSPARC->spDFT_GVec[3*i+2] + k3;
		eig_cos1 = eig_cos2 = eig_cos3 = pSPARC->FDweights_D2[0];
		eig_sin1 = eig_sin2 = eig_sin3 = 0.0;
		for (int p = 1; p < FDn + 1; p++){
			eig_cos1 += 2.0*cos(KGx*dx*p)*pSPARC->FDweights_D2[p];
			eig_cos2 += 2.0*cos(KGy*dy*p)*pSPARC->FDweights_D2[p];
			eig_cos3 += 2.0*cos(KGz*dz*p)*pSPARC->FDweights_D2[p];
			eig_sin1 += sin(KGx*dx*p)*pSPARC->FDweights_D1[p];
			eig_sin2 += sin(KGy*dy*p)*pSPARC->FDweights_D1[p];
			eig_sin3 += sin(KGz*dz*p)*pSPARC->FDweights_D1[p];
		}
		pSPARC->spDFT_Geigen_kin_nl[i+shift] = -0.5*(pSPARC->lapcT[0]/(dx*dx)*eig_cos1 - 8.0*pSPARC->lapcT[1]/(dx*dy)*eig_sin1*eig_sin2 + 
                                                     -8.0*pSPARC->lapcT[2]/(dx*dz)*eig_sin1*eig_sin3 + pSPARC->lapcT[4]/(dy*dy)*eig_cos2 + 
                       						         -8.0*pSPARC->lapcT[5]/(dy*dz)*eig_sin2*eig_sin3 + pSPARC->lapcT[8]/(dz*dz)*eig_cos3);
	}*/

}





// Calculate nonlocal contribution of planewaves
void spDFT_nonlocal_energy_stress(SPARC_OBJ *pSPARC, int kpt) {
	double _Complex *alpha, *beta, *x_rc, *x_rld_rc;
	double dx = pSPARC->delta_x;
	double dy = pSPARC->delta_y;
	double dz = pSPARC->delta_z;
	double Lx = pSPARC->range_x;
	double Ly = pSPARC->range_y;
	double Lz = pSPARC->range_z;
	int NG = pSPARC->spDFT_NG;
	int DMnx = pSPARC->Nx_d_dmcomm;
	int DMny = pSPARC->Ny_d_dmcomm;
	int DMVertices_x =  pSPARC->DMVertices_dmcomm[0];
	int DMVertices_y =  pSPARC->DMVertices_dmcomm[2];
	int DMVertices_z =  pSPARC->DMVertices_dmcomm[4];
	double volume = Lx * Ly * Lz * pSPARC->Jacbdet;
	double k1 = pSPARC->k1_loc[kpt];
	double k2 = pSPARC->k2_loc[kpt];
	double k3 = pSPARC->k3_loc[kpt];
	int shift = kpt*NG;
	int ncol = NG;
	int tnproj = 0;
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
  		tnproj += pSPARC->nlocProj[ityp].nproj;
	}
	alpha = (double _Complex *) calloc( tnproj * ncol, sizeof(double _Complex));
	beta = (double _Complex *) calloc( tnproj * 6 * ncol, sizeof(double _Complex));
	int shift_atom = 0;
	int shift_proj = 0;
	
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
   		if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen in some psps
		int nproj = pSPARC->nlocProj[ityp].nproj;
		double x0_ref = pSPARC->atom_pos[shift_atom];
        double y0_ref = pSPARC->atom_pos[shift_atom + 1];
        double z0_ref = pSPARC->atom_pos[shift_atom + 2];

    	for (int iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
			double x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
           	double y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
           	double z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
			if (!( (fabs(x0_i - x0_ref) < TEMP_TOL || fabs(fabs(x0_i - x0_ref) - Lx) < TEMP_TOL) && (fabs(y0_i - y0_ref) < TEMP_TOL || fabs(fabs(y0_i - y0_ref) - Ly) < TEMP_TOL) && (fabs(z0_i - z0_ref) < TEMP_TOL || fabs(fabs(z0_i - z0_ref) - Lz) < TEMP_TOL))) continue;

			double _Complex a = pSPARC->dV;
       	 	double _Complex b = 1.0;
			int ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
           	x_rc = (double _Complex *) malloc( ndc * ncol * sizeof(double _Complex));
			x_rld_rc = (double _Complex *) malloc( ndc * 6 * ncol * sizeof(double _Complex));
			//int atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
			for (int n = 0; n < ncol; n++) {
				int index = 3*n;
				double KGx = pSPARC->spDFT_GVec[index] + k1;
				double KGy = pSPARC->spDFT_GVec[index+1] + k2;
				double KGz = pSPARC->spDFT_GVec[index+2] + k3;
				double KG1 = pSPARC->gradT[0]*KGx + pSPARC->gradT[3]*KGy + pSPARC->gradT[6]*KGz;
				double KG2 = pSPARC->gradT[1]*KGx + pSPARC->gradT[4]*KGy + pSPARC->gradT[7]*KGz;
				double KG3 = pSPARC->gradT[2]*KGx + pSPARC->gradT[5]*KGy + pSPARC->gradT[8]*KGz;
				int sindex = 6*n*ndc;
               	for (int i = 0; i < ndc; i++) {
					int indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
               		int k_DM = indx / (DMnx * DMny);
               		int j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
               		int i_DM = indx % DMnx;
               		double dx1 = (i_DM + DMVertices_x) * dx - x0_i;
               		double dx2 = (j_DM + DMVertices_y) * dy - y0_i;
               		double dx3 = (k_DM + DMVertices_z) * dz - z0_i;
					nonCart2Cart_coord(pSPARC, &dx1, &dx2, &dx3);
					double KG1X1R = KG1 * dx1;
					double KG1X2R = KG1 * dx2;
					double KG1X3R = KG1 * dx3;
					double KG2X2R = KG2 * dx2;
					double KG2X3R = KG2 * dx3;
					double KG3X3R = KG3 * dx3;
					double KGXR = KG1X1R + KG2X2R + KG3X3R;
					double cosKGXR = cos(KGXR);
					double sinKGXR = sin(KGXR);
	                x_rc[n*ndc+i] = cosKGXR + sinKGXR * I;
					double _Complex temp = -sinKGXR + cosKGXR * I;
					x_rld_rc[sindex+i] = temp * KG1X1R;
					x_rld_rc[sindex+ndc+i] = temp * KG1X2R;
					x_rld_rc[sindex+2*ndc+i] = temp * KG1X3R;
					x_rld_rc[sindex+3*ndc+i] = temp * KG2X2R;
					x_rld_rc[sindex+4*ndc+i] = temp * KG2X3R;
					x_rld_rc[sindex+5*ndc+i] = temp * KG3X3R;
    	        }
			}
			
			cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, ncol, ndc,
                  		&a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, x_rc, ndc, &b,
						alpha+shift_proj*ncol, nproj);
			cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, 6*ncol, ndc,
                   		&a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, x_rld_rc, ndc, &b,
                   		beta+shift_proj*6*ncol, nproj);
			free(x_rc);
			free(x_rld_rc);
		}
		shift_proj += nproj;
		shift_atom += 3*pSPARC->nAtomv[ityp];
	}

	// if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
   	MPI_Comm_size(pSPARC->dmcomm, &commsize);
    if (commsize > 1) {
       	MPI_Allreduce(MPI_IN_PLACE, alpha, tnproj * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
		MPI_Allreduce(MPI_IN_PLACE, beta, tnproj * 6 * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

	// go over all atoms and multiply gamma_Jl to the inner product
    int count = 0, count_stress = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
		if (! pSPARC->nlocProj[ityp].nproj) continue;
       	int lloc = pSPARC->localPsd[ityp];
       	int lmax = pSPARC->psd[ityp].lmax;
		int nproj = pSPARC->nlocProj[ityp].nproj;
       	for (int n = 0; n < ncol; n++) {
			int ldispl = 0;
            for (int l = 0; l <= lmax; l++) {
                // skip the local l
                if (l == lloc) {
                    ldispl += pSPARC->psd[ityp].ppl[l];
                    continue;
                }
                for (int np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    for (int m = -l; m <= l; m++) {
						double a1 = creal(alpha[count]);
						double a2 = cimag(alpha[count]);
						double coef = a1*a1 + a2*a2;
						double coef11 = coef + 2*(a1*creal(beta[count_stress]) + a2*cimag(beta[count_stress]));
						double coef12 = 2*(a1*creal(beta[count_stress+nproj]) + a2*cimag(beta[count_stress+nproj]));
						double coef13 = 2*(a1*creal(beta[count_stress+2*nproj]) + a2*cimag(beta[count_stress+2*nproj]));
						double coef22 = coef + 2*(a1*creal(beta[count_stress+3*nproj]) + a2*cimag(beta[count_stress+3*nproj]));
						double coef23 = 2*(a1*creal(beta[count_stress+4*nproj]) + a2*cimag(beta[count_stress+4*nproj]));
						double coef33 = coef + 2*(a1*creal(beta[count_stress+5*nproj]) + a2*cimag(beta[count_stress+5*nproj]));
						double Geigen = (coef * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;
						double Gstress11 = (coef11 * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;
						double Gstress12 = (coef12 * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;
						double Gstress13 = (coef13 * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;
						double Gstress22 = (coef22 * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;
						double Gstress23 = (coef23 * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;
						double Gstress33 = (coef33 * pSPARC->psd[ityp].Gamma[ldispl+np])*pSPARC->nAtomv[ityp]/volume;			
						int index = shift + n;
						pSPARC->spDFT_Geigen_kin_nl[index] += Geigen;
						int index_stress = 6*index;
						pSPARC->spDFT_Gstress_kin_nl[index_stress] += Gstress11;
						pSPARC->spDFT_Gstress_kin_nl[index_stress+1] += Gstress12;
						pSPARC->spDFT_Gstress_kin_nl[index_stress+2] += Gstress13;
						pSPARC->spDFT_Gstress_kin_nl[index_stress+3] += Gstress22;
						pSPARC->spDFT_Gstress_kin_nl[index_stress+4] += Gstress23;
						pSPARC->spDFT_Gstress_kin_nl[index_stress+5] += Gstress33;
						count++; count_stress++;
                    }
                }
                ldispl += pSPARC->psd[ityp].ppl[l];
            }
			count_stress += 5*nproj;
        }
   	}
	free(alpha);
	free(beta);
}





// Calculate nonlocal contribution of planewaves
void spDFT_nonlocal_energy_pressure(SPARC_OBJ *pSPARC, int kpt, int ncol, int *indices_G, int *indices, int *COUNT_idn) {
	double _Complex *alpha, *beta, *x_rc, *x_rld_rc;
	double dx = pSPARC->delta_x;
	double dy = pSPARC->delta_y;
	double dz = pSPARC->delta_z;
	double Lx = pSPARC->range_x;
	double Ly = pSPARC->range_y;
	double Lz = pSPARC->range_z;
	int NG = pSPARC->spDFT_NG;
	int DMnx = pSPARC->Nx_d_dmcomm;
	int DMny = pSPARC->Ny_d_dmcomm;
	int DMVertices_x =  pSPARC->DMVertices_dmcomm[0];
	int DMVertices_y =  pSPARC->DMVertices_dmcomm[2];
	int DMVertices_z =  pSPARC->DMVertices_dmcomm[4];
	double volume = Lx * Ly * Lz * pSPARC->Jacbdet;
	double k1 = pSPARC->k1_loc[kpt];
	double k2 = pSPARC->k2_loc[kpt];
	double k3 = pSPARC->k3_loc[kpt];
	int shift = kpt*NG;
	memset(pSPARC->spDFT_Gpres_nl+shift, 0, sizeof(double) * NG);
	int tnproj = 0;
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
  		tnproj += pSPARC->nlocProj[ityp].nproj;
	}
	alpha = (double _Complex *) calloc( tnproj * ncol, sizeof(double _Complex));
	beta = (double _Complex *) calloc( tnproj * ncol, sizeof(double _Complex));
	int shift_atom = 0;	
	int shift_proj = 0;
	
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
  		if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen in some psps
		int nproj = pSPARC->nlocProj[ityp].nproj;
		double x0_ref = pSPARC->atom_pos[shift_atom];
        double y0_ref = pSPARC->atom_pos[shift_atom + 1];
        double z0_ref = pSPARC->atom_pos[shift_atom + 2];
       	for (int iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
			double x0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
           	double y0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
           	double z0_i = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
			if (!( (fabs(x0_i - x0_ref) < TEMP_TOL || fabs(fabs(x0_i - x0_ref) - Lx) < TEMP_TOL) && (fabs(y0_i - y0_ref) < TEMP_TOL || fabs(fabs(y0_i - y0_ref) - Ly) < TEMP_TOL) && (fabs(z0_i - z0_ref) < TEMP_TOL || fabs(fabs(z0_i - z0_ref) - Lz) < TEMP_TOL))) continue;
			
			double _Complex a = pSPARC->dV;
       	 	double _Complex b = 1.0;
			int ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
           	x_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
			x_rld_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
			//int atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
			for (int n = 0; n < ncol; n++) {
				int index = 3*indices_G[n];
				double KG1 = pSPARC->spDFT_GVec[index] + k1;
				double KG2 = pSPARC->spDFT_GVec[index+1] + k2;
				double KG3 = pSPARC->spDFT_GVec[index+2] + k3;
            	for (int i = 0; i < ndc; i++) {
					int indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
               		int k_DM = indx / (DMnx * DMny);
               		int j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
               		int i_DM = indx % DMnx;
               		double x1 = (i_DM + DMVertices_x) * dx;
               		double x2 = (j_DM + DMVertices_y) * dy;
               		double x3 = (k_DM + DMVertices_z) * dz;
					double KGXR = KG1*(x1-x0_i) + KG2*(x2-y0_i) + KG3*(x3-z0_i);
					double cosKGXR = cos(KGXR);
					double sinKGXR = sin(KGXR);
                	x_rc[n*ndc+i] = cosKGXR + sinKGXR * I;
					x_rld_rc[n*ndc+i] = (-sinKGXR + cosKGXR * I) * KGXR;
             	}
            }
			cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, ncol, ndc, 
               	   		&a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, x_rc, ndc, &b, 
               			alpha+shift_proj * ncol, nproj);
			cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, ncol, ndc, 
               	   		&a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, x_rld_rc, ndc, &b, 
               			beta+shift_proj * ncol, nproj);
			free(x_rc);
			free(x_rld_rc);
		}
		shift_proj += nproj;
		shift_atom += 3*pSPARC->nAtomv[ityp];
	}

	// if there are domain parallelization over each band, we need to sum over all processes over domain comm
    int commsize;
   	MPI_Comm_size(pSPARC->dmcomm, &commsize);
    if (commsize > 1) {
    	MPI_Allreduce(MPI_IN_PLACE, alpha, tnproj * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
		MPI_Allreduce(MPI_IN_PLACE, beta, tnproj * ncol, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

	// go over all atoms and multiply gamma_Jl to the inner product
    int count = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
    	int lloc = pSPARC->localPsd[ityp];
    	int lmax = pSPARC->psd[ityp].lmax;
    	for (int n = 0; n < ncol; n++) {
         	int ldispl = 0;
           	for (int l = 0; l <= lmax; l++) {
           		// skip the local l
           		if (l == lloc) {
               		ldispl += pSPARC->psd[ityp].ppl[l];
               		continue;
           		}
                for (int np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                    for (int m = -l; m <= l; m++) {
						double a1 = creal(alpha[count]);
						double a2 = cimag(alpha[count]);
						double b1 = creal(beta[count]);
						double b2 = cimag(beta[count]);
						double coef1 = a1*a1 + a2*a2;
						double coef2 = coef1 + 2*(a1*b1 + a2*b2);
						double Geigen = (coef1 * pSPARC->psd[ityp].Gamma[ldispl+np]) * pSPARC->nAtomv[ityp]/volume;
						double Gpres = (coef2 * pSPARC->psd[ityp].Gamma[ldispl+np]) * pSPARC->nAtomv[ityp]/volume;
						for (int count_idn = 0; count_idn < COUNT_idn[n]; count_idn++) {
							int index = indices[n] + shift + count_idn;
                           	pSPARC->spDFT_Geigen_kin_nl[index] += Geigen;
							pSPARC->spDFT_Gpres_nl[index] += Gpres;
						}
						count++;
                    }
                }
                ldispl += pSPARC->psd[ityp].ppl[l];
            }
        }
   	}
	free(alpha);
	free(beta);
}





// Calculate electron density due to plane waves
void spDFT_electrondensity_homo(SPARC_OBJ *pSPARC, double *rho) {
	if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
	int Nspinor =  pSPARC->Nspinor_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int NG = pSPARC->spDFT_NG;
	int DMnd = pSPARC->Nd_d_dmcomm; 
	double volume = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;
	double *dens_homo = (double *) calloc(pSPARC->Nspinor, sizeof(double));

	for (int spinor = 0; spinor < Nspinor; spinor++){
		int spinor_shift = spinor + pSPARC->spinor_start_indx;
		int shift0 = spinor * Nkpt* NG;
		for (int kpt = 0; kpt < Nkpt; kpt++) {
			double woccfac = (pSPARC->occfac * (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts))/volume;
			int shift = shift0 + kpt*NG;
			for (int i = 0; i < NG; i++) {
				dens_homo[spinor_shift] += (pSPARC->spDFT_Gocc1[i+shift] - pSPARC->spDFT_Gocc2[i+shift]) * woccfac;
			}
		}
	}

	// sum over spin comm group
    if(pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, dens_homo, pSPARC->Nspinor, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, dens_homo, pSPARC->Nspinor, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

	for (int spinor = 0; spinor < pSPARC->Nspinor; spinor ++) {
		for (int i = 0; i < DMnd; i++) {
			rho[i+spinor*DMnd] += dens_homo[spinor];
		}
	}

	free(dens_homo);
}




// Calculate total eigenvalue contribution of plane waves
void spDFT_eigen_total(SPARC_OBJ *pSPARC) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

	double volume = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;
	MPI_Comm comm = pSPARC->dmcomm;
	int Nspin = pSPARC->Nspin_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int NG = pSPARC->spDFT_NG;
	int Ns = pSPARC->Nstates;
	int DMnd = pSPARC->Nd_d_dmcomm;

	for (int spn = 0; spn < Nspin; spn++) {
		double Veff_avg = 0.0;
		int shift = (pSPARC->spin_start_indx + spn) * DMnd;
		int range = shift + DMnd;
		for (int i = shift; i < range; i++) {
			Veff_avg +=  pSPARC->Veff_loc_dmcomm[i];
		}
		Veff_avg *= (pSPARC->dV/volume);
		MPI_Allreduce(MPI_IN_PLACE, &Veff_avg, 1, MPI_DOUBLE, MPI_SUM, comm);
    	
		int spn_disp = spn*Nkpt*NG;
		int spin_shift1 = spn*Nkpt*Ns;
		for (int kpt = 0; kpt < Nkpt; kpt++) {
			if (pSPARC->spDFT_isesplit_const == 0){
				int shift = spn*Nkpt + kpt;
				int spkpt_shift1 = spin_shift1 + kpt*Ns;
				pSPARC->spDFT_Ec[shift] = pSPARC->lambda_sorted[(Ns-1)+spkpt_shift1] - 10.0 * pSPARC->spDFT_tau_s;
			}
			int kpt_disp = kpt*NG;
			for (int i = 0; i < NG; i++) {
				pSPARC->spDFT_Geigen[i+kpt_disp+spn_disp] = pSPARC->spDFT_Geigen_kin_nl[i+kpt_disp] + Veff_avg;
			}
		}
	}
}








// Calculate minimum and maximum eigenvalues among plane waves
void spDFT_eigen_minmax(SPARC_OBJ *pSPARC, double *eigmin, double *eigmax) {
	if(pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

	for(int spn = 0; spn < pSPARC->Nspin_spincomm; spn++) {
        int spn_disp = spn*pSPARC->Nkpts_kptcomm*pSPARC->spDFT_NG;
        for(int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++){
            if(pSPARC->spDFT_Geigen[spn_disp + kpt*pSPARC->spDFT_NG] < *eigmin)
                *eigmin = pSPARC->spDFT_Geigen[spn_disp + kpt*pSPARC->spDFT_NG];
            if(pSPARC->spDFT_Geigen[spn_disp + (kpt+1)*pSPARC->spDFT_NG-1] > *eigmax)
                *eigmax = pSPARC->spDFT_Geigen[spn_disp + (kpt+1)*pSPARC->spDFT_NG-1];
        }
    }

}





double sigmoid(double x, double shift1, double shift2, double smearing) {
    double result = 1.0 / (1.0 + exp((x - shift1 - shift2)/smearing));
	if (result < TEMP_TOL)
		return 0.0;
	else
		return result;
}





// Occupation constrint for Fermi level calulation 
double occ_constraint_spDFT(SPARC_OBJ *pSPARC, double lambda_f) {
	int Nspin =  pSPARC->Nspin_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int Ns = pSPARC->Nstates;
    int NG = pSPARC->spDFT_NG;
	double tau_s = pSPARC->spDFT_tau_s;
	double tau_e = 1.0/pSPARC->Beta;
    double g = 0.0, Ne = pSPARC->NegCharge;
	double Ec;

    for (int spn = 0; spn < Nspin; spn++) {
		int spin_shift1 = spn*Nkpt*Ns;
		int spin_shift2 = spn*Nkpt*NG;
        for (int kpt = 0; kpt < Nkpt; kpt++) {
			double wkpt = pSPARC->kptWts_loc[kpt];
			int spkpt_shift1 = spin_shift1 + kpt*Ns;
			int spkpt_shift2 = spin_shift2 + kpt*NG;
			int shift = spn*Nkpt + kpt;
			if(pSPARC->spDFT_isesplit_const == 1)
				Ec = pSPARC->spDFT_esplit[shift] + lambda_f;
			else 
				Ec = pSPARC->spDFT_Ec[shift];
			for (int i = 0; i < Ns; i++) {
               	g += wkpt * (sigmoid(pSPARC->lambda[i+spkpt_shift1], Ec, 0.0, tau_s) * 
			                 sigmoid(pSPARC->lambda[i+spkpt_shift1], 0.0, lambda_f, tau_e));
            }

			for (int i = 0; i < NG; i++) {
                g += wkpt * sigmoid(pSPARC->spDFT_Geigen[i+spkpt_shift2], 0.0, lambda_f, tau_e) * 
					 (1.0 - sigmoid(pSPARC->spDFT_Geigen[i+spkpt_shift2], Ec, 0.0, tau_s));
            }

         }
    }
    
    g *= pSPARC->occfac / pSPARC->Nkpts; // find average
    if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find g
        MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }
    
    if (pSPARC->npkpt != 1) { // sum over processes with the same rank in kptcomm to find g
        MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }
    return g + Ne;
}







// Calculation of electronic occupations
void spDFT_occupation(SPARC_OBJ *pSPARC, double lambda_f) {
	if(pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

	int Nspin =  pSPARC->Nspin_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int Ns = pSPARC->Nstates;
    int NG = pSPARC->spDFT_NG;
	double tau_s = pSPARC->spDFT_tau_s;
	double tau_e = 1.0/pSPARC->Beta;

    for (int spn = 0; spn < Nspin; spn++) {
		int spin_shift1 = spn*Nkpt*Ns;
		int spin_shift2 = spn*Nkpt*NG;
        for (int kpt = 0; kpt < Nkpt; kpt++) {
			//double wkpt = pSPARC->kptWts_loc[kpt];
			int spkpt_shift1 = spin_shift1 + kpt*Ns;
			int spkpt_shift2 = spin_shift2 + kpt*NG;
			int shift = spn*Nkpt + kpt;
			if(pSPARC->spDFT_isesplit_const == 1)
				pSPARC->spDFT_Ec[shift] = pSPARC->spDFT_esplit[shift] + lambda_f;
            for (int i = 0; i < Ns; i++) {
                pSPARC->occ[i+spkpt_shift1] = sigmoid(pSPARC->lambda[i+spkpt_shift1], pSPARC->spDFT_Ec[shift], 0.0, tau_s) * 
							                  sigmoid(pSPARC->lambda[i+spkpt_shift1], 0.0, lambda_f, tau_e);
            }

			for (int i = 0; i < NG; i++) {
                pSPARC->spDFT_Gocc1[i+spkpt_shift2] = sigmoid(pSPARC->spDFT_Geigen[i+spkpt_shift2], 0.0, lambda_f, tau_e); 
				pSPARC->spDFT_Gocc2[i+spkpt_shift2] = pSPARC->spDFT_Gocc1[i+spkpt_shift2] * sigmoid(pSPARC->spDFT_Geigen[i+spkpt_shift2], pSPARC->spDFT_Ec[shift], 0.0, tau_s);
            }

         }
    }
    
}


double partialoccupation_sp(double x, double x0, double y0) {
	double result = 1.0;
	if (x > -36.0) // exp(-36)*3 < 1e-15
    	result +=  exp(x);
	if ((x0*x + y0) > -36.0)
		result += exp(x0*x + y0);
	if (((x0+1.0)*x + y0) > -36.0)
		result += exp((x0+1.0)*x + y0);

	result = 1.0/result;
	
	if (result < TEMP_TOL)
		return TEMP_TOL;
	else if ((1.0 - result) <  TEMP_TOL)
		return (1.0 - TEMP_TOL);
	else 
		return result;
}



double entropy_sp_limx0(double x, double x0, double y0) {
	if (x < TEMP_TOL)
		return 0.0;
	else
		 return (x/(1.0 + x0)) * (1.0 - y0 - log(x));
}



double entropy_FD(double x) {
    if (x > TEMP_TOL && (1.0-x) > TEMP_TOL)
		return (-x * log(x) - (1.0 - x) * log(1.0 - x));
	else 
		return 0.0;
}


double generate_grid(double x1, double x2, int nnodes, double *nodes) {
    double h1 = (x2-x1)/(nnodes-1);
	for (int i = 0; i < nnodes; i++){
		nodes[i] = x1 + i*h1;
	}
	return h1;
}


int find_xref_entropyref(double occ, double *x) {
	int i;
	for (i = 0; i < 15; i++) {
		if (occ >= x[i] && occ < x[i+1]) {
			break;
		}
	}
	/* clamp: when occ >= x[15], use last valid interval [x[14], x[15]] */
	if (i >= 15) i = 14;
	return i;
}



double numeric_intg_trapz_uniform(int nnodes, double shift, double h, double *f) {
	double g = 0.0;
	for (int i = 0; i < nnodes; i++){
		g += f[i];
	}
	g = shift + (g - f[0]/2.0 - f[nnodes-1]/2.0) * h;

	return g;
}



double numeric_intg_trapz_nonuniform(int nnodes, double h, double occ, double x_ref, double entropy_ref, double *dentropy) {
	double entropy = 0.0, x = x_ref;
	int i;
	for (i = 0; i < nnodes; i++){
		x = x_ref + i*h;
		if (x <= occ)
			entropy += dentropy[i];
		else 
			break;
	}
	double dentropy_occ = dentropy[i] + ((dentropy[i] - dentropy[i-1])/h) * (occ - x);
	entropy = entropy_ref + (entropy - dentropy[0]/2.0 - dentropy[i-1]/2.0) * h + (dentropy[i-1]/2.0 + dentropy_occ/2.0) * (occ - x + h);

	return entropy;
}



// Electronic entropy calculation in spDFT
double spDFT_entropy(SPARC_OBJ *pSPARC, int kpt, int spn) {
	if(pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return 0.0;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int Nkpt = pSPARC->Nkpts_kptcomm;
	//int Nspin = pSPARC->Nspin_spincomm;
	int Ns = pSPARC->Nstates;
	int NG = pSPARC->spDFT_NG;

	double mu = pSPARC->Efermi;
	//const double *eigval_sp1 = pSPARC->lambda + kpt*Ns + spn*Nkpt*Ns;
	double *Q = pSPARC->occ + kpt*Ns + spn*Nkpt*Ns;
	double *P1 = pSPARC->spDFT_Gocc1 + kpt*NG + spn*Nkpt*NG;
	double *P2 = pSPARC->spDFT_Gocc2 + kpt*NG + spn*Nkpt*NG;
	double tau_e = 1.0/pSPARC->Beta;
	double tau_s = pSPARC->spDFT_tau_s;
	
	// Set up constants
	double esplit = pSPARC->spDFT_Ec[spn*Nkpt + kpt] - mu;//eigval_sp1[Ns-1] - mu - 10.0 * tau_s;
	double A = tau_e/tau_s;
	double lgB = -esplit/tau_s;
	double lg_epsil = -34.0; // log(epsilon)
	double dentropy_meshsize = -1e-5;
	
	// Determine bounds
	double dentropy_min, dentropy_max;
	double Sdot_0 = lg_epsil - log(3.0);
	if (((1.0-A)*Sdot_0) >= lgB)
		dentropy_min = Sdot_0;
	else
		dentropy_min = (lg_epsil - log(3.0) - lgB)/A;

	Sdot_0 = -(lg_epsil + lgB)/(1.0+A);
	if ((-A*Sdot_0) <= lgB)
		dentropy_max = Sdot_0;
	else
		dentropy_max = -lg_epsil;
	
#ifdef DEBUG
	if(!rank)
		printf("Entropy calculation: Sdot_min = %f, Sdot_max = %f\n",dentropy_min, dentropy_max);
#endif

	// Find occupations between the bounds found above
	int nnodes = ceil((dentropy_min-dentropy_max)/dentropy_meshsize)+1;
	double *dentropy_grid = (double *) malloc(nnodes * sizeof(double));
	double *occ_grid = (double *) malloc(nnodes * sizeof(double));
	double *dspline_grid = (double *) malloc(nnodes * sizeof(double));
	generate_grid(dentropy_max, dentropy_min, nnodes, dentropy_grid);
	for (int i = 0; i < nnodes; i++){
		occ_grid[i] = partialoccupation_sp(dentropy_grid[i], A, lgB);
	}
	
	// Cubic spline derivative for dentropy between x1 and xN
	getYD_gen(dentropy_grid, occ_grid, dspline_grid, nnodes);

	// Entropy at occ \in [1e-15 1e-1]
	int nnodes_fine = 10001;
	double *occ_grid_fine = (double *) malloc(nnodes_fine * sizeof(double));
	double *dentropy_grid_fine = (double *) malloc(15 * nnodes_fine * sizeof(double));
	double *x = (double *) malloc(16 * sizeof(double));
	double *entropy_x = (double *) malloc(16 * sizeof(double));

	x[0] = occ_grid[0];
	entropy_x[0] = 0.0; // a practically unoccupied state
	x[1] = 1e-14;
	for (int i = 1; i <= 14; i++) {
		double occ_meshsize	= generate_grid(x[i-1], x[i], nnodes_fine, occ_grid_fine);
    	SplineInterp(occ_grid, dentropy_grid, nnodes, occ_grid_fine, dentropy_grid_fine + (i-1)*nnodes_fine, nnodes_fine, dspline_grid);
		entropy_x[i] = numeric_intg_trapz_uniform(nnodes_fine, entropy_x[i-1], occ_meshsize, dentropy_grid_fine + (i-1)*nnodes_fine);
		x[i+1] = x[i]*10.0;
	}
	x[15] = occ_grid[nnodes-1];
	double occ_meshsize	= generate_grid(x[14], x[15], nnodes_fine, occ_grid_fine);
    SplineInterp(occ_grid, dentropy_grid, nnodes, occ_grid_fine, dentropy_grid_fine + 14*nnodes_fine, nnodes_fine, dspline_grid);
	dentropy_grid_fine[15 * nnodes_fine-1] = dentropy_min; 
	entropy_x[15] = numeric_intg_trapz_uniform(nnodes_fine, entropy_x[14], occ_meshsize, dentropy_grid_fine + 14*nnodes_fine);
	
	// Entropy corresponding to KS like orbitals (sorted in descending order)
	double entropy_Q = 0.0;
	for (int i = Ns-1; i >= 0; i--){
		if (Q[i] >= x[0] && Q[i] <= x[15]){
			int index = find_xref_entropyref(Q[i], x);
			double occ_meshsize = (x[index+1] - x[index])/(nnodes_fine-1);
			double entropy = numeric_intg_trapz_nonuniform(nnodes_fine, occ_meshsize, Q[i], x[index], entropy_x[index], dentropy_grid_fine + index*nnodes_fine);
			entropy_Q += entropy;
		}
	}
	
	// Entropy corresponding to plane waves (P2 sorted in descending order)
	double entropy_P1, entropy_P2;
	entropy_P1 = entropy_P2 = 0.0;
	for (int i = NG-1; i >= 0; i--){
		entropy_P1 += entropy_FD(P1[i]);
		if (P2[i] >= x[0] && P2[i] <= x[15]){
			int index = find_xref_entropyref(P2[i], x);
			double occ_meshsize = (x[index+1] - x[index])/(nnodes_fine-1);
			double entropy = numeric_intg_trapz_nonuniform(nnodes_fine, occ_meshsize, P2[i], x[index], entropy_x[index], dentropy_grid_fine + index*nnodes_fine);
			entropy_P2 += entropy;
		}
	}
	
	double total_entropy = entropy_Q + entropy_P1 - entropy_P2;

	free(dentropy_grid);
	free(occ_grid);
	free(dspline_grid);
	free(occ_grid_fine);
	free(dentropy_grid_fine);
	free(x);
	free(entropy_x);

	return total_entropy;
}



/*
double spDFT_entropy(SPARC_OBJ *pSPARC, int kpt, int spn) {
	if(pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return 0.0;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int Nkpt = pSPARC->Nkpts_kptcomm;
	int Nspin = pSPARC->Nspin_spincomm;
	int Ns = pSPARC->Nstates;
	int NG = pSPARC->spDFT_NG;

	double mu = pSPARC->Efermi;
	const double *eigval_sp1 = pSPARC->lambda + kpt*Ns + spn*Nkpt*Ns;
	const double *Q = pSPARC->occ + kpt*Ns + spn*Nkpt*Ns;
	const double *P1 = pSPARC->spDFT_Gocc1 + kpt*NG + spn*Nkpt*NG;
	const double *P2 = pSPARC->spDFT_Gocc2 + kpt*NG + spn*Nkpt*NG;
	double tau_e = 1.0/pSPARC->Beta;
	double tau_s = pSPARC->spDFT_tau_s;
	
	// Set up constants
	double esplit = pSPARC->spDFT_Ec[spn*Nkpt + kpt] - mu;//eigval_sp1[Ns-1] - mu - 10.0 * tau_s;
	double A = tau_e/tau_s;
	double lgB = -esplit/tau_s;
	double lg_epsil = -37.0; // log(epsilon)
	double dentropy_meshsize = -1e-5;
	double occ_meshsize = 1e-7;

	// Cubic spline derivative for dentropy between x1 and xN
	double dentropy_min = (1.0/A) * (lg_epsil - lgB);
	double dentropy_max = -(1.0/(A+1)) * (lg_epsil + lgB);
	int nnodes = ceil((dentropy_min-dentropy_max)/dentropy_meshsize)+1;
	double *dentropy_grid = (double *) malloc(nnodes * sizeof(double));
	double *occ_grid = (double *) malloc(nnodes * sizeof(double));
	double *dspline_grid = (double *) malloc(nnodes * sizeof(double));
	generate_grid(dentropy_max, dentropy_min, nnodes, dentropy_grid);
	for (int i = 0; i < nnodes; i++){
		occ_grid[i] = partialoccupation_sp(dentropy_grid[i], A, lgB);
		//if (!rank)
		//	printf("%20.15E\n",occ_grid[i]);
	}

	if(!rank)
		printf("%f,%f,%20.15E,%20.15E\n",dentropy_min, dentropy_max, occ_grid[0], occ_grid[nnodes-1]);

	getYD_gen(occ_grid, dentropy_grid, dspline_grid, nnodes);

	// Entropy at x1 and xN
	double entropy_x1 = entropy_sp_limx0(occ_grid[0], A, lgB); // entropy at x1
	
	int nnodes_fine = ceil((occ_grid[nnodes-1]-occ_grid[0])/occ_meshsize)+1;
   	double *occ_grid_fine = (double *) malloc(nnodes_fine * sizeof(double));
	double *dentropy_grid_fine = (double *) malloc(nnodes_fine * sizeof(double));
	double grid_size = generate_grid(occ_grid[0], occ_grid[nnodes-1], nnodes_fine, occ_grid_fine);
	SplineInterpNonuniform(occ_grid, dentropy_grid, nnodes, occ_grid_fine, dentropy_grid_fine, nnodes_fine, dspline_grid);
	double entropy_xN = numeric_intg_trapz_uniform(nnodes_fine, entropy_x1, occ_meshsize, dentropy_grid_fine); // entropy at xN

	// Entropy corresponding to KS like orbitals (sorted in descending order)
	double entropy_Q_1, entropy_Q_2, entropy_Q_3;
	entropy_Q_1 = entropy_Q_2 = entropy_Q_3 = 0.0;
	double Q_ref = occ_grid[0];
	double entropy_Q_ref = entropy_x1;
	for (int i = Ns-1; i >= 0; i--){
		if (Q[i] > 0 && Q[i] <= occ_grid[0]){
			entropy_Q_1 += entropy_sp_limx0(Q[i], A, lgB);
		} else if (Q[i] > occ_grid[0] && Q[i] < occ_grid[nnodes-1]){
			nnodes_fine = ceil((Q[i] - Q_ref)/occ_meshsize)+1;
			if (nnodes_fine == 1){
				entropy_Q_2 += entropy_Q_ref;
				Q_ref = Q[i];
			} else{
    			double grid_size = generate_grid(Q_ref, Q[i], nnodes_fine, occ_grid_fine);
    			SplineInterpNonuniform(occ_grid, dentropy_grid, nnodes, occ_grid_fine, dentropy_grid_fine, nnodes_fine, dspline_grid);
				double entropy = numeric_intg_trapz_uniform(nnodes_fine, entropy_Q_ref, occ_meshsize, dentropy_grid_fine);
				entropy_Q_2 += entropy;
				Q_ref = Q[i];
				entropy_Q_ref = entropy;
			}
		} else if (Q[i] >= occ_grid[nnodes-1] && Q[i] <= 1) {
			entropy_Q_3 += entropy_xN - entropy_FD(occ_grid[nnodes-1]) + entropy_FD(Q[i]);
		}
	}
	double entropy_Q = entropy_Q_1 + entropy_Q_2 + entropy_Q_3;

	// Entropy corresponding to plane waves (P2 sorted in descending order)
	double entropy_P1, entropy_P2_1, entropy_P2_2, entropy_P2_3;
	entropy_P1 = entropy_P2_1 = entropy_P2_2 = entropy_P2_3 = 0.0;
	double P2_ref = occ_grid[0];
	double entropy_P2_ref = entropy_x1;
	for (int i = NG-1; i >= 0 ; i--){
		entropy_P1 += entropy_FD(P1[i]);
		if (P2[i] > 0 && P2[i] <= occ_grid[0]){
			entropy_P2_1 += entropy_sp_limx0(P2[i], A, lgB);
		} else if (P2[i] > occ_grid[0] && P2[i] < occ_grid[nnodes-1]){
			nnodes_fine = ceil((P2[i]-P2_ref)/occ_meshsize)+1;
			if(nnodes_fine == 1){
				entropy_P2_2 += entropy_P2_ref;
				P2_ref = P2[i];
			} else{ 
    			double grid_size = generate_grid(P2_ref, P2[i], nnodes_fine, occ_grid_fine);
    			SplineInterpNonuniform(occ_grid, dentropy_grid, nnodes, occ_grid_fine, dentropy_grid_fine, nnodes_fine, dspline_grid);
				double entropy = numeric_intg_trapz_uniform(nnodes_fine, entropy_P2_ref, occ_meshsize, dentropy_grid_fine);
				entropy_P2_2 += entropy;
				P2_ref = P2[i];
				entropy_P2_ref = entropy;
			}
		} else if (P2[i] >= occ_grid[nnodes-1] && P2[i] <= 1) {
			entropy_P2_3 += entropy_xN - entropy_FD(occ_grid[nnodes-1]) + entropy_FD(P2[i]);
		}
	}
	double entropy_P2 = entropy_P2_1 + entropy_P2_2 + entropy_P2_3;
	//printf("%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",entropy_Q_1, entropy_Q_2, entropy_Q_3, entropy_P2_1, entropy_P2_2, entropy_P2_3, entropy_P1);
	
	double total_entropy = entropy_Q + entropy_P1 - entropy_P2;

	free(dentropy_grid);
	free(occ_grid);
	free(dspline_grid);
	free(occ_grid_fine);
	free(dentropy_grid_fine);

	return total_entropy;
}



*/


// Stress contribution from homogenous gas in spDFT
void spDFT_stress(SPARC_OBJ *pSPARC) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

	pSPARC->spDFT_stress[0] = pSPARC->spDFT_stress[1] = pSPARC->spDFT_stress[2] = pSPARC->spDFT_stress[3] = pSPARC->spDFT_stress[4] = pSPARC->spDFT_stress[5] = 0.0;
    int Nspin = pSPARC->Nspin_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int NG = pSPARC->spDFT_NG;	
    double occfac = pSPARC->occfac;
	 double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

	for (int spn = 0; spn < Nspin; spn++) {
        for (int kpt = 0; kpt < Nkpt; kpt++) {
            double woccfac = occfac * pSPARC->kptWts_loc[kpt]/(pSPARC->Nkpts*cell_measure);
			int shift = kpt*NG + spn*Nkpt*NG;
			int shift0 = kpt*6*NG;
            for (int i = 0; i < NG; i++) {
				for (int j = 0; j < 6; j++) {
                	pSPARC->spDFT_stress[j] -= woccfac * (pSPARC->spDFT_Gocc1[i+shift] - pSPARC->spDFT_Gocc2[i+shift]) *
                                           	pSPARC->spDFT_Gstress_kin_nl[shift0+6*i+j];
            	}
			}
        }
    }
    if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find Eband
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->spDFT_stress, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }
    if (pSPARC->npkpt != 1) { // sum over processes with the same rank in kptcomm to find Eband
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->spDFT_stress, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }   
}









// Pressure contribution from homogenous gas in spDFT
void spDFT_pressure_homo(SPARC_OBJ *pSPARC) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;

	pSPARC->spDFT_pres_homo = 0.0;
    int Nspin = pSPARC->Nspin_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
	int NG = pSPARC->spDFT_NG;
    double occfac = pSPARC->occfac;

    for (int spn = 0; spn < Nspin; spn++) {
        for (int kpt = 0; kpt < Nkpt; kpt++) {
            double woccfac = occfac * pSPARC->kptWts_loc[kpt];
			int shift = kpt*NG + spn*Nkpt*NG;
			int shift0 = kpt*NG;
            for (int i = 0; i < NG; i++) {
                pSPARC->spDFT_pres_homo += woccfac * (pSPARC->spDFT_Gocc1[i+shift] - pSPARC->spDFT_Gocc2[i+shift]) *
                                           pSPARC->spDFT_Gpres_nl[i+shift0];
            }
        }
    }
    pSPARC->spDFT_pres_homo /= (-pSPARC->Nkpts);
    if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find Eband
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->spDFT_pres_homo, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }
    if (pSPARC->npkpt != 1) { // sum over processes with the same rank in kptcomm to find Eband
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->spDFT_pres_homo, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }
   
}







void spDFT_occcheck(SPARC_OBJ *pSPARC, int rank) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;
	
	int Nspin = pSPARC->Nspin_spincomm;
	int Nkpt = pSPARC->Nkpts_kptcomm;
    int NG = pSPARC->spDFT_NG;

	double eigmax = pSPARC->spDFT_Geigen[NG-1];
	double occmin = pSPARC->spDFT_Gocc1[NG-1] - pSPARC->spDFT_Gocc2[NG-1];
	for(int spn = 0; spn < Nspin; spn++) {
       	int spn_disp = spn*Nkpt*NG;
       	for(int kpt = 0; kpt < Nkpt; kpt++){
			int disp = spn_disp + (kpt+1)*NG-1;
           	if(pSPARC->spDFT_Geigen[disp] > eigmax){
               	eigmax = pSPARC->spDFT_Geigen[disp];
				occmin = pSPARC->spDFT_Gocc1[disp] - pSPARC->spDFT_Gocc2[disp];
			}
    	}
    }

	if (pSPARC->npspin != 1) {
        MPI_Allreduce(MPI_IN_PLACE, &eigmax, 1, MPI_DOUBLE, MPI_MAX, pSPARC->spin_bridge_comm);
		MPI_Allreduce(MPI_IN_PLACE, &occmin, 1, MPI_DOUBLE, MPI_MIN, pSPARC->spin_bridge_comm);
    }
        
    if (pSPARC->npkpt != 1) { 
        MPI_Allreduce(MPI_IN_PLACE, &eigmax, 1, MPI_DOUBLE, MPI_MAX, pSPARC->kpt_bridge_comm);
		MPI_Allreduce(MPI_IN_PLACE, &occmin, 1, MPI_DOUBLE, MPI_MIN, pSPARC->kpt_bridge_comm);
    }

#ifdef DEBUG
	if (!rank){
		printf("\nThe energy and occupation of the highest-energy planewave are: %10.15f, %10.15E\n",eigmax, occmin);
	}
#endif

	if(occmin > 10*pSPARC->spDFT_tol_occ) {
		if(!rank)
			printf("Warning: The occupation of the highest PW has exceeded 10x the tolerance and hence updating the maximum PW\n");
		spDFT_finalization(pSPARC);
		spDFT_initialization(pSPARC, fabs(pSPARC->Efermi));
		spDFT_eigen_kinetic_nonlocal(pSPARC);
	}
	
}





void spDFT_write(SPARC_OBJ *pSPARC, int rank) {
	int rank_kptcomm;
	MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0 || rank_kptcomm != 0) return;

	char spDFTFilename[L_STRING];
    if (rank == 0) snprintf(spDFTFilename, L_STRING, "%s", pSPARC->spDFTFilename);
    
    FILE *output_fp;
    // first create an empty file
    if (rank == 0) {
        output_fp = fopen(spDFTFilename,"w");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",spDFTFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp, ":Esplit(Ha):\n");
        fclose(output_fp);   
    }


	int Nspn = pSPARC->Nspin_spincomm;
	int Nk = pSPARC->Nkpts_kptcomm;
	int sendcount, *recvcounts, *displs;
    double *recvbuf_Ec, *Ec_all;
    sendcount = 0;
    recvcounts = NULL;
    displs = NULL;
    recvbuf_Ec = NULL;
	Ec_all = NULL;
	int *Nk_i = (int *) malloc(pSPARC->npkpt * sizeof(int));
	int *displs_all = (int *) malloc((pSPARC->npkpt+1) * sizeof(int));
    
    // first collect Ec over spin
    if (pSPARC->npspin > 1) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->spincomm_index == 0) { 
            recvbuf_Ec = (double *) malloc(pSPARC->Nspin * Nk * sizeof(double));
            recvcounts  = (int *)   malloc(pSPARC->npspin * sizeof(int)); // npspin is 2
            displs      = (int *)   malloc((pSPARC->npspin+1) * sizeof(int)); 
            int i;
            displs[0] = 0;
            for (i = 0; i < pSPARC->npspin; i++) {
                recvcounts[i] = Nspn * Nk;
                displs[i+1] = displs[i] + recvcounts[i];
            }
        } 
        // set up send info
        sendcount = Nspn * Nk;
        MPI_Gatherv(pSPARC->spDFT_Ec, sendcount, MPI_DOUBLE,
                    recvbuf_Ec, recvcounts, displs,
                    MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
        if (pSPARC->spincomm_index == 0) { 
            free(recvcounts);
            free(displs);
        }
    } else {
        recvbuf_Ec = pSPARC->spDFT_Ec;
    }
 
      

    // next collect eigval/occ over all kpoints
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->kptcomm_index == 0) {
            int i;
            Ec_all = (double *) malloc(pSPARC->Nspin * pSPARC->Nkpts_sym * sizeof(double));
            recvcounts = (int *) malloc(pSPARC->npkpt * sizeof(int));
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
               		   0, pSPARC->kpt_bridge_comm);
            displs_all[0] = 0;
            for (i = 0; i < pSPARC->npkpt; i++) {
                recvcounts[i] = Nk_i[i] * pSPARC->Nspin;
                displs_all[i+1] = displs_all[i] + recvcounts[i];
            }
        } else {
			MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
              			0, pSPARC->kpt_bridge_comm);
		}
        // set up send info
        sendcount = pSPARC->Nspin * Nk;
        MPI_Gatherv(recvbuf_Ec, sendcount, MPI_DOUBLE,
                    Ec_all, recvcounts, displs_all,
                    MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
        if (pSPARC->kptcomm_index == 0) {
            free(recvcounts);
        }
    } else {
        Nk_i[0] = Nk; // only one kptcomm
        displs_all[0] = 0;
        Ec_all = recvbuf_Ec;
    }

    // let root process print split energy to .spDFT file
    if (pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            // write to .spDFT file
            output_fp = fopen(spDFTFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",spDFTFilename);
                exit(EXIT_FAILURE);
            }
            int k, Kcomm_indx;
            if (pSPARC->Nspin == 1) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        fprintf(output_fp, "%.15f \n", 
                               Ec_all[displs_all[Kcomm_indx] + k] - pSPARC->Efermi);
                    }
                }
            } else if (pSPARC->Nspin == 2) {
                for (Kcomm_indx = 0; Kcomm_indx < pSPARC->npkpt; Kcomm_indx++) {
                    int Nk_Kcomm_indx = Nk_i[Kcomm_indx];
                    for (k = 0; k < Nk_Kcomm_indx; k++) {
                        fprintf(output_fp, "%.15f %.15f\n", 
                                Ec_all[displs_all[Kcomm_indx] + k] - pSPARC->Efermi,
 								Ec_all[displs_all[Kcomm_indx] + Nk_Kcomm_indx + k] - pSPARC->Efermi);
                    }
                }
            }
            fclose(output_fp);
        }
    }

	free(Nk_i);
    free(displs_all);

    if (pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0) { 
            free(recvbuf_Ec);
        }
    }

    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
        if (pSPARC->kptcomm_index == 0) {
            free(Ec_all);
        }
    }

}




void spDFT_read(SPARC_OBJ *pSPARC) {
	int rank_kptcomm, rank;
	MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0 || rank_kptcomm != 0) return;	

	int nval_eng = pSPARC->Nspin * pSPARC->Nkpts_sym;
	double *esplit = (double *) malloc( nval_eng * sizeof(double));
	FILE *spdft_fp = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(!rank){
		spdft_fp = fopen(pSPARC->spDFTFilename,"r");
		if (spdft_fp == NULL) {
			printf("\nERROR: SP_ISESPLIT_CONST: 1 reads the splitting energies from \"%s\", which cannot be opened.\n", pSPARC->spDFTFilename);
			exit(EXIT_FAILURE);
		}

		char str[L_STRING];
		while (fscanf(spdft_fp,"%s",str) != EOF){
			if (strcmpi(str, ":Esplit(Ha):") == 0){
				if (pSPARC->Nspin == 1){
					for (int i = 0; i < nval_eng; i++){
						if (fscanf(spdft_fp,"%lf",&esplit[i]) != 1) {
							printf("\nERROR: could not read the splitting energies from \"%s\".\n", pSPARC->spDFTFilename);
							exit(EXIT_FAILURE);
						}
					}
				} else if (pSPARC->Nspin == 2) {
					for (int i = 0; i < pSPARC->Nkpts_sym; i++){
						if (fscanf(spdft_fp,"%lf %lf",&esplit[2*i], &esplit[2*i+1]) != 2) {
							printf("\nERROR: could not read the splitting energies from \"%s\".\n", pSPARC->spDFTFilename);
							exit(EXIT_FAILURE);
						}
					}
				}
			}
		}
		fclose(spdft_fp);
    }
		
	int Nspn = pSPARC->Nspin_spincomm;
	int Nk = pSPARC->Nkpts_kptcomm;
	int *sendcounts, recvcount, *displs;
    recvcount = 0;
    sendcounts = NULL;
    displs = NULL;
    double *recvbuf_esplit;
	recvbuf_esplit = NULL;
	int *Nk_i = (int *) malloc(pSPARC->npkpt * sizeof(int));

	// First scatter splitting energies over all kpoints
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
		recvbuf_esplit = (double *) malloc(pSPARC->Nspin * Nk * sizeof(double));
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->kptcomm_index == 0) {
			int i;
            sendcounts = (int *) malloc(pSPARC->npkpt * sizeof(int));
			displs      = (int *)   malloc((pSPARC->npkpt+1) * sizeof(int));
            // collect all the number of kpoints assigned to each kptcomm
            MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
               		   0, pSPARC->kpt_bridge_comm);
            displs[0] = 0;
            for (i = 0; i < pSPARC->npkpt; i++) {
                sendcounts[i] = Nk_i[i] * pSPARC->Nspin;
                displs[i+1] = displs[i] + sendcounts[i];
            }
        } else {
			MPI_Gather(&Nk, 1, MPI_INT, Nk_i, 1, MPI_INT,
              			0, pSPARC->kpt_bridge_comm);
            
		}
        // set up send info
        recvcount = pSPARC->Nspin * Nk;
        MPI_Scatterv(esplit, sendcounts, displs, MPI_DOUBLE,
                     recvbuf_esplit, recvcount, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
       
		if (pSPARC->kptcomm_index == 0) {
            free(sendcounts);
			free(displs);
        }
    } else {
        Nk_i[0] = Nk; // only one kptcomm
        recvbuf_esplit = esplit;
    }

	if (pSPARC->npspin > 1) {
        // set up receive buffer and receive counts in kptcomm roots with spin up
        if (pSPARC->spincomm_index == 0) { 
            sendcounts  = (int *)   malloc(pSPARC->npspin * sizeof(int)); // npspin is 2
            displs      = (int *)   malloc((pSPARC->npspin+1) * sizeof(int)); 
            int i;
            displs[0] = 0;
            for (i = 0; i < pSPARC->npspin; i++) {
                sendcounts[i] = Nspn * Nk;
                displs[i+1] = displs[i] + sendcounts[i];
            }
        } 
        // set up send info
        recvcount = Nspn * Nk;
        MPI_Scatterv(recvbuf_esplit, sendcounts, displs, MPI_DOUBLE,
                    pSPARC->spDFT_esplit, recvcount, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
        if (pSPARC->spincomm_index == 0) { 
            free(sendcounts);
            free(displs);
        }
    } else {
		for (int i = 0; i < Nspn*Nk; i++)
        	pSPARC->spDFT_esplit[i] = recvbuf_esplit[i];
    }

	if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {
		free(recvbuf_esplit);
	}
	free(Nk_i);
    free(esplit);
}




// Free variables at the end of each scf cycle
void spDFT_free_scfvar(SPARC_OBJ *pSPARC) {
	if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
    	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
    		if (! pSPARC->nlocProj[ityp].nproj) {
        		free(pSPARC->nlocProj[ityp].Chi_c);                    
            	continue;
        	}
        	for (int iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
        		free(pSPARC->nlocProj[ityp].Chi_c[iat]);
        	}
        	free(pSPARC->nlocProj[ityp].Chi_c);
    	}
	}
}






// Free variables at the end of the simulation
void spDFT_finalization(SPARC_OBJ *pSPARC) {
	if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return;

	free(pSPARC->spDFT_GVec);
	free(pSPARC->spDFT_Gocc1);
	free(pSPARC->spDFT_Gocc2);
	free(pSPARC->spDFT_Geigen);
	free(pSPARC->spDFT_Geigen_kin_nl);
	if (pSPARC->Calc_stress == 1)
		free(pSPARC->spDFT_Gstress_kin_nl);
	else 
		free(pSPARC->spDFT_Gpres_nl);
	free(pSPARC->spDFT_Ec);

	if(pSPARC->spDFT_isesplit_const) {
		free(pSPARC->spDFT_esplit);
	}

}

