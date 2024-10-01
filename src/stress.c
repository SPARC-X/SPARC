/**
 * @file    stress.c
 * @brief   This file contains the functions for calculating stress.
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "stress.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"
#include "mGGAstress.h"
#include "vdWDFstress.h"
#include "d3forceStress.h"
#include "exactExchangeStress.h"
#include "spinOrbitCoupling.h"
#include "sqProperties.h"
#include "cyclix_stress.h"
#include "forces.h"
#include "exchangeCorrelation.h"

#ifdef SPARCX_ACCEL
	#include "accel.h"
	#include "accel_kpt.h"
#endif

#define TEMP_TOL 1e-12


/*
@ brief: function to calculate ionic stress/pressure and add to the electronic stress/pressure
*/
void Calculate_ionic_stress(SPARC_OBJ *pSPARC){
    if (pSPARC->CyclixFlag) {
        Calculate_ionic_stress_cyclix(pSPARC);
    } else {
        Calculate_ionic_stress_linear(pSPARC);
    }
}

void Calculate_ionic_stress_linear(SPARC_OBJ *pSPARC){
    double *stress_i, *avgvel, mass_tot = 0.0;
    stress_i = (double*) calloc(6, sizeof(double));
    avgvel = (double*) calloc(3, sizeof(double));
    int count, ityp, atm, j, k, index;
    count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
	    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
		    avgvel[0] += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3];
		    avgvel[1] += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 1];
		    avgvel[2] += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 2];
		    mass_tot += pSPARC->Mass[ityp];
		    count += 1;
	    }
	}
	
    for(j = 0; j < 3; j++)
        avgvel[j] /= mass_tot;
    
    index = 0;
    for(j = 0; j < 3; j++){
        for(k = j; k < 3; k++){
            count = 0;
            for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
	            for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
	                stress_i[index] -= pSPARC->Mass[ityp] * (pSPARC->ion_vel[count * 3 + j] - avgvel[j]) * (pSPARC->ion_vel[count * 3 + k] - avgvel[k]);
	                count++;
	            }
	        }
	        index++;
	    }
    }
    
    // Determine ionic stress and pressure
 	double cell_measure = pSPARC->Jacbdet;
    if(pSPARC->BCx == 0)
        cell_measure *= pSPARC->range_x;
    if(pSPARC->BCy == 0)
        cell_measure *= pSPARC->range_y;
    if(pSPARC->BCz == 0)
        cell_measure *= pSPARC->range_z;

    for(j = 0; j < 6; j++){
 	    pSPARC->stress_i[j] = stress_i[j]/cell_measure;
 	    pSPARC->stress[j] += pSPARC->stress_i[j];
    }
    
    if (pSPARC->BC == 2) {
        pSPARC->pres_i = -1.0*(pSPARC->stress_i[0] + pSPARC->stress_i[3] + pSPARC->stress_i[5])/3.0;
        pSPARC->pres += pSPARC->pres_i;
    }
    
    free(stress_i);
    free(avgvel);
}


/*
@ brief: function to calculate electronic stress
*/
void Calculate_electronic_stress(SPARC_OBJ *pSPARC) {
    if (pSPARC->CyclixFlag) {
        Calculate_electronic_stress_cyclix(pSPARC);
    } else {
        Calculate_electronic_stress_linear(pSPARC);
    }
}


void Calculate_electronic_stress_linear(SPARC_OBJ *pSPARC) {
	int i, rank;
#ifdef DEBUG
    double t1, t2;
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // find exchange-correlation components of stress
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    Calculate_XC_stress(pSPARC);
    if (pSPARC->ixc[2]) {
        if (pSPARC->isGammaPoint) { // metaGGA stress psi term is related to wavefunction psi directly; it needs to be computed outside of function Calculate_XC_stress
            Calculate_XC_stress_mGGA_psi_term(pSPARC); // the function is in file mgga/mGGAstress.c
        }
        else {
            Calculate_XC_stress_mGGA_psi_term_kpt(pSPARC); // the function is in file mgga/mGGAstress.c
        }
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating exchange-correlation stress components: %.3f ms\n", (t2 - t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // find local stress components
    Calculate_local_stress(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating local stress components: %.3f ms\n", (t2 - t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // find nonlocal + kinetic stress components
    Calculate_nonlocal_kinetic_stress(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating nonlocal+kinetic stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif

    if (pSPARC->usefock > 0) {
        // find exact exchange stress components
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif        
        Calculate_exact_exchange_stress(pSPARC);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank) printf("Time for calculating exact exchange stress components: %.3f ms\n", (t2 - t1)*1e3);
        #endif
    }    

    // find stress
 	if(!rank){ 	    
        for(i = 0; i < 6; i++){
            pSPARC->stress[i] = (pSPARC->stress_k[i] + pSPARC->stress_xc[i] + pSPARC->stress_nl[i] + pSPARC->stress_el[i]);
            if (pSPARC->usefock > 0) 
                pSPARC->stress[i] += pSPARC->stress_exx[i];
        }
        
        if (pSPARC->BC == 2) {
 		    pSPARC->pres = -1.0*(pSPARC->stress[0] + pSPARC->stress[3] + pSPARC->stress[5])/3.0;
 		}

        if (pSPARC->BC == 3){
            if (pSPARC->BCx == 0 && pSPARC->BCy == 0){
                pSPARC->stress[2] = pSPARC->stress[4] = pSPARC->stress[5] = 0.0;
            } else if (pSPARC->BCx == 0 && pSPARC->BCz == 0){
                pSPARC->stress[1] = pSPARC->stress[3] = pSPARC->stress[4] = 0.0;
            } else if (pSPARC->BCy == 0 && pSPARC->BCz == 0){
                pSPARC->stress[0] = pSPARC->stress[1] = pSPARC->stress[2] = 0.0;
            }
        } else if (pSPARC->BC == 4){
            if (pSPARC->BCx == 0){
                pSPARC->stress[1] = pSPARC->stress[2] = pSPARC->stress[3] =pSPARC->stress[4] = pSPARC->stress[5] = 0.0;
            } else if (pSPARC->BCy == 0){
                pSPARC->stress[0] = pSPARC->stress[1] = pSPARC->stress[2] =pSPARC->stress[4] = pSPARC->stress[5] = 0.0;
            } else if (pSPARC->BCz == 0){
                pSPARC->stress[0] = pSPARC->stress[1] = pSPARC->stress[2] =pSPARC->stress[3] = pSPARC->stress[4] = 0.0;
            }
        }


    }

#ifdef DEBUG    
    if (!rank) {
        printf("\nElectronic contribution to stress");
        PrintStress(pSPARC, pSPARC->stress, NULL); 
    }
#endif

}


/*
* @brief: find stress contributions from exchange-correlation
*         due to non-linear core correction (NLCC).
*/
void Calculate_XC_stress_nlcc(SPARC_OBJ *pSPARC, double *stress_xc_nlcc) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int rank_comm_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_comm_world);
#ifdef DEBUG
    if (!rank_comm_world) 
        printf("Start calculating NLCC exchange-correlation components of stress ...\n");
#endif

    int ityp, iat, i, j, k, p, ip, jp, kp, di, dj, dk, i_DM, j_DM, k_DM, FDn, count, count_interp,
        DMnx, DMny, nx, ny, nz, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *pshifty, *pshiftz, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x, y, z, *R, *R_interp;
    double rchrg;
    double x1_R1, x2_R2, x3_R3;
    double t1, t2, t_sort = 0.0;

    FDn = pSPARC->order / 2;
    
    DMnx = pSPARC->Nx_d;
    DMny = pSPARC->Ny_d;

    // Create indices for laplacian
    pshifty = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz = (int *)malloc( (FDn+1) * sizeof(int));
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));
    if (pshifty == NULL || pshiftz == NULL || 
        pshifty_ex == NULL || pshiftz_ex == NULL) {
        printf("\nMemory allocation failed in local forces!\n");
        exit(EXIT_FAILURE);
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];

            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            // number of finite-difference nodes in each direction of extended_rb (rb + order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes
            // number of finite-difference nodes in each direction of extended_extended_rb (rb + order) region
            nx2p = nxp + pSPARC->order;
            ny2p = nyp + pSPARC->order;
            nz2p = nzp + pSPARC->order;
            nd_2ex = nx2p * ny2p * nz2p; // total number of nodes
            
            // radii^2 of the finite difference grids of the extended_extended_rb region
            R  = (double *)malloc(sizeof(double) * nd_2ex);
            assert(R != NULL);
            
            // left corner of the 2FDn-extended-rb-region
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->order;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->order;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->order;
                        
            // relative coordinate of image atoms
            x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            if(pSPARC->cell_typ == 0) {    
                for (k = 0; k < nz2p; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < ny2p; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nx2p; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } else {
                for (k = 0; k < nz2p; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < ny2p; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nx2p; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt(pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                          + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z) );
                            //R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } 

            // VJ = (double *)malloc( nd_2ex * sizeof(double) );
            double *rhocJ = (double *)calloc( nd_2ex,sizeof(double) );
            assert(rhocJ != NULL);
            
            // avoid interpolating positions larger than rchrg
            R_interp = (double *)malloc( count_interp * sizeof(double) );
            ind_interp = (int *)malloc( count_interp * sizeof(int) );
            double *rhocJ_interp = (double *)calloc(count_interp, sizeof(double));
            count = 0;
            for (i = 0; i < nd_2ex; i++) {
                if (R[i] <= rchrg) {
                    ind_interp[count] = i; // store index
                    R_interp[count] = R[i]; // store radius value
                    count++;
                }
            }
            
            t1 = MPI_Wtime();
            SplineInterpMain(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rho_c_table, pSPARC->psd[ityp].size, 
                         R_interp, rhocJ_interp, count_interp, pSPARC->psd[ityp].SplineRhocD,pSPARC->psd[ityp].is_r_uniform);
            t2 = MPI_Wtime();
            t_sort += t2 - t1;

            for (i = 0; i < count_interp; i++) {
                rhocJ[ind_interp[i]] = rhocJ_interp[i];
            }

            free(rhocJ_interp); rhocJ_interp = NULL;
            free(R_interp); R_interp = NULL;
            free(ind_interp); ind_interp = NULL;
            free(R); R = NULL;
            
            // shift vectors initialized
            pshifty[0] = pshiftz[0] = pshifty_ex[0] = pshiftz_ex[0] = 0;
            for (p = 1; p <= FDn; p++) {
                pshifty[p] = p * nxp;
                pshiftz[p] = pshifty[p] * nyp;
                pshifty_ex[p] = p * nx2p;
                pshiftz_ex[p] = pshifty_ex[p] *ny2p;
            }
            
            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];

            // calculate drhocJ, 3 components
            double *drhocJ_x = malloc(nd_ex * sizeof(double));
            double *drhocJ_y = malloc(nd_ex * sizeof(double));
            double *drhocJ_z = malloc(nd_ex * sizeof(double));
            assert(drhocJ_x != NULL && drhocJ_y != NULL && drhocJ_z != NULL);
            for (int k2p = FDn, kp = 0; k2p < nz2p-FDn; k2p++,kp++) {
                int kshift_2p = k2p * nx2p * ny2p;
                int kshift_p = kp * nxp * nyp;
                for (int j2p = FDn, jp = 0; j2p < ny2p-FDn; j2p++,jp++) {
                    int jshift_2p = kshift_2p + j2p * nx2p;
                    int jshift_p = kshift_p + jp * nxp;
                    for (int i2p = FDn, ip = 0; i2p < nx2p-FDn; i2p++,ip++) {
                        int ishift_2p = jshift_2p + i2p;
                        int ishift_p = jshift_p + ip;
                        double drhocJ_x_val, drhocJ_y_val, drhocJ_z_val;
                        drhocJ_x_val = drhocJ_y_val = drhocJ_z_val = 0.0;
                        for (p = 1; p <= FDn; p++) {
                            drhocJ_x_val += (rhocJ[ishift_2p+p] - rhocJ[ishift_2p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            drhocJ_y_val += (rhocJ[ishift_2p+pshifty_ex[p]] - rhocJ[ishift_2p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            drhocJ_z_val += (rhocJ[ishift_2p+pshiftz_ex[p]] - rhocJ[ishift_2p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];
                        }
                        drhocJ_x[ishift_p] = drhocJ_x_val;
                        drhocJ_y[ishift_p] = drhocJ_y_val;
                        drhocJ_z[ishift_p] = drhocJ_z_val;
                    }
                }
            }

            // // find int Vxc(x) * drhocJ(x) dx
            double *Vxc = pSPARC->XCPotential;
            for (k = 0, kp = FDn, k_DM = dk; k < nz; k++, kp++, k_DM++) {
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                for (j = 0, jp = FDn, j_DM = dj; j < ny; j++, jp++, j_DM++) {
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    for (i = 0, ip = FDn, i_DM = di; i < nx; i++, ip++, i_DM++) {
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        x1_R1 = (i_DM + pSPARC->DMVertices[0]) * pSPARC->delta_x - x0_i;
                        x2_R2 = (j_DM + pSPARC->DMVertices[2]) * pSPARC->delta_y - y0_i;
                        x3_R3 = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;
                        if (pSPARC->cell_typ != 0)
                            nonCart2Cart_coord(pSPARC, &x1_R1, &x2_R2, &x3_R3);
                        double drhocJ_x_val = drhocJ_x[ishift_p];
                        double drhocJ_y_val = drhocJ_y[ishift_p];
                        double drhocJ_z_val = drhocJ_z[ishift_p];
                        if (pSPARC->cell_typ != 0)
                            nonCart2Cart_grad(pSPARC, &drhocJ_x_val, &drhocJ_y_val, &drhocJ_z_val);
                        double Vxc_val;
                        if (pSPARC->spin_typ == 0)
                            Vxc_val = Vxc[ishift_DM];
                        else
                            Vxc_val = 0.5 * (Vxc[ishift_DM] + Vxc[pSPARC->Nd_d+ishift_DM]);
                        stress_xc_nlcc[0] += drhocJ_x_val * x1_R1 * Vxc_val;
                        stress_xc_nlcc[1] += drhocJ_x_val * x2_R2 * Vxc_val;
                        stress_xc_nlcc[2] += drhocJ_x_val * x3_R3 * Vxc_val;
                        stress_xc_nlcc[3] += drhocJ_y_val * x2_R2 * Vxc_val;
                        stress_xc_nlcc[4] += drhocJ_y_val * x3_R3 * Vxc_val;
                        stress_xc_nlcc[5] += drhocJ_z_val * x3_R3 * Vxc_val;
                    }
                }
            }
            free(rhocJ);
            free(drhocJ_x);
            free(drhocJ_y);
            free(drhocJ_z);
        }
    }

    for(i = 0; i < 6; i++)
        stress_xc_nlcc[i] *= pSPARC->dV;
    t1 = MPI_Wtime();
    // sum over all domains
    MPI_Allreduce(MPI_IN_PLACE, stress_xc_nlcc, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;
       for(int i = 0; i < 6; i++)
            stress_xc_nlcc[i] /= cell_measure; 
    }

#ifdef DEBUG
    if (!rank_comm_world) { 
        printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
        printf("NLCC XC contribution to stress");
        PrintStress(pSPARC, stress_xc_nlcc, NULL); 
    }
#endif

    free(pshifty);
    free(pshiftz);
    free(pshifty_ex);
    free(pshiftz_ex);
}


/*
* @brief: find stress contributions from exchange-correlation
*/
void Calculate_XC_stress(SPARC_OBJ *pSPARC) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);

#ifdef DEBUG
    int rank_comm_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_comm_world);
    if (!rank_comm_world) printf("Start calculating exchange-correlation components of stress ...\n");
#endif

    pSPARC->stress_xc[0] = pSPARC->stress_xc[3] = pSPARC->stress_xc[5] = pSPARC->Exc - pSPARC->Exc_corr;
    pSPARC->stress_xc[1] = pSPARC->stress_xc[2] = pSPARC->stress_xc[4] = 0.0;

    if(pSPARC->isgradient){    
        int len_tot, i, DMnd;
        DMnd = pSPARC->Nd_d;
        len_tot = pSPARC->Nspdentd * DMnd;
        double *Drho_x, *Drho_y, *Drho_z;
        double *stress_xc;
        stress_xc = (double *) calloc(6, sizeof(double));
        
        Drho_x = (double *)malloc( len_tot * sizeof(double));
        Drho_y = (double *)malloc( len_tot * sizeof(double));
        Drho_z = (double *)malloc( len_tot * sizeof(double));
        assert(Drho_x != NULL && Drho_y != NULL && Drho_z != NULL);

        double *rho = (double *)malloc(len_tot * sizeof(double) );
        add_rho_core(pSPARC, pSPARC->electronDens, rho, pSPARC->Nspdentd);

        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspdentd, 0.0, rho, DMnd, Drho_x, DMnd, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspdentd, 0.0, rho, DMnd, Drho_y, DMnd, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspdentd, 0.0, rho, DMnd, Drho_z, DMnd, 2, pSPARC->dmcomm_phi);
        
        free(rho);

        if(pSPARC->cell_typ == 0){
            for(i = 0; i < len_tot; i++){
                stress_xc[0] += Drho_x[i] * Drho_x[i] * pSPARC->Dxcdgrho[i];
                stress_xc[1] += Drho_x[i] * Drho_y[i] * pSPARC->Dxcdgrho[i];
                stress_xc[2] += Drho_x[i] * Drho_z[i] * pSPARC->Dxcdgrho[i];
                stress_xc[3] += Drho_y[i] * Drho_y[i] * pSPARC->Dxcdgrho[i];
                stress_xc[4] += Drho_y[i] * Drho_z[i] * pSPARC->Dxcdgrho[i];
                stress_xc[5] += Drho_z[i] * Drho_z[i] * pSPARC->Dxcdgrho[i];    
            }
            for(i = 0; i < 6; i++)
                stress_xc[i] *= pSPARC->dV;
            
            // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
            MPI_Allreduce(MPI_IN_PLACE, stress_xc, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
        } else{
            for(i = 0; i < len_tot; i++){
                nonCart2Cart_grad(pSPARC, Drho_x + i, Drho_y + i, Drho_z + i);
                stress_xc[0] += Drho_x[i] * Drho_x[i] * pSPARC->Dxcdgrho[i];
                stress_xc[1] += Drho_x[i] * Drho_y[i] * pSPARC->Dxcdgrho[i];
                stress_xc[2] += Drho_x[i] * Drho_z[i] * pSPARC->Dxcdgrho[i];
                stress_xc[3] += Drho_y[i] * Drho_y[i] * pSPARC->Dxcdgrho[i];
                stress_xc[4] += Drho_y[i] * Drho_z[i] * pSPARC->Dxcdgrho[i];
                stress_xc[5] += Drho_z[i] * Drho_z[i] * pSPARC->Dxcdgrho[i];  
            }
            for(i = 0; i < 6; i++)
                stress_xc[i] *= pSPARC->dV;
            
            // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
            MPI_Allreduce(MPI_IN_PLACE, stress_xc, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
        }
        
        for(i = 0; i < 6; i++)
            pSPARC->stress_xc[i] -= stress_xc[i];
        
        // deallocate
        free(stress_xc);
        free(Drho_x); free(Drho_y); free(Drho_z);
    }

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

       for(int i = 0; i < 6; i++)
            pSPARC->stress_xc[i] /= cell_measure; 

    }

    if (pSPARC->NLCC_flag) {
        // double stress_xc_nlcc[6] = {0,0,0,0,0,0};
        double *stress_xc_nlcc = (double *)calloc(6,sizeof(double));
        assert(stress_xc_nlcc != NULL);
        Calculate_XC_stress_nlcc(pSPARC, stress_xc_nlcc);
        for(int i = 0; i < 6; i++)
            pSPARC->stress_xc[i] += stress_xc_nlcc[i];
        free(stress_xc_nlcc);
    }

    if (pSPARC->d3Flag == 1) {
        d3_grad_cell_stress(pSPARC); 
    }

    if (pSPARC->ixc[3] != 0) { // either vdW_DF1 or vdW_DF2
        Calculate_XC_stress_vdWDF(pSPARC); // the function is in file vdW/vdWDF/vdWDF.c
    }

#ifdef DEBUG    
    if (!rank_comm_world) {
        printf("\nXC contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_xc, NULL); 
    }
#endif
}




/*
@ brief: function to calculate the local stress components
*/

void Calculate_local_stress(SPARC_OBJ *pSPARC) {
#define INDEX_EX(i,j,k) ((k)*nx2p*ny2p+(j)*nx2p+(i))
#define INDEX_Dx1(i,j,k) ((k)*nx2p*nyp+(j)*nx2p+(i))
#define INDEX_Dx2(i,j,k) ((k)*nxp*ny2p+(j)*nxp+(i))
#define INDEX_Dx3(i,j,k) ((k)*nxp*nyp+(j)*nxp+(i))
#define dV1(i,j,k) dV1[(k)*(nx2p)*(nyp)+(j)*(nx2p)+(i)]
#define dV2(i,j,k) dV2[(k)*(nxp)*(ny2p)+(j)*(nxp)+(i)]
#define dV3(i,j,k) dV3[(k)*(nxp)*(nyp)+(j)*(nxp)+(i)]
#define dV_ref1(i,j,k) dV_ref1[(k)*(nx2p)*(nyp)+(j)*(nx2p)+(i)]
#define dV_ref2(i,j,k) dV_ref2[(k)*(nxp)*(ny2p)+(j)*(nxp)+(i)]
#define dV_ref3(i,j,k) dV_ref3[(k)*(nxp)*(nyp)+(j)*(nxp)+(i)]   

	if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    
    int ityp, iat, i, j, k, p, ip, jp, kp, ip2, jp2, kp2, di, dj, dk, i_DM, j_DM, k_DM, FDn, count, count_interp,
        DMnx, DMny, DMnd, nx, ny, nz, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *pshifty, *pshiftz, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x, y, z, *R,
           *VJ, *VJ_ref, *bJ, *bJ_ref, 
           DbJ_x_val, DbJ_ref_x_val, DbJ_y_val, DbJ_ref_y_val, DbJ_z_val, DbJ_ref_z_val, 
           DVJ_x_val, DVJ_y_val, DVJ_z_val, DVJ_ref_x_val, DVJ_ref_y_val, DVJ_ref_z_val,
           *R_interp, *VJ_interp;
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    double temp1, temp2, temp3, temp_x, temp_y, temp_z;
    double x1_R1, x2_R2, x3_R3;
    
    double *stress_el, *stress_corr;
    
    stress_el = (double*) calloc(6, sizeof(double));
    stress_corr = (double*) calloc(6,  sizeof(double));

    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (!rank) printf("Start calculating local components of stress ...\n");
#endif
    ////////////////////////////
    double t1, t2, t_sort = 0.0;
    ////////////////////////////
    double *Lap_wt, *Lap_stencil;
    
    FDn = pSPARC->order / 2;
    w2_diag = (pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0] + pSPARC->D2_stencil_coeffs_z[0]) * -inv_4PI;
    if(pSPARC->cell_typ == 0){
        Lap_wt = (double *)malloc((3*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt;
    } else{
        Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt+5;
    }
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, -inv_4PI);

    // Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;
    DMnx = pSPARC->Nx_d; DMny = pSPARC->Ny_d; // DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d;
    
    // shift vectors initialized
    pshifty = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz = (int *)malloc( (FDn+1) * sizeof(int));
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));

    // find gradient of phi
    double *Dphi_x, *Dphi_y, *Dphi_z;
    Dphi_x = (double *)malloc( DMnd * sizeof(double));
    Dphi_y = (double *)malloc( DMnd * sizeof(double));
    Dphi_z = (double *)malloc( DMnd * sizeof(double));
    
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, DMnd, Dphi_x, DMnd, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, DMnd, Dphi_y, DMnd, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, DMnd, Dphi_z, DMnd, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd; i++){
        nonCart2Cart_grad(pSPARC, Dphi_x + i, Dphi_y + i, Dphi_z + i);
        temp1 = 0.5 * (pSPARC->psdChrgDens[i] - pSPARC->electronDens[i]) * pSPARC->elecstPotential[i];
        stress_el[0] += inv_4PI * Dphi_x[i] * Dphi_x[i] + temp1;
        stress_el[1] += inv_4PI * Dphi_x[i] * Dphi_y[i];
        stress_el[2] += inv_4PI * Dphi_x[i] * Dphi_z[i];
        stress_el[3] += inv_4PI * Dphi_y[i] * Dphi_y[i] + temp1;
        stress_el[4] += inv_4PI * Dphi_y[i] * Dphi_z[i];
        stress_el[5] += inv_4PI * Dphi_z[i] * Dphi_z[i] + temp1;
    }

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;

            // number of finite-difference nodes in each direction of extended_rb (rb + order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes
            // number of finite-difference nodes in each direction of extended_extended_rb (rb + order) region
            nx2p = nxp + pSPARC->order;
            ny2p = nyp + pSPARC->order;
            nz2p = nzp + pSPARC->order;
            nd_2ex = nx2p * ny2p * nz2p; // total number of nodes
            
            // radii^2 of the finite difference grids of the extended_extended_rb region
            R  = (double *)malloc(sizeof(double) * nd_2ex);
            if (R == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            }
            
            // left corner of the 2FDn-extended-rb-region
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->order;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->order;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->order;
                        
            // relative coordinate of image atoms
            x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            if(pSPARC->cell_typ == 0) {    
                for (k = 0; k < nz2p; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < ny2p; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nx2p; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } else {
                for (k = 0; k < nz2p; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < ny2p; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nx2p; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt(pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                          + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z) );
                            //R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } 
            
            VJ_ref = (double *)malloc( nd_2ex * sizeof(double) );
            if (VJ_ref == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            // Calculate pseudopotential reference
            Calculate_Pseudopot_Ref(R, nd_2ex, pSPARC->REFERENCE_CUTOFF, -pSPARC->Znucl[ityp], VJ_ref);
            
            VJ = (double *)malloc( nd_2ex * sizeof(double) );
            if (VJ == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            // avoid sorting positions larger than rchrg
            VJ_interp = (double *)malloc( count_interp * sizeof(double) );
            R_interp = (double *)malloc( count_interp * sizeof(double) );
            ind_interp = (int *)malloc( count_interp * sizeof(int) );
            count = 0;
            for (i = 0; i < nd_2ex; i++) {
                if (R[i] <= rchrg) {
                    ind_interp[count] = i; // store index
                    R_interp[count] = R[i]; // store radius value
                    count++;
                } else {
                    VJ[i] = -pSPARC->Znucl[ityp] / R[i];
                }
            }
            
            t1 = MPI_Wtime();
            
            // sort R_interp and then apply cubic spline interpolation to find VJ
            // notice here we extract out positions within radius rchrg
            if (pSPARC->psd[ityp].is_r_uniform == 1) {
                SplineInterpUniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
                                    R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD); 
            } else {
                SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
                                    R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD); 
            }

            t2 = MPI_Wtime();
            t_sort += t2 - t1;
            
            for (i = 0; i < count_interp; i++) {
                if (R_interp[i] < TEMP_TOL) {
                    VJ[ind_interp[i]] = pSPARC->psd[ityp].Vloc_0;
                } else {
                    VJ[ind_interp[i]] =  VJ_interp[i]/R_interp[i];
                }
            }
            free(VJ_interp); VJ_interp = NULL;
            free(R_interp); R_interp = NULL;
            free(ind_interp); ind_interp = NULL;
            free(R); R = NULL;
            
            // shift vectors initialized
            pshifty[0] = pshiftz[0] = pshifty_ex[0] = pshiftz_ex[0] = 0;
            for (p = 1; p <= FDn; p++) {
                pshifty[p] = p * nxp;
                pshiftz[p] = pshifty[p] * nyp;
                pshifty_ex[p] = p * nx2p;
                pshiftz_ex[p] = pshifty_ex[p] *ny2p;
            }
                    
            // calculate pseudocharge density bJ and bJ_ref in the FDn+rb-domain
            bJ = (double *)malloc( nd_ex * sizeof(double) );
            bJ_ref = (double *)malloc( nd_ex * sizeof(double) );
            if (bJ == NULL || bJ_ref == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            double xin = pSPARC->xin + pSPARC->Atom_Influence_local[ityp].xs[iat] * pSPARC->delta_x;       
            Calc_lapV(pSPARC, VJ, FDn, nx2p, ny2p, nz2p, nxp, nyp, nzp, Lap_wt, w2_diag, xin, -inv_4PI, bJ);
            Calc_lapV(pSPARC, VJ_ref, FDn, nx2p, ny2p, nz2p, nxp, nyp, nzp, Lap_wt, w2_diag, xin, -inv_4PI, bJ_ref);

            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];
            if(pSPARC->cell_typ == 0){
                for(kp = FDn, kp2 = pSPARC->order, k_DM = dk; kp2 < nzp; kp++, kp2++, k_DM++) {
                    int kshift_DM = k_DM * DMnx * DMny;
                    int kshift_2p = kp2 * nx2p * ny2p;
                    int kshift_p = kp * nxp * nyp;
                    for(jp = FDn, jp2 = pSPARC->order, j_DM = dj; jp2 < nyp; jp++, jp2++, j_DM++) {
                        int jshift_DM = kshift_DM + j_DM * DMnx;
                        int jshift_2p = kshift_2p + jp2 * nx2p;
                        int jshift_p = kshift_p + jp * nxp;
                        //#pragma simd
                        for(ip = FDn, ip2 = pSPARC->order, i_DM = di; ip2 < nxp ; ip++, ip2++, i_DM++) {
                            int ishift_DM = jshift_DM + i_DM;
                            int ishift_2p = jshift_2p + ip2;
                            int ishift_p = jshift_p + ip;
                            DbJ_x_val = DbJ_y_val = DbJ_z_val = 0.0;
                            DbJ_ref_x_val = DbJ_ref_y_val = DbJ_ref_z_val = 0.0;
                            DVJ_x_val = DVJ_y_val = DVJ_z_val = 0.0;
                            DVJ_ref_x_val = DVJ_ref_y_val = DVJ_ref_z_val = 0.0;

                            for (p = 1; p <= FDn; p++) {
                                DbJ_x_val += (bJ[ishift_p+p] - bJ[ishift_p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DbJ_y_val += (bJ[ishift_p+pshifty[p]] - bJ[ishift_p-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DbJ_z_val += (bJ[ishift_p+pshiftz[p]] - bJ[ishift_p-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                                DbJ_ref_x_val += (bJ_ref[ishift_p+p] - bJ_ref[ishift_p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DbJ_ref_y_val += (bJ_ref[ishift_p+pshifty[p]] - bJ_ref[ishift_p-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DbJ_ref_z_val += (bJ_ref[ishift_p+pshiftz[p]] - bJ_ref[ishift_p-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                                DVJ_x_val += (VJ[ishift_2p+p] - VJ[ishift_2p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DVJ_y_val += (VJ[ishift_2p+pshifty_ex[p]] - VJ[ishift_2p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DVJ_z_val += (VJ[ishift_2p+pshiftz_ex[p]] - VJ[ishift_2p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                                DVJ_ref_x_val += (VJ_ref[ishift_2p+p] - VJ_ref[ishift_2p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DVJ_ref_y_val += (VJ_ref[ishift_2p+pshifty_ex[p]] - VJ_ref[ishift_2p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DVJ_ref_z_val += (VJ_ref[ishift_2p+pshiftz_ex[p]] - VJ_ref[ishift_2p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                            }
                            
                            // find integrals in the stress expression
                            x1_R1 = (i_DM + pSPARC->DMVertices[0]) * pSPARC->delta_x - x0_i;
                            x2_R2 = (j_DM + pSPARC->DMVertices[2]) * pSPARC->delta_y - y0_i;
                            x3_R3 = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;

                            stress_el[0] += DbJ_x_val * x1_R1 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[1] += DbJ_x_val * x2_R2 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[2] += DbJ_x_val * x3_R3 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[3] += DbJ_y_val * x2_R2 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[4] += DbJ_y_val * x3_R3 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[5] += DbJ_z_val * x3_R3 * pSPARC->elecstPotential[ishift_DM];

                            temp1 = pSPARC->Vc[ishift_DM] - VJ_ref[ishift_2p];
                            temp2 = pSPARC->Vc[ishift_DM];
                            temp3 = pSPARC->psdChrgDens[ishift_DM] + pSPARC->psdChrgDens_ref[ishift_DM];
                            temp_x = DbJ_ref_x_val*temp1 + DbJ_x_val*temp2 + (DVJ_ref_x_val-DVJ_x_val)*temp3 - DVJ_ref_x_val*bJ_ref[ishift_p];
                            temp_y = DbJ_ref_y_val*temp1 + DbJ_y_val*temp2 + (DVJ_ref_y_val-DVJ_y_val)*temp3 - DVJ_ref_y_val*bJ_ref[ishift_p];
                            temp_z = DbJ_ref_z_val*temp1 + DbJ_z_val*temp2 + (DVJ_ref_z_val-DVJ_z_val)*temp3 - DVJ_ref_z_val*bJ_ref[ishift_p];
                            
                            stress_corr[0] += temp_x * x1_R1;
                            stress_corr[1] += temp_x * x2_R2;
                            stress_corr[2] += temp_x * x3_R3;
                            stress_corr[3] += temp_y * x2_R2;
                            stress_corr[4] += temp_y * x3_R3;
                            stress_corr[5] += temp_z * x3_R3;
                        }
                    }
                }
            } else {

                for(kp = FDn, kp2 = pSPARC->order, k_DM = dk; kp2 < nzp; kp++, kp2++, k_DM++) {
                    int kshift_DM = k_DM * DMnx * DMny;
                    int kshift_2p = kp2 * nx2p * ny2p;
                    int kshift_p = kp * nxp * nyp;
                    for(jp = FDn, jp2 = pSPARC->order, j_DM = dj; jp2 < nyp; jp++, jp2++, j_DM++) {
                        int jshift_DM = kshift_DM + j_DM * DMnx;
                        int jshift_2p = kshift_2p + jp2 * nx2p;
                        int jshift_p = kshift_p + jp * nxp;
                        //#pragma simd
                        for(ip = FDn, ip2 = pSPARC->order, i_DM = di; ip2 < nxp ; ip++, ip2++, i_DM++) {
                            int ishift_DM = jshift_DM + i_DM;
                            int ishift_2p = jshift_2p + ip2;
                            int ishift_p = jshift_p + ip;
                            DbJ_x_val = DbJ_y_val = DbJ_z_val = 0.0;
                            DbJ_ref_x_val = DbJ_ref_y_val = DbJ_ref_z_val = 0.0;
                            DVJ_x_val = DVJ_y_val = DVJ_z_val = 0.0;
                            DVJ_ref_x_val = DVJ_ref_y_val = DVJ_ref_z_val = 0.0;

                            for (p = 1; p <= FDn; p++) {
                                DbJ_x_val += (bJ[ishift_p+p] - bJ[ishift_p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DbJ_y_val += (bJ[ishift_p+pshifty[p]] - bJ[ishift_p-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DbJ_z_val += (bJ[ishift_p+pshiftz[p]] - bJ[ishift_p-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                                DbJ_ref_x_val += (bJ_ref[ishift_p+p] - bJ_ref[ishift_p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DbJ_ref_y_val += (bJ_ref[ishift_p+pshifty[p]] - bJ_ref[ishift_p-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DbJ_ref_z_val += (bJ_ref[ishift_p+pshiftz[p]] - bJ_ref[ishift_p-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                                DVJ_x_val += (VJ[ishift_2p+p] - VJ[ishift_2p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DVJ_y_val += (VJ[ishift_2p+pshifty_ex[p]] - VJ[ishift_2p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DVJ_z_val += (VJ[ishift_2p+pshiftz_ex[p]] - VJ[ishift_2p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                                DVJ_ref_x_val += (VJ_ref[ishift_2p+p] - VJ_ref[ishift_2p-p]) * pSPARC->D1_stencil_coeffs_x[p];
                                DVJ_ref_y_val += (VJ_ref[ishift_2p+pshifty_ex[p]] - VJ_ref[ishift_2p-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                                DVJ_ref_z_val += (VJ_ref[ishift_2p+pshiftz_ex[p]] - VJ_ref[ishift_2p-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                            }
                            
                            // find integrals in the stress expression
                            x1_R1 = (i_DM + pSPARC->DMVertices[0]) * pSPARC->delta_x - x0_i;
                            x2_R2 = (j_DM + pSPARC->DMVertices[2]) * pSPARC->delta_y - y0_i;
                            x3_R3 = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;

                            nonCart2Cart_coord(pSPARC, &x1_R1, &x2_R2, &x3_R3);
                            nonCart2Cart_grad(pSPARC, &DbJ_x_val, &DbJ_y_val, &DbJ_z_val);
                            nonCart2Cart_grad(pSPARC, &DbJ_ref_x_val, &DbJ_ref_y_val, &DbJ_ref_z_val);
                            nonCart2Cart_grad(pSPARC, &DVJ_x_val, &DVJ_y_val, &DVJ_z_val);
                            nonCart2Cart_grad(pSPARC, &DVJ_ref_x_val, &DVJ_ref_y_val, &DVJ_ref_z_val);

                            stress_el[0] += DbJ_x_val * x1_R1 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[1] += DbJ_x_val * x2_R2 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[2] += DbJ_x_val * x3_R3 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[3] += DbJ_y_val * x2_R2 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[4] += DbJ_y_val * x3_R3 * pSPARC->elecstPotential[ishift_DM];
                            stress_el[5] += DbJ_z_val * x3_R3 * pSPARC->elecstPotential[ishift_DM];

                            temp1 = pSPARC->Vc[ishift_DM] - VJ_ref[ishift_2p];
                            temp2 = pSPARC->Vc[ishift_DM];
                            temp3 = pSPARC->psdChrgDens[ishift_DM] + pSPARC->psdChrgDens_ref[ishift_DM];
                            temp_x = DbJ_ref_x_val*temp1 + DbJ_x_val*temp2 + (DVJ_ref_x_val-DVJ_x_val)*temp3 - DVJ_ref_x_val*bJ_ref[ishift_p];
                            temp_y = DbJ_ref_y_val*temp1 + DbJ_y_val*temp2 + (DVJ_ref_y_val-DVJ_y_val)*temp3 - DVJ_ref_y_val*bJ_ref[ishift_p];
                            temp_z = DbJ_ref_z_val*temp1 + DbJ_z_val*temp2 + (DVJ_ref_z_val-DVJ_z_val)*temp3 - DVJ_ref_z_val*bJ_ref[ishift_p];
                            
                            stress_corr[0] += temp_x * x1_R1;
                            stress_corr[1] += temp_x * x2_R2;
                            stress_corr[2] += temp_x * x3_R3;
                            stress_corr[3] += temp_y * x2_R2;
                            stress_corr[4] += temp_y * x3_R3;
                            stress_corr[5] += temp_z * x3_R3;
                        }
                    }
                }
            }  

            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
        }   
    }
    
    for(i = 0; i < 6; i++)
        pSPARC->stress_el[i] = (stress_el[i] + 0.5 * stress_corr[i]) * pSPARC->dV;

    t1 = MPI_Wtime();
    // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
    MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_el, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();
    
    pSPARC->stress_el[0] += pSPARC->Esc;
    pSPARC->stress_el[3] += pSPARC->Esc;
    pSPARC->stress_el[5] += pSPARC->Esc;

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

       for(int i = 0; i < 6; i++)
            pSPARC->stress_el[i] /= cell_measure; 

    }

#ifdef DEBUG
    if (!rank){ 
        printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
        printf("\nElectrostatics contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_el, NULL); 
    }
#endif
    
    //deallocate
    free(Lap_wt);
    free(Dphi_x);
    free(Dphi_y);
    free(Dphi_z);
    free(pshifty); free(pshiftz);
    free(pshifty_ex); free(pshiftz_ex);
    free(stress_el);
    free(stress_corr);

#undef INDEX_EX
#undef INDEX_Dx1
#undef INDEX_Dx2
#undef INDEX_Dx3
#undef dV1
#undef dV2
#undef dV3
#undef dV_ref1
#undef dV_ref2
#undef dV_ref3    

}

/**
 * @brief    Calculate nonlocal stress components.
 */
void Calculate_nonlocal_kinetic_stress(SPARC_OBJ *pSPARC) {
    if (pSPARC->sqAmbientFlag == 1 || pSPARC->sqHighTFlag == 1) {
        Calculate_nonlocal_kinetic_stress_SQ(pSPARC);   
    } else if (pSPARC->isGammaPoint) {
    #ifdef SPARCX_ACCEL
        if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && pSPARC->spin_typ <= 1 && pSPARC->SQ3Flag == 0 && (pSPARC->Nd_d_dmcomm == pSPARC->Nd || pSPARC->useACCELGT))
        {
        ACCEL_Calculate_nonlocal_kinetic_stress_linear(pSPARC);
        } else
    #endif
        {
            Calculate_nonlocal_kinetic_stress_linear(pSPARC);
        }
    } else {
    #ifdef SPARCX_ACCEL
        if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && pSPARC->spin_typ <= 1 && pSPARC->SQ3Flag == 0 && (pSPARC->Nd_d_dmcomm == pSPARC->Nd || pSPARC->useACCELGT))
        {
            ACCEL_Calculate_nonlocal_kinetic_stress_kpt(pSPARC);
        } else
    #endif
        {
            Calculate_nonlocal_kinetic_stress_kpt(pSPARC);
        }
    }
 }


/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_linear(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int i, ncol, DMnd, DMndsp;
    int dim, dim2, count, size_k, spinor, Nspinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    size_k = DMndsp * ncol;
    
    double *alpha, *beta, *dpsi_full;
    double *dpsi_xi, *dpsi_x1, *dpsi_x2, *dpsi_x3, *dpsi_xi_lv;
    double energy_nl = 0.0, stress_k[6], stress_nl[6];
    
    for (i = 0; i < 6; i++) stress_nl[i] = stress_k[i] = 0;

    dpsi_full = (double *)malloc( 3 * size_k * sizeof(double) );  // dpsi_x, dpsi_y, dpsi_z in cartesian coordinates
    assert(dpsi_full != NULL);
    
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * 7 * Nspinor, sizeof(double));
    assert(alpha != NULL);
    
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif

    // find gradient of psi
    if (pSPARC->cell_typ == 0){
        for (dim = 0; dim < 3; dim++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                // find dPsi in direction dim
                dpsi_xi = dpsi_full + dim * size_k;
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, 
                                    pSPARC->Xorb+spinor*DMnd, DMndsp, dpsi_xi+spinor*DMnd, DMndsp, dim, pSPARC->dmcomm);
            }
        }
    } else {
        dpsi_xi_lv = (double *)malloc( size_k * sizeof(double) );  // dpsi_x, dpsi_y, dpsi_z along lattice vecotrs
        assert(dpsi_xi_lv != NULL);
        dpsi_x1 = dpsi_full;
        dpsi_x2 = dpsi_full + size_k;
        dpsi_x3 = dpsi_full + 2 * size_k;
        for (dim = 0; dim < 3; dim++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                // find dPsi in direction dim along lattice vector directions
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, 
                                    pSPARC->Xorb+spinor*DMnd, DMndsp, dpsi_xi_lv+spinor*DMnd, DMndsp, dim, pSPARC->dmcomm);
            }
            // find dPsi in direction dim in cartesian coordinates
            for (i = 0; i < size_k; i++) {
                if (dim == 0) {
                    dpsi_x1[i] = pSPARC->gradT[0]*dpsi_xi_lv[i];
                    dpsi_x2[i] = pSPARC->gradT[1]*dpsi_xi_lv[i];
                    dpsi_x3[i] = pSPARC->gradT[2]*dpsi_xi_lv[i];
                } else {
                    dpsi_x1[i] += pSPARC->gradT[0+3*dim]*dpsi_xi_lv[i];
                    dpsi_x2[i] += pSPARC->gradT[1+3*dim]*dpsi_xi_lv[i];
                    dpsi_x3[i] += pSPARC->gradT[2+3*dim]*dpsi_xi_lv[i];
                }
            }
        }
        free(dpsi_xi_lv);
    }

    // find <chi_Jlm, psi> 
    beta = alpha;
    Compute_Integral_psi_Chi(pSPARC, beta, pSPARC->Xorb);

    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    count = 1;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = dpsi_full + dim * size_k;
        for (dim2 = dim; dim2 < 3; dim2++) {
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_Chi_StXmRjp_beta_Dpsi(pSPARC, dpsi_xi, beta, dim2);
            count ++;
        }
    }
    
    // Kinetic stress
    Compute_stress_tensor_kinetic(pSPARC, dpsi_full, stress_k);
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        // MPI_Allreduce will fail randomly in valgrind test for unknown reason
        MPI_Request req0, req3;
        MPI_Status  sta0, sta3;
        MPI_Iallreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * 7, 
                        MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm, &req0);
        MPI_Iallreduce(MPI_IN_PLACE, stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm, &req3);
        MPI_Wait(&req0, &sta0);
        MPI_Wait(&req3, &sta3);
    }

    /* calculate nonlocal stress */
    Compute_stress_tensor_nloc_by_integrals(pSPARC, stress_nl, alpha);
    
    /* calculate nonlocal energy */
    energy_nl = Compute_Nonlocal_Energy_by_integrals(pSPARC, alpha);
    free(alpha);
    
    for(i = 0; i < 6; i++)
        stress_nl[i] *= pSPARC->occfac * 2.0;
    
    energy_nl *= pSPARC->occfac/pSPARC->dV;   

    pSPARC->stress_nl[0] = stress_nl[0] - energy_nl;
    pSPARC->stress_nl[1] = stress_nl[1];
    pSPARC->stress_nl[2] = stress_nl[2];
    pSPARC->stress_nl[3] = stress_nl[3] - energy_nl;
    pSPARC->stress_nl[4] = stress_nl[4];
    pSPARC->stress_nl[5] = stress_nl[5] - energy_nl;
    for(i = 0; i < 6; i++)
        pSPARC->stress_k[i] = stress_k[i];
    
    // sum over all spin
    if (pSPARC->npspin > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);        
    }

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

        for(i = 0; i < 6; i++) {
            pSPARC->stress_nl[i] /= cell_measure;
            pSPARC->stress_k[i] /= cell_measure;
        }

    }

#ifdef DEBUG    
    if (!rank){
        printf("\nNon-local contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_nl, NULL);
        printf("\nKinetic contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_k, NULL);  
    } 
#endif
}



/**
 * @brief   Calculate <ChiSC_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 */
void Compute_Integral_Chi_StXmRjp_beta_Dpsi(SPARC_OBJ *pSPARC, double *dpsi_xi, double *beta, int dim2) 
{
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, spinorshift;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double *dpsi_xi_rc, *dpsi_ptr, *dpsi_xi_rc_ptr;
    double R1, R2, R3, x1_R1, x2_R2, x3_R3, StXmRjp;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
            R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
            R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
            dpsi_xi_rc = (double *)malloc( ndc * ncol * sizeof(double));
            assert(dpsi_xi_rc);
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMndsp + spinor * DMnd;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                        k_DM = indx / (DMnx * DMny);
                        j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                        i_DM = indx % DMnx;
                        x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                        x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                        x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                        StXmRjp = pSPARC->LatUVec[0+dim2] * x1_R1 + pSPARC->LatUVec[3+dim2] * x2_R2 + pSPARC->LatUVec[6+dim2] * x3_R3;
                        *(dpsi_xi_rc_ptr + i) = *(dpsi_ptr + indx) * StXmRjp;
                    }
                }
                spinorshift = pSPARC->IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                            dpsi_xi_rc, ndc, 1.0, beta+spinorshift+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
            }
            free(dpsi_xi_rc);
        }
    }
}

/**
 * @brief   Compute kinetic stress tensor
 */
void Compute_stress_tensor_kinetic(SPARC_OBJ *pSPARC, double *dpsi_full, double *stress_k) 
{
    int ncol, DMnd, Ns;
    int Nspinor, DMndsp, size_k;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    size_k = DMndsp * ncol;
    Ns = pSPARC->Nstates;
    
    double *dpsi_xi, *dpsi_xj, *dpsi_xi_ptr, *dpsi_xj_ptr;
    int count, dim, dim2, n, spinor, i;

    count = 0;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = dpsi_full + dim * size_k;
        for (dim2 = dim; dim2 < 3; dim2++) {
            dpsi_xj = dpsi_full + dim2 * size_k;
            double temp_k = 0;
            for(n = 0; n < ncol; n++){
                for (spinor = 0; spinor < Nspinor; spinor++) {
                    double dpsii_dpsij = 0;
                    dpsi_xi_ptr = dpsi_xi + n * DMndsp + spinor * DMnd; // dpsi_xi
                    dpsi_xj_ptr = dpsi_xj + n * DMndsp + spinor * DMnd; // dpsi_xj

                    for(i = 0; i < DMnd; i++){
                        dpsii_dpsij += *(dpsi_xi_ptr + i) * *(dpsi_xj_ptr + i);
                    }
                    double *occ = pSPARC->occ;
                    if (pSPARC->spin_typ == 1) occ += spinor * Ns;
                    double g_nk = occ[n + pSPARC->band_start_indx];
                    temp_k += dpsii_dpsij * g_nk;
                }
            }
            stress_k[count] -= pSPARC->occfac * temp_k;
            count ++;
        }
    }
}


/**
 * @brief   Compute nonlocal Energy with spin-orbit coupling
 */
double Compute_Nonlocal_Energy_by_integrals(SPARC_OBJ *pSPARC, double *alpha)
{
    int n, np, ldispl, ityp, iat, Ns;
    int count, l, m, lmax, spinor, Nspinor;
    double eJ, temp_e, temp2_e, g_nk, gamma_jl, energy_nl;
    Ns = pSPARC->Nstates;
    Nspinor = pSPARC->Nspinor_spincomm;

    energy_nl = 0.0;
    count = 0;
    for (spinor = 0; spinor < Nspinor; spinor++) {
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                eJ = 0.0; 
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    double *occ = pSPARC->occ;
                    if (pSPARC->spin_typ == 1) occ += spinor * Ns;
                    g_nk = occ[n];
                    temp2_e = 0.0; 

                    // scalar relativistic term
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == pSPARC->localPsd[ityp]) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            temp_e = 0.0; 
                            for (m = -l; m <= l; m++) {
                                temp_e += alpha[count] * alpha[count];
                                count++;
                            }
                            gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                            temp2_e += temp_e * gamma_jl;
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }   
                    eJ += temp2_e * g_nk;
                }
                energy_nl += eJ;
            }
        }
    }
    return energy_nl;
}

/**
 * @brief   Compute nonlocal stress tensor
 */
void Compute_stress_tensor_nloc_by_integrals(SPARC_OBJ *pSPARC, double *stress_nl, double *alpha)
{
    int i, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, l, m, lmax, spinor, Nspinor;
    int ppl, *IP_displ;
    double g_nk, SJ[6], temp2_s[6], gamma_Jl = 0;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;    
    Nspinor = pSPARC->Nspinor_spincomm;
    IP_displ = pSPARC->IP_displ;

    double *beta1_x1, *beta2_x1, *beta3_x1, *beta2_x2, *beta3_x2, *beta3_x3; 
    beta1_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor;
    beta2_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor * 2;
    beta3_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor * 3;
    beta2_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor * 4;
    beta3_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor * 5;
    beta3_x3 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor * 6;

    count = 0;
    for (spinor = 0; spinor < Nspinor; spinor++) {
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                for(i = 0; i < 6; i++) SJ[i] = 0.0;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    double *occ = pSPARC->occ;
                    if (pSPARC->spin_typ == 1) occ += spinor * Ns;
                    g_nk = occ[n];

                    for(i = 0; i < 6; i++) temp2_s[i] = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        ppl = pSPARC->psd[ityp].ppl[l];
                        // skip the local l
                        if (l == pSPARC->localPsd[ityp]) {
                            ldispl += ppl;
                            continue;
                        }
                        for (np = 0; np < ppl; np++) {
                            for (m = -l; m <= l; m++) {
                                gamma_Jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                temp2_s[0] += gamma_Jl * alpha[count] * beta1_x1[count];
                                temp2_s[1] += gamma_Jl * alpha[count] * beta2_x1[count];
                                temp2_s[2] += gamma_Jl * alpha[count] * beta3_x1[count];
                                temp2_s[3] += gamma_Jl * alpha[count] * beta2_x2[count];
                                temp2_s[4] += gamma_Jl * alpha[count] * beta3_x2[count];
                                temp2_s[5] += gamma_Jl * alpha[count] * beta3_x3[count];
                                count++;
                            }
                        }
                        ldispl += ppl;
                    }
                    for(i = 0; i < 6; i++)
                        SJ[i] += temp2_s[i] * g_nk;
                }
                for(i = 0; i < 6; i++)
                    stress_nl[i] -= SJ[i];
            }
        }
    }
}

/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_kpt(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int i, ncol, DMnd, DMndsp;
    int dim, dim2, count, kpt, Nk, size_k, spinor, Nspinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    size_k = DMndsp * ncol;
    
    double _Complex *alpha, *alpha_so1, *alpha_so2, *beta, *dpsi_full;
    alpha = alpha_so1 = alpha_so2 = NULL;
    double _Complex *dpsi_xi, *dpsi_x1, *dpsi_x2, *dpsi_x3, *dpsi_xi_lv;    
    double energy_nl = 0.0, stress_k[6], stress_nl[6];
    
    for (i = 0; i < 6; i++) stress_nl[i] = stress_k[i] = 0;

    dpsi_full = (double _Complex *)malloc( 3 * size_k * Nk * sizeof(double _Complex) );  // dpsi_x, dpsi_y, dpsi_z in cartesian coordinates
    assert(dpsi_full != NULL);
    
    alpha = (double _Complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * 7 * Nspinor, sizeof(double _Complex));
    assert(alpha != NULL);
    if (pSPARC->SOC_Flag == 1) {
        alpha_so1 = (double _Complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * 7 * Nspinor, sizeof(double _Complex));
        alpha_so2 = (double _Complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * 7 * Nspinor, sizeof(double _Complex));
        assert(alpha_so1 != NULL && alpha_so2 != NULL);
    }
    
    double k1, k2, k3, kpt_vec;
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif

    // find gradient of psi
    if (pSPARC->cell_typ == 0){
        for (dim = 0; dim < 3; dim++) {
            // count = 0;
            for(kpt = 0; kpt < Nk; kpt++) {
                k1 = pSPARC->k1_loc[kpt];
                k2 = pSPARC->k2_loc[kpt];
                k3 = pSPARC->k3_loc[kpt];
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                for (spinor = 0; spinor < Nspinor; spinor++) {
                    // find dPsi in direction dim
                    dpsi_xi = dpsi_full + dim * size_k * Nk;
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+kpt*size_k+spinor*DMnd, DMndsp, 
                                                dpsi_xi+kpt*size_k+spinor*DMnd, DMndsp, dim, &kpt_vec, pSPARC->dmcomm);
                }
            }
        }
    } else {
        dpsi_xi_lv = (double _Complex *)malloc( size_k * sizeof(double _Complex) );  // dpsi_x, dpsi_y, dpsi_z along lattice vecotrs
        assert(dpsi_xi_lv != NULL);
        dpsi_x1 = dpsi_full;
        dpsi_x2 = dpsi_full + size_k * Nk;
        dpsi_x3 = dpsi_full + 2 * size_k * Nk;
        for (dim = 0; dim < 3; dim++) {
            // count = 0;
            for(kpt = 0; kpt < Nk; kpt++) {
                k1 = pSPARC->k1_loc[kpt];
                k2 = pSPARC->k2_loc[kpt];
                k3 = pSPARC->k3_loc[kpt];
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                for (spinor = 0; spinor < Nspinor; spinor++) {
                    // find dPsi in direction dim along lattice vector directions
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, 
                                            pSPARC->Xorb_kpt+kpt*size_k+spinor*DMnd, DMndsp, 
                                            dpsi_xi_lv+spinor*DMnd, DMndsp, dim, &kpt_vec, pSPARC->dmcomm);
                }
                // find dPsi in direction dim in cartesian coordinates
                for (i = 0; i < size_k; i++) {
                    if (dim == 0) {
                        dpsi_x1[i + kpt*size_k] = pSPARC->gradT[0]*dpsi_xi_lv[i];
                        dpsi_x2[i + kpt*size_k] = pSPARC->gradT[1]*dpsi_xi_lv[i];
                        dpsi_x3[i + kpt*size_k] = pSPARC->gradT[2]*dpsi_xi_lv[i];
                    } else {
                        dpsi_x1[i + kpt*size_k] += pSPARC->gradT[0+3*dim]*dpsi_xi_lv[i];
                        dpsi_x2[i + kpt*size_k] += pSPARC->gradT[1+3*dim]*dpsi_xi_lv[i];
                        dpsi_x3[i + kpt*size_k] += pSPARC->gradT[2+3*dim]*dpsi_xi_lv[i];
                    }
                }
            }
        }
        free(dpsi_xi_lv);
    }

    // find <chi_Jlm, psi> 
    for(kpt = 0; kpt < Nk; kpt++){
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * kpt;
        Compute_Integral_psi_Chi_kpt(pSPARC, beta, pSPARC->Xorb_kpt+kpt*size_k, kpt, "SC");
        if (pSPARC->SOC_Flag == 0) continue;
        beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * kpt;
        Compute_Integral_psi_Chi_kpt(pSPARC, beta, pSPARC->Xorb_kpt+kpt*size_k, kpt, "SO1");
        beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * kpt;            
        Compute_Integral_psi_Chi_kpt(pSPARC, beta, pSPARC->Xorb_kpt+kpt*size_k, kpt, "SO2");
    }

    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    count = 1;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = dpsi_full + dim * size_k * Nk;
        for (dim2 = dim; dim2 < 3; dim2++) {
            for(kpt = 0; kpt < Nk; kpt++) {
                beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * (Nk * count + kpt);
                Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(pSPARC, dpsi_xi+kpt*size_k, beta, kpt, dim2, "SC");
                if (pSPARC->SOC_Flag == 0) continue;
                beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * count + kpt);
                Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(pSPARC, dpsi_xi+kpt*size_k, beta, kpt, dim2, "SO1");
                beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * count + kpt);
                Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(pSPARC, dpsi_xi+kpt*size_k, beta, kpt, dim2, "SO2");
            }
            count ++;
        }
    }
    
    // Kinetic stress
    Compute_stress_tensor_kinetic_kpt(pSPARC, dpsi_full, stress_k);
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        // MPI_Allreduce will fail randomly in valgrind test for unknown reason
        MPI_Request req0, req1, req2, req3;
        MPI_Status  sta0, sta1, sta2, sta3;
        MPI_Iallreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * Nspinor * 7, 
                        MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm, &req0);
        MPI_Iallreduce(MPI_IN_PLACE, stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm, &req3);
        MPI_Wait(&req0, &sta0);
        MPI_Wait(&req3, &sta3);
        if (pSPARC->SOC_Flag == 1) {
            MPI_Iallreduce(MPI_IN_PLACE, alpha_so1, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * Nspinor * 7, 
                            MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm, &req1);
            MPI_Iallreduce(MPI_IN_PLACE, alpha_so2, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * Nspinor * 7, 
                            MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm, &req2);
            MPI_Wait(&req1, &sta1);
            MPI_Wait(&req2, &sta2);
        }
    }

    /* calculate nonlocal stress */
    Compute_stress_tensor_nloc_by_integrals_kpt(pSPARC, stress_nl, alpha, "SC");
    if (pSPARC->SOC_Flag == 1) {
        Compute_stress_tensor_nloc_by_integrals_kpt(pSPARC, stress_nl, alpha_so1, "SO1");
        Compute_stress_tensor_nloc_by_integrals_kpt(pSPARC, stress_nl, alpha_so2, "SO2");
    }
    
    /* calculate nonlocal energy */
    energy_nl += Compute_Nonlocal_Energy_by_integrals_kpt(pSPARC, alpha,"SC");
    free(alpha);
    if (pSPARC->SOC_Flag == 1) {
        energy_nl += Compute_Nonlocal_Energy_by_integrals_kpt(pSPARC, alpha_so1,"SO1");
        energy_nl += Compute_Nonlocal_Energy_by_integrals_kpt(pSPARC, alpha_so2,"SO2");
        free(alpha_so1);
        free(alpha_so2);
    }
    
    for(i = 0; i < 6; i++)
        stress_nl[i] *= pSPARC->occfac * 2.0;
    
    energy_nl *= pSPARC->occfac/pSPARC->dV;   

    pSPARC->stress_nl[0] = stress_nl[0] - energy_nl;
    pSPARC->stress_nl[1] = stress_nl[1];
    pSPARC->stress_nl[2] = stress_nl[2];
    pSPARC->stress_nl[3] = stress_nl[3] - energy_nl;
    pSPARC->stress_nl[4] = stress_nl[4];
    pSPARC->stress_nl[5] = stress_nl[5] - energy_nl;
    for(i = 0; i < 6; i++)
        pSPARC->stress_k[i] = stress_k[i];
    
    // sum over all spin
    if (pSPARC->npspin > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }


    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        MPI_Allreduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);        
    }

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

        for(i = 0; i < 6; i++) {
            pSPARC->stress_nl[i] /= cell_measure;
            pSPARC->stress_k[i] /= cell_measure;
        }

    }

#ifdef DEBUG    
    if (!rank){
        printf("\nNon-local contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_nl, NULL);
        printf("\nKinetic contribution to stress");
        PrintStress(pSPARC, pSPARC->stress_k, NULL);  
    } 
#endif
}


/**
 * @brief   Calculate <ChiSC_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int kpt, int dim2, char *option) 
{
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndsp, spinorshift, nproj, ispinor, *IP_displ;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;

    double _Complex *dpsi_xi_rc, *dpsi_ptr, *dpsi_xi_rc_ptr;
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, R1, R2, R3, x1_R1, x2_R2, x3_R3, StXmRjp;
    double _Complex bloch_fac, b, **Chi = NULL;
    
    k1 = pSPARC->k1_loc[kpt];
    k2 = pSPARC->k2_loc[kpt];
    k3 = pSPARC->k3_loc[kpt];

    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        nproj = !strcmpi(option, "SC") ? pSPARC->nlocProj[ityp].nproj : pSPARC->nlocProj[ityp].nprojso_ext;
        if (!strcmpi(option, "SC")) 
            Chi = pSPARC->nlocProj[ityp].Chi_c;
        else if (!strcmpi(option, "SO1")) 
            Chi = pSPARC->nlocProj[ityp].Chisowt0;

        if (! nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
            R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
            R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
            theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
            bloch_fac = cos(theta) + sin(theta) * I;
            b = 1.0;
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
            dpsi_xi_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
            assert(dpsi_xi_rc);
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            for (spinor = 0; spinor < Nspinor; spinor++) {
                if (!strcmpi(option, "SO2")) 
                    Chi = (spinor == 0) ? pSPARC->nlocProj[ityp].Chisowtnl : pSPARC->nlocProj[ityp].Chisowtl; 
                ispinor = !strcmpi(option, "SO2") ? (1 - spinor) : spinor;

                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMndsp + ispinor * DMnd;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                        k_DM = indx / (DMnx * DMny);
                        j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                        i_DM = indx % DMnx;
                        x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                        x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                        x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                        StXmRjp = pSPARC->LatUVec[0+dim2] * x1_R1 + pSPARC->LatUVec[3+dim2] * x2_R2 + pSPARC->LatUVec[6+dim2] * x3_R3;
                        *(dpsi_xi_rc_ptr + i) = *(dpsi_ptr + indx) * StXmRjp;
                    }
                }

                spinorshift = IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Chi[iat], ndc, 
                            dpsi_xi_rc, ndc, &b, beta+spinorshift+IP_displ[atom_index]*ncol, nproj);                        
            }
            free(dpsi_xi_rc);
        }
    }
}

/**
 * @brief   Compute kinetic stress tensor
 */
void Compute_stress_tensor_kinetic_kpt(SPARC_OBJ *pSPARC, double _Complex * dpsi_full, double *stress_k) 
{
    int ncol, DMnd, Ns;
    int Nspinor, DMndsp, Nk, size_k;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    size_k = DMndsp * ncol;
    Nk = pSPARC->Nkpts_kptcomm;
    Ns = pSPARC->Nstates;
    
    double _Complex *dpsi_xi, *dpsi_xj, *dpsi_xi_ptr, *dpsi_xj_ptr;
    int count, dim, dim2, kpt, n, spinor, i;

    count = 0;
    for (dim = 0; dim < 3; dim++) {
        dpsi_xi = dpsi_full + dim * size_k * Nk;
        for (dim2 = dim; dim2 < 3; dim2++) {
            dpsi_xj = dpsi_full + dim2 * size_k * Nk;
            for(kpt = 0; kpt < Nk; kpt++) {
                double temp_k = 0;
                for(n = 0; n < ncol; n++){
                    for (spinor = 0; spinor < Nspinor; spinor++) {
                        double dpsii_dpsij = 0;
                        dpsi_xi_ptr = dpsi_xi + kpt * size_k + n * DMndsp + spinor * DMnd; // dpsi_xi
                        dpsi_xj_ptr = dpsi_xj + kpt * size_k + n * DMndsp + spinor * DMnd; // dpsi_xj

                        for(i = 0; i < DMnd; i++){
                            dpsii_dpsij += creal(*(dpsi_xi_ptr + i)) * creal(*(dpsi_xj_ptr + i)) + cimag(*(dpsi_xi_ptr + i)) * cimag(*(dpsi_xj_ptr + i));
                        }
                        double *occ = pSPARC->occ + kpt*Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                        double g_nk = occ[n + pSPARC->band_start_indx];
                        temp_k += dpsii_dpsij * g_nk;
                    }
                }
                stress_k[count] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k;
            }
            count ++;
        }
    }
}


/**
 * @brief   Compute nonlocal energy with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
double Compute_Nonlocal_Energy_by_integrals_kpt(SPARC_OBJ *pSPARC, double _Complex *alpha, char *option)
{
    int k, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, l, m, lmax, Nk, spinor, Nspinor, shift;
    int l_start, mexclude, ppl, *IP_displ;
    double energy_nl, eJ, temp_e, temp2_e, g_nk, kptwt, alpha_r, alpha_i, beta_r, beta_i, scaled_gamma_Jl = 0;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    Nspinor = pSPARC->Nspinor_spincomm;

    l_start = !strcmpi(option, "SC") ? 0 : 1;
    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;
    shift = IP_displ[pSPARC->n_atom] * ncol;

    count = 0;
    energy_nl = 0.0;
    for (k = 0; k < Nk; k++) {
        for (spinor = 0; spinor < Nspinor; spinor++) {
            double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    eJ = 0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        double *occ = pSPARC->occ + k*Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                        g_nk = occ[n];

                        temp2_e = 0.0; 
                        ldispl = 0;
                        for (l = l_start; l <= lmax; l++) {
                            mexclude = !strcmpi(option, "SC") ? (l+1) : (!strcmpi(option, "SO1") ? 0 : l);
                            ppl = !strcmpi(option, "SC") ? pSPARC->psd[ityp].ppl[l] : pSPARC->psd[ityp].ppl_soc[l-1];

                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += ppl;
                                continue;
                            }
                            for (np = 0; np < ppl; np++) {
                                for (m = -l; m <= l; m++) {
                                    if (m == mexclude) continue;
                                    if (!strcmpi(option, "SC")) scaled_gamma_Jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                    else if (!strcmpi(option, "SO1")) scaled_gamma_Jl = spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                                    else if (!strcmpi(option, "SO2")) scaled_gamma_Jl = 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np];

                                    if (!strcmpi(option, "SO2")) {
                                        alpha_r = creal(alpha[count]);       alpha_i = cimag(alpha[count]);
                                        beta_r  = creal(alpha[count+shift]); beta_i  = cimag(alpha[count+shift]);
                                        temp_e =  2*(alpha_r * beta_r + alpha_i * beta_i);
                                    } else {
                                        alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                        temp_e = alpha_r*alpha_r + alpha_i*alpha_i;
                                    }
                                    temp2_e += temp_e * scaled_gamma_Jl;
                                    count++;
                                }
                            }
                            ldispl += ppl;
                        }
                        eJ += temp2_e * g_nk;
                    }
                    
                    kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                    energy_nl += kptwt * eJ;
                }
            }
            if (!strcmpi(option, "SO2")) {
                count += shift;
                break;
            }
        }
    } 
    return energy_nl;    
}


/**
 * @brief   Compute nonlocal stress tensor with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_stress_tensor_nloc_by_integrals_kpt(SPARC_OBJ *pSPARC, double *stress_nl, double _Complex *alpha, char *option)
{
    int i, k, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, l, m, lmax, Nk, spinor, Nspinor;
    int l_start, mexclude, ppl, *IP_displ;
    double g_nk, kptwt, alpha_r, alpha_i, SJ[6], temp2_s[6], scaled_gamma_Jl = 0;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    Nspinor = pSPARC->Nspinor_spincomm;

    l_start = !strcmpi(option, "SC") ? 0 : 1;
    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    double _Complex *beta1_x1, *beta2_x1, *beta3_x1, *beta2_x2, *beta3_x2, *beta3_x3; 
    beta1_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk;
    beta2_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk * 2;
    beta3_x1 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk * 3;
    beta2_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk * 4;
    beta3_x2 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk * 5;
    beta3_x3 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk * 6;

    count = 0;
    for (k = 0; k < Nk; k++) {
        for (spinor = 0; spinor < Nspinor; spinor++) {
            double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    for(i = 0; i < 6; i++) SJ[i] = 0.0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        double *occ = pSPARC->occ + k*Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                        g_nk = occ[n];

                        for(i = 0; i < 6; i++) temp2_s[i] = 0.0;
                        ldispl = 0;
                        for (l = l_start; l <= lmax; l++) {
                            mexclude = !strcmpi(option, "SC") ? (l+1) : (!strcmpi(option, "SO1") ? 0 : l);
                            ppl = !strcmpi(option, "SC") ? pSPARC->psd[ityp].ppl[l] : pSPARC->psd[ityp].ppl_soc[l-1];

                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += ppl;
                                continue;
                            }
                            for (np = 0; np < ppl; np++) {
                                for (m = -l; m <= l; m++) {
                                    if (m == mexclude) continue;
                                    if (!strcmpi(option, "SC")) scaled_gamma_Jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                    else if (!strcmpi(option, "SO1")) scaled_gamma_Jl = spinorfac*0.5*m*pSPARC->psd[ityp].Gamma_soc[ldispl+np];
                                    else if (!strcmpi(option, "SO2")) scaled_gamma_Jl = 0.5*sqrt(l*(l+1)-m*(m+1))*pSPARC->psd[ityp].Gamma_soc[ldispl+np];

                                    alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                    temp2_s[0] += scaled_gamma_Jl * (alpha_r * creal(beta1_x1[count]) - alpha_i * cimag(beta1_x1[count]));
                                    temp2_s[1] += scaled_gamma_Jl * (alpha_r * creal(beta2_x1[count]) - alpha_i * cimag(beta2_x1[count]));
                                    temp2_s[2] += scaled_gamma_Jl * (alpha_r * creal(beta3_x1[count]) - alpha_i * cimag(beta3_x1[count]));
                                    temp2_s[3] += scaled_gamma_Jl * (alpha_r * creal(beta2_x2[count]) - alpha_i * cimag(beta2_x2[count]));
                                    temp2_s[4] += scaled_gamma_Jl * (alpha_r * creal(beta3_x2[count]) - alpha_i * cimag(beta3_x2[count]));
                                    temp2_s[5] += scaled_gamma_Jl * (alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]));
                                    count++;
                                }
                            }
                            ldispl += ppl;
                        }
                        for(i = 0; i < 6; i++)
                            SJ[i] += temp2_s[i] * g_nk;
                    }
                    
                    kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                    for(i = 0; i < 6; i++)
                        stress_nl[i] -= kptwt * SJ[i];
                }
            }
        }
    } 
}



/*
@ brief: function to print stress tensor
*/
void PrintStress (SPARC_OBJ *pSPARC, double *stress, FILE *fp) {

    if (fp == NULL){
        if (pSPARC->BC == 2){
            printf(" (GPa): \n %.15f %.15f %.15f \n %.15f %.15f %.15f \n %.15f %.15f %.15f\n", stress[0]*CONST_HA_BOHR3_GPA, stress[1]*CONST_HA_BOHR3_GPA, stress[2]*CONST_HA_BOHR3_GPA, stress[1]*CONST_HA_BOHR3_GPA, stress[3]*CONST_HA_BOHR3_GPA, stress[4]*CONST_HA_BOHR3_GPA, stress[2]*CONST_HA_BOHR3_GPA, stress[4]*CONST_HA_BOHR3_GPA, stress[5]*CONST_HA_BOHR3_GPA);
        } else if (pSPARC->BC == 3){
            if (pSPARC->BCx == 0 && pSPARC->BCy == 0){
                printf(" (Ha/Bohr**2): \n %.15f %.15f \n %.15f %.15f \n", stress[0], stress[1], stress[1], stress[3]);
                printf("Stress equiv. to all periodic (GPa): \n %.15f %.15f \n %.15f %.15f \n", stress[0]*CONST_HA_BOHR3_GPA/pSPARC->range_z, stress[1]*CONST_HA_BOHR3_GPA/pSPARC->range_z, stress[1]*CONST_HA_BOHR3_GPA/pSPARC->range_z, stress[3]*CONST_HA_BOHR3_GPA/pSPARC->range_z);
            } else if (pSPARC->BCx == 0 && pSPARC->BCz == 0){
                printf(" (Ha/Bohr**2): \n %.15f %.15f \n %.15f %.15f \n", stress[0], stress[2], stress[2], stress[5]);
                printf("Stress equiv. to all periodic (GPa): \n %.15f %.15f \n %.15f %.15f \n", stress[0]*CONST_HA_BOHR3_GPA/pSPARC->range_y, stress[2]*CONST_HA_BOHR3_GPA/pSPARC->range_y, stress[2]*CONST_HA_BOHR3_GPA/pSPARC->range_y, stress[5]*CONST_HA_BOHR3_GPA/pSPARC->range_y);
            } else if (pSPARC->BCy == 0 && pSPARC->BCz == 0){
                printf(" (Ha/Bohr**2): \n %.15f %.15f \n %.15f %.15f \n", stress[3], stress[4], stress[4], stress[5]);
                printf("Stress equiv. to all periodic (GPa): \n %.15f %.15f \n %.15f %.15f \n", stress[3]*CONST_HA_BOHR3_GPA/pSPARC->range_x, stress[4]*CONST_HA_BOHR3_GPA/pSPARC->range_x, stress[4]*CONST_HA_BOHR3_GPA/pSPARC->range_x, stress[5]*CONST_HA_BOHR3_GPA/pSPARC->range_x);
            }
        } else if (pSPARC->BC == 4){
            if (pSPARC->BCx == 0){
                printf(" (Ha/Bohr): \n %.15f \n", stress[0]);
                printf("Stress equiv. to all periodic (GPa): \n %.15f \n", stress[0]*CONST_HA_BOHR3_GPA/(pSPARC->range_y * pSPARC->range_z));
            } else if (pSPARC->BCy == 0){
                printf(" (Ha/Bohr): \n %.15f \n", stress[3]);
                printf("Stress equiv. to all periodic (GPa): \n %.15f \n", stress[3]*CONST_HA_BOHR3_GPA/(pSPARC->range_x * pSPARC->range_z));
            } else if (pSPARC->BCz == 0){
                printf(" (Ha/Bohr): \n %.15f \n", stress[5]);
                printf("Stress equiv. to all periodic (GPa): \n %.15f \n", stress[5]*CONST_HA_BOHR3_GPA/(pSPARC->range_x * pSPARC->range_y));
            }
        }  else if (pSPARC->BC >= 5 && pSPARC->BC <= 7){
            printf(" (Ha/Bohr): \n %.15f \n", stress[5]);
            printf("Stress equiv. to all periodic (GPa): \n %.15f \n", stress[5]*CONST_HA_BOHR3_GPA/(M_PI * ( pow(pSPARC->xout,2.0) - pow(pSPARC->xin,2.0) ) ) );
        }
    } else {
        if (pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
            if (pSPARC->BC == 2){
                fprintf(fp, " (GPa): \n%18.10E %18.10E %18.10E \n%18.10E %18.10E %18.10E \n%18.10E %18.10E %18.10E\n", stress[0]*CONST_HA_BOHR3_GPA, stress[1]*CONST_HA_BOHR3_GPA, stress[2]*CONST_HA_BOHR3_GPA, stress[1]*CONST_HA_BOHR3_GPA, stress[3]*CONST_HA_BOHR3_GPA, stress[4]*CONST_HA_BOHR3_GPA, stress[2]*CONST_HA_BOHR3_GPA, stress[4]*CONST_HA_BOHR3_GPA, stress[5]*CONST_HA_BOHR3_GPA);
            } else if (pSPARC->BC == 3){
                if (pSPARC->BCx == 0 && pSPARC->BCy == 0){
                    fprintf(fp, " (Ha/Bohr**2): \n%18.10E %18.10E \n%18.10E %18.10E \n", stress[0], stress[1], stress[1], stress[3]);
                    fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E %18.10E \n%18.10E %18.10E \n", stress[0]*CONST_HA_BOHR3_GPA/pSPARC->range_z, stress[1]*CONST_HA_BOHR3_GPA/pSPARC->range_z, stress[1]*CONST_HA_BOHR3_GPA/pSPARC->range_z, stress[3]*CONST_HA_BOHR3_GPA/pSPARC->range_z);
                } else if (pSPARC->BCx == 0 && pSPARC->BCz == 0){
                    fprintf(fp, " (Ha/Bohr**2): \n%18.10E %18.10E \n%18.10E %18.10E \n", stress[0], stress[2], stress[2], stress[5]);
                    fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E %18.10E \n%18.10E %18.10E \n", stress[0]*CONST_HA_BOHR3_GPA/pSPARC->range_y, stress[2]*CONST_HA_BOHR3_GPA/pSPARC->range_y, stress[2]*CONST_HA_BOHR3_GPA/pSPARC->range_y, stress[5]*CONST_HA_BOHR3_GPA/pSPARC->range_y);
                } else if (pSPARC->BCy == 0 && pSPARC->BCz == 0){
                    fprintf(fp, " (Ha/Bohr**2): \n%18.10E %18.10E \n%18.10E %18.10E \n", stress[3], stress[4], stress[4], stress[5]);
                    fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E %18.10E \n%18.10E %18.10E \n", stress[3]*CONST_HA_BOHR3_GPA/pSPARC->range_x, stress[4]*CONST_HA_BOHR3_GPA/pSPARC->range_x, stress[4]*CONST_HA_BOHR3_GPA/pSPARC->range_x, stress[5]*CONST_HA_BOHR3_GPA/pSPARC->range_x);
                }
            } else if (pSPARC->BC == 4){
                if (pSPARC->BCx == 0){
                    fprintf(fp, " (Ha/Bohr): \n%18.10E \n", stress[0]);
                    fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E \n", stress[0]*CONST_HA_BOHR3_GPA/(pSPARC->range_y * pSPARC->range_z));
                } else if (pSPARC->BCy == 0){
                    fprintf(fp, " (Ha/Bohr): \n%18.10E \n", stress[3]);
                    fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E \n", stress[3]*CONST_HA_BOHR3_GPA/(pSPARC->range_x * pSPARC->range_z));
                } else if (pSPARC->BCz == 0){
                    fprintf(fp, " (Ha/Bohr): \n%18.10E \n", stress[5]);
                    fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E \n", stress[5]*CONST_HA_BOHR3_GPA/(pSPARC->range_x * pSPARC->range_y));
                }
            } else if (pSPARC->BC >= 5 && pSPARC->BC <= 7){
                fprintf(fp, " (Ha/Bohr): \n%18.10E \n", stress[5]);
                fprintf(fp, "Stress equiv. to all periodic (GPa): \n%18.10E \n", stress[5]*CONST_HA_BOHR3_GPA/(M_PI * ( pow(pSPARC->xout,2.0) - pow(pSPARC->xin,2.0) ) ) );
            }
        } else {
            if (pSPARC->BC == 2){
                fprintf(fp, "%18.10E %18.10E %18.10E \n%18.10E %18.10E %18.10E \n%18.10E %18.10E %18.10E\n", stress[0]*CONST_HA_BOHR3_GPA, stress[1]*CONST_HA_BOHR3_GPA, stress[2]*CONST_HA_BOHR3_GPA, stress[1]*CONST_HA_BOHR3_GPA, stress[3]*CONST_HA_BOHR3_GPA, stress[4]*CONST_HA_BOHR3_GPA, stress[2]*CONST_HA_BOHR3_GPA, stress[4]*CONST_HA_BOHR3_GPA, stress[5]*CONST_HA_BOHR3_GPA);
            } else if (pSPARC->BC == 3){
                if (pSPARC->BCx == 0 && pSPARC->BCy == 0){
                    fprintf(fp, "%18.10E %18.10E \n%18.10E %18.10E \n", stress[0], stress[1], stress[1], stress[3]);
                } else if (pSPARC->BCx == 0 && pSPARC->BCz == 0){
                    fprintf(fp, "%18.10E %18.10E \n%18.10E %18.10E \n", stress[0], stress[2], stress[2], stress[5]);
                } else if (pSPARC->BCy == 0 && pSPARC->BCz == 0){
                    fprintf(fp, "%18.10E %18.10E \n%18.10E %18.10E \n", stress[3], stress[4], stress[4], stress[5]);
                }
            } else if (pSPARC->BC == 4){
                if (pSPARC->BCx == 0){
                    fprintf(fp, "%18.10E \n", stress[0]);
                } else if (pSPARC->BCy == 0){
                    fprintf(fp, "%18.10E \n", stress[3]);
                } else if (pSPARC->BCz == 0){
                    fprintf(fp, "%18.10E \n", stress[5]);
                }
            } else if (pSPARC->BC >= 5 && pSPARC->BC <= 7){
                fprintf(fp, "%18.10E \n", stress[5]);
            }
        }
        
    }

}


/**
 * @brief Convert stress component from Ha/Bohr**3 (or Ha/Bohr**2 for slabs, Ha/Bohr for wires) to
 *        GPa. Note that for the Dirichlet directions, we directly scale by the domain size, which
 *        is not physical in general. 
 * 
 * @param Stress Stress component in a.u. (Ha/Bohr**3, or Ha/Bohr**2, or Ha/Bohr,
 *               which will be returned in Unit)

 * @param cellsizes Cell sizes in all three direction.
 * @param BCs Boundary condition, 0 - periodic, 1 - dirichlet
 * @param origUnit (OUTPUT) Unit for the original stress.
 * @return double Stress component in GPa.
 */
double convertStressToGPa(double Stress, double cellsizes[3], int BCs[3], char origUnit[16]) {
    double scale = 1.0; // scale the stress by the dimensions of the dirichlet directions
    int nperiods = 3;
    if (BCs[0] == 1) { scale *= cellsizes[0]; nperiods--; }
    if (BCs[1] == 1) { scale *= cellsizes[1]; nperiods--; }
    if (BCs[2] == 1) { scale *= cellsizes[2]; nperiods--; }

    // scale and then convert to GPa
    double StressGPa = Stress / scale * CONST_HA_BOHR3_GPA; 

    // find the original unit
    if (nperiods == 1)
        snprintf(origUnit, 16, "Ha/Bohr");
    else
        snprintf(origUnit, 16, "Ha/Bohr**%d", nperiods);

    return StressGPa;
}



