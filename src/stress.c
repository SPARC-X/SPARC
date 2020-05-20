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

#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "stress.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecOrth.h"
#include "lapVecNonOrth.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"

#define TEMP_TOL 1e-12


/*
@ brief: function to calculate ionic stress/pressure and add to the electronic stress/pressure
*/


void Calculate_ionic_stress(SPARC_OBJ *pSPARC){
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
    if(pSPARC->BCy == 0)
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
	int i, rank;
    double t1, t2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // find exchange-correlation components of stress
    t1 = MPI_Wtime();
    Calculate_XC_stress(pSPARC);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating exchange-correlation stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif
    
    // find local stress components
    t1 = MPI_Wtime();
    Calculate_local_stress(pSPARC);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating local stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif
    
    // find nonlocal + kinetic stress components
    t1 = MPI_Wtime();
    if (pSPARC->isGammaPoint)
        Calculate_nonlocal_kinetic_stress(pSPARC);
    else
        Calculate_nonlocal_kinetic_stress_kpt(pSPARC);    
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating nonlocal+kinetic stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif

    // find stress
 	if(!rank){ 	    
        for(i = 0; i < 6; i++){
            pSPARC->stress[i] = (pSPARC->stress_k[i] + pSPARC->stress_xc[i] + pSPARC->stress_nl[i] + pSPARC->stress_el[i]);
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
*/
void Calculate_XC_stress(SPARC_OBJ *pSPARC) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (!rank) printf("Start calculating exchange-correlation components of stress ...\n");
#endif

    if(strcmpi(pSPARC->XC,"LDA_PW") == 0 || strcmpi(pSPARC->XC,"LDA_PZ") == 0){
        pSPARC->stress_xc[0] = pSPARC->stress_xc[3] = pSPARC->stress_xc[5] = pSPARC->Exc - pSPARC->Exc_corr;
        pSPARC->stress_xc[1] = pSPARC->stress_xc[2] = pSPARC->stress_xc[4] = 0.0;
    } else if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0){
        pSPARC->stress_xc[0] = pSPARC->stress_xc[3] = pSPARC->stress_xc[5] = pSPARC->Exc - pSPARC->Exc_corr;
        pSPARC->stress_xc[1] = pSPARC->stress_xc[2] = pSPARC->stress_xc[4] = 0.0;
        int DMnd, i;
        DMnd = (2*pSPARC->Nspin - 1) * pSPARC->Nd_d;
        double *Drho_x, *Drho_y, *Drho_z;
        double *stress_xc;
        stress_xc = (double *) calloc(6, sizeof(double));
        
        Drho_x = (double *)malloc( DMnd * sizeof(double));
        Drho_y = (double *)malloc( DMnd * sizeof(double));
        Drho_z = (double *)malloc( DMnd * sizeof(double));
    
        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, pSPARC->electronDens, Drho_x, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, pSPARC->electronDens, Drho_y, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, pSPARC->electronDens, Drho_z, 2, pSPARC->dmcomm_phi);
        
        if(pSPARC->cell_typ == 0){
            for(i = 0; i < DMnd; i++){
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
            for(i = 0; i < DMnd; i++){
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
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_z;

       for(int i = 0; i < 6; i++)
            pSPARC->stress_xc[i] /= cell_measure; 

    }


#ifdef DEBUG    
    if (!rank) {
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
    
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_z, 2, pSPARC->dmcomm_phi);
    
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
        if(pSPARC->BCy == 0)
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
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, j, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, i_DM, j_DM, k_DM, dim, atom_index, count, l, m, lmax, spn_i, nspin, size_s;
    nspin = pSPARC->Nspin_spincomm; // number of spin in my spin communicator
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates; // total number of bands
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    size_s = ncol * DMnd;

    double *alpha, *beta, *beta1, *beta2, *beta3, *psi_ptr, *dpsi_ptr, *psi_rc, *dpsi_x1_rc, *dpsi_x2_rc, *dpsi_x3_rc, *psi_rc_ptr, *dpsi_x1_rc_ptr, *dpsi_x2_rc_ptr, *dpsi_x3_rc_ptr, *dpsi_full, R1, R2, R3, x1_R1, x2_R2, x3_R3;
    double *SJ, eJ, temp_e, *temp_s, temp2_e, *temp2_s, g_nk, *beta1_x1, *beta1_x2, *beta1_x3, *beta2_x1, *beta2_x2, *beta2_x3, *beta3_x1, *beta3_x2, *beta3_x3, gamma_jl;
    double *dpsi_ptr2, dpsi1_dpsi1, dpsi1_dpsi2, dpsi1_dpsi3, dpsi2_dpsi2, dpsi2_dpsi3, dpsi3_dpsi3;
    
    //double ts, te;

    double energy_nl = 0.0, *stress_k, *stress_nl;
    stress_k = (double*) calloc(6, sizeof(double));
    stress_nl = (double*) calloc(9, sizeof(double));
    SJ = (double*) malloc(9 * sizeof(double));
    temp_s = (double*) malloc(9 * sizeof(double));
    temp2_s = (double*) malloc(9 * sizeof(double));
    
    int len_psi = DMnd * ncol * nspin;
    dpsi_full = (double *)malloc( len_psi * sizeof(double) );
    if (dpsi_full == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 10, sizeof(double));
    
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                psi_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                
                /* first find inner product <Psi_n, Chi_Jlm> */
                for (n = 0; n < ncol; n++) {
                    psi_ptr = pSPARC->Xorb + spn_i * size_s + n * DMnd;
                    psi_rc_ptr = psi_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(psi_rc_ptr + i) = *(psi_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, pSPARC->dV, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                            psi_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiplied dV to get inner-product
                free(psi_rc);    
            }
        }
        count++;
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
            // find dPsi in direction dim
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, pSPARC->Yorb+spn_i*size_s, dim, pSPARC->dmcomm);
            beta1 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * (3*dim+1) + count);
            beta2 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * (3*dim+2) + count);
            beta3 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * (3*dim+3) + count);
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
                    R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                    dpsi_x1_rc = (double *)malloc( ndc * ncol * sizeof(double));
                    dpsi_x2_rc = (double *)malloc( ndc * ncol * sizeof(double));
                    dpsi_x3_rc = (double *)malloc( ndc * ncol * sizeof(double));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++) {
                        dpsi_ptr = pSPARC->Yorb + spn_i * size_s + n * DMnd;
                        dpsi_x1_rc_ptr = dpsi_x1_rc + n * ndc;
                        dpsi_x2_rc_ptr = dpsi_x2_rc + n * ndc;
                        dpsi_x3_rc_ptr = dpsi_x3_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                            k_DM = indx / (DMnx * DMny);
                            j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                            i_DM = indx % DMnx;
                            x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                            x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                            x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                            
                            *(dpsi_x1_rc_ptr + i) = *(dpsi_ptr + indx) * x1_R1;
                            *(dpsi_x2_rc_ptr + i) = *(dpsi_ptr + indx) * x2_R2;
                            *(dpsi_x3_rc_ptr + i) = *(dpsi_ptr + indx) * x3_R3;
                        }
                    }
                
                    /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                     *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                     *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                     *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                     */      
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                                dpsi_x1_rc, ndc, 1.0, beta1+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                                dpsi_x2_rc, ndc, 1.0, beta2+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                                dpsi_x3_rc, ndc, 1.0, beta3+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);                        
                    free(dpsi_x1_rc); free(dpsi_x2_rc); free(dpsi_x3_rc);
                }
            }
            
            // Kinetic stress
            if(dim == 0){
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, dpsi_full+spn_i*size_s, 1, pSPARC->dmcomm);
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_full + spn_i * size_s + n * DMnd; // dpsi_2
                    dpsi1_dpsi1 = dpsi1_dpsi2 = dpsi2_dpsi2 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi1 += *(dpsi_ptr + i) * *(dpsi_ptr + i);
                        dpsi1_dpsi2 += *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                        dpsi2_dpsi2 += *(dpsi_ptr2 + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_k[0] += dpsi1_dpsi1 * g_nk;
                    stress_k[1] += dpsi1_dpsi2 * g_nk;
                    stress_k[3] += dpsi2_dpsi2 * g_nk;
                }
                
                stress_k[0] *= -(2.0/pSPARC->Nspin);
                stress_k[1] *= -(2.0/pSPARC->Nspin);
                stress_k[3] *= -(2.0/pSPARC->Nspin);

                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, dpsi_full+spn_i*size_s, 2, pSPARC->dmcomm);
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + spn_i * size_s + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_full + spn_i * size_s + n * DMnd; // dpsi_3
                    dpsi1_dpsi3 = dpsi3_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi3 += *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                        dpsi3_dpsi3 += *(dpsi_ptr2 + i) * *(dpsi_ptr2 + i);
                    }
                    g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                    stress_k[2] += dpsi1_dpsi3 * g_nk;
                    stress_k[5] += dpsi3_dpsi3 * g_nk;
                }
                stress_k[2] *= -(2.0/pSPARC->Nspin);
                stress_k[5] *= -(2.0/pSPARC->Nspin);
            } else if(dim == 1){
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + spn_i * size_s + n * DMnd; // dpsi_2
                    dpsi_ptr2 = dpsi_full + spn_i * size_s + n * DMnd; // dpsi_3
                    dpsi2_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi2_dpsi3 += *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    stress_k[4] += dpsi2_dpsi3 * pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
                }
                stress_k[4] *= -(2.0/pSPARC->Nspin);
            }
            count++;
        }    
    }
    
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 10, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal stress */
    // go over all atoms and find nonlocal stress
    beta1_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin;
    beta1_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 2;
    beta1_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 3;
    beta2_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 4;
    beta2_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 5;
    beta2_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 6;
    beta3_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 7;
    beta3_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 8;
    beta3_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 9;
    count = 0;
    
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                eJ = 0.0; for(i = 0; i < 9; i++) SJ[i] = 0.0;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    g_nk = pSPARC->occ[spn_i*Ns + n]; // TODO: for k-points calculation, use occ[n+k*Ns]
                    temp2_e = 0.0; for(i = 0; i < 9; i++) temp2_s[i] = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == pSPARC->localPsd[ityp]) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            temp_e = 0.0; for(i = 0; i < 9; i++) temp_s[i] = 0.0;
                            for (m = -l; m <= l; m++) {
                                temp_e += alpha[count] * alpha[count];
                                temp_s[0] += alpha[count] * beta1_x1[count]; temp_s[1] += alpha[count] * beta1_x2[count]; temp_s[2] += alpha[count] * beta1_x3[count];
                                temp_s[3] += alpha[count] * beta2_x1[count]; temp_s[4] += alpha[count] * beta2_x2[count]; temp_s[5] += alpha[count] * beta2_x3[count];
                                temp_s[6] += alpha[count] * beta3_x1[count]; temp_s[7] += alpha[count] * beta3_x2[count]; temp_s[8] += alpha[count] * beta3_x3[count];
                                count++;
                            }
                            gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                            temp2_e += temp_e * gamma_jl;
                            for(i = 0; i < 9; i++)
                                temp2_s[i] += temp_s[i] * gamma_jl;
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }
                    eJ += temp2_e * g_nk;
                    for(i = 0; i < 9; i++)
                        SJ[i] += temp2_s[i] * g_nk;
                }
                
                energy_nl -= eJ;
                for(i = 0; i < 9; i++)
                    stress_nl[i] -= SJ[i];
            }
        }
    }    
    
    for(i = 0; i < 9; i++)
        stress_nl[i] *= (2.0/pSPARC->Nspin) * 2.0;
    
    energy_nl *= (2.0/pSPARC->Nspin)/pSPARC->dV;   
    
    
    if(pSPARC->cell_typ == 0){
        pSPARC->stress_nl[0] = stress_nl[0] + energy_nl;
        pSPARC->stress_nl[1] = stress_nl[1];
        pSPARC->stress_nl[2] = stress_nl[2];
        pSPARC->stress_nl[3] = stress_nl[4] + energy_nl;
        pSPARC->stress_nl[4] = stress_nl[5];
        pSPARC->stress_nl[5] = stress_nl[8] + energy_nl;
        for(i = 0; i < 6; i++)
            pSPARC->stress_k[i] = stress_k[i];
    } else {
        double *c_g, *c_c;
        c_g = (double*) malloc(9*sizeof(double));
        c_c = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
                c_c[3*i+j] = pSPARC->LatUVec[3*j+i];
            }
        }
        // sigma_ab = sum(i,j) (grad^T(a,i) * LatUVec^T(b,j) * grad_i_psi (x-R)_j)
        pSPARC->stress_nl[0] = c_g[0] * (c_c[0] * stress_nl[0] + c_c[1] * stress_nl[1] + c_c[2] * stress_nl[2]) +
                               c_g[1] * (c_c[0] * stress_nl[3] + c_c[1] * stress_nl[4] + c_c[2] * stress_nl[5]) + 
                               c_g[2] * (c_c[0] * stress_nl[6] + c_c[1] * stress_nl[7] + c_c[2] * stress_nl[8]) + energy_nl;                       
        pSPARC->stress_nl[1] = c_g[0] * (c_c[3] * stress_nl[0] + c_c[4] * stress_nl[1] + c_c[5] * stress_nl[2]) +
                               c_g[1] * (c_c[3] * stress_nl[3] + c_c[4] * stress_nl[4] + c_c[5] * stress_nl[5]) + 
                               c_g[2] * (c_c[3] * stress_nl[6] + c_c[4] * stress_nl[7] + c_c[5] * stress_nl[8]);
        pSPARC->stress_nl[2] = c_g[0] * (c_c[6] * stress_nl[0] + c_c[7] * stress_nl[1] + c_c[8] * stress_nl[2]) +
                               c_g[1] * (c_c[6] * stress_nl[3] + c_c[7] * stress_nl[4] + c_c[8] * stress_nl[5]) + 
                               c_g[2] * (c_c[6] * stress_nl[6] + c_c[7] * stress_nl[7] + c_c[8] * stress_nl[8]); 
        pSPARC->stress_nl[3] = c_g[3] * (c_c[3] * stress_nl[0] + c_c[4] * stress_nl[1] + c_c[5] * stress_nl[2]) +
                               c_g[4] * (c_c[3] * stress_nl[3] + c_c[4] * stress_nl[4] + c_c[5] * stress_nl[5]) + 
                               c_g[5] * (c_c[3] * stress_nl[6] + c_c[4] * stress_nl[7] + c_c[5] * stress_nl[8]) + energy_nl;
        pSPARC->stress_nl[4] = c_g[3] * (c_c[6] * stress_nl[0] + c_c[7] * stress_nl[1] + c_c[8] * stress_nl[2]) +
                               c_g[4] * (c_c[6] * stress_nl[3] + c_c[7] * stress_nl[4] + c_c[8] * stress_nl[5]) + 
                               c_g[5] * (c_c[6] * stress_nl[6] + c_c[7] * stress_nl[7] + c_c[8] * stress_nl[8]);
        pSPARC->stress_nl[5] = c_g[6] * (c_c[6] * stress_nl[0] + c_c[7] * stress_nl[1] + c_c[8] * stress_nl[2]) +
                               c_g[7] * (c_c[6] * stress_nl[3] + c_c[7] * stress_nl[4] + c_c[8] * stress_nl[5]) + 
                               c_g[8] * (c_c[6] * stress_nl[6] + c_c[7] * stress_nl[7] + c_c[8] * stress_nl[8]) + energy_nl;

        pSPARC->stress_k[0] = c_g[0] * (c_g[0] * stress_k[0] + c_g[1] * stress_k[1] + c_g[2] * stress_k[2]) +
                              c_g[1] * (c_g[0] * stress_k[1] + c_g[1] * stress_k[3] + c_g[2] * stress_k[4]) +
                              c_g[2] * (c_g[0] * stress_k[2] + c_g[1] * stress_k[4] + c_g[2] * stress_k[5]);
        pSPARC->stress_k[1] = c_g[0] * (c_g[3] * stress_k[0] + c_g[4] * stress_k[1] + c_g[5] * stress_k[2]) +
                              c_g[1] * (c_g[3] * stress_k[1] + c_g[4] * stress_k[3] + c_g[5] * stress_k[4]) +
                              c_g[2] * (c_g[3] * stress_k[2] + c_g[4] * stress_k[4] + c_g[5] * stress_k[5]);
        pSPARC->stress_k[2] = c_g[0] * (c_g[6] * stress_k[0] + c_g[7] * stress_k[1] + c_g[8] * stress_k[2]) +
                              c_g[1] * (c_g[6] * stress_k[1] + c_g[7] * stress_k[3] + c_g[8] * stress_k[4]) +
                              c_g[2] * (c_g[6] * stress_k[2] + c_g[7] * stress_k[4] + c_g[8] * stress_k[5]);
        pSPARC->stress_k[3] = c_g[3] * (c_g[3] * stress_k[0] + c_g[4] * stress_k[1] + c_g[5] * stress_k[2]) +
                              c_g[4] * (c_g[3] * stress_k[1] + c_g[4] * stress_k[3] + c_g[5] * stress_k[4]) +
                              c_g[5] * (c_g[3] * stress_k[2] + c_g[4] * stress_k[4] + c_g[5] * stress_k[5]);
        pSPARC->stress_k[4] = c_g[3] * (c_g[6] * stress_k[0] + c_g[7] * stress_k[1] + c_g[8] * stress_k[2]) +
                              c_g[4] * (c_g[6] * stress_k[1] + c_g[7] * stress_k[3] + c_g[8] * stress_k[4]) +
                              c_g[5] * (c_g[6] * stress_k[2] + c_g[7] * stress_k[4] + c_g[8] * stress_k[5]);
        pSPARC->stress_k[5] = c_g[6] * (c_g[6] * stress_k[0] + c_g[7] * stress_k[1] + c_g[8] * stress_k[2]) +
                              c_g[7] * (c_g[6] * stress_k[1] + c_g[7] * stress_k[3] + c_g[8] * stress_k[4]) +
                              c_g[8] * (c_g[6] * stress_k[2] + c_g[7] * stress_k[4] + c_g[8] * stress_k[5]);                                                                                   
  
        free(c_g); free(c_c);
    }


    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }
    
    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }


    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_z;

       for(int i = 0; i < 6; i++) {
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
    
    
    free(alpha);
    free(stress_k);
    free(stress_nl);
    free(SJ);
    free(temp_s);
    free(temp2_s);
}


/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_kpt(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, j, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, i_DM, j_DM, k_DM, dim, atom_index, count, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    
    double complex *alpha, *beta, *beta1, *beta2, *beta3, *psi_ptr, *dpsi_ptr, *dpsi_ptr2, *psi_rc, *dpsi_x1_rc, *dpsi_x2_rc, *dpsi_x3_rc, *psi_rc_ptr, *dpsi_x1_rc_ptr, *dpsi_x2_rc_ptr, *dpsi_x3_rc_ptr, *dpsi_full;
    double complex *beta1_x1, *beta1_x2, *beta1_x3, *beta2_x1, *beta2_x2, *beta2_x3, *beta3_x1, *beta3_x2, *beta3_x3;
    double *SJ, eJ, *temp_k, temp_e, *temp_s, temp2_e, *temp2_s, g_nk, gamma_jl, kptwt,  R1, R2, R3, x1_R1, x2_R2, x3_R3;
    double dpsi1_dpsi1, dpsi1_dpsi2, dpsi1_dpsi3, dpsi2_dpsi2, dpsi2_dpsi3, dpsi3_dpsi3;

    double energy_nl = 0.0, *stress_k, *stress_nl;
    temp_k = (double*) malloc(6 * sizeof(double));
    stress_k = (double*) calloc(6, sizeof(double));
    stress_nl = (double*) calloc(9, sizeof(double));
    SJ = (double*) malloc(9 * sizeof(double));
    temp_s = (double*) malloc(9 * sizeof(double));
    temp2_s = (double*) malloc(9 * sizeof(double));
    
    dpsi_full = (double complex *)malloc( size_s * nspin * sizeof(double complex) );
    if (dpsi_full == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 10, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta;
    double complex bloch_fac, a, b;
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++){
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (!pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                    R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
                    bloch_fac = cos(theta) - sin(theta) * I;
                    a = bloch_fac * pSPARC->dV;
                    b = 1.0;
                    
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                    psi_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    
                    /* first find inner product <Psi_n, Chi_Jlm> */
                    for (n = 0; n < ncol; n++) {
                        psi_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                        psi_rc_ptr = psi_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(psi_rc_ptr + i) = conj(*(psi_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                        }
                    }
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                psi_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiplied dV to get inner-product
                    free(psi_rc);    
                }
            }
            count++;
        }
    }       
    
    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
            for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                k1 = pSPARC->k1_loc[kpt];
                k2 = pSPARC->k2_loc[kpt];
                k3 = pSPARC->k3_loc[kpt];
                // find dPsi in direction dim
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt+spn_i*size_s+kpt*size_k, dim, kpt, pSPARC->dmcomm);
                beta1 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * (3*dim + 1) + count);
                beta2 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * (3*dim + 2) + count);
                beta3 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin * (3*dim + 3) + count);
                
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                    for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                        R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
                        R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                        R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                        theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
                        bloch_fac = cos(theta) + sin(theta) * I;
                        b = 1.0;
                        ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                        dpsi_x1_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                        dpsi_x2_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                        dpsi_x3_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                        atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                        for (n = 0; n < ncol; n++) {
                            dpsi_ptr = pSPARC->Yorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                            dpsi_x1_rc_ptr = dpsi_x1_rc + n * ndc;
                            dpsi_x2_rc_ptr = dpsi_x2_rc + n * ndc;
                            dpsi_x3_rc_ptr = dpsi_x3_rc + n * ndc;
                            for (i = 0; i < ndc; i++) {
                                indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                                k_DM = indx / (DMnx * DMny);
                                j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                                i_DM = indx % DMnx;
                                x1_R1 = (i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1;
                                x2_R2 = (j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2;
                                x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                                *(dpsi_x1_rc_ptr + i) = *(dpsi_ptr + indx) * x1_R1;
                                *(dpsi_x2_rc_ptr + i) = *(dpsi_ptr + indx) * x2_R2;
                                *(dpsi_x3_rc_ptr + i) = *(dpsi_ptr + indx) * x3_R3;
                            }
                        }
                    
                        /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                         *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                         *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                         *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).

                         */      
                        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                    dpsi_x1_rc, ndc, &b, beta1+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                    dpsi_x2_rc, ndc, &b, beta2+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);
                        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                    dpsi_x3_rc, ndc, &b, beta3+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);                        
                        free(dpsi_x1_rc); free(dpsi_x2_rc); free(dpsi_x3_rc);
                    }
                }
                // Kinetic stress
                if(dim == 0){
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, dpsi_full+spn_i*size_s+kpt*size_k, 1, kpt, pSPARC->dmcomm);
                    //ts = MPI_Wtime();
                    temp_k[0] = temp_k[1] = temp_k[3] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_1
                        dpsi_ptr2 = dpsi_full + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_2
                        dpsi1_dpsi1 = dpsi1_dpsi2 = dpsi2_dpsi2 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi1_dpsi1 += creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr + i));
                            dpsi1_dpsi2 += creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i));
                            dpsi2_dpsi2 += creal(*(dpsi_ptr2 + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr2 + i)) * cimag(*(dpsi_ptr2 + i));
                        }
                        g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                        temp_k[0] += dpsi1_dpsi1 * g_nk;
                        temp_k[1] += dpsi1_dpsi2 * g_nk;
                        temp_k[3] += dpsi2_dpsi2 * g_nk;
                    }
                    stress_k[0] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[0];
                    stress_k[1] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[1];
                    stress_k[3] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[3];

                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, dpsi_full+spn_i*size_s+kpt*size_k, 2, kpt, pSPARC->dmcomm);
                    temp_k[2] = temp_k[5] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_1
                        dpsi_ptr2 = dpsi_full + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_3
                        dpsi1_dpsi3 = dpsi3_dpsi3 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi1_dpsi3 += creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i));
                            dpsi3_dpsi3 += creal(*(dpsi_ptr2 + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr2 + i)) * cimag(*(dpsi_ptr2 + i));
                        }
                        g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                        temp_k[2] += dpsi1_dpsi3 * g_nk;
                        temp_k[5] += dpsi3_dpsi3 * g_nk;
                    }
                    stress_k[2] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[2];
                    stress_k[5] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[5];
                } else if(dim == 1){
                    temp_k[4] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_2
                        dpsi_ptr2 = dpsi_full + spn_i * size_s + kpt * size_k + n * DMnd; // dpsi_3
                        dpsi2_dpsi3 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi2_dpsi3 += creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i));
                        }
                        temp_k[4] += dpsi2_dpsi3 * pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                    }
                    stress_k[4] -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[4];
                }
                count++;
            }
        }    
    }
    
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 10, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, stress_k, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal stress */
    // go over all atoms and find nonlocal stress
    beta1_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin;
    beta1_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 2;
    beta1_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 3;
    beta2_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 4;
    beta2_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 5;
    beta2_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 6;
    beta3_x1 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 7;
    beta3_x2 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 8;
    beta3_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 9;
    count = 0;
    
    double alpha_r, alpha_i;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    eJ = 0.0; for(i = 0; i < 9; i++) SJ[i] = 0.0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                        temp2_e = 0.0; for(i = 0; i < 9; i++) temp2_s[i] = 0.0;
                        ldispl = 0;
                        for (l = 0; l <= lmax; l++) {
                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += pSPARC->psd[ityp].ppl[l];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                temp_e = 0.0; for(i = 0; i < 9; i++) temp_s[i] = 0.0;
                                for (m = -l; m <= l; m++) {
                                    alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                    temp_e += pow(alpha_r, 2.0) + pow(alpha_i, 2.0);
                                    temp_s[0] += alpha_r * creal(beta1_x1[count]) - alpha_i * cimag(beta1_x1[count]);
                                    temp_s[1] += alpha_r * creal(beta1_x2[count]) - alpha_i * cimag(beta1_x2[count]);
                                    temp_s[2] += alpha_r * creal(beta1_x3[count]) - alpha_i * cimag(beta1_x3[count]);
                                    temp_s[3] += alpha_r * creal(beta2_x1[count]) - alpha_i * cimag(beta2_x1[count]);
                                    temp_s[4] += alpha_r * creal(beta2_x2[count]) - alpha_i * cimag(beta2_x2[count]);
                                    temp_s[5] += alpha_r * creal(beta2_x3[count]) - alpha_i * cimag(beta2_x3[count]);
                                    temp_s[6] += alpha_r * creal(beta3_x1[count]) - alpha_i * cimag(beta3_x1[count]);
                                    temp_s[7] += alpha_r * creal(beta3_x2[count]) - alpha_i * cimag(beta3_x2[count]);
                                    temp_s[8] += alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]);
                                    count++;
                                }
                                gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                temp2_e += temp_e * gamma_jl;
                                for(i = 0; i < 9; i++)
                                    temp2_s[i] += temp_s[i] * gamma_jl;
                            }
                            ldispl += pSPARC->psd[ityp].ppl[l];
                        }
                        eJ += temp2_e * g_nk;
                        for(i = 0; i < 9; i++)
                            SJ[i] += temp2_s[i] * g_nk;
                    }
                    
                    kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                    energy_nl -= kptwt * eJ;
                    for(i = 0; i < 9; i++)
                        stress_nl[i] -= kptwt * SJ[i];
                }
            }
        }
    }     
    
    for(i = 0; i < 9; i++)
        stress_nl[i] *= (2.0/pSPARC->Nspin) * 2.0;
    
    energy_nl *= (2.0/pSPARC->Nspin)/pSPARC->dV;   
    
    
    if(pSPARC->cell_typ == 0){
        pSPARC->stress_nl[0] = stress_nl[0] + energy_nl;
        pSPARC->stress_nl[1] = stress_nl[1];
        pSPARC->stress_nl[2] = stress_nl[2];
        pSPARC->stress_nl[3] = stress_nl[4] + energy_nl;
        pSPARC->stress_nl[4] = stress_nl[5];
        pSPARC->stress_nl[5] = stress_nl[8] + energy_nl;
        for(i = 0; i < 6; i++)
            pSPARC->stress_k[i] = stress_k[i];
    } else {
        double *c_g, *c_c;
        c_g = (double*) malloc(9*sizeof(double));
        c_c = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
                c_c[3*i+j] = pSPARC->LatUVec[3*j+i];
            }
        }
        pSPARC->stress_nl[0] = c_g[0] * (c_c[0] * stress_nl[0] + c_c[1] * stress_nl[1] + c_c[2] * stress_nl[2]) +
                               c_g[1] * (c_c[0] * stress_nl[3] + c_c[1] * stress_nl[4] + c_c[2] * stress_nl[5]) + 
                               c_g[2] * (c_c[0] * stress_nl[6] + c_c[1] * stress_nl[7] + c_c[2] * stress_nl[8]) + energy_nl;                       
        pSPARC->stress_nl[1] = c_g[0] * (c_c[3] * stress_nl[0] + c_c[4] * stress_nl[1] + c_c[5] * stress_nl[2]) +
                               c_g[1] * (c_c[3] * stress_nl[3] + c_c[4] * stress_nl[4] + c_c[5] * stress_nl[5]) + 
                               c_g[2] * (c_c[3] * stress_nl[6] + c_c[4] * stress_nl[7] + c_c[5] * stress_nl[8]);
        pSPARC->stress_nl[2] = c_g[0] * (c_c[6] * stress_nl[0] + c_c[7] * stress_nl[1] + c_c[8] * stress_nl[2]) +
                               c_g[1] * (c_c[6] * stress_nl[3] + c_c[7] * stress_nl[4] + c_c[8] * stress_nl[5]) + 
                               c_g[2] * (c_c[6] * stress_nl[6] + c_c[7] * stress_nl[7] + c_c[8] * stress_nl[8]); 
        pSPARC->stress_nl[3] = c_g[3] * (c_c[3] * stress_nl[0] + c_c[4] * stress_nl[1] + c_c[5] * stress_nl[2]) +
                               c_g[4] * (c_c[3] * stress_nl[3] + c_c[4] * stress_nl[4] + c_c[5] * stress_nl[5]) + 
                               c_g[5] * (c_c[3] * stress_nl[6] + c_c[4] * stress_nl[7] + c_c[5] * stress_nl[8]) + energy_nl;
        pSPARC->stress_nl[4] = c_g[3] * (c_c[6] * stress_nl[0] + c_c[7] * stress_nl[1] + c_c[8] * stress_nl[2]) +
                               c_g[4] * (c_c[6] * stress_nl[3] + c_c[7] * stress_nl[4] + c_c[8] * stress_nl[5]) + 
                               c_g[5] * (c_c[6] * stress_nl[6] + c_c[7] * stress_nl[7] + c_c[8] * stress_nl[8]);
        pSPARC->stress_nl[5] = c_g[6] * (c_c[6] * stress_nl[0] + c_c[7] * stress_nl[1] + c_c[8] * stress_nl[2]) +
                               c_g[7] * (c_c[6] * stress_nl[3] + c_c[7] * stress_nl[4] + c_c[8] * stress_nl[5]) + 
                               c_g[8] * (c_c[6] * stress_nl[6] + c_c[7] * stress_nl[7] + c_c[8] * stress_nl[8]) + energy_nl;

        pSPARC->stress_k[0] = c_g[0] * (c_g[0] * stress_k[0] + c_g[1] * stress_k[1] + c_g[2] * stress_k[2]) +
                              c_g[1] * (c_g[0] * stress_k[1] + c_g[1] * stress_k[3] + c_g[2] * stress_k[4]) +
                              c_g[2] * (c_g[0] * stress_k[2] + c_g[1] * stress_k[4] + c_g[2] * stress_k[5]);
        pSPARC->stress_k[1] = c_g[0] * (c_g[3] * stress_k[0] + c_g[4] * stress_k[1] + c_g[5] * stress_k[2]) +
                              c_g[1] * (c_g[3] * stress_k[1] + c_g[4] * stress_k[3] + c_g[5] * stress_k[4]) +
                              c_g[2] * (c_g[3] * stress_k[2] + c_g[4] * stress_k[4] + c_g[5] * stress_k[5]);
        pSPARC->stress_k[2] = c_g[0] * (c_g[6] * stress_k[0] + c_g[7] * stress_k[1] + c_g[8] * stress_k[2]) +
                              c_g[1] * (c_g[6] * stress_k[1] + c_g[7] * stress_k[3] + c_g[8] * stress_k[4]) +
                              c_g[2] * (c_g[6] * stress_k[2] + c_g[7] * stress_k[4] + c_g[8] * stress_k[5]);
        pSPARC->stress_k[3] = c_g[3] * (c_g[3] * stress_k[0] + c_g[4] * stress_k[1] + c_g[5] * stress_k[2]) +
                              c_g[4] * (c_g[3] * stress_k[1] + c_g[4] * stress_k[3] + c_g[5] * stress_k[4]) +
                              c_g[5] * (c_g[3] * stress_k[2] + c_g[4] * stress_k[4] + c_g[5] * stress_k[5]);
        pSPARC->stress_k[4] = c_g[3] * (c_g[6] * stress_k[0] + c_g[7] * stress_k[1] + c_g[8] * stress_k[2]) +
                              c_g[4] * (c_g[6] * stress_k[1] + c_g[7] * stress_k[3] + c_g[8] * stress_k[4]) +
                              c_g[5] * (c_g[6] * stress_k[2] + c_g[7] * stress_k[4] + c_g[8] * stress_k[5]);
        pSPARC->stress_k[5] = c_g[6] * (c_g[6] * stress_k[0] + c_g[7] * stress_k[1] + c_g[8] * stress_k[2]) +
                              c_g[7] * (c_g[6] * stress_k[1] + c_g[7] * stress_k[3] + c_g[8] * stress_k[4]) +
                              c_g[8] * (c_g[6] * stress_k[2] + c_g[7] * stress_k[4] + c_g[8] * stress_k[5]);                                                                                   
  
        free(c_g); free(c_c);
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }


    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        }
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
            MPI_Reduce(MPI_IN_PLACE, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(pSPARC->stress_nl, pSPARC->stress_nl, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);

            MPI_Reduce(pSPARC->stress_k, pSPARC->stress_k, 6, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_z;

       for(int i = 0; i < 6; i++) {
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
    
    
    free(alpha);
    free(temp_k);
    free(stress_k);
    free(stress_nl);
    free(SJ);
    free(temp_s);
    free(temp2_s);
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
            }
        }
        
    }

}
