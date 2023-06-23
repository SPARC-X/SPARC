/**
 * @file    stress.c
 * @brief   This file contains the functions for calculating stress in cyclix symmetry.
 *
 * @author  abhiraj sharma <asharma424@gatech.edu>
 			Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "cyclix_stress.h"
#include "cyclix_tools.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"
#include "stress.h"
#include "cyclix_forces.h"
#include "spinOrbitCoupling.h"

#define TEMP_TOL 1e-12


/*
@ brief: function to calculate ionic stress/pressure and add to the electronic stress/pressure
*/


void Calculate_ionic_stress_cyclix(SPARC_OBJ *pSPARC){
    double stress_i = 0.0, avgvel = 0.0, mass_tot = 0.0;
    int ityp, atm, count;
    
    count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
	    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
		    avgvel += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 2];
		    mass_tot += pSPARC->Mass[ityp];
		    count ++;
	    }
	}
	
    avgvel /= mass_tot;
    
    count = 0;
    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
        for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
            stress_i -= pSPARC->Mass[ityp] * pow((pSPARC->ion_vel[count * 3 + 2] - avgvel), 2.0);
            count++;
        }
    }

    // Determine ionic stress and pressure
 	double cell_measure = pSPARC->range_z;
    
    pSPARC->stress_i[5] = (2*M_PI/pSPARC->range_y) * stress_i/cell_measure;
 	pSPARC->stress[5] += pSPARC->stress_i[5];

}


/*
@ brief: function to calculate electronic stress
*/

void Calculate_electronic_stress_cyclix(SPARC_OBJ *pSPARC) {
	int rank;
    double t1, t2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // find exchange-correlation components of stress
    t1 = MPI_Wtime();
    Calculate_XC_stress_cyclix(pSPARC);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating exchange-correlation stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif
    
    // find local stress components
    t1 = MPI_Wtime();
    Calculate_local_stress_cyclix(pSPARC);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating local stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif
    
    // find nonlocal + kinetic stress components
    t1 = MPI_Wtime();
    if(pSPARC->isGammaPoint) {
        Calculate_nonlocal_kinetic_stress_cyclix(pSPARC);
    } else {
        if (pSPARC->Nspinor == 1) {
            Calculate_nonlocal_kinetic_stress_kpt_cyclix(pSPARC); 
        } else {
            Calculate_nonlocal_kinetic_stress_kpt_spinor_cyclix(pSPARC);
        }
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating nonlocal+kinetic stress components: %.3f ms\n", (t2 - t1)*1e3);
#endif

    // find stress
 	if(!rank){        
        pSPARC->stress[5] = pSPARC->stress_k[5] + pSPARC->stress_xc[5] + pSPARC->stress_nl[5] + pSPARC->stress_el[5];
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
void Calculate_XC_stress_nlcc_cyclix(SPARC_OBJ *pSPARC, double *stress_xc_nlcc) {
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
        nx, ny, nz, nd, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *pshifty, *pshiftz, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x, y, z, *R, *R_interp;
    double rchrg;
    double x3_R3;
    double t1, t2, t_sort = 0.0;

    FDn = pSPARC->order / 2;
    
    int DMnx = pSPARC->Nx_d;
    int DMny = pSPARC->Ny_d;
    // DMnz = pSPARC->Nz_d;
    // DMnd = pSPARC->Nd_d;
    //DMnd = (2*pSPARC->Nspin - 1) * pSPARC->Nd_d;

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

            // original atom index this image atom corresponds to
            //int atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            nd = nx * ny * nz;
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
            //x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            //y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            //z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
            
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            for (k = kcor; k < kcor+nz2p; k++) {
                z = k * pSPARC->delta_z;
                for (j = jcor; j < jcor+ny2p; j++) {
                    y = j * pSPARC->delta_y;
                    for (i = icor; i < icor+nx2p; i++) {
                        x = pSPARC->xin + i * pSPARC->delta_x;
                        CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &R[count]);
                        if (R[count] <= rchrg) count_interp++;
                        count++;
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
            
            //double xin = pSPARC->xin + pSPARC->Atom_Influence_local[ityp].xs[iat] * pSPARC->delta_x;       

            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];

            // calculate drhocJ, 3 components
            double *drhocJ_z = malloc(nd_ex * sizeof(double));
            assert(drhocJ_z != NULL);
            for (int k2p = FDn, kp = 0; k2p < nz2p-FDn; k2p++,kp++) {
                int kshift_2p = k2p * nx2p * ny2p;
                int kshift_p = kp * nxp * nyp;
                for (int j2p = FDn, jp = 0; j2p < ny2p-FDn; j2p++,jp++) {
                    int jshift_2p = kshift_2p + j2p * nx2p;
                    int jshift_p = kshift_p + jp * nxp;
                    for (int i2p = FDn, ip = 0; i2p < nx2p-FDn; i2p++,ip++) {
                        int ishift_2p = jshift_2p + i2p;
                        int ishift_p = jshift_p + ip;
                        double drhocJ_z_val = 0.0;
                        Dpseudopot_cyclix_z(pSPARC, rhocJ, FDn, ishift_2p, pshifty_ex, pshiftz_ex, &drhocJ_z_val);
                        drhocJ_z[ishift_p] = drhocJ_z_val;
                    }
                }
            }

            // // find int Vxc(x) * drhocJ(x) dx
            double *Vxc = pSPARC->XCPotential;
            for (k = 0, kp = FDn, k_DM = dk; k < nz; k++, kp++, k_DM++) {
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                //int kshift = k * nx * ny;  
                for (j = 0, jp = FDn, j_DM = dj; j < ny; j++, jp++, j_DM++) {
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    //int jshift = kshift + j * nx;
                    for (i = 0, ip = FDn, i_DM = di; i < nx; i++, ip++, i_DM++) {
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        //int ishift = jshift + i;
                        x3_R3 = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;
                        double drhocJ_z_val = drhocJ_z[ishift_p];
                        double Vxc_val;
                        if (pSPARC->Nspin == 1)
                            Vxc_val = Vxc[ishift_DM];
                        else
                            Vxc_val = 0.5 * (Vxc[ishift_DM] + Vxc[pSPARC->Nd_d+ishift_DM]);
                        *stress_xc_nlcc += drhocJ_z_val * x3_R3 * Vxc_val * pSPARC->Intgwt_phi[ishift_DM];
                    }
                }
            }
            free(rhocJ);
            free(drhocJ_z);
        }
    }

    t1 = MPI_Wtime();
    // sum over all domains
    MPI_Allreduce(MPI_IN_PLACE, stress_xc_nlcc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->range_z;
        *stress_xc_nlcc /= cell_measure;
        *stress_xc_nlcc *= 2.0 * M_PI/pSPARC->range_y;
    }

#ifdef DEBUG
    if (!rank_comm_world) { 
        printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
        printf("NLCC XC contribution to stress: %f", *stress_xc_nlcc); 
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
void Calculate_XC_stress_cyclix(SPARC_OBJ *pSPARC) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (!rank) printf("Start calculating exchange-correlation components of stress ...\n");
#endif

    if(strcmp(pSPARC->XC,"LDA_PW") == 0 || strcmp(pSPARC->XC,"LDA_PZ") == 0){
        pSPARC->stress_xc[5] = pSPARC->Exc - pSPARC->Exc_corr;
    } else if(strcmp(pSPARC->XC,"GGA_PBE") == 0 || strcmp(pSPARC->XC,"GGA_RPBE") == 0 || strcmp(pSPARC->XC,"GGA_PBEsol") == 0){
        pSPARC->stress_xc[5] = pSPARC->Exc - pSPARC->Exc_corr;
        int DMnd, i, count;
        DMnd = (2*pSPARC->Nspin - 1) * pSPARC->Nd_d;
        double *Drho_z;
        double stress_xc = 0.0;
        
        Drho_z = (double *)malloc( DMnd * sizeof(double));
        double *rho;
        if (pSPARC->NLCC_flag) {
            rho = (double *)malloc(DMnd * sizeof(double) );
            for (i = 0; i < DMnd; i++)
                rho[i] = pSPARC->electronDens[i] + pSPARC->electronDens_core[i];
        } else {
            rho = pSPARC->electronDens;
        }

        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);

        if (pSPARC->NLCC_flag) free(rho);
        
        count = 0;
        for (int spn = 0; spn < 2*pSPARC->Nspin - 1; spn++) {
            for(i = 0; i < pSPARC->Nd_d; i++){
                stress_xc += Drho_z[count] * Drho_z[count] * pSPARC->Dxcdgrho[count] * pSPARC->Intgwt_phi[i];
		        count++;
	        }    
        }
        
        // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
        MPI_Allreduce(MPI_IN_PLACE, &stress_xc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    
        pSPARC->stress_xc[5] -= stress_xc;
        
        // deallocate
        free(Drho_z);
    }


    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->range_z;
        pSPARC->stress_xc[5] /= cell_measure;
        pSPARC->stress_xc[5] *= 2.0 * M_PI/pSPARC->range_y;
    }

    if (pSPARC->NLCC_flag) {
        double stress_xc_nlcc = 0;
        Calculate_XC_stress_nlcc_cyclix(pSPARC, &stress_xc_nlcc);
        pSPARC->stress_xc[5] += stress_xc_nlcc;
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

void Calculate_local_stress_cyclix(SPARC_OBJ *pSPARC) {

	if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    
    int ityp, iat, i, j, k, p, ip, jp, kp, ip2, jp2, kp2, di, dj, dk, i_DM, j_DM, k_DM, FDn, count, count_interp,
        DMnx, DMny, DMnd, nx, ny, nz, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *pshifty, *pshifty_ex, *pshiftz, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x, y, z, *R, *VJ, *VJ_ref, *bJ, *bJ_ref, DbJ_z_val, DbJ_ref_z_val, DVJ_z_val, DVJ_ref_z_val,
           *R_interp, *VJ_interp;
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    double temp1, temp2, temp3, temp_z;
    double x3_R3;
    
    double stress_el = 0.0, stress_corr = 0.0;
    
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
    w2_diag = (pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_z[0]) * -inv_4PI;

    Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
    Lap_stencil = Lap_wt+5;
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, -inv_4PI);

    // Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;
    DMnx = pSPARC->Nx_d; DMny = pSPARC->Ny_d; // DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d;
    
    // shift vectors initialized
    pshifty = (int *)malloc( (FDn+1) * sizeof(int));
    pshifty_ex = (int *)malloc( (FDn+1) * sizeof(int));

    pshiftz = (int *)malloc( (FDn+1) * sizeof(int));
    pshiftz_ex = (int *)malloc( (FDn+1) * sizeof(int));

    // find gradient of phi
    double *Dphi_z;
    Dphi_z = (double *)malloc( DMnd * sizeof(double));
    
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, pSPARC->elecstPotential, Dphi_z, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd; i++){
        temp1 = 0.5 * (pSPARC->psdChrgDens[i] - pSPARC->electronDens[i]) * pSPARC->elecstPotential[i];
        stress_el += (inv_4PI * Dphi_z[i] * Dphi_z[i] + temp1) * pSPARC->Intgwt_phi[i];
    }
    
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];
            
            // original atom index this image atom corresponds to
            // atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            // nd = nx * ny * nz;
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
            //x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            //y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            //z0_i_shift =  z0_i - pSPARC->delta_z * kcor;
   
            // find distance between atom and finite-difference grids
            count = 0; count_interp = 0;
            for (k = kcor; k < kcor+nz2p; k++) {
                z = k * pSPARC->delta_z;
                for (j = jcor; j < jcor+ny2p; j++) {
                    y = j * pSPARC->delta_y;
                    for (i = icor; i < icor+nx2p; i++) {
                        x = pSPARC->xin + i * pSPARC->delta_x;
                        CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &R[count]);
                        if (R[count] <= rchrg) count_interp++;
                        count++;
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
            //printf("rank = %d, R[%d] = %.13e\n", rank, len_interp-1, R[len_interp-1]); // R is not sorted!
            SortSplineInterp(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
                             R_interp, VJ_interp, count_interp, pSPARC->psd[ityp].SplinerVlocD);

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

            double xin = pSPARC->xin + (pSPARC->Atom_Influence_local[ityp].xs[iat] - FDn) * pSPARC->delta_x;       
            Calc_lapV(pSPARC, VJ, FDn, nx2p, ny2p, nz2p, nxp, nyp, nzp, Lap_wt, w2_diag, xin, -inv_4PI, bJ);
            Calc_lapV(pSPARC, VJ_ref, FDn, nx2p, ny2p, nz2p, nxp, nyp, nzp, Lap_wt, w2_diag, xin, -inv_4PI, bJ_ref);

            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];
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
                        DbJ_z_val = 0.0;
                        DbJ_ref_z_val = 0.0;
                        DVJ_z_val = 0.0;
                        DVJ_ref_z_val = 0.0;

            			Dpseudopot_cyclix_z(pSPARC, bJ, FDn, ishift_p, pshifty, pshiftz, &DbJ_z_val);
            			Dpseudopot_cyclix_z(pSPARC, bJ_ref, FDn, ishift_p, pshifty, pshiftz, &DbJ_ref_z_val);
            			Dpseudopot_cyclix_z(pSPARC, VJ, FDn, ishift_2p, pshifty_ex, pshiftz_ex, &DVJ_z_val);
            			Dpseudopot_cyclix_z(pSPARC, VJ_ref, FDn, ishift_2p, pshifty_ex, pshiftz_ex, &DVJ_ref_z_val);
                          
                        // find integrals in the stress expression
                        x3_R3 = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;

                        stress_el += DbJ_z_val *  pSPARC->elecstPotential[ishift_DM] * x3_R3 * pSPARC->Intgwt_phi[ishift_DM];

                        temp1 = pSPARC->Vc[ishift_DM] - VJ_ref[ishift_2p];
                        temp2 = pSPARC->Vc[ishift_DM];
                        temp3 = pSPARC->psdChrgDens[ishift_DM] + pSPARC->psdChrgDens_ref[ishift_DM];
                        temp_z = DbJ_ref_z_val*temp1 + DbJ_z_val*temp2 + (DVJ_ref_z_val-DVJ_z_val)*temp3 - DVJ_ref_z_val*bJ_ref[ishift_p];
                        
                        stress_corr += temp_z * x3_R3 * pSPARC->Intgwt_phi[ishift_DM];
                    }
                }
            }
  

            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
        }
    }
    
    stress_el += 0.5 * stress_corr;

    t1 = MPI_Wtime();
    // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
    MPI_Allreduce(MPI_IN_PLACE, &stress_el, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();
    
    pSPARC->stress_el[5] = stress_el + pSPARC->Esc;

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->range_z;
        pSPARC->stress_el[5] /= cell_measure;
        pSPARC->stress_el[5] *= 2.0 * M_PI/pSPARC->range_y;
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
    free(Dphi_z);
    free(pshifty);
    free(pshiftz);
    free(pshifty_ex);
    free(pshiftz_ex);    

}



void Dpseudopot_cyclix_z(SPARC_OBJ *pSPARC, double *VJ, int FDn, int ishift_p, int *pshifty, int *pshiftz, double *DVJ_z_val) {
    double c31 = -pSPARC->twist;
    double DY, DZ;
    for (int p = 1; p <= FDn; p++) {
        DY = (VJ[ishift_p+pshifty[p]] - VJ[ishift_p-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
        DZ = (VJ[ishift_p+pshiftz[p]] - VJ[ishift_p-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];
        *DVJ_z_val += c31 * DY + DZ;
    }    
}


/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_cyclix(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, k_DM, atom_index, count, l, m, lmax, spn_i, nspin, size_s;
    nspin = pSPARC->Nspin_spincomm; // number of spin in my spin communicator
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates; // total number of bands
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    // DMnz = pSPARC->Nz_d_dmcomm;
    size_s = ncol * DMnd;

    double *alpha, *beta, *beta3, *psi_ptr, *dpsi_ptr, *psi_rc, *dpsi_x3_rc, *psi_rc_ptr, *dpsi_x3_rc_ptr, R3, x3_R3;
    double SJ, eJ, temp_e, temp_s, temp2_e, temp2_s, g_nk, *beta3_x3, gamma_jl;
    double dpsi3_dpsi3;
    
    //double ts, te;

    double energy_nl = 0.0, stress_k = 0.0, stress_nl = 0.0;
    
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 2, sizeof(double));
    
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            //lmax = pSPARC->psd[ityp].lmax;
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
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi_cyclix[iat], ndc, 
                            psi_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiplied dV to get inner-product
                free(psi_rc);    
            }
        }
        count++;
    }    
    
    /* find inner product <Chi_Jlm, dPsi_3.(z-R_J3)> */
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        // find dPsi in z-direction
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, pSPARC->Yorb, 2, pSPARC->dmcomm);
        beta3 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin + count);
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                dpsi_x3_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = pSPARC->Yorb + n * DMnd;
                    dpsi_x3_rc_ptr = dpsi_x3_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                        k_DM = indx / (DMnx * DMny);
                        x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                        //nonCart2Cart_coord(pSPARC, &x1_Ri, &x2_Ri, &x3_Ri);
                        *(dpsi_x3_rc_ptr + i) = *(dpsi_ptr + indx) * x3_R3;
                    }
                }
            
                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                 *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                 *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                 *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                 */      
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi_cyclix[iat], ndc, 
                            dpsi_x3_rc, ndc, 1.0, beta3+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);                        
                free(dpsi_x3_rc);
            }
        }
        
        // Kinetic stress
        
        for(n = 0; n < ncol; n++){
            dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_1
            dpsi3_dpsi3 = 0.0;
            for(i = 0; i < DMnd; i++){
                dpsi3_dpsi3 += *(dpsi_ptr + i) * *(dpsi_ptr + i) * pSPARC->Intgwt_psi[i];
            }
            g_nk = pSPARC->occ[spn_i*Ns + n + pSPARC->band_start_indx];
            stress_k += dpsi3_dpsi3 * g_nk;
        }
        stress_k *= -(2.0/pSPARC->Nspin);

        count++;
    }
    
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 2, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal stress */
    // go over all atoms and find nonlocal stress
    beta3_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin;
    count = 0;
    
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                eJ = 0.0; SJ = 0.0;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    g_nk = pSPARC->occ[spn_i*Ns + n]; // TODO: for k-points calculation, use occ[n+k*Ns]
                    temp2_e = 0.0; temp2_s = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == pSPARC->localPsd[ityp]) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            temp_e = 0.0; temp_s = 0.0;
                            for (m = -l; m <= l; m++) {
                                temp_e += alpha[count] * alpha[count];
                                temp_s += alpha[count] * beta3_x3[count];
                                count++;
                            }
                            gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                            temp2_e += temp_e * gamma_jl;
                            temp2_s += temp_s * gamma_jl;
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }
                    eJ += temp2_e * g_nk;
                    SJ += temp2_s * g_nk;
                }
                
                energy_nl -= eJ;
                stress_nl -= SJ;
            }
        }
    }    
    
    stress_nl *= (2.0/pSPARC->Nspin) * 2.0;
    
    energy_nl *= (2.0/pSPARC->Nspin);   
    
    stress_nl += energy_nl;

    // sum over all spin
    if (pSPARC->npspin > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }
    
    // sum over all bands
    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    pSPARC->stress_nl[5] = stress_nl;
    pSPARC->stress_k[5] = stress_k;

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->range_z;
        pSPARC->stress_nl[5] /= cell_measure;
        pSPARC->stress_nl[5] *= 2.0 * M_PI/pSPARC->range_y;
        pSPARC->stress_k[5] /= cell_measure;
        pSPARC->stress_k[5] *= 2.0 * M_PI/pSPARC->range_y;

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
}


/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_kpt_cyclix(SPARC_OBJ *pSPARC)
{ 
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, k_DM, atom_index, count, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    // DMnz = pSPARC->Nz_d_dmcomm;
    
    double _Complex *alpha, *beta, *beta3, *psi_ptr, *dpsi_ptr, *psi_rc, *dpsi_x3_rc, *psi_rc_ptr, *dpsi_x3_rc_ptr;
    double _Complex *beta3_x3;
    double SJ, eJ, temp_k, temp_e, temp_s, temp2_e, temp2_s, g_nk, gamma_jl, kptwt, R1, R2, R3, x3_R3;
    double dpsi3_dpsi3;
    
    //double ts, te;

    double energy_nl = 0.0, stress_k = 0.0, stress_nl = 0.0;
    
    alpha = (double _Complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 2, sizeof(double _Complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, kpt_vec[3], theta;
    double _Complex bloch_fac, a, b;
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
                //lmax = pSPARC->psd[ityp].lmax;
                if (!pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3  ];
                    R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    theta = -k1 * (floor(R1/Lx) * Lx) - k2 * (floor(R2/Ly) * Ly) - k3 * (floor(R3/Lz) * Lz);
                    bloch_fac = cos(theta) - sin(theta) * I;
                    a = bloch_fac;
                    b = 1.0;
                    
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                    psi_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    
                    /* first find inner product <Psi_n, Chi_Jlm> */
                    for (n = 0; n < ncol; n++) {
                        psi_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                        psi_rc_ptr = psi_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(psi_rc_ptr + i) = conj(*(psi_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                        }
                    }
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &a, pSPARC->nlocProj[ityp].Chi_c_cyclix[iat], ndc, 
                                psi_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiplied dV to get inner-product
                    free(psi_rc);    
                }
            }
            count++;
        }
    }       
    
    /* find inner product <Chi_Jlm, dPsi_3.(z-R_J3)> */
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            kpt_vec[0] = k1 = pSPARC->k1_loc[kpt];
            kpt_vec[1] = k2 = pSPARC->k2_loc[kpt];
            kpt_vec[2] = k3 = pSPARC->k3_loc[kpt];
            // find dPsi in direction dim
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, 2, kpt_vec, pSPARC->dmcomm);
            beta3 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin + count);
            
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
                    dpsi_x3_rc = (double _Complex *)malloc( ndc * ncol * sizeof(double _Complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++) {
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd;
                        dpsi_x3_rc_ptr = dpsi_x3_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                            k_DM = indx / (DMnx * DMny);
                            x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                            //nonCart2Cart_coord(pSPARC, &x1_Ri, &x2_Ri, &x3_Ri);
                            *(dpsi_x3_rc_ptr + i) = *(dpsi_ptr + indx) * x3_R3;
                        }
                    }
                
                    /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                     *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                     *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                     *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).

                     */      
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c_cyclix[iat], ndc, 
                                dpsi_x3_rc, ndc, &b, beta3+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj);                        
                    free(dpsi_x3_rc);
                }
            }
            // Kinetic stress
            temp_k = 0.0;
            for(n = 0; n < ncol; n++){
                dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_1
                dpsi3_dpsi3 = 0.0;
                for(i = 0; i < DMnd; i++){
                    dpsi3_dpsi3 += (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr + i))) * pSPARC->Intgwt_psi[i];
                }
                g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                temp_k += dpsi3_dpsi3 * g_nk;
            }
            stress_k -= (2.0/pSPARC->Nspin) * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k;
         
            count++;
        }
    }
    
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 2, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal stress */
    // go over all atoms and find nonlocal stress
    beta3_x3 = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin;
    count = 0;
    
    double alpha_r, alpha_i;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                    eJ = 0.0; SJ = 0.0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                        temp2_e = 0.0; temp2_s = 0.0;
                        ldispl = 0;
                        for (l = 0; l <= lmax; l++) {
                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += pSPARC->psd[ityp].ppl[l];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                temp_e = 0.0; temp_s = 0.0;
                                for (m = -l; m <= l; m++) {
                                    alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                    temp_e += pow(alpha_r, 2.0) + pow(alpha_i, 2.0);
                                    temp_s += alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]);
                                    count++;
                                }
                                gamma_jl = pSPARC->psd[ityp].Gamma[ldispl+np];
                                temp2_e += temp_e * gamma_jl;
                                temp2_s += temp_s * gamma_jl;
                            }
                            ldispl += pSPARC->psd[ityp].ppl[l];
                        }
                        eJ += temp2_e * g_nk;
                        SJ += temp2_s * g_nk;
                    }
                    
                    kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                    energy_nl -= kptwt * eJ;
                    stress_nl -= kptwt * SJ;
                }
            }
        }
    }     
    
    stress_nl *= (2.0/pSPARC->Nspin) * 2.0;
    
    energy_nl *= (2.0/pSPARC->Nspin);   
    
    stress_nl += energy_nl;
    
    // sum over all spin
    if (pSPARC->npspin > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

    // sum over all kpoints
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);           
    }

    pSPARC->stress_nl[5] = stress_nl;
    pSPARC->stress_k[5] = stress_k;

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->range_z;
        pSPARC->stress_nl[5] /= cell_measure;
        pSPARC->stress_nl[5] *= 2.0 * M_PI/pSPARC->range_y;
        pSPARC->stress_k[5] /= cell_measure;
        pSPARC->stress_k[5] *= 2.0 * M_PI/pSPARC->range_y;

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
}


/**
 * @brief    Calculate nonlocal + kinetic components of stress.
 */
void Calculate_nonlocal_kinetic_stress_kpt_spinor_cyclix(SPARC_OBJ *pSPARC)
{ 
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, k_DM, atom_index, count, l, m, lmax, kpt, Nk, size_k, spn_i, nspin, size_s;
    int DMndbyNspinor, Nspinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor;
    
    double _Complex *alpha, *alpha_so1, *alpha_so2, *beta, *beta3, *psi_ptr, *dpsi_ptr, *psi_rc, *dpsi_x3_rc, *psi_rc_ptr, *dpsi_x3_rc_ptr;
    double _Complex *beta3_x3;
    double SJ, eJ, temp_k, temp_e, temp_s, temp2_e, temp2_s, g_nk, gamma_jl, kptwt, R1, R2, R3, x3_R3;
    double dpsi3_dpsi3;

    double energy_nl = 0.0, stress_k = 0.0, stress_nl = 0.0;

    alpha = (double _Complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 2 * Nspinor, sizeof(double _Complex));
    alpha_so1 = (double _Complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 2 * Nspinor, sizeof(double _Complex));
    alpha_so2 = (double _Complex *)calloc( pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 2 * Nspinor, sizeof(double _Complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, kpt_vec[3], theta;
    double _Complex bloch_fac, a, b;
#ifdef DEBUG 
    if (!rank) printf("Start calculating stress contributions from kinetic and nonlocal psp. \n");
#endif

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < Nk; kpt++){
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SC");
            beta = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO1");
            beta = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * count;
            Compute_Integral_psi_Chi_kpt(pSPARC, beta, spn_i, kpt, "SO2");
            count++;
        }
    }       
    
    /* find inner product <Chi_Jlm, dPsi_3.(z-R_J3)> */
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            kpt_vec[0] = k1 = pSPARC->k1_loc[kpt];
            kpt_vec[1] = k2 = pSPARC->k2_loc[kpt];
            kpt_vec[2] = k3 = pSPARC->k3_loc[kpt];
            // find dPsi in direction dim
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, 2, kpt_vec, pSPARC->dmcomm);
            
            beta3 = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin + count);
            Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt_cyclix(pSPARC, pSPARC->Yorb_kpt, beta3, spn_i, kpt, "SC");
            beta3 = alpha_so1 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin + count);
            Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt_cyclix(pSPARC, pSPARC->Yorb_kpt, beta3, spn_i, kpt, "SO1");
            beta3 = alpha_so2 + pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nspinor * (Nk * nspin + count);
            Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt_cyclix(pSPARC, pSPARC->Yorb_kpt, beta3, spn_i, kpt, "SO2");

            // Kinetic stress
            temp_k = 0.0;
            for(n = 0; n < ncol; n++){
                dpsi3_dpsi3 = 0.0;
                for (int spinor = 0; spinor < Nspinor; spinor++) {
                    dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd + spinor * DMndbyNspinor; // dpsi_1
                    for(i = 0; i < DMndbyNspinor; i++){
                        dpsi3_dpsi3 += (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr + i))) * pSPARC->Intgwt_psi[i];
                    }
                }
                g_nk = pSPARC->occ[spn_i*Nk*Ns + kpt*Ns + n + pSPARC->band_start_indx];
                temp_k += dpsi3_dpsi3 * g_nk;
            }
            stress_k -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k;
            count++;
        }
    }
    
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 2 * Nspinor, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, alpha_so1, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 2 * Nspinor, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, alpha_so2, pSPARC->IP_displ_SOC[pSPARC->n_atom] * ncol * Nk * nspin * 2 * Nspinor, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal stress */
    Compute_stress_tensor_nloc_by_integrals_cyclix(pSPARC, &stress_nl, alpha, "SC");
    Compute_stress_tensor_nloc_by_integrals_cyclix(pSPARC, &stress_nl, alpha_so1, "SO1");
    Compute_stress_tensor_nloc_by_integrals_cyclix(pSPARC, &stress_nl, alpha_so2, "SO2");
    
    energy_nl = Compute_Nonlocal_Energy_by_integrals(pSPARC, alpha, alpha_so1, alpha_so2);

    stress_nl *= pSPARC->occfac * 2.0;    
    energy_nl *= pSPARC->occfac;    
    stress_nl -= energy_nl;

    // sum over all spin
    if (pSPARC->npspin > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm); 
    }


    // sum over all kpoints
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    // sum over all bands
    if (pSPARC->npband > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, &stress_nl, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        MPI_Allreduce(MPI_IN_PLACE, &stress_k, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm); 
    }

    pSPARC->stress_nl[5] = stress_nl;
    pSPARC->stress_k[5] = stress_k;

    if (!rank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->range_z;
        pSPARC->stress_nl[5] /= cell_measure;
        pSPARC->stress_nl[5] *= 2.0 * M_PI/pSPARC->range_y;
        pSPARC->stress_k[5] /= cell_measure;
        pSPARC->stress_k[5] *= 2.0 * M_PI/pSPARC->range_y;
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
    free(alpha_so1);
    free(alpha_so2);
}


/**
 * @brief   Calculate <ChiSC_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt_cyclix(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int spn_i, int kpt, char *option) 
{
    int i, n, ndc, ityp, iat, ncol, DMnd, atom_index;
    int spinor, Nspinor, DMndbyNspinor, spinorshift, nproj, ispinor, *IP_displ;
    int indx, i_DM, j_DM, k_DM, DMnx, DMny;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    DMndbyNspinor = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    Nspinor = pSPARC->Nspinor;

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
            Chi = pSPARC->nlocProj[ityp].Chi_c_cyclix;
        else if (!strcmpi(option, "SO1")) 
            Chi = pSPARC->nlocProj[ityp].Chisowt0_cyclix;

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
                    Chi = (spinor == 0) ? pSPARC->nlocProj[ityp].Chisowtnl_cyclix : pSPARC->nlocProj[ityp].Chisowtl_cyclix; 
                ispinor = !strcmpi(option, "SO2") ? (1 - spinor) : spinor;

                for (n = 0; n < ncol; n++) {
                    dpsi_ptr = dpsi_xi + n * DMnd + ispinor * DMndbyNspinor;
                    dpsi_xi_rc_ptr = dpsi_xi_rc + n * ndc;

                    for (i = 0; i < ndc; i++) {
                        indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                        k_DM = indx / (DMnx * DMny);
                        x3_R3 = (k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3;
                        //nonCart2Cart_coord(pSPARC, &x1_Ri, &x2_Ri, &x3_Ri);
                        *(dpsi_xi_rc_ptr + i) = *(dpsi_ptr + indx) * x3_R3;
                    }
                }
            
                /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).

                */      
                spinorshift = IP_displ[pSPARC->n_atom] * ncol * spinor;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, nproj, ncol, ndc, &bloch_fac, Chi[iat], ndc, 
                            dpsi_xi_rc, ndc, &b, beta+spinorshift+IP_displ[atom_index]*ncol, nproj);                        
            }
            free(dpsi_xi_rc);
        }
    }
}



void Compute_stress_tensor_nloc_by_integrals_cyclix(SPARC_OBJ *pSPARC, double *stress_nl, double _Complex *alpha, char *option)
{
    int i, k, n, np, ldispl, ityp, iat, ncol, Ns;
    int count, l, m, lmax, Nk, spn_i, nspin, spinor, Nspinor;
    int l_start, mexclude, ppl, *IP_displ;
    double g_nk, kptwt, alpha_r, alpha_i, SJ, temp2_s, scaled_gamma_Jl = 0;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    Nspinor = pSPARC->Nspinor;

    l_start = !strcmpi(option, "SC") ? 0 : 1;
    IP_displ = !strcmpi(option, "SC") ? pSPARC->IP_displ : pSPARC->IP_displ_SOC;

    double _Complex *beta3_x3 = alpha + IP_displ[pSPARC->n_atom]*ncol*Nspinor*Nk*nspin;

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (spinor = 0; spinor < Nspinor; spinor++) {
                double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                    lmax = pSPARC->psd[ityp].lmax;
                    for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
                        SJ = 0.0;
                        for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                            g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                            temp2_s = 0.0;
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
                                        temp2_s += scaled_gamma_Jl * (alpha_r * creal(beta3_x3[count]) - alpha_i * cimag(beta3_x3[count]));
                                        count++;
                                    }
                                }
                                ldispl += ppl;
                            }
                            SJ += temp2_s * g_nk;
                        }
                        
                        kptwt = pSPARC->kptWts_loc[k] / pSPARC->Nkpts;
                        *stress_nl -= kptwt * SJ;
                    }
                }
            }
        }
    }
}