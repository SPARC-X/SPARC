/**
 * @file    pressure.c
 * @brief   This file contains the functions for calculating pressure.
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

#include "pressure.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "lapVecRoutines.h"
#include "tools.h" 
#include "isddft.h"
#include "initialization.h"
#include "electrostatics.h"
#include "energy.h"
#include "vdWDFstress.h"
#include "d3forceStress.h"
#include "MGGAstress.h"
#include "spinOrbitCoupling.h"
#include "exactExchangePressure.h"
#include "sqProperties.h"

#define TEMP_TOL 1e-12


/*
 * @brief: function to calculate the electronic pressure
 */
void Calculate_electronic_pressure(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    	
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    
    // find exchange-correlation component of pressure
    Calculate_XC_pressure(pSPARC);
    if ((pSPARC->mGGAflag == 1) && (pSPARC->countPotentialCalculate > 1)) { // metaGGA pressure is related to wavefunction psi directly; it needs to be computed outside of function Calculate_XC_pressure
        if (pSPARC->isGammaPoint) {
            Calculate_XC_stress_mGGA_psi_term(pSPARC); // the function is in file mgga/mgga.c
        }
        else {
            Calculate_XC_stress_mGGA_psi_term_kpt(pSPARC); // the function is in file mgga/mgga.c
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating exchnage-correlation pressure components: %.3f ms\n", (t2 - t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // find local pressure components
    Calculate_local_pressure(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating local pressure components: %.3f ms\n", (t2 - t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // find nonlocal pressure components
    Calculate_nonlocal_pressure(pSPARC);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Time for calculating nonlocal pressure components: %.3f ms\n", (t2 - t1)*1e3);
#endif
    
    if (pSPARC->usefock > 0) {
        // find exact exchange pressure components
    #ifdef DEBUG
        t1 = MPI_Wtime();
    #endif    
        Calculate_exact_exchange_pressure(pSPARC);
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank) printf("Time for calculating exact exchange pressure components: %.3f ms\n", (t2 - t1)*1e3);
    #endif
    }    

	// find total pressure    
 	if(!rank){
 	    // Define measure of unit cell
 		double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;
        		
 		pSPARC->pres = (-2 * (pSPARC->Eband + pSPARC->Escc) + pSPARC->pres_xc + pSPARC->pres_el + pSPARC->pres_nl);
        if (pSPARC->usefock > 0) pSPARC->pres += pSPARC->pres_exx; 
 		pSPARC->pres /= (-3 * cell_measure); // measure = volume for 3D, area for 2D, and length for 1D.
 	}
#ifdef DEBUG    
    if (!rank) {
        printf("Electronic contribution to pressure = %.15f Ha/Bohr^3, %.15f GPa\n", pSPARC->pres, pSPARC->pres*CONST_HA_BOHR3_GPA);
    }
#endif
}



/*
 * @brief: find pressure contribution from exchange-correlation
 */
void Calculate_XC_pressure(SPARC_OBJ *pSPARC) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (!rank) printf("Start calculating exchange-correlation components of pressure ...\n");
#endif

    if(strcmpi(pSPARC->XC,"LDA_PW") == 0 || strcmpi(pSPARC->XC,"LDA_PZ") == 0){
        pSPARC->pres_xc = 3 * pSPARC->Exc - pSPARC->Exc_corr;
    } else if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0 
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0
            || strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0 || strcmpi(pSPARC->XC,"SCAN") == 0){
        pSPARC->pres_xc = 3 * pSPARC->Exc - pSPARC->Exc_corr;
        int DMnd, i;
        DMnd = (2*pSPARC->Nspin - 1) * pSPARC->Nd_d;
        double *Drho_x, *Drho_y, *Drho_z, *lapcT;
        double pres_xc;
        Drho_x = (double *)malloc( DMnd * sizeof(double));
        Drho_y = (double *)malloc( DMnd * sizeof(double));
        Drho_z = (double *)malloc( DMnd * sizeof(double));
    
        double *rho;
        if (pSPARC->NLCC_flag) {
            rho = (double *)malloc(DMnd * sizeof(double) );
            for (i = 0; i < DMnd; i++)
                rho[i] = pSPARC->electronDens[i] + pSPARC->electronDens_core[i];
        } else {
            rho = pSPARC->electronDens;
        }

        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, rho, Drho_x, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, rho, Drho_y, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, pSPARC->Nd_d, pSPARC->DMVertices, (2*pSPARC->Nspin - 1), 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);
        
        if (pSPARC->NLCC_flag) free(rho);

        pres_xc = 0.0;
        
        if(pSPARC->cell_typ == 0){
            for(i = 0; i < DMnd; i++){
                pres_xc += (Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i]) * pSPARC->Dxcdgrho[i];
            }
            pres_xc *= pSPARC->dV;
            
            // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
            MPI_Allreduce(MPI_IN_PLACE, &pres_xc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
        } else{
            lapcT = (double *) malloc(6 * sizeof(double));
            lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
            lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
            for(i = 0; i < DMnd; i++){
                pres_xc += (Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                            Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i])) * pSPARC->Dxcdgrho[i]; 
            }
            pres_xc *= pSPARC->dV;
            
            // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
            MPI_Allreduce(MPI_IN_PLACE, &pres_xc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
            free(lapcT);
        }
        
        pSPARC->pres_xc -= pres_xc;
        
        // deallocate
        free(Drho_x); free(Drho_y); free(Drho_z);
    } 

    if (pSPARC->d3Flag == 1) {
        // if (!rank) printf("XC pressure before d3 %9.6E\n", pSPARC->pres_xc);
        d3_grad_cell_stress(pSPARC);
        // if (!rank) printf("XC pressure after d3 %9.6E\n", pSPARC->pres_xc);
    }

    if (pSPARC->vdWDFFlag != 0) { // either vdW_DF1 or vdW_DF2, compute the contribution of nonlinear correlation of vdWDF on stress/pressure
        Calculate_XC_stress_vdWDF(pSPARC); // the function is in file vdW/vdWDF/vdWDF.c
    }

    if (pSPARC->NLCC_flag) {
        double Calculate_XC_pressure_nlcc(SPARC_OBJ *pSPARC);
        double pres_xc_nlcc = Calculate_XC_pressure_nlcc(pSPARC);
        pSPARC->pres_xc += pres_xc_nlcc;
    } 
}


/*
* @brief: find stress contributions from exchange-correlation
*         due to non-linear core correction (NLCC).
*/
double Calculate_XC_pressure_nlcc(SPARC_OBJ *pSPARC) {
    double pres_xc_nlcc = 0.0;
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return pres_xc_nlcc;
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
                        if (pSPARC->Nspin == 1)
                            Vxc_val = Vxc[ishift_DM];
                        else
                            Vxc_val = 0.5 * (Vxc[ishift_DM] + Vxc[pSPARC->Nd_d+ishift_DM]);
                        pres_xc_nlcc += (drhocJ_x_val*x1_R1 + drhocJ_y_val*x2_R2 + drhocJ_z_val*x3_R3) * Vxc_val;
                    }
                }
            }
            free(rhocJ);
            free(drhocJ_x);
            free(drhocJ_y);
            free(drhocJ_z);
        }
    }

    pres_xc_nlcc *= pSPARC->dV;
    
    t1 = MPI_Wtime();
    // sum over all domains
    MPI_Allreduce(MPI_IN_PLACE, &pres_xc_nlcc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (!rank_comm_world) { 
        printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
    }
#endif

    free(pshifty);
    free(pshiftz);
    free(pshifty_ex);
    free(pshiftz_ex);
    return pres_xc_nlcc;
}


/*
 * @brief: function to calculate the local pressure components for non orthogonal cell
 */
void Calculate_local_pressure(SPARC_OBJ *pSPARC) {
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
        icor, jcor, kcor, indx_ex, indx_2ex, indx_DM, *pshifty, *pshiftz, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x, y, z, *R,
           *VJ, *VJ_ref, *bJ, *bJ_ref, 
           DbJ_x_val, DbJ_ref_x_val, DbJ_y_val, DbJ_ref_y_val, DbJ_z_val, DbJ_ref_z_val, 
           DVJ_x_val, DVJ_y_val, DVJ_z_val, DVJ_ref_x_val, DVJ_ref_y_val, DVJ_ref_z_val,
           *R_interp, *VJ_interp;
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    double temp1 = 0.0, temp2 = 0.0, temp3;
    double temp_xx = 0.0, temp_yy = 0.0, temp_zz = 0.0, temp_xy = 0.0, temp_xz = 0.0, temp_yz = 0.0;
    double pressure_el = 0.0, pressure_corr = 0.0;
    double x1_R1, x2_R2, x3_R3;
    
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (!rank) printf("Start calculating local components of pressure ...\n");
#endif

    double t1, t2, t_sort = 0.0;
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

    DMnx = pSPARC->Nx_d; DMny = pSPARC->Ny_d; 
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

    if(pSPARC->cell_typ == 0){
        for(i = 0; i < DMnd; i++){
        	temp1 += Dphi_x[i] * Dphi_x[i] + Dphi_y[i] * Dphi_y[i] + Dphi_z[i] * Dphi_z[i];
        	temp2 += (pSPARC->electronDens[i] + 3 * pSPARC->psdChrgDens[i]) * pSPARC->elecstPotential[i];
        }
        pressure_el += inv_4PI * temp1 + 0.5 * temp2;
    } else {
        for(i = 0; i < DMnd; i++){
            temp_xx += Dphi_x[i] * Dphi_x[i];
            temp_yy += Dphi_y[i] * Dphi_y[i];
            temp_zz += Dphi_z[i] * Dphi_z[i];
            temp_xy += Dphi_x[i] * Dphi_y[i];
            temp_xz += Dphi_x[i] * Dphi_z[i];
            temp_yz += Dphi_y[i] * Dphi_z[i];        
        	temp1 += (pSPARC->electronDens[i] + 3 * pSPARC->psdChrgDens[i]) * pSPARC->elecstPotential[i];
        }
        pressure_el += inv_4PI * ( pSPARC->lapcT[0]*temp_xx + pSPARC->lapcT[4]*temp_yy + pSPARC->lapcT[8]*temp_zz
                                 + 2*(pSPARC->lapcT[1]*temp_xy + pSPARC->lapcT[2]*temp_xz + pSPARC->lapcT[5]*temp_yz) );
        
        pressure_el += 0.5 * temp1;
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
            for(kp = FDn, kp2 = pSPARC->order, k_DM = dk; kp2 < nzp; kp++, kp2++, k_DM++) {
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_2p = kp2 * nx2p * ny2p;
                int kshift_p = kp * nxp * nyp;
                for(jp = FDn, jp2 = pSPARC->order, j_DM = dj; jp2 < nyp; jp++, jp2++, j_DM++) {
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_2p = kshift_2p + jp2 * nx2p;
                    int jshift_p = kshift_p + jp * nxp;
                    for(ip = FDn, ip2 = pSPARC->order, i_DM = di; ip2 < nxp; ip++, ip2++, i_DM++) {
                        DbJ_x_val = DbJ_y_val = DbJ_z_val = 0.0;
                        DbJ_ref_x_val = DbJ_ref_y_val = DbJ_ref_z_val = 0.0;
                        DVJ_x_val = DVJ_y_val = DVJ_z_val = 0.0;
                        DVJ_ref_x_val = DVJ_ref_y_val = DVJ_ref_z_val = 0.0;

                        indx_ex = jshift_p + ip;
                        indx_2ex = jshift_2p + ip2;
                        for (p = 1; p <= FDn; p++) {
                            DbJ_x_val += (bJ[indx_ex+p] - bJ[indx_ex-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            DbJ_y_val += (bJ[indx_ex+pshifty[p]] - bJ[indx_ex-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            DbJ_z_val += (bJ[indx_ex+pshiftz[p]] - bJ[indx_ex-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                            DbJ_ref_x_val += (bJ_ref[indx_ex+p] - bJ_ref[indx_ex-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            DbJ_ref_y_val += (bJ_ref[indx_ex+pshifty[p]] - bJ_ref[indx_ex-pshifty[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            DbJ_ref_z_val += (bJ_ref[indx_ex+pshiftz[p]] - bJ_ref[indx_ex-pshiftz[p]]) * pSPARC->D1_stencil_coeffs_z[p];

							DVJ_x_val += (VJ[indx_2ex+p] - VJ[indx_2ex-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            DVJ_y_val += (VJ[indx_2ex+pshifty_ex[p]] - VJ[indx_2ex-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            DVJ_z_val += (VJ[indx_2ex+pshiftz_ex[p]] - VJ[indx_2ex-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];

                            DVJ_ref_x_val += (VJ_ref[indx_2ex+p] - VJ_ref[indx_2ex-p]) * pSPARC->D1_stencil_coeffs_x[p];
                            DVJ_ref_y_val += (VJ_ref[indx_2ex+pshifty_ex[p]] - VJ_ref[indx_2ex-pshifty_ex[p]]) * pSPARC->D1_stencil_coeffs_y[p];
                            DVJ_ref_z_val += (VJ_ref[indx_2ex+pshiftz_ex[p]] - VJ_ref[indx_2ex-pshiftz_ex[p]]) * pSPARC->D1_stencil_coeffs_z[p];

						}
                        
                        // find integrals in the pressure expression
                        indx_DM = jshift_DM + i_DM;
                        
                        x1_R1 = (i_DM + pSPARC->DMVertices[0]) * pSPARC->delta_x - x0_i;
                        x2_R2 = (j_DM + pSPARC->DMVertices[2]) * pSPARC->delta_y - y0_i;
                        x3_R3 = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;

                        pressure_el += (DbJ_x_val * x1_R1 + DbJ_y_val * x2_R2 + DbJ_z_val * x3_R3) * pSPARC->elecstPotential[indx_DM]; //- 0.5 * VJ[indx_2ex]) ;
                                           //-(DVJ_x_val * x1_R1 + DVJ_y_val * x2_R2 + DVJ_z_val * x3_R3) * 0.5 * bJ[indx_ex];

                        temp1 = pSPARC->Vc[indx_DM] - VJ_ref[indx_2ex];
                        temp2 = pSPARC->Vc[indx_DM];  //+ VJ[indx_2ex];
                        temp3 = pSPARC->psdChrgDens[indx_DM] + pSPARC->psdChrgDens_ref[indx_DM];
                        
                        pressure_corr += ( DbJ_ref_x_val*temp1 + DbJ_x_val*temp2 + (DVJ_ref_x_val-DVJ_x_val)*temp3 - DVJ_ref_x_val*bJ_ref[indx_ex] ) * x1_R1; 
                                          
                        pressure_corr += ( DbJ_ref_y_val*temp1 + DbJ_y_val*temp2 + (DVJ_ref_y_val-DVJ_y_val)*temp3 - DVJ_ref_y_val*bJ_ref[indx_ex] ) * x2_R2; 
                                          
                       	pressure_corr += ( DbJ_ref_z_val*temp1 + DbJ_z_val*temp2 + (DVJ_ref_z_val-DVJ_z_val)*temp3 - DVJ_ref_z_val*bJ_ref[indx_ex] ) * x3_R3; 
                       	                     
                    }
                }
            }

            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
        }   
    }
    
    pSPARC->pres_el = (pressure_el + 0.5 * pressure_corr ) * pSPARC->dV;
    t1 = MPI_Wtime();
    // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
    MPI_Allreduce(MPI_IN_PLACE, &pSPARC->pres_el, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();
    
    pSPARC->pres_el += 3 * pSPARC->Esc;
#ifdef DEBUG
    if (!rank){ 
        printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
        printf("Pressure contribution from electrostatics: %.15f Ha\n",pSPARC->pres_el);
    }    
#endif
    
    //deallocate
    free(Lap_wt);
    free(Dphi_x);
    free(Dphi_y);
    free(Dphi_z);
    free(pshifty); free(pshiftz);
    free(pshifty_ex); free(pshiftz_ex);

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
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure(SPARC_OBJ *pSPARC) {
    if (pSPARC->isGammaPoint) {
        if (pSPARC->SQFlag == 1) 
            Calculate_nonlocal_pressure_SQ(pSPARC);
        else
            Calculate_nonlocal_pressure_linear(pSPARC);
    } else {
        if (pSPARC->Nspinor == 1)
            Calculate_nonlocal_pressure_kpt(pSPARC);  
        else if (pSPARC->Nspinor == 2)
            Calculate_nonlocal_pressure_kpt_spinor(pSPARC);
    }
}

/**
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure_linear(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, i_DM, j_DM, k_DM, dim, atom_index, count, l, m, lmax, spn_i, nspin, size_s;
    nspin = pSPARC->Nspin_spincomm; // number of spin in my spin communicator
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates; // total number of bands
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    size_s = ncol * DMnd;
    
    double pressure_nloc = 0.0, *alpha, *beta, *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr, R1, R2, R3;
    double pJ, eJ, temp_e, temp_p, temp2_e, temp2_p, g_nk, *beta_x, *beta_y,
           *beta_z;
    
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 4, sizeof(double));
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal pressure\n");
#endif
    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * count;
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            if (!pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                x_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                
                /* first find inner product <Psi_n, Chi_Jlm>, and <Chi_Jlm, Psi_n> */
                for (n = 0; n < ncol; n++) {
                    x_ptr = pSPARC->Xorb + spn_i * size_s + n * DMnd;
                    x_rc_ptr = x_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        *(x_rc_ptr + i) = *(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                    }
                }
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, pSPARC->dV, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                            x_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiply dV to get inner-product
                free(x_rc);
                
            }
        }
        count++;
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
        // find dPsi in direction dim
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spn_i*size_s, pSPARC->Yorb, dim, pSPARC->dmcomm);
            beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (nspin * (dim+1) + count);
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    R1 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3];
                    R2 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1];
                    R3 = pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2];
                    ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
                    dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    for (n = 0; n < ncol; n++) {
                        dx_ptr = pSPARC->Yorb + n * DMnd;
                        dx_rc_ptr = dx_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                            if (dim == 0){
                            	i_DM = indx % DMnx;
                            	*(dx_rc_ptr + i) = *(dx_ptr + indx) * ((i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1);
                            } else if(dim == 1){
                            	k_DM = indx / (DMnx * DMny);
                    			j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                            	*(dx_rc_ptr + i) = *(dx_ptr + indx) * ((j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2);
                            } else {
                            	k_DM = indx / (DMnx * DMny);
                    			*(dx_rc_ptr + i) = *(dx_ptr + indx) * ((k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3);
                            }
                            
                        }
                    }

                
                    /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                     *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                     *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                     *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                     */      
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, 1.0, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                                dx_rc, ndc, 1.0, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); 
                    free(dx_rc);
                }
            }
            count++;
        }    
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * nspin * 4, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal pressure */
    // go over all atoms and find nonlocal pressure
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*nspin * 3;
    count = 0;
    
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            lmax = pSPARC->psd[ityp].lmax;
            for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            	eJ = pJ = 0.0;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    g_nk = pSPARC->occ[spn_i*Ns + n];
                    temp2_e = temp2_p = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == pSPARC->localPsd[ityp]) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            temp_e = temp_p = 0.0;
                            for (m = -l; m <= l; m++) {
                                temp_e += alpha[count] * alpha[count];
                                temp_p += alpha[count] * (beta_x[count] + beta_y[count] + beta_z[count]);
                                count++;
                            }
                            temp2_e += temp_e * pSPARC->psd[ityp].Gamma[ldispl+np];
                            temp2_p += temp_p * pSPARC->psd[ityp].Gamma[ldispl+np];
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }
                    eJ += temp2_e * g_nk;
                    pJ += temp2_p * g_nk;
                }
                
                pressure_nloc -= (2.0/pSPARC->Nspin) * 2.0 *  (pJ + 0.5 * eJ/pSPARC->dV);
            }
        }
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }


    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }
    
    if (!rank) {
        pSPARC->pres_nl = pressure_nloc;
    }

#ifdef DEBUG    
    if (!rank){
        printf("Pressure contribution from nonlocal pseudopotential: = %.15f Ha\n", pSPARC->pres_nl);
    }    
#endif
    
    
    free(alpha);
}


/**
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure_kpt(SPARC_OBJ *pSPARC)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, k, n, np, ldispl, ndc, ityp, iat, ncol, Ns, DMnd, DMnx, DMny, indx, i_DM, j_DM, k_DM, dim, count, l, m, lmax, atom_index, kpt, Nk, size_k, spn_i, nspin, size_s;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nspin = pSPARC->Nspin_spincomm;
    size_k = DMnd * ncol;
    size_s = size_k * Nk;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    
    double complex *alpha, *beta, *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr, *beta_x, *beta_y, *beta_z;
    double R1, R2, R3, pJ, eJ, temp_e, temp_p, temp2_e, temp2_p, g_nk;
    double pressure_nloc = 0.0;
    
    alpha = (double complex *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4, sizeof(double complex));
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double k1, k2, k3, theta, kpt_vec;
    double complex bloch_fac, a, b;
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal pressure\n");
#endif

    count = 0;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++){
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
                    x_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                    atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                    
                    /* first find inner product <Psi_n, Chi_Jlm>, and <Chi_Jlm, Psi_n> */
                    for (n = 0; n < ncol; n++) {
                        x_ptr = pSPARC->Xorb_kpt + spn_i * size_s + kpt * size_k + n * DMnd;
                        x_rc_ptr = x_rc + n * ndc;
                        for (i = 0; i < ndc; i++) {
                            *(x_rc_ptr + i) = conj(*(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]));
                        }
                    }
                    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &a, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                x_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiply dV to get inner-product
                    free(x_rc);
                }
            }
            count++;
        }
    }    
    
    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    for (dim = 0; dim < 3; dim++) {
        count = 0;
        for(spn_i = 0; spn_i < nspin; spn_i++) {
            for(kpt = 0; kpt < Nk; kpt++) {
                k1 = pSPARC->k1_loc[kpt];
                k2 = pSPARC->k2_loc[kpt];
                k3 = pSPARC->k3_loc[kpt];
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                // find dPsi in direction dim
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spn_i*size_s+kpt*size_k, pSPARC->Yorb_kpt, dim, kpt_vec, pSPARC->dmcomm);
                beta = alpha + pSPARC->IP_displ[pSPARC->n_atom] * ncol * (Nk * nspin* (dim + 1) + count);
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
                        dx_rc = (double complex *)malloc( ndc * ncol * sizeof(double complex));
                        atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                        for (n = 0; n < ncol; n++) {
                            dx_ptr = pSPARC->Yorb_kpt + n * DMnd;
                            dx_rc_ptr = dx_rc + n * ndc;
                            for (i = 0; i < ndc; i++) {
                                indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                                if (dim == 0){
                                	i_DM = indx % DMnx;
                                	*(dx_rc_ptr + i) = *(dx_ptr + indx) * ((i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - R1);
                                } else if(dim == 1){
                                	k_DM = indx / (DMnx * DMny);
                        			j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
                                	*(dx_rc_ptr + i) = *(dx_ptr + indx) * ((j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - R2);
                                } else {
                                	k_DM = indx / (DMnx * DMny);
                        			*(dx_rc_ptr + i) = *(dx_ptr + indx) * ((k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - R3);
                                }
                                
                            }
                        }

                    
                        /* Note: in principle we need to multiply dV to get inner-product, however, since Psi is normalized 
                         *       in the l2-norm instead of L2-norm, each psi value has to be multiplied by 1/sqrt(dV) to
                         *       recover the actual value. Considering this, we only multiply dV in one of the inner product
                         *       and the other dV is canceled by the product of two scaling factors, 1/sqrt(dV) and 1/sqrt(dV).
                         */      
                        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, &bloch_fac, pSPARC->nlocProj[ityp].Chi_c[iat], ndc, 
                                    dx_rc, ndc, &b, beta+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); 
                        free(dx_rc);
                    }
                }
                count++;
            }
        }      
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * Nk * nspin * 4, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal pressure */
    // go over all atoms and find nonlocal pressure
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*Nk*nspin * 3;
    count = 0;
    
    double alpha_r, alpha_i;
    for(spn_i = 0; spn_i < nspin; spn_i++) {
        for (k = 0; k < Nk; k++) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
                lmax = pSPARC->psd[ityp].lmax;
                for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {                
                	eJ = pJ = 0.0;
                    for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                        g_nk = pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                        temp2_e = temp2_p = 0.0;
                        ldispl = 0;
                        for (l = 0; l <= lmax; l++) {
                            // skip the local l
                            if (l == pSPARC->localPsd[ityp]) {
                                ldispl += pSPARC->psd[ityp].ppl[l];
                                continue;
                            }
                            for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                                temp_e = temp_p = 0.0;
                                for (m = -l; m <= l; m++) {
                                    alpha_r = creal(alpha[count]); alpha_i = cimag(alpha[count]);
                                    temp_e += pow(alpha_r, 2.0) + pow(alpha_i, 2.0);
                                    temp_p += alpha_r * creal(beta_x[count] + beta_y[count] + beta_z[count]) - alpha_i * cimag(beta_x[count] + beta_y[count] + beta_z[count]);
                                    count++;
                                }
                                temp2_e += temp_e * pSPARC->psd[ityp].Gamma[ldispl+np];
                                temp2_p += temp_p * pSPARC->psd[ityp].Gamma[ldispl+np];
                            }
                            ldispl += pSPARC->psd[ityp].ppl[l];
                        }
                        eJ += temp2_e * g_nk;
                        pJ += temp2_p * g_nk;
                    }
                    
                    pressure_nloc -= (2.0/pSPARC->Nspin) * 2.0 * pSPARC->kptWts_loc[k] / pSPARC->Nkpts * (pJ + 0.5 * eJ/pSPARC->dV);
                }
            }
        }
    }
        
    // sum over all spin
    if (pSPARC->npspin > 1) {    
        if (pSPARC->spincomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        }
    }
        
    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        }
    }

    // sum over all bands
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0){
            MPI_Reduce(MPI_IN_PLACE, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        } else{
            MPI_Reduce(&pressure_nloc, &pressure_nloc, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        }
    }
    
    if (!rank) {
        pSPARC->pres_nl = pressure_nloc;
    }

#ifdef DEBUG    
    if (!rank){
        printf("Pressure contribution from nonlocal pseudopotential: = %.15f Ha\n", pSPARC->pres_nl);
    }    
#endif
    
    
    free(alpha);
}
