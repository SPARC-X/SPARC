/**
 * @file    stress.c
 * @brief   This file contains the functions for calculating stress and pressure.
 *
 * @author  abhiraj sharma <asharma424@gatech.edu>
 			Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <mpi.h>

#include "stress.h"
#include "gradVecRoutines.h"
#include "tools.h" 
#include "isddft.h"

#define TEMP_TOL 1e-12
#define max(a,b) ((a)>(b)?(a):(b))
/*
@ brief: function to calculate the pressure
*/

void Calculate_pressure(SPARC_OBJ *pSPARC) {
	int rank;
    double t1, t2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // find local pressure components
    t1 = MPI_Wtime();
    Calculate_local_pressure_orthCell(pSPARC);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating local pressure components: %.3f ms\n", (t2 - t1)*1e3);
#endif
    
    // find nonlocal pressure components
    t1 = MPI_Wtime();
    Calculate_nonlocal_pressure(pSPARC);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("Time for calculating nonlocal pressure components: %.3f ms\n", (t2 - t1)*1e3);
#endif

	// find total pressure    
 	if(!rank){
 		pSPARC->pressure += (-2 * pSPARC->Eband + 3 * pSPARC->Exc - pSPARC->Exc_corr + 3 * pSPARC->Esc);
 		pSPARC->pressure /= (-3 * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z);
 	}
#ifdef DEBUG    
    if (!rank) {
        printf("Pressure = %.15f Ha/Bohr^3, %.15f GPa\n", pSPARC->pressure, pSPARC->pressure*29421.02648438959);
    }
#endif

}


/*
@ brief: function to calculate the local pressure components
*/

void Calculate_local_pressure_orthCell(SPARC_OBJ *pSPARC) {
	if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    
    int ityp, iat, i, j, k, p, ip, jp, kp, ip2, jp2, kp2, di, dj, dk, i_DM, j_DM, k_DM, FDn, count, count_interp,
        len_interp, Nx, Ny, Nz, DMnx, DMny, DMnz, DMnd, nx, ny, nz, nd, nxp, nyp, nzp, nd_ex, nx2p, ny2p, nz2p, nd_2ex, 
        icor, jcor, kcor, *Ind_sort, atom_index, indx_ex, indx_2ex, indx_DM, *pshifty, *pshiftz, *pshifty_ex, *pshiftz_ex, *ind_interp;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x2, y2, z2, *R, *R_sort,
           *VJ, *VJ_sort, *VJ_ref, *psdChrgDens_ref, *bJ, *bJ_ref, bJ_val, bJ_ref_val, 
           DbJ_x_val, DbJ_ref_x_val, DbJ_y_val, DbJ_ref_y_val, DbJ_z_val, DbJ_ref_z_val, 
           DVJ_x_val, DVJ_y_val, DVJ_z_val, DVJ_ref_x_val, DVJ_ref_y_val, DVJ_ref_z_val,
           *R_interp, *VJ_interp;
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    double temp1 = 0.0, temp2 = 0.0, temp3;
    double pressure_local = 0.0, pressure_corr = 0.0;
    double xRi, yRi, zRi;
    
    //double rr;
    
    
    double rb;
    int count_rb, count_rb2, *Xindx, *Yindx, *Zindx, *Xindx2, *Yindx2, *Zindx2, fx, fy, fz;
    
    
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (!rank) printf("Start calculating local components of pressure ...\n");
#endif
    ////////////////////////////
    double t1, t2, t_sort = 0.0;
    ////////////////////////////
    
    FDn = pSPARC->order/2;
    w2_diag = pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0] + pSPARC->D2_stencil_coeffs_z[0];
 
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;
    DMnx = pSPARC->Nx_d; DMny = pSPARC->Ny_d; DMnz = pSPARC->Nz_d;
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
    	temp1 += Dphi_x[i] * Dphi_x[i] + Dphi_y[i] * Dphi_y[i] + Dphi_z[i] * Dphi_z[i];
    	temp2 += (pSPARC->electronDens[i] + 3 * pSPARC->psdChrgDens[i]) * pSPARC->elecstPotential[i];
    }
    pressure_local += inv_4PI * temp1 + 0.5 * temp2;
    
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        rb = max(pSPARC->CUTOFF_x[ityp], max(pSPARC->CUTOFF_y[ityp], pSPARC->CUTOFF_z[ityp]));
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];
            
            // original atom index this image atom corresponds to
            atom_index = pSPARC->Atom_Influence_local[ityp].atom_index[iat];
            
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
            Xindx = (int*) malloc(nd_2ex * sizeof(int));
            Yindx = (int*) malloc(nd_2ex * sizeof(int));
            Zindx = (int*) malloc(nd_2ex * sizeof(int));
            Xindx2 = (int*) malloc(nd_2ex * sizeof(int));
            Yindx2 = (int*) malloc(nd_2ex * sizeof(int));
            Zindx2 = (int*) malloc(nd_2ex * sizeof(int));
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
            count = 0; count_interp = 0, count_rb = 0;
            for (k = 0; k < nz2p; k++) {
                z2 = k * pSPARC->delta_z - z0_i_shift; 
                z2 *= z2;
                for (j = 0; j < ny2p; j++) {
                    y2 = j * pSPARC->delta_y - y0_i_shift; 
                    y2 *= y2;
                    for (i = 0; i < nx2p; i++) {
                        x2 = i * pSPARC->delta_x - x0_i_shift; 
                        x2 *= x2;
                        R[count] = sqrt(x2 + y2 + z2);
                        if (R[count] <= rchrg) 
                            count_interp++;
                        if(R[count] <= rb && k >=FDn && k < nz2p-FDn && j >=FDn && j < ny2p-FDn && i >=FDn && i < nx2p-FDn){
                            Xindx[count_rb] = i;
                            Yindx[count_rb] = j;
                            Zindx[count_rb] = k;
                            count_rb++;
                        }                
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
            bJ = (double *)calloc( nd_ex, sizeof(double) );
            bJ_ref = (double *)calloc( nd_ex, sizeof(double) );
            if (bJ == NULL || bJ_ref == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            
            
            count_rb2 = 0;
            for(k = 0; k < count_rb; k++){
                indx_2ex = Zindx[k] * (nx2p * ny2p) + Yindx[k] * nx2p + Xindx[k];
                bJ_val = VJ[indx_2ex] * w2_diag;
                bJ_ref_val = VJ_ref[indx_2ex] * w2_diag;
                for (p = 1; p <= FDn; p++) {
                    bJ_val += ( (VJ[indx_2ex+p] + VJ[indx_2ex-p]) * pSPARC->D2_stencil_coeffs_x[p]
                              + (VJ[indx_2ex+pshifty_ex[p]] + VJ[indx_2ex-pshifty_ex[p]]) * pSPARC->D2_stencil_coeffs_y[p]
                              + (VJ[indx_2ex+pshiftz_ex[p]] + VJ[indx_2ex-pshiftz_ex[p]]) * pSPARC->D2_stencil_coeffs_z[p]);
                    bJ_ref_val += ( (VJ_ref[indx_2ex+p] + VJ_ref[indx_2ex-p]) * pSPARC->D2_stencil_coeffs_x[p]
                                  + (VJ_ref[indx_2ex+pshifty_ex[p]] + VJ_ref[indx_2ex-pshifty_ex[p]]) * pSPARC->D2_stencil_coeffs_y[p]
                                  + (VJ_ref[indx_2ex+pshiftz_ex[p]] + VJ_ref[indx_2ex-pshiftz_ex[p]]) * pSPARC->D2_stencil_coeffs_z[p]);
                }
                
                fz = Zindx[k] - FDn;
                fy = Yindx[k] - FDn;
                fx = Xindx[k] - FDn;
                
                if(fz >=FDn && fz < nzp-FDn && fy >=FDn && fy < nyp-FDn && fx >=FDn && fx < nxp-FDn){
                    Xindx2[count_rb2] = fx;
                    Yindx2[count_rb2] = fy;
                    Zindx2[count_rb2] = fz;
                    Xindx[count_rb2] = Xindx[k];
                    Yindx[count_rb2] = Yindx[k];
                    Zindx[count_rb2] = Zindx[k];
                    count_rb2++;
                }
                
                
                indx_ex = (fz) * (nxp * nyp) + (fy) * nxp + (fx);
                
                
                bJ[indx_ex] = -(inv_4PI * bJ_val);
                bJ_ref[indx_ex] = -(inv_4PI * bJ_ref_val);
            }
            
            
            
            
            
            
            
            
            /*x0_i_shift =  x0_i - pSPARC->delta_x * (icor + FDn); 
            y0_i_shift =  y0_i - pSPARC->delta_y * (jcor + FDn);
            z0_i_shift =  z0_i - pSPARC->delta_z * (kcor + FDn);
            */
            
            /*rr = sqrt(pow(i*pSPARC->delta_x-x0_i_shift,2) + pow(j*pSPARC->delta_y-y0_i_shift,2) + pow(k*pSPARC->delta_z-z0_i_shift,2));
                        if(rr >= pSPARC->CUTOFF_x[ityp]){ 
                            printf("rr %.2f, bJ_val %.15f, bJ_ref_val %.15f \n",rr,bJ_val,bJ_ref_val);
                        }*/   
            
            
            
            /*for (k = 0, kp = FDn; k < nzp; k++, kp++) {
                for (j = 0, jp = FDn; j < nyp; j++, jp++) {
                    for (i = 0, ip = FDn; i < nxp; i++, ip++) {
	                    indx_2ex = kp * (nx2p * ny2p) + jp * nx2p + ip;
                        bJ_val = VJ[indx_2ex] * w2_diag;
                        bJ_ref_val = VJ_ref[indx_2ex] * w2_diag;
                        for (p = 1; p <= FDn; p++) {
                            bJ_val += ( (VJ[indx_2ex+p] + VJ[indx_2ex-p]) * pSPARC->D2_stencil_coeffs_x[p]
                                      + (VJ[indx_2ex+pshifty_ex[p]] + VJ[indx_2ex-pshifty_ex[p]]) * pSPARC->D2_stencil_coeffs_y[p]
                                      + (VJ[indx_2ex+pshiftz_ex[p]] + VJ[indx_2ex-pshiftz_ex[p]]) * pSPARC->D2_stencil_coeffs_z[p]);
                            bJ_ref_val += ( (VJ_ref[indx_2ex+p] + VJ_ref[indx_2ex-p]) * pSPARC->D2_stencil_coeffs_x[p]
                                          + (VJ_ref[indx_2ex+pshifty_ex[p]] + VJ_ref[indx_2ex-pshifty_ex[p]]) * pSPARC->D2_stencil_coeffs_y[p]
                                          + (VJ_ref[indx_2ex+pshiftz_ex[p]] + VJ_ref[indx_2ex-pshiftz_ex[p]]) * pSPARC->D2_stencil_coeffs_z[p]);
                        }
                        
                         
                        
                        indx_ex = k * (nxp * nyp) + j * nxp + i;
                        bJ[indx_ex] = -(inv_4PI * bJ_val);
                        bJ_ref[indx_ex] = -(inv_4PI * bJ_ref_val);
                    }
                }
            }*/
            
            // calculate gradient of bJ, bJ_ref, VJ, VJ_ref in the rb-domain
            dk = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dj = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            di = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];
            
            
            
            
            
            
            for(k = 0; k < count_rb2; k++) {
                DbJ_x_val = DbJ_y_val = DbJ_z_val = 0.0;
                DbJ_ref_x_val = DbJ_ref_y_val = DbJ_ref_z_val = 0.0;
                DVJ_x_val = DVJ_y_val = DVJ_z_val = 0.0;
                DVJ_ref_x_val = DVJ_ref_y_val = DVJ_ref_z_val = 0.0;
                indx_ex = Zindx2[k] * (nxp * nyp) + Yindx2[k] * nxp + Xindx2[k];
                indx_2ex = Zindx[k] * (nx2p * ny2p) + Yindx[k] * nx2p + Xindx[k];
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
                i_DM = di + (Xindx2[k]-FDn);
                j_DM = dj + (Yindx2[k]-FDn);
                k_DM = dk + (Zindx2[k]-FDn);
                indx_DM = k_DM * DMnx * DMny + j_DM * DMnx + i_DM;
                
                xRi = (i_DM + pSPARC->DMVertices[0]) * pSPARC->delta_x - x0_i;
                yRi = (j_DM + pSPARC->DMVertices[2]) * pSPARC->delta_y - y0_i;
                zRi = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;

                pressure_local += (DbJ_x_val * xRi + DbJ_y_val * yRi + DbJ_z_val * zRi) * pSPARC->elecstPotential[indx_DM]; //- 0.5 * VJ[indx_2ex]) ;
                                   //-(DVJ_x_val * xRi + DVJ_y_val * yRi + DVJ_z_val * zRi) * 0.5 * bJ[indx_ex];

                temp1 = pSPARC->Vc[indx_DM] - VJ_ref[indx_2ex];
                temp2 = pSPARC->Vc[indx_DM] ;  //+ VJ[indx_2ex];
                temp3 = pSPARC->psdChrgDens[indx_DM] + pSPARC->psdChrgDens_ref[indx_DM];
                
                pressure_corr += ( DbJ_ref_x_val*temp1 + DbJ_x_val*temp2 + (DVJ_ref_x_val-DVJ_x_val)*temp3 - DVJ_ref_x_val*bJ_ref[indx_ex] ) * xRi; 
                                  // + DVJ_x_val * bJ[indx_ex]
                pressure_corr += ( DbJ_ref_y_val*temp1 + DbJ_y_val*temp2 + (DVJ_ref_y_val-DVJ_y_val)*temp3 - DVJ_ref_y_val*bJ_ref[indx_ex] ) * yRi; 
                                  // + DVJ_y_val * bJ[indx_ex]
               	pressure_corr += ( DbJ_ref_z_val*temp1 + DbJ_z_val*temp2 + (DVJ_ref_z_val-DVJ_z_val)*temp3 - DVJ_ref_z_val*bJ_ref[indx_ex] ) * zRi; 
               	                  //  + DVJ_z_val * bJ[indx_ex]    
                 
            }
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            /*for(kp = FDn, kp2 = pSPARC->order, k_DM = dk; kp2 < nzp; kp++, kp2++, k_DM++) {
                for(jp = FDn, jp2 = pSPARC->order, j_DM = dj; jp2 < nyp; jp++, jp2++, j_DM++) {
                    for(ip = FDn, ip2 = pSPARC->order, i_DM = di; ip2 < nxp; ip++, ip2++, i_DM++) {
                        DbJ_x_val = DbJ_y_val = DbJ_z_val = 0.0;
                        DbJ_ref_x_val = DbJ_ref_y_val = DbJ_ref_z_val = 0.0;
                        DVJ_x_val = DVJ_y_val = DVJ_z_val = 0.0;
                        DVJ_ref_x_val = DVJ_ref_y_val = DVJ_ref_z_val = 0.0;

                        indx_ex = kp * (nxp * nyp) + jp * nxp + ip;
                        indx_2ex = kp2 * (nx2p * ny2p) + jp2 * nx2p + ip2;
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
                        indx_DM = k_DM * DMnx * DMny + j_DM * DMnx + i_DM;
                        
                        xRi = (i_DM + pSPARC->DMVertices[0]) * pSPARC->delta_x - x0_i;
                        yRi = (j_DM + pSPARC->DMVertices[2]) * pSPARC->delta_y - y0_i;
                        zRi = (k_DM + pSPARC->DMVertices[4]) * pSPARC->delta_z - z0_i;

                        pressure_local += (DbJ_x_val * xRi + DbJ_y_val * yRi + DbJ_z_val * zRi) * pSPARC->elecstPotential[indx_DM]; //- 0.5 * VJ[indx_2ex]) ;
                                           //-(DVJ_x_val * xRi + DVJ_y_val * yRi + DVJ_z_val * zRi) * 0.5 * bJ[indx_ex];

                        temp1 = pSPARC->Vc[indx_DM] - VJ_ref[indx_2ex];
                        temp2 = pSPARC->Vc[indx_DM] ;  //+ VJ[indx_2ex];
                        temp3 = pSPARC->psdChrgDens[indx_DM] + pSPARC->psdChrgDens_ref[indx_DM];
                        
                        pressure_corr += ( DbJ_ref_x_val*temp1 + DbJ_x_val*temp2 + (DVJ_ref_x_val-DVJ_x_val)*temp3 - DVJ_ref_x_val*bJ_ref[indx_ex] ) * xRi; 
                                          // + DVJ_x_val * bJ[indx_ex]
                        pressure_corr += ( DbJ_ref_y_val*temp1 + DbJ_y_val*temp2 + (DVJ_ref_y_val-DVJ_y_val)*temp3 - DVJ_ref_y_val*bJ_ref[indx_ex] ) * yRi; 
                                          // + DVJ_y_val * bJ[indx_ex]
                       	pressure_corr += ( DbJ_ref_z_val*temp1 + DbJ_z_val*temp2 + (DVJ_ref_z_val-DVJ_z_val)*temp3 - DVJ_ref_z_val*bJ_ref[indx_ex] ) * zRi; 
                       	                  //  + DVJ_z_val * bJ[indx_ex]    
                    }
                }
            }*/

            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
            free(Xindx);free(Yindx);free(Zindx);
            free(Xindx2);free(Yindx2);free(Zindx2);
        }   
    }
    
    pSPARC->pressure = (pressure_local + 0.5 * pressure_corr ) * pSPARC->dV ;

    free(Dphi_x);
    free(Dphi_y);
    free(Dphi_z);
    
    t1 = MPI_Wtime();
    // do Allreduce/Reduce to find total integral // TODO: check if there's only 1 process, then skip this
    MPI_Allreduce(MPI_IN_PLACE, &pSPARC->pressure, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank){ 
        printf("time for sorting and interpolate pseudopotential: %.3f ms, time for Allreduce/Reduce: %.3f ms \n", t_sort*1e3, (t2-t1)*1e3);
        printf("Local pressure: %.15f Ha/Bohr^3\n",pSPARC->pressure);
    }    
#endif
    
    //deallocate
    free(pshifty); free(pshiftz);
    free(pshifty_ex); free(pshiftz_ex);

}


/**
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure(SPARC_OBJ *pSPARC)
{
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i, j, k, n, np, ldispl, ndc, ityp, iat, ncol, DMnd, DMnx, DMny, DMnz, indx, i_DM, j_DM, k_DM, dim, atom_index, count, l, m, lmax;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnx = pSPARC->Nx_d_dmcomm;
    DMny = pSPARC->Ny_d_dmcomm;
    DMnz = pSPARC->Nz_d_dmcomm;
    
    double pressure_nloc = 0.0, *alpha, *alpha2, *beta, *x_ptr, *dx_ptr, *x_rc, *dx_rc, *x_rc_ptr, *dx_rc_ptr;
    double pJ, val, val_x, val_y, val_z, val2, val2_x, val2_y, val2_z, g_nk, *beta_x, *beta_y,
           *beta_z, *alpha_J, *beta_Jx, *beta_Jy, *beta_Jz;
    
    alpha = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol * 4, sizeof(double));
    //alpha2 = (double *)calloc( pSPARC->IP_displ[pSPARC->n_atom] * ncol, sizeof(double));
#ifdef DEBUG 
    if (!rank) printf("Start Calculating nonlocal pressure\n");
#endif
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        //lmax = pSPARC->psd[ityp].lmax;
        if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
            ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat];
            x_rc = (double *)malloc( ndc * ncol * sizeof(double));
            atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
            
            /* first find inner product <Psi_n, Chi_Jlm>, and <Chi_Jlm, Psi_n> */
            for (n = 0; n < ncol; n++) {
                x_ptr = pSPARC->Xorb + n * DMnd;
                x_rc_ptr = x_rc + n * ndc;
                for (i = 0; i < ndc; i++) {
                    // x_rc[n*ndc+i] = pSPARC->Xorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                    *(x_rc_ptr + i) = *(x_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]);
                }
            }
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, pSPARC->nlocProj[ityp].nproj, ncol, ndc, pSPARC->dV, pSPARC->nlocProj[ityp].Chi[iat], ndc, 
                        x_rc, ndc, 1.0, alpha+pSPARC->IP_displ[atom_index]*ncol, pSPARC->nlocProj[ityp].nproj); // multiply dV to get inner-product
            //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ncol, pSPARC->nlocProj[ityp].nproj, ndc, 1.0, x_rc, ndc, 
            //            pSPARC->nlocProj[ityp].Chi[iat], ndc, 1.0, alpha2+pSPARC->IP_displ[atom_index]*ncol, ncol); // this calculates <Psi_n, Chi_Jlm>
            free(x_rc);
            
        }
    }
    
    /* find inner product <Chi_Jlm, dPsi_n.(x-R_J)> */
    for (dim = 0; dim < 3; dim++) {
        // find dPsi in direction dim
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb, pSPARC->Yorb, dim, pSPARC->dmcomm);
        beta = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol*(dim+1);
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            //lmax = pSPARC->psd[ityp].lmax;
            if (! pSPARC->nlocProj[ityp].nproj) continue; // this is typical for hydrogen
            for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                ndc = pSPARC->Atom_Influence_nloc[ityp].ndc[iat]; 
                dx_rc = (double *)malloc( ndc * ncol * sizeof(double));
                atom_index = pSPARC->Atom_Influence_nloc[ityp].atom_index[iat];
                for (n = 0; n < ncol; n++) {
                    dx_ptr = pSPARC->Yorb + n * DMnd;
                    dx_rc_ptr = dx_rc + n * ndc;
                    for (i = 0; i < ndc; i++) {
                        // dx_rc[n*ndc+i] = pSPARC->Yorb[n*DMnd+pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i]];
                        if (dim == 0){
                        	indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                		    i_DM = indx % DMnx;
                        	*(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i])
                        					 * ((i_DM + pSPARC->DMVertices_dmcomm[0]) * pSPARC->delta_x - pSPARC->Atom_Influence_nloc[ityp].coords[iat*3]);
                        } else if(dim == 1){
                        	indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                			k_DM = indx / (DMnx * DMny);
                			j_DM = (indx - k_DM * (DMnx * DMny)) / DMnx;
          
                        	*(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i])
                        					 * ((j_DM + pSPARC->DMVertices_dmcomm[2]) * pSPARC->delta_y - pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+1]);
                        } else {
                        	indx = pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i];
                			k_DM = indx / (DMnx * DMny);
                			*(dx_rc_ptr + i) = *(dx_ptr + pSPARC->Atom_Influence_nloc[ityp].grid_pos[iat][i])
                        					 * ((k_DM + pSPARC->DMVertices_dmcomm[4]) * pSPARC->delta_z - pSPARC->Atom_Influence_nloc[ityp].coords[iat*3+2]);
                        
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
    }

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, alpha, pSPARC->IP_displ[pSPARC->n_atom] * ncol * 4, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
        //MPI_Allreduce(MPI_IN_PLACE, alpha2, pSPARC->IP_displ[pSPARC->n_atom] * ncol, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    /* calculate nonlocal pressure */
    // go over all atoms and find nonlocal pressure
    beta_x = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol;
    beta_y = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol * 2;
    beta_z = alpha + pSPARC->IP_displ[pSPARC->n_atom]*ncol * 3;
    count = 0; atom_index = 0;
    
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        lmax = pSPARC->psd[ityp].lmax;
        for (iat = 0; iat < pSPARC->nAtomv[ityp]; iat++) {
            for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
            	pJ = 0.0;
                for (n = pSPARC->band_start_indx; n <= pSPARC->band_end_indx; n++) {
                    g_nk = pSPARC->occ[n]; // TODO: for k-points calculation, use occ[n+k*Ns]
                    val2 = val2_x = val2_y = val2_z = 0.0;
                    ldispl = 0;
                    for (l = 0; l <= lmax; l++) {
                        // skip the local l
                        if (l == pSPARC->localPsd[ityp]) {
                            ldispl += pSPARC->psd[ityp].ppl[l];
                            continue;
                        }
                        for (np = 0; np < pSPARC->psd[ityp].ppl[l]; np++) {
                            val = val_x = val_y = val_z = 0.0;
                            for (m = -l; m <= l; m++) {
                                val += alpha[count] * alpha[count]/pSPARC->dV;
                                val_x += alpha[count] * beta_x[count];
                                val_y += alpha[count] * beta_y[count];
                                val_z += alpha[count] * beta_z[count];
                                count++;
                            }
                            val2 += val * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_x += val_x * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_y += val_y * pSPARC->psd[ityp].Gamma[ldispl+np];
                            val2_z += val_z * pSPARC->psd[ityp].Gamma[ldispl+np];
                        }
                        ldispl += pSPARC->psd[ityp].ppl[l];
                    }
                    pJ += (0.5 * val2 + val2_x + val2_y + val2_z) * g_nk;
                }
                
                if (pSPARC->Nkpts > 1) {
                    pressure_nloc -= 4.0 * pSPARC->kptWts_loc[k] / pSPARC->Nkpts * pJ;
                } else {
                    pressure_nloc -= 4.0 * pJ;
                }
                atom_index++;
            }
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
    
#ifdef DEBUG    
    if (!rank){
        printf("pressure_nloc = %.15f Ha/Bohr^3\n", pressure_nloc);
    }    
#endif
    
    if (!rank) {
        pSPARC->pressure += pressure_nloc;
    }
    
    free(alpha);
    //free(alpha2);
}
