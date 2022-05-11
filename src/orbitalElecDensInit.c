/**
 * @file    orbitalElecDensInit.c
 * @brief   This file contains the functions for electron density initialization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
/* LAPACK, LAPACKE routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <lapacke.h>
#endif

#include "orbitalElecDensInit.h"
#include "isddft.h"
#include "tools.h"

#define max(x,y) ((x)>(y)?(x):(y))


/**
 * @brief   Initialze electron density.
 */
void Init_electronDensity(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Initializing electron density ... \n");
#endif
    
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        int  i, DMnd;
        DMnd = pSPARC->Nd_d * (2*pSPARC->Nspin - 1);
        // for 1st Relax step/ MDstep, set electron density to be sum of atomic potentials
        if( (pSPARC->elecgs_Count - pSPARC->StressCount) == 0){
            // TODO: implement restart based on previous MD electron density. Things to consider:
            //if (pSPARC->RestartFlag) {
                // 1) Each processor stores the density in its memory in a separate file at the end of MD (same frequency as the main restart file).
                // 2) After all processors have printed their density, a counter in main restart file will be updated.
                // 3) If the counter says "success" then use the density from previous step as guess otherwise start from a guess based on electronDens_at.
                // 4) Change (pSPARC->elecgs_Count + !pSPARC->RestartFlag) > 3  condition to pSPARC->elecgs_Count >= 3, below
                //printf("\n\n Implement density extrapolation when restart flag is on!! \n\n");
            //} else {
            for (i = 0; i < DMnd; i++)
                pSPARC->electronDens[i] = pSPARC->electronDens_at[i];   
            //}
            // Storing atom position needed for charge extrapolation in future Relax/MD steps
			if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1){
            	for(i = 0; i < 3 * pSPARC->n_atom; i++)
        			pSPARC->atom_pos_0dt[i] = pSPARC->atom_pos[i];
			}
        } else {
            if( (pSPARC->elecgs_Count - pSPARC->StressCount) >= 3 && (pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1)){
#ifdef DEBUG
        		if(!rank)
        		    printf("Using charge extrapolation for density guess\n");
#endif
        		// Test if needed by using unscaled atomic charge density 
                // double scal_fac = (pSPARC->NetCharge - pSPARC->PosCharge) / pSPARC->NegCharge;    
        		// for(i = 0; i < DMnd; i++)
            	//	pSPARC->electronDens_at[i] /= scal_fac;

                // Perform charge extrapolation using scaled rho_at     
       			for(i = 0; i < pSPARC->Nd_d; i++){
            		pSPARC->electronDens[i] = pSPARC->electronDens_at[i] + pSPARC->delectronDens[i]; // extrapolated density for the next step
               		if(pSPARC->electronDens[i] < 0.0)
                        pSPARC->electronDens[i] = pSPARC->xc_rhotol; // 1e-14
        		}

                double rho_mag;
                if(pSPARC->spin_typ != 0){
                    for(i = 0; i < pSPARC->Nd_d; i++){
                        rho_mag = pSPARC->electronDens[pSPARC->Nd_d+i] - pSPARC->electronDens[2*pSPARC->Nd_d+i]; // from previous step
                        pSPARC->electronDens[pSPARC->Nd_d+i] = (pSPARC->electronDens[i] + rho_mag)/2.0;
                        pSPARC->electronDens[2*pSPARC->Nd_d+i] = (pSPARC->electronDens[i] - rho_mag)/2.0;
                    }
                }
        	}
            
            // Scale density
			double int_rho = 0.0, vscal;        
        	for (i = 0; i < pSPARC->Nd_d; i++) {
           		int_rho += pSPARC->electronDens[i];
           	}
            int_rho *= pSPARC->dV;
            MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi); 	
            vscal = pSPARC->PosCharge / int_rho;
            for (i = 0; i < DMnd; i++)
            	pSPARC->electronDens[i] *= vscal;
		}
	}
}

/*
 * @brief function to perform charge extrapolation to provide better rho_guess for future relax/MD steps
 * @ref   Ab initio molecular dynamics, a simple algorithm for charge extrapolation
 */
// TODO: Check if the FtF matrix is singular and if it is then decide on how to do extrapolation or not to do at all
void elecDensExtrapolation(SPARC_OBJ *pSPARC) {
	// processors that are not in the dmcomm_phi will remain idle
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    int nd, ii, matrank, count = 0, atm;
    double alpha, beta;
    for (nd = 0; nd < pSPARC->Nd_d; nd++){
        pSPARC->delectronDens_2dt[nd] = pSPARC->delectronDens_1dt[nd];
        pSPARC->delectronDens_1dt[nd] = pSPARC->delectronDens_0dt[nd];
        pSPARC->delectronDens_0dt[nd] = pSPARC->electronDens[nd] - pSPARC->electronDens_at[nd];
    }
    if(pSPARC->MDFlag == 1){
	    for(atm = 0; atm < pSPARC->n_atom; atm++){
	    	if(pSPARC->MDCount == 1){
	    		pSPARC->atom_pos_nm[count * 3] = pSPARC->atom_pos[count * 3];
		    	pSPARC->atom_pos_nm[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1];
			    pSPARC->atom_pos_nm[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2];
		    	count ++;
		    } else{
			    pSPARC->atom_pos_nm[count * 3] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3];
			    pSPARC->atom_pos_nm[count * 3 + 1] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3 + 1];
			    pSPARC->atom_pos_nm[count * 3 + 2] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3 + 2];
			    count ++;
		    }         
	    }
    } else if(pSPARC->RelaxFlag == 1){
        for(atm = 0; atm < pSPARC->n_atom; atm++){
		    if((pSPARC->elecgs_Count - pSPARC->StressCount) == 1){
			    pSPARC->atom_pos_nm[count * 3] = pSPARC->atom_pos[count * 3];
			    pSPARC->atom_pos_nm[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1];
			    pSPARC->atom_pos_nm[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2];
                count ++;
		    } else{
			    pSPARC->atom_pos_nm[count * 3] += pSPARC->Relax_fac * pSPARC->d[count * 3] * pSPARC->mvAtmConstraint[count * 3];
			    pSPARC->atom_pos_nm[count * 3 + 1] += pSPARC->Relax_fac * pSPARC->d[count * 3 + 1] * pSPARC->mvAtmConstraint[count * 3 + 1];
			    pSPARC->atom_pos_nm[count * 3 + 2] += pSPARC->Relax_fac * pSPARC->d[count * 3 + 2] * pSPARC->mvAtmConstraint[count * 3 + 2];
			    count ++;
		    }
	    }
    }
	if((pSPARC->elecgs_Count - pSPARC->StressCount) >= 3){ 
       double *FtF, *Ftf, *s, temp1, temp2, temp3; // 2x2 matrix and 2x1 vectors in (FtF)*(svec) = Ftf
        FtF = (double *)calloc( 4 , sizeof(double) );
        Ftf = (double *)calloc( 2 , sizeof(double) );
        s = (double *)calloc( 2 , sizeof(double) );
        for(ii = 0; ii < 3 * pSPARC->n_atom; ii++){
            temp1 = pSPARC->atom_pos_0dt[ii] - pSPARC->atom_pos_1dt[ii];
            temp2 = pSPARC->atom_pos_1dt[ii] - pSPARC->atom_pos_2dt[ii];
            temp3 = pSPARC->atom_pos_nm[ii] - pSPARC->atom_pos_0dt[ii];
            FtF[0] += temp1*temp1;
            FtF[1] += temp1*temp2;
            FtF[3] += temp2*temp2;
            Ftf[0] += temp1*temp3;
            Ftf[1] += temp2*temp3;
        }
        FtF[2] = FtF[1];
        // Find inv(FtF)*Ftf, LAPACKE_dgelsd stores the answer in Ftf vector
        LAPACKE_dgelsd(LAPACK_COL_MAJOR, 2, 2, 1, FtF, 2, Ftf, 2, s, -1.0, &matrank);
        alpha = Ftf[0];
        beta = Ftf[1];
        // Extrapolation 
        for (nd = 0; nd < pSPARC->Nd_d; nd++)
            pSPARC->delectronDens[nd] = (1 + alpha) * pSPARC->delectronDens_0dt[nd] + (beta - alpha) * pSPARC->delectronDens_1dt[nd] - beta * pSPARC->delectronDens_2dt[nd];
        free(FtF);
        free(Ftf);
        free(s);            
    }
    for(ii = 0; ii < 3 * pSPARC->n_atom; ii++){
        pSPARC->atom_pos_2dt[ii] = pSPARC->atom_pos_1dt[ii];
        pSPARC->atom_pos_1dt[ii] = pSPARC->atom_pos_0dt[ii];
        pSPARC->atom_pos_0dt[ii] = pSPARC->atom_pos_nm[ii];
    }
}


/**
 * @brief   initialize Kohn-Sham orbitals.
 */
void Init_orbital(SPARC_OBJ *pSPARC)
{
    if (pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Initializing Kohn-Sham orbitals ... \n");
#endif

    int k, n, i, DMnd, size_k, len_tot, size_s, spn_i, spinor;
#ifdef DEBUG
    double t1, t2;
#endif
    // Multiply a factor for a spinor wavefunction
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    size_k = DMnd * pSPARC->Nband_bandcomm;
    // notice that in processors not for orbital calculations len_tot = 0
    size_s = size_k * pSPARC->Nkpts_kptcomm;
    
    // Multiply a factor for a spin polarized calculation
    len_tot = size_s * pSPARC->Nspin_spincomm;
    
    // for 1st Relax step, set electron density to be sum of atomic potentials
    if((pSPARC->elecgs_Count) == 0){
        if (pSPARC->isGammaPoint){
            // allocate memory in the very first relax/MD step
            pSPARC->Xorb = (double *)malloc( len_tot * sizeof(double) );
            pSPARC->Yorb = (double *)malloc( size_k * sizeof(double) );
            if (pSPARC->Xorb == NULL || pSPARC->Yorb == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            }

            // set random initial orbitals
            // notes: 1. process not in dmcomm will have 0 row of bands, hence no orbitals assigned
            //        2. Xorb in different kptcomms will have the same random matrix if the comm sizes are identical
            //        3. we're also forcing all kpoints to have the same initial orbitals
#ifdef DEBUG            
            t1 = MPI_Wtime();
#endif
            if (pSPARC->FixRandSeed == 1) {
                int gridsizes[3];
                gridsizes[0] = pSPARC->Nx;
                gridsizes[1] = pSPARC->Ny;
                gridsizes[2] = pSPARC->Nz;
                //int size_kg = pSPARC->Nd * pSPARC->Nstates;
                int size_sg = pSPARC->Nd * pSPARC->Nstates;
                for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    //int kg  = pSPARC->kpt_start_indx; // global kpt index
                    int sg  = pSPARC->spin_start_indx + spn_i; // global spin index
                    for (n = 0; n < pSPARC->Nband_bandcomm; n++) {
                        int ng = pSPARC->band_start_indx + n; // global band index
                        int shift_g = sg * size_sg + ng * pSPARC->Nd; // global shift
                        int shift   = spn_i * size_s + n  * DMnd; // local shift
                        double *Psi_kn = pSPARC->Xorb + shift;
                        SeededRandVec(Psi_kn, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, shift_g);
                    }
                }
            } else {
                for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    SetRandMat(pSPARC->Xorb + spn_i*size_s , DMnd, pSPARC->Nband_bandcomm, -0.5, 0.5, pSPARC->spincomm);
                }
            }
        } else {
            // allocate memory in the very first relax/MD step
            pSPARC->Xorb_kpt = (double complex *) malloc( len_tot * sizeof(double complex) );
            pSPARC->Yorb_kpt = (double complex *) malloc( size_k * sizeof(double complex) );
            if (pSPARC->Xorb_kpt == NULL || pSPARC->Yorb_kpt == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            }

            // set random initial orbitals
            // notes: 1. process not in dmcomm will have 0 row of bands, hence no orbitals assigned
            //        2. Xorb in different kptcomms will have the same random matrix if the comm sizes are identical
            //        3. we're also forcing all kpoints to have the same initial orbitals
#ifdef DEBUG            
            t1 = MPI_Wtime();
#endif                
            if (pSPARC->FixRandSeed == 1) {
                int gridsizes[3];
                gridsizes[0] = pSPARC->Nx;
                gridsizes[1] = pSPARC->Ny;
                gridsizes[2] = pSPARC->Nz;
                int size_kg = pSPARC->Nd * pSPARC->Nspinor * pSPARC->Nstates;
                int size_sg = size_kg * pSPARC->Nkpts_sym;
                for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    int sg  = pSPARC->spin_start_indx + spn_i; // global spin index
                    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
                        int kg  = pSPARC->kpt_start_indx + k; // global kpt index
                        for (n = 0; n < pSPARC->Nband_bandcomm; n++) {
                            int ng = pSPARC->band_start_indx + n; // global band index
                            int shift_g = sg * size_sg + kg * size_kg + ng * pSPARC->Nd * pSPARC->Nspinor; // global shift
                            int shift   = spn_i * size_s + k  * size_k  + n  * DMnd; // local shift
                            for (spinor = 0; spinor < pSPARC->Nspinor; spinor ++) {
                                double complex *Psi_kn = pSPARC->Xorb_kpt + shift + spinor * pSPARC->Nd_d_dmcomm;
                                SeededRandVec_complex(Psi_kn, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, shift_g);
                            }
                        }
                    }
                }    
            } else {
                for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    for (i = 0; i < pSPARC->Nkpts_kptcomm; i++) {
                        SetRandMat_complex(pSPARC->Xorb_kpt + i*size_k + spn_i*size_s, DMnd, pSPARC->Nband_bandcomm, -0.5, 0.5, pSPARC->spincomm);
                    }
                }    
            }
        }        
#ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank) printf("Finished setting random orbitals. Time taken: %.3f ms\n",(t2-t1)*1e3);
#endif
    } else {
        // TODO: implement Kohn-Sham orbital extrapolation here!
    }
}
