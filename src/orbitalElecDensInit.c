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
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <string.h> 
/* LAPACK, LAPACKE routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <lapacke.h>
#endif

#include "orbitalElecDensInit.h"
#include "isddft.h"
#include "tools.h"
#include "electronDensity.h"
#include "parallelization.h"
#include "readfiles.h"
#include "initialization.h"
#include "md.h"
#define max(x,y) ((x)>(y)?(x):(y))


/**
 * @brief Read vector(s) from Cube file(s) and distribute in comm.
 * 
 * @param pSPARC 
 * @param filenames An array of n file names.
 * @param data_dist The pointer to the final distributed vector.
 * @param n Number of files (also number of columns of data_dist).
 * @param DMverts Domain vertices of the local distributed domain.
 * @param comm Communicator with a Cartesian topology.
 */
void read_cube_and_dist_vec(
    SPARC_OBJ *pSPARC, char filenames[3][L_STRING+L_PSD], double *data_dist, int n,
    int DMverts[6], MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // initialize D2D for distributing the vectors from root to comm
    MPI_Comm send_comm;
    if (rank != 0) {
        send_comm = MPI_COMM_NULL;
    } else {
        int dims[3] = {1,1,1}, periods[3] = {1,1,1};
        // create a cartesian topology on one process (rank 0)
        MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &send_comm);
    }
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int sdims[3], sVert[6], rdims[3], periods[3], coords[3];
    sdims[0] = sdims[1] = sdims[2] = 1;
    sVert[0] = 0; sVert[1] = gridsizes[0] - 1;
    sVert[2] = 0; sVert[3] = gridsizes[1] - 1;
    sVert[4] = 0; sVert[5] = gridsizes[2] - 1;

    MPI_Cart_get(comm, 3, rdims, periods, coords);
    D2D_OBJ d2d_sender, d2d_recvr;
    Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, sVert, DMverts, send_comm, sdims, comm, rdims, comm);

    int DMnx = DMverts[1] - DMverts[0] + 1;
    int DMny = DMverts[3] - DMverts[2] + 1;
    int DMnz = DMverts[5] - DMverts[4] + 1;
    int DMnd = DMnx * DMny * DMnz;

    double *vec_in = NULL;
    // loop over the n filenames
    for (int i = 0; i < n; i++) {
        // let root process read the density from file
        if (rank == 0) {
            vec_in = read_vec_cube(pSPARC, filenames[i]);
        }

        // distribute the density from root to comm
        D2D(&d2d_sender, &d2d_recvr, gridsizes, sVert, vec_in, DMverts, data_dist+i*DMnd, send_comm, sdims, comm, rdims, comm, sizeof(double));

        if (rank == 0) free(vec_in);
    }

    Free_D2D_Target(&d2d_sender, &d2d_recvr, send_comm, comm);

    if (rank == 0) {
        MPI_Comm_free(&send_comm);
    }
}



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
        int DMnd = pSPARC->Nd_d;
        // for 1st Relax step/ MDstep, set initial electron density
        if( (pSPARC->elecgs_Count - pSPARC->StressCount) == 0){
            // read initial density from file 
            if (pSPARC->BandStructFlag == 1 ||  pSPARC->readInitDens) {
                char inputDensFnames[3][L_STRING+L_PSD];
                // set up input density filename
                if (rank == 0) {
                    char INPUT_DIR[L_PSD];
                    extract_path_from_file(pSPARC->filename, INPUT_DIR, L_PSD);
                    combine_path_filename(INPUT_DIR, pSPARC->InDensTCubFilename, inputDensFnames[0], L_STRING+L_PSD);
                    combine_path_filename(INPUT_DIR, pSPARC->InDensUCubFilename, inputDensFnames[1], L_STRING+L_PSD);
                    combine_path_filename(INPUT_DIR, pSPARC->InDensDCubFilename, inputDensFnames[2], L_STRING+L_PSD);
                }

                int nFileToRead = pSPARC->densfilecount;
                read_cube_and_dist_vec(
                    pSPARC, inputDensFnames, pSPARC->electronDens, nFileToRead,
                    pSPARC->DMVertices, pSPARC->dmcomm_phi
                );
            } else {
                // TODO: implement restart based on previous MD electron density. Things to consider:
                // 1) Each processor stores the density in its memory in a separate file at the end of MD (same frequency as the main restart file).
                // 2) After all processors have printed their density, a counter in main restart file will be updated.
                // 3) If the counter says "success" then use the density from previous step as guess otherwise start from a guess based on electronDens_at.
                // 4) Change (pSPARC->elecgs_Count + !pSPARC->RestartFlag) > 3  condition to pSPARC->elecgs_Count >= 3, below
                
                // copy initial electron density
                memcpy(pSPARC->electronDens, pSPARC->electronDens_at, DMnd * sizeof(double));
                if (pSPARC->usefock)
                    memcpy(pSPARC->electronDens_pbe, pSPARC->electronDens_at, DMnd * sizeof(double));
                // get intial magnetization
                if (pSPARC->spin_typ == 1) {
                    memcpy(pSPARC->mag, pSPARC->mag_at, DMnd * sizeof(double));
                    Calculate_diagonal_Density(pSPARC, DMnd, pSPARC->mag, pSPARC->electronDens, pSPARC->electronDens+DMnd, pSPARC->electronDens+2*DMnd);
                } else if (pSPARC->spin_typ == 2) {
                    memcpy(pSPARC->mag+DMnd, pSPARC->mag_at, DMnd * 3 * sizeof(double));
                    Calculate_Magnorm(pSPARC, DMnd, pSPARC->mag+DMnd*1, pSPARC->mag+DMnd*2, pSPARC->mag+DMnd*3, pSPARC->mag);
                    Calculate_diagonal_Density(pSPARC, DMnd, pSPARC->mag, pSPARC->electronDens, pSPARC->electronDens+DMnd, pSPARC->electronDens+2*DMnd);
                }
            }
            
            // Storing atom position needed for charge extrapolation in future Relax/MD steps
			if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1){
            	for(int i = 0; i < 3 * pSPARC->n_atom; i++)
        			pSPARC->atom_pos_0dt[i] = pSPARC->atom_pos[i];
			}
        } else {
            if( (pSPARC->elecgs_Count - pSPARC->StressCount) >= 3 && (pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1)){
            #ifdef DEBUG
        		if(!rank)
        		    printf("Using charge extrapolation for density guess\n");
            #endif

                // Perform charge extrapolation using scaled rho_at     
       			for(int i = 0; i < DMnd; i++){
            		pSPARC->electronDens[i] = pSPARC->electronDens_at[i] + pSPARC->delectronDens[i]; // extrapolated density for the next step
               		if(pSPARC->electronDens[i] < 0.0)
                        pSPARC->electronDens[i] = pSPARC->xc_rhotol; // 1e-14
        		}
        	}
            
            // Scale density
			double int_rho = 0.0, vscal;        
        	if (pSPARC->CyclixFlag) {
                for (int i = 0; i < DMnd; i++) {
                    int_rho += pSPARC->electronDens[i] * pSPARC->Intgwt_phi[i];
                }
            } else {
                for (int i = 0; i < DMnd; i++) {
                    int_rho += pSPARC->electronDens[i];
                }
                int_rho *= pSPARC->dV;
            }
            MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi); 	
            vscal = pSPARC->PosCharge / int_rho;
            for (int i = 0; i < DMnd; i++)
            	pSPARC->electronDens[i] *= vscal;
            
            if (pSPARC->spin_typ != 0) {
                for(int i = 0; i < DMnd; i++){
                    double rho_mag = pSPARC->mag[i];
                    pSPARC->electronDens[DMnd+i] = (pSPARC->electronDens[i] + rho_mag)/2.0;
                    pSPARC->electronDens[2*DMnd+i] = (pSPARC->electronDens[i] - rho_mag)/2.0;
                }
            }
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
    double alpha, beta, *coord_temp1, *coord_temp2;
    for (nd = 0; nd < pSPARC->Nd_d; nd++){
        pSPARC->delectronDens_2dt[nd] = pSPARC->delectronDens_1dt[nd];
        pSPARC->delectronDens_1dt[nd] = pSPARC->delectronDens_0dt[nd];
        double *electronDens = pSPARC->usefock%2 ? pSPARC->electronDens_pbe : pSPARC->electronDens;        
        pSPARC->delectronDens_0dt[nd] = electronDens[nd] - pSPARC->electronDens_at[nd];
    }
    if(pSPARC->MDFlag == 1){
        if(pSPARC->MDCount == 1) {
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm[count * 3] = pSPARC->atom_pos[count * 3];
                pSPARC->atom_pos_nm[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1];
                pSPARC->atom_pos_nm[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2];
                count ++;
            }   
        } else{
            coord_temp1 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));
            coord_temp2 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));

            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                coord_temp1[atm] = pSPARC->atom_pos_nm[atm];
                coord_temp2[atm] = pSPARC->atom_pos[atm];
            }
            wraparound_dynamics(pSPARC, coord_temp1, 0);
            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    Cart2nonCart_coord(pSPARC, coord_temp2+3*atm, coord_temp2+3*atm+1, coord_temp2+3*atm+2);
                    Cart2nonCart_coord(pSPARC, pSPARC->atom_pos_nm+3*atm, pSPARC->atom_pos_nm+3*atm+1, pSPARC->atom_pos_nm+3*atm+2);
                }
            }
            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm[atm] += coord_temp2[atm] - coord_temp1[atm];
            }

            

            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    nonCart2Cart_coord(pSPARC, pSPARC->atom_pos_nm+3*atm, pSPARC->atom_pos_nm+3*atm+1, pSPARC->atom_pos_nm+3*atm+2);
                }
            }

            free(coord_temp1);
            free(coord_temp2);
        }   
    } else if(pSPARC->RelaxFlag == 1){
        if((pSPARC->elecgs_Count - pSPARC->StressCount) == 1) {
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm[count * 3] = pSPARC->atom_pos[count * 3];
                pSPARC->atom_pos_nm[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1];
                pSPARC->atom_pos_nm[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2];
                count ++;
            }
        } else{
            coord_temp1 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));
            coord_temp2 = (double *) malloc(3*pSPARC->n_atom*sizeof(double));

            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                coord_temp1[atm] = pSPARC->atom_pos_nm[atm];
                coord_temp2[atm] = pSPARC->atom_pos[atm];
            }
            wraparound_dynamics(pSPARC, coord_temp1, 0);
            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    Cart2nonCart_coord(pSPARC, coord_temp2+3*atm, coord_temp2+3*atm+1, coord_temp2+3*atm+2);
                    Cart2nonCart_coord(pSPARC, pSPARC->atom_pos_nm+3*atm, pSPARC->atom_pos_nm+3*atm+1, pSPARC->atom_pos_nm+3*atm+2);
                }
            }
            for(atm = 0; atm < 3*pSPARC->n_atom; atm++){
                pSPARC->atom_pos_nm[atm] += coord_temp2[atm] - coord_temp1[atm];
            }

            if(pSPARC->cell_typ != 0){
                for(atm = 0; atm < pSPARC->n_atom; atm++)
                    nonCart2Cart_coord(pSPARC, pSPARC->atom_pos_nm+3*atm, pSPARC->atom_pos_nm+3*atm+1, pSPARC->atom_pos_nm+3*atm+2);
            }

            free(coord_temp1);
            free(coord_temp2);
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

    int k, n, DMnd, DMndsp, size_k, len_tot, spinor;
#ifdef DEBUG
    double t1, t2;
#endif
    // Multiply a factor for a spinor wavefunction
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    size_k = DMndsp * pSPARC->Nband_bandcomm;
    // notice that in processors not for orbital calculations len_tot = 0
    len_tot = size_k * pSPARC->Nkpts_kptcomm;
    
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};

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
                int Ndsp = pSPARC->Nd * pSPARC->Nspinor;
                
                for (n = 0; n < pSPARC->Nband_bandcomm; n++) {
                    int ng = pSPARC->band_start_indx + n; // global band index
                    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
                        int spinorg = pSPARC->spinor_start_indx + spinor;
                        int shift_g = ng * Ndsp + spinorg * pSPARC->Nd; // global shift
                        int shift   = n  * DMndsp + spinor * DMnd; // local shift
                        double *Psi_kn = pSPARC->Xorb + shift;
                        SeededRandVec(Psi_kn, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, shift_g);
                    }
                }
            } else {
                SetRandMat(pSPARC->Xorb, DMndsp, pSPARC->Nband_bandcomm, -0.5, 0.5, pSPARC->spincomm);
            }
        } else {
            // allocate memory in the very first relax/MD step
            pSPARC->Xorb_kpt = (double _Complex *) malloc( len_tot * sizeof(double _Complex) );
            pSPARC->Yorb_kpt = (double _Complex *) malloc( size_k * sizeof(double _Complex) );
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
                int Ndsp = pSPARC->Nd * pSPARC->Nspinor;
                int size_kg = Ndsp * pSPARC->Nstates;

                for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
                    int kg  = pSPARC->kpt_start_indx + k; // global kpt index
                    for (n = 0; n < pSPARC->Nband_bandcomm; n++) {
                        int ng = pSPARC->band_start_indx + n; // global band index
                        for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
                            int spinorg = pSPARC->spinor_start_indx + spinor;
                            int shift_g = kg * size_kg + ng * Ndsp + spinorg * pSPARC->Nd; // global shift
                            int shift   = k  * size_k  + n  * DMndsp + spinor * DMnd; // local shift
                            double _Complex *Psi_kn = pSPARC->Xorb_kpt + shift;
                            SeededRandVec_complex(Psi_kn, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, shift_g);
                        }
                    }
                }
                
            } else {
                SetRandMat_complex(pSPARC->Xorb_kpt, DMndsp, pSPARC->Nband_bandcomm*pSPARC->Nkpts_kptcomm, -0.5, 0.5, pSPARC->spincomm); 
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
