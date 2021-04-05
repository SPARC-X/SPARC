/**
 * @file    md.c
 * @brief   This file contains the functions for performing molecular dynamics.
 *
 * @author  Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "md.h"
#include "isddft.h"
#include "orbitalElecDensInit.h"
#include "initialization.h"
#include "electronicGroundState.h"
#include "stress.h"
#include "tools.h"

#define max(a,b) ((a)>(b)?(a):(b))

//TODO: Implement the case when some atom coordinates are held fixed
/**
  @ brief: Main function of MD
**/
void main_MD(SPARC_OBJ *pSPARC) {
	int rank;
	int print_restart_typ = 0;
	double t_init, t_acc, *avgvel, *maxvel, *mindis;
	t_init = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	avgvel = (double *)malloc(pSPARC->Ntypes * sizeof(double) );
	maxvel = (double *)malloc(pSPARC->Ntypes * sizeof(double) );
	mindis = (double *)malloc(pSPARC->Ntypes*(pSPARC->Ntypes+1)/2 * sizeof(double) );

	// Check whether the restart has to be performed
	if(pSPARC->RestartFlag != 0){
		// Check if .restart file present
		if(rank == 0){
		    FILE *rst_fp = NULL;
		    if( access(pSPARC->restart_Filename, F_OK ) != -1 )
				rst_fp = fopen(pSPARC->restart_Filename,"r");
			else if( access(pSPARC->restartC_Filename, F_OK ) != -1 )
				rst_fp = fopen(pSPARC->restartC_Filename,"r");
			else
				rst_fp = fopen(pSPARC->restartP_Filename,"r");
	    
	    	if(rst_fp == NULL)
	        	pSPARC->RestartFlag = 0;
	    }
	    MPI_Bcast(&pSPARC->RestartFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

	    if (pSPARC->RestartFlag != 0) {
			RestartMD(pSPARC);
			int atm;
			if(pSPARC->cell_typ != 0){
	            for(atm = 0; atm < pSPARC->n_atom; atm++){
	                Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
		        }
	        }
	    }
	}

	// Initialize the MD for the very first step only
	Calculate_electronicGroundState(pSPARC);
	Initialize_MD(pSPARC);

	pSPARC->MD_maxStep = pSPARC->restartCount + pSPARC->MD_Nstep;

	// File output_md stores all the desirable properties from a MD run
	FILE *output_md, *output_fp;
	if (pSPARC->PrintMDout == 1 && !rank && pSPARC->MD_Nstep > 0){
        output_md = fopen(pSPARC->MDFilename,"w");
        if (output_md == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->MDFilename);
            exit(EXIT_FAILURE);
        }
        pSPARC->MDCount = -1;
        Print_fullMD(pSPARC, output_md, avgvel, maxvel, mindis); // prints the QOI in the output_md file
        pSPARC->MDCount++;

        if(pSPARC->RestartFlag == 0){
        	fprintf(output_md,":MDSTEP: %d\n", 1);
        	fprintf(output_md,":MDTM: %.2f\n", (MPI_Wtime() - t_init));
        	MD_QOI(pSPARC, avgvel, maxvel, mindis); // calculates the quantities of interest in an MD simulation
        	Print_fullMD(pSPARC, output_md, avgvel, maxvel, mindis); // prints the QOI in the output_md file
        }

        fclose(output_md);
    }

    if (!rank && pSPARC->MD_Nstep > 0) {
        output_fp = fopen(pSPARC->OutFilename,"a");
	    if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"MD step time                       :  %.3f (sec)\n", (MPI_Wtime() - t_init));
        fclose(output_fp);
    }

    pSPARC->MDCount++;
	pSPARC->elecgs_Count++;

    // Perform MD until maxStep is reached or total walltime is hit
	int Count = pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0); // Count is the MD step no. to be performed
	int check1 = (pSPARC->PrintMDout == 1 && !rank);
	int check2 = (pSPARC->Printrestart == 1 && !rank);
	t_acc = (MPI_Wtime() - t_init)/60;// tracks time in minutes
   	while(Count <= pSPARC->MD_maxStep && (t_acc + 1.0*(MPI_Wtime() - t_init)/60) < pSPARC->TWtime){
   		t_init = MPI_Wtime();
#ifdef DEBUG
   		if(!rank) printf(":MDSTEP: %d\n", Count);
#endif
		if (check1){
			output_md = fopen(pSPARC->MDFilename,"a+");
			if (output_md == NULL) {
			    printf("\nCannot open file \"%s\"\n",pSPARC->MDFilename);
			    exit(EXIT_FAILURE);
			}
			fprintf(output_md,":MDSTEP: %d\n", Count);
		}

		if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0)
			NVT_NH(pSPARC);
		else if(strcmpi(pSPARC->MDMeth,"NVE") == 0)
			NVE(pSPARC);
		else if(strcmpi(pSPARC->MDMeth,"NVK_G") == 0)
		    NVK_G(pSPARC);
		else{
			if (!rank){
				printf("\nCannot recognize MDMeth = \"%s\"\n",pSPARC->MDMeth);
                printf("MDMeth (MD Method) must be one of the following:\n\tNVT_NH\t NVE\n");
            }
            exit(EXIT_FAILURE);
        }

        MD_QOI(pSPARC, avgvel, maxvel, mindis); // calculates the quantities of interest in an MD simulation

        if(check1) {
            fprintf(output_md,":MDTM: %.2f\n", MPI_Wtime() - t_init);
           	Print_fullMD(pSPARC, output_md, avgvel, maxvel, mindis); // prints the QOI in the .aimd file
        } if(check2 && !(Count % pSPARC->Printrestart_fq)) // printrestart_fq is the frequency at which the restart file is written
			PrintMD(pSPARC, 1, print_restart_typ);
        
        if(access("SPARC.stop", F_OK ) != -1 ){ // If a .stop file exists in the folder then the run will be terminated
			pSPARC->MDCount++;
			break;
		}
		if(check1)
    		fclose(output_md);
#ifdef DEBUG
   		if (!rank) printf("Time taken by MDSTEP %d: %.3f s.\n", Count, (MPI_Wtime() - t_init));
#endif
		if(!rank){
			output_fp = fopen(pSPARC->OutFilename,"a");
		    if (output_fp == NULL) {
	            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
	            exit(EXIT_FAILURE);
	        }
	        fprintf(output_fp,"MD step time                       :  %.3f (sec)\n", (MPI_Wtime() - t_init));
	        fclose(output_fp);
	    }

		pSPARC->MDCount ++;
		Count ++;
		t_acc += (MPI_Wtime() - t_init)/60;
	} // end while loop
	if(check2){
		pSPARC->MDCount --;
		print_restart_typ = 1;
		PrintMD(pSPARC, 1, print_restart_typ);
	}
    free(avgvel);
    free(maxvel);
    free(mindis);
}

/**
  @ brief: function to initialize velocities and accelerations for Molecular Dynamics (MD)
 **/
void Initialize_MD(SPARC_OBJ *pSPARC) {
	int ityp, atm, count, rand_seed = 5;

	if (pSPARC->ion_vel_dstr_rand == 1) {
		// add a time-dependent component to the seed
		rand_seed += (int)MPI_Wtime();
	}

	pSPARC->ion_accel = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	if (pSPARC->ion_accel == NULL) {
        printf("\nCannot allocate memory for ion acceleration array!\n");
        exit(EXIT_FAILURE);
    }

    int count_x = 0;
    int count_y = 0;
    int count_z = 0;
    count = 0;
    for (atm = 0; atm < pSPARC->n_atom; atm++) {
        count_x += pSPARC->mvAtmConstraint[3 * count];
        count_y += pSPARC->mvAtmConstraint[3 * count + 1];
        count_z += pSPARC->mvAtmConstraint[3 * count + 2];
        count++;
    }
    pSPARC->dof = (count_x - (count_x == pSPARC->n_atom)) + (count_y - (count_y == pSPARC->n_atom))
                + (count_z - (count_z == pSPARC->n_atom)); // TODO: Create a user defined variable dof with this as a default value
	
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++)
		pSPARC->Mass[ityp] *= pSPARC->amu2au; // mass in atomic units
    
    pSPARC->MD_dt *= pSPARC->fs2atu; // time in atomic time units

	// Variables for NVT_NH
	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0 && pSPARC->RestartFlag != 1){
		pSPARC->snose = 0.0;
		pSPARC->xi_nose = 0.0;
		pSPARC->thermos_Ti = pSPARC->ion_T;
		pSPARC->thermos_T  = pSPARC->thermos_Ti;
	}

	if(pSPARC->RestartFlag == 0){
	    double mass_sum = 0.0, mvsum_x, mvsum_y, mvsum_z;
	    mvsum_x = mvsum_y = mvsum_z = 0.0;
	    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		    mass_sum += pSPARC->Mass[ityp] * pSPARC->nAtomv[ityp];
	    }
	    if(pSPARC->ion_vel_dstr != 3){ // initial velocity of ions is not explicitly provided by the user
		    pSPARC->ion_vel = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
		    if (pSPARC->ion_vel == NULL) {
            	printf("\nCannot allocate memory for ion velocity array!\n");
            	exit(EXIT_FAILURE);
        	}
        }
	    // Uniform velocity distribution
	    if(pSPARC->ion_vel_dstr == 1){
		    double vel_cm, s = 2.0, x = 0.0, y = 0.0; // vel_cm: center of mass velocity in Bohr/atu
		    vel_cm = sqrt((pSPARC->dof * pSPARC->ion_T * pSPARC->kB)/mass_sum);
		    count = 0;
		    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
			    srand(ityp + rand_seed);// random seed (default 5). Seed has to be common across all processors!!
			    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
				    while(s>1.0){
					    x = 2.0 * ((double)(rand()) / (double)(RAND_MAX)) - 1; // random number between -1 and +1
					    y = 2.0 * ((double)(rand()) / (double)(RAND_MAX)) - 1;
					    s = x * x + y * y;
				    }
				    pSPARC->ion_vel[count * 3 + 2] = vel_cm * (1.0 - 2.0 * s);
				    s = 2.0 * sqrt(1.0 - s);
				    pSPARC->ion_vel[count * 3 + 1] = s * y * vel_cm;
				    pSPARC->ion_vel[count * 3] = s * x * vel_cm;
				    mvsum_x += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3];
				    mvsum_y += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 1];
				    mvsum_z += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 2];
				    count += 1;
			    }
		    }
	    }else if(pSPARC->ion_vel_dstr == 2) // Maxwell-Boltzmann distribution of velocity (Default!)
	    {
		    count = 0;
		    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
			    srand(ityp + rand_seed);// random seed (default 5). Seed has to be common across all processors!!
			    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
				    pSPARC->ion_vel[count * 3] = sqrt(pSPARC->kB * pSPARC->ion_T/pSPARC->Mass[ityp]) * cos(2 * M_PI * ((double)(rand()) / (double)(RAND_MAX)));
				    pSPARC->ion_vel[count * 3] *= sqrt(-2.0 * log((double)(rand()) / (double)(RAND_MAX)));
				    pSPARC->ion_vel[count * 3 + 1] = sqrt(pSPARC->kB * pSPARC->ion_T/pSPARC->Mass[ityp]) * cos(2 * M_PI * ((double)(rand()) / (double)(RAND_MAX)));
				    pSPARC->ion_vel[count * 3 + 1] *= sqrt(-2.0 * log((double)(rand()) / (double)(RAND_MAX)));
				    pSPARC->ion_vel[count * 3 + 2] = sqrt(pSPARC->kB * pSPARC->ion_T/pSPARC->Mass[ityp]) * cos( 2 * M_PI * ((double)(rand()) / (double)(RAND_MAX)));
				    pSPARC->ion_vel[count * 3 + 2] *= sqrt(-2.0 * log((double)(rand()) / (double)(RAND_MAX)));
				    mvsum_x += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3];
				    mvsum_y += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 1];
				    mvsum_z += pSPARC->Mass[ityp] * pSPARC->ion_vel[count * 3 + 2];
				    count += 1;
			    }
		    }
	    }

	    // Remove translation (rotation not possible in case of PBC) TODO: angular velocity in other types of boundary conditions
	    pSPARC->KE = 0.0;
	    count = 0;
	    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			    pSPARC->ion_vel[count * 3] -= mvsum_x/mass_sum;
			    pSPARC->ion_vel[count * 3 + 1] -= mvsum_y/mass_sum;
			    pSPARC->ion_vel[count * 3 + 2] -= mvsum_z/mass_sum;
			    count += 1;
		    }
	    }

	    // Zeroing the velocity for fixed atoms
	    count = 0;
	    pSPARC->KE = 0.0;
	    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			    pSPARC->ion_vel[count * 3] *= pSPARC->mvAtmConstraint[count * 3];
			    pSPARC->ion_vel[count * 3 + 1] *= pSPARC->mvAtmConstraint[count * 3 + 1];
			    pSPARC->ion_vel[count * 3 + 2] *= pSPARC->mvAtmConstraint[count * 3 + 2];
			    pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			    count += 1;
		    }
	    }

	    // Rescale velocities
	    double rescal_fac ;
	    rescal_fac = sqrt(pSPARC->dof * pSPARC->kB * pSPARC->ion_T/(2.0 * pSPARC->KE)) ;
	    pSPARC->KE = 0.0;
	    count = 0;
	    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		    for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			    pSPARC->ion_vel[count * 3] *= rescal_fac;
			    pSPARC->ion_vel[count * 3 + 1] *= rescal_fac;
			    pSPARC->ion_vel[count * 3 + 2] *= rescal_fac;
			    pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			    count += 1;
		    }
	    }
	}
	count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->ion_accel[count * 3] = pSPARC->forces[count * 3]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 1] = pSPARC->forces[count * 3 + 1]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 2] = pSPARC->forces[count * 3 + 2]/pSPARC->Mass[ityp];
			count += 1;
		}
	}

#ifdef DEBUG
	// Statistics to be set to zero initially
	pSPARC->mean_TE_ext = pSPARC->std_TE_ext = 0.0;
	pSPARC->mean_elec_T = pSPARC->mean_ion_T = pSPARC->mean_TE = pSPARC->mean_KE = pSPARC->mean_PE = pSPARC->mean_U = pSPARC->mean_Entropy = 0.0;
	pSPARC->std_elec_T = pSPARC->std_ion_T = pSPARC->std_TE = pSPARC->std_KE = pSPARC->std_PE = pSPARC->std_U = pSPARC->std_Entropy = 0.0;
#endif
}

/*
@ brief function to perform NVT MD simulation with Nose hoover thermostat wherein number of particles, volume and temperature are kept constant (equivalent to ionmov = 8 in ABINIT)
*/
void NVT_NH(SPARC_OBJ *pSPARC) {
	double fsnose;
	// First step velocity Verlet
	VVerlet1(pSPARC);
	// Update thermostat
	pSPARC->thermos_T = pSPARC->thermos_Ti + ((pSPARC->thermos_Tf - pSPARC->thermos_Ti)/(pSPARC->MD_Nstep)) * (pSPARC->MDCount);
	fsnose = (pSPARC->v2nose - pSPARC->dof * pSPARC->kB * pSPARC->thermos_T)/pSPARC->qmass;
	pSPARC->snose += pSPARC->MD_dt * (pSPARC->xi_nose + 0.5 * pSPARC->MD_dt * fsnose);
	pSPARC->xi_nose += 0.5 * pSPARC->MD_dt * fsnose;
	// Charge extrapolation (for better rho_guess)
	elecDensExtrapolation(pSPARC);
	// Check position of atom near the boundary and apply wraparound in case of PBC, otherwise show error if the atom is too close to the boundary for bounded domain
	Check_atomlocation(pSPARC);
	// Compute DFT energy and forces by solving Kohn-Sham eigenvalue problem
	Calculate_electronicGroundState(pSPARC);
	pSPARC->elecgs_Count++;
	// Second step velocity Verlet
	VVerlet2(pSPARC);
}

/*
@ brief: function to perform first step of velocity verlet algorithm
*/
void VVerlet1(SPARC_OBJ* pSPARC) {
	int ityp, atm, count = 0;
	pSPARC->v2nose = 0.0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->v2nose += pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			pSPARC->ion_vel[count * 3] += 0.5 * pSPARC->MD_dt * (pSPARC->ion_accel[count * 3] - pSPARC->xi_nose * pSPARC->ion_vel[count * 3]);
			pSPARC->ion_vel[count * 3 + 1] += 0.5 * pSPARC->MD_dt * (pSPARC->ion_accel[count * 3 + 1] - pSPARC->xi_nose * pSPARC->ion_vel[count * 3 + 1]);
			pSPARC->ion_vel[count * 3 + 2] += 0.5 * pSPARC->MD_dt * (pSPARC->ion_accel[count * 3 + 2] - pSPARC->xi_nose * pSPARC->ion_vel[count * 3 + 2]);
			// r_(t+dt) = r_t + dt*v_(t+dt/2)
			pSPARC->atom_pos[count * 3] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3];
			pSPARC->atom_pos[count * 3 + 1] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3 + 1];
			pSPARC->atom_pos[count * 3 + 2] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3 + 2];
			count ++;
		}
	}
}

/*
@ brief: function to perform second step of velocity verlet algorithm
*/
void VVerlet2(SPARC_OBJ* pSPARC) {
	int ityp, atm, count = 0, ready = 0, kk, jj;
	double *vel_temp, *vonose, *hnose, *binose, xin_nose, xio, delxi, dnose ;
	vel_temp = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	vonose = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	hnose = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	binose = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	xin_nose = pSPARC->xi_nose;
	pSPARC->v2nose = 0.0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->v2nose += pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			//a(t+dt) = F(t+dt)/M
			pSPARC->ion_accel[count * 3] = pSPARC->forces[count * 3]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 1] = pSPARC->forces[count * 3 + 1]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 2] = pSPARC->forces[count * 3 + 2]/pSPARC->Mass[ityp];
			vel_temp[count * 3] = pSPARC->ion_vel[count * 3];
			vel_temp[count * 3 + 1] = pSPARC->ion_vel[count * 3 + 1];
			vel_temp[count * 3 + 2] = pSPARC->ion_vel[count * 3 + 2];
			count ++;
		}
	}
	do {
		xio = xin_nose ;
		delxi = 0.0 ;
		count = 0;
		for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
			for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
				vonose[count * 3] = vel_temp[count * 3] ;
				vonose[count * 3 + 1] = vel_temp[count * 3 + 1] ;
				vonose[count * 3 + 2] = vel_temp[count * 3 + 2] ;
				hnose[count * 3] = -0.5 * pSPARC->MD_dt * (pSPARC->ion_accel[count * 3] - xio * vonose[count * 3]) - (pSPARC->ion_vel[count * 3] - vonose[count * 3]);
				hnose[count * 3 + 1] = -0.5 * pSPARC->MD_dt * (pSPARC->ion_accel[count * 3 + 1] - xio * vonose[count * 3 + 1]) - (pSPARC->ion_vel[count * 3 + 1] - vonose[count * 3 + 1]);
				hnose[count * 3 + 2] = -0.5 * pSPARC->MD_dt * (pSPARC->ion_accel[count * 3 + 2] - xio * vonose[count * 3 + 2]) - (pSPARC->ion_vel[count * 3 + 2] - vonose[count * 3 + 2]);
				binose[count * 3] = vonose[count * 3] * pSPARC->MD_dt * pSPARC->Mass[ityp]/pSPARC->qmass ;
				delxi += hnose[count * 3] * binose[count * 3] ;
				binose[count * 3 + 1] = vonose[count * 3 + 1] * pSPARC->MD_dt * pSPARC->Mass[ityp]/pSPARC->qmass ;
				delxi += hnose[count * 3 + 1] * binose[count * 3 + 1] ;
				binose[count * 3 + 2] = vonose[count * 3 + 2] * pSPARC->MD_dt * pSPARC->Mass[ityp]/pSPARC->qmass ;
				delxi += hnose[count * 3 + 2] * binose[count * 3 + 2] ;
				count ++;
			}
		}
		dnose = -1.0 * (0.5 * xio * pSPARC->MD_dt + 1.0) ;
		delxi += -1.0 * dnose * ((-1.0 * pSPARC->v2nose + pSPARC->dof * pSPARC->kB * pSPARC->thermos_T) * 0.5 * pSPARC->MD_dt/pSPARC->qmass - (pSPARC->xi_nose - xio)) ;
		delxi /= (-0.5 * pow(pSPARC->MD_dt,2.0) * pSPARC->v2nose/pSPARC->qmass + dnose) ;
		pSPARC->v2nose = 0.0 ;
		count = 0 ;
		for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
			for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
				vel_temp[count * 3] += (hnose[count * 3] + 0.5 * pSPARC->MD_dt * vonose[count * 3] * delxi)/dnose ;
				vel_temp[count * 3 + 1] += (hnose[count * 3 + 1] + 0.5 * pSPARC->MD_dt * vonose[count * 3 + 1] * delxi)/dnose ;
				vel_temp[count * 3 + 2] += (hnose[count * 3 + 2] + 0.5 * pSPARC->MD_dt * vonose[count * 3 + 2] * delxi)/dnose ;
				pSPARC->v2nose += pSPARC->Mass[ityp] * (pow(vel_temp[count * 3], 2.0) + pow(vel_temp[count * 3 + 1], 2.0) + pow(vel_temp[count * 3 + 2], 2.0));
				count ++;
			}
		}
		xin_nose = xio + delxi ;
		ready = 1 ;
		//Test for convergence
		kk = -1, jj = 0;
		do{
			kk = kk + 1 ;
			if(kk >= pSPARC->n_atom){
				kk = 0;
				jj ++;
			}
			if(kk < pSPARC->n_atom && jj < 3){
				//if (fabs(vel_temp[kk + jj*pSPARC->n_atom]) < 1e-50)
				// vel_temp[kk + jj*pSPARC->n_atom] = 1e-50 ;
				if(fabs((vel_temp[kk + jj * pSPARC->n_atom] - vonose[kk + jj * pSPARC->n_atom])/vel_temp[kk + jj * pSPARC->n_atom]) > 1e-7)
					ready = 0 ;
			}
			else{
				//if (fabs(xin_nose) < 1e-50)
				//  xin_nose = 1e-50 ;
				if(fabs((xin_nose - xio)/xin_nose) > 1e-7)
					ready = 0 ;
			}
		} while(kk < pSPARC->n_atom && jj < 3 && ready);
	} while (ready == 0);

	pSPARC->xi_nose = xin_nose ;
	count = 0;
	pSPARC->KE = 0.0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->ion_vel[count * 3] = vel_temp[count * 3];
			pSPARC->ion_vel[count * 3 + 1] = vel_temp[count * 3 + 1];
			pSPARC->ion_vel[count * 3 + 2] = vel_temp[count * 3 + 2];
			pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			count ++;
		}
	}
	free(vel_temp);
	free(vonose);
	free(binose);
	free(hnose);
}

/**
  @ brief:  Perform molecular dynamics keeping number of particles, volume of the cell and total energy(P.E.(calculated quantum mechanically) + K.E. of ions(Calculated classically)) constant.
 **/
void NVE(SPARC_OBJ *pSPARC) {
	// Leapfrog step - 1
	Leapfrog_part1(pSPARC);
	// Charge extrapolation (for better rho_guess)
	elecDensExtrapolation(pSPARC);
	// Check position of atom near the boundary and apply wraparound in case of PBC, otherwise show error if the atom is too close to the boundary for bounded domain
	Check_atomlocation(pSPARC);
	// Compute DFT energy and forces by solving Kohn-Sham eigenvalue problem
	Calculate_electronicGroundState(pSPARC);
	pSPARC->elecgs_Count++;
	// Leapfrog step (part-2)
	Leapfrog_part2(pSPARC);
}

/*
@ brief: function to update position of atoms using Leapfrog method
*/
void Leapfrog_part1(SPARC_OBJ *pSPARC) {
	/********************************************************
	// Leapfrog algorithm
	PART-I (Implemented in this function)
	v_(t+dt/2) = v_t + 0.5*dt*a_t
	r_(t+dt) = r_t + dt*v_(t+dt/2)

	PART-II (Implemented in the next function, after computing
	a_(t+dt) for current positions)
	v_(t+dt) = v_(t+dt/2) + 0.5*dt*a_(t+dt)
	 **********************************************************/
	int count = 0, atm;
	for(atm = 0; atm < pSPARC->n_atom; atm++){
		// v_(t+dt/2) = v_t + 0.5*dt*a_t
		pSPARC->ion_vel[count * 3] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3];
		pSPARC->ion_vel[count * 3 + 1] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3 + 1];
		pSPARC->ion_vel[count * 3 + 2] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3 + 2];
		// r_(t+dt) = r_t + dt*v_(t+dt/2)
		pSPARC->atom_pos[count * 3] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3];
		pSPARC->atom_pos[count * 3 + 1] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3 + 1];
		pSPARC->atom_pos[count * 3 + 2] += pSPARC->MD_dt * pSPARC->ion_vel[count * 3 + 2];
		count ++;
	}

}

/*
 @ brief: function to update velocity of atoms using Leapfrog method
*/
void Leapfrog_part2(SPARC_OBJ *pSPARC) {
	/************************************
	  PART-II Leapfrog algorithm
	  v_(t+dt) = v_(t+dt/2) + 0.5*dt*a_(t+dt)
	 *************************************/
	int count = 0, ityp, atm;
	pSPARC->KE = 0.0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->ion_accel[count * 3] = pSPARC->forces[count * 3]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 1] = pSPARC->forces[count * 3 + 1]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 2] = pSPARC->forces[count * 3 + 2]/pSPARC->Mass[ityp];
			pSPARC->ion_vel[count * 3] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3];
			pSPARC->ion_vel[count * 3 + 1] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3 + 1];
			pSPARC->ion_vel[count * 3 + 2] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3 + 2];
			pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			count ++;
		}
	}

}


/**
  @ brief:  Perform molecular dynamics keeping number of particles, volume of the cell and kinetic energy constant i.e. NVK with Gaussian thermostat.
            It is based on the implementation in ABINIT (ionmov=12)
 **/
void NVK_G(SPARC_OBJ *pSPARC) {
	// Calculate velocity at next half time step
	Calc_vel1_G(pSPARC);

	// Charge extrapolation (for better rho_guess)
	elecDensExtrapolation(pSPARC);

	// Check position of atom near the boundary and apply wraparound in case of PBC, otherwise show error if the atom is too close to the boundary for bounded domain
	Check_atomlocation(pSPARC);

	// Compute DFT energy and forces by solving Kohn-Sham eigenvalue problem
	Calculate_electronicGroundState(pSPARC);
	pSPARC->elecgs_Count++;

	// Calculate velocity at next full time step
	Calc_vel2_G(pSPARC);
}


/*
 @ brief: calculate velocity at next half time step for isokinetic ensemble with Gaussian thermostat
*/

void Calc_vel1_G(SPARC_OBJ *pSPARC) {
    double v2gauss = 0.0;
    double a = 0.0;
    double b = 0.0;
    int ityp, atm, count = 0;

    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
		    v2gauss += (pSPARC->ion_vel[count * 3] * pSPARC->ion_vel[count * 3] +
		                pSPARC->ion_vel[count * 3 + 1] * pSPARC->ion_vel[count * 3 + 1] +
		                pSPARC->ion_vel[count * 3 + 2] * pSPARC->ion_vel[count * 3 + 2]) * pSPARC->Mass[ityp] ;

		    a += pSPARC->forces[count * 3] * pSPARC->ion_vel[count * 3] +
		         pSPARC->forces[count * 3 + 1] * pSPARC->ion_vel[count * 3 + 1] +
		         pSPARC->forces[count * 3 + 2] * pSPARC->ion_vel[count * 3 + 2] ;

		    b += pSPARC->forces[count * 3] * pSPARC->ion_accel[count * 3] +
		         pSPARC->forces[count * 3 + 1] * pSPARC->ion_accel[count * 3 + 1] +
		         pSPARC->forces[count * 3 + 2] * pSPARC->ion_accel[count * 3 + 2] ;

		    count ++;
		}
	}

	a /= v2gauss;
	b /= v2gauss;

	double sqb = sqrt(b);
	double as = sqb * pSPARC->MD_dt/2.0;
	if(as > 300.0)
	    as = 300.0;

	double s1 = cosh(as);
    double s2 = sinh(as);
    double s = a * (s1-1.0)/b + s2/sqb;
    double scdot = a * s2/sqb + s1;

    count = 0;
    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
		    pSPARC->ion_vel[count * 3] = (pSPARC->ion_vel[count * 3] + pSPARC->ion_accel[count * 3] * s)/scdot;
		    pSPARC->ion_vel[count * 3 + 1] = (pSPARC->ion_vel[count * 3 + 1] + pSPARC->ion_accel[count * 3 + 1] * s)/scdot;
		    pSPARC->ion_vel[count * 3 + 2] = (pSPARC->ion_vel[count * 3 + 2] + pSPARC->ion_accel[count * 3 + 2] * s)/scdot;

		    pSPARC->atom_pos[count * 3] += pSPARC->ion_vel[count * 3] * pSPARC->MD_dt;
		    pSPARC->atom_pos[count * 3 + 1] += pSPARC->ion_vel[count * 3 + 1] * pSPARC->MD_dt;
		    pSPARC->atom_pos[count * 3 + 2] += pSPARC->ion_vel[count * 3 + 2] * pSPARC->MD_dt;
		    count ++;
		}
	}
}


/*
 @ brief: calculate velocity at next full time step for isokinetic ensemble with Gaussian thermostat
*/

void Calc_vel2_G(SPARC_OBJ *pSPARC) {
    double v2gauss = 0.0;
    double a = 0.0;
    double b = 0.0;
    int ityp, atm, count = 0;

    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
		    pSPARC->ion_accel[count * 3] = pSPARC->forces[count * 3]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 1] = pSPARC->forces[count * 3 + 1]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 2] = pSPARC->forces[count * 3 + 2]/pSPARC->Mass[ityp];

		    v2gauss += (pSPARC->ion_vel[count * 3] * pSPARC->ion_vel[count * 3] +
		                pSPARC->ion_vel[count * 3 + 1] * pSPARC->ion_vel[count * 3 + 1] +
		                pSPARC->ion_vel[count * 3 + 2] * pSPARC->ion_vel[count * 3 + 2]) * pSPARC->Mass[ityp] ;

		    a += pSPARC->forces[count * 3] * pSPARC->ion_vel[count * 3] +
		         pSPARC->forces[count * 3 + 1] * pSPARC->ion_vel[count * 3 + 1] +
		         pSPARC->forces[count * 3 + 2] * pSPARC->ion_vel[count * 3 + 2] ;

		    b += pSPARC->forces[count * 3] * pSPARC->ion_accel[count * 3] +
		         pSPARC->forces[count * 3 + 1] * pSPARC->ion_accel[count * 3 + 1] +
		         pSPARC->forces[count * 3 + 2] * pSPARC->ion_accel[count * 3 + 2] ;

		    count ++;
		}
	}

	a /= v2gauss;
	b /= v2gauss;

	double sqb = sqrt(b);
	double as = sqb * pSPARC->MD_dt/2.0;
	if(as > 300.0)
	    as = 300.0;

	double s1 = cosh(as);
    double s2 = sinh(as);
    double s = a * (s1-1.0)/b + s2/sqb;
    double scdot = a * s2/sqb + s1;

    count = 0;
    pSPARC->KE = 0.0;
    for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
		    pSPARC->ion_vel[count * 3] = (pSPARC->ion_vel[count * 3] + pSPARC->ion_accel[count * 3] * s)/scdot;
		    pSPARC->ion_vel[count * 3 + 1] = (pSPARC->ion_vel[count * 3 + 1] + pSPARC->ion_accel[count * 3 + 1] * s)/scdot;
		    pSPARC->ion_vel[count * 3 + 2] = (pSPARC->ion_vel[count * 3 + 2] + pSPARC->ion_accel[count * 3 + 2] * s)/scdot;
		    pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
		    count ++;
		}
	}
}



/*
 @ brief: function to wrap around atom positions that lie outside main domain in case of PBC and check if the atoms are too close to the boundary in case of bounded domain
*/
void Check_atomlocation(SPARC_OBJ *pSPARC) {
    int rank, ityp, i, atm, atm2, count, dir = 0, maxdir = 3, BC;
	double length, temp, rc1 = 0.0, rc2 = 0.0, *rc, tol = 0.5;// Change tol according to the situation
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	rc = (double *)malloc(pSPARC->Ntypes * sizeof(double) );
	for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rc[ityp] = 0.0;
        for(i = 0; i <= pSPARC->psd[ityp].lmax; i++)
        	rc[ityp] = max(rc[ityp], pSPARC->psd[ityp].rc[i]);
    }
    // Check whether the two atoms are closer than rc
	for(atm = 0; atm < pSPARC->n_atom - 1; atm++){
		count = 0;
		for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
			count += pSPARC->nAtomv[ityp];
			if(atm < count){
				rc1 = rc[ityp];
                break;
			}else
				continue;
		}
		for(atm2 = atm + 1; atm2 < pSPARC->n_atom; atm2++){
			count = 0;
			for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
				count += pSPARC->nAtomv[ityp];
				if(atm2 < count){
					rc2 = rc[ityp];
                   break;
				}else
					continue;
			}
			temp = fabs(sqrt(pow(pSPARC->atom_pos[3 * atm] - pSPARC->atom_pos[3 * atm2],2.0) + pow(pSPARC->atom_pos[3 * atm + 1] - pSPARC->atom_pos[3 * atm2 + 1],2.0) + pow(pSPARC->atom_pos[3 * atm + 2] - pSPARC->atom_pos[3 * atm2 + 2],2.0) ));
			if(temp < (1 - tol) * (rc1 + rc2)){
				if(!rank)
					printf("WARNING: Atoms too close to each other with interatomic distance of %E Bohr\n",temp);
				atm2 = pSPARC->n_atom;
				atm = pSPARC->n_atom - 1;
			}
		}
	}
	free(rc);

	// Convert Cart to nonCart coordinates for non orthogonal cell
    if(pSPARC->cell_typ != 0){
        for(atm = 0; atm < pSPARC->n_atom; atm++)
	        Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
    }

	while(dir < maxdir){
		if(dir == 0){
			length = pSPARC->range_x;
			BC = pSPARC->BCx;
		}
		else if(dir == 1){
			length = pSPARC->range_y;
			BC = pSPARC->BCy;
		}
		else if(dir == 2){
			length = pSPARC->range_z;
			BC = pSPARC->BCz;
		}
		if(BC == 1){
			for(atm = 0; atm < pSPARC->n_atom; atm++){
				if(pSPARC->atom_pos[atm * 3 + dir] >= length || pSPARC->atom_pos[atm * 3 + dir] < 0){
					if(!rank)
						printf("Error: Atom number %d has crossed the boundary in %d direction",atm, dir);
					exit(EXIT_FAILURE);
				}
			}
		}else if(BC == 0){
		// TODO: Assumed that atom moves less than one cell length, generalize in future
			for(atm = 0; atm < pSPARC->n_atom; atm++){
				if(pSPARC->atom_pos[atm * 3 + dir] >= length)
					pSPARC->atom_pos[atm * 3 + dir] -= length;
				else if(pSPARC->atom_pos[atm * 3 + dir] < 0)
					pSPARC->atom_pos[atm * 3 + dir] += length;
			}
		}
		dir ++;
	}
}

/*
 @ brief: function to write all relevant DFT quantities generated during MD simulation
*/
void Print_fullMD(SPARC_OBJ *pSPARC, FILE *output_md, double *avgvel, double *maxvel, double *mindis) {
    int atm;

    // Print Description of all variables
    if(pSPARC->MDCount == -1){
    	fprintf(output_md,":Description: \n\n");
    	fprintf(output_md,":Desc_R: Atom positions in Cartesian coordinates. Unit=Bohr \n");
    	fprintf(output_md,":Desc_V: Atomic velocities in Cartesian coordinates. Unit=Bohr/atu \n"
    					  "     where atu is the atomic unit of time, hbar/Ha \n");
    	fprintf(output_md,":Desc_F: Atomic forces in Cartesian coordinates. Unit=Ha/Bohr \n");
    	fprintf(output_md,":Desc_MDTM: MD time. Unit=second \n");
    	fprintf(output_md,":Desc_TEL: Electronic temperature. Unit=Kelvin \n");
    	fprintf(output_md,":Desc_TIO: Ionic temperature. Unit=Kelvin \n");
    	fprintf(output_md,":Desc_TEN: Total energy. TEN = KEN + FEN. Unit=Ha/atom \n");
    	fprintf(output_md,":Desc_KEN: Ionic kinetic energy. Unit=Ha/atom \n");
    	fprintf(output_md,":Desc_FEN: Free energy. FEN = UEN - TSEN. Unit=Ha/atom \n");
    	fprintf(output_md,":Desc_UEN: Internal energy. Unit=Ha/atom \n");
    	fprintf(output_md,":Desc_TSEN: Electronic entropic energy. Unit=Ha/atom \n");
    	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
    		fprintf(output_md,":Desc_TENX: Total energy of extended system. Unit=Ha/atom \n");
    	}
    	if(pSPARC->Calc_stress == 1){
	    	fprintf(output_md,":Desc_STRESS: Stress, excluding ion-kinetic contribution. Unit=GPa(all periodic),Ha/Bohr**2(surface),Ha/Bohr(wire) \n");
	    	fprintf(output_md,":Desc_STRIO: Ion-kinetic stress in cartesian coordinate. Unit=GPa(all periodic),Ha/Bohr**2(surface),Ha/Bohr(wire) \n");
	    }
	    if((pSPARC->Calc_pres == 1 || pSPARC->Calc_stress == 1) && pSPARC->BC == 2){
	    	fprintf(output_md,":Desc_PRESIO: Ion-kinetic pressure in cartesian coordinate. Unit=GPa \n");
	    	fprintf(output_md,":Desc_PRES: Pressure, excluding ion-kinetic contribution. Unit=GPa \n");
	    	fprintf(output_md,":Desc_PRESIG: Pressure N k T/V of ideal gas at temperature T = TIO. Unit=GPa \n"
         					  "     where N = number of particles, k = Boltzmann constant, V = volume\n");
	    }

	    #ifdef DEBUG
	    fprintf(output_md,":Desc_ST: (DEBUG mode only) Tags ending in 'ST' describe statistics. Printed are the mean and standard deviation, respectively. \n");
    	#endif

    	fprintf(output_md,":Desc_AVGV: Average of the speed of all ions of the same type. Unit=Bohr/atu \n");
    	fprintf(output_md,":Desc_MAXV: Maximum of the speed of all ions of the same type. Unit=Bohr/atu \n");
    	fprintf(output_md,":Desc_MIND: Minimum of the distance of all ions of the same type. Unit=Bohr \n");
    	fprintf(output_md, "\n\n");
    } else{

	    // Print Temperature and energy
	    fprintf(output_md,":TEL: %.15g\n", pSPARC->elec_T);
	    fprintf(output_md,":TIO: %.15g\n", pSPARC->ion_T);
		fprintf(output_md,":TEN: %18.10E\n", pSPARC->TE);
		fprintf(output_md,":KEN: %18.10E\n", pSPARC->KE);
		fprintf(output_md,":FEN: %18.10E\n", pSPARC->PE);
		fprintf(output_md,":UEN: %18.10E\n", pSPARC->PE - pSPARC->Entropy/pSPARC->n_atom);
		fprintf(output_md,":TSEN:%18.10E\n", pSPARC->Entropy/pSPARC->n_atom);
		if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
			fprintf(output_md,":TENX:%18.10E \n", pSPARC->TE_ext);
		}

	    // Print atomic position
	    if(pSPARC->PrintAtomPosFlag){
		    fprintf(output_md,":R:\n");
		    for(atm = 0; atm < pSPARC->n_atom; atm++){
		    	fprintf(output_md,"%18.10E %18.10E %18.10E\n", pSPARC->atom_pos[3 * atm], pSPARC->atom_pos[3 * atm + 1], pSPARC->atom_pos[3 * atm + 2]);
		    }
		}

	    // Print velocity
	    if(pSPARC->PrintAtomVelFlag){
		    fprintf(output_md,":V:\n");
		    for(atm = 0; atm < pSPARC->n_atom; atm++){
		    	fprintf(output_md,"%18.10E %18.10E %18.10E\n", pSPARC->ion_vel[3 * atm], pSPARC->ion_vel[3 * atm + 1], pSPARC->ion_vel[3 * atm + 2]);
		    }
		}

		// Print Forces
		if(pSPARC->PrintForceFlag){
			fprintf(output_md,":F:\n");
		    for(atm = 0; atm < pSPARC->n_atom; atm++){
		    	fprintf(output_md,"%18.10E %18.10E %18.10E\n", pSPARC->forces[3 * atm], pSPARC->forces[3 * atm + 1], pSPARC->forces[3 * atm + 2]);
		    }
		}

	    // Print stress
	    if(pSPARC->Calc_stress == 1){
	        fprintf(output_md,":STRIO:\n");
	        PrintStress (pSPARC, pSPARC->stress_i, output_md);
	        fprintf(output_md,":STRESS:\n");
	        double stress_e[6]; // electronic stress
            for (int i = 0; i < 6; i++) 
                stress_e[i] = pSPARC->stress[i] - pSPARC->stress_i[i];
            PrintStress (pSPARC, stress_e, output_md);
	    }

	    // print pressure
	    if ((pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1) && pSPARC->BC == 2) {
	        // find pressure of ideal gas: NkT/V
	        // Define measure of unit cell
	 		double cell_measure = pSPARC->Jacbdet;
	        if(pSPARC->BCx == 0)
	            cell_measure *= pSPARC->range_x;
	        if(pSPARC->BCy == 0)
	            cell_measure *= pSPARC->range_y;
	        if(pSPARC->BCz == 0)
	            cell_measure *= pSPARC->range_z;
	        double pres_ig = 0.0;
	        pres_ig = pSPARC->n_atom * pSPARC->kB * pSPARC->ion_T / cell_measure;

	        fprintf(output_md,":PRESIO: %18.10E\n", pSPARC->pres_i*CONST_HA_BOHR3_GPA);
	        fprintf(output_md,":PRES:   %18.10E\n", (pSPARC->pres-pSPARC->pres_i)*CONST_HA_BOHR3_GPA);
	        fprintf(output_md,":PRESIG: %18.10E\n", pres_ig*CONST_HA_BOHR3_GPA); // Ideal Gas

		}

	#ifdef DEBUG
	    // Print Statistical properties
	    fprintf(output_md,":TELST: %18.10E %18.10E\n", pSPARC->mean_elec_T, pSPARC->std_elec_T);
	    fprintf(output_md,":TIOST: %18.10E %18.10E\n", pSPARC->mean_ion_T, pSPARC->std_ion_T);
		fprintf(output_md,":TENST: %18.10E %18.10E\n", pSPARC->mean_TE, pSPARC->std_TE);
		fprintf(output_md,":KENST: %18.10E %18.10E\n", pSPARC->mean_KE, pSPARC->std_KE);
		fprintf(output_md,":FENST: %18.10E %18.10E\n", pSPARC->mean_PE, pSPARC->std_PE);
		fprintf(output_md,":UENST: %18.10E %18.10E\n", pSPARC->mean_U, pSPARC->std_U);
		fprintf(output_md,":TSENST: %18.10E %18.10E\n", pSPARC->mean_Entropy, pSPARC->std_Entropy);
		if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
			fprintf(output_md,":TENXST: %18.10E %18.10E\n", pSPARC->mean_TE_ext, pSPARC->std_TE_ext);
		}
		fprintf(output_md,":AVGV:\n");
		for(atm = 0; atm < pSPARC->Ntypes; atm++)
			fprintf(output_md," %18.10E\n",avgvel[atm]);
		fprintf(output_md,":MAXV:\n");
		for(atm = 0; atm < pSPARC->Ntypes; atm++)
			fprintf(output_md," %18.10E\n",maxvel[atm]);
	#endif
		fprintf(output_md,":MIND:\n");
		int Nintr = pSPARC->Ntypes * (pSPARC->Ntypes + 1) / 2;
		for(atm = 0; atm < Nintr; atm++)
			fprintf(output_md," %18.10E\n",mindis[atm]);
	}
}

/*
 @ brief function to evaluate the qunatities of interest in a MD simulation
*/
void MD_QOI(SPARC_OBJ *pSPARC, double *avgvel, double *maxvel, double *mindis) {
	// Compute MD energies (TE=KE+PE)/atom and temperature
	pSPARC->ion_T = 2 * pSPARC->KE /(pSPARC->kB * pSPARC->dof);
	if(pSPARC->ion_elec_eqT == 1){
		pSPARC->elec_T = pSPARC->ion_T;
		pSPARC->Beta = 1.0/(pSPARC->elec_T * pSPARC->kB);
	}
	pSPARC->PE = pSPARC->Etot / pSPARC->n_atom;
	pSPARC->KE = pSPARC->KE/pSPARC->n_atom;
	pSPARC->TE = (pSPARC->PE + pSPARC->KE);
	// Extended System (Ionic system + Thermostat) energy
	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
		pSPARC->TE_ext = (0.5 * pSPARC->qmass * pow(pSPARC->xi_nose, 2.0) + pSPARC->dof * pSPARC->kB * pSPARC->thermos_T * pSPARC->snose)/pSPARC->n_atom + pSPARC->TE;
	}
    // Compute Ionic stress/pressure
	if(pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1)
	    Calculate_ionic_stress(pSPARC);

	//	Calculate_stress(pSPARC);
#ifdef DEBUG
	// MD Statistics
	double mean_TE_old, mean_KE_old, mean_PE_old, mean_U_old, mean_Eent_old, mean_Ti_old, mean_Te_old;
	int Count = pSPARC->MDCount + (pSPARC->RestartFlag == 0) ;
	mean_Te_old = pSPARC->mean_elec_T;
	mean_Ti_old = pSPARC->mean_ion_T;
	mean_TE_old = pSPARC->mean_TE;
	mean_KE_old = pSPARC->mean_KE;
	mean_PE_old = pSPARC->mean_PE;
	mean_U_old = pSPARC->mean_U;
	mean_Eent_old = pSPARC->mean_Entropy;
	pSPARC->mean_elec_T = (mean_Te_old * (Count - 1) + pSPARC->elec_T)/ Count;
	pSPARC->mean_ion_T = (mean_Ti_old * (Count - 1) + pSPARC->ion_T)/ Count;
	pSPARC->mean_TE = (mean_TE_old * (Count - 1) + pSPARC->TE)/ Count;
	pSPARC->mean_KE = (mean_KE_old * (Count - 1) + pSPARC->KE)/ Count;
	pSPARC->mean_PE = (mean_PE_old * (Count - 1) + pSPARC->PE)/ Count;
	pSPARC->mean_U = (mean_U_old * (Count - 1) + pSPARC->PE - pSPARC->Entropy/pSPARC->n_atom)/ Count;
	pSPARC->mean_Entropy = (mean_Eent_old * (Count - 1) + pSPARC->Entropy/pSPARC->n_atom)/ Count;
	pSPARC->std_elec_T = sqrt(fabs( ((pow(pSPARC->std_elec_T,2.0) + pow(mean_Te_old,2.0)) * (Count - 1) + pow(pSPARC->elec_T,2.0))/Count - pow(pSPARC->mean_elec_T,2.0) ));
	pSPARC->std_ion_T = sqrt(fabs( ((pow(pSPARC->std_ion_T,2.0) + pow(mean_Ti_old,2.0)) * (Count - 1) + pow(pSPARC->ion_T,2.0))/Count - pow(pSPARC->mean_ion_T,2.0) ));
	pSPARC->std_TE = sqrt(fabs( ((pow(pSPARC->std_TE,2.0) + pow(mean_TE_old,2.0)) * (Count - 1) + pow(pSPARC->TE,2.0))/Count - pow(pSPARC->mean_TE,2.0) ));
	pSPARC->std_KE = sqrt(fabs( ((pow(pSPARC->std_KE,2.0) + pow(mean_KE_old,2.0)) * (Count - 1) + pow(pSPARC->KE,2.0))/Count - pow(pSPARC->mean_KE,2.0) ));
	pSPARC->std_PE = sqrt(fabs( ((pow(pSPARC->std_PE,2.0) + pow(mean_PE_old,2.0)) * (Count - 1) + pow(pSPARC->PE,2.0))/Count - pow(pSPARC->mean_PE,2.0) ));
	pSPARC->std_U = sqrt(fabs( ((pow(pSPARC->std_U,2.0) + pow(mean_U_old,2.0)) * (Count - 1) + pow(pSPARC->PE - pSPARC->Entropy/pSPARC->n_atom,2.0))/Count - pow(pSPARC->mean_U,2.0) ));
	pSPARC->std_Entropy = sqrt(fabs( ((pow(pSPARC->std_Entropy,2.0) + pow(mean_Eent_old,2.0)) * (Count - 1) + pow(pSPARC->Entropy/pSPARC->n_atom,2.0))/Count - pow(pSPARC->mean_Entropy,2.0) ));
	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
		double mean_TEx_old = pSPARC->mean_TE_ext;
		pSPARC->mean_TE_ext = (mean_TEx_old * (Count - 1) + pSPARC->TE_ext)/ Count;
		pSPARC->std_TE_ext = sqrt(fabs( ((pow(pSPARC->std_TE_ext,2.0) + pow(mean_TEx_old,2.0)) * (Count - 1) + pow(pSPARC->TE_ext,2.0))/Count - pow(pSPARC->mean_TE_ext,2.0) ));
	}
#endif
	
	// Average and maximum speed
	int ityp, atm, cc = 0;
	double temp;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		maxvel[ityp] = 0.0;
		avgvel[ityp] = 0.0;
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			temp = fabs(sqrt(pow(pSPARC->ion_vel[3 * cc],2.0) + pow(pSPARC->ion_vel[3 * cc + 1],2.0) + pow(pSPARC->ion_vel[3 * cc + 2],2.0)));
			if(temp > maxvel[ityp])
				maxvel[ityp] = temp;
			avgvel[ityp] += temp;
			cc += 1;
		}
		avgvel[ityp] /= pSPARC->nAtomv[ityp];
	}

	// Average and minimum distance
	//*avgdis  = pow((pSPARC->range_x * pSPARC->range_y * pSPARC->range_z)/pSPARC->n_atom,1/3.0);
	int atm2, ityp2, cc2;
	cc = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
	    mindis[ityp] = 1000000000.0;
		for(atm = 0; atm < pSPARC->nAtomv[ityp] - 1; atm++){
			for(atm2 = atm + 1; atm2 < pSPARC->nAtomv[ityp]; atm2++){
				temp = fabs(sqrt(pow(pSPARC->atom_pos[3 * (atm + cc)] - pSPARC->atom_pos[3 * (atm2 + cc)],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 1] - pSPARC->atom_pos[3 * (atm2 + cc) + 1],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 2] - pSPARC->atom_pos[3 * (atm2 + cc) + 2],2.0) ));
				if(temp < mindis[ityp])
					mindis[ityp] = temp;
			}
		}
		cc += pSPARC->nAtomv[ityp];
	}
	cc = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes - 1; ityp++){
		cc2 = pSPARC->nAtomv[ityp] + cc;
		for(ityp2 = ityp + 1; ityp2 < pSPARC->Ntypes; ityp2++){
			mindis[pSPARC->Ntypes - 1 + ityp * (pSPARC->Ntypes - 1) + ityp2 - ityp] = 1000000000.0;
			for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
				for(atm2 = 0; atm2 < pSPARC->nAtomv[ityp2]; atm2++){
					temp = fabs(sqrt(pow(pSPARC->atom_pos[3 * (atm + cc)] - pSPARC->atom_pos[3 * (atm2 + cc2)],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 1] - pSPARC->atom_pos[3 * (atm2 + cc2) + 1],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 2] - pSPARC->atom_pos[3 * (atm2 + cc2) + 2],2.0) ));
					if(temp < mindis[pSPARC->Ntypes - 1 + ityp * (pSPARC->Ntypes - 1) + ityp2 - ityp])
						mindis[pSPARC->Ntypes - 1 + ityp * (pSPARC->Ntypes - 1) + ityp2 - ityp] = temp;
				}
			}
			cc2 += pSPARC->nAtomv[ityp2];
		}
		cc += pSPARC->nAtomv[ityp];
	}
}

/*
 @ brief: function to write all relevant quantities needed for MD restart
*/
void PrintMD(SPARC_OBJ *pSPARC, int Flag, int print_restart_typ) {
    FILE *mdout;
    if(!Flag){
    	mdout = fopen(pSPARC->restartC_Filename,"r+");
    	if(mdout == NULL){
        	printf("\nCannot open file \"%s\"\n",pSPARC->restartC_Filename);
        	exit(EXIT_FAILURE);
    	}
    	// Update MD Count
    	fprintf(mdout,":STOPCOUNT: %d\n", pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
    	fclose(mdout);

    }
    else{
    	if (print_restart_typ == 0) {
    		// Transfer the restart content to a file before overwriting
    		Rename_restart(pSPARC);
    		// Overwrite in the restart file
			mdout = fopen(pSPARC->restartC_Filename,"w");
		}
		else
			mdout = fopen(pSPARC->restart_Filename,"w");

    	// Print restart Count
    	fprintf(mdout,":MDSTEP: %d\n", pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
    	// Print atomic position
    	int atm;
    	fprintf(mdout,":R(Bohr):\n");
    	for(atm = 0; atm < pSPARC->n_atom; atm++){
    		fprintf(mdout,"%18.10E %18.10E %18.10E\n", pSPARC->atom_pos[3 * atm], pSPARC->atom_pos[3 * atm + 1], pSPARC->atom_pos[3 * atm + 2]);
    	}

    	// Print velocity
    	fprintf(mdout,":V(Bohr/atu):\n");
    	for(atm = 0; atm < pSPARC->n_atom; atm++){
    		fprintf(mdout,"%18.10E %18.10E %18.10E\n", pSPARC->ion_vel[3 * atm], pSPARC->ion_vel[3 * atm + 1], pSPARC->ion_vel[3 * atm + 2]);
    	}
    	// Print extended system parameters in case of NVT
    	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
    		fprintf(mdout,":snose: %.15g\n", pSPARC->snose);
    		fprintf(mdout,":xinose: %.15g\n", pSPARC->xi_nose);
    		fprintf(mdout,":TTHRMI(K): %.15g\n", pSPARC->thermos_T);
    	}
    	// Print temperature
    	fprintf(mdout,":TEL(K): %.15g\n", pSPARC->elec_T);
    	fprintf(mdout,":TIO(K): %.15g\n", pSPARC->ion_T);

		fclose(mdout);
	}
}

/*
@ brief function to read the restart file for MD restart
*/
void RestartMD(SPARC_OBJ *pSPARC) {
	int rank, position, l_buff = 0;
#ifdef DEBUG
	double t1, t2;
#endif
	char *buff;
	FILE *rst_fp = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Open the restart file
    if(!rank){
    	//char rst_Filename[L_STRING];
    	if( access(pSPARC->restart_Filename, F_OK ) != -1 )
			rst_fp = fopen(pSPARC->restart_Filename,"r");
		else if( access(pSPARC->restartC_Filename, F_OK ) != -1 )
			rst_fp = fopen(pSPARC->restartC_Filename,"r");
		else
			rst_fp = fopen(pSPARC->restartP_Filename,"r");
    }
#ifdef DEBUG
	if(!rank)
    	printf("Reading .restart file for MD\n");
#endif
    // Allocate memory for dynamic variables
	pSPARC->ion_vel = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	if (pSPARC->ion_vel == NULL) {
        printf("\nCannot allocate memory for ion velocity array!\n");
        exit(EXIT_FAILURE);
    }
	
	// Allocate memory for Pack and Unpack to be used later for broadcasting
	if(pSPARC->RestartFlag == 1)
	    l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + 5) * sizeof(double);
	else if(pSPARC->RestartFlag == -1)
	    l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + 3) * sizeof(double);

    buff = (char *)malloc( l_buff*sizeof(char) );
    if (buff == NULL) {
        printf("\nmemory cannot be allocated for buffer\n");
        exit(EXIT_FAILURE);
    }
	if(!rank){
		char str[L_STRING];
		int atm;
		while (fscanf(rst_fp,"%s",str) != EOF){
			if (strcmpi(str, ":STOPCOUNT:") == 0)
				fscanf(rst_fp,"%d",&pSPARC->StopCount);
			else if (strcmpi(str,":MDSTEP:") == 0)
				fscanf(rst_fp,"%d",&pSPARC->restartCount);
			else if (strcmpi(str,":R(Bohr):") == 0)
				for(atm = 0; atm < pSPARC->n_atom; atm++)
					fscanf(rst_fp,"%lf %lf %lf", &pSPARC->atom_pos[3 * atm], &pSPARC->atom_pos[3 * atm + 1], &pSPARC->atom_pos[3 * atm + 2]);
			else if (strcmpi(str,":V(Bohr/atu):") == 0)
				for(atm = 0; atm < pSPARC->n_atom; atm++)
					fscanf(rst_fp,"%lf %lf %lf", &pSPARC->ion_vel[3 * atm], &pSPARC->ion_vel[3 * atm + 1], &pSPARC->ion_vel[3 * atm + 2]);
			else if (strcmpi(str,":snose:") == 0 && pSPARC->RestartFlag == 1)
				fscanf(rst_fp,"%lf", &pSPARC->snose);
			else if (strcmpi(str,":xinose:") == 0 && pSPARC->RestartFlag == 1)
				fscanf(rst_fp,"%lf", &pSPARC->xi_nose);
			else if (strcmpi(str,":TEL(K):") == 0 && pSPARC->RestartFlag == 1)
				fscanf(rst_fp,"%lf", &pSPARC->elec_T);
			else if (strcmpi(str,":TIO(K):") == 0 && pSPARC->RestartFlag == 1)
				fscanf(rst_fp,"%lf", &pSPARC->ion_T);
			else if (strcmpi(str,":TTHRMI(K):") == 0 && pSPARC->RestartFlag == 1)
				fscanf(rst_fp,"%lf", &pSPARC->thermos_Ti);
		}
		fclose(rst_fp);

		// Pack the variables
		position = 0;
        MPI_Pack(&pSPARC->StopCount, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(&pSPARC->restartCount, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->atom_pos, 3*pSPARC->n_atom, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->ion_vel, 3*pSPARC->n_atom, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        if(pSPARC->RestartFlag == 1){
            MPI_Pack(&pSPARC->elec_T, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->ion_T, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
            	MPI_Pack(&pSPARC->snose, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->xi_nose, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->thermos_Ti, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            }
        }

        // broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
	} else{
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        // broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 1) printf(GRN "MPI_Bcast (.restart MD) packed buff of length %d took %.3f ms\n" RESET, l_buff,(t2-t1)*1000);
#endif
		// unpack the variables
        position = 0;
        MPI_Unpack(buff, l_buff, &position, &pSPARC->StopCount, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, &pSPARC->restartCount, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->atom_pos, 3*pSPARC->n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->ion_vel, 3*pSPARC->n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
        if(pSPARC->RestartFlag == 1){
            MPI_Unpack(buff, l_buff, &position, &pSPARC->elec_T, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->ion_T, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
        	MPI_Unpack(buff, l_buff, &position, &pSPARC->snose, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        	MPI_Unpack(buff, l_buff, &position, &pSPARC->xi_nose, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        	MPI_Unpack(buff, l_buff, &position, &pSPARC->thermos_Ti, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

	}
	if(pSPARC->RestartFlag == 1)
	    pSPARC->Beta = 1.0/(pSPARC->elec_T * pSPARC->kB);
	free(buff);
}



/*
@ brief: function to rename the restart file
*/
void Rename_restart(SPARC_OBJ *pSPARC) {
	if( access(pSPARC->restartC_Filename, F_OK ) != -1 )
	    rename(pSPARC->restartC_Filename, pSPARC->restartP_Filename);
}

