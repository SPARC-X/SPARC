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
#include "pressure.h"
#include "relax.h"
#include "electrostatics.h"
#include "eigenSolver.h" // Mesh2ChebDegree

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
		else if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0)
		    NPT_NH(pSPARC);
		else if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0)
			NPT_NP(pSPARC);
		else{
			if (!rank){
				printf("\nCannot recognize MDMeth = \"%s\"\n",pSPARC->MDMeth);
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
	// Variables for NPT_NH
	if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0 && pSPARC->RestartFlag != 1){
		int i;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if((pSPARC->NPT_NHnnos == 0) || (pSPARC->NPT_NHnnos > L_QMASS)) {
		    if (!rank) {
			    printf("Amount of thermostat variables cannot be zero or larger than %d. Please write valid amount of thermo variable (int) at head of input line.\n", L_QMASS);
			    printf("Example as below\n");
			    printf("NPT_NH_QMASS: 4 1500.0 1500.0 1500.0 1500.0\n");
			    exit(EXIT_FAILURE);
		    }
		}
        for (int subscript_NPTNH_qmass = 0; subscript_NPTNH_qmass < pSPARC->NPT_NHnnos; subscript_NPTNH_qmass++){
            if (pSPARC->NPT_NHqmass[subscript_NPTNH_qmass] == 0.0 && !rank){
                printf("Mass of thermostat variable %d cannot be zero. Please input valid amount of mass of thermo variable.\n", subscript_NPTNH_qmass);
                exit(EXIT_FAILURE);
            }
        }
        if(pSPARC->NPT_NHbmass == 0.0) {
            if (!rank) {
                printf("Mass of barostat variable cannot be zero. Please input valid amount of mass of baro variable.\n");
                exit(EXIT_FAILURE);
            }
        }
		if ((pSPARC->NPTscaleVecs[0] == 1) && (pSPARC->NPTscaleVecs[1] == 1) && (pSPARC->NPTscaleVecs[2] == 1)) pSPARC->NPTisotropicFlag = 1;
		else pSPARC->NPTisotropicFlag = 0;

		for (i = 0; i < pSPARC->NPT_NHnnos; i++) {
			pSPARC->glogs[i] = 0.0;
			pSPARC->vlogs[i] = 0.0;
			pSPARC->xlogs[i] = 0.0;
		}
		pSPARC->vlogv = 0.0;
		pSPARC->thermos_Ti = pSPARC->ion_T;
		pSPARC->thermos_T  = pSPARC->thermos_Ti;
		pSPARC->prtarget /= 29421.02648438959; // transfer from GPa to Ha/Bohr^3
		pSPARC->scale = 1.0;

	    pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
		pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
		pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
		pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
	}
	else if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0) { // restart
		if ((pSPARC->NPTscaleVecs[0] == 1) && (pSPARC->NPTscaleVecs[1] == 1) && (pSPARC->NPTscaleVecs[2] == 1)) pSPARC->NPTisotropicFlag = 1;
		else pSPARC->NPTisotropicFlag = 0;
		int i;
		for (i = 0; i < pSPARC->NPT_NHnnos; i++) {
			pSPARC->glogs[i] = 0.0;
		}
		// pSPARC->thermos_Ti = pSPARC->ion_T; // ion_T is decided by the kinetic energy of particles at that time step, changing with time
		pSPARC->thermos_Ti = pSPARC->elec_T;
		pSPARC->thermos_T  = pSPARC->thermos_Ti;
		pSPARC->prtarget /= 29421.02648438959; // transfer from GPa to Ha/Bohr^3

		pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
		pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
		pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
		pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
		Calculate_ionic_stress(pSPARC);
	}
	// Variables for NPT_NP
	if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 && pSPARC->RestartFlag != 1){
		int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x*pSPARC->range_y*pSPARC->range_z;
		pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
		pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
		pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
        pSPARC->maxTimeIter = 30;

        if(pSPARC->NPT_NP_bmass == 0.0) {
            if (!rank) {
                printf("Mass of barostat variable cannot be zero. Please input valid amount of mass of baro variable.\n");
                exit(EXIT_FAILURE);
            }
        }
        pSPARC->range_x_velo = 0.0; 
        pSPARC->G_NPT_NP[0] = pow(pSPARC->range_x, 2.0);
        pSPARC->range_y_velo = 0.0; 
        pSPARC->G_NPT_NP[1] = pow(pSPARC->range_y, 2.0);
        pSPARC->range_z_velo = 0.0; 
        pSPARC->G_NPT_NP[2] = pow(pSPARC->range_z, 2.0);

	    pSPARC->scale = 1.0; // better to move to initialization.c
	    // pSPARC->prtarget = 6.0 // input variable, unit in GPa
	    pSPARC->prtarget /= 29421.02648438959; // transfer from GPa to Ha/Bohr^3

	    if(pSPARC->NPT_NP_qmass == 0.0) {
            if (!rank) {
                printf("Mass of thermostat variable cannot be zero. Please input valid amount of mass of thermo variable.\n");
                exit(EXIT_FAILURE);
            }
        }
		if ((pSPARC->NPTscaleVecs[0] == 0) || (pSPARC->NPTscaleVecs[1] == 0) || (pSPARC->NPTscaleVecs[2] == 0)) {
			if (!rank) {
                printf("NPT-NP does not support specifying the rescaled lattice vectors!\n");
                exit(EXIT_FAILURE);
            }
		}
		
        pSPARC->S_NPT_NP = 1.0; 
        pSPARC->Sv_NPT_NP = 0.0; 

        pSPARC->thermos_Ti = pSPARC->ion_T;
		pSPARC->thermos_T  = pSPARC->thermos_Ti;

		pSPARC->Pm_NPT_NP[0] = pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)/pSPARC->S_NPT_NP*2*pSPARC->range_x*pSPARC->range_x_velo/pSPARC->G_NPT_NP[0];
		pSPARC->Pm_NPT_NP[1] = pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)/pSPARC->S_NPT_NP*2*pSPARC->range_y*pSPARC->range_y_velo/pSPARC->G_NPT_NP[1];
		pSPARC->Pm_NPT_NP[2] = pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)/pSPARC->S_NPT_NP*2*pSPARC->range_z*pSPARC->range_z_velo/pSPARC->G_NPT_NP[2];
    }
    else if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) { // restart
		int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x*pSPARC->range_y*pSPARC->range_z;
		pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
		pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
		pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
        pSPARC->maxTimeIter = 30;
        pSPARC->G_NPT_NP[0] = pSPARC->range_x * pSPARC->range_x;
        pSPARC->G_NPT_NP[1] = pSPARC->range_y * pSPARC->range_y;
        pSPARC->G_NPT_NP[2] = pSPARC->range_z * pSPARC->range_z;

        pSPARC->thermos_Ti = pSPARC->elec_T;
		pSPARC->thermos_T  = pSPARC->thermos_Ti;
        pSPARC->prtarget /= 29421.02648438959; // transfer from GPa to Ha/Bohr^3
		Calculate_ionic_stress(pSPARC);
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
@ brief function to perform NPT MD simulation with Nose hoover chain.
wherein number of particles, pressure and temperature are kept constant (equivalent to ionmov = 13, optcell = 1 in ABINIT)
reference: Glenn J. Martyna , Mark E. Tuckerman , Douglas J. Tobias & Michael L. Klein (1996).
Explicit reversible integrators for extended systems dynamics, Molecular Physics, 87:5
Tuckerman, M. E., Alejandre, J., López-Rendón, R., Jochim, A. L., & Martyna, G. J. (2006). 
A Liouville-operator derived measure-preserving integrator for molecular dynamics simulations in the isothermal–isobaric ensemble. 
Journal of Physics A: Mathematical and General, 39(19), 5629.
*/
void NPT_NH (SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(&pSPARC->pres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pSPARC->pres_i, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if ((pSPARC->MDCount != 1) || (pSPARC->RestartFlag == 1)) {
        // Update velocity of particles in the second half timestep
	    AccelVelocityParticle(pSPARC);
	    // Update velocity of virtual thermo and baro variables in the second half timestep
	    IsoPress(pSPARC);
	}

	// Update velocity of virtual thermo and baro variables in the first half timestep
	IsoPress(pSPARC);
	// Update velocity of particles in the first half timestep
	AccelVelocityParticle(pSPARC);
	// Update position of particles and size of cell in the timestep
	PositionParticleCell(pSPARC);
    // Reinitialize mesh size and related variables after changing size of cell
    reinitialize_mesh_NPT(pSPARC);
    
	// Charge extrapolation (for better rho_guess)
	elecDensExtrapolation(pSPARC);
	// Check position of atom near the boundary and apply wraparound in case of PBC, otherwise show error if the atom is too close to the boundary for bounded domain
	Check_atomlocation(pSPARC);
	// Compute DFT energy and forces by solving Kohn-Sham eigenvalue problem
	Calculate_electronicGroundState(pSPARC);
	pSPARC->elecgs_Count++;
    #ifdef DEBUG
        // Calculate Hamiltonian of the system.
        hamiltonian_NPT_NH(pSPARC);
        if (rank == 0){
            printf("\nend NPT timestep %d\n", pSPARC->MDCount + 1);
        }
    #endif

}

/*
@ brief function to update accelerations and velocities of particles changed by forces in NPT
*/
void AccelVelocityParticle (SPARC_OBJ *pSPARC) {
	int ityp, atm, count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->ion_accel[count * 3] = pSPARC->forces[count * 3]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 1] = pSPARC->forces[count * 3 + 1]/pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 2] = pSPARC->forces[count * 3 + 2]/pSPARC->Mass[ityp];
			pSPARC->ion_vel[count * 3] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3];
			pSPARC->ion_vel[count * 3 + 1] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3 + 1];
			pSPARC->ion_vel[count * 3 + 2] += 0.5 * pSPARC->MD_dt * pSPARC->ion_accel[count * 3 + 2];
			count ++;
		}
	}
}


/*
@ brief function to update accelerations, velocities and positions of virtual thermo and barostat variables in NPT
*/
void IsoPress(SPARC_OBJ *pSPARC) {
	int i, atm, ityp;
    int count = 0;
	double scale, gn1kt, odnf, modnf, ktemp;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    pSPARC->KE = 0.0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			count ++;
		}
	}

	pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
	scale = 1.0;
	ktemp = pSPARC->kB * pSPARC->thermos_T;
    gn1kt = (double)(pSPARC->dof + 1) * ktemp;

    modnf = 3.0/(double)pSPARC->dof;
    odnf = 1.0 + 3.0/(double)pSPARC->dof;
    // update accelerations of the virtual variables, 1st
    double glogv = 0.0;
	pSPARC->glogs[0] = ((2.0*pSPARC->KE + pSPARC->NPT_NHbmass * pow(pSPARC->vlogv, 2.0) - gn1kt) - pSPARC->vlogs[1]*pSPARC->vlogs[0]*pSPARC->NPT_NHqmass[0]) / pSPARC->NPT_NHqmass[0];
	glogv = (3*pSPARC->volumeCell*((pSPARC->pres - pSPARC->pres_i) - pSPARC->prtarget) + modnf*2*pSPARC->KE - pSPARC->vlogs[0]*pSPARC->vlogv*pSPARC->NPT_NHbmass) / pSPARC->NPT_NHbmass;
    // printf("\nrank %d glogv%12.9f = (odnf%12.6f*2*KE%12.9f+3*(pres%12.9f-prtarget%12.6f)*ucvol%12.9f)/bmass%12.6f\n", rank, glogv,odnf,pSPARC->KE,(pSPARC->pres - pSPARC->pres_i),pSPARC->prtarget,ucvol,pSPARC->NPT_NHbmass);

    // update velocities of the virtual variables, 1st
    double alocal;
    double nowvlogv = pSPARC->vlogv;
	for (i = 0; i < pSPARC->NPT_NHnnos; i++){
        pSPARC->vlogs[i] += pSPARC->MD_dt / 4.0 * pSPARC->glogs[i];
    }
	nowvlogv += pSPARC->MD_dt / 4.0 * glogv;
    // update accelerations of baro variable, 2nd
    alocal = exp(-pSPARC->MD_dt / 2.0 * (pSPARC->vlogs[0] + odnf * nowvlogv));
    scale = scale * alocal;
    pSPARC->KE = pSPARC->KE * pow(alocal, 2.0);
	glogv = (3*pSPARC->volumeCell*((pSPARC->pres - pSPARC->pres_i) - pSPARC->prtarget) + modnf*2*pSPARC->KE - pSPARC->vlogs[0]*nowvlogv*pSPARC->NPT_NHbmass) / pSPARC->NPT_NHbmass;
    // printf("\nrank %d glogv%12.9f = (odnf%12.6f*2*KE%12.9f+3*(pres%12.9f-prtarget%12.6f)*ucvol%12.9f)/bmass%12.6f\n", rank, glogv,odnf,pSPARC->KE,(pSPARC->pres - pSPARC->pres_i),pSPARC->prtarget,ucvol,pSPARC->NPT_NHbmass);
    // update thermostat position, for computing Hamiltonian
    for (i = 0; i < pSPARC->NPT_NHnnos; i++){
    	pSPARC->xlogs[i] += pSPARC->vlogs[i] * pSPARC->MD_dt / 2;
    }
    // update velocities of the baro variables, 2nd
	nowvlogv += pSPARC->MD_dt / 4.0 * glogv;
    // update accelerations of thermal variable, 2nd
	pSPARC->glogs[0] = ((2.0*pSPARC->KE + pSPARC->NPT_NHbmass * pow(pSPARC->vlogv, 2.0) - gn1kt) - pSPARC->vlogs[1]*pSPARC->vlogs[0]*pSPARC->NPT_NHqmass[0]) / pSPARC->NPT_NHqmass[0];
	for (i = 0; i < pSPARC->NPT_NHnnos; i++) {
    	pSPARC->vlogs[i] += pSPARC->MD_dt / 4.0 * pSPARC->glogs[i];
    }
    for (i = 1; i < pSPARC->NPT_NHnnos - 1; i++) {
		pSPARC->glogs[i] = (pSPARC->NPT_NHqmass[i - 1] * pow(pSPARC->vlogs[i - 1], 2.0) - ktemp - pSPARC->vlogs[i + 1]*pSPARC->vlogs[i]*pSPARC->NPT_NHqmass[i]) / pSPARC->NPT_NHqmass[i];
	}
	pSPARC->glogs[pSPARC->NPT_NHnnos - 1] = (pSPARC->NPT_NHqmass[pSPARC->NPT_NHnnos - 2]*pow(pSPARC->vlogs[pSPARC->NPT_NHnnos - 2], 2.0) - ktemp) / pSPARC->NPT_NHqmass[pSPARC->NPT_NHnnos - 1];
    // update velocities of particles
    count = 0;
    for (atm = 0; atm < pSPARC->n_atom; atm++){
		pSPARC->ion_vel[count * 3] *= scale;
		pSPARC->ion_vel[count * 3 + 1] *= scale;
		pSPARC->ion_vel[count * 3 + 2] *= scale;
		count ++;
	}
    // send calculated variables back to pSPARC
    pSPARC->vlogv = nowvlogv;
    #ifdef DEBUG
    if (rank == 0){
        printf("rank %d", rank);
        for (i = 0; i < pSPARC->NPT_NHnnos; i++){
            printf("\nvlogs[%d] is  %12.9f; glogs[%d] is  %12.9f \n", i, pSPARC->vlogs[i], i, pSPARC->glogs[i]);
        }
        printf("\nglogv is %12.9f\n", glogv);
        printf("\nvlogv is %12.9f\n", nowvlogv);
    }
    #endif
}

/*
@ brief function to update positions of particles and size of a cell in NPT
*/
void PositionParticleCell(SPARC_OBJ *pSPARC) {
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // update particle positions
	if (pSPARC->NPTisotropicFlag == 1) {
		int count, atm;
		double mttk_aloc, mttk_aloc2, polysh, mttk_bloc;
		mttk_aloc = exp(pSPARC->MD_dt / 2.0 * pSPARC->vlogv);
		mttk_aloc2 = pow(pSPARC->vlogv * pSPARC->MD_dt / 2.0, 2.0);

		polysh = (((1.0 / (6.0*20.0*42.0*72.0) * mttk_aloc2 + 1.0 / (6.0*20.0*42.0)) * mttk_aloc2 + 1.0 / (6.0*20.0)) * mttk_aloc2 + 1.0 / 6.0) * mttk_aloc2 + 1.0;
		mttk_bloc = mttk_aloc * polysh * pSPARC->MD_dt;
		count = 0;
		for(atm = 0; atm < pSPARC->n_atom; atm++){
			pSPARC->atom_pos[count * 3] = pSPARC->atom_pos[count * 3] * pow(mttk_aloc, 2.0) + pSPARC->ion_vel[count * 3] * mttk_bloc;
			pSPARC->atom_pos[count * 3 + 1] = pSPARC->atom_pos[count * 3 + 1] * pow(mttk_aloc, 2.0) + pSPARC->ion_vel[count * 3 + 1] * mttk_bloc;
			pSPARC->atom_pos[count * 3 + 2] = pSPARC->atom_pos[count * 3 + 2] * pow(mttk_aloc, 2.0) + pSPARC->ion_vel[count * 3 + 2] * mttk_bloc;
			count ++;
		}
    	// update size of cell
    	pSPARC->scale = exp(pSPARC->MD_dt * pSPARC->vlogv);
		pSPARC->range_x *= pSPARC->scale;
    	pSPARC->range_y *= pSPARC->scale;
    	pSPARC->range_z *= pSPARC->scale;
	}
	else { // only 1 or 2 lattice vectors can be rescaled
		int dim = pSPARC->NPTscaleVecs[0] + pSPARC->NPTscaleVecs[1] + pSPARC->NPTscaleVecs[2];
		double rescale = 3.0/(double)dim;

		int count, atm;
		double mttk_aloc, mttk_aloc2, polysh, mttk_bloc;
		mttk_aloc = exp(pSPARC->MD_dt / 2.0 * pSPARC->vlogv * rescale);
		mttk_aloc2 = pow(pSPARC->vlogv * pSPARC->MD_dt / 2.0 * rescale, 2.0);

		polysh = (((1.0 / (6.0*20.0*42.0*72.0) * mttk_aloc2 + 1.0 / (6.0*20.0*42.0)) * mttk_aloc2 + 1.0 / (6.0*20.0)) * mttk_aloc2 + 1.0 / 6.0) * mttk_aloc2 + 1.0;
		mttk_bloc = mttk_aloc * polysh * pSPARC->MD_dt;

		double *LatUVec = pSPARC->LatUVec; double *gradT = pSPARC->gradT;
		double carCoord[3]; double nonCarCoord[3]; // cartisian and reduced coordinate
		double carVelo[3]; double nonCarVelo[3]; // cartisian and reduced velocity
		count = 0;
		for(atm = 0; atm < pSPARC->n_atom; atm++){
			carCoord[0] = pSPARC->atom_pos[count * 3]; carCoord[1] = pSPARC->atom_pos[count * 3 + 1]; carCoord[2] = pSPARC->atom_pos[count * 3 + 2];
			carVelo[0] = pSPARC->ion_vel[count * 3]; carVelo[1] = pSPARC->ion_vel[count * 3 + 1]; carVelo[2] = pSPARC->ion_vel[count * 3 + 2];

			Cart2nonCart(gradT, carCoord, nonCarCoord);
			Cart2nonCart(gradT, carVelo, nonCarVelo);

			if (pSPARC->NPTscaleVecs[0] == 1) nonCarCoord[0] = nonCarCoord[0] * pow(mttk_aloc, 2.0) + nonCarVelo[0] * mttk_bloc;
			else nonCarCoord[0] = nonCarCoord[0] + nonCarVelo[0]*pSPARC->MD_dt;

			if (pSPARC->NPTscaleVecs[1] == 1) nonCarCoord[1] = nonCarCoord[1] * pow(mttk_aloc, 2.0) + nonCarVelo[1] * mttk_bloc;
			else nonCarCoord[1] = nonCarCoord[1] + nonCarVelo[1]*pSPARC->MD_dt;

			if (pSPARC->NPTscaleVecs[2] == 1) nonCarCoord[2] = nonCarCoord[2] * pow(mttk_aloc, 2.0) + nonCarVelo[2] * mttk_bloc;
			else nonCarCoord[2] = nonCarCoord[2] + nonCarVelo[2]*pSPARC->MD_dt;

			nonCart2Cart(LatUVec, carCoord, nonCarCoord);

			pSPARC->atom_pos[count * 3] = carCoord[0]; pSPARC->atom_pos[count * 3 + 1] = carCoord[1]; pSPARC->atom_pos[count * 3 + 2] = carCoord[2];
			count ++;
		}
    	// update size of cell
    	pSPARC->scale = exp(pSPARC->MD_dt * pSPARC->vlogv * rescale);
		if (pSPARC->NPTscaleVecs[0] == 1) pSPARC->range_x *= pSPARC->scale;
    	if (pSPARC->NPTscaleVecs[1] == 1) pSPARC->range_y *= pSPARC->scale;
    	if (pSPARC->NPTscaleVecs[2] == 1) pSPARC->range_z *= pSPARC->scale;
	}
    #ifdef DEBUG
        if(rank == 0){
            printf("rank %d", rank);
	        printf("scale of cell is %12.9f\n", pSPARC->scale);
	        printf("pSPARC->range is (%12.9f, %12.9f, %12.9f)\n", pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
        }
    #endif
}

/**
 * @brief   Write the re-initialized parameters into the output file.
 */
void write_output_reinit_NPT(SPARC_OBJ *pSPARC) {
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    FILE *output_fp = fopen(pSPARC->OutFilename,"a");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
        exit(EXIT_FAILURE);
    }
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                         Reinitialized parameters                          \n");
    fprintf(output_fp,"***************************************************************************\n");
	if (pSPARC->Flag_latvec_scale == 0)
		fprintf(output_fp,"CELL: %.15g %.15g %.15g \n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
	else
		fprintf(output_fp,"LATVEC_SCALE: %.15g %.15g %.15g \n",pSPARC->range_x/pSPARC->initialLatVecLength[0],pSPARC->range_y/pSPARC->initialLatVecLength[1],pSPARC->range_z/pSPARC->initialLatVecLength[2]);
    fprintf(output_fp,"CHEB_DEGREE: %d\n",pSPARC->ChebDegree);
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                             Reinitialization                              \n");
    fprintf(output_fp,"***************************************************************************\n");
    if ( (fabs(pSPARC->delta_x-pSPARC->delta_y) <=1e-12) && (fabs(pSPARC->delta_x-pSPARC->delta_z) <=1e-12)
        && (fabs(pSPARC->delta_y-pSPARC->delta_z) <=1e-12) ) {
        fprintf(output_fp,"Mesh spacing                       :  %.6g (Bohr)\n",pSPARC->delta_x);
    } else {
        fprintf(output_fp,"Mesh spacing in x-direction        :  %.6g (Bohr)\n",pSPARC->delta_x);
        fprintf(output_fp,"Mesh spacing in y-direction        :  %.6g (Bohr)\n",pSPARC->delta_y);
        fprintf(output_fp,"Mesh spacing in z direction        :  %.6g (Bohr)\n",pSPARC->delta_z);
    }

    fclose(output_fp);
}

/*
@ brief reinitialize related variables after the size changing of cell.
*/
void reinitialize_mesh_NPT(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


#ifdef DEBUG
        double t1, t2;
#endif

    int p;
#ifdef DEBUG
    // scaling factor
    double scal = pSPARC->scale; // the ratio between length
    if(rank == 0){
    	printf("scal: %12.6f\n", scal);
    }
#endif
    
	pSPARC->delta_x = pSPARC->range_x/(pSPARC->numIntervals_x);
	pSPARC->delta_y = pSPARC->range_y/(pSPARC->numIntervals_y);
	pSPARC->delta_z = pSPARC->range_z/(pSPARC->numIntervals_z);
    

    pSPARC->dV = pSPARC->delta_x * pSPARC->delta_y * pSPARC->delta_z * pSPARC->Jacbdet;

#ifdef DEBUG
    if (!rank) {
        // printf("Volume: %12.6f\n", vol);
        printf("CELL  : %12.6f\t%12.6f\t%12.6f\n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
        printf("COORD : \n");
        for (int i = 0; i < 3 * pSPARC->n_atom; i++) {
            printf("%12.6f\t",pSPARC->atom_pos[i]);
            if (i%3==2 && i>0) printf("\n");
        }
        printf("\n");
    }
#endif

    int FDn = pSPARC->order / 2;

    // 1st derivative weights including mesh
    double dx_inv, dy_inv, dz_inv;
    dx_inv = 1.0 / pSPARC->delta_x;
    dy_inv = 1.0 / pSPARC->delta_y;
    dz_inv = 1.0 / pSPARC->delta_z;
    for (p = 1; p < FDn + 1; p++) {
        pSPARC->D1_stencil_coeffs_x[p] = pSPARC->FDweights_D1[p] * dx_inv;
        pSPARC->D1_stencil_coeffs_y[p] = pSPARC->FDweights_D1[p] * dy_inv;
        pSPARC->D1_stencil_coeffs_z[p] = pSPARC->FDweights_D1[p] * dz_inv;
    }

    // 2nd derivative weights including mesh
    double dx2_inv, dy2_inv, dz2_inv;
    dx2_inv = 1.0 / (pSPARC->delta_x * pSPARC->delta_x);
    dy2_inv = 1.0 / (pSPARC->delta_y * pSPARC->delta_y);
    dz2_inv = 1.0 / (pSPARC->delta_z * pSPARC->delta_z);

    // Stencil coefficients for mixed derivatives
    if (pSPARC->cell_typ == 0) {
        for (p = 0; p < FDn + 1; p++) {
            pSPARC->D2_stencil_coeffs_x[p] = pSPARC->FDweights_D2[p] * dx2_inv;
            pSPARC->D2_stencil_coeffs_y[p] = pSPARC->FDweights_D2[p] * dy2_inv;
            pSPARC->D2_stencil_coeffs_z[p] = pSPARC->FDweights_D2[p] * dz2_inv;
        }
    } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
        for (p = 0; p < FDn + 1; p++) {
            pSPARC->D2_stencil_coeffs_x[p] = pSPARC->lapcT[0] * pSPARC->FDweights_D2[p] * dx2_inv;
            pSPARC->D2_stencil_coeffs_y[p] = pSPARC->lapcT[4] * pSPARC->FDweights_D2[p] * dy2_inv;
            pSPARC->D2_stencil_coeffs_z[p] = pSPARC->lapcT[8] * pSPARC->FDweights_D2[p] * dz2_inv;
            pSPARC->D2_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_12 d/dx(df/dy)
            pSPARC->D2_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_13 d/dx(df/dz)
            pSPARC->D2_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // 2*T_23 d/dy(df/dz)
            pSPARC->D1_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dy_inv; // d/dx(2*T_12 df/dy) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
            pSPARC->D1_stencil_coeffs_yx[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // d/dy(2*T_12 df/dx) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
            pSPARC->D1_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dz_inv; // d/dx(2*T_13 df/dz) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
            pSPARC->D1_stencil_coeffs_zx[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // d/dz(2*T_13 df/dx) used in d/dz(2*T_13 df/dz + 2*T_23 df/dy)
            pSPARC->D1_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dz_inv; // d/dy(2*T_23 df/dz) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
            pSPARC->D1_stencil_coeffs_zy[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // d/dz(2*T_23 df/dy) used in d/dz(2*T_12 df/dx + 2*T_23 df/dy)
        }
    }

    // maximum eigenvalue of -0.5 * Lap (only with periodic boundary conditions)
    if(pSPARC->cell_typ == 0) {
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        pSPARC->MaxEigVal_mhalfLap = pSPARC->D2_stencil_coeffs_x[0]
                                   + pSPARC->D2_stencil_coeffs_y[0]
                                   + pSPARC->D2_stencil_coeffs_z[0];
        double scal_x, scal_y, scal_z;
        scal_x = (pSPARC->Nx - pSPARC->Nx % 2) / (double) pSPARC->Nx;
        scal_y = (pSPARC->Ny - pSPARC->Ny % 2) / (double) pSPARC->Ny;
        scal_z = (pSPARC->Nz - pSPARC->Nz % 2) / (double) pSPARC->Nz;
        for (int p = 1; p < FDn + 1; p++) {
            pSPARC->MaxEigVal_mhalfLap += 2.0 * (pSPARC->D2_stencil_coeffs_x[p] * cos(M_PI*p*scal_x)
                                               + pSPARC->D2_stencil_coeffs_y[p] * cos(M_PI*p*scal_y)
                                               + pSPARC->D2_stencil_coeffs_z[p] * cos(M_PI*p*scal_z));
        }
        pSPARC->MaxEigVal_mhalfLap *= -0.5;
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("Max eigenvalue of -0.5*Lap is %.13f, time taken: %.3f ms\n",
            pSPARC->MaxEigVal_mhalfLap, (t2-t1)*1e3);
#endif
    }

    double h_eff = 0.0;
    if (fabs(pSPARC->delta_x - pSPARC->delta_y) < 1E-12 &&
        fabs(pSPARC->delta_y - pSPARC->delta_z) < 1E-12) {
        h_eff = pSPARC->delta_x;
    } else {
        // find effective mesh s.t. it has same spectral width
        h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
    }

    // find Chebyshev polynomial degree based on max eigenvalue (spectral width)
    if (pSPARC->ChebDegree < 0) {
        pSPARC->ChebDegree = Mesh2ChebDegree(h_eff);
#ifdef DEBUG
        if (!rank && h_eff < 0.1) {
            printf("#WARNING: for mesh less than 0.1, the default Chebyshev polynomial degree might not be enought!\n");
        }
        if (!rank) printf("h_eff = %.2f, npl = %d\n", h_eff,pSPARC->ChebDegree);
#endif
    } else {
#ifdef DEBUG
        if (!rank) printf("Chebyshev polynomial degree (provided by user): npl = %d\n",pSPARC->ChebDegree);
#endif
    }

    // default Kerker tolerance
    if (pSPARC->TOL_PRECOND < 0.0) { // kerker tol not provided by user
        pSPARC->TOL_PRECOND = (h_eff * h_eff) * 1e-3;
    }


    // re-calculate k-point grid
    Calculate_kpoints(pSPARC);

    // re-calculate local k-points array
    if (pSPARC->Nkpts >= 1 && pSPARC->kptcomm_index != -1) {
        Calculate_local_kpoints(pSPARC);
    }

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // re-calculate pseudocharge density cutoff ("rb")
    Calculate_PseudochargeCutoff(pSPARC);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Calculating rb for all atom types took %.3f ms\n",(t2-t1)*1000);
#endif


    // write reinitialized parameters into output file
    if (rank == 0) {
        write_output_reinit_NPT(pSPARC);
    }

}


/* 
@ brief: calculate Hamiltonian of the NPT system.
*/
void hamiltonian_NPT_NH(SPARC_OBJ *pSPARC){
	double kineticBaro, kineticTher, potentialBaro, potentialTher;
	double hamiltonian;
	int i;

	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
	hamiltonian = pSPARC->KE + pSPARC->Etot;
	kineticBaro = pow(pSPARC->vlogv, 2.0) * pSPARC->NPT_NHbmass * 0.5;
	kineticTher = 0.0;
	for (i = 0; i < pSPARC->NPT_NHnnos; i++){
        kineticTher += pow(pSPARC->vlogs[i], 2.0) * pSPARC->NPT_NHqmass[i] * 0.5;
	}
    double ktemp;
	pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
	ktemp = pSPARC->kB * pSPARC->thermos_T;
	potentialBaro = pSPARC->prtarget * pSPARC->volumeCell;
	potentialTher = (double)pSPARC->dof * ktemp * pSPARC->xlogs[0];
    for (i = 1; i < pSPARC->NPT_NHnnos; i++){
    	potentialTher += ktemp * pSPARC->xlogs[i];
    }
    if (rank == 0) {
    	printf("\n");
    	printf("rank %d", rank);
    	printf("ENERGY of time step %d\n", pSPARC->MDCount + 1);
	    printf("kinetic energy (Ha)             : %12.9f\n", pSPARC->KE);
	    printf("potential energy (Ha)           : %12.9f\n", pSPARC->Etot);
	    printf("barostat kinetic energy (Ha)    : %12.9f\n", kineticBaro);
	    printf("thermostat kinetic energy (Ha)  : %12.9f\n", kineticTher);
	    printf("barostat potential energy (Ha)  : %12.9f\n", potentialBaro);
	    printf("thermostat potential energy (Ha): %12.9f\n", potentialTher);
	    hamiltonian += kineticBaro + kineticTher + potentialBaro + potentialTher;
	    printf("Hamiltonian (Ha)                : %12.9f\n", hamiltonian);
	}
}



/*
@ brief function to perform NPT MD simulation with Nose-Poincare, wherein number of particles, pressure and temperature are kept constant
Reference: Hernández, E. "Metric-tensor flexible-cell algorithm for isothermal–isobaric molecular dynamics simulations." 
The Journal of Chemical Physics 115, no. 22 (2001): 10282-10290.
*/
void NPT_NP (SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(&pSPARC->pres, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pSPARC->pres_i, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// calculate Hamiltonian of the NPT_NP system.
	initialize_Hamiltonian(pSPARC);
	// updating momentums of thermostat and barostat variables and particles in the first half step in NPT_NP.
	updateMomentum_FirstHalf(pSPARC);
	// updating momentums of thermostat and barostat variables and particles in the second half step in NPT_NP.
	updateMomentum_SecondHalf(pSPARC);
	// updating value of thermostat variable and length of cell and positions of particles in NPT_NP.
	updatePosition(pSPARC);
	// Reinitialize mesh size and related variables after changing size of cell
    reinitialize_mesh_NPT(pSPARC);

	// Charge extrapolation (for better rho_guess)
	elecDensExtrapolation(pSPARC);
	// Check position of atom near the boundary and apply wraparound in case of PBC, otherwise show error if the atom is too close to the boundary for bounded domain
	Check_atomlocation(pSPARC);
	// Compute DFT energy and forces by solving Kohn-Sham eigenvalue problem
	Calculate_electronicGroundState(pSPARC);
	pSPARC->elecgs_Count++;
	#ifdef DEBUG
		if (!rank) printf("\nend NPT_NP timestep %d\n", pSPARC->MDCount + 1);
	#endif
}

/*
 @ brief: initialize momentum of barostat variables Pm, and calculate Hamiltonian of the system
*/
void initialize_Hamiltonian(SPARC_OBJ *pSPARC){
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double ktemp;

	pSPARC->Pm_NPT_NP[0] = pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0) / pSPARC->S_NPT_NP * 2.0 *pSPARC->range_x*pSPARC->range_x_velo / pow(pSPARC->G_NPT_NP[0], 2.0);
	pSPARC->Pm_NPT_NP[1] = pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0) / pSPARC->S_NPT_NP * 2.0 *pSPARC->range_y*pSPARC->range_y_velo / pow(pSPARC->G_NPT_NP[1], 2.0);
	pSPARC->Pm_NPT_NP[2] = pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0) / pSPARC->S_NPT_NP * 2.0 *pSPARC->range_z*pSPARC->range_z_velo / pow(pSPARC->G_NPT_NP[2], 2.0);

	pSPARC->KE = 0.0;
	int ityp, atm;
	int count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			count ++;
		}
	}

	pSPARC->Kther = 0.5 * pSPARC->NPT_NP_qmass * pow(pSPARC->Sv_NPT_NP, 2.0);
	ktemp = pSPARC->kB * pSPARC->thermos_T;
	pSPARC->Uther = (double)pSPARC->dof * log(pSPARC->S_NPT_NP) * ktemp;

	double a[3];
	for (int i = 0; i < 3; i++){
		a[i] = 1.0 / (pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)) * pSPARC->Pm_NPT_NP[i] * pSPARC->G_NPT_NP[i];
	}
	pSPARC->Kbaro = 0.5 * pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0) * (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
	pSPARC->Ubaro = pSPARC->prtarget * pSPARC->volumeCell;

	pSPARC->Hamiltonian_NPT_NP = pSPARC->Etot;
	pSPARC->Hamiltonian_NPT_NP += pSPARC->KE + pSPARC->Kther + pSPARC->Uther + pSPARC->Kbaro + pSPARC->Ubaro;
	if ((pSPARC->MDCount == 1)  && (pSPARC->RestartFlag != 1)) {
		pSPARC->init_Hamil_NPT_NP = pSPARC->Hamiltonian_NPT_NP;
	}
	#ifdef DEBUG
	if (rank == 0) {
    	printf("\n");
    	printf("rank %d", rank);
    	printf("ENERGY of time step %d\n", pSPARC->MDCount + 1);
	    printf("kinetic energy (Ha)             : %12.9f\n", pSPARC->KE);
	    printf("potential energy (Ha)           : %12.9f\n", pSPARC->Etot);
	    printf("barostat kinetic energy (Ha)    : %12.9f\n", pSPARC->Kbaro);
	    printf("thermostat kinetic energy (Ha)  : %12.9f\n", pSPARC->Kther);
	    printf("barostat potential energy (Ha)  : %12.9f\n", pSPARC->Ubaro);
	    printf("thermostat potential energy (Ha): %12.9f\n", pSPARC->Uther);
	    printf("Hamiltonian (Ha)                : %12.9f\n", pSPARC->Hamiltonian_NPT_NP);
	}
	#endif
}

/*
 @ brief: update momentum of thermostat and barostat variables in a half step, update momentum/mass (not velocity!) of particles in a step
*/
void updateMomentum_FirstHalf(SPARC_OBJ *pSPARC) {
	double ktemp;
	double Ga1[3];
	double B[3];
	double Ga3[3];
	double PmA[3];
	double Sa;

	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// update momentum of thermostat variable in a half step
	ktemp = pSPARC->kB * pSPARC->thermos_T;
	Sa = (pSPARC->KE - pSPARC->Etot - pSPARC->dof*ktemp*(log(pSPARC->S_NPT_NP) + 1) - pSPARC->Kbaro - pSPARC->Ubaro - pSPARC->Kther + pSPARC->init_Hamil_NPT_NP)/pSPARC->NPT_NP_qmass;
	pSPARC->Sv_NPT_NP += Sa * pSPARC->MD_dt / 2.0;
	#ifdef DEBUG
	if (rank == 0) {
		printf("Sa is %12.9f\n", Sa);
		printf("Sv_NPT_NP in the 1st half step is %12.9f\n", pSPARC->Sv_NPT_NP);
	}
	#endif
	// update momentum of barostat variables in a half step
	for (int i = 0; i < 3; i++){
		Ga1[i] = -(pSPARC->pres - pSPARC->pres_i) * pSPARC->volumeCell / 2.0 / pSPARC->G_NPT_NP[i];
		B[i] = 1.0 / (pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)) * pow(pSPARC->Pm_NPT_NP[i],2.0) * pSPARC->G_NPT_NP[i];
		Ga3[i] = (pSPARC->prtarget * pSPARC->volumeCell / 2.0 - pSPARC->Kbaro) / pSPARC->G_NPT_NP[i];
		PmA[i] = Ga1[i] + B[i] + Ga3[i];
		pSPARC->Pm_NPT_NP[i] -= pSPARC->MD_dt * pSPARC->S_NPT_NP / 2.0 * PmA[i];
		#ifdef DEBUG
		if (rank == 0){
			printf("PmA[%d] is %12.9f\n", i, PmA[i]);
			printf("pSPARC->Pm_NPT_NP[%d] in 1st half step is %12.9f\n", i, pSPARC->Pm_NPT_NP[i]);
		}
		#endif
	}
	// update momentum/mass of particles (reminder: not velocity!) in a step
	int ityp, atm;
	int count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->ion_accel[count * 3] = pSPARC->forces[count * 3] / pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 1] = pSPARC->forces[count * 3 + 1] / pSPARC->Mass[ityp];
			pSPARC->ion_accel[count * 3 + 2] = pSPARC->forces[count * 3 + 2] / pSPARC->Mass[ityp];
			pSPARC->ion_vel[count * 3] = pSPARC->ion_vel[count * 3] * pSPARC->S_NPT_NP + pSPARC->MD_dt * pSPARC->S_NPT_NP * pSPARC->ion_accel[count * 3];
			pSPARC->ion_vel[count * 3 + 1] = pSPARC->ion_vel[count * 3 + 1] * pSPARC->S_NPT_NP + pSPARC->MD_dt * pSPARC->S_NPT_NP * pSPARC->ion_accel[count * 3 + 1];
			pSPARC->ion_vel[count * 3 + 2] = pSPARC->ion_vel[count * 3 + 2] * pSPARC->S_NPT_NP + pSPARC->MD_dt * pSPARC->S_NPT_NP * pSPARC->ion_accel[count * 3 + 2]; 
			// for now they are not velocity!
			count ++;
		}
	}
}

/*
 @ brief: update momentum of thermostat and barostat variables in the second half step
*/
void updateMomentum_SecondHalf(SPARC_OBJ *pSPARC) {
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double PmTmp[3], PmNew[3];
	int i;
	for (i = 0; i < 3; i++){
		PmTmp[i] = pSPARC->Pm_NPT_NP[i];
	}
	// update momentum of barostat variables in the second half time step
	int judge = 0;
	int timeIter = 0;
	double G1[3], Gatmp1[3], Gatmp2[3], Gatmp3[3], PmAtmp[3];
	double KbaroTmp = 0;
	while (judge == 0) {
		timeIter++;
		KbaroTmp = 0;
		for (i = 0; i < 3; i++) {
			G1[i] = 1 / (pSPARC->NPT_NP_bmass * pow(pSPARC->volumeCell, 2.0)) * PmTmp[i] * pSPARC->G_NPT_NP[i];
			KbaroTmp += (pSPARC->NPT_NP_bmass * pow(pSPARC->volumeCell, 2.0)) / 2 * pow(G1[i],2.0);
		}
		for (i = 0; i < 3; i++) {
			Gatmp1[i] = -(pSPARC->pres - pSPARC->pres_i) * pSPARC->volumeCell / 2.0 / pSPARC->G_NPT_NP[i];
			Gatmp2[i] = G1[i] * PmTmp[i];
			Gatmp3[i] = (pSPARC->prtarget * pSPARC->volumeCell / 2.0 - KbaroTmp) / pSPARC->G_NPT_NP[i];
			PmAtmp[i] = Gatmp1[i] + Gatmp2[i] + Gatmp3[i];
			PmNew[i] = pSPARC->Pm_NPT_NP[i] - pSPARC->MD_dt / 2.0 * pSPARC->S_NPT_NP * PmAtmp[i];
		}
		for (i = 0; i < 3; i++) {
			judge = 1;
			if (fabs(PmNew[i] - PmTmp[i]) > 1e-7){
				judge = 0;
			}
			PmTmp[i] = PmNew[i];
		}
		if (timeIter > pSPARC->maxTimeIter){
			judge = 1;
			if (rank == 0)
				printf("Reminder: The barostat momentum Pm_NPT_NP does not converge in %d timesteps.\n", pSPARC->maxTimeIter);
		}
	}
	for (i = 0; i < 3; i++) {
		pSPARC->Pm_NPT_NP[i] = PmTmp[i];
		#ifdef DEBUG
		if (rank == 0)
			printf("pSPARC->Pm_NPT_NP[%d] in 2nd half step is %12.9f\n", i, pSPARC->Pm_NPT_NP[i]);
		#endif
	}

	// update thermostat velocity in the second half time step
	pSPARC->KE = 0.0;
	int ityp, atm;
	int count = 0;
	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->KE += 0.5 * pSPARC->Mass[ityp] * (pow(pSPARC->ion_vel[count * 3], 2.0) + pow(pSPARC->ion_vel[count * 3 + 1], 2.0) + pow(pSPARC->ion_vel[count * 3 + 2], 2.0));
			count ++;
		}
	}
	pSPARC->KE /= pow(pSPARC->S_NPT_NP, 2.0); // from momentum/mass to true velocity
	pSPARC->Kbaro = 0.0;
	for (i = 0; i < 3; i++) {
		G1[i] = 1.0 / (pSPARC->NPT_NP_bmass * pow(pSPARC->volumeCell, 2.0)) * PmTmp[i] * pSPARC->G_NPT_NP[i];
		pSPARC->Kbaro += (pSPARC->NPT_NP_bmass * pow(pSPARC->volumeCell, 2.0)) / 2.0 * pow(G1[i],2.0);
	}
	double factor;
	double ktemp = pSPARC->kB * pSPARC->thermos_T;
	factor = pSPARC->MD_dt / 2.0 * (pSPARC->dof*ktemp*(log(pSPARC->S_NPT_NP) + 1) - pSPARC->KE + pSPARC->Etot + pSPARC->Kbaro + pSPARC->Ubaro - pSPARC->init_Hamil_NPT_NP) - pSPARC->NPT_NP_qmass*pSPARC->Sv_NPT_NP;
	#ifdef DEBUG
	if (rank == 0)
		printf("factor is %12.9f\n", factor);
	#endif
	if ((1.0 - factor*pSPARC->MD_dt/pSPARC->NPT_NP_qmass) < 0.0){
		if (rank == 0)
			printf("The mass of thermostat variable NPT_NP_qmass is too small. Please try a larger thermostat mass.");
		exit(EXIT_FAILURE);
	}
	pSPARC->Sv_NPT_NP = -2.0 * factor / (pSPARC->NPT_NP_qmass * (1.0 + sqrt(1.0 - factor * pSPARC->MD_dt / pSPARC->NPT_NP_qmass)));
	#ifdef DEBUG
	if (rank == 0) 
		printf("Sv_NPT_NP in the 2nd half step is %12.9f\n", pSPARC->Sv_NPT_NP);
	#endif
}

/*
 @ brief: update positions of particles, value of thermostat variable and barostat variables in the step
*/
void updatePosition(SPARC_OBJ *pSPARC) {
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int judge = 0;
	// update value of thermostat variable S_NPT_NP
	double Stemp = pSPARC->S_NPT_NP;
	double Snew;
	int timeIter = 0;
	while (judge == 0) {
		timeIter++;
		Snew = pSPARC->S_NPT_NP + pSPARC->MD_dt / 2.0 * (pSPARC->S_NPT_NP + Stemp) * pSPARC->Sv_NPT_NP;
		if (fabs(Snew - Stemp) < 1e-7) {
			judge = 1;
		}
		Stemp = Snew;
		if (timeIter > pSPARC->maxTimeIter) {
			judge = 1;
			if (rank == 0)
				printf("Reminder: The value of thermostat variable S_NPT_NP does not converge in %d iterations.\n", pSPARC->maxTimeIter);
		}
	}
	#ifdef DEBUG
	if (rank == 0)
		printf("Stemp is %12.9f\n", Stemp);
	#endif
	// update values of barostat variables G_NPT_NP
	double Gtmp[3], Gnew[3], Gpig[3], GpigOld[3];
	int i;
	for (i = 0; i < 3; i++) {
		Gtmp[i] = pSPARC->G_NPT_NP[i];
		GpigOld[i] = pow(pSPARC->G_NPT_NP[i], 2.0) * pSPARC->Pm_NPT_NP[i];
	}
	judge = 0; timeIter = 0;
	while (judge == 0){
		timeIter++;
		for (i = 0; i < 3; i++) {
			Gpig[i] = pow(Gtmp[i], 2.0) * pSPARC->Pm_NPT_NP[i];
			Gnew[i] = pSPARC->G_NPT_NP[i] + pSPARC->MD_dt / 2.0 * (pSPARC->S_NPT_NP/(pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0))*GpigOld[i] + Stemp/(pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0))*Gpig[i]);
		}
		judge = 1;
		for (i = 0; i < 3; i++) {
			if (fabs(Gnew[i] - Gtmp[i]) > 1e-7) {
				judge = 0;
			}
			Gtmp[i] = Gnew[i];
		}
		if (timeIter > pSPARC->maxTimeIter) {
			judge = 1;
			if (rank == 0)
				printf("Reminder: The barostat variables G_NPT_NP do not converge in %d iterations.\n", pSPARC->maxTimeIter);
		}
	}
	double G3[3];
	for (i = 0; i < 3; i++) {
		pSPARC->G_NPT_NP[i] = Gtmp[i];
		G3[i] = pow(Gtmp[i], 2.0) * pSPARC->Pm_NPT_NP[i];
	}
	#ifdef DEBUG
	if (rank == 0) {
		printf("pSPARC->G_NPT_NP[0] is %12.9f\n", pSPARC->G_NPT_NP[0]);
		printf("pSPARC->G_NPT_NP[1] is %12.9f\n", pSPARC->G_NPT_NP[1]);
		printf("pSPARC->G_NPT_NP[2] is %12.9f\n", pSPARC->G_NPT_NP[2]);
	}
	#endif
	// update side lengths of cells and velocities of them
	pSPARC->scale = sqrt(pSPARC->G_NPT_NP[0]) / pSPARC->range_x; // isotropic expansion
	#ifdef DEBUG
	if (rank == 0)
		printf("pSPARC->scale is %12.9f.\n", pSPARC->scale);
	#endif
	pSPARC->range_x = sqrt(pSPARC->G_NPT_NP[0]);
	pSPARC->range_y = sqrt(pSPARC->G_NPT_NP[1]);
	pSPARC->range_z = sqrt(pSPARC->G_NPT_NP[2]);
	pSPARC->volumeCell = pSPARC->Jacbdet*pSPARC->range_x*pSPARC->range_y*pSPARC->range_z;
	pSPARC->range_x_velo = G3[0] * Stemp / (2.0*pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)) / pSPARC->range_x;
	pSPARC->range_y_velo = G3[1] * Stemp / (2.0*pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)) / pSPARC->range_y;
	pSPARC->range_z_velo = G3[2] * Stemp / (2.0*pSPARC->NPT_NP_bmass*pow(pSPARC->volumeCell, 2.0)) / pSPARC->range_z;
	// update positions of particles, and restore the values of particle velocities
	int count = 0;
	int atm;
	for(atm = 0; atm < pSPARC->n_atom; atm++){
		pSPARC->atom_pos[count * 3] = (pSPARC->atom_pos[count * 3]) * pSPARC->scale + pSPARC->MD_dt/2.0 * (pSPARC->ion_vel[count * 3]/pSPARC->S_NPT_NP + pSPARC->ion_vel[count * 3]/Stemp); //
		pSPARC->atom_pos[count * 3 + 1] = (pSPARC->atom_pos[count * 3 + 1]) * pSPARC->scale + pSPARC->MD_dt/2.0 * (pSPARC->ion_vel[count * 3 + 1]/pSPARC->S_NPT_NP + pSPARC->ion_vel[count * 3 + 1]/Stemp); //
		pSPARC->atom_pos[count * 3 + 2] = (pSPARC->atom_pos[count * 3 + 2]) * pSPARC->scale + pSPARC->MD_dt/2.0 * (pSPARC->ion_vel[count * 3 + 2]/pSPARC->S_NPT_NP + pSPARC->ion_vel[count * 3 + 2]/Stemp); //
		pSPARC->ion_vel[count * 3] /= Stemp;
		pSPARC->ion_vel[count * 3 + 1] /= Stemp;
		pSPARC->ion_vel[count * 3 + 2] /= Stemp; 
		count ++;
	}
	pSPARC->S_NPT_NP = Stemp;
}

/**
 * @ brief: function to convert non cartesian to cartesian coordinates, from initialization.c
 */
void nonCart2Cart(double *LatUVec, double *carCoord, double *nonCarCoord) {
    carCoord[0] = LatUVec[0] * nonCarCoord[0] + LatUVec[3] * nonCarCoord[1] + LatUVec[6] * nonCarCoord[2];
    carCoord[1] = LatUVec[1] * nonCarCoord[0] + LatUVec[4] * nonCarCoord[1] + LatUVec[7] * nonCarCoord[2];
    carCoord[2] = LatUVec[2] * nonCarCoord[0] + LatUVec[5] * nonCarCoord[1] + LatUVec[8] * nonCarCoord[2];
}

/**
 * @brief: function to convert cartesian to non cartesian coordinates, from initialization.c
 */
void Cart2nonCart(double *gradT, double *carCoord, double *nonCarCoord) {
    nonCarCoord[0] = gradT[0] * carCoord[0] + gradT[1] * carCoord[1] + gradT[2] * carCoord[2];
    nonCarCoord[1] = gradT[3] * carCoord[0] + gradT[4] * carCoord[1] + gradT[5] * carCoord[2];
    nonCarCoord[2] = gradT[6] * carCoord[0] + gradT[7] * carCoord[1] + gradT[8] * carCoord[2];
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
						printf("ERROR: Atom number %d has crossed the boundary in %d direction",atm, dir);
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
		fprintf(output_md,":Desc_KENIG: Kinetic energy: 3/2 N k T of ideal gas at temperature T = TIO. Unit=Ha/atom \n"
						  "     where N = number of particles, k = Boltzmann constant\n");
    	fprintf(output_md,":Desc_FEN: Free energy F = U - TS. FEN = UEN + TSEN. Unit=Ha/atom \n");
    	fprintf(output_md,":Desc_UEN: Internal energy. Unit=Ha/atom \n");
    	fprintf(output_md,":Desc_TSEN: Electronic entropic contribution -TS to free energy F = U - TS. Unit=Ha/atom \n");
    	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
    		fprintf(output_md,":Desc_TENX: Total energy of extended system. Unit=Ha/atom \n");
    	}
		if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0 || strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			if (pSPARC->Flag_latvec_scale == 0)
				fprintf(output_md,":Desc_CELL: lengths of three lattice vectors. Unit = Bohr \n");
			else
				fprintf(output_md,":Desc_LATVEC_SCALE: ratio of cell lattice vectors over input lattice vector. Unit = 1 \n");
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
		double ken_ig = 0.0;
		ken_ig = 3.0/2.0 * pSPARC->n_atom * pSPARC->kB * pSPARC->ion_T;

	    // Print Temperature and energy
	    fprintf(output_md,":TEL: %.15g\n", pSPARC->elec_T);
	    fprintf(output_md,":TIO: %.15g\n", pSPARC->ion_T);
		fprintf(output_md,":TEN: %18.10E\n", pSPARC->TE);
		fprintf(output_md,":KEN: %18.10E\n", pSPARC->KE);
		fprintf(output_md,":KENIG:%18.10E\n",ken_ig/pSPARC->n_atom);
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

		// Print length of lattice vectors
		if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0 || strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			if (pSPARC->Flag_latvec_scale == 0)
				fprintf(output_md,":CELL: %18.10E %18.10E %18.10E\n", pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
			else
				fprintf(output_md,":LATVEC_SCALE: %18.10E %18.10E %18.10E\n", pSPARC->range_x/pSPARC->initialLatVecLength[0], pSPARC->range_y/pSPARC->initialLatVecLength[1], pSPARC->range_z/pSPARC->initialLatVecLength[2]);
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
		// Print extended system parameters in case of NPT-NH
    	if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0){
    		int thenos;
    		fprintf(mdout,":NPT_NH_QMASS: %d", pSPARC->NPT_NHnnos);
    		for (thenos = 0; thenos < pSPARC->NPT_NHnnos; thenos++) {
    			if (thenos%5 == 0){
                        fprintf(mdout,"\n");
                    }
                    fprintf(mdout," %.15g",pSPARC->NPT_NHqmass[thenos]);
    		}
    		fprintf(mdout,"\n");
    		fprintf(mdout,":NPT_NH_BMASS: %.15g\n",pSPARC->NPT_NHbmass);
    		fprintf(mdout,":NPT_NH_vlogs: %d", pSPARC->NPT_NHnnos); // velocities of virtual thermal parameters
    		for (thenos = 0; thenos < pSPARC->NPT_NHnnos; thenos++) {
    			if (thenos%5 == 0){
                        fprintf(mdout,"\n");
                    }
                    fprintf(mdout," %.15g", pSPARC->vlogs[thenos]);
    		}
    		fprintf(mdout,"\n");
    		fprintf(mdout,":NPT_NH_vlogv: %.15g\n", pSPARC->vlogv); // velocities of the virtual baro parameter
    		fprintf(mdout,":NPT_NH_xlogs: %d", pSPARC->NPT_NHnnos); // positions of virtual thermal parameters
    		for (thenos = 0; thenos < pSPARC->NPT_NHnnos; thenos++) {
    			if (thenos%5 == 0){
                        fprintf(mdout,"\n");
                    }
                    fprintf(mdout," %.15g", pSPARC->xlogs[thenos]);
    		}
    		fprintf(mdout,"\n");
    		if (pSPARC->Flag_latvec_scale == 0)
    			fprintf(mdout,":CELL: %.15g %.15g %.15g\n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z); //(no variable for position of barostat variable)
			else 
				fprintf(mdout,":LATVEC_SCALE: %.15g %.15g %.15g\n",pSPARC->range_x/pSPARC->initialLatVecLength[0],pSPARC->range_y/pSPARC->initialLatVecLength[1],pSPARC->range_z/pSPARC->initialLatVecLength[2]);
    		fprintf(mdout,":TARGET_PRESSURE: %.15g GPa\n",pSPARC->prtarget * 29421.02648438959);
    	}
    	// Print extended system parameters in case of NPT-NP
    	if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
    		fprintf(mdout,":NPT_NP_QMASS: %.15g\n", pSPARC->NPT_NP_qmass);
    		fprintf(mdout,":NPT_NP_BMASS: %.15g\n", pSPARC->NPT_NP_bmass);
    		fprintf(mdout,":NPT_NP_Sv: %.15g\n", pSPARC->Sv_NPT_NP); // velocity of virtual thermal parameter
    		fprintf(mdout,":NPT_NP_Pm: %.15g %.15g %.15g\n", pSPARC->Pm_NPT_NP[0], pSPARC->Pm_NPT_NP[1], pSPARC->Pm_NPT_NP[2]); // velocity of virtual baro parameter
    		fprintf(mdout,":NPT_NP_S: %.15g\n", pSPARC->S_NPT_NP); // value of virtual thermal parameter
    		fprintf(mdout,":NPT_NP_range_x_velo: %.15g\n", pSPARC->range_x_velo); // velocity of virtual x baro parameter
    		fprintf(mdout,":NPT_NP_range_y_velo: %.15g\n", pSPARC->range_y_velo); // velocity of virtual y baro parameter
    		fprintf(mdout,":NPT_NP_range_z_velo: %.15g\n", pSPARC->range_z_velo); // velocity of virtual z baro parameter
    		if (pSPARC->Flag_latvec_scale == 0)
    			fprintf(mdout,":CELL: %.15g %.15g %.15g\n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z); //(no variable for position of barostat variable)
			else 
				fprintf(mdout,":LATVEC_SCALE: %.15g %.15g %.15g\n",pSPARC->range_x/pSPARC->initialLatVecLength[0],pSPARC->range_y/pSPARC->initialLatVecLength[1],pSPARC->range_z/pSPARC->initialLatVecLength[2]);
    		fprintf(mdout,":TARGET_PRESSURE: %.15g GPa\n",pSPARC->prtarget * 29421.02648438959);
    		fprintf(mdout,":NPT_NP_ini_Hamiltonian: %.15g\n", pSPARC->init_Hamil_NPT_NP);
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
	if(pSPARC->RestartFlag == 1){
	    // l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + 5) * sizeof(double);
	    if (strcmpi(pSPARC->MDMeth,"NPT_NH") == 0){
	    	l_buff = (2 + 1) * sizeof(int) + (6 * pSPARC->n_atom + (5 + 3*pSPARC->NPT_NHnnos + 8)) * sizeof(double);
	    }
	    else if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
	    	l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + (5 + 13)) * sizeof(double);
	    }
	    else {
	    	l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + 5) * sizeof(double);
	    }
	}
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
			if (strcmpi(pSPARC->MDMeth,"NPT_NH") == 0) {
				if (strcmpi(str,":NPT_NH_QMASS:") == 0) { 
            		fscanf(rst_fp,"%d",&pSPARC->NPT_NHnnos);
            		for (int subscript_NPTNH_qmass = 0; subscript_NPTNH_qmass < pSPARC->NPT_NHnnos; subscript_NPTNH_qmass++){
            		    fscanf(rst_fp,"%lf",&pSPARC->NPT_NHqmass[subscript_NPTNH_qmass]);
            		}
            		fscanf(rst_fp, "%*[^\n]\n");
            	}
            	else if (strcmpi(str,":NPT_NH_vlogs:") == 0) { 
            		fscanf(rst_fp,"%d",&pSPARC->NPT_NHnnos);
            		for (int subscript_NPTNH_qmass = 0; subscript_NPTNH_qmass < pSPARC->NPT_NHnnos; subscript_NPTNH_qmass++){
            		    fscanf(rst_fp,"%lf",&pSPARC->vlogs[subscript_NPTNH_qmass]);
            		}
            		fscanf(rst_fp, "%*[^\n]\n");
            	}
            	else if (strcmpi(str,":NPT_NH_xlogs:") == 0) { 
            		fscanf(rst_fp,"%d",&pSPARC->NPT_NHnnos);
            		for (int subscript_NPTNH_qmass = 0; subscript_NPTNH_qmass < pSPARC->NPT_NHnnos; subscript_NPTNH_qmass++){
            		    fscanf(rst_fp,"%lf",&pSPARC->xlogs[subscript_NPTNH_qmass]);
            		}
            		fscanf(rst_fp, "%*[^\n]\n");
            	}

            	else if (strcmpi(str,":NPT_NH_BMASS:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->NPT_NHbmass);
            	else if (strcmpi(str,":NPT_NH_vlogv:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->vlogv);
            	else if (strcmpi(str,":CELL:") == 0) {
            		double nowRange_x, nowRange_y, nowRange_z;
        		    fscanf(rst_fp,"%lf", &nowRange_x); fscanf(rst_fp,"%lf", &nowRange_y); fscanf(rst_fp,"%lf", &nowRange_z);
        		    fscanf(rst_fp, "%*[^\n]\n");

        		    if (pSPARC->NPTscaleVecs[0] == 1) pSPARC->scale = nowRange_x / pSPARC->range_x; // now NPT_NH only support expanding with a constant ratio
					else if (pSPARC->NPTscaleVecs[1] == 1) pSPARC->scale = nowRange_y / pSPARC->range_y; 
					else pSPARC->scale = nowRange_z / pSPARC->range_z;
        		    
        		    pSPARC->range_x = nowRange_x;
        		    pSPARC->range_y = nowRange_y;
        		    pSPARC->range_z = nowRange_z;
        		}
				else if (strcmpi(str,":LATVEC_SCALE:") == 0) {
					double nowLatScale_x, nowLatScale_y, nowLatScale_z;
					fscanf(rst_fp,"%lf", &nowLatScale_x); fscanf(rst_fp,"%lf", &nowLatScale_y); fscanf(rst_fp,"%lf", &nowLatScale_z);
					fscanf(rst_fp, "%*[^\n]\n");

					double nowRange_x, nowRange_y, nowRange_z;
					pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
					pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
					pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
					nowRange_x = pSPARC->initialLatVecLength[0]*nowLatScale_x;
					nowRange_y = pSPARC->initialLatVecLength[1]*nowLatScale_y;
					nowRange_z = pSPARC->initialLatVecLength[2]*nowLatScale_z;

					if (pSPARC->NPTscaleVecs[0] == 1) pSPARC->scale = nowRange_x / pSPARC->range_x; // now NPT_NH only support expanding with a constant ratio
					else if (pSPARC->NPTscaleVecs[1] == 1) pSPARC->scale = nowRange_y / pSPARC->range_y; 
					else pSPARC->scale = nowRange_z / pSPARC->range_z;
        		    
        		    pSPARC->range_x = nowRange_x;
        		    pSPARC->range_y = nowRange_y;
        		    pSPARC->range_z = nowRange_z;
				}
        		else if (strcmpi(str,":TARGET_PRESSURE:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->prtarget);
			}
			if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) {
            	if (strcmpi(str,":NPT_NP_QMASS:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->NPT_NP_qmass);
            	else if (strcmpi(str,":NPT_NP_Sv:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->Sv_NPT_NP);
            	else if (strcmpi(str,":NPT_NP_S:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->S_NPT_NP);
            	else if (strcmpi(str,":NPT_NP_BMASS:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->NPT_NP_bmass);
            	else if (strcmpi(str,":NPT_NP_range_x_velo:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->range_x_velo);
            	else if (strcmpi(str,":NPT_NP_range_y_velo:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->range_y_velo);
            	else if (strcmpi(str,":NPT_NP_range_z_velo:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->range_z_velo);
            	else if (strcmpi(str,":CELL:") == 0) {
            		double nowRange_x, nowRange_y, nowRange_z;
        		    fscanf(rst_fp,"%lf", &nowRange_x); fscanf(rst_fp,"%lf", &nowRange_y); fscanf(rst_fp,"%lf", &nowRange_z);
        		    fscanf(rst_fp, "%*[^\n]\n");

        		    pSPARC->scale = nowRange_x / pSPARC->range_x; // now NPT_NP only support homogeneous expansion,
        		    // compute scale from x is enough
        		    pSPARC->range_x = nowRange_x;
        		    pSPARC->range_y = nowRange_y;
        		    pSPARC->range_z = nowRange_z;
            	}
				else if (strcmpi(str,":LATVEC_SCALE:") == 0) {
					double nowLatScale_x, nowLatScale_y, nowLatScale_z;
					fscanf(rst_fp,"%lf", &nowLatScale_x); fscanf(rst_fp,"%lf", &nowLatScale_y); fscanf(rst_fp,"%lf", &nowLatScale_z);
					fscanf(rst_fp, "%*[^\n]\n");

					double nowRange_x, nowRange_y, nowRange_z;
					pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
					pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
					pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
					nowRange_x = pSPARC->initialLatVecLength[0]*nowLatScale_x;
					nowRange_y = pSPARC->initialLatVecLength[1]*nowLatScale_y;
					nowRange_z = pSPARC->initialLatVecLength[2]*nowLatScale_z;
					pSPARC->scale = nowRange_x / pSPARC->range_x; // now NPT_NP only support homogeneous expansion,
        		    // compute scale from x is enough
        		    pSPARC->range_x = nowRange_x;
        		    pSPARC->range_y = nowRange_y;
        		    pSPARC->range_z = nowRange_z;
				}
            	else if (strcmpi(str,":TARGET_PRESSURE:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->prtarget);
            	else if (strcmpi(str,":NPT_NP_ini_Hamiltonian:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->init_Hamil_NPT_NP);
			}
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
			else if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0){
            	MPI_Pack(&pSPARC->NPT_NHnnos, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(pSPARC->NPT_NHqmass, pSPARC->NPT_NHnnos, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(pSPARC->vlogs, pSPARC->NPT_NHnnos, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(pSPARC->xlogs, pSPARC->NPT_NHnnos, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);

            	MPI_Pack(&pSPARC->NPT_NHbmass, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->vlogv, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->scale, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_x, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_y, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_z, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->prtarget, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            }
            else if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
            	MPI_Pack(&pSPARC->NPT_NP_qmass, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->Sv_NPT_NP, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->S_NPT_NP, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->NPT_NP_bmass, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_x_velo, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_y_velo, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_z_velo, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->scale, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_x, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_y, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->range_z, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->prtarget, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->init_Hamil_NPT_NP, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
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
        if (rank == 0) printf(GRN "MPI_Bcast (.restart MD) packed buff of length %d took %.3f ms\n" RESET, l_buff,(t2-t1)*1000);
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
			else if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0){
            	MPI_Unpack(buff, l_buff, &position, &pSPARC->NPT_NHnnos, 1, MPI_INT, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, pSPARC->NPT_NHqmass, pSPARC->NPT_NHnnos, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, pSPARC->vlogs, pSPARC->NPT_NHnnos, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, pSPARC->xlogs, pSPARC->NPT_NHnnos, MPI_DOUBLE, MPI_COMM_WORLD);

        		MPI_Unpack(buff, l_buff, &position, &pSPARC->NPT_NHbmass, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->vlogv, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->scale, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_x, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_y, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_z, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->prtarget, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            else if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
            	MPI_Unpack(buff, l_buff, &position, &pSPARC->NPT_NP_qmass, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->Sv_NPT_NP, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->S_NPT_NP, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        		MPI_Unpack(buff, l_buff, &position, &pSPARC->NPT_NP_bmass, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_x_velo, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_y_velo, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_z_velo, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->scale, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_x, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_y, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->range_z, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->prtarget, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->init_Hamil_NPT_NP, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

	}
	if(pSPARC->RestartFlag == 1) {
	    pSPARC->Beta = 1.0/(pSPARC->elec_T * pSPARC->kB);
	    if((strcmpi(pSPARC->MDMeth,"NPT_NH") == 0) || (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0)) {
            reinitialize_mesh_NPT(pSPARC);
        }
	}
	free(buff);
}



/*
@ brief: function to rename the restart file
*/
void Rename_restart(SPARC_OBJ *pSPARC) {
	if( access(pSPARC->restartC_Filename, F_OK ) != -1 )
	    rename(pSPARC->restartC_Filename, pSPARC->restartP_Filename);
}

