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
#include <stdbool.h>
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h> 
    #include <lapacke.h>
#endif

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
#include "readfiles.h"
#include "eigenSolver.h" // Mesh2ChebDegree
#include "readfiles.h"
#include "sparc_mlff_interface.h"

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
	pSPARC->t_run_md = MPI_Wtime();

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

	Calculate_Properties(pSPARC);
	//Calculate_electronicGroundState(pSPARC);
	Initialize_MD(pSPARC);

	
	pSPARC->MD_maxStep = pSPARC->restartCount + pSPARC->MD_Nstep;

	int check1 = (pSPARC->PrintMDout == 1 && !rank);
	int check2 = (pSPARC->Printrestart == 1 && !rank);

	if((strcmpi(pSPARC->MDMeth,"NPT_NP") != 0) && (strcmpi(pSPARC->MDMeth,"NPH") != 0)){  // For NPT_NP and NPH, a similar but slightly modified printing scheme is run, as the printing instances in this below setup is not compatible with NPT_NP and NPH ensemble, i.e. the printing setup in the below scheme (if condition) if used for NPT_NP and NPH would be corresponding to time = t in positions and potential energies, but only time = t-dt/2 for momenta, velocities and kinetic energies. So the quantities are not in-sync when being printed to 'log' or 'aimd' file, so this setup is done separately for NPT_NP and NPH with printing happening when all quantities are in-sync and belong to same time = t
		FILE *output_md, *output_fp;
		// File output_md stores all the desirable properties from a MD run
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

	} else {  // Carry out the above procedure in slightly modified printing manner for NPT_NP and NPH ensemble
		// File output_md stores all the desirable properties from a MD run
		FILE *output_md;

		if (pSPARC->PrintMDout == 1 && !rank && pSPARC->MD_Nstep > 0){
			output_md = fopen(pSPARC->MDFilename,"w");
			if (output_md == NULL) {
				printf("\nCannot open file \"%s\"\n",pSPARC->MDFilename);
				exit(EXIT_FAILURE);
			}
			pSPARC->MDCount = -1;
			Print_fullMD(pSPARC, output_md, avgvel, maxvel, mindis); // prints the QOI in the output_md file
			pSPARC->MDCount++;

			// if(pSPARC->RestartFlag == 0){
			// 	fprintf(output_md,":MDSTEP: %d\n", 1);
			// 	fprintf(output_md,":MDTM: %.2f\n", (MPI_Wtime() - t_init));
			// 	MD_QOI(pSPARC, avgvel, maxvel, mindis); // calculates the quantities of interest in an MD simulation
			// 	Print_fullMD(pSPARC, output_md, avgvel, maxvel, mindis); // prints the QOI in the output_md file
			// }

			fclose(output_md);
		}

		pSPARC->MDCount++;
		pSPARC->elecgs_Count++;

		// Perform MD until maxStep is reached or total walltime is hit
		int Count = pSPARC->MDCount + pSPARC->restartCount; // Count is the MD step no. to be performed
		#ifdef DEBUG
			if(!rank) printf(":MDSTEP: %d\n", Count);
		#endif

		t_acc = (MPI_Wtime() - t_init)/60;// tracks time in minutes
		while(Count <= pSPARC->MD_maxStep && (t_acc + 1.0*(MPI_Wtime() - t_init)/60) < pSPARC->TWtime){
			t_init = MPI_Wtime();
			
			NPT_NP_and_NPH(pSPARC, avgvel, maxvel, mindis);

			// This is true, restart happens from the instance when position are at time = t,  whereas momenta are lagging behind by dt/2, so they belong to time = t-dt/2  (unlike printing all quantities to an MD file, which happens when all quantities belong to time = t,  and in-sync with each other)
			if(check2 && !(Count % pSPARC->Printrestart_fq)) // printrestart_fq is the frequency at which the restart file is written
				PrintMD(pSPARC, 1, print_restart_typ);
			
			if(access("SPARC.stop", F_OK ) != -1 ){ // If a .stop file exists in the folder then the run will be terminated
				pSPARC->MDCount++;
				break;
			}
			pSPARC->MDCount ++;  
			Count ++;  
			t_acc += (MPI_Wtime() - t_init)/60;
		}  // end while loop

	}


	if(check2){
		pSPARC->MDCount --;
		print_restart_typ = 1;
		PrintMD(pSPARC, 1, print_restart_typ);
	}
    free(avgvel);
    free(maxvel);
    free(mindis);
	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH") == 0){
		free(pSPARC->ion_vel_fractional);
		free(pSPARC->Pm_ion);
	}
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
		// pSPARC->thermos_Ti = pSPARC->elec_T; // It comes from restart file
		pSPARC->thermos_T  = pSPARC->thermos_Ti;
		pSPARC->prtarget /= 29421.02648438959; // transfer from GPa to Ha/Bohr^3

		pSPARC->volumeCell = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
		pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0]*pSPARC->LatVec[0] + pSPARC->LatVec[1]*pSPARC->LatVec[1] + pSPARC->LatVec[2]*pSPARC->LatVec[2]);
		pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3]*pSPARC->LatVec[3] + pSPARC->LatVec[4]*pSPARC->LatVec[4] + pSPARC->LatVec[5]*pSPARC->LatVec[5]);
		pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6]*pSPARC->LatVec[6] + pSPARC->LatVec[7]*pSPARC->LatVec[7] + pSPARC->LatVec[8]*pSPARC->LatVec[8]);
		Calculate_ionic_stress(pSPARC);
	}
	// Variables for NPT_NP and NPH
	if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH") == 0) && pSPARC->RestartFlag != 1){
		int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        pSPARC->volumeCell = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;
		if (pSPARC->Flag_latvec_scale == 1){
			pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0] * pSPARC->LatVec[0] + pSPARC->LatVec[1] * pSPARC->LatVec[1] + pSPARC->LatVec[2]* pSPARC->LatVec[2]);
			pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3] * pSPARC->LatVec[3] + pSPARC->LatVec[4] * pSPARC->LatVec[4] + pSPARC->LatVec[5] * pSPARC->LatVec[5]);
			pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6] * pSPARC->LatVec[6] + pSPARC->LatVec[7] * pSPARC->LatVec[7] + pSPARC->LatVec[8] * pSPARC->LatVec[8]);
		}
		else{
			pSPARC->initialLatVecLength[0] = 1; pSPARC->initialLatVecLength[1] = 1; pSPARC->initialLatVecLength[2] = 1;
		}
		
		pSPARC->maxTimeIter = 100;

        if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			if(pSPARC->NPT_NP_bmass == 0.0) {
				if (!rank) {
					printf("Mass of barostat variable cannot be zero in NPT_NP. Please input valid amount of mass of baro variable.\n");
					exit(EXIT_FAILURE);
				}
			}
		}

		if(strcmpi(pSPARC->MDMeth,"NPH") == 0){
			if(pSPARC->NPH_bmass == 0.0) {
				if (!rank) {
					printf("Mass of barostat variable cannot be zero in NPH. Please input valid amount of mass of baro variable.\n");
					exit(EXIT_FAILURE);
				}
			}
		}

		fetch_MD_cell_ingredients(pSPARC, false);
	    pSPARC->pressure_external /= CONST_HA_BOHR3_GPA; // transfer from GPa to Ha/Bohr^3
		for (int i = 0; i < 6; i++){
			pSPARC->stress_external[i] /= CONST_HA_BOHR3_GPA; // transfer from GPa to Ha/Bohr^3
		}
		pSPARC->external_stress_cartesian[0] = pSPARC->stress_external[0]; pSPARC->external_stress_cartesian[4] = pSPARC->stress_external[1]; pSPARC->external_stress_cartesian[8] = pSPARC->stress_external[2];
		pSPARC->external_stress_cartesian[1] = pSPARC->stress_external[3]; pSPARC->external_stress_cartesian[2] = pSPARC->stress_external[4]; pSPARC->external_stress_cartesian[3] = pSPARC->stress_external[3];
		pSPARC->external_stress_cartesian[5] = pSPARC->stress_external[5]; pSPARC->external_stress_cartesian[6] = pSPARC->stress_external[4]; pSPARC->external_stress_cartesian[7] = pSPARC->stress_external[5];
		
		if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 && (pSPARC->NPT_NP_qmass == 0.0)){
            if (!rank) {
                printf("Mass of thermostat variable cannot be zero in NPT_NP ensemble. Please input valid amount of mass of thermo variable\n");
                exit(EXIT_FAILURE);
            }
        }

		if(strcmpi(pSPARC->MDMeth,"NPH") == 0){
			pSPARC->NPT_NP_qmass = 0; // Internally we are setting NPT_NP_qmass to 0; as rest of the functionality is entirely same as NPT_NP so we are using the same functions for NPH
			if (!rank) {
                printf("Mass of thermostat variable is zero in NPH ensemble\n");
            }
		}
	
        pSPARC->SNOSE[0] = 1.0;  // S variable of the thermostat @ current timestep (t)
        pSPARC->SNOSE[1] = 0.0;  // Ps/Ms or initial velocity of thermostat	
		pSPARC->SNOSE[2] = pSPARC->SNOSE[0];  // S variable of the thermostat @ previous timestep (t-dt)
		
        pSPARC->thermos_Ti = pSPARC->ion_T;
		pSPARC->thermos_T  = pSPARC->thermos_Ti;

		pSPARC->ion_vel_fractional = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
		if (pSPARC->ion_vel_fractional == NULL) {
			fprintf(stderr, "Error: Memory allocation failed for fractional ionic velocity array.\n");
			exit(EXIT_FAILURE);
		}
		pSPARC->Pm_ion = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
		if (pSPARC->Pm_ion == NULL) {
			fprintf(stderr, "Error: Memory allocation failed for ionic momentum array.\n");
			exit(EXIT_FAILURE);
		}
    }
    else if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH") == 0) { // restart
		int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		pSPARC->maxTimeIter = 100;
		
		//fetch_MD_cell_ingredients_restart(pSPARC);

        pSPARC->pressure_external /= CONST_HA_BOHR3_GPA; // transfer from GPa to Ha/Bohr^3
		for (int i = 0; i < 6; i++){
			pSPARC->stress_external[i] /= CONST_HA_BOHR3_GPA; // transfer from GPa to Ha/Bohr^3
		}// transfer from GPa to Ha/Bohr^3
		pSPARC->external_stress_cartesian[0] = pSPARC->stress_external[0]; pSPARC->external_stress_cartesian[4] = pSPARC->stress_external[1]; pSPARC->external_stress_cartesian[8] = pSPARC->stress_external[2];
		pSPARC->external_stress_cartesian[1] = pSPARC->stress_external[3]; pSPARC->external_stress_cartesian[2] = pSPARC->stress_external[4]; pSPARC->external_stress_cartesian[3] = pSPARC->stress_external[3];
		pSPARC->external_stress_cartesian[5] = pSPARC->stress_external[5]; pSPARC->external_stress_cartesian[6] = pSPARC->stress_external[4]; pSPARC->external_stress_cartesian[7] = pSPARC->stress_external[5];
		
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, pSPARC->volumeCell, pSPARC->external_stress_cartesian, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, pSPARC->external_stress_lattice, 3);
		
		//double temp_mat[9];
		//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, pSPARC->volumeCell, pSPARC->external_stress_cartesian, 3, pSPARC->reciprocal_lattice, 3, 0.0, temp_mat, 3);
		//cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_lattice, 3, temp_mat, 3, 0.0, pSPARC->external_stress_lattice, 3);
	

		if(strcmpi(pSPARC->MDMeth,"NPH")==0){
			pSPARC->NPT_NP_qmass = 0;
			pSPARC->SNOSE[0] = 1.0;
			pSPARC->SNOSE[1] = 0.0;
			pSPARC->SNOSE[2] = pSPARC->SNOSE[0];
		}
		// pSPARC->thermos_Ti = pSPARC->elec_T;
		pSPARC->thermos_T = pSPARC->thermos_Ti; // It comes from restart file!

		//Calculate_ionic_stress(pSPARC);

		//Pm_ion memory allocation already done within 'RestartMD' function as that is being MPI communicated
		pSPARC->ion_vel_fractional = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
		if (pSPARC->ion_vel_fractional == NULL) {
			fprintf(stderr, "Error: Memory allocation failed for ionic fractional velocity array.\n");
			exit(EXIT_FAILURE);
		}
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
		pSPARC->mean_elec_T = pSPARC->mean_ion_T = pSPARC->mean_TE = pSPARC->mean_KE = pSPARC->mean_PE = pSPARC->mean_U = pSPARC->mean_Entropy = 0.0; pSPARC->mean_internal_pressure = 0.0;
		pSPARC->std_elec_T = pSPARC->std_ion_T = pSPARC->std_TE = pSPARC->std_KE = pSPARC->std_PE = pSPARC->std_U = pSPARC->std_Entropy = 0.0; pSPARC->std_internal_pressure = 0.0;
		for (int i = 0; i < 9; i++){
			pSPARC->mean_total_internal_stress[i] = 0.0;
			pSPARC->std_total_internal_stress[i] = 0.0;
		}
	
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
	Calculate_Properties(pSPARC);
	//Calculate_electronicGroundState(pSPARC);
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
	Calculate_Properties(pSPARC);
	//Calculate_electronicGroundState(pSPARC);
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
	Calculate_Properties(pSPARC);
	//Calculate_electronicGroundState(pSPSPARC);
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
	Calculate_Properties(pSPARC);
	//Calculate_electronicGroundState(pSPARC);
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

    // scaling factor
	int p;
#ifdef DEBUG
	int i;
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
        for (i = 0; i < 3 * pSPARC->n_atom; i++) {
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
            printf("WARNING: for mesh less than 0.1, the default Chebyshev polynomial degree might not be enought!\n");
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
        if (pSPARC->sqAmbientFlag == 0 && pSPARC->sqHighTFlag == 0 && pSPARC->OFDFTFlag == 0) {
            Calculate_local_kpoints(pSPARC);
        }
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
	hamiltonian += kineticBaro + kineticTher + potentialBaro + potentialTher;
	pSPARC->Hamiltonian_NPT_NH = hamiltonian;
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
	    printf("Hamiltonian (Ha)                : %12.9f\n", hamiltonian);
	}
}



/** 
 *@brief function to perform NPT MD simulation with Nose-Poincare, wherein number of particles, pressure and temperature are kept constant.
Also performs NPH MD simulation , wherein number of particles, pressure and enthalpy (or generalized enthalpy of Thurston in case of anisotropic external stress) are kept constant. 
NPH in this scheme is same as NPT_NP wherein Mass of thermostat is zero. So same functions are used.
Reference for NPT_NP and NPH: Hernández, E. "Metric-tensor flexible-cell algorithm for isothermal–isobaric molecular dynamics simulations." .Hereafter referred to as the 'Hernandez paper'.
The Journal of Chemical Physics 115, no. 22 (2001): 10282-10290.
Additional Reference for NPH: Souza, Ivo, and JoséLuís Martins. "Metric tensor as the dynamical variable for variable-cell-shape molecular dynamics." 
Physical Review B 55.14 (1997): 8733.
*/
void NPT_NP_and_NPH(SPARC_OBJ *pSPARC, double *avgvel, double *maxvel, double *mindis) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Bcast(&pSPARC->stress, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pSPARC->stress_i, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	//Calculate initial hamitonian
	if ((pSPARC->MDCount == 1)  && (pSPARC->RestartFlag != 1)){
		NPT_NP_and_NPH_init_hamiltonian(pSPARC);
	}
	//NPT_NPH_main routine
	NPT_NPH_main(pSPARC, avgvel, maxvel, mindis);
	// Reinitialize mesh size and related variables after changing size of cell
    reinitialize_mesh_NPT(pSPARC);
	// Charge extrapolation (for better rho_guess)
	elecDensExtrapolation(pSPARC);
	// Check position of atom near the boundary and apply wraparound in case of PBC, otherwise show error if the atom is too close to the boundary for bounded domain
	Check_atomlocation(pSPARC);
	// Compute DFT energy and forces by solving Kohn-Sham eigenvalue problem
	Calculate_Properties(pSPARC);
	//Calculate_electronicGroundState(pSPARC);
	pSPARC->elecgs_Count++;
}

/** 
 *@brief function to compute transpose of a matrix and adds it to the original matrix
*/
void transpose_and_add(double *matrix1){
	//performs matrix1 = matrix1 + transpose(matrix1)
	// Diagonals: m[i][i] = m[i][i] + m[i][i]
	matrix1[0] *= 2.0; matrix1[4] *= 2.0; matrix1[8] *= 2.0;

	// Off-diagonals: sum then assign to both sides
	double s01 = matrix1[1] + matrix1[3]; // (0,1) + (1,0)
	double s02 = matrix1[2] + matrix1[6]; // (0,2) + (2,0)
	double s12 = matrix1[5] + matrix1[7]; // (1,2) + (2,1)

	matrix1[1] = matrix1[3] = s01;
	matrix1[2] = matrix1[6] = s02;
	matrix1[5] = matrix1[7] = s12;
}

/** 
 *@brief function to be called during 'restart' of NPT_NP and NPH to initialize the cell ingredients such as 'full_lattice' (lattice vectors scaled by LATVEC scale, or LatUVec scaled by cell), reciprocal lattice, metric tensor, reciprocal metic tensor, angle between cell lattice vectors, cell volume, etc
In essence similar to 'fetch_MD_cell_ingredients' below, but called during 'restart' of NPT_NP and NPH
*/
void fetch_MD_cell_ingredients_restart(SPARC_OBJ *pSPARC){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double old_cell[9]; double oldLatVec[9];

	//Compute Cell lattice vectors (scaled by LATVEC scale)
	if (pSPARC->Flag_latvec_scale == 0){
		for (int i = 0; i < 3; i++) {
			pSPARC->full_lattice[i] = pSPARC->LatUVec[i] * pSPARC->range_x;
			pSPARC->full_lattice[i+3] = pSPARC->LatUVec[i + 3] * pSPARC->range_y;
			pSPARC->full_lattice[i+6] = pSPARC->LatUVec[i + 6] * pSPARC->range_z;
		}
	}
	else{
		for (int i = 0; i < 3; i++) {
			pSPARC->full_lattice[i] = pSPARC->LatVec[i] * pSPARC->latvec_scale_x;
			pSPARC->full_lattice[i+3] = pSPARC->LatVec[i+3] * pSPARC->latvec_scale_y;
			pSPARC->full_lattice[i+6] = pSPARC->LatVec[i+6] * pSPARC->latvec_scale_z;
		}

	}
	//Compute metric_tensor (G) in real-space (not reciprocal space)
	cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, pSPARC->full_lattice, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->metric_tensor, 3);
	
	if (pSPARC->Flag_latvec_scale == 1){ // for Flag_latvec_scale == 0,  the update for 'range_x,y,z' has already happened within restart function
		pSPARC->range_x = sqrt( pSPARC->metric_tensor[0] );
		pSPARC->range_y = sqrt( pSPARC->metric_tensor[4] );
		pSPARC->range_z = sqrt( pSPARC->metric_tensor[8] );

		pSPARC->initialLatVecLength[0] = sqrt(pSPARC->LatVec[0] * pSPARC->LatVec[0] + pSPARC->LatVec[1] * pSPARC->LatVec[1] + pSPARC->LatVec[2]* pSPARC->LatVec[2]);
		pSPARC->initialLatVecLength[1] = sqrt(pSPARC->LatVec[3] * pSPARC->LatVec[3] + pSPARC->LatVec[4] * pSPARC->LatVec[4] + pSPARC->LatVec[5] * pSPARC->LatVec[5]);
		pSPARC->initialLatVecLength[2] = sqrt(pSPARC->LatVec[6] * pSPARC->LatVec[6] + pSPARC->LatVec[7] * pSPARC->LatVec[7] + pSPARC->LatVec[8] * pSPARC->LatVec[8]);
	}
	else{
		pSPARC->initialLatVecLength[0] = 1; pSPARC->initialLatVecLength[1] = 1; pSPARC->initialLatVecLength[2] = 1;
	}

	//Assign full_lattice to LatVec for updating various quantities in 'Cart2nonCart_transformMat'
	for (int i = 0; i < 9; i++){oldLatVec[i] = pSPARC->LatVec[i];}
	for (int i = 0; i < 9; i++){pSPARC->LatVec[i] = pSPARC->full_lattice[i];}
	
	//Update LatUVec, Jacbdet, metricT, gradT, lapcT  (this function updates all quantites based on 'LatVec' variable as input,  but we actually want it based on 'full_lattice' variable, which contains all the cell lattice vectors updates,  so we do the step above this to assign full_lattice to LatVec)
	Cart2nonCart_transformMat(pSPARC);
	pSPARC->volumeCell = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;

	//Reassign LatVec it old values
	for (int i = 0; i < 9; i++){pSPARC->LatVec[i] = oldLatVec[i];}
	
	// Update/Calculate new angles between lattice vectors  (only for inference, not used anywhere in the code)
	double cos_gamma_new = pSPARC->metric_tensor[1] / (pSPARC->range_x * pSPARC->range_y); 
	double cos_beta_new = pSPARC->metric_tensor[2] / (pSPARC->range_x * pSPARC->range_z);
	double cos_alpha_new = pSPARC->metric_tensor[5] / (pSPARC->range_y * pSPARC->range_z);
		
	pSPARC->angle_12 = acos(cos_gamma_new) * 180 / M_PI;
	pSPARC->angle_13 = acos(cos_beta_new) * 180 / M_PI;
	pSPARC->angle_23 = acos(cos_alpha_new) * 180 / M_PI;

	//Update reciprocal lattice vectors, reciprocal metric tensor
	for (int i = 0; i < 9; i++){
		old_cell[i] = pSPARC->full_lattice[i];
	}
	
	// Calculate the inverse of lattice-vector
	double U[9]; double S[3]; double VT[9]; double superb[2]; double temp_mat[9];
	double S_inv[9] = {0};

	/* Calculating pinv(LatVec): This is necessary because for extremely skewed LV, the LV maybe close to singular */
	// Compute SVD of LatVec(LV) first: LV = U * S * V^T
	LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', 3, 3, old_cell, 3, S, U, 3, VT, 3, superb);

	// First compute S^{-1}
	for (int i = 0; i < 3; i++) {
		if (S[i] > 1e-12) S_inv[i * 3 + i] = 1.0 / S[i];
	}

	// Compute LV^{-1} = V * S^{-1} * U^T
	// Note that since full_lattice is rowMajor,  reciprocal_lattice is columnMajor
	// i.e. the reciprocal lattice vectors (of full_lattice) are stored column-wise 
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, VT, 3, S_inv, 3, 0.0, temp_mat, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, temp_mat, 3, U, 3, 0.0, pSPARC->reciprocal_lattice, 3);
	// Now computing reciprocal metric tensor, 
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_lattice, 3, pSPARC->reciprocal_lattice, 3, 0.0, pSPARC->reciprocal_metric_tensor, 3);
	
}

/** 
 *@brief function to initialize the cell ingredients for NPT_NP or NPH, and for updating the cell ingredients in each step.
To compute 'full_lattice' (lattice vectors 'LatVec' scaled by LATVEC SCALE or LatUVec scaled by cell), and corresponding:  reciprocal_lattice, metric_tensor, reciprocal_matric_tensor, initialLatVecAngles, rotation_matrix, angle between cell lattice vectors, velocity of cell lattice vectors, update volume of the cell etc
*/
void fetch_MD_cell_ingredients(SPARC_OBJ *pSPARC, bool update_cell){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double old_cell[9]; double new_cell[9]; 

	if (update_cell == false){ // Update_cell == false only initializes the cell parameters and basic key ingredients used in NPT_NP and NPH ensemble; such as full_lattice, metric_tensor, reciprocal_metric_tensor ...etc,  to be used when doing first step of MD
	
		for (int i = 0; i < 3; i++) {
			pSPARC->full_lattice[i] = pSPARC->LatUVec[i] * pSPARC->range_x;
			pSPARC->full_lattice[i+3] = pSPARC->LatUVec[i + 3] * pSPARC->range_y;
			pSPARC->full_lattice[i+6] = pSPARC->LatUVec[i + 6] * pSPARC->range_z;
		}

		//Compute metric_tensor (G) in real-space (not reciprocal space)
		cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, pSPARC->full_lattice, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->metric_tensor, 3);
	
		// Cosine of Angles between cell lattice vectors (useful in case of restricting lattice vectors rotation in NPT_NP and NPH simulations)
		pSPARC->initialLatVecAngles[0] = pSPARC->metric_tensor[5] / (pSPARC->range_y * pSPARC->range_z);  // cos_alpha (angle between b and c)
		pSPARC->initialLatVecAngles[1] = pSPARC->metric_tensor[2] / (pSPARC->range_x * pSPARC->range_z);  // cos_beta (angle between a and c)
		pSPARC->initialLatVecAngles[2] = pSPARC->metric_tensor[1] / (pSPARC->range_x * pSPARC->range_y);  // cos_gamma (angle between a and b)

		pSPARC->angle_12 = acos(pSPARC->initialLatVecAngles[2]) * 180 / M_PI;
		pSPARC->angle_13 = acos(pSPARC->initialLatVecAngles[1]) * 180 / M_PI;
		pSPARC->angle_23 = acos(pSPARC->initialLatVecAngles[0]) * 180 / M_PI;
		
	}

	// Rotated cell, such that the lattice vectors are: LatVec1 = (a,0,0);  LatVec2 = (b1, b2, 0); LatVec3 = (c1,c2,c3)
	new_cell[0] = sqrt(pSPARC->metric_tensor[0]);
	new_cell[1] = 0;
	new_cell[2] = 0;

	new_cell[3] = pSPARC->metric_tensor[3] / sqrt(pSPARC->metric_tensor[0]);
	new_cell[4] = sqrt( pSPARC->metric_tensor[4]- pSPARC->metric_tensor[3] * pSPARC->metric_tensor[3] / pSPARC->metric_tensor[0]);
	new_cell[5] = 0;

	new_cell[6] = pSPARC->metric_tensor[6] / sqrt(pSPARC->metric_tensor[0]);
	new_cell[7] = ( pSPARC->metric_tensor[0] * pSPARC->metric_tensor[7] - pSPARC->metric_tensor[3] * pSPARC->metric_tensor[6]) / sqrt(pSPARC->metric_tensor[0] * pSPARC->metric_tensor[0] * pSPARC->metric_tensor[4] - pSPARC->metric_tensor[0] * pSPARC->metric_tensor[3] * pSPARC->metric_tensor[3]);          
    new_cell[8] = sqrt(pSPARC->metric_tensor[8] - new_cell[6] * new_cell[6] - new_cell[7] * new_cell[7]);

	if (update_cell == true){ //update_cell == true is used to update the lattice_vectors of full cell, reciprocal metric tensor, reciprocal lattice vectors, cell volume, cell lattice vectors velocities and basically all the parameters that needs to be updated accounting the change in cell lattice vectors lengths and angles;  to be used after the first step of MD (and at the end of each MD step).
		//Update full cell's lattice vectors
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, new_cell, 3, pSPARC->rotation_matrix, 3, 0.0, pSPARC->full_lattice, 3);
	
		//Update range_x, range_y, range_z, volumeCell, 
		pSPARC->range_x = sqrt( pSPARC->metric_tensor[0] );
		pSPARC->range_y = sqrt( pSPARC->metric_tensor[4] );
		pSPARC->range_z = sqrt( pSPARC->metric_tensor[8] );

		//Assign full_lattice to LatVec for updating various quantities in 'Cart2nonCart_transformMat'
		double oldLatVec[9];
		for (int i = 0; i < 9; i++){oldLatVec[i] = pSPARC->LatVec[i];}
		for (int i = 0; i < 9; i++){pSPARC->LatVec[i] = pSPARC->full_lattice[i];}
		
		//Update LatUVec, Jacbdet, metricT, gradT, lapcT  (this function updates all quantites based on 'LatVec' variable as input,  but we actually want it based on 'full_lattice' variable, which contains all the cell lattice vectors updates,  so we do the step above this to assign full_lattice to LatVec)
		Cart2nonCart_transformMat(pSPARC);
		pSPARC->volumeCell = pSPARC->Jacbdet * pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;

		//Reassign LatVec it old values
		for (int i = 0; i < 9; i++){pSPARC->LatVec[i] = oldLatVec[i];}

		// Update/Calculate new angles between lattice vectors
		pSPARC->angle_12 = acos( pSPARC->metric_tensor[1] / (pSPARC->range_x * pSPARC->range_y) ) * 180 / M_PI;
		pSPARC->angle_13 = acos( pSPARC->metric_tensor[2] / (pSPARC->range_x * pSPARC->range_z) ) * 180 / M_PI;
		pSPARC->angle_23 = acos( pSPARC->metric_tensor[5] / (pSPARC->range_y * pSPARC->range_z) ) * 180 / M_PI;

		//Update LATVEC_SCALE and LatVec
		if (pSPARC->Flag_latvec_scale == 1){
			// LatVec just accounts for change in orientation/angles
			for (int i = 0; i < 3; i++){
				pSPARC->LatVec[i] = pSPARC->LatUVec[i] * pSPARC->initialLatVecLength[0];  
				pSPARC->LatVec[i+3] = pSPARC->LatUVec[i+3] * pSPARC->initialLatVecLength[1]; 
				pSPARC->LatVec[i+6] = pSPARC->LatUVec[i+6] * pSPARC->initialLatVecLength[2];
			}

			// LATVEC_SCALE accounts for change in lengths
			pSPARC->latvec_scale_x = pSPARC->range_x / pSPARC->initialLatVecLength[0];  
			pSPARC->latvec_scale_y = pSPARC->range_y / pSPARC->initialLatVecLength[1];
			pSPARC->latvec_scale_z = pSPARC->range_z / pSPARC->initialLatVecLength[2];
			
		}

	}
	//Update reciprocal lattice vectors, reciprocal metric tensor
	for (int i = 0; i < 9; i++){
		old_cell[i] = pSPARC->full_lattice[i];
	}
	
	// Calculate the inverse of lattice-vector
	double U[9]; double S[3]; double VT[9]; double superb[2]; double temp_mat[9];
	double S_inv[9] = {0};

	/* Calculating pinv(LatVec): This is necessary because for extremely skewed LV, the LV maybe close to singular */
	// Compute SVD of LatVec(LV) first: LV = U * S * V^T
	LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', 3, 3, old_cell, 3, S, U, 3, VT, 3, superb);

	// First compute S^{-1}
	for (int i = 0; i < 3; i++) {
		if (S[i] > 1e-12) S_inv[i * 3 + i] = 1.0 / S[i];
	}

	// Compute LV^{-1} = V * S^{-1} * U^T
	// Note that since full_lattice is rowMajor,  reciprocal_lattice is columnMajor
	// i.e. the reciprocal lattice vectors (of full_lattice) are stored column-wise 
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, VT, 3, S_inv, 3, 0.0, temp_mat, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, temp_mat, 3, U, 3, 0.0, pSPARC->reciprocal_lattice, 3);
	// Now computing reciprocal metric tensor, 
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_lattice, 3, pSPARC->reciprocal_lattice, 3, 0.0, pSPARC->reciprocal_metric_tensor, 3);

	if (update_cell == false){
		//Rotation_matrix = reciprocal_lattice@new_cell    as we want:  new_cell@Rotation_matrix.T = old_cell;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_lattice, 3, new_cell, 3, 0.0, pSPARC->rotation_matrix, 3); 

		//Initiating cell lattice vectors velocity as zero
		for (int i = 0; i < 9; i++){
			pSPARC->lattice_avg_velo[i] = 0.0;
		}
	}

	//If updating the cell, then also calculate the velocity of cell lattice vectors  (only useful in 1st MD step (if starting with nonzero lattice vector velocities:  but as of now this step is not really useful, as we are by default only supporting zero initializing for lattice vector velocities))
	else {
		double temp_mat_2[9] = {0.0};  double temp_mat_3[9];

		for (int i = 0; i < 9; i++){
			temp_mat_3[i] = pSPARC->Pm_metric_tensor[i];
		}

		if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			cblas_dscal(9, pSPARC->SNOSE[2] / ( pSPARC->NPT_NP_bmass * pSPARC->volumeCell * pSPARC->volumeCell ), temp_mat_3, 1);
		}
		else {
			cblas_dscal(9, pSPARC->SNOSE[2] / ( pSPARC->NPH_bmass * pSPARC->volumeCell * pSPARC->volumeCell ), temp_mat_3, 1);
		}
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->metric_tensor, 3, temp_mat_3, 3, 0.0, temp_mat_2, 3);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, temp_mat_2, 3, pSPARC->metric_tensor, 3, 0.0, temp_mat_3, 3);

		for (int i = 0; i < 9; i++){
			temp_mat_2[i] = 0.0;
		}

		temp_mat_2[0] = 0.5 * temp_mat_3[0] / (new_cell[0]);
		temp_mat_2[1] = 0.0;
		temp_mat_2[2] = 0.0;

		temp_mat_2[3] = (temp_mat_3[3] - temp_mat_2[0] * new_cell[3]) / new_cell[0];
		temp_mat_2[4] = (0.5 * temp_mat_3[4] - temp_mat_2[3] * new_cell[3]) / new_cell[4];
		temp_mat_2[5] = 0.0;

		temp_mat_2[6] = (temp_mat_3[6] - temp_mat_2[0] * new_cell[6]) / new_cell[0];
		temp_mat_2[7] = (temp_mat_3[7] - temp_mat_2[6] * new_cell[3] - temp_mat_2[3] * new_cell[6] - temp_mat_2[4] * new_cell[7]) / new_cell[4];
		temp_mat_2[8] = (0.5 * temp_mat_3[8] - temp_mat_2[6] * new_cell[6] - temp_mat_2[7] * new_cell[7]) / new_cell[8];

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, temp_mat_2, 3, pSPARC->rotation_matrix, 3, 0.0, pSPARC->lattice_avg_velo, 3);
	}
}

/** 
 *@brief function to compute the kinetic energy (and temperature) of the ionic particles in NPT_NP and NPH ensemble
*/
void Calculate_Ionic_particles_Kinetic_energy(SPARC_OBJ *pSPARC){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double vector[3]={0.0};
	pSPARC->KE = 0.0;

	int count = 0;
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
		for (int atm = 0; atm < pSPARC->nAtomv[ityp]; atm++) {
			cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, pSPARC->reciprocal_metric_tensor, 3, &pSPARC->Pm_ion[count*3], 1, 0.0, vector, 1);                          
			pSPARC->KE += cblas_ddot(3, vector, 1, &pSPARC->Pm_ion[count*3], 1) / pSPARC->Mass[ityp];
			count++;
		}
	}
	
	pSPARC->KE = 0.5 * pSPARC->KE / ( pSPARC->SNOSE[0] * pSPARC->SNOSE[0] );
	pSPARC->ion_T = 2 * pSPARC->KE /(pSPARC->kB * pSPARC->dof); //Update Ionic temperature;
}

/** 
 *@brief function to compute the kinetic stress of the ionic particles in NPT_NP and NPH ensemble
*/
void Calculate_Kinetic_stress(SPARC_OBJ *pSPARC){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	for (int i = 0; i < 9; i++){
		pSPARC->kinetic_stress[i] = 0.0; //Initialize kinetic stress to 0
	}
	
	int count = 0;
	for(int ityp = 0; ityp < pSPARC->Ntypes; ityp++){
		for(int atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
			pSPARC->kinetic_stress[0] += pSPARC->Mass[ityp] * pSPARC->ion_vel_fractional[count*3] * pSPARC->ion_vel_fractional[count*3];
			pSPARC->kinetic_stress[1] += pSPARC->Mass[ityp] * pSPARC->ion_vel_fractional[count*3] * pSPARC->ion_vel_fractional[count*3+1];
			pSPARC->kinetic_stress[2] += pSPARC->Mass[ityp] * pSPARC->ion_vel_fractional[count*3] * pSPARC->ion_vel_fractional[count*3+2];
			pSPARC->kinetic_stress[4] += pSPARC->Mass[ityp] * pSPARC->ion_vel_fractional[count*3+1] * pSPARC->ion_vel_fractional[count*3+1];
			pSPARC->kinetic_stress[5] += pSPARC->Mass[ityp] * pSPARC->ion_vel_fractional[count*3+1] * pSPARC->ion_vel_fractional[count*3+2];
			pSPARC->kinetic_stress[8] += pSPARC->Mass[ityp] * pSPARC->ion_vel_fractional[count*3+2] * pSPARC->ion_vel_fractional[count*3+2];
			count++;
		}
	}
	
	pSPARC->kinetic_stress[3] = pSPARC->kinetic_stress[1];
	pSPARC->kinetic_stress[6] = pSPARC->kinetic_stress[2];
	pSPARC->kinetic_stress[7] = pSPARC->kinetic_stress[5];

	cblas_dscal(9, 0.5 / (pSPARC->SNOSE[0] * pSPARC->SNOSE[0]), pSPARC->kinetic_stress, 1);
}

/** 
 *@brief function to compute the initial hamiltonian of the system in NPT_NP and NPH ensemble.
Only called once during the 1st step of MD. Not to be called after restart.
*/
void NPT_NP_and_NPH_init_hamiltonian(SPARC_OBJ *pSPARC){
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	//Initialize some useful constants
	double baro_const1;
	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		baro_const1 = pSPARC->NPT_NP_bmass * pSPARC->volumeCell * pSPARC->volumeCell; // M_G*det(G) in the Hernandez paper
	}
	else {
		baro_const1 = pSPARC->NPH_bmass * pSPARC->volumeCell * pSPARC->volumeCell; // M_G*det(G) in the Hernandez paper
	}
	double baro_const2 = baro_const1 / pSPARC->SNOSE[2];
	double baro_const3 = 1.0 / baro_const1;
	double ktemp = pSPARC->kB * pSPARC->thermos_T;

	//Initialize empty temporary matrices and vectors 
	double temp_mat[9]; double temp_mat_a[9]; double temp_mat_b[9];
	
	// ------------------------------------- BEGIN: Calculating Hamiltonian (Eqn 10)----------------------------------//

	// Calculating kinetic energy of ions
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, pSPARC->SNOSE[2], pSPARC->ion_vel, 3, pSPARC->reciprocal_lattice, 3, 0.0, pSPARC->ion_vel_fractional, 3); 
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0, pSPARC->ion_vel_fractional, 3, pSPARC->metric_tensor, 3, 0.0, pSPARC->Pm_ion, 3); 
	int count = 0;
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
		cblas_dscal(3 * pSPARC->nAtomv[ityp], pSPARC->Mass[ityp], &pSPARC->Pm_ion[count], 1);
		count = count + 3 * pSPARC->nAtomv[ityp];
	}
	// Calculate kinetic energy and kinetic stress
	Calculate_Kinetic_stress(pSPARC); //Calculate kinetic stress with the initial distribution of Ionic particles velocity
	Calculate_Ionic_particles_Kinetic_energy(pSPARC);  //Term 1 in Eqn.10 Hernandez paper

	// Calculating thermostat energies
	pSPARC->Kther = 0.5 * pSPARC->NPT_NP_qmass * pSPARC->SNOSE[1] * pSPARC->SNOSE[1]; //Kinetic;  Term 6 in Eqn.10 Hernandez paper
	pSPARC->Uther = (double)pSPARC->dof * log(pSPARC->SNOSE[0]) * ktemp; //Entropic;  Term 7 in Eqn.10 Hernandez paper
	

	// Calculating barostat energies 
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1.0, pSPARC->full_lattice, 3, pSPARC->lattice_avg_velo, 3, 0.0, pSPARC->Pm_metric_tensor, 3);
	transpose_and_add(pSPARC->Pm_metric_tensor);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_metric_tensor, 3, pSPARC->Pm_metric_tensor, 3, 0.0, temp_mat, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, temp_mat, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, pSPARC->Pm_metric_tensor, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, baro_const3 * baro_const2, pSPARC->Pm_metric_tensor, 3, pSPARC->metric_tensor, 3, 0.0, temp_mat_a, 3);
	
	//Since these are 3x3 matrix, transposing without for loop (for avoiding loop overhead cost)
	temp_mat_b[0] = temp_mat_a[0]; temp_mat_b[4] = temp_mat_a[4]; temp_mat_b[8] = temp_mat_a[8];
	temp_mat_b[1] = temp_mat_a[3]; temp_mat_b[2] = temp_mat_a[6]; temp_mat_b[5] = temp_mat_a[7];
	temp_mat_b[3] = temp_mat_a[1]; temp_mat_b[6] = temp_mat_a[2]; temp_mat_b[7] = temp_mat_a[5]; 

	pSPARC->Kbaro = 0.5 * baro_const1 * cblas_ddot(9, temp_mat_a, 1, temp_mat_b, 1);//Kinetic;  Term 3 in Eqn.10 in Hernandez paper
	pSPARC->Ubaro = pSPARC->pressure_external * pSPARC->volumeCell; //Potential;  Term 4 in Eqn.10 in Hernandez paper
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, pSPARC->volumeCell, pSPARC->external_stress_cartesian, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, pSPARC->external_stress_lattice, 3);
	
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, pSPARC->volumeCell, pSPARC->external_stress_cartesian, 3, pSPARC->reciprocal_lattice, 3, 0.0, temp_mat, 3);
	//cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_lattice, 3, temp_mat, 3, 0.0, pSPARC->external_stress_lattice, 3);
	
	pSPARC->Ubaro += 0.5 * cblas_ddot(9, pSPARC->external_stress_lattice, 1, pSPARC->metric_tensor, 1); //Potential;  Term 5 in Eqn.10 in Hernandez paper 


	double sumAllHamilTerms;
	sumAllHamilTerms = pSPARC->KE + pSPARC->Etot + pSPARC->Kther + pSPARC->Uther + pSPARC->Kbaro + pSPARC->Ubaro;  //Etot is 2nd term in Eqn. 10 in Hernandez paper

	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		if ((pSPARC->MDCount == 1)  && (pSPARC->RestartFlag != 1)) {
			pSPARC->init_Hamil_NPT_NP = sumAllHamilTerms;  //Initial Hamiltonian H0
		}
		pSPARC->Hamiltonian_NPT_NP = pSPARC->SNOSE[0] * (sumAllHamilTerms - pSPARC->init_Hamil_NPT_NP);  //Eqn. 8 in the Hernandez paper
	}
	else {
		if ((pSPARC->MDCount == 1)  && (pSPARC->RestartFlag != 1)) {
			pSPARC->init_Hamil_NPH = sumAllHamilTerms;     //Initial Hamiltonian H0
		}
		pSPARC->Hamiltonian_NPH = pSPARC->SNOSE[0] * (sumAllHamilTerms - pSPARC->init_Hamil_NPH);  //Eqn. 8 in the Hernandez paper
	}
	
	#ifdef DEBUG
	if (rank == 0) {
		printf("within init");
    	printf("\n");
    	printf("rank %d", rank);
    	printf("ENERGY of time step %d\n", pSPARC->MDCount + 1);
	    printf("kinetic energy (Ha)             : %12.9f\n", pSPARC->KE);
	    printf("potential energy (Ha)           : %12.9f\n", pSPARC->Etot);
	    printf("barostat kinetic energy (Ha)    : %12.9f\n", pSPARC->Kbaro);
	    printf("thermostat kinetic energy (Ha)  : %12.9f\n", pSPARC->Kther);
	    printf("barostat potential energy (Ha)  : %12.9f\n", pSPARC->Ubaro);
	    printf("thermostat potential energy (Ha): %12.9f\n", pSPARC->Uther);
	    printf("Sum of all energy terms (Ha)    : %12.9f\n", sumAllHamilTerms);
		if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			printf("Hamiltonian (Ha)                : %12.9f\n", pSPARC->Hamiltonian_NPT_NP);
		}
		else {
			printf("Hamiltonian (Ha)                : %12.9f\n", pSPARC->Hamiltonian_NPH);
		}
	}
	#endif
	// ------------------------------------- END: Calculating Hamiltonian (Eqn 10)----------------------------------//

}

/** 
 *@brief function is the main function of NPT_NP and NPH ensemble. 
 It solves Equations 18a to 18g in the  Hernández, E. "Metric-tensor flexible-cell algorithm for isothermal–isobaric molecular dynamics simulations." .Hereafter referred to as the 'Hernandez paper'.: 
 		  -updates momentum and position of the ionic particles
		  -updates momentum and position of the thermostat
		  -update momentum and position of the barostat
		  -calculate Hamiltonian of the system
*/
void NPT_NPH_main(SPARC_OBJ *pSPARC, double *avgvel, double *maxvel, double *mindis) {
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double ktemp = pSPARC->kB * pSPARC->thermos_T;
	int count;
	
	//Initialize empty temporary matrices and vectors 
	double temp_mat[9]; double temp_mat_a[9]; double temp_mat_b[9]; double temp_Pm_mat[9];
	double internal_stress_cartesian[9];

	//Initialize some useful constants
	double baro_const1; int NPT_NPH_ANGLES;  int NPT_NPHconstraintFlag; int NPT_NPHscaleVecs[3]={0};
	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		NPT_NPH_ANGLES = pSPARC->NPT_NP_ANGLES;
		NPT_NPHconstraintFlag = pSPARC->NPTconstraintFlag;
		NPT_NPHscaleVecs[0] = pSPARC->NPTscaleVecs[0]; NPT_NPHscaleVecs[1] = pSPARC->NPTscaleVecs[1]; NPT_NPHscaleVecs[2] = pSPARC->NPTscaleVecs[2];
		
		baro_const1 = pSPARC->NPT_NP_bmass * pSPARC->volumeCell * pSPARC->volumeCell; // M_G*det(G) in the Hernandez paper
	}
	else{
		NPT_NPH_ANGLES = pSPARC->NPH_ANGLES;
		NPT_NPHconstraintFlag = pSPARC->NPHconstraintFlag;
		NPT_NPHscaleVecs[0] = pSPARC->NPHscaleVecs[0]; NPT_NPHscaleVecs[1] = pSPARC->NPHscaleVecs[1]; NPT_NPHscaleVecs[2] = pSPARC->NPHscaleVecs[2];
		
		baro_const1 = pSPARC->NPH_bmass * pSPARC->volumeCell * pSPARC->volumeCell; // M_G*det(G) in the Hernandez paper
	}

	double baro_const3 = 1.0 / baro_const1;

	if (pSPARC->MDCount == 1){ // Do some checks/setup
		//Initialize constraint stress to 0
		for (int i = 0; i < 9; i++){
			pSPARC->constraint_stress[i] = 0.0;
		}
	}

	// ------------------------------------- BEGIN: Updating Momenta by half step (Eqns. 18g, 18h, 18i)----------------------------------//
	double factor;
	// bring the momenta of the thermostat variable in time-sync with the positions (since mometa are delayed by dt/2)
	// This corresponds Eqn. 18G in the Hernandez paper (Skip this step if doing NPH since thermostat mass = 0)
	if (pSPARC->NPT_NP_qmass > 0){
		pSPARC->Kther = 0.5 * pSPARC->NPT_NP_qmass * pSPARC->SNOSE[1] * pSPARC->SNOSE[1]; //Kinetic
		factor = (pSPARC->KE - pSPARC->Etot - pSPARC->dof * ktemp * (log(pSPARC->SNOSE[0]) + 1) - pSPARC->Kbaro - pSPARC->Ubaro - pSPARC->Kther  + pSPARC->init_Hamil_NPT_NP) ;
		pSPARC->SNOSE[1] += 0.5 * factor * pSPARC->MD_dt / pSPARC->NPT_NP_qmass;
		#ifdef DEBUG
		if (rank == 0) {
			printf("factor Eqn.18G is %12.9f\n", factor);
			printf("SNOSE[1] in the 1st half step is %12.9f\n", pSPARC->SNOSE[1]);
		}
		#endif
		// Update the kinetic energy of the thermostat
		
		pSPARC->Kther = 0.5 * pSPARC->NPT_NP_qmass * pSPARC->SNOSE[1] * pSPARC->SNOSE[1]; //Kinetic
		pSPARC->Uther = (double)pSPARC->dof * log(pSPARC->SNOSE[0]) * ktemp; //Entropic;  Term 7 in Eqn.10 Hernandez paper
		
	}
	
	// bring the momenta of the barostat variable in time-sync with the positions (since mometa are delayed by dt/2)
	// This setup corresponds Eqn. 18h in the Hernandez paper 
	internal_stress_cartesian[0] = pSPARC->stress[0]; internal_stress_cartesian[4] = pSPARC->stress[3]; internal_stress_cartesian[8] = pSPARC->stress[5];
	internal_stress_cartesian[1] = pSPARC->stress[1]; internal_stress_cartesian[2] = pSPARC->stress[2]; internal_stress_cartesian[3] = pSPARC->stress[1];
	internal_stress_cartesian[5] = pSPARC->stress[4]; internal_stress_cartesian[6] = pSPARC->stress[2]; internal_stress_cartesian[7] = pSPARC->stress[4];
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, internal_stress_cartesian, 3, pSPARC->reciprocal_lattice, 3, 0.0, temp_mat_b, 3);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 0.5 * pSPARC->volumeCell, pSPARC->reciprocal_lattice, 3, temp_mat_b, 3, 0.0, pSPARC->internal_stress_fractional, 3); //Multiplying by volume since the stress is stored in the units of Ha/bohr^3

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, baro_const3, pSPARC->Pm_metric_tensor, 3, pSPARC->metric_tensor, 3, 0.0, temp_mat_a, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, temp_mat_a, 3, pSPARC->Pm_metric_tensor, 3, 0.0, temp_mat_b, 3);

	for (int i = 0; i < 9; i++){
		temp_Pm_mat[i] = pSPARC->Pm_metric_tensor[i];
	}

	// Eqn. 18h  Hernandez paper
	for (int i = 0; i < 9; i++){
		temp_mat[i] = (pSPARC->internal_stress_fractional[i] - pSPARC->kinetic_stress[i]) + (temp_mat_b[i]);
		temp_mat[i] += (0.5 * pSPARC->pressure_external * pSPARC->volumeCell - pSPARC->Kbaro) * pSPARC->reciprocal_metric_tensor[i] + 0.5 * pSPARC->external_stress_lattice[i];
		pSPARC->Pm_metric_tensor[i] -=  0.5 * pSPARC->MD_dt * pSPARC->SNOSE[0] * temp_mat[i];
	} 
	

	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		if (pSPARC->NPT_NP_ANGLES == 0){
			compute_constraint_stress(pSPARC, NPT_NPHconstraintFlag, NPT_NPHscaleVecs);
		}
		
	}
	else {
		if (pSPARC->NPH_ANGLES == 0){
			compute_constraint_stress(pSPARC, NPT_NPHconstraintFlag, NPT_NPHscaleVecs);
		}
	}

	//This block below seems redundant, since we already update the momenta based on constraints in function: compute_constraint_stress
	for (int i = 0; i < 9; i++){
		temp_mat[i] = pSPARC->internal_stress_fractional[i] + pSPARC->constraint_stress[i] - pSPARC->kinetic_stress[i] + temp_mat_b[i];
		temp_mat[i] += (0.5 * pSPARC->pressure_external * pSPARC->volumeCell - pSPARC->Kbaro) * pSPARC->reciprocal_metric_tensor[i] + 0.5 * pSPARC->external_stress_lattice[i];
		pSPARC->Pm_metric_tensor[i] = temp_Pm_mat[i] - 0.5 * pSPARC->MD_dt * pSPARC->SNOSE[0] * temp_mat[i];
	} 	

	//Now impose the constraints on it
	for (int i = 0; i < 3; i++){ // If any of the lattice vectors is constrained from being rescaled, set its momentum to 0
		if (NPT_NPHscaleVecs[i] == 0){
			pSPARC->Pm_metric_tensor[4 * i] = 0; //setting that corresponding diagonal element to 0
		}
	}

	if ( (NPT_NPHscaleVecs[2]) == 0 && (fabs(pSPARC->initialLatVecAngles[0]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[1]) <= 1e-12) ){ //Special constraint 1: a and b can vary, c must not vary. Further, the angle between c and a, or between c and b, must be 90 degrees.
		pSPARC->Pm_metric_tensor[8] = 0;
		pSPARC->Pm_metric_tensor[7] = 0;
		pSPARC->Pm_metric_tensor[6] = 0;
		pSPARC->Pm_metric_tensor[5] = 0;
		pSPARC->Pm_metric_tensor[2] = 0;
	}
	else if ( (NPT_NPHscaleVecs[1]) == 0 && (fabs(pSPARC->initialLatVecAngles[0]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[2]) <= 1e-12) ){ //Special constraint 2: a and c can vary, b must not vary. Further, the angle between b and a, or between b and c, must be 90 degrees.
		pSPARC->Pm_metric_tensor[4] = 0;
		pSPARC->Pm_metric_tensor[1] = 0;
		pSPARC->Pm_metric_tensor[3] = 0;
		pSPARC->Pm_metric_tensor[5] = 0;
		pSPARC->Pm_metric_tensor[7] = 0;
	}
	else if ( (NPT_NPHscaleVecs[0]) == 0 && (fabs(pSPARC->initialLatVecAngles[1]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[2]) <= 1e-12) ){ //Special constraint 1: b and c can vary, a must not vary. Further, the angle between a and b, or between a and c, must be 90 degrees.
		pSPARC->Pm_metric_tensor[0] = 0;
		pSPARC->Pm_metric_tensor[1] = 0;
		pSPARC->Pm_metric_tensor[2] = 0;
		pSPARC->Pm_metric_tensor[3] = 0;
		pSPARC->Pm_metric_tensor[6] = 0;
	}


	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, baro_const3, pSPARC->Pm_metric_tensor, 3, pSPARC->metric_tensor, 3, 0.0, temp_mat_a, 3);
	//Since these are 3x3 matrix, transposing without for loop (for avoiding loop overhead cost)
	temp_mat_b[0] = temp_mat_a[0]; temp_mat_b[4] = temp_mat_a[4]; temp_mat_b[8] = temp_mat_a[8];
	temp_mat_b[1] = temp_mat_a[3]; temp_mat_b[2] = temp_mat_a[6]; temp_mat_b[5] = temp_mat_a[7];
	temp_mat_b[3] = temp_mat_a[1]; temp_mat_b[6] = temp_mat_a[2]; temp_mat_b[7] = temp_mat_a[5]; 

	//Update the kinetic energy of the barostat
	pSPARC->Kbaro = 0.5 * baro_const1 * cblas_ddot(9, temp_mat_a, 1, temp_mat_b, 1);//Kinetic
	pSPARC->Ubaro = pSPARC->pressure_external * pSPARC->volumeCell + 0.5 * cblas_ddot(9, pSPARC->external_stress_lattice, 1, pSPARC->metric_tensor, 1); //Potential
	
	
	// bring the momenta of the Ionic particles in time-sync with the positions (since mometa are delayed by dt/2)
	// This setup corresponds to Eqn. 18i in Hernandez paper
	double *ion_forces_fractional = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
	if (ion_forces_fractional == NULL) {
		fprintf(stderr, "Error: Memory allocation failed for ionic forces fractional array.\n");
		exit(EXIT_FAILURE);
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, pSPARC->n_atom, 3, 3, 1.0, pSPARC->forces, 3, pSPARC->full_lattice, 3, 0.0, ion_forces_fractional, 3); 
	// Eqn. 18i: momentum += 0.5 * dt * S * D2C
	cblas_daxpy(3 * pSPARC->n_atom, 0.5 * pSPARC->MD_dt * pSPARC->SNOSE[0], ion_forces_fractional, 1, pSPARC->Pm_ion, 1);
	//Update the kinetic energy of the Ionic particles
	Calculate_Ionic_particles_Kinetic_energy(pSPARC);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0, pSPARC->Pm_ion, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, pSPARC->ion_vel_fractional, 3); 
	count = 0;
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
		cblas_dscal(3 * pSPARC->nAtomv[ityp], 1.0 / pSPARC->Mass[ityp], &pSPARC->ion_vel_fractional[count], 1);
		count += 3 * pSPARC->nAtomv[ityp];
	}
	// Calculate kinetic stress
	Calculate_Kinetic_stress(pSPARC);
	
	// Calculate total internal pressure
	for (int i = 0; i < 9; i++){
		pSPARC->total_internal_stress[i] = ( pSPARC->kinetic_stress[i] - pSPARC->internal_stress_fractional[i] - pSPARC->constraint_stress[i] ) / pSPARC->volumeCell; 
	}	
	pSPARC->internal_pressure = 2.0 / 3.0 * cblas_ddot(9, pSPARC->total_internal_stress, 1, pSPARC->metric_tensor, 1);
	
	//Update Hamiltonian
	double sumAllHamilTerms = pSPARC->KE + pSPARC->Etot + pSPARC->Kther + pSPARC->Uther + pSPARC->Kbaro + pSPARC->Ubaro; 
	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){

		//TODO: The three lines below which are currently commented can be uncommented in case and checked in future. They are placed here if one wants to check forming the initial hamiltonian at t = 0, currently the initial hamiltonian is formed at t = -dt/2

		//  if ((pSPARC->MDCount == 1)  && (pSPARC->RestartFlag != 1)) {
		//  	pSPARC->init_Hamil_NPT_NP = sumAllHamilTerms ;
		//  }
		pSPARC->Hamiltonian_NPT_NP = pSPARC->SNOSE[0] * (sumAllHamilTerms - pSPARC->init_Hamil_NPT_NP);  //Eqn. 8 in the Hernandez paper
	}
	else {

		//TODO: The three lines below which are currently commented can be uncommented in case and checked in future. They are placed here if one wants to check forming the initial hamiltonian at t = 0, currently the initial hamiltonian is formed at t = -dt/2

		//  if ((pSPARC->MDCount == 1)  && (pSPARC->RestartFlag != 1)) {
		//  	pSPARC->init_Hamil_NPH = sumAllHamilTerms ;
		//  }
		pSPARC->Hamiltonian_NPH = pSPARC->SNOSE[0] * (sumAllHamilTerms - pSPARC->init_Hamil_NPH);  //Eqn. 8 in the Hernandez paper
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
	    printf("Sum of all energy terms (Ha)    : %12.9f\n", sumAllHamilTerms);
		if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			printf("Hamiltonian (Ha)                : %12.9f\n", pSPARC->Hamiltonian_NPT_NP);
		}
		else {
			printf("Hamiltonian (Ha)                : %12.9f\n", pSPARC->Hamiltonian_NPH);
		}
	}
	#endif

	// For printing, convert the total internal stress, constraint stress and the kinetic stress to cartesian coordinates
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->full_lattice, 3, pSPARC->total_internal_stress, 3, 0.0, temp_mat, 3);  //Already divided by volume earlier
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 2.0, temp_mat, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->total_internal_stress, 3); //Mutliplied by 2 so as to remove the effect of previous divisions by 2 in the kinetic_stress, internal_stress_fractional
	
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0 / pSPARC->volumeCell, pSPARC->full_lattice, 3, pSPARC->kinetic_stress, 3, 0.0, temp_mat, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 2.0, temp_mat, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->kinetic_stress, 3); //Mutliplied by 2 so as to remove the effect of previous division by 2 in the kinetic_stress, and negated so as to follow SPARC convention of ion-kinetic stress

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0 / pSPARC->volumeCell, pSPARC->full_lattice, 3, pSPARC->constraint_stress, 3, 0.0, temp_mat, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 2.0, temp_mat, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->constraint_stress, 3); 

	// ------------------------------------- END: Updating Momenta by half step (Eqns. 18g, 18h, 18i)----------------------------------//
	

	// ------------------------------------- BEGIN: PRINTING FOR NPT_NP OR NPH ENSEMBLE (this marks the end of one timestep) --------------------------------------//

	// Obtain updated velocities in cartesian coordinates for use in MD_QOI function
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0 / pSPARC->SNOSE[0], pSPARC->ion_vel_fractional, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->ion_vel, 3); 
	
	#ifdef DEBUG
		if (!rank) printf("\nend NPT_NP timestep %d\n", pSPARC->MDCount);
	#endif
	int Count = pSPARC->MDCount + pSPARC->restartCount; 
	int check1 = (pSPARC->PrintMDout == 1 && !rank);
	
	FILE *output_md;
	FILE *output_fp;

	MD_QOI(pSPARC, avgvel, maxvel, mindis); // calculates the quantities of interest in an MD simulation
	if(check1) {
		output_md = fopen(pSPARC->MDFilename,"a+");
		if (output_md == NULL) {
			printf("\nCannot open file \"%s\"\n",pSPARC->MDFilename);
			exit(EXIT_FAILURE);
		}
		fprintf(output_md,"\n\n");
		fprintf(output_md,":MDSTEP: %d\n", Count);
		fprintf(output_md,":MDTM: %.2f\n", MPI_Wtime() - pSPARC->t_run_md);
		Print_fullMD(pSPARC, output_md, avgvel, maxvel, mindis); // prints the QOI in the .aimd file
		fclose(output_md);
	} 
	
	#ifdef DEBUG
		if (!rank) printf("Time taken by MDSTEP %d: %.3f s.\n", Count, (MPI_Wtime() - pSPARC->t_run_md));
	#endif
	if(!rank){
		output_fp = fopen(pSPARC->OutFilename,"a");
		if (output_fp == NULL) {
			printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
			exit(EXIT_FAILURE);
		}
		fprintf(output_fp,"MD step time       :  %.3f (sec)\n", (MPI_Wtime() - pSPARC->t_run_md));
		fclose(output_fp);
	}

	//	(this marks the start of MDCount^{th} step ), but do not move pSPARC->MDCount increment here, it is affecting other functions
	pSPARC->t_run_md = MPI_Wtime();
	#ifdef DEBUG
		if(!rank) printf(":MDSTEP: %d\n",  pSPARC->MDCount + pSPARC->restartCount);
	#endif
	
	// ------------------------------------- END: PRINTING FOR NPT_NP OR NPH ENSEMBLE (this marks the beginning of new timestep) --------------------------------------//


	// ------------------------------------- BEGIN: Updating Momenta by second half step (Eqns. 18a, 18b, 18c)	
	//				   And updating the position variable of the themostat if doing NPT_NP emsemble (Eqn. 18d) ----------------------------------//
				
	// Now we update/Step-up the momenta of the Ionic particles by dt/2
	// This setup corresponds to Eqn. 18a in Hernandez paper
	cblas_daxpy(3 * pSPARC->n_atom, 0.5 * pSPARC->MD_dt * pSPARC->SNOSE[0], ion_forces_fractional, 1, pSPARC->Pm_ion, 1);
	free(ion_forces_fractional);
	//Update the kinetic energy of the Ionic particles
	Calculate_Ionic_particles_Kinetic_energy(pSPARC);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0, pSPARC->Pm_ion, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, pSPARC->ion_vel_fractional, 3); 
	count = 0;
	for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
		cblas_dscal(3 * pSPARC->nAtomv[ityp], 1.0 / pSPARC->Mass[ityp], &pSPARC->ion_vel_fractional[count], 1);
		count = count + 3 * pSPARC->nAtomv[ityp];
	}
	// Calculate kinetic stress
	Calculate_Kinetic_stress(pSPARC);	
	
	// Now we update/Step-up the momenta of the barostat by dt/2
	// This setup corresponds to Eqn. 18b in the Hernandez paper
	// This needs to be solved iteratively
	Update_metric_tensor_momenta_iteratively_half_step(pSPARC);

	//Now impose the constraints on it
	for (int i = 0; i < 3; i++){ // If any of the lattice vectors is constrained from being rescaled, set its momentum to 0
		if (NPT_NPHscaleVecs[i] == 0){
			pSPARC->Pm_metric_tensor[4 * i] = 0; //setting that corresponding diagonal element to 0
		}
	}

	if ( (NPT_NPHscaleVecs[2]) == 0 && (fabs(pSPARC->initialLatVecAngles[0]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[1]) <= 1e-12) ){ //Special constraint 1: a and b can vary, c must not vary. Further, the angle between c and a, or between c and b, must be 90 degrees.
		pSPARC->Pm_metric_tensor[8] = 0;
		pSPARC->Pm_metric_tensor[7] = 0;
		pSPARC->Pm_metric_tensor[6] = 0;
		pSPARC->Pm_metric_tensor[5] = 0;
		pSPARC->Pm_metric_tensor[2] = 0;
	}
	else if ( (NPT_NPHscaleVecs[1]) == 0 && (fabs(pSPARC->initialLatVecAngles[0]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[2]) <= 1e-12) ){ //Special constraint 2: a and c can vary, b must not vary. Further, the angle between b and a, or between b and c, must be 90 degrees.
		pSPARC->Pm_metric_tensor[4] = 0;
		pSPARC->Pm_metric_tensor[1] = 0;
		pSPARC->Pm_metric_tensor[3] = 0;
		pSPARC->Pm_metric_tensor[5] = 0;
		pSPARC->Pm_metric_tensor[7] = 0;
	}
	else if ( (NPT_NPHscaleVecs[0]) == 0 && (fabs(pSPARC->initialLatVecAngles[1]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[2]) <= 1e-12) ){ //Special constraint 1: b and c can vary, a must not vary. Further, the angle between a and b, or between a and c, must be 90 degrees.
		pSPARC->Pm_metric_tensor[0] = 0;
		pSPARC->Pm_metric_tensor[1] = 0;
		pSPARC->Pm_metric_tensor[2] = 0;
		pSPARC->Pm_metric_tensor[3] = 0;
		pSPARC->Pm_metric_tensor[6] = 0;
	}


	// Now we update/Step-up the momenta of the thermostat by dt/2
	// This setup corresponds to Eqn. 18c in the Hernandez paper
	// Skip this step if doing NPH ensemble
	if (pSPARC->NPT_NP_qmass > 0){
		double factor;
		factor = 0.5 * pSPARC->MD_dt *( pSPARC->dof * ktemp * ( log(pSPARC->SNOSE[0]) + 1 ) - pSPARC->KE + pSPARC->Etot + pSPARC->Kbaro + pSPARC->Ubaro - pSPARC->init_Hamil_NPT_NP ) - pSPARC->NPT_NP_qmass * pSPARC->SNOSE[1];
	
		#ifdef DEBUG
		if (rank == 0)
			printf("factor is %12.9f\n", factor);
		#endif
		if ((1.0 - factor * pSPARC->MD_dt / pSPARC->NPT_NP_qmass) < 0.0){
			if (rank == 0)
				printf("WARNING:  The mass of thermostat variable NPT_NP_qmass is too small. Please try a larger thermostat mass.");
		}
		
		pSPARC->SNOSE[1] = -2.0 * factor / (pSPARC->NPT_NP_qmass * (1.0 + sqrt(1.0 - factor * pSPARC->MD_dt / pSPARC->NPT_NP_qmass)));
		#ifdef DEBUG
		if (rank == 0) 
			printf("SNOSE[1] in the 2nd half step is %12.9f\n", pSPARC->SNOSE[1]);
		#endif
	}


	// Now we update/Step-up the Position of the thermostat by dt/2
	// This setup corresponds to Eqn. 18d in the Hernandez paper
	double S_new = 1.0; double S_temp = pSPARC->SNOSE[0];
	int TimeIter = 0;
	if (pSPARC->NPT_NP_qmass > 0){ // This gets executes when running NPT_NP ensemble  (skip if running NPH ensemble)
		while (1) {
			S_new = pSPARC->SNOSE[0] + 0.5 * pSPARC->MD_dt * (pSPARC->SNOSE[0] + S_temp) * pSPARC->SNOSE[1];
			if (fabs(S_temp - S_new) < 1e-7) {
				break; 
			}
			S_temp = S_new;
			TimeIter++;
			//it iteration count exceeds max iteration counts, exit 
			if (TimeIter > pSPARC->maxTimeIter){
				if (rank == 0){
					printf("%15.7f \n", S_new);		
					printf("Reminder: The themostat position SNOSE[0] does not converge to %e tolerance in %d timesteps : stopping\n", 1e-7, pSPARC->maxTimeIter );
					exit(1);
				}
			}
		}
		#ifdef DEBUG
		if (rank == 0)
			printf("S_temp is %12.9f\n", S_temp);
		#endif
	}
	else{ // This gets executes when running NPH ensemble
		pSPARC->SNOSE[0] = 1.0;
		pSPARC->SNOSE[1] = 0.0;
	}
	
	// ------------------------------------- END: Updating Momenta by second half step (Eqns. 18a, 18b, 18c)	
	//				   END: And updating the position variable of the themostat if doing NPT_NP emsemble (Eqn. 18d) ----------------------------------//
		 

	// ------------------------------------- BEGIN: Updating Components of the metric tensor (Eqn. 18e)----------------------------------//
	//							And updating the atomic positions  (Eqn. 18f), and restoring ionic velocties ----------------------------//	
	
	pSPARC->Kbaro = pSPARC->Kbaro * pSPARC->volumeCell * pSPARC->volumeCell;

	//Eqn. 18e is implicitly solved in the below subroutine
	Update_metric_tensor_components_iteratively_full_step(pSPARC, S_new, NPT_NPH_ANGLES, NPT_NPHconstraintFlag); 

	//Before updating cell parameters, convert cartesian atomic position coordinates to fractional
	double *atom_pos_fractional = (double *)malloc(3 * pSPARC->n_atom * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0, pSPARC->atom_pos, 3, pSPARC->reciprocal_lattice, 3, 0.0, atom_pos_fractional, 3); 
	//Update cell parameters: lattice vectors, reciprocal lattice vectors, volume of the cell, reciprocal metric tensor, cell lattice velocities using the updated metric tensor
	fetch_MD_cell_ingredients(pSPARC, true);
	//Update Atomic position in fractional coordinates (Eqn. 18f in Hernandez paper)
	cblas_daxpy(3 * pSPARC->n_atom, 0.5 * pSPARC->MD_dt * (1.0 / pSPARC->SNOSE[0] + 1.0 / S_new), pSPARC->ion_vel_fractional, 1, atom_pos_fractional, 1);
	// Now reconvert atomic positions to cartesian coordinates
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0, atom_pos_fractional, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->atom_pos, 3); 
	free(atom_pos_fractional);
	// Now reconvert velocity of ions to cartesian coordinates (this step is not actually needed, as these ionic velocities are not used anywhere in NPT_NP)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pSPARC->n_atom, 3, 3, 1.0 / S_new, pSPARC->ion_vel_fractional, 3, pSPARC->full_lattice, 3, 0.0, pSPARC->ion_vel, 3); 
	//cblas_dscal(3 * pSPARC->n_atom, 1.0 / S_new, pSPARC->ion_vel_fractional, 1);


	//Update the Kinetic and Potential energy of the barostat based on new metric tensor and new cell volume
	pSPARC->Kbaro = pSPARC->Kbaro / ( pSPARC->volumeCell * pSPARC->volumeCell );  //Kinetic
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, pSPARC->volumeCell, pSPARC->external_stress_cartesian, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, pSPARC->external_stress_lattice, 3);
	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, pSPARC->volumeCell, pSPARC->external_stress_cartesian, 3, pSPARC->reciprocal_lattice, 3, 0.0, temp_mat, 3);
	//cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_lattice, 3, temp_mat, 3, 0.0, pSPARC->external_stress_lattice, 3);
	
	pSPARC->Ubaro = pSPARC->pressure_external * pSPARC->volumeCell + 0.5 * cblas_ddot(9, pSPARC->external_stress_lattice, 1, pSPARC->metric_tensor, 1);  //Potential
	
	//Update kinetic energy and kinetic stress based on new S
	pSPARC->KE *=  (pSPARC->SNOSE[0] * pSPARC->SNOSE[0]) / ( S_new * S_new );
	
	for (int i = 0; i < 9; i++){
		pSPARC->kinetic_stress[i] *= (pSPARC->SNOSE[0] * pSPARC->SNOSE[0]) / ( S_new * S_new );
	}

	// Update SNOSE[0] to S_new
	if (pSPARC->NPT_NP_qmass > 0){
		pSPARC->SNOSE[2] = pSPARC->SNOSE[0];
		pSPARC->SNOSE[0] = S_new;
	}
	pSPARC->ion_T = 2 * pSPARC->KE /(pSPARC->kB * pSPARC->dof); //Update Ionic temperature;
	// ------------------------------------- END: Updating Components of the metric tensor (Eqn. 18e)----------------------------------//
	//						END:And updating the atomic positions  (Eqn. 18f), and restoring ionic velocties --------------------------//	
}
	

/** 
 *@brief function to impose constraints on the update of barostat momentum. To be called once per NPT_NP or NPH step.
If and When this function is called, it imposes a universal constraint of all angles between lattice vectors being FIXED; 
This is on top of other constraints explicitly imposed in the function through 'if' condition
*/
void compute_constraint_stress(SPARC_OBJ *pSPARC, int NPT_NPHconstraintFlag, int *NPT_NPHscaleVecs){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//Initialize some useful constants
	double a_norm; double b_norm; double c_norm; 
	double da_dt_norm; double db_dt_norm; double dc_dt_norm;
	double baro_const4; double thermo_const1;
	
	//Initialize empty temporary matrices and vectors 
	double gpi[9]; double gpig[9]; double constraint_velocity[9];

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->metric_tensor, 3,pSPARC->Pm_metric_tensor, 3, 0.0, gpi, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, gpi, 3,pSPARC->metric_tensor, 3, 0.0, gpig, 3);

	a_norm = sqrt(pSPARC->full_lattice[0] * pSPARC->full_lattice[0] + pSPARC->full_lattice[1] * pSPARC->full_lattice[1] + pSPARC->full_lattice[2] * pSPARC->full_lattice[2]);
	b_norm = sqrt(pSPARC->full_lattice[3] * pSPARC->full_lattice[3] + pSPARC->full_lattice[4] * pSPARC->full_lattice[4] + pSPARC->full_lattice[5] * pSPARC->full_lattice[5]);
	c_norm = sqrt(pSPARC->full_lattice[6] * pSPARC->full_lattice[6] + pSPARC->full_lattice[7] * pSPARC->full_lattice[7] + pSPARC->full_lattice[8] * pSPARC->full_lattice[8]);
	
	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		baro_const4 = pSPARC->SNOSE[0] / (pSPARC->NPT_NP_bmass * pSPARC->volumeCell * pSPARC->volumeCell); // M_G*det(G) in the Hernandez paper
	}
	else{
		baro_const4 = pSPARC->SNOSE[0] / (pSPARC->NPH_bmass * pSPARC->volumeCell * pSPARC->volumeCell); // M_G*det(G) in the Hernandez paper
	}
	
	da_dt_norm = baro_const4 * gpig[0] / (2.0 * a_norm);
	db_dt_norm = baro_const4 * gpig[4] / (2.0 * b_norm);
	dc_dt_norm = baro_const4 * gpig[8] / (2.0 * c_norm);

	for (int i = 0; i < 3; i++){ // If any of the lattice vectors is constrained from being rescaled, set its momentum to 0
		if (NPT_NPHscaleVecs[i] == 0){
			gpig[4 * i] = 0; //setting that corresponding diagonal element to 0
		}
	}

	if (NPT_NPHconstraintFlag == 4){ // Cell vectors are constrained to ISOTROPIC scaling;  and all angles between lattice vectors are FIXED
		gpig[0] = (gpig[0] + gpig[4] + gpig[8]) / 3;
		gpig[4] = gpig[0]; 
		gpig[8] = gpig[0];
	}

	else if (NPT_NPHconstraintFlag == 1){ // |a| = |b| and |c| can vary independently;  and all angles between lattice vectors are FIXED
		gpig[0] = (gpig[0] + gpig[4]) / 2;
		gpig[4] = gpig[0]; 
	}

	else if (NPT_NPHconstraintFlag == 2){ // |a| = |c| and |b| can vary independently;  and all angles between lattice vectors are FIXED
		gpig[0] = (gpig[0] + gpig[8]) / 2;
		gpig[8] = gpig[0]; 
	}
	
	else if (NPT_NPHconstraintFlag == 3){ // |b| = |c| and |a| can vary independently;  and all angles between lattice vectors are FIXED
		gpig[4] = (gpig[4] + gpig[8]) / 2;
		gpig[8] = gpig[4]; 
	}
	
	if ( (NPT_NPHscaleVecs[2]) == 0 && (fabs(pSPARC->initialLatVecAngles[0]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[1]) <= 1e-12) ){ // Only |a| and |b| varies, no changes in third direction i.e |c| is fixed;  and all angles between lattice vectors are FIXED, and further the lattice vector 'c' must be orthogonal to lattice vectors 'a' and 'b'
		gpig[8] = 0;
		gpig[7] = 0;
		gpig[6] = 0;
		gpig[5] = 0;
		gpig[2] = 0;
	}

	else if ( (NPT_NPHscaleVecs[1]) == 0 && (fabs(pSPARC->initialLatVecAngles[0]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[2]) <= 1e-12) ){ // Only |a| and |c| varies, no changes in third direction i.e |b| is fixed;  and all angles between lattice vectors are FIXED, and further the lattice vector 'b' must be orthogonal to lattice vectors 'a' and 'c'
		gpig[4] = 0;
		gpig[1] = 0;
		gpig[3] = 0;
		gpig[5] = 0;
		gpig[7] = 0;
	}

	else if ( (NPT_NPHscaleVecs[0]) == 0 && (fabs(pSPARC->initialLatVecAngles[1]) <= 1e-12) && (fabs(pSPARC->initialLatVecAngles[2]) <= 1e-12) ){ // Only |a| and |b| varies, no changes in third direction i.e |c| is fixed;  and all angles between lattice vectors are FIXED, and further the lattice vector 'a' must be orthogonal to lattice vectors 'b' and 'c'
		gpig[0] = 0;
		gpig[1] = 0;
		gpig[2] = 0;
		gpig[3] = 0;
		gpig[6] = 0;
	}

	constraint_velocity[0] = gpig[0] * baro_const4;
	constraint_velocity[4] = gpig[4] * baro_const4;
	constraint_velocity[8] = gpig[8] * baro_const4;
	
	//Impose the constraint on angles being FIXED
	constraint_velocity[1] = (da_dt_norm * b_norm + a_norm * db_dt_norm) * pSPARC->initialLatVecAngles[2];
	constraint_velocity[2] = (da_dt_norm * c_norm + a_norm * dc_dt_norm) * pSPARC->initialLatVecAngles[1];
	constraint_velocity[5] = (db_dt_norm * c_norm + b_norm * dc_dt_norm) * pSPARC->initialLatVecAngles[0];

	constraint_velocity[3] = constraint_velocity[1];
	constraint_velocity[6] = constraint_velocity[2];
	constraint_velocity[7] = constraint_velocity[5];

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->reciprocal_metric_tensor, 3, constraint_velocity, 3, 0.0, gpi, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0 / baro_const4, gpi, 3, pSPARC->reciprocal_metric_tensor, 3, 0.0, gpig, 3);

	thermo_const1 = 2.0 / (pSPARC->MD_dt * pSPARC->SNOSE[0]);

	// Calculating constraint stress
	for (int i = 0; i < 9; i++){
		pSPARC->constraint_stress[i] = thermo_const1 * (pSPARC->Pm_metric_tensor[i] - gpig[i]);
		pSPARC->Pm_metric_tensor[i] = gpig[i];
	}
}

/** 
 *@brief function to update the Metric tensor components in NPT_NP and NPH ensemble iteratively in one single step (from time = t to t+dt).
This is Eqn. 18e in the Hernandez paper 
To be called once per NPT_NP or NPH step.
*/
void Update_metric_tensor_components_iteratively_full_step(SPARC_OBJ *pSPARC, double S_new, int NPT_NPH_ANGLES, int NPT_NPHconstraintFlag){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//Initialize some useful constants
	bool converged;  // determine whether the iteration converges or not
	int TimeIter = 0; // current iteration count
	const double tolerance = 1e-7; // tolerance criterion for checking convergence
	double baro_const11;
	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		baro_const11 = 0.5 * pSPARC->MD_dt / (pSPARC->NPT_NP_bmass);
	}
	else{
		baro_const11 = 0.5 * pSPARC->MD_dt / (pSPARC->NPH_bmass);
	}

	double baro_const6 = baro_const11 * pSPARC->SNOSE[0] / (pSPARC->volumeCell * pSPARC->volumeCell);
	double baro_const7;
	double det_temp_metric_tensor;
	double a_norm; double b_norm; double c_norm; 

	//Initialize empty temporary matrices and vectors 
	double gpi[9]; double gpig[9]; double gpig_old[9];
	double temp_metric_tensor[9]; double new_metric_tensor[9];

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, pSPARC->metric_tensor, 3, pSPARC->Pm_metric_tensor, 3, 0.0, gpi, 3);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, gpi, 3, pSPARC->metric_tensor, 3, 0.0, gpig, 3);

	for (int i = 0; i < 9; i++){
		temp_metric_tensor[i] = pSPARC->metric_tensor[i];
		gpig_old[i] = gpig[i];
	}

	while (1){

		det_temp_metric_tensor = temp_metric_tensor[0] * ( temp_metric_tensor[4] * temp_metric_tensor[8] - temp_metric_tensor[7] * temp_metric_tensor[5] ) 
							   + temp_metric_tensor[1] * ( temp_metric_tensor[5] * temp_metric_tensor[6] - temp_metric_tensor[3] * temp_metric_tensor[8] )
							   + temp_metric_tensor[2] * ( temp_metric_tensor[3] * temp_metric_tensor[7] - temp_metric_tensor[6] * temp_metric_tensor[4] ) ;
							  
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, temp_metric_tensor, 3, pSPARC->Pm_metric_tensor, 3, 0.0, gpi, 3);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, gpi, 3, temp_metric_tensor, 3, 0.0, gpig, 3);
				 
		baro_const7 = baro_const11 * S_new / det_temp_metric_tensor;
		
		for (int i = 0; i < 9; i++){
			new_metric_tensor[i] = pSPARC->metric_tensor[i] + baro_const6 * gpig_old[i] + baro_const7 * gpig[i];
		}

		converged = true;
		for (int i = 0; i < 9; i++){
			if ( fabs(new_metric_tensor[i] - temp_metric_tensor[i]) > tolerance ){
				converged = false;
				break;
			}
		}

		if (converged){
			break;
		} else{
			cblas_dscal(9, 0.5 , new_metric_tensor, 1);
			transpose_and_add(new_metric_tensor);
			for (int i = 0; i < 9; i++){
				temp_metric_tensor[i] = new_metric_tensor[i];
			}
		}

		TimeIter++;
		//it iteration count exceeds max iteration counts, exit 
		if (TimeIter > pSPARC->maxTimeIter){
			for(int row = 0; row < 3; row++){
				if (rank == 0){
					printf("%15.7f %15.7f %15.7f\n",new_metric_tensor[row * 3], 
								new_metric_tensor[row * 3 + 1], new_metric_tensor[row * 3 + 2]);
					printf("Reminder The barostat components: metric_tensor does not converge to %e tolerance in %d timesteps : stopping\n", tolerance, pSPARC->maxTimeIter );
					exit(1);
				}
			}
		}
	}

	if (NPT_NPH_ANGLES == 0){
		if (NPT_NPHconstraintFlag == 4){
			new_metric_tensor[0] = ( new_metric_tensor[0] + new_metric_tensor[4] + new_metric_tensor[8]) / 3.0;
			new_metric_tensor[4] = new_metric_tensor[0];
				new_metric_tensor[8] = new_metric_tensor[0];
		}

		else if (NPT_NPHconstraintFlag == 1){
			new_metric_tensor[0] = ( new_metric_tensor[0] + new_metric_tensor[4] ) / 2.0;
			new_metric_tensor[4] = new_metric_tensor[0];
		}

		else if (NPT_NPHconstraintFlag == 2){
			new_metric_tensor[0] = ( new_metric_tensor[0] + new_metric_tensor[8] ) / 2.0;
			new_metric_tensor[8] = new_metric_tensor[0];
		}

		else if (NPT_NPHconstraintFlag == 3){
			new_metric_tensor[4] = ( new_metric_tensor[4] + new_metric_tensor[8] ) / 2.0;
			new_metric_tensor[8] = new_metric_tensor[4];
		}

		a_norm = sqrt( new_metric_tensor[0] );
		b_norm = sqrt( new_metric_tensor[4] );
		c_norm = sqrt( new_metric_tensor[8] );

		new_metric_tensor[1] = a_norm * b_norm * pSPARC->initialLatVecAngles[2];
		new_metric_tensor[2] = a_norm * c_norm * pSPARC->initialLatVecAngles[1];
		new_metric_tensor[5] = b_norm * c_norm * pSPARC->initialLatVecAngles[0];

		new_metric_tensor[3] = new_metric_tensor[1];
		new_metric_tensor[6] = new_metric_tensor[2];
		new_metric_tensor[7] = new_metric_tensor[5];

		for (int i = 0; i < 9; i++){
			temp_metric_tensor[i] = new_metric_tensor[i];
		}	
	}	

	for (int i = 0; i < 9; i++){
		pSPARC->metric_tensor[i] = temp_metric_tensor[i];
	}

	#ifdef DEBUG
	if (rank == 0) {
		for (int i = 0; i < 9; i++){
			printf("pSPARC->metric_tensor[%d] is %12.9f\n", i, pSPARC->metric_tensor[i]);
		}
	}
	#endif

}

/** 
 *@brief function to update the momentum metric tensor (barostat variable) in NPT_NP and NPH ensemble iteratively in one half step (from time = t to t+dt/2).
This is Eqn. 18b in the Hernandez paper 
To be called once per NPT_NP or NPH step. The other half update happens within the 'NPT_NP_and_NPH_main' subroutine
*/
void Update_metric_tensor_momenta_iteratively_half_step(SPARC_OBJ *pSPARC){
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//Initialize some useful constants
	bool converged;  // determine whether the iteration converges or not
	int TimeIter = 0;  // current iteration count
	const double tolerance = 1e-12; // tolerance criterion for checking convergence
	double baro_const0, baro_const3;

	if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
		baro_const0 = 0.5 * (pSPARC->NPT_NP_bmass * pSPARC->volumeCell * pSPARC->volumeCell);
		baro_const3 = 0.5 / baro_const0;
		
	}
	else{
		baro_const0 = 0.5 * (pSPARC->NPH_bmass * pSPARC->volumeCell * pSPARC->volumeCell);
		baro_const3 = 0.5 / baro_const0;
		
	}
	
	//Initialize empty temporary matrices and vectors 
	double temp_mat[9]; 
	double temp_mat_1[9]; double temp_mat_2[9]; double temp_mat_3[9]; 
	double temp_Pm_metric_tensor[9]; double new_Pm_metric_tensor[9] = {0.0};

	for (int i = 0; i < 9; i++){
		temp_Pm_metric_tensor[i] = pSPARC->Pm_metric_tensor[i];
	}

	// Now iteratively solve equation 18b (i.e. find the new momenta of the barostat at time t+dt/2)
	while (1){
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, baro_const3, temp_Pm_metric_tensor, 3,  pSPARC->metric_tensor, 3, 0.0, temp_mat_1, 3);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, temp_mat_1, 3, temp_Pm_metric_tensor, 3, 0.0, temp_mat_2, 3);
	
		//Transpose
		temp_mat_3[0] = temp_mat_1[0]; temp_mat_3[4] = temp_mat_1[4]; temp_mat_3[8] = temp_mat_1[8];
		temp_mat_3[1] = temp_mat_1[3]; temp_mat_3[2] = temp_mat_1[6]; temp_mat_3[5] = temp_mat_1[7];
		temp_mat_3[3] = temp_mat_1[1]; temp_mat_3[6] = temp_mat_1[2]; temp_mat_3[7] = temp_mat_1[5]; 

		pSPARC->Kbaro = baro_const0 * cblas_ddot(9, temp_mat_1, 1, temp_mat_3, 1);

		// Eqn. 18b  Hernandez paper
		for (int i = 0; i < 9; i++){
			temp_mat[i] = pSPARC->internal_stress_fractional[i] - pSPARC->kinetic_stress[i] + temp_mat_2[i];
			temp_mat[i] += (0.5 * pSPARC->pressure_external * pSPARC->volumeCell - pSPARC->Kbaro) * pSPARC->reciprocal_metric_tensor[i] + 0.5 * pSPARC->external_stress_lattice[i];
			new_Pm_metric_tensor[i] =  pSPARC->Pm_metric_tensor[i]  - 0.5 * pSPARC->MD_dt * pSPARC->SNOSE[0] * temp_mat[i];
		} 

		cblas_dscal(9, 0.5 , new_Pm_metric_tensor, 1);
		transpose_and_add(new_Pm_metric_tensor);

		//Check convergence
		converged = true;
		for (int i = 0; i < 9; i++){
			if ( fabs(new_Pm_metric_tensor[i] - temp_Pm_metric_tensor[i]) > tolerance ){
				converged = false;
				break;
			}
		}

		// if yes then break; else resolve
		if (converged){
			break;
		} else{
			for (int i = 0; i < 9; i++){
				temp_Pm_metric_tensor[i] = new_Pm_metric_tensor[i];
			}
		}

		TimeIter++;
		//it iteration count exceeds max iteration counts, exit 
		if (TimeIter > pSPARC->maxTimeIter){
			for(int row = 0; row < 3; row++){
				if (rank == 0){
					printf("%15.7f %15.7f %15.7f\n",new_Pm_metric_tensor[row * 3 + 0], 
								new_Pm_metric_tensor[row * 3 + 1], new_Pm_metric_tensor[row * 3 + 2]);
					printf("Reminder The barostat momentum Pm_metric_tensor does not converge to %e tolerance in %d timesteps : stopping\n", tolerance, pSPARC->maxTimeIter );
					exit(1);
				}
			}
		}

	}

	// As converged,  update the metric tensor momenta
	for (int i = 0; i < 9; i++){
		pSPARC->Pm_metric_tensor[i] = temp_Pm_metric_tensor[i];
		#ifdef DEBUG
		if (rank == 0)
				printf("pSPARC->Pm_metric_tensor[%d] in 2nd half step is %12.9f\n", i, pSPARC->Pm_metric_tensor[i]);
		#endif
	}
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
* @ brief: function to check if the atoms are too close to the boundary in case of bounded domain or to each other in general
*/
void Check_atomlocation(SPARC_OBJ *pSPARC) {
    int rank, ityp, i, atm, atm2, count;
	double temp, rc1 = 0.0, rc2 = 0.0, *rc, tol = 0.5;// Change tol according to the situation
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

	wraparound_dynamics(pSPARC, pSPARC->atom_pos, 1);
}

/*
* @ brief: function to wraparound atom positions for PBC
*/
void wraparound_dynamics(SPARC_OBJ *pSPARC, double *coord, int opt) {

	int rank, atm, dir = 0, maxdir = 3, BC;
	double length, coord_temp;// Change tol according to the situation
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	// Convert Cart to nonCart coordinates for non orthogonal cell
    if(pSPARC->cell_typ != 0){
        for(atm = 0; atm < pSPARC->n_atom; atm++)
	        Cart2nonCart_coord(pSPARC, coord+3*atm, coord+3*atm+1, coord+3*atm+2);
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
			if (!((pSPARC->CyclixFlag) && (dir == 0))) { // if it is in cyclix coordinate and in x (radial) direction, we will not check it
                for(atm = 0; atm < pSPARC->n_atom; atm++){
                    if(coord[atm * 3 + dir] >= length || coord[atm * 3 + dir] < 0){
                        if(!rank)
                            printf("ERROR: Atom number %d has crossed the boundary in %d direction",atm, dir);
                        exit(EXIT_FAILURE);
                    }
                }
            } 
		} else if(BC == 0){
			for(atm = 0; atm < pSPARC->n_atom; atm++){
				coord_temp = *(coord+3*atm+dir);
            	if (coord_temp < 0.0 || coord_temp >= length)
            	{
                	coord_temp = fmod(coord_temp,length);
					double new_coord = coord_temp + (coord_temp<0.0)*length;
					double shift = new_coord - *(coord+3*atm+dir);
					*(coord+3*atm+dir) = new_coord;
					if (pSPARC->CyclixFlag && opt == 1){
						wraparound_velocity(pSPARC, shift, dir, 3*atm);
					}
				}
			}
		}
		dir ++;
	}
}

/*
* @ brief: function to wraparound velocities in MD and displacement vectors in relaxation for PBC
*/
void wraparound_velocity(SPARC_OBJ *pSPARC, double shift, int dir, int loc) {

	if (dir == 1){
		if (pSPARC->MDFlag == 1){
			double vx = pSPARC->ion_vel[loc]; double vy = pSPARC->ion_vel[loc+1];
			pSPARC->ion_vel[loc] = cos(shift) * vx -sin(shift) * vy;
			pSPARC->ion_vel[loc+1] = sin(shift) * vx + cos(shift) * vy;
		}
		else if (pSPARC->RelaxFlag == 1){
			double vx = pSPARC->d[loc]; double vy = pSPARC->d[loc+1];
			pSPARC->d[loc] = cos(shift) * vx -sin(shift) * vy;
			pSPARC->d[loc+1] = sin(shift) * vx + cos(shift) * vy;
		}	
	} else if (dir == 2){
		if (pSPARC->MDFlag == 1){
			double vx = pSPARC->ion_vel[loc]; double vy = pSPARC->ion_vel[loc+1];
			pSPARC->ion_vel[loc] = cos(pSPARC->twist*shift) * vx -sin(pSPARC->twist*shift) * vy;
			pSPARC->ion_vel[loc+1] = sin(pSPARC->twist*shift) * vx + cos(pSPARC->twist*shift) * vy;
		}
		else if (pSPARC->RelaxFlag == 1){
			double vx = pSPARC->d[loc]; double vy = pSPARC->d[loc+1];
			pSPARC->d[loc] = cos(pSPARC->twist*shift) * vx -sin(pSPARC->twist*shift) * vy;
			pSPARC->d[loc+1] = sin(pSPARC->twist*shift) * vx + cos(pSPARC->twist*shift) * vy;
		}	
	}
		
}



/*
 @ brief: function to write all relevant DFT quantities generated during MD simulation
*/
void Print_fullMD(SPARC_OBJ *pSPARC, FILE *output_md, double *avgvel, double *maxvel, double *mindis) {
    (void)avgvel; (void)maxvel; // only used under #ifdef DEBUG (:AVGV:/:MAXV: printout below)
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
		if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0){
			if (pSPARC->Flag_latvec_scale == 0)
				fprintf(output_md,":Desc_CELL: lengths of three lattice vectors. Unit = Bohr \n");
			else
				fprintf(output_md,":Desc_LATVEC_SCALE: ratio of cell lattice vectors over input lattice vector. Unit = 1 \n");
			fprintf(output_md,":Desc_NPT_NH_HAMIL: Hamiltonian of the NPT_NH system, formula (5.4) in (M. E. Tuckerman et al, 2006). Unit = Ha \n");
		}

		if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH")==0){
			fprintf(output_md,":Desc_ANGLES: angles between cell lattice vectors. Format: (angle between 1 and 2,  angle between 1 and 3,  angle between 2 and 3). Unit = Degree \n");
			fprintf(output_md,":Desc_VOLUME: Volume of the full lattice. Unit = Bohr^3 \n");
			if (pSPARC->Flag_latvec_scale == 0){
				fprintf(output_md,":Desc_CELL: lengths of three lattice vectors. Unit = Bohr \n");
				fprintf(output_md,":Desc_LatUVec: Lattice vectors of unit length, purpose: to represent change in angles during %s ensemble. Unit = 1 \n",pSPARC->MDMeth);
			} else {
				fprintf(output_md,":Desc_LATVEC_SCALE: ratio of length of cell lattice vectors over length of input lattice vectors. Unit = 1 \n");
				fprintf(output_md,":Desc_LatVec: Lattice vectors, purpose: only to represent change in angles during %s ensemble (has same magnitude as initial LatVec). Unit = Bohr \n",pSPARC->MDMeth);
			}
			if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
				fprintf(output_md,":Desc_NPT_NP_HAMIL: Hamiltonian of the NPT_NP system, formula (10) in (E. Hernandez, 2001). Unit = Ha \n");
				fprintf(output_md,":Desc_SNOSE[0]: Position variable of the thermostat\n");
				fprintf(output_md,":Desc_SNOSE[1]: Velocity variable of the thermostat\n");
			} else{
				fprintf(output_md,":Desc_NPH_HAMIL: Hamiltonian of the NPH system, formula (10) in (E. Hernandez, 2001) with M_s (thermostat Mass) = 0 and S = 1, so no thermostat contribution. Unit = Ha \n");
				fprintf(output_md,":Desc_NPH_Enthalpy: Enthalpy of the NPH system (or generalized enthalpy in case of anisotropic stress). This quantity is same as NPH Hamiltonian (NPH_HAMIL) plus the initial NPH hamiltonian. Unit = Ha \n");	
			}
		}
    	if(pSPARC->Calc_stress == 1){
	    	fprintf(output_md,":Desc_STRESS: Stress, excluding ion-kinetic contribution. Unit=GPa(all periodic),Ha/Bohr**2(surface),Ha/Bohr(wire) \n"); //Calculated different in NPT_NP and NPH ensemble (basically not explicitly subtracting center of mass velocity, but assuming it to be 0, as per the (E.Hernandez, 2001) paper formula)
	    	fprintf(output_md,":Desc_STRIO: Ion-kinetic stress in cartesian coordinate. Unit=GPa(all periodic),Ha/Bohr**2(surface),Ha/Bohr(wire) \n"); //Calculated different in NPT_NP and NPH ensemble (basically not explicitly subtracting center of mass velocity, but assuming it to be 0, as per the (E.Hernandez, 2001) paper formula)
			if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH")==0){
				fprintf(output_md,":Desc_CONSTRESS: Constraint stress (due to restricting cell's full flexibility) in cartesian coordinate. Unit=GPa(all periodic),Ha/Bohr**2(surface),Ha/Bohr(wire) \n");
				fprintf(output_md,":Desc_TOTSTRESS: Total internal stress in cartesian coordinate (also accounting stresses due to any constraint on cell flexibility). Unit=GPa(all periodic),Ha/Bohr**2(surface),Ha/Bohr(wire) \n"); //Calculated as: kinetic_stress (positive convention) - electronic stress - constraint stress;  as per (E. Hernandez, 2001) paper
			}
	    }
	    if((pSPARC->Calc_pres == 1 || pSPARC->Calc_stress == 1) && pSPARC->BC == 2){
	    	fprintf(output_md,":Desc_PRESIO: Ion-kinetic pressure in cartesian coordinate. Unit=GPa \n");
	    	fprintf(output_md,":Desc_PRES: Pressure, excluding ion-kinetic contribution. Unit=GPa \n");
			if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH")==0){
				fprintf(output_md,":Desc_CONPRES: Constraint pressure (pressure due to restricting cell flexibility). Unit=GPa \n"); 
				fprintf(output_md,":Desc_TOTPRES: Total internal pressure (also accounting pressure due to any constraint on cell flexibility). Unit=GPa \n"); 
			}
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
		fprintf(output_md,":TWIST: %.15g\n", pSPARC->twist);
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
		if(strcmpi(pSPARC->MDMeth,"NPT_NH") == 0)
			fprintf(output_md,":NPT_NH_HAMIL:%18.10E \n", pSPARC->Hamiltonian_NPT_NH);
		if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
			fprintf(output_md,":NPT_NP_HAMIL:%18.10E \n", pSPARC->Hamiltonian_NPT_NP);
			fprintf(output_md,":SNOSE[0]:%18.10E \n", pSPARC->SNOSE[0]);
			fprintf(output_md,":SNOSE[1]:%18.10E \n", pSPARC->SNOSE[1]);
		} if(strcmpi(pSPARC->MDMeth,"NPH") == 0){
			fprintf(output_md,":NPH_HAMIL:%18.10E \n", pSPARC->Hamiltonian_NPH); 
			fprintf(output_md,":NPH_ENTHALPY:%18.10E \n", pSPARC->Hamiltonian_NPH +  pSPARC->init_Hamil_NPH); 
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
		if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH") == 0){
			fprintf(output_md,":ANGLES:\n");
			fprintf(output_md," %.3f %.3f %.3f\n", pSPARC->angle_12, pSPARC->angle_13, pSPARC->angle_23);
			fprintf(output_md,":VOLUME: %18.10E\n", pSPARC->volumeCell);
			if (pSPARC->Flag_latvec_scale == 0){
				fprintf(output_md,":CELL:\n");
				fprintf(output_md,"%18.10E %18.10E %18.10E\n", pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
				fprintf(output_md,":LatUVec: \n");
				fprintf(output_md," %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->LatUVec[0],pSPARC->LatUVec[1],pSPARC->LatUVec[2] 
																												  , pSPARC->LatUVec[3],pSPARC->LatUVec[4],pSPARC->LatUVec[5]
																												  , pSPARC->LatUVec[6],pSPARC->LatUVec[7],pSPARC->LatUVec[8]);
			} else {
				fprintf(output_md,":LATVEC_SCALE:\n");
				fprintf(output_md," %18.10E %18.10E %18.10E\n", pSPARC->range_x / pSPARC->initialLatVecLength[0], pSPARC->range_y / pSPARC->initialLatVecLength[1], pSPARC->range_z / pSPARC->initialLatVecLength[2]);
				fprintf(output_md,":LatVec:\n");
				fprintf(output_md," %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->LatVec[0],pSPARC->LatVec[1],pSPARC->LatVec[2] 
																												  , pSPARC->LatVec[3],pSPARC->LatVec[4],pSPARC->LatVec[5]
																												  , pSPARC->LatVec[6],pSPARC->LatVec[7],pSPARC->LatVec[8]);
			}
		}

	    // Print stress
	    if(pSPARC->Calc_stress == 1){
			if(strcmpi(pSPARC->MDMeth,"NPT_NP") != 0 && strcmpi(pSPARC->MDMeth,"NPH") != 0){
				fprintf(output_md,":STRIO:\n");
				PrintStress (pSPARC, pSPARC->stress_i, output_md);
				fprintf(output_md,":STRESS:\n");
				double stress_e[6]; // electronic stress
				for (int i = 0; i < 6; i++) 
					stress_e[i] = pSPARC->stress[i] - pSPARC->stress_i[i];
				PrintStress (pSPARC, stress_e, output_md);
			}

			else{ //Trigger NPT_NP or NPH ensemble stress printing
				fprintf(output_md,":STRIO:\n"); //Ion-kinetic stress 
				double temp_stress[6];
				temp_stress[0] = pSPARC->kinetic_stress[0]; temp_stress[1] = pSPARC->kinetic_stress[1]; temp_stress[2] = pSPARC->kinetic_stress[2];
				temp_stress[3] = pSPARC->kinetic_stress[4]; temp_stress[4] = pSPARC->kinetic_stress[5]; temp_stress[5] = pSPARC->kinetic_stress[8];
				PrintStress (pSPARC, temp_stress, output_md);  //Print kinetic stress as defined in the (E. Hernandez, 2001) paper
				fprintf(output_md,":STRESS:\n");
				double stress_e[6]; // electronic stress
				for (int i = 0; i < 6; i++) 
					stress_e[i] = pSPARC->stress[i];  // stress_i or ionic stress function is never called in NPT_NP or NPH ensemble, so pSPARC->stress does not include contribution from it, and thus no need to subtract stress_i from pSPARC->stress, as done in other ensembles
				PrintStress (pSPARC, stress_e, output_md);
				fprintf(output_md,":CONSTRESS:\n");  //Constraint stress
				temp_stress[0] = pSPARC->constraint_stress[0]; temp_stress[1] = pSPARC->constraint_stress[1]; temp_stress[2] = pSPARC->constraint_stress[2];
				temp_stress[3] = pSPARC->constraint_stress[4]; temp_stress[4] = pSPARC->constraint_stress[5]; temp_stress[5] = pSPARC->constraint_stress[8];
				PrintStress (pSPARC, temp_stress, output_md);
				fprintf(output_md,":TOTSTRESS:\n");	//Print total internal stress accouting for the constraints in NPT_NP or NPH ensemble
				temp_stress[0] = pSPARC->total_internal_stress[0]; temp_stress[1] = pSPARC->total_internal_stress[1]; temp_stress[2] = pSPARC->total_internal_stress[2];
				temp_stress[3] = pSPARC->total_internal_stress[4]; temp_stress[4] = pSPARC->total_internal_stress[5]; temp_stress[5] = pSPARC->total_internal_stress[8];
				PrintStress (pSPARC, temp_stress, output_md);
			}
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

			if(strcmpi(pSPARC->MDMeth,"NPT_NP") != 0 && strcmpi(pSPARC->MDMeth,"NPH") != 0){
				fprintf(output_md,":PRESIO: %18.10E\n", pSPARC->pres_i * CONST_HA_BOHR3_GPA);
				fprintf(output_md,":PRES:   %18.10E\n", (pSPARC->pres-pSPARC->pres_i) * CONST_HA_BOHR3_GPA);
			}
			else{ //Trigger NPT_NP or NPH ensemble stress printing
				double ion_kinetic_pressure, constraint_pressure;
				ion_kinetic_pressure = 1.0 / 3.0 * (pSPARC->kinetic_stress[0] + pSPARC->kinetic_stress[4] + pSPARC->kinetic_stress[8]);  //Kinetic stress is already multiplied by 2
				constraint_pressure = 1.0 / 3.0 * (pSPARC->constraint_stress[0] + pSPARC->constraint_stress[4] + pSPARC->constraint_stress[8]);  //Constraint stress is already multiplied by 2
				fprintf(output_md,":PRESIO:  %18.10E\n", ion_kinetic_pressure * CONST_HA_BOHR3_GPA);
				fprintf(output_md,":PRES:    %18.10E\n", (pSPARC->internal_pressure - ion_kinetic_pressure + constraint_pressure) * CONST_HA_BOHR3_GPA); 
				fprintf(output_md,":CONPRES: %18.10E\n", constraint_pressure * CONST_HA_BOHR3_GPA);
				fprintf(output_md,":TOTPRES: %18.10E\n", pSPARC->internal_pressure * CONST_HA_BOHR3_GPA); //Also accounts for pressure due to constraint stress
			}

	        fprintf(output_md,":PRESIG:  %18.10E\n", pres_ig * CONST_HA_BOHR3_GPA); // Ideal Gas

		}

	#ifdef DEBUG
	    // Print Statistical properties
	    fprintf(output_md,":TELST: %18.10E %18.10E\n", pSPARC->mean_elec_T, pSPARC->std_elec_T);
	    fprintf(output_md,":TIOST: %18.10E %18.10E\n", pSPARC->mean_ion_T, pSPARC->std_ion_T);
		fprintf(output_md,":PREST: %18.10E %18.10E\n", pSPARC->mean_internal_pressure * CONST_HA_BOHR3_GPA, pSPARC->std_internal_pressure * CONST_HA_BOHR3_GPA);
		fprintf(output_md,":MEAN_TOTSTRESS:\n");
		fprintf(output_md," %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->mean_total_internal_stress[0] * CONST_HA_BOHR3_GPA, pSPARC->mean_total_internal_stress[1] * CONST_HA_BOHR3_GPA, pSPARC->mean_total_internal_stress[2] * CONST_HA_BOHR3_GPA 
																										  , pSPARC->mean_total_internal_stress[3] * CONST_HA_BOHR3_GPA, pSPARC->mean_total_internal_stress[4] * CONST_HA_BOHR3_GPA, pSPARC->mean_total_internal_stress[5] * CONST_HA_BOHR3_GPA
																										  , pSPARC->mean_total_internal_stress[6] * CONST_HA_BOHR3_GPA, pSPARC->mean_total_internal_stress[7] * CONST_HA_BOHR3_GPA, pSPARC->mean_total_internal_stress[8] * CONST_HA_BOHR3_GPA);
		fprintf(output_md,":STD_TOTSTRESS:\n");
		fprintf(output_md," %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->std_total_internal_stress[0] * CONST_HA_BOHR3_GPA, pSPARC->std_total_internal_stress[1] * CONST_HA_BOHR3_GPA, pSPARC->std_total_internal_stress[2] * CONST_HA_BOHR3_GPA
																										  , pSPARC->std_total_internal_stress[3] * CONST_HA_BOHR3_GPA, pSPARC->std_total_internal_stress[4] * CONST_HA_BOHR3_GPA, pSPARC->std_total_internal_stress[5] * CONST_HA_BOHR3_GPA
																										  , pSPARC->std_total_internal_stress[6] * CONST_HA_BOHR3_GPA, pSPARC->std_total_internal_stress[7] * CONST_HA_BOHR3_GPA, pSPARC->std_total_internal_stress[8] * CONST_HA_BOHR3_GPA);
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
		char elemType1[8], elemType2[8]; 
		int typeIndex, typeIndex2, pairIndex;
		for (typeIndex = 0; typeIndex < pSPARC->Ntypes; typeIndex++) {
			find_element(elemType1, &pSPARC->atomType[L_ATMTYPE*typeIndex]);
			fprintf(output_md,"%s - %s: %18.10E\n", elemType1, elemType1, mindis[typeIndex]);
		}
		pairIndex = 0;
		for(typeIndex = 0; typeIndex < pSPARC->Ntypes - 1; typeIndex++) {
			find_element(elemType1, &pSPARC->atomType[L_ATMTYPE*typeIndex]);
			for (typeIndex2 = typeIndex + 1; typeIndex2 < pSPARC->Ntypes; typeIndex2++) {
				find_element(elemType2, &pSPARC->atomType[L_ATMTYPE*typeIndex2]);
				fprintf(output_md,"%s - %s: %18.10E\n", elemType1, elemType2, mindis[pairIndex + pSPARC->Ntypes]);
				pairIndex++;
			}
		}
	}
}

/*
 @ brief function to evaluate the quantities of interest in a MD simulation
*/
void MD_QOI(SPARC_OBJ *pSPARC, double *avgvel, double *maxvel, double *mindis) {
	// Compute MD energies (TE=KE+PE)/atom and temperature
	pSPARC->ion_T = 2 * pSPARC->KE /(pSPARC->kB * pSPARC->dof);
	if(pSPARC->ion_elec_eqT == 1){
		pSPARC->elec_T = pSPARC->ion_T;
		pSPARC->Beta = 1.0/(pSPARC->elec_T * pSPARC->kB);
	}
	pSPARC->PE = pSPARC->Etot / pSPARC->n_atom;
	pSPARC->KE = pSPARC->KE / pSPARC->n_atom;
	pSPARC->TE = (pSPARC->PE + pSPARC->KE);
	// Extended System (Ionic system + Thermostat) energy
	if(strcmpi(pSPARC->MDMeth,"NVT_NH") == 0){
		pSPARC->TE_ext = (0.5 * pSPARC->qmass * pow(pSPARC->xi_nose, 2.0) + pSPARC->dof * pSPARC->kB * pSPARC->thermos_T * pSPARC->snose)/pSPARC->n_atom + pSPARC->TE;
	}

    // Compute Ionic stress/pressure
	if(strcmpi(pSPARC->MDMeth,"NPT_NP") != 0 && strcmpi(pSPARC->MDMeth,"NPH") != 0){
		if(pSPARC->Calc_stress == 1 || pSPARC->Calc_pres == 1)
			Calculate_ionic_stress(pSPARC);
	}
	
	//	Calculate_stress(pSPARC);
#ifdef DEBUG
	// MD Statistics
	double mean_TE_old, mean_KE_old, mean_PE_old, mean_U_old, mean_Eent_old, mean_Ti_old, mean_Te_old, mean_internal_pressure_old, mean_total_internal_stress_old[9];
	int Count;
	if(strcmpi(pSPARC->MDMeth,"NPT_NP") != 0 && strcmpi(pSPARC->MDMeth,"NPH") != 0){
		Count = pSPARC->MDCount + (pSPARC->RestartFlag == 0) ;
	}
	else{
		Count = pSPARC->MDCount;
	}
	mean_Te_old = pSPARC->mean_elec_T;
	mean_Ti_old = pSPARC->mean_ion_T;
	mean_TE_old = pSPARC->mean_TE;
	mean_KE_old = pSPARC->mean_KE;
	mean_PE_old = pSPARC->mean_PE;
	mean_U_old = pSPARC->mean_U;
	mean_Eent_old = pSPARC->mean_Entropy;
	mean_internal_pressure_old = pSPARC->mean_internal_pressure;
	for (int i = 0; i < 9; i++){mean_total_internal_stress_old[i] = pSPARC->mean_total_internal_stress[i];}
	pSPARC->mean_elec_T = (mean_Te_old * (Count - 1) + pSPARC->elec_T)/ Count;
	pSPARC->mean_ion_T = (mean_Ti_old * (Count - 1) + pSPARC->ion_T)/ Count;
	pSPARC->mean_internal_pressure = (mean_internal_pressure_old * (Count - 1) + pSPARC->internal_pressure) / Count;
	for (int i = 0; i < 9; i++){pSPARC->mean_total_internal_stress[i] = (mean_total_internal_stress_old[i] * (Count - 1) + pSPARC->total_internal_stress[i]) / Count;}
	pSPARC->mean_TE = (mean_TE_old * (Count - 1) + pSPARC->TE)/ Count;
	pSPARC->mean_KE = (mean_KE_old * (Count - 1) + pSPARC->KE)/ Count;
	pSPARC->mean_PE = (mean_PE_old * (Count - 1) + pSPARC->PE)/ Count;
	pSPARC->mean_U = (mean_U_old * (Count - 1) + pSPARC->PE - pSPARC->Entropy/pSPARC->n_atom)/ Count;
	pSPARC->mean_Entropy = (mean_Eent_old * (Count - 1) + pSPARC->Entropy/pSPARC->n_atom)/ Count;
	pSPARC->std_elec_T = sqrt(fabs( ((pow(pSPARC->std_elec_T,2.0) + pow(mean_Te_old,2.0)) * (Count - 1) + pow(pSPARC->elec_T,2.0))/Count - pow(pSPARC->mean_elec_T,2.0) ));
	pSPARC->std_ion_T = sqrt(fabs( ((pow(pSPARC->std_ion_T,2.0) + pow(mean_Ti_old,2.0)) * (Count - 1) + pow(pSPARC->ion_T,2.0))/Count - pow(pSPARC->mean_ion_T,2.0) ));
	pSPARC->std_internal_pressure = sqrt(fabs( ((pow(pSPARC->std_internal_pressure,2.0) + pow(mean_internal_pressure_old,2.0)) * (Count - 1) + pow(pSPARC->internal_pressure,2.0))/Count - pow(pSPARC->mean_internal_pressure,2.0) ));
	for (int i = 0; i < 9; i++){pSPARC->std_total_internal_stress[i] = sqrt(fabs( ((pow(pSPARC->std_total_internal_stress[i],2.0) + pow(mean_total_internal_stress_old[i],2.0)) * (Count - 1) + pow(pSPARC->total_internal_stress[i],2.0))/Count - pow(pSPARC->mean_total_internal_stress[i],2.0) ));}
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
	int atm2, ityp2, cc2, pairIndex;
	cc = 0;

	// Store the cell as a 3x3 lattice vector
	double *lattice = (double *)calloc(9, sizeof(double));
	double cell[3] = {pSPARC->range_x, pSPARC->range_y, pSPARC->range_z};
	double dr[3];
    int row;
	for (row = 0; row < 3; row++) {
        lattice[row*3] = pSPARC->LatUVec[row*3] * cell[row];
        lattice[row*3 + 1] = pSPARC->LatUVec[row*3 + 1] * cell[row];
        lattice[row*3 + 2] = pSPARC->LatUVec[row*3 + 2] * cell[row];
    }

	// Calculate the inverse of lattice-vector
	double LV_inv[9], U[9], S[3], VT[9], superb[2], temp_mat[9], LatVec[9];
    double S_inv[9] = {0};
	int m = 3;

	for (int JJ = 0; JJ < 9; JJ++) {
        LatVec[JJ] = lattice[JJ];
    }

	/* Calculating pinv(LatVec): This is necessary because for extremely skewed LV, the LV maybe close to singular */
	// Compute SVD of LatVec(LV) first: LV = U * S * V^T
	LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, m, LatVec, m, S, U, m, VT, m, superb);

	// First compute S^{-1}
	for (int i = 0; i < 3; i++) {
		if (S[i] > 1e-12) S_inv[i * 3 + i] = 1.0/S[i];
	}

	// Compute LV^{-1} = V * S^{-1} * U^T
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, m, m, 1.0, VT, m, S_inv, m, 0.0, temp_mat, m);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, temp_mat, m, U, m, 0.0, LV_inv, m);

	for(ityp = 0; ityp < pSPARC->Ntypes; ityp++){
	    mindis[ityp] = 1000000000.0;
		for(atm = 0; atm < pSPARC->nAtomv[ityp] - 1; atm++){
			for(atm2 = atm + 1; atm2 < pSPARC->nAtomv[ityp]; atm2++){
				// temp = fabs(sqrt(pow(pSPARC->atom_pos[3 * (atm + cc)] - pSPARC->atom_pos[3 * (atm2 + cc)],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 1] - pSPARC->atom_pos[3 * (atm2 + cc) + 1],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 2] - pSPARC->atom_pos[3 * (atm2 + cc) + 2],2.0) ));
				
				// Vector connecting atoms atm and atm2
				dr[0] = pSPARC->atom_pos[3 * (atm + cc)] - pSPARC->atom_pos[3 * (atm2 + cc)];
				dr[1] = pSPARC->atom_pos[3 * (atm + cc) + 1] - pSPARC->atom_pos[3 * (atm2 + cc) + 1];
				dr[2] = pSPARC->atom_pos[3 * (atm + cc) + 2] - pSPARC->atom_pos[3 * (atm2 + cc) + 2];

				double frac[3], dr_pbc[3];

				// Minimum Image Convention (MIC) in fractional coordinates
				// frac = LV^{-1} * dr
				cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, LV_inv, m, dr, 1, 0.0, frac, 1);

				// Wrap into [-0.5, 0.5)
				frac[0] -= round(frac[0]);
				frac[1] -= round(frac[1]); 
				frac[2] -= round(frac[2]);

				// dr_pbc = lattice * frac
				cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, lattice, 3, frac, 1, 0.0, dr_pbc, 1);

				// Calculate distance
				temp = sqrt(dr_pbc[0]*dr_pbc[0] + dr_pbc[1]*dr_pbc[1] + dr_pbc[2]*dr_pbc[2]);

				if(temp < mindis[ityp])
					mindis[ityp] = temp;
			}
		}
		cc += pSPARC->nAtomv[ityp];
	}
	cc = 0;
	pairIndex = pSPARC->Ntypes;
	for(ityp = 0; ityp < pSPARC->Ntypes - 1; ityp++){
		cc2 = pSPARC->nAtomv[ityp] + cc;
		for(ityp2 = ityp + 1; ityp2 < pSPARC->Ntypes; ityp2++){
			// mindis[pSPARC->Ntypes - 1 + ityp*(pSPARC->Ntypes-1 + pSPARC->Ntypes-1-(ityp - 1))/2 + ityp2 - ityp] = 1000000000.0;
			mindis[pairIndex] = 1000000000.0;
			for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++){
				for(atm2 = 0; atm2 < pSPARC->nAtomv[ityp2]; atm2++){
					// temp = fabs(sqrt(pow(pSPARC->atom_pos[3 * (atm + cc)] - pSPARC->atom_pos[3 * (atm2 + cc2)],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 1] - pSPARC->atom_pos[3 * (atm2 + cc2) + 1],2.0) + pow(pSPARC->atom_pos[3 * (atm + cc) + 2] - pSPARC->atom_pos[3 * (atm2 + cc2) + 2],2.0) ));
					
					// Vector connecting atoms atm and atm2
					dr[0] = pSPARC->atom_pos[3 * (atm + cc)] - pSPARC->atom_pos[3 * (atm2 + cc2)];
					dr[1] = pSPARC->atom_pos[3 * (atm + cc) + 1] - pSPARC->atom_pos[3 * (atm2 + cc2) + 1];
					dr[2] = pSPARC->atom_pos[3 * (atm + cc) + 2] - pSPARC->atom_pos[3 * (atm2 + cc2) + 2];

					double frac[3], dr_pbc[3];

					// Minimum Image Convention (MIC) in fractional coordinates
					// frac = LV^{-1} * dr
					cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, LV_inv, m, dr, 1, 0.0, frac, 1);

					// Wrap into [-0.5, 0.5)
					frac[0] -= round(frac[0]);
					frac[1] -= round(frac[1]);
					frac[2] -= round(frac[2]);

					// dr_pbc = lattice * frac
					cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, lattice, 3, frac, 1, 0.0, dr_pbc, 1);

					// Calculate distance
					temp = sqrt(dr_pbc[0]*dr_pbc[0] + dr_pbc[1]*dr_pbc[1] + dr_pbc[2]*dr_pbc[2]);

					if(temp < mindis[pairIndex])
						mindis[pairIndex] = temp;
				}
			}
			pairIndex++;
			cc2 += pSPARC->nAtomv[ityp2];
		}
		cc += pSPARC->nAtomv[ityp];
	}
	free(lattice);
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
		if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
    		fprintf(mdout,":STOPCOUNT: %d\n", pSPARC->MDCount + pSPARC->restartCount);
		}
		else{
			fprintf(mdout,":STOPCOUNT: %d\n", pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
		}
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
		printf("\n");
		printf("\n");
		if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
			fprintf(mdout,":MDSTEP: %d\n", pSPARC->MDCount + pSPARC->restartCount );
		}
		else{
    		fprintf(mdout,":MDSTEP: %d\n", pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
		}
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
		if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
			fprintf(mdout,":Pm_ion:\n");
			for(atm = 0; atm < pSPARC->n_atom; atm++){
				fprintf(mdout,"%18.10E %18.10E %18.10E\n", pSPARC->Pm_ion[3 * atm], pSPARC->Pm_ion[3 * atm + 1], pSPARC->Pm_ion[3 * atm + 2]);
			}
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
    		fprintf(mdout,":TTHRMI(K): %.15g\n", pSPARC->thermos_Ti);
			fprintf(mdout,":TARGET_PRESSURE: %.15g GPa\n",pSPARC->prtarget * 29421.02648438959);
    	}
    	// Print extended system parameters in case of NPT-NP or NPH ensemble
    	if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
			if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
				fprintf(mdout,":NPT_NP_QMASS: %.15g\n", pSPARC->NPT_NP_qmass);
				fprintf(mdout,":NPT_NP_BMASS: %.15g\n", pSPARC->NPT_NP_bmass);
				fprintf(mdout,":NPT_NP_SNOSE[0]: %.15g\n", pSPARC->SNOSE[0]); // value of virtual thermal parameter at current timestep
				fprintf(mdout,":NPT_NP_SNOSE[1]: %.15g\n", pSPARC->SNOSE[1]); // velocity of virtual thermal parameter
				fprintf(mdout,":NPT_NP_SNOSE[2]: %.15g\n", pSPARC->SNOSE[2]); // value of virtual thermal parameter at previous timestep
				fprintf(mdout,":NPT_NP_INIT_Hamiltonian: %.15g\n", pSPARC->init_Hamil_NPT_NP);
			}
			else if (strcmpi(pSPARC->MDMeth,"NPH") == 0){
				fprintf(mdout,":NPH_BMASS: %.15g\n", pSPARC->NPH_bmass);
				fprintf(mdout,":NPH_INIT_Hamiltonian: %.15g\n", pSPARC->init_Hamil_NPH);
			}
			fprintf(mdout,":KE: %.15g\n",pSPARC->KE);
			fprintf(mdout,":Kbaro: %.15g\n",pSPARC->Kbaro);
			fprintf(mdout,":Ubaro: %.15g\n",pSPARC->Ubaro);
			fprintf(mdout,":kinetic_stress: \n %.15g %.15g %.15g \n %.15g %.15g %.15g \n %.15g %.15g %.15g\n", pSPARC->kinetic_stress[0], pSPARC->kinetic_stress[1], pSPARC->kinetic_stress[2]
																									  		   , pSPARC->kinetic_stress[3], pSPARC->kinetic_stress[4], pSPARC->kinetic_stress[5]
																									  		   , pSPARC->kinetic_stress[6], pSPARC->kinetic_stress[7], pSPARC->kinetic_stress[8]); 
    		fprintf(mdout,":Pm_metric_tensor: \n %.15g %.15g %.15g \n %.15g %.15g %.15g \n %.15g %.15g %.15g\n", pSPARC->Pm_metric_tensor[0], pSPARC->Pm_metric_tensor[1], pSPARC->Pm_metric_tensor[2]
																									  		   , pSPARC->Pm_metric_tensor[3], pSPARC->Pm_metric_tensor[4], pSPARC->Pm_metric_tensor[5]
																									  		   , pSPARC->Pm_metric_tensor[6], pSPARC->Pm_metric_tensor[7], pSPARC->Pm_metric_tensor[8]); // velocity of lattice
    		if (pSPARC->Flag_latvec_scale == 0){
				fprintf(mdout,":CELL: %18.10E %18.10E %18.10E\n", pSPARC->range_x, pSPARC->range_y, pSPARC->range_z);
				fprintf(mdout,":LatUVec: \n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->LatUVec[0], pSPARC->LatUVec[1], pSPARC->LatUVec[2] 
																													      , pSPARC->LatUVec[3], pSPARC->LatUVec[4], pSPARC->LatUVec[5]
																													      , pSPARC->LatUVec[6], pSPARC->LatUVec[7], pSPARC->LatUVec[8]);
			} else {
				fprintf(mdout,":LATVEC_SCALE: %18.10E %18.10E %18.10E\n", pSPARC->range_x/pSPARC->initialLatVecLength[0], pSPARC->range_y/pSPARC->initialLatVecLength[1], pSPARC->range_z/pSPARC->initialLatVecLength[2]);
				fprintf(mdout,":LatVec: \n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->LatVec[0], pSPARC->LatVec[1], pSPARC->LatVec[2] 
																													     , pSPARC->LatVec[3], pSPARC->LatVec[4], pSPARC->LatVec[5]
																													     , pSPARC->LatVec[6], pSPARC->LatVec[7], pSPARC->LatVec[8]);
			}
    		fprintf(mdout,":INITIAL_ANGLES: %18.10E %18.10E %18.10E\n", acos(pSPARC->initialLatVecAngles[0]) * 180 / M_PI, acos(pSPARC->initialLatVecAngles[1]) * 180 / M_PI, acos(pSPARC->initialLatVecAngles[2]) * 180 / M_PI );
			fprintf(mdout,":ROTATION_MATRIX: \n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n %18.10E %18.10E %18.10E\n", pSPARC->rotation_matrix[0], pSPARC->rotation_matrix[1], pSPARC->rotation_matrix[2] 
																													  	      , pSPARC->rotation_matrix[3], pSPARC->rotation_matrix[4], pSPARC->rotation_matrix[5]
																													          , pSPARC->rotation_matrix[6], pSPARC->rotation_matrix[7], pSPARC->rotation_matrix[8]);
			fprintf(mdout,":TTHRMI(K): %.15g\n", pSPARC->thermos_Ti);
			fprintf(mdout,":EXTERNAL_PRESSURE: %.15g\n", pSPARC->pressure_external * CONST_HA_BOHR3_GPA);
			fprintf(mdout,":EXTERNAL_STRESS: %.15g %.15g %.15g %.15g %.15g %.15g GPa\n",pSPARC->stress_external[0] * CONST_HA_BOHR3_GPA
                                                                                       ,pSPARC->stress_external[1] * CONST_HA_BOHR3_GPA
                                                                                       ,pSPARC->stress_external[2] * CONST_HA_BOHR3_GPA
                                                                                       ,pSPARC->stress_external[3] * CONST_HA_BOHR3_GPA
                                                                                       ,pSPARC->stress_external[4] * CONST_HA_BOHR3_GPA
                                                                                       ,pSPARC->stress_external[5] * CONST_HA_BOHR3_GPA);		
    	}
    	// Print temperature
    	fprintf(mdout,":TEL(K): %18.10E\n", pSPARC->elec_T);
    	fprintf(mdout,":TIO(K): %18.10E\n", pSPARC->ion_T);
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


	if(strcmpi(pSPARC->MDMeth,"NPT_NP") == 0 || strcmpi(pSPARC->MDMeth,"NPH") == 0){
		pSPARC->Pm_ion = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
		if (pSPARC->Pm_ion == NULL) {
			printf("\nCannot allocate memory for ion momentum array!\n");
			exit(EXIT_FAILURE);
    	}
	}
	
	// Allocate memory for Pack and Unpack to be used later for broadcasting
	if(pSPARC->RestartFlag == 1){
	    // l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + 5) * sizeof(double);
	    if (strcmpi(pSPARC->MDMeth,"NPT_NH") == 0){
	    	l_buff = (2 + 1) * sizeof(int) + (6 * pSPARC->n_atom + (5 + 3*pSPARC->NPT_NHnnos + 9 + 20)) * sizeof(double);
	    }
	    else if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
	    	l_buff = 2 * sizeof(int) + (9 * pSPARC->n_atom + (5 + 56 + 32)) * sizeof(double);
	    } 
		else if (strcmpi(pSPARC->MDMeth,"NPH") == 0){
	    	l_buff = 2 * sizeof(int) + (9 * pSPARC->n_atom + (5 + 52 + 32)) * sizeof(double);
	    } 
	    else {
	    	l_buff = 2 * sizeof(int) + (6 * pSPARC->n_atom + 5 + 22) * sizeof(double);
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
			else if (strcmpi(str,":R(Bohr):") == 0){
				for(atm = 0; atm < pSPARC->n_atom; atm++){
					fscanf(rst_fp,"%lf %lf %lf", &pSPARC->atom_pos[3 * atm], &pSPARC->atom_pos[3 * atm + 1], &pSPARC->atom_pos[3 * atm + 2]);
				}
			} else if (strcmpi(str,":V(Bohr/atu):") == 0){
					for(atm = 0; atm < pSPARC->n_atom; atm++){
						fscanf(rst_fp,"%lf %lf %lf", &pSPARC->ion_vel[3 * atm], &pSPARC->ion_vel[3 * atm + 1], &pSPARC->ion_vel[3 * atm + 2]);
				}
			} else if (strcmpi(str,":Pm_ion:") == 0){
					for(atm = 0; atm < pSPARC->n_atom; atm++){
						fscanf(rst_fp,"%lf %lf %lf", &pSPARC->Pm_ion[3 * atm], &pSPARC->Pm_ion[3 * atm + 1], &pSPARC->Pm_ion[3 * atm + 2]);
				}
			}
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
				else if (strcmpi(str,":TTHRMI(K):") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->thermos_Ti);
        		else if (strcmpi(str,":TARGET_PRESSURE:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->prtarget);
			}
			if ((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
				if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
					if (strcmpi(str,":NPT_NP_SNOSE[0]:") == 0)
						fscanf(rst_fp,"%lf", &pSPARC->SNOSE[0]);
					else if (strcmpi(str,":NPT_NP_SNOSE[1]:") == 0)
						fscanf(rst_fp,"%lf", &pSPARC->SNOSE[1]);
					else if (strcmpi(str,":NPT_NP_SNOSE[2]:") == 0)
						fscanf(rst_fp,"%lf", &pSPARC->SNOSE[2]);
					else if (strcmpi(str,":NPT_NP_INIT_Hamiltonian:") == 0)
						fscanf(rst_fp,"%lf", &pSPARC->init_Hamil_NPT_NP);
				}
				else if (strcmpi(pSPARC->MDMeth,"NPH") == 0){
					if (strcmpi(str,":NPH_INIT_Hamiltonian:") == 0)
            			fscanf(rst_fp,"%lf", &pSPARC->init_Hamil_NPH);
				}
				if (strcmpi(str,":KE:") == 0)
					fscanf(rst_fp,"%lf", &pSPARC->KE);
				else if (strcmpi(str,":Kbaro:") == 0)
					fscanf(rst_fp,"%lf", &pSPARC->Kbaro);
				else if (strcmpi(str,":Ubaro:") == 0)
					fscanf(rst_fp,"%lf", &pSPARC->Ubaro);
				else if (strcmpi(str,":kinetic_stress:") == 0){
					fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[0]); fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[1]); fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[2]);
					fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[3]); fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[4]); fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[5]);
					fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[6]); fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[7]); fscanf(rst_fp,"%lf", &pSPARC->kinetic_stress[8]);
				} else if (strcmpi(str,":Pm_metric_tensor:") == 0){
					fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[0]); fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[1]); fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[2]);
					fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[3]); fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[4]); fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[5]);
					fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[6]); fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[7]); fscanf(rst_fp,"%lf", &pSPARC->Pm_metric_tensor[8]);
				} else if (strcmpi(str,":INITIAL_ANGLES:") == 0){
					fscanf(rst_fp,"%lf", &pSPARC->initialLatVecAngles[0]); fscanf(rst_fp,"%lf", &pSPARC->initialLatVecAngles[1]); fscanf(rst_fp,"%lf", &pSPARC->initialLatVecAngles[2]);
					for (int i = 0; i < 3; i++){pSPARC->initialLatVecAngles[i] = cos(M_PI / 180 * pSPARC->initialLatVecAngles[i]);}
				} else if (strcmpi(str,":CELL:") == 0) {
        		    fscanf(rst_fp,"%lf", &pSPARC->range_x); fscanf(rst_fp,"%lf", &pSPARC->range_y); fscanf(rst_fp,"%lf", &pSPARC->range_z);
            	} else if (strcmpi(str,":LatUVec:") == 0) {
					fscanf(rst_fp,"%lf", &pSPARC->LatUVec[0]); fscanf(rst_fp,"%lf", &pSPARC->LatUVec[1]); fscanf(rst_fp,"%lf", &pSPARC->LatUVec[2]);
					fscanf(rst_fp,"%lf", &pSPARC->LatUVec[3]); fscanf(rst_fp,"%lf", &pSPARC->LatUVec[4]); fscanf(rst_fp,"%lf", &pSPARC->LatUVec[5]);
					fscanf(rst_fp,"%lf", &pSPARC->LatUVec[6]); fscanf(rst_fp,"%lf", &pSPARC->LatUVec[7]); fscanf(rst_fp,"%lf", &pSPARC->LatUVec[8]);
				} else if (strcmpi(str,":LATVEC_SCALE:") == 0) {
					fscanf(rst_fp,"%lf", &pSPARC->latvec_scale_x); fscanf(rst_fp,"%lf", &pSPARC->latvec_scale_y); fscanf(rst_fp,"%lf", &pSPARC->latvec_scale_z);
					
					fscanf(rst_fp, "%*[^\n]\n");
				} else if (strcmpi(str,":LatVec:") == 0){
					fscanf(rst_fp,"%lf", &pSPARC->LatVec[0]); fscanf(rst_fp,"%lf", &pSPARC->LatVec[1]); fscanf(rst_fp,"%lf", &pSPARC->LatVec[2]);
					fscanf(rst_fp,"%lf", &pSPARC->LatVec[3]); fscanf(rst_fp,"%lf", &pSPARC->LatVec[4]); fscanf(rst_fp,"%lf", &pSPARC->LatVec[5]);
					fscanf(rst_fp,"%lf", &pSPARC->LatVec[6]); fscanf(rst_fp,"%lf", &pSPARC->LatVec[7]); fscanf(rst_fp,"%lf", &pSPARC->LatVec[8]);
				} else if (strcmpi(str,":ROTATION_MATRIX:") == 0){
					fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[0]); fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[1]); fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[2]);
					fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[3]); fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[4]); fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[5]);
					fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[6]); fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[7]); fscanf(rst_fp,"%lf", &pSPARC->rotation_matrix[8]);
				} else if (strcmpi(str,":TTHRMI(K):") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->thermos_Ti);
            	else if (strcmpi(str,":EXTERNAL_PRESSURE:") == 0)
            		fscanf(rst_fp,"%lf", &pSPARC->pressure_external);
				else if (strcmpi(str,":EXTERNAL_STRESS:") == 0){
					fscanf(rst_fp,"%lf", &pSPARC->stress_external[0]); fscanf(rst_fp,"%lf", &pSPARC->stress_external[1]); fscanf(rst_fp,"%lf", &pSPARC->stress_external[2]);
					fscanf(rst_fp,"%lf", &pSPARC->stress_external[3]); fscanf(rst_fp,"%lf", &pSPARC->stress_external[4]); fscanf(rst_fp,"%lf", &pSPARC->stress_external[5]);
				}
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
			if ((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
				MPI_Pack(pSPARC->Pm_ion, 3*pSPARC->n_atom, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
			}
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
				MPI_Pack(&pSPARC->thermos_Ti, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->prtarget, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            }
            else if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
				if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
					MPI_Pack(pSPARC->SNOSE, 3, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(&pSPARC->init_Hamil_NPT_NP, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				}
				else if (strcmpi(pSPARC->MDMeth,"NPH") == 0){
					MPI_Pack(&pSPARC->init_Hamil_NPH, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				}
				MPI_Pack(&pSPARC->KE, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(&pSPARC->Kbaro, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(&pSPARC->Ubaro, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(pSPARC->kinetic_stress, 9, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(pSPARC->Pm_metric_tensor, 9, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(pSPARC->initialLatVecAngles, 3, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				if (pSPARC->Flag_latvec_scale == 0){
					MPI_Pack(&pSPARC->range_x, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(&pSPARC->range_y, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(&pSPARC->range_z, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(pSPARC->LatUVec, 9, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				}
				else if (pSPARC->Flag_latvec_scale == 1){
					MPI_Pack(&pSPARC->latvec_scale_x, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(&pSPARC->latvec_scale_y, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(&pSPARC->latvec_scale_z, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
					MPI_Pack(pSPARC->LatVec, 9, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				}
				MPI_Pack(pSPARC->rotation_matrix, 9, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(&pSPARC->thermos_Ti, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            	MPI_Pack(&pSPARC->pressure_external, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
				MPI_Pack(pSPARC->stress_external, 6, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
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
        MPI_Unpack(buff, l_buff, &position, pSPARC->atom_pos, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->ion_vel, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
		if(pSPARC->RestartFlag == 1){
			if ((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
				MPI_Unpack(buff, l_buff, &position, pSPARC->Pm_ion, 3 * pSPARC->n_atom, MPI_DOUBLE, MPI_COMM_WORLD);
			}
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
				MPI_Unpack(buff, l_buff, &position, &pSPARC->thermos_Ti, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        		MPI_Unpack(buff, l_buff, &position, &pSPARC->prtarget, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            else if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0) || (strcmpi(pSPARC->MDMeth,"NPH") == 0)){
				if (strcmpi(pSPARC->MDMeth,"NPT_NP") == 0){
					MPI_Unpack(buff, l_buff, &position, pSPARC->SNOSE, 3, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, &pSPARC->init_Hamil_NPT_NP, 1, MPI_DOUBLE, MPI_COMM_WORLD);
				}
				else if (strcmpi(pSPARC->MDMeth,"NPH") == 0){
					MPI_Unpack(buff, l_buff, &position, &pSPARC->init_Hamil_NPH, 1, MPI_DOUBLE, MPI_COMM_WORLD);
				}
				MPI_Unpack(buff, l_buff, &position, &pSPARC->KE, 1, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, &pSPARC->Kbaro, 1, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, &pSPARC->Ubaro, 1, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, pSPARC->kinetic_stress, 9, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, pSPARC->Pm_metric_tensor, 9, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, pSPARC->initialLatVecAngles, 3, MPI_DOUBLE, MPI_COMM_WORLD);
				if (pSPARC->Flag_latvec_scale == 0){
					MPI_Unpack(buff, l_buff, &position, &pSPARC->range_x, 1, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, &pSPARC->range_y, 1, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, &pSPARC->range_z, 1, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, pSPARC->LatUVec, 9, MPI_DOUBLE, MPI_COMM_WORLD);
				}
				else if (pSPARC->Flag_latvec_scale == 1){
					MPI_Unpack(buff, l_buff, &position, &pSPARC->latvec_scale_x, 1, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, &pSPARC->latvec_scale_y, 1, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, &pSPARC->latvec_scale_z, 1, MPI_DOUBLE, MPI_COMM_WORLD);
					MPI_Unpack(buff, l_buff, &position, pSPARC->LatVec, 9, MPI_DOUBLE, MPI_COMM_WORLD);
				}
				MPI_Unpack(buff, l_buff, &position, pSPARC->rotation_matrix, 9, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, &pSPARC->thermos_Ti, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            	MPI_Unpack(buff, l_buff, &position, &pSPARC->pressure_external, 1, MPI_DOUBLE, MPI_COMM_WORLD);
				MPI_Unpack(buff, l_buff, &position, pSPARC->stress_external, 6, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

	}
	if(pSPARC->RestartFlag == 1) {
	    pSPARC->Beta = 1.0/(pSPARC->elec_T * pSPARC->kB);
	    if((strcmpi(pSPARC->MDMeth,"NPT_NH") == 0)) { 
			    reinitialize_mesh_NPT(pSPARC);
        }
		// For NPT_NP and NPH the 'reinitialize_mesh_NPT' function is being called from 'fetch_MD_cell_ingredients_restart' after calculating new Jacbdet, cell volume etc
		if((strcmpi(pSPARC->MDMeth,"NPT_NP") == 0)||(strcmpi(pSPARC->MDMeth,"NPH") == 0) ){
			fetch_MD_cell_ingredients_restart(pSPARC);
			//Now reinitialize mesh based on calculated Jacbdet and other ingredients
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
