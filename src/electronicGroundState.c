/**
 * @file    electronicGroundState.c
 * @brief   This file contains the functions for electronic Ground-state calculation.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "electronicGroundState.h"
#include "initialization.h"
#include "electrostatics.h"
#include "orbitalElecDensInit.h"
#include "nlocVecRoutines.h"
#include "exchangeCorrelation.h"
#include "electronDensity.h"
#include "eigenSolver.h"
#include "eigenSolverKpt.h" 
#include "mixing.h"
#include "occupation.h"
#include "energy.h"
#include "tools.h"
#include "parallelization.h"
#include "isddft.h"
#include "finalization.h"
#include "forces.h"
#include "stress.h"
#include "pressure.h"
#include "exactExchange.h"
#include "d3correction.h"
#include "d3forceStress.h"
#include "mGGAtauTransferTauVxc.h"
#include "spinOrbitCoupling.h"
#include "sq.h"
#include "sqFinalization.h"
#include "sqEnergy.h"
#include "sqDensity.h"
#include "sqProperties.h"
#include "sqParallelization.h"
#include "sqNlocVecRoutines.h"
#include "printing.h"

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
#endif

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))
#define TEMP_TOL 1e-12

/**
 * @brief   Calculate the ground state energy and forces for fixed atom positions.  
 */
void Calculate_electronicGroundState(SPARC_OBJ *pSPARC) {
    int rank, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    FILE *output_fp, *static_fp;
    double t1, t2;
    
#ifdef DEBUG
    if (rank == 0) printf("Start ground-state calculation.\n");
#endif
    
    // Check if Reference Cutoff > 0.5 * nearest neighbor distance
    // Check if Reference Cutoff < mesh spacing
    if (rank == 0) {
        double nn = compute_nearest_neighbor_dist(pSPARC, 'N');
        int SCF_ind;
        if (pSPARC->MDFlag == 1)
            SCF_ind = pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0);
        else if(pSPARC->RelaxFlag >= 1)
            SCF_ind = pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0);
        else
            SCF_ind = 1;
        if (pSPARC->REFERENCE_CUTOFF > 0.5*nn) {
            printf("\nWARNING: REFERENCE _CUFOFF (%.6f Bohr) > 1/2 nn (nearest neighbor) distance (%.6f Bohr) in SCF#%d\n",
                        pSPARC->REFERENCE_CUTOFF, 0.5*nn,  SCF_ind);
        }
        if (pSPARC->REFERENCE_CUTOFF < pSPARC->delta_x ||
            pSPARC->REFERENCE_CUTOFF < pSPARC->delta_y ||
            pSPARC->REFERENCE_CUTOFF < pSPARC->delta_z ) {
            printf("\nWARNING: REFERENCE _CUFOFF (%.6f Bohr) < MESH_SPACING (dx %.6f Bohr, dy %.6f Bohr, dz %.6f Bohr) in SCF#%d\n",
                        pSPARC->REFERENCE_CUTOFF, pSPARC->delta_x, pSPARC->delta_y, pSPARC->delta_z, SCF_ind );
        }
    }
    
	// initialize the history variables of Anderson mixing
	if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
	    memset(pSPARC->mixing_hist_Xk, 0, sizeof(double)* pSPARC->Nd_d * pSPARC->Nspden * pSPARC->MixingHistory);
	    memset(pSPARC->mixing_hist_Fk, 0, sizeof(double)* pSPARC->Nd_d * pSPARC->Nspden * pSPARC->MixingHistory);
    }
    
    Calculate_EGS_elecDensEnergy(pSPARC);

    // write energies into output file   
    if (!rank && pSPARC->Verbosity) {
        output_fp = fopen(pSPARC->OutFilename,"a");
	    if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"====================================================================\n");
        fprintf(output_fp,"                    Energy and force calculation                    \n");
        fprintf(output_fp,"====================================================================\n");
        fprintf(output_fp,"Free energy per atom               :%18.10E (Ha/atom)\n", pSPARC->Etot / pSPARC->n_atom);
        fprintf(output_fp,"Total free energy                  :%18.10E (Ha)\n", pSPARC->Etot);
        fprintf(output_fp,"Band structure energy              :%18.10E (Ha)\n", pSPARC->Eband);
        fprintf(output_fp,"Exchange correlation energy        :%18.10E (Ha)\n", pSPARC->Exc);
        fprintf(output_fp,"Self and correction energy         :%18.10E (Ha)\n", pSPARC->Esc);
        fprintf(output_fp,"-Entropy*kb*T                      :%18.10E (Ha)\n", pSPARC->Entropy);
        fprintf(output_fp,"Fermi level                        :%18.10E (Ha)\n", pSPARC->Efermi);
        if (pSPARC->d3Flag == 1) {
        	fprintf(output_fp,"DFT-D3 correction                  :%18.10E (Ha)\n", pSPARC->d3Energy[0]);
        }
        if (pSPARC->ixc[3] != 0) {
            fprintf(output_fp,"vdWDF energy                       :%18.10E (Ha)\n", pSPARC->vdWDFenergy);
        }
        fclose(output_fp);
        // for static calculation, print energy to .static file
        if (pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
            if (pSPARC->PrintForceFlag == 1 || pSPARC->PrintAtomPosFlag == 1) {
                static_fp = fopen(pSPARC->StaticFilename,"a");
                if (static_fp == NULL) {
                    printf("\nCannot open file \"%s\"\n",pSPARC->StaticFilename);
                    exit(EXIT_FAILURE);
                }
                fprintf(static_fp,"Total free energy (Ha): %.15E\n", pSPARC->Etot);
                fclose(static_fp);
            }
        }
    }
    
    t1 = MPI_Wtime();
    // calculate forces
    Calculate_EGS_Forces(pSPARC);
    t2 = MPI_Wtime();
    
    // calculate atom magnetization
    if (pSPARC->spin_typ) CalculateAtomMag(pSPARC);

    // write forces into .static file if required
    if (rank == 0 && pSPARC->Verbosity && pSPARC->PrintForceFlag == 1 && pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
        static_fp = fopen(pSPARC->StaticFilename,"a");
        if (static_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->StaticFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(static_fp, "Atomic forces (Ha/Bohr):\n");
        for (i = 0; i < pSPARC->n_atom; i++) {
            fprintf(static_fp,"%18.10E %18.10E %18.10E\n",
                    pSPARC->forces[3*i],pSPARC->forces[3*i+1],pSPARC->forces[3*i+2]);
        }
        if (pSPARC->spin_typ == 1) {        
            fprintf(static_fp, "Atomic magnetization along Z-dir within Radius 2 Bohr: (Bohr magneton)\n");
            for (i = 0; i < pSPARC->n_atom; i++) {
                fprintf(static_fp,"%18.10E\n", pSPARC->AtomMag[i]);
            }
        }
        if (pSPARC->spin_typ == 2) {        
            fprintf(static_fp, "Atomic magnetization along X,Y,Z-dir within Radius 2 Bohr: (Bohr magneton)\n");
            for (i = 0; i < pSPARC->n_atom; i++) {
                fprintf(static_fp,"%18.10E %18.10E %18.10E\n",
                        pSPARC->AtomMag[3*i],pSPARC->AtomMag[3*i+1],pSPARC->AtomMag[3*i+2]);
            }
        }
        fclose(static_fp);
    }
    
    if(!rank && pSPARC->Verbosity) {
    	output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        double avgF = 0.0, maxF = 0.0, temp;
        for (i = 0; i< pSPARC->n_atom; i++){
        	temp = fabs(sqrt(pow(pSPARC->forces[3*i  ],2.0) 
        	               + pow(pSPARC->forces[3*i+1],2.0) 
        	               + pow(pSPARC->forces[3*i+2],2.0)));
        	avgF += temp;
        	if (temp > maxF) maxF = temp;	
        }
        avgF /= pSPARC->n_atom;
		fprintf(output_fp,"RMS force                          :%18.10E (Ha/Bohr)\n",avgF);
        fprintf(output_fp,"Maximum force                      :%18.10E (Ha/Bohr)\n",maxF);
        fprintf(output_fp,"Time for force calculation         :  %.3f (sec)\n",t2-t1);
        fclose(output_fp);
    }
    
    // Calculate Stress and pressure
    if(pSPARC->Calc_stress == 1){
        t1 = MPI_Wtime();
        Calculate_electronic_stress(pSPARC);
        // if (pSPARC->d3Flag == 1) d3_grad_cell_stress(pSPARC); // move this function into Calculate_electronic_stress?
        t2 = MPI_Wtime();
        if(!rank && pSPARC->Verbosity) {
            // write stress to .static file
            if (pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0) {
                static_fp = fopen(pSPARC->StaticFilename,"a");
                if (static_fp == NULL) {
                    printf("\nCannot open file \"%s\"\n",pSPARC->StaticFilename);
                    exit(EXIT_FAILURE);
                }
                if(pSPARC->MDFlag == 0 && pSPARC->RelaxFlag == 0){
                    fprintf(static_fp,"Stress");
                    PrintStress (pSPARC, pSPARC->stress, static_fp);
                }
                fclose(static_fp);
            }
            // write max stress to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            double maxS = 0.0, temp;
            if (pSPARC->CyclixFlag) {
                maxS = fabs(pSPARC->stress[5]);
            } else {
                for (i = 0; i< 6; i++){
                    temp = fabs(pSPARC->stress[i]);
                    if(temp > maxS)
                        maxS = temp;	
                }
            }

            if (pSPARC->BC == 2){
                fprintf(output_fp,"Pressure                           :%18.10E (GPa)\n",pSPARC->pres*CONST_HA_BOHR3_GPA);
                fprintf(output_fp,"Maximum stress                     :%18.10E (GPa)\n",maxS*CONST_HA_BOHR3_GPA);
            } else{
                double cellsizes[3] = {pSPARC->range_x, pSPARC->range_y, pSPARC->range_z};
                int BCs[3] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
                char stressUnit[16];
                double maxSGPa = convertStressToGPa(maxS, cellsizes, BCs, stressUnit);
                fprintf(output_fp,"Maximum stress                     :%18.10E (%s)\n",maxS,stressUnit);
                fprintf(output_fp,"Maximum stress equiv. to periodic  :%18.10E (GPa)\n",maxSGPa);
            }
            fprintf(output_fp,"Time for stress calculation        :  %.3f (sec)\n",t2-t1);
            fclose(output_fp);
        }
    } else if(pSPARC->Calc_pres == 1){
        t1 = MPI_Wtime();
        Calculate_electronic_pressure(pSPARC);
        // if (pSPARC->d3Flag == 1) d3_grad_cell_stress(pSPARC); // add the D3 contribution on pressure to total pressure. move this function into Calculate_electronic_pressure?
        t2 = MPI_Wtime();
        if(!rank && pSPARC->Verbosity) {
        	output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"Pressure                           :%18.10E (GPa)\n",pSPARC->pres*CONST_HA_BOHR3_GPA);
            fprintf(output_fp,"Time for pressure calculation      :  %.3f (sec)\n",t2-t1);
            fclose(output_fp);
        }
    }

    if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1){
        // force are only correct in intersection of all dmcomm and dmcomm_phi
		MPI_Bcast(pSPARC->forces, 3*pSPARC->n_atom, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    // convert non cartesian atom coordinates to cartesian 
	    if(pSPARC->cell_typ != 0){
	        for(i = 0; i < pSPARC->n_atom; i++)
	            nonCart2Cart_coord(pSPARC, &pSPARC->atom_pos[3*i], &pSPARC->atom_pos[3*i+1], &pSPARC->atom_pos[3*i+2]);	
		}
	}
	
	// Free the scf variables
    if (pSPARC->SQFlag == 1) {
        Free_scfvar_SQ(pSPARC);
    } else {
        Free_scfvar(pSPARC);
    }

    // print final electron density
    if (pSPARC->PrintElecDensFlag == 1) {
        #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        printElecDens(pSPARC);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Time for printing density: %.3f ms\n", (t2-t1)*1e3);
        #endif
    }

    // print final eigenvalues and occupations
    if (pSPARC->PrintEigenFlag == 1) {
        #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        printEigen(pSPARC);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Time for printing eigenvalues: %.3f ms\n", (t2-t1)*1e3);
        #endif
    }

    // print final orbitals
    if (pSPARC->PrintPsiFlag[0] == 1) {
        #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        print_orbitals(pSPARC);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Time for printing orbitals: %.3f ms\n", (t2-t1)*1e3);
        #endif
    }

    // print energy density
    if (pSPARC->PrintEnergyDensFlag == 1) {
        #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
        #endif
        printEnergyDensity(pSPARC);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("Time for printing energy density: %.3f ms\n", (t2-t1)*1e3);
        #endif
    }
}


/**
 * @brief   Calculate electronic ground state electron density and energy 
 *          for fixed atom positions.  
 */
void Calculate_EGS_elecDensEnergy(SPARC_OBJ *pSPARC) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    double t1, t2;
    if (rank == 0) printf("Calculating electron density ... \n");
    t1 = MPI_Wtime();
#endif
    
    // find atoms that influence the process domain
    GetInfluencingAtoms(pSPARC);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nFinding influencing atoms took %.3f ms\n",(t2-t1)*1000); 
    t1 = MPI_Wtime();
#endif
    
    // calculate pseudocharge density b
    Generate_PseudoChargeDensity(pSPARC);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculating b & b_ref took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();   
#endif

    if (pSPARC->SQFlag == 1) {
        SQ_OBJ *pSQ = pSPARC->pSQ;
        GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc_kptcomm, 
                        pSQ->DMVertices_PR, pSQ->dmcomm_SQ);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nFinding nonlocal influencing atoms in dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif

        CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj_kptcomm, 
                        pSPARC->Atom_Influence_nloc_kptcomm, pSQ->DMVertices_PR, pSQ->dmcomm_SQ);

    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nCalculating nonlocal projectors in dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif

        GetNonlocalProjectorsForNode(pSPARC, pSPARC->nlocProj_kptcomm, &pSPARC->nlocProj_SQ, 
                        pSPARC->Atom_Influence_nloc_kptcomm, &pSPARC->Atom_Influence_nloc_SQ, pSQ->dmcomm_SQ);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nGetting nonlocal projectors for each node in dmcomm_SQ took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif
        
        // TODO: Add correction term 
        if(pSPARC->SQ_correction == 1) {
            OverlapCorrection_SQ(pSPARC);
            OverlapCorrection_forces_SQ(pSPARC);
        }
    } else {
        // find atoms that have nonlocal influence the process domain (of psi-domain)
        GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc, pSPARC->DMVertices_dmcomm, 
                                pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nFinding nonlocal influencing atoms in psi-domain took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif
        
        // calculate nonlocal projectors in psi-domain
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                        pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        else
            CalculateNonlocalProjectors_kpt(pSPARC, &pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                            pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);	                            
        
        if (pSPARC->SOC_Flag) {
            CalculateNonlocalProjectors_SOC(pSPARC, pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                            pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
            CreateChiSOMatrix(pSPARC, pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                            pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nCalculating nonlocal projectors in psi-domain took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();   
    #endif
        
        // find atoms that have nonlocal influence the process domain (of kptcomm_topo)
        GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->DMVertices_kptcomm, 
                                pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nFinding nonlocal influencing atoms in kptcomm_topo took %.3f ms\n",(t2-t1)*1000);
        t1 = MPI_Wtime();
    #endif
        
        // calculate nonlocal projectors in kptcomm_topo
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                        pSPARC->DMVertices_kptcomm, 
                                        pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        else
            CalculateNonlocalProjectors_kpt(pSPARC, &pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                            pSPARC->DMVertices_kptcomm, 
                                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);								    
        
        if (pSPARC->SOC_Flag) {
            CalculateNonlocalProjectors_SOC(pSPARC, pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                            pSPARC->DMVertices_kptcomm, 
                                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
            CreateChiSOMatrix(pSPARC, pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf("\nCalculating nonlocal projectors in kptcomm_topo took %.3f ms\n",(t2-t1)*1000);   
    #endif
        
        // initialize orbitals psi
        Init_orbital(pSPARC);
    }

    // initialize electron density rho (initial guess)
    Init_electronDensity(pSPARC);

    // solve KS-DFT equations using Chebyshev filtered subspace iteration
    scf(pSPARC);

    // DFT-D3 correction: it does not depend on SCF
    if (pSPARC->d3Flag == 1) {
        #ifdef DEBUG
            t1 = MPI_Wtime();
        #endif
        d3_energy_gradient(pSPARC);
        #ifdef DEBUG
            t2 = MPI_Wtime();
            if (rank == 0) printf("Time for D3 calculation:    %.3f (sec)\n",t2-t1);
        #endif
    }
}


/**
 * @brief   Chebyshev filtered subpace iteration method for KS-DFT self-consistent field (SCF) calculations.
 */
void scf(SPARC_OBJ *pSPARC)
{
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
    FILE *output_fp;
    
    #ifdef USE_EVA_MODULE
    // TODO: Remove pSPARC->cell_typ when EVA support non-orthogonal cells
    EVA_buff_init(pSPARC->order, pSPARC->cell_typ);
    #endif
    
	#ifdef DEBUG
    if (rank == 0) printf("Start SCF calculation ... \n");
	#endif
    
    if (!rank && pSPARC->Verbosity) {
        output_fp = fopen(pSPARC->OutFilename,"a");
	    if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        
        if(pSPARC->spin_typ == 0)
            fprintf(output_fp,"===================================================================\n");
        else if(pSPARC->spin_typ == 1)
            fprintf(output_fp,"========================================================================================\n");
        else
            fprintf(output_fp,"======================================================================================================================\n");
            

        if(pSPARC->MDFlag == 1)
            fprintf(output_fp,"                    Self Consistent Field (SCF#%d)                     \n",
                    pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
        else if(pSPARC->RelaxFlag >= 1)
            fprintf(output_fp,"                    Self Consistent Field (SCF#%d)                     \n",
                    pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
        else
            fprintf(output_fp,"                    Self Consistent Field (SCF#%d)                     \n",1);
        
        if(pSPARC->spin_typ == 0)
            fprintf(output_fp,"===================================================================\n");
        else if(pSPARC->spin_typ == 1)
            fprintf(output_fp,"========================================================================================\n");
        else
            fprintf(output_fp,"======================================================================================================================\n");

        if(pSPARC->spin_typ == 0)
            fprintf(output_fp,"Iteration     Free Energy (Ha/atom)   SCF Error        Timing (sec)\n");
        else if(pSPARC->spin_typ == 1)
            fprintf(output_fp,"Iteration     Free Energy (Ha/atom)    Magnetization     SCF Error        Timing (sec)\n");
        else
            fprintf(output_fp,"Iteration     Free Energy (Ha/atom)            Magnetization (tot,x,y,z)                 SCF Error        Timing (sec)\n");

        fclose(output_fp);
    }
    
    #ifdef DEBUG
    double t1, t2;
    #endif
    
    int DMnd = pSPARC->Nd_d;
    int i;
    // solve the poisson equation for electrostatic potential, "phi"
    Calculate_elecstPotential(pSPARC);

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    // calculate xc potential (LDA), "Vxc"
    Calculate_Vxc(pSPARC);
    pSPARC->countPotentialCalculate++;
	#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, XC calculation took %.3f ms\n", rank, (t2-t1)*1e3); 
    t1 = MPI_Wtime(); 
	#endif 
    
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);
    
    // initialize mixing_hist_xk (and mixing_hist_xkm1)
    Update_mixing_hist_xk(pSPARC);
    
    if (pSPARC->SQFlag == 1) {
        for (i = 0; i < pSPARC->Nspden; i++)
            TransferVeff_phi2sq(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->pSQ->Veff_loc_SQ + i*pSPARC->Nd_d_dmcomm);
    } else {
        // transfer Veff_loc from "phi-domain" to "psi-domain"
        for (i = 0; i < pSPARC->Nspden; i++)
            Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
    }
    
	#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("rank = %d, Veff calculation and Bcast (non-blocking) took %.3f ms\n",rank,(t2-t1)*1e3); 
    }
	#endif

    scf_loop(pSPARC);
    if (pSPARC->usefock > 0) {
        pSPARC->usefock ++;
        Exact_Exchange_loop(pSPARC);
        pSPARC->usefock ++;
    }
}

/**
 * @brief   KS-DFT self-consistent field (SCF) calculations.
 */
void scf_loop(SPARC_OBJ *pSPARC) {
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    int DMnd = pSPARC->Nd_d;    
    int i, k, SCFcount;

#ifdef DEBUG
    int spn_i;
    int Nk = pSPARC->Nkpts_kptcomm;
    int Ns = pSPARC->Nstates;
#endif

    double error, dEtot, dEband;
    double t_scf_s, t_scf_e, t_cum_scf;
    double Veff_mean[4];

#ifdef DEBUG
    double t1, t2;
#endif
    
    FILE *output_fp;

    t_cum_scf = 0.0;
    error = pSPARC->TOL_SCF + 1.0;
    dEtot = dEband = pSPARC->TOL_SCF + 1.0;    
    if (pSPARC->usefock > 1) pSPARC->MINIT_SCF = 1;

    // 1st SCF will perform Chebyshev filtering several times, "count" keeps track 
    // of number of Chebyshev filtering calls, while SCFcount keeps track of # of
    // electron density updates
    SCFcount = 0;

    while (SCFcount < pSPARC->MAXIT_SCF) {
#ifdef DEBUG
        if (rank == 0) {
            printf("-------------\n"
                   "SCF iter %d  \n"
                   "-------------\n", SCFcount+1);
        }
#endif
        
        // used for QE scf error, save input rho_in and phi_in
        if (pSPARC->scf_err_type == 1) {
            memcpy(pSPARC->phi_dmcomm_phi_in, pSPARC->elecstPotential, DMnd * sizeof(double));
            memcpy(pSPARC->rho_dmcomm_phi_in, pSPARC->electronDens   , DMnd * sizeof(double));
		}

        // update Veff_loc_dmcomm_phi_in
        if (pSPARC->MixingVariable == 1) {
            double *Veff_out = (pSPARC->spin_typ == 2) ? pSPARC->Veff_dia_loc_dmcomm_phi : pSPARC->Veff_loc_dmcomm_phi;
            memcpy(pSPARC->Veff_loc_dmcomm_phi_in, Veff_out, sizeof(double)*pSPARC->Nd_d*pSPARC->Nspdend);
        }
        // update electronDens_in
        if (pSPARC->spin_typ > 0 && pSPARC->MixingVariable == 0) {
            memcpy(pSPARC->electronDens_in, pSPARC->electronDens, pSPARC->Nspdentd*DMnd * sizeof(double));
        }

		// start scf timer
        t_scf_s = MPI_Wtime();
        
        if (pSPARC->SQFlag == 1)
            Calculate_elecDens_SQ(pSPARC, SCFcount);
        else {
            Calculate_elecDens(rank, pSPARC, SCFcount, error);
            if ((pSPARC->ixc[2]) && (pSPARC->countPotentialCalculate))
                compute_Kinetic_Density_Tau_Transfer_phi(pSPARC);
        }

        // Calculate net magnetization for spin polarized calculations
        if (pSPARC->spin_typ != 0)
            Calculate_magnetization(pSPARC);
        
        if (pSPARC->MixingVariable == 0) { // density mixing
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif
            dEband = pSPARC->Eband;
            dEtot  = pSPARC->Etot;
            // Calculate/estimate system energies
            // Note: we estimate energy based on rho_in, which is shown
            //       to make energy convergence faster. But note that 
            //       this functional is not variational.
            //       An alternative way is to add a correction term to express
            //       everything in rho_out (to correct the rho_in terms in band 
            //       struc energy). This is done once SCF is converged. Therefore
            //       the final reported energy is variational.            
            if(pSPARC->spin_typ == 0) {
                Calculate_Free_Energy(pSPARC, pSPARC->mixing_hist_xk);
            } else {
                Calculate_Free_Energy(pSPARC, pSPARC->electronDens_in);                
            }
            
            dEband = fabs(dEband - pSPARC->Eband) / pSPARC->n_atom;
            dEtot  = fabs(dEtot  - pSPARC->Etot ) / pSPARC->n_atom;

		    #ifdef DEBUG
            t2 = MPI_Wtime();
            if(!rank) 
            printf("rank = %d, Calculating/Estimating energy took %.3f ms, "
                   "Etot = %.9f, dEtot = %.3e, dEband = %.3e\n", 
                   rank,(t2-t1)*1e3,pSPARC->Etot,dEtot,dEband);
		    #endif    
        }

        // update potential for potential mixing only
        if (pSPARC->MixingVariable == 1) { // potential mixing
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif

            // solve the poisson equation for electrostatic potential, "phi"
            Calculate_elecstPotential(pSPARC);

		    #ifdef DEBUG
            t2 = MPI_Wtime();
            if(!rank) printf("rank = %d, Solving poisson equation took %.3f ms\n", rank, (t2 - t1) * 1e3);
		    t1 = MPI_Wtime();
            #endif
            
            // calculate xc potential (LDA, PW92), "Vxc"
            Calculate_Vxc(pSPARC);
		    pSPARC->countPotentialCalculate++;
		    #ifdef DEBUG
            t2 = MPI_Wtime();
            if(!rank) printf("rank = %d, Calculating Vxc took %.3f ms\n", rank, (t2 - t1) * 1e3);
		    #endif            
            
            // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
            Calculate_Veff_loc_dmcomm_phi(pSPARC);
        }

        if (pSPARC->MixingVariable == 1) { // potential mixing
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif
            dEband = pSPARC->Eband;
            dEtot  = pSPARC->Etot;

            // Calculate/estimate system energies
            Calculate_Free_Energy(pSPARC, pSPARC->electronDens);
            
            // Self Consistency Correction to energy when we evaluate energy based on rho_out
            double Escc = 0.0;
            double *Veff_out = (pSPARC->spin_typ == 2) ? pSPARC->Veff_dia_loc_dmcomm_phi : pSPARC->Veff_loc_dmcomm_phi;
            int ncol = (pSPARC->spin_typ == 0) ? 1 : 2;
            Escc = Calculate_Escc(
                pSPARC, DMnd, ncol, Veff_out, 
                pSPARC->Veff_loc_dmcomm_phi_in, pSPARC->electronDens + ((pSPARC->spin_typ > 0) ? DMnd : 0),
                pSPARC->dmcomm_phi
            );
            pSPARC->Escc = Escc;

            #ifdef DEBUG
            if(!rank) printf("Escc = %.3e\n", Escc);
            #endif

            // add self consistency correction to total energy
            pSPARC->Etot = pSPARC->Etot + Escc;
            
            dEband = fabs(dEband - pSPARC->Eband) / pSPARC->n_atom;
            dEtot  = fabs(dEtot  - pSPARC->Etot ) / pSPARC->n_atom;
            
		    #ifdef DEBUG
            t2 = MPI_Wtime();
            if(!rank) 
            printf("rank = %d, Calculating/Estimating energy took %.3f ms, "
                   "Etot = %.9f, dEtot = %.3e, dEband = %.3e\n", 
                   rank,(t2-t1)*1e3,pSPARC->Etot,dEtot,dEband);
		    #endif
        }

        // scf convergence flag
        int scf_conv = 0;

        // find SCF error
        if (pSPARC->scf_err_type == 0) { // default
            Evaluate_scf_error(pSPARC, &error, &scf_conv);
        } else if (pSPARC->scf_err_type == 1) { // QE scf err: conv_thr
            Evaluate_QE_scf_error(pSPARC, &error, &scf_conv);
        }
        
        // check if Etot is NaN
        if (pSPARC->Etot != pSPARC->Etot) {
            if (!rank) printf("ERROR: Etot is NaN\n");
            exit(EXIT_FAILURE);
        }

        // Apply mixing and preconditioner, if required
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif

        // find mean value of Veff, and shift Veff by -mean(Veff)
        if (pSPARC->MixingVariable == 1) { // potential mixing 
            shiting_Veff_mean(pSPARC, pSPARC->Veff_loc_dmcomm_phi, pSPARC->Nspden, Veff_mean, 0, -1);
        }

        Mixing(pSPARC, SCFcount);

        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank) printf("rank = %d, Mixing (+ precond) took %.3f ms\n", rank, (t2 - t1) * 1e3);
        #endif

        if (pSPARC->MixingVariable == 1) { // potential mixing, add veff_mean back
            // shift the next input veff so that it's integral is
            // equal to that of the current output veff for periodic systems
            // shift Veff by +mean(Veff)
            shiting_Veff_mean(pSPARC, pSPARC->Veff_loc_dmcomm_phi, pSPARC->Nspden, Veff_mean, 1, 1);
        } else if (pSPARC->MixingVariable == 0) { // recalculate potential for density mixing 
            // solve the poisson equation for electrostatic potential, "phi"
            Calculate_elecstPotential(pSPARC);
            Calculate_Vxc(pSPARC);
            pSPARC->countPotentialCalculate++;
            Calculate_Veff_loc_dmcomm_phi(pSPARC);
        }

        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif

        if (pSPARC->SQFlag == 1) {
            for (i = 0; i < pSPARC->Nspden; i++)
                TransferVeff_phi2sq(pSPARC, pSPARC->Veff_loc_dmcomm_phi, pSPARC->pSQ->Veff_loc_SQ);
        } else {
            // transfer Veff_loc from "phi-domain" to "psi-domain"
            for (i = 0; i < pSPARC->Nspden; i++)
                Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
        }

        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank) 
            printf("rank = %d, Transfering Veff from phi-domain to psi-domain took %.3f ms\n", 
                rank, (t2 - t1) * 1e3);
        #endif 
        
        SCFcount++;
        t_scf_e = MPI_Wtime();
        #ifdef DEBUG        
        if (!rank) printf("\nThis SCF took %.3f ms, scf error = %.3e\n", (t_scf_e-t_scf_s)*1e3, error);
        #endif

        t_cum_scf += t_scf_e-t_scf_s;
        if (!rank) {
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            if(pSPARC->spin_typ == 0)
                fprintf(output_fp,"%-6d      %18.10E        %.3E        %.3f\n", 
                        SCFcount, pSPARC->Etot/pSPARC->n_atom, error, t_cum_scf);
            else if(pSPARC->spin_typ == 1)
                fprintf(output_fp,"%-6d      %18.10E        %11.4E        %.3E        %.3f\n", 
                            SCFcount, pSPARC->Etot/pSPARC->n_atom, pSPARC->netM[0], error, t_cum_scf);
            else if(pSPARC->spin_typ == 2)
                fprintf(output_fp,"%-6d      %18.10E    %11.4E, %11.4E, %11.4E, %11.4E     %.3E        %.3f\n", 
                            SCFcount, pSPARC->Etot/pSPARC->n_atom, 
                            pSPARC->netM[0], pSPARC->netM[1], pSPARC->netM[2], pSPARC->netM[3], error, t_cum_scf);

            fclose(output_fp);
            t_cum_scf = 0.0;
        }

        // STOP scf if scf error is less than the given tolerance
        if (scf_conv && SCFcount >= pSPARC->MINIT_SCF) break;
    }
    
    if (!rank) {
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"Total number of SCF: %-6d\n",SCFcount);
        // for density mixing, extra poisson solve is needed
        if (pSPARC->scf_err_type == 1 && pSPARC->MixingVariable == 0) { 
            fprintf(output_fp,"Extra time for evaluating QE SCF Error: %.3f (sec)\n", pSPARC->t_qe_extra);
        }
        fclose(output_fp);
    }

    #ifdef USE_EVA_MODULE
    EVA_buff_finalize();
    #endif
    
	// check occupation (if Nstates is large enough)
    if (pSPARC->SQFlag == 1) {
        SQ_OBJ *pSQ = pSPARC->pSQ;
        if (pSQ->dmcomm_SQ != MPI_COMM_NULL) {
            // Find occupation corresponding to maximum eigenvalue
            double maxeig, occ_maxeig;
            maxeig = pSQ->maxeig[0];
            for (k = 1; k < pSQ->DMnd_SQ; k++) {
                if (pSQ->maxeig[k] > maxeig)
                    maxeig = pSQ->maxeig[k];
            }
            occ_maxeig = 2.0 * smearing_function(pSPARC->Beta, maxeig, pSPARC->Efermi, pSPARC->elec_T_type);

            #ifdef DEBUG
            if (!rank) printf("Max eigenvalue and corresponding occupation number: %.15f, %.15f\n",maxeig,occ_maxeig);
            #endif

            if (occ_maxeig > pSPARC->SQ_tol_occ) {
                if (!rank)  printf("WARNING: Occupation number corresponding to maximum "
                            "eigenvalue exceeded the tolerance of: %E\n", pSPARC->SQ_tol_occ);
            }
        }
    } else {
        #ifdef DEBUG
        int spin_maxocc = 0, k_maxocc = 0;
        double maxocc = -1.0;
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++){
            for (k = 0; k < Nk; k++) {
                int ind = pSPARC->Nstates-1;
                ind = max(ind,0);
                double g_ind = pSPARC->occ_sorted[spn_i*Ns*Nk + k*Ns + ind];
                
                if (g_ind > maxocc) {
                    maxocc = g_ind;
                    spin_maxocc = spn_i;
                    k_maxocc = k;
                }

                // #ifdef DEBUG
                if(!rank) {
                    int nocc_print = min(200,pSPARC->Nstates - pSPARC->Nelectron/2 + 10);
                    nocc_print = min(nocc_print, pSPARC->Nstates);
                    printf("The last %d occupations of kpoints #%d are (Nelectron = %d):\n", nocc_print, k+1, pSPARC->Nelectron);
                    for (i = 0; i < nocc_print; i++) {
                        printf("lambda[%4d] = %18.14f, occ[%4d] = %18.14f\n", 
                                Ns - nocc_print + i + 1, 
                                pSPARC->lambda_sorted[spn_i*Ns*Nk + k*Ns + Ns - nocc_print + i],
                                Ns - nocc_print + i + 1, 
                                pSPARC->occfac * pSPARC->occ_sorted[spn_i*Ns*Nk + k*Ns + Ns - nocc_print + i]);
                    }
                }
            }
        }

        // print occ(0.9*NSTATES)] and occ(NSTATES) in DEBUG mode (only for the k point that gives max occ)
        spn_i = spin_maxocc;
        k = k_maxocc;
        if(!rank) {
            double k1_red, k2_red, k3_red;
            if (pSPARC->BC != 1) {
                k1_red = pSPARC->k1_loc[k]*pSPARC->range_x/(2.0*M_PI);
                k2_red = pSPARC->k2_loc[k]*pSPARC->range_y/(2.0*M_PI);
                k3_red = pSPARC->k3_loc[k]*pSPARC->range_z/(2.0*M_PI);
            } else {
                k1_red = k2_red = k3_red = 0.0;
            }
            int ind_90percent = round(pSPARC->Nstates * 0.90) - 1;
            int ind_100percent = pSPARC->Nstates - 1;
            double g_ind_90percent = pSPARC->occ_sorted[spn_i*Ns*Nk + k*Ns + ind_90percent];
            double g_ind_100percent = pSPARC->occ_sorted[spn_i*Ns*Nk + k*Ns + ind_100percent];
            // write to .out file
            if (pSPARC->BC != 1) printf("\nk = [%.3f, %.3f, %.3f]\n", k1_red, k2_red, k3_red);
            printf("Occupation of state %d (90%%) = %.15f.\n"
                "Occupation of state %d (100%%) = %.15f.\n",
                ind_90percent+1, (3.0-pSPARC->Nspin)/pSPARC->Nspinor * g_ind_90percent,
                ind_100percent+1, (3.0-pSPARC->Nspin)/pSPARC->Nspinor * g_ind_100percent);
        }
        #endif
    }

    // check if scf is converged
    double TOL = (pSPARC->usefock % 2 == 1) ? pSPARC->TOL_SCF_INIT : pSPARC->TOL_SCF;
    if (error > TOL) {
    	if(!rank) {
            printf("WARNING: SCF#%d did not converge to desired accuracy!\n",
            		pSPARC->MDFlag ? pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0) : pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"WARNING: SCF#%d did not converge to desired accuracy!\n",
            		pSPARC->MDFlag ? pSPARC->MDCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0) : pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
            fclose(output_fp);
        }
    }
}


/**
 * @ brief Evaluate SCF error.
 *
 *         Depending on whether it's density mixing or potential mixing, 
 *         the scf error is defines as
 *           scf error := || rho  -  rho_old || / ||  rho ||, or
 *           scf error := || Veff - Veff_old || / || Veff ||.
 *         Note that for Periodic BC, Veff and Veff_old here are shifted 
 *         so that their mean value is 0.
 */
void Evaluate_scf_error(SPARC_OBJ *pSPARC, double *scf_error, int *scf_conv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int DMnd = pSPARC->Nd_d;
    int len = pSPARC->Nspdend * DMnd;

    double *var_in = NULL, *var_out = NULL;    
    if (pSPARC->MixingVariable == 0) {
        var_in = (pSPARC->spin_typ) ? (pSPARC->electronDens_in + DMnd) : pSPARC->mixing_hist_xk;
        var_out = pSPARC->electronDens + ((pSPARC->spin_typ > 0) ? DMnd : 0);
    } else if (pSPARC->MixingVariable == 1) {
        var_in = pSPARC->Veff_loc_dmcomm_phi_in;
        var_out = (pSPARC->spin_typ == 2) ? pSPARC->Veff_dia_loc_dmcomm_phi : pSPARC->Veff_loc_dmcomm_phi;
    }
    
    double sbuf[2] = {0, 0};
    for (int i = 0; i < len; i++) {
        double temp = var_out[i] - var_in[i];
        sbuf[0] += var_out[i] * var_out[i];
        sbuf[1] += temp * temp;
    }

    double error;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        MPI_Allreduce(MPI_IN_PLACE, sbuf, 2, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
        error = sqrt(sbuf[1] / sbuf[0]);
    }
    MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // output
    *scf_error = error;
    *scf_conv  = (pSPARC->usefock % 2 == 1) 
                ? ((int) (error < pSPARC->TOL_SCF_INIT)) : ((int) (error < pSPARC->TOL_SCF));
}



/**
 * @brief Evaluate the scf error defined in Quantum Espresso.
 *
 *        Find the scf error defined in Quantum Espresso. QE implements 
 *        Eq.(A.7) of the reference paper, with a slight modification: 
 *          conv_thr = 4 \pi e^2 \Omega \sum_G |\Delta \rho(G)|^2 / G^2
 *        This is equivalent to 2 * Eq.(A.6) in the reference paper.
 *
 * @ref   P Giannozzi et al, J. Phys.:Condens. Matter 21(2009) 395502.
 */
void Evaluate_QE_scf_error(SPARC_OBJ *pSPARC, double *scf_error, int *scf_conv) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // update phi_out for density mixing
    if (pSPARC->MixingVariable == 0) { // desity mixing
        double t1, t2;
        t1 = MPI_Wtime();
        // solve the poisson equation for electrostatic potential, "phi"
        Calculate_elecstPotential(pSPARC);
        t2 = MPI_Wtime();
        if (!rank) printf("QE scf error: update phi_out took %.3f ms\n", (t2-t1)*1e3); 
        pSPARC->t_qe_extra += (t2 - t1);
    }

    double error = 0.0;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        int i;
        int DMnd = pSPARC->Nd_d;
        double loc_err = 0.0;
        for (i = 0; i < DMnd; i++) {
            loc_err += (pSPARC->electronDens[i]    - pSPARC->rho_dmcomm_phi_in[i]) * 
                       (pSPARC->elecstPotential[i] - pSPARC->phi_dmcomm_phi_in[i]);
        }
        loc_err = fabs(loc_err * pSPARC->dV); // in case error is not numerically positive
        MPI_Allreduce(&loc_err, &error, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    }

    MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);   
    // output
    *scf_error = error;
    *scf_conv  = (pSPARC->usefock % 2 == 1) 
                ? ((int) (error < pSPARC->TOL_SCF_INIT)) : ((int) (error < pSPARC->TOL_SCF));
}



/**
* @ brief find net magnetization of the system
**/
void Calculate_magnetization(SPARC_OBJ *pSPARC)
{
    if(pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int DMnd = pSPARC->Nd_d;
    for (int n = 0; n < pSPARC->Nmag; n++) {
        pSPARC->netM[n] = 0;
        for (int i = 0; i < DMnd; i++) {
            if (pSPARC->CyclixFlag)
                pSPARC->netM[n] += pSPARC->mag[i+n*DMnd] * pSPARC->Intgwt_phi[i];
            else
                pSPARC->netM[n] += pSPARC->mag[i+n*DMnd] * pSPARC->dV;
        }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, pSPARC->netM, pSPARC->Nmag, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
}    

/**
 * @brief   Calculate Veff_loc = phi + Vxc (in phi-domain).
 */
void Calculate_Veff_loc_dmcomm_phi(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int DMnd = pSPARC->Nd_d;

    if (pSPARC->spin_typ == 2) {
        for (int i = 0; i < DMnd; i++) {
            // complete 4 terms
            pSPARC->Veff_loc_dmcomm_phi[i] = pSPARC->XCPotential_nc[i] + pSPARC->elecstPotential[i];
            pSPARC->Veff_loc_dmcomm_phi[i+DMnd] = pSPARC->XCPotential_nc[i+DMnd] + pSPARC->elecstPotential[i];
            pSPARC->Veff_loc_dmcomm_phi[i+2*DMnd] = pSPARC->XCPotential_nc[i+2*DMnd];
            pSPARC->Veff_loc_dmcomm_phi[i+3*DMnd] = pSPARC->XCPotential_nc[i+3*DMnd];

            // diagonal 2 terms
            if (pSPARC->MixingVariable == 1) {
                pSPARC->Veff_dia_loc_dmcomm_phi[i] = pSPARC->XCPotential[i] + pSPARC->elecstPotential[i];
                pSPARC->Veff_dia_loc_dmcomm_phi[i+DMnd] = pSPARC->XCPotential[i+DMnd] + pSPARC->elecstPotential[i];
            }
        }        
    } else {
        assert(pSPARC->Nspdend == 1 || pSPARC->Nspdend == 2);
        for (int n = 0; n < pSPARC->Nspdend; n++) {
            int shift = n * DMnd;
            for (int i = 0; i < DMnd; i++) {
                pSPARC->Veff_loc_dmcomm_phi[shift + i] = pSPARC->XCPotential[shift + i] + pSPARC->elecstPotential[i];
            }
        }
    }
}



/**
 * @brief   Update mixing_hist_xk.
 */
// TODO: check if this function is necessary!
void Update_mixing_hist_xk(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;    
    double Veff_mean[4];
    if (pSPARC->MixingVariable == 0) {
        memcpy(pSPARC->mixing_hist_xkm1, pSPARC->electronDens, sizeof(double)*pSPARC->Nd_d);
        if (pSPARC->spin_typ > 0) {
            int ncol = (pSPARC->spin_typ == 2 ? 3 : 1);
            memcpy(pSPARC->mixing_hist_xkm1+pSPARC->Nd_d, pSPARC->mag + (pSPARC->spin_typ == 2) * pSPARC->Nd_d, sizeof(double)*pSPARC->Nd_d*ncol);
        }         
    } else {
        memcpy(pSPARC->mixing_hist_xkm1, pSPARC->Veff_loc_dmcomm_phi, sizeof(double)*pSPARC->Nd_d*pSPARC->Nspden);
        shiting_Veff_mean(pSPARC, pSPARC->mixing_hist_xkm1, pSPARC->Nspden, Veff_mean, 0, -1);
        memcpy(pSPARC->mixing_hist_xk, pSPARC->mixing_hist_xkm1, sizeof(double)*pSPARC->Nd_d*pSPARC->Nspden);
    }
    memcpy(pSPARC->mixing_hist_xk, pSPARC->mixing_hist_xkm1, sizeof(double)*pSPARC->Nd_d*pSPARC->Nspden);
}


void shiting_Veff_mean(SPARC_OBJ *pSPARC, double *Veff, int ncol, double *Veff_mean, int option, int dir)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    // when option = 0, calculate Veff_mean, option = 1, don't do it
    // shift by dir * Veff_mean
    assert(option == 0 || option == 1);
    assert(dir == 1 || dir == -1);
    // user need ensure Veff_mean at least with size ncol*1
    if (pSPARC->BC != 2) {
        memset(Veff_mean, 0, sizeof(double)*ncol);
        return;
    }

    int DMnd = pSPARC->Nd_d;
    if (option == 0) {
        for (int i = 0; i < ncol; i++) {
            VectorSum(Veff + i*DMnd, DMnd, Veff_mean+i, pSPARC->dmcomm_phi);
            Veff_mean[i] /= pSPARC->Nd;
        }
        if (pSPARC->spin_typ == 1) {
            Veff_mean[0] = Veff_mean[1] = (Veff_mean[0] + Veff_mean[1])/2.0;
        }
    }
    
    for (int i = 0; i < ncol; i++) {
        VectorShift(Veff + i*DMnd, DMnd, dir * Veff_mean[i], pSPARC->dmcomm_phi);
    }
}




/**
 * @brief   Transfer Veff_loc from phi-domain to psi-domain.
 *
 *          Use DD2DD (Domain Decomposition to Domain Decomposition) to 
 *          do the transmision between phi-domain and the dmcomm that 
 *          contains root process, and then broadcast to all dmcomms.
 */
void Transfer_Veff_loc(SPARC_OBJ *pSPARC, double *Veff_phi_domain, double *Veff_psi_domain) 
{
#ifdef DEBUG
    double t1, t2;
#endif
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Transmitting Veff_loc from phi-domain to psi-domain (LOCAL) ...\n");
#endif    
    //void DD2DD(SPARC_OBJ *pSPARC, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert, double *rdata, 
    //       MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims)
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx_phi; sdims[1] = pSPARC->npNdy_phi; sdims[2] = pSPARC->npNdz_phi;
    rdims[0] = pSPARC->npNdx; rdims[1] = pSPARC->npNdy; rdims[2] = pSPARC->npNdz;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    D2D(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, Veff_phi_domain, 
        pSPARC->DMVertices_dmcomm, Veff_psi_domain, pSPARC->dmcomm_phi, sdims, 
        (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, 
        rdims, MPI_COMM_WORLD, sizeof(double));
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: D2D took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first kptcomms in each spincomm
    if (pSPARC->npspin > 1 && pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index == 0) {
        MPI_Bcast(Veff_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: bcast btw/ spincomms of 1st kptcomm took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first bandcomms in each kptcomm
    if (pSPARC->spincomm_index >= 0 && pSPARC->npkpt > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->bandcomm_index == 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(Veff_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: bcast btw/ kptcomms of 1st bandcomm took %.3f ms\n",(t2-t1)*1e3);
#endif

    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast below on Quartz cluster, this Barrier fixed the issue (why?)

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Bcast phi from first bandcomm to all other bandcomms
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(Veff_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->blacscomm);    
    }
    pSPARC->req_veff_loc = MPI_REQUEST_NULL;
    
    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast above on Quartz cluster, this Barrier fixed the issue (why?)

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: mpi_bcast (count = %d) to all bandcomms took %.3f ms\n",pSPARC->Nd_d_dmcomm,(t2-t1)*1e3);
#endif

    if ((pSPARC->countPotentialCalculate > 1) && (pSPARC->ixc[2]))
        Transfer_vxcMGGA3_phi_psi(pSPARC, pSPARC->vxcMGGA3, pSPARC->vxcMGGA3_loc_dmcomm); // only transfer the potential they are going to use
}



/**
 * @brief   Transfer electron density from psi-domain to phi-domain.
 */
void TransferDensity(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv)
{
#ifdef DEBUG
    double t1, t2;
#endif
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int sdims[3], rdims[3], gridsizes[3];
    sdims[0] = pSPARC->npNdx; sdims[1] = pSPARC->npNdy; sdims[2] = pSPARC->npNdz;
    rdims[0] = pSPARC->npNdx_phi; rdims[1] = pSPARC->npNdy_phi; rdims[2] = pSPARC->npNdz_phi;
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    D2D(&pSPARC->d2d_dmcomm, &pSPARC->d2d_dmcomm_phi, gridsizes, pSPARC->DMVertices_dmcomm, rho_send, 
        pSPARC->DMVertices, rho_recv, (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, sdims, 
        pSPARC->dmcomm_phi, rdims, MPI_COMM_WORLD, sizeof(double));
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, D2D took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}


/**
 * @brief   Calculate magnetization of atoms
 */
void CalculateAtomMag(SPARC_OBJ *pSPARC)
{
    if(pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int ncol = (pSPARC->spin_typ == 2) ? 3 : 1;
    // reset AtomMag
    memset(pSPARC->AtomMag, 0, sizeof(double) * pSPARC->n_atom * ncol);
    double rc = 2.0;
    int DMnd = pSPARC->Nd_d;
    int DMnx = pSPARC->Nx_d;
    int DMny = pSPARC->Ny_d;

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;

    double DMxs = pSPARC->xin + pSPARC->DMVertices[0] * pSPARC->delta_x;
    double DMxe = pSPARC->xin + pSPARC->DMVertices[1] * pSPARC->delta_x; // note that this is not the actual edge, add BCx to get actual domain edge
    double DMys = pSPARC->DMVertices[2] * pSPARC->delta_y;
    double DMye = pSPARC->DMVertices[3] * pSPARC->delta_y; // note that this is not the actual edge, add BCx to get actual domain edge
    double DMzs = pSPARC->DMVertices[4] * pSPARC->delta_z;
    double DMze = pSPARC->DMVertices[5] * pSPARC->delta_z; // note that this is not the actual edge, add BCx to get actual domain edge
    double dis = 0;

    for (int iatm = 0; iatm < pSPARC->n_atom; iatm++) {
        double x0 = pSPARC->atom_pos[3*iatm];
        double y0 = pSPARC->atom_pos[3*iatm+1];
        double z0 = pSPARC->atom_pos[3*iatm+2];

        int ppmin, ppmax, qqmin, qqmax, rrmin, rrmax;
        ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;

        if (pSPARC->BCx == 0) {
            ppmax = floor((rc + Lx - x0) / Lx + TEMP_TOL);
            ppmin = -floor((rc + x0) / Lx + TEMP_TOL);
        }
        if (pSPARC->BCy == 0) {
            qqmax = floor((rc + Ly - y0) / Ly + TEMP_TOL);
            qqmin = -floor((rc + y0) / Ly + TEMP_TOL);
        }
        if (pSPARC->BCz == 0) {
            rrmax = floor((rc + Lz - z0) / Lz + TEMP_TOL);
            rrmin = -floor((rc + z0) / Lz + TEMP_TOL);
        }

        // check how many of it's images interacts with the local distributed domain
        for (int rr = rrmin; rr <= rrmax; rr++) {
            double z0_i = z0 + Lz * rr; // z coord of image atom
            if ((z0_i < DMzs - rc) || (z0_i >= DMze + rc)) continue;
            for (int qq = qqmin; qq <= qqmax; qq++) {
                double y0_i = y0 + Ly * qq; // y coord of image atom
                if ((y0_i < DMys - rc) || (y0_i >= DMye + rc)) continue;
                for (int pp = ppmin; pp <= ppmax; pp++) {
                    double x0_i = x0 + Lx * pp; // x coord of image atom
                    if ((x0_i < DMxs - rc) || (x0_i >= DMxe + rc)) continue;
                    
                    // find start & end nodes of the rc-region of the image atom
                    // This way, we try to make sure all points inside rc-region
                    // is strictly less that rc distance away from the image atom
                    double rc_xl = ceil( (x0_i - pSPARC->xin - rc)/pSPARC->delta_x);
                    double rc_xr = floor((x0_i - pSPARC->xin + rc)/pSPARC->delta_x);
                    double rc_yl = ceil( (y0_i - rc)/pSPARC->delta_y);
                    double rc_yr = floor((y0_i + rc)/pSPARC->delta_y);
                    double rc_zl = ceil( (z0_i - rc)/pSPARC->delta_z);
                    double rc_zr = floor((z0_i + rc)/pSPARC->delta_z);

                    // find overlap of rc-region of the image and the local dist. domain
                    double xs = max(pSPARC->DMVertices[0], rc_xl);
                    double xe = min(pSPARC->DMVertices[1], rc_xr);
                    double ys = max(pSPARC->DMVertices[2], rc_yl);
                    double ye = min(pSPARC->DMVertices[3], rc_yr);
                    double zs = max(pSPARC->DMVertices[4], rc_zl);
                    double ze = min(pSPARC->DMVertices[5], rc_zr);

                    for (int k = zs; k <= ze; k++) {
                        double z = k * pSPARC->delta_z;
                        for (int j = ys; j < ye; j++) {
                            double y = j * pSPARC->delta_y;
                            for (int i = xs; i < xe; i++) {
                                double x = pSPARC->xin + i * pSPARC->delta_x;
                                CalculateDistance(pSPARC, x, y, z, x0_i, y0_i, z0_i, &dis);
                                // check if the rc sphere of this image intersects with current domain
                                if (dis <= rc) {
                                    int k_DM = k - pSPARC->DMVertices[4];
                                    int j_DM = j - pSPARC->DMVertices[2];
                                    int i_DM = i - pSPARC->DMVertices[0];
                                    int indx = k_DM * (DMnx * DMny) + j_DM * DMnx + i_DM;
                                    if (pSPARC->spin_typ == 2) {
                                        pSPARC->AtomMag[3*iatm  ] += pSPARC->mag[indx + DMnd]   * pSPARC->dV;   // magx
                                        pSPARC->AtomMag[3*iatm+1] += pSPARC->mag[indx + 2*DMnd] * pSPARC->dV; // magy
                                        pSPARC->AtomMag[3*iatm+2] += pSPARC->mag[indx + 3*DMnd] * pSPARC->dV; // magz
                                    } 
                                    if (pSPARC->spin_typ == 1) {
                                        pSPARC->AtomMag[iatm] += pSPARC->mag[indx] * pSPARC->dV;   // magz
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, pSPARC->AtomMag, pSPARC->n_atom*ncol, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
}