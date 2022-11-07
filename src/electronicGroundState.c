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
#include "mixing.h" // AndersonExtrapolation
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
	    memset(pSPARC->mixing_hist_Xk, 0, sizeof(double)* pSPARC->Nd_d * pSPARC->Nspin * pSPARC->MixingHistory);
	    memset(pSPARC->mixing_hist_Fk, 0, sizeof(double)* pSPARC->Nd_d * pSPARC->Nspin * pSPARC->MixingHistory);
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
        if (pSPARC->vdWDFFlag != 0) {
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
            for (i = 0; i< 6; i++){
            	temp = fabs(pSPARC->stress[i]);
            	if(temp > maxS)
            		maxS = temp;	
            }
            if (pSPARC->BC == 2){
                fprintf(output_fp,"Pressure                           :%18.10E (GPa)\n",pSPARC->pres*CONST_HA_BOHR3_GPA);
                fprintf(output_fp,"Maximum stress                     :%18.10E (GPa)\n",maxS*CONST_HA_BOHR3_GPA);
            } else{
                fprintf(output_fp,"Maximum stress                     :%18.10E (a.u.)\n",maxS);
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
        else
            fprintf(output_fp,"========================================================================================\n");
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
        else
            fprintf(output_fp,"========================================================================================\n");
        if(pSPARC->spin_typ == 0)
            fprintf(output_fp,"Iteration     Free Energy (Ha/atom)   SCF Error        Timing (sec)\n");
        else
            fprintf(output_fp,"Iteration     Free Energy (Ha/atom)    Magnetization     SCF Error        Timing (sec)\n");
        fclose(output_fp);
    }
    
    #ifdef DEBUG
    double t1, t2;
    #endif
    
    int DMnd = pSPARC->Nd_d;
    int NspinDMnd = pSPARC->Nspin * DMnd;
    int i;
    // solve the poisson equation for electrostatic potential, "phi"
    Calculate_elecstPotential(pSPARC);

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    // calculate xc potential (LDA), "Vxc"
    Calculate_Vxc(pSPARC);
	#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, XC calculation took %.3f ms\n", rank, (t2-t1)*1e3); 
    t1 = MPI_Wtime(); 
	#endif 
    
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);
    double veff_mean;
    veff_mean = 0.0;
    // for potential mixing with PBC, calculate mean(veff)
    if (pSPARC->MixingVariable == 1)  { // potential mixing
        if (pSPARC->BC == 2 || pSPARC->BC == 0) {
            VectorSum(pSPARC->Veff_loc_dmcomm_phi, NspinDMnd, &veff_mean, pSPARC->dmcomm_phi);
            veff_mean /= ((double) (pSPARC->Nd * pSPARC->Nspin));
        }
    }
    pSPARC->veff_mean = veff_mean;

    // initialize mixing_hist_xk (and mixing_hist_xkm1)
    Update_mixing_hist_xk(pSPARC, veff_mean);
    
    if (pSPARC->SQFlag == 1) {
        TransferVeff_phi2sq(pSPARC, pSPARC->Veff_loc_dmcomm_phi, pSPARC->pSQ->Veff_loc_SQ);
    } else {
        // transfer Veff_loc from "phi-domain" to "psi-domain"
        for (i = 0; i < pSPARC->Nspin; i++)
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
    int NspinDMnd = pSPARC->Nspin * DMnd;
    int sindx_rho = (pSPARC->Nspin == 2) ? DMnd : 0;
    int i, k, SCFcount;

#ifdef DEBUG
    int spn_i;
    int Nk = pSPARC->Nkpts_kptcomm;
    int Ns = pSPARC->Nstates;
#endif

    double error, dEtot, dEband, veff_mean;
    double t_scf_s, t_scf_e, t_cum_scf;

#ifdef DEBUG
    double t1, t2;
#endif
    
    FILE *output_fp;

    t_cum_scf = 0.0;
    error = pSPARC->TOL_SCF + 1.0;
    dEtot = dEband = pSPARC->TOL_SCF + 1.0;
    veff_mean = pSPARC->veff_mean;
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

		// start scf timer
        t_scf_s = MPI_Wtime();
        
        if (pSPARC->SQFlag == 1)
            Calculate_elecDens_SQ(pSPARC, SCFcount);
        else
            Calculate_elecDens(rank, pSPARC, SCFcount, error);

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
            if(pSPARC->spin_typ != 0) {
                double *rho_in = (double *)malloc(3 * DMnd * sizeof(double));
                for (int i = 0; i < DMnd; i++) {
                    rho_in[i] = pSPARC->mixing_hist_xk[i] + pSPARC->mixing_hist_xk[DMnd+i];
                    rho_in[DMnd+i] = pSPARC->mixing_hist_xk[i];
                    rho_in[2*DMnd+i] = pSPARC->mixing_hist_xk[DMnd+i];
                }
                Calculate_Free_Energy(pSPARC, rho_in);
                free(rho_in);
            } else
                Calculate_Free_Energy(pSPARC, pSPARC->mixing_hist_xk);
            
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
            // for potential mixing, the history vector is shifted, so we need to add veff_mean back
            if (pSPARC->BC == 2 || pSPARC->BC == 0) {
                VectorShift(pSPARC->Veff_loc_dmcomm_phi_in, NspinDMnd, veff_mean, pSPARC->dmcomm_phi);
            }

            Escc = Calculate_Escc(
                pSPARC, NspinDMnd, pSPARC->Veff_loc_dmcomm_phi, 
                pSPARC->Veff_loc_dmcomm_phi_in, pSPARC->electronDens + sindx_rho,
                pSPARC->dmcomm_phi
            );
            pSPARC->Escc = Escc;

            // remove veff_mean again
            if (pSPARC->BC == 2 || pSPARC->BC == 0) {
                VectorShift(pSPARC->Veff_loc_dmcomm_phi_in, NspinDMnd, -veff_mean, pSPARC->dmcomm_phi);
            }

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
            
        // find mean value of Veff, and shift Veff by -mean(Veff)
        if (pSPARC->MixingVariable == 1 && pSPARC->BC == 2) { // potential mixing 
            // find mean of Veff
            VectorSum(pSPARC->Veff_loc_dmcomm_phi, NspinDMnd, &veff_mean, pSPARC->dmcomm_phi);
            veff_mean /= ((double) (pSPARC->Nd * pSPARC->Nspin));
            // shift Veff by -mean(Veff) before mixing and calculating scf error
            VectorShift(pSPARC->Veff_loc_dmcomm_phi, NspinDMnd, -veff_mean, pSPARC->dmcomm_phi);
            pSPARC->veff_mean = veff_mean;
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

        // stop the loop if SCF error is smaller than tolerance
        // if (dEtot < pSPARC->TOL_SCF && dEband < pSPARC->TOL_SCF) break;

        // STOP scf if scf error is less than the given tolerance
        if (scf_conv && SCFcount >= pSPARC->MINIT_SCF-1) {
            SCFcount++;
            t_scf_e = MPI_Wtime();
			#ifdef DEBUG
            if (!rank) 
            	printf("\nThis SCF took %.3f ms, scf error = %.3e\n", 
            	       (t_scf_e-t_scf_s)*1e3, error);
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
                else
                    fprintf(output_fp,"%-6d      %18.10E        %11.4E        %.3E        %.3f\n", 
                            SCFcount, pSPARC->Etot/pSPARC->n_atom, pSPARC->netM, error, t_cum_scf);
                fclose(output_fp);
                t_cum_scf = 0.0;
            }
            break;
        }

        // Apply mixing and preconditioner, if required
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif

        Mixing(pSPARC, SCFcount);

        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank) printf("rank = %d, Mixing (+ precond) took %.3f ms\n", rank, (t2 - t1) * 1e3);
        #endif

        if (pSPARC->MixingVariable == 1 && pSPARC->BC == 2) { // potential mixing, add veff_mean back
            // shift the next input veff so that it's integral is
            // equal to that of the current output veff for periodic systems
            // shift Veff by +mean(Veff)
            VectorShift(pSPARC->Veff_loc_dmcomm_phi, NspinDMnd, veff_mean, pSPARC->dmcomm_phi);
        } else if (pSPARC->MixingVariable == 0) { // recalculate potential for density mixing 
            // solve the poisson equation for electrostatic potential, "phi"
            Calculate_elecstPotential(pSPARC);
            Calculate_Vxc(pSPARC);
            Calculate_Veff_loc_dmcomm_phi(pSPARC);
        }

        if (pSPARC->SQFlag == 1) {
            for (i = 0; i < pSPARC->Nspin; i++)
                TransferVeff_phi2sq(pSPARC, pSPARC->Veff_loc_dmcomm_phi, pSPARC->pSQ->Veff_loc_SQ);
        } else {
            // transfer Veff_loc from "phi-domain" to "psi-domain"
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif

            for (i = 0; i < pSPARC->Nspin; i++)
                Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
            
            #ifdef DEBUG
            t2 = MPI_Wtime();
            if(!rank) 
                printf("rank = %d, Transfering Veff from phi-domain to psi-domain took %.3f ms\n", 
                    rank, (t2 - t1) * 1e3);
            #endif  
        }

		#ifdef DEBUG
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
            else
                fprintf(output_fp,"%-6d      %18.10E        %11.4E        %.3E        %.3f\n", 
                            SCFcount, pSPARC->Etot/pSPARC->n_atom, pSPARC->netM, error, t_cum_scf);
            fclose(output_fp);
            t_cum_scf = 0.0;
        }
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

    // Calculate actual energy for density mixing
    if (pSPARC->MixingVariable == 0) {
        // store/update input Veff for density mixing
        if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
            for (i = 0; i < NspinDMnd; i++) 
                pSPARC->Veff_loc_dmcomm_phi_in[i] = pSPARC->Veff_loc_dmcomm_phi[i];
        }
        
        // update potential
        Calculate_elecstPotential(pSPARC);
        Calculate_Vxc(pSPARC);
        Calculate_Veff_loc_dmcomm_phi(pSPARC);
        
        // here potential is consistent with density
        Calculate_Free_Energy(pSPARC, pSPARC->electronDens); 
        double Escc = Calculate_Escc(
            pSPARC, NspinDMnd, pSPARC->Veff_loc_dmcomm_phi, 
            pSPARC->Veff_loc_dmcomm_phi_in, pSPARC->electronDens + sindx_rho,
            pSPARC->dmcomm_phi
        );
        pSPARC->Escc = Escc;
        pSPARC->Etot = pSPARC->Etot + Escc;
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
            for (k = 1; k < pSQ->Nd_d_SQ; k++) {
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
                                (3.0-pSPARC->Nspin)/pSPARC->Nspinor * pSPARC->occ_sorted[spn_i*Ns*Nk + k*Ns + Ns - nocc_print + i]);
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
    double error, sbuf[2], rbuf[2], temp;
    int i, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int DMnd = pSPARC->Nd_d;
    int NspinDMnd = pSPARC->Nspin * DMnd;
    int sindx_rho = (pSPARC->Nspin == 2) ? DMnd : 0;
    
    error = 0.0;
    sbuf[0] = sbuf[1] = 0.0;    
    rbuf[0] = rbuf[1] = 0.0;

    if (pSPARC->MixingVariable == 0) {        // density mixing
        for (i = 0; i < NspinDMnd; i++) {
            temp     = pSPARC->electronDens[sindx_rho + i] - pSPARC->mixing_hist_xk[i];
            sbuf[0] += pSPARC->electronDens[sindx_rho + i] * pSPARC->electronDens[sindx_rho + i];
            sbuf[1] += temp * temp;
        }
    } else if (pSPARC->MixingVariable == 1) { // potential mixing 
        for (i = 0; i < NspinDMnd; i++) {
            //temp = (pSPARC->Veff_loc_dmcomm_phi[i] - veff_mean) - pSPARC->mixing_hist_xk[i];
            temp     = pSPARC->Veff_loc_dmcomm_phi[i] - pSPARC->mixing_hist_xk[i];
            // TODO: should we calculate the norm of the shifted v_out?
            sbuf[0] += pSPARC->Veff_loc_dmcomm_phi[i] * pSPARC->Veff_loc_dmcomm_phi[i];
            sbuf[1] += temp * temp;
        }
    } else {
        if (!rank) {
            printf("Cannot recogonize mixing variable option %d\n", pSPARC->MixingVariable);
            exit(EXIT_FAILURE);
        }
    }

    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        // MPI_Allreduce(sbuf, rbuf, 2, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
        MPI_Reduce(sbuf, rbuf, 2, MPI_DOUBLE, MPI_SUM, 0, pSPARC->dmcomm_phi);
        error = sqrt(rbuf[1] / rbuf[0]);
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
        MPI_Reduce(&loc_err, &error, 1, MPI_DOUBLE, MPI_SUM, 0, pSPARC->dmcomm_phi);
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

    double int_rhoup = 0.0, int_rhodn = 0.0;
    double spn_int[2], spn_sum[2] = {0.0,0.0};
    int DMnd = pSPARC->Nd_d, i;
    
    for (i = 0; i < DMnd; i++) {
        int_rhoup += pSPARC->electronDens[DMnd+i];
        int_rhodn += pSPARC->electronDens[2*DMnd+i];
    }

    int_rhoup *= pSPARC->dV;
    int_rhodn *= pSPARC->dV;
    
    spn_int[0] = int_rhoup; spn_int[1] = int_rhodn;
    MPI_Allreduce(spn_int, spn_sum, 2, MPI_DOUBLE,
                  MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Nelectron_up = spn_sum[0];
    pSPARC->Nelectron_dn = spn_sum[1];         
    pSPARC->netM = spn_sum[0] - spn_sum[1];
}    

/**
 * @brief   Calculate Veff_loc = phi + Vxc (in phi-domain).
 */
void Calculate_Veff_loc_dmcomm_phi(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    unsigned i, spn_i, sindx;
    for (spn_i = 0; spn_i < pSPARC->Nspin; spn_i++) {
        sindx = spn_i * pSPARC->Nd_d;
        for (i = 0; i < pSPARC->Nd_d; i++) {
            pSPARC->Veff_loc_dmcomm_phi[sindx + i] = pSPARC->XCPotential[sindx + i] + pSPARC->elecstPotential[i];
        }
    }
}



/**
 * @brief   Update mixing_hist_xk.
 */
// TODO: check if this function is necessary!
void Update_mixing_hist_xk(SPARC_OBJ *pSPARC, double veff_mean) 
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    unsigned i;
    if (pSPARC->spin_typ == 0) {
        if (pSPARC->MixingVariable == 0) {        // density mixing
            for (i = 0; i < pSPARC->Nd_d; i++) {
                pSPARC->mixing_hist_xk[i] = pSPARC->mixing_hist_xkm1[i]
                                          = pSPARC->electronDens[i];
            }
        } else if (pSPARC->MixingVariable == 1) { // potential mixing
            for (i = 0; i < pSPARC->Nd_d; i++) {
                pSPARC->mixing_hist_xk[i] = pSPARC->mixing_hist_xkm1[i]
                                          = pSPARC->Veff_loc_dmcomm_phi[i] - veff_mean;
            }
        }
    } else {
        if (pSPARC->MixingVariable == 0) {        // density mixing
            for (i = 0; i < 2 * pSPARC->Nd_d; i++) {
                pSPARC->mixing_hist_xk[i] = pSPARC->mixing_hist_xkm1[i]
                                          = pSPARC->electronDens[pSPARC->Nd_d + i];
            }
        } else if (pSPARC->MixingVariable == 1) { // potential mixing
            for (i = 0; i < 2 * pSPARC->Nd_d; i++) {
                pSPARC->mixing_hist_xk[i] = pSPARC->mixing_hist_xkm1[i]
                                          = pSPARC->Veff_loc_dmcomm_phi[i] - veff_mean;
            }
        }
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
        rdims, MPI_COMM_WORLD);
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
        pSPARC->dmcomm_phi, rdims, MPI_COMM_WORLD);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, D2D took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}
