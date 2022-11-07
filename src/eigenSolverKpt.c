/**
 * @file    eigenSolverKpt.c
 * @brief   This file contains the functions for eigenSolver.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #define MKL_Complex16 double complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/* ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif

#include "eigenSolverKpt.h"
#include "eigenSolver.h"
#include "tools.h" 
#include "linearSolver.h" // Lanczos
#include "lapVecRoutinesKpt.h"
#include "hamiltonianVecRoutines.h"
#include "occupation.h"
#include "isddft.h"
#include "parallelization.h"
#include "linearAlgebra.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
int CheFSI_use_EVA_Kpt = -1;
#endif

// static int SCFcount_;

/*
 @ brief: Main function of Chebyshev filtering 
*/

void eigSolve_CheFSI_kpt(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error) {
    // Set up for CheFSI function
    if(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return;
    
    int count, kpt, Ncheb = 1; // do Ncheb times Chebyshev filtering for all scfs > 0
    pSPARC->Ncheb = (pSPARC->elecgs_Count == 0 && SCFcount > 0) ? Ncheb : 1;
    double lambda_cutoff = 0;
    double complex *x0;
    
    // SCFcount_ = SCFcount;

    int spn_i;

    // this is for varying chebyshev polynomial degrees 
    //int npl_min, npl_max, count_chebfilt, count_cheb_npl;
    double log_TOL, log_err0;
    
    log_TOL = log(pSPARC->TOL_SCF);
    log_err0 = 0.0;
    //count_chebfilt = 0;
    //count_cheb_npl = 0; // this is to find sum of chebyshev degrees over all scf iterations
    
    // TODO: Change to (elecgs_Count > 0) once the previous electron density is used during restart
    if(pSPARC->elecgs_Count == 0 && SCFcount == 0){
        // pSPARC->eigmin = (double *) malloc(pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * sizeof (double));
        // pSPARC->eigmax = (double *) malloc(pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * sizeof (double));
    }
    if(pSPARC->elecgs_Count > 0 || pSPARC->usefock > 1) 
        pSPARC->rhoTrigger = 1;
    
#ifdef DEBUG
    double t1, t2;
#endif
    x0 = pSPARC->Lanczos_x0_complex;
    
    if(SCFcount == 0){
        pSPARC->npl_max = pSPARC->ChebDegree;
        pSPARC->npl_min = max(pSPARC->ChebDegree / 4, 12);
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        // set up initial guess for Lanczos
        if (pSPARC->kptcomm_topo != MPI_COMM_NULL)
            SetRandMat_complex(x0, pSPARC->Nd_d_kptcomm*pSPARC->Nspinor, 1, 0.0, 1.0, pSPARC->kptcomm_topo); // TODO: change for FixRandSeed = 1
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("\nTime for setting up initial guess for Lanczos: %.3f ms\n", (t2-t1)*1e3);
#endif
        count = 0;
    } else if (SCFcount == 1)
    {   
        count = pSPARC->rhoTrigger;
    } else
    {
        count = pSPARC->rhoTrigger + (SCFcount-1) * pSPARC->Ncheb;
    }   
                
    
    while(count < pSPARC->rhoTrigger + SCFcount*pSPARC->Ncheb){
        
        for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            // each kpt group take care of the kpts assigned to it
            for (kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
                // perform CheFSI algorithm, including
                // 1) Find Chebyshev filtering bounds 
                // 2) Chebyshev filtering,          3) Projection, 
                // 4) Solve projected eigenproblem, 5) Subspace rotation
                CheFSI_kpt(pSPARC, lambda_cutoff, x0, count, kpt, spn_i);
            }
        }
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        
        // ** calculate fermi energy ** //
        //if(pSPARC->kptcomm_index < 0) return;
               
        // find global minimum and global maximum eigenvalue
        double eigmin_g = pSPARC->lambda_sorted[0];
        double eigmax_g = pSPARC->lambda_sorted[pSPARC->Nstates-1];
        int spn_disp;
        for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            spn_disp = spn_i*pSPARC->Nkpts_kptcomm*pSPARC->Nstates;
            for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++){
                if(pSPARC->lambda_sorted[spn_disp + kpt*pSPARC->Nstates] < eigmin_g)
                    eigmin_g = pSPARC->lambda_sorted[spn_disp + kpt*pSPARC->Nstates];
                if(pSPARC->lambda_sorted[spn_disp + (kpt+1)*pSPARC->Nstates-1] > eigmax_g)
                    eigmax_g = pSPARC->lambda_sorted[spn_disp + (kpt+1)*pSPARC->Nstates-1];
            }
        }
        
        if (pSPARC->npspin != 1) { // find min/max over processes with the same rank in spincomm
            MPI_Allreduce(MPI_IN_PLACE, &eigmin_g, 1, MPI_DOUBLE, MPI_MIN, pSPARC->spin_bridge_comm);
            MPI_Allreduce(MPI_IN_PLACE, &eigmax_g, 1, MPI_DOUBLE, MPI_MAX, pSPARC->spin_bridge_comm);
        }
        
        if (pSPARC->npkpt != 1) { // find min/max over processes with the same rank in kptcomm to find g
            MPI_Allreduce(MPI_IN_PLACE, &eigmin_g, 1, MPI_DOUBLE, MPI_MIN, pSPARC->kpt_bridge_comm);
            MPI_Allreduce(MPI_IN_PLACE, &eigmax_g, 1, MPI_DOUBLE, MPI_MAX, pSPARC->kpt_bridge_comm);
        }
        
        pSPARC->Efermi = Calculate_occupation(pSPARC, eigmin_g - 1, eigmax_g + 1, 1e-12, 100); 
        
        // check occupation (if Nstates is large enough) for every SCF
        // for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        //     spn_disp = spn_i*pSPARC->Nkpts_kptcomm*pSPARC->Nstates;
        //     for (kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
        //         // check if occ(0.90*Ns) <= 1e-6, otherwise give warning
        //         int ind = round(pSPARC->Nstates * 0.90) - 1;
        //         ind = max(ind,0);
        //         double g_ind = pSPARC->occ_sorted[spn_disp + kpt*pSPARC->Nstates + ind];
        //         if (fabs((3.0-pSPARC->Nspin) * g_ind) > 1e-6) {
        //             if(!rank) {
        //                 printf("WARNING: lowest occupation not small enough, consider increasing number of states!\n"
        //                        "    occ(%d) = %.15f\n", ind+1,
        //                        (3.0-pSPARC->Nspin) * g_ind);
        //                 // write to .out file
        //                 output_fp = fopen(pSPARC->OutFilename,"a");
        //                 if (output_fp == NULL) {
        //                     printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
        //                     exit(EXIT_FAILURE);
        //                 }
        //                 fprintf(output_fp,
        //                        "WARNING: lowest occupation not small enough, consider increasing number of states!\n"
        //                        "    occ(%d) = %.15f\n", ind+1,
        //                        (3.0-pSPARC->Nspin) * g_ind);
        //                 fclose(output_fp);
        //             }
        //         }
        //     }
        // }

#ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) {
            printf("rank = %d, Efermi = %16.12f"
                   " calculate fermi energy took %.3f ms\n", 
                   rank, pSPARC->Efermi, (t2-t1)*1e3);
        }
#endif
        
        count++;
    }    

    // adjust chebyshev polynomial degree if CheFSI optimization is turned on
    if (pSPARC->CheFSI_Optmz) {
        //if (max(dEtot,dEband) > 1e-2) {
        if (error > 1e-2) {
            pSPARC->ChebDegree = pSPARC->npl_max;
            //log_err0 = log(dEband);
#ifdef DEBUG
            if(!rank) 
                printf("************************************************************************\n"
                       "Original Chebyshev polyn degree: %d, for next SCF chebdegree: %d\n"
                       "************************************************************************\n", 
                       pSPARC->npl_max, pSPARC->ChebDegree);
#endif
        } else {
            pSPARC->ChebDegree = pSPARC->npl_min + (int)((pSPARC->npl_max - pSPARC->npl_min)/(log_err0 - log_TOL) * (log(error) - log_TOL));
            pSPARC->ChebDegree = min(pSPARC->npl_max, pSPARC->ChebDegree);
            pSPARC->ChebDegree = max(pSPARC->npl_min, pSPARC->ChebDegree);
#ifdef DEBUG
            if(!rank) printf("************************************************************************\n"
                             "Original Chebyshev polyn degree: %d, for next SCF chebdegree: %d\n"
                             "************************************************************************\n",
                             pSPARC->npl_max, pSPARC->ChebDegree);
#endif
        }
    }
}



/**
 * @brief   Apply Chebyshev-filtered subspace iteration steps.
 */
void CheFSI_kpt(SPARC_OBJ *pSPARC, double lambda_cutoff, double complex *x0, int count, int kpt, int spn_i)
{
    int rank, rank_spincomm, nproc_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    
    // determine the bounds for performing chebyshev filtering
    Chebyshevfilter_constants_kpt(pSPARC, x0, &lambda_cutoff, &pSPARC->eigmin[spn_i*pSPARC->Nkpts_kptcomm + kpt], &pSPARC->eigmax[spn_i*pSPARC->Nkpts_kptcomm + kpt], count, kpt, spn_i);

#ifdef DEBUG
            if (!rank && kpt == 0) {
                printf("\n Chebfilt %d, in Chebyshev filtering, lambda_cutoff = %f,"
                       " lowerbound = %f, upperbound = %f\n\n",
                       count+1, lambda_cutoff, pSPARC->eigmin[spn_i*pSPARC->Nkpts_kptcomm + kpt], pSPARC->eigmax[spn_i*pSPARC->Nkpts_kptcomm + kpt]);
            }
#endif

    #ifdef DEBUG
    double t1, t2, t3;
    #endif
    double t_temp;
    int DMnd, size_k, size_s;
    DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    size_k = DMnd * pSPARC->Nband_bandcomm;
    size_s = size_k * pSPARC->Nkpts_kptcomm;

    // ** Chebyshev filtering ** //
    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    ChebyshevFiltering_kpt(pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s, 
                   pSPARC->Yorb_kpt, pSPARC->Nband_bandcomm, 
                   pSPARC->ChebDegree, lambda_cutoff, pSPARC->eigmax[spn_i*pSPARC->Nkpts_kptcomm + kpt], pSPARC->eigmin[spn_i*pSPARC->Nkpts_kptcomm + kpt], kpt, spn_i,
                   pSPARC->dmcomm, &t_temp);
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && kpt == 0) 
        printf("Total time for Chebyshev filtering (%d columns, degree = %d): %.3f ms\n", 
                pSPARC->Nband_bandcomm, pSPARC->ChebDegree, (t2-t1)*1e3);
    #endif
    
    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    // ** calculate projected Hamiltonian and overlap matrix ** //
    #ifdef USE_DP_SUBEIG
    DP_Project_Hamiltonian_kpt(
        pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Yorb_kpt, 
        pSPARC->Hp_kpt, pSPARC->Mp_kpt, spn_i, kpt
    );
    #else
    // allocate memory for block cyclic format of the wavefunction
    if (pSPARC->npband > 1) {
        pSPARC->Yorb_BLCYC_kpt = (double complex *)malloc(
            pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double complex));
        assert(pSPARC->Yorb_BLCYC_kpt != NULL);
    }
    Project_Hamiltonian_kpt(pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Yorb_kpt, 
                        pSPARC->Hp_kpt, pSPARC->Mp_kpt, kpt, spn_i, pSPARC->dmcomm);
    #endif
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && kpt == 0) printf("Total time for projection: %.3f ms\n", (t2-t1)*1e3);
    t1 = MPI_Wtime();
    #endif
    
    // ** solve the generalized eigenvalue problem Hp * Q = Mp * Q * Lambda **//
    #ifdef USE_DP_SUBEIG
    DP_Solve_Generalized_EigenProblem_kpt(pSPARC, kpt, spn_i);
    #else
    Solve_Generalized_EigenProblem_kpt(pSPARC, kpt, spn_i);
    #endif
    
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif

    // if eigvals are calculated in root process, then bcast the eigvals
    if (pSPARC->useLAPACK == 1 && nproc_kptcomm > 1) {
        MPI_Bcast(pSPARC->lambda, pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, 
                  MPI_DOUBLE, 0, pSPARC->kptcomm); // TODO: bcast in blacscomm if possible
    }
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank_spincomm && spn_i == 0 && kpt == 0) { 
        printf("==generalized eigenproblem: bcast eigvals took %.3f ms\n", (t2-t3)*1e3);
        printf("Total time for solving generalized eigenvalue problem: %.3f ms\n", 
                (t2-t1)*1e3);
    }
    t1 = MPI_Wtime();
    #endif
    
    // ** subspace rotation ** //
    #ifdef USE_DP_SUBEIG
    DP_Subspace_Rotation_kpt(pSPARC, pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s);
    #else
    double complex *YQ_BLCYC;
    if (pSPARC->npband > 1) {
        YQ_BLCYC = (double complex *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double complex));
        assert(YQ_BLCYC != NULL);
    } else {
        YQ_BLCYC = pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s;
    }
    // ScaLAPACK stores the eigenvectors in Q
    Subspace_Rotation_kpt(pSPARC, pSPARC->Yorb_BLCYC_kpt, pSPARC->Q_kpt, 
                      YQ_BLCYC, pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s, kpt, spn_i);
    
    if (pSPARC->npband > 1) {
        free(YQ_BLCYC);
        free(pSPARC->Yorb_BLCYC_kpt);
        pSPARC->Yorb_BLCYC_kpt = NULL;
    }
    #endif
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && kpt == 0) printf("Total time for subspace rotation: %.3f ms\n", (t2-t1)*1e3);
    #endif
}



/**
 * @brief   Find Chebyshev filtering bounds and cutoff constants.
 */
void Chebyshevfilter_constants_kpt(
    SPARC_OBJ *pSPARC, double complex *x0, double *lambda_cutoff, double *eigmin, 
    double *eigmax, int count, int kpt, int spn_i
)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    #ifdef DEBUG
    double t1, t2;
    #endif
    double temp;
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx;         
    sdims[1] = pSPARC->npNdy;         
    sdims[2] = pSPARC->npNdz; 
    rdims[0] = pSPARC->npNdx_kptcomm; 
    rdims[1] = pSPARC->npNdy_kptcomm; 
    rdims[2] = pSPARC->npNdz_kptcomm;
    //size_k = pSPARC->Nd_d_dmcomm * pSPARC->Nband_bandcomm;
    int sg  = pSPARC->spin_start_indx + spn_i;

    // ** find smallest and largest eigenvalue of the Hamiltonian ** //
    if (count == 0) 
    {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        if (pSPARC->chefsibound_flag == 0 || pSPARC->chefsibound_flag == 1) { // 0 - default, 1 - always call Lanczos on H
            // estimate both max and min eigenval of H using Lanczos
            D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
                sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm);
            // If exchange-correlation is SCAN, GGA_PBE will be the exc used in 1st SCF; it is unnecessary to transform a zero vector
            Lanczos_kpt(pSPARC, pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, 
                    pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->nlocProj_kptcomm, 
                    eigmin, eigmax, x0, pSPARC->TOL_LANCZOS, pSPARC->TOL_LANCZOS, 
                    1000, kpt, spn_i, pSPARC->kptcomm_topo, &pSPARC->req_veff_loc);
            *eigmax *= 1.01; // add 1% buffer
        } else {
            double eigmin_lap;
            // find min eigval of Lap, estimate eigmax of H by -0.5 * eigmin_lap   
            Lanczos_laplacian_kpt(pSPARC, pSPARC->DMVertices_kptcomm, &eigmin_lap, &temp,
                    x0, pSPARC->TOL_LANCZOS, 1e10, 1000, kpt, spn_i, pSPARC->kptcomm_topo); 
            *eigmax = -0.5 * eigmin_lap;
            *eigmax *= 1.01; // add 1% buffer
            *eigmin = -2.0; // TODO: tune this value
        }

        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0 && kpt == 0) {
            printf("rank = %3d, Lanczos took %.3f ms, eigmin = %.12f, eigmax = %.12f\n", 
                   rank, (t2-t1)*1e3, *eigmin, *eigmax);
        }
        #endif

        *eigmin -= 0.1; // for safety

    } else if (count >= pSPARC->rhoTrigger) {
        *eigmin = pSPARC->lambda_sorted[spn_i*pSPARC->Nstates*pSPARC->Nkpts_kptcomm + kpt*pSPARC->Nstates]; // take previous eigmin
        
        if (pSPARC->chefsibound_flag == 1 || ((count == pSPARC->rhoTrigger) && (strcmpi(pSPARC->XC, "SCAN") == 0))) { // 1 - always call Lanczos on H; the other condition is for SCAN: 
        //the first SCF is PBE, the second is SCAN, so it is necessary to do Lanczos again in 2nd SCF
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif
            // estimate both max eigenval of H using Lanczos
            D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
                sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm);
            if (strcmpi(pSPARC->XC, "SCAN") == 0) { // transfer vxcMGGA3 of this spin to kptcomm, it is moved from file mgga/mgga.c to here.
                // printf("rank %d, joined SCAN Lanczos, pSPARC->countPotentialCalculate %d\n", rank, pSPARC->countPotentialCalculate);
                D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, 
                pSPARC->DMVertices_dmcomm, pSPARC->vxcMGGA3_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                pSPARC->DMVertices_kptcomm, pSPARC->vxcMGGA3_loc_kptcomm, 
                pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
                sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm);
            }
            Lanczos_kpt(pSPARC, pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, 
                    pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->nlocProj_kptcomm, 
                    &temp, eigmax, x0, 1e10, pSPARC->TOL_LANCZOS, 
                    1000, kpt, spn_i, pSPARC->kptcomm_topo, &pSPARC->req_veff_loc);
            *eigmax *= 1.01; // add 1% buffer
            #ifdef DEBUG
            t2 = MPI_Wtime();
            if (rank == 0 && kpt == 0) {
                printf("rank = %3d, Lanczos took %.3f ms, eigmin = %.12f, eigmax = %.12f\n", 
                   rank, (t2-t1)*1e3, *eigmin, *eigmax);
            }
            #endif
        } 
    }
    
    if (pSPARC->elecgs_Count == 0 && count == 0) 
        *lambda_cutoff = 0.5 * (*eigmin + *eigmax);
    else{
        //*lambda_cutoff = pSPARC->Efermi + log(1e6-1) / pSPARC->Beta + 0.1;
        *lambda_cutoff = pSPARC->lambda_sorted[spn_i*pSPARC->Nstates*pSPARC->Nkpts_kptcomm + (kpt+1)*pSPARC->Nstates-1] + 0.1;
    }
}



/**
 * @brief   Perform Chebyshev filtering.
 */
void ChebyshevFiltering_kpt(
    SPARC_OBJ *pSPARC, int *DMVertices, double complex *X, double complex *Y, int ncol, 
    int m, double a, double b, double a0, int kpt, int spn_i, MPI_Comm comm, 
    double *time_info
) 
{   
    if (comm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;
    // a0: minimum eigval, b: maxinum eigval, a: cutoff eigval
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    #ifdef DEBUG   
    if(!rank && kpt == 0) printf("Start Chebyshev filtering ... \n");
    #endif

    double t1, t2;
    *time_info = 0.0;

    double e, c, sigma, sigma1, sigma2, gamma, vscal, vscal2;
    double complex *Ynew;
    int i, j, DMnd, len_tot;
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);
    DMnd *= pSPARC->Nspinor;   
    len_tot = DMnd * ncol;    
    e = 0.5 * (b - a);
    c = 0.5 * (b + a);
    sigma = sigma1 = e / (a0 - c);
    gamma = 2.0 / sigma1;

    t1 = MPI_Wtime();
    // find Y = (H - c*I)X
    int sg  = pSPARC->spin_start_indx + spn_i;
    Hamiltonian_vectors_mult_kpt(
        pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, -c, X, Y, spn_i, kpt, comm
    );
    t2 = MPI_Wtime();
    *time_info += t2 - t1;
        
    // scale Y by (sigma1 / e)
    vscal = sigma1 / e;
    for (i = 0; i < len_tot; i++) Y[i] *= vscal;
      
    Ynew = (double complex *)malloc( len_tot * sizeof(double complex));

    for (j = 1; j < m; j++) {
        sigma2 = 1.0 / (gamma - sigma);
        
        t1 = MPI_Wtime();
        // Ynew = (H - c*I)Y
        Hamiltonian_vectors_mult_kpt(
            pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
            pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, -c, Y, Ynew, spn_i, kpt, comm
        );
        t2 = MPI_Wtime();
        *time_info += t2 - t1;
        
        // Ynew = (2*sigma2/e) * Ynew - (sigma*sigma2) * X, then update X and Y
        vscal = 2.0 * sigma2 / e; vscal2 = sigma * sigma2;

        for (i = 0; i < len_tot; i++) {
            //Ynew[i] = vscal * Ynew[i] - vscal2 * X[i];
            Ynew[i] *= vscal;
            Ynew[i] -= vscal2 * X[i];
            X[i] = Y[i];
            Y[i] = Ynew[i];
        }        
        sigma = sigma2;
    } 
    free(Ynew);
}

#ifdef USE_DP_SUBEIG
struct DP_CheFSI_kpt_s
{
    int nproc_row;                // Number of processes in process row, == comm size of pSPARC->blacscomm
    int nproc_kpt;                // Number of processes in kpt_comm 
    int rank_row;                 // Rank of this process in process row, == rank in pSPARC->blacscomm
    int rank_kpt;                 // Rank of this process in kpt_comm;
    int Ns_bp;                    // Number of bands this process has in the original band parallelization (BP), 
                                  // == number of local states (bands) in SPARC == pSPARC->{band_end_indx-band_start_indx} + 1
    int Ns_dp;                    // Number of bands this process has in the converted domain parallelization (DP),
                                  // == number of total states (bands) in SPARC == pSPARC->Nstates
    int Nd_bp;                    // Number of FD points this process has in the original band parallelization (BP), == pSPARC->Nd_d_dmcomm
    int Nd_dp;                    // Number of FD points this process has after converted to domain parallelization (DP)
    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int desc_Hp_local[9];         // descriptor for Hp_local on each ictxt_blacs_topo
    int desc_Mp_local[9];         // descriptor for Mp_local on each ictxt_blacs_topo
    int desc_eig_vecs[9];         // descriptor for eig_vecs on each ictxt_blacs_topo
    #endif
    int *Ns_bp_displs;            // Size nproc_row+1, the pSPARC->band_start_indx on each process in pSPARC->blacscomm
    int *Nd_dp_displs;            // Size nproc_row+1, displacements of FD points for each process in DP
    int *bp2dp_sendcnts;          // BP to DP send counts
    int *bp2dp_sdispls;           // BP to DP displacements
    int *dp2bp_sendcnts;          // DP to BP send counts
    int *dp2bp_sdispls;           // DP to BP send displacements
    double _Complex *Y_packbuf;   // Y pack buffer
    double _Complex *HY_packbuf;  // HY pack buffer
    double _Complex *Y_dp;        // Y block in DP
    double _Complex *HY_dp;       // HY block in DP
    double _Complex *Mp_local;    // Local Mp result
    double _Complex *Hp_local;    // Local Hp result
    double _Complex *eig_vecs;    // Eigen vectors from solving generalized eigenproblem
    MPI_Comm kpt_comm;            // MPI communicator that contains all active processes in pSPARC->kptcomm
};
typedef struct DP_CheFSI_kpt_s* DP_CheFSI_kpt_t;

static int calc_block_spos(const int len, const int nblk, const int iblk)
{
	if (iblk < 0 || iblk > nblk) return -1;
	int rem = len % nblk;
	int bs0 = len / nblk;
	int bs1 = bs0 + 1;
	if (iblk < rem) return (bs1 * iblk);
    else return (bs0 * iblk + rem);
}

/**
 * @brief   Initialize domain parallelization data structures for calculating projected Hamiltonian,  
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI_kpt().
 */
void init_DP_CheFSI_kpt(SPARC_OBJ *pSPARC)
{
    int proc_active = (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) ? 0 : 1;
    
    DP_CheFSI_kpt_t DP_CheFSI_kpt = (DP_CheFSI_kpt_t) malloc(sizeof(struct DP_CheFSI_kpt_s));
    
    // Split the kpt_comm for all active processes in pSPARC->kptcomm
    int nproc_kpt, rank_kpt;
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kpt);
    MPI_Comm_split(pSPARC->kptcomm, proc_active, rank_kpt, &DP_CheFSI_kpt->kpt_comm);
    if (proc_active == 0)
    {
        free(DP_CheFSI_kpt);
        MPI_Comm_free(&DP_CheFSI_kpt->kpt_comm);
        pSPARC->DP_CheFSI_kpt = NULL;
        return;
    } else {
        MPI_Comm_size(DP_CheFSI_kpt->kpt_comm, &nproc_kpt);
        MPI_Comm_rank(DP_CheFSI_kpt->kpt_comm, &rank_kpt);
    }
    
    // Get the original band parallelization parameters
    int nproc_row, rank_row;
    MPI_Comm_size(pSPARC->blacscomm, &nproc_row);
    MPI_Comm_rank(pSPARC->blacscomm, &rank_row);
    int Ns_bp = pSPARC->band_end_indx - pSPARC->band_start_indx + 1;
    int Ns_dp = pSPARC->Nstates;
    int Nd_bp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    
    // The number of bands on each process could be different, we need to gather them
    int *Ns_bp_displs = (int*) malloc(sizeof(int) * (nproc_row + 1));
    assert(Ns_bp_displs != NULL);
    MPI_Allgather(
        &pSPARC->band_start_indx, 1, MPI_INT, Ns_bp_displs, 1, 
        MPI_INT, pSPARC->blacscomm
    );
    Ns_bp_displs[nproc_row] = pSPARC->Nstates;
    
    // Nd_bp is the same for all processes in blacscomm, partition it in the same way
    int *Nd_dp_displs = (int*) malloc(sizeof(int) * (nproc_row + 1));
    assert(Nd_dp_displs != NULL);
    for (int i = 0; i <= nproc_row; i++)
        Nd_dp_displs[i] = calc_block_spos(Nd_bp, nproc_row, i);
    int Nd_dp = Nd_dp_displs[rank_row + 1] - Nd_dp_displs[rank_row];
    
    // Calculate MPI_Alltoallv send and recv parameters
    int *bp2dp_sendcnts = (int*) malloc(sizeof(int) * nproc_row);
    int *bp2dp_sdispls  = (int*) malloc(sizeof(int) * nproc_row);
    int *dp2bp_sendcnts = (int*) malloc(sizeof(int) * nproc_row);
    int *dp2bp_sdispls  = (int*) malloc(sizeof(int) * nproc_row);
    assert(bp2dp_sendcnts != NULL && bp2dp_sdispls != NULL);
    assert(dp2bp_sendcnts != NULL && dp2bp_sdispls != NULL);
    bp2dp_sendcnts[0] = Ns_bp * Nd_dp_displs[1];
    dp2bp_sendcnts[0] = Nd_dp * Ns_bp_displs[1];
    bp2dp_sdispls[0]  = 0;
    dp2bp_sdispls[0]  = 0;
    for (int i = 1; i < nproc_row; i++)
    {
        bp2dp_sendcnts[i] = Ns_bp * (Nd_dp_displs[i + 1] - Nd_dp_displs[i]);
        dp2bp_sendcnts[i] = Nd_dp * (Ns_bp_displs[i + 1] - Ns_bp_displs[i]);
        bp2dp_sdispls[i]  = bp2dp_sdispls[i - 1] + bp2dp_sendcnts[i - 1];
        dp2bp_sdispls[i]  = dp2bp_sdispls[i - 1] + dp2bp_sendcnts[i - 1];
    }
    
    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int ZERO = 0, info;
    // descriptors for Hp_local, Mp_local and eig_vecs (set block size to Ns_dp)
    descinit_(DP_CheFSI_kpt->desc_Hp_local, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
        &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
    assert(info == 0);
    descinit_(DP_CheFSI_kpt->desc_Mp_local, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
        &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
    assert(info == 0);
    descinit_(DP_CheFSI_kpt->desc_eig_vecs, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
        &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
    assert(info == 0);
    #endif

    // Copy parameters and pointers to DP_CheFSI_kpt
    DP_CheFSI_kpt->nproc_row = nproc_row;
    DP_CheFSI_kpt->nproc_kpt = nproc_kpt;
    DP_CheFSI_kpt->rank_row  = rank_row;
    DP_CheFSI_kpt->rank_kpt  = rank_kpt;
    DP_CheFSI_kpt->Ns_bp     = Ns_bp;
    DP_CheFSI_kpt->Ns_dp     = Ns_dp;
    DP_CheFSI_kpt->Nd_bp     = Nd_bp;
    DP_CheFSI_kpt->Nd_dp     = Nd_dp;
    DP_CheFSI_kpt->Ns_bp_displs   = Ns_bp_displs;
    DP_CheFSI_kpt->Nd_dp_displs   = Nd_dp_displs;
    DP_CheFSI_kpt->bp2dp_sendcnts = bp2dp_sendcnts;
    DP_CheFSI_kpt->bp2dp_sdispls  = bp2dp_sdispls;
    DP_CheFSI_kpt->dp2bp_sendcnts = dp2bp_sendcnts;
    DP_CheFSI_kpt->dp2bp_sdispls  = dp2bp_sdispls;
    size_t Nd_Ns_bp_msize = sizeof(double _Complex) * Nd_bp * Ns_bp;
    size_t Nd_Ns_dp_msize = sizeof(double _Complex) * Nd_dp * Ns_dp;
    size_t Ns_dp_2_msize  = sizeof(double _Complex) * Ns_dp * Ns_dp;
    DP_CheFSI_kpt->Y_dp       = (double _Complex*) malloc(Nd_Ns_dp_msize);
    DP_CheFSI_kpt->HY_dp      = (double _Complex*) malloc(Nd_Ns_dp_msize);
    DP_CheFSI_kpt->Y_packbuf  = (double _Complex*) malloc(Nd_Ns_bp_msize);
    DP_CheFSI_kpt->HY_packbuf = (double _Complex*) malloc(Nd_Ns_bp_msize);
    DP_CheFSI_kpt->Mp_local   = (double _Complex*) malloc(Ns_dp_2_msize);
    DP_CheFSI_kpt->Hp_local   = (double _Complex*) malloc(Ns_dp_2_msize);
    DP_CheFSI_kpt->eig_vecs   = (double _Complex*) malloc(Ns_dp_2_msize);
    assert(DP_CheFSI_kpt->Y_dp != NULL && DP_CheFSI_kpt->HY_dp != NULL);
    assert(DP_CheFSI_kpt->Y_packbuf  != NULL);
    assert(DP_CheFSI_kpt->HY_packbuf != NULL);
    assert(DP_CheFSI_kpt->Mp_local   != NULL);
    assert(DP_CheFSI_kpt->Hp_local   != NULL);
    assert(DP_CheFSI_kpt->eig_vecs   != NULL);
    pSPARC->DP_CheFSI_kpt = (void*) DP_CheFSI_kpt;
}

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix with domain parallelization
 *          data partitioning. 
 *
 *          Hp = Y' * H * Y,  Mp = Y' * Y. The tall-skinny column-matrices Y and HY 
 *          are partitioned by rows only in this function. MPI_Alltoallv is used to 
 *          convert the data layout from band parallelization to domain parallelization
 *          in each original domain parallelization part (blacscomm). Then we need 2 
 *          MPI_Reduce to get the final Hp and Mp on rank 0 of each kpt_comm.
 */
void DP_Project_Hamiltonian_kpt(SPARC_OBJ *pSPARC, int *DMVertices, double complex *Y, double complex *Hp, double complex *Mp, int spn_i, int kpt)
{
    DP_CheFSI_kpt_t DP_CheFSI_kpt = (DP_CheFSI_kpt_t) pSPARC->DP_CheFSI_kpt;
    if (DP_CheFSI_kpt == NULL) return;
    
    double st, et, st0, et0;
    int rank_kpt = DP_CheFSI_kpt->rank_kpt;
    
    st0 = MPI_Wtime();
    
    // Calculate H * Y, copied from Project_Hamiltonian
    int sg = pSPARC->spin_start_indx + spn_i;
    int DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    int size_k = DMnd * pSPARC->Nband_bandcomm;
    int size_s = size_k * pSPARC->Nkpts_kptcomm;
    double *Veff_loc_sg = pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm;
    double complex *HY = pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s;
    st = MPI_Wtime();
    Hamiltonian_vectors_mult_kpt(
        pSPARC, DMnd, DMVertices, 
        Veff_loc_sg, pSPARC->Atom_Influence_nloc, 
        pSPARC->nlocProj, pSPARC->Nband_bandcomm, 
        0.0, Y, HY, spn_i, kpt, pSPARC->dmcomm
    );
    
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_kpt, rank 0, calc HY used %.3lf ms\n", 1000.0 * (et - st));
    #endif
    
    // Use MPI_Alltoallv to convert from band parallelization to domain parallelization
    // Note: Y and HY are stored in column major!
    st = MPI_Wtime();
    double _Complex *Y_dp  = DP_CheFSI_kpt->Y_dp;
    double _Complex *HY_dp = DP_CheFSI_kpt->HY_dp;
    BP2DP(
        pSPARC->blacscomm, DP_CheFSI_kpt->nproc_row,
        DP_CheFSI_kpt->Nd_bp, DP_CheFSI_kpt->Ns_bp, DP_CheFSI_kpt->Nd_dp_displs,
        DP_CheFSI_kpt->bp2dp_sendcnts, DP_CheFSI_kpt->bp2dp_sdispls,
        DP_CheFSI_kpt->dp2bp_sendcnts, DP_CheFSI_kpt->dp2bp_sdispls,
        sizeof(double _Complex),  Y,  DP_CheFSI_kpt->Y_packbuf,  Y_dp
    );
    BP2DP(
        pSPARC->blacscomm, DP_CheFSI_kpt->nproc_row,
        DP_CheFSI_kpt->Nd_bp, DP_CheFSI_kpt->Ns_bp, DP_CheFSI_kpt->Nd_dp_displs,
        DP_CheFSI_kpt->bp2dp_sendcnts, DP_CheFSI_kpt->bp2dp_sdispls,
        DP_CheFSI_kpt->dp2bp_sendcnts, DP_CheFSI_kpt->dp2bp_sdispls,
        sizeof(double _Complex), HY, DP_CheFSI_kpt->HY_packbuf, HY_dp
    );
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_kpt, rank 0, put local Y & HY into GTMatrix used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    // Local zgemm, Y and HY are Nd_dp-by-Ns_dp column-major matrices, we need Y^T * (H)Y
    st = MPI_Wtime();    
    int Nd_dp = DP_CheFSI_kpt->Nd_dp;
    int Ns_dp = DP_CheFSI_kpt->Ns_dp;
    double _Complex *Mp_local = DP_CheFSI_kpt->Mp_local;
    double _Complex *Hp_local = DP_CheFSI_kpt->Hp_local;
    double _Complex dc_one    = 1.0;
    double _Complex dc_zero   = 0.0;
    cblas_zgemm(
        CblasColMajor, CblasConjTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        &dc_one, Y_dp, Nd_dp, Y_dp, Nd_dp, 
        &dc_zero, Mp_local, Ns_dp
    );
    cblas_zgemm(
        CblasColMajor, CblasConjTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        &dc_one, Y_dp, Nd_dp, HY_dp, Nd_dp, 
        &dc_zero, Hp_local, Ns_dp
    );
    
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_kpt, rank 0, local zgemm for Hp & Mp used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    // Reduce to Mp & Hp
    st = MPI_Wtime();
    int Ns_dp_2 = Ns_dp * Ns_dp;
    MPI_Request req0, req1;
    MPI_Status  sta0, sta1;
    if (DP_CheFSI_kpt->rank_kpt == 0)
    {
        MPI_Ireduce(
            MPI_IN_PLACE, Mp_local, Ns_dp_2, MPI_C_DOUBLE_COMPLEX, 
            MPI_SUM, 0, DP_CheFSI_kpt->kpt_comm, &req0
        );
        MPI_Ireduce(
            MPI_IN_PLACE, Hp_local, Ns_dp_2, MPI_C_DOUBLE_COMPLEX, 
            MPI_SUM, 0, DP_CheFSI_kpt->kpt_comm, &req1
        );
    } else {
        MPI_Ireduce(
            Mp_local, NULL, Ns_dp_2, MPI_C_DOUBLE_COMPLEX, 
            MPI_SUM, 0, DP_CheFSI_kpt->kpt_comm, &req0
        );
        MPI_Ireduce(
            Hp_local, NULL, Ns_dp_2, MPI_C_DOUBLE_COMPLEX, 
            MPI_SUM, 0, DP_CheFSI_kpt->kpt_comm, &req1
        );
    }
    MPI_Wait(&req0, &sta0);
    MPI_Wait(&req1, &sta1);
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_kpt, rank 0, reduce for Hp & Mp used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    et0 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_kpt, rank 0, Project_Hamiltonian_DP_kpt used %.3lf ms\n", 1000.0 * (et0 - st0));
    #endif
}

/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x using domain parallelization
 *          data partitioning. 
 *
 *          Rank 0 of each kpt_comm will solve this generalized eigenproblem locally and 
 *          then broadcast the obtained eigenvectors to all other processes in this kpt_comm.
 */
void DP_Solve_Generalized_EigenProblem_kpt(SPARC_OBJ *pSPARC, int kpt, int spn_i)
{
    DP_CheFSI_kpt_t DP_CheFSI_kpt = (DP_CheFSI_kpt_t) pSPARC->DP_CheFSI_kpt;
    if (DP_CheFSI_kpt == NULL) return;
    
    if (pSPARC->useLAPACK == 1)
    {
        int Ns_dp = DP_CheFSI_kpt->Ns_dp;
        int rank_kpt = DP_CheFSI_kpt->rank_kpt;
        double _Complex *eig_vecs = DP_CheFSI_kpt->eig_vecs;
        double st = MPI_Wtime();
        if (rank_kpt == 0)
        {
            double _Complex *Hp_local = DP_CheFSI_kpt->Hp_local;
            double _Complex *Mp_local = DP_CheFSI_kpt->Mp_local;
            double *eig_val = pSPARC->lambda + kpt*pSPARC->Nstates + spn_i*pSPARC->Nkpts_kptcomm*pSPARC->Nstates;
            LAPACKE_zhegvd(
                LAPACK_COL_MAJOR, 1, 'V', 'U', Ns_dp, 
                Hp_local, Ns_dp, Mp_local, Ns_dp, eig_val
            );
            copy_mat_blk(sizeof(double _Complex), Hp_local, Ns_dp, Ns_dp, Ns_dp, eig_vecs, Ns_dp);
        }
        double et0 = MPI_Wtime();
        MPI_Bcast(eig_vecs, Ns_dp * Ns_dp, MPI_C_DOUBLE_COMPLEX, 0, DP_CheFSI_kpt->kpt_comm);
        double et1 = MPI_Wtime();
        #ifdef DEBUG
        if (rank_kpt == 0) printf("Rank 0, DP_Solve_Generalized_EigenProblem_kpt used %.3lf ms, LAPACKE_zhegvd used %.3lf ms\n", 1000.0 * (et1 - st), 1000.0 * (et0 - st));
        #endif
    } else {
        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        int rank_dmcomm = -1;
        if (pSPARC->dmcomm != MPI_COMM_NULL) 
            MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
        // Hp and Mp is only correct at the first blacscomm
        if (rank_dmcomm == 0) {
            int ONE = 1;
            // Step 1: redistribute DP_CheFSI_kpt->Hp_local and DP_CheFSI_kpt->Mp_local on rank_kpt == 0 to ScaLAPACK format 
            pzgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, DP_CheFSI_kpt->Hp_local, &ONE, &ONE, 
                      DP_CheFSI_kpt->desc_Hp_local, pSPARC->Hp_kpt, &ONE, &ONE, 
                      pSPARC->desc_Hp_BLCYC, &pSPARC->ictxt_blacs_topo);
            pzgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, DP_CheFSI_kpt->Mp_local, &ONE, &ONE, 
                      DP_CheFSI_kpt->desc_Mp_local, pSPARC->Mp_kpt, &ONE, &ONE, 
                      pSPARC->desc_Mp_BLCYC, &pSPARC->ictxt_blacs_topo);

            // Step 2: use pzsygvx_ to solve the generalized eigenproblem
            Solve_Generalized_EigenProblem_kpt(pSPARC, kpt, spn_i);
            
            // Step 3: redistribute the obtained eigenvectors from ScaLAPACK format to DP_CheFSI_kpt->eig_vecs on rank_kpt == 0
            pzgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Q_kpt, &ONE, &ONE, 
                      pSPARC->desc_Q_BLCYC, DP_CheFSI_kpt->eig_vecs, &ONE, &ONE, 
                      DP_CheFSI_kpt->desc_eig_vecs, &pSPARC->ictxt_blacs_topo);
        }

        int Ns_dp = DP_CheFSI_kpt->Ns_dp;
        MPI_Bcast(DP_CheFSI_kpt->eig_vecs, Ns_dp * Ns_dp, MPI_C_DOUBLE_COMPLEX, 0, DP_CheFSI_kpt->kpt_comm);
        
        if (pSPARC->npNd > 1 && pSPARC->bandcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
            double *eig_val = pSPARC->lambda + kpt*Ns_dp + spn_i*pSPARC->Nkpts_kptcomm*Ns_dp;
            MPI_Bcast(eig_val, Ns_dp, MPI_DOUBLE, 0, pSPARC->dmcomm);
        }

        #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        if (DP_CheFSI_kpt->rank_kpt == 0) printf("[FATAL] Subspace eigenproblem should be solved using ScaLAPACK but ScaLAPACK is not compiled\n");
        exit(255);
        #endif  // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    }
}

/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals, using domain parallelization
 *          data partitioning. 
 *
 *          This is just to perform a matrix-matrix multiplication: PsiQ = YPsi* Q, here 
 *          Psi == Y, Q == eigvec. Note that Y and YQ are in domain parallelization data 
 *          layout. We use MPI_Alltoallv to convert the obtained YQ back to the band + domain 
 *          parallelization format in SPARC and copy the transformed YQ to Psi_rot. 
 */
void DP_Subspace_Rotation_kpt(SPARC_OBJ *pSPARC, double complex *Psi_rot)
{
    DP_CheFSI_kpt_t DP_CheFSI_kpt = (DP_CheFSI_kpt_t) pSPARC->DP_CheFSI_kpt;
    if (DP_CheFSI_kpt == NULL) return;
    
    int rank_kpt = DP_CheFSI_kpt->rank_kpt;
    double st, et0, et1;
    
    // Psi == Y, Q == eig_vecs, we store Psi * Q in HY
    st = MPI_Wtime();
    int Nd_dp = DP_CheFSI_kpt->Nd_dp;
    int Ns_dp = DP_CheFSI_kpt->Ns_dp;
    double _Complex *Y_dp     = DP_CheFSI_kpt->Y_dp;
    double _Complex *YQ_dp    = DP_CheFSI_kpt->HY_dp;
    double _Complex *eig_vecs = DP_CheFSI_kpt->eig_vecs;
    double _Complex dc_one    = 1.0;
    double _Complex dc_zero   = 0.0;
    cblas_zgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        Nd_dp, Ns_dp, Ns_dp, 
        &dc_one, Y_dp, Nd_dp, eig_vecs, Ns_dp,
        &dc_zero, YQ_dp, Nd_dp
    );
    et0 = MPI_Wtime();
    
    // Redistribute Psi * Q back into band + domain format using MPI_Alltoallv
    DP2BP(
        pSPARC->blacscomm, DP_CheFSI_kpt->nproc_row,
        DP_CheFSI_kpt->Nd_bp, DP_CheFSI_kpt->Ns_bp, DP_CheFSI_kpt->Nd_dp_displs,
        DP_CheFSI_kpt->bp2dp_sendcnts, DP_CheFSI_kpt->bp2dp_sdispls,
        DP_CheFSI_kpt->dp2bp_sendcnts, DP_CheFSI_kpt->dp2bp_sdispls,
        sizeof(double _Complex), YQ_dp, DP_CheFSI_kpt->Y_packbuf, Psi_rot
    );
    // Synchronize here to prevent some processes run too fast and enter next Project_Hamiltonian_GTM too earlier
    MPI_Barrier(DP_CheFSI_kpt->kpt_comm);
    et1 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0) printf("Rank 0, DP_Subspace_Rotation_kpt used %.3lf ms, redist PsiQ used %.3lf ms\n\n", 1000.0 * (et1 - st), 1000.0 * (et1 - et0));
    #endif
}

/**
 * @brief   Free domain parallelization data structures for calculating projected Hamiltonian, 
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI_kpt().
 */
void free_DP_CheFSI_kpt(SPARC_OBJ *pSPARC)
{
    DP_CheFSI_kpt_t DP_CheFSI_kpt = (DP_CheFSI_kpt_t) pSPARC->DP_CheFSI_kpt;
    if (DP_CheFSI_kpt == NULL) return;
    
    free(DP_CheFSI_kpt->Ns_bp_displs);
    free(DP_CheFSI_kpt->Nd_dp_displs);
    free(DP_CheFSI_kpt->bp2dp_sendcnts);
    free(DP_CheFSI_kpt->bp2dp_sdispls);
    free(DP_CheFSI_kpt->dp2bp_sendcnts);
    free(DP_CheFSI_kpt->dp2bp_sdispls);
    free(DP_CheFSI_kpt->Y_packbuf);
    free(DP_CheFSI_kpt->HY_packbuf);
    free(DP_CheFSI_kpt->Y_dp);
    free(DP_CheFSI_kpt->HY_dp);
    free(DP_CheFSI_kpt->Mp_local);
    free(DP_CheFSI_kpt->Hp_local);
    free(DP_CheFSI_kpt->eig_vecs);
    MPI_Comm_free(&DP_CheFSI_kpt->kpt_comm);
    
    free(DP_CheFSI_kpt);
    pSPARC->DP_CheFSI_kpt = NULL;
}

#endif  // End of #ifdef USE_DP_SUBEIG

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix.
 *
 *          Hp = Y' * H * Y, 
 *          Mp = Y' * Y.
 */
void Project_Hamiltonian_kpt(SPARC_OBJ *pSPARC, int *DMVertices, double complex *Y, 
                         double complex *Hp, double complex *Mp, int kpt, int spn_i, MPI_Comm comm) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->bandcomm_index < 0 || comm == MPI_COMM_NULL) return;
    //if (comm == MPI_COMM_NULL) return;

    int nproc_dmcomm, rank;
    MPI_Comm_size(comm, &nproc_dmcomm);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #ifdef DEBUG
    double t1, t2, t3, t4;
    #endif

    int Nd_blacscomm;
    int sg  = pSPARC->spin_start_indx + spn_i;
    Nd_blacscomm = pSPARC->is_domain_uniform ? pSPARC->Nd : pSPARC->Nd_d_dmcomm;
    Nd_blacscomm *= pSPARC->Nspinor;

    int ONE = 1;
    double complex alpha = 1.0, beta = 0.0;
    /* Calculate Mp = Y' * Y */
    
    #ifdef DEBUG
    t3 = MPI_Wtime();

    t1 = MPI_Wtime();
    #endif
    if (pSPARC->npband > 1) {
        // distribute orbitals into block cyclic format
        pzgemr2d_(&Nd_blacscomm, &pSPARC->Nstates, Y, &ONE, &ONE, pSPARC->desc_orbitals,
                  pSPARC->Yorb_BLCYC_kpt, &ONE, &ONE, pSPARC->desc_orb_BLCYC, &pSPARC->ictxt_blacs); 
    } else {
        pSPARC->Yorb_BLCYC_kpt = Y;
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();  
    if(!rank && spn_i == 0 && kpt == 0) 
        printf("rank = %2d, Distribute orbital to block cyclic format took %.3f ms\n", 
                rank, (t2 - t1)*1e3);          
    t1 = MPI_Wtime();
    #endif
    
    if (pSPARC->npband > 1) {
        #ifdef DEBUG    
        if (!rank && spn_i == 0 && kpt == 0) printf("rank = %d, STARTING PZGEMM ...\n",rank);
        #endif    
        // perform matrix multiplication using ScaLAPACK routines
        pzgemm_("C", "N", &pSPARC->Nstates, &pSPARC->Nstates, &Nd_blacscomm, &alpha, 
                pSPARC->Yorb_BLCYC_kpt, &ONE, &ONE, pSPARC->desc_orb_BLCYC,
                pSPARC->Yorb_BLCYC_kpt, &ONE, &ONE, pSPARC->desc_orb_BLCYC, &beta, Mp, 
                &ONE, &ONE, pSPARC->desc_Mp_BLCYC);
    } else {
        #ifdef DEBUG    
        if (!rank && spn_i == 0 && kpt == 0) printf("rank = %d, STARTING ZGEMM ...\n",rank);
        #endif 
        cblas_zgemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            pSPARC->Nstates, pSPARC->Nstates, Nd_blacscomm,
            &alpha, pSPARC->Yorb_BLCYC_kpt, Nd_blacscomm, pSPARC->Yorb_BLCYC_kpt, Nd_blacscomm, 
            &beta, Mp, pSPARC->Nstates
        );
    }
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) 
        printf("rank = %2d, Psi'*Psi in block cyclic format in each blacscomm took %.3f ms\n", 
                rank, (t2 - t1)*1e3); 
    t1 = MPI_Wtime();
    #endif
    
    if (nproc_dmcomm > 1 && !pSPARC->is_domain_uniform) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Mp, pSPARC->nr_Mp_BLCYC*pSPARC->nc_Mp_BLCYC, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    #ifdef DEBUG
    t2 = MPI_Wtime();
    t4 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, Allreduce to sum Psi'*Psi over dmcomm took %.3f ms\n", 
                     rank, (t2 - t1)*1e3); 
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, Distribute data + matrix mult took %.3f ms\n", 
                     rank, (t4 - t3)*1e3);
    #endif

    /* Calculate Hp = Y' * HY */
    // first find HY
    double complex *HY_BLCYC;
    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    // save HY in Xorb
    int DMnd = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor;
    int size_k = DMnd * pSPARC->Nband_bandcomm;
    int size_s = size_k * pSPARC->Nkpts_kptcomm;
    Hamiltonian_vectors_mult_kpt(
        pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, pSPARC->Atom_Influence_nloc, 
        pSPARC->nlocProj, pSPARC->Nband_bandcomm, 0.0, Y, pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s, spn_i, kpt, pSPARC->dmcomm
    );
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, finding HY took %.3f ms\n", rank, (t2 - t1)*1e3);   
    t1 = MPI_Wtime();
    #endif    
    
    if (pSPARC->npband > 1) {
        // distribute HY
        HY_BLCYC = (double complex *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double complex));
        pzgemr2d_(&Nd_blacscomm, &pSPARC->Nstates, pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s, &ONE, &ONE, 
                  pSPARC->desc_orbitals, HY_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, 
                  &pSPARC->ictxt_blacs);
    } else {
        HY_BLCYC = pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s;
    }
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, distributing HY into block cyclic form took %.3f ms\n", 
                     rank, (t2 - t1)*1e3);  
    t1 = MPI_Wtime();
    #endif

    if (pSPARC->npband > 1) {
        // perform matrix multiplication Y' * HY using ScaLAPACK routines
        pzgemm_("C", "N", &pSPARC->Nstates, &pSPARC->Nstates, &Nd_blacscomm, &alpha, 
                pSPARC->Yorb_BLCYC_kpt, &ONE, &ONE, pSPARC->desc_orb_BLCYC, HY_BLCYC, 
                &ONE, &ONE, pSPARC->desc_orb_BLCYC, &beta, Hp, &ONE, &ONE, 
                pSPARC->desc_Hp_BLCYC);
    } else{
        cblas_zgemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            pSPARC->Nstates, pSPARC->Nstates, Nd_blacscomm,
            &alpha, pSPARC->Yorb_BLCYC_kpt, Nd_blacscomm, HY_BLCYC, Nd_blacscomm, 
            &beta, Hp, pSPARC->Nstates
        );
    }

    if (nproc_dmcomm > 1 && !pSPARC->is_domain_uniform) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Hp, pSPARC->nr_Hp_BLCYC*pSPARC->nc_Hp_BLCYC, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
    }

    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, finding Y'*HY took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif
    if (pSPARC->npband > 1) {
        free(HY_BLCYC);
    }
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}



/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x.
 *
 *          Note: Hp = Psi' * H * Psi, Mp = Psi' * Psi. Also note that both Hp and 
 *                Mp are distributed block cyclically.
 *          
 *          TODO: At some point it is better to use ELPA (https://elpa.mpcdf.mpg.de/) 
 *                for solving subspace eigenvalue problem, which can provide up to 
 *                3x speedup.
 */
void Solve_Generalized_EigenProblem_kpt(SPARC_OBJ *pSPARC, int kpt, int spn_i) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

    #ifdef DEBUG
    if (!rank && spn_i == 0 && kpt == 0) printf("Start solving generalized eigenvalue problem ...\n");
    #endif
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int nproc_dmcomm;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    #ifdef DEBUG
    double t1, t2;
    #endif

    if (pSPARC->useLAPACK == 1) {
        int info = 0;
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        if ((!pSPARC->is_domain_uniform && !pSPARC->bandcomm_index) ||
            (pSPARC->is_domain_uniform && !rank_kptcomm)) {
            info = LAPACKE_zhegvd(LAPACK_COL_MAJOR,1,'V','U',pSPARC->Nstates,pSPARC->Hp_kpt,
                          pSPARC->Nstates,pSPARC->Mp_kpt,pSPARC->Nstates,
                          pSPARC->lambda + kpt*pSPARC->Nstates + spn_i*pSPARC->Nkpts_kptcomm*pSPARC->Nstates);
        }
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank_spincomm && spn_i == 0 && kpt == 0) {
            printf("==generalized eigenproblem: "
                   "info = %d, solving generalized eigenproblem using LAPACKE_zhegvd: %.3f ms\n", 
                   info, (t2 - t1)*1e3);
        }
        #else
        (void) info; // suppress -Wunused-but-set-variable warning
        #endif

        int ONE = 1;
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        // distribute eigenvectors to block cyclic format
        pzgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp_kpt, &ONE, &ONE, 
                  pSPARC->desc_Hp_BLCYC, pSPARC->Q_kpt, &ONE, &ONE, 
                  pSPARC->desc_Q_BLCYC, &pSPARC->ictxt_blacs_topo);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank_spincomm && spn_i == 0 && kpt == 0) {
            printf("==generalized eigenproblem: "
                   "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
                   (t2 - t1)*1e3);
        }
        #endif
    } else {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        int ONE = 1, il = 1, iu = 1, *ifail, info, N, M, NZ;
        // double complex *work;
        double vl = 0.0, vu = 0.0, abstol, orfac;
        
        ifail = (int *)malloc(pSPARC->Nstates * sizeof(int));
        N = pSPARC->Nstates;
        orfac = pSPARC->eig_paral_orfac;
        #ifdef DEBUG
        if(!rank) printf("rank = %d, orfac = %.3e\n", rank, orfac);
        #endif 
        // this setting yields the most orthogonal eigenvectors
        abstol = pdlamch_(&pSPARC->ictxt_blacs_topo, "U");

        pzhegvx_subcomm_ (
                &ONE, "V", "A", "U", &N, pSPARC->Hp_kpt, &ONE, 
                &ONE, pSPARC->desc_Hp_BLCYC, pSPARC->Mp_kpt, &ONE, &ONE, 
                pSPARC->desc_Mp_BLCYC, &vl, &vu, &il, &iu, &abstol, &M, 
                &NZ, pSPARC->lambda + kpt*N + spn_i*pSPARC->Nkpts_kptcomm*N, &orfac, 
                pSPARC->Q_kpt, &ONE, &ONE, pSPARC->desc_Q_BLCYC, ifail, &info,
                pSPARC->blacscomm, pSPARC->eig_paral_subdims, pSPARC->eig_paral_blksz);

        if (info != 0 && !rank) {
            printf("\nError in solving generalized eigenproblem! info = %d\n", info);
        }
        
        #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank && spn_i == 0 && kpt == 0) {
            printf("rank = %d, info = %d, ifail[0] = %d, time for solving generalized eigenproblem : %.3f ms\n", 
                    rank, info, ifail[0], (t2 - t1)*1e3);
            printf("rank = %d, after calling pzhegvx, Nstates = %d\n", rank, N);
        }
        #endif
        free(ifail);
    }
#else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("[FATAL] Subspace eigenproblem are using ScaLAPACK routines but ScaLAPACK is not compiled\n");
    if (rank == 0) printf("\nPlease turn on USE_MKL or USE_SCALAPACK!\n");
    exit(255);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
} 



/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals.
 *
 *          This is just to perform a matrix-matrix multiplication: Psi = Psi * Q.
 *          Note that Psi, Q and PsiQ are distributed block cyclically, Psi_rot is
 *          the band + domain parallelization format of PsiQ.
 */
void Subspace_Rotation_kpt(SPARC_OBJ *pSPARC, double complex *Psi, double complex  *Q, double complex *PsiQ, double complex *Psi_rot, int kpt, int spn_i)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int Nd_blacscomm, ONE = 1;
    Nd_blacscomm = pSPARC->is_domain_uniform ? pSPARC->Nd : pSPARC->Nd_d_dmcomm;
    Nd_blacscomm *= pSPARC->Nspinor;

    double complex alpha = 1.0, beta = 0.0;

    #ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
    #endif

    if (pSPARC->npband > 1) {
        // perform matrix multiplication Psi * Q using ScaLAPACK routines
        pzgemm_("N", "N", &Nd_blacscomm, &pSPARC->Nstates, &pSPARC->Nstates, &alpha, 
                Psi, &ONE, &ONE, pSPARC->desc_orb_BLCYC, Q, &ONE, &ONE, 
                pSPARC->desc_Q_BLCYC, &beta, PsiQ, &ONE, &ONE, pSPARC->desc_orb_BLCYC);
    } else {
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            Nd_blacscomm, pSPARC->Nstates, pSPARC->Nstates, 
            &alpha, Psi, Nd_blacscomm, Q, pSPARC->Nstates,
            &beta, PsiQ, Nd_blacscomm
        );
    }

    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, subspace rotation using ScaLAPACK took %.3f ms\n", 
                     rank, (t2 - t1)*1e3); 
    t1 = MPI_Wtime();
    #endif
    
    if (pSPARC->npband > 1) {
        // distribute rotated orbitals from block cyclic format back into 
        // original format (band + domain)
        pzgemr2d_(&Nd_blacscomm, &pSPARC->Nstates, PsiQ, &ONE, &ONE, 
                  pSPARC->desc_orb_BLCYC, Psi_rot, &ONE, &ONE, 
                  pSPARC->desc_orbitals, &pSPARC->ictxt_blacs);
    }
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) 
        printf("rank = %2d, Distributing orbital back into band + domain format took %.3f ms\n", 
                rank, (t2 - t1)*1e3); 
    #endif
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}



/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Hamiltonian corresponding to the given k-point.  
 */
void Lanczos_kpt(const SPARC_OBJ *pSPARC, int *DMVertices, double *Veff_loc,
             ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
             double *eigmin, double *eigmax, double complex *x0, double TOL_min, double TOL_max, 
             int MAXIT, int kpt, int spn_i, MPI_Comm comm, MPI_Request *req_veff_loc) 
{
    #ifdef DEBUG
    double t1, t2;
    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    #ifdef DEBUG
    if (rank == 0 && spn_i == 0 && kpt == 0) printf("\nStart Lanczos algorithm ...\n");
    #endif

    if (comm == MPI_COMM_NULL) {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        // receive computed eigmin and eigmax from root processors in the Cart topology
        double Bbuf[2];

        // the non-root processes do nothing
        MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm_inter);
        // this is inefficient since many proc already have the info.
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm); 
        
        *eigmin = Bbuf[0]; *eigmax = Bbuf[1];

        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank && spn_i == 0 && kpt == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
        #endif
        return;
    }

    double vscal, err_eigmin, err_eigmax, eigmin_pre, eigmax_pre;
    double complex *V_j, *V_jm1, *V_jp1;
    double *a, *b, *d, *e;
    int i, j, DMnd;
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);
    DMnd *= pSPARC->Nspinor;

    V_j   = (double complex *)malloc( DMnd * sizeof(double complex));
    V_jm1 = (double complex *)malloc( DMnd * sizeof(double complex));
    V_jp1 = (double complex *)malloc( DMnd * sizeof(double complex));
    a     = (double*)malloc( (MAXIT+1) * sizeof(double));
    b     = (double*)malloc( (MAXIT+1) * sizeof(double));
    d     = (double*)malloc( (MAXIT+1) * sizeof(double));
    e     = (double*)malloc( (MAXIT+1) * sizeof(double));

    /* set random initial guess vector V_jm1 with unit 2-norm */
#ifdef DEBUG
    srand(rank+1);
    //srand(rank+(int)MPI_Wtime());
#else
    srand(rank+1); 
    //srand(rank+1+(int)MPI_Wtime());
#endif    
    double rand_min = -1.0, rand_max = 1.0;
    for (i = 0; i < DMnd; i++) {
        //V_jm1[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        //TODO: [1,...,1] might be a better guess for it's closer to the eigvec for Lap 
        //      with zero (or ~= zero) eigval, and since min eig is harder to converge
        //      this will make the overall convergence faster.
        //V_jm1[i] = 1.0 - 1e-3 + 2e-3 * (double) rand() / RAND_MAX; 
        V_jm1[i] = x0[i];
    }
    
    // TODO: if input initial guess is unit norm, then there's no need for the following
    Vector2Norm_complex(V_jm1, DMnd, &vscal, comm); // find norm of V_jm1
    vscal = 1.0 / vscal;
    // scale the random guess vector s.t. V_jm1 has unit 2-norm
    for (i = 0; i < DMnd; i++)
        V_jm1[i] *= vscal;

    // calculate V_j = H * V_jm1, TODO: check if Veff_loc is available
    // TODO: remove if not using nonblocking communication
#ifdef DEBUG    
    t1 = MPI_Wtime();
#endif
    MPI_Wait(req_veff_loc, MPI_STATUS_IGNORE);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("Wait for veff to be bcasted took %.3f ms\n", (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    Hamiltonian_vectors_mult_kpt(
        pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
        nlocProj, 1, 0.0, V_jm1, V_j, spn_i, kpt, comm
    );

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, One H*x took %.3f ms\n", rank, (t2-t1)*1e3);   
#endif
    // find dot product of V_jm1 and V_j, and store the value in a[0]
    VectorDotProduct_complex(V_jm1, V_j, DMnd, &a[0], comm);

    // orthogonalize V_jm1 and V_j
    for (i = 0; i < DMnd; i++)
        V_j[i] -= a[0] * V_jm1[i];
    
    // find norm of V_j
    Vector2Norm_complex(V_j, DMnd, &b[0], comm); 
    
    if (!b[0]) {
        // if ||V_j|| = 0, pick an arbitrary vector with unit norm that's orthogonal to V_jm1
        rand_min = -1.0, rand_max = 1.0;
        for (i = 0; i < DMnd; i++) {
            V_j[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        }
        // orthogonalize V_j and V_jm1
        VectorDotProduct_complex(V_j, V_jm1, DMnd, &a[0], comm);
        for (i = 0; i < DMnd; i++)
            V_j[i] -= a[0] * V_jm1[i];
        // find norm of V_j
        Vector2Norm_complex(V_j, DMnd, &b[0], comm);
    }

    // scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (i = 0; i < DMnd; i++) 
        V_j[i] *= vscal;

    eigmin_pre = *eigmin = 0.0;
    eigmax_pre = *eigmax = 0.0;
    err_eigmin = TOL_min + 1.0;
    err_eigmax = TOL_max + 1.0;
    j = 0;
    while ((err_eigmin > TOL_min || err_eigmax > TOL_max) && j < MAXIT) 
    {
        //t1 = MPI_Wtime();        
        // V_{j+1} = H * V_j
        Hamiltonian_vectors_mult_kpt(
            pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
            nlocProj, 1, 0.0, V_j, V_jp1, spn_i, kpt, comm
        );

        // a[j+1] = <V_j, V_{j+1}>
        VectorDotProduct_complex(V_j, V_jp1, DMnd, &a[j+1], comm);

        for (i = 0; i < DMnd; i++) {
            // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
            V_jp1[i] -= (a[j+1] * V_j[i] + b[j] * V_jm1[i]);    
            // update V_{j-1}, i.e., V_{j-1} := V_j
            V_jm1[i] = V_j[i];
        }
        
        Vector2Norm_complex(V_jp1, DMnd, &b[j+1], comm);
        if (!b[j+1]) {
            break;
        }
        
        vscal = 1.0 / b[j+1];
        // update V_j := V_{j+1} / ||V_{j+1}||
        for (i = 0; i < DMnd; i++)
            V_j[i] = V_jp1[i] * vscal;

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        for (i = 0; i < j+2; i++) {
            d[i] = a[i];
            e[i] = b[i];
        }
        
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        
        if (!LAPACKE_dsterf(j+2, d, e)) {
            *eigmin = d[0];
            *eigmax = d[j+1];
        } else {
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }
        
        #ifdef DEBUG
        t2 = MPI_Wtime();
        #endif
        
        err_eigmin = fabs(*eigmin - eigmin_pre);
        err_eigmax = fabs(*eigmax - eigmax_pre);

        eigmin_pre = *eigmin;
        eigmax_pre = *eigmax;

        j++;
    }

#ifdef DEBUG
    if (rank == 0 && spn_i == 0 && kpt == 0) {
        printf("    Lanczos iter %d, eigmin  = %.9f, eigmax = %.9f, err_eigmin = %.3e, err_eigmax = %.3e\n",j,*eigmin, *eigmax,err_eigmin,err_eigmax);
    }
#endif
    
    if (pSPARC->kptcomm_inter != MPI_COMM_NULL) {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif

        // broadcast the computed eigmin and eigmax from root to processors not in the Cart topology
        int rank_kptcomm = -1;
        MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

        double Bbuf[2];
        Bbuf[0] = *eigmin; Bbuf[1] = *eigmax;
        if (!rank_kptcomm) {
            // the root process will broadcast the values
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_ROOT, pSPARC->kptcomm_inter);
        } else {
            // the non-root processes do nothing
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kptcomm_inter);
        }
        //MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm);

#ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank && spn_i == 0 && kpt == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
#endif
    }
    
    free(V_j); free(V_jm1); free(V_jp1);
    free(a); free(b); free(d); free(e);
}



/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Lap.  
 */
void Lanczos_laplacian_kpt(
    const SPARC_OBJ *pSPARC, const int *DMVertices, double *eigmin, 
    double *eigmax, double complex *x0, const double TOL_min, const double TOL_max, 
    const int MAXIT, int kpt, int spn_i, MPI_Comm comm
) 
{
    #ifdef DEBUG
    double t1, t2;
    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
    if (rank == 0 && spn_i == 0 && kpt == 0) printf("\nStart Lanczos algorithm ...\n");
    #endif
    if (comm == MPI_COMM_NULL) {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        // receive computed eigmin and eigmax from root processors in the Cart topology
        double Bbuf[2];
        // the non-root processes do nothing
        MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm_inter);
        // this is inefficient since many proc already have the info.
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm); 
        
        *eigmin = Bbuf[0]; *eigmax = Bbuf[1];

        #ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank && spn_i == 0 && kpt == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
        #endif
        return;
    }

    double vscal, err_eigmin, err_eigmax, eigmin_pre, eigmax_pre;
    double complex *V_j, *V_jm1, *V_jp1;
    double *a, *b, *d, *e;
    int i, j, DMnd;
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);
    DMnd *= pSPARC->Nspinor;
    
    V_j   = (double complex *)malloc( DMnd * sizeof(double complex));
    V_jm1 = (double complex *)malloc( DMnd * sizeof(double complex));
    V_jp1 = (double complex *)malloc( DMnd * sizeof(double complex));
    a     = (double*)malloc( (MAXIT+1) * sizeof(double));
    b     = (double*)malloc( (MAXIT+1) * sizeof(double));
    d     = (double*)malloc( (MAXIT+1) * sizeof(double));
    e     = (double*)malloc( (MAXIT+1) * sizeof(double));

    /* set random initial guess vector V_jm1 with unit 2-norm */
#ifdef DEBUG
    srand(rank+1);
    //srand(rank+(int)MPI_Wtime());
#else
    srand(rank+1); 
    //srand(rank+1+(int)MPI_Wtime());
#endif    
    double rand_min = -1.0, rand_max = 1.0;
    for (i = 0; i < DMnd; i++) {
        //V_jm1[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        //TODO: [1,...,1] might be a better guess for it's closer to the eigvec for Lap 
        //      with zero (or ~= zero) eigval, and since min eig is harder to converge
        //      this will make the overall convergence faster.
        //V_jm1[i] = 1.0 - 1e-3 + 2e-3 * (double) rand() / RAND_MAX; 
        V_jm1[i] = x0[i];
    }
    
    Vector2Norm_complex(V_jm1, DMnd, &vscal, comm); // find norm of V_jm1
    vscal = 1.0 / vscal;
    // scale the random guess vector s.t. V_jm1 has unit 2-norm
    for (i = 0; i < DMnd; i++) 
        V_jm1[i] *= vscal;

    // calculate V_j = H * V_jm1
    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    Lap_vec_mult_kpt(pSPARC, DMnd, DMVertices, 1, 0.0, V_jm1, V_j, kpt, comm);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank && spn_i == 0 && kpt == 0) printf("rank = %2d, One H*x took %.3f ms\n", rank, (t2-t1)*1e3);   
#endif
    // find dot product of V_jm1 and V_j, and store the value in a[0]
    VectorDotProduct_complex(V_jm1, V_j, DMnd, &a[0], comm);

    // orthogonalize V_jm1 and V_j
    for (i = 0; i < DMnd; i++)
        V_j[i] -= a[0] * V_jm1[i];
    
    // find norm of V_j
    Vector2Norm_complex(V_j, DMnd, &b[0], comm); 
    
    if (!b[0]) {
        // if ||V_j|| = 0, pick an arbitrary vector with unit norm that's orthogonal to V_jm1
        rand_min = -1.0, rand_max = 1.0;
        for (i = 0; i < DMnd; i++) {
            V_j[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        }
        // orthogonalize V_j and V_jm1
        VectorDotProduct_complex(V_j, V_jm1, DMnd, &a[0], comm);
        for (i = 0; i < DMnd; i++)
            V_j[i] -= a[0] * V_jm1[i];
        // find norm of V_j
        Vector2Norm_complex(V_j, DMnd, &b[0], comm);
    }

    // scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (i = 0; i < DMnd; i++) 
        V_j[i] *= vscal;

    eigmin_pre = *eigmin = 0.0;
    eigmax_pre = *eigmax = 0.0;
    err_eigmin = TOL_min + 1.0;
    err_eigmax = TOL_max + 1.0;
    j = 0;
    while ((err_eigmin > TOL_min || err_eigmax > TOL_max) && j < MAXIT) {
        //t1 = MPI_Wtime();        
        // V_{j+1} = H * V_j
		Lap_vec_mult_kpt(pSPARC, DMnd, DMVertices, 1, 0.0, V_j, V_jp1, kpt, comm);

        // a[j+1] = <V_j, V_{j+1}>
        VectorDotProduct_complex(V_j, V_jp1, DMnd, &a[j+1], comm);

        for (i = 0; i < DMnd; i++) {
            // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
            V_jp1[i] -= (a[j+1] * V_j[i] + b[j] * V_jm1[i]);    
            // update V_{j-1}, i.e., V_{j-1} := V_j
            V_jm1[i] = V_j[i];
        }

        Vector2Norm_complex(V_jp1, DMnd, &b[j+1], comm);
        if (!b[j+1]) {
            break;
        }
        
        vscal = 1.0 / b[j+1];
        // update V_j := V_{j+1} / ||V_{j+1}||
        for (i = 0; i < DMnd; i++)
            V_j[i] = V_jp1[i] * vscal;

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        for (i = 0; i < j+2; i++) {
            d[i] = a[i];
            e[i] = b[i];
        }
        
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        
        if (!LAPACKE_dsterf(j+2, d, e)) {
            *eigmin = d[0];
            *eigmax = d[j+1];
        } else {
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }
        
        #ifdef DEBUG
        t2 = MPI_Wtime();
        #endif
        
        err_eigmin = fabs(*eigmin - eigmin_pre);
        err_eigmax = fabs(*eigmax - eigmax_pre);

        eigmin_pre = *eigmin;
        eigmax_pre = *eigmax;

        j++;
    }
#ifdef DEBUG
    if (rank == 0 && spn_i == 0 && kpt == 0) {
        printf("    Lanczos iter %d, eigmin  = %.9f, eigmax = %.9f, err_eigmin = %.3e, err_eigmax = %.3e\n",j,*eigmin, *eigmax,err_eigmin,err_eigmax);
    }
#endif
    
    if (pSPARC->kptcomm_inter != MPI_COMM_NULL) {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif

        // broadcast the computed eigmin and eigmax from root to processors not in the Cart topology
        int rank_kptcomm = -1;
        MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

        double Bbuf[2];
        Bbuf[0] = *eigmin; Bbuf[1] = *eigmax;
        if (!rank_kptcomm) {
            // the root process will broadcast the values
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_ROOT, pSPARC->kptcomm_inter);
        } else {
            // the non-root processes do nothing
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kptcomm_inter);
        }
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm);

#ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank && spn_i == 0 && kpt == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
#endif
    }
    
    free(V_j); free(V_jm1); free(V_jp1);
    free(a); free(b); free(d); free(e);
}


