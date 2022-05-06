/***
 * @file    exactExchange.c
 * @brief   This file contains the functions for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <complex.h>
#include <limits.h>
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #define MKL_Complex16 double complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/** ScaLAPACK routines */
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
#ifdef USE_FFTW
    #include <fftw3.h>
#endif

#include "exactExchange.h"
#include "exactExchangeKpt.h"
#include "tools.h"
#include "parallelization.h"
#include "electronicGroundState.h"
#include "exchangeCorrelation.h"
#include "lapVecRoutines.h"
#include "lapVecOrth.h"
#include "lapVecNonOrth.h"
#include "nlocVecRoutines.h"
#include "linearSolver.h"
#include "electrostatics.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Outer loop of SCF using Vexx (exact exchange potential)
 */
void Exact_Exchange_loop(SPARC_OBJ *pSPARC) {
    int i, k, n, rank, DMnd, Nband, blacs_size, kpt_size, spn_i;
    double t1, t2;
    FILE *output_fp;

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /************************ Exact exchange potential parameters ************************/
    int count_xx = 0;
    double Eband_pre = pSPARC->Eband, err_Exx = pSPARC->TOL_FOCK + 1;
    pSPARC->Exxtime = pSPARC->ACEtime = 0.0;
    
    // blacscomm contains all processes with the same rank_dmcomm
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_size);

    /************************* Update Veff copied from SCF code **************************/
    #ifdef DEBUG
    if(!rank) 
        printf("\nStart evaluating Exact Exchange potential!\n");
    #endif  

    // calculate xc potential (LDA), "Vxc"
    t1 = MPI_Wtime(); 
    Calculate_Vxc(pSPARC);
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (rank == 0) printf("rank = %d, XC calculation took %.3f ms\n", rank, (t2-t1)*1e3); 
    #endif 

    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);

    double veff_mean;
    veff_mean = 0.0;
    // for potential mixing with PBC, calculate mean(veff)
    if (pSPARC->MixingVariable == 1)  { // potential mixing
        if (pSPARC->BC == 2 || pSPARC->BC == 0) {
            VectorSum(pSPARC->Veff_loc_dmcomm_phi, pSPARC->Nspin*pSPARC->Nd_d, &veff_mean, pSPARC->dmcomm_phi);
            veff_mean /= ((double) (pSPARC->Nd * pSPARC->Nspin));
        }
    }
    pSPARC->veff_mean = veff_mean;

    // initialize mixing_hist_xk (and mixing_hist_xkm1)
    Update_mixing_hist_xk(pSPARC, veff_mean);

    // transfer Veff_loc from "phi-domain" to "psi-domain"
    t1 = MPI_Wtime();
    for (i = 0; i < pSPARC->Nspin; i++)
        Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*pSPARC->Nd_d, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
    t2 = MPI_Wtime();

    #ifdef DEBUG
    if(!rank) 
        printf("rank = %d, Transfering Veff from phi-domain to psi-domain took %.3f ms\n", 
               rank, (t2 - t1) * 1e3);
    #endif 

    /******************************* Hartre-Fock outer loop ******************************/
    while ((err_Exx > pSPARC->TOL_FOCK && count_xx < pSPARC->MAXIT_FOCK) || count_xx < pSPARC->MINIT_FOCK) {
    #ifdef DEBUG
    if(!rank) 
        printf("\nHartree-Fock Outer Loop: %d \n",count_xx + 1);
    #endif  

        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"\nNo.%d Exx outer loop: \n", count_xx + 1);
            fclose(output_fp);
        }

        if (count_xx > 0) {
            if (pSPARC->MixingVariable == 1 && (pSPARC->BC == 2 || pSPARC->BC == 0)) { // potential mixing, add veff_mean back
                VectorShift(pSPARC->Veff_loc_dmcomm_phi, pSPARC->Nspin*pSPARC->Nd_d, pSPARC->veff_mean, pSPARC->dmcomm_phi);
            }

            // transfer Veff_loc from "phi-domain" to "psi-domain"
            t1 = MPI_Wtime();
            for (i = 0; i < pSPARC->Nspin; i++)
                Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*pSPARC->Nd_d, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
            t2 = MPI_Wtime();

            #ifdef DEBUG
            if(!rank) 
                printf("rank = %d, Transfering Veff from phi-domain to psi-domain took %.3f ms\n", 
                    rank, (t2 - t1) * 1e3);
            #endif 
        }

        if (pSPARC->ACEFlag == 0) {
            if (pSPARC->isGammaPoint == 1) {
                // Gathering all outer orbitals into each band comm
                t1 = MPI_Wtime();
                gather_psi_occ_outer(pSPARC, pSPARC->psi_outer, pSPARC->occ_outer);
                t2 = MPI_Wtime();
                #ifdef DEBUG
                if(!rank) 
                    printf("\nGathering all bands of psi_outer to each dmcomm took : %.3f ms\n", (t2-t1)*1e3);
                #endif 
            } else {
                // Gathering all outer orbitals and outer occ
                t1 = MPI_Wtime();
                gather_psi_occ_outer_kpt(pSPARC, pSPARC->psi_outer_kpt, pSPARC->occ_outer);
                t2 = MPI_Wtime();
                #ifdef DEBUG
                if(!rank) 
                    printf("\nGathering all bands and all kpoints of psi_outer and occupations to each dmcomm took : %.3f ms\n", (t2-t1)*1e3);
                #endif 
            }
        } else {
            #ifdef DEBUG
            if(!rank) printf("\nStart to create ACE operator!\n");
            #endif  
            t1 = MPI_Wtime();
            // create ACE operator 
            if (pSPARC->isGammaPoint == 1) {
                allocate_ACE(pSPARC);
                int xi_shift = pSPARC->Nstates_occ[0] * DMnd;
                for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    ACE_operator(pSPARC, pSPARC->Xorb + spn_i*DMnd*Nband, pSPARC->occ + spn_i*pSPARC->Nstates, spn_i, pSPARC->Xi + spn_i*xi_shift);
                }
            } else {
                gather_psi_occ_outer_kpt(pSPARC, pSPARC->psi_outer_kpt, pSPARC->occ_outer);
                allocate_ACE_kpt(pSPARC);
                // ACE_operator_kpt(pSPARC);
                int xi_shift = DMnd * pSPARC->Nstates_occ[0] * pSPARC->Nkpts_kptcomm;
                int psi_outer_shift = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
                int psi_shift = DMnd * pSPARC->Nband_bandcomm * pSPARC->Nkpts_kptcomm;
                int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

                for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    ACE_operator_kpt(pSPARC, pSPARC->psi_outer_kpt + spn_i*psi_outer_shift,
                        pSPARC->Xorb_kpt + spn_i*psi_shift, pSPARC->occ_outer + spn_i*occ_outer_shift, 
                        spn_i, pSPARC->Xi_kpt + spn_i*xi_shift);
                }
            }
            t2 = MPI_Wtime();
            pSPARC->ACEtime += (t2 - t1);
            #ifdef DEBUG
            if(!rank) printf("\nCreating ACE operator took %.3f ms!\n", (t2 - t1)*1e3);
            #endif  
        }

        // transfer psi_outer from "psi-domain" to "phi-domain" in No-ACE case 
        // transfer Xi from "psi-domain" to "phi-domain" in ACE case 
        t1 = MPI_Wtime();
        
        if (pSPARC->ACEFlag == 0) {
            if (pSPARC->isGammaPoint == 1) {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nstates*pSPARC->Nspin_spincomm, pSPARC->psi_outer, pSPARC->psi_outer_kptcomm_topo);    
            } else {
                Transfer_dmcomm_to_kptcomm_topo_complex(pSPARC, pSPARC->Nstates*pSPARC->Nkpts_hf_red*pSPARC->Nspin_spincomm, pSPARC->psi_outer_kpt, pSPARC->psi_outer_kptcomm_topo_kpt);
            }
            
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank) 
                printf("\nTransfering all bands of psi_outer to kptcomm_topo took : %.3f ms\n", (t2-t1)*1e3);
            #endif  
        } else {
            int sum_Nsocc = 0;
            for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
                sum_Nsocc += pSPARC->Nstates_occ[spn_i];

            if (pSPARC->isGammaPoint == 1) {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, sum_Nsocc, pSPARC->Xi, pSPARC->Xi_kptcomm_topo);
            } else {
                Transfer_dmcomm_to_kptcomm_topo_complex(pSPARC, sum_Nsocc*pSPARC->Nkpts_kptcomm, pSPARC->Xi_kpt, pSPARC->Xi_kptcomm_topo_kpt);
            }

            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank) 
                printf("\nTransfering Xi to kptcomm_topo took : %.3f ms\n", (t2-t1)*1e3);
            #endif  
        }

        // compute exact exchange energy estimation with psi_outer
        // Eexx saves negative exact exchange energy without hybrid mixing
        if (pSPARC->isGammaPoint == 1)
            evaluate_exact_exchange_energy(pSPARC);
        else
            evaluate_exact_exchange_energy_kpt(pSPARC);
        
        scf_loop(pSPARC);

        err_Exx = fabs(Eband_pre - pSPARC->Eband)/pSPARC->n_atom;
        MPI_Bcast(&err_Exx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);                 // TODO: Create bridge comm 

        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"Exx outer loop error: %.10e \n",err_Exx);
            fclose(output_fp);
        }

        Eband_pre = pSPARC->Eband;
        count_xx ++;
    }

    #ifdef DEBUG
    if(!rank) 
        printf("\nFinished outer loop in %d steps!\n", count_xx);
    #endif  

    if (err_Exx > pSPARC->TOL_FOCK) {
        if(!rank) {
            printf("WARNING: EXX outer loop did not converge to desired accuracy!\n");
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"WARNING: EXX outer loop did not converge to desired accuracy!\n");
            fclose(output_fp);
        }
    }

    // update the converged exact exchange energy
    pSPARC->Exc -= pSPARC->Eexx;
    pSPARC->Etot += 2 * pSPARC->Eexx;
    if (pSPARC->isGammaPoint == 1)
        evaluate_exact_exchange_energy(pSPARC);
    else
        evaluate_exact_exchange_energy_kpt(pSPARC);

    pSPARC->Exc += pSPARC->Eexx;
    pSPARC->Etot -= 2*pSPARC->Eexx;

    #ifdef DEBUG
    if(!rank && pSPARC->ACEFlag == 1) {
        printf("\n== Exact exchange Timing: creating ACE: %.3f ms\tapply ACE: %.3f ms\n",
            pSPARC->ACEtime*1e3, pSPARC->Exxtime*1e3);
    }
    if(!rank && pSPARC->ACEFlag == 0) {
        printf("\n== Exact exchange Timing: apply Vx takes    %.3f ms\n", pSPARC->Exxtime*1e3);
    }
    #endif  
}


/**
 * @brief   Initialization of all variables for exact exchange.
 */
void init_exx(SPARC_OBJ *pSPARC) {
    int i, j, rank, DMnd, Nband, len_full_tot, Ns_full_total, blacs_size, kpt_bridge_size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    
    find_local_kpthf(pSPARC);                                                                           // determine local number kpts for HF
    len_full_tot = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red * pSPARC->Nspin_spincomm;                  // total length across all kpts.
    Ns_full_total = pSPARC->Nstates * pSPARC->Nkpts_sym * pSPARC->Nspin_spincomm;                           // total length across all kpts.

    if (pSPARC->ACEFlag == 0) {
        if (pSPARC->isGammaPoint == 1) {
            // Storage of psi_outer in dmcomm
            pSPARC->psi_outer = (double *)calloc( len_full_tot , sizeof(double) ); 
            // Storage of psi_outer in kptcomm_topo
            pSPARC->psi_outer_kptcomm_topo = 
                    (double *)calloc(pSPARC->Nd_d_kptcomm * pSPARC->Nstates * pSPARC->Nspin_spincomm , sizeof(double));
            assert (pSPARC->psi_outer != NULL && pSPARC->psi_outer_kptcomm_topo != NULL);
            
            pSPARC->occ_outer = (double *)calloc(Ns_full_total, sizeof(double));
            assert(pSPARC->occ_outer != NULL);
        } else {
            // Storage of psi_outer_kpt in dmcomm
            pSPARC->psi_outer_kpt = (double _Complex *)calloc( len_full_tot , sizeof(double _Complex) ); 
            // Storage of psi_outer in kptcomm_topo
            pSPARC->psi_outer_kptcomm_topo_kpt = 
                    (double _Complex *)calloc(pSPARC->Nd_d_kptcomm * pSPARC->Nstates * pSPARC->Nkpts_hf_red * pSPARC->Nspin_spincomm, sizeof(double _Complex));
            assert (pSPARC->psi_outer_kpt != NULL && pSPARC->psi_outer_kptcomm_topo_kpt != NULL);
            
            pSPARC->occ_outer = (double *)calloc(Ns_full_total, sizeof(double));
            assert(pSPARC->occ_outer != NULL);
        }
    } else {
        if (pSPARC->isGammaPoint == 1) {
            pSPARC->Nstates_occ[0] = pSPARC->Nstates_occ[1] = 0;
        } else {
            pSPARC->Nstates_occ[0] = pSPARC->Nstates_occ[1] = 0;
            pSPARC->psi_outer_kpt = (double _Complex *)calloc( len_full_tot , sizeof(double _Complex) );                    
            pSPARC->occ_outer = (double *)calloc(Ns_full_total, sizeof(double));
            assert(pSPARC->psi_outer_kpt != NULL && pSPARC->occ_outer != NULL);
        }
    }
    
    find_k_shift(pSPARC);
    kshift_phasefactor(pSPARC);

    if (pSPARC->EXXMeth_Flag == 0) {
        if (pSPARC->EXXDiv_Flag == 1) 
            auxiliary_constant(pSPARC);

        // compute the constant coefficients for solving Poisson's equation using FFT
        // For even conjugate space (FFT), only half of the coefficients are needed
        if (pSPARC->dmcomm != MPI_COMM_NULL || pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            if (pSPARC->isGammaPoint == 1) {
                compute_pois_fft_const(pSPARC);
            } else {
                compute_pois_fft_const_kpt(pSPARC);
            }
        }
    }

    // create kpttopo_dmcomm_inter
    pSPARC->flag_kpttopo_dm = 0;
    create_kpttopo_dmcomm_inter(pSPARC);
}

/**
 * @brief   Memory free of all variables for exact exchange.
 */
void free_exx(SPARC_OBJ *pSPARC) {
    int blacs_size, kpt_bridge_size;
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);
    
    free(pSPARC->kpthf_flag_kptcomm);
    free(pSPARC->Nkpts_hf_list);
    if (pSPARC->ACEFlag == 0) {
        if (pSPARC->isGammaPoint == 1) {
            free(pSPARC->psi_outer);
            free(pSPARC->psi_outer_kptcomm_topo);
            free(pSPARC->occ_outer);
        } else {
            free(pSPARC->psi_outer_kpt);
            free(pSPARC->psi_outer_kptcomm_topo_kpt);
            free(pSPARC->occ_outer);
        }
    } else {
        if (pSPARC->isGammaPoint == 1) {
            if (pSPARC->spincomm_index >= 0) {
                free(pSPARC->Xi);
                free(pSPARC->Xi_kptcomm_topo);
            }
        } else {
            free(pSPARC->psi_outer_kpt);
            free(pSPARC->occ_outer);
            if (pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index >= 0) {
                free(pSPARC->Xi_kpt);
                free(pSPARC->Xi_kptcomm_topo_kpt);
            }
        }
    }

    if (pSPARC->EXXMeth_Flag == 0) {
        if (pSPARC->dmcomm != MPI_COMM_NULL || pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            free(pSPARC->pois_FFT_const);
            if (pSPARC->Calc_stress == 1) {
                free(pSPARC->pois_FFT_const_stress);
                if (pSPARC->EXXDiv_Flag == 0)
                    free(pSPARC->pois_FFT_const_stress2);
            } else if (pSPARC->Calc_pres == 1) {
                if (pSPARC->EXXDiv_Flag != 0)
                    free(pSPARC->pois_FFT_const_press);
            }
        }
    }

    if (pSPARC->Nkpts_shift > 1) {
        free(pSPARC->neg_phase);
        free(pSPARC->pos_phase);
    }

    if (pSPARC->kpttopo_dmcomm_inter != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->kpttopo_dmcomm_inter);
}

/**
 * @brief   Create ACE operator in dmcomm
 *
 *          Using occupied + extra orbitals to construct the ACE operator 
 *          Due to the symmetry of ACE operator, only Ns_occ(Ns_occ+1)/2
 *          Poisson's equations need to be solved.
 *          Note: there is no band parallelization when usign ACE, so there 
 *          is only 1 dmcomm for each k-point. 
 */
void ACE_operator(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer, int spn_i, double *Xi) {
    int i, j, k, rank, nproc_dmcomm, Ns, dims[3], DMnd, ONE = 1, Ns_occ;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ, occ_i, occ_j, *rhs, *Vi, *M, t1, t2, alpha = 1.0, beta = 0.0;
    /******************************************************************************/

    if (pSPARC->dmcomm == MPI_COMM_NULL) return;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DMnd = pSPARC->Nd_d_dmcomm;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
    Ns_occ = pSPARC->Nstates_occ[spn_i];
    num_rhs = (Ns_occ*(Ns_occ+1))/2;
    Ns = pSPARC->Nstates;

    /* EXXMem_batch could be any positive integer to define the maximum number of 
       Poisson's equations to be solved at one time. The smaller the EXXMem_batch, 
       the less memory are required, but also the longer the running time. This part
       of code could be directly applied to NON-ACE part. */
    
    batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                        num_rhs : pSPARC->EXXMem_batch * nproc_dmcomm;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required

    rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
    assert(rhs != NULL && Vi != NULL);

    /*************** Solve all Poisson's equation and apply to psi ****************/
    rhs_list_i = (int*) calloc(num_rhs, sizeof(int)); 
    rhs_list_j = (int*) calloc(num_rhs, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    count = 0;
    for (i = 0; i < Ns_occ; i++) {
        for (j = i; j < Ns_occ; j++) {
            rhs_list_i[count] = i;
            rhs_list_j[count] = j;
            count ++;
        }
    }

    for (i = 0; i < Ns_occ*DMnd; i++) Xi[i] = 0.0;

    t1 = MPI_Wtime();
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            for (k = 0; k < DMnd; k++) {
                rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd] * psi_outer[k + i*DMnd];
            }
        }
        
        poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, pSPARC->dmcomm);
        
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            
            occ_i = occ_outer[i];
            occ_j = occ_outer[j];

            for (k = 0; k < DMnd; k++) {
                Xi[k + i*DMnd] -= occ_j * psi_outer[k + j*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            }

            if (i != j) {
                for (k = 0; k < DMnd; k++) {
                    Xi[k + j*DMnd] -= occ_i * psi_outer[k + i*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                }
            }
        }
    }

    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs);
    free(Vi);

    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && !spn_i) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif
    /******************************************************************************/

    M = (double *)calloc(Ns_occ * Ns_occ, sizeof(double));
    assert(M != NULL);
    
    t1 = MPI_Wtime();
    // perform matrix multiplication psi' * W using ScaLAPACK routines
    cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Ns_occ, Ns_occ, DMnd,
                1.0, psi_outer, DMnd, Xi, DMnd, 0.0, M, Ns_occ);

    if (nproc_dmcomm > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, M, Ns_occ*Ns_occ,
                      MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && !spn_i) printf("rank = %2d, finding M = psi'* W took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif

    // perform Cholesky Factorization on -M
    // M = chol(-M), upper triangular matrix
    for (i = 0; i < Ns_occ*Ns_occ; i++) M[i] = -1.0 * M[i];

    t1 = MPI_Wtime();
    int info = 0;
    info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', Ns_occ, M, Ns_occ);
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (!rank && !spn_i) 
        printf("==Cholesky Factorization: "
               "info = %d, computing Cholesky Factorization using LAPACKE_dpotrf: %.3f ms\n", 
               info, (t2 - t1)*1e3);
    #endif

    // Xi = WM^(-1)
    t1 = MPI_Wtime();
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, 
                CblasNonUnit, DMnd, Ns_occ, 1.0, M, Ns_occ, Xi, DMnd);    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (!rank && !spn_i) 
        printf("==Triangular matrix equation: "
               "Solving triangular matrix equation using cblas_dtrsm: %.3f ms\n", (t2 - t1)*1e3);
    #endif

    free(M);
}


/**
 * @brief   Evaluating Exact Exchange potential
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ncol, int DMnd, double *Hx, int spin, MPI_Comm comm) {        
    int i, j, k, rank, Lanczos_flag, dims[3];
    double *psi, *Xi, t1, t2, *occ;
    
    MPI_Comm_rank(comm, &rank);
    Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
    /********************************************************************/

    int xi_shift = pSPARC->Nstates_occ[0] * DMnd;
    int psi_outer_shift = DMnd * pSPARC->Nstates;
    int occ_outer_shift = pSPARC->Nstates;

    t1 = MPI_Wtime();
    if (pSPARC->ACEFlag == 0) {
        if (Lanczos_flag == 0) {
            dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
        } else {
            dims[0] = pSPARC->npNdx_kptcomm; dims[1] = pSPARC->npNdy_kptcomm; dims[2] = pSPARC->npNdz_kptcomm;
        }
        occ = pSPARC->occ_outer + spin * occ_outer_shift;
        psi = (Lanczos_flag == 0) ? pSPARC->psi_outer + spin* psi_outer_shift : pSPARC->psi_outer_kptcomm_topo + spin* psi_outer_shift;
        evaluate_exact_exchange_potential(pSPARC, X, ncol, DMnd, dims, occ, psi, Hx, comm);

    } else {
        Xi = (Lanczos_flag == 0) ? pSPARC->Xi + spin * xi_shift : pSPARC->Xi_kptcomm_topo + spin * xi_shift;
        evaluate_exact_exchange_potential_ACE(pSPARC, X, ncol, DMnd, Xi, Hx, spin, comm);
    }

    t2 = MPI_Wtime();
    pSPARC->Exxtime +=(t2-t1);
}


/**
 * @brief   Evaluate Exact Exchange potential using non-ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param dims            3 dimensions of comm processes grid
 * @param occ_outer       Full set of occ_outer occupations
 * @param psi_outer       Full set of psi_outer orbitals
 * @param Hx              Result of Hx plus fock operator times X 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ncol, int DMnd, int *dims, 
                                    double *occ_outer, double *psi_outer, double *Hx, MPI_Comm comm)
{
    int i, j, k, rank, Ns, num_rhs, *rhs_list_i, *rhs_list_j;
    int size, batch_num_rhs, NL, base, loop;
    double occ, *rhs, *Vi, hyb_mixing, occ_alpha;

    Ns = pSPARC->Nstates;
    hyb_mixing = pSPARC->hyb_mixing;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/
    rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
    rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (i = 0; i < ncol; i++) {
        for (j = 0; j < Ns; j++) {
            if (occ_outer[j] > 1e-6) {
                rhs_list_i[count] = i;
                rhs_list_j[count] = j;
                count++;
            }
        }
    }
    num_rhs = count;

    if (num_rhs == 0) {
        free(rhs_list_i);
        free(rhs_list_j);
        return;
    }

    batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                        num_rhs : pSPARC->EXXMem_batch * size;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required                        
    rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
    assert(rhs != NULL && Vi != NULL);

    /*************** Solve all Poisson's equation and apply to X ****************/    
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            for (k = 0; k < DMnd; k++) {
                rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd] * X[k + i*DMnd];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, comm);

        // Apply exact exchange potential to vector X
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            occ = occ_outer[j];
            occ_alpha = occ * hyb_mixing;
            for (k = 0; k < DMnd; k++) {
                Hx[k + i*DMnd] -= occ_alpha * psi_outer[k + j*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            }
        }
    }

    
    free(rhs);
    free(Vi);
    free(rhs_list_i);
    free(rhs_list_j);
}



/**
 * @brief   Evaluate Exact Exchange potential using ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param Xi              Xi of ACE operator 
 * @param Hx              Result of Hx plus Vx times X
 * @param spin            Local spin index
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_ACE(SPARC_OBJ *pSPARC, 
    double *X, int ncol, int DMnd, double *Xi, double *Hx, int spin, MPI_Comm comm) 
{
    int rank, size, Ns_occ;
    Ns_occ = pSPARC->Nstates_occ[spin];
    double *Xi_times_psi = (double *) calloc(Ns_occ * ncol, sizeof(double));
    assert(Xi_times_psi != NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/

    // perform matrix multiplication Xi' * X using ScaLAPACK routines
    cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Ns_occ, ncol, DMnd,
                1.0, Xi, DMnd, X, DMnd, 0.0, Xi_times_psi, Ns_occ);

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Ns_occ*ncol, 
                      MPI_DOUBLE, MPI_SUM, comm);
    }

    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Ns_occ,
                -pSPARC->hyb_mixing, Xi, DMnd, Xi_times_psi, Ns_occ, 1.0, Hx, DMnd);

    free(Xi_times_psi);
}



/**
 * @brief   Evaluate Exact Exchange Energy
 */
void evaluate_exact_exchange_energy(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, rank, size, spn_i;
    int Ns, ncol, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, *rhs, *Vi, t1, t2, *psi_outer, temp, *occ_outer, *psi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    ncol = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;
    pSPARC->Eexx = 0.0;

    int xi_shift = DMnd * pSPARC->Nstates_occ[0] * pSPARC->Nkpts_kptcomm;
    int psi_outer_shift = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
    int psi_shift = DMnd * pSPARC->Nband_bandcomm * pSPARC->Nkpts_kptcomm;
    int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    t1 = MPI_Wtime();
    if (pSPARC->ACEFlag == 0) {
        dims[0] = pSPARC->npNdx; 
        dims[1] = pSPARC->npNdy; 
        dims[2] = pSPARC->npNdz;

        int *rhs_list_i, *rhs_list_j;
        rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
        rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
        assert(rhs_list_i != NULL && rhs_list_j != NULL);

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            psi_outer = pSPARC->psi_outer + spn_i * psi_outer_shift;
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb + spn_i * psi_shift;

            // Find the number of Poisson's equation required to be solved
            // Using the occupation threshold 1e-6
            int count = 0;
            for (i = 0; i < ncol; i++) {
                for (j = 0; j < Ns; j++) {
                    if (occ_outer[i] + occ_outer[j] > 1e-6) {
                        rhs_list_i[count] = i;
                        rhs_list_j[count] = j;
                        count++;
                    }
                }
            }
            num_rhs = count;
            if (num_rhs == 0) continue;            

            batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                            num_rhs : pSPARC->EXXMem_batch * size;
        
            NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
            rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
            Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
            assert(rhs != NULL && Vi != NULL);

            for (loop = 0; loop < NL; loop ++) {
                base = batch_num_rhs*loop;
                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    for (k = 0; k < DMnd; k++) {
                        rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd] * psi[k + i*DMnd];
                        // rhs[k + j*DMnd + i*DMnd*Ns] = psi_outer[k + j*DMnd] * pSPARC->Xorb[k + i*DMnd];
                    }
                }

                // Solve all Poisson's equation 
                poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, comm);

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    
                    occ_i = occ_outer[i + pSPARC->band_start_indx];
                    occ_j = occ_outer[j];

                    // TODO: use a temp array to reduce the MPI_Allreduce time to 1
                    temp = 0.0;
                    for (k = 0; k < DMnd; k++){
                        temp += rhs[k + (count-base)*DMnd] * Vi[k + (count-base)*DMnd];
                        // temp += rhs[k + j*DMnd + i*DMnd*Ns] * Vi[k + j*DMnd + i*DMnd*Ns];
                    }
                    if (size > 1)
                        MPI_Allreduce(MPI_IN_PLACE, &temp, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
                    pSPARC->Eexx += occ_i * occ_j * temp;
                }
            }

            free(rhs);
            free(Vi);
        }
        free(rhs_list_i);
        free(rhs_list_j);

        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        pSPARC->Eexx /= pSPARC->dV;

    } else {
        
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            int Ns_occ = pSPARC->Nstates_occ[spn_i];
            // printf("Ns_occ %d\n", Ns_occ);

            double *Xi_times_psi = (double *) calloc(Ns_occ * Ns_occ, sizeof(double));
            assert(Xi_times_psi != NULL);

            // perform matrix multiplication psi' * X using ScaLAPACK routines
            cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Ns_occ, Ns_occ, DMnd,
                        1.0, pSPARC->Xorb + spn_i * psi_shift, DMnd, pSPARC->Xi + spn_i * xi_shift, DMnd, 0.0, Xi_times_psi, Ns_occ);

            if (size > 1) {
                // sum over all processors in dmcomm
                MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Ns_occ*Ns_occ, 
                            MPI_DOUBLE, MPI_SUM, comm);
            }

            for (i = 0; i < Ns_occ; i++) {
                temp = 0.0;
                for (j = 0; j < Ns_occ; j++) {
                    temp += Xi_times_psi[i+j*Ns_occ] * Xi_times_psi[i+j*Ns_occ];
                }
                temp *= pSPARC->occ[i + Ns*spn_i];
                pSPARC->Eexx += temp;
            }

            free(Xi_times_psi);
        }
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    pSPARC->Eexx /= (pSPARC->Nspin + 0.0);
    pSPARC->Eexx *= -pSPARC->hyb_mixing;

    t2 = MPI_Wtime();
#ifdef DEBUG
if(!grank) 
    printf("\nEvaluating Exact exchange energy took: %.3f ms\nExact exchange energy %.6f.\n", (t2-t1)*1e3, pSPARC->Eexx);
#endif  
}



/**
 * @brief   Solving Poisson's equation using FFT or CG
 *          
 *          This function only works for solving Poisson's equation with real right hand side
 */
void poissonSolve(SPARC_OBJ *pSPARC, double *rhs, double *pois_FFT_const, 
                    int ncol, int DMnd, int *dims, double *Vi, MPI_Comm comm) 
{
    int i, j, k, lsize, lrank, size_s, ncolp;
    int *sendcounts, *sdispls, *recvcounts, *rdispls, **DMVertices, *ncolpp;
    int coord_comm[3], gridsizes[3], DNx, DNy, DNz, DNd, Nd, Nx, Ny, Nz, Ns, Ndc;
    double *rhs_loc, *Vi_loc, *rhs_loc_order, *Vi_loc_order, *f;

    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);
    Ns = pSPARC->Nstates;
    size_s = DMnd * ncol;                                                                     // it is DMnd * Nband in other parts
    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz; 
    Ndc = Nz * Ny * (Nx/2+1);
    ncolp = ncol / lsize + ((lrank < ncol % lsize) ? 1 : 0);
    /********************************************************************/

    if (lsize > 1){
        // variables for RHS storage
        rhs_loc = (double*) malloc(sizeof(double) * ncolp * Nd);
        rhs_loc_order = (double*) malloc(sizeof(double) * Nd * ncolp);

        // number of columns per proc
        ncolpp = (int*) malloc(sizeof(int) * lsize);

        // variables for alltoallv
        sendcounts = (int*) malloc(sizeof(int)*lsize);
        sdispls = (int*) malloc(sizeof(int)*lsize);
        recvcounts = (int*) malloc(sizeof(int)*lsize);
        rdispls = (int*) malloc(sizeof(int)*lsize);
        DMVertices = (int**) malloc(sizeof(int*)*lsize);
        assert(rhs_loc != NULL && rhs_loc_order != NULL && ncolpp != NULL && 
               sendcounts != NULL && sdispls != NULL && recvcounts != NULL && 
               rdispls != NULL && DMVertices!= NULL);

        for (k = 0; k < lsize; k++) {
            DMVertices[k] = (int*) malloc(sizeof(int)*6);
            assert(DMVertices[k] != NULL);
        }
        /********************************************************************/
        
        // separate equations to different processes in the dmcomm or kptcomm_topo                 
        for (i = 0; i < lsize; i++) {
            ncolpp[i] = ncol / lsize + ((i < ncol % lsize) ? 1 : 0);
        }

        // this part of codes copied from parallelization.c
        gridsizes[0] = Nx; gridsizes[1] = Ny; gridsizes[2] = Nz;
        // compute variables required by gatherv and scatterv
        for (i = 0; i < lsize; i++) {
            MPI_Cart_coords(comm, i, 3, coord_comm);
            // find size of distributed domain over comm
            DNx = block_decompose(gridsizes[0], dims[0], coord_comm[0]);
            DNy = block_decompose(gridsizes[1], dims[1], coord_comm[1]);
            DNz = block_decompose(gridsizes[2], dims[2], coord_comm[2]);
            DNd = DNx * DNy * DNz;
            // Here DMVertices [1][3][5] is not the same as they are in parallelization
            DMVertices[i][0] = block_decompose_nstart(gridsizes[0], dims[0], coord_comm[0]);
            DMVertices[i][1] = DNx;                                                                                     // stores number of nodes instead of coordinates of end nodes
            DMVertices[i][2] = block_decompose_nstart(gridsizes[1], dims[1], coord_comm[1]);
            DMVertices[i][3] = DNy;                                                                                     // stores number of nodes instead of coordinates of end nodes
            DMVertices[i][4] = block_decompose_nstart(gridsizes[2], dims[2], coord_comm[2]);
            DMVertices[i][5] = DNz;                                                                                     // stores number of nodes instead of coordinates of end nodes
        }

        sdispls[0] = 0;
        rdispls[0] = 0;
        for (i = 0; i < lsize; i++) {
            sendcounts[i] = ncolpp[i] * DMnd;
            recvcounts[i] = ncolp * DMVertices[i][1] * DMVertices[i][3] * DMVertices[i][5];
            if (i < lsize - 1) {
                sdispls[i+1] = sdispls[i] + sendcounts[i];
                rdispls[i+1] = rdispls[i] + recvcounts[i];
            }
        }
        /********************************************************************/

        MPI_Alltoallv(rhs, sendcounts, sdispls, MPI_DOUBLE, 
                        rhs_loc, recvcounts, rdispls, MPI_DOUBLE, comm);

        // rhs_full needs rearrangement
        rearrange_rhs((void *) rhs_loc, ncolp, DMVertices, rdispls, lsize, 
                        Nx, Ny, Nd, (void *) rhs_loc_order, sizeof(double));

        free(rhs_loc);
        // variable for local result Vi
        Vi_loc = (double*) malloc(sizeof(double)* Nd * ncolp);
        assert(Vi_loc != NULL);
    } else {
        // if the size of comm is 1, there is no need to scatter and rearrange the results
        rhs_loc_order = rhs;
        Vi_loc = Vi;
    }   

    if (pSPARC->EXXMeth_Flag == 0) {                                                // Solve in Fourier Space
        pois_fft(pSPARC, rhs_loc_order, pois_FFT_const, ncolp, Vi_loc);
    } else {
        f = malloc(sizeof(double) * ncolp * Nd);
        assert(f != NULL);
        poisson_RHS_local(pSPARC, rhs_loc_order, f, Nd, ncolp);
        pois_linearsolver(pSPARC, f, ncolp, Vi_loc);                                // Solve in Real Space
        free(f);
    }
    
    if (lsize > 1)  
        free(rhs_loc_order);

    if (lsize > 1) {
        Vi_loc_order = (double*) malloc(sizeof(double)* Nd * ncolp);
        assert(Vi_loc_order != NULL);

        // Vi_loc needs rearrangement
        rearrange_Vi((void *) Vi_loc, ncolp, DMVertices, lsize, 
                        Nx, Ny, Nd, (void *) Vi_loc_order, sizeof(double));

        MPI_Alltoallv(Vi_loc_order, recvcounts, rdispls, MPI_DOUBLE, 
                    Vi, sendcounts, sdispls, MPI_DOUBLE, comm);

        free(Vi_loc_order);
    }

    /********************************************************************/
    if (lsize > 1){
        free(Vi_loc);
        free(ncolpp);
        free(sendcounts);
        free(sdispls);
        free(recvcounts);
        free(rdispls);
        for (k = 0; k < lsize; k++) 
            free(DMVertices[k]);
        free(DMVertices);
    }
}



/**
 * @brief   Rearrange Vi after receiving from root comm
 *          
 *          Vi was ordered in continuous way and now is ordered in block-separated way
 *          Note: using unit_size to control whether it's double or double _Complex 
 */
void rearrange_Vi(void *Vi_full, int ncol, int **DMVertices, int size_comm, 
                    int Nx, int Ny, int Nd, void *Vi_full_order, int unit_size) {
#define Vi_full_(i,j,k, l) Vi_full_[(i) + (j) * (Nx)+ (k) * (Nx) * (Ny) + (l) * (Nd)]
    if (ncol == 0) return;
    int ii, i, j, k, l, t, p, seg, *coord;
    /********************************************************************/
    if (unit_size == 8) {
        double *Vi_full_order_ = (double *) Vi_full_order;
        double *Vi_full_ = (double *) Vi_full;
        p = 0;
        for (t = 0; t < size_comm; t++){
            coord = DMVertices[t];
            for (l = 0; l < ncol; l++) {
                for (k = coord[4]; k < coord[4] + coord[5]; k++)
                    for (j = coord[2]; j < coord[2] + coord[3]; j++)
                        for (i = coord[0]; i < coord[0] + coord[1]; i++) {
                            Vi_full_order_[p++] = Vi_full_(i, j, k, l);
                        }
            }
        }
    }

    if (unit_size == 16) {
        double _Complex *Vi_full_order_ = (double _Complex *) Vi_full_order;
        double _Complex *Vi_full_ = (double _Complex *) Vi_full;
        p = 0;
        for (t = 0; t < size_comm; t++){
            coord = DMVertices[t];
            for (l = 0; l < ncol; l++) {
                for (k = coord[4]; k < coord[4] + coord[5]; k++)
                    for (j = coord[2]; j < coord[2] + coord[3]; j++)
                        for (i = coord[0]; i < coord[0] + coord[1]; i++) {
                            Vi_full_order_[p++] = Vi_full_(i, j, k, l);
                        }
            }
        }
    }
#undef Vi_full_
}


/**
 * @brief   Rearrange rhs after receiving from other comms
 *          
 *          Vi was ordered in block-separated way and now is ordered in continuous way
 *          Note: using unit_size to control whether it's double or double _Complex 
 */
void rearrange_rhs(void *rhs_full, int ncol, int **DMVertices, int *displs, int size_comm, 
                    int Nx, int Ny, int Nd, void *rhs_full_order, int unit_size) {
#define rhs_full_order_(i,j,k,l) rhs_full_order_[(i) + (j) * (Nx)+ (k) * (Nx) * (Ny) + (l) * (Nd)]
    if (ncol == 0) return;
    int ii, i, j, k, l, t, p, q, *coord, start, end, seg;
    /********************************************************************/

    if (unit_size == 8) {
        double *rhs_full_order_ = (double *) rhs_full_order;
        double *rhs_full_ = (double *) rhs_full;
        for (t = 0; t < size_comm; t++) {
            coord = DMVertices[t];
            seg = coord[1] * coord[3] * coord[5];
            start = displs[t];
            for (ii = start; ii < start + seg; ii++) {
                p = ii - start;                     // relative coordinates
                q = p % coord[1];                   // relative x
                i = coord[0] + q;
                p -= q; p /= coord[1];
                q = p % coord[3];                   // relative y
                j = coord[2] + q;
                p -= q; p /= coord[3];              // p is relative z
                k = coord[4] + p;
                for (l = 0; l < ncol; l++) {
                    rhs_full_order_(i, j, k, l) = rhs_full_[ii + l * seg];
                }
            }
        }
    }

    if (unit_size == 16) {
        double _Complex *rhs_full_order_ = (double _Complex*) rhs_full_order;
        double _Complex *rhs_full_ = (double _Complex*) rhs_full;
        for (t = 0; t < size_comm; t++) {
            coord = DMVertices[t];
            seg = coord[1] * coord[3] * coord[5];
            start = displs[t];
            for (ii = start; ii < start + seg; ii++) {
                p = ii - start;                     // relative coordinates
                q = p % coord[1];                   // relative x
                i = coord[0] + q;
                p -= q; p /= coord[1];
                q = p % coord[3];                   // relative y
                j = coord[2] + q;
                p -= q; p /= coord[3];              // p is relative z
                k = coord[4] + p;
                for (l = 0; l < ncol; l++) {
                    rhs_full_order_(i, j, k, l) = rhs_full_[ii + l * seg];
                }
            }
        }
    }
#undef rhs_full_order_
}


/**
 * @brief   preprocessing RHS of Poisson's equation depends on the method for Exact Exchange
 */
void poisson_RHS_local(SPARC_OBJ *pSPARC, double *rhs, double *f, int Nd, int ncolp) {
    if (ncolp == 0) return;
    int i;
    double *d_cor;
    /********************************************************************/
    
    for (i = 0; i < Nd * ncolp; i++) {
        f[i] = rhs[i] * (4.0 * M_PI);                                       // linear solver solves equation -Lap x = 4pi * rhs
    }
    d_cor = (double *)malloc( Nd * ncolp * sizeof(double) );
    assert(d_cor != NULL);
    MultipoleExpansion_phi_local(pSPARC, f, d_cor, Nd, ncolp);
    for (i = 0; i < Nd * ncolp; i++) f[i] -= d_cor[i];
    free(d_cor);
}


/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs_loc_order     complete RHS of poisson's equations without parallelization. 
 * @param pois_FFT_const    constant for solving possion's equations
 * @param ncolp             Number of poisson's equations to be solved.
 * @param Vi_loc            complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_fft(SPARC_OBJ *pSPARC, double *rhs_loc_order, double *pois_FFT_const, int ncolp, double *Vi_loc) {
    if (ncolp == 0) return;
    int i, j, Nd, Nx, Ny, Nz, Ndc;
    double _Complex *rhs_bar;

    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz; 
    Ndc = Nz * Ny * (Nx/2+1);
    rhs_bar = (double _Complex*) malloc(sizeof(double _Complex) * Ndc * ncolp);
    assert(rhs_bar != NULL);
    /********************************************************************/

    // FFT
#if defined(USE_MKL)
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 

    for (i = 0; i < ncolp; i++)
        MKL_MDFFT_real(rhs_loc_order + i * Nd, dim_sizes, strides_out, rhs_bar + i * Ndc);
#elif defined(USE_FFTW)
    int dim_sizes[3] = {Nz, Ny, Nx};

    for (i = 0; i < ncolp; i++)
        FFTW_MDFFT_real(dim_sizes, rhs_loc_order + i * Nd, rhs_bar + i * Ndc);
#endif

    // multiplied by alpha
    for (j = 0; j < ncolp; j++) {
        for (i = 0; i < Ndc; i++) {
            rhs_bar[i + j*Ndc] = creal(rhs_bar[i + j*Ndc]) * pois_FFT_const[i] 
                                + (cimag(rhs_bar[i + j*Ndc]) * pois_FFT_const[i]) * I;
        }
    }

    // iFFT
#if defined(USE_MKL)
    for (i = 0; i < ncolp; i++)
        MKL_MDiFFT_real(rhs_bar + i * Ndc, dim_sizes, strides_out, Vi_loc + i * Nd);
#elif defined(USE_FFTW)
    for (i = 0; i < ncolp; i++)
        FFTW_MDiFFT_real(dim_sizes, rhs_bar + i * Ndc, Vi_loc + i * Nd);
#endif

    free(rhs_bar);
}


/**
 * @brief   Solve Poisson's equation using linear solver (e.g. CG) in Real Space
 */
void pois_linearsolver(SPARC_OBJ *pSPARC, double *rhs_loc_order, int ncolp, double *Vi_loc) {
    if (ncolp == 0) return;
    int i, Nd;
    int dims[3] = {1,1,1}, periods[3] = {0,0,0}, DMVertices[6];
    MPI_Comm self_topo;
    Nd = pSPARC->Nd;
    DMVertices[0] = 0;
    DMVertices[2] = 0;
    DMVertices[4] = 0;
    DMVertices[1] = pSPARC->Nx - 1;
    DMVertices[3] = pSPARC->Ny - 1;
    DMVertices[5] = pSPARC->Nz - 1;
    /********************************************************************/

    // Create a communicator for each single processor
    MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &self_topo); // 0 is to reorder rank

    void (*Ax)(const SPARC_OBJ *, const int, const int *, const int, const double, double *, double *, MPI_Comm) = Lap_vec_mult; // Lap_vec_mult is defined in lapVecRoutines.c

    for (i = 0; i < ncolp*Nd; i++) Vi_loc[i] = 0;                    // TODO: use better initial guess

    for (i = 0; i < ncolp; i++) 
        CG(pSPARC, Ax, Nd, Nd, DMVertices, Vi_loc + i*Nd, 
            rhs_loc_order + i*Nd, 1e-8, pSPARC->MAXIT_POISSON, self_topo);

    MPI_Comm_free(&self_topo);
}


/**
 * @brief   preprocessing RHS of Poisson's equation by multipole expansion on single process
 */
void MultipoleExpansion_phi_local(SPARC_OBJ *pSPARC, double *f, double *d_cor, int Nd, int ncolp) {
#define d_cor(i,j,k,n) d_cor[(n)*Nd+(k)*Nx*Ny+(j)*Nx+(i)]
#define phi(i,j,k,n) phi[(n)*nd_phi+(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]
    
    int LMAX = 6, l, m, n, i, j, k, p, count, index, Q_len, Nx, Ny, Nz,
        FDn, nbr_i, is, ie, js, je,
        ks, ke, nx_cor, ny_cor, nz_cor, nd_cor, is_phi, ie_phi, js_phi, je_phi,
        ks_phi, ke_phi, nx_phi, ny_phi, nz_phi, nd_phi, i_phi, j_phi, k_phi,
        i_DM, j_DM, k_DM, DMCorVert[6][6];
    double *Qlm, Lx, Ly, Lz, *r_pos_x, *r_pos_y, *r_pos_z, *r_pos_r, *r_pow_l,
           *Ylm, *phi, x, y, z, r, x2, y2, z2;
    
    FDn = pSPARC->order / 2;

    Lx = pSPARC->range_x;
    Ly = pSPARC->range_y;
    Lz = pSPARC->range_z;
    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    /********************************************************************/

    /* find multipole moments Qlm */
    r_pos_x = (double *)malloc( sizeof(double) * Nd );
    r_pos_y = (double *)malloc( sizeof(double) * Nd );
    r_pos_z = (double *)malloc( sizeof(double) * Nd );
    r_pos_r = (double *)malloc( sizeof(double) * Nd );
    assert(r_pos_x != NULL && r_pos_y != NULL && 
           r_pos_z != NULL && r_pos_r != NULL);

    // find distance between the center of the domain and finite-difference grids
    count = 0; 
    for (k = 0; k < Nz; k++) {
        z = k * pSPARC->delta_z - Lz/2.0; 
        z2 = z * z;
        for (j = 0; j < Ny; j++) {
            y = j * pSPARC->delta_y - Ly/2.0; 
            y2 = y * y;
            for (i = 0; i < Nx; i++) {
                x = i * pSPARC->delta_x - Lx/2.0;
                x2 = x * x; 
                r_pos_x[count] = x;
                r_pos_y[count] = y;
                r_pos_z[count] = z;
                r_pos_r[count] = sqrt(x2 + y2 + z2);
                count++;
            }
        }
    }    

    Ylm = (double *)malloc( sizeof(double) * Nd );
    Q_len = (LMAX+1)*(LMAX+1);
    Qlm = (double *)calloc( Q_len * ncolp, sizeof(double) );
    r_pow_l = (double *)malloc( sizeof(double) * Nd );
    assert(Ylm != NULL && Qlm != NULL && r_pow_l != NULL);

    for (i = 0; i < Nd; i++) r_pow_l[i] = 1.0; // init to 1
    index = 0;
    for (l = 0; l <= LMAX; l++) {
        // find r^l
        if (l) {
            for (i = 0; i < Nd; i++) r_pow_l[i] *= r_pos_r[i];
        }
        for (m = -l; m <= l; m++) {
            RealSphericalHarmonic(Nd, r_pos_x, r_pos_y, r_pos_z, r_pos_r, l, m, Ylm);
            for (j = 0; j < ncolp; j++) {
                Qlm[index + j * Q_len] = 0.0;
                for (i = 0; i < Nd; i++) 
                    Qlm[index + j * Q_len] += r_pow_l[i] * f[i + j * Nd] * Ylm[i];
                Qlm[index + j * Q_len] *= pSPARC->dV;
            }
            index++;
        }
    } 
    free(r_pos_x); free(r_pos_y); free(r_pos_z); free(r_pos_r); free(Ylm); free(r_pow_l);


    /* find "charge correction" (boudary correction) */
    // define the “correction domain” which contributes to the charge correction. i.e. 0 to FDn-1 and
    // nx-FDn nx-1 in each direction.
    DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
    DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;  
    DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
    DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
    DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
    DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

    for (i = 0; i < Nd*ncolp; i++) d_cor[i] = 0.0; // init correction to 0
       
    // find correction contribution from each side
    for (nbr_i = 0; nbr_i < 6; nbr_i++) {
        is = DMCorVert[nbr_i][0];
        ie = DMCorVert[nbr_i][1];
        js = DMCorVert[nbr_i][2];
        je = DMCorVert[nbr_i][3];
        ks = DMCorVert[nbr_i][4];
        ke = DMCorVert[nbr_i][5];
        
        // find the region of phi that have contribution to the correction domain
        is_phi = is; ie_phi = ie;
        js_phi = js; je_phi = je;
        ks_phi = ks; ke_phi = ke;
        switch (nbr_i) {
            case 0:
                is_phi = is - FDn; ie_phi = -1; break;
            case 1:
                is_phi = Nx; ie_phi = ie + FDn; break;
            case 2:
                js_phi = js - FDn; je_phi = -1; break;
            case 3:
                js_phi = Ny; je_phi = je + FDn; break;
            case 4:
                ks_phi = ks - FDn; ke_phi = -1; break;
            case 5:
                ks_phi = Nz; ke_phi = ke + FDn; break; 
        }
        
        nx_phi = ie_phi - is_phi + 1;
        ny_phi = je_phi - js_phi + 1;
        nz_phi = ke_phi - ks_phi + 1;
        nd_phi = nx_phi * ny_phi * nz_phi;
        
        // calculate electrostatic potential "phi" inside
        phi = (double *)calloc( nd_phi * ncolp, sizeof(double) );
        Ylm = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_x = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_y = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_z = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_r = (double *)malloc( sizeof(double) * nd_phi );
        r_pow_l = (double *)malloc( sizeof(double) * nd_phi );
        assert(phi != NULL && Ylm != NULL && r_pos_x != NULL && r_pos_y != NULL 
            && r_pos_z != NULL && r_pos_r != NULL && r_pow_l != NULL);

        count = 0;
        for (k = 0; k < nz_phi; k++) {
            z = (k + ks_phi) * pSPARC->delta_z - Lz*0.5; 
            for (j = 0; j < ny_phi; j++) {
                y = (j + js_phi) * pSPARC->delta_y - Ly*0.5;
                for (i = 0; i < nx_phi; i++) {
                    x = (i + is_phi) * pSPARC->delta_x - Lx*0.5;
                    r = sqrt(x * x + y * y + z * z);
                    r_pos_x[count] = x;
                    r_pos_y[count] = y;
                    r_pos_z[count] = z;
                    r_pos_r[count] = r;
                    count++;
                }
            }
        } 
        
        for (i = 0; i < nd_phi; i++) r_pow_l[i] = 1.0; // init r_pow_l to 1
        index = 0;
        for (l = 0; l <= LMAX; l++) {
            // find r^(l+1)
            for (i = 0; i < nd_phi; i++) r_pow_l[i] *= r_pos_r[i];
            for (m = -l; m <= l; m++) {
                RealSphericalHarmonic(nd_phi, r_pos_x, r_pos_y, r_pos_z, r_pos_r, l, m, Ylm);
                for (j = 0; j < ncolp; j++) {
                    for (i = 0; i < nd_phi; i++) 
                        phi[i + j* nd_phi] += 1.0 / ((2*l+1) * r_pow_l[i]) * Ylm[i] * Qlm[index + j * Q_len];
                }
                index++;
            }
        } 
        free(Ylm); free(r_pos_x); free(r_pos_y); free(r_pos_z); free(r_pos_r); free(r_pow_l);

        // calculate the correction "d_cor"
        for (n = 0; n < ncolp; n++) {
            for (k = ks; k <= ke; k++) {
                k_phi = k - ks_phi;
                for (j = js; j <= je; j++) {
                    j_phi = j - js_phi;
                    for (i = is; i <= ie; i++) {
                        i_phi = i - is_phi;
                        for (p = 1; p <= FDn; p++) {
                            switch (nbr_i) {
                                case 0:
                                    if ((i-p) < 0) 
                                        d_cor(i,j,k,n) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi,n);
                                    break;
                                case 1:
                                    if ((i+p) >= Nx) 
                                        d_cor(i,j,k,n) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi,n);
                                    break;
                                case 2:
                                    if ((j-p) < 0) 
                                        d_cor(i,j,k,n) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi,n);
                                    break;
                                case 3:
                                    if ((j+p) >= Ny) 
                                        d_cor(i,j,k,n) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi,n);
                                    break;
                                case 4:
                                    if ((k-p) < 0) 
                                        d_cor(i,j,k,n) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p,n);
                                    break;
                                case 5:
                                    if ((k+p) >= Nz) 
                                        d_cor(i,j,k,n) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p,n);
                                    break;
                            }
                        }
                    }
                }
            }
        }
        
        free(phi);
    }
    free(Qlm);

#undef d_cor
#undef phi
}



/**
 * @brief   Transfer vectors from dmcomm to kptcomm_topo for Lancozs algorithm
 *
 *          Used to transfer psi_outer in case of no-ACE method and transfer 
 *          Xi (of ACE operator) in case of ACE method from dmcomm to kptcomm_topo
 */
void Transfer_dmcomm_to_kptcomm_topo(SPARC_OBJ *pSPARC, int ncols, double *psi_outer, double *psi_outer_kptcomm_topo) {
    int i, gridsizes[3], sdims[3], rdims[3];

    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx;         
    sdims[1] = pSPARC->npNdy;         
    sdims[2] = pSPARC->npNdz; 
    rdims[0] = pSPARC->npNdx_kptcomm; 
    rdims[1] = pSPARC->npNdy_kptcomm; 
    rdims[2] = pSPARC->npNdz_kptcomm;
    /********************************************************************/

    // Transferring all bands of psi_outer to kptcomm_topo for Lanczos
    for (i = 0; i < ncols; i++) {
        D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, 
            pSPARC->DMVertices_dmcomm, psi_outer + pSPARC->Nd_d_dmcomm * i,
            pSPARC->DMVertices_kptcomm, psi_outer_kptcomm_topo + pSPARC->Nd_d_kptcomm * i,
            pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL, sdims, 
            pSPARC->kptcomm_topo, rdims, 
            pSPARC->kptcomm);
    }
}



/**
 * @brief   Gather psi_outers in other bandcomms
 *
 *          The default comm is blacscomm
 */
void gather_psi_occ_outer(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer) 
{
    int i, j, k, grank, lrank, lsize, size_s, Ns, DMnd, Nband, spn_i;
    int sendcount, *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    size_s = DMnd * Nband;
    Ns = pSPARC->Nstates;

    int DMndNband = DMnd * Nband;
    int DMndNs = DMnd * Ns;
    int Nstotal = Ns * pSPARC->Nspin_spincomm;
    int shift = pSPARC->band_start_indx * DMnd;
    // Save orbitals and occupations and to construct exact exchange operator
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
        for (i = 0; i < DMndNband; i++) 
            pSPARC->psi_outer[i + shift + spn_i*DMndNs] = pSPARC->Xorb[i + spn_i*DMndNband];
    for (i = 0; i < Nstotal; i++) 
        pSPARC->occ_outer[i] = pSPARC->occ[i];
    /********************************************************************/
    
    if (lsize > 1) {
        recvcounts = (int*) malloc(sizeof(int)* lsize);
        displs = (int*) malloc(sizeof(int)* lsize);
        assert(recvcounts != NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (Ns - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < lsize; i++){
            recvcounts[i] = (i < (Ns / NB) ? NB : (i == (Ns / NB) ? (Ns % NB) : 0)) * DMnd;
            if (i != (lsize-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
            MPI_Allgatherv(MPI_IN_PLACE, 1, MPI_DOUBLE, psi_outer + spn_i*DMndNs, 
                recvcounts, displs, MPI_DOUBLE, pSPARC->blacscomm);   
        
        free(recvcounts);
        free(displs); 
    }

    /********************************************************************/
    if (pSPARC->flag_kpttopo_dm && pSPARC->ACEFlag == 0) {
        int rank_kptcomm_topo;
        int NsNkNsp = pSPARC->Nstates * pSPARC->Nkpts_sym * pSPARC->Nspin_spincomm;
        MPI_Comm_rank(pSPARC->kptcomm_topo, &rank_kptcomm_topo);
        if (pSPARC->flag_kpttopo_dm_type == 1) {
            if (!rank_kptcomm_topo)
                MPI_Bcast(pSPARC->occ_outer, NsNkNsp, MPI_DOUBLE, MPI_ROOT, pSPARC->kpttopo_dmcomm_inter);
            else
                MPI_Bcast(pSPARC->occ_outer, NsNkNsp, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kpttopo_dmcomm_inter);
        } else {
            MPI_Bcast(pSPARC->occ_outer, NsNkNsp, MPI_DOUBLE, 0, pSPARC->kpttopo_dmcomm_inter);
        }
    }
}


/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 *          Auxiliary function - Method by Gygi and Baldereschi
 *          DOI: 10.1103/PhysRevB.34.4405
 */
void compute_pois_fft_const(SPARC_OBJ *pSPARC) {
#define alpha(i,j,k) pSPARC->pois_FFT_const[(k)*Nxy+(j)*Nxh+(i)]
#define alpha1(i,j,k) pSPARC->pois_FFT_const_stress[(k)*Nxy+(j)*Nxh+(i)]
#define alpha2(i,j,k) pSPARC->pois_FFT_const_stress2[(k)*Nxy+(j)*Nxh+(i)]
#define beta(i,j,k) pSPARC->pois_FFT_const_press[(k)*Nxy+(j)*Nxh+(i)]
    int i, j, k, Nx, Nxh, Ny, Nz, Ndc, Nxy;
    double L1, L2, L3, V, R_c, G[3], g[3], G2, omega, omega2;

    Nx = pSPARC->Nx;
    Nxh = pSPARC->Nx / 2 + 1;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    Ndc = Nz * Ny * Nxh;
    Nxy = Ny * Nxh;

    // When BC is Dirichelet, one more FD node is incldued.
    // Real length of each side has to be added by 1 mesh length.
    L1 = pSPARC->delta_x * Nx;
    L2 = pSPARC->delta_y * Ny;
    L3 = pSPARC->delta_z * Nz;

    V =  L1 * L2 * L3 * pSPARC->Jacbdet * pSPARC->Nkpts_hf;       // Nk_hf * unit cell volume
    R_c = pow(3*V/(4*M_PI),(1.0/3));
    omega = pSPARC->hyb_range_fock;
    omega2 = omega * omega;

    pSPARC->pois_FFT_const = (double *)malloc(sizeof(double) * Ndc);
    assert(pSPARC->pois_FFT_const != NULL);
    // allocate space for stress
    if (pSPARC->Calc_stress == 1) {
        pSPARC->pois_FFT_const_stress = (double *)malloc(sizeof(double) * Ndc);
        assert(pSPARC->pois_FFT_const_stress != NULL);
        if (pSPARC->EXXDiv_Flag == 0) {
            pSPARC->pois_FFT_const_stress2 = (double *)malloc(sizeof(double) * Ndc);
            assert(pSPARC->pois_FFT_const_stress2 != NULL);
        }
    } else if (pSPARC->Calc_pres == 1) {
        if (pSPARC->EXXDiv_Flag != 0) {
            pSPARC->pois_FFT_const_press = (double *)malloc(sizeof(double) * Ndc);
            assert(pSPARC->pois_FFT_const_press != NULL);
        }
    }
    /********************************************************************/

    // spherical truncation
    if (pSPARC->EXXDiv_Flag == 0) {
        for (k = 0; k < Nz; k++) {
            for (j = 0; j < Ny; j++) {
                for (i = 0; i < Nxh; i++) {
                    // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                    g[0] = g[1] = g[2] = 0.0;
                    G[0] = i*2*M_PI/L1;
                    G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                    G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                    matrixTimesVec_3d(pSPARC->lapcT, G, g);
                    G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                    double x = R_c*sqrt(G2);
                    if (fabs(G2) > 1e-4) {
                        alpha(i,j,k) = 4*M_PI*(1-cos(x))/G2;
                        if (pSPARC->Calc_stress == 1) {
                            double G4 = G2*G2;
                            alpha1(i,j,k) = 4*M_PI*( 1-cos(x)- x/2*sin(x) )/G4;
                            alpha2(i,j,k) = 4*M_PI*( x/2*sin(x) )/G2/3;
                        }
                    } else {
                        alpha(i,j,k) = 2*M_PI*(R_c * R_c);
                        if (pSPARC->Calc_stress == 1) {
                            alpha1(i,j,k) = 4*M_PI*pow(R_c,4)/24;
                            alpha2(i,j,k) = 4*M_PI*( R_c*R_c/2 )/3;
                        }
                    }
                }
            }
        }
        return;
    }

    // auxiliary function
    if (pSPARC->EXXDiv_Flag == 1) {
        for (k = 0; k < Nz; k++) {
            for (j = 0; j < Ny; j++) {
                for (i = 0; i < Nxh; i++) {
                    // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                    g[0] = g[1] = g[2] = 0.0;
                    G[0] = i*2*M_PI/L1;
                    G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                    G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                    matrixTimesVec_3d(pSPARC->lapcT, G, g);
                    G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                    double x = -0.25/omega2*G2;
                    if (fabs(G2) > 1e-4) {
                        if (omega > 0) {
                            alpha(i,j,k) = 4*M_PI/G2 * (1 - exp(x));
                            if (pSPARC->Calc_stress == 1) {
                                double G4 = G2*G2;
                                alpha1(i,j,k) = 4*M_PI*(1 - exp(x)*(1-x))/G4 /4;
                            } else if (pSPARC->Calc_pres == 1) {
                                beta(i,j,k) = 4*M_PI*(1 - exp(x)*(1-x))/G2 /4;
                            }
                        } else {
                            alpha(i,j,k) = 4*M_PI/G2;
                            if (pSPARC->Calc_stress == 1) {
                                double G4 = G2*G2;
                                alpha1(i,j,k) = 4*M_PI/G4 /4;
                            } else if (pSPARC->Calc_pres == 1) {
                                beta(i,j,k) = 4*M_PI/G2 /4;
                            }
                        }
                    } else {
                        if (omega > 0) {
                            alpha(i,j,k) = 4*M_PI*(pSPARC->const_aux + 0.25/omega2);
                            if (pSPARC->Calc_stress == 1) {
                                alpha1(i,j,k) = 0;
                            } else if (pSPARC->Calc_pres == 1) {
                                beta(i,j,k) = 0;
                            }
                        } else {
                            alpha(i,j,k) = 4*M_PI*pSPARC->const_aux;
                            if (pSPARC->Calc_stress == 1) {
                                alpha1(i,j,k) = 0;
                            } else if (pSPARC->Calc_pres == 1) {
                                beta(i,j,k) = 0;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // ERFC short ranged screened 
    if (pSPARC->EXXDiv_Flag == 2) {
        for (k = 0; k < Nz; k++) {
            for (j = 0; j < Ny; j++) {
                for (i = 0; i < Nxh; i++) {
                    // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                    g[0] = g[1] = g[2] = 0.0;
                    G[0] = i*2*M_PI/L1;
                    G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                    G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                    matrixTimesVec_3d(pSPARC->lapcT, G, g);
                    G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                    double x = -G2/4.0/omega2;
                    if (fabs(G2) > 1e-4) {
                        alpha(i,j,k) = 4*M_PI*(1-exp(x))/G2;
                        if (pSPARC->Calc_stress == 1) {
                            double G4 = G2*G2;
                            alpha1(i,j,k) = 4*M_PI*( 1-exp(x)*(1-x) )/G4;
                        } else if (pSPARC->Calc_pres == 1) {
                            beta(i,j,k) = 4*M_PI*( 1-exp(x)*(1-x) )/G2;
                        }
                    } else {
                        alpha(i,j,k) = M_PI/omega2;
                        if (pSPARC->Calc_stress == 1) {
                            alpha1(i,j,k) = 0;
                        } else if (pSPARC->Calc_pres == 1) {
                            beta(i,j,k) = 0;
                        }
                    }
                }
            }
        }
        return;
    }

#undef alpha
#undef alpha1
#undef alpha2
#undef beta
}

/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 * 
 *          Note: using Finite difference approximation of G2
 *          Note: it's not used in the code. Only use exact G2. 
 */
void compute_pois_fft_const_FD(SPARC_OBJ *pSPARC) 
{
#define alpha(i,j,k) alpha[(k)*Nxy+(j)*Nxh+(i)]
    int FDn, i, j, k, p, Nx, Nxh, Ny, Nz, Ndc, Nxy;
    double *w2_x, *w2_y, *w2_z, V, R_c, *alpha;

    Nx = pSPARC->Nx;
    Nxh = pSPARC->Nx / 2 + 1;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    Ndc = Nz * Ny * Nxh;
    Nxy = Ny * Nxh;
    FDn = pSPARC->order / 2;
    // scaled finite difference coefficients
    w2_x = pSPARC->D2_stencil_coeffs_x;
    w2_y = pSPARC->D2_stencil_coeffs_y;
    w2_z = pSPARC->D2_stencil_coeffs_z;

    pSPARC->pois_FFT_const = (double *)malloc(sizeof(double) * Ndc);
    assert(pSPARC->pois_FFT_const != NULL);
    alpha = pSPARC->pois_FFT_const;
    /********************************************************************/

    for (k = 0; k < Nz; k++) {
        for (j = 0; j < Ny; j++) {
            for (i = 0; i < Nxh; i++) {
                alpha(i,j,k) = w2_x[0] + w2_y[0] + w2_z[0];
                for (p = 1; p < FDn + 1; p++){
                    alpha(i,j,k) += 2*(w2_x[p]*cos(2*M_PI/Nx*p*i) 
                        + w2_y[p]*cos(2*M_PI/Ny*p*j) + w2_z[p]*cos(2*M_PI/Nz*p*k));
                }
            }
        }
    }
    alpha[0] = -1;                                                  // For taking sqrt, will change in the end 
    V = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z;        // Volume of the domain
    R_c = pow(3*V/(4*M_PI),(1.0/3));

    // For singularity issue, use (1-cos(R_c*sqrt(G^2)))/G^2
    for (k = 0; k < Nz; k++) {
        for (j = 0; j < Ny; j++) {
            for (i = 0; i < Nxh; i++) {
                alpha(i,j,k) = -4*M_PI*(1-cos(R_c*sqrt(-alpha(i,j,k))))/alpha(i,j,k);
            }
        }
    }
    alpha[0] = 2*M_PI*(R_c * R_c);
#undef alpha
}


/**
 * @brief   Create communicators between k-point topology and dmcomm.
 * 
 * Notes:   Occupations are only correct within all dmcomm processes.
 *          When not using ACE method, this communicator might be required. 
 */
void create_kpttopo_dmcomm_inter(SPARC_OBJ *pSPARC) 
{
    int i, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (pSPARC->kptcomm_topo != MPI_COMM_NULL && pSPARC->ACEFlag == 0) {
        int nproc_kptcomm;
        int nproc_kptcomm_topo = pSPARC->npNdx_kptcomm * pSPARC->npNdy_kptcomm * pSPARC->npNdz_kptcomm;
        MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);

        int size_bandcomm = nproc_kptcomm / pSPARC->npband;
        int npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
        int *ndm_list = (int *) calloc(sizeof(int), nproc_kptcomm_topo);
        assert(ndm_list != NULL);

    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif
        int count_ndm = 0;
        for (i = 0; i < nproc_kptcomm_topo; i++) {
            if (i >= pSPARC->npband * size_bandcomm || i % size_bandcomm >= npNd) 
                ndm_list[count_ndm++] = i;
        }
        if (count_ndm == 0) {
            pSPARC->kpttopo_dmcomm_inter = MPI_COMM_NULL;
            pSPARC->flag_kpttopo_dm = 0;
        } else {
            pSPARC->flag_kpttopo_dm = 1;
            MPI_Group kpttopo_group, kpttopo_group_excl, kpttopo_group_incl;
            MPI_Comm kpttopo_excl, kpttopo_incl;
            MPI_Comm_group(pSPARC->kptcomm_topo, &kpttopo_group);
            MPI_Group_incl(kpttopo_group, count_ndm, ndm_list, &kpttopo_group_excl);
            MPI_Group_excl(kpttopo_group, count_ndm, ndm_list, &kpttopo_group_incl);

            MPI_Comm_create_group(pSPARC->kptcomm_topo, kpttopo_group_excl, 110, &kpttopo_excl);
            MPI_Comm_create_group(pSPARC->kptcomm_topo, kpttopo_group_incl, 110, &kpttopo_incl);
            // yong excl zhiyao yige list
            if (kpttopo_excl != MPI_COMM_NULL) {
                MPI_Intercomm_create(kpttopo_excl, 0, pSPARC->kptcomm_topo, 0, 111, &pSPARC->kpttopo_dmcomm_inter);
                pSPARC->flag_kpttopo_dm_type = 2;               // - recv
            } else {
                MPI_Intercomm_create(kpttopo_incl, 0, pSPARC->kptcomm_topo, ndm_list[0], 111, &pSPARC->kpttopo_dmcomm_inter);
                pSPARC->flag_kpttopo_dm_type = 1;               // - send
            }

            if (kpttopo_excl != MPI_COMM_NULL)
                MPI_Comm_free(&kpttopo_excl);
            if (kpttopo_incl != MPI_COMM_NULL)
                MPI_Comm_free(&kpttopo_incl);
            MPI_Group_free(&kpttopo_group);
            MPI_Group_free(&kpttopo_group_excl);
            MPI_Group_free(&kpttopo_group_incl);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!rank && count_ndm > 0) {
            printf("\nThere are %d processes need to get correct occupations in each kptcomm_topo.\n",count_ndm);
            printf("\n--set up kpttopo_dmcomm_inter took %.3f ms\n",(t2-t1)*1000);
        }
    #endif  

        free(ndm_list);
    } else {
        pSPARC->kpttopo_dmcomm_inter = MPI_COMM_NULL;
        pSPARC->flag_kpttopo_dm = 0;
    }
}

/**
 * @brief   Compute the coefficient for G = 0 term in auxiliary function method
 * 
 *          Note: Using QE's method. Not the original one by Gygi 
 */
void auxiliary_constant(SPARC_OBJ *pSPARC) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int Nx, Ny, Nz, i, j, k, ihf, jhf, khf, Kx_hf, Ky_hf, Kz_hf;
    double L1, L2, L3, V, alpha, scaled_intf, sumfGq;
    double tpiblx, tpibly, tpiblz, q[3], g[3], G[3], modGq, omega, omega2;

    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;

    Kx_hf = pSPARC->Kx_hf;
    Ky_hf = pSPARC->Ky_hf;
    Kz_hf = pSPARC->Kz_hf;

    // When BC is Dirichelet, one more FD node is incldued.
    // Real length of each side has to be added by 1 mesh length.
    L1 = pSPARC->delta_x * Nx;
    L2 = pSPARC->delta_y * Ny;
    L3 = pSPARC->delta_z * Nz;

    tpiblx = 2 * M_PI / L1;
    tpibly = 2 * M_PI / L2;
    tpiblz = 2 * M_PI / L3;

    int N[3] = {Nx, Ny, Nz};
    double tpibl[3] = {tpiblx, tpibly, tpiblz};
    int nkpt[3] = {pSPARC->Kx, pSPARC->Ky, pSPARC->Kz};
    double ecut = ecut_estimate(pSPARC->delta_x, pSPARC->delta_y, pSPARC->delta_z);
    alpha = 10.0/(2.0*ecut);

#ifdef DEBUG
    if(!rank)
        printf("Ecut estimation is %.2f Ha (%.2f Ry) and alpha within auxiliary function is %.6f\n", ecut, 2*ecut, alpha);
#endif
    
    V =  L1 * L2 * L3 * pSPARC->Jacbdet;    // volume of unit cell
    scaled_intf = V*pSPARC->Nkpts_hf/(4*M_PI*sqrt(M_PI*alpha));
    omega = pSPARC->hyb_range_fock;
    omega2 = omega * omega;

    sumfGq = 0;
    for (khf = 0; khf < Kz_hf; khf++) {
        for (jhf = 0; jhf < Ky_hf; jhf++) {
            for (ihf = 0; ihf < Kx_hf; ihf++) {
                q[0] = ihf * tpiblx / Kx_hf;
                q[1] = jhf * tpibly / Ky_hf;
                q[2] = khf * tpiblz / Kz_hf;
                
                for (k = 0; k < Nz; k++) {
                    for (j = 0; j < Ny; j++) {
                        for (i = 0; i < Nx; i++) {
                            // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                            g[0] = g[1] = g[2] = 0.0;
                            G[0] = (i < Nx/2+1) ? (i*tpiblx) : ((i-Nx)*tpiblx);
                            G[1] = (j < Ny/2+1) ? (j*tpibly) : ((j-Ny)*tpibly);
                            G[2] = (k < Nz/2+1) ? (k*tpiblz) : ((k-Nz)*tpiblz);

                            G[0] += q[0];
                            G[1] += q[1];
                            G[2] += q[2];
                            matrixTimesVec_3d(pSPARC->lapcT, G, g);
                            modGq = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                            
                            if (modGq > 1e-8) {
                                if (omega < 0)
                                    sumfGq += exp(-alpha*modGq)/modGq;
                                else
                                    sumfGq += exp(-alpha*modGq)/modGq*(1-exp(-modGq/4.0/omega2));
                            }
                        }
                    }
                }
            }
        }
    }
    

    if (omega < 0) {
        pSPARC->const_aux = scaled_intf + alpha - sumfGq;    
    } else {
        sumfGq += 0.25/omega2;
        int nqq, iq;
        double dq, q_, qq, aa;

        nqq = 100000;
        dq = 5.0/sqrt(alpha)/nqq;
        aa = 0;
        for (iq = 0; iq < nqq; iq++) {
            q_ = dq*(iq+0.5);
            qq = q_*q_;
            aa = aa - exp( -alpha * qq) * exp(-0.25*qq/omega2)*dq;
        }
        aa = 2.0*aa/M_PI + 1.0/sqrt(M_PI*alpha);
        scaled_intf = V*pSPARC->Nkpts_hf/(4*M_PI)*aa;
        pSPARC->const_aux = scaled_intf - sumfGq;
    }
        
#ifdef DEBUG
    if(!rank)
        printf("The constant for zero G (auxiliary function) is %.6f\n", pSPARC->const_aux);
#endif
}


/**
 * @brief   Estimation of Ecut by (pi/h)^2/2
 */
double ecut_estimate(double hx, double hy, double hz)
{
    double dx2_inv, dy2_inv, dz2_inv, h_eff, ecut;

    dx2_inv = 1.0/(hx * hx);
    dy2_inv = 1.0/(hy * hy);
    dz2_inv = 1.0/(hz * hz);
    h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
    
    ecut = (M_PI/h_eff) * (M_PI/h_eff) / 2;     // Ecut rho
    ecut = ecut / 4.0;                          // Ecut wfc
    ecut = ecut * 0.9;                          // by experience
    return ecut;
}


/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0) return;
    int i, rank, DMnd, Ns_occ, Ns, spn_i, Ns_occ_temp[2];
    int sum_temp, sum;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    sum_temp = sum = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        // construct ACE operator only using occupied states
        Ns_occ_temp[spn_i] = 0;
        for (i = 0; i < Ns; i++)
            if (pSPARC->occ[i + spn_i*Ns] > 1e-6) Ns_occ_temp[spn_i]++;
        Ns_occ_temp[spn_i] += pSPARC->EXXACEVal_state;
        Ns_occ_temp[spn_i] = min(Ns_occ_temp[spn_i], pSPARC->Nstates);                      // Ensure Ns_occ is less or equal to Nstates
    }
    
    // Note: occupations are only correct in dmcomm. We also need it to be correct in 
    // kptcomm_topo. Generally, all processes stores Ace opetaor need to be correct.
    MPI_Bcast(Ns_occ_temp, 2, MPI_INT, 0, pSPARC->kptcomm);
    
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        sum_temp += Ns_occ_temp[spn_i];
        sum += pSPARC->Nstates_occ[spn_i];
    }

    // If number of occupied states changed, need to reallocate memory space
    if (sum_temp != sum) { 
        if (sum > 0) {
        #ifdef DEBUG
        if(!rank) 
            printf("\nNumber of occupied states + Extra states changed : %d\n", sum_temp);
        #endif  
            free(pSPARC->Xi);
            free(pSPARC->Xi_kptcomm_topo);
            pSPARC->Xi = NULL;
            pSPARC->Xi_kptcomm_topo = NULL;
        } else {
        #ifdef DEBUG
        if(!rank) 
            printf("\nStarts to use %d states to create ACE operator.\n", sum_temp);
        #endif  
        }

        // Xi, ACE operator
        pSPARC->Xi = (double *)malloc(DMnd * sum_temp * sizeof(double));
        // Storage of ACE operator in kptcomm_topo
        pSPARC->Xi_kptcomm_topo = 
                (double *)calloc(pSPARC->Nd_d_kptcomm * sum_temp , sizeof(double));
        assert(pSPARC->Xi != NULL && pSPARC->Xi_kptcomm_topo != NULL);    
    }

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        pSPARC->Nstates_occ[spn_i] = Ns_occ_temp[spn_i];
}

/**
 * @brief   Compute exact exchange energy density
 */
void computeExactExchangeEnergyDensity(SPARC_OBJ *pSPARC, double *Exxrho)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, Nband, DMnd, Ns, spn_i, Nk, kpt, count;
    int n, nstart, nend, sg, size_s, size_k, spinDMnd;
    double *X, *Vexx, g_nk, t1, t2;
    double _Complex *X_kpt, *Vexx_kpt;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    spinDMnd = (pSPARC->spin_typ == 0) ? DMnd : 2*DMnd;
    comm = pSPARC->dmcomm;
    memset(Exxrho, 0, sizeof(double) * DMnd * (2*pSPARC->Nspin-1));

    if (pSPARC->isGammaPoint == 1) {
        size_s = DMnd * Nband;
        Vexx = (double *) calloc(sizeof(double), DMnd * Nband);
        assert(Vexx != NULL);
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            sg  = pSPARC->spin_start_indx + spn_i;
            X = pSPARC->Xorb + spn_i*size_s;

            memset(Vexx, 0, sizeof(double)*size_s);
            exact_exchange_potential(pSPARC, X, Nband, DMnd, Vexx, spn_i, comm);

            count = 0;
            for (n = nstart; n <= nend; n++) {
                g_nk = pSPARC->occ[n+spn_i*Ns];
                for (i = 0; i < DMnd; i++, count++) {
                    // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                    // only total in case of non-spin-polarized calculation
                    // different from electron density 
                    Exxrho[sg*DMnd + i] += g_nk * X[count] * Vexx[count];
                }
            }
        }
        free(Vexx);
    } else {
        size_k = DMnd * Nband;
        size_s = size_k * Nk;
        Vexx_kpt = (double _Complex *) calloc(sizeof(double _Complex), DMnd * Nband);
        assert(Vexx_kpt != NULL);
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            sg  = pSPARC->spin_start_indx + spn_i;
            for (kpt = 0; kpt < Nk; kpt++) {
                X_kpt = pSPARC->Xorb_kpt + kpt*size_k + spn_i*size_s;

                memset(Vexx_kpt, 0, sizeof(double _Complex) * DMnd * Nband);
                exact_exchange_potential_kpt(pSPARC, X_kpt, Nband, DMnd, Vexx_kpt, spn_i, kpt, comm);
                
                count = 0;
                for (n = nstart; n <= nend; n++) {
                    g_nk = (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * pSPARC->occ[spn_i*Nk*Ns+kpt*Ns+n];
                    for (i = 0; i < DMnd; i++, count++) {
                        // first column spin up, second colum spin down, last column total in case of spin-polarized calculation
                        // only total in case of non-spin-polarized calculation
                        // different from electron density 
                        Exxrho[sg*DMnd + i] += g_nk * creal(conj(X_kpt[count]) * Vexx_kpt[count]);
                    }
                }
            }
        }
        free(Vexx_kpt);
    }


    // sum over spin comm group
    if(pSPARC->npspin > 1) {
        t1 = MPI_Wtime();
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(Exxrho, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("rank = %d, --- Calculate kinetic energy density: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all k-point groups
    if (pSPARC->npkpt > 1 && pSPARC->spincomm_index == 0) {    
        t1 = MPI_Wtime();
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(Exxrho, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("rank = %d, --- Calculate exact exchange energy density: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0) {
        t1 = MPI_Wtime();
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(Exxrho, Exxrho, spinDMnd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (rank == 0) printf("rank = %d, --- Calculate exact exchange energy density: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }

    double vscal = 1.0 / pSPARC->dV * pSPARC->hyb_mixing;
    if (pSPARC->spin_typ == 0) {
        for (i = 0; i < DMnd; i++) {
            Exxrho[i] *= vscal;
        }
    } else {
        vscal *= 0.5;       // spin factor
        for (i = 0; i < 2*DMnd; i++) {
            Exxrho[i] *= vscal;
        }
        // Total Kinetic energy density 
        for (i = 0; i < DMnd; i++) {
            Exxrho[i+2*DMnd] = Exxrho[i] + Exxrho[i+DMnd];
        }
    }
#ifdef DEBUG
    double Exx = 0.0;
    for (i = 0; i < DMnd; i++) {
        Exx += Exxrho[i + pSPARC->spin_typ*2*DMnd];
    }
    if (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) {
        MPI_Allreduce(MPI_IN_PLACE, &Exx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    Exx *= pSPARC->dV;
    if (!rank) printf("\nExact exchange energy from energy density: %f\n"
                        "Exact exchange energy calculated directly: %f\n", Exx, -pSPARC->hyb_mixing*pSPARC->Eexx);
#endif
}