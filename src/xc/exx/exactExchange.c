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
    #define MKL_Complex16 double _Complex
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
#include "lapVecRoutines.h"
#include "linearSolver.h"
#include "exactExchangeKpt.h"
#include "tools.h"
#include "parallelization.h"
#include "electronicGroundState.h"
#include "exchangeCorrelation.h"
#include "exactExchangeInitialization.h"
#include "electrostatics.h"
#include "sqHighTDensity.h"
#include "sqParallelization.h"
#include "sqHighTExactExchange.h"
#include "kroneckerLaplacian.h"

#ifdef ACCELGT
#include "accel.h"
#include "cuda_exactexchange.h"
#include "cuda_linearAlgebra.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Outer loop of SCF using Vexx (exact exchange potential)
 */
void Exact_Exchange_loop(SPARC_OBJ *pSPARC) {
    int i, rank;
    double t1, t2, ACE_time = 0.0;
    FILE *output_fp;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int DMnd = pSPARC->Nd_d;

    /************************ Exact exchange potential parameters ************************/
    int count_xx = 0;
    double Eexx_pre = pSPARC->Eexx, err_Exx = pSPARC->TOL_FOCK + 1;
    pSPARC->Exxtime = pSPARC->ACEtime = pSPARC->Exxtime_comm = 0.0;
    pSPARC->Exxtime_solver = pSPARC->Exxtime_memload = pSPARC->Exxtime_cyc = 0;
    pSPARC->Exxtime_rhs = pSPARC->Exxtime_move = 0;

    /************************* Update Veff copied from SCF code **************************/
    #ifdef DEBUG
    if(!rank) 
        printf("\nStart evaluating Exact Exchange !\n");
    fflush(stdout);
    #endif  

    // update pbe density
    memcpy(pSPARC->electronDens_pbe, pSPARC->electronDens, DMnd * pSPARC->Nspdentd * sizeof(double));

    // initialize the history variables of Anderson mixing
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        memset(pSPARC->mixing_hist_Xk, 0, sizeof(double)* pSPARC->Nd_d * pSPARC->Nspden * pSPARC->MixingHistory);
        memset(pSPARC->mixing_hist_Fk, 0, sizeof(double)* pSPARC->Nd_d * pSPARC->Nspden * pSPARC->MixingHistory);
    }

    // for the first outer loop with SQ.
    if (pSPARC->sqHighTFlag) {
        t1 = MPI_Wtime();
        pSPARC->usefock--;
        calculate_density_matrix_SQ_highT(pSPARC);
        pSPARC->usefock++;
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank) 
            printf("rank = %d, calculating density matrix took %.3f ms\n",rank,(t2-t1)*1e3); 
        #endif 
    }

    // calculate xc potential (LDA), "Vxc"
    t1 = MPI_Wtime(); 
    Calculate_Vxc(pSPARC);
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (rank == 0) printf("rank = %d, XC calculation took %.3f ms\n", rank, (t2-t1)*1e3); 
    #endif 
    
    t1 = MPI_Wtime();
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);

    // initialize mixing_hist_xk (and mixing_hist_xkm1)
    Update_mixing_hist_xk(pSPARC);

    if (pSPARC->sqHighTFlag == 1) {
        for (i = 0; i < pSPARC->Nspden; i++)
            TransferVeff_phi2sq(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->pSQ->Veff_loc_SQ + i * pSPARC->pSQ->DMnd_SQ);
    } else {
        // transfer Veff_loc from "phi-domain" to "psi-domain"
        for (i = 0; i < pSPARC->Nspden; i++)
            Transfer_Veff_loc(pSPARC, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
    }

    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank) 
        printf("rank = %d, Veff calculation and Bcast (non-blocking) took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif 

    /******************************* Hartre-Fock outer loop ******************************/
    while (count_xx < pSPARC->MAXIT_FOCK) {
    #ifdef DEBUG
    if(!rank) 
        printf("\nHartree-Fock Outer Loop: %d \n",count_xx + 1);
    fflush(stdout);
    #endif  

        if (pSPARC->sqHighTFlag) {
            t1 = MPI_Wtime();
            collect_col_of_Density_matrix(pSPARC);
            if (pSPARC->ExxAcc == 1 && pSPARC->SQ_highT_hybrid_gauss_mem == 1) {
                compute_exx_potential_SQ(pSPARC);
            }
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if (pSPARC->ExxAcc == 1 && pSPARC->SQ_highT_hybrid_gauss_mem == 1) {
                if(!rank) printf("\nCollecting columns of density matrix and calculating exx potential for SQ hybrid took : %.3f ms\n", (t2-t1)*1e3);
            } else {
                if(!rank) printf("\nCollecting columns of density matrix for SQ hybrid took : %.3f ms\n", (t2-t1)*1e3);
            }
            #endif 
            pSPARC->ACEtime += (t2 - t1);
            ACE_time = (t2 - t1);
        } else {
            if (pSPARC->ExxAcc == 0) {
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
            } else if (pSPARC->ExxAcc == 1) {
                #ifdef DEBUG
                if(!rank) printf("\nStart to create ACE operator!\n");
                #endif  
                t1 = MPI_Wtime();
                // create ACE operator 
                if (pSPARC->isGammaPoint == 1) {
                    allocate_ACE(pSPARC);                
                    ACE_operator(pSPARC, pSPARC->Xorb, pSPARC->occ, pSPARC->Xi);
                } else {
                    gather_psi_occ_outer_kpt(pSPARC, pSPARC->psi_outer_kpt, pSPARC->occ_outer);
                    allocate_ACE_kpt(pSPARC);                
                    ACE_operator_kpt(pSPARC, pSPARC->Xorb_kpt, pSPARC->occ_outer, pSPARC->Xi_kpt);
                }
                t2 = MPI_Wtime();
                pSPARC->ACEtime += (t2 - t1);
                ACE_time = (t2 - t1);
                #ifdef DEBUG
                if(!rank) printf("\nCreating ACE operator took %.3f ms!\n", (t2 - t1)*1e3);
                #endif
            }
        }

        // transfer psi_outer from "psi-domain" to "phi-domain" in No-ACE case 
        // transfer Xi from "psi-domain" to "phi-domain" in ACE case 
        t1 = MPI_Wtime();
        if (!pSPARC->sqHighTFlag && pSPARC->ExxAcc == 0) {
            if (pSPARC->isGammaPoint == 1) {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates, pSPARC->psi_outer, pSPARC->psi_outer_kptcomm_topo, sizeof(double));    
            } else {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates*pSPARC->Nkpts_hf_red, pSPARC->psi_outer_kpt, pSPARC->psi_outer_kptcomm_topo_kpt, sizeof(double _Complex));
            }
            
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank) 
                printf("\nTransfering all bands of psi_outer to kptcomm_topo took : %.3f ms\n", (t2-t1)*1e3);
            #endif  
        } else if (!pSPARC->sqHighTFlag && pSPARC->ExxAcc == 1) {
            if (pSPARC->isGammaPoint == 1) {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates_occ, pSPARC->Xi, pSPARC->Xi_kptcomm_topo, sizeof(double));
            } else {
                Transfer_dmcomm_to_kptcomm_topo(pSPARC, pSPARC->Nspinor_spincomm, pSPARC->Nstates_occ*pSPARC->Nkpts_kptcomm, pSPARC->Xi_kpt, pSPARC->Xi_kptcomm_topo_kpt, sizeof(double _Complex));                
            }

            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank) 
                printf("\nTransfering Xi to kptcomm_topo took : %.3f ms\n", (t2-t1)*1e3);
            #endif  
        }

        // compute exact exchange energy estimation with psi_outer
        // Eexx saves negative exact exchange energy without hybrid mixing
        if (pSPARC->sqHighTFlag == 1) {
            exact_exchange_energy_SQ(pSPARC);
        } else {
            exact_exchange_energy(pSPARC);
        }

        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (pSPARC->sqHighTFlag == 1) {
                fprintf(output_fp,"\nNo.%d Exx outer loop. Basis timing: %.3f (sec)\n", count_xx + 1, ACE_time);
            } else if (pSPARC->ExxAcc == 0) {
                fprintf(output_fp,"\nNo.%d Exx outer loop. \n", count_xx + 1);
            } else {
                fprintf(output_fp,"\nNo.%d Exx outer loop. ACE timing: %.3f (sec)\n", count_xx + 1, ACE_time);
            }
            fclose(output_fp);
        }

        scf_loop(pSPARC);        

        Eexx_pre = pSPARC->Eexx;
        // update the final exact exchange energy
        pSPARC->Exc -= pSPARC->Eexx;
        pSPARC->Etot += 2 * pSPARC->Eexx;

        // compute exact exchange energy
        if (pSPARC->sqHighTFlag == 1) {
            calculate_density_matrix_SQ_highT(pSPARC);
            exact_exchange_energy_SQ(pSPARC);
        } else {
            exact_exchange_energy(pSPARC);
        }
        pSPARC->Exc += pSPARC->Eexx;
        pSPARC->Etot -= 2*pSPARC->Eexx;

        // error evaluation
        err_Exx = fabs(Eexx_pre - pSPARC->Eexx)/pSPARC->n_atom;
        MPI_Bcast(&err_Exx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);                 // TODO: Create bridge comm 
        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"Exx outer loop error: %.10e \n",err_Exx);
            fclose(output_fp);
        }
        if (err_Exx < pSPARC->TOL_FOCK && (count_xx+1) >= pSPARC->MINIT_FOCK) break;        
        pSPARC->fock_err = err_Exx;
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

    #ifdef DEBUG
    if(!rank && pSPARC->sqHighTFlag && pSPARC->ExxAcc == 0) {
        printf("\n== Exact exchange Timing in SQ (Hsub routine) takes   %.3f ms\n", pSPARC->Exxtime*1e3);
    }
    if(!rank && pSPARC->sqHighTFlag && pSPARC->ExxAcc == 1) {
        printf("\n== Exact exchange Timing in SQ (Hsub routine) takes %.3f ms\tcalculating potential takes %.3f ms\n", pSPARC->Exxtime*1e3, pSPARC->ACEtime*1e3);
    }
    if(!rank && !pSPARC->sqHighTFlag && pSPARC->ExxAcc == 1) {
        printf("\n== Exact exchange Timing: creating ACE: %.3f ms\tapply ACE: %.3f ms\t Alltoallv takes %.3f ms\n",
            pSPARC->ACEtime*1e3, pSPARC->Exxtime*1e3, pSPARC->Exxtime_comm*1e3);
        printf(  "                          serial solver takes %.3f ms, cyc communication: %.3f ms, gpu loaded %.3f ms\n", 
            pSPARC->Exxtime_solver*1e3, pSPARC->Exxtime_cyc*1e3, pSPARC->Exxtime_memload*1e3);
        printf(  "                          creating rhs takes %.3f ms, rhs formating takes %.3f ms\n", pSPARC->Exxtime_rhs*1e3, pSPARC->Exxtime_move*1e3);
    }
    if(!rank && !pSPARC->sqHighTFlag && pSPARC->ExxAcc == 0) {
        printf("\n== Exact exchange Timing: apply Vx takes    %.3f ms\n", pSPARC->Exxtime*1e3);
    }
    #endif  
}


/**
 * @brief   Evaluating Exact Exchange potential
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ldx, int ncol, int DMnd, double *Hx, int ldhx, int spin, MPI_Comm comm) 
{
    int rank, Lanczos_flag;
    double *Xi, t1, t2, *occ;
    
    MPI_Comm_rank(comm, &rank);
    Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
    /********************************************************************/

    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int occ_outer_shift = pSPARC->Nstates;

    t1 = MPI_Wtime();
    if (pSPARC->ExxAcc == 0) {
        occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spin * occ_outer_shift) : pSPARC->occ_outer;
        double *psi_outer = (Lanczos_flag == 0) ? pSPARC->psi_outer + spin* DMnd : pSPARC->psi_outer_kptcomm_topo + spin* DMnd;
        evaluate_exact_exchange_potential(pSPARC, X, ldx, ncol, DMnd, occ, psi_outer, DMndsp, Hx, ldhx, comm);
    } else {
        Xi = (Lanczos_flag == 0) ? pSPARC->Xi + spin * DMnd : pSPARC->Xi_kptcomm_topo + spin * DMnd;
        evaluate_exact_exchange_potential_ACE(pSPARC, X, ldx, ncol, DMnd, Xi, DMndsp, Hx, ldhx, comm);
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
 * @param occ_outer       Full set of occ_outer occupations
 * @param psi_outer       Full set of psi_outer orbitals
 * @param Hx              Result of Hx plus fock operator times X 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ldx, int ncol, int DMnd, 
                                    double *occ_outer, double *psi_outer, int ldpo, double *Hx, int ldhx, MPI_Comm comm)
{
    int i, j, k, rank, Ns, num_rhs, *rhs_list_i, *rhs_list_j;
    int size, batch_num_rhs, NL, base, loop;
    double occ, *rhs, *sol, exx_frac, occ_alpha;

    Ns = pSPARC->Nstates;
    exx_frac = pSPARC->exx_frac;
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

    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * size;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required                        
    rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    sol = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
    assert(rhs != NULL && sol != NULL);

    /*************** Solve all Poisson's equation and apply to X ****************/    
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            for (k = 0; k < DMnd; k++) {
                rhs[k + (count-base)*DMnd] = psi_outer[k + j*ldpo] * X[k + i*ldx];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, comm);

        // Apply exact exchange potential to vector X
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            occ = occ_outer[j];
            occ_alpha = occ * exx_frac;
            for (k = 0; k < DMnd; k++) {
                Hx[k + i*ldhx] -= occ_alpha * psi_outer[k + j*ldpo] * sol[k + (count-base)*DMnd] / pSPARC->dV;
            }
        }
    }

    
    free(rhs);
    free(sol);
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
void evaluate_exact_exchange_potential_ACE(SPARC_OBJ *pSPARC, double *X, int ldx, 
    int ncol, int DMnd, double *Xi, int ldxi, double *Hx, int ldhx, MPI_Comm comm) 
{
    int rank, size, Nstates_occ;
    Nstates_occ = pSPARC->Nstates_occ;
    double *Xi_times_psi = (double *) malloc(Nstates_occ * ncol * sizeof(double));
    assert(Xi_times_psi != NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/

    // perform matrix multiplication Xi' * X using ScaLAPACK routines
    if (ncol != 1) {
        cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Nstates_occ, ncol, DMnd,
                    1.0, Xi, ldxi, X, ldx, 0.0, Xi_times_psi, Nstates_occ);
    } else {
        cblas_dgemv( CblasColMajor, CblasTrans, DMnd, Nstates_occ, 1.0, 
                    Xi, ldxi, X, 1, 0.0, Xi_times_psi, 1);
    }

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nstates_occ*ncol, 
                      MPI_DOUBLE, MPI_SUM, comm);
    }

    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    if (ncol != 1) {
        cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Nstates_occ,
                    -pSPARC->exx_frac, Xi, ldxi, Xi_times_psi, Nstates_occ, 1.0, Hx, ldhx);
    } else {
        cblas_dgemv( CblasColMajor, CblasNoTrans, DMnd, Nstates_occ, -pSPARC->exx_frac, 
                    Xi, ldxi, Xi_times_psi, 1, 1.0, Hx, 1);
    }

    free(Xi_times_psi);
}


void exact_exchange_energy(SPARC_OBJ *pSPARC)
{
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    if (pSPARC->isGammaPoint == 1) {
        evaluate_exact_exchange_energy(pSPARC);
    } else {
        evaluate_exact_exchange_energy_kpt(pSPARC);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
if(!rank) 
    printf("\nEvaluating Exact exchange energy took: %.3f ms\nExact exchange energy %.6f.\n", (t2-t1)*1e3, pSPARC->Eexx);
#endif  
}

/**
 * @brief   Evaluate Exact Exchange Energy
 */
void evaluate_exact_exchange_energy(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, rank, size;
    int Ns, ncol, DMnd, DMndsp, num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, *rhs, *sol, *psi_outer, temp, *occ_outer, *psi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Ns = pSPARC->Nstates;
    ncol = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;
    pSPARC->Eexx = 0.0;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (pSPARC->ExxAcc == 0) {
        int *rhs_list_i, *rhs_list_j;
        rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
        rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
        assert(rhs_list_i != NULL && rhs_list_j != NULL);

        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            psi_outer = pSPARC->psi_outer + spinor * DMnd;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * Ns) : pSPARC->occ_outer;
            psi = pSPARC->Xorb + spinor * DMnd;

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

            batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                            num_rhs : pSPARC->ExxMemBatch * size;
        
            NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
            rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
            sol = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
            assert(rhs != NULL && sol != NULL);

            for (loop = 0; loop < NL; loop ++) {
                base = batch_num_rhs*loop;
                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    for (k = 0; k < DMnd; k++) {
                        rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMndsp] * psi[k + i*DMndsp];
                    }
                }

                // Solve all Poisson's equation 
                poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, comm);

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    
                    occ_i = occ_outer[i + pSPARC->band_start_indx];
                    occ_j = occ_outer[j];

                    // TODO: use a temp array to reduce the MPI_Allreduce time to 1
                    temp = 0.0;
                    for (k = 0; k < DMnd; k++){
                        temp += rhs[k + (count-base)*DMnd] * sol[k + (count-base)*DMnd];
                    }
                    if (size > 1)
                        MPI_Allreduce(MPI_IN_PLACE, &temp, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
                    pSPARC->Eexx += occ_i * occ_j * temp;
                }
            }

            free(rhs);
            free(sol);
        }
        free(rhs_list_i);
        free(rhs_list_j);

        pSPARC->Eexx /= pSPARC->dV;

    } else {
        int Nstates_occ = pSPARC->Nstates_occ;
        int Nband_bandcomm_M = pSPARC->Nband_bandcomm_M;

        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            if (Nband_bandcomm_M == 0) continue;
            double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
            double *Xi_times_psi = (double *) calloc(Nband_bandcomm_M * Nstates_occ, sizeof(double));
            assert(Xi_times_psi != NULL);
        
            #ifdef ACCELGT
            if (pSPARC->useACCEL == 1) {
                ACCEL_DGEMM( CblasColMajor, CblasTrans, CblasNoTrans, Nband_bandcomm_M, Nstates_occ, DMnd,
                            1.0, pSPARC->Xorb + spinor * DMnd, DMndsp, pSPARC->Xi + spinor * DMnd, 
                            DMndsp, 0.0, Xi_times_psi, Nband_bandcomm_M);
            } else
            #endif // ACCELGT
            {
                // perform matrix multiplication psi' * X using ScaLAPACK routines
                cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Nband_bandcomm_M, Nstates_occ, DMnd,
                            1.0, pSPARC->Xorb + spinor * DMnd, DMndsp, pSPARC->Xi + spinor * DMnd, 
                            DMndsp, 0.0, Xi_times_psi, Nband_bandcomm_M);
            }

            if (size > 1) {
                // sum over all processors in dmcomm
                MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nband_bandcomm_M*Nstates_occ, 
                            MPI_DOUBLE, MPI_SUM, comm);
            }

            for (i = 0; i < Nband_bandcomm_M; i++) {
                temp = 0.0;
                for (j = 0; j < Nstates_occ; j++) {
                    temp += Xi_times_psi[i+j*Nband_bandcomm_M] * Xi_times_psi[i+j*Nband_bandcomm_M];
                }
                temp *= occ[i + pSPARC->band_start_indx];
                pSPARC->Eexx += temp;
            }

            free(Xi_times_psi);
        }
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    pSPARC->Eexx /= (pSPARC->Nspin + 0.0);
    pSPARC->Eexx *= -pSPARC->exx_frac;
}



/**
 * @brief   Solving Poisson's equation using FFT or KRON
 *          
 *          This function only works for solving Poisson's equation with real right hand side
 *          option: 0 - solve poissons equation,       1 - solve with pois_const_stress
 *                  2 - solve with pois_const_stress2, 3 - solve with pois_const_press
 */
void poissonSolve(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, 
                int ncol, int DMnd, double *sol, MPI_Comm comm) 
{    
    int size, rank;
    int *sendcounts, *sdispls, *recvcounts, *rdispls;
    double *rhs_loc, *sol_loc, *rhs_loc_order, *sol_loc_order;
    sendcounts = sdispls = recvcounts = rdispls = NULL;
    rhs_loc = sol_loc = rhs_loc_order = sol_loc_order = NULL;    
    int (*DMVertices)[6] = NULL;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);    
    int Nd = pSPARC->Nd;
    int Nx = pSPARC->Nx; 
    int Ny = pSPARC->Ny; 
    int Nz = pSPARC->Nz;
    int gridsizes[3] = {Nx, Ny, Nz};
    int ncolp = ncol / size + ((rank < ncol % size) ? 1 : 0);
    /********************************************************************/

#ifdef DEBUG
	double t1, t2;
#endif  

    if (size > 1){
        // variables for RHS storage
        rhs_loc = (double*) malloc(sizeof(double) * ncolp * Nd);
        rhs_loc_order = (double*) malloc(sizeof(double) * Nd * ncolp);
        DMVertices = (int (*)[6]) malloc(sizeof(int[6])*size);

        // variables for alltoallv
        sendcounts = (int*) malloc(sizeof(int)*size);
        sdispls = (int*) malloc(sizeof(int)*size);
        recvcounts = (int*) malloc(sizeof(int)*size);
        rdispls = (int*) malloc(sizeof(int)*size);
        assert(rhs_loc != NULL && rhs_loc_order != NULL && 
               sendcounts != NULL && sdispls != NULL && recvcounts != NULL && rdispls != NULL);
                
        parallel_info_dp2bp(gridsizes, DMnd, ncol, DMVertices, sendcounts, sdispls, recvcounts, rdispls, NULL, 1, comm);
        /********************************************************************/

#ifdef DEBUG
	t1 = MPI_Wtime();
#endif  
        MPI_Alltoallv(rhs, sendcounts, sdispls, MPI_DOUBLE, 
                        rhs_loc, recvcounts, rdispls, MPI_DOUBLE, comm);

#ifdef DEBUG
	t2 = MPI_Wtime();
    pSPARC->Exxtime_comm += (t2-t1);
#endif  

        // rhs_full needs rearrangement
        block_dp_to_cart((void *) rhs_loc, ncolp, DMVertices, rdispls, size, 
                        Nx, Ny, Nd, (void *) rhs_loc_order, sizeof(double));

        free(rhs_loc);
        // variable for local result sol
        sol_loc = (double*) malloc(sizeof(double)* Nd * ncolp);
        assert(sol_loc != NULL);
    } else {
        // if the size of comm is 1, there is no need to scatter and rearrange the results
        rhs_loc_order = rhs;
        sol_loc = sol;
    }   

#ifdef DEBUG
	t1 = MPI_Wtime();
#endif 
    if (pSPARC->ExxMethod == 0) {
        // solve by fft
        pois_fft(pSPARC, rhs_loc_order, pois_const, ncolp, sol_loc);
    } else {
        // solve by kron
        pois_kron(pSPARC, rhs_loc_order, pois_const, ncolp, sol_loc);
    }

#ifdef DEBUG
	t2 = MPI_Wtime();
    pSPARC->Exxtime_solver += (t2-t1);
#endif  

    if (size > 1)  
        free(rhs_loc_order);

    if (size > 1) {
        sol_loc_order = (double*) malloc(sizeof(double)* Nd * ncolp);
        assert(sol_loc_order != NULL);

        // sol_loc needs rearrangement
        cart_to_block_dp((void *) sol_loc, ncolp, DMVertices, size, 
                        Nx, Ny, Nd, (void *) sol_loc_order, sizeof(double));

#ifdef DEBUG
	t1 = MPI_Wtime();
#endif  
        MPI_Alltoallv(sol_loc_order, recvcounts, rdispls, MPI_DOUBLE, 
                    sol, sendcounts, sdispls, MPI_DOUBLE, comm);

#ifdef DEBUG
	t2 = MPI_Wtime();
    pSPARC->Exxtime_comm += (t2-t1);
#endif  
        free(sol_loc_order);
    }

    /********************************************************************/
    if (size > 1){
        free(sol_loc);        
        free(sendcounts);
        free(sdispls);
        free(recvcounts);
        free(rdispls);
        free(DMVertices);
    }
}


void parallel_info_dp2bp(int gridsizes[3], int DMnd, int ncol, 
    int (*DMVertices)[6], int *sendcounts, int *sdispls, int *recvcounts, int *rdispls, int *kq_shift, int sing_size, MPI_Comm cart_comm)
{
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);
    
    int ncolp = ncol / size + ((rank < ncol % size) ? 1 : 0);
    int *ncolpp = (int*) malloc(sizeof(int) * size);
    /********************************************************************/
    
    // separate equations to different processes in the dmcomm or kptcomm_topo                 
    for (int i = 0; i < size; i++) {
        ncolpp[i] = ncol / size + ((i < ncol % size) ? 1 : 0);
    }

    if (kq_shift) {
        for (int i = 0; i < rank; i++) *kq_shift += ncolpp[i];
    }

    // compute variables required by gatherv and scatterv
    for (int i = 0; i < size; i++) {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get(cart_comm, 3, dims, periods, coords);
        int coord_comm[3];
        MPI_Cart_coords(cart_comm, i, 3, coord_comm);
        // find size of distributed domain over comm
        int DNx = block_decompose(gridsizes[0], dims[0], coord_comm[0]);
        int DNy = block_decompose(gridsizes[1], dims[1], coord_comm[1]);
        int DNz = block_decompose(gridsizes[2], dims[2], coord_comm[2]);
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
    for (int i = 0; i < size; i++) {
        sendcounts[i] = ncolpp[i] * DMnd * sing_size;
        recvcounts[i] = ncolp * DMVertices[i][1] * DMVertices[i][3] * DMVertices[i][5] * sing_size;
        if (i < size - 1) {
            sdispls[i+1] = sdispls[i] + sendcounts[i];
            rdispls[i+1] = rdispls[i] + recvcounts[i];
        }
    }

    free(ncolpp);    
}


/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_kron(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, int ncol, double *sol)
{
    if (ncol == 0) return;
    int Nd = pSPARC->Nd;
    KRON_LAP* kron_lap = pSPARC->kron_lap_exx[0];
    
    if (pSPARC->BC == 1) {
        double *rhs_ = (double *) malloc(sizeof(double)*Nd);
        double *d_cor = (double *) malloc(sizeof(double)*Nd);
        assert(rhs_ != NULL && d_cor != NULL);
        int DMVertices[6] = {0, pSPARC->Nx-1, 0, pSPARC->Ny-1, 0, pSPARC->Nz-1};

        for (int n = 0; n < ncol; n++) {
            for (int i = 0; i < Nd; i++) rhs_[i] = -4*M_PI*rhs[i + n*Nd];
            apply_multipole_expansion(pSPARC, pSPARC->MpExp_exx, 
                pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, DMVertices, rhs_, d_cor, MPI_COMM_SELF);
            for (int i = 0; i < Nd; i++) rhs_[i] -= d_cor[i];
            LAP_KRON(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                    rhs_, kron_lap->inv_eig, sol + n*Nd);
        }
        free(rhs_);
        free(d_cor);
    } else {
        for (int n = 0; n < ncol; n++) {
            LAP_KRON(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                    rhs + n*Nd, pois_const, sol + n*Nd);
        }
    }
}



/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const    constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_fft(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, int ncol, double *sol) {
    if (ncol == 0) return;    

    int Nd = pSPARC->Nd;
    int Nx = pSPARC->Nx, Ny = pSPARC->Ny, Nz = pSPARC->Nz; 
    int Ndc = Nz * Ny * (Nx/2+1);
    double _Complex *rhs_bar = (double _Complex*) malloc(sizeof(double _Complex) * Ndc * ncol);
    assert(rhs_bar != NULL);

#if defined(USE_MKL)
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 
#elif defined(USE_FFTW)
    int dim_sizes[3] = {Nz, Ny, Nx};
    int inembed[3] = {Nz, Ny, Nx};
    int onembed[3] = {Nz, Ny, (Nx/2+1)};
#endif
    /********************************************************************/
    // FFT
    #if defined(USE_MKL)
        MKL_MDFFT_batch_real(rhs, ncol, dim_sizes, Nd, rhs_bar, strides_out, Ndc);
    #elif defined(USE_FFTW)
        FFTW_MDFFT_batch_real(dim_sizes, ncol, rhs, inembed, Nd, rhs_bar, onembed, Ndc);
    #endif

    int cnt = 0;
    for (int n = 0; n < ncol; n++) {
        // multiplied by alpha
        for (int i = 0; i < Ndc; i++) {
            rhs_bar[cnt] = creal(rhs_bar[cnt]) * pois_const[i] 
                                + (cimag(rhs_bar[cnt]) * pois_const[i]) * I;
            cnt++;
        }
    }
    
    // iFFT
    #if defined(USE_MKL)
        // MKL_MDiFFT_real(rhs_bar, dim_sizes, strides_out, sol + n * Nd);
        MKL_MDiFFT_batch_real(rhs_bar, ncol, dim_sizes, Ndc, strides_out, sol, Nd);
    #elif defined(USE_FFTW)        
        // FFTW_MDiFFT_real(dim_sizes, rhs_bar, sol + n * Nd);
        FFTW_MDiFFT_batch_real(dim_sizes, ncol, rhs_bar, onembed, Ndc, sol, inembed, Nd);
    #endif
    free(rhs_bar);
}

/**
 * @brief   Gather psi_outers in other bandcomms
 *
 *          The default comm is blacscomm
 */
void gather_psi_occ_outer(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer) 
{
    int i, grank, lrank, lsize, Ns, DMnd, DMndsp, Nband;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);

    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Nband = pSPARC->Nband_bandcomm;    
    Ns = pSPARC->Nstates;

    int NsNsp = Ns * pSPARC->Nspin_spincomm;
    int shift = pSPARC->band_start_indx * DMndsp;
    // Save orbitals and occupations and to construct exact exchange operator
    copy_mat_blk(sizeof(double), pSPARC->Xorb, DMndsp, DMndsp, Nband, psi_outer+shift, DMndsp);
    gather_blacscomm(pSPARC, DMndsp, Ns, psi_outer);
    
    for (i = 0; i < NsNsp; i++) 
        occ_outer[i] = pSPARC->occ[i];
    /********************************************************************/
    if (pSPARC->flag_kpttopo_dm && pSPARC->ExxAcc == 0) {
        int rank_kptcomm_topo;
        MPI_Comm_rank(pSPARC->kptcomm_topo, &rank_kptcomm_topo);
        if (pSPARC->flag_kpttopo_dm_type == 1) {
            if (!rank_kptcomm_topo)
                MPI_Bcast(occ_outer, NsNsp, MPI_DOUBLE, MPI_ROOT, pSPARC->kpttopo_dmcomm_inter);
            else
                MPI_Bcast(occ_outer, NsNsp, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kpttopo_dmcomm_inter);
        } else {
            MPI_Bcast(occ_outer, NsNsp, MPI_DOUBLE, 0, pSPARC->kpttopo_dmcomm_inter);
        }
    }
}



/**
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double *vec)
{
    if (pSPARC->blacscomm == MPI_COMM_NULL) return;
    int i, grank, lrank, lsize;
    int *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);    

    if (lsize > 1) {
        recvcounts = (int*) malloc(sizeof(int)* lsize);
        displs = (int*) malloc(sizeof(int)* lsize);
        assert(recvcounts != NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < lsize; i++){
            recvcounts[i] = (i < (Ncol / NB) ? NB : (i == (Ncol / NB) ? (Ncol % NB) : 0)) * Nrow;
            if (i != (lsize-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }

        MPI_Allgatherv(MPI_IN_PLACE, 1, MPI_DOUBLE, vec, 
            recvcounts, displs, MPI_DOUBLE, pSPARC->blacscomm);   
        
        free(recvcounts);
        free(displs); 
    }
}


/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE(SPARC_OBJ *pSPARC) {
    int i, rank, DMnd, DMndsp, Nstates_occ, Ns, spn_i;    

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
        
    Nstates_occ = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        // construct ACE operator only using occupied states
        int Nstates_occ_temp = 0;
        for (i = 0; i < Ns; i++)
            if (pSPARC->occ[i + spn_i*Ns] > 1e-6) Nstates_occ_temp ++;
        pSPARC->Nstates_occ_list[spn_i] = Nstates_occ_temp;
        Nstates_occ = max(Nstates_occ, Nstates_occ_temp);
    }
    Nstates_occ += pSPARC->EeeAceValState;
    Nstates_occ = min(Nstates_occ, pSPARC->Nstates);                      // Ensure Nstates_occ is less or equal to Nstates        
    
    // Note: occupations are only correct in dmcomm.
    MPI_Allreduce(MPI_IN_PLACE, &Nstates_occ, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    if (pSPARC->spincomm_index < 0) return;

    // If number of occupied states changed, need to reallocate memory space
    if (Nstates_occ != pSPARC->Nstates_occ) { 
        if (pSPARC->Nstates_occ > 0) {
        #ifdef DEBUG
        if(!rank) 
            printf("\nNumber of occupied states + Extra states changed : %d\n", Nstates_occ);
        #endif  
            free(pSPARC->Xi);
            free(pSPARC->Xi_kptcomm_topo);
            pSPARC->Xi = NULL;
            pSPARC->Xi_kptcomm_topo = NULL;
        } else {
        #ifdef DEBUG
        if(!rank) 
            printf("\nStarts to use %d states to create ACE operator.\n", Nstates_occ);
        #endif  
        }

        // Xi, ACE operator
        pSPARC->Xi = (double *)malloc(DMndsp * Nstates_occ * sizeof(double));
        // Storage of ACE operator in kptcomm_topo
        pSPARC->Xi_kptcomm_topo = 
                (double *)calloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_spincomm * Nstates_occ , sizeof(double));
        assert(pSPARC->Xi != NULL && pSPARC->Xi_kptcomm_topo != NULL);    
        pSPARC->Nstates_occ = Nstates_occ;
    }
    
    if (Nstates_occ < pSPARC->band_start_indx)
        pSPARC->Nband_bandcomm_M = 0;
    else {
        pSPARC->Nband_bandcomm_M = min(pSPARC->Nband_bandcomm, Nstates_occ - pSPARC->band_start_indx);
    }
}


/**
 * @brief   Create ACE operator in dmcomm
 *
 *          Using occupied + extra orbitals to construct the ACE operator 
 *          Due to the symmetry of ACE operator, only half Poisson's 
 *          equations need to be solved.
 */
void ACE_operator(SPARC_OBJ *pSPARC, double *psi, double *occ, double *Xi)
{
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    t1 = MPI_Wtime();
#endif

#ifdef ACCELGT
    if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && pSPARC->spin_typ <= 1)
    {
        ACCEL_solving_for_Xi(pSPARC, psi, occ, Xi);
    } else 
#endif 
    {
        solving_for_Xi(pSPARC, psi, occ, Xi);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) 
    printf("solving_for_Xi took: %.3f ms\n", (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif  

    calculate_ACE_operator(pSPARC, psi, Xi);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) 
    printf("calculate_ACE_operator took: %.3f ms\n", (t2-t1)*1e3);
#endif  
}


void solving_for_Xi(SPARC_OBJ *pSPARC, double *psi, double *occ, double *Xi) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank, nproc_dmcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    int Nband_M = pSPARC->Nband_bandcomm_M;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int Nstates_occ = pSPARC->Nstates_occ;    

    memset(Xi, 0, sizeof(double) * Nstates_occ*DMndsp);
    // if Nband==0 here, Xi_ won't be used anyway
    double *Xi_ = Xi + pSPARC->band_start_indx * DMndsp;

    int nproc_blacscomm = pSPARC->npband;
    int reps = (nproc_blacscomm == 1) ? 0 : ((nproc_blacscomm - 2) / 2 + 1); // ceil((nproc_blacscomm-1)/2)
    int Nband_max = (pSPARC->Nstates - 1) / pSPARC->npband + 1;

    MPI_Request reqs[2];
    double *psi_storage1, *psi_storage2;
    psi_storage1 = psi_storage2 = NULL;
    if (reps > 0) {
        psi_storage1 = (double *) malloc(sizeof(double) * DMnd * Nband_max);
        psi_storage2 = (double *) malloc(sizeof(double) * DMnd * Nband_max);
    }
    
    double t1, t2, t_comm;
    t_comm = 0;
    for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        // in case of hydrogen 
        if (pSPARC->Nstates_occ_list[min(spinor, pSPARC->Nspin_spincomm-1)] == 0) continue;

        for (int rep = 0; rep <= reps; rep++) {
            if (rep == 0) {
                if (reps > 0) {
                    t1 = MPI_Wtime();
                    // first gather the orbitals in the rotation way
                    if (DMnd != DMndsp) {
                        copy_mat_blk(sizeof(double), psi + spinor*DMnd, DMndsp, DMnd, pSPARC->Nband_bandcomm, psi_storage2, DMnd);
                        transfer_orbitals_blacscomm(pSPARC, psi_storage2, psi_storage1, rep, reqs, sizeof(double));
                    } else {
                        transfer_orbitals_blacscomm(pSPARC, psi, psi_storage1, rep, reqs, sizeof(double));
                    }
                    t2 = MPI_Wtime();
                    t_comm += (t2 - t1);
                }
                // solve poisson's equations 
                double *occ_ = (pSPARC->spin_typ == 1) ? (occ + spinor*pSPARC->Nstates) : occ;
                solve_half_local_poissons_equation_apply2Xi(pSPARC, Nband_M, psi + spinor*DMnd, DMndsp, occ_, Xi_ + spinor*DMnd, DMndsp);
            } else {
                t1 = MPI_Wtime();
                int res = MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
                assert(res == MPI_SUCCESS);

                double *sendbuff = (rep%2==1) ? psi_storage1 : psi_storage2;
                double *recvbuff = (rep%2==1) ? psi_storage2 : psi_storage1;
                if (rep != reps) {
                    // first gather the orbitals in the rotation way
                    transfer_orbitals_blacscomm(pSPARC, sendbuff, recvbuff, rep, reqs, sizeof(double));
                }
                t2 = MPI_Wtime();
                t_comm += (t2 - t1);

                // solve poisson's equations 
                double *occ_ = (pSPARC->spin_typ == 1) ? (occ + spinor*pSPARC->Nstates) : occ;
                solve_allpair_poissons_equation_apply2Xi(pSPARC, Nband_M, psi + spinor*DMnd, DMndsp, sendbuff, DMnd, occ_, Xi + spinor*DMnd, DMndsp, rep);
            }
        }
    }

    #ifdef DEBUG
        if (!rank) printf("transferring orbitals in rotation wise took %.3f ms\n", t_comm*1e3);
    #endif
    pSPARC->Exxtime_cyc += t_comm;

    if (reps > 0) {
        free(psi_storage1);
        free(psi_storage2);
    }

    // Allreduce is unstable in valgrind test
    if (nproc_blacscomm > 1) {
        int res = MPI_Allreduce_overload(MPI_IN_PLACE, Xi, DMndsp*Nstates_occ, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        if (res != MPI_SUCCESS) {
            printf("ERROR: OOM in MPI_Allreduce_overload!\n");
            exit(EXIT_FAILURE);
        }
    }
}

void calculate_ACE_operator(SPARC_OBJ *pSPARC, double *psi, double *Xi) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank, rank_dmcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);

    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int Nstates_occ = pSPARC->Nstates_occ;

    double *M = (double *) malloc(Nstates_occ*Nstates_occ* sizeof(double));
    assert(M != NULL);

    double t1, t2;
    for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++ ) {
        // memset(M, 0, Nstates_occ*Nstates_occ* sizeof(double));

        // in case of hydrogen 
        if (pSPARC->Nstates_occ_list[min(spinor, pSPARC->Nspin_spincomm)] == 0) continue;

        t1 = MPI_Wtime();
        #ifdef ACCELGT
        if (pSPARC->useACCEL == 1) {
            ACCEL_DGEMM( CblasColMajor, CblasTrans, CblasNoTrans, Nstates_occ, pSPARC->Nband_bandcomm_M, DMnd,
                        1.0, Xi + spinor*DMnd, DMndsp, psi + spinor*DMnd, DMndsp, 
                        0.0, M + pSPARC->band_start_indx*Nstates_occ, Nstates_occ);
        } else
        #endif // ACCELGT
        {
            cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Nstates_occ, pSPARC->Nband_bandcomm_M, DMnd,
                        1.0, Xi + spinor*DMnd, DMndsp, psi + spinor*DMnd, DMndsp, 
                        0.0, M + pSPARC->band_start_indx*Nstates_occ, Nstates_occ);
        }

        if (pSPARC->npNd > 1) {
            MPI_Allreduce(MPI_IN_PLACE, M + pSPARC->band_start_indx*Nstates_occ, 
                            Nstates_occ*pSPARC->Nband_bandcomm_M, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
        }
        gather_blacscomm(pSPARC, Nstates_occ, Nstates_occ, M);        

        t2 = MPI_Wtime();
    #ifdef DEBUG
        if(!rank && !spinor) printf("rank = %2d, finding M = psi'* W took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif

        // perform Cholesky Factorization on -M
        // M = chol(-M), upper triangular matrix
        int info = 0;
        if (rank_dmcomm == 0) {
            for (int i = 0; i < Nstates_occ*Nstates_occ; i++) M[i] = -1.0 * M[i];

            #ifdef ACCELGT
            if (pSPARC->useACCEL == 1) {
                info = DPOTRF(LAPACK_COL_MAJOR, 'U', Nstates_occ, M, Nstates_occ);                
            } else
            #endif // ACCELGT
            {
                info = LAPACKE_dpotrf (LAPACK_COL_MAJOR, 'U', Nstates_occ, M, Nstates_occ);
            }
        }

        if (pSPARC->npNd > 1) {
            MPI_Bcast(M, Nstates_occ*Nstates_occ, MPI_DOUBLE, 0, pSPARC->dmcomm);
        }

        t1 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank && !spinor) 
            printf("==Cholesky Factorization: "
                "info = %d, computing Cholesky Factorization using dpotrf: %.3f ms\n", 
                info, (t1 - t2)*1e3);
    #endif

        #ifdef ACCELGT
        if (pSPARC->useACCEL == 1) {
            ACCEL_DTRSM(CblasColMajor, CblasRight, CblasUpper,
                CblasNoTrans, CblasNonUnit, DMnd, Nstates_occ, 1.0, M, Nstates_occ, Xi + spinor*DMnd, DMndsp);
            
        } else
        #endif // ACCELGT
        {
            cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper,
                CblasNoTrans, CblasNonUnit, DMnd, Nstates_occ, 1.0, M, Nstates_occ, Xi + spinor*DMnd, DMndsp);
        }
        t2 = MPI_Wtime();
    #ifdef DEBUG
        if (!rank && !spinor) 
            printf("==Triangular matrix equation: "
                "Solving triangular matrix equation using dtrsm: %.3f ms\n", (t2 - t1)*1e3);
    #endif
    }
    free(M);
}


/**
 * @brief   Solve half of poissons equation locally and apply to Xi
 */
void solve_half_local_poissons_equation_apply2Xi(SPARC_OBJ *pSPARC, int ncol, double *psi, int ldp, double *occ, double *Xi, int ldxi)
{
    int i, j, k, rank, Nband, DMnd;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ_i, occ_j, *rhs, *sol;

#ifdef DEBUG
    double t1, t2, tt1, tt2;
#endif

    /******************************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DMnd = pSPARC->Nd_d_dmcomm;        
    Nband = pSPARC->Nband_bandcomm;
    if (ncol == 0) return;

    rhs_list_i = (int*) calloc(Nband * ncol, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nband * ncol, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // get index for rhs of poissons equations
    count = 0;
    for (j = 0; j < Nband; j++) {
        if (occ[j + pSPARC->band_start_indx] < 1e-6) continue;
        for (i = j; i < ncol; i++) {             
            rhs_list_i[count] = i;
            rhs_list_j[count] = j;
            count ++;
        }
    }
    num_rhs = count;
    if (num_rhs == 0) {
        free(rhs_list_i);
        free(rhs_list_j);
        return;
    }

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    
    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                    num_rhs : pSPARC->ExxMemBatch * pSPARC->npNd;
                    
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
    rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    sol = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
    assert(rhs != NULL && sol != NULL);

    /*************** Solve all Poisson's equation and find M ****************/
    for (loop = 0; loop < NL; loop ++) {
    
    #ifdef DEBUG
        tt1 = MPI_Wtime();
    #endif
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            for (k = 0; k < DMnd; k++) {
                rhs[k + (count-base)*DMnd] = psi[k + j*ldp] * psi[k + i*ldp];
            }
        }
    #ifdef DEBUG
        tt2 = MPI_Wtime();
        pSPARC->Exxtime_rhs += (tt2-tt1);
    #endif

        poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, pSPARC->dmcomm);

    #ifdef DEBUG
        tt1 = MPI_Wtime();
    #endif

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            
            occ_i = occ[i + pSPARC->band_start_indx];
            occ_j = occ[j + pSPARC->band_start_indx];

            for (k = 0; k < DMnd; k++) {
                Xi[k + i*ldxi] -= occ_j * psi[k + j*ldp] * sol[k + (count-base)*DMnd] / pSPARC->dV;                
            }
            
            if (i != j && occ_i > 1e-6) {
                for (k = 0; k < DMnd; k++) {
                    Xi[k + j*ldxi] -= occ_i * psi[k + i*ldp] * sol[k + (count-base)*DMnd] / pSPARC->dV;
                }
            }
        }
    #ifdef DEBUG
        tt2 = MPI_Wtime();
        pSPARC->Exxtime_rhs += (tt2-tt1);
    #endif
    }
    free(rhs);
    free(sol);

    free(rhs_list_i);
    free(rhs_list_j);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",rank,(t2-t1)*1e3); 
#endif
}

/**
 * @brief   transfer orbitals in a cyclic rotation way to save memory
 */
void transfer_orbitals_blacscomm(SPARC_OBJ *pSPARC, void *sendbuff, void *recvbuff, int shift, MPI_Request *reqs, int unit_size)
{
    assert(unit_size == sizeof(double) || unit_size == sizeof(double _Complex));
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    int size, rank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);

    int DMnd = pSPARC->Nd_d_dmcomm;    
    int Ns = pSPARC->Nstates;
    int NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    int srank = (rank-shift+size)%size;
    int rrank = (rank-shift-1+size)%size;
    int Nband_send = srank < (Ns / NB) ? NB : (srank == (Ns / NB) ? (Ns % NB) : 0);
    int Nband_recv = rrank < (Ns / NB) ? NB : (rrank == (Ns / NB) ? (Ns % NB) : 0);
    int rneighbor = (rank+1)%size;
    int lneighbor = (rank-1+size)%size;

    if (unit_size == sizeof(double)) {
        MPI_Irecv(recvbuff, DMnd*Nband_recv, MPI_DOUBLE, lneighbor, 111, blacscomm, &reqs[1]);
        MPI_Isend(sendbuff, DMnd*Nband_send, MPI_DOUBLE, rneighbor, 111, blacscomm, &reqs[0]);
    } else {
        MPI_Irecv(recvbuff, DMnd*Nband_recv, MPI_DOUBLE_COMPLEX, lneighbor, 111, blacscomm, &reqs[1]);
        MPI_Isend(sendbuff, DMnd*Nband_send, MPI_DOUBLE_COMPLEX, rneighbor, 111, blacscomm, &reqs[0]);
    }
    
}


/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi(
    SPARC_OBJ *pSPARC, int ncol, double *psi, int ldp, double *psi_storage, int ldps, double *occ, double *Xi, int ldxi, int shift)
{
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    int size, rank, grank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);

    // when blacscomm is composed of even number of processors
    // second half of them will solve one less rep than the first half to avoid repetition
    int reps = (size - 2) / 2 + 1;
    if (size%2 == 0 && rank >= size/2 && shift == reps) return;

    int i, j, k, nproc_dmcomm, Ns, DMnd;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ_i, occ_j, *rhs, *sol, *Xi_l, *Xi_r;

#ifdef DEBUG
    double t1, t2, tt1, tt2;
#endif

    /******************************************************************************/

    if (ncol == 0) return;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);
    DMnd = pSPARC->Nd_d_dmcomm;    
    Ns = pSPARC->Nstates;    

    int source = (rank-shift+size)%size;
    int NB = (Ns - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    int Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
    int band_start_indx_source = source * NB;

    rhs_list_i = (int*) calloc(Nband_source * ncol, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nband_source * ncol, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // get index for rhs of poissons equations
    count = 0;
    for (j = 0; j < Nband_source; j++) {
        occ_j = occ[j + band_start_indx_source];
        for (i = 0; i < ncol; i++) {     
            occ_i = occ[i + pSPARC->band_start_indx];
            if (occ_j < 1e-6 && (occ_i < 1e-6 || j + band_start_indx_source >= pSPARC->Nstates_occ)) continue;
            rhs_list_i[count] = i;
            rhs_list_j[count] = j;
            count ++;
        }
    }
    num_rhs = count;

    Xi_l = Xi + pSPARC->band_start_indx * ldxi;
    Xi_r = Xi + band_start_indx_source * ldxi;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    if (num_rhs > 0) {
        batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * nproc_dmcomm;
                        
        NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
        rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
        sol = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && sol != NULL);

        /*************** Solve all Poisson's equation and find M ****************/
        for (loop = 0; loop < NL; loop ++) {

        #ifdef DEBUG
            tt1 = MPI_Wtime();
        #endif

            base = batch_num_rhs*loop;
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*ldps] * psi[k + i*ldp];
                }
            }

        #ifdef DEBUG
            tt2 = MPI_Wtime();
            pSPARC->Exxtime_rhs += (tt2-tt1);
        #endif

            poissonSolve(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, pSPARC->dmcomm);
            
        #ifdef DEBUG
            tt1 = MPI_Wtime();
        #endif

            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ[i + pSPARC->band_start_indx];
                occ_j = occ[j + band_start_indx_source];

                if (occ_j > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi_l[k + i*ldxi] -= occ_j * psi_storage[k + j*ldps] * sol[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }

                if (occ_i > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi_r[k + j*ldxi] -= occ_i * psi[k + i*ldp] * sol[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }
            }
        
        #ifdef DEBUG
            tt2 = MPI_Wtime();
            pSPARC->Exxtime_rhs += (tt2-tt1);
        #endif

        }
        free(rhs);
        free(sol);
    }

    free(rhs_list_i);
    free(rhs_list_j);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!grank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",grank,(t2-t1)*1e3); 
#endif
}