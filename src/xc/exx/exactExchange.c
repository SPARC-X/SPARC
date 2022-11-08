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
#include "lapVecRoutines.h"
#include "linearSolver.h"
#include "exactExchangeKpt.h"
#include "tools.h"
#include "parallelization.h"
#include "electronicGroundState.h"
#include "exchangeCorrelation.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)


/**
 * @brief   Outer loop of SCF using Vexx (exact exchange potential)
 */
void Exact_Exchange_loop(SPARC_OBJ *pSPARC) {
    int i, rank, DMnd, blacs_size, kpt_size, spn_i;
    double t1, t2, ACE_time = 0.0;
    FILE *output_fp;

    DMnd = pSPARC->Nd_d_dmcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /************************ Exact exchange potential parameters ************************/
    int count_xx = 0;
    double Eexx_pre = pSPARC->Eexx, err_Exx = pSPARC->TOL_FOCK + 1;
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
    while (count_xx < pSPARC->MAXIT_FOCK) {
    #ifdef DEBUG
    if(!rank) 
        printf("\nHartree-Fock Outer Loop: %d \n",count_xx + 1);
    #endif  

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
                int psi_shift = DMnd * pSPARC->Nband_bandcomm;
                for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    ACE_operator(pSPARC, pSPARC->Xorb + spn_i*psi_shift, 
                        pSPARC->occ + spn_i*pSPARC->Nstates, spn_i, pSPARC->Xi + spn_i*xi_shift);
                }
            } else {
                gather_psi_occ_outer_kpt(pSPARC, pSPARC->psi_outer_kpt, pSPARC->occ_outer);
                allocate_ACE_kpt(pSPARC);
                int xi_shift = DMnd * pSPARC->Nstates_occ[0] * pSPARC->Nkpts_kptcomm;
                int psi_shift = DMnd * pSPARC->Nband_bandcomm * pSPARC->Nkpts_kptcomm;
                int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

                for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                    ACE_operator_kpt(pSPARC, pSPARC->Xorb_kpt + spn_i*psi_shift, 
                        pSPARC->occ_outer + spn_i*occ_outer_shift, spn_i, pSPARC->Xi_kpt + spn_i*xi_shift);
                }
            }
            t2 = MPI_Wtime();
            pSPARC->ACEtime += (t2 - t1);
            ACE_time = (t2 - t1);
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

        if(count_xx > 0) {
            err_Exx = fabs(Eexx_pre - pSPARC->Eexx)/pSPARC->n_atom;
            MPI_Bcast(&err_Exx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);                 // TODO: Create bridge comm 
            if(!rank) {
                // write to .out file
                output_fp = fopen(pSPARC->OutFilename,"a");
                fprintf(output_fp,"Exx outer loop error: %.10e \n",err_Exx);
                fclose(output_fp);
            }
            if (err_Exx < pSPARC->TOL_FOCK && count_xx >= pSPARC->MINIT_FOCK) break;
        }
        Eexx_pre = pSPARC->Eexx;

        if(!rank) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (pSPARC->ACEFlag == 0)
                fprintf(output_fp,"\nNo.%d Exx outer loop. \n", count_xx + 1);
            else 
                fprintf(output_fp,"\nNo.%d Exx outer loop. ACE timing: %.3f (sec)\n", count_xx + 1, ACE_time);
            fclose(output_fp);
        }

        scf_loop(pSPARC);
        count_xx ++;
    }

    #ifdef DEBUG
    if(!rank) 
        printf("\nFinished outer loop in %d steps!\n", count_xx);
    #endif  

    if (count_xx == pSPARC->MAXIT_FOCK) {
        // update the final exact exchange energy if necessary
        pSPARC->Exc -= pSPARC->Eexx;
        pSPARC->Etot += 2 * pSPARC->Eexx;
        if (pSPARC->isGammaPoint == 1)
            evaluate_exact_exchange_energy(pSPARC);
        else
            evaluate_exact_exchange_energy_kpt(pSPARC);

        pSPARC->Exc += pSPARC->Eexx;
        pSPARC->Etot -= 2*pSPARC->Eexx;

        err_Exx = fabs(Eexx_pre - pSPARC->Eexx)/pSPARC->n_atom;
        MPI_Bcast(&err_Exx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);                 // TODO: Create bridge comm 
        if(!rank && count_xx > 0) {
            // write to .out file
            output_fp = fopen(pSPARC->OutFilename,"a");
            fprintf(output_fp,"Exx outer loop error: %.10e \n",err_Exx);
            fclose(output_fp);
        }
    }

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
 * @brief   Evaluating Exact Exchange potential
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ncol, int DMnd, double *Hx, int spin, MPI_Comm comm) {        
    int rank, Lanczos_flag, dims[3];
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
    double occ, *rhs, *Vi, exx_frac, occ_alpha;

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
            occ_alpha = occ * exx_frac;
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
    if (ncol != 1) {
        cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Ns_occ, ncol, DMnd,
                    1.0, Xi, DMnd, X, DMnd, 0.0, Xi_times_psi, Ns_occ);
    } else {
        cblas_dgemv( CblasColMajor, CblasTrans, DMnd, Ns_occ, 1.0, 
                    Xi, DMnd, X, 1, 0.0, Xi_times_psi, 1);
    }

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Ns_occ*ncol, 
                      MPI_DOUBLE, MPI_SUM, comm);
    }

    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    if (ncol != 1) {
        cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Ns_occ,
                    -pSPARC->exx_frac, Xi, DMnd, Xi_times_psi, Ns_occ, 1.0, Hx, DMnd);
    } else {
        cblas_dgemv( CblasColMajor, CblasNoTrans, DMnd, Ns_occ, -pSPARC->exx_frac, 
                    Xi, DMnd, Xi_times_psi, 1, 1.0, Hx, 1);
    }

    free(Xi_times_psi);
}



/**
 * @brief   Evaluate Exact Exchange Energy
 */
void evaluate_exact_exchange_energy(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, rank, size, spn_i;
    int Ns, ncol, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, *rhs, *Vi, *psi_outer, temp, *occ_outer, *psi;
    MPI_Comm comm;

#ifdef DEBUG
    double t1, t2;
#endif

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

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
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

        pSPARC->Eexx /= pSPARC->dV;

    } else {
        
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            int Ns_occ = pSPARC->Nstates_occ[spn_i];
            int Nband_bandcomm_M = pSPARC->Nband_bandcomm_M[spn_i];
            if (Nband_bandcomm_M == 0) {continue;};

            double *Xi_times_psi = (double *) calloc(Nband_bandcomm_M * Ns_occ, sizeof(double));
            assert(Xi_times_psi != NULL);

            // perform matrix multiplication psi' * X using ScaLAPACK routines
            cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, Nband_bandcomm_M, Ns_occ, DMnd,
                        1.0, pSPARC->Xorb + spn_i * psi_shift, DMnd, pSPARC->Xi + spn_i * xi_shift, 
                        DMnd, 0.0, Xi_times_psi, Nband_bandcomm_M);

            if (size > 1) {
                // sum over all processors in dmcomm
                MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nband_bandcomm_M*Ns_occ, 
                            MPI_DOUBLE, MPI_SUM, comm);
            }

            for (i = 0; i < Nband_bandcomm_M; i++) {
                temp = 0.0;
                for (j = 0; j < Ns_occ; j++) {
                    temp += Xi_times_psi[i+j*Nband_bandcomm_M] * Xi_times_psi[i+j*Nband_bandcomm_M];
                }
                temp *= pSPARC->occ[i + pSPARC->band_start_indx + Ns*spn_i];
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

#ifdef DEBUG
    t2 = MPI_Wtime();
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
    int i, k, lsize, lrank, ncolp;
    int *sendcounts, *sdispls, *recvcounts, *rdispls, **DMVertices, *ncolpp;
    int coord_comm[3], gridsizes[3], DNx, DNy, DNz, Nd, Nx, Ny, Nz;
    double *rhs_loc, *Vi_loc, *rhs_loc_order, *Vi_loc_order, *f;
    sendcounts = sdispls = recvcounts = rdispls = ncolpp = NULL;
    rhs_loc = Vi_loc = rhs_loc_order = Vi_loc_order = f = NULL;
    DMVertices = NULL;

    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);    
    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz;     
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
    int i, j, k, l, t, p, *coord;
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
    int ii, i, j, k, l, t, p, q, *coord, start, seg;
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
        ks, ke, is_phi, ie_phi, js_phi, je_phi,
        ks_phi, ke_phi, nx_phi, ny_phi, nz_phi, nd_phi, i_phi, j_phi, k_phi,
        DMCorVert[6][6];
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
    int i, grank, lrank, lsize, Ns, DMnd, Nband, spn_i;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;    
    Ns = pSPARC->Nstates;

    int DMndNband = DMnd * Nband;
    int DMndNs = DMnd * Ns;
    int Nstotal = Ns * pSPARC->Nspin_spincomm;
    int shift = pSPARC->band_start_indx * DMnd;
    // Save orbitals and occupations and to construct exact exchange operator
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
        for (i = 0; i < DMndNband; i++) 
            psi_outer[i + shift + spn_i*DMndNs] = pSPARC->Xorb[i + spn_i*DMndNband];
    for (i = 0; i < Nstotal; i++) 
        occ_outer[i] = pSPARC->occ[i];

    /********************************************************************/
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
        gather_blacscomm(pSPARC, Ns, psi_outer + spn_i*DMndNs);

    /********************************************************************/
    if (pSPARC->flag_kpttopo_dm && pSPARC->ACEFlag == 0) {
        int rank_kptcomm_topo;
        int NsNkNsp = pSPARC->Nstates * pSPARC->Nkpts_sym * pSPARC->Nspin_spincomm;
        MPI_Comm_rank(pSPARC->kptcomm_topo, &rank_kptcomm_topo);
        if (pSPARC->flag_kpttopo_dm_type == 1) {
            if (!rank_kptcomm_topo)
                MPI_Bcast(occ_outer, NsNkNsp, MPI_DOUBLE, MPI_ROOT, pSPARC->kpttopo_dmcomm_inter);
            else
                MPI_Bcast(occ_outer, NsNkNsp, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kpttopo_dmcomm_inter);
        } else {
            MPI_Bcast(occ_outer, NsNkNsp, MPI_DOUBLE, 0, pSPARC->kpttopo_dmcomm_inter);
        }
    }
}



/**
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm(SPARC_OBJ *pSPARC, int Ncol, double *vec)
{
    if (pSPARC->blacscomm == MPI_COMM_NULL) return;
    int i, grank, lrank, lsize, DMnd;
    int *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &lrank);
    MPI_Comm_size(pSPARC->blacscomm, &lsize);

    DMnd = pSPARC->Nd_d_dmcomm;

    if (lsize > 1) {
        recvcounts = (int*) malloc(sizeof(int)* lsize);
        displs = (int*) malloc(sizeof(int)* lsize);
        assert(recvcounts != NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < lsize; i++){
            recvcounts[i] = (i < (Ncol / NB) ? NB : (i == (Ncol / NB) ? (Ncol % NB) : 0)) * DMnd;
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
    if (pSPARC->spincomm_index < 0) return;
    int i, rank, DMnd, Ns_occ, Ns, spn_i, Ns_occ_temp[2];
    int sum_temp, sum;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMnd *= pSPARC->Nspinor;
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

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        Ns_occ = pSPARC->Nstates_occ[spn_i] = Ns_occ_temp[spn_i];
        if (Ns_occ < pSPARC->band_start_indx)
            pSPARC->Nband_bandcomm_M[spn_i] = 0;
        else {
            pSPARC->Nband_bandcomm_M[spn_i] = min(pSPARC->Nband_bandcomm, Ns_occ - pSPARC->band_start_indx);
        }        
    }

#if defined(USE_MKL) || defined(USE_SCALAPACK)
    // create SCALAPACK information for ACE operator
    int nprow, npcol, myrow, mycol;
    // get coord of each process in original context
    Cblacs_gridinfo( pSPARC->ictxt_blacs, &nprow, &npcol, &myrow, &mycol );
    int ZERO = 0, mb, nb, llda, info;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        Ns_occ = pSPARC->Nstates_occ[spn_i];
        mb = max(1, Ns_occ);
        nb = mb;
        // nb = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // equal to ceil(Nstates/npband), for int only
        // set up descriptor for storage of orbitals in ictxt_blacs (original)
        llda = mb;
        if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
            descinit_(pSPARC->desc_M[spn_i], &Ns_occ, &Ns_occ,
                    &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs, &llda, &info);
            pSPARC->nrows_M[spn_i] = numroc_( &Ns_occ, &mb, &myrow, &ZERO, &nprow);
            pSPARC->ncols_M[spn_i] = numroc_( &Ns_occ, &nb, &mycol, &ZERO, &npcol);
        } else {
            for (i = 0; i < 9; i++)
                pSPARC->desc_M[spn_i][i] = -1;
            pSPARC->nrows_M[spn_i] = pSPARC->ncols_M[spn_i] = 0;
        }

        // descriptor for Xi 
        mb = max(1, DMnd);
        nb = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // equal to ceil(Nstates/npband), for int only
        // set up descriptor for storage of orbitals in ictxt_blacs (original)
        llda = max(1, DMnd);
        if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
            descinit_(pSPARC->desc_Xi[spn_i], &DMnd, &Ns_occ,
                    &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs, &llda, &info);
        } else {
            for (i = 0; i < 9; i++)
                pSPARC->desc_Xi[spn_i][i] = -1;
        }
    }
#endif
}



/**
 * @brief   Create ACE operator in dmcomm
 *
 *          Using occupied + extra orbitals to construct the ACE operator 
 *          Due to the symmetry of ACE operator, only half Poisson's 
 *          equations need to be solved.
 */
void ACE_operator(SPARC_OBJ *pSPARC, double *psi, double *occ, int spn_i, double *Xi) 
{
    int i, rank, nproc_dmcomm, Nband_M, DMnd, ONE = 1, Ns_occ;    
    double *M, t1, t2, alpha = 1.0, beta = 0.0, *Xi_, *psi_storage1, *psi_storage2, t_comm;
    /******************************************************************************/

    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    Nband_M = pSPARC->Nband_bandcomm_M[spn_i];
    DMnd = pSPARC->Nd_d_dmcomm;
    Ns_occ = pSPARC->Nstates_occ[spn_i];    

    memset(Xi, 0, sizeof(double) * Ns_occ*DMnd);
    // if Nband==0 here, Xi_ won't be used anyway
    Xi_ = Xi + pSPARC->band_start_indx * DMnd;

    int nproc_blacscomm = pSPARC->npband;
    int reps = (nproc_blacscomm == 1) ? 0 : ((nproc_blacscomm - 2) / 2 + 1); // ceil((nproc_blacscomm-1)/2)
    int Nband_max = (pSPARC->Nstates - 1) / pSPARC->npband + 1;

    MPI_Request reqs[2];
    psi_storage1 = psi_storage2 = NULL;
    if (reps > 0) {
        psi_storage1 = (double *) calloc(sizeof(double), DMnd * Nband_max);
        psi_storage2 = (double *) calloc(sizeof(double), DMnd * Nband_max);
    }
    
    t_comm = 0;
    for (int rep = 0; rep <= reps; rep++) {
        if (rep == 0) {
            if (reps > 0) {
                t1 = MPI_Wtime();
                // first gather the orbitals in the rotation way
                transfer_orbitals_blacscomm(pSPARC, psi, psi_storage1, rep, reqs);
                t2 = MPI_Wtime();
                t_comm += (t2 - t1);
            }
            // solve poisson's equations 
            solve_half_local_poissons_equation_apply2Xi(pSPARC, Nband_M, psi, occ, Xi_);
        } else {
            t1 = MPI_Wtime();
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            double *sendbuff = (rep%2==1) ? psi_storage1 : psi_storage2;
            double *recvbuff = (rep%2==1) ? psi_storage2 : psi_storage1;
            if (rep != reps) {
                // first gather the orbitals in the rotation way
                transfer_orbitals_blacscomm(pSPARC, sendbuff, recvbuff, rep, reqs);
            }
            t2 = MPI_Wtime();
            t_comm += (t2 - t1);

            // solve poisson's equations 
            solve_allpair_poissons_equation_apply2Xi(pSPARC, Nband_M, psi, sendbuff, occ, Xi, rep, Ns_occ);
        }
    }

    #ifdef DEBUG
        if (!rank) printf("transferring orbitals in rotation wise took %.3f ms\n", t_comm*1e3);
    #endif

    if (reps > 0) {
        free(psi_storage1);
        free(psi_storage2);
    }

    // MPI_Allreduce(MPI_IN_PLACE, Xi, DMnd*Ns_occ, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    // Allreduce is unstable in valgrind test
    if (nproc_blacscomm > 1) {
        MPI_Request req;
        MPI_Status  sta;
        MPI_Iallreduce(MPI_IN_PLACE, Xi, DMnd*Ns_occ, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm, &req);
        MPI_Wait(&req, &sta);
    }

    /******************************************************************************/
    int nrows_M = pSPARC->nrows_M[spn_i];
    int ncols_M = pSPARC->ncols_M[spn_i];
    M = (double *) calloc(nrows_M * ncols_M, sizeof(double));
    assert(M != NULL);

    t1 = MPI_Wtime();
    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    // perform matrix multiplication psi' * W using ScaLAPACK routines    
    pdgemm_("T", "N", &Ns_occ, &Ns_occ, &DMnd, &alpha, 
            psi, &ONE, &ONE, pSPARC->desc_Xi[spn_i], Xi_, 
            &ONE, &ONE, pSPARC->desc_Xi[spn_i], &beta, M, &ONE, &ONE, 
            pSPARC->desc_M[spn_i]);
    #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    // add the implementation without SCALAPACK
    exit(255);
    #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    if (nproc_dmcomm > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, M, nrows_M*ncols_M,
                      MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && !spn_i) printf("rank = %2d, finding M = psi'* W took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif

    // perform Cholesky Factorization on -M
    // M = chol(-M), upper triangular matrix
    for (i = 0; i < nrows_M*ncols_M; i++) M[i] = -1.0 * M[i];

    t1 = MPI_Wtime();
    int info = 0;
    if (nrows_M*ncols_M > 0) {
        info = LAPACKE_dpotrf (LAPACK_COL_MAJOR, 'U', Ns_occ, M, Ns_occ);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (!rank && !spn_i) 
        printf("==Cholesky Factorization: "
               "info = %d, computing Cholesky Factorization using LAPACKE_dpotrf: %.3f ms\n", 
               info, (t2 - t1)*1e3);
    #else
    (void) info; // suppress unused var warning
    #endif

    // Xi = WM^(-1)
    t1 = MPI_Wtime();
    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    pdtrsm_("R", "U", "N", "N", &DMnd, &Ns_occ, &alpha, 
                M, &ONE, &ONE, pSPARC->desc_M[spn_i], 
                Xi_, &ONE, &ONE, pSPARC->desc_Xi[spn_i]);
    #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    // add the implementation without SCALAPACK
    exit(255);
    #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (!rank && !spn_i) 
        printf("==Triangular matrix equation: "
               "Solving triangular matrix equation using cblas_dtrsm: %.3f ms\n", (t2 - t1)*1e3);
    #endif

    free(M);

    // gather all columns of Xi
    gather_blacscomm(pSPARC, Ns_occ, Xi);
}

/**
 * @brief   Solve half of poissons equation locally and apply to Xi
 */
void solve_half_local_poissons_equation_apply2Xi(SPARC_OBJ *pSPARC, int ncol, double *psi, double *occ, double *Xi)
{
    int i, j, k, rank, dims[3], Nband, DMnd;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ_i, occ_j, *rhs, *Vi;

#ifdef DEBUG
    double t1, t2;
#endif

    /******************************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DMnd = pSPARC->Nd_d_dmcomm;    
    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
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

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    if (num_rhs > 0) {
        batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                        num_rhs : pSPARC->EXXMem_batch * pSPARC->npNd;
                        
        NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
        rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
        Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && Vi != NULL);

        /*************** Solve all Poisson's equation and find M ****************/
        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi[k + j*DMnd] * psi[k + i*DMnd];
                }
            }
            
            poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, pSPARC->dmcomm);

            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ[i + pSPARC->band_start_indx];
                occ_j = occ[j + pSPARC->band_start_indx];

                for (k = 0; k < DMnd; k++) {
                    Xi[k + i*DMnd] -= occ_j * psi[k + j*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                }

                if (i != j && occ_i > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi[k + j*DMnd] -= occ_i * psi[k + i*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }
            }
        }
        free(rhs);
        free(Vi);
    }

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
void transfer_orbitals_blacscomm(SPARC_OBJ *pSPARC, double *sendbuff, double *recvbuff, int shift, MPI_Request *reqs)
{
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

    MPI_Irecv(recvbuff, DMnd*Nband_recv, MPI_DOUBLE, lneighbor, 111, blacscomm, &reqs[1]);
    MPI_Isend(sendbuff, DMnd*Nband_send, MPI_DOUBLE, rneighbor, 111, blacscomm, &reqs[0]);
}


/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi(
    SPARC_OBJ *pSPARC, int ncol, double *psi, double *psi_storage, double *occ, double *Xi, int shift, int Ns_occ)
{
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    int size, rank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);

    // when blacscomm is composed of even number of processors
    // second half of them will solve one less rep than the first half to avoid repetition
    int reps = (size - 2) / 2 + 1;
    if (size%2 == 0 && rank >= size/2 && shift == reps) return;

    int i, j, k, nproc_dmcomm, Ns, dims[3], DMnd;
    int *rhs_list_i, *rhs_list_j, num_rhs, count, loop, batch_num_rhs, NL, base;
    double occ_i, occ_j, *rhs, *Vi, *Xi_l, *Xi_r;

#ifdef DEBUG
    double t1, t2;
#endif

    /******************************************************************************/

    if (ncol == 0) return;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);
    DMnd = pSPARC->Nd_d_dmcomm;    
    Ns = pSPARC->Nstates;
    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;

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
            if (occ_j < 1e-6 && (occ_i < 1e-6 || j + band_start_indx_source >= Ns_occ)) continue;
            rhs_list_i[count] = i;
            rhs_list_j[count] = j;
            count ++;
        }
    }
    num_rhs = count;

    Xi_l = Xi + pSPARC->band_start_indx * DMnd;
    Xi_r = Xi + band_start_indx_source * DMnd;

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    if (num_rhs > 0) {
        batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                        num_rhs : pSPARC->EXXMem_batch * nproc_dmcomm;
                        
        NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required
        rhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
        Vi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && Vi != NULL);

        /*************** Solve all Poisson's equation and find M ****************/
        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*DMnd] * psi[k + i*DMnd];
                }
            }
            
            poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, pSPARC->dmcomm);
            
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ[i + pSPARC->band_start_indx];
                occ_j = occ[j + band_start_indx_source];

                if (occ_j > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi_l[k + i*DMnd] -= occ_j * psi_storage[k + j*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }

                if (occ_i > 1e-6) {
                    for (k = 0; k < DMnd; k++) {
                        Xi_r[k + j*DMnd] -= occ_i * psi[k + i*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                    }
                }
            }
        }
        free(rhs);
        free(Vi);
    }

    free(rhs_list_i);
    free(rhs_list_j);
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif
}
