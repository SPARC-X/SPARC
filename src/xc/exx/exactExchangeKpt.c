/***
 * @file    exactExchangeKpt.c
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
#include "exactExchangeKpt.h"
#include "tools.h"
#include "parallelization.h"
#include "kroneckerLaplacian.h"

#ifdef ACCELGT
#include "cuda_accel_kpt.h"
#include "cuda_exactexchange.h"
#include "cuda_linearAlgebra.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL (1e-14)


/**
 * @brief   Create ACE operator in dmcomm for each k-point
 *
 *          Using occupied + extra orbitals to construct the ACE operator for each k-point.
 *          Note: there is no band parallelization when usign ACE, so there is only 1 dmcomm
 *          for each k-point. 
 */
void ACE_operator_kpt(SPARC_OBJ *pSPARC, 
    double _Complex *psi, double *occ_outer, double _Complex *Xi_kpt)
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
        ACCEL_solving_for_Xi_kpt(pSPARC, psi, occ_outer, Xi_kpt);
    } else 
#endif 
    {
        solving_for_Xi_kpt(pSPARC, psi, occ_outer, Xi_kpt);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) 
    printf("solving_for_Xi took: %.3f ms\n", (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif  

    calculate_ACE_operator_kpt(pSPARC, psi, Xi_kpt);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) 
    printf("calculate_ACE_operator took: %.3f ms\n", (t2-t1)*1e3);
#endif  
}


void solving_for_Xi_kpt(SPARC_OBJ *pSPARC, 
    double _Complex *psi, double *occ_outer, double _Complex *Xi_kpt)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int Nband = pSPARC->Nband_bandcomm;
    int Ns = pSPARC->Nstates;
    int Nband_M = pSPARC->Nband_bandcomm_M;
    int DMnd = pSPARC->Nd_d_dmcomm;     
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int Nstates_occ = pSPARC->Nstates_occ;
    int size_k = DMndsp * Nband;
    int size_k_xi = DMndsp * Nstates_occ;
    int size_k_ps = DMnd * Nband;
    /******************************************************************************/

    int kpt_bridge_comm_rank, kpt_bridge_comm_size;
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_comm_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_comm_size);
    int blacscomm_rank, blacscomm_size;
    MPI_Comm_rank(pSPARC->blacscomm, &blacscomm_rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacscomm_size);

    int reps_band = pSPARC->npband - 1;
    int Nband_max = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
    int reps_kpt = pSPARC->npkpt - 1;
    int Nkpthf_red_max = pSPARC->Nkpts_hf_kptcomm;
    for (int k = 0; k < pSPARC->npkpt; k++) 
        Nkpthf_red_max = max(Nkpthf_red_max, pSPARC->Nkpts_hf_list[k]);

    // starts to create Xi
    memset(Xi_kpt, 0, sizeof(double _Complex) * size_k_xi * pSPARC->Nkpts_kptcomm);
    double _Complex *psi_storage1_kpt, *psi_storage2_kpt = NULL;
    psi_storage1_kpt = (double _Complex *) malloc(sizeof(double _Complex) * DMnd * Nband * Nkpthf_red_max);
    assert(psi_storage1_kpt != NULL);
    if (reps_kpt > 0) {
        psi_storage2_kpt = (double _Complex *) malloc(sizeof(double _Complex) * DMnd * Nband * Nkpthf_red_max);
        assert(psi_storage2_kpt != NULL);
    }
    
    double _Complex *psi_storage1_band, *psi_storage2_band;
    psi_storage1_band = psi_storage2_band = NULL;
    if (reps_band > 0) {
        psi_storage1_band = (double _Complex *) malloc(sizeof(double _Complex) * DMnd * Nband_max);
        psi_storage2_band = (double _Complex *) malloc(sizeof(double _Complex) * DMnd * Nband_max);
        assert(psi_storage1_band != NULL && psi_storage2_band != NULL);
    }

    double t_comm = 0, t1, t2;
    MPI_Request reqs_kpt[2], reqs_band[2];

    for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        // in case of hydrogen 
        if (pSPARC->Nstates_occ_list[min(spinor, pSPARC->Nspin_spincomm)] == 0) continue;

        // extract and store all the orbitals for hybrid calculation
        int count = 0;
        for (int k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
            if (!pSPARC->kpthf_flag_kptcomm[k]) continue;
            copy_mat_blk(sizeof(double _Complex), psi + k*size_k + spinor*DMnd, DMndsp, DMnd, Nband, psi_storage1_kpt + count*size_k_ps, DMnd);
            count ++;
        }

        for (int rep_kpt = 0; rep_kpt <= reps_kpt; rep_kpt++) {
            // transfer the orbitals in the rotation way across kpt_bridge_comm
            double _Complex *sendbuff_kpt, *recvbuff_kpt;
            if (rep_kpt == 0) {
                sendbuff_kpt = psi_storage1_kpt;
                if (reps_kpt > 0) {
                    recvbuff_kpt = psi_storage2_kpt;
                    t1 = MPI_Wtime();
                    transfer_orbitals_kptbridgecomm(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt, sizeof(double _Complex));
                    t2 = MPI_Wtime();
                    t_comm += (t2 - t1);
                }
            } else {
                t1 = MPI_Wtime();
                MPI_Waitall(2, reqs_kpt, MPI_STATUSES_IGNORE);
                sendbuff_kpt = (rep_kpt%2==1) ? psi_storage2_kpt : psi_storage1_kpt;
                recvbuff_kpt = (rep_kpt%2==1) ? psi_storage1_kpt : psi_storage2_kpt;
                if (rep_kpt != reps_kpt) {
                    transfer_orbitals_kptbridgecomm(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt, sizeof(double _Complex));
                }
                t2 = MPI_Wtime();
                t_comm += (t2 - t1);
            }

            int source_kpt = (kpt_bridge_comm_rank-rep_kpt+kpt_bridge_comm_size)%kpt_bridge_comm_size;
            int nkpthf_red = pSPARC->Nkpts_hf_list[source_kpt];
            int kpthf_start_indx = pSPARC->kpthf_start_indx_list[source_kpt];
            
            for (int k = 0; k < nkpthf_red; k++) {
                int k_indx = k + kpthf_start_indx;
                int counts = pSPARC->kpthfred2kpthf[k_indx][0];
                
                double _Complex *sendbuff_band, *recvbuff_band;
                for (int rep_band = 0; rep_band <= reps_band; rep_band++) {
                    if (rep_band == 0) {
                        sendbuff_band = sendbuff_kpt + k*size_k_ps;
                        if (reps_band > 0) {
                            t1 = MPI_Wtime();
                            recvbuff_band = psi_storage1_band;
                            transfer_orbitals_blacscomm(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band, sizeof(double _Complex));
                            t2 = MPI_Wtime();
                            t_comm += (t2 - t1);
                        }
                    } else {
                        t1 = MPI_Wtime();
                        MPI_Waitall(2, reqs_band, MPI_STATUSES_IGNORE);
                        sendbuff_band = (rep_band%2==1) ? psi_storage1_band : psi_storage2_band;
                        recvbuff_band = (rep_band%2==1) ? psi_storage2_band : psi_storage1_band;
                        if (rep_band != reps_band) {
                            // transfer the orbitals in the rotation way across blacscomm
                            transfer_orbitals_blacscomm(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band, sizeof(double _Complex));
                        }
                        t2 = MPI_Wtime();
                        t_comm += (t2 - t1);
                    }
                    
                    for (int count = 0; count < counts; count++) {
                        int kpt_q = pSPARC->kpthfred2kpthf[k_indx][count+1];
                        int ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
                        double *occ = occ_outer + ll * Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * pSPARC->Nkpts_sym * Ns;
                        // solve poisson's equations 
                        for (int kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k ++) {
                            solve_allpair_poissons_equation_apply2Xi_kpt(pSPARC, Nband_M, 
                                psi+spinor*DMnd, DMndsp, sendbuff_band, DMnd, occ, Xi_kpt+spinor*DMnd, DMndsp, kpt_k, kpt_q, rep_band);
                        }
                    }
                }
            }
        }
    }
    free(psi_storage1_kpt);
    if (reps_kpt > 0) {
        free(psi_storage2_kpt);
    }
    if (reps_band > 0) {
        free(psi_storage1_band);
        free(psi_storage2_band);
    }
    
    #ifdef DEBUG
        if (!rank) printf("transferring orbitals in rotation wise took %.3f ms\n", t_comm*1e3);
    #endif
    pSPARC->Exxtime_cyc += t_comm;
}



void calculate_ACE_operator_kpt(SPARC_OBJ *pSPARC, double _Complex *psi, double _Complex *Xi_kpt)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank, rank_dmcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
    int Nband = pSPARC->Nband_bandcomm;
    int DMnd = pSPARC->Nd_d_dmcomm;     
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int Nstates_occ = pSPARC->Nstates_occ;
    int size_k = DMndsp * Nband;
    int size_k_xi = DMndsp * Nstates_occ;    

    double _Complex *M = (double _Complex *)calloc(Nstates_occ * Nstates_occ, sizeof(double _Complex));
    assert(M != NULL);
        
    double _Complex alpha = 1.0, beta = 0.0;
    double t1, t2;

    // i -- kpt_k, j -- kpt_q
    for (int kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k ++) {
        double _Complex *Xi_kpt_ = Xi_kpt + kpt_k*size_k_xi;

        gather_blacscomm_kpt(pSPARC, DMndsp, Nstates_occ, Xi_kpt_);

        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
            // in case of hydrogen 
            if (pSPARC->Nstates_occ_list[min(spinor, pSPARC->Nspin_spincomm-1)] == 0) continue;

            t1 = MPI_Wtime();
            #ifdef ACCELGT
            if (pSPARC->useACCEL == 1) {
                ACCEL_ZGEMM (CblasColMajor, CblasConjTrans, CblasNoTrans, 
                    Nstates_occ, pSPARC->Nband_bandcomm_M, DMnd, 
                    &alpha, Xi_kpt_ + spinor*DMnd, DMndsp, psi + kpt_k*size_k + spinor*DMnd, DMndsp,
                    &beta, M + pSPARC->band_start_indx*Nstates_occ, Nstates_occ);
            } else
            #endif // ACCELGT
            {
                cblas_zgemm (CblasColMajor, CblasConjTrans, CblasNoTrans, 
                    Nstates_occ, pSPARC->Nband_bandcomm_M, DMnd, 
                    &alpha, Xi_kpt_ + spinor*DMnd, DMndsp, psi + kpt_k*size_k + spinor*DMnd, DMndsp,
                    &beta, M + pSPARC->band_start_indx*Nstates_occ, Nstates_occ);
            }

            if (pSPARC->npNd > 1) {
                // sum over all processors in dmcomm
                MPI_Allreduce(MPI_IN_PLACE, M + pSPARC->band_start_indx*Nstates_occ, 
                            Nstates_occ*pSPARC->Nband_bandcomm_M, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
            }

            gather_blacscomm_kpt(pSPARC, Nstates_occ, Nstates_occ, M);

            t2 = MPI_Wtime();
        #ifdef DEBUG
            if(!rank && !spinor) printf("rank = %2d, finding M = psi'* W took %.3f ms\n",rank,(t2-t1)*1e3); 
        #endif

            // perform Cholesky Factorization on -M
            // M = chol(-M), upper triangular matrix
            // M = -0.5*(M + M') seems to lead to numerical issue. 
            // TODO: Check if it's required to "symmetrize" M.             
            int info = 0;
            if (rank_dmcomm == 0) {
                for (int i = 0; i < Nstates_occ * Nstates_occ; i++)   M[i] = -M[i];

                #ifdef ACCELGT
                if (pSPARC->useACCEL == 1) {
                    info = ZPOTRF(LAPACK_COL_MAJOR, 'U', Nstates_occ, M, Nstates_occ);
                } else 
                #endif // ACCELGT
                {
                    info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', Nstates_occ, M, Nstates_occ);
                }
            }

            if (pSPARC->npNd > 1) {
                MPI_Bcast(M, Nstates_occ*Nstates_occ, MPI_DOUBLE_COMPLEX, 0, pSPARC->dmcomm);
            }
                        
            t1 = MPI_Wtime();
            #ifdef DEBUG
            if (!rank && !spinor) 
                printf("==Cholesky Factorization: "
                    "info = %d, computing Cholesky Factorization using zpotrf: %.3f ms\n", 
                    info, (t1 - t2)*1e3);
            #endif

            // Xi = WM^(-1)
            #ifdef ACCELGT
            if (pSPARC->useACCEL == 1) {
                ACCEL_ZTRSM(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, 
                        CblasNonUnit, DMnd, Nstates_occ, &alpha, M, Nstates_occ, Xi_kpt_ + spinor*DMnd, DMndsp);
            } else 
            #endif // ACCELGT
            {
                cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, 
                        CblasNonUnit, DMnd, Nstates_occ, &alpha, M, Nstates_occ, Xi_kpt_ + spinor*DMnd, DMndsp);
            }

            t2 = MPI_Wtime();
            #ifdef DEBUG
            if (!rank && !spinor) 
                printf("==Triangular matrix equation: "
                    "Solving triangular matrix equation using ztrsm: %.3f ms\n", (t2 - t1)*1e3);
            #endif
        }
    }
    free(M);
}


/**
 * @brief   Evaluating Exact Exchange potential in k-point case
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ldx, int ncol, int DMnd, double _Complex *Hx, int ldhx, int spin, int kpt, MPI_Comm comm) 
{        
    int rank, Lanczos_flag;
    double _Complex *psi_outer, *Xi;
    double t1, t2, *occ_outer;
    
    MPI_Comm_rank(comm, &rank);
    Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
    /********************************************************************/

    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;    
    int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

    t1 = MPI_Wtime();
    if (pSPARC->ExxAcc == 0) {
        occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spin * occ_outer_shift) : pSPARC->occ_outer;
        psi_outer = (Lanczos_flag == 0) ? pSPARC->psi_outer_kpt + spin * DMnd : pSPARC->psi_outer_kptcomm_topo_kpt + spin * DMnd ;
        evaluate_exact_exchange_potential_kpt(pSPARC, X, ldx, ncol, DMnd, occ_outer, psi_outer, DMndsp, Hx, ldhx, kpt, comm);

    } else {
        Xi = (Lanczos_flag == 0) ? pSPARC->Xi_kpt + spin * DMnd : pSPARC->Xi_kptcomm_topo_kpt + spin * DMnd;
        evaluate_exact_exchange_potential_ACE_kpt(pSPARC, X, ldx, ncol, DMnd, Xi, DMndsp, Hx, ldhx, kpt, comm);
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
 * @param kpt_k           Local index of each k-point 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, double _Complex *X, int ldx, 
        int ncol, int DMnd, double *occ_outer, 
        double _Complex *psi_outer, int ldpo, double _Complex *Hx, int ldhx, int kpt_k, MPI_Comm comm) 
{
    int i, j, k, l, ll, ll_red, rank, Ns, num_rhs;
    int size, batch_num_rhs, NL, base, loop, Nkpts_hf;
    int *rhs_list_i, *rhs_list_j, *rhs_list_l;
    double occ, exx_frac, occ_alpha;
    double _Complex *rhs, *sol;

    Ns = pSPARC->Nstates;
    Nkpts_hf = pSPARC->Nkpts_hf;
    exx_frac = pSPARC->exx_frac;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);

    int size_k_po = ldpo * Ns;
    /********************************************************************/
    rhs_list_i = (int*) malloc(ncol * Ns * Nkpts_hf * sizeof(int)); 
    rhs_list_j = (int*) malloc(ncol * Ns * Nkpts_hf * sizeof(int)); 
    rhs_list_l = (int*) malloc(ncol * Ns * Nkpts_hf * sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_l != NULL);
    
    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (i = 0; i < ncol; i++) {
        for (l = 0; l < Nkpts_hf; l++) {
            ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
            ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
            for (j = 0; j < Ns; j++) {
                if (occ_outer[j + ll * Ns] > 1e-6) {
                    rhs_list_i[count] = i;              // col indx of X
                    rhs_list_j[count] = j;              // band indx of psi_outer
                    rhs_list_l[count] = l;              // k-point indx of psi_outer
                    count++;
                }
            }
        }
    }
    num_rhs = count;

    if (num_rhs == 0) {
        free(rhs_list_i);
        free(rhs_list_j);
        free(rhs_list_l);
        return;
    }

    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * size;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                                 // number of loops required                        

    rhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                        // right hand sides of Poisson's equation
    sol = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                         // the solution for each rhs
    assert(rhs != NULL && sol != NULL);

    /*************** Solve all Poisson's equation and apply to X ****************/    
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            l = rhs_list_l[count];
            ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
            ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
            if (pSPARC->kpthf_pn[l] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_outer[k + j*ldpo + ll_red*size_k_po]) * X[k + i*ldx];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_outer[k + j*ldpo + ll_red*size_k_po] * X[k + i*ldx];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, kpt_k + pSPARC->kpt_start_indx, l, comm);

        // Apply exact exchange potential to vector X
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            l = rhs_list_l[count];
            ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
            ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
            occ = occ_outer[j + ll * Ns];
            occ_alpha = occ * exx_frac;
            if (pSPARC->kpthf_pn[l] == 1) {
                for (k = 0; k < DMnd; k++) 
                    Hx[k + i*ldhx] -= occ_alpha * pSPARC->kptWts_hf * psi_outer[k + j*ldpo + ll_red*size_k_po] * sol[k + (count-base)*DMnd] / pSPARC->dV;
            } else {
                for (k = 0; k < DMnd; k++) 
                    Hx[k + i*ldhx] -= occ_alpha * pSPARC->kptWts_hf * conj(psi_outer[k + j*ldpo + ll_red*size_k_po]) * sol[k + (count-base)*DMnd] / pSPARC->dV;
            }
        }
    }

    free(rhs);
    free(sol);
    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_l);
}



/**
 * @brief   Evaluate Exact Exchange potential using ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param Xi              Xi of ACE operator 
 * @param Hx              Result of Hx plus fock operator times X 
 * @param kpt_k           Local index of each k-point 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_ACE_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ldx, int ncol, int DMnd, double _Complex *Xi, int ldxi,
        double _Complex *Hx, int ldhx, int kpt, MPI_Comm comm) 
{
    int rank, size, size_k_xi, Nstates_occ;
    Nstates_occ = pSPARC->Nstates_occ;
    size_k_xi = ldxi * Nstates_occ;
    double _Complex alpha = 1.0, beta = 0.0;    
    double _Complex *Xi_times_psi = (double _Complex *) malloc(Nstates_occ * ncol * sizeof(double _Complex));
    assert(Xi_times_psi != NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/

    // perform matrix multiplication Xi' * X using ScaLAPACK routines
    if (ncol != 1) {
        cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Nstates_occ, ncol, DMnd,
                    &alpha, Xi + kpt*size_k_xi, ldxi, X, ldx, &beta, Xi_times_psi, Nstates_occ);
    } else {
        cblas_zgemv( CblasColMajor, CblasConjTrans, DMnd, Nstates_occ, 
                    &alpha, Xi + kpt*size_k_xi, ldxi, X, 1, &beta, Xi_times_psi, 1);
    }

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nstates_occ*ncol, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }

    alpha = -pSPARC->exx_frac;
    beta = 1.0;
    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    if (ncol != 1) {
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Nstates_occ,
                    &alpha, Xi + kpt*size_k_xi, ldxi, Xi_times_psi, Nstates_occ, &beta, Hx, ldhx);
    } else {
        cblas_zgemv( CblasColMajor, CblasNoTrans, DMnd, Nstates_occ, 
                    &alpha, Xi + kpt*size_k_xi, ldxi, Xi_times_psi, 1, &beta, Hx, 1);
    }

    free(Xi_times_psi);
}


/**
 * @brief   Evaluate Exact Exchange Energy in k-point case.
 */
void evaluate_exact_exchange_energy_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, ll_red, m, grank, rank, size;
    int Ns, Ns_loc, DMnd, DMndsp, num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, *occ_outer;
    double _Complex *rhs, *sol, *psi_outer, *psi, *xi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_loc = pSPARC->Nkpts_kptcomm;
    Ns_loc = Nband * Nkpts_loc;         // for Xorb
    comm = pSPARC->dmcomm;
    pSPARC->Eexx = 0.0;

    int Nkpts_hf = pSPARC->Nkpts_hf;    
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (pSPARC->ExxAcc == 0) {
        int size_k = DMndsp * Nband;
        int size_k_ = DMndsp * Ns;
        int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

        int *rhs_list_i, *rhs_list_j, *rhs_list_l, *rhs_list_m;
        rhs_list_i = (int*) malloc(Ns_loc * Ns * Nkpts_hf * sizeof(int)); 
        rhs_list_j = (int*) malloc(Ns_loc * Ns * Nkpts_hf * sizeof(int)); 
        rhs_list_l = (int*) malloc(Ns_loc * Ns * Nkpts_hf * sizeof(int)); 
        rhs_list_m = (int*) malloc(Ns_loc * Ns * Nkpts_hf * sizeof(int)); 
        assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_l != NULL && rhs_list_m != NULL);

        for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
            psi_outer = pSPARC->psi_outer_kpt + spinor * DMnd;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * occ_outer_shift) : pSPARC->occ_outer;
            psi = pSPARC->Xorb_kpt + spinor * DMnd;
        
            // Find the number of Poisson's equation required to be solved
            // Using the occupation threshold 1e-6
            int count = 0;
            for (m = 0; m < Nkpts_loc; m++) {
                for (i = 0; i < Nband; i++) {
                    for (l = 0; l < Nkpts_hf; l++) {
                        ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                        for (j = 0; j < Ns; j++) {
                            if (occ_outer[j + ll * Ns] > 1e-6 && 
                                occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns] > 1e-6 ) {
                                rhs_list_i[count] = i;              // col indx of Xorb
                                rhs_list_j[count] = j;              // band indx of psi_outer
                                rhs_list_l[count] = l;              // k-point indx of psi_outer
                                rhs_list_m[count] = m;              // k-point indx of Xorb
                                count++;
                            }
                        }
                    }
                }
            }
            num_rhs = count;

            if (count > 0) {
                batch_num_rhs = pSPARC->ExxMemBatch == 0 ? num_rhs : pSPARC->ExxMemBatch * size;
                NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required                        

                rhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                            // right hand sides of Poisson's equation
                sol = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                             // the solution for each rhs
                assert(rhs != NULL && sol != NULL);

                for (loop = 0; loop < NL; loop ++) {
                    base = batch_num_rhs*loop;
                    for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                        i = rhs_list_i[count];                      // col indx of Xorb
                        j = rhs_list_j[count];                      // band indx of psi_outer
                        l = rhs_list_l[count];                      // k-point indx of psi_outer
                        m = rhs_list_m[count];                      // k-point indx of Xorb
                        ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                        ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
                        if (pSPARC->kpthf_pn[l] == 1) {
                            for (k = 0; k < DMnd; k++) 
                                rhs[k + (count-base)*DMnd] = conj(psi_outer[k + j*DMndsp + ll_red*size_k_]) * psi[k + i*DMndsp + m*size_k];
                        } else {
                            for (k = 0; k < DMnd; k++) 
                                rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMndsp + ll_red*size_k_] * psi[k + i*DMndsp + m*size_k];
                        }
                    }

                    // Solve all Poisson's equation 
                    poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, m + pSPARC->kpt_start_indx, l, comm);

                    for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                        i = rhs_list_i[count];
                        j = rhs_list_j[count];
                        l = rhs_list_l[count];
                        m = rhs_list_m[count];
                        ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                        
                        occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                        occ_j = occ_outer[j + ll * Ns];

                        // TODO: use a temp array to reduce the MPI_Allreduce time to 1
                        double _Complex temp = 0.0;
                        for (k = 0; k < DMnd; k++){
                            temp += (conj(rhs[k + (count-base)*DMnd]) * sol[k + (count-base)*DMnd]);
                        }
                        if (size > 1)
                            MPI_Allreduce(MPI_IN_PLACE, &temp, 1,  MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
                        pSPARC->Eexx += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * creal(temp);
                    }
                }

                free(rhs);
                free(sol);
            }
        }
        free(rhs_list_i);
        free(rhs_list_j);
        free(rhs_list_l);
        free(rhs_list_m);
        pSPARC->Eexx /= pSPARC->dV;

    } else {
        int size_k = DMndsp * Nband;        
        int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;
        int Nstates_occ = pSPARC->Nstates_occ;
        int size_k_xi = DMndsp * Nstates_occ;
        int Nband_bandcomm_M = pSPARC->Nband_bandcomm_M;
        double _Complex alpha = 1.0, beta = 0.0;
        
        double _Complex *Xi_times_psi = (double _Complex *) malloc(Nband_bandcomm_M * Nstates_occ * sizeof(double _Complex));
        assert(Xi_times_psi != NULL);

        for (int kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k++) {
            if (Nband_bandcomm_M == 0) continue;
            psi = pSPARC->Xorb_kpt + kpt_k*size_k;
            xi = pSPARC->Xi_kpt + kpt_k*size_k_xi;

            for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
                occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * occ_outer_shift) : pSPARC->occ_outer;

                #ifdef ACCELGT
                if (pSPARC->useACCEL == 1) {
                    ACCEL_ZGEMM( CblasColMajor, CblasConjTrans, CblasNoTrans, Nband_bandcomm_M, Nstates_occ, DMnd,
                                &alpha, psi + spinor*DMnd, DMndsp, xi + spinor*DMnd, DMndsp, &beta, Xi_times_psi, Nband_bandcomm_M);
                } else
                #endif // ACCELGT
                {
                    // perform matrix multiplication psi' * X using ScaLAPACK routines
                    cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Nband_bandcomm_M, Nstates_occ, DMnd,
                                &alpha, psi + spinor*DMnd, DMndsp, xi + spinor*DMnd, DMndsp, &beta, Xi_times_psi, Nband_bandcomm_M);
                }

                if (size > 1) {
                    // sum over all processors in dmcomm
                    MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nband_bandcomm_M*Nstates_occ, 
                                MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
                }

                for (i = 0; i < Nband_bandcomm_M; i++) {
                    double temp = 0.0;
                    for (j = 0; j < Nstates_occ; j++) {
                        temp += creal(conj(Xi_times_psi[i+j*Nband_bandcomm_M]) * Xi_times_psi[i+j*Nband_bandcomm_M]);
                    }
                    temp *= occ_outer[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
                    pSPARC->Eexx += pSPARC->kptWts_loc[kpt_k] * temp / pSPARC->Nkpts ;
                }
            }
        }
        free(Xi_times_psi);
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    pSPARC->Eexx /= (pSPARC->Nspin + 0.0);
    pSPARC->Eexx *= -pSPARC->exx_frac;
}


/**
 * @brief   Solving Poisson's equation using FFT 
 *          
 *          This function only works for solving Poisson's equation with complex right hand side
 *          Note: only FFT method is supported in current version. 
 *          TODO: add method in real space. 
 */
void poissonSolve_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_const, int ncol, int DMnd, 
                double _Complex *sol, int kpt_k, int kpt_q, MPI_Comm comm) 
{
    int size, rank;
    int *sendcounts, *sdispls, *recvcounts, *rdispls;
    sendcounts = sdispls = recvcounts = rdispls = NULL;
    double _Complex *rhs_loc, *sol_loc, *rhs_loc_order, *sol_loc_order;
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
    int kq_shift = 0;                                                           // start index for k&q for each process
    /********************************************************************/

#ifdef DEBUG
	double t1, t2;
#endif  

    if (size > 1){
        // variables for RHS storage
        rhs_loc = (double _Complex *) malloc(sizeof(double _Complex) * ncolp * Nd);
        rhs_loc_order = (double _Complex *) malloc(sizeof(double _Complex) * Nd * ncolp);
        DMVertices = (int (*)[6]) malloc(sizeof(int[6])*size);

        // variables for alltoallv
        sendcounts = (int*) malloc(sizeof(int)*size);
        sdispls = (int*) malloc(sizeof(int)*size);
        recvcounts = (int*) malloc(sizeof(int)*size);
        rdispls = (int*) malloc(sizeof(int)*size);
        assert(rhs_loc != NULL && rhs_loc_order != NULL && 
               sendcounts != NULL && sdispls != NULL && recvcounts != NULL && rdispls != NULL);
        
        parallel_info_dp2bp(gridsizes, DMnd, ncol, DMVertices, sendcounts, sdispls, recvcounts, rdispls, &kq_shift, 1, comm);
        /********************************************************************/

#ifdef DEBUG
	t1 = MPI_Wtime();
#endif  
        MPI_Alltoallv(rhs, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, 
                        rhs_loc, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, comm);
#ifdef DEBUG
	t2 = MPI_Wtime();
    pSPARC->Exxtime_comm += (t2-t1);
#endif  
        // rhs_full needs rearrangement
        block_dp_to_cart((void *) rhs_loc, ncolp, DMVertices, rdispls, size, 
                        Nx, Ny, Nd, (void *) rhs_loc_order, sizeof(double _Complex));

        free(rhs_loc);
        // variable for local result sol
        sol_loc = (double _Complex *) malloc(sizeof(double _Complex) * Nd * ncolp);
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
        pois_fft_kpt(pSPARC, rhs_loc_order, pois_const, ncolp, sol_loc, kpt_k, kpt_q);
    } else {
        // solve by kron
        pois_kron_kpt(pSPARC, rhs_loc_order, pois_const, ncolp, sol_loc, kpt_k, kpt_q);
    }

#ifdef DEBUG
	t2 = MPI_Wtime();
    pSPARC->Exxtime_solver += (t2-t1);
#endif  

    if (size > 1)
        free(rhs_loc_order);

    if (size > 1) {
        sol_loc_order = (double _Complex *) malloc(sizeof(double _Complex)* Nd * ncolp);
        assert(sol_loc_order != NULL);

        // sol_loc needs rearrangement
        cart_to_block_dp((void *) sol_loc, ncolp, DMVertices, size, 
                        Nx, Ny, Nd, (void *) sol_loc_order, sizeof(double _Complex));

#ifdef DEBUG
	t1 = MPI_Wtime();
#endif  

        MPI_Alltoallv(sol_loc_order, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, 
                    sol, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, comm);

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



/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space - in k-point case. 
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * @param kpt_k             global index of k Bloch wave vector
 * @param kpt_q             global index of q Bloch wave vector
 * Note:                    Assuming the RHS is periodic with Bloch wave vector (k - q). 
 * Note:                    This function is complete localized. 
 */
void pois_fft_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_const, 
                    int ncol, double _Complex *sol, int kpt_k, int kpt_q) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]
    if (ncol == 0) return;

    int Nd = pSPARC->Nd;
    int Nx = pSPARC->Nx, Ny = pSPARC->Ny, Nz = pSPARC->Nz; 
    double _Complex *rhs_bar = (double _Complex*) malloc(sizeof(double _Complex) * Nd * ncol);
    assert(rhs_bar != NULL);
    int *Kptshift_map = pSPARC->Kptshift_map;
    double *alpha;

    // don't want to change rhs
    double _Complex *rhs_  = (double _Complex *) malloc(sizeof(double _Complex) * Nd * ncol);    

#if defined(USE_MKL)
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*Nx, Nx, 1};
#elif defined(USE_FFTW)
    int dim_sizes[3] = {Nz, Ny, Nx};
#endif
    /********************************************************************/
    
    memcpy(rhs_, rhs, sizeof(double _Complex) * Nd * ncol);
    apply_phase_factor(pSPARC, rhs_, ncol, "N", kpt_k, kpt_q);

    // FFT
#if defined(USE_MKL)
    MKL_MDFFT_batch(rhs_, ncol, dim_sizes, Nd, rhs_bar, strides_out, Nd);
#elif defined(USE_FFTW)
    FFTW_MDFFT_batch(dim_sizes, ncol, rhs_, Nd, rhs_bar, Nd);
#endif

    int cnt = 0;
    for (int n = 0; n < ncol; n++) {
        int l = Kptshift_map(kpt_k, kpt_q);
        if (!l) {
            alpha = pois_const + Nd * (pSPARC->Nkpts_shift - 1);
        } else {
            alpha = pois_const + Nd * (l - 1);
        }
        // multiplied by alpha
        for (int i = 0; i < Nd; i++) {
            rhs_bar[cnt] = creal(rhs_bar[cnt]) * alpha[i] 
                                + (cimag(rhs_bar[cnt]) * alpha[i]) * I;
            cnt++;
        }
    }

    // iFFT
#if defined(USE_MKL)    
    MKL_MDiFFT_batch(rhs_bar, ncol, dim_sizes, Nd, sol, strides_out, Nd);
#elif defined(USE_FFTW)
    FFTW_MDiFFT_batch(dim_sizes, ncol, rhs_bar, Nd, sol, Nd);
#endif

    apply_phase_factor(pSPARC, sol, ncol, "P", kpt_k, kpt_q);

    free(rhs_bar);
    free(rhs_);
#undef Kptshift_map
}


/**
 * @brief   Solve Poisson's equation using Kronecker product Lapacian
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * @param kpt_k             global index of k Bloch wave vector
 * @param kpt_q             global index of q Bloch wave vector
 * Note:                    Assuming the RHS is periodic with Bloch wave vector (k - q). 
 * Note:                    This function is complete localized. 
 */
void pois_kron_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_const, 
                    int ncol, double _Complex *sol, int kpt_k, int kpt_q) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]
    if (ncol == 0) return;

    int Nd = pSPARC->Nd;
    int Nx = pSPARC->Nx, Ny = pSPARC->Ny, Nz = pSPARC->Nz;     
    int *Kptshift_map = pSPARC->Kptshift_map;
    /********************************************************************/

    for (int n = 0; n < ncol; n++) {        
        int l = Kptshift_map(kpt_k, kpt_q);
        KRON_LAP* kron_lap; double *alpha;
        if (!l) {
            alpha = pois_const + Nd * (pSPARC->Nkpts_shift - 1);
            kron_lap = pSPARC->kron_lap_exx[pSPARC->Nkpts_shift - 1];
        } else {
            alpha = pois_const + Nd * (l - 1);
            kron_lap = pSPARC->kron_lap_exx[l-1];
        }
        LAP_KRON_COMPLEX(Nx, Ny, Nz, kron_lap->Vx_kpt, kron_lap->Vy_kpt, kron_lap->Vz_kpt,
                 kron_lap->VyH_kpt, kron_lap->VzH_kpt, rhs + n*Nd, alpha, sol + n*Nd);
    }    
#undef Kptshift_map
}



/**
 * @brief   Gather psi_outer_kpt and occupations in other bandcomms and kptcomms
 * 
 *          In need of communication between k-point topology and dmcomm, additional 
 *          communication for occupations is required. 
 */
void gather_psi_occ_outer_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_outer_kpt, double *occ_outer) {
    int i, k, grank, blacs_rank, blacs_size, kpt_bridge_rank, kpt_bridge_size;
    int Ns, DMnd, Nband, Nkpts_hf_kptcomm, spn_i;
    int sendcount, *recvcounts, *displs;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &blacs_rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nkpts_hf_kptcomm = pSPARC->Nkpts_hf_kptcomm;

    int size_k = DMndsp * Nband;
    int size_k_ = DMndsp * Ns;    
    int shift_k_ = pSPARC->kpthf_start_indx * size_k_;
    int shift_s = pSPARC->band_start_indx * DMndsp;
    int shift_k_occ = pSPARC->kpt_start_indx * Ns;                  // gather all occ for all kpts
    
    int shift_spn_occ_outer = Ns * pSPARC->Nkpts_sym;
    int shift_spn_occ = Ns * pSPARC->Nkpts_kptcomm;

    // local arrangement of psi
    if (pSPARC->ExxAcc == 0) {
        int count = 0;
        for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
            if (!pSPARC->kpthf_flag_kptcomm[k]) continue;
            copy_mat_blk(sizeof(double _Complex), pSPARC->Xorb_kpt + k*size_k, DMndsp, 
                DMndsp, Nband, pSPARC->psi_outer_kpt + shift_s + count * size_k_ + shift_k_, DMndsp);
            count ++;
        }
    }

    // local arrangement of occ
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        for (i = 0; i < Ns * pSPARC->Nkpts_kptcomm; i++) 
            pSPARC->occ_outer[i + shift_k_occ + spn_i * shift_spn_occ_outer] = pSPARC->occ[i + spn_i * shift_spn_occ];
    }

    /********************************************************************/

    if (pSPARC->ExxAcc == 0) {
        if (blacs_size > 1) {
            // First step, gather all required bands across blacscomm within each kptcomm
            for (i = 0 ; i < Nkpts_hf_kptcomm; i ++) 
                gather_blacscomm_kpt(pSPARC, DMndsp, Ns, psi_outer_kpt + i * size_k_ + shift_k_);
        }

        if (kpt_bridge_size > 1) {
            // Second step, gather all required kpts across kptcomm
            gather_kptbridgecomm_kpt(pSPARC, DMndsp, Ns, psi_outer_kpt);
        }
    }

    /********************************************************************/
    // Gather occupations if needed
    if (kpt_bridge_size > 1) {
        int Nkpts_by_npkpt = pSPARC->Nkpts_sym / pSPARC->npkpt;
        int Nkpts_mod_npkpt = pSPARC->Nkpts_sym % pSPARC->npkpt;
        recvcounts = (int*) malloc(sizeof(int)* kpt_bridge_size);
        displs = (int*) malloc(sizeof(int)* kpt_bridge_size);
        assert(recvcounts !=NULL && displs != NULL);
        displs[0] = 0;
        for (i = 0; i < kpt_bridge_size; i++){
            recvcounts[i] = (Nkpts_by_npkpt + (int) (i < Nkpts_mod_npkpt)) * Ns;
            if (i != (kpt_bridge_size-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }
        sendcount = 1;

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
            MPI_Allgatherv(MPI_IN_PLACE, sendcount, MPI_DOUBLE, occ_outer + spn_i * shift_spn_occ_outer, 
                recvcounts, displs, MPI_DOUBLE, pSPARC->kpt_bridge_comm);   

        free(recvcounts);
        free(displs); 
    }

    /********************************************************************/
    if (pSPARC->flag_kpttopo_dm && pSPARC->ExxAcc == 0) {
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
 * @brief   Apply phase factor by Bloch wave vector shifts. 
 * 
 * @param vec           vectors to be applied phase factors
 * @param ncol          number of columns of vectors
 * @param NorP          "N" is for negative phase factor, "P" is for positive.
 * @param kpt_k         global k-point index for k
 * @param kpt_q         global k-point index for q
 * Note:                Assuming the shifts is introduced by wave vector (k - q)
 */
void apply_phase_factor(SPARC_OBJ *pSPARC, double _Complex *vec, int ncol, char *NorP, int kpt_k, int kpt_q) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]

    int i, l, col, Nd, rank;
    int *Kptshift_map;
    double _Complex *phase;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Nd = pSPARC->Nd;

    if (*NorP == 'N') {
        phase = pSPARC->neg_phase;
    } else if(*NorP == 'P') {
        phase = pSPARC->pos_phase;
    } else {
        if (!rank) printf("ERROR: Please only use \"N\" or \"P\" for apply_phase_factor function\n");
        exit(EXIT_FAILURE);
    }
    Kptshift_map = pSPARC->Kptshift_map;

    for (col = 0; col < ncol; col ++) {
        l = Kptshift_map(kpt_k, kpt_q);
        if (!l) continue;                           // nothing for 0 shift
        for (i = 0; i < Nd; i++) 
            vec[i + col * Nd] *= phase[i + (l-1) * Nd];
    }

#undef Kptshift_map
}

/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE_kpt(SPARC_OBJ *pSPARC) {
    int rank, i, j, spn_i;    

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int Ns = pSPARC->Nstates;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int occ_shift = pSPARC->Nstates * pSPARC->Nkpts_kptcomm;

    int Nstates_occ = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        double *occ = pSPARC->occ + spn_i*occ_shift;

        // Find maximal number of occupied states among all k-points
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            int Nstates_occ_temp = 0;
            for (j = 0; j < Ns; j++) 
                if (occ[j + kpt*Ns] > 1e-6) Nstates_occ_temp ++;
            Nstates_occ = max(Nstates_occ, Nstates_occ_temp);
        }

        // find number of occupied states among all HF k-points
        int count = 0;
        for (int kpt = 0; kpt < pSPARC->Nkpts_hf; kpt++) {
            int ll = pSPARC->kpthf_ind[kpt];                  // ll w.r.t. Nkpts_sym, for occ
            occ = pSPARC->occ_outer + ll * Ns + spn_i * pSPARC->Nstates * pSPARC->Nkpts_sym;
            for (i = 0; i < Ns; i++) {
                if (occ[i] > 1e-6) count++;
            }
        }
        pSPARC->Nstates_occ_list[spn_i] = count;
    }
    // Apply ExxAceVal_state here
    Nstates_occ += pSPARC->EeeAceValState;
    Nstates_occ = min(Nstates_occ, pSPARC->Nstates);                      // Ensure Nstates_occ is less or equal to Nstates        

    // Note: occupations are only correct in dmcomm. 
    MPI_Allreduce(MPI_IN_PLACE, &Nstates_occ, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return;

    // If number of occupied states changed, need to reallocate memory space
    if (Nstates_occ != pSPARC->Nstates_occ) { 
        if (pSPARC->Nstates_occ > 0) {
        #ifdef DEBUG
        if(!rank) 
            printf("\nTotal number of occupied states + Extra states changed : %d\n", Nstates_occ);
        #endif  
            free(pSPARC->Xi_kpt);
            free(pSPARC->Xi_kptcomm_topo_kpt);
            pSPARC->Xi_kpt = NULL;
            pSPARC->Xi_kptcomm_topo_kpt = NULL;
        } else {
        #ifdef DEBUG
        if(!rank) 
            printf("\nStart to use %d states to create ACE operator for each k-point and spin.\n", Nstates_occ);
        #endif  
        }

        // Xi, ACE operator
        pSPARC->Xi_kpt = (double _Complex *) malloc(DMndsp * Nstates_occ * pSPARC->Nkpts_kptcomm * sizeof(double _Complex));
        // Storage of ACE operator in kptcomm_topo
        pSPARC->Xi_kptcomm_topo_kpt = 
                (double _Complex *)malloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_spincomm * Nstates_occ * pSPARC->Nkpts_kptcomm * sizeof(double _Complex));
        assert(pSPARC->Xi_kpt != NULL && pSPARC->Xi_kptcomm_topo_kpt != NULL);
        pSPARC->Nstates_occ = Nstates_occ;
    }
    
    if (Nstates_occ < pSPARC->band_start_indx)
        pSPARC->Nband_bandcomm_M = 0;
    else {
        pSPARC->Nband_bandcomm_M = min(pSPARC->Nband_bandcomm, Nstates_occ - pSPARC->band_start_indx);
    }
}


/**
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm_kpt(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double _Complex *vec) 
{
    if (pSPARC->blacscomm == MPI_COMM_NULL) return;

    int i, grank, blacs_rank, blacs_size;
    int sendcount, *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &blacs_rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    
    if (blacs_size > 1) {
        recvcounts = (int*) malloc(sizeof(int)* blacs_size);
        displs = (int*) malloc(sizeof(int)* blacs_size);
        assert(recvcounts !=NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < blacs_size; i++){
            recvcounts[i] = (i < (Ncol / NB) ? NB : (i == (Ncol / NB) ? (Ncol % NB) : 0)) * Nrow;
            if (i != (blacs_size-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }
        sendcount = 1;
        MPI_Allgatherv(MPI_IN_PLACE, sendcount, MPI_DOUBLE_COMPLEX, vec, 
            recvcounts, displs, MPI_DOUBLE_COMPLEX, pSPARC->blacscomm);

        free(recvcounts);
        free(displs); 
    }
}

/**
 * @brief   Gather orbitals shape vectors across kpt_bridge_comm
 */
void gather_kptbridgecomm_kpt(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double _Complex *vec)
{
    if (pSPARC->kpt_bridge_comm == MPI_COMM_NULL) return;

    int kpt_bridge_rank, kpt_bridge_size;
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    int i, size_k, sendcount, *recvcounts, *displs;
    size_k = Nrow * Ncol;
    
    recvcounts = (int*) malloc(sizeof(int)* kpt_bridge_size);
    displs = (int*) malloc(sizeof(int)* kpt_bridge_size);
    assert(recvcounts !=NULL && displs != NULL);
    displs[0] = 0;
    for (i = 0; i < kpt_bridge_size; i++){
        recvcounts[i] = pSPARC->Nkpts_hf_list[i] * size_k;
        if (i != (kpt_bridge_size-1))
            displs[i+1] = displs[i] + recvcounts[i];
    }
    sendcount = 1;

    MPI_Allgatherv(MPI_IN_PLACE, sendcount, MPI_DOUBLE_COMPLEX, vec, 
        recvcounts, displs, MPI_DOUBLE_COMPLEX, pSPARC->kpt_bridge_comm);   

    free(recvcounts);
    free(displs); 
}


/**
 * @brief   transfer orbitals in a cyclic rotation way to save memory
 */
void transfer_orbitals_kptbridgecomm(SPARC_OBJ *pSPARC, 
        void *sendbuff, void *recvbuff, int shift, MPI_Request *reqs, int unit_size)
{
    assert(unit_size == sizeof(double) || unit_size == sizeof(double _Complex));
    MPI_Comm kpt_bridge_comm = pSPARC->kpt_bridge_comm;
    if (kpt_bridge_comm == MPI_COMM_NULL) return;
    int size, rank;
    MPI_Comm_size(kpt_bridge_comm, &size);
    MPI_Comm_rank(kpt_bridge_comm, &rank);

    int DMnd = pSPARC->Nd_d_dmcomm;
    int Nband = pSPARC->Nband_bandcomm;
    int size_k = DMnd * Nband;

    int srank = (rank-shift+size)%size;
    int rrank = (rank-shift-1+size)%size;
    int Nkpt_hf_send = pSPARC->Nkpts_hf_list[srank];
    int Nkpt_hf_recv = pSPARC->Nkpts_hf_list[rrank];
    int rneighbor = (rank+1)%size;
    int lneighbor = (rank-1+size)%size;
    
    if (unit_size == sizeof(double)) {
        MPI_Irecv(recvbuff, size_k*Nkpt_hf_recv, MPI_DOUBLE, lneighbor, 111, kpt_bridge_comm, &reqs[1]);
        MPI_Isend(sendbuff, size_k*Nkpt_hf_send, MPI_DOUBLE, rneighbor, 111, kpt_bridge_comm, &reqs[0]);
    } else {
        MPI_Irecv(recvbuff, size_k*Nkpt_hf_recv, MPI_DOUBLE_COMPLEX, lneighbor, 111, kpt_bridge_comm, &reqs[1]);
        MPI_Isend(sendbuff, size_k*Nkpt_hf_send, MPI_DOUBLE_COMPLEX, rneighbor, 111, kpt_bridge_comm, &reqs[0]);
    }

}

/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi_kpt(SPARC_OBJ *pSPARC, 
    int ncol, double _Complex *psi, int ldp, double _Complex *psi_storage, int ldps, double *occ, double _Complex *Xi_kpt, int ldxi, 
    int kpt_k, int kpt_q, int shift)
{
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    if (ncol == 0) return;
    int size, rank, grank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);

    int i, j, k;
    int num_rhs, count, loop, batch_num_rhs, NL, base;
    int *rhs_list_i, *rhs_list_j;
    double occ_j;
    double _Complex *rhs, *sol, *Xi_kpt_ki;
    /******************************************************************************/

    int DMnd = pSPARC->Nd_d_dmcomm;    
    int Ns = pSPARC->Nstates;    
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;    
    int size_k = ldp * pSPARC->Nband_bandcomm;
    int size_k_xi = ldxi * pSPARC->Nstates_occ;

    int source = (rank-shift+size)%size;
    int NB = (Ns - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    int Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
    int band_start_indx_source = source * NB;

    rhs_list_i = (int*) malloc(Nband_source * ncol * Nkpts_kptcomm * sizeof(int)); 
    rhs_list_j = (int*) malloc(Nband_source * ncol * Nkpts_kptcomm * sizeof(int));     
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    count = 0;    
    for (i = 0; i < ncol; i++) {
        for (j = 0; j < Nband_source; j++) {
            occ_j = occ[j+band_start_indx_source];
            if (occ_j < 1e-6) continue;
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

    /* ExxMemBatch could be any positive integer to define the maximum number of 
    Poisson's equations to be solved at one time. The smaller the ExxMemBatch, 
    the less memory are required, but also the longer the running time. This part
    of code could be directly applied to NON-ACE part. */
    
    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                        num_rhs : pSPARC->ExxMemBatch * pSPARC->npNd;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required

    rhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    sol = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                          // the solution for each rhs
    assert(rhs != NULL && sol != NULL);

#ifdef DEBUG
    double t1, t2, tt1, tt2;
    t1 = MPI_Wtime();
#endif
    
    /*************** Solve all Poisson's equation and apply to psi ****************/
    for (loop = 0; loop < NL; loop ++) {
        
    #ifdef DEBUG
        tt1 = MPI_Wtime();
    #endif

        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];              // band indx of psi_outer with k vector
            j = rhs_list_j[count];              // band indx of psi_outer with q vector                        
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_storage[k + j*ldps]) * psi[k + i*ldp + kpt_k*size_k];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*ldps] * psi[k + i*ldp + kpt_k*size_k];
            }
        }
    
    #ifdef DEBUG
        tt2 = MPI_Wtime();
        pSPARC->Exxtime_rhs += (tt2-tt1);
    #endif

        poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_const, count-base, DMnd, sol, kpt_k + pSPARC->kpt_start_indx, kpt_q, pSPARC->dmcomm);    
    
    #ifdef DEBUG
        tt1 = MPI_Wtime();
    #endif
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];              // band indx of psi_outer with k vector
            j = rhs_list_j[count];              // band indx of psi_outer with q vector

            occ_j = occ[j+band_start_indx_source];
            Xi_kpt_ki = Xi_kpt + pSPARC->band_start_indx * ldxi + kpt_k*size_k_xi;
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    Xi_kpt_ki[k + i*ldxi] -= pSPARC->kptWts_hf * occ_j * psi_storage[k + j*ldps] * sol[k + (count-base)*DMnd] / pSPARC->dV;
            } else {
                for (k = 0; k < DMnd; k++) 
                    Xi_kpt_ki[k + i*ldxi] -= pSPARC->kptWts_hf * occ_j * conj(psi_storage[k + j*ldps]) * sol[k + (count-base)*DMnd] / pSPARC->dV;
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
    if(!grank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",grank,(t2-t1)*1e3); 
#endif
}
