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

#define TEMP_TOL (1e-14)


/**
 * @brief   Create ACE operator in dmcomm for each k-point
 *
 *          Using occupied + extra orbitals to construct the ACE operator for each k-point.
 *          Note: there is no band parallelization when usign ACE, so there is only 1 dmcomm
 *          for each k-point. 
 */
void ACE_operator_kpt(SPARC_OBJ *pSPARC, 
    double _Complex *psi_outer, double _Complex *psi, double *occ_outer, int spn_i, double _Complex *Xi_kpt)
{
    int i, j, k, l, ll, ll_red, kpt_k, kpt_q, rank, nproc_dmcomm, Ns, Ns_occ, dims[3], DMnd, ONE = 1;
    int num_rhs, count, loop, batch_num_rhs, NL, base, Nkpts_hf, DMndNs, DMndNsocc;
    int *rhs_list_i, *rhs_list_j, *rhs_list_l, *kpt_k_list, *kpt_q_list;
    double occ, occ_i, occ_j, *M_real, t1, t2;
    double _Complex *rhs, *Vi, *M;
    /******************************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->dmcomm == MPI_COMM_NULL) return;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
    Nkpts_hf = pSPARC->Nkpts_hf;
    DMndNs = DMnd * Ns;
    Ns_occ = pSPARC->Nstates_occ[spn_i];    
    /******************************************************************************/

    rhs_list_i = (int*) calloc(Ns_occ * Ns * Nkpts_hf, sizeof(int)); 
    rhs_list_j = (int*) calloc(Ns_occ * Ns * Nkpts_hf, sizeof(int)); 
    rhs_list_l = (int*) calloc(Ns_occ * Ns * Nkpts_hf, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_l != NULL);

    DMndNsocc = DMnd * Ns_occ;
    for (i = 0; i < DMnd * Ns_occ * pSPARC->Nkpts_kptcomm; i++) 
        Xi_kpt[i] = 0.0;

    // i -- kpt_k, j -- kpt_q
    for (kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k ++) {
        count = 0;
        for (kpt_q = 0; kpt_q < Nkpts_hf; kpt_q ++) {
            ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
            // construct ACE operator using all orbitals of Xorb_kpt
            for (i = 0; i < Ns_occ; i++) {
                for (j = 0; j < pSPARC->Nstates; j++) {
                    if (occ_outer[j + ll * Ns] > 1e-6) {
                        rhs_list_i[count] = i;              // band indx of psi_outer with k vector
                        rhs_list_j[count] = j;              // band indx of psi_outer with q vector
                        rhs_list_l[count] = kpt_q;          // k-point indx of q vector
                        count ++;
                    }
                }
            }
        }
        
        num_rhs = count;
        if (num_rhs == 0) continue;

        /* EXXMem_batch could be any positive integer to define the maximum number of 
        Poisson's equations to be solved at one time. The smaller the EXXMem_batch, 
        the less memory are required, but also the longer the running time. This part
        of code could be directly applied to NON-ACE part. */
        
        batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                            num_rhs : pSPARC->EXXMem_batch * nproc_dmcomm;
        NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required

        rhs = (double _Complex *)calloc(sizeof(double _Complex) , DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
        Vi = (double _Complex *)calloc(sizeof(double _Complex) , DMnd * batch_num_rhs);                          // the solution for each rhs
        kpt_k_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of k vector 
        kpt_q_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of q vector 
        assert(rhs != NULL && Vi != NULL && kpt_k_list != NULL && kpt_q_list != NULL);

        for (i = 0; i < batch_num_rhs; i ++) kpt_k_list[i] = kpt_k + pSPARC->kpt_start_indx;
        /*************** Solve all Poisson's equation and apply to psi ****************/
        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];              // band indx of psi_outer with k vector
                j = rhs_list_j[count];              // band indx of psi_outer with q vector
                l = rhs_list_l[count];              // k-point indx of q vector
                ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
                kpt_q_list[count-base] = l;
                if (pSPARC->kpthf_pn[l] == 1) {
                    for (k = 0; k < DMnd; k++) 
                        rhs[k + (count-base)*DMnd] = conj(psi_outer[k + j*DMnd + ll_red*DMndNs]) * psi[k + i*DMnd + kpt_k*DMndNs];
                } else {
                    for (k = 0; k < DMnd; k++) 
                        rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd + ll_red*DMndNs] * psi[k + i*DMnd + kpt_k*DMndNs];
                }
            }
            
            poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, kpt_k_list, kpt_q_list, pSPARC->dmcomm);    
            
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];              // band indx of psi_outer with k vector
                j = rhs_list_j[count];              // band indx of psi_outer with q vector
                l = rhs_list_l[count];              // k-point indx of q vector
                ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi

                occ_j = occ_outer[j + ll * Ns];
                if (pSPARC->kpthf_pn[l] == 1) {
                    for (k = 0; k < DMnd; k++) 
                        Xi_kpt[k + i*DMnd + kpt_k*DMndNsocc] -= pSPARC->kptWts_hf * occ_j * psi_outer[k + j*DMnd + ll_red*DMndNs] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                } else {
                    for (k = 0; k < DMnd; k++) 
                        Xi_kpt[k + i*DMnd + kpt_k*DMndNsocc] -= pSPARC->kptWts_hf * occ_j * conj(psi_outer[k + j*DMnd + ll_red*DMndNs]) * Vi[k + (count-base)*DMnd] / pSPARC->dV;
                }
            }
        }

        free(rhs);
        free(Vi);
        free(kpt_k_list);
        free(kpt_q_list);

        
        /******************************************************************************/

        M = (double _Complex *)calloc(Ns_occ * Ns_occ, sizeof(double _Complex));
        assert(M != NULL);
        
        double _Complex alpha = 1.0;
        double _Complex beta = 0.0;
        t1 = MPI_Wtime();
        // perform matrix multiplication psi' * W using ScaLAPACK routines
        cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Ns_occ, Ns_occ, DMnd,
                    &alpha, psi + kpt_k*DMndNs, DMnd, Xi_kpt + kpt_k*DMndNsocc, DMnd, &beta, M, Ns_occ);

        if (nproc_dmcomm > 1) {
            // sum over all processors in dmcomm
            MPI_Allreduce(MPI_IN_PLACE, M, Ns_occ*Ns_occ,
                        MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
        }
    
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank && !spn_i) printf("rank = %2d, finding M = psi'* W took %.3f ms\n",rank,(t2-t1)*1e3); 
        #endif

        // perform Cholesky Factorization on -M
        // M = chol(-M), upper triangular matrix
        // M = -0.5*(M + M') seems to lead to numerical issue. 
        // TODO: Check if it's required to "symmetrize" M. 
        for (i = 0; i < Ns_occ * Ns_occ; i++)   M[i] = -M[i];

        t1 = MPI_Wtime();
        int info = 0;
        info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', Ns_occ, M, Ns_occ);
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && !spn_i) 
            printf("==Cholesky Factorization: "
                "info = %d, computing Cholesky Factorization using LAPACKE_zpotrf: %.3f ms\n", 
                info, (t2 - t1)*1e3);
        #endif

        // Xi = WM^(-1)
        t1 = MPI_Wtime();
        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, 
                    CblasNonUnit, DMnd, Ns_occ, &alpha, M, Ns_occ, Xi_kpt + kpt_k*DMndNsocc, DMnd);

        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && !spn_i) 
            printf("==Triangular matrix equation: "
                "Solving triangular matrix equation using cblas_ztrsm: %.3f ms\n", (t2 - t1)*1e3);
        #endif

        free(M);
    }

    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_l);
}


/**
 * @brief   Evaluating Exact Exchange potential in k-point case
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ncol, int DMnd, double _Complex *Hx, int spin, int kpt, MPI_Comm comm) 
{        
    int i, j, k, rank, Lanczos_flag, dims[3];
    double _Complex *psi_outer, *Xi;
    double t1, t2, *occ_outer;
    
    MPI_Comm_rank(comm, &rank);
    Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
    /********************************************************************/

    int xi_shift = pSPARC->Nstates_occ[0] * DMnd * pSPARC->Nkpts_kptcomm;
    int psi_outer_shift = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
    int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

    t1 = MPI_Wtime();
    if (pSPARC->ACEFlag == 0) {
        if (Lanczos_flag == 0) {
            dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
        } else {
            dims[0] = pSPARC->npNdx_kptcomm; dims[1] = pSPARC->npNdy_kptcomm; dims[2] = pSPARC->npNdz_kptcomm;
        }
        occ_outer = pSPARC->occ_outer + spin * occ_outer_shift;
        psi_outer = (Lanczos_flag == 0) ? pSPARC->psi_outer_kpt + spin * psi_outer_shift : pSPARC->psi_outer_kptcomm_topo_kpt + spin * psi_outer_shift ;
        evaluate_exact_exchange_potential_kpt(pSPARC, X, ncol, DMnd, dims, occ_outer, psi_outer, Hx, kpt, comm);

    } else {
        Xi = (Lanczos_flag == 0) ? pSPARC->Xi_kpt + spin * xi_shift : pSPARC->Xi_kptcomm_topo_kpt + spin * xi_shift;
        evaluate_exact_exchange_potential_ACE_kpt(pSPARC, X, ncol, DMnd, Xi, Hx, spin, kpt, comm);
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
 * @param kpt_k           Local index of each k-point 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, double _Complex *X, 
        int ncol, int DMnd, int *dims, double *occ_outer, 
        double _Complex *psi_outer, double _Complex *Hx, int kpt_k, MPI_Comm comm) 
{
    int i, j, k, l, ll, ll_red, rank, Ns, num_rhs;
    int size, batch_num_rhs, NL, base, loop, Nkpts_hf, Nkpts_hf_red;
    int *rhs_list_i, *rhs_list_j, *rhs_list_l, *kpt_k_list, *kpt_q_list;
    double occ, hyb_mixing, occ_alpha;
    double _Complex *rhs, *Vi;

    Ns = pSPARC->Nstates;
    Nkpts_hf = pSPARC->Nkpts_hf;
    Nkpts_hf_red = pSPARC->Nkpts_hf_red;
    hyb_mixing = pSPARC->hyb_mixing;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);

    int DMndNs = DMnd * Ns;
    /********************************************************************/
    rhs_list_i = (int*) calloc(ncol * Ns * Nkpts_hf, sizeof(int)); 
    rhs_list_j = (int*) calloc(ncol * Ns * Nkpts_hf, sizeof(int)); 
    rhs_list_l = (int*) calloc(ncol * Ns * Nkpts_hf, sizeof(int)); 
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

    batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                        num_rhs : pSPARC->EXXMem_batch * size;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                                 // number of loops required                        

    rhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                        // right hand sides of Poisson's equation
    Vi = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                         // the solution for each rhs
    kpt_k_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of k vector 
    kpt_q_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of q vector 
    assert(rhs != NULL && Vi != NULL && kpt_k_list != NULL && kpt_q_list != NULL);

    for (i = 0; i < batch_num_rhs; i ++) kpt_k_list[i] = kpt_k + pSPARC->kpt_start_indx;
    /*************** Solve all Poisson's equation and apply to X ****************/    
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            l = rhs_list_l[count];
            ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
            ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
            kpt_q_list[count-base] = l;
            if (pSPARC->kpthf_pn[l] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_outer[k + j*DMnd + ll_red*DMndNs]) * X[k + i*DMnd];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd + ll_red*DMndNs] * X[k + i*DMnd];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, kpt_k_list, kpt_q_list, comm);

        // Apply exact exchange potential to vector X
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            l = rhs_list_l[count];
            ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
            ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
            occ = occ_outer[j + ll * Ns];
            occ_alpha = occ * hyb_mixing;
            if (pSPARC->kpthf_pn[l] == 1) {
                for (k = 0; k < DMnd; k++) 
                    Hx[k + i*DMnd] -= occ_alpha * pSPARC->kptWts_hf * psi_outer[k + j*DMnd + ll_red*DMndNs] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            } else {
                for (k = 0; k < DMnd; k++) 
                    Hx[k + i*DMnd] -= occ_alpha * pSPARC->kptWts_hf * conj(psi_outer[k + j*DMnd + ll_red*DMndNs]) * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            }
        }
    }

    free(rhs);
    free(Vi);
    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_l);
    free(kpt_k_list);
    free(kpt_q_list);
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
        double _Complex *X, int ncol, int DMnd, double _Complex *Xi, 
        double _Complex *Hx, int spin, int kpt, MPI_Comm comm) 
{
    int rank, size, DMndNs_occ, Ns_occ;
    Ns_occ = pSPARC->Nstates_occ[spin];
    DMndNs_occ = DMnd * Ns_occ;
    double _Complex alpha = 1.0;
    double _Complex beta = 0.0;
    double _Complex *Xi_times_psi = (double _Complex *) calloc(Ns_occ * ncol, sizeof(double _Complex));
    assert(Xi_times_psi != NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/

    // perform matrix multiplication Xi' * X using ScaLAPACK routines
    cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Ns_occ, ncol, DMnd,
                &alpha, Xi + kpt*DMndNs_occ, DMnd, X, DMnd, &beta, Xi_times_psi, Ns_occ);

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Ns_occ*ncol, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }

    alpha = -pSPARC->hyb_mixing;
    beta = 1.0;
    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Ns_occ,
                &alpha, Xi + kpt*DMndNs_occ, DMnd, Xi_times_psi, Ns_occ, &beta, Hx, DMnd);

    free(Xi_times_psi);
}


/**
 * @brief   Evaluate Exact Exchange Energy in k-point case.
 */
void evaluate_exact_exchange_energy_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, ll_red, m, grank, rank, size, spn_i;
    int Ns, Ns_loc, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, t1, t2, *occ_outer;
    double _Complex *rhs, *Vi, *psi_outer, *psi, *xi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_loc = pSPARC->Nkpts_kptcomm;
    Ns_loc = Nband * Nkpts_loc;         // for Xorb
    comm = pSPARC->dmcomm;
    pSPARC->Eexx = 0.0;

    int size_k = DMnd * Ns;
    int shift_k = pSPARC->kpthf_start_indx * Ns * DMnd;
    int shift_s = pSPARC->band_start_indx * DMnd;
    int Nkpts_hf = pSPARC->Nkpts_hf;
    int Nkpts_hf_red = pSPARC->Nkpts_hf_red;

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

        int *rhs_list_i, *rhs_list_j, *rhs_list_l, *rhs_list_m, *kpt_k_list, *kpt_q_list;
        rhs_list_i = (int*) calloc(Ns_loc * Ns * Nkpts_hf, sizeof(int)); 
        rhs_list_j = (int*) calloc(Ns_loc * Ns * Nkpts_hf, sizeof(int)); 
        rhs_list_l = (int*) calloc(Ns_loc * Ns * Nkpts_hf, sizeof(int)); 
        rhs_list_m = (int*) calloc(Ns_loc * Ns * Nkpts_hf, sizeof(int)); 
        assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_l != NULL && rhs_list_m != NULL);

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            psi_outer = pSPARC->psi_outer_kpt + spn_i * psi_outer_shift;
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;
        
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
                batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                                num_rhs : pSPARC->EXXMem_batch * size;
                NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required                        

                rhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                            // right hand sides of Poisson's equation
                Vi = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                             // the solution for each rhs
                kpt_k_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of k vector 
                kpt_q_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of q vector 
                assert(rhs != NULL && Vi != NULL && kpt_k_list != NULL && kpt_q_list != NULL);

                for (loop = 0; loop < NL; loop ++) {
                    base = batch_num_rhs*loop;
                    for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                        i = rhs_list_i[count];                      // col indx of Xorb
                        j = rhs_list_j[count];                      // band indx of psi_outer
                        l = rhs_list_l[count];                      // k-point indx of psi_outer
                        m = rhs_list_m[count];                      // k-point indx of Xorb
                        ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                        ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
                        kpt_k_list[count-base] = m + pSPARC->kpt_start_indx;
                        kpt_q_list[count-base] = l;
                        if (pSPARC->kpthf_pn[l] == 1) {
                            for (k = 0; k < DMnd; k++) 
                                rhs[k + (count-base)*DMnd] = conj(psi_outer[k + j*DMnd + ll_red*size_k]) * psi[k + i*DMnd + m*DMnd*Nband];
                        } else {
                            for (k = 0; k < DMnd; k++) 
                                rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd + ll_red*size_k] * psi[k + i*DMnd + m*DMnd*Nband];
                        }
                    }

                    // Solve all Poisson's equation 
                    poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, kpt_k_list, kpt_q_list, comm);

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
                            temp += (conj(rhs[k + (count-base)*DMnd]) * Vi[k + (count-base)*DMnd]);
                        }
                        if (size > 1)
                            MPI_Allreduce(MPI_IN_PLACE, &temp, 1,  MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->dmcomm);
                        pSPARC->Eexx += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * creal(temp);
                    }
                }

                free(rhs);
                free(Vi);
                free(kpt_k_list);
                free(kpt_q_list);
            }
        }
        free(rhs_list_i);
        free(rhs_list_j);
        free(rhs_list_l);
        free(rhs_list_m);

        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        pSPARC->Eexx /= pSPARC->dV;

    } else {

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;
            xi = pSPARC->Xi_kpt + spn_i * xi_shift;

            int Ns_occ = pSPARC->Nstates_occ[spn_i];
            int DMndNsocc = DMnd * Ns_occ;
            int DMndNs = DMnd * Ns;
            double _Complex alpha = 1.0;
            double _Complex beta = 0.0;
            double _Complex *Xi_times_psi = (double _Complex *) calloc(Ns_occ * Ns_occ, sizeof(double _Complex));
            assert(Xi_times_psi != NULL);
            int kpt_k;

            for (kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k++) {
                // perform matrix multiplication psi' * X using ScaLAPACK routines
                cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Ns_occ, Ns_occ, DMnd,
                            &alpha, psi + kpt_k*DMndNs, DMnd, 
                            xi + kpt_k*DMndNsocc, DMnd, &beta, Xi_times_psi, Ns_occ);

                if (size > 1) {
                    // sum over all processors in dmcomm
                    MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Ns_occ*Ns_occ, 
                                MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
                }

                for (i = 0; i < Ns_occ; i++) {
                    double temp = 0.0;
                    for (j = 0; j < Ns_occ; j++) {
                        temp += creal(conj(Xi_times_psi[i+j*Ns_occ]) * Xi_times_psi[i+j*Ns_occ]);
                    }
                    temp *= occ_outer[i + (kpt_k + pSPARC->kpt_start_indx) * Ns];
                    pSPARC->Eexx += pSPARC->kptWts_loc[kpt_k] * temp / pSPARC->Nkpts ;
                }
            }
            free(Xi_times_psi);
        }
    }

    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, &pSPARC->Eexx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
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
 * @brief   Solving Poisson's equation using FFT 
 *          
 *          This function only works for solving Poisson's equation with complex right hand side
 *          Note: only FFT method is supported in current version. 
 *          TODO: add method in real space. 
 */
void poissonSolve_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_FFT_const, int ncol, int DMnd, 
                int *dims, double _Complex *Vi, int *kpt_k_list, int *kpt_q_list, MPI_Comm comm) 
{
    int i, j, k, lsize, lrank, size_s, ncolp;
    int *sendcounts, *sdispls, *recvcounts, *rdispls, **DMVertices, *ncolpp;
    int coord_comm[3], gridsizes[3], DNx, DNy, DNz, DNd, Nd, Nx, Ny, Nz, Ns, kq_shift;
    double _Complex *rhs_loc, *Vi_loc, *rhs_loc_order, *Vi_loc_order, *f;

    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);
    Ns = pSPARC->Nstates;
    size_s = DMnd * ncol;                                                   // it is DMnd * Nband in other parts
    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz; 
    ncolp = ncol / lsize + ((lrank < ncol % lsize) ? 1 : 0);
    kq_shift = 0;                                                           // start index for k&q for each process
    /********************************************************************/

    if (lsize > 1){
        // variables for RHS storage
        rhs_loc = (double _Complex *) malloc(sizeof(double _Complex) * ncolp * Nd);
        rhs_loc_order = (double _Complex *) malloc(sizeof(double _Complex) * Nd * ncolp);

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

        for (i = 0; i < lrank; i++) 
            kq_shift += ncolpp[i];

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

        MPI_Alltoallv(rhs, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, 
                        rhs_loc, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, comm);

        // rhs_full needs rearrangement
        rearrange_rhs((void *) rhs_loc, ncolp, DMVertices, rdispls, lsize, 
                        Nx, Ny, Nd, (void *) rhs_loc_order, sizeof(double _Complex));

        free(rhs_loc);
        // variable for local result Vi
        Vi_loc = (double _Complex *) malloc(sizeof(double _Complex) * Nd * ncolp);
        assert(Vi_loc != NULL);
    } else {
        // if the size of comm is 1, there is no need to scatter and rearrange the results
        // when applying phase factor, rhs will be changed. So get a copy. 
        rhs_loc_order = (double _Complex *) malloc(sizeof(double _Complex) * Nd * ncolp);
        assert(rhs_loc_order != NULL);
        for (i = 0; i < Nd * ncolp; i++)
            rhs_loc_order[i] = rhs[i];
        Vi_loc = Vi;
    }   

    if (pSPARC->EXXMeth_Flag == 0) {                                                // Solve in Fourier Space
        pois_fft_kpt(pSPARC, rhs_loc_order, pois_FFT_const, ncolp, Vi_loc, kpt_k_list + kq_shift, kpt_q_list + kq_shift);
    } else {
        // TODO: Add method for solving in real space.
    }
    
    free(rhs_loc_order);

    if (lsize > 1) {
        Vi_loc_order = (double _Complex *) malloc(sizeof(double _Complex)* Nd * ncolp);
        assert(Vi_loc_order != NULL);

        // Vi_loc needs rearrangement
        rearrange_Vi((void *) Vi_loc, ncolp, DMVertices, lsize, 
                        Nx, Ny, Nd, (void *) Vi_loc_order, sizeof(double _Complex));

        MPI_Alltoallv(Vi_loc_order, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, 
                    Vi, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, comm);

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
 * @brief   Solve Poisson's equation using FFT in Fourier Space - in k-point case. 
 * 
 * @param rhs_loc_order     complete RHS of poisson's equations without parallelization. 
 * @param pois_FFT_const    constant for solving possion's equations
 * @param ncolp             Number of poisson's equations to be solved.
 * @param Vi_loc            complete solutions of poisson's equations without parallelization. 
 * @param kpt_k_list        List of global index of k Bloch wave vector
 * @param kpt_q_list        List of global index of q Bloch wave vector
 * Note:                    Assuming the RHS is periodic with Bloch wave vector (k - q). 
 * Note:                    This function is complete localized. 
 */
void pois_fft_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs_loc_order, double *pois_FFT_const, 
                    int ncolp, double _Complex *Vi_loc, int *kpt_k_list, int *kpt_q_list) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]
    if (ncolp == 0) return;
    int i, j, l, Nd, Nx, Ny, Nz;
    int *Kptshift_map;
    double *alpha;
    double _Complex *rhs_bar;

    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx; Ny = pSPARC->Ny; Nz = pSPARC->Nz; 
    rhs_bar = (double _Complex*) malloc(sizeof(double _Complex) * Nd * ncolp);
    assert(rhs_bar != NULL);
    Kptshift_map = pSPARC->Kptshift_map;
    /********************************************************************/
    
    apply_phase_factor(pSPARC, rhs_loc_order, ncolp, "N", kpt_k_list, kpt_q_list);
    
    // FFT
#if defined(USE_MKL)
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*Nx, Nx, 1};

    for (i = 0; i < ncolp; i++)
        MKL_MDFFT(rhs_loc_order + i * Nd, dim_sizes, strides_out, rhs_bar + i * Nd);
#elif defined(USE_FFTW)
    int dim_sizes[3] = {Nz, Ny, Nx};

    for (i = 0; i < ncolp; i++)
        FFTW_MDFFT(dim_sizes, rhs_loc_order + i * Nd, rhs_bar + i * Nd);
#endif

    // multiplied by alpha
    for (j = 0; j < ncolp; j++) {
        l = Kptshift_map(kpt_k_list[j], kpt_q_list[j]);
        if (!l) {
            alpha = pois_FFT_const + Nd * (pSPARC->Nkpts_shift - 1);
        } else {
            alpha = pois_FFT_const + Nd * (l - 1);
        }
        for (i = 0; i < Nd; i++) {
            rhs_bar[i + j*Nd] = creal(rhs_bar[i + j*Nd]) * alpha[i] 
                                + (cimag(rhs_bar[i + j*Nd]) * alpha[i]) * I;
        }
    }

    // iFFT
#if defined(USE_MKL)
    for (i = 0; i < ncolp; i++)
        MKL_MDiFFT(rhs_bar + i * Nd, dim_sizes, strides_out, Vi_loc + i * Nd);
#elif defined(USE_FFTW)
    for (i = 0; i < ncolp; i++)
        FFTW_MDiFFT(dim_sizes, rhs_bar + i * Nd, Vi_loc + i * Nd);
#endif

    apply_phase_factor(pSPARC, Vi_loc, ncolp, "P", kpt_k_list, kpt_q_list);

    free(rhs_bar);
#undef Kptshift_map
}


/**
 * @brief   Transfer complex vectors from dmcomm to kptcomm_topo for Lancozs algorithm
 *
 *          Used to transfer psi_outer_kpt in case of no-ACE method and transfer 
 *          Xi_kpt (of ACE operator) in case of ACE method from dmcomm to kptcomm_topo
 */
void Transfer_dmcomm_to_kptcomm_topo_complex(SPARC_OBJ *pSPARC, 
        int ncols, double _Complex *psi_outer_kpt, double _Complex *psi_outer_kptcomm_topo_kpt) 
{
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
        D2D_kpt(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, 
            pSPARC->DMVertices_dmcomm, psi_outer_kpt + pSPARC->Nd_d_dmcomm * i,
            pSPARC->DMVertices_kptcomm, psi_outer_kptcomm_topo_kpt + pSPARC->Nd_d_kptcomm * i,
            pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL, sdims, 
            pSPARC->kptcomm_topo, rdims, 
            pSPARC->kptcomm);
    }
}


/**
 * @brief   Gather psi_outer_kpt and occupations in other bandcomms and kptcomms
 * 
 *          In need of communication between k-point topology and dmcomm, additional 
 *          communication for occupations is required. 
 */
void gather_psi_occ_outer_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_outer_kpt, double *occ_outer) {
    int i, j, k, grank, blacs_rank, blacs_size, kpt_bridge_rank, kpt_bridge_size;
    int Ns, DMnd, Nband, Nkpts_hf_kptcomm, spn_i;
    int sendcount, *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &blacs_rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nkpts_hf_kptcomm = pSPARC->Nkpts_hf_kptcomm;

    int size_k = DMnd * Nband;
    int size_k_ = DMnd * Ns;
    int shift_k = pSPARC->kpthf_start_indx * Ns * DMnd;
    int shift_s = pSPARC->band_start_indx * DMnd;
    int shift_k_occ = pSPARC->kpt_start_indx * Ns;                  // gather all occ for all kpts
    
    int shift_spn_psi_outer = DMnd * Ns * pSPARC->Nkpts_hf_red;
    int shift_spn_psi = DMnd * Nband * pSPARC->Nkpts_kptcomm;
    int shift_spn_occ_outer = Ns * pSPARC->Nkpts_sym;
    int shift_spn_occ = Ns * pSPARC->Nkpts_kptcomm;

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        int count = 0;
        for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
            if (!pSPARC->kpthf_flag_kptcomm[k]) continue;
            for (i = 0; i < size_k; i++)  
                pSPARC->psi_outer_kpt[i + shift_s + count * size_k_ + shift_k + spn_i * shift_spn_psi_outer] = pSPARC->Xorb_kpt[i + k * size_k + spn_i * shift_spn_psi];
            count ++;
        }
        
        for (i = 0; i < Ns * pSPARC->Nkpts_kptcomm; i++) 
            pSPARC->occ_outer[i + shift_k_occ + spn_i * shift_spn_occ_outer] = pSPARC->occ[i + spn_i * shift_spn_occ];
    }

    /********************************************************************/

    if (blacs_size > 1) {
        // First step, gather all required bands across blacscomm within each kptcomm
        recvcounts = (int*) malloc(sizeof(int)* blacs_size);
        displs = (int*) malloc(sizeof(int)* blacs_size);
        assert(recvcounts !=NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (Ns - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < blacs_size; i++){
            recvcounts[i] = (i < (Ns / NB) ? NB : (i == (Ns / NB) ? (Ns % NB) : 0)) * DMnd;
            if (i != (blacs_size-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }
        sendcount = 1;

        // Gather Nkpts_hf_kptcomm times. TODO: optimize to only gather once but needs rearrangement
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
            for (i = 0 ; i < Nkpts_hf_kptcomm; i ++) 
                MPI_Allgatherv(MPI_IN_PLACE, sendcount, MPI_DOUBLE_COMPLEX, psi_outer_kpt + i * size_k_ + shift_k + spn_i * shift_spn_psi_outer, 
                    recvcounts, displs, MPI_DOUBLE_COMPLEX, pSPARC->blacscomm);

        free(recvcounts);
        free(displs); 
    }

    if (kpt_bridge_size > 1) {
        // Second step, gather all required kpts across kptcomm
        recvcounts = (int*) malloc(sizeof(int)* kpt_bridge_size);
        displs = (int*) malloc(sizeof(int)* kpt_bridge_size);
        assert(recvcounts !=NULL && displs != NULL);
        displs[0] = 0;
        for (i = 0; i < kpt_bridge_size; i++){
            recvcounts[i] = pSPARC->Nkpts_hf_list[i] * size_k_;
            if (i != (kpt_bridge_size-1))
                displs[i+1] = displs[i] + recvcounts[i];
        }
        sendcount = 1;

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
            MPI_Allgatherv(MPI_IN_PLACE, sendcount, MPI_DOUBLE_COMPLEX, psi_outer_kpt + spn_i * shift_spn_psi_outer, 
                recvcounts, displs, MPI_DOUBLE_COMPLEX, pSPARC->kpt_bridge_comm);   

        free(recvcounts);
        free(displs); 
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
 */
void compute_pois_fft_const_kpt(SPARC_OBJ *pSPARC) {
#define alpha(i,j,k,l) pSPARC->pois_FFT_const[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]
#define alpha1(i,j,k,l) pSPARC->pois_FFT_const_stress[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]
#define alpha2(i,j,k,l) pSPARC->pois_FFT_const_stress2[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]
#define beta(i,j,k,l) pSPARC->pois_FFT_const_press[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]
    int i, j, k, l, Nx, Ny, Nz, Nd, Nxy, Nkpts_shift;
    double L1, L2, L3, V, R_c, G[3], g[3], G2, omega, omega2;

    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;

    // When BC is Dirichelet, one more FD node is incldued.
    // Real length of each side has to be added by 1 mesh length.
    // Only happens for  sheet and wire. 
    L1 = pSPARC->delta_x * Nx;
    L2 = pSPARC->delta_y * Ny;
    L3 = pSPARC->delta_z * Nz;

    Nd = pSPARC->Nd;
    Nxy = Ny * Nx;
    Nkpts_shift = pSPARC->Nkpts_shift;
    V =  L1 * L2 * L3 * pSPARC->Jacbdet * pSPARC->Nkpts_hf;       // Nk_hf * unit cell volume
    R_c = pow(3*V/(4*M_PI),(1.0/3));
    omega = pSPARC->hyb_range_fock;
    omega2 = omega * omega;

    pSPARC->pois_FFT_const = (double *)malloc(sizeof(double) * Nd * Nkpts_shift);
    assert(pSPARC->pois_FFT_const != NULL);
    // allocate space for stress
    if (pSPARC->Calc_stress == 1) {
        pSPARC->pois_FFT_const_stress = (double *)malloc(sizeof(double) * Nd * Nkpts_shift);
        assert(pSPARC->pois_FFT_const_stress != NULL);
        if (pSPARC->EXXDiv_Flag == 0) {
            pSPARC->pois_FFT_const_stress2 = (double *)malloc(sizeof(double) * Nd * Nkpts_shift);
            assert(pSPARC->pois_FFT_const_stress2 != NULL);
        }
    } else if (pSPARC->Calc_pres == 1) {
        if (pSPARC->EXXDiv_Flag != 0) {
            pSPARC->pois_FFT_const_press = (double *)malloc(sizeof(double) * Nd * Nkpts_shift);
            assert(pSPARC->pois_FFT_const_press != NULL);
        }
    }
    /********************************************************************/

    // spherical truncation
    if (pSPARC->EXXDiv_Flag == 0) {
        for (l = 0; l < Nkpts_shift; l++) {
            for (k = 0; k < Nz; k++) {
                for (j = 0; j < Ny; j++) {
                    for (i = 0; i < Nx; i++) {
                        // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                        g[0] = g[1] = g[2] = 0.0;
                        G[0] = (i < Nx/2+1) ? (i*2*M_PI/L1) : ((i-Nx)*2*M_PI/L1);
                        G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                        G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                        if (l < Nkpts_shift - 1) {                                                      // The last shift is all zeros
                            G[0] += pSPARC->k1_shift[l];
                            G[1] += pSPARC->k2_shift[l];
                            G[2] += pSPARC->k3_shift[l];
                        }
                        matrixTimesVec_3d(pSPARC->lapcT, G, g);
                        G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                        if (fabs(G2) > 1e-4) {                                              // 1e-4 is the tolerance from ABINIT
                            alpha(i,j,k,l) = 4*M_PI*(1-cos(R_c*sqrt(G2)))/G2;
                            if (pSPARC->Calc_stress == 1) {
                                double x = R_c*sqrt(G2);
                                double G4 = G2*G2;
                                alpha1(i,j,k,l) = 4*M_PI*( 1-cos(x)- x/2*sin(x) )/G4;
                                alpha2(i,j,k,l) = 4*M_PI*( x/2*sin(x) )/G2/3;
                            }
                        } else {
                            alpha(i,j,k,l) = 2*M_PI*(R_c * R_c);
                            if (pSPARC->Calc_stress == 1) {
                                alpha1(i,j,k,l) = 4*M_PI*pow(R_c,4)/24;
                                alpha2(i,j,k,l) = 4*M_PI*( R_c*R_c/2 )/3;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // auxiliary function
    if (pSPARC->EXXDiv_Flag == 1) {
        for (l = 0; l < Nkpts_shift; l++) {
            for (k = 0; k < Nz; k++) {
                for (j = 0; j < Ny; j++) {
                    for (i = 0; i < Nx; i++) {
                        // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                        g[0] = g[1] = g[2] = 0.0;
                        G[0] = (i < Nx/2+1) ? (i*2*M_PI/L1) : ((i-Nx)*2*M_PI/L1);
                        G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                        G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                        if (l < Nkpts_shift - 1) {                                                      // The last shift is all zeros
                            G[0] += pSPARC->k1_shift[l];
                            G[1] += pSPARC->k2_shift[l];
                            G[2] += pSPARC->k3_shift[l];
                        }
                        matrixTimesVec_3d(pSPARC->lapcT, G, g);
                        G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                        double x = -0.25/omega2*G2;
                        if (fabs(G2) > 1e-4) {                                              // 1e-4 is the tolerance from ABINIT
                            if (omega > 0) {
                                alpha(i,j,k,l) = 4*M_PI/G2 * (1 - exp(-0.25/omega2*G2));
                                if (pSPARC->Calc_stress == 1) {
                                    double G4 = G2*G2;
                                    alpha1(i,j,k,l) = 4*M_PI*(1 - exp(x)*(1-x))/G4 /4;
                                } else if (pSPARC->Calc_pres == 1) {
                                    beta(i,j,k,l) = 4*M_PI*(1 - exp(x)*(1-x))/G2 /4;
                                }
                            } else {
                                alpha(i,j,k,l) = 4*M_PI/G2;
                                if (pSPARC->Calc_stress == 1) {
                                    double G4 = G2*G2;
                                    alpha1(i,j,k,l) = 4*M_PI/G4 /4;
                                } else if (pSPARC->Calc_pres == 1) {
                                    beta(i,j,k,l) = 4*M_PI/G2 /4;
                                }
                            }
                        } else {
                            if (omega > 0) {
                                alpha(i,j,k,l) = 4*M_PI*(pSPARC->const_aux + 0.25/omega2);
                                if (pSPARC->Calc_stress == 1) {
                                    alpha1(i,j,k,l) = 0;                                    
                                } else if (pSPARC->Calc_pres == 1) {
                                    beta(i,j,k,l) = 0;
                                }
                            } else {
                                alpha(i,j,k,l) = 4*M_PI*pSPARC->const_aux;
                                if (pSPARC->Calc_stress == 1) {
                                    alpha1(i,j,k,l) = 0;                                                                        
                                } else if (pSPARC->Calc_pres == 1) {
                                    beta(i,j,k,l) = 0;
                                }
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
        for (l = 0; l < Nkpts_shift; l++) {
            for (k = 0; k < Nz; k++) {
                for (j = 0; j < Ny; j++) {
                    for (i = 0; i < Nx; i++) {
                        // G = [(k1-1)*2*pi/L1, (k2-1)*2*pi/L2, (k3-1)*2*pi/L3];
                        g[0] = g[1] = g[2] = 0.0;
                        G[0] = (i < Nx/2+1) ? (i*2*M_PI/L1) : ((i-Nx)*2*M_PI/L1);
                        G[1] = (j < Ny/2+1) ? (j*2*M_PI/L2) : ((j-Ny)*2*M_PI/L2);
                        G[2] = (k < Nz/2+1) ? (k*2*M_PI/L3) : ((k-Nz)*2*M_PI/L3);
                        if (l < Nkpts_shift - 1) {                                                      // The last shift is all zeros
                            G[0] += pSPARC->k1_shift[l];
                            G[1] += pSPARC->k2_shift[l];
                            G[2] += pSPARC->k3_shift[l];
                        }
                        matrixTimesVec_3d(pSPARC->lapcT, G, g);
                        G2 = G[0] * g[0] + G[1] * g[1] + G[2] * g[2];
                        double x = -G2/4.0/omega2;
                        if (fabs(G2) > 1e-4) {                                              // 1e-4 is the tolerance from ABINIT
                            alpha(i,j,k,l) = 4*M_PI*(1-exp(x))/G2;
                            if (pSPARC->Calc_stress == 1) {
                                double G4 = G2*G2;
                                alpha1(i,j,k,l) = 4*M_PI*( 1-exp(x)*(1-x) )/G4;
                            } else if (pSPARC->Calc_pres == 1) {
                                beta(i,j,k,l) = 4*M_PI*( 1-exp(x)*(1-x) )/G2;
                            }
                        } else {
                            alpha(i,j,k,l) = M_PI/omega2;
                            if (pSPARC->Calc_stress == 1) {
                                alpha1(i,j,k,l) = 0;
                            } else if (pSPARC->Calc_pres == 1) {
                                beta(i,j,k,l) = 0;
                            }
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
 * @brief   Find out the unique Bloch vector shifts (k-q)
 */
void find_k_shift(SPARC_OBJ *pSPARC) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]
    int rank, k_ind, q_ind, i, count, flag;
    double *k1_shift, *k2_shift, *k3_shift;
    double k1_shift_temp, k2_shift_temp, k3_shift_temp;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    k1_shift = (double *) calloc(sizeof(double), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    k2_shift = (double *) calloc(sizeof(double), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    k3_shift = (double *) calloc(sizeof(double), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    pSPARC->Kptshift_map = (int*) calloc(sizeof(int), pSPARC->Nkpts_sym*pSPARC->Nkpts_hf);
    assert(k1_shift != NULL && k2_shift != NULL && k3_shift != NULL && pSPARC->Kptshift_map != NULL);

    int *Kptshift_map = pSPARC->Kptshift_map;
    // k_shift[0] = {0,0,0}. remove later.
    k1_shift[0] = k2_shift[0] = k3_shift[0] = 0.0;
    count = 1;
    for (q_ind = 0; q_ind < pSPARC->Nkpts_hf; q_ind ++) {
        for (k_ind = 0; k_ind < pSPARC->Nkpts_sym; k_ind ++) {
            
            k1_shift_temp = pSPARC->k1[k_ind] - pSPARC->k1_hf[q_ind];
            k2_shift_temp = pSPARC->k2[k_ind] - pSPARC->k2_hf[q_ind];
            k3_shift_temp = pSPARC->k3[k_ind] - pSPARC->k3_hf[q_ind];
            if (fabs(k1_shift_temp) < TEMP_TOL && fabs(k2_shift_temp) < TEMP_TOL 
                && fabs(k3_shift_temp) < TEMP_TOL) {
                Kptshift_map(k_ind, q_ind) = 0;
                continue;
            }

            // flag - 0, new shift;
            flag = 0;
            for (i = 0; i < count; i++) {
                if (fabs(k1_shift[i] - k1_shift_temp) < TEMP_TOL && 
                    fabs(k2_shift[i] - k2_shift_temp) < TEMP_TOL &&
                    fabs(k3_shift[i] - k3_shift_temp) < TEMP_TOL) {
                    flag = i;
                    break;
                }
            }
            if (!flag) {
                k1_shift[count] = k1_shift_temp;
                k2_shift[count] = k2_shift_temp;
                k3_shift[count] = k3_shift_temp;
                Kptshift_map(k_ind, q_ind) = count++;
            } else {
                Kptshift_map(k_ind, q_ind) = flag;
            }
        }
    }
    pSPARC->Nkpts_shift = count;
    // No need to store [0,0,0] shift, which is always there. 
    pSPARC->k1_shift = (double *) calloc(sizeof(double), count - 1);
    pSPARC->k2_shift = (double *) calloc(sizeof(double), count - 1);
    pSPARC->k3_shift = (double *) calloc(sizeof(double), count - 1);
    assert(pSPARC->k1_shift !=NULL && pSPARC->k2_shift != NULL && pSPARC->k3_shift != NULL);

    for (i = 0; i < pSPARC->Nkpts_shift - 1; i++) {
        pSPARC->k1_shift[i] = k1_shift[i+1];
        pSPARC->k2_shift[i] = k2_shift[i+1];
        pSPARC->k3_shift[i] = k3_shift[i+1];
    }

    free(k1_shift);
    free(k2_shift);
    free(k3_shift);
#ifdef DEBUG
    if (!rank) printf("Unique Block vector shift (k-q), Nkpts_shift = %d\n", pSPARC->Nkpts_shift);
    for (int nk = 0; nk < min(pSPARC->Nkpts_shift - 1, 10); nk++) {
        double tpiblx = 2 * M_PI / pSPARC->range_x;
        double tpibly = 2 * M_PI / pSPARC->range_y;
        double tpiblz = 2 * M_PI / pSPARC->range_z;
        if (!rank) printf("k1_shift[%2d]: %8.4f, k2_shift[%2d]: %8.4f, k3_shift[%2d]: %8.4f\n",
            nk,pSPARC->k1_shift[nk]/tpiblx,nk,pSPARC->k2_shift[nk]/tpibly,nk,pSPARC->k3_shift[nk]/tpiblz);
    }
    if (pSPARC->Nkpts_shift - 1 <= 10) {
        if (!rank) printf("k1_shift[%2d]: %8.4f, k2_shift[%2d]: %8.4f, k3_shift[%2d]: %8.4f\n",
            pSPARC->Nkpts_shift - 1,0.0,pSPARC->Nkpts_shift - 1,0.0,pSPARC->Nkpts_shift - 1,0.0);
    }
#endif
#undef Kptshift_map
}


/**
 * @brief   Find out the positive shift exp(i*(k-q)*r) and negative shift exp(-i*(k-q)*r)
 *          for each unique Bloch shifts
 */
void kshift_phasefactor(SPARC_OBJ *pSPARC) {
    if (pSPARC->Nkpts_shift == 1) return;
#define neg_phase(i,j,k,l) neg_phase[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]
#define pos_phase(i,j,k,l) pos_phase[(l)*Nd+(k)*Nxy+(j)*Nx+(i)]

    int i, j, k, l, Nx, Ny, Nxy, Nz, Nd, Nkpts_shift;
    double L1, L2, L3, rx, ry, rz, dot_prod;
    double _Complex *neg_phase, *pos_phase;

    L1 = pSPARC->range_x;
    L2 = pSPARC->range_y;
    L3 = pSPARC->range_z;
    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    Nxy = Nx * Ny;
    Nd = pSPARC->Nd;
    Nkpts_shift = pSPARC->Nkpts_shift;

    pSPARC->neg_phase = (double _Complex *) calloc(sizeof(double _Complex), Nd * (Nkpts_shift - 1));
    pSPARC->pos_phase = (double _Complex *) calloc(sizeof(double _Complex), Nd * (Nkpts_shift - 1));
    assert(pSPARC->neg_phase != NULL && pSPARC->pos_phase != NULL);
    neg_phase = pSPARC->neg_phase;
    pos_phase = pSPARC->pos_phase;

    for (l = 0; l < Nkpts_shift - 1; l ++) {
        for (k = 0; k < Nz; k ++) {
            for (j = 0; j < Ny; j ++) {
                for (i = 0; i < Nx; i ++) {
                    rx = i * L1 / Nx;
                    ry = j * L2 / Ny;
                    rz = k * L3 / Nz;
                    dot_prod = rx * pSPARC->k1_shift[l] + ry * pSPARC->k2_shift[l] + rz * pSPARC->k3_shift[l];
                    // neg_phase(:,i) = exp(-1i*r*k_shift');
                    neg_phase(i,j,k,l) = cos(dot_prod) - I * sin(dot_prod);
                    // pos_phase(:,i) = exp(1i*r*k_shift');
                    pos_phase(i,j,k,l) = cos(dot_prod) + I * sin(dot_prod);
                }
            }
        }
    }
#undef neg_phase
#undef pos_phase
}


/**
 * @brief   Find out the k-point for hartree-fock exact exchange in local process
 */
void find_local_kpthf(SPARC_OBJ *pSPARC) 
{
    int k, nk, nk_hf, count, *list, kpt_bridge_size, kpt_bridge_rank, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    pSPARC->kpthf_flag_kptcomm = (int *) calloc(pSPARC->Nkpts_kptcomm, sizeof(int));
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (nk_hf = 0; nk_hf < pSPARC->Nkpts_hf_red; nk_hf ++) {
            if (pSPARC->kpts_hf_red_list[nk_hf] == (pSPARC->kpt_start_indx + k)){
                pSPARC->kpthf_flag_kptcomm[k] = 1;
                count ++;
                break;
            }
        }
    }
    pSPARC->Nkpts_hf_kptcomm = count;
    pSPARC->Nkpts_hf_list = (int *) calloc(sizeof(int), kpt_bridge_size);
    MPI_Allgather(&count, 1, MPI_INT, pSPARC->Nkpts_hf_list, 1, MPI_INT, pSPARC->kpt_bridge_comm);
    pSPARC->kpthf_start_indx = 0;                                               // shift in terms of Nkpts_hf_red
    for (k = 0; k < kpt_bridge_rank; k++)
        pSPARC->kpthf_start_indx += pSPARC->Nkpts_hf_list[k];

#ifdef DEBUG
    if (!rank) printf("Number of k-point orbitals gathered from each k-point processes:\n");
    for (k = 0; k < kpt_bridge_size; k++) {
        if (!rank) printf("Nkpts_hf_list [%d]: %d\n",k,pSPARC->Nkpts_hf_list[k]);
    }
#endif
}


/**
 * @brief   Apply phase factor by Bloch wave vector shifts. 
 * 
 * @param vec           vectors to be applied phase factors
 * @param ncol          number of columns of vectors
 * @param NorP          "N" is for negative phase factor, "P" is for positive.
 * @param kpt_k_list    list of global k-point index for k
 * @param kpt_q_list    list of global k-point index for q
 * Note:                Assuming the shifts is introduced by wave vector (k - q)
 */
void apply_phase_factor(SPARC_OBJ *pSPARC, double _Complex *vec, int ncol, char *NorP, int *kpt_k_list, int *kpt_q_list) 
{
#define Kptshift_map(i,j) Kptshift_map[i+j*pSPARC->Nkpts_sym]

    int i, l, col, Nd, Nx, Ny, Nz, Nxy, rank;
    int *Kptshift_map;
    double _Complex *phase;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Nd = pSPARC->Nd;
    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;
    Nxy = Nx * Ny;

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
        l = Kptshift_map(kpt_k_list[col], kpt_q_list[col]);
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
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return;
    int rank, i, j, spn_i, Ns_occ_temp[2], sum_temp, sum;
    int Ns, DMnd, Nsocc_max_loc, Nsocc_temp, Ns_occ;
    double *occ;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    sum_temp = sum = 0;
    int occ_shift = pSPARC->Nstates * pSPARC->Nkpts_kptcomm;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        occ = pSPARC->occ + spn_i*occ_shift;

        // Find maximal number of occupied states among all k-points
        for (i = 0; i < pSPARC->Nkpts_kptcomm; i++) {
            Nsocc_temp = 0;
            for (j = 0; j < Ns; j++) {
                if (occ[j + i*Ns] > 1e-6)
                    Nsocc_temp ++;
            }
            if (!i)
                Nsocc_max_loc = Nsocc_temp;
            else
                Nsocc_max_loc = max(Nsocc_max_loc, Nsocc_temp);
        }

        MPI_Allreduce(&Nsocc_max_loc, &Ns_occ, 1, 
                    MPI_INT, MPI_MAX, pSPARC->kpt_bridge_comm);

        // Apply ExxAceVal_state here
        Ns_occ += pSPARC->EXXACEVal_state;
        Ns_occ = min(Ns_occ, pSPARC->Nstates);                      // Ensure Ns_occ is less or equal to Nstates

        Ns_occ_temp[spn_i] = Ns_occ;
    }

    // Note: occupations are only correct in dmcomm. We also need it to be correct in 
    // kptcomm_topo. Generally, all processes which need to store Xi.
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
            printf("\nTotal number of occupied states + Extra states changed : %d\n", sum_temp);
        #endif  
            free(pSPARC->Xi_kpt);
            free(pSPARC->Xi_kptcomm_topo_kpt);
            pSPARC->Xi_kpt = NULL;
            pSPARC->Xi_kptcomm_topo_kpt = NULL;
        } else {
        #ifdef DEBUG
        if(!rank) 
            printf("\nStart to use %d states to create ACE operator for each k-point and spin.\n", sum_temp);
        #endif  
        }

        // Xi, ACE operator
        pSPARC->Xi_kpt = (double _Complex *) calloc(DMnd * sum_temp * pSPARC->Nkpts_kptcomm, sizeof(double _Complex));
        // Storage of ACE operator in kptcomm_topo
        pSPARC->Xi_kptcomm_topo_kpt = 
                (double _Complex *)calloc(pSPARC->Nd_d_kptcomm * sum_temp * pSPARC->Nkpts_kptcomm, sizeof(double _Complex));
        assert(pSPARC->Xi_kpt != NULL && pSPARC->Xi_kptcomm_topo_kpt != NULL);
    }

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
        pSPARC->Nstates_occ[spn_i] = Ns_occ_temp[spn_i];
}