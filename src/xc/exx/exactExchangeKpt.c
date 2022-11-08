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
    double _Complex *psi, double *occ_outer, int spn_i, double _Complex *Xi_kpt)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, k, ll, kpt_k, kpt_q, count, rank, Ns, Ns_occ, DMnd, ONE = 1;
    int DMndNsocc, Nband, Nband_M, size_k;
    double *occ, t1, t2, t_comm;
    double _Complex *M, *Xi_kpt_;
    double _Complex *psi_storage1_kpt, *psi_storage2_kpt, *psi_storage1_band, *psi_storage2_band;
    psi_storage1_kpt = psi_storage2_kpt = psi_storage1_band = psi_storage2_band = NULL;
    double _Complex *sendbuff_kpt, *recvbuff_kpt, *sendbuff_band, *recvbuff_band;
    MPI_Request reqs_kpt[2], reqs_band[2];
    /******************************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Nband = pSPARC->Nband_bandcomm;
    Ns = pSPARC->Nstates;
    Nband_M = pSPARC->Nband_bandcomm_M[spn_i];
    DMnd = pSPARC->Nd_d_dmcomm;     
    Ns_occ = pSPARC->Nstates_occ[spn_i];
    size_k = DMnd * Nband;
    DMndNsocc = DMnd * Ns_occ;
    /******************************************************************************/

    int kpt_bridge_comm_rank, kpt_bridge_comm_size;
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_comm_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_comm_size);
    int blacscomm_rank, blacscomm_size;
    MPI_Comm_rank(pSPARC->blacscomm, &blacscomm_rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacscomm_size);

    // check the occupations for kpthf
    // if all occupations are 0 for kpthf, then Xi is zero. 
    // Avoid the bug in solving LAPACKE_zpotrf
    count = 0;
    for (kpt_q = 0; kpt_q < pSPARC->Nkpts_hf; kpt_q++) {
        ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
        occ = occ_outer + ll * Ns;
        for (i = 0; i < Ns; i++) {
            if (occ[i] > 1e-6) count++;
        }
    }
    if (!count) return;

    int reps_band = pSPARC->npband - 1;
    int Nband_max = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
    int reps_kpt = pSPARC->npkpt - 1;
    int Nkpthf_red_max = pSPARC->Nkpts_hf_kptcomm;
    for (k = 0; k < pSPARC->npkpt; k++) 
        Nkpthf_red_max = max(Nkpthf_red_max, pSPARC->Nkpts_hf_list[k]);

    // starts to create Xi
    t_comm = 0;
    memset(Xi_kpt, 0, sizeof(double _Complex) * DMndNsocc * pSPARC->Nkpts_kptcomm);
    psi_storage1_kpt = (double complex *) calloc(sizeof(double complex), DMnd * Nband * Nkpthf_red_max);
    assert(psi_storage1_kpt != NULL);
    if (reps_kpt > 0) {
        psi_storage2_kpt = (double complex *) calloc(sizeof(double complex), DMnd * Nband * Nkpthf_red_max);
        assert(psi_storage2_kpt != NULL);
    }
    // extract and store all the orbitals for hybrid calculation
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        if (!pSPARC->kpthf_flag_kptcomm[k]) continue;
        for (i = 0; i < size_k; i++)  
            psi_storage1_kpt[i + count * size_k] = psi[i + k * size_k];
        count ++;
    }
    if (reps_band > 0) {
        psi_storage1_band = (double complex *) calloc(sizeof(double complex), DMnd * Nband_max);
        psi_storage2_band = (double complex *) calloc(sizeof(double complex), DMnd * Nband_max);
        assert(psi_storage1_band != NULL && psi_storage2_band != NULL);
    }

    for (int rep_kpt = 0; rep_kpt <= reps_kpt; rep_kpt++) {
        // transfer the orbitals in the rotation way across kpt_bridge_comm
        if (rep_kpt == 0) {
            sendbuff_kpt = psi_storage1_kpt;
            if (reps_kpt > 0) {
                recvbuff_kpt = psi_storage2_kpt;
                t1 = MPI_Wtime();
                transfer_orbitals_kptbridgecomm_kpt(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt);
                t2 = MPI_Wtime();
                t_comm += (t2 - t1);
            }
        } else {
            t1 = MPI_Wtime();
            MPI_Waitall(2, reqs_kpt, MPI_STATUSES_IGNORE);
            sendbuff_kpt = (rep_kpt%2==1) ? psi_storage2_kpt : psi_storage1_kpt;
            recvbuff_kpt = (rep_kpt%2==1) ? psi_storage1_kpt : psi_storage2_kpt;
            if (rep_kpt != reps_kpt) {
                transfer_orbitals_kptbridgecomm_kpt(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt);
            }
            t2 = MPI_Wtime();
            t_comm += (t2 - t1);
        }

        int source_kpt = (kpt_bridge_comm_rank-rep_kpt+kpt_bridge_comm_size)%kpt_bridge_comm_size;
        int nkpthf_red = pSPARC->Nkpts_hf_list[source_kpt];
        int kpthf_start_indx = pSPARC->kpthf_start_indx_list[source_kpt];
        
        for (k = 0; k < nkpthf_red; k++) {
            int k_indx = k + kpthf_start_indx;
            int counts = pSPARC->kpthfred2kpthf[k_indx][0];

            for (int rep_band = 0; rep_band <= reps_band; rep_band++) {
                if (rep_band == 0) {
                    // psi_storage_band = psi_storage_kpt + k*size_k;
                    sendbuff_band = sendbuff_kpt + k*size_k;
                    if (reps_band > 0) {
                        t1 = MPI_Wtime();
                        recvbuff_band = psi_storage1_band;
                        transfer_orbitals_blacscomm_kpt(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band);
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
                        transfer_orbitals_blacscomm_kpt(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band);
                    }
                    t2 = MPI_Wtime();
                    t_comm += (t2 - t1);
                }
                
                for (count = 0; count < counts; count++) {
                    kpt_q = pSPARC->kpthfred2kpthf[k_indx][count+1];
                    ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
                    occ = occ_outer + ll * Ns;
                    // solve poisson's equations 
                    solve_allpair_poissons_equation_apply2Xi_kpt(pSPARC, Nband_M, 
                        psi, sendbuff_band, occ, Xi_kpt, kpt_q, rep_band, Ns_occ);
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

    // i -- kpt_k, j -- kpt_q
    for (kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k ++) {
        Xi_kpt_ = Xi_kpt + pSPARC->band_start_indx*DMnd + kpt_k*DMndNsocc;

        int nrows_M = pSPARC->nrows_M[spn_i];
        int ncols_M = pSPARC->ncols_M[spn_i];
        M = (double _Complex *)calloc(nrows_M * ncols_M, sizeof(double _Complex));
        assert(M != NULL);
        
        double _Complex alpha = 1.0;
        double _Complex beta = 0.0;
        t1 = MPI_Wtime();
        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        // perform matrix multiplication psi' * W using ScaLAPACK routines
        pzgemm_("C", "N", &Ns_occ, &Ns_occ, &DMnd, &alpha, 
                psi + kpt_k*size_k, &ONE, &ONE, pSPARC->desc_Xi[spn_i], 
                Xi_kpt_, &ONE, &ONE, pSPARC->desc_Xi[spn_i], 
                &beta, M,  &ONE, &ONE, pSPARC->desc_M[spn_i]);
        #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        // add the implementation without SCALAPACK
        exit(255);
        #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

        if (pSPARC->npNd > 1) {
            // sum over all processors in dmcomm
            MPI_Allreduce(MPI_IN_PLACE, M, nrows_M * ncols_M,
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
        for (i = 0; i < nrows_M * ncols_M; i++)   M[i] = -M[i];

        // t1 = MPI_Wtime();
        int info = 0;
        if (nrows_M*ncols_M > 0) {
            info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U', Ns_occ, M, Ns_occ);
            assert(info == 0);
        }
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && !spn_i) 
            printf("==Cholesky Factorization: "
                "info = %d, computing Cholesky Factorization using LAPACKE_zpotrf: %.3f ms\n", 
                info, (t2 - t1)*1e3);
        #endif

        // Xi = WM^(-1)
        t1 = MPI_Wtime();
        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        pztrsm_("R", "U", "N", "N", &DMnd, &Ns_occ, &alpha, 
                M, &ONE, &ONE, pSPARC->desc_M[spn_i], 
                Xi_kpt_, &ONE, &ONE, pSPARC->desc_Xi[spn_i]);
        #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        // add the implementation without SCALAPACK
        exit(255);
        #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && !spn_i) 
            printf("==Triangular matrix equation: "
                "Solving triangular matrix equation using cblas_ztrsm: %.3f ms\n", (t2 - t1)*1e3);
        #endif

        free(M);

        // gather all columns of Xi
        gather_blacscomm_kpt(pSPARC, Ns_occ, Xi_kpt + kpt_k*DMndNsocc);
    }
}


/**
 * @brief   Evaluating Exact Exchange potential in k-point case
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ncol, int DMnd, double _Complex *Hx, int spin, int kpt, MPI_Comm comm) 
{        
    int rank, Lanczos_flag, dims[3];
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
    int size, batch_num_rhs, NL, base, loop, Nkpts_hf;
    int *rhs_list_i, *rhs_list_j, *rhs_list_l, *kpt_k_list, *kpt_q_list;
    double occ, exx_frac, occ_alpha;
    double _Complex *rhs, *Vi;

    Ns = pSPARC->Nstates;
    Nkpts_hf = pSPARC->Nkpts_hf;
    exx_frac = pSPARC->exx_frac;
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
            occ_alpha = occ * exx_frac;
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
    if (ncol != 1) {
        cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Ns_occ, ncol, DMnd,
                    &alpha, Xi + kpt*DMndNs_occ, DMnd, X, DMnd, &beta, Xi_times_psi, Ns_occ);
    } else {
        cblas_zgemv( CblasColMajor, CblasConjTrans, DMnd, Ns_occ, 
                    &alpha, Xi + kpt*DMndNs_occ, DMnd, X, 1, &beta, Xi_times_psi, 1);
    }

    if (size > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Ns_occ*ncol, 
                      MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    }

    alpha = -pSPARC->exx_frac;
    beta = 1.0;
    // perform matrix multiplication Xi * (Xi'*X) using ScaLAPACK routines
    if (ncol != 1) {
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, ncol, Ns_occ,
                    &alpha, Xi + kpt*DMndNs_occ, DMnd, Xi_times_psi, Ns_occ, &beta, Hx, DMnd);
    } else {
        cblas_zgemv( CblasColMajor, CblasNoTrans, DMnd, Ns_occ, 
                    &alpha, Xi + kpt*DMndNs_occ, DMnd, Xi_times_psi, 1, &beta, Hx, 1);
    }

    free(Xi_times_psi);
}


/**
 * @brief   Evaluate Exact Exchange Energy in k-point case.
 */
void evaluate_exact_exchange_energy_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, ll_red, m, grank, rank, size, spn_i;
    int Ns, Ns_loc, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base;
    double occ_i, occ_j, *occ_outer;
    double _Complex *rhs, *Vi, *psi_outer, *psi, *xi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
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

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    if (pSPARC->ACEFlag == 0) {
        int size_k = DMnd * Nband;
        int size_k_ = DMnd * Ns;
        int psi_outer_shift = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
        int psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
        int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

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
                                rhs[k + (count-base)*DMnd] = conj(psi_outer[k + j*DMnd + ll_red*size_k_]) * psi[k + i*DMnd + m*size_k];
                        } else {
                            for (k = 0; k < DMnd; k++) 
                                rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd + ll_red*size_k_] * psi[k + i*DMnd + m*size_k];
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
        pSPARC->Eexx /= pSPARC->dV;

    } else {
        int size_k = DMnd*Nband;
        int xi_shift = DMnd * pSPARC->Nstates_occ[0] * pSPARC->Nkpts_kptcomm;
        int psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
        int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;

        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            int Ns_occ = pSPARC->Nstates_occ[spn_i];
            int Nband_bandcomm_M = pSPARC->Nband_bandcomm_M[spn_i];
            if (Nband_bandcomm_M == 0) continue;
            
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;
            xi = pSPARC->Xi_kpt + spn_i * xi_shift;
            int DMndNsocc = DMnd * Ns_occ;
            double _Complex alpha = 1.0;
            double _Complex beta = 0.0;
            double _Complex *Xi_times_psi = (double _Complex *) calloc(Nband_bandcomm_M * Ns_occ, sizeof(double _Complex));
            assert(Xi_times_psi != NULL);
            int kpt_k;

            for (kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k++) {
                // perform matrix multiplication psi' * X using ScaLAPACK routines
                cblas_zgemm( CblasColMajor, CblasConjTrans, CblasNoTrans, Nband_bandcomm_M, Ns_occ, DMnd,
                            &alpha, psi + kpt_k*size_k, DMnd, 
                            xi + kpt_k*DMndNsocc, DMnd, &beta, Xi_times_psi, Nband_bandcomm_M);

                if (size > 1) {
                    // sum over all processors in dmcomm
                    MPI_Allreduce(MPI_IN_PLACE, Xi_times_psi, Nband_bandcomm_M*Ns_occ, 
                                MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
                }

                for (i = 0; i < Nband_bandcomm_M; i++) {
                    double temp = 0.0;
                    for (j = 0; j < Ns_occ; j++) {
                        temp += creal(conj(Xi_times_psi[i+j*Nband_bandcomm_M]) * Xi_times_psi[i+j*Nband_bandcomm_M]);
                    }
                    temp *= occ_outer[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
                    pSPARC->Eexx += pSPARC->kptWts_loc[kpt_k] * temp / pSPARC->Nkpts ;
                    if (pSPARC->Eexx != pSPARC->Eexx) printf("temp %f, occ %f\n", temp, occ_outer[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns]);
                }
            }
            free(Xi_times_psi);
        }
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

#ifdef DEBUG
    t2 = MPI_Wtime();
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
    int i, k, lsize, lrank, ncolp;
    int *sendcounts, *sdispls, *recvcounts, *rdispls, **DMVertices, *ncolpp;
    int coord_comm[3], gridsizes[3], DNx, DNy, DNz, Nd, Nx, Ny, Nz, kq_shift;
    double _Complex *rhs_loc, *Vi_loc, *rhs_loc_order, *Vi_loc_order, *f;
    sendcounts = sdispls = recvcounts = rdispls = ncolpp = NULL;
    rhs_loc = Vi_loc = rhs_loc_order = Vi_loc_order = f = NULL;
    DMVertices = NULL;

    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);
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
    int i, k, grank, blacs_rank, blacs_size, kpt_bridge_rank, kpt_bridge_size;
    int Ns, DMnd, Nband, Nkpts_hf_kptcomm, spn_i;
    int sendcount, *recvcounts, *displs;

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
    // int shift_k = pSPARC->kpthf_start_indx * Nband * DMnd;
    int shift_k_ = pSPARC->kpthf_start_indx * Ns * DMnd;
    int shift_s = pSPARC->band_start_indx * DMnd;
    int shift_k_occ = pSPARC->kpt_start_indx * Ns;                  // gather all occ for all kpts
    
    // int shift_spn_psi_outer = DMnd * Nband * pSPARC->Nkpts_hf_red;
    int shift_spn_psi_outer_ = DMnd * Ns * pSPARC->Nkpts_hf_red;
    int shift_spn_psi = DMnd * Nband * pSPARC->Nkpts_kptcomm;
    int shift_spn_occ_outer = Ns * pSPARC->Nkpts_sym;
    int shift_spn_occ = Ns * pSPARC->Nkpts_kptcomm;

    // local arrangement of psi
    if (pSPARC->ACEFlag == 0) {
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            int count = 0;
            for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
                if (!pSPARC->kpthf_flag_kptcomm[k]) continue;
                for (i = 0; i < size_k; i++)  
                    pSPARC->psi_outer_kpt[i + shift_s + count * size_k_ + shift_k_ + spn_i * shift_spn_psi_outer_] = pSPARC->Xorb_kpt[i + k * size_k + spn_i * shift_spn_psi];
                count ++;
            }
        }
    }

    // local arrangement of occ
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        for (i = 0; i < Ns * pSPARC->Nkpts_kptcomm; i++) 
            pSPARC->occ_outer[i + shift_k_occ + spn_i * shift_spn_occ_outer] = pSPARC->occ[i + spn_i * shift_spn_occ];
    }

    /********************************************************************/

    if (pSPARC->ACEFlag == 0) {
        if (blacs_size > 1) {
            // First step, gather all required bands across blacscomm within each kptcomm
            for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
                for (i = 0 ; i < Nkpts_hf_kptcomm; i ++) 
                    gather_blacscomm_kpt(pSPARC, Ns, psi_outer_kpt + i * size_k_ + shift_k_ + spn_i * shift_spn_psi_outer_);
        }

        if (kpt_bridge_size > 1) {
            // Second step, gather all required kpts across kptcomm
            for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) 
                gather_kptbridgecomm_kpt(pSPARC, Ns, psi_outer_kpt + spn_i * shift_spn_psi_outer_);
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
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm_kpt(SPARC_OBJ *pSPARC, int Ncol, double _Complex *vec) 
{
    if (pSPARC->blacscomm == MPI_COMM_NULL) return;

    int i, grank, blacs_rank, blacs_size, DMnd;
    int sendcount, *recvcounts, *displs, NB;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(pSPARC->blacscomm, &blacs_rank);
    MPI_Comm_size(pSPARC->blacscomm, &blacs_size);

    DMnd = pSPARC->Nd_d_dmcomm;

    if (blacs_size > 1) {
        recvcounts = (int*) malloc(sizeof(int)* blacs_size);
        displs = (int*) malloc(sizeof(int)* blacs_size);
        assert(recvcounts !=NULL && displs != NULL);

        // gather all bands, this part of code copied from parallelization.c
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
        displs[0] = 0;
        for (i = 0; i < blacs_size; i++){
            recvcounts[i] = (i < (Ncol / NB) ? NB : (i == (Ncol / NB) ? (Ncol % NB) : 0)) * DMnd;
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
void gather_kptbridgecomm_kpt(SPARC_OBJ *pSPARC, int Ncol, double _Complex *vec)
{
    if (pSPARC->kpt_bridge_comm == MPI_COMM_NULL) return;

    int kpt_bridge_rank, kpt_bridge_size;
    MPI_Comm_rank(pSPARC->kpt_bridge_comm, &kpt_bridge_rank);
    MPI_Comm_size(pSPARC->kpt_bridge_comm, &kpt_bridge_size);

    int i, DMnd, size_k, sendcount, *recvcounts, *displs;
    DMnd = pSPARC->Nd_d_dmcomm;
    size_k = DMnd * Ncol;
    
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
void transfer_orbitals_blacscomm_kpt(SPARC_OBJ *pSPARC, double complex *sendbuff, double complex *recvbuff, int shift, MPI_Request *reqs)
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

    MPI_Irecv(recvbuff, DMnd*Nband_recv, MPI_DOUBLE_COMPLEX, lneighbor, 111, blacscomm, &reqs[1]);
    MPI_Isend(sendbuff, DMnd*Nband_send, MPI_DOUBLE_COMPLEX, rneighbor, 111, blacscomm, &reqs[0]);
}


/**
 * @brief   transfer orbitals in a cyclic rotation way to save memory
 */
void transfer_orbitals_kptbridgecomm_kpt(SPARC_OBJ *pSPARC, 
    double complex *sendbuff, double complex *recvbuff, int shift, MPI_Request *reqs)
{
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
    
    MPI_Irecv(recvbuff, size_k*Nkpt_hf_recv, MPI_DOUBLE_COMPLEX, lneighbor, 111, kpt_bridge_comm, &reqs[1]);
    MPI_Isend(sendbuff, size_k*Nkpt_hf_send, MPI_DOUBLE_COMPLEX, rneighbor, 111, kpt_bridge_comm, &reqs[0]);

}

/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi_kpt(SPARC_OBJ *pSPARC, 
    int ncol, double complex *psi, double complex *psi_storage, double *occ, double complex *Xi_kpt, int kpt_q, int shift, int Ns_occ)
{
    MPI_Comm blacscomm = pSPARC->blacscomm;
    if (blacscomm == MPI_COMM_NULL) return;
    if (ncol == 0) return;
    int size, rank;
    MPI_Comm_size(blacscomm, &size);
    MPI_Comm_rank(blacscomm, &rank);

    int i, j, k, kpt_k, Ns, dims[3], DMnd;
    int num_rhs, count, loop, batch_num_rhs, NL, base, Nkpts_kptcomm, Nband, DMndNsocc;
    int *rhs_list_i, *rhs_list_j, *rhs_list_k, *kpt_k_list, *kpt_q_list;
    double occ_j;
    double _Complex *rhs, *Vi, *Xi_kpt_ki;
    /******************************************************************************/

    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    dims[0] = pSPARC->npNdx; dims[1] = pSPARC->npNdy; dims[2] = pSPARC->npNdz;
    DMndNsocc = DMnd * Ns_occ;
    Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    Nband = pSPARC->Nband_bandcomm;
    int size_k = DMnd * Nband;

    int source = (rank-shift+size)%size;
    int NB = (Ns - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    int Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
    int band_start_indx_source = source * NB;

    rhs_list_i = (int*) calloc(Nband_source * ncol * Nkpts_kptcomm, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nband_source * ncol * Nkpts_kptcomm, sizeof(int)); 
    rhs_list_k = (int*) calloc(Nband_source * ncol * Nkpts_kptcomm, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_k != NULL);

    count = 0;
    for (kpt_k = 0; kpt_k < Nkpts_kptcomm; kpt_k ++) {
        for (i = 0; i < ncol; i++) {
            for (j = 0; j < Nband_source; j++) {
                occ_j = occ[j+band_start_indx_source];
                if (occ_j < 1e-6) continue;
                rhs_list_i[count] = i;
                rhs_list_j[count] = j;
                rhs_list_k[count] = kpt_k;
                count ++;
            }
        }
    }

    num_rhs = count;
    if (num_rhs == 0) {
        free(rhs_list_i);
        free(rhs_list_j);
        free(rhs_list_k);
        return;
    }

    /* EXXMem_batch could be any positive integer to define the maximum number of 
    Poisson's equations to be solved at one time. The smaller the EXXMem_batch, 
    the less memory are required, but also the longer the running time. This part
    of code could be directly applied to NON-ACE part. */
    
    batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                        num_rhs : pSPARC->EXXMem_batch * pSPARC->npNd;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required

    rhs = (double _Complex *)calloc(sizeof(double _Complex) , DMnd * batch_num_rhs);                         // right hand sides of Poisson's equation
    Vi = (double _Complex *)calloc(sizeof(double _Complex) , DMnd * batch_num_rhs);                          // the solution for each rhs
    kpt_k_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of k vector 
    kpt_q_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of q vector 
    assert(rhs != NULL && Vi != NULL && kpt_k_list != NULL && kpt_q_list != NULL);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif
    for (i = 0; i < batch_num_rhs; i ++) kpt_q_list[i] = kpt_q;
    /*************** Solve all Poisson's equation and apply to psi ****************/
    for (loop = 0; loop < NL; loop ++) {
        base = batch_num_rhs*loop;
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];              // band indx of psi_outer with k vector
            j = rhs_list_j[count];              // band indx of psi_outer with q vector
            kpt_k = rhs_list_k[count];              // k-point indx of k vector
            // ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
            // ll_red = pSPARC->kpthf_ind_red[l];          // ll_red w.r.t. Nkpts_hf_red, for psi
            kpt_k_list[count-base] = kpt_k + pSPARC->kpt_start_indx;
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_storage[k + j*DMnd]) * psi[k + i*DMnd + kpt_k*size_k];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*DMnd] * psi[k + i*DMnd + kpt_k*size_k];
            }
        }
        
        poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const, count-base, DMnd, dims, Vi, kpt_k_list, kpt_q_list, pSPARC->dmcomm);    
        
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];              // band indx of psi_outer with k vector
            j = rhs_list_j[count];              // band indx of psi_outer with q vector
            kpt_k = rhs_list_k[count];              // k-point indx of k vector

            occ_j = occ[j+band_start_indx_source];
            Xi_kpt_ki = Xi_kpt + pSPARC->band_start_indx * DMnd + kpt_k*DMndNsocc;
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    Xi_kpt_ki[k + i*DMnd] -= pSPARC->kptWts_hf * occ_j * psi_storage[k + j*DMnd] * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            } else {
                for (k = 0; k < DMnd; k++) 
                    Xi_kpt_ki[k + i*DMnd] -= pSPARC->kptWts_hf * occ_j * conj(psi_storage[k + j*DMnd]) * Vi[k + (count-base)*DMnd] / pSPARC->dV;
            }
        }
    }

    free(rhs);
    free(Vi);
    free(kpt_k_list);
    free(kpt_q_list);

    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_k);

    #ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("rank = %2d, solving Poisson's equations took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif
}
