/***
 * @file    exactExchangeProperties.c
 * @brief   This file contains the functions for Exact Exchange properties.
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
#include "exactExchangeProperties.h"
#include "tools.h"
#include "lapVecRoutines.h"
#include "lapVecOrth.h"
#include "lapVecNonOrth.h"
#include "nlocVecRoutines.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "stress.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress(SPARC_OBJ *pSPARC) {
    if (pSPARC->isGammaPoint) {
        Calculate_exact_exchange_stress_linear(pSPARC);
    } else {
        Calculate_exact_exchange_stress_kpt(pSPARC);
    }
}

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_linear(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, rank, size, spn_i;
    int Ns, ncol, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, *rhs, *phi, *psi_outer, *psi_outer_, temp, *occ_outer, *occ_outer_, *psi;
    double *Drhs, *Dphi_1, *Dphi_2, stress_exx[7];
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    ncol = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;

    int xi_shift = DMnd * pSPARC->Nstates_occ[0] * pSPARC->Nkpts_kptcomm;
    int psi_outer_shift = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
    int psi_shift = DMnd * pSPARC->Nband_bandcomm * pSPARC->Nkpts_kptcomm;
    int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;
    /********************************************************************/
    for (i = 0; i < 7; i++) stress_exx[i] = 0.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;

    int *rhs_list_i, *rhs_list_j;
    rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
    rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    psi_outer_ = (pSPARC->ACEFlag == 1) ? pSPARC->Xorb : pSPARC->psi_outer;
    occ_outer_ = (pSPARC->ACEFlag == 1) ? pSPARC->occ : pSPARC->occ_outer;
    mflag = pSPARC->EXXDiv_Flag;

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        psi_outer = psi_outer_ + spn_i * psi_outer_shift;
        occ_outer = occ_outer_ + spn_i * occ_outer_shift;
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
        phi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && phi != NULL);

        // space for gradients of rhs
        Drhs = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);
        assert(Drhs != NULL);

        // space for gradients of phi
        Dphi_1 = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);
        Dphi_2 = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);
        assert(Dphi_1 != NULL && Dphi_2 != NULL);

        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd] * psi[k + i*DMnd];
                }
            }

            // Solve all Poisson's equation 
            poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const_stress, count-base, DMnd, dims, phi, comm);
            
            // component (1,1)
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, rhs, Drhs, 0, comm);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, phi, Dphi_1, 0, comm);
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    // Drhs saves grad_x of rho, Dphi_1 saves grad_x of phi
                    stress_exx[0] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_1[k + (count-base)*DMnd];
                }
            }

            // component (1,2)
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, phi, Dphi_1, 1, comm);
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    // Drhs saves grad_x of rhs, Dphi_1 saves grad_y of phi
                    stress_exx[1] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_1[k + (count-base)*DMnd];
                }
            }

            // component (1,3)
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, phi, Dphi_2, 2, comm);
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    // Drhs saves grad_x of rhs, Dphi_2 saves grad_z of phi
                    stress_exx[2] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_2[k + (count-base)*DMnd];
                }
            }

            // component (2,2)
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, rhs, Drhs, 1, comm);
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    // Drhs saves grad_y of rhs, Dphi_1 saves grad_y of phi
                    stress_exx[3] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_1[k + (count-base)*DMnd];
                }
            }

            // component (2,3)
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    // Drhs saves grad_y of rhs, Dphi_2 saves grad_z of phi
                    stress_exx[4] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_2[k + (count-base)*DMnd];
                }
            }

            // component (3,3)
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, rhs, Drhs, 2, comm);
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    // Drhs saves grad_z of rhs, Dphi_2 saves grad_z of phi
                    stress_exx[5] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_2[k + (count-base)*DMnd];
                }
            }

            // additional term for spherical truncation
            if (mflag == 0) {
                poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const_stress2, count-base, DMnd, dims, phi, comm);
                for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    
                    occ_i = occ_outer[i + pSPARC->band_start_indx];
                    occ_j = occ_outer[j];

                    for (k = 0; k < DMnd; k++){
                        stress_exx[6] += occ_i * occ_j * rhs[k + (count-base)*DMnd] * phi[k + (count-base)*DMnd];
                    }
                }
            }
        }
        free(rhs);
        free(phi);
        free(Drhs);
        free(Dphi_1);
        free(Dphi_2);
    }
    free(rhs_list_i);
    free(rhs_list_j);

    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7,  MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }
    
    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    // convert to cartesian coordinates
    if (pSPARC->cell_typ == 0) {
        pSPARC->stress_exx[0] = stress_exx[0];
        pSPARC->stress_exx[1] = stress_exx[1];
        pSPARC->stress_exx[2] = stress_exx[2];
        pSPARC->stress_exx[3] = stress_exx[3];
        pSPARC->stress_exx[4] = stress_exx[4];
        pSPARC->stress_exx[5] = stress_exx[5];
    } else {
        double *c_g;
        c_g = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
            }
        }
        pSPARC->stress_exx[0] = c_g[0] * (c_g[0] * stress_exx[0] + c_g[1] * stress_exx[1] + c_g[2] * stress_exx[2]) +
                                c_g[1] * (c_g[0] * stress_exx[1] + c_g[1] * stress_exx[3] + c_g[2] * stress_exx[4]) +
                                c_g[2] * (c_g[0] * stress_exx[2] + c_g[1] * stress_exx[4] + c_g[2] * stress_exx[5]);
        pSPARC->stress_exx[1] = c_g[0] * (c_g[3] * stress_exx[0] + c_g[4] * stress_exx[1] + c_g[5] * stress_exx[2]) +
                                c_g[1] * (c_g[3] * stress_exx[1] + c_g[4] * stress_exx[3] + c_g[5] * stress_exx[4]) +
                                c_g[2] * (c_g[3] * stress_exx[2] + c_g[4] * stress_exx[4] + c_g[5] * stress_exx[5]);
        pSPARC->stress_exx[2] = c_g[0] * (c_g[6] * stress_exx[0] + c_g[7] * stress_exx[1] + c_g[8] * stress_exx[2]) +
                                c_g[1] * (c_g[6] * stress_exx[1] + c_g[7] * stress_exx[3] + c_g[8] * stress_exx[4]) +
                                c_g[2] * (c_g[6] * stress_exx[2] + c_g[7] * stress_exx[4] + c_g[8] * stress_exx[5]);
        pSPARC->stress_exx[3] = c_g[3] * (c_g[3] * stress_exx[0] + c_g[4] * stress_exx[1] + c_g[5] * stress_exx[2]) +
                                c_g[4] * (c_g[3] * stress_exx[1] + c_g[4] * stress_exx[3] + c_g[5] * stress_exx[4]) +
                                c_g[5] * (c_g[3] * stress_exx[2] + c_g[4] * stress_exx[4] + c_g[5] * stress_exx[5]);
        pSPARC->stress_exx[4] = c_g[3] * (c_g[6] * stress_exx[0] + c_g[7] * stress_exx[1] + c_g[8] * stress_exx[2]) +
                                c_g[4] * (c_g[6] * stress_exx[1] + c_g[7] * stress_exx[3] + c_g[8] * stress_exx[4]) +
                                c_g[5] * (c_g[6] * stress_exx[2] + c_g[7] * stress_exx[4] + c_g[8] * stress_exx[5]);
        pSPARC->stress_exx[5] = c_g[6] * (c_g[6] * stress_exx[0] + c_g[7] * stress_exx[1] + c_g[8] * stress_exx[2]) +
                                c_g[7] * (c_g[6] * stress_exx[1] + c_g[7] * stress_exx[3] + c_g[8] * stress_exx[4]) +
                                c_g[8] * (c_g[6] * stress_exx[2] + c_g[7] * stress_exx[4] + c_g[8] * stress_exx[5]);
        free(c_g);
    }

    if (mflag == 0) {
        pSPARC->stress_exx[0] += stress_exx[6];
        pSPARC->stress_exx[3] += stress_exx[6];
        pSPARC->stress_exx[5] += stress_exx[6];
    }

    for (i = 0; i < 6; i++) 
        pSPARC->stress_exx[i] *= (-pSPARC->hyb_mixing/pSPARC->dV/pSPARC->Nspin);

    pSPARC->stress_exx[0] = 2*pSPARC->stress_exx[0] - 2*pSPARC->Eexx + (mflag == 1)*pSPARC->Eexx/2;
    pSPARC->stress_exx[1] = 2*pSPARC->stress_exx[1];
    pSPARC->stress_exx[2] = 2*pSPARC->stress_exx[2];
    pSPARC->stress_exx[3] = 2*pSPARC->stress_exx[3] - 2*pSPARC->Eexx + (mflag == 1)*pSPARC->Eexx/2;
    pSPARC->stress_exx[4] = 2*pSPARC->stress_exx[4];
    pSPARC->stress_exx[5] = 2*pSPARC->stress_exx[5] - 2*pSPARC->Eexx + (mflag == 1)*pSPARC->Eexx/2;

    if (!grank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

        for(int i = 0; i < 6; i++) {
            pSPARC->stress_exx[i] /= cell_measure;            
        }
    }
    
#ifdef DEBUG
if(!grank) {
    printf("\nExact exchange contribution to stress");
    PrintStress(pSPARC, pSPARC->stress_exx, NULL);  
}
#endif  
}

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, ll_red, m, grank, rank, size, spn_i, n;
    int Ns, Ns_loc, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, *occ_outer, stress_exx[7], kpt_vec;
    double _Complex *rhs, *psi_outer, *psi, *phi;
    double _Complex *Drhs,*Dphi_1, *Dphi_2;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_loc = pSPARC->Nkpts_kptcomm;
    Ns_loc = Nband * Nkpts_loc;         // for Xorb
    comm = pSPARC->dmcomm;

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
    for (i = 0; i < 7; i++) stress_exx[i] = 0.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;
    mflag = pSPARC->EXXDiv_Flag;
    
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
            phi = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                            // the solution for each rhs
            kpt_k_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of k vector 
            kpt_q_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of q vector 
            assert(rhs != NULL && phi != NULL && kpt_k_list != NULL && kpt_q_list != NULL);

            // space for gradients of rhs
            Drhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);

            // space for gradients of phi
            Dphi_1 = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);
            Dphi_2 = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);
            assert(Dphi_1 != NULL && Dphi_2 != NULL);

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
                poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const_stress, count-base, DMnd, dims, phi, kpt_k_list, kpt_q_list, comm);

                // component (1,1)
                for (n = 0; n < count-base; n++) {
                    kpt_vec = pSPARC->k1[kpt_k_list[n]] - pSPARC->k1_hf[kpt_q_list[n]];
                    // calculate gradients of rhs
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, rhs+n*DMnd, Drhs+n*DMnd, 0, kpt_vec, comm);
                    // calculate gradients of phi
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, phi+n*DMnd, Dphi_1+n*DMnd, 0, kpt_vec, comm);
                }

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        // Drhs saves grad_x of rhs, Dphi_1 saves grad_x of phi
                        stress_exx[0] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_1[k + (count-base)*DMnd]);
                    }
                }
                
                // component (1,2)
                for (n = 0; n < count-base; n++) {
                    kpt_vec = pSPARC->k2[kpt_k_list[n]] - pSPARC->k2_hf[kpt_q_list[n]];
                    // calculate gradients of phi
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, phi+n*DMnd, Dphi_1+n*DMnd, 1, kpt_vec, comm);
                }

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        // Drhs saves grad_x of rhs, Dphi_1 saves grad_y of phi
                        stress_exx[1] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_1[k + (count-base)*DMnd]);
                    }
                }

                // component (1,3)
                for (n = 0; n < count-base; n++) {
                    kpt_vec = pSPARC->k3[kpt_k_list[n]] - pSPARC->k3_hf[kpt_q_list[n]];
                    // calculate gradients of phi
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, phi+n*DMnd, Dphi_2+n*DMnd, 2, kpt_vec, comm);
                }

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        // Drhs saves grad_x of rhs, Dphi_2 saves grad_z of phi
                        stress_exx[2] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
                    }
                }

                // component (2,2)
                for (n = 0; n < count-base; n++) {
                    kpt_vec = pSPARC->k2[kpt_k_list[n]] - pSPARC->k2_hf[kpt_q_list[n]];
                    // calculate gradients of rhs
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, rhs+n*DMnd, Drhs+n*DMnd, 1, kpt_vec, comm);
                }

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        // Drhs saves grad_y of rhs, Dphi_1 saves grad_y of phi
                        stress_exx[3] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_1[k + (count-base)*DMnd]);
                    }
                }

                // component (2,3)
                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        // Drhs saves grad_y of rhs, Dphi_2 saves grad_z of phi
                        stress_exx[4] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
                    }
                }

                // component (3,3)
                for (n = 0; n < count-base; n++) {
                    kpt_vec = pSPARC->k3[kpt_k_list[n]] - pSPARC->k3_hf[kpt_q_list[n]];
                    // calculate gradients of rhs
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, rhs+n*DMnd, Drhs+n*DMnd, 2, kpt_vec, comm);
                }

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        // Drhs saves grad_z of rhs, Dphi_2 saves grad_y of phi
                        stress_exx[5] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
                    }
                }

                // additional term for spherical truncation
                if (mflag == 0) {
                    poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const_stress2, count-base, DMnd, dims, phi, kpt_k_list, kpt_q_list, comm);
                    
                    for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                        i = rhs_list_i[count];
                        j = rhs_list_j[count];
                        l = rhs_list_l[count];
                        m = rhs_list_m[count];
                        ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                        
                        occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                        occ_j = occ_outer[j + ll * Ns];
                        
                        for (k = 0; k < DMnd; k++){
                            stress_exx[6] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                            creal(conj(rhs[k + (count-base)*DMnd]) * phi[k + (count-base)*DMnd]);
                        }
                    }
                }
            }

            free(rhs);
            free(phi);
            free(kpt_k_list);
            free(kpt_q_list);
            free(Drhs);
            free(Dphi_1);
            free(Dphi_2);
        }
    }
    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_l);
    free(rhs_list_m);

    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7,  MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->npkpt > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_exx, 7, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    // convert to cartesian coordinates
    if (pSPARC->cell_typ == 0) {
        pSPARC->stress_exx[0] = stress_exx[0];
        pSPARC->stress_exx[1] = stress_exx[1];
        pSPARC->stress_exx[2] = stress_exx[2];
        pSPARC->stress_exx[3] = stress_exx[3];
        pSPARC->stress_exx[4] = stress_exx[4];
        pSPARC->stress_exx[5] = stress_exx[5];
    } else {
        double *c_g;
        c_g = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
            }
        }
        pSPARC->stress_exx[0] = c_g[0] * (c_g[0] * stress_exx[0] + c_g[1] * stress_exx[1] + c_g[2] * stress_exx[2]) +
                                c_g[1] * (c_g[0] * stress_exx[1] + c_g[1] * stress_exx[3] + c_g[2] * stress_exx[4]) +
                                c_g[2] * (c_g[0] * stress_exx[2] + c_g[1] * stress_exx[4] + c_g[2] * stress_exx[5]);
        pSPARC->stress_exx[1] = c_g[0] * (c_g[3] * stress_exx[0] + c_g[4] * stress_exx[1] + c_g[5] * stress_exx[2]) +
                                c_g[1] * (c_g[3] * stress_exx[1] + c_g[4] * stress_exx[3] + c_g[5] * stress_exx[4]) +
                                c_g[2] * (c_g[3] * stress_exx[2] + c_g[4] * stress_exx[4] + c_g[5] * stress_exx[5]);
        pSPARC->stress_exx[2] = c_g[0] * (c_g[6] * stress_exx[0] + c_g[7] * stress_exx[1] + c_g[8] * stress_exx[2]) +
                                c_g[1] * (c_g[6] * stress_exx[1] + c_g[7] * stress_exx[3] + c_g[8] * stress_exx[4]) +
                                c_g[2] * (c_g[6] * stress_exx[2] + c_g[7] * stress_exx[4] + c_g[8] * stress_exx[5]);
        pSPARC->stress_exx[3] = c_g[3] * (c_g[3] * stress_exx[0] + c_g[4] * stress_exx[1] + c_g[5] * stress_exx[2]) +
                                c_g[4] * (c_g[3] * stress_exx[1] + c_g[4] * stress_exx[3] + c_g[5] * stress_exx[4]) +
                                c_g[5] * (c_g[3] * stress_exx[2] + c_g[4] * stress_exx[4] + c_g[5] * stress_exx[5]);
        pSPARC->stress_exx[4] = c_g[3] * (c_g[6] * stress_exx[0] + c_g[7] * stress_exx[1] + c_g[8] * stress_exx[2]) +
                                c_g[4] * (c_g[6] * stress_exx[1] + c_g[7] * stress_exx[3] + c_g[8] * stress_exx[4]) +
                                c_g[5] * (c_g[6] * stress_exx[2] + c_g[7] * stress_exx[4] + c_g[8] * stress_exx[5]);
        pSPARC->stress_exx[5] = c_g[6] * (c_g[6] * stress_exx[0] + c_g[7] * stress_exx[1] + c_g[8] * stress_exx[2]) +
                                c_g[7] * (c_g[6] * stress_exx[1] + c_g[7] * stress_exx[3] + c_g[8] * stress_exx[4]) +
                                c_g[8] * (c_g[6] * stress_exx[2] + c_g[7] * stress_exx[4] + c_g[8] * stress_exx[5]);
        free(c_g);
    }

    if (mflag == 0) {
        pSPARC->stress_exx[0] += stress_exx[6];
        pSPARC->stress_exx[3] += stress_exx[6];
        pSPARC->stress_exx[5] += stress_exx[6];
    }

    for (i = 0; i < 6; i++) 
        pSPARC->stress_exx[i] *= (-pSPARC->hyb_mixing/pSPARC->dV/pSPARC->Nspin);

    pSPARC->stress_exx[0] = 2*pSPARC->stress_exx[0] - 2*pSPARC->Eexx + (mflag == 1)*pSPARC->Eexx/2;
    pSPARC->stress_exx[1] = 2*pSPARC->stress_exx[1];
    pSPARC->stress_exx[2] = 2*pSPARC->stress_exx[2];
    pSPARC->stress_exx[3] = 2*pSPARC->stress_exx[3] - 2*pSPARC->Eexx + (mflag == 1)*pSPARC->Eexx/2;
    pSPARC->stress_exx[4] = 2*pSPARC->stress_exx[4];
    pSPARC->stress_exx[5] = 2*pSPARC->stress_exx[5] - 2*pSPARC->Eexx + (mflag == 1)*pSPARC->Eexx/2;

    if (!grank) {
        // Define measure of unit cell
        double cell_measure = pSPARC->Jacbdet;
        if(pSPARC->BCx == 0)
            cell_measure *= pSPARC->range_x;
        if(pSPARC->BCy == 0)
            cell_measure *= pSPARC->range_y;
        if(pSPARC->BCz == 0)
            cell_measure *= pSPARC->range_z;

        for(int i = 0; i < 6; i++) {
            pSPARC->stress_exx[i] /= cell_measure;            
        }
    }

#ifdef DEBUG
if(!grank) {
    printf("\nExact exchange contribution to stress");
    PrintStress(pSPARC, pSPARC->stress_exx, NULL);  
}
#endif  
}



/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure(SPARC_OBJ *pSPARC) {
    if (pSPARC->isGammaPoint) {
        Calculate_exact_exchange_pressure_linear(pSPARC);
    } else {
        Calculate_exact_exchange_pressure_kpt(pSPARC);
    }
}

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_linear(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, rank, size, spn_i;
    int Ns, ncol, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, *rhs, *phi, *psi_outer, *psi_outer_, temp, *occ_outer, *occ_outer_, *psi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    ncol = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int xi_shift = DMnd * pSPARC->Nstates_occ[0] * pSPARC->Nkpts_kptcomm;
    int psi_outer_shift = DMnd * pSPARC->Nstates * pSPARC->Nkpts_hf_red;
    int psi_shift = DMnd * pSPARC->Nband_bandcomm * pSPARC->Nkpts_kptcomm;
    int occ_outer_shift = pSPARC->Nstates * pSPARC->Nkpts_sym;
    /********************************************************************/
    double pres_exx = 0.0;
    mflag = pSPARC->EXXDiv_Flag;
    
    if (mflag == 0) {
        pSPARC->pres_exx = 0;
#ifdef DEBUG    
    if (!grank){
        printf("Uncounted pressure contribution from exact exchange: = %.15f Ha\n", pSPARC->pres_exx);
    }
#endif
        return;
    }

    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;

    int *rhs_list_i, *rhs_list_j;
    rhs_list_i = (int*) calloc(ncol * Ns, sizeof(int)); 
    rhs_list_j = (int*) calloc(ncol * Ns, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    psi_outer_ = (pSPARC->ACEFlag == 1) ? pSPARC->Xorb : pSPARC->psi_outer;
    occ_outer_ = (pSPARC->ACEFlag == 1) ? pSPARC->occ : pSPARC->occ_outer;

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        psi_outer = psi_outer_ + spn_i * psi_outer_shift;
        occ_outer = occ_outer_ + spn_i * occ_outer_shift;
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
        phi = (double *)malloc(sizeof(double) * DMnd * batch_num_rhs);                          // the solution for each rhs
        assert(rhs != NULL && phi != NULL);

        for (loop = 0; loop < NL; loop ++) {
            base = batch_num_rhs*loop;
            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                for (k = 0; k < DMnd; k++) {
                    rhs[k + (count-base)*DMnd] = psi_outer[k + j*DMnd] * psi[k + i*DMnd];
                }
            }

            // Solve all Poisson's equation 
            poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const_press, count-base, DMnd, dims, phi, comm);

            for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                
                occ_i = occ_outer[i + pSPARC->band_start_indx];
                occ_j = occ_outer[j];

                for (k = 0; k < DMnd; k++){
                    pres_exx += occ_i * occ_j * rhs[k + (count-base)*DMnd] * phi[k + (count-base)*DMnd];
                }
            }
        }
        free(rhs);
        free(phi);
    }
    free(rhs_list_i);
    free(rhs_list_j);

    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }
    
    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }
    
    pres_exx *= (-pSPARC->hyb_mixing/pSPARC->dV/pSPARC->Nspin);
    
    if (mflag == 1)
        pres_exx += pSPARC->Eexx*3/4;
    
    pSPARC->pres_exx = 2*pres_exx - 2*pSPARC->Eexx;

#ifdef DEBUG    
    if (!grank){
        printf("Uncounted pressure contribution from exact exchange: = %.15f Ha\n", pSPARC->pres_exx);
    }    
#endif
}

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, ll_red, m, grank, rank, size, spn_i, n;
    int Ns, Ns_loc, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, *occ_outer, kpt_vec, pres_exx;
    double _Complex *rhs, *psi_outer, *psi, *phi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_loc = pSPARC->Nkpts_kptcomm;
    Ns_loc = Nband * Nkpts_loc;         // for Xorb
    comm = pSPARC->dmcomm;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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
    mflag = pSPARC->EXXDiv_Flag;
    if (mflag == 0) {
        pSPARC->pres_exx = 0;
#ifdef DEBUG    
    if (!grank){
        printf("Uncounted pressure contribution from exact exchange: = %.15f Ha\n", pSPARC->pres_exx);
    }
#endif
        return;
    }

    pres_exx = 0;
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
            phi = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                            // the solution for each rhs
            kpt_k_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of k vector 
            kpt_q_list = (int *) calloc (sizeof(int), batch_num_rhs);                                                   // list of q vector 
            assert(rhs != NULL && phi != NULL && kpt_k_list != NULL && kpt_q_list != NULL);

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
                poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const_press, count-base, DMnd, dims, phi, kpt_k_list, kpt_q_list, comm);

                for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                    i = rhs_list_i[count];
                    j = rhs_list_j[count];
                    l = rhs_list_l[count];
                    m = rhs_list_m[count];
                    ll = pSPARC->kpthf_ind[l];                  // ll w.r.t. Nkpts_sym, for occ
                    
                    occ_i = occ_outer[pSPARC->band_start_indx + i + (m + pSPARC->kpt_start_indx) * Ns];
                    occ_j = occ_outer[j + ll * Ns];
                    
                    for (k = 0; k < DMnd; k++){
                        pres_exx += pSPARC->kptWts_hf * pSPARC->kptWts_loc[m] / pSPARC->Nkpts * occ_i * occ_j * 
                                         creal(conj(rhs[k + (count-base)*DMnd]) * phi[k + (count-base)*DMnd]);
                    }
                }
            }

            free(rhs);
            free(phi);            
            free(kpt_k_list);
            free(kpt_q_list);
        }
    }
    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_l);
    free(rhs_list_m);

    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1,  MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->npkpt > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

    if (pSPARC->npspin > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &pres_exx, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }

    pres_exx *= (-pSPARC->hyb_mixing/pSPARC->dV/pSPARC->Nspin);

    if (mflag == 1)
        pres_exx += pSPARC->Eexx*3/4;
    
    pSPARC->pres_exx = 2*pres_exx - 2*pSPARC->Eexx;

#ifdef DEBUG    
    if (!grank){
        printf("Uncounted pressure contribution from exact exchange: = %.15f Ha\n", pSPARC->pres_exx);
    }
#endif
}