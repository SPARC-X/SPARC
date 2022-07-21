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
    int i, j, grank, rank, size, spn_i;
    int Ns, Nband, DMnd, psi_shift, occ_outer_shift, psi_outer_shift, mflag;
    double *psi_outer, *occ_outer, *psi, stress_exx[7];
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;
    mflag = pSPARC->EXXDiv_Flag;
    /********************************************************************/
    for (i = 0; i < 7; i++) stress_exx[i] = 0.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        if (pSPARC->ACEFlag == 0) {
            psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
            occ_outer_shift = Ns * pSPARC->Nkpts_sym;
            psi_outer_shift = DMnd * pSPARC->Nkpts_hf_red * Ns;
            psi = pSPARC->Xorb + spn_i * psi_shift;
            psi_outer = pSPARC->psi_outer + spn_i * psi_outer_shift;
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            Calculate_exact_exchange_stress_linear_nonACE(pSPARC, psi_outer, psi, occ_outer, stress_exx);
        } else {
            psi = pSPARC->Xorb + spn_i * DMnd * Nband;
            occ_outer = pSPARC->occ + spn_i * Ns;
            Calculate_exact_exchange_stress_linear_ACE(pSPARC, psi, occ_outer, stress_exx);
        }
    }

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
void Calculate_exact_exchange_stress_linear_nonACE(SPARC_OBJ *pSPARC, 
        double *psi_outer, double *psi, double *occ_outer, double *stress_exx)
{
    solve_allpair_poissons_equation_stress(pSPARC, 
        psi_outer, psi, occ_outer, pSPARC->Nstates, 0, stress_exx);
}


/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_linear_ACE(SPARC_OBJ *pSPARC, 
        double *psi, double *occ, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int grank, DMnd, reps, NB, Ns, source, Nband_source, band_start_indx_source;
    double *psi_storage1, *psi_storage2, *sendbuff, *recvbuff;
    MPI_Request reqs[2];

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    DMnd = pSPARC->Nd_d_dmcomm;
    reps = pSPARC->npband - 1;
    NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
    Ns = pSPARC->Nstates;

    MPI_Comm blacscomm = pSPARC->blacscomm;
    int rank, size;
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_size(blacscomm, &size);
    /********************************************************************/
    
    if (reps > 0) {
        psi_storage1 = (double *) calloc(sizeof(double), NB*DMnd);
        psi_storage2 = (double *) calloc(sizeof(double), NB*DMnd);
    }

    for (int rep = 0; rep <= reps; rep ++) {
        source = (rank-rep+size)%size;
        Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
        band_start_indx_source = source * NB;
        if (rep == 0) {
            if (reps > 0) {
                transfer_orbitals_blacscomm(pSPARC, psi, psi_storage1, rep, reqs);
            }
            sendbuff = psi;
        } else {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            // first gather the orbitals in the rotation way
            sendbuff = (rep%2==1) ? psi_storage1 : psi_storage2;
            recvbuff = (rep%2==1) ? psi_storage2 : psi_storage1;
            if (rep != reps) {
                transfer_orbitals_blacscomm(pSPARC, sendbuff, recvbuff, rep, reqs);
            }
        }
        solve_allpair_poissons_equation_stress(pSPARC, sendbuff, psi, occ, 
            Nband_source, band_start_indx_source, stress_exx);
    }

    if (reps > 0) {
        free(psi_storage1);
        free(psi_storage2);
    }
}

/**
 * @brief   Calculate Exact Exchange stress
 */
void solve_allpair_poissons_equation_stress(SPARC_OBJ *pSPARC, 
    double *psi_storage, double *psi, double *occ, int Nband_source, 
    int band_start_indx_source, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    MPI_Comm blacscomm = pSPARC->blacscomm;
    MPI_Comm comm = pSPARC->dmcomm;
    int i, j, k, grank, rank, size;
    int Ns, Nband, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag, band_start_indx;
    double occ_i, occ_j, *rhs, *phi, temp, *Drhs, *Dphi_1, *Dphi_2;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_size(blacscomm, &size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    mflag = pSPARC->EXXDiv_Flag;
    band_start_indx = pSPARC->band_start_indx;
    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;
    /********************************************************************/

    int *rhs_list_i, *rhs_list_j;
    rhs_list_i = (int*) calloc(Nband * Nband_source, sizeof(int));
    rhs_list_j = (int*) calloc(Nband * Nband_source, sizeof(int));
    assert(rhs_list_i != NULL && rhs_list_j != NULL);
    
    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (i = 0; i < Nband; i++) {
        for (j = 0; j < Nband_source; j++) {
            if (occ[i + band_start_indx]> 1e-6 && occ[j + band_start_indx_source] > 1e-6) {
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
                    num_rhs : pSPARC->EXXMem_batch * pSPARC->npNd;

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
                rhs[k + (count-base)*DMnd] = psi_storage[k + j*DMnd] * psi[k + i*DMnd];
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
            
            occ_i = occ[i + band_start_indx];
            occ_j = occ[j + band_start_indx_source];

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
            
            occ_i = occ[i + band_start_indx];
            occ_j = occ[j + band_start_indx_source];

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
            
            occ_i = occ[i + band_start_indx];
            occ_j = occ[j + band_start_indx_source];

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
            
            occ_i = occ[i + band_start_indx];
            occ_j = occ[j + band_start_indx_source];

            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_y of rhs, Dphi_1 saves grad_y of phi
                stress_exx[3] += occ_i * occ_j * Drhs[k + (count-base)*DMnd] * Dphi_1[k + (count-base)*DMnd];
            }
        }

        // component (2,3)
        for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            
            occ_i = occ[i + band_start_indx];
            occ_j = occ[j + band_start_indx_source];

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
            
            occ_i = occ[i + band_start_indx];
            occ_j = occ[j + band_start_indx_source];

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
                
                occ_i = occ[i + band_start_indx];
                occ_j = occ[j + band_start_indx_source];

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
    free(rhs_list_i);
    free(rhs_list_j);
}


/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, grank, rank, size, spn_i;
    int Ns, Nband, DMnd, mflag;
    double *occ_outer, stress_exx[7], kpt_vec;
    double _Complex *psi_outer, *psi;
    MPI_Comm comm = pSPARC->dmcomm;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;    
    mflag = pSPARC->EXXDiv_Flag;
    /********************************************************************/
    for (i = 0; i < 7; i++) stress_exx[i] = 0.0;

    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        if (pSPARC->ACEFlag == 0) {
            int psi_outer_shift = DMnd * Ns * pSPARC->Nkpts_hf_red;
            int psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
            int occ_outer_shift = Ns * pSPARC->Nkpts_sym;
            
            psi_outer = pSPARC->psi_outer_kpt + spn_i * psi_outer_shift;
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;

            Calculate_exact_exchange_stress_kpt_nonACE(pSPARC, psi_outer, occ_outer, psi, stress_exx);
        } else {
            int psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
            int occ_outer_shift = Ns * pSPARC->Nkpts_sym;
            
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;

            Calculate_exact_exchange_stress_kpt_ACE(pSPARC, psi, occ_outer, stress_exx);
        }
    }

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
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt_nonACE(SPARC_OBJ *pSPARC, 
    double complex *psi_outer, double *occ_outer, double complex *psi, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int k, grank, kpt_q;
    int Ns, DMnd, size_k_, count;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    size_k_ = Ns * DMnd;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    
    for (k = 0; k < pSPARC->Nkpts_hf_red; k++) {
        int counts = pSPARC->kpthfred2kpthf[k][0];
        for (count = 0; count < counts; count++) {
            kpt_q = pSPARC->kpthfred2kpthf[k][count+1];
            // solve poisson's equations 
            solve_allpair_poissons_equation_stress_kpt(pSPARC, psi_outer + k*size_k_, 
                psi, occ_outer, Ns, 0, kpt_q, stress_exx);
        }
    }
}


/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt_ACE(SPARC_OBJ *pSPARC, 
    double complex *psi, double *occ_outer, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, kpt_q;
    int Ns, DMnd, Nband, NB, size_k, count;
    int source, Nband_source, band_start_indx_source;
    double complex *psi_storage1_kpt, *psi_storage2_kpt, *psi_storage1_band, *psi_storage2_band;
    double complex *sendbuff_kpt, *recvbuff_kpt, *sendbuff_band, *recvbuff_band;
    MPI_Comm blacscomm = pSPARC->blacscomm;
    MPI_Request reqs_kpt[2], reqs_band[2];

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    size_k = Nband * DMnd;
    NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
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
    for (k = 0; k < pSPARC->npkpt; k++) 
        Nkpthf_red_max = max(Nkpthf_red_max, pSPARC->Nkpts_hf_list[k]);

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
                transfer_orbitals_kptbridgecomm_kpt(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt);
            }
        } else {
            MPI_Waitall(2, reqs_kpt, MPI_STATUSES_IGNORE);
            sendbuff_kpt = (rep_kpt%2==1) ? psi_storage2_kpt : psi_storage1_kpt;
            recvbuff_kpt = (rep_kpt%2==1) ? psi_storage1_kpt : psi_storage2_kpt;
            if (rep_kpt != reps_kpt) {
                transfer_orbitals_kptbridgecomm_kpt(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt);
            }
        }

        int source_kpt = (kpt_bridge_comm_rank-rep_kpt+kpt_bridge_comm_size)%kpt_bridge_comm_size;
        int nkpthf_red = pSPARC->Nkpts_hf_list[source_kpt];
        int kpthf_start_indx = pSPARC->kpthf_start_indx_list[source_kpt];
        
        for (k = 0; k < nkpthf_red; k++) {
            int k_indx = k + kpthf_start_indx;
            int counts = pSPARC->kpthfred2kpthf[k_indx][0];

            for (int rep_band = 0; rep_band <= reps_band; rep_band++) {
                // transfer the orbitals in the rotation way across blacscomm
                if (rep_band == 0) {
                    sendbuff_band = sendbuff_kpt + k*size_k;
                    if (reps_band > 0) {
                        recvbuff_band = psi_storage1_band;
                        transfer_orbitals_blacscomm_kpt(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band);
                    }
                } else {                    
                    MPI_Waitall(2, reqs_band, MPI_STATUSES_IGNORE);
                    sendbuff_band = (rep_band%2==1) ? psi_storage1_band : psi_storage2_band;
                    recvbuff_band = (rep_band%2==1) ? psi_storage2_band : psi_storage1_band;
                    if (rep_band != reps_band) {
                        transfer_orbitals_blacscomm_kpt(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band);
                    }
                }
                
                source = (blacscomm_rank-rep_band+blacscomm_size)%blacscomm_size;
                Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
                band_start_indx_source = source * NB;

                for (count = 0; count < counts; count++) {
                    kpt_q = pSPARC->kpthfred2kpthf[k_indx][count+1];
                    // solve poisson's equations 
                    solve_allpair_poissons_equation_stress_kpt(pSPARC, sendbuff_band, 
                        psi, occ_outer, Nband_source, band_start_indx_source, kpt_q, stress_exx);
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
}

/**
 * @brief   Calculate Exact Exchange stress
 */
void solve_allpair_poissons_equation_stress_kpt(SPARC_OBJ *pSPARC, double complex *psi_storage, 
    double complex *psi, double *occ, int Nband_source, int band_start_indx_source, int kpt_q, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, grank, rank, size, n, kpt_k;
    int Ns, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, kpt_vec;
    double _Complex *rhs, *phi, *Drhs,*Dphi_1, *Dphi_2;
    MPI_Comm comm = pSPARC->dmcomm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
    int size_k = DMnd * Nband;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;
    mflag = pSPARC->EXXDiv_Flag;
    
    int *rhs_list_i, *rhs_list_j, *rhs_list_k, *kpt_k_list, *kpt_q_list;
    rhs_list_i = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    rhs_list_k = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_k != NULL);
    
    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (kpt_k = 0; kpt_k < Nkpts_kptcomm; kpt_k ++) {
        for (i = 0; i < Nband; i++) {
            for (j = 0; j < Nband_source; j++) {
                occ_j = occ[j + band_start_indx_source + ll * Ns];
                occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
                
                if (occ_j > 1e-6 && occ_i > 1e-6) {
                    rhs_list_i[count] = i;
                    rhs_list_j[count] = j;
                    rhs_list_k[count] = kpt_k;
                    count ++;
                }
            }
        }
    }
    num_rhs = count;
    
    if (!num_rhs) {
        free(rhs_list_i);
        free(rhs_list_j);
        free(rhs_list_k);
        return;
    }
    batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                    num_rhs : pSPARC->EXXMem_batch * pSPARC->npNd;
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
            kpt_k = rhs_list_k[count];                  // k-point indx of k vector
            kpt_k_list[count-base] = kpt_k + pSPARC->kpt_start_indx;
            kpt_q_list[count-base] = kpt_q;
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_storage[k + j*DMnd]) * psi[k + i*DMnd + kpt_k*size_k];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*DMnd] * psi[k + i*DMnd + kpt_k*size_k];
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
            kpt_k = rhs_list_k[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_x of rhs, Dphi_1 saves grad_x of phi
                stress_exx[0] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
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
            kpt_k = rhs_list_k[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_x of rhs, Dphi_1 saves grad_y of phi
                stress_exx[1] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
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
            kpt_k = rhs_list_k[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_x of rhs, Dphi_2 saves grad_z of phi
                stress_exx[2] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
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
            kpt_k = rhs_list_k[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_y of rhs, Dphi_1 saves grad_y of phi
                stress_exx[3] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_1[k + (count-base)*DMnd]);
            }
        }

        // component (2,3)
        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            kpt_k = rhs_list_k[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_y of rhs, Dphi_2 saves grad_z of phi
                stress_exx[4] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
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
            kpt_k = rhs_list_k[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_z of rhs, Dphi_2 saves grad_y of phi
                stress_exx[5] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
            }
        }

        // additional term for spherical truncation
        if (mflag == 0) {
            poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const_stress2, count-base, DMnd, dims, phi, kpt_k_list, kpt_q_list, comm);
            
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
                kpt_k = rhs_list_k[count];
                occ_j = occ[j + band_start_indx_source + ll * Ns];
                occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
                
                for (k = 0; k < DMnd; k++){
                    stress_exx[6] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
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
    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_k);
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
    int i, j, grank, rank, size, spn_i;
    int Ns, Nband, DMnd, psi_shift, occ_outer_shift, psi_outer_shift, mflag;
    double *psi_outer, *occ_outer, *psi, pres_exx;
    MPI_Comm comm = pSPARC->dmcomm;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    mflag = pSPARC->EXXDiv_Flag;
    /********************************************************************/    
    
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
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        if (pSPARC->ACEFlag == 0) {
            psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
            occ_outer_shift = Ns * pSPARC->Nkpts_sym;
            psi_outer_shift = DMnd * pSPARC->Nkpts_hf_red * Ns;
            psi = pSPARC->Xorb + spn_i * psi_shift;
            psi_outer = pSPARC->psi_outer + spn_i * psi_outer_shift;
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            Calculate_exact_exchange_pressure_linear_nonACE(pSPARC, psi_outer, psi, occ_outer, &pres_exx);
        } else {
            psi = pSPARC->Xorb + spn_i * DMnd * Nband;
            occ_outer = pSPARC->occ + spn_i * Ns;
            Calculate_exact_exchange_pressure_linear_ACE(pSPARC, psi, occ_outer, &pres_exx);
        }
    }

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
void Calculate_exact_exchange_pressure_linear_nonACE(SPARC_OBJ *pSPARC, 
    double *psi_outer, double *occ, double *psi, double *pres_exx) 
{   
    solve_allpair_poissons_equation_pressure(pSPARC, 
        psi_outer, psi, occ, pSPARC->Nstates, 0, pres_exx);
}


/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_linear_ACE(SPARC_OBJ *pSPARC, 
    double *psi, double *occ, double *pres_exx) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int grank, DMnd, reps, NB, Ns, source, Nband_source, band_start_indx_source;
    double *psi_storage1, *psi_storage2, *sendbuff, *recvbuff;
    MPI_Request reqs[2];
    
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    DMnd = pSPARC->Nd_d_dmcomm;
    reps = pSPARC->npband - 1;
    NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
    Ns = pSPARC->Nstates;

    MPI_Comm blacscomm = pSPARC->blacscomm;
    int rank, size;
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_size(blacscomm, &size);
    /********************************************************************/

    if (reps > 0) {
        psi_storage1 = (double *) calloc(sizeof(double), NB*DMnd);
        psi_storage2 = (double *) calloc(sizeof(double), NB*DMnd);
    }

    for (int rep = 0; rep <= reps; rep ++) {
        source = (rank-rep+size)%size;
        Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
        band_start_indx_source = source * NB;
        if (rep == 0) {
            if (reps > 0) {
                transfer_orbitals_blacscomm(pSPARC, psi, psi_storage1, rep, reqs);
            }
            sendbuff = psi;
        } else {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            // first gather the orbitals in the rotation way
            sendbuff = (rep%2==1) ? psi_storage1 : psi_storage2;
            recvbuff = (rep%2==1) ? psi_storage2 : psi_storage1;
            if (rep != reps) {
                transfer_orbitals_blacscomm(pSPARC, sendbuff, recvbuff, rep, reqs);
            }
        }
        solve_allpair_poissons_equation_pressure(pSPARC, sendbuff, psi, occ, 
            Nband_source, band_start_indx_source, pres_exx);
    }
    
    if (reps > 0) {
        free(psi_storage1);
        free(psi_storage2);
    }
}

/**
 * @brief   Calculate Exact Exchange pressure
 */
void solve_allpair_poissons_equation_pressure(SPARC_OBJ *pSPARC, 
    double *psi_storage, double *psi, double *occ, int Nband_source, 
    int band_start_indx_source, double *pres_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank;
    int Ns, Nband, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, *rhs, *phi;
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    comm = pSPARC->dmcomm;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    /********************************************************************/
    mflag = pSPARC->EXXDiv_Flag;
    
    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;

    int *rhs_list_i, *rhs_list_j;
    rhs_list_i = (int*) calloc(Nband * Nband_source, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nband * Nband_source, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL);

    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (i = 0; i < Nband; i++) {
        for (j = 0; j < Nband_source; j++) {
            if (occ[i + pSPARC->band_start_indx] > 1e-6 
             && occ[j + band_start_indx_source] > 1e-6) {
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
                    num_rhs : pSPARC->EXXMem_batch * pSPARC->npNd;

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
                rhs[k + (count-base)*DMnd] = psi_storage[k + j*DMnd] * psi[k + i*DMnd];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve(pSPARC, rhs, pSPARC->pois_FFT_const_press, count-base, DMnd, dims, phi, comm);

        for (count = base; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            
            occ_i = occ[i + pSPARC->band_start_indx];
            occ_j = occ[j + band_start_indx_source];

            for (k = 0; k < DMnd; k++){
                *pres_exx += occ_i * occ_j * rhs[k + (count-base)*DMnd] * phi[k + (count-base)*DMnd];
            }
        }
    }
    free(rhs);
    free(phi);
    free(rhs_list_i);
    free(rhs_list_j);
}


/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt(SPARC_OBJ *pSPARC) 
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, grank, rank, size, spn_i;
    int Ns, Nband, DMnd, mflag;
    double *occ_outer, pres_exx, kpt_vec;
    double _Complex *psi_outer, *psi;
    MPI_Comm comm = pSPARC->dmcomm;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;    
    mflag = pSPARC->EXXDiv_Flag;
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
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        if (pSPARC->ACEFlag == 0) {
            int psi_outer_shift = DMnd * Ns * pSPARC->Nkpts_hf_red;
            int psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
            int occ_outer_shift = Ns * pSPARC->Nkpts_sym;
            
            psi_outer = pSPARC->psi_outer_kpt + spn_i * psi_outer_shift;
            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;

            Calculate_exact_exchange_pressure_kpt_nonACE(pSPARC, psi_outer, occ_outer, psi, &pres_exx);
        } else {
            int psi_shift = DMnd * Nband * pSPARC->Nkpts_kptcomm;
            int occ_outer_shift = Ns * pSPARC->Nkpts_sym;

            occ_outer = pSPARC->occ_outer + spn_i * occ_outer_shift;
            psi = pSPARC->Xorb_kpt + spn_i * psi_shift;

            Calculate_exact_exchange_pressure_kpt_ACE(pSPARC, psi, occ_outer, &pres_exx);
        }
    }
    
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

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt_nonACE(SPARC_OBJ *pSPARC, 
    double complex *psi_outer, double *occ_outer, double complex *psi, double *pres_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int k, kpt_q;
    int Ns, DMnd, size_k_, count;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    size_k_ = Ns * DMnd;
    /********************************************************************/    
    
    for (k = 0; k < pSPARC->Nkpts_hf_red; k++) {
        int counts = pSPARC->kpthfred2kpthf[k][0];
        for (count = 0; count < counts; count++) {
            kpt_q = pSPARC->kpthfred2kpthf[k][count+1];
            // solve poisson's equations 
            solve_allpair_poissons_equation_pressure_kpt(pSPARC, psi_outer + k*size_k_, 
                psi, occ_outer, Ns, 0, kpt_q, pres_exx);
        }
    }
}

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt_ACE(SPARC_OBJ *pSPARC, 
    double complex *psi, double *occ_outer, double *pres_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, grank, kpt_q;
    int Ns, DMnd, Nband, NB, size_k, count;
    int source, Nband_source, band_start_indx_source;
    double complex *psi_storage1_kpt, *psi_storage2_kpt, *psi_storage1_band, *psi_storage2_band;
    double complex *sendbuff_kpt, *recvbuff_kpt, *sendbuff_band, *recvbuff_band;
    MPI_Comm blacscomm = pSPARC->blacscomm;
    MPI_Request reqs_kpt[2], reqs_band[2];

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    size_k = Nband * DMnd;
    NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
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
    for (k = 0; k < pSPARC->npkpt; k++) 
        Nkpthf_red_max = max(Nkpthf_red_max, pSPARC->Nkpts_hf_list[k]);

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
                transfer_orbitals_kptbridgecomm_kpt(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt);
            }
        } else {
            MPI_Waitall(2, reqs_kpt, MPI_STATUSES_IGNORE);
            sendbuff_kpt = (rep_kpt%2==1) ? psi_storage2_kpt : psi_storage1_kpt;
            recvbuff_kpt = (rep_kpt%2==1) ? psi_storage1_kpt : psi_storage2_kpt;
            if (rep_kpt != reps_kpt) {
                transfer_orbitals_kptbridgecomm_kpt(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt);
            }
        }

        int source_kpt = (kpt_bridge_comm_rank-rep_kpt+kpt_bridge_comm_size)%kpt_bridge_comm_size;
        int nkpthf_red = pSPARC->Nkpts_hf_list[source_kpt];
        int kpthf_start_indx = pSPARC->kpthf_start_indx_list[source_kpt];
        
        for (k = 0; k < nkpthf_red; k++) {
            int k_indx = k + kpthf_start_indx;
            int counts = pSPARC->kpthfred2kpthf[k_indx][0];

            for (int rep_band = 0; rep_band <= reps_band; rep_band++) {
                // transfer the orbitals in the rotation way across blacscomm
                if (rep_band == 0) {
                    sendbuff_band = sendbuff_kpt + k*size_k;
                    if (reps_band > 0) {
                        recvbuff_band = psi_storage1_band;
                        transfer_orbitals_blacscomm_kpt(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band);
                    }
                } else {
                    MPI_Waitall(2, reqs_band, MPI_STATUSES_IGNORE);
                    sendbuff_band = (rep_band%2==1) ? psi_storage1_band : psi_storage2_band;
                    recvbuff_band = (rep_band%2==1) ? psi_storage2_band : psi_storage1_band;
                    if (rep_band != reps_band) {
                        transfer_orbitals_blacscomm_kpt(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band);
                    }
                }
                
                source = (blacscomm_rank-rep_band+blacscomm_size)%blacscomm_size;
                Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
                band_start_indx_source = source * NB;

                for (count = 0; count < counts; count++) {
                    kpt_q = pSPARC->kpthfred2kpthf[k_indx][count+1];
                    // solve poisson's equations 
                    solve_allpair_poissons_equation_pressure_kpt(pSPARC, sendbuff_band, 
                        psi, occ_outer, Nband_source, band_start_indx_source, kpt_q, pres_exx);
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
}

/**
 * @brief   Calculate Exact Exchange pressure
 */
void solve_allpair_poissons_equation_pressure_kpt(SPARC_OBJ *pSPARC, double complex *psi_storage, 
    double complex *psi, double *occ, int Nband_source, int band_start_indx_source, int kpt_q, double *pres_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, l, ll, grank, rank, size, n, kpt_k;
    int Ns, DMnd, dims[3], num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, kpt_vec;
    double _Complex *rhs, *phi;
    MPI_Comm comm = pSPARC->dmcomm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
    int size_k = DMnd * Nband;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    /********************************************************************/

    mflag = pSPARC->EXXDiv_Flag;
    dims[0] = pSPARC->npNdx; 
    dims[1] = pSPARC->npNdy; 
    dims[2] = pSPARC->npNdz;
    
    int *rhs_list_i, *rhs_list_j, *rhs_list_k, *kpt_k_list, *kpt_q_list;
    rhs_list_i = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    rhs_list_k = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    assert(rhs_list_i != NULL && rhs_list_j != NULL && rhs_list_k != NULL);

    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (kpt_k = 0; kpt_k < Nkpts_kptcomm; kpt_k ++) {
        for (i = 0; i < Nband; i++) {
            for (j = 0; j < Nband_source; j++) {
                occ_j = occ[j + band_start_indx_source + ll * Ns];
                occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
                
                if (occ_j > 1e-6 && occ_i > 1e-6) {
                    rhs_list_i[count] = i;
                    rhs_list_j[count] = j;
                    rhs_list_k[count] = kpt_k;
                    count ++;
                }
            }
        }
    }
    num_rhs = count;

    if (count == 0) {
        free(rhs_list_i);
        free(rhs_list_j);
        free(rhs_list_k);
        return;
    }
    batch_num_rhs = pSPARC->EXXMem_batch == 0 ? 
                    num_rhs : pSPARC->EXXMem_batch * pSPARC->npNd;
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
            kpt_k = rhs_list_k[count];                  // k-point indx of k vector
            kpt_k_list[count-base] = kpt_k + pSPARC->kpt_start_indx;
            kpt_q_list[count-base] = kpt_q;
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_storage[k + j*DMnd]) * psi[k + i*DMnd + kpt_k*size_k];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*DMnd] * psi[k + i*DMnd + kpt_k*size_k];
            }
        }
        
        // Solve all Poisson's equation 
        poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_FFT_const_press, count-base, DMnd, dims, phi, kpt_k_list, kpt_q_list, comm);

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];                      // col indx of Xorb
            j = rhs_list_j[count];                      // band indx of psi_outer
            kpt_k = rhs_list_k[count];                  // k-point indx of k vector
            kpt_k_list[count-base] = kpt_k + pSPARC->kpt_start_indx;
            kpt_q_list[count-base] = kpt_q;
            
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + (kpt_k + pSPARC->kpt_start_indx) * Ns];
            
            for (k = 0; k < DMnd; k++){
                *pres_exx += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(rhs[k + (count-base)*DMnd]) * phi[k + (count-base)*DMnd]);
            }
        }
    }

    free(rhs);
    free(phi);            
    free(kpt_k_list);
    free(kpt_q_list);
    free(rhs_list_i);
    free(rhs_list_j);
    free(rhs_list_k);
}