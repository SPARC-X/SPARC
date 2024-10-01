/***
 * @file    exactExchangeStress.c
 * @brief   This file contains the functions for Exact Exchange Stress.
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

#include "exactExchange.h"
#include "exactExchangeKpt.h"
#include "exactExchangeStress.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "stress.h"
#include "tools.h"
#include "sqHighTExactExchange.h"

#ifdef ACCELGT
#include "cuda_exactexchange.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


#define TEMP_TOL (1e-12)

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress(SPARC_OBJ *pSPARC) {
    if (pSPARC->sqHighTFlag == 1) {
        Calculate_exact_exchange_stress_SQ(pSPARC);
    } else if (pSPARC->isGammaPoint) {
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
    int i, j, grank, rank, size, spinor;
    int Ns, DMnd, mflag;
    double *psi_outer, *occ_outer, *psi, stress_exx[7];
    MPI_Comm comm;

    DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Ns = pSPARC->Nstates;
    comm = pSPARC->dmcomm;
    mflag = pSPARC->ExxDivFlag;
    /********************************************************************/
    for (i = 0; i < 7; i++) stress_exx[i] = 0.0;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (pSPARC->ExxAcc == 0) {            
            psi = pSPARC->Xorb + spinor * DMnd;
            psi_outer = pSPARC->psi_outer + spinor * DMnd;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * Ns) : pSPARC->occ_outer;
            Calculate_exact_exchange_stress_linear_nonACE(pSPARC, psi_outer, DMndsp, psi, DMndsp, occ_outer, stress_exx);
        } else {
            psi = pSPARC->Xorb + spinor * DMnd;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
        #ifdef ACCELGT
            if (pSPARC->useACCEL == 1) {
                CUDA_Calculate_exact_exchange_stress_linear_ACE(pSPARC, psi, DMndsp, occ_outer, stress_exx);
            } else
        #endif // ACCELGT
            {
                Calculate_exact_exchange_stress_linear_ACE(pSPARC, psi, DMndsp, occ_outer, stress_exx);   
            }
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
        pSPARC->stress_exx[i] *= (-pSPARC->exx_frac/pSPARC->dV/pSPARC->Nspin);

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
        double *psi_outer, int ldpo, double *psi, int ldp, double *occ_outer, double *stress_exx)
{
    solve_allpair_poissons_equation_stress(pSPARC, 
        psi_outer, ldpo, psi, ldp, occ_outer, pSPARC->Nstates, 0, stress_exx);
}


/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_linear_ACE(SPARC_OBJ *pSPARC, 
        double *psi, int ldp, double *occ, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int grank, DMnd, rep, reps, NB, Ns, source, Nband_source, band_start_indx_source;
    double *psi_storage1, *psi_storage2, *sendbuff, *recvbuff;
    psi_storage1 = psi_storage2 = sendbuff = recvbuff = NULL;
    MPI_Request reqs[2];

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
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

    for (rep = 0; rep <= reps; rep ++) {
        source = (rank-rep+size)%size;
        Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
        band_start_indx_source = source * NB;
        if (rep == 0) {
            if (reps > 0) {
                if (DMnd != DMndsp) {
                    copy_mat_blk(sizeof(double), psi, DMndsp, DMnd, pSPARC->Nband_bandcomm, psi_storage2, DMnd);
                    transfer_orbitals_blacscomm(pSPARC, psi_storage2, psi_storage1, rep, reqs, sizeof(double));
                    sendbuff = psi_storage2;
                } else {
                    transfer_orbitals_blacscomm(pSPARC, psi, psi_storage1, rep, reqs, sizeof(double));
                    sendbuff = psi;
                }
            } else {
                sendbuff = psi;
            }
        } else {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            // first gather the orbitals in the rotation way
            sendbuff = (rep%2==1) ? psi_storage1 : psi_storage2;
            recvbuff = (rep%2==1) ? psi_storage2 : psi_storage1;
            if (rep != reps) {
                transfer_orbitals_blacscomm(pSPARC, sendbuff, recvbuff, rep, reqs, sizeof(double));
            }
        }
        if (sendbuff == psi) {
            solve_allpair_poissons_equation_stress(pSPARC, psi, DMndsp, psi, DMndsp, occ, 
                Nband_source, band_start_indx_source, stress_exx);
        } else {
            solve_allpair_poissons_equation_stress(pSPARC, sendbuff, DMnd, psi, DMndsp, occ, 
                Nband_source, band_start_indx_source, stress_exx);
        }
        
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
    double *psi_storage, int ldps, double *psi, int ldp, double *occ, int Nband_source, 
    int band_start_indx_source, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    MPI_Comm blacscomm = pSPARC->blacscomm;
    MPI_Comm comm = pSPARC->dmcomm;
    int i, j, k, grank, rank, size;
    int Nband, DMnd, num_rhs, batch_num_rhs, NL, loop, base, mflag, band_start_indx;
    double occ_i, occ_j, *rhs, *phi, *Drhs, *Dphi_1, *Dphi_2;

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    MPI_Comm_rank(blacscomm, &rank);
    MPI_Comm_size(blacscomm, &size);

    DMnd = pSPARC->Nd_d_dmcomm;
    Nband = pSPARC->Nband_bandcomm;
    mflag = pSPARC->ExxDivFlag;
    band_start_indx = pSPARC->band_start_indx;
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

    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                    num_rhs : pSPARC->ExxMemBatch * pSPARC->npNd;

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
                rhs[k + (count-base)*DMnd] = psi_storage[k + j*ldps] * psi[k + i*ldp];
            }
        }

        // Solve all Poisson's equation 
        poissonSolve(pSPARC, rhs, pSPARC->pois_const_stress, count-base, DMnd, phi, comm);
        
        // component (1,1)
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, rhs, DMnd, Drhs, DMnd, 0, comm);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, phi, DMnd, Dphi_1, DMnd, 0, comm);
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
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, phi, DMnd, Dphi_1, DMnd, 1, comm);
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
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, phi, DMnd, Dphi_2, DMnd, 2, comm);
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
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, rhs, DMnd, Drhs, DMnd, 1, comm);
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
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, count-base, 0.0, rhs, DMnd, Drhs, DMnd, 2, comm);
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
            poissonSolve(pSPARC, rhs, pSPARC->pois_const_stress2, count-base, DMnd, phi, comm);
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
    int i, j, grank, rank, size;
    int Ns, DMnd, mflag;
    double *occ_outer, stress_exx[7];
    double _Complex *psi_outer, *psi;
    MPI_Comm comm = pSPARC->dmcomm;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    Ns = pSPARC->Nstates;
    mflag = pSPARC->ExxDivFlag;
    /********************************************************************/
    for (i = 0; i < 7; i++) stress_exx[i] = 0.0;

    for (int spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        if (pSPARC->ExxAcc == 0) {
            psi_outer = pSPARC->psi_outer_kpt + spinor * DMnd;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * Ns * pSPARC->Nkpts_sym) : pSPARC->occ_outer;
            psi = pSPARC->Xorb_kpt + spinor * DMnd;

            Calculate_exact_exchange_stress_kpt_nonACE(pSPARC, psi_outer, DMndsp, psi, DMndsp, occ_outer, stress_exx);
        } else {
            int occ_outer_shift = Ns * pSPARC->Nkpts_sym;
            occ_outer = (pSPARC->spin_typ == 1) ? (pSPARC->occ_outer + spinor * occ_outer_shift) : pSPARC->occ_outer;
            psi = pSPARC->Xorb_kpt + spinor * DMnd;

        #ifdef ACCELGT
            if (pSPARC->useACCEL == 1) {
                CUDA_Calculate_exact_exchange_stress_kpt_ACE(pSPARC, psi, DMndsp, occ_outer, stress_exx);                
            } else
        #endif // ACCELGT
            {
                Calculate_exact_exchange_stress_kpt_ACE(pSPARC, psi, DMndsp, occ_outer, stress_exx);
            }
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
        pSPARC->stress_exx[i] *= (-pSPARC->exx_frac/pSPARC->dV/pSPARC->Nspin);

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
    double _Complex *psi_outer, int ldpo, double _Complex *psi, int ldp, double *occ_outer, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int k, grank, kpt_q;
    int Ns, size_k_, count;

    Ns = pSPARC->Nstates;
    size_k_ = Ns * ldpo;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    
    for (k = 0; k < pSPARC->Nkpts_hf_red; k++) {
        int counts = pSPARC->kpthfred2kpthf[k][0];
        for (count = 0; count < counts; count++) {
            kpt_q = pSPARC->kpthfred2kpthf[k][count+1];
            // solve poisson's equations 
            for (int kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k ++) {
                solve_allpair_poissons_equation_stress_kpt(pSPARC, psi_outer + k*size_k_, ldpo, 
                    psi, ldp, occ_outer, Ns, 0, kpt_k, kpt_q, stress_exx);
            }
        }
    }
}


/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt_ACE(SPARC_OBJ *pSPARC, 
    double _Complex *psi, int ldp, double *occ_outer, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int k, grank, kpt_q;
    int Ns, DMnd, Nband, NB, size_k, count, rep_kpt, rep_band;
    int source, Nband_source, band_start_indx_source;
    double _Complex *psi_storage1_kpt, *psi_storage2_kpt, *psi_storage1_band, *psi_storage2_band;
    double _Complex *sendbuff_kpt, *recvbuff_kpt, *sendbuff_band, *recvbuff_band;
    psi_storage1_kpt = psi_storage2_kpt = psi_storage1_band = psi_storage2_band = NULL;
    sendbuff_kpt = recvbuff_kpt = sendbuff_band = recvbuff_band = NULL;
    MPI_Request reqs_kpt[2], reqs_band[2];

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    Nband = pSPARC->Nband_bandcomm;
    size_k = Nband * ldp;
    int size_k_ps = DMnd * Nband;
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

    psi_storage1_kpt = (double _Complex *) calloc(sizeof(double _Complex), DMnd * Nband * Nkpthf_red_max);
    assert(psi_storage1_kpt != NULL);
    if (reps_kpt > 0) {
        psi_storage2_kpt = (double _Complex *) calloc(sizeof(double _Complex), DMnd * Nband * Nkpthf_red_max);
        assert(psi_storage2_kpt != NULL);
    }
    // extract and store all the orbitals for hybrid calculation
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        if (!pSPARC->kpthf_flag_kptcomm[k]) continue;
        copy_mat_blk(sizeof(double _Complex), psi + k*size_k, ldp, DMnd, Nband, psi_storage1_kpt + count*size_k_ps, DMnd);
        count ++;
    }
    if (reps_band > 0) {
        psi_storage1_band = (double _Complex *) calloc(sizeof(double _Complex), DMnd * Nband_max);
        psi_storage2_band = (double _Complex *) calloc(sizeof(double _Complex), DMnd * Nband_max);
        assert(psi_storage1_band != NULL && psi_storage2_band != NULL);
    }

    for (rep_kpt = 0; rep_kpt <= reps_kpt; rep_kpt++) {

        // transfer the orbitals in the rotation way across kpt_bridge_comm
        if (rep_kpt == 0) {
            sendbuff_kpt = psi_storage1_kpt;
            if (reps_kpt > 0) {
                recvbuff_kpt = psi_storage2_kpt;
                transfer_orbitals_kptbridgecomm(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt, sizeof(double _Complex));
            }
        } else {
            MPI_Waitall(2, reqs_kpt, MPI_STATUSES_IGNORE);
            sendbuff_kpt = (rep_kpt%2==1) ? psi_storage2_kpt : psi_storage1_kpt;
            recvbuff_kpt = (rep_kpt%2==1) ? psi_storage1_kpt : psi_storage2_kpt;
            if (rep_kpt != reps_kpt) {
                transfer_orbitals_kptbridgecomm(pSPARC, sendbuff_kpt, recvbuff_kpt, rep_kpt, reqs_kpt, sizeof(double _Complex));
            }
        }

        int source_kpt = (kpt_bridge_comm_rank-rep_kpt+kpt_bridge_comm_size)%kpt_bridge_comm_size;
        int nkpthf_red = pSPARC->Nkpts_hf_list[source_kpt];
        int kpthf_start_indx = pSPARC->kpthf_start_indx_list[source_kpt];
        
        for (k = 0; k < nkpthf_red; k++) {
            int k_indx = k + kpthf_start_indx;
            int counts = pSPARC->kpthfred2kpthf[k_indx][0];

            for (rep_band = 0; rep_band <= reps_band; rep_band++) {
                // transfer the orbitals in the rotation way across blacscomm
                if (rep_band == 0) {
                    sendbuff_band = sendbuff_kpt + k*size_k_ps;
                    if (reps_band > 0) {
                        recvbuff_band = psi_storage1_band;
                        transfer_orbitals_blacscomm(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band, sizeof(double _Complex));
                    }
                } else {                    
                    MPI_Waitall(2, reqs_band, MPI_STATUSES_IGNORE);
                    sendbuff_band = (rep_band%2==1) ? psi_storage1_band : psi_storage2_band;
                    recvbuff_band = (rep_band%2==1) ? psi_storage2_band : psi_storage1_band;
                    if (rep_band != reps_band) {
                        transfer_orbitals_blacscomm(pSPARC, sendbuff_band, recvbuff_band, rep_band, reqs_band, sizeof(double _Complex));
                    }
                }
                
                source = (blacscomm_rank-rep_band+blacscomm_size)%blacscomm_size;
                Nband_source = source < (Ns / NB) ? NB : (source == (Ns / NB) ? (Ns % NB) : 0);
                band_start_indx_source = source * NB;

                for (count = 0; count < counts; count++) {
                    kpt_q = pSPARC->kpthfred2kpthf[k_indx][count+1];
                    // solve poisson's equations 
                    for (int kpt_k = 0; kpt_k < pSPARC->Nkpts_kptcomm; kpt_k ++) {
                        solve_allpair_poissons_equation_stress_kpt(pSPARC, sendbuff_band, DMnd, 
                            psi, ldp, occ_outer, Nband_source, band_start_indx_source, kpt_k, kpt_q, stress_exx);
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
}

/**
 * @brief   Calculate Exact Exchange stress
 */
void solve_allpair_poissons_equation_stress_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_storage, int ldps, 
    double _Complex *psi, int ldp, double *occ, int Nband_source, int band_start_indx_source, int kpt_k, int kpt_q, double *stress_exx)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int i, j, k, ll, grank, rank, size, n;
    int Ns, DMnd, num_rhs, batch_num_rhs, NL, loop, base, mflag;
    double occ_i, occ_j, kpt_vec;
    double _Complex *rhs, *phi, *Drhs,*Dphi_1, *Dphi_2;
    MPI_Comm comm = pSPARC->dmcomm;

    DMnd = pSPARC->Nd_d_dmcomm;
    Ns = pSPARC->Nstates;
    int Nband = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    ll = pSPARC->kpthf_ind[kpt_q];                  // ll w.r.t. Nkpts_sym, for occ
    int size_k = ldp * Nband;
    int kpt_k_ = kpt_k + pSPARC->kpt_start_indx;
    /********************************************************************/

    MPI_Comm_rank(MPI_COMM_WORLD, &grank);    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    mflag = pSPARC->ExxDivFlag;
    
    int *rhs_list_i, *rhs_list_j;
    rhs_list_i = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int)); 
    rhs_list_j = (int*) calloc(Nkpts_kptcomm * Nband * Nband_source, sizeof(int));     
    assert(rhs_list_i != NULL && rhs_list_j != NULL);
    
    // Find the number of Poisson's equation required to be solved
    // Using the occupation threshold 1e-6
    int count = 0;
    for (i = 0; i < Nband; i++) {
        for (j = 0; j < Nband_source; j++) {
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
            if (occ_j > 1e-6 && occ_i > 1e-6) {
                rhs_list_i[count] = i;
                rhs_list_j[count] = j;
                count ++;
            }
        }
    }
    num_rhs = count;
    
    if (!num_rhs) {
        free(rhs_list_i);
        free(rhs_list_j);
        return;
    }
    batch_num_rhs = pSPARC->ExxMemBatch == 0 ? 
                    num_rhs : pSPARC->ExxMemBatch * pSPARC->npNd;
    NL = (num_rhs - 1) / batch_num_rhs + 1;                                                // number of loops required                        

    rhs = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                            // right hand sides of Poisson's equation
    phi = (double _Complex *)malloc(sizeof(double _Complex) * DMnd * batch_num_rhs);                            // the solution for each rhs
    assert(rhs != NULL && phi != NULL);

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
            if (pSPARC->kpthf_pn[kpt_q] == 1) {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = conj(psi_storage[k + j*ldps]) * psi[k + i*ldp + kpt_k*size_k];
            } else {
                for (k = 0; k < DMnd; k++) 
                    rhs[k + (count-base)*DMnd] = psi_storage[k + j*ldps] * psi[k + i*ldp + kpt_k*size_k];
            }
        }
        
        // Solve all Poisson's equation 
        poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_const_stress, count-base, DMnd, phi, kpt_k_, kpt_q, comm);

        // component (1,1)
        for (n = 0; n < count-base; n++) {
            kpt_vec = pSPARC->k1[kpt_k_] - pSPARC->k1_hf[kpt_q];
            // calculate gradients of rhs
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, rhs+n*DMnd, DMnd, Drhs+n*DMnd, DMnd, 0, &kpt_vec, comm);
            // calculate gradients of phi
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, phi+n*DMnd, DMnd, Dphi_1+n*DMnd, DMnd, 0, &kpt_vec, comm);
        }

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_x of rhs, Dphi_1 saves grad_x of phi
                stress_exx[0] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_1[k + (count-base)*DMnd]);
            }
        }
        
        // component (1,2)
        for (n = 0; n < count-base; n++) {
            kpt_vec = pSPARC->k2[kpt_k_] - pSPARC->k2_hf[kpt_q];
            // calculate gradients of phi
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, phi+n*DMnd, DMnd, Dphi_1+n*DMnd, DMnd, 1, &kpt_vec, comm);
        }

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];            
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_x of rhs, Dphi_1 saves grad_y of phi
                stress_exx[1] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_1[k + (count-base)*DMnd]);
            }
        }

        // component (1,3)
        for (n = 0; n < count-base; n++) {
            kpt_vec = pSPARC->k3[kpt_k_] - pSPARC->k3_hf[kpt_q];
            // calculate gradients of phi
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, phi+n*DMnd, DMnd, Dphi_2+n*DMnd, DMnd, 2, &kpt_vec, comm);
        }

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];            
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_x of rhs, Dphi_2 saves grad_z of phi
                stress_exx[2] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
            }
        }

        // component (2,2)
        for (n = 0; n < count-base; n++) {
            kpt_vec = pSPARC->k2[kpt_k_] - pSPARC->k2_hf[kpt_q];
            // calculate gradients of rhs
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, rhs+n*DMnd, DMnd, Drhs+n*DMnd, DMnd, 1, &kpt_vec, comm);
        }

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
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
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_y of rhs, Dphi_2 saves grad_z of phi
                stress_exx[4] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
            }
        }

        // component (3,3)
        for (n = 0; n < count-base; n++) {
            kpt_vec = pSPARC->k3[kpt_k_] - pSPARC->k3_hf[kpt_q];
            // calculate gradients of rhs
            Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, 1, 0.0, rhs+n*DMnd, DMnd, Drhs+n*DMnd, DMnd, 2, &kpt_vec, comm);
        }

        for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
            i = rhs_list_i[count];
            j = rhs_list_j[count];            
            occ_j = occ[j + band_start_indx_source + ll * Ns];
            occ_i = occ[i + pSPARC->band_start_indx + kpt_k_ * Ns];
            
            for (k = 0; k < DMnd; k++){
                // Drhs saves grad_z of rhs, Dphi_2 saves grad_y of phi
                stress_exx[5] += pSPARC->kptWts_hf * pSPARC->kptWts_loc[kpt_k] / pSPARC->Nkpts * occ_i * occ_j * 
                                    creal(conj(Drhs[k + (count-base)*DMnd]) * Dphi_2[k + (count-base)*DMnd]);
            }
        }

        // additional term for spherical truncation
        if (mflag == 0) {
            poissonSolve_kpt(pSPARC, rhs, pSPARC->pois_const_stress2, count-base, DMnd, phi, kpt_k_, kpt_q, comm);
            
            for (count = batch_num_rhs*loop; count < min(batch_num_rhs*(loop+1),num_rhs); count++) {
                i = rhs_list_i[count];
                j = rhs_list_j[count];
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
    free(Drhs);
    free(Dphi_1);
    free(Dphi_2);
    free(rhs_list_i);
    free(rhs_list_j);
}
