/**
 * @file    MGGAstress.c
 * @brief   This file contains the metaGGA term of Hamiltonian*Wavefunction, 
 * \int_{Omega} rho * (\part epsilon_{xc} / \part tau) * [\sum_{n=1}^N_s \int_{BZ} g_n(2\nabla_{alpha} psi_n_k * \nabla_{beta} psi_n_k) / |V_{BZ}| dk] dx
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "isddft.h"
#include "mGGAstress.h"
#include "tools.h"
#include "parallelization.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"

/**
 * @brief   compute the metaGGA psi stress term,
 * called by Calculate_electronic_stress in stress.c and Calculate_electronic_pressure in pressure.c
 * Attention: it is NOT the full contrubution of metaGGA on stress!
 */
void Calculate_XC_stress_mGGA_psi_term(SPARC_OBJ *pSPARC) {
    // currently there is no spin-polarized
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return; // mimic Calculate_nonlocal_kinetic_stress_linear in stress.c ???
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int ncol, Ns, DMnd, DMndsp, dim, count, spinor, n, i, j;
    double *dpsi_ptr, *dpsi_ptr2, dpsi1_dpsi1, g_nk, dpsi1_dpsi2, dpsi1_dpsi3, dpsi2_dpsi3, dpsi2_dpsi2, dpsi3_dpsi3;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates; // total number of bands
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    double *stress_mGGA_psi, *stress_mGGA_psi_cartesian, *dpsi_full; // the metaGGA term directly related to all wave functions psi
    stress_mGGA_psi = (double*) calloc(6, sizeof(double));
    stress_mGGA_psi_cartesian = (double*) calloc(6, sizeof(double));
    dpsi_full = (double *)malloc( DMnd * ncol * sizeof(double) );
    if (dpsi_full == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    double *vxcMGGA3_loc = pSPARC->vxcMGGA3_loc_dmcomm; // local rho*d\epsilon_{xc} / d\tau array
    for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor++) {
        int sg = pSPARC->spinor_start_indx + spinor;
        count = 0;
        for (dim = 0; dim < 3; dim++) {
            // find dPsi in direction dim
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spinor*DMnd, DMndsp, pSPARC->Yorb, DMnd, dim, pSPARC->dmcomm);
            // Kinetic stress
            if(dim == 0){
                // component (1,1)
                // find d(Psi*Ds)                
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_1
                    dpsi1_dpsi1 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi1 += vxcMGGA3_loc[sg*DMnd + i] * *(dpsi_ptr + i) * *(dpsi_ptr + i);
                    }                    
                    double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
                    g_nk = occ[n + pSPARC->band_start_indx];
                    stress_mGGA_psi[0] += dpsi1_dpsi1 * g_nk; // component (1,1)
                }

                // component (1,2)
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spinor*DMnd, DMndsp, dpsi_full, DMnd, 1, pSPARC->dmcomm);
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_2
                    dpsi1_dpsi2 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi2 += vxcMGGA3_loc[sg*DMnd + i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
                    g_nk = occ[n + pSPARC->band_start_indx];
                    stress_mGGA_psi[1] += dpsi1_dpsi2 * g_nk; // component (1,2)
                }
                

                // component (1,3)
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb+spinor*DMnd, DMndsp, dpsi_full, DMnd, 2, pSPARC->dmcomm);
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_1
                    dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_3
                    dpsi1_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi1_dpsi3 += vxcMGGA3_loc[sg*DMnd + i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
                    g_nk = occ[n + pSPARC->band_start_indx];
                    stress_mGGA_psi[2] += dpsi1_dpsi3 * g_nk; // component (1,3)
                }
            } else if(dim == 1){
                // component (2,3)
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_2
                    dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_3
                    dpsi2_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi2_dpsi3 += vxcMGGA3_loc[sg*DMnd + i] * *(dpsi_ptr + i) * *(dpsi_ptr2 + i);
                    }
                    double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
                    g_nk = occ[n + pSPARC->band_start_indx];
                    stress_mGGA_psi[4] += dpsi2_dpsi3 * g_nk; // component (2,3)
                }

                // component (2,2)
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_1
                    dpsi2_dpsi2 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi2_dpsi2 += vxcMGGA3_loc[sg*DMnd + i] * *(dpsi_ptr + i) * *(dpsi_ptr + i);
                    }
                    double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
                    g_nk = occ[n + pSPARC->band_start_indx];
                    stress_mGGA_psi[3] += dpsi2_dpsi2 * g_nk; // component (2,2)
                }
            } else if (dim == 2) {
                // component (3,3)                
                for(n = 0; n < ncol; n++){
                    dpsi_ptr = pSPARC->Yorb + n * DMnd; // dpsi_1
                    dpsi3_dpsi3 = 0.0;
                    for(i = 0; i < DMnd; i++){
                        dpsi3_dpsi3 += vxcMGGA3_loc[sg*DMnd + i] * *(dpsi_ptr + i) * *(dpsi_ptr + i);
                    }
                    double *occ = (pSPARC->spin_typ == 1) ? (pSPARC->occ + spinor * Ns) : pSPARC->occ;
                    g_nk = occ[n + pSPARC->band_start_indx];
                    stress_mGGA_psi[5] += dpsi3_dpsi3 * g_nk; // component (3,3)
                }
            }
        }    
        count++;
    }

    stress_mGGA_psi[0] *= -pSPARC->occfac; // component (1,1)
    stress_mGGA_psi[1] *= -pSPARC->occfac; // component (1,2)
    stress_mGGA_psi[2] *= -pSPARC->occfac; // component (1,3)
    stress_mGGA_psi[3] *= -pSPARC->occfac; // component (2,2)
    stress_mGGA_psi[4] *= -pSPARC->occfac; // component (2,3)
    stress_mGGA_psi[5] *= -pSPARC->occfac; // component (3,3)

    free(dpsi_full);
    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if(pSPARC->cell_typ == 0){
        for(i = 0; i < 6; i++)
            stress_mGGA_psi_cartesian[i] = stress_mGGA_psi[i];
    } else {
        double *c_g, *c_c;
        c_g = (double*) malloc(9*sizeof(double));
        c_c = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
                c_c[3*i+j] = pSPARC->LatUVec[3*j+i];
            }
        }

        stress_mGGA_psi_cartesian[0] = c_g[0] * (c_g[0] * stress_mGGA_psi[0] + c_g[1] * stress_mGGA_psi[1] + c_g[2] * stress_mGGA_psi[2]) +
                              c_g[1] * (c_g[0] * stress_mGGA_psi[1] + c_g[1] * stress_mGGA_psi[3] + c_g[2] * stress_mGGA_psi[4]) +
                              c_g[2] * (c_g[0] * stress_mGGA_psi[2] + c_g[1] * stress_mGGA_psi[4] + c_g[2] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[1] = c_g[0] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                              c_g[1] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                              c_g[2] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[2] = c_g[0] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                              c_g[1] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                              c_g[2] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[3] = c_g[3] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                              c_g[4] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                              c_g[5] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[4] = c_g[3] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                              c_g[4] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                              c_g[5] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[5] = c_g[6] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                              c_g[7] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                              c_g[8] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);                                                                                   
  
        free(c_g); free(c_c);
    }


    // sum over all spin
    if (pSPARC->npspin > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }
    
    // sum over all bands
    if (pSPARC->npband > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->Calc_stress == 1) {
        if (!rank) {
            // Define measure of unit cell
            double cell_measure = pSPARC->Jacbdet;
            if(pSPARC->BCx == 0)
                cell_measure *= pSPARC->range_x;
            if(pSPARC->BCy == 0)
                cell_measure *= pSPARC->range_y;
            if(pSPARC->BCz == 0)
                cell_measure *= pSPARC->range_z;
            printf("\npSPARC->dV=%12.9E, cell_measure=%12.9E\n", pSPARC->dV, cell_measure);
            for(i = 0; i < 6; i++) {
                stress_mGGA_psi_cartesian[i] /= cell_measure;
            }
            #ifdef DEBUG
            printf("\nmetaGGA psi stress term (which is not the total contribution of metaGGA on stress!) (GPa):\n");
            printf("%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n", 
                    stress_mGGA_psi_cartesian[0]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[3]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[5]*CONST_HA_BOHR3_GPA);
            #endif
            // add metaGGA psi stress term to exchange-correlation stress
            pSPARC->stress_xc[0] += stress_mGGA_psi_cartesian[0]; // (1,1)
            pSPARC->stress_xc[1] += stress_mGGA_psi_cartesian[1]; // (1,2)
            pSPARC->stress_xc[2] += stress_mGGA_psi_cartesian[2]; // (1,3)
            pSPARC->stress_xc[3] += stress_mGGA_psi_cartesian[3]; // (2,2)
            pSPARC->stress_xc[4] += stress_mGGA_psi_cartesian[4]; // (2,3)
            pSPARC->stress_xc[5] += stress_mGGA_psi_cartesian[5]; // (3,3)
        }
    } 

    if (pSPARC->Calc_pres == 1) {
        if(!rank) {
            pSPARC->pres_xc += (stress_mGGA_psi_cartesian[0] + stress_mGGA_psi_cartesian[3] + stress_mGGA_psi_cartesian[5]);
        }
    }
    free(stress_mGGA_psi);
    free(stress_mGGA_psi_cartesian);
}

/**
 * @brief   compute the metaGGA psi stress term, which is 
 * \int_{Omega}(\rho\frac{\partial \epsilon_{xc}}{\partial \tau})[\sum_{n=1}^N_s g_n(2\nabla_{\alpha}\psi_n * \nabla_{\beta}\psi_n)] dx
 * called by Calculate_electronic_stress in stress.c
 * Attention: it is NOT the full contrubution of metaGGA on stress!
 */
void Calculate_XC_stress_mGGA_psi_term_kpt(SPARC_OBJ *pSPARC) {
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i, j, n, ncol, Ns, DMnd, DMndsp, dim, count, kpt, Nk, size_k, spinor, Nspinor;
    ncol = pSPARC->Nband_bandcomm; // number of bands assigned
    Ns = pSPARC->Nstates;
    DMnd = pSPARC->Nd_d_dmcomm;
    Nspinor = pSPARC->Nspinor_spincomm;
    DMndsp = DMnd * Nspinor;
    Nk = pSPARC->Nkpts_kptcomm;    
    size_k = DMndsp * ncol;    

    double _Complex *dpsi_ptr, *dpsi_ptr2, *dpsi_full;
    double *temp_k, *stress_mGGA_psi, *stress_mGGA_psi_cartesian, g_nk;
    double dpsi1_dpsi1, dpsi1_dpsi2, dpsi1_dpsi3, dpsi2_dpsi2, dpsi2_dpsi3, dpsi3_dpsi3;

    temp_k = (double*) malloc(6 * sizeof(double));
    stress_mGGA_psi = (double*) calloc(6, sizeof(double));
    stress_mGGA_psi_cartesian = (double*) calloc(6, sizeof(double));
    
    dpsi_full = (double _Complex *)malloc( DMnd * ncol * sizeof(double _Complex) );
    if (dpsi_full == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    double *vxcMGGA3_loc = pSPARC->vxcMGGA3_loc_dmcomm; // local rho*d\epsilon_{xc} / d\tau array
    
    double k1, k2, k3, kpt_vec;


    count = 0;
    for(spinor = 0; spinor < Nspinor; spinor++) {
        int sg = pSPARC->spinor_start_indx + spinor;
        for(kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            k1 = pSPARC->k1_loc[kpt];
            k2 = pSPARC->k2_loc[kpt];
            k3 = pSPARC->k3_loc[kpt];
            for (dim = 0; dim < 3; dim++) {
                // find dPsi in direction dim
                kpt_vec = (dim == 0) ? k1 : ((dim == 1) ? k2 : k3);
                Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spinor*DMnd+kpt*size_k, DMndsp, pSPARC->Yorb_kpt, DMnd, dim, &kpt_vec, pSPARC->dmcomm);

                // Kinetic stress
                if(dim == 0){
                    kpt_vec = k2;
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spinor*DMnd+kpt*size_k, DMndsp, dpsi_full, DMnd, 1, &kpt_vec, pSPARC->dmcomm);
                    //ts = MPI_Wtime();
                    temp_k[0] = temp_k[1] = temp_k[3] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_1
                        dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_2
                        dpsi1_dpsi1 = dpsi1_dpsi2 = dpsi2_dpsi2 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi1_dpsi1 += vxcMGGA3_loc[sg*DMnd + i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr + i)));
                            dpsi1_dpsi2 += vxcMGGA3_loc[sg*DMnd + i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i)));
                            dpsi2_dpsi2 += vxcMGGA3_loc[sg*DMnd + i] * (creal(*(dpsi_ptr2 + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr2 + i)) * cimag(*(dpsi_ptr2 + i)));
                        }
                        double *occ = pSPARC->occ + kpt*Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                        g_nk = occ[n + pSPARC->band_start_indx];
                        temp_k[0] += dpsi1_dpsi1 * g_nk;
                        temp_k[1] += dpsi1_dpsi2 * g_nk;
                        temp_k[3] += dpsi2_dpsi2 * g_nk;
                    }
                    stress_mGGA_psi[0] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[0];
                    stress_mGGA_psi[1] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[1];
                    stress_mGGA_psi[3] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[3];

                    kpt_vec = k3;
                    Gradient_vectors_dir_kpt(pSPARC, DMnd, pSPARC->DMVertices_dmcomm, ncol, 0.0, pSPARC->Xorb_kpt+spinor*DMnd+kpt*size_k, DMndsp, dpsi_full, DMnd, 2, &kpt_vec, pSPARC->dmcomm);
                    temp_k[2] = temp_k[5] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_1
                        dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_3
                        dpsi1_dpsi3 = dpsi3_dpsi3 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi1_dpsi3 += vxcMGGA3_loc[sg*DMnd + i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i)));
                            dpsi3_dpsi3 += vxcMGGA3_loc[sg*DMnd + i] * (creal(*(dpsi_ptr2 + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr2 + i)) * cimag(*(dpsi_ptr2 + i)));
                        }
                        double *occ = pSPARC->occ + kpt*Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                        g_nk = occ[n + pSPARC->band_start_indx];
                        temp_k[2] += dpsi1_dpsi3 * g_nk;
                        temp_k[5] += dpsi3_dpsi3 * g_nk;
                    }
                    stress_mGGA_psi[2] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[2];
                    stress_mGGA_psi[5] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[5];
                } else if(dim == 1){
                    temp_k[4] = 0.0;
                    for(n = 0; n < ncol; n++){
                        dpsi_ptr = pSPARC->Yorb_kpt + n * DMnd; // dpsi_2
                        dpsi_ptr2 = dpsi_full + n * DMnd; // dpsi_3
                        dpsi2_dpsi3 = 0.0;
                        for(i = 0; i < DMnd; i++){
                            dpsi2_dpsi3 += vxcMGGA3_loc[sg*DMnd + i] * (creal(*(dpsi_ptr + i)) * creal(*(dpsi_ptr2 + i)) + cimag(*(dpsi_ptr + i)) * cimag(*(dpsi_ptr2 + i)));
                        }
                        double *occ = pSPARC->occ + kpt*Ns;
                        if (pSPARC->spin_typ == 1) occ += spinor * Nk * Ns;
                        g_nk = occ[n + pSPARC->band_start_indx];
                        temp_k[4] += dpsi2_dpsi3 * g_nk;
                    }
                    stress_mGGA_psi[4] -= pSPARC->occfac * pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts * temp_k[4];
                }
            }
            count++;
        }    
    }
    
    free(dpsi_full);

    if (pSPARC->npNd > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi, 6, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }

    if(pSPARC->cell_typ == 0){
        for(i = 0; i < 6; i++)
            stress_mGGA_psi_cartesian[i] = stress_mGGA_psi[i];
    } else {
        double *c_g, *c_c;
        c_g = (double*) malloc(9*sizeof(double));
        c_c = (double*) malloc(9*sizeof(double));
        for(i = 0; i < 3; i++){
            for(j = 0; j < 3; j++){
                c_g[3*i+j] = pSPARC->gradT[3*j+i];
                c_c[3*i+j] = pSPARC->LatUVec[3*j+i];
            }
        }
        stress_mGGA_psi_cartesian[0] = c_g[0] * (c_g[0] * stress_mGGA_psi[0] + c_g[1] * stress_mGGA_psi[1] + c_g[2] * stress_mGGA_psi[2]) +
                                       c_g[1] * (c_g[0] * stress_mGGA_psi[1] + c_g[1] * stress_mGGA_psi[3] + c_g[2] * stress_mGGA_psi[4]) +
                                       c_g[2] * (c_g[0] * stress_mGGA_psi[2] + c_g[1] * stress_mGGA_psi[4] + c_g[2] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[1] = c_g[0] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                                       c_g[1] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                                       c_g[2] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[2] = c_g[0] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                                       c_g[1] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                                       c_g[2] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[3] = c_g[3] * (c_g[3] * stress_mGGA_psi[0] + c_g[4] * stress_mGGA_psi[1] + c_g[5] * stress_mGGA_psi[2]) +
                                       c_g[4] * (c_g[3] * stress_mGGA_psi[1] + c_g[4] * stress_mGGA_psi[3] + c_g[5] * stress_mGGA_psi[4]) +
                                       c_g[5] * (c_g[3] * stress_mGGA_psi[2] + c_g[4] * stress_mGGA_psi[4] + c_g[5] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[4] = c_g[3] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                                       c_g[4] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                                       c_g[5] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);
        stress_mGGA_psi_cartesian[5] = c_g[6] * (c_g[6] * stress_mGGA_psi[0] + c_g[7] * stress_mGGA_psi[1] + c_g[8] * stress_mGGA_psi[2]) +
                                       c_g[7] * (c_g[6] * stress_mGGA_psi[1] + c_g[7] * stress_mGGA_psi[3] + c_g[8] * stress_mGGA_psi[4]) +
                                       c_g[8] * (c_g[6] * stress_mGGA_psi[2] + c_g[7] * stress_mGGA_psi[4] + c_g[8] * stress_mGGA_psi[5]);                                                                                   
  
        free(c_g); free(c_c);
    }    
    
    // sum over all spin
    if (pSPARC->npspin > 1) {    

        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
    }
    // sum over all kpoints
    if (pSPARC->npkpt > 1) {    
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }
    // sum over all bands
    if (pSPARC->npband > 1) {
        MPI_Allreduce(MPI_IN_PLACE, stress_mGGA_psi_cartesian, 6, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

    if (pSPARC->Calc_stress == 1) {
        if (!rank) {
            // Define measure of unit cell
            double cell_measure = pSPARC->Jacbdet;
            if(pSPARC->BCx == 0)
                cell_measure *= pSPARC->range_x;
            if(pSPARC->BCy == 0)
                cell_measure *= pSPARC->range_y;
            if(pSPARC->BCz == 0)
                cell_measure *= pSPARC->range_z;
            printf("\npSPARC->dV=%12.9E, cell_measure=%12.9E\n", pSPARC->dV, cell_measure);
            for(i = 0; i < 6; i++) {
                stress_mGGA_psi_cartesian[i] /= cell_measure;
            }
            #ifdef DEBUG
            printf("\nmetaGGA psi stress term (which is not the total contribution of metaGGA on stress!) (GPa):\n");
            printf("%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n%18.14f %18.14f %18.14f \n", 
                    stress_mGGA_psi_cartesian[0]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[1]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[3]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, 
                    stress_mGGA_psi_cartesian[2]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[4]*CONST_HA_BOHR3_GPA, stress_mGGA_psi_cartesian[5]*CONST_HA_BOHR3_GPA);
            #endif
            // add metaGGA psi stress term to exchange-correlation stress
            pSPARC->stress_xc[0] += stress_mGGA_psi_cartesian[0]; // (1,1)
            pSPARC->stress_xc[1] += stress_mGGA_psi_cartesian[1]; // (1,2)
            pSPARC->stress_xc[2] += stress_mGGA_psi_cartesian[2]; // (1,3)
            pSPARC->stress_xc[3] += stress_mGGA_psi_cartesian[3]; // (2,2)
            pSPARC->stress_xc[4] += stress_mGGA_psi_cartesian[4]; // (2,3)
            pSPARC->stress_xc[5] += stress_mGGA_psi_cartesian[5]; // (3,3)
        }
    } 

    if (pSPARC->Calc_pres == 1) {
        if(!rank) {
            pSPARC->pres_xc += (stress_mGGA_psi_cartesian[0] + stress_mGGA_psi_cartesian[3] + stress_mGGA_psi_cartesian[5]);
        }
    }
    free(temp_k);
    
    free(stress_mGGA_psi);
    free(stress_mGGA_psi_cartesian);
}