/***
 * @file    exactExchangeStress.h
 * @brief   This file contains the function declarations for Exact Exchange Stress.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGESTRESS_H
#define EXACTEXCHANGESTRESS_H

#include "isddft.h"


/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_linear(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_linear_nonACE(SPARC_OBJ *pSPARC, 
        double *psi_outer, int ldpo, double *psi, int ldp, double *occ_outer, double *stress_exx);

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_linear_ACE(SPARC_OBJ *pSPARC, 
        double *psi, int ldp, double *occ, double *stress_exx);

/**
 * @brief   Calculate Exact Exchange stress
 */
void solve_allpair_poissons_equation_stress(SPARC_OBJ *pSPARC, 
    double *psi_storage, int ldps, double *psi, int ldp, double *occ, int Nband_source, 
    int band_start_indx_source, double *stress_exx);

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt_nonACE(SPARC_OBJ *pSPARC, 
    double _Complex *psi_outer, int ldpo, double _Complex *psi, int ldp, double *occ_outer, double *stress_exx);

/**
 * @brief   Calculate Exact Exchange stress
 */
void Calculate_exact_exchange_stress_kpt_ACE(SPARC_OBJ *pSPARC, 
    double _Complex *psi, int ldp, double *occ_outer, double *stress_exx);

/**
 * @brief   Calculate Exact Exchange stress
 */
void solve_allpair_poissons_equation_stress_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_storage, int ldps, 
    double _Complex *psi, int ldp, double *occ, int Nband_source, int band_start_indx_source, int kpt_k, int kpt_q, double *stress_exx);

#endif // EXACTEXCHANGESTRESS_H 