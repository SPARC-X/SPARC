/***
 * @file    exactExchangePressure.h
 * @brief   This file contains the function declarations for Exact Exchange pressure.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGEPRESSURE_H
#define EXACTEXCHANGEPRESSURE_H

#include "isddft.h"

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_linear(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_linear_nonACE(SPARC_OBJ *pSPARC, 
    double *psi_outer, int ldpo, double *psi, int ldp, double *occ, double *pres_exx);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_linear_ACE(SPARC_OBJ *pSPARC, 
    double *psi, int ldp, double *occ, double *pres_exx);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void solve_allpair_poissons_equation_pressure(SPARC_OBJ *pSPARC, 
    double *psi_storage, int ldps, double *psi, int ldp, double *occ, int Nband_source, 
    int band_start_indx_source, double *pres_exx);


/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt_nonACE(SPARC_OBJ *pSPARC, 
    double _Complex *psi_outer, int ldpo, double _Complex *psi, int ldp, double *occ_outer, double *pres_exx);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void Calculate_exact_exchange_pressure_kpt_ACE(SPARC_OBJ *pSPARC, 
    double _Complex *psi, int ldp, double *occ_outer, double *pres_exx);

/**
 * @brief   Calculate Exact Exchange pressure
 */
void solve_allpair_poissons_equation_pressure_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_storage, int ldps, 
    double _Complex *psi, int ldp, double *occ, int Nband_source, int band_start_indx_source, int kpt_k, int kpt_q, double *pres_exx);

#endif // EXACTEXCHANGEPRESSURE_H 