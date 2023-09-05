/**
 * @file    pressure.h
 * @brief   This file contains the function declarations for calculating pressure.
 *
 * @authors Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef PRESSURE_H
#define PRESSURE_H 


#include "isddft.h"


/**
 * @brief    Calculate quantum mechanical pressure
 */

void Calculate_electronic_pressure(SPARC_OBJ *pSPARC);


/*
* @brief: find pressure contributions from exchange-correlation
*/
void Calculate_XC_pressure(SPARC_OBJ *pSPARC);



/**
 * @brief    Calculate local pressure components.
 */

void Calculate_local_pressure(SPARC_OBJ *pSPARC);
 
void Calculate_local_pressure_orthCell(SPARC_OBJ *pSPARC);
void Calculate_local_pressure_nonorthCell(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_pressure_linear(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_pressure_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate <ChiSC_Jlm, (x-RJ')_beta, DPsi_n> for spinor psi
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_XmRjp_beta_Dpsi(SPARC_OBJ *pSPARC, double *dpsi_xi, double *beta, int dim2);
void Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int kpt, int dim2, char *option);

/**
 * @brief   Compute nonlocal pressure with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_pressure_nloc_by_integrals(SPARC_OBJ *pSPARC, double *pressure_nloc, double *alpha);
void Compute_pressure_nloc_by_integrals_kpt(SPARC_OBJ *pSPARC, double *pressure_nloc, double _Complex *alpha, char *option);

#endif // PRESSURE_H 
