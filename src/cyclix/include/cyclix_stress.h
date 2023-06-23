/**
 * @file    cyclix_stress.h
 * @brief   This file contains the function declarations for calculating stress in 1D periodic cells of cyclix.
 *
 * @author  abhiraj sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef CYCLIX_STRESS_H
#define CYCLIX_STRESS_H 


#include "isddft.h"


/**
 * @brief    Calculate quantum mechanical stress
 */

/* Calculate ionic stress*/
void Calculate_ionic_stress_cyclix(SPARC_OBJ *pSPARC);

/* Calculate stress from the electronic part */
void Calculate_electronic_stress_cyclix(SPARC_OBJ *pSPARC);

/*
* @brief: find stress contributions from exchange-correlation
*/
void Calculate_XC_stress_nlcc_cyclix(SPARC_OBJ *pSPARC, double *stress_xc_nlcc);
void Calculate_XC_stress_cyclix(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate local stress components.
 */
void Calculate_local_stress_cyclix(SPARC_OBJ *pSPARC);

void Dpseudopot_cyclix_z(SPARC_OBJ *pSPARC, double *VJ, int FDn, int ishift_p, int *pshifty, int *pshiftz, double *DVJ_z_val);

/**
 * @brief    Calculate nonlocal stress components for gamma point.
 */
void Calculate_nonlocal_kinetic_stress_cyclix(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate nonlocal stress components for k-points.
 */
void Calculate_nonlocal_kinetic_stress_kpt_cyclix(SPARC_OBJ *pSPARC);

void Calculate_nonlocal_kinetic_stress_kpt_spinor_cyclix(SPARC_OBJ *pSPARC);

void Compute_stress_tensor_nloc_by_integrals_cyclix(SPARC_OBJ *pSPARC, double *stress_nl, double _Complex *alpha, char *option);

void Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt_cyclix(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int spn_i, int kpt, char *option);

#endif // CYCLIX_STRESS_H 
