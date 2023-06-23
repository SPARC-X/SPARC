/**
 * @file    cyclix_forces.h
 * @brief   This file contains functions for force calculation in systems with cyclix geometry
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#ifndef CYCLIX_FORCES_H
#define CYCLIX_FORCES_H 

#include "isddft.h"

/**
 * @brief    Calculate nonlocal force components for cyclix system with gamma point.
 */
void Calculate_nonlocal_forces_cyclix(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate nonlocal force components for cyclix system with kpts.
 */
void Calculate_nonlocal_forces_kpt_cyclix(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate local force components
 */ 
void Calculate_local_forces_cyclix(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate xc force components
 */ 
void Calculate_forces_xc_cyclix(SPARC_OBJ *pSPARC, double *forces_xc);

/**
* @ brief: function to calculate derivative of the pseudopotential
*/ 
void Dpseudopot_cyclix(SPARC_OBJ *pSPARC, double *VJ, int FDn, int ishift_p, int *pshifty_ex, int *pshiftz_ex, double *DVJ_x_val, double *DVJ_y_val, double *DVJ_z_val, double xs, double ys, double zs);

/**
* @ brief: function to apply rotation matrix on the force vector
*/ 

void Rotate_vector_cyclix(SPARC_OBJ *pSPARC, double *fx, double *fy);


/**
* @ brief: function to apply rotation matrix on the force vector (complex type)
*/ 

void Rotate_vector_complex_cyclix(SPARC_OBJ *pSPARC, double _Complex *fx, double _Complex *fy);


/**
 * @brief    Calculate nonlocal force components with kpts in case of spinor wave function.
 */
void Calculate_nonlocal_forces_kpt_spinor_cyclix(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate <Psi_n, Chi_Jlm> for spinor force
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_psi_Chi_kpt_cyclix(SPARC_OBJ *pSPARC, double _Complex *beta, int spn_i, int kpt, char *option);


void Compute_Integral_Chi_Dpsixy_kpt_cyclix(SPARC_OBJ *pSPARC, 
    double _Complex *dpsix, double _Complex *dpsiy, double _Complex *betax, double _Complex *betay, int spn_i, int kpt, char *option);


void Compute_Integral_Chi_Dpsiz_kpt_cyclix(SPARC_OBJ *pSPARC, double _Complex *dpsi, double _Complex *beta, int spn_i, int kpt, char *option);

#endif // CYCLIX_FORCES_H
