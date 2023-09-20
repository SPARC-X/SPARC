/**
 * @file    forces.h
 * @brief   This file contains the function declarations for calculating forces.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef FORCES_H
#define FORCES_H 


#include "isddft.h"


/**
 * @brief    Calculate atomic forces.
 */
void Calculate_EGS_Forces(SPARC_OBJ *pSPARC);

/**
 * @brief    Calculate local force components.
 */
void Calculate_local_forces(SPARC_OBJ *pSPARC);
void Calculate_local_forces_linear(SPARC_OBJ *pSPARC);

/**
 * @brief    Calculate xc force components for nonlinear core 
 *           correction (NLCC).
 */ 
void Calculate_forces_xc(SPARC_OBJ *pSPARC, double *forces_xc);
void Calculate_forces_xc_linear(SPARC_OBJ *pSPARC, double *forces_xc);

/**
 * @brief    Calculate nonlocal force components.
 */
void Calculate_nonlocal_forces(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_forces_linear(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_forces_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate <Psi_n, Chi_Jlm> for spinor force
 */
void Compute_Integral_psi_Chi(SPARC_OBJ *pSPARC, double *beta, double *Xorb);
void Compute_Integral_psi_Chi_kpt(SPARC_OBJ *pSPARC, double _Complex *beta, double _Complex *Xorb_kpt, int kpt, char *option);

/**
 * @brief   Calculate <Chi_Jlm, DPsi_n> for spinor force
 */
void Compute_Integral_Chi_Dpsi(SPARC_OBJ *pSPARC, double *dpsi, double *beta);
void Compute_Integral_Chi_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi, double _Complex *beta, int kpt, char *option);

/**
 * @brief   Compute nonlocal forces using stored integrals
 */
void Compute_force_nloc_by_integrals(SPARC_OBJ *pSPARC, double *force_nloc, double *alpha);
void Compute_force_nloc_by_integrals_kpt(SPARC_OBJ *pSPARC, double *force_nloc, double _Complex *alpha, char *option);

/**
 * @brief    Symmetrize the force components so that sum of forces is zero.
 */
void Symmetrize_forces(SPARC_OBJ *pSPARC);

#endif // FORCES_H 
