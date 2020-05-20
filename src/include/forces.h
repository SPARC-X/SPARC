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
//void Calculate_Atomicforces(SPARC_OBJ *pSPARC);
void Calculate_EGS_Forces(SPARC_OBJ *pSPARC);

/**
 * @brief    Calculate local force components.
 */
void Calculate_local_forces(SPARC_OBJ *pSPARC);
void Calculate_local_forces_linear(SPARC_OBJ *pSPARC);

/**
 * @brief    Calculate nonlocal force components.
 */
 
void Calculate_nonlocal_forces(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_forces_linear(SPARC_OBJ *pSPARC);

/**
 * @brief    Calculate nonlocal force components with kpts.
 */
void Calculate_nonlocal_forces_kpt(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_forces_kpt_linear(SPARC_OBJ *pSPARC);

/**
 * @brief    Symmetrize the force components so that sum of forces is zero.
 */
void Symmetrize_forces(SPARC_OBJ *pSPARC);


#endif // FORCES_H 
