/**
 * @file    stress.h
 * @brief   This file contains the function declarations for calculating stress and pressure.
 *
 * @authors Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef STRESS_H
#define STRESS_H 


#include "isddft.h"


/**
 * @brief    Calculate quantum mechanical stress
 */

/* Calculate ionic stress*/
void Calculate_ionic_stress(SPARC_OBJ *pSPARC);

void Calculate_electronic_stress(SPARC_OBJ *pSPARC);

/*
* @brief: find stress contributions from exchange-correlation
*/
void Calculate_XC_stress(SPARC_OBJ *pSPARC);


/**
 * @brief    Calculate local stress components.
 */
void Calculate_local_stress(SPARC_OBJ *pSPARC);




/**
 * @brief    Calculate nonlocal stress components.
 */
void Calculate_nonlocal_kinetic_stress(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_kinetic_stress_linear(SPARC_OBJ *pSPARC);
void Calculate_nonlocal_kinetic_stress_kpt(SPARC_OBJ *pSPARC);

/*
@ brief: function to print stress tensor
*/
void PrintStress (SPARC_OBJ *pSPARC, double *stress, FILE *fp);

#endif // STRESS_H 
