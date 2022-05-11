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

#endif // PRESSURE_H 
