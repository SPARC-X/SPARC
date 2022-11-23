/**
 * @file    MGGAstress.h
 * @brief   This file contains the function declarations for metaGGA stress.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MGGA_STR
#define MGGA_STR

#include "isddft.h"

/**
 * @brief   compute the metaGGA psi stress term, gamma point
 */
void Calculate_XC_stress_mGGA_psi_term(SPARC_OBJ *pSPARC);

/**
 * @brief   compute the metaGGA psi stress term, k-point
 */
void Calculate_XC_stress_mGGA_psi_term_kpt(SPARC_OBJ *pSPARC);

// void Calculate_XC_Pres_mGGA_psi_term(SPARC_OBJ *pSPARC);

// void Calculate_XC_Pres_mGGA_psi_term_kpt(SPARC_OBJ *pSPARC);

#endif