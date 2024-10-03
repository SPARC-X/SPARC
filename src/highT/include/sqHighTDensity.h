/**
 * @file    sqHighTDensity.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQHIGHTDENSITY_H
#define SQHIGHTDENSITY_H 

#include "isddft.h"

/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ_highT(SPARC_OBJ *pSPARC, int SCFcount);

/**
 * @brief   Compute all columns of density matrix using Gauss Quadrature
 */
void calculate_density_matrix_SQ_highT(SPARC_OBJ *pSPARC);

#endif 