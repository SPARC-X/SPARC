/**
 * @file    sqDensity.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQDENSITY_H
#define SQDENSITY_H 

#include "isddft.h"

/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ(SPARC_OBJ *pSPARC, int SCFcount);

void find_Efermi_SQ(SPARC_OBJ *pSPARC, int SCFcount);

void Calculate_elecDens_Gauss(SPARC_OBJ *pSPARC);


/**
 * @brief   Compute column of density matrix using Gauss Quadrature
 */
void Gauss_density_matrix_col(SPARC_OBJ *pSPARC, int Nd, int npl, double *DMcol, double *V, double *w, double *D);

#endif 