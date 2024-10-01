/**
 * @file    sqEnergy.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQENERGY_H
#define SQENERGY_H 

#include "isddft.h"

/**
 * @brief   Calculate Band energy with SQ method
 */
double Calculate_Eband_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate electronic entropy with SQ method
 */
double Calculate_electronicEntropy_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate band energy with Gauss quadrature in SQ method
 */
double Calculate_Eband_SQ_Gauss(SPARC_OBJ *pSPARC, int DMnd, MPI_Comm comm);

/**
 * @brief   Occupation constraints with Gauss quadrature
 */
double occ_constraint_SQ_gauss(SPARC_OBJ *pSPARC, double lambda_f);

/**
 * @brief   Band energy function
 * 
 * TODO:    Can be removed and simply replace the code.
 */
double UbandFunc(double t, double lambda_f, double bet, int type);

/**
 * @brief   Entropy function
 */
double EentFunc(double t, double lambda_f, double bet, int type);


/**
 * @brief   Calculate band energy with Gauss quadrature in SQ method
 */
double Calculate_Eband_SQ_Gauss(SPARC_OBJ *pSPARC, int DMnd, MPI_Comm comm);


/**
 * @brief   Calculate electronic entropy with Gauss quadrature in SQ method
 */
double Calculate_electronicEntropy_SQ_Gauss(SPARC_OBJ *pSPARC, int DMnd, MPI_Comm comm);

/**
 * @brief   Compute repulsive energy correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_SQ(SPARC_OBJ *pSPARC);

#endif