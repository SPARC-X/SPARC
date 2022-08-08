/**
 * @file    vdWDFstress.h
 * @brief   This file contains declaration of functions for computing stress from vdF-DF1 and vdW-DF2 non-linear correlation.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Dion, Max, Henrik Rydberg, Elsebeth Schröder, David C. Langreth, and Bengt I. Lundqvist. 
 * "Van der Waals density functional for general geometries." 
 * Physical review letters 92, no. 24 (2004): 246401.
 * Román-Pérez, Guillermo, and José M. Soler. 
 * "Efficient implementation of a van der Waals density functional: application to double-wall carbon nanotubes." 
 * Physical review letters 103, no. 9 (2009): 096102.
 * Lee, Kyuho, Éamonn D. Murray, Lingzhu Kong, Bengt I. Lundqvist, and David C. Langreth. 
 * "Higher-accuracy van der Waals density functional." Physical Review B 82, no. 8 (2010): 081101.
 * Thonhauser, T., S. Zuluaga, C. A. Arter, K. Berland, E. Schröder, and P. Hyldgaard. 
 * "Spin signature of nonlocal correlation binding in metal-organic frameworks." 
 * Physical review letters 115, no. 13 (2015): 136402.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef vdWDF_STR
#define vdWDF_STR

#include "isddft.h"
#include "exchangeCorrelation.h"

/**
 * @brief compute the stress term containing grad(rho)
 */
void vdWDF_stress_gradient(SPARC_OBJ *pSPARC, double *stressGrad);

/**
 * @brief compute the stress term containing grad(rho), spin-polarized case
 */
void spin_vdWDF_stress_gradient(SPARC_OBJ *pSPARC, double *stressGrad);

/**
 * @brief compute the value of all derivative of kernel functions (210 functions for all model energy ratio pairs) on every reciprocal grid point
 * based on the distance between the reciprocal point and the center O, called by vdWDF_stress_kernel
 */
void interpolate_dKerneldK(SPARC_OBJ *pSPARC, double **dKerneldLength);

/**
 * @brief compute the stress term containing derivative of kernel functions
 */
void vdWDF_stress_kernel(SPARC_OBJ *pSPARC, double *stressKernel);

/*
 Functions below is for calculating vdWDF stress, called by Calculate_XC_stress in stress.c
*/
void Calculate_XC_stress_vdWDF(SPARC_OBJ *pSPARC);

#endif