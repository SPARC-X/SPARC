/**
 * @file    sqProperties.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQPROPERTIES_H
#define SQPROPERTIES_H 

#include "isddft.h"

/**
 * @brief   Calculate nonlocal forces using SQ method
 */
void Calculate_nonlocal_forces_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate nonlocal term of pressure in SQ method
 */
void Calculate_nonlocal_pressure_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate nonlocal and kinetic term of stress in SQ method
 */
void Calculate_nonlocal_kinetic_stress_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute repulsive force correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_forces_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate second derivative of 1 point
 */
double D2_1point(double *vec, int N, double *D2_stencil_coeffs, int FDn, int stride);

/**
 * @brief   Calculate first derivative of 1 point
 */
double D1_1point(double *vec, int N, double *D1_stencil_coeffs, int FDn, int stride);

/**
 * @brief symmetrize SQ stress
 */
void symmetrize_stress_SQ(double *stress_out, double *stress_in);

#endif