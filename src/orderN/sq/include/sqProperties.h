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

/**
 * @brief   Calculate electron density using SQ method
 */
void Calculate_elecDens_SQ(SPARC_OBJ *pSPARC, int SCFcount);

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
 * @brief   Calculate nonlocal forces using SQ method
 */
void Calculate_nonlocal_forces_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute chebyshev coefficients Ci (length npl_c+1) to fit the function "func"
 * 
 * NOTE:    THIS FUNCTION CURRENTLY USES DISCRETE ORTHOGONALITY PROPERTY OF CHEBYSHEV POLYNOMIALS WHICH ONLY GIVE "approximate" 
 *          COEFFICIENTS. SO IF THE FUNCTION IS NOT SMOOTH ENOUGH THE COEFFICIENTS WILL NOT BE ACCURATE ENOUGH. 
 *          SO THIS FUNCTION HAS TO BE REPLACED WITH ITS CONTINUOUS COUNTER PART WHICH EVALUATES OSCILLATORY INTEGRALS AND USES FFT.
 */
void ChebyshevCoeff_SQ(int npl_c, double *Ci, double *d, double (*fun)(double, double, double, int), 
                const double beta, const double lambda_f, const int type);

/**
 * @brief   Lanczos algorithm computing only minimal eigenvales
 * 
 * TODO:    There are 2 different Lanczos algorithm with confusing names. Change them or combine them
 */
void LanczosAlgorithm_new(SPARC_OBJ *pSPARC, int i, int j, int k, double *lambda_min, int nd, int choice);

/**
 * @brief   Tridiagonal solver for eigenvalues. Part of Lanczos algorithm.
 */
void TridiagEigenSolve_new(SPARC_OBJ *pSPARC, double *diag, double *subdiag, int n, double *lambda_min, int choice, int nd);

/**
 * @brief   Calculate nonlocal term of pressure in SQ method
 */
void Calculate_nonlocal_pressure_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate nonlocal and kinetic term of stress in SQ method
 */
void Calculate_nonlocal_kinetic_stress_SQ(SPARC_OBJ *pSPARC);


/**
 * @brief   Compute repulsive energy correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute repulsive force correction for atoms having rc-overlap
 * 
 * TODO:    Add implementation from SQDFT 
 */
void OverlapCorrection_forces_SQ(SPARC_OBJ *pSPARC);



void Gauss_density_matrix_col(SPARC_OBJ *pSPARC, int Nd, int npl, double ***DMcol, double *V, double *w, double *D);


void Clenshaw_curtis_density_matrix_col(SPARC_OBJ *pSPARC, double ***DMcol, int i, int j, int k, int nd);

#endif