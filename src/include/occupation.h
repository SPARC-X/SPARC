/**
 * @file    occupation.h
 * @brief   This file contains function declarations for calculating 
 *          occupations.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
 
#ifndef OCCUPATION_H
#define OCCUPATION_H

#include "isddft.h"


/**
 * @brief   Find occupation.
 */
double Calculate_occupation(SPARC_OBJ *pSPARC, double x1, double x2, double tol, int max_iter);


/**
 * @brief   Find fermi level using Brent's method.
 *
 * @ref     W.H. Press, Numerical recepies 3rd edition: The art of scientific 
 *          computing, Cambridge university press, 2007.
 */
double Calculate_FermiLevel(SPARC_OBJ *pSPARC, double x1, double x2, double tol, int max_iter, 
    double (*constraint)(SPARC_OBJ*, double));

/**
 * @brief   Collect all eigenvalues from other k-points and the other spin
 */
void collect_all_lambda(SPARC_OBJ *pSPARC, double *totalLambdaArray);

/**
 * @brief   Find fermi level using Brent's method, but calling local_occ_constraint to replace occ_constraint
 *
 * @ref     W.H. Press, Numerical recepies 3rd edition: The art of scientific 
 *          computing, Cambridge university press, 2007.
 */
double local_Calculate_FermiLevel(SPARC_OBJ *pSPARC, double x1, double x2, double *totalLambdaArray, double tol, int max_iter,
                                  double (*constraint)(SPARC_OBJ *, double, double *));
/**
 * @brief   Fermi Dirac function.
 */
//inline double smearing_FermiDirac(double beta, double lambda, double lambda_f);
// static inline double smearing_FermiDirac(double beta, double lambda, double lambda_f)
// {	
//     return 1.0/(1.0+exp(beta*(lambda-lambda_f)));
// }


/**
 * @brief   Gaussian smearing function.
 */
// static inline double smearing_Gaussian(double beta, double lambda, double lambda_f)
// {	
//     return 0.5 * ( 1.0 - erf(beta * (lambda - lambda_f)));
// }


/**
 * @brief   Smearing function.
 *
 *          Currently there are two types of smearing functions implemented, 
 *			Gaussian smearing function and Fermi-Dirac smearing function.
 */
double smearing_function(double beta, double lambda, double lambda_f, int type);


/**
 * @brief   With given fermi energy guess, this function returns the difference  
 *          between the negative charges of the system and the integral of the
 *          occupations in the Brillouin zone.
 *
 */
double occ_constraint(SPARC_OBJ *pSPARC, double lambda_f);

/**
 * @brief   Similar usage to occ_constraint, but this function does not need Allreduce
 *
 */
double local_occ_constraint(SPARC_OBJ *pSPARC, double lambda_f, double *totalLambdaArray);


#endif // OCCUPATION_H
