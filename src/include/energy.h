/**
 * @file    energy.h
 * @brief   This file contains function declarations for calculating
 *          system energies.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
 
#ifndef ENERGY_H
#define ENERGY_H

#include "isddft.h"


/**
 * @brief   Calculate free energy.
 */
//void Calculate_Free_Energy(SPARC_OBJ *pSPARC, double *ETOT_DIFF, double *EBAND_DIFF);
void Calculate_Free_Energy(SPARC_OBJ *pSPARC, double *electronDens);


/**
 * @brief   Calculate band energy.
 */
double Calculate_Eband(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate electronic entropy.  
 */
double Calculate_electronicEntropy(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate entropy term for the provide eigenvalues.
 *
 *          There are several choices of entropy terms depending on 
 *          what smearing method is used. 
 *          For Fermi-Dirac smearing: 
 *              S(occ) = -[occ * ln(occ) + (1-occ) * ln(1-occ)],
 *          For Gaussian smearing:
 *              S(lambda) = 1/sqrt(pi) * exp(-((lambda - lambda_f)/sigma)^2),
 *          For Methfessel-Paxton with Hermite polynomial of degree 2N,
 *              S(lambda) = 1/2 * A_N * H_2N(x) * exp(-x^2),
 *          where x = (lambda - lambda_f)/sigma, H_n are the Hermite polynomial
 *          of degree n, A_N is given in ref: M. Methfessel and A.T. Paxton (1989). 
 *          Note: when N = 0, MP is equivalent to gaussian smearing. Currently 
 *          not implemented. 
 *
 * @param lambda     Eigenvalue.
 * @param lambda_f   Fermi level.
 * @param occ        Occupation number corresponding to the eigenvalue (only 
 *                   used in Fermi-Dirac smearing).
 * @param beta       beta := 1/sigma.   
 * @param type       Smearing type. type = 0: Fermi-Dirac, type = 1: Gassian (MP0),
 *                   type > 1: Methfessel-Paxton (N = type - 1).
 */
double Calculate_entropy_term(
    const double *lambda,    const double *occ,
    const double lambda_f,   const int n_start,       
    const int n_end,         const double beta,       
    const int type
);


/**
 * @brief   Calculate self consistent correction to free energy.
 *
 *          The energy functional contains a band structure energy term,
 *          where both rho_in and rho_out are involved. This correction 
 *          term is to remove the mixed terms and replace with pure 
 *          rho_out based term.
 *            Escc = integral((Veff[rho_out] - Veff[rho_in]) * rho_out).
 */
//double Calculate_Escc(SPARC_OBJ *pSPARC, double *electronDens);


/**
 * @brief   Calculate self consistent correction to free energy.
 *
 *          The energy functional contains a band structure energy term,
 *          where both rho_in and rho_out are involved. This correction 
 *          term is to remove the mixed terms and replace with pure 
 *          rho_out based term.
 *            Escc = integral((Veff[rho_out] - Veff[rho_in]) * rho_out).
 */
double Calculate_Escc(
    SPARC_OBJ *pSPARC, const int DMnd, const int ncol,
    const double *Veff_out, const double *Veff_in,
    const double *rho_out,  MPI_Comm comm
);


#endif // ENERGY_H


