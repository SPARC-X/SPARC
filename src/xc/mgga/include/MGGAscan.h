/**
 * @file    MGGAscan.h
 * @brief   This file contains the function declarations for SCAN functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SCAN_H
#define SCAN_H 

#include "isddft.h"

/**
 * @brief   the main function in scan.c, compute \epsilon and potentials of SCAN functional
 *          
 * @param rho             electron density vector
 * @param normDrho        size of gradient of electron density |grad n|
 * @param tau             the kinetic density tau vector
 * @param e_xc            the exchange-correlation energy density epsilon
 * @param vxcMGGA1        d(n epsilon)/d(n)
 * @param vxcMGGA2        d(n epsilon)/d(|grad n|)
 * @param vxcMGGA3        d(n epsilon)/d(tau)
 */
void SCAN_EnergyDens_Potential(SPARC_OBJ *pSPARC, double *rho, double *normDrho, double *tau, double *e_xc, double *vxcMGGA1, double *vxcMGGA2, double *vxcMGGA3);

/**
 * @brief   compute important medium variables: s and ds/dn, ds/d|grad n|; \alpha and d\alpha/dn, d\alpha/d(grad n) and d\alpha/d\tau
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param normDrho        size of gradient of electron density |grad n|
 * @param tau             the kinetic density tau vector
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 */
void basic_MGGA_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau);

/**
 * @brief   compute exchange part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 * @param epsilonx        exchange energy density
 * @param vx1             d(n epsilon_x)/dn
 * @param vx2             d(n epsilon_x)/d|grad n|
 * @param vx3             d(n epsilon_x)/d tau
 */
void scanx(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx1, double *vx2, double *vx3);

/**
 * @brief   compute correlation part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 * @param epsilonc        correlation energy density
 * @param vc1             d(n epsilon_c)/dn
 * @param vc2             d(n epsilon_c)/d|grad n|
 * @param vc3             d(n epsilon_c)/d tau
 */
void scanc(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonc, double *vc1, double *vc2, double *vc3);

#endif // SCAN_H