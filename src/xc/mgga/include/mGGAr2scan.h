/**
 * @file    mGGAr2scan.h
 * @brief   This file contains the function declarations for r2SCAN functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef R2SCAN_H
#define R2SCAN_H 

#include "isddft.h"

/**
 * @brief   One of the four main functions in mGGAscan.c, compute \epsilon_x and potentials of exchange of SCAN. Called by Calculate_Vxc in exchangeCorrelation.c
 *          
 * @param DMnd            number of grid points in the processor in phi domain
 * @param rho             electron density vector
 * @param sigma           square of length of gradient of electron density |grad n|^2
 * @param tau             the kinetic density tau vector
 * @param ex              the exchange energy density epsilon_x
 * @param vx              d(n epsilon)/d(n)
 * @param v2x             d(n epsilon)/d(|grad n|)/|grad n|
 * @param v3x             d(n epsilon)/d(tau)
 */
void r2scanx(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

/**
 * @brief   compute important medium variables: s and ds/dn, ds/d|grad n|; \alpha and d\alpha/dn, d\alpha/d(grad n) and d\alpha/d\tau
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param normDrho        size of gradient of electron density |grad n|
 * @param tau             the kinetic density tau vector
 * @param p_dpdn_dpddn    the medium variable p, and dp/dn, dp/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 */
void basic_r2scanx_variables(int length, double *rho, double *normDrho, double *tau, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau);

/**
 * @brief   compute exchange part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param p_dpdn_dpddn    the medium variable p, and dp/dn, dp/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 * @param epsilonx        exchange energy density
 * @param vx1             d(n epsilon_x)/dn
 * @param vx2             d(n epsilon_x)/d|grad n|
 * @param vx3             d(n epsilon_x)/d tau
 */
void Calculate_r2scanx(int length, double *rho, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx, double *v2x, double *v3x);

/**
 * @brief   One of the four main functions in mGGAscan.c, compute \epsilon_c and potentials of correlation of SCAN. Called by Calculate_Vxc in exchangeCorrelation.c
 *          
 * @param DMnd            number of grid points in the processor in phi domain
 * @param rho             electron density vector
 * @param sigma           square of length of gradient of electron density |grad n|^2
 * @param tau             the kinetic density tau vector
 * @param ec              the correlation energy density epsilon_c
 * @param vc              d(n epsilon)/d(n)
 * @param v2c             d(n epsilon)/d(|grad n|)/|grad n|
 * @param v3c             d(n epsilon)/d(tau)
 */
void r2scanc(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

/**
 * @brief   compute important medium variables: s and ds/dn, ds/d|grad n|; \alpha and d\alpha/dn, d\alpha/d(grad n) and d\alpha/d\tau
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param normDrho        size of gradient of electron density |grad n|
 * @param tau             the kinetic density tau vector
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|
 * @param p_dpdn_dpddn    the medium variable p, and dp/dn, dp/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 */
void basic_r2scanc_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau);

/**
 * @brief   compute correlation part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|
 * @param p_dpdn_dpddn    the medium variable p, and dp/dn, dp/d|grad n|
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau
 * @param epsilonc        correlation energy density
 * @param vc              d(n epsilon_c)/dn
 * @param v2c             d(n epsilon_c)/d|grad n|
 * @param v3c             d(n epsilon_c)/d tau
 */
void Calculate_r2scanc(int length, double *rho, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau, double *epsilonc, double *vc, double *v2c, double *v3c);

/**
 * @brief   One of the four main functions in mGGAscan.c, compute \epsilon_x and potentials of exchange of SCAN. Called by Calculate_Vxc in exchangeCorrelation.c
 *          
 * @param DMnd            number of grid points in the processor in phi domain
 * @param rho             electron density vector
 * @param sigma           square of length of gradient of electron density |grad n|^2
 * @param tau             the kinetic density tau vector
 * @param ex              the exchange energy density epsilon_x
 * @param vx              d(n epsilon)/d(n); two vectors: up/dn
 * @param v2x             d(n epsilon)/d(|grad n|)/|grad n|; two vectors: up/dn
 * @param v3x             d(n epsilon)/d(tau); two vectors: up/dn
 */
void r2scanx_spin(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

/**
 * @brief   One of the four main functions in mGGAscan.c, compute \epsilon_c and potentials of correlation of SCAN. Called by Calculate_Vxc in exchangeCorrelation.c
 *          
 * @param DMnd            number of grid points in the processor in phi domain
 * @param rho             electron density vector
 * @param sigma           square of length of gradient of electron density |grad n|^2
 * @param tau             the kinetic density tau vector
 * @param ec              the correlation energy density epsilon_c
 * @param vc              d(n epsilon)/d(n), two vectors: up/dn
 * @param v2c             d(n epsilon)/d(|grad n|)/|grad n|, one vector
 * @param v3c             d(n epsilon)/d(tau), one vector
 */
void r2scanc_spin(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

/**
 * @brief   compute important medium variables: s and ds/dn, ds/d|grad n|; \alpha and d\alpha/dn, d\alpha/d(grad n) and d\alpha/d\tau
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector, n = n_up + n_dn
 * @param normDrho        length of gradient of electron density |grad n|
 * @param tau             kinetic density tau vector, tau = tau_up + tau_dn
 * 
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|, from the whole system
 * @param p_dpdn_dpddn    the medium variable p, and dp/dn, dp/d|grad n|, from the whole system
 * @param alpha_dadnup_dadndn_daddn_dadtau        the medium variable alpha, and d alpha/dn_up, d alpha/dn_dn, d alpha/d|grad n|, d alpha/d tau
 * @param zeta_dzetadnup_dzetadndn    the medium varible zeta, and d zeta/dn_up, d zeta/dn_dn, from the whole system
 */
void basic_r2scanc_spin_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn);

/**
 * @brief   compute correlation part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector, n = n_up + n_dn
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|, from the whole system
 * @param p_dpdn_dpddn    the medium variable p, and dp/dn, dp/d|grad n|, from the whole system
 * @param alpha_dadnup_dadndn_daddn_dadtau        the medium variable alpha, and d alpha/dn_up, d alpha/dn_dn, d alpha/d|grad n|, d alpha/d tau
 * @param zeta_dzetadnup_dzetadndn    the medium varible zeta, and d zeta/dn_up, d zeta/dn_dn, from the whole system
 * 
 * @param epsilonc        correlation energy density
 * @param vc              0~Nd_d-1: d(n epsilon_c)/dn_up; Nd_d~2*Nd_d-1: d(n epsilon_c)/dn_dn
 * @param v2c             d(n epsilon_c)/d|grad n|
 * @param v3c             d(n epsilon_c)/d tau
 */
void Calculate_r2scanc_spin(int length, double *rho, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc, double *v2c, double *v3c);

#endif // R2SCAN_H