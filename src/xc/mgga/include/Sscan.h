/**
 * @file    Sscan.h
 * @brief   This file contains the function declarations for SCAN functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief   the main function in scan.c, compute \epsilon and potentials of SCAN functional
 *          In this test script, length replaced pSPARC. DMnd is the only variable depends on pSPARC. Maybe pSPARC pointer can be removed.
 * @param rho             electron density vector, structure: 0~Nd_d-1: total electron density; Nd_d~2*Nd_d-1: up density; 2*Nd_d~3*Nd_d-1: down density
 * @param normDrho        length of gradient of electron density |grad n|, structure is similar to rho
 * @param tau             the kinetic density tau vector, structure is similar to rho
 * 
 * @param e_xc            the exchange-correlation energy density epsilon, has one column, epsilon
 * @param vxcMGGA1        d(n epsilon)/d(n), has two columns: d(n epsilon)/dn_up = d (n epsilon_x)/dn_up + d (n epsilon_c)/dn_up; d(n epsilon)/dn_dn = d (n epsilon_x)/dn_dn + d (n epsilon_c)/dn_dn
 * @param vxcMGGA2        d(n epsilon)/d(|grad n|), has three columns: d(n epsilon_c)/d|grad n|, d(n epsilon_x)/d|grad n_up|, d(n epsilon_x)/d|grad n_dn|
 * @param vxcMGGA3        d(n epsilon)/d(tau), has two columns: d(n epsilon)/dtau_up = d(n epsilon_x)/dtau_up + d(n epsilon_c)/dtau; d(n epsilon)/dtau_dn = d(n epsilon_x)/dtau_dn + d(n epsilon_c)/dtau
 */
void SSCAN_EnergyDens_Potential(SPARC_OBJ *pSPARC, double *rho, double *normDrho, double *tau, double *e_xc, double *vxcMGGA1, double *vxcMGGA2, double *vxcMGGA3);

/**
 * @brief   compute important medium variables: s and ds/dn, ds/d|grad n|; \alpha and d\alpha/dn, d\alpha/d(grad n) and d\alpha/d\tau
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector, 0~Nd_d-1: up density; Nd_d~2*Nd_d-1: down density
 * @param normDrho        length of gradient of electron density, 0~Nd_d-1: |grad n_up|; Nd_d~2*Nd_d-1: |grad n_dn|
 * @param tau             the kinetic density tau vector, 0~Nd_d-1: tau_up; Nd_d~2*Nd_d-1: tau_dn
 * 
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|, 0~Nd_d-1: s variables for up; Nd_d~2*Nd_d-1: s variables for down
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau, 0~Nd_d-1: alpha variables for up; Nd_d~2*Nd_d-1: alpha variables for down
 */
void basic_MGSGA_variables_exchange(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau);

/**
 * @brief   compute exchange part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector, 2*n_up ir 2*n_dn
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|, from 2*n_up or 2*n_dn
 * @param alpha_dadn_daddn_dadtau        the medium variable alpha, and d alpha/dn, d alpha/d|grad n|, d alpha/d tau, from 2*up or 2*dn
 * 
 * @param epsilonx        exchange energy density, 0~Nd_d-1: epsilon_up; Nd_d~2*Nd_d-1: epsilon_down
 * @param vx1             0~Nd_d-1: d(n epsilon_x)/dn_up; Nd_d~2*Nd_d-1: d(n epsilon_x)/dn_dn
 * @param vx2             0~Nd_d-1: d(n epsilon_x)/d|grad n_up|; Nd_d~2*Nd_d-1: d(n epsilon_x)/d|grad n_dn|
 * @param vx3             0~Nd_d-1: d(n epsilon_x)/d tau_up; Nd_d~2*Nd_d-1: d(n epsilon_x)/d tau_dn
 */
void sscanx(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx1, double *vx2, double *vx3);

/**
 * @brief   compute important medium variables: s and ds/dn, ds/d|grad n|; \alpha and d\alpha/dn, d\alpha/d(grad n) and d\alpha/d\tau
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector, n = n_up + n_dn
 * @param normDrho        length of gradient of electron density |grad n|
 * @param tau             kinetic density tau vector, tau = tau_up + tau_dn
 * 
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|, from the whole system
 * @param alpha_dadnup_dadndn_daddn_dadtau        the medium variable alpha, and d alpha/dn_up, d alpha/dn_dn, d alpha/d|grad n|, d alpha/d tau
 * @param zeta_dzetadnup_dzetadndn    the medium varible zeta, and d zeta/dn_up, d zeta/dn_dn, from the whole system
 */
void basic_MGSGA_variables_correlation(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn);

/**
 * @brief   compute correlation part of \epsilon and potential of SCAN
 *          
 * @param length          length of the rho vector (also the length of |grad n| and tau)
 * @param rho             electron density vector, n = n_up + n_dn
 * @param s_dsdn_dsddn    the medium variable s, and ds/dn, ds/d|grad n|, from the whole system
 * @param alpha_dadnup_dadndn_daddn_dadtau        the medium variable alpha, and d alpha/dn_up, d alpha/dn_dn, d alpha/d|grad n|, d alpha/d tau
 * @param zeta_dzetadnup_dzetadndn    the medium varible zeta, and d zeta/dn_up, d zeta/dn_dn, from the whole system
 * 
 * @param epsilonc        correlation energy density
 * @param vc1             0~Nd_d-1: d(n epsilon_c)/dn_up; Nd_d~2*Nd_d-1: d(n epsilon_c)/dn_dn
 * @param vc2             d(n epsilon_c)/d|grad n|
 * @param vc3             d(n epsilon_c)/d tau
 */
void sscanc(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc1, double *vc2, double *vc3);

