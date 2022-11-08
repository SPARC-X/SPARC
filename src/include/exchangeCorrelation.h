/**
 * @file    exchangeCorrelation.h
 * @brief   This file declares the functions for calculating exchange-correlation components.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef XC_H
#define XC_H

#include "isddft.h"


/**
* @brief Structure to hold the constants used in xc functional
**/
typedef struct _XCCST_OBJ {
double alpha_zeta2; 
double alpha_zeta;
double beta;
double fsec_inv;
double kappa_pbe;
double mu;
double rsfac;
double kappa;
double mu_divkappa_pbe;
double mu_divkappa;

double ec0_aa; double ec1_aa; double mac_aa;
double ec0_a1; double ec1_a1; double mac_a1;
double ec0_b1; double ec1_b1; double mac_b1;
double ec0_b2; double ec1_b2; double mac_b2;
double ec0_b3; double ec1_b3; double mac_b3;
double ec0_b4; double ec1_b4; double mac_b4;

double piinv;
double third;
double twom1_3;
double sixpi2_1_3;
double sixpi2m1_3;
double threefourth_divpi;
double gamma;
double gamma_inv;
double beta_gamma;
double factf_zeta;
double factfp_zeta;
double coeff_tt;
double sq_rsfac;
double sq_rsfac_inv;
} XCCST_OBJ;


/**
@ brief: function to initialize the constants used in the xc functionals 
**/
void xc_constants_init(XCCST_OBJ *xc_cst, SPARC_OBJ *pSPARC);


/**
* @brief  Calculate exchange correlation potential
**/
void Calculate_Vxc(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate the XC potential using LDA.  
 *
 *          This function calls appropriate XC potential calculation routine.
 */
void Calculate_Vxc_LDA(SPARC_OBJ *pSPARC, double *rho);


/**
 * @brief   Calculate the LSDA Perdew-Wang XC potential.
 * 
 *          This function implements LSDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Vxc_LSDA_PW(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);


/**
 * @brief   Calculate the XC potential using LDA.  
 *
 *          This function implements LDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Vxc_LDA_PW(SPARC_OBJ *pSPARC, double *rho);


/**
 * @brief   Calculate the LDA Perdew-Zunger XC potential.
 *
 *          This function implements LDA Ceperley-Alder Perdew-Zunger 
 *          exchange-correlation potential.
 */
void Calculate_Vxc_LDA_PZ(SPARC_OBJ *pSPARC, double *rho);


/**
 * @brief   Calculate the XC potential using GGA.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_GGA(SPARC_OBJ *pSPARC, double *rho);


/**
 * @brief   Calculate the XC potential using GGA_PBE.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_GGA_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);


/**
 * @brief   Calculate the XC potential using GSGA_PBE.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_GSGA_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);


/**
* @brief  Calculate exchange correlation energy
**/
void Calculate_Exc(SPARC_OBJ *pSPARC, double *electronDens);


/**
 * @brief   Calculate the LDA XC energy.  
 *
 *          This function calls appropriate LDA exchange-correlation energy calculation routine.
 */
void Calculate_Exc_LDA(SPARC_OBJ *pSPARC, double *electronDens);


/**
 * @brief   Calculate the XC energy using LDA.  
 *
 *          This function implements LDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Exc_LDA_PW(SPARC_OBJ *pSPARC, double *electronDens);

/**
 * @brief   Calculate the XC energy using LSDA.  
 *
 *          This function implements LSDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Exc_LSDA_PW(SPARC_OBJ *pSPARC, double *electronDens);

/**
 * @brief   Calculate the LDA Perdew-Zunger XC energy.  
 *
 *          This function implements LDA Ceperley-Alder Perdew-Zunger
 *          exchange-correlation potential.
 */
void Calculate_Exc_LDA_PZ(SPARC_OBJ *pSPARC, double *electronDens);


/**
 * @brief   Calculate the GGA XC energy.  
 *
 *          This function calls appropriate LDA exchange-correlation energy.
 */
void Calculate_Exc_GGA(SPARC_OBJ *pSPARC, double *electronDens);


/**
 * @brief   Calculate the GGA Perdew-Burje-Ernzerhof XC energy.
 */
void Calculate_Exc_GGA_PBE(SPARC_OBJ *pSPARC, double *electronDens);

/**
 * @brief   Calculate the GSGA Perdew-Burje-Ernzerhof XC energy.
 */
void Calculate_Exc_GSGA_PBE(SPARC_OBJ *pSPARC, double *electronDens);

/**
 * @brief   Calculate PBE short ranged exchange
 *          Taken from Quantum Espresson
 */
void pbexsr(double rho, double grho, double omega, double *e_xc_sr, double *XCPotential_sr, double *Dxcdgrho_sr);

/**
 * @brief   Calculate PBE short ranged enhancement factor
 *          Taken from Quantum Espresson
 */
void wpbe_analy_erfc_approx_grad(double rho, double s, double omega, double *Fx_wpbe, double *d1rfx, double *d1sfx);

/**
 * @brief  Calculate exchange correlation energy density
 **/
void Calculate_xc_energy_density(SPARC_OBJ *pSPARC, double *ExcRho);

#endif // XC_H


