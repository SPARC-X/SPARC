/**
 * @file    ExchangeCorrelation.c
 * @brief   This file contains the functions for calculating exchange-correlation components.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "exchangeCorrelation.h"
#include "initialization.h"
#include "isddft.h"
#include "gradVecRoutines.h"
#include "tools.h"
#include "vdWDFexchangeLinearCorre.h"
#include "vdWDFnonlinearCorre.h"
#include "MGGAexchangeCorrelation.h"



/**
@ brief: function to initialize the constants used in the xc functionals 
**/
void xc_constants_init(XCCST_OBJ *xc_cst, SPARC_OBJ *pSPARC) {
    xc_cst->alpha_zeta2 = 1.0 - 1.0e-6; xc_cst->alpha_zeta = 1.0 - 1.0e-6; // ABINIT
    //xc_cst->alpha_zeta2 = 1.0; xc_cst->alpha_zeta = 1.0; // LIBXC
    if(strcmpi(pSPARC->XC,"GGA_PBEsol") == 0) {
		xc_cst->beta = 0.046;
		xc_cst->mu = 10.0/81.0;//0.1235
	} else {
		xc_cst->beta = 0.066725;
    	//xc_cst->beta = 0.06672455060314922;
		xc_cst->mu = 0.2195149727645171;
	}
    xc_cst->fsec_inv = 1.0/1.709921;
    xc_cst->kappa_pbe = 0.804;
    xc_cst->rsfac = 0.6203504908994000;
    xc_cst->kappa = xc_cst->kappa_pbe;
    xc_cst->mu_divkappa_pbe = xc_cst->mu/xc_cst->kappa_pbe;
    xc_cst->mu_divkappa = xc_cst->mu_divkappa_pbe;

    xc_cst->ec0_aa = 0.031091; xc_cst->ec1_aa = 0.015545; xc_cst->mac_aa = 0.016887; // ABINIT
    //xc_cst->ec0_aa = 0.0310907; xc_cst->ec1_aa = 0.01554535; xc_cst->mac_aa = 0.0168869; // LIBXC
    xc_cst->ec0_a1 = 0.21370;  xc_cst->ec1_a1 = 0.20548;  xc_cst->mac_a1 = 0.11125;
    xc_cst->ec0_b1 = 7.5957;  xc_cst->ec1_b1 = 14.1189;  xc_cst->mac_b1 = 10.357;
    xc_cst->ec0_b2 = 3.5876;   xc_cst->ec1_b2 = 6.1977;   xc_cst->mac_b2 = 3.6231;
    xc_cst->ec0_b3 = 1.6382;   xc_cst->ec1_b3 = 3.3662;   xc_cst->mac_b3 = 0.88026;
    xc_cst->ec0_b4 = 0.49294;  xc_cst->ec1_b4 = 0.62517;  xc_cst->mac_b4 = 0.49671;

    // Constants
    xc_cst->piinv = 1.0/M_PI;
    xc_cst->third = 1.0/3.0;
    xc_cst->twom1_3 = pow(2.0,-xc_cst->third);
    xc_cst->sixpi2_1_3 = pow(6.0*M_PI*M_PI,xc_cst->third);
    xc_cst->sixpi2m1_3 = 1.0/xc_cst->sixpi2_1_3;
    xc_cst->threefourth_divpi = (3.0/4.0) * xc_cst->piinv;
    xc_cst->gamma = (1.0 - log(2.0)) * pow(xc_cst->piinv,2.0);
    xc_cst->gamma_inv = 1.0/xc_cst->gamma;
    //beta_gamma = xc_cst->beta * xc_cst->gamma_inv;
    xc_cst->factf_zeta = 1.0/(pow(2.0,(4.0/3.0)) - 2.0);
    xc_cst->factfp_zeta = 4.0/3.0 * xc_cst->factf_zeta * xc_cst->alpha_zeta2;
    xc_cst->coeff_tt = 1.0/(4.0 * 4.0 * xc_cst->piinv * pow(3.0*M_PI*M_PI,xc_cst->third));
    xc_cst->sq_rsfac = sqrt(xc_cst->rsfac);
    xc_cst->sq_rsfac_inv = 1.0/xc_cst->sq_rsfac;
}



/**
* @brief  Calculate exchange correlation potential
**/
void Calculate_Vxc(SPARC_OBJ *pSPARC)
{
    double *rho;
    int sz_rho, i;
    sz_rho = pSPARC->Nd_d * (2*pSPARC->Nspin-1);
    rho = (double *)malloc(sz_rho * sizeof(double) );
    for (i = 0; i < sz_rho; i++){
        rho[i] = pSPARC->electronDens[i];
        // for non-linear core correction, use rho+rho_core to evaluate Vxc[rho+rho_core]
        if (pSPARC->NLCC_flag)
            rho[i] += pSPARC->electronDens_core[i];
        if(rho[i] < pSPARC->xc_rhotol)
            rho[i] = pSPARC->xc_rhotol;
    }

    if (pSPARC->spin_typ != 0) {
        for(i = 0; i < pSPARC->Nd_d; i++)
            rho[i] = rho[pSPARC->Nd_d + i] + rho[2*pSPARC->Nd_d + i];
    }

    if(strcmpi(pSPARC->XC,"LDA_PW") == 0 || strcmpi(pSPARC->XC,"LDA_PZ") == 0)
        Calculate_Vxc_LDA(pSPARC, rho);
    else if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
        || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HSE") == 0
        || strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0)
        Calculate_Vxc_GGA(pSPARC, rho);
    else if(strcmpi(pSPARC->XC,"HF") == 0) {
        if (pSPARC->usefock%2 == 1)
            Calculate_Vxc_GGA(pSPARC, rho);
        else {
            for (i = 0; i < pSPARC->Nd_d; i++) {
                pSPARC->e_xc[i] = 0.0;
                pSPARC->Dxcdgrho[i] = 0.0;
                pSPARC->XCPotential[i] = 0.0;
            }
        }
    } else if(pSPARC->mGGAflag == 1) {
        Calculate_transfer_Vxc_MGGA(pSPARC, rho);
    } else {
        printf("Cannot recognize the XC option provided!\n");
        exit(EXIT_FAILURE);
    }
    free(rho);
}



/**
 * @brief   Calculate the XC potential using LDA.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_LDA(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->spin_typ == 0){ // spin unpolarized
        if(strcmpi(pSPARC->XC,"LDA_PW") == 0) {
            // Perdew-Wang exchange correlation 
            Calculate_Vxc_LDA_PW(pSPARC, rho);
        } else if (strcmpi(pSPARC->XC,"LDA_PZ") == 0) {
            // Perdew-Zunger exchange correlation 
            Calculate_Vxc_LDA_PZ(pSPARC, rho);
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    } else {
        if(strcmpi(pSPARC->XC,"LDA_PW") == 0) {
            // Initialize constants    
            XCCST_OBJ xc_cst;
            xc_constants_init(&xc_cst, pSPARC);
            // Perdew-Wang exchange correlation 
            Calculate_Vxc_LSDA_PW(pSPARC, &xc_cst, rho);
        } else if (strcmpi(pSPARC->XC,"LDA_PZ") == 0) {
            printf("Currently only LDA_PW available\n");
            exit(EXIT_FAILURE);
            // Perdew-Zunger exchange correlation 
            // Calculate_Vxc_LSDA_PZ(pSPARC);
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    }
}



/**
 * @brief   Calculate the LDA Perdew-Wang XC potential.
 * 
 *          This function implements LDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Vxc_LDA_PW(SPARC_OBJ *pSPARC, double *rho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vxc (LDA) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    
    int i;
    double p, A, alpha1, beta1, beta2, beta3, beta4, C3, C31; // parameters (constants)
    double rs, G1, G2, rho_cbrt, rs_sqrt, rs_pow_1p5, rs_pow_p, rs_pow_pplus1;
    
    // correlation parameters
    p = 1.0; 
    A = 0.031091;
    alpha1 = 0.21370;
    beta1 = 7.5957;
    beta2 = 3.5876;
    beta3 = 1.6382;
    beta4 = 0.49294;
    
    // exchange parameter
    C3 = 0.9847450218426965; // (3/pi)^(1/3)
    C31 = 0.6203504908993999; // (3/4pi)^(1/3)
    
    for (i = 0; i < pSPARC->Nd_d; i++) {
        rho_cbrt = cbrt(rho[i]);
        rs = C31 / rho_cbrt; // rs = (3/(4*pi*rho))^(1/3)
        rs_sqrt = sqrt(rs); // rs^0.5
        rs_pow_1p5 = rs * rs_sqrt; // rs^1.5
        rs_pow_p = rs; // rs^p, where p = 1, pow function is slow (~100x slower than add/sub)
        rs_pow_pplus1 = rs_pow_p * rs; // rs^(p+1)
        G1 = log(1.0+1.0/(2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1 )));
        G2 = 2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1);
        
        pSPARC->XCPotential[i] = -2.0*A*(1.0+alpha1*rs) * G1 
	                             -(rs/3.0)*( -2.0*A*alpha1 * G1 + (2.0*A*(1.0+alpha1*rs) * (A*(beta1/rs_sqrt + 2.0*beta2 + 3.0*beta3*rs_sqrt + 2.0*(p+1.0)*beta4*rs_pow_p))) / (G2 * (G2 + 1.0)) );
        pSPARC->XCPotential[i] -= C3 * rho_cbrt; // add exchange potential 
    }
}


/**
 * @brief   Calculate the LSDA Perdew-Wang XC potential.
 * 
 *          This function implements LSDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Vxc_LSDA_PW(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vxc (LSDA_PW) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }

    int DMnd, i, spn_i;
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, zeta, zetp, zetm, zetpm1_3, zetmm1_3, rhotmo6, rhoto6;
    double exc, rhomot, ex_lsd, rs, sqr_rs, rsm1_2;
    double ec0_q0, ec0_q1, ec0_q1p, ec0_den, ec0_log, ecrs0, decrs0_drs, mac_q0, mac_q1, mac_q1p, mac_den, mac_log, macrs, dmacrs_drs;
    double ec1_q0, ec1_q1, ec1_q1p, ec1_den, ec1_log, ecrs1, decrs1_drs, zetp_1_3, zetm_1_3, f_zeta, fp_zeta, zeta4;
    double gcrs, ecrs, dgcrs_drs, decrs_drs, dfzeta4_dzeta, decrs_dzeta, vxcadd;
    
    DMnd = pSPARC->Nd_d;
    for(i = 0; i < DMnd; i++) {
        //pSPARC->electronDens[i] += 1e-50;
        //pSPARC->electronDens[DMnd+i] += 1e-50;
        //pSPARC->electronDens[2*DMnd+i] += 1e-50;
        rhom1_3 = pow(rho[i],-xc_cst->third);
        rhotot_inv = pow(rhom1_3,3.0);
        zeta = (rho[DMnd+i] - rho[2*DMnd+i]) * rhotot_inv;
        zetp = 1.0 + zeta * xc_cst->alpha_zeta;
        zetm = 1.0 - zeta * xc_cst->alpha_zeta;
        zetpm1_3 = pow(zetp,-xc_cst->third);
        zetmm1_3 = pow(zetm,-xc_cst->third);
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // -----------------------------------------------------------------------
        // First take care of the exchange part of the functional
        exc = 0.0;
        for(spn_i = 0; spn_i < 2; spn_i++){
            rho_updn = rho[DMnd + spn_i*DMnd + i]; 
            rho_updnm1_3 = pow(rho_updn, -xc_cst->third);
            rhomot = rho_updnm1_3;
            ex_lsd = -xc_cst->threefourth_divpi * xc_cst->sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            pSPARC->XCPotential[spn_i*DMnd + i] = (4.0/3.0) * ex_lsd;
            exc += ex_lsd * rho_updn;
        }
        pSPARC->e_xc[i] = exc * rhotot_inv;

        // -----------------------------------------------------------------------------
        // Then takes care of the LSD correlation part of the functional

        rs = xc_cst->rsfac * rhom1_3;
        sqr_rs = xc_cst->sq_rsfac * rhotmo6;
        rsm1_2 = xc_cst->sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD

        ec0_q0 = -2.0 * xc_cst->ec0_aa * (1.0 + xc_cst->ec0_a1 * rs);
        ec0_q1 = 2.0 * xc_cst->ec0_aa *(xc_cst->ec0_b1 * sqr_rs + xc_cst->ec0_b2 * rs + xc_cst->ec0_b3 * rs * sqr_rs + xc_cst->ec0_b4 * rs * rs);
        ec0_q1p = xc_cst->ec0_aa * (xc_cst->ec0_b1 * rsm1_2 + 2.0 * xc_cst->ec0_b2 + 3.0 * xc_cst->ec0_b3 * sqr_rs + 4.0 * xc_cst->ec0_b4 * rs);
        ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        ecrs0 = ec0_q0 * ec0_log;
        decrs0_drs = -2.0 * xc_cst->ec0_aa * xc_cst->ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        mac_q0 = -2.0 * xc_cst->mac_aa * (1.0 + xc_cst->mac_a1 * rs);
        mac_q1 = 2.0 * xc_cst->mac_aa * (xc_cst->mac_b1 * sqr_rs + xc_cst->mac_b2 * rs + xc_cst->mac_b3 * rs * sqr_rs + xc_cst->mac_b4 * rs * rs);
        mac_q1p = xc_cst->mac_aa * (xc_cst->mac_b1 * rsm1_2 + 2.0 * xc_cst->mac_b2 + 3.0 * xc_cst->mac_b3 * sqr_rs + 4.0 * xc_cst->mac_b4 * rs);
        mac_den = 1.0/(mac_q1 * mac_q1 + mac_q1);
        mac_log = -log( mac_q1 * mac_q1 * mac_den );
        macrs = mac_q0 * mac_log;
        dmacrs_drs = -2.0 * xc_cst->mac_aa * xc_cst->mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

        ec1_q0 = -2.0 * xc_cst->ec1_aa * (1.0 + xc_cst->ec1_a1 * rs);
        ec1_q1 = 2.0 * xc_cst->ec1_aa * (xc_cst->ec1_b1 * sqr_rs + xc_cst->ec1_b2 * rs + xc_cst->ec1_b3 * rs * sqr_rs + xc_cst->ec1_b4 * rs * rs);
        ec1_q1p = xc_cst->ec1_aa * (xc_cst->ec1_b1 * rsm1_2 + 2.0 * xc_cst->ec1_b2 + 3.0 * xc_cst->ec1_b3 * sqr_rs + 4.0 * xc_cst->ec1_b4 * rs);
        ec1_den = 1.0/(ec1_q1 * ec1_q1 + ec1_q1);
        ec1_log = -log( ec1_q1 * ec1_q1 * ec1_den );
        ecrs1 = ec1_q0 * ec1_log;
        decrs1_drs = -2.0 * xc_cst->ec1_aa * xc_cst->ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;
        
        // xc_cst->alpha_zeta is introduced in order to remove singularities for fully polarized systems.
        zetp_1_3 = (1.0 + zeta * xc_cst->alpha_zeta) * pow(zetpm1_3,2.0);
        zetm_1_3 = (1.0 - zeta * xc_cst->alpha_zeta) * pow(zetmm1_3,2.0);

        f_zeta = ( (1.0 + zeta * xc_cst->alpha_zeta2) * zetp_1_3 + (1.0 - zeta * xc_cst->alpha_zeta2) * zetm_1_3 - 2.0 ) * xc_cst->factf_zeta;
        fp_zeta = ( zetp_1_3 - zetm_1_3 ) * xc_cst->factfp_zeta;
        zeta4 = pow(zeta, 4.0);

        gcrs = ecrs1 - ecrs0 + macrs * xc_cst->fsec_inv;
        ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * xc_cst->fsec_inv);
        dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * xc_cst->fsec_inv;
        decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * xc_cst->fsec_inv);
        dfzeta4_dzeta = 4.0 * pow(zeta,3.0) * f_zeta + fp_zeta * zeta4;
        decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * xc_cst->fsec_inv;

        pSPARC->e_xc[i] += ecrs;
        vxcadd = ecrs - rs * xc_cst->third * decrs_drs - zeta * decrs_dzeta;
        pSPARC->XCPotential[i] += vxcadd + decrs_dzeta;
        pSPARC->XCPotential[DMnd+i] += vxcadd - decrs_dzeta;
    }
}



/**
 * @brief   Calculate the LDA Perdew-Zunger XC potential.
 *
 *          This function implements LDA Ceperley-Alder Perdew-Zunger 
 *          exchange-correlation potential.
 */
void Calculate_Vxc_LDA_PZ(SPARC_OBJ *pSPARC, double *rho) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int  i;
    double C3, rhoi, rs;
    double A,B,C,D,gamma1,beta1,beta2,Vxci;

    A = 0.0311;
    B = -0.048 ;
    C = 0.002 ;
    D = -0.0116 ;
    gamma1 = -0.1423 ;
    beta1 = 1.0529 ;
    beta2 = 0.3334 ; 

    C3 = 0.9847450218427;

    for (i = 0; i < pSPARC->Nd_d; i++) {
        rhoi = rho[i];
        //if (rhoi == 0.0) {
        //    Vxci = 0.0 ;     
        //} else {	     
        rs = pow(0.75/(M_PI*rhoi),(1.0/3.0));
        if (rs<1.0) {
            Vxci = log(rs)*(A+(2.0/3.0)*C*rs) + (B-(1.0/3.0)*A) + (1.0/3.0)*(2.0*D-C)*rs; 
        } else {
            Vxci = (gamma1 + (7.0/6.0)*gamma1*beta1*pow(rs,0.5) + (4.0/3.0)*gamma1*beta2*rs)/pow(1+beta1*pow(rs,0.5)+beta2*rs,2.0) ;		  
        }
        //} 	
        Vxci = Vxci - C3*pow(rhoi,1.0/3.0) ;     
        pSPARC->XCPotential[i] = Vxci;        
    }
}



/**
 * @brief   Calculate the XC potential using GGA.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_GGA(SPARC_OBJ *pSPARC, double *rho)
{
    // Initialize constants    
    XCCST_OBJ xc_cst;
    xc_constants_init(&xc_cst, pSPARC);

    if (pSPARC->spin_typ == 0) { // spin unpolarized
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0) 
            // Perdew-Burke Ernzerhof exchange-correlation 
            Calculate_Vxc_GGA_PBE(pSPARC, &xc_cst, rho);
        else if (strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0) { // it can also be replaced by pSPARC->vdWDFFlag != 0
            Calculate_Vxc_vdWExchangeLinearCorre(pSPARC, &xc_cst, rho); // compute energy and potential of Zhang-Yang revised PBE exchange + LDA PW91 correlation
            Calculate_nonLinearCorr_E_V_vdWDF(pSPARC, rho); // the function is in /vdW/vdWDF/vdWDF.c
        }
        else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }    
    } else {
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0)
            // Perdew-Burke Ernzerhof exchange-correlation 
            Calculate_Vxc_GSGA_PBE(pSPARC, &xc_cst, rho);
        else if (strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0) {
            Calculate_Vxc_vdWExchangeLinearCorre(pSPARC, &xc_cst, rho); // compute energy and potential of Zhang-Yang revised PBE exchange + LDA PW91 correlation
            Calculate_nonLinearCorr_E_V_SvdWDF(pSPARC, rho); // the function is in /vdW/vdWDF/vdWDF.c
        }
        else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    }
}


/**
 * @brief   Calculate the XC potential using GGA_PBE.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_GGA_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vxc (GGA_PBE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    
    //double beta, kappa_pbe, mu, rsfac, kappa, mu_divkappa_pbe, mu_divkappa;
    //double ec0_aa, ec0_a1, ec0_b1, ec0_b2, ec0_b3, ec0_b4;
    //double pi, piinv, third, twom1_3, sixpi2_1_3, sixpi2m1_3, threefourth_divpi, gamma, gamma_inv, coeff_tt, sq_rsfac, sq_rsfac_inv;
    double rho_updn, rho_updnm1_3, rhom1_3, rhotot_inv, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, exc, rs, sqr_rs, rsm1_2;
    double ec0_q0, ec0_q1, ec0_q1p, ec0_den, ec0_log, ecrs0, decrs0_drs, ecrs, decrs_drs;
    double phi_zeta_inv, phi3_zeta, gamphi3inv, bb, dbb_drs, exp_pbe, cc, dcc_dbb, dcc_drs, coeff_aa, aa, daa_drs;
    double grrho2, dtt_dg, tt, xx, dxx_drs, dxx_dtt, pade_den, pade, dpade_dxx, dpade_drs, dpade_dtt, coeff_qq, qq, dqq_drs, dqq_dtt;
    double arg_rr, div_rr, rr, drr_dqq, drr_drs, drr_dtt, hh, dhh_dtt, dhh_drs, drhohh_drho; 

    double temp1, temp2, temp3;
    int DMnd, i;
    DMnd = pSPARC->Nd_d;
    
    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    Drho_x = (double *) malloc(DMnd * sizeof(double));
    Drho_y = (double *) malloc(DMnd * sizeof(double));
    Drho_z = (double *) malloc(DMnd * sizeof(double));
    DDrho_x = (double *) malloc(DMnd * sizeof(double));
    DDrho_y = (double *) malloc(DMnd * sizeof(double));
    DDrho_z = (double *) malloc(DMnd * sizeof(double));
    sigma = (double *) malloc(DMnd * sizeof(double));
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);
    
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        lapcT = (double *) malloc(6 * sizeof(double));
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(i = 0; i < DMnd; i++){
            sigma[i] = Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                       Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i]); 
        }
        free(lapcT);
    } else {
        for(i = 0; i < DMnd; i++){
            sigma[i] = Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i];
        }
    }

    // Compute exchange and correlation
    //phi_zeta = 1.0;
    //phip_zeta = 0.0;
    phi_zeta_inv = 1.0;
    //phi_logder = 0.0;
    phi3_zeta = 1.0;
    gamphi3inv = xc_cst->gamma_inv;
    //phipp_zeta = (-2.0/9.0) * alpha_zeta * alpha_zeta;

    for(i = 0; i < DMnd; i++){
        rho_updn = rho[i]/2.0;
        rho_updnm1_3 = pow(rho_updn, -xc_cst->third);
        rhom1_3 = xc_cst->twom1_3 * rho_updnm1_3;

        rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // First take care of the exchange part of the functional
        rhomot = rho_updnm1_3;
        ex_lsd = -xc_cst->threefourth_divpi * xc_cst->sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        // Perdew-Burke-Ernzerhof GGA, exchange part
        rho_inv = rhomot * rhomot * rhomot;
        coeffss = (1.0/4.0) * xc_cst->sixpi2m1_3 * xc_cst->sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        ss = (sigma[i]/4.0) * coeffss; // s^2

        if (strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0 || strcmpi(pSPARC->XC,"SCAN") == 0) { 
            divss = 1.0/(1.0 + xc_cst->mu_divkappa * ss);
            dfxdss = xc_cst->mu * (divss * divss);
            //d2fxdss2 = -xc_cst->mu * 2.0 * xc_cst->mu_divkappa * (divss * divss * divss);
        } else if (strcmpi(pSPARC->XC,"GGA_RPBE") == 0) {
            divss = exp(-xc_cst->mu_divkappa * ss);
            dfxdss = xc_cst->mu * divss;
            //d2fxdss2 = -xc_cst->mu * xc_cst->mu_divkappa * divss;
        } else {
            printf("Unrecognized GGA functional: %s\n",pSPARC->XC);
            exit(EXIT_FAILURE);
        }
        
        fx = 1.0 + xc_cst->kappa * (1.0 - divss);
            
        ex_gga = ex_lsd * fx;
        dssdn = (-8.0/3.0) * (ss * rho_inv);
        dfxdn = dfxdss * dssdn;
        pSPARC->XCPotential[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);

        dssdg = 2.0 * coeffss;
        dfxdg = dfxdss * dssdg;
        pSPARC->Dxcdgrho[i] = 0.5 * ex_lsd * rho_updn * dfxdg;
        exc = ex_gga * rho_updn;

        // If non spin-polarized, treat spin down contribution now, similar to spin up
        exc = exc * 2.0;
        pSPARC->e_xc[i] = exc * rhotot_inv;

        if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"PBE0") == 0) {
            pSPARC->e_xc[i] *=  (1.0 - pSPARC->exx_frac);
            pSPARC->Dxcdgrho[i] *= (1.0 - pSPARC->exx_frac);
            pSPARC->XCPotential[i] *= (1.0 - pSPARC->exx_frac);
        }

        if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"HSE") == 0) {
            double e_xc_sr, Dxcdgrho_sr, XCPotential_sr;
            // Use the same strategy as \rho for \grho here. 
            // Without this threshold, numerical issue will make simulation fail. 
            if (sigma[i] < 1E-14) sigma[i] = 1E-14;
            pbexsr(rho[i], sigma[i], pSPARC->hyb_range_pbe, &e_xc_sr, &XCPotential_sr, &Dxcdgrho_sr);
            pSPARC->e_xc[i] -=  pSPARC->exx_frac * e_xc_sr / rho[i];
            pSPARC->XCPotential[i] -= pSPARC->exx_frac * XCPotential_sr;
            pSPARC->Dxcdgrho[i] -= pSPARC->exx_frac * Dxcdgrho_sr;
        }
        // WARNING: Dxcdgrho = 0.5 * dvxcdgrho1 here in M-SPARC!! But the same in the end. 

        // Then takes care of the LSD correlation part of the functional
        rs = xc_cst->rsfac * rhom1_3;
        sqr_rs = xc_cst->sq_rsfac * rhotmo6;
        rsm1_2 = xc_cst->sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        ec0_q0 = -2.0 * xc_cst->ec0_aa * (1.0 + xc_cst->ec0_a1 * rs);
        ec0_q1 = 2.0 * xc_cst->ec0_aa * (xc_cst->ec0_b1 * sqr_rs + xc_cst->ec0_b2 * rs + xc_cst->ec0_b3 * rs * sqr_rs + xc_cst->ec0_b4 * rs * rs);
        ec0_q1p = xc_cst->ec0_aa * (xc_cst->ec0_b1 * rsm1_2 + 2.0 * xc_cst->ec0_b2 + 3.0 * xc_cst->ec0_b3 * sqr_rs + 4.0 * xc_cst->ec0_b4 * rs);
        ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        ecrs0 = ec0_q0 * ec0_log;
        decrs0_drs = -2.0 * xc_cst->ec0_aa * xc_cst->ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        ecrs = ecrs0;
        decrs_drs = decrs0_drs;
        //decrs_dzeta = 0.0;
        //zeta = 0.0;

        // Add LSD correlation functional to GGA exchange functional
        pSPARC->e_xc[i] += ecrs;
        pSPARC->XCPotential[i] += ecrs - (rs/3.0) * decrs_drs;

        // Eventually add the GGA correlation part of the PBE functional
        // Note : the computation of the potential in the spin-unpolarized
        // case could be optimized much further. Other optimizations are left to do.

        // From ec to bb
        bb = ecrs * gamphi3inv;
        dbb_drs = decrs_drs * gamphi3inv;
        // dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        exp_pbe = exp(-bb);
        cc = 1.0/(exp_pbe - 1.0);
        dcc_dbb = cc * cc * exp_pbe;
        dcc_drs = dcc_dbb * dbb_drs;
        // dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        coeff_aa = xc_cst->beta * xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv;
        aa = coeff_aa * cc;
        daa_drs = coeff_aa * dcc_drs;
        //daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        grrho2 = sigma[i];
        dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * xc_cst->coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        xx = aa * tt;
        dxx_drs = daa_drs * tt;
        //dxx_dzeta = daa_dzeta * tt;
        dxx_dtt = aa;

        // From xx to pade
        pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        pade = (1.0 + xx) * pade_den;
        dpade_dxx = -xx * (2.0 + xx) * pow(pade_den,2);
        dpade_drs = dpade_dxx * dxx_drs;
        dpade_dtt = dpade_dxx * dxx_dtt;
        //dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        qq = coeff_qq * pade;
        dqq_drs = coeff_qq * dpade_drs;
        dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        //dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        arg_rr = 1.0 + xc_cst->beta * xc_cst->gamma_inv * qq;
        div_rr = 1.0/arg_rr;
        rr = xc_cst->gamma * log(arg_rr);
        drr_dqq = xc_cst->beta * div_rr;
        drr_drs = drr_dqq * dqq_drs;
        drr_dtt = drr_dqq * dqq_dtt;
        //drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        hh = phi3_zeta * rr;
        dhh_drs = phi3_zeta * drr_drs;
        dhh_dtt = phi3_zeta * drr_dtt;
        //dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        pSPARC->e_xc[i] += hh;

        // From hh to the derivative of the energy wrt the density
        drhohh_drho = hh - xc_cst->third * rs * dhh_drs - (7.0/3.0) * tt * dhh_dtt; //- zeta * dhh_dzeta 
        pSPARC->XCPotential[i] += drhohh_drho;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        pSPARC->Dxcdgrho[i] += (rho[i] * dtt_dg * dhh_dtt);    
    }
    
    // for(i = 0; i < DMnd; i++){
    //     if(pSPARC->electronDens[i] == 0.0){
    //         pSPARC->XCPotential[i] = 0.0;
    //         pSPARC->e_xc[i] = 0.0;
    //         pSPARC->Dxcdgrho[i] = 0.0;
    //     }
    // }

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * pSPARC->Dxcdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * pSPARC->Dxcdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * pSPARC->Dxcdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
        for(i = 0; i < DMnd; i++){
            Drho_x[i] *= pSPARC->Dxcdgrho[i];
            Drho_y[i] *= pSPARC->Dxcdgrho[i];
            Drho_z[i] *= pSPARC->Dxcdgrho[i];
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd; i++){
        //if(pSPARC->electronDens[i] != 0.0)
        pSPARC->XCPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
    }

    // Deallocate memory
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
}



/**
 * @brief   Calculate the XC potential using GSGA_PBE.  
 *
 *          This function calls appropriate XC potential.
 */
void Calculate_Vxc_GSGA_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) 
        printf("Start calculating Vxc (GSGA_PBE) ...\n");
#endif 
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return;
    }
    
    double rhom1_3, rhotot_inv, zeta, zetp, zetm, zetpm1_3, zetmm1_3;
    double rho_updn, rho_updnm1_3, rhotmo6, rhoto6, rhomot, ex_lsd, rho_inv, coeffss, ss;
    double divss, dfxdss, fx, ex_gga, dssdn, dfxdn, dssdg, dfxdg, exc, rs, sqr_rs, rsm1_2;
    double ec0_q0, ec0_q1, ec0_q1p, ec0_den, ec0_log, ecrs0, decrs0_drs, mac_q0, mac_q1, mac_q1p, mac_den, mac_log, macrs, dmacrs_drs;
    double ec1_q0, ec1_q1, ec1_q1p, ec1_den, ec1_log, ecrs1, decrs1_drs, zetp_1_3, zetm_1_3, f_zeta, fp_zeta, zeta4;
    double gcrs, ecrs, dgcrs_drs, decrs_drs, dfzeta4_dzeta, decrs_dzeta, vxcadd;
    double phi_zeta, phip_zeta, phi_zeta_inv, phi_logder, phi3_zeta, gamphi3inv;
    double bb, dbb_drs, dbb_dzeta, exp_pbe, cc, dcc_dbb, dcc_drs, dcc_dzeta, coeff_aa, aa, daa_drs, daa_dzeta;
    double grrho2, dtt_dg, tt, xx, dxx_drs, dxx_dzeta, dxx_dtt, pade_den, pade, dpade_dxx, dpade_drs, dpade_dtt, dpade_dzeta;
    double coeff_qq, qq, dqq_drs, dqq_dtt, dqq_dzeta, arg_rr, div_rr, rr, drr_dqq, drr_drs, drr_dtt, drr_dzeta;
    double hh, dhh_dtt, dhh_drs, dhh_dzeta, drhohh_drho;

    double temp1, temp2, temp3;
    int DMnd, i, spn_i;
    DMnd = pSPARC->Nd_d;
    
    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;

    Drho_x = (double *) malloc(3*DMnd * sizeof(double));
    Drho_y = (double *) malloc(3*DMnd * sizeof(double));
    Drho_z = (double *) malloc(3*DMnd * sizeof(double));
    DDrho_x = (double *) malloc(3*DMnd * sizeof(double));
    DDrho_y = (double *) malloc(3*DMnd * sizeof(double));
    DDrho_z = (double *) malloc(3*DMnd * sizeof(double));
    sigma = (double *) malloc(3*DMnd * sizeof(double));
    
    // Gradient of total electron density
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, rho, Drho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, rho, Drho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, rho, Drho_z, 2, pSPARC->dmcomm_phi);
    
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        lapcT = (double *) malloc(6 * sizeof(double));
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(i = 0; i < 3*DMnd; i++){
            sigma[i] = Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                       Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i]);
        }
        free(lapcT);
    } else {
        for(i = 0; i < 3*DMnd; i++){
            sigma[i] = Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i];
        }
    }
    
    for(i = 0; i < DMnd; i++) {
        // pSPARC->electronDens[i] += 1e-50;
        // pSPARC->electronDens[DMnd + i] += 1e-50;
        // pSPARC->electronDens[2*DMnd + i] += 1e-50;
        rhom1_3 = pow(rho[i],-xc_cst->third);
        rhotot_inv = pow(rhom1_3,3.0);
        zeta = (rho[DMnd + i] - rho[2*DMnd + i]) * rhotot_inv;
        zetp = 1.0 + zeta * xc_cst->alpha_zeta;
        zetm = 1.0 - zeta * xc_cst->alpha_zeta;
        zetpm1_3 = pow(zetp,-xc_cst->third);
        zetmm1_3 = pow(zetm,-xc_cst->third);
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // First take care of the exchange part of the functional
        exc = 0.0;
        for(spn_i = 0; spn_i < 2; spn_i++){
            rho_updn = rho[DMnd + spn_i*DMnd + i];
            rho_updnm1_3 = pow(rho_updn, -xc_cst->third);
            rhomot = rho_updnm1_3;
            ex_lsd = -xc_cst->threefourth_divpi * xc_cst->sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            rho_inv = rhomot * rhomot * rhomot;
            coeffss = (1.0/4.0) * xc_cst->sixpi2m1_3 * xc_cst->sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
            ss = sigma[DMnd + spn_i*DMnd + i] * coeffss;
            
			if (strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
                || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0 || strcmpi(pSPARC->XC,"SCAN") == 0) {
            	divss = 1.0/(1.0 + xc_cst->mu_divkappa * ss);
            	dfxdss = xc_cst->mu * (divss * divss);
            	//d2fxdss2 = -xc_cst->mu * 2.0 * xc_cst->mu_divkappa * (divss * divss * divss);
        	} else if (strcmpi(pSPARC->XC,"GGA_RPBE") == 0) {
            	divss = exp(-xc_cst->mu_divkappa * ss);
            	dfxdss = xc_cst->mu * divss;
            	//d2fxdss2 = -xc_cst->mu * xc_cst->mu_divkappa * divss;
        	} else {
                printf("Unrecognized XC functional: %s\n", pSPARC->XC);
                exit(EXIT_FAILURE);
            }
			//divss = 1.0/(1.0 + xc_cst->mu_divkappa * ss);
            //dfxdss = xc_cst->mu * (divss * divss);
            
			fx = 1.0 + xc_cst->kappa * (1.0 - divss);
            ex_gga = ex_lsd * fx;
            dssdn = (-8.0/3.0) * (ss * rho_inv);
            dfxdn = dfxdss * dssdn;
            pSPARC->XCPotential[spn_i*DMnd + i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);

            dssdg = 2.0 * coeffss;
            dfxdg = dfxdss * dssdg;
            pSPARC->Dxcdgrho[DMnd + spn_i*DMnd + i] = ex_lsd * rho_updn * dfxdg;
            exc += ex_gga * rho_updn;

            // Hybrid functional 
            if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"PBE0") == 0) {
                exc -= pSPARC->exx_frac * ex_gga * rho_updn;
                pSPARC->Dxcdgrho[DMnd + spn_i*DMnd + i] *= (1.0 - pSPARC->exx_frac);
                pSPARC->XCPotential[spn_i*DMnd + i] *= (1.0 - pSPARC->exx_frac);
            }

            if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"HSE") == 0) {
                double e_xc_sr, Dxcdgrho_sr, XCPotential_sr;
                // Use the same strategy as \rho for \grho here. 
                // Without this threshold, numerical issue will make simulation fail. 
                if (sigma[DMnd + spn_i*DMnd + i] < 1E-14) sigma[DMnd + spn_i*DMnd + i] = 1E-14;
                pbexsr(rho[DMnd + spn_i*DMnd + i] * 2.0, sigma[DMnd + spn_i*DMnd + i] * 4.0, pSPARC->hyb_range_pbe, &e_xc_sr, &XCPotential_sr, &Dxcdgrho_sr);
                exc -=  pSPARC->exx_frac * e_xc_sr / 2.0;
                pSPARC->XCPotential[spn_i*DMnd + i] -= pSPARC->exx_frac * XCPotential_sr;
                pSPARC->Dxcdgrho[DMnd + spn_i*DMnd + i] -= pSPARC->exx_frac * Dxcdgrho_sr * 2.0;
            }
        }
        pSPARC->e_xc[i] = exc * rhotot_inv;

        // Then takes care of the LSD correlation part of the functional
        rs = xc_cst->rsfac * rhom1_3;
        sqr_rs = xc_cst->sq_rsfac * rhotmo6;
        rsm1_2 = xc_cst->sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        ec0_q0 = -2.0 * xc_cst->ec0_aa * (1.0 + xc_cst->ec0_a1 * rs);
        ec0_q1 = 2.0 * xc_cst->ec0_aa *(xc_cst->ec0_b1 * sqr_rs + xc_cst->ec0_b2 * rs + xc_cst->ec0_b3 * rs * sqr_rs + xc_cst->ec0_b4 * rs * rs);
        ec0_q1p = xc_cst->ec0_aa * (xc_cst->ec0_b1 * rsm1_2 + 2.0 * xc_cst->ec0_b2 + 3.0 * xc_cst->ec0_b3 * sqr_rs + 4.0 * xc_cst->ec0_b4 * rs);
        ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        ecrs0 = ec0_q0 * ec0_log;
        decrs0_drs = -2.0 * xc_cst->ec0_aa * xc_cst->ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        mac_q0 = -2.0 * xc_cst->mac_aa * (1.0 + xc_cst->mac_a1 * rs);
        mac_q1 = 2.0 * xc_cst->mac_aa * (xc_cst->mac_b1 * sqr_rs + xc_cst->mac_b2 * rs + xc_cst->mac_b3 * rs * sqr_rs + xc_cst->mac_b4 * rs * rs);
        mac_q1p = xc_cst->mac_aa * (xc_cst->mac_b1 * rsm1_2 + 2.0 * xc_cst->mac_b2 + 3.0 * xc_cst->mac_b3 * sqr_rs + 4.0 * xc_cst->mac_b4 * rs);
        mac_den = 1.0/(mac_q1 * mac_q1 + mac_q1);
        mac_log = -log( mac_q1 * mac_q1 * mac_den );
        macrs = mac_q0 * mac_log;
        dmacrs_drs = -2.0 * xc_cst->mac_aa * xc_cst->mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

        //zeta = (rho(:,2) - rho(:,3)) .* rhotot_inv;
        ec1_q0 = -2.0 * xc_cst->ec1_aa * (1.0 + xc_cst->ec1_a1 * rs);
        ec1_q1 = 2.0 * xc_cst->ec1_aa * (xc_cst->ec1_b1 * sqr_rs + xc_cst->ec1_b2 * rs + xc_cst->ec1_b3 * rs * sqr_rs + xc_cst->ec1_b4 * rs * rs);
        ec1_q1p = xc_cst->ec1_aa * (xc_cst->ec1_b1 * rsm1_2 + 2.0 * xc_cst->ec1_b2 + 3.0 * xc_cst->ec1_b3 * sqr_rs + 4.0 * xc_cst->ec1_b4 * rs);
        ec1_den = 1.0/(ec1_q1 * ec1_q1 + ec1_q1);
        ec1_log = -log( ec1_q1 * ec1_q1 * ec1_den );
        ecrs1 = ec1_q0 * ec1_log;
        decrs1_drs = -2.0 * xc_cst->ec1_aa * xc_cst->ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;
        
        // xc_cst->alpha_zeta is introduced in order to remove singularities for fully polarized systems.
        zetp_1_3 = (1.0 + zeta * xc_cst->alpha_zeta) * pow(zetpm1_3,2.0);
        zetm_1_3 = (1.0 - zeta * xc_cst->alpha_zeta) * pow(zetmm1_3,2.0);

        f_zeta = ( (1.0 + zeta * xc_cst->alpha_zeta2) * zetp_1_3 + (1.0 - zeta * xc_cst->alpha_zeta2) * zetm_1_3 - 2.0 ) * xc_cst->factf_zeta;
        fp_zeta = ( zetp_1_3 - zetm_1_3 ) * xc_cst->factfp_zeta;
        zeta4 = pow(zeta, 4.0);

        gcrs = ecrs1 - ecrs0 + macrs * xc_cst->fsec_inv;
        ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * xc_cst->fsec_inv);
        dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * xc_cst->fsec_inv;
        decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * xc_cst->fsec_inv);
        dfzeta4_dzeta = 4.0 * pow(zeta,3.0) * f_zeta + fp_zeta * zeta4;
        decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * xc_cst->fsec_inv;

        pSPARC->e_xc[i] += ecrs;
        vxcadd = ecrs - rs * xc_cst->third * decrs_drs - zeta * decrs_dzeta;
        pSPARC->XCPotential[i] += vxcadd + decrs_dzeta;
        pSPARC->XCPotential[DMnd+i] += vxcadd - decrs_dzeta;

        // Eventually add the GGA correlation part of the PBE functional
        // The definition of phi has been slightly changed, because
        // the original PBE one gives divergent behaviour for fully polarized points
        
        phi_zeta = ( zetpm1_3 * (1.0 + zeta * xc_cst->alpha_zeta) + zetmm1_3 * (1.0 - zeta * xc_cst->alpha_zeta)) * 0.5;
        phip_zeta = (zetpm1_3 - zetmm1_3) * xc_cst->third * xc_cst->alpha_zeta;
        phi_zeta_inv = 1.0/phi_zeta;
        phi_logder = phip_zeta * phi_zeta_inv;
        phi3_zeta = phi_zeta * phi_zeta * phi_zeta;
        gamphi3inv = xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv * phi_zeta_inv;        
        
        // From ec to bb
        bb = ecrs * gamphi3inv;
        dbb_drs = decrs_drs * gamphi3inv;
        dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        exp_pbe = exp(-bb);
        cc = 1.0/(exp_pbe - 1.0);
        dcc_dbb = cc * cc * exp_pbe;
        dcc_drs = dcc_dbb * dbb_drs;
        dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        coeff_aa = xc_cst->beta * xc_cst->gamma_inv * phi_zeta_inv * phi_zeta_inv;
        aa = coeff_aa * cc;
        daa_drs = coeff_aa * dcc_drs;
        daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        grrho2 = sigma[i];
        dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * xc_cst->coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        xx = aa * tt;
        dxx_drs = daa_drs * tt;
        dxx_dzeta = daa_dzeta * tt;
        dxx_dtt = aa;

        // From xx to pade
        pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        pade = (1.0 + xx) * pade_den;
        dpade_dxx = -xx * (2.0 + xx) * pow(pade_den,2.0);
        dpade_drs = dpade_dxx * dxx_drs;
        dpade_dtt = dpade_dxx * dxx_dtt;
        dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        qq = coeff_qq * pade;
        dqq_drs = coeff_qq * dpade_drs;
        dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        arg_rr = 1.0 + xc_cst->beta * xc_cst->gamma_inv * qq;
        div_rr = 1.0/arg_rr;
        rr = xc_cst->gamma * log(arg_rr);
        drr_dqq = xc_cst->beta * div_rr;
        drr_drs = drr_dqq * dqq_drs;
        drr_dtt = drr_dqq * dqq_dtt;
        drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        hh = phi3_zeta * rr;
        dhh_drs = phi3_zeta * drr_drs;
        dhh_dtt = phi3_zeta * drr_dtt;
        dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        pSPARC->e_xc[i] += hh;

        // From hh to the derivative of the energy wrt the density
        drhohh_drho = hh - xc_cst->third * rs * dhh_drs - zeta * dhh_dzeta - (7.0/3.0) * tt * dhh_dtt; 
        pSPARC->XCPotential[i] += drhohh_drho + dhh_dzeta;
        pSPARC->XCPotential[DMnd + i] += drhohh_drho - dhh_dzeta;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        pSPARC->Dxcdgrho[i] = (rho[i] * dtt_dg * dhh_dtt);
    }

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < 3*DMnd; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * pSPARC->Dxcdgrho[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * pSPARC->Dxcdgrho[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * pSPARC->Dxcdgrho[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
       for(i = 0; i < 3*DMnd; i++){
            Drho_x[i] *= pSPARC->Dxcdgrho[i];
            Drho_y[i] *= pSPARC->Dxcdgrho[i];
            Drho_z[i] *= pSPARC->Dxcdgrho[i];
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd; i++){
        pSPARC->XCPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i] - DDrho_x[DMnd + i] - DDrho_y[DMnd + i] - DDrho_z[DMnd + i];
        pSPARC->XCPotential[DMnd + i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i] - DDrho_x[2*DMnd + i] - DDrho_y[2*DMnd + i] - DDrho_z[2*DMnd + i];
    }  

    // Deallocate memory
    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);
}




//// old implementation
//void Calculate_Vxc_LDA(SPARC_OBJ *pSPARC) {
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    if (rank == 0) {
//        printf("Start calculating Vxc (LDA) ...\n");
//    }    
//    if (pSPARC->dmcomm_phi == MPI_COMM_NULL)        return; 
//  int  i,j,k;
//  double C3,rhoi;
//  double A,alpha1,beta1,beta2,beta3,beta4,Vxci;
//  int xcor,ycor,zcor,lxdim,lydim,lzdim;
//  double p;
//  p = 1.0; A = 0.031091; alpha1 = 0.21370;
//  beta1 = 7.5957; beta2 = 3.5876; beta3 = 1.6382;  beta4 = 0.49294;
//  C3 = 0.9847450218427; // (3/pi)^(1/3) = 0.9847450218426965

//    xcor = 0;
//    ycor = 0;
//    zcor = 0;
//    lxdim = pSPARC->Nx_d;
//    lydim = pSPARC->Ny_d;
//    lzdim = pSPARC->Nz_d;
//    int count = 0;
//    for(k=zcor; k<zcor+lzdim; k++)
//        for(j=ycor; j<ycor+lydim; j++)
//            for(i=xcor; i<xcor+lxdim; i++)
//                {
//                    rhoi = pSPARC->electronDens[count];
//                    if (rhoi==0) {
//                        Vxci = 0.0;
//                    } else {	
//                        Vxci = pow((0.75/(M_PI*rhoi)),(1.0/3.0)) ; // rs
//                        Vxci = (-2.0*A*(1.0+alpha1*Vxci))*log(1.0+1.0/(2.0*A*(beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0))))) 
//                        - (Vxci/3.0)*(-2.0*A*alpha1*log(1.0+1.0/(2.0*A*(beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0))))) 
//                        - ( (-2.0*A*(1.0+alpha1*Vxci))*(A*( beta1*pow(Vxci,-0.5)+ 2.0*beta2 + 3.0*beta3*pow(Vxci,0.5) + 2.0*(p+1.0)*beta4*pow(Vxci,p) ))) 
//                         /( (2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0)) ) )*
//                            (2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0)) ) )+
//                            (2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0)) ) )
//                          ) 
//                        
//                        ) ;
//                    } 	
//                    Vxci = Vxci - C3*pow(rhoi,1.0/3.0) ; // add exchange potential     
//                    pSPARC->XCPotential[count] = Vxci;  
//                    count++;
//                }

//    if (rank == 0) {
//        printf("rank = %d, cbrt(0.325649642516) = %f, pow(0.325649642516,1.0/3.0) = %f\n", rank, cbrt(0.325649642516), pow(0.325649642516,1.0/3.0));     
//    }
//}


/**
 * @brief  Calculate exchange correlation energy
 **/
void Calculate_Exc(SPARC_OBJ *pSPARC, double *electronDens)
{
    double *rho;
    int sz_rho, i;
    sz_rho = pSPARC->Nd_d * (2*pSPARC->Nspin-1);
    rho = (double *)malloc(sz_rho * sizeof(double) );
    for (i = 0; i < sz_rho; i++){
        rho[i] = pSPARC->electronDens[i];
        // for non-linear core correction, use rho+rho_core to evaluate Vxc[rho+rho_core]
        if (pSPARC->NLCC_flag)
            rho[i] += pSPARC->electronDens_core[i];
        if(rho[i] < pSPARC->xc_rhotol)
            rho[i] = pSPARC->xc_rhotol;
    }

    if(strcmpi(pSPARC->XC,"LDA_PW") == 0 || strcmpi(pSPARC->XC,"LDA_PZ") == 0)
        Calculate_Exc_LDA(pSPARC, rho);
    else if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
        || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0
        || strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0)
        Calculate_Exc_GGA(pSPARC, rho);
    else if(pSPARC->mGGAflag == 1) {
        Calculate_Exc_MGGA(pSPARC, rho);
    }
    else {
        printf("Cannot recognize the XC option provided!\n");
        exit(EXIT_FAILURE);
    }

    free(rho);
}


/**
 * @brief   Calculate the LDA XC energy.  
 *
 *          This function calls appropriate LDA exchange-correlation energy.
 */
void Calculate_Exc_LDA(SPARC_OBJ *pSPARC, double *electronDens)
{
    if(pSPARC->spin_typ == 0) { // spin unpolarized
        if(strcmpi(pSPARC->XC,"LDA_PW") == 0) {
            // Perdew-Wang exchange correlation 
            Calculate_Exc_LDA_PW(pSPARC, electronDens);
        } else if (strcmpi(pSPARC->XC,"LDA_PZ") == 0) {
            // Perdew-Zunger exchange correlation 
            Calculate_Exc_LDA_PZ(pSPARC, electronDens);
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    } else {
        if(strcmpi(pSPARC->XC,"LDA_PW") == 0) {
            // Perdew-Wang exchange correlation 
            Calculate_Exc_LSDA_PW(pSPARC, electronDens);
        } else if (strcmpi(pSPARC->XC,"LDA_PZ") == 0) {
            printf("Under development! Currently only LDA_PW available\n");
            exit(EXIT_FAILURE);
            // Perdew-Zunger exchange correlation 
            //Calculate_Exc_LDA_PZ(pSPARC, electronDens);
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    }    
}



/**
 * @brief   Calculate the XC energy using LDA.  
 *
 *          This function implements LDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Exc_LDA_PW(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    #ifdef DEBUG
    double t1, t2, t3;
    t1 = MPI_Wtime();
    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int i;
    double A, alpha1, beta1, beta2, beta3, beta4, C2, C31;
    double rs, rho_cbrt, rs_sqrt, rs_pow_1p5, rs_pow_p, rs_pow_pplus1, ec, Exc;
    
    // correlation parameters
    // p = 1.0; 
    A = 0.031091;
    alpha1 = 0.21370;
    beta1 = 7.5957;
    beta2 = 3.5876;
    beta3 = 1.6382;
    beta4 = 0.49294;
    
    // exchange parameters
    C2 = 0.738558766382022; // 3/4 * (3/pi)^(1/3)
    C31 = 0.6203504908993999; // (3/4pi)^(1/3)
    
    Exc = 0.0;
    ec = 0.0;
    
    for (i = 0; i < pSPARC->Nd_d; i++) {
        rho_cbrt = cbrt(electronDens[i]);
        // calculate correlation energy
        //if (electronDens[i] != 0.0) {
        rs = C31 / rho_cbrt; // rs = (3/(4*pi*rho))^(1/3)
        rs_sqrt = sqrt(rs); // rs^0.5
        rs_pow_1p5 = rs * rs_sqrt; // rs^1.5
        rs_pow_p = rs; // rs^p, where p = 1, pow function is slow (~100x slower than add/sub)
        rs_pow_pplus1 = rs_pow_p * rs; // rs^(p+1)
        ec = -2.0 * A * (1 + alpha1 * rs) * log(1.0 + 1.0 / (2.0 * A * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs_pow_1p5 + beta4 * rs_pow_pplus1)));
        Exc += ec * electronDens[i];
        //} 
        // add exchange potential 
        Exc -= C2 * rho_cbrt * electronDens[i]; // Ex = -C2 * integral(rho^(4/3))
    }
    Exc *= pSPARC->dV;
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    #endif

    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;

#ifdef DEBUG
    t3 = MPI_Wtime();
    if (!rank) printf("rank = %d, Exc = %18.14f , local calculation time: %.3f ms, Allreduce time: %.3f ms, Total time: %.3f ms\n", rank, Exc,(t2-t1)*1e3, (t3-t2)*1e3, (t3-t1)*1e3);
#endif
}


/**
 * @brief   Calculate the XC energy using LSDA.  
 *
 *          This function implements LSDA Ceperley-Alder Perdew-Wang 
 *          exchange-correlation potential (PW92).
 */
void Calculate_Exc_LSDA_PW(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int i;
    double Exc = 0.0;
    for (i = 0; i < pSPARC->Nd_d; i++) {
        Exc += electronDens[i] * pSPARC->e_xc[i]; 
    }
    
    Exc *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
}




/**
 * @brief   Calculate the LDA Perdew-Zunger XC energy.  
 *
 *          This function implements LDA Ceperley-Alder Perdew-Zunger
 *          exchange-correlation potential.
 */
void Calculate_Exc_LDA_PZ(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int i;
    double A,B,C,D,gamma1,beta1,beta2,C2,Exc,rhoi,ex,ec,rs;

    A = 0.0311;
    B = -0.048 ;
    C = 0.002 ;
    D = -0.0116 ;
    gamma1 = -0.1423 ;
    beta1 = 1.0529 ;
    beta2 = 0.3334 ; 
    C2 = 0.73855876638202;
    
    ec = ex = Exc = 0.0;
    for (i = 0; i < pSPARC->Nd_d; i++) {
        rhoi = electronDens[i];
        ex = -C2*pow(rhoi,1.0/3.0);
        //if (rhoi == 0) {
        //    Ec = 0.0;
        //} else {
        rs = pow(0.75/(M_PI*rhoi),(1.0/3.0));
        if (rs<1.0) {
            ec = A*log(rs) + B + C*rs*log(rs) + D*rs;
        } else {
            ec = gamma1/(1.0+beta1*pow(rs,0.5)+beta2*rs);
        }
        //}
        Exc += (ex+ec)*rhoi; 
    }
    
    Exc *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
}


/**
 * @brief   Calculate the GGA XC energy.  
 *
 *          This function calls appropriate LDA exchange-correlation energy.
 */
void Calculate_Exc_GGA(SPARC_OBJ *pSPARC, double *electronDens)
{
    if(pSPARC->spin_typ == 0) { // spin unpolarized
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0) {
            // Perdew-Burke-Ernzerhof exchange correlation 
            Calculate_Exc_GGA_PBE(pSPARC, electronDens);
        } else if (strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0) {
            Calculate_Exc_GGA_vdWDF_ExchangeLinearCorre(pSPARC, electronDens); // actually the function has no difference from Calculate_Exc_GGA_PBE. Maybe thery can be unified.
            Add_Exc_vdWDF(pSPARC); // the function is in /vdW/vdWDF/vdWDF.c
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    } else {
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0 || strcmpi(pSPARC->XC,"HSE") == 0) {
            // Perdew-Burke-Ernzerhof exchange correlation 
            Calculate_Exc_GSGA_PBE(pSPARC, electronDens);
        } else if (strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0) {
            Calculate_Exc_GSGA_vdWDF_ExchangeLinearCorre(pSPARC, electronDens); // actually the function has no difference from Calculate_Exc_GSGA_PBE. Maybe thery can be unified.
            Add_Exc_vdWDF(pSPARC); // the function is in /vdW/vdWDF/vdWDF.c
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    }    
}

/**
 * @brief   Calculate the GGA Perdew-Burje-Ernzerhof XC energy.
 */
void Calculate_Exc_GGA_PBE(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int i;
    double Exc = 0.0;
    for (i = 0; i < pSPARC->Nd_d; i++) {
        //if(electronDens[i] != 0)
        Exc += electronDens[i] * pSPARC->e_xc[i]; 
    }
    
    Exc *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
}


/**
 * @brief   Calculate the GSGA Perdew-Burje-Ernzerhof XC energy.
 */
void Calculate_Exc_GSGA_PBE(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int i;
    double Exc = 0.0;
    for (i = 0; i < pSPARC->Nd_d; i++) {
        //if(electronDens[i] != 0)
        Exc += electronDens[i] * pSPARC->e_xc[i]; 
    }
    
    Exc *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
}


/**
 * @brief   Calculate PBE short ranged exchange
 *          Taken from Quantum Espresson
 */
void pbexsr(double rho, double grho, double omega, double *e_xc_sr, double *XCPotential_sr, double *Dxcdgrho_sr)
{
    double us, ax, f1, alpha, rs, vx;
    double aa, rr, ex, s2, s, fx, d1x, d2x, dsdn, dsdg;

    us = 0.161620459673995492;
    ax = -0.738558766382022406;
    f1 = -1.10783814957303361;
    alpha = 2.0/3.0;
        
    rs = pow(rho,1.0/3.0);
    vx = (4.0/3.0)*f1*alpha*rs;
    
    aa = grho;
    rr = 1.0/(rho*rs);
    ex = ax/rr;
    s2 = aa*rr*rr*us*us;
    
    s = sqrt(s2);
    if (s > 8.3)
        s = 8.572844 - 18.796223/s2;
    
    wpbe_analy_erfc_approx_grad(rho, s, omega, &fx, &d1x, &d2x);
    
    *e_xc_sr  = ex*fx;
    dsdn  = -4.0/3.0*s/rho;
    *XCPotential_sr = vx*fx + (dsdn*d2x+d1x)*ex;
    dsdg  = us*rr;
    *Dxcdgrho_sr = ex/sqrt(aa)*dsdg*d2x;
}


/**
 * @brief   Calculate PBE short ranged enhancement factor
 *          Taken from Quantum Espresson
 */
void wpbe_analy_erfc_approx_grad(double rho, double s, double omega, double *Fx_wpbe, double *d1rfx, double *d1sfx)
{
    double r36, r64, r81, r256, r384, r27, r128, r144, r288, r324;
    double r729, r20, r32, r243, r2187, r6561, r40, r12, r25, r30; 
    double r54, r75, r105, r135, r1215, r15309;
    double f12, f13, f14, f32, f34, f94, f98, f1516;
    double ea1, ea2, ea3, ea4, ea5, ea6, ea7, ea8, eb1; 
    double A, A2, A3, A12, A32, A52, B, C, D, E, pi2, srpi; 
    double Ha1, Ha2, Ha3, Ha4, Ha5, Fc1, Fc2, EGa1, EGa2, EGa3; 
    double expei1, expei2, expei3, expei4, EGscut, wcutoff, expfcutoff, xkf; 
    double w, w2, w3, w4, w5, w6, w7, w8, d1rw, X, s2, s3, s4, s5, s6;
    double Hnum, Hden, H, d1sHnum, d1sHden, d1sH, F, d1sF;
    double Hsbw,Hsbw2,Hsbw3,Hsbw4,Hsbw12,Hsbw32,Hsbw52,Hsbw72,d1sHsbw,d1rHsbw;
    double DHsbw,DHsbw2,DHsbw3,DHsbw4,DHsbw5,DHsbw12,DHsbw32,DHsbw52,DHsbw72;
    double DHsbw92,HsbwA94,HsbwA942,HsbwA943,HsbwA945,HsbwA9412,DHs,DHs2,DHs3;
    double DHs4,DHs72,DHs92,d1sDHs,DHsw,DHsw2,DHsw52,DHsw72,d1rDHsw;
    double G_a, d1sG_a, G_b, d1sG_b, EG, d1sEG;
    double term2, d1sterm2, term3, d1sterm3, d1rterm3, term4, d1sterm4, d1rterm4, term5, d1sterm5, d1rterm5;
    double t10 = 0.0, t10d1, d1st10 = 0.0, d1rt10 = 0.0, piexperf, expei;
    double piexperfd1, d1spiexperf, d1rpiexperf, expeid1, d1sexpei, d1rexpei;
    double t1, d1st1, d1rt1, term1, d1sterm1, d1rterm1, term1d1;
    double np1, d1rnp1, np2, d1rnp2, f2, f2d1, d1sf2, d1rf2;
    double f3, f3d1, d1sf3, d1rf3, f4, f4d1, d1sf4, d1rf4, f5, f5d1, d1sf5;
    double d1rf5, f6, f6d1, d1sf6, d1rf6, f7, d1sf7, d1rf7, f8, f8d1, d1sf8, d1rf8;
    double f9, f9d1, d1sf9, d1rf9, t2t9, d1st2t9, d1rt2t9;

    // Real numbers 
    r36 = 36;
    r64 = 64;
    r81 = 81;
    r256 = 256;
    r384 = 384;

    r27 = 27;
    r128 = 128;
    r144 = 144;
    r288 = 288;
    r324 = 324;
    r729  = 729;

    r20 = 20;
    r32 = 32;
    r243 = 243;
    r2187 = 2187;
    r6561 = 6561;
    r40 = 40;

    r12 = 12;
    r25 = 25;
    r30 = 30;
    r54 = 54;
    r75 = 75;
    r105 = 105;
    r135 = 135;
    r1215 = 1215;
    r15309  = 15309;
    
    // General constants
    f12    = 0.5;
    f13    = 1.0/3.0;
    f14    = 0.25;
    f32    = 1.5;
    f34    = 0.75;
    f94    = 2.25;
    f98    = 1.125;
    f1516  = 15.0/16.0;
    pi2    = M_PI*M_PI;
    srpi   = sqrt(M_PI);

    // Constants from fit
    ea1 = -1.128223946706117;
    ea2 = 1.452736265762971;
    ea3 = -1.243162299390327;
    ea4 = 0.971824836115601;
    ea5 = -0.568861079687373;
    ea6 = 0.246880514820192;
    ea7 = -0.065032363850763;
    ea8 = 0.008401793031216;
    eb1 = 1.455915450052607;

    // Constants for PBE hole
    A      =  1.0161144;
    B      = -3.7170836e-1;
    C      = -7.7215461e-2;
    D      =  5.7786348e-1;
    E      = -5.1955731e-2;

    // Constants for fit of H(s) (PBE)
    Ha1    = 9.79681e-3;
    Ha2    = 4.10834e-2;
    Ha3    = 1.87440e-1;
    Ha4    = 1.20824e-3;
    Ha5    = 3.47188e-2;

    // Constants for F(H) (PBE)
    Fc1    = 6.4753871;
    Fc2    = 4.7965830e-1;

    // Constants for polynomial expansion for EG for small s
    EGa1   = -2.628417880e-2;
    EGa2   = -7.117647788e-2;
    EGa3   =  8.534541323e-2;

    // Constants for large x expansion of exp(x)*ei(-x)
    expei1 = 4.03640;
    expei2 = 1.15198;
    expei3 = 5.03627;
    expei4 = 4.19160;

    // Cutoff criterion below which to use polynomial expansion
    EGscut     = 8.0e-2;
    wcutoff    = 1.4e1;
    expfcutoff = 7.0e2;

    xkf    = pow((3*pi2*rho),f13);

    A2 = A*A;
    A3 = A2*A;
    A12 = sqrt(A);
    A32 = A12*A;
    A52 = A32*A;

    w     = omega / xkf;
    w2    = w * w;
    w3    = w2 * w;
    w4    = w2 * w2;
    w5    = w3 * w2;
    w6    = w5 * w;
    w7    = w6 * w;
    w8    = w7 * w;

    d1rw  = -(1.0/(3.0*rho))*w;

    X      = - 8.0/9.0;
    s2     = s*s;
    s3     = s2*s;
    s4     = s2*s2;
    s5     = s4*s;
    s6     = s5*s;

    // Calculate wPBE enhancement factor;
    Hnum    = Ha1*s2 + Ha2*s4;
    Hden    = 1.0 + Ha3*s4 + Ha4*s5 + Ha5*s6;
    H       = Hnum/Hden;
    d1sHnum = 2*Ha1*s + 4*Ha2*s3;
    d1sHden = 4*Ha3*s3 + 5*Ha4*s4 + 6*Ha5*s5;
    d1sH    = (Hden*d1sHnum - Hnum*d1sHden) / (Hden*Hden);
    F      = Fc1*H + Fc2;
    d1sF   = Fc1*d1sH;

    // Change exp1nt of Gaussian if we're using the simple approx.;
    if (w > wcutoff)
        eb1 = 2.0;

    // Calculate helper variables (should be moved later on);
    Hsbw = s2*H + eb1*w2;
    Hsbw2 = Hsbw*Hsbw;
    Hsbw3 = Hsbw2*Hsbw;
    Hsbw4 = Hsbw3*Hsbw;
    Hsbw12 = sqrt(Hsbw);
    Hsbw32 = Hsbw12*Hsbw;
    Hsbw52 = Hsbw32*Hsbw;
    Hsbw72 = Hsbw52*Hsbw;

    d1sHsbw  = d1sH*s2 + 2*s*H;
    d1rHsbw  = 2*eb1*d1rw*w;

    DHsbw = D + s2*H + eb1*w2;
    DHsbw2 = DHsbw*DHsbw;
    DHsbw3 = DHsbw2*DHsbw;
    DHsbw4 = DHsbw3*DHsbw;
    DHsbw5 = DHsbw4*DHsbw;
    DHsbw12 = sqrt(DHsbw);
    DHsbw32 = DHsbw12*DHsbw;
    DHsbw52 = DHsbw32*DHsbw;
    DHsbw72 = DHsbw52*DHsbw;
    DHsbw92 = DHsbw72*DHsbw;

    HsbwA94   = f94 * Hsbw / A;
    HsbwA942  = HsbwA94*HsbwA94;
    HsbwA943  = HsbwA942*HsbwA94;
    HsbwA945  = HsbwA943*HsbwA942;
    HsbwA9412 = sqrt(HsbwA94);

    DHs    = D + s2*H;
    DHs2   = DHs*DHs;
    DHs3   = DHs2*DHs;
    DHs4   = DHs3*DHs;
    DHs72  = DHs3*sqrt(DHs);
    DHs92  = DHs72*DHs;

    d1sDHs = 2*s*H + s2*d1sH;

    DHsw   = DHs + w2;
    DHsw2  = DHsw*DHsw;
    DHsw52 = sqrt(DHsw)*DHsw2;
    DHsw72 = DHsw52*DHsw;

    d1rDHsw = 2*d1rw*w;

    if (s > EGscut) {
        G_a    = srpi*(15*E+6*C*(1+F*s2)*DHs+4*B*(DHs2)+8*A*(DHs3))*(1/(16*DHs72))
                - f34*M_PI*sqrt(A) * exp(f94*H*s2/A)*(1 - erf(f32*s*sqrt(H/A)));
        d1sG_a = (1/r32)*srpi * ((r36*(2*H + d1sH*s) / (A12*sqrt(H/A)))+ (1/DHs92) 
                * (-8*A*d1sDHs*DHs3 - r105*d1sDHs*E-r30*C*d1sDHs*DHs*(1+s2*F)
                +r12*DHs2*(-B*d1sDHs + C*s*(d1sF*s + 2*F)))-((r54*exp(f94*H*s2/A)
                *srpi*s*(2*H+d1sH*s)*erfc(f32*sqrt(H/A)*s))/ A12));
        G_b    = (f1516 * srpi * s2) / DHs72;
        d1sG_b = (15*srpi*s*(4*DHs - 7*d1sDHs*s))/(r32*DHs92);
        EG     = - (f34*M_PI + G_a) / G_b;
        d1sEG  = (-4*d1sG_a*G_b + d1sG_b*(4*G_a + 3*M_PI))/(4*G_b*G_b);

    } else {
        EG    = EGa1 + EGa2*s2 + EGa3*s4;
        d1sEG = 2*EGa2*s + 4*EGa3*s3;   
    }

    // Calculate the terms needed in any case  
    term2 = (DHs2*B + DHs*C + 2*E + DHs*s2*C*F + 2*s2*EG)/(2*DHs3);

    d1sterm2 = (-6*d1sDHs*(EG*s2 + E) + DHs2 * (-d1sDHs*B + s*C*(d1sF*s + 2*F)) 
            + 2*DHs * (2*EG*s - d1sDHs*C + s2 * (d1sEG - d1sDHs*C*F))) / (2*DHs4);

    term3 = - w  * (4*DHsw2*B + 6*DHsw*C + 15*E + 6*DHsw*s2*C*F + 15*s2*EG) / (8*DHs*DHsw52);

    d1sterm3 = w * (2*d1sDHs*DHsw * (4*DHsw2*B + 6*DHsw*C + 15*E + 3*s2*(5*EG + 2*DHsw*C*F)) 
            + DHs * (r75*d1sDHs*(EG*s2 + E) + 4*DHsw2*(d1sDHs*B - 3*s*C*(d1sF*s + 2*F))   
            - 6*DHsw*(-3*d1sDHs*C + s*(10*EG + 5*d1sEG*s - 3*d1sDHs*s*C*F))))    
            / (16*DHs2*DHsw72);

    d1rterm3 = (-2*d1rw*DHsw * (4*DHsw2*B + 6*DHsw*C + 15*E + 3*s2*(5*EG + 2*DHsw*C*F)) 
            + w * d1rDHsw * (r75*(EG*s2 + E) + 2*DHsw*(2*DHsw*B + 9*C + 9*s2*C*F))) 
            / (16*DHs*DHsw72);

    term4 = - w3 * (DHsw*C + 5*E + DHsw*s2*C*F + 5*s2*EG) / (2*DHs2*DHsw52);

    d1sterm4 = (w3 * (4*d1sDHs*DHsw * (DHsw*C + 5*E + s2 * (5*EG + DHsw*C*F))
            + DHs * (r25*d1sDHs*(EG*s2 + E) - 2*DHsw2*s*C*(d1sF*s + 2*F) 
            + DHsw * (3*d1sDHs*C + s*(-r20*EG - 10*d1sEG*s + 3*d1sDHs*s*C*F)))))
            / (4*DHs3*DHsw72);

    d1rterm4 = (w2 * (-6*d1rw*DHsw * (DHsw*C + 5*E + s2 * (5*EG + DHsw*C*F))
            + w * d1rDHsw * (r25*(EG*s2 + E) + 3*DHsw*C*(1 + s2*F))))  
            / (4*DHs2*DHsw72);

    term5 = - w5 * (E + s2*EG) / (DHs3*DHsw52);

    d1sterm5 = (w5 * (6*d1sDHs*DHsw*(EG*s2 + E) + DHs * (-2*DHsw*s * 
            (2*EG + d1sEG*s) + 5*d1sDHs * (EG*s2 + E)))) / (2*DHs4*DHsw72);

    d1rterm5 = (w4 * 5*(EG*s2 + E) * (-2*d1rw*DHsw + d1rDHsw * w)) / (2*DHs3*DHsw72);



    if (s > 0.0 || w > 0.0) {
        t10    = (f12)*A*log(Hsbw / DHsbw);
        t10d1  = f12*A*(1.0/Hsbw - 1.0/DHsbw);
        d1st10 = d1sHsbw*t10d1;
        d1rt10 = d1rHsbw*t10d1;
    }
    
    // Calculate exp(x)*f(x) depending on size of x
    if (HsbwA94 < expfcutoff) {
        piexperf = M_PI*exp(HsbwA94)*erfc(HsbwA9412);
        expei    = exp(HsbwA94)*(-expint(1, HsbwA94));
    } else {
        piexperf = M_PI*(1.0/(srpi*HsbwA9412) - 1.0/(2.0*sqrt(M_PI*HsbwA943)) + 3.0/(4.0*sqrt(M_PI*HsbwA945)));
        expei  = - (1.0/HsbwA94) * (HsbwA942 + expei1*HsbwA94 + expei2)/(HsbwA942 + expei3*HsbwA94 + expei4);
    }

    // Calculate the derivatives (based on the orig. expression)
    piexperfd1  = - (3.0*srpi*sqrt(Hsbw/A))/(2.0*Hsbw) + (9.0*piexperf)/(4.0*A);
    d1spiexperf = d1sHsbw*piexperfd1;
    d1rpiexperf = d1rHsbw*piexperfd1;

    expeid1  = f14*(4.0/Hsbw + (9.0*expei)/A);
    d1sexpei = d1sHsbw*expeid1;
    d1rexpei = d1rHsbw*expeid1;

    // gai 
    if (w == 0) {
        // Fall back to original expression for the PBE hole
        t1 = -f12*A*expei;
        d1st1 = -f12*A*d1sexpei;
        d1rt1 = -f12*A*d1rexpei;
        
        if (s > 0.0) {
            term1    = t1 + t10;
            d1sterm1 = d1st1 + d1st10;
            d1rterm1 = d1rt1 + d1rt10;
            *Fx_wpbe = X * (term1 + term2);
            *d1sfx = X * (d1sterm1 + d1sterm2);
            *d1rfx = X * d1rterm1;
        } else {
            *Fx_wpbe = 1.0;
            // TODO    This is checked to be true for term1
            //         How about the other terms???
            *d1sfx   = 0.0;
            *d1rfx   = 0.0;
        }
        
    } else if (w > wcutoff) {
        // Use simple Gaussian approximation for large w
        term1 = -f12*A*(expei+log(DHsbw)-log(Hsbw));
        term1d1  = - A/(2*DHsbw) - f98*expei;
        d1sterm1 = d1sHsbw*term1d1;
        d1rterm1 = d1rHsbw*term1d1;

        *Fx_wpbe = X * (term1 + term2 + term3 + term4 + term5);
        *d1sfx = X * (d1sterm1 + d1sterm2 + d1sterm3 + d1sterm4 + d1sterm5);
        *d1rfx = X * (d1rterm1 + d1rterm3 + d1rterm4 + d1rterm5);

    } else {
        // For everything else, use the full blown expression
        // First, we calculate the polynomials for the first term
        np1    = -f32*ea1*A12*w + r27*ea3*w3/(8*A12) - r243*ea5*w5/(r32*A32) + r2187*ea7*w7/(r128*A52);
        d1rnp1 = - f32*ea1*d1rw*A12 + (r81*ea3*d1rw*w2)/(8*A12) - 
                (r1215*ea5*d1rw*w4)/(r32*A32) + (r15309*ea7*d1rw*w6)/(r128*A52);
        np2 = -A + f94*ea2*w2 - r81*ea4*w4/(16*A) + r729*ea6*w6/(r64*A2) - r6561*ea8*w8/(r256*A3);
        d1rnp2 =   f12*(9*ea2*d1rw*w) - (r81*ea4*d1rw*w3)/(4*A)    
                + (r2187*ea6*d1rw*w5)/(r32*A2) - (r6561*ea8*d1rw*w7)/(r32*A3);

        // The first term is
        t1    = f12*(np1*piexperf + np2*expei);
        d1st1 = f12*(d1spiexperf*np1 + d1sexpei*np2);
        d1rt1 = f12*(d1rnp2*expei + d1rpiexperf*np1 + d1rexpei*np2 + d1rnp1*piexperf);

        // The factors for the main polynomoal in w and their derivatives
        f2    = (f12)*ea1*srpi*A / DHsbw12;
        f2d1  = - ea1*srpi*A / (4*DHsbw32);
        d1sf2 = d1sHsbw*f2d1;
        d1rf2 = d1rHsbw*f2d1;

        f3    = (f12)*ea2*A / DHsbw;
        f3d1  = - ea2*A / (2*DHsbw2);
        d1sf3 = d1sHsbw*f3d1;
        d1rf3 = d1rHsbw*f3d1;

        f4    =  ea3*srpi*(-f98 / Hsbw12 + f14*A / DHsbw32);
        f4d1  = ea3*srpi*((9/(16*Hsbw32))- (3*A/(8*DHsbw52)));
        d1sf4 = d1sHsbw*f4d1;
        d1rf4 = d1rHsbw*f4d1;

        f5    = ea4*(1/r128) * (-r144*(1/Hsbw) + r64*(1/DHsbw2)*A);
        f5d1  = ea4*((f98/Hsbw2)-(A/DHsbw3));
        d1sf5 = d1sHsbw*f5d1;
        d1rf5 = d1rHsbw*f5d1;

        f6    = ea5*(3*srpi*(3*DHsbw52*(9*Hsbw-2*A) + 4*Hsbw32*A2)) / (r32*DHsbw52*Hsbw32*A);
        f6d1  = ea5*srpi*((r27/(r32*Hsbw52))-(r81/(r64*Hsbw32*A))-((15*A)/(16*DHsbw72)));
        d1sf6 = d1sHsbw*f6d1;
        d1rf6 = d1rHsbw*f6d1;

        f7    = ea6*(((r32*A)/DHsbw3 + (-r36 + (r81*s2*H)/A)/Hsbw2)) / r32;
        d1sf7 = ea6*(3*(r27*d1sH*DHsbw4*Hsbw*s2 + 8*d1sHsbw*A*(3*DHsbw4 - 4*Hsbw3*A) + 
                r54*DHsbw4*s*(Hsbw - d1sHsbw*s)*H))/(r32*DHsbw4*Hsbw3*A);
        d1rf7 = ea6*d1rHsbw*((f94/Hsbw3)-((3*A)/DHsbw4)-((r81*s2*H)/(16*Hsbw3*A)));

        f8    = ea7*(-3*srpi*(-r40*Hsbw52*A3+9*DHsbw72*(r27*Hsbw2-6*Hsbw*A+4*A2))) / (r128 * DHsbw72*Hsbw52*A2);
        f8d1  = ea7*srpi*((r135/(r64*Hsbw72)) + (r729/(r256*Hsbw32*A2))  
                -(r243/(r128*Hsbw52*A))-((r105*A)/(r32*DHsbw92)));
        d1sf8 = d1sHsbw*f8d1;
        d1rf8 = d1rHsbw*f8d1;

        f9    = (r324*ea6*eb1*DHsbw4*Hsbw*A + ea8*(r384*Hsbw3*A3 + DHsbw4*(-r729*Hsbw2 
                + r324*Hsbw*A - r288*A2))) / (r128*DHsbw4*Hsbw3*A2);
        f9d1  = -((r81*ea6*eb1)/(16*Hsbw3*A)) + ea8*((r27/(4*Hsbw4))+(r729/(r128*Hsbw2*A2)) 
                -(r81/(16*Hsbw3*A))-((r12*A/DHsbw5)));
        d1sf9 = d1sHsbw*f9d1;
        d1rf9 = d1rHsbw*f9d1;

        t2t9    = f2*w  + f3*w2 + f4*w3 + f5*w4 + f6*w5 + f7*w6 + f8*w7 + f9*w8;
        d1st2t9 = d1sf2*w + d1sf3*w2 + d1sf4*w3 + d1sf5*w4 
                + d1sf6*w5 + d1sf7*w6 + d1sf8*w7 + d1sf9*w8;
        d1rt2t9 = d1rw*f2 + d1rf2*w + 2*d1rw*f3*w + d1rf3*w2 + 3*d1rw*f4*w2 
                + d1rf4*w3 + 4*d1rw*f5*w3 + d1rf5*w4 + 5*d1rw*f6*w4 
                + d1rf6*w5 + 6*d1rw*f7*w5 + d1rf7*w6 + 7*d1rw*f8*w6 
                + d1rf8*w7 + 8*d1rw*f9*w7 + d1rf9*w8;

        // The final value of term1 for 0 < omega < wcutoff is:
        term1 = t1 + t2t9 + t10;
        d1sterm1 = d1st1 + d1st2t9 + d1st10;
        d1rterm1 = d1rt1 + d1rt2t9 + d1rt10;
        
        // The final value for the enhancement factor and its derivatives is:
        *Fx_wpbe = X * (term1 + term2 + term3 + term4 + term5);
        *d1sfx = X * (d1sterm1 + d1sterm2 + d1sterm3 + d1sterm4 + d1sterm5);
        *d1rfx = X * (d1rterm1 + d1rterm3 + d1rterm4 + d1rterm5);
    }
}


/**
 * @brief  Calculate exchange correlation energy density
 **/
void Calculate_xc_energy_density(SPARC_OBJ *pSPARC, double *ExcRho)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    double *rho;
    int sz_rho, i, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sz_rho = pSPARC->Nd_d;
    rho = (double *)malloc(sz_rho * sizeof(double) );
    for (i = 0; i < sz_rho; i++){
        rho[i] = pSPARC->electronDens[i];
        // for non-linear core correction, use rho+rho_core to evaluate Vxc[rho+rho_core]
        if (pSPARC->NLCC_flag)
            rho[i] += pSPARC->electronDens_core[i];
        if(rho[i] < pSPARC->xc_rhotol)
            rho[i] = pSPARC->xc_rhotol;
    }

    // TODO: Add ExcRho for spin-up and down
    for (i = 0; i < pSPARC->Nd_d; i++) {
        ExcRho[i] = pSPARC->e_xc[i] * rho[i];
    }

    free(rho);
#ifdef DEBUG
    double Exc = 0;
    for (i = 0; i < pSPARC->Nd_d; i++)
        Exc += ExcRho[i];
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    Exc *= pSPARC->dV;
    if (!rank) printf("Exchange correlation energy (without hybrid) from Exc energy density: %f\n", Exc);
#endif
}

