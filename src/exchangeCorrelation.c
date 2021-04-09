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
        if(pSPARC->electronDens[i] < pSPARC->xc_rhotol)
            rho[i] = pSPARC->xc_rhotol;
        else
            rho[i] = pSPARC->electronDens[i];
    }

    // for non-linear core correction, use rho+rho_core to evaluate Vxc[rho+rho_core]
    if (pSPARC->NLCC_flag) {
        for (i = 0; i < sz_rho; i++){
            rho[i] += pSPARC->electronDens_core[i];
        }
    }

    if (pSPARC->spin_typ != 0) {
        for(i = 0; i < pSPARC->Nd_d; i++)
            rho[i] = rho[pSPARC->Nd_d + i] + rho[2*pSPARC->Nd_d + i];
    }

    if(strcmpi(pSPARC->XC,"LDA_PW") == 0 || strcmpi(pSPARC->XC,"LDA_PZ") == 0)
    Calculate_Vxc_LDA(pSPARC, rho);
    else if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0)
    Calculate_Vxc_GGA(pSPARC, rho);
    else {
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
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0)
            // Perdew-Burke Ernzerhof exchange-correlation 
            Calculate_Vxc_GGA_PBE(pSPARC, &xc_cst, rho);
        else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }    
    } else {
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0)
            // Perdew-Burke Ernzerhof exchange-correlation 
            Calculate_Vxc_GSGA_PBE(pSPARC, &xc_cst, rho);
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

        if (strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0) {
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
            
			if (strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0) {
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
        if(electronDens[i] < pSPARC->xc_rhotol)
            rho[i] = pSPARC->xc_rhotol;
        else
            rho[i] = electronDens[i];
    }
    
    // for non-linear core correction, use rho+rho_core to evaluate Vxc[rho+rho_core]
    if (pSPARC->NLCC_flag) {
        for (i = 0; i < sz_rho; i++){
            rho[i] += pSPARC->electronDens_core[i];
        }
    }

    if(strcmpi(pSPARC->XC,"LDA_PW") == 0 || strcmpi(pSPARC->XC,"LDA_PZ") == 0)
    Calculate_Exc_LDA(pSPARC, rho);
    else if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0)
    Calculate_Exc_GGA(pSPARC, rho);
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

    double t1, t2, t3;

    t1 = MPI_Wtime();

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
    
    t2 = MPI_Wtime();    

    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;

    t3 = MPI_Wtime();
#ifdef DEBUG
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
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0) {
            // Perdew-Burke-Ernzerhof exchange correlation 
            Calculate_Exc_GGA_PBE(pSPARC, electronDens);
        } else {
            printf("Cannot recognize the XC option provided!\n");
            exit(EXIT_FAILURE);
        }
    } else {
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0) {
            // Perdew-Burke-Ernzerhof exchange correlation 
            Calculate_Exc_GSGA_PBE(pSPARC, electronDens);
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
