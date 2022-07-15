/**
 * @file    MGGAscan.c
 * @brief   This file contains the functions for scan functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Sun, Jianwei, Adrienn Ruzsinszky, and John P. Perdew.
 * "Strongly constrained and appropriately normed semilocal density functional." 
 * Physical review letters 115, no. 3 (2015): 036402.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "MGGAscan.h"
#include "isddft.h"


void SCAN_EnergyDens_Potential(SPARC_OBJ *pSPARC, double *rho, double *normDrho, double *tau, double *e_xc, double *vxcMGGA1, double *vxcMGGA2, double *vxcMGGA3) { 
    // the main function in scan.c, compute \epsilon and potentials of SCAN functional
    int DMnd = pSPARC->Nd_d;
    
    double **s_dsdn_dsddn = (double**)malloc(sizeof(double*)*3); // variable s, and ds/dn, ds/d|\nabla n|
    int i;
    for (i = 0; i < 3; i++) {
        s_dsdn_dsddn[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(s_dsdn_dsddn[i] != NULL);
    }
    double **alpha_dadn_daddn_dadtau = (double**)malloc(sizeof(double*)*4); // variable \alpha, and d\alpha/dn, d\alpha/d|\nabla n|, d\alpha/d\tau
    for (i = 0; i < 4; i++) {
        alpha_dadn_daddn_dadtau[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alpha_dadn_daddn_dadtau[i] != NULL);
    }

    basic_MGGA_variables(DMnd, rho, normDrho, tau, s_dsdn_dsddn, alpha_dadn_daddn_dadtau);

    double *epsilonx, *vx1, *vx2, *vx3;
    epsilonx = (double *) malloc(DMnd * sizeof(double));
    vx1 = (double *) malloc(DMnd * sizeof(double));
    vx2 = (double *) malloc(DMnd * sizeof(double));
    vx3 = (double *) malloc(DMnd * sizeof(double));

    scanx(DMnd, rho, s_dsdn_dsddn, alpha_dadn_daddn_dadtau, epsilonx, vx1, vx2, vx3);

    double *epsilonc, *vc1, *vc2, *vc3;
    epsilonc = (double *) malloc(DMnd * sizeof(double));
    vc1 = (double *) malloc(DMnd * sizeof(double));
    vc2 = (double *) malloc(DMnd * sizeof(double));
    vc3 = (double *) malloc(DMnd * sizeof(double));

    scanc(DMnd, rho, s_dsdn_dsddn, alpha_dadn_daddn_dadtau, epsilonc, vc1, vc2, vc3);

    for (i = 0; i < DMnd; i++) {
        e_xc[i] = epsilonx[i] + epsilonc[i];
        vxcMGGA1[i] = vx1[i] + vc1[i];
        vxcMGGA2[i] = vx2[i] + vc2[i];
        vxcMGGA3[i] = vx3[i] + vc3[i];
    }

    for (i = 0; i < 3; i++) {
        free(s_dsdn_dsddn[i]);
    }
    free(s_dsdn_dsddn);
    for (i = 0; i < 4; i++) {
        free(alpha_dadn_daddn_dadtau[i]);
    }
    free(alpha_dadn_daddn_dadtau);

    free(epsilonx); free(vx1); free(vx2); free(vx3);
    free(epsilonc); free(vc1); free(vc2); free(vc3);
}

void basic_MGGA_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau) {
    int i;
    double threeMPi2_1o3 = pow(3.0*M_PI*M_PI, 1.0/3.0);
    double threeMPi2_2o3 = threeMPi2_1o3*threeMPi2_1o3;
    for (i = 0; i < length; i++) {
        s_dsdn_dsddn[0][i] = normDrho[i] / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0));
        double tauw = normDrho[i]*normDrho[i] / (8*rho[i]);
        double tauUnif = 3.0/10.0 * threeMPi2_2o3 * pow(rho[i], 5.0/3.0);
        alpha_dadn_daddn_dadtau[0][i] = (tau[i] - tauw) / tauUnif;

        s_dsdn_dsddn[1][i] = -2.0*normDrho[i] / (3.0 * threeMPi2_1o3 * pow(rho[i], 7.0/3.0)); // ds/dn
        s_dsdn_dsddn[2][i] = 1.0 / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0)); // ds/d|\nabla n|
        double DtauwDn = -normDrho[i]*normDrho[i] / (8*rho[i]*rho[i]);
        double DtauwDDn = normDrho[i] / (4*rho[i]);
        double DtauUnifDn = threeMPi2_2o3 / 2.0 * pow(rho[i], 2.0/3.0);
        alpha_dadn_daddn_dadtau[1][i] = (-DtauwDn*tauUnif - (tau[i] - tauw)*DtauUnifDn) / (tauUnif*tauUnif); // d\alpha/dn
        alpha_dadn_daddn_dadtau[2][i] = (-DtauwDDn) / tauUnif; // d\alpha/d|\nabla n|
        alpha_dadn_daddn_dadtau[3][i] = 1.0 / tauUnif; // d\alpha/d\tau
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, s %.7E, dsdn %.7E, dsddn %.7E, alpha %.7E, dadn %.7E, daddn %.7E, dadtau %.7E\n", i, 
    //         s_dsdn_dsddn[0][i], s_dsdn_dsddn[1][i], s_dsdn_dsddn[2][i], 
    //         alpha_dadn_daddn_dadtau[0][i], alpha_dadn_daddn_dadtau[1][i], alpha_dadn_daddn_dadtau[2][i],alpha_dadn_daddn_dadtau[3][i]);
    // }
}

void scanx(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx1, double *vx2, double *vx3) {
    // constants for h_x^1
    double k1 = 0.065;
    double mu_ak = 10.0/81.0;
    double b2 = sqrt(5913.0/405000.0);
    double b1 = 511.0/13500.0/(2.0*b2);
    double b3 = 0.5;
    double b4 = mu_ak*mu_ak/k1 - 1606.0/18225.0 - b1*b1;
    // constant h_x^0
    double hx0 = 1.174;
    // constants for switching function f_x
    double c1x = 0.667;
    double c2x = 0.8;
    double dx = 1.24;
    // constants for Fx, which is mixing of h_x^0 and h_x^1
    double a1 = 4.9479;
    int i;
    for (i = 0; i < length; i++) {
        double epsilon_xUnif = -3.0/(4.0*M_PI) * pow(3.0*M_PI*M_PI * rho[i], 1.0/3.0);
        // compose h_x^1
        double s = s_dsdn_dsddn[0][i];
        double s2 = s*s;
        double alpha = alpha_dadn_daddn_dadtau[0][i];
        double term1 = 1.0 + b4*s2/mu_ak*exp(-fabs(b4)*s2/mu_ak);
        double xFir = mu_ak*s2 * term1;
        double term3 = 2.0*(b1*s2 + b2*(1.0 - alpha)*exp(-b3*(1.0 - alpha)*(1.0 - alpha)));
        double xSec = (term3/2.0)*(term3/2.0);
        double hx1 = 1.0 + k1 - k1/(1.0 + (xFir + xSec)/k1); // x = xFir + xSec
        double fx;
        if (alpha > 1.0) {
            fx = -dx*exp(c2x / (1.0 - alpha));
        }
        else {
            fx = exp(-c1x*alpha / (1.0 - alpha));
        } // when \alpha == 1.0, fx should be 0.0
        double sqrt_s = sqrt(s);
        double gx = 1.0 - exp(-a1/sqrt_s);
        double Fx = (hx1 + fx*(hx0 - hx1))*gx;
        epsilonx[i] = epsilon_xUnif*Fx;

        double term2 = s2*(b4/mu_ak*exp(-fabs(b4)*s2/mu_ak) + b4*s2/mu_ak*exp(-fabs(b4)*s2/mu_ak)*(-fabs(b4)/mu_ak));
        double term4 = b2*(-exp(-b3*(1.0 - alpha)*(1.0 - alpha)) + (1.0 - alpha)*exp(-b3*(1.0 - alpha)*(1.0 - alpha))*(2*b3*(1.0 - alpha)));
        // printf("point %3d, term1 %.7E, term2 %.7E, term3 %.7E, term4 %.7E\n", i, term1, term2, term3, term4);
        double DxDs = 2*s*(mu_ak*(term1 + term2) + b1*term3);
        double DxDalpha = term3*term4;
        double DxDn = s_dsdn_dsddn[1][i]*DxDs + alpha_dadn_daddn_dadtau[1][i]*DxDalpha;
        double DxDDn = s_dsdn_dsddn[2][i]*DxDs + alpha_dadn_daddn_dadtau[2][i]*DxDalpha;
        double DxDtau = alpha_dadn_daddn_dadtau[3][i]*DxDalpha;
        // printf("point %3d, DxDn %.7E, DxDDn %.7E, DxDtau %.7E\n", i, DxDn, DxDDn, DxDtau);

        double DgxDn = -exp(-a1/sqrt_s)*(a1/2.0/sqrt_s/s)*s_dsdn_dsddn[1][i];
        double DgxDDn = -exp(-a1/sqrt_s)*(a1/2.0/sqrt_s/s)*s_dsdn_dsddn[2][i];
        double Dhx1Dx = 1.0 / (1.0 + (xFir + xSec)/k1) / (1.0 + (xFir + xSec)/k1);
        double Dhx1Dn = DxDn*Dhx1Dx;
        double Dhx1DDn = DxDDn*Dhx1Dx;
        double Dhx1Dtau = DxDtau*Dhx1Dx;
        // printf("point %3d, Dhx1Dn %.7E, Dhx1DDn %.7E, Dhx1Dtau %.7E\n", i, Dhx1Dn, Dhx1DDn, Dhx1Dtau);
        double DfxDalpha;
        if (alpha > 1.0) {
            DfxDalpha = -dx*exp(c2x/(1.0 - alpha)) * (c2x/(1.0 - alpha)/(1.0 - alpha));
        }
        else {
            DfxDalpha = exp(-c1x*alpha/(1.0 - alpha)) * (-c1x/(1.0 - alpha)/(1 - alpha));
        }
        double DfxDn = DfxDalpha*alpha_dadn_daddn_dadtau[1][i];
        double DfxDDn = DfxDalpha*alpha_dadn_daddn_dadtau[2][i];
        double DfxDtau = DfxDalpha*alpha_dadn_daddn_dadtau[3][i];
        double DFxDn = (hx1 + fx*(hx0 - hx1))*DgxDn + gx*(1.0 - fx)*Dhx1Dn + gx*(hx0 - hx1)*DfxDn;
        double DFxDDn = (hx1 + fx*(hx0 - hx1))*DgxDDn + gx*(1.0 - fx)*Dhx1DDn + gx*(hx0 - hx1)*DfxDDn;
        double DFxDtau = gx*(1.0 - fx)*Dhx1Dtau + gx*(hx0 - hx1)*DfxDtau;
        // printf("point %3d, DFxDn %.7E, DFxDDn %.7E, DFxDtau %.7E\n", i, DFxDn, DFxDDn, DFxDtau);
        // solve variant of n*epsilon_x^{unif}*F_x
        double Depsilon_xUnifDn = -pow(3.0*M_PI*M_PI, 1.0/3.0) / (4.0*M_PI) * pow(rho[i], -2.0/3.0);
        vx1[i] = (epsilon_xUnif + rho[i]*Depsilon_xUnifDn)*Fx + rho[i]*epsilon_xUnif*DFxDn;
        vx2[i] = rho[i]*epsilon_xUnif*DFxDDn;
        vx3[i] = rho[i]*epsilon_xUnif*DFxDtau;
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, epsilon_x %.7E, vx1 %.7E, vx2 %.7E, vx3 %.7E\n", i, epsilonx[i], vx1[i], vx2[i], vx3[i]);
    // }
}

void scanc(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonc, double *vc1, double *vc2, double *vc3) {
    // constants for epsilon_c^0
    double b1c = 0.0285764;
    double b2c = 0.0889;
    double b3c = 0.125541;
    double betaConst = 0.06672455060314922;
    double betaRsInf = betaConst*0.1/0.1778;
    double f0 = -0.9;
    // constants for epsilon_c LSDA1
    double p = 1.0;
    double AA = 0.0310907 ;
	double alpha1 = 0.21370 ;
	double beta1 = 7.5957 ;
	double beta2 = 3.5876 ;
	double beta3 = 1.6382 ;
	double beta4 = 0.49294 ;
    // constants for H1 of epsilon_c^1
    double r = 0.031091;
    // constants for switching function f_c
    double c1c = 0.64;
    double c2c = 1.5;
    double dc = 0.7;
    int i;
    for (i = 0; i < length; i++) {
        double zeta = 0; // now SCAN does not contain spin
        double phi = (pow(1.0 + zeta, 2.0/3.0) + pow(1.0 - zeta, 2.0/3.0)) / 2.0; // Since there is no spin, phi should be equal to 1
        double dx = (pow(1.0 + zeta, 4.0/3.0) + pow(1.0 - zeta, 4.0/3.0)) / 2.0; // Since there is no spin, dx should be equal to 1
        double s = s_dsdn_dsddn[0][i];
        double alpha = alpha_dadn_daddn_dadtau[0][i];
        double rs = pow(0.75/(M_PI*rho[i]), 1.0/3.0);
        // epsilon_c^0 (\alpha approach 0)
        double ecLDA0 = -b1c / (1.0 + b2c*sqrt(rs) + b3c*rs);
        double cx0 = -3.0/(4.0*M_PI) * pow(9.0*M_PI/4.0, 1.0/3.0);
        double Gc = (1.0 - 2.3631*(dx - 1.0)) * (1.0 - pow(zeta, 12.0));
        double w0 = exp(-ecLDA0/b1c) - 1.0;
        double xiInf0 = pow(3.0*M_PI*M_PI/16.0, 2.0/3.0) * (betaRsInf*1.0/(cx0 - f0)); // \xi_{r_s->\inf}(\zeta=0), 0.128026
        double gInf0s = pow(1.0 + 4.0*xiInf0*s*s, -0.25);
        double H0 = b1c*log(1.0 + w0*(1.0 - gInf0s));
        double ec0 = (ecLDA0 + H0)*Gc;
        // epsilon_c^1 (\alpha approach 1)
        double sqrRs = sqrt(rs);
        double rsmHalf = 1.0/sqrRs;
        double beta = betaConst * (1.0 + 0.1*rs) / (1.0 + 0.1778*rs);
        // epsilon_c LSDA1
        double ec_q0 = -2.0*AA*(1.0 + alpha1*rs);
        double ec_q1 = 2.0*AA*(beta1*sqrRs + beta2*rs + beta3*rs*sqrRs + beta4*rs*rs);
        double ec_q1p = AA*(beta1*rsmHalf + 2.0*beta2 + 3.0*beta3*sqrRs + 4.0*beta4*rs);
        double ec_den = 1.0 / (ec_q1*ec_q1 + ec_q1);
        double ec_log = -log(ec_q1*ec_q1*ec_den);
        double ec_lsda1 = ec_q0*ec_log;
        double Dec_lsda1Drs = -2.0*AA*alpha1*ec_log - ec_q0*ec_q1p*ec_den;
        // H1
        double rPhi3 = r*phi*phi*phi;
        double w1 = exp(-ec_lsda1/rPhi3) - 1.0;
        double A = beta / (r*w1);
        double t = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * s/(phi*sqrRs);
        double g = pow(1.0 + 4.0*A*t*t, -0.25);
        double H1 = rPhi3 * log(1.0 + w1*(1.0 - g));
        double ec1 = ec_lsda1 + H1;
        // printf("point %3d, ec0 %.7E, ec1 %.7E\n", i, ec0, ec1);
        // interpolate and extrapolate epsilon_c
        double fc;
        if (alpha > 1.0) {
            fc = -dc*exp(c2c / (1.0 - alpha));
        }
        else {
            fc = exp(-c1c*alpha / (1.0 - alpha));
        }
        epsilonc[i] = ec1 + fc*(ec0 - ec1);
        // compute variation of epsilon_c^0
        double DzetaDn = 0.0; // no spin
        double DrsDn = -4.0*M_PI/9.0 * pow(4.0*M_PI/3.0*rho[i], -4.0/3.0);
        double DdxDn = (4.0/3.0*pow(1.0 + zeta, 1.0/3.0) - 4.0/3.0*pow(1.0 - zeta, 1.0/3.0))*DzetaDn; // when there is no spin, it should be 0
        double DGcDn = -2.3631*DdxDn*(1.0 - pow(zeta, 12.0)) + (1.0 - 2.3631*(dx - 1))*(12.0*pow(zeta, 11.0)*DzetaDn);
        double DgInf0sDs = -0.25*pow(1.0 + 4.0*xiInf0*s*s, -1.25) * (4.0*xiInf0*2.0*s);
        double DgInf0sDn = DgInf0sDs * s_dsdn_dsddn[1][i];
        double DgInf0sDDn = DgInf0sDs * s_dsdn_dsddn[2][i];
        double DecLDA0Dn = b1c*(0.5*b2c/sqrRs + b3c) / pow(1.0 + b2c*sqrRs + b3c*rs, 2.0) * DrsDn;
        double Dw0Dn = (w0 + 1.0) * (-DecLDA0Dn/b1c);
        double DH0Dn = b1c*(Dw0Dn*(1.0 - gInf0s) - w0*DgInf0sDn) / (1.0 + w0*(1.0 - gInf0s));
        double DH0DDn = b1c*(-w0*DgInf0sDDn) / (1.0 + w0*(1.0 - gInf0s));
        double Dec0Dn = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0 + H0)*DGcDn;
        double Dec0DDn = DH0DDn*Gc;
        // compute variation of epsilon_c^1

        double denominatorInLogLSDA1 = 2.0*AA*(beta1*sqrRs + beta2*rs + beta3*sqrRs*rs + beta4*pow(rs, p + 1.0));
        double Dec_lsda1Dn = -(rs/rho[i]/3.0) * (-2.0*AA*alpha1*log(1.0 + 1.0/denominatorInLogLSDA1)
            -((-2.0*AA*(1.0 + alpha1*rs))*(AA*(beta1/sqrRs + 2.0*beta2 + 3.0*beta3*sqrRs + 2.0*(p + 1.0)*beta4*pow(rs, p))))
            / (denominatorInLogLSDA1*denominatorInLogLSDA1 + denominatorInLogLSDA1)); // from LDA_PW. If spin is added, the formula needs to be modified!
        // printf("point %3d, epsilonc %.7E, Dec0Dn %.7E, Dec_lsda1Dn %.7E\n", i, epsilonc[i], Dec0Dn, Dec_lsda1Dn);
        double DbetaDn = 0.066725*(0.1*(1.0 + 0.1778*rs) - 0.1778*(1.0 + 0.1*rs)) / (1.0 + 0.1778*rs) / (1.0 + 0.1778*rs) * DrsDn;
        double DphiDn = 0.5*(2.0/3.0*pow(1.0 + zeta, -1.0/3.0) - 2.0/3.0*pow(1.0 - zeta, -1.0/3.0)) * DzetaDn; // no spin, it should be 0
        double DtDn = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * (phi*sqrRs*s_dsdn_dsddn[1][i] - s*(DphiDn*sqrRs + phi*DrsDn/(2.0*sqrRs))) / (phi*phi*rs);
        double DtDDn = t*s_dsdn_dsddn[2][i]/s;
        double Dw1Dn = (w1 + 1.0) * (-(rPhi3*Dec_lsda1Dn - r*ec_lsda1*(3.0*phi*phi*DphiDn)) / rPhi3 / rPhi3);
        double DADn = (w1*DbetaDn - beta*Dw1Dn) / (r*w1*w1);
        double DgDn = -0.25*pow(1.0 + 4.0*A*t*t, -1.25) * (4.0*(DADn*t*t + 2.0*A*t*DtDn));
        double DgDDn = -0.25*pow(1.0 + 4.0*A*t*t, -1.25) * (4.0*2.0*A*t*DtDDn);
        double DH1Dn = 3*r*phi*phi*DphiDn*log(1.0 + w1*(1.0 - g)) + rPhi3*(Dw1Dn*(1.0 - g) - w1*DgDn) / (1.0 + w1*(1.0 - g));
        double DH1DDn = rPhi3*(-w1*DgDDn) / (1.0 + w1*(1.0 - g));
        double Dec1Dn = Dec_lsda1Dn + DH1Dn;
        double Dec1DDn = DH1DDn;
        // printf("point %3d, DH1Dn %.7E, DH1DDn %.7E\n", i, DH1Dn, DH1DDn);

        // compute variation of f_c and epsilon_c
        double DfcDalpha;
        if (alpha > 1.0) {
            DfcDalpha = fc*(c2c/(1.0 - alpha)/(1.0 - alpha));
        }
        else {
            DfcDalpha = fc*(-c1c/(1.0 - alpha)/(1.0 - alpha));
        }
        double DfcDn = DfcDalpha*alpha_dadn_daddn_dadtau[1][i];
        double DfcDDn = DfcDalpha*alpha_dadn_daddn_dadtau[2][i];
        double DfcDtau = DfcDalpha*alpha_dadn_daddn_dadtau[3][i];
        double DepsiloncDn = Dec1Dn + fc*(Dec0Dn - Dec1Dn) + DfcDn*(ec0 - ec1);
        double DepsiloncDDn = Dec1DDn + fc*(Dec0DDn -Dec1DDn) + DfcDDn*(ec0 - ec1);
        double DepsiloncDtau = DfcDtau*(ec0 - ec1);
        vc1[i] = epsilonc[i] + rho[i]*DepsiloncDn;
        vc2[i] = rho[i]*DepsiloncDDn;
        vc3[i] = rho[i]*DepsiloncDtau;
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, epsilon_c %.7E, vc1 %.7E, vc2 %.7E, vc3 %.7E\n", i, epsilonc[i], vc1[i], vc2[i], vc3[i]);
    // }
}

