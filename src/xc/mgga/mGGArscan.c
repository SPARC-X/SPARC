/**
 * @file    MGGArscan.c
 * @brief   This file contains the functions for scan functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Bartók, Albert P., and Jonathan R. Yates. "Regularized SCAN functional." 
 * The Journal of chemical physics 150, no. 16 (2019): 161101.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "mGGArscan.h"
#include "isddft.h"


void rscanx(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x) {
    double **s_dsdn_dsddn = (double**)malloc(sizeof(double*)*3); // variable s, and ds/dn, ds/d|\nabla n|
    double *normDrho = (double*)malloc(DMnd * sizeof(double));
    int i;
    for (i = 0; i < 3; i++) {
        s_dsdn_dsddn[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(s_dsdn_dsddn[i] != NULL);
    }
    double **alphaP_dadn_daddn_dadtau = (double**)malloc(sizeof(double*)*4); // variable \alpha, and d\alpha/dn, d\alpha/d|\nabla n|, d\alpha/d\tau
    for (i = 0; i < 4; i++) {
        alphaP_dadn_daddn_dadtau[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alphaP_dadn_daddn_dadtau[i] != NULL);
    }

    for (i = 0; i < DMnd; i++) {
        normDrho[i] = sqrt(sigma[i]);
    }

    basic_rscan_variables(DMnd, rho, normDrho, tau, s_dsdn_dsddn, alphaP_dadn_daddn_dadtau);

    Calculate_rscanx(DMnd, rho, s_dsdn_dsddn, alphaP_dadn_daddn_dadtau, ex, vx, v2x, v3x);

    for (i = 0; i < DMnd; i++) {
        v2x[i] = v2x[i] / normDrho[i];
    }

    for (i = 0; i < 3; i++) {
        free(s_dsdn_dsddn[i]);
    }
    free(s_dsdn_dsddn);
    free(normDrho);
    for (i = 0; i < 4; i++) {
        free(alphaP_dadn_daddn_dadtau[i]);
    }
    free(alphaP_dadn_daddn_dadtau);
}

void basic_rscan_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alphaP_dadn_daddn_dadtau) {
    int i;
    double threeMPi2_1o3 = pow(3.0*M_PI*M_PI, 1.0/3.0);
    double threeMPi2_2o3 = threeMPi2_1o3*threeMPi2_1o3;
    for (i = 0; i < length; i++) {
        s_dsdn_dsddn[0][i] = normDrho[i] / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0));
        double tauw = normDrho[i]*normDrho[i] / (8*rho[i]);
        double tauUnif = 3.0/10.0 * threeMPi2_2o3 * pow(rho[i], 5.0/3.0);
        double alpha = (tau[i] - tauw) / (tauUnif + 1e-4);
        double alpha2 = alpha*alpha;
        alphaP_dadn_daddn_dadtau[0][i] = alpha2*alpha / (alpha2 + 1e-3);

        s_dsdn_dsddn[1][i] = -2.0*normDrho[i] / (3.0 * threeMPi2_1o3 * pow(rho[i], 7.0/3.0)); // ds/dn
        s_dsdn_dsddn[2][i] = 1.0 / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0)); // ds/d|\nabla n|
        double DtauwDn = -normDrho[i]*normDrho[i] / (8*rho[i]*rho[i]);
        double DtauwDDn = normDrho[i] / (4*rho[i]);
        double DtauUnifDn = threeMPi2_2o3 / 2.0 * pow(rho[i], 2.0/3.0);
        double DalphaDn = (-DtauwDn*(tauUnif + 1e-4) - (tau[i] - tauw)*DtauUnifDn) / ((tauUnif + 1e-4)*(tauUnif + 1e-4));
        double DalphaDDn = (-DtauwDDn) / (tauUnif + 1e-4);
        double DalphaDtau = 1.0 / (tauUnif + 1e-4);
        double DalphaPDalpha = (3.0*alpha2*(alpha2 + 1e-3) - alpha2*alpha*(2.0*alpha)) / ((alpha2 + 1e-3)*(alpha2 + 1e-3));
        alphaP_dadn_daddn_dadtau[1][i] = DalphaPDalpha*DalphaDn; // d\alpha/dn
        alphaP_dadn_daddn_dadtau[2][i] = DalphaPDalpha*DalphaDDn; // d\alpha/d|\nabla n|
        alphaP_dadn_daddn_dadtau[3][i] = DalphaPDalpha*DalphaDtau; // d\alpha/d\tau
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, s %.7E, dsdn %.7E, dsddn %.7E, alpha %.7E, dadn %.7E, daddn %.7E, dadtau %.7E\n", i, 
    //         s_dsdn_dsddn[0][i], s_dsdn_dsddn[1][i], s_dsdn_dsddn[2][i], 
    //         alpha_dadn_daddn_dadtau[0][i], alpha_dadn_daddn_dadtau[1][i], alpha_dadn_daddn_dadtau[2][i],alpha_dadn_daddn_dadtau[3][i]);
    // }
}

void Calculate_rscanx(int length, double *rho, double **s_dsdn_dsddn, double **alphaP_dadn_daddn_dadtau, double *epsilonx, double *vx, double *v2x, double *v3x) {
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
        double alphaP = alphaP_dadn_daddn_dadtau[0][i];
        double term1 = 1.0 + b4*s2/mu_ak*exp(-fabs(b4)*s2/mu_ak);
        double xFir = mu_ak*s2 * term1;
        double term3 = 2.0*(b1*s2 + b2*(1.0 - alphaP)*exp(-b3*(1.0 - alphaP)*(1.0 - alphaP)));
        double xSec = (term3/2.0)*(term3/2.0);
        double hx1 = 1.0 + k1 - k1/(1.0 + (xFir + xSec)/k1); // x = xFir + xSec
        double fx;
        if (alphaP > 2.5) {
            fx = -dx*exp(c2x / (1.0 - alphaP));
        }
        else if (alphaP > 0.0) {
            double alpha2 = alphaP *alphaP; double alpha3 = alpha2*alphaP;
            double alpha4 = alpha3*alphaP; double alpha5 = alpha4*alphaP;
            double alpha6 = alpha5*alphaP; double alpha7 = alpha6*alphaP;
            fx = 1.0 + (-0.667)*alphaP + (-0.4445555)*alpha2 + (-0.663086601049)*alpha3 + 1.451297044490*alpha4 
                + (-0.887998041597)*alpha5 + 0.234528941479*alpha6 + (-0.023185843322)*alpha7;
        }
        else {
            fx = exp(-c1x*alphaP / (1.0 - alphaP));
        }
        double sqrt_s = sqrt(s);
        double gx = 1.0 - exp(-a1/sqrt_s);
        double Fx = (hx1 + fx*(hx0 - hx1))*gx;
        epsilonx[i] = epsilon_xUnif*Fx;

        double term2 = s2*(b4/mu_ak*exp(-fabs(b4)*s2/mu_ak) + b4*s2/mu_ak*exp(-fabs(b4)*s2/mu_ak)*(-fabs(b4)/mu_ak));
        double term4 = b2*(-exp(-b3*(1.0 - alphaP)*(1.0 - alphaP)) + (1.0 - alphaP)*exp(-b3*(1.0 - alphaP)*(1.0 - alphaP))*(2*b3*(1.0 - alphaP)));
        // printf("point %3d, term1 %.7E, term2 %.7E, term3 %.7E, term4 %.7E\n", i, term1, term2, term3, term4);
        double DxDs = 2*s*(mu_ak*(term1 + term2) + b1*term3);
        double DxDalpha = term3*term4;
        double DxDn = s_dsdn_dsddn[1][i]*DxDs + alphaP_dadn_daddn_dadtau[1][i]*DxDalpha;
        double DxDDn = s_dsdn_dsddn[2][i]*DxDs + alphaP_dadn_daddn_dadtau[2][i]*DxDalpha;
        double DxDtau = alphaP_dadn_daddn_dadtau[3][i]*DxDalpha;
        // printf("point %3d, DxDn %.7E, DxDDn %.7E, DxDtau %.7E\n", i, DxDn, DxDDn, DxDtau);

        double DgxDn = -exp(-a1/sqrt_s)*(a1/2.0/sqrt_s/s)*s_dsdn_dsddn[1][i];
        double DgxDDn = -exp(-a1/sqrt_s)*(a1/2.0/sqrt_s/s)*s_dsdn_dsddn[2][i];
        double Dhx1Dx = 1.0 / (1.0 + (xFir + xSec)/k1) / (1.0 + (xFir + xSec)/k1);
        double Dhx1Dn = DxDn*Dhx1Dx;
        double Dhx1DDn = DxDDn*Dhx1Dx;
        double Dhx1Dtau = DxDtau*Dhx1Dx;
        // printf("point %3d, Dhx1Dn %.7E, Dhx1DDn %.7E, Dhx1Dtau %.7E\n", i, Dhx1Dn, Dhx1DDn, Dhx1Dtau);
        double DfxDalpha;
        if (alphaP > 2.5) {
            DfxDalpha = -dx*exp(c2x/(1.0 - alphaP)) * (c2x/(1.0 - alphaP)/(1.0 - alphaP));
        }
        else if (alphaP > 0.0) {
            double alpha2 = alphaP *alphaP; double alpha3 = alpha2*alphaP;
            double alpha4 = alpha3*alphaP; double alpha5 = alpha4*alphaP;
            double alpha6 = alpha5*alphaP;
            DfxDalpha = (-0.667) + (-0.4445555)*alphaP*2.0 + (-0.663086601049)*alpha2*3.0 + 1.451297044490*alpha3*4.0
                + (-0.887998041597)*alpha4*5.0 + 0.234528941479*alpha5*6.0 + (-0.023185843322)*alpha6*7.0;
        }
        else {
            DfxDalpha = exp(-c1x*alphaP/(1.0 - alphaP)) * (-c1x/(1.0 - alphaP)/(1 - alphaP));
        }
        double DfxDn = DfxDalpha*alphaP_dadn_daddn_dadtau[1][i];
        double DfxDDn = DfxDalpha*alphaP_dadn_daddn_dadtau[2][i];
        double DfxDtau = DfxDalpha*alphaP_dadn_daddn_dadtau[3][i];
        double DFxDn = (hx1 + fx*(hx0 - hx1))*DgxDn + gx*(1.0 - fx)*Dhx1Dn + gx*(hx0 - hx1)*DfxDn;
        double DFxDDn = (hx1 + fx*(hx0 - hx1))*DgxDDn + gx*(1.0 - fx)*Dhx1DDn + gx*(hx0 - hx1)*DfxDDn;
        double DFxDtau = gx*(1.0 - fx)*Dhx1Dtau + gx*(hx0 - hx1)*DfxDtau;
        // printf("point %3d, DFxDn %.7E, DFxDDn %.7E, DFxDtau %.7E\n", i, DFxDn, DFxDDn, DFxDtau);
        // solve variant of n*epsilon_x^{unif}*F_x
        double Depsilon_xUnifDn = -pow(3.0*M_PI*M_PI, 1.0/3.0) / (4.0*M_PI) * pow(rho[i], -2.0/3.0);
        vx[i] = (epsilon_xUnif + rho[i]*Depsilon_xUnifDn)*Fx + rho[i]*epsilon_xUnif*DFxDn;
        v2x[i] = rho[i]*epsilon_xUnif*DFxDDn;
        v3x[i] = rho[i]*epsilon_xUnif*DFxDtau;
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, epsilon_x %.7E, vx1 %.7E, vx2 %.7E, vx3 %.7E\n", i, epsilonx[i], vx1[i], vx2[i], vx3[i]);
    // }
}

void rscanc(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c) {
    double **s_dsdn_dsddn = (double**)malloc(sizeof(double*)*3); // variable s, and ds/dn, ds/d|\nabla n|
    double *normDrho = (double*)malloc(DMnd * sizeof(double));
    int i;
    for (i = 0; i < 3; i++) {
        s_dsdn_dsddn[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(s_dsdn_dsddn[i] != NULL);
    }
    double **alphaP_dadn_daddn_dadtau = (double**)malloc(sizeof(double*)*4); // variable \alpha, and d\alpha/dn, d\alpha/d|\nabla n|, d\alpha/d\tau
    for (i = 0; i < 4; i++) {
        alphaP_dadn_daddn_dadtau[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alphaP_dadn_daddn_dadtau[i] != NULL);
    }

    for (i = 0; i < DMnd; i++) {
        normDrho[i] = sqrt(sigma[i]);
    }

    basic_rscan_variables(DMnd, rho, normDrho, tau, s_dsdn_dsddn, alphaP_dadn_daddn_dadtau);

    Calculate_rscanc(DMnd, rho, s_dsdn_dsddn, alphaP_dadn_daddn_dadtau, ec, vc, v2c, v3c);

    for (i = 0; i < DMnd; i++) {
        v2c[i] = v2c[i] / normDrho[i];
    }

    for (i = 0; i < 3; i++) {
        free(s_dsdn_dsddn[i]);
    }
    free(s_dsdn_dsddn);
    free(normDrho);
    for (i = 0; i < 4; i++) {
        free(alphaP_dadn_daddn_dadtau[i]);
    }
    free(alphaP_dadn_daddn_dadtau);
}

void Calculate_rscanc(int length, double *rho, double **s_dsdn_dsddn, double **alphaP_dadn_daddn_dadtau, double *epsilonc, double *vc, double *v2c, double *v3c) {
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
        double alphaP = alphaP_dadn_daddn_dadtau[0][i];
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
        double beta = betaConst * (1.0 + 0.1*rs) / (1.0 + 0.1778*rs);
        // epsilon_c LSDA1
        double ec_q0 = -2.0*AA*(1.0 + alpha1*rs);
        double ec_q1 = 2.0*AA*(beta1*sqrRs + beta2*rs + beta3*rs*sqrRs + beta4*rs*rs);
        double ec_den = 1.0 / (ec_q1*ec_q1 + ec_q1);
        double ec_log = -log(ec_q1*ec_q1*ec_den);
        double ec_lsda1 = ec_q0*ec_log;
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
        if (alphaP > 2.5) {
            fc = -dc*exp(c2c / (1.0 - alphaP));
        }
        else if (alphaP > 0.0) {
            double alpha2 = alphaP *alphaP; double alpha3 = alpha2*alphaP;
            double alpha4 = alpha3*alphaP; double alpha5 = alpha4*alphaP;
            double alpha6 = alpha5*alphaP; double alpha7 = alpha6*alphaP;
            fc = 1.0 + (-0.64)*alphaP + (-0.4352)*alpha2 + (-1.535685604549)*alpha3 + 3.061560252175*alpha4
                + (-1.915710236206)*alpha5 + 0.516884468372*alpha6 + (-0.051848879792)*alpha7;
        }
        else {
            fc = exp(-c1c*alphaP / (1.0 - alphaP));
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
        if (alphaP > 2.5) {
            DfcDalpha = fc*(c2c/(1.0 - alphaP)/(1.0 - alphaP));
        }
        else if (alphaP > 0) {
            double alpha2 = alphaP *alphaP; double alpha3 = alpha2*alphaP;
            double alpha4 = alpha3*alphaP; double alpha5 = alpha4*alphaP;
            double alpha6 = alpha5*alphaP;
            DfcDalpha = (-0.64) + (-0.4352)*alphaP*2.0 + (-1.535685604549)*alpha2*3.0 + 3.061560252175*alpha3*4.0 
                + (-1.915710236206)*alpha4*5.0 + 0.516884468372*alpha5*6.0 + (-0.051848879792)*alpha6*7.0;
        }
        else {
            DfcDalpha = fc*(-c1c/(1.0 - alphaP)/(1.0 - alphaP));
        }
        double DfcDn = DfcDalpha*alphaP_dadn_daddn_dadtau[1][i];
        double DfcDDn = DfcDalpha*alphaP_dadn_daddn_dadtau[2][i];
        double DfcDtau = DfcDalpha*alphaP_dadn_daddn_dadtau[3][i];
        double DepsiloncDn = Dec1Dn + fc*(Dec0Dn - Dec1Dn) + DfcDn*(ec0 - ec1);
        double DepsiloncDDn = Dec1DDn + fc*(Dec0DDn -Dec1DDn) + DfcDDn*(ec0 - ec1);
        double DepsiloncDtau = DfcDtau*(ec0 - ec1);
        vc[i] = epsilonc[i] + rho[i]*DepsiloncDn;
        v2c[i] = rho[i]*DepsiloncDDn;
        v3c[i] = rho[i]*DepsiloncDtau;
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, epsilon_c %.7E, vc1 %.7E, vc2 %.7E, vc3 %.7E\n", i, epsilonc[i], vc1[i], vc2[i], vc3[i]);
    // }
}


void rscanx_spin(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x) { 
    // the main function in scan.c, compute \epsilon and potentials of SCAN functional
    double *ex_up = (double*)malloc(sizeof(double)*DMnd);
    double *vx_up = (double*)malloc(sizeof(double)*DMnd);
    double *v2x_up = (double*)malloc(sizeof(double)*DMnd);
    double *v3x_up = (double*)malloc(sizeof(double)*DMnd);
    double *ex_dn = (double*)malloc(sizeof(double)*DMnd);
    double *vx_dn = (double*)malloc(sizeof(double)*DMnd);
    double *v2x_dn = (double*)malloc(sizeof(double)*DMnd);
    double *v3x_dn = (double*)malloc(sizeof(double)*DMnd);

    double *rho_up = (double*)malloc(sizeof(double)*DMnd);
    double *sigma_up = (double*)malloc(sizeof(double)*DMnd);
    double *tau_up = (double*)malloc(sizeof(double)*DMnd);
    double *rho_dn = (double*)malloc(sizeof(double)*DMnd);
    double *sigma_dn = (double*)malloc(sizeof(double)*DMnd);
    double *tau_dn = (double*)malloc(sizeof(double)*DMnd);

    int i;
    for (i = 0; i < DMnd; i++) {
        rho_up[i] = rho[i + DMnd] * 2.0;
        rho_dn[i] = rho[i + 2*DMnd] * 2.0;
        sigma_up[i] = sigma[i + DMnd] * 4.0;
        sigma_dn[i] = sigma[i + 2*DMnd] * 4.0;
        tau_up[i] = tau[i + DMnd] * 2.0;
        tau_dn[i] = tau[i + 2*DMnd] * 2.0;
    }

    rscanx(DMnd, rho_up, sigma_up, tau_up, ex_up, vx_up, v2x_up, v3x_up);
    rscanx(DMnd, rho_dn, sigma_dn, tau_dn, ex_dn, vx_dn, v2x_dn, v3x_dn);

    for (i = 0; i < DMnd; i++) { // this loop can be optimized
        ex[i] = (ex_up[i]*rho[i + DMnd] + ex_dn[i]*rho[i + 2*DMnd]) / rho[i];
        vx[i] = vx_up[i];
        vx[DMnd + i] = vx_dn[i];
        v2x[i] = v2x_up[i] * 2.0;
        v2x[DMnd + i] = v2x_dn[i] * 2.0;
        v3x[i] = v3x_up[i];
        v3x[DMnd + i] = v3x_dn[i];
    }

    free(ex_up); free(vx_up); free(v2x_up); free(v3x_up);
    free(ex_dn); free(vx_dn); free(v2x_dn); free(v3x_dn);

    free(rho_up); free(sigma_up); free(tau_up);
    free(rho_dn); free(sigma_dn); free(tau_dn);
}

void rscanc_spin(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c) { 
    double **s_dsdn_dsddn_correlation = (double**)malloc(sizeof(double*)*3); // variable s, and ds/dn, ds/d|\nabla n|. n = n_up + n_dn
    double *normDrho = (double*)malloc(3*DMnd * sizeof(double));
    int i;
    for (i = 0; i < 3; i++) {
        s_dsdn_dsddn_correlation[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(s_dsdn_dsddn_correlation[i] != NULL);
    }
    double **alphaP_dadnup_dadndn_daddn_dadtau_correlation = (double**)malloc(sizeof(double*)*5); // variable \alpha, and d\alpha/dn_up, d\alpha/dn_dn, d\alpha/d|\nabla n|, d\alpha/d\tau. n = n_up + n_dn, tau = tau_up + tau_dn
    for (i = 0; i < 5; i++) {
        alphaP_dadnup_dadndn_daddn_dadtau_correlation[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alphaP_dadnup_dadndn_daddn_dadtau_correlation[i] != NULL);
    }
    double **zeta_dzetadnup_dzetadndn = (double**)malloc(sizeof(double*)*3);
    for (i = 0; i < 3; i++) {
        zeta_dzetadnup_dzetadndn[i] = (double*)malloc(sizeof(double*)*DMnd);
        assert(zeta_dzetadnup_dzetadndn[i] != NULL);
    }

    for (i = 0; i < 3*DMnd; i++) {
        normDrho[i] = sqrt(sigma[i]);
    }

    basic_rscanc_spin_variables(DMnd, rho, normDrho, tau, s_dsdn_dsddn_correlation, alphaP_dadnup_dadndn_daddn_dadtau_correlation, zeta_dzetadnup_dzetadndn);

    Calculate_rscanc_spin(DMnd, rho, s_dsdn_dsddn_correlation, alphaP_dadnup_dadndn_daddn_dadtau_correlation, zeta_dzetadnup_dzetadndn, ec, vc, v2c, v3c);

    for (i = 0; i < DMnd; i++) {
        v2c[i] = v2c[i] / normDrho[i];
    }

    for (i = 0; i < 3; i++) {
        free(s_dsdn_dsddn_correlation[i]);
    }
    free(s_dsdn_dsddn_correlation);
    free(normDrho);
    for (i = 0; i < 5; i++) {
        free(alphaP_dadnup_dadndn_daddn_dadtau_correlation[i]);
    }
    free(alphaP_dadnup_dadndn_daddn_dadtau_correlation);
    for (i = 0; i < 3; i++) {
        free(zeta_dzetadnup_dzetadndn[i]);
    }
    free(zeta_dzetadnup_dzetadndn);
}

void basic_rscanc_spin_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alphaP_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn) { // this function serves for exchange computation
    int i;
    double threeMPi2_1o3 = pow(3.0*M_PI*M_PI, 1.0/3.0);
    double threeMPi2_2o3 = threeMPi2_1o3*threeMPi2_1o3;
    double theRho, theNormDrho, theTau;
    double theZeta;
    double ds, DdsDnup, DdsDndn;
    double alpha, alpha2, DalphaDnup, DalphaDndn, DalphaDDn, DalphaDtau, DalphaPDalpha;
    double tauw, tauUnif, DtauwDn, DtauwDDn, DtauUnifDnup, DtauUnifDndn;
    for (i = 0; i < length; i++) {
        theRho = rho[i];
        theNormDrho = normDrho[i];
        theTau = tau[i];
        s_dsdn_dsddn[0][i] = theNormDrho / (2.0 * threeMPi2_1o3 * pow(theRho, 4.0/3.0));
        zeta_dzetadnup_dzetadndn[0][i] = (rho[length+i] - rho[length*2+i]) / theRho;
        theZeta = zeta_dzetadnup_dzetadndn[0][i];
        tauw = theNormDrho*theNormDrho / (8.0*theRho);
        ds = (pow(1.0+theZeta, 5.0/3.0) + pow(1-theZeta, 5.0/3.0)) / 2.0;
        tauUnif = 3.0/10.0 * threeMPi2_2o3 * pow(theRho, 5.0/3.0) * ds;
        alpha = (theTau - tauw) / (tauUnif + 1e-4);
        alpha2 = alpha*alpha;
        alphaP_dadnup_dadndn_daddn_dadtau[0][i] = alpha2*alpha / (alpha2 + 1e-3);

        s_dsdn_dsddn[1][i] = -2.0*theNormDrho / (3.0 * threeMPi2_1o3 * pow(theRho, 7.0/3.0)); // ds/dn
        s_dsdn_dsddn[2][i] = 1.0 / (2.0 * threeMPi2_1o3 * pow(theRho, 4.0/3.0)); // ds/d|\nabla n|
        zeta_dzetadnup_dzetadndn[1][i] = 2.0*rho[length*2+i] / (theRho*theRho);
        zeta_dzetadnup_dzetadndn[2][i] = -2.0*rho[length+i] / (theRho*theRho);
        DtauwDn = -theNormDrho*theNormDrho / (8*theRho*theRho);
        DtauwDDn = theNormDrho / (4*theRho);
        DdsDnup = 5.0/3.0 * (pow(1.0+theZeta, 2.0/3.0) - pow(1.0-theZeta, 2.0/3.0)) * zeta_dzetadnup_dzetadndn[1][i] / 2.0;
        DdsDndn = 5.0/3.0 * (pow(1.0+theZeta, 2.0/3.0) - pow(1.0-theZeta, 2.0/3.0)) * zeta_dzetadnup_dzetadndn[2][i] / 2.0;
        DtauUnifDnup = threeMPi2_2o3 / 2.0 * pow(theRho, 2.0/3.0) * ds + 3.0/10.0 * threeMPi2_2o3 * pow(theRho, 5.0/3.0) * DdsDnup;
        DtauUnifDndn = threeMPi2_2o3 / 2.0 * pow(theRho, 2.0/3.0) * ds + 3.0/10.0 * threeMPi2_2o3 * pow(theRho, 5.0/3.0) * DdsDndn;
        DalphaDnup = (-DtauwDn*(tauUnif + 1e-4) - (tau[i] - tauw)*DtauUnifDnup) / ((tauUnif + 1e-4)*(tauUnif + 1e-4));
        DalphaDndn = (-DtauwDn*(tauUnif + 1e-4) - (tau[i] - tauw)*DtauUnifDndn) / ((tauUnif + 1e-4)*(tauUnif + 1e-4));
        DalphaDDn = (-DtauwDDn) / (tauUnif + 1e-4);
        DalphaDtau = 1.0 / (tauUnif + 1e-4);
        DalphaPDalpha = (3.0*alpha2*(alpha2 + 1e-3) - alpha2*alpha*(2.0*alpha)) / ((alpha2 + 1e-3)*(alpha2 + 1e-3));
        alphaP_dadnup_dadndn_daddn_dadtau[1][i] = DalphaPDalpha*DalphaDnup; // d\alphaP/dnup
        alphaP_dadnup_dadndn_daddn_dadtau[2][i] = DalphaPDalpha*DalphaDndn; // d\alphaP/dndn
        alphaP_dadnup_dadndn_daddn_dadtau[3][i] = DalphaPDalpha*DalphaDDn; // d\alphaP/d|\nabla n|
        alphaP_dadnup_dadndn_daddn_dadtau[4][i] = DalphaPDalpha*DalphaDtau; // d\alphaP/d\tau
    }
    // for (i = 0; i < length; i++) {
    //     printf("point %3d, s %.7E, dsdn %.7E, dsddn %.7E, alpha %.7E, dadn %.7E, daddn %.7E, dadtau %.7E\n", i, 
    //         s_dsdn_dsddn[0][i], s_dsdn_dsddn[1][i], s_dsdn_dsddn[2][i], 
    //         alpha_dadn_daddn_dadtau[0][i], alpha_dadn_daddn_dadtau[1][i], alpha_dadn_daddn_dadtau[2][i],alpha_dadn_daddn_dadtau[3][i]);
    // }
}

void Calculate_rscanc_spin(int length, double *rho, double **s_dsdn_dsddn, double **alphaP_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc, double *v2c, double *v3c) {
    double zeta, phi, dx, s, alphaP, rs;
    double ecLDA0, cx0, Gc, w0, xiInf0, gInf0s, H0, ec0;
    // constants for epsilon_c^0
    double b1c = 0.0285764;
    double b2c = 0.0889;
    double b3c = 0.125541;
    double betaConst = 0.06672455060314922;
    double betaRsInf = betaConst*0.1/0.1778;
    double f0 = -0.9;
    // constants for epsilon_c LSDA1
    double ecrs0_q0, ecrs0_q1, ecrs0_q1p, ecrs0_den, ecrs0_log, ecrs0, Decrs0_Drs;
    double AA0 = 0.0310907 ;
	double alpha10 = 0.21370 ;
	double beta10 = 7.5957 ;
	double beta20 = 3.5876 ;
	double beta30 = 1.6382 ;
	double beta40 = 0.49294 ;
    double ac_q0, ac_q1, ac_q1p, ac_den, ac_log, ac, Dac_Drs;
    double AAac = 0.0168869 ;
	double alpha1ac = 0.11125 ;
	double beta1ac = 10.357 ;
	double beta2ac = 3.6231 ;
	double beta3ac = 0.88026 ;
	double beta4ac = 0.49671 ;
    double ecrs1_q0, ecrs1_q1, ecrs1_q1p, ecrs1_den, ecrs1_log, ecrs1, Decrs1_Drs;
    double AA1 = 0.01554535 ;
	double alpha11 = 0.20548 ;
	double beta11 = 14.1189 ;
	double beta21 = 6.1977 ;
	double beta31 = 3.3662 ;
	double beta41 = 0.62517 ;
    double f_zeta, fp_zeta, zeta4, gcrs, ec_lsda1;
    // constants for H1 of epsilon_c^1
    double r = 0.031091;
    double rPhi3, w1, A, t, g, H1, ec1;
    // constants for switching function f_c
    double c1c = 0.64;
    double c2c = 1.5;
    double dc = 0.7;

    double DzetaDnup, DzetaDndn, DrsDn, DdxDnup, DdxDndn, DGcDnup, DGcDndn, DgInf0sDs, DgInf0sDn, DgInf0sDDn, DecLDA0Dn, Dw0Dn, DH0Dn, DH0DDn, Dec0Dnup, Dec0Dndn, Dec0DDn;
    double DgcrsDrs, Dec_lsda1_Drs, Dfzeta4_Dzeta, Dec_lsda1_Dzeta, Dec_lsda1Dnup, Dec_lsda1Dndn;
    double DbetaDn, DphiDnup, DphiDndn, DtDnup, DtDndn, DtDDn, Dw1Dnup, Dw1Dndn, DADnup, DADndn, DgDnup, DgDndn, DgDDn, DH1Dnup, DH1Dndn, DH1DDn, Dec1Dnup, Dec1Dndn, Dec1DDn;
    double DfcDnup, DfcDndn, DfcDDn, DfcDtau, DepsiloncDnup, DepsiloncDndn, DepsiloncDDn, DepsiloncDtau;
    int i;
    for (i = 0; i < length; i++) {
        zeta = zeta_dzetadnup_dzetadndn[0][i];
        phi = (pow(1.0 + zeta, 2.0/3.0) + pow(1.0 - zeta, 2.0/3.0)) / 2.0;
        dx = (pow(1.0 + zeta, 4.0/3.0) + pow(1.0 - zeta, 4.0/3.0)) / 2.0;
        s = s_dsdn_dsddn[0][i];
        alphaP = alphaP_dadnup_dadndn_daddn_dadtau[0][i];
        rs = pow(0.75/(M_PI*rho[i]), 1.0/3.0);
        // epsilon_c^0 (\alpha approach 0)
        ecLDA0 = -b1c / (1.0 + b2c*sqrt(rs) + b3c*rs);
        cx0 = -3.0/(4.0*M_PI) * pow(9.0*M_PI/4.0, 1.0/3.0);
        Gc = (1.0 - 2.3631*(dx - 1.0)) * (1.0 - pow(zeta, 12.0));
        w0 = exp(-ecLDA0/b1c) - 1.0;
        xiInf0 = pow(3.0*M_PI*M_PI/16.0, 2.0/3.0) * (betaRsInf*1.0/(cx0 - f0)); // \xi_{r_s->\inf}(\zeta=0), 0.128026
        gInf0s = pow(1.0 + 4.0*xiInf0*s*s, -0.25);
        H0 = b1c*log(1.0 + w0*(1.0 - gInf0s));
        ec0 = (ecLDA0 + H0)*Gc;
        // epsilon_c^1 (\alpha approach 1)
        double sqrRs = sqrt(rs);
        double rsmHalf = 1.0/sqrRs;
        double beta = betaConst * (1.0 + 0.1*rs) / (1.0 + 0.1778*rs);
        // epsilon_c LSDA1
        ecrs0_q0 = -2.0*AA0*(1.0 + alpha10*rs);
        ecrs0_q1 = 2.0*AA0*(beta10*sqrRs + beta20*rs + beta30*rs*sqrRs + beta40*rs*rs);
        ecrs0_q1p = AA0*(beta10*rsmHalf + 2.0*beta20 + 3.0*beta30*sqrRs + 4.0*beta40*rs);
        ecrs0_den = 1.0 / (ecrs0_q1*ecrs0_q1 + ecrs0_q1);
        ecrs0_log = -log(ecrs0_q1*ecrs0_q1*ecrs0_den);
        ecrs0 = ecrs0_q0*ecrs0_log;
        Decrs0_Drs = -2.0*AA0*alpha10*ecrs0_log - ecrs0_q0*ecrs0_q1p*ecrs0_den;

        ac_q0 = -2.0*AAac*(1.0 + alpha1ac*rs);
        ac_q1 = 2.0*AAac*(beta1ac*sqrRs + beta2ac*rs + beta3ac*rs*sqrRs + beta4ac*rs*rs);
        ac_q1p = AAac*(beta1ac*rsmHalf + 2.0*beta2ac + 3.0*beta3ac*sqrRs + 4.0*beta4ac*rs);
        ac_den = 1.0 / (ac_q1*ac_q1 + ac_q1);
        ac_log = -log(ac_q1*ac_q1*ac_den);
        ac = ac_q0*ac_log;
        Dac_Drs = -2.0*AAac*alpha1ac*ac_log - ac_q0*ac_q1p*ac_den;

        ecrs1_q0 = -2.0*AA1*(1.0 + alpha11*rs);
        ecrs1_q1 = 2.0*AA1*(beta11*sqrRs + beta21*rs + beta31*rs*sqrRs + beta41*rs*rs);
        ecrs1_q1p = AA1*(beta11*rsmHalf + 2.0*beta21 + 3.0*beta31*sqrRs + 4.0*beta41*rs);
        ecrs1_den = 1.0 / (ecrs1_q1*ecrs1_q1 + ecrs1_q1);
        ecrs1_log = -log(ecrs1_q1*ecrs1_q1*ecrs1_den);
        ecrs1 = ecrs1_q0*ecrs1_log;
        Decrs1_Drs = -2.0*AA1*alpha11*ecrs1_log - ecrs1_q0*ecrs1_q1p*ecrs1_den;

        f_zeta = (pow(1.0+zeta, 4.0/3.0) + pow(1.0-zeta, 4.0/3.0) - 2.0 ) / (pow(2.0, 4.0/3.0) - 2.0);
	    fp_zeta = (pow(1.0+zeta, 1.0/3.0) - pow(1.0-zeta, 1.0/3.0)) * 4.0/3.0 / (pow(2.0, 4.0/3.0) - 2.0);
	    zeta4 = pow(zeta, 4.0);
	    gcrs = ecrs1 - ecrs0 + ac/1.709921;
	    ec_lsda1 = ecrs0 + f_zeta * (zeta4 * gcrs - ac/1.709921);
        // H1
        rPhi3 = r*phi*phi*phi;
        w1 = exp(-ec_lsda1/rPhi3) - 1.0;
        A = beta / (r*w1);
        t = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * s/(phi*sqrRs);
        g = pow(1.0 + 4.0*A*t*t, -0.25);
        H1 = rPhi3 * log(1.0 + w1*(1.0 - g));
        ec1 = ec_lsda1 + H1;
        // printf("point %3d, ec0 %.7E, ec1 %.7E\n", i, ec0, ec1);
        // interpolate and extrapolate epsilon_c
        double fc;
        if (alphaP > 2.5) {
            fc = -dc*exp(c2c / (1.0 - alphaP));
        }
        else if (alphaP > 0.0) {
            double alpha2, alpha3, alpha4, alpha5, alpha6, alpha7;
            alpha2 = alphaP *alphaP; alpha3 = alpha2*alphaP;
            alpha4 = alpha3*alphaP; alpha5 = alpha4*alphaP;
            alpha6 = alpha5*alphaP; alpha7 = alpha6*alphaP;
            fc = 1.0 + (-0.64)*alphaP + (-0.4352)*alpha2 + (-1.535685604549)*alpha3 + 3.061560252175*alpha4
                + (-1.915710236206)*alpha5 + 0.516884468372*alpha6 + (-0.051848879792)*alpha7;
        }
        else {
            fc = exp(-c1c*alphaP / (1.0 - alphaP));
        }
        epsilonc[i] = ec1 + fc*(ec0 - ec1);
        // compute variation of epsilon_c^0
        DzetaDnup = zeta_dzetadnup_dzetadndn[1][i];
        DzetaDndn = zeta_dzetadnup_dzetadndn[2][i];
        DrsDn = -4.0*M_PI/9.0 * pow(4.0*M_PI/3.0*rho[i], -4.0/3.0);
        DdxDnup = (4.0/3.0*pow(1.0 + zeta, 1.0/3.0) - 4.0/3.0*pow(1.0 - zeta, 1.0/3.0))/2.0 * DzetaDnup;
        DdxDndn = (4.0/3.0*pow(1.0 + zeta, 1.0/3.0) - 4.0/3.0*pow(1.0 - zeta, 1.0/3.0))/2.0 * DzetaDndn;
        DGcDnup = -2.3631*DdxDnup*(1.0 - pow(zeta, 12.0)) + (1.0 - 2.3631*(dx - 1))*(-12.0*pow(zeta, 11.0)*DzetaDnup);
        // if (i==4) {
        //     printf("point 4, DdxDnup %10.8f, DzetaDnup %10.8f\n", DdxDnup, DzetaDnup);
        // }
        DGcDndn = -2.3631*DdxDndn*(1.0 - pow(zeta, 12.0)) + (1.0 - 2.3631*(dx - 1))*(-12.0*pow(zeta, 11.0)*DzetaDndn);
        DgInf0sDs = -0.25*pow(1.0 + 4.0*xiInf0*s*s, -1.25) * (4.0*xiInf0*2.0*s);
        DgInf0sDn = DgInf0sDs * s_dsdn_dsddn[1][i];
        DgInf0sDDn = DgInf0sDs * s_dsdn_dsddn[2][i];
        DecLDA0Dn = b1c*(0.5*b2c/sqrRs + b3c) / pow(1.0 + b2c*sqrRs + b3c*rs, 2.0) * DrsDn;
        Dw0Dn = (w0 + 1.0) * (-DecLDA0Dn/b1c);
        DH0Dn = b1c*(Dw0Dn*(1.0 - gInf0s) - w0*DgInf0sDn) / (1.0 + w0*(1.0 - gInf0s));
        DH0DDn = b1c*(-w0*DgInf0sDDn) / (1.0 + w0*(1.0 - gInf0s));
        // double Dec0Dn = (DecLDA0Dn + DH0Dn)*Gc + (ec0 + H0)*DGcDn;
        Dec0Dnup = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0 + H0)*DGcDnup;
        // if (i==4) {
        //     printf("point 4, DecLDA0Dn %10.8f, DH0Dn %10.8f, Gc %10.8f, ecLDA0 %10.8f, DGcDnup %10.8f\n", DecLDA0Dn, DH0Dn, Gc, ecLDA0, DGcDnup);
        // }
        Dec0Dndn = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0 + H0)*DGcDndn;
        Dec0DDn = DH0DDn*Gc;
        // compute variation of epsilon_c^1
        DgcrsDrs = Decrs1_Drs - Decrs0_Drs + Dac_Drs/1.709921;
	    Dec_lsda1_Drs = Decrs0_Drs + f_zeta * (zeta4 * DgcrsDrs - Dac_Drs/1.709921);
        // if (i==4) {
        //     printf("point 4, Decrs0_Drs %10.8f, f_zeta %10.8f, zeta4 %10.8f, DgcrsDrs %10.8f, Dac_Drs %10.8f\n", Decrs0_Drs, f_zeta, zeta4, DgcrsDrs, Dac_Drs);
        // }
	    Dfzeta4_Dzeta = 4.0 * (zeta*zeta*zeta) * f_zeta + fp_zeta * zeta4;
	    Dec_lsda1_Dzeta = Dfzeta4_Dzeta * gcrs - fp_zeta * ac/1.709921;
	    Dec_lsda1Dnup = (- rs * 1.0/3.0 * Dec_lsda1_Drs - zeta * Dec_lsda1_Dzeta + Dec_lsda1_Dzeta) / rho[i];
	    Dec_lsda1Dndn = (- rs * 1.0/3.0 * Dec_lsda1_Drs - zeta * Dec_lsda1_Dzeta - Dec_lsda1_Dzeta) / rho[i];
        // if (i==4) {
        //     printf("point 4, rs %10.8f, Dec_lsda1_Drs %10.8f, zeta %10.8f, Dec_lsda1_Dzeta %10.8f\n", rs, Dec_lsda1_Drs, zeta, Dec_lsda1_Dzeta);
        // }
        // double denominatorInLogLSDA1 = 2.0*AA*(beta1*sqrRs + beta2*rs + beta3*sqrRs*rs + beta4*pow(rs, p + 1.0));
        // double Dec_lsda1Dn = -(rs/rho[i]/3.0) * (-2.0*AA*alpha1*log(1.0 + 1.0/denominatorInLogLSDA1)
        //     -((-2.0*AA*(1.0 + alpha1*rs))*(AA*(beta1/sqrRs + 2.0*beta2 + 3.0*beta3*sqrRs + 2.0*(p + 1.0)*beta4*pow(rs, p))))
        //     / (denominatorInLogLSDA1*denominatorInLogLSDA1 + denominatorInLogLSDA1)); // from LDA_PW. If spin is added, the formula needs to be modified!
        // printf("point %3d, epsilonc %.7E, Dec0Dn %.7E, Dec_lsda1Dn %.7E\n", i, epsilonc[i], Dec0Dn, Dec_lsda1Dn);
        DbetaDn = 0.066725*(0.1*(1.0 + 0.1778*rs) - 0.1778*(1.0 + 0.1*rs)) / (1.0 + 0.1778*rs) / (1.0 + 0.1778*rs) * DrsDn;
        DphiDnup = 0.5*(2.0/3.0*pow(1.0 + zeta, -1.0/3.0) - 2.0/3.0*pow(1.0 - zeta, -1.0/3.0)) * DzetaDnup;
        DphiDndn = 0.5*(2.0/3.0*pow(1.0 + zeta, -1.0/3.0) - 2.0/3.0*pow(1.0 - zeta, -1.0/3.0)) * DzetaDndn;
        DtDnup = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * (phi*sqrRs*s_dsdn_dsddn[1][i] - s*(DphiDnup*sqrRs + phi*DrsDn/(2.0*sqrRs))) / (phi*phi*rs);
        DtDndn = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * (phi*sqrRs*s_dsdn_dsddn[1][i] - s*(DphiDndn*sqrRs + phi*DrsDn/(2.0*sqrRs))) / (phi*phi*rs);
        DtDDn = t*s_dsdn_dsddn[2][i]/s;
        Dw1Dnup = (w1 + 1.0) * (-(rPhi3*Dec_lsda1Dnup - r*ec_lsda1*(3.0*phi*phi*DphiDnup)) / rPhi3 / rPhi3);
        Dw1Dndn = (w1 + 1.0) * (-(rPhi3*Dec_lsda1Dndn - r*ec_lsda1*(3.0*phi*phi*DphiDndn)) / rPhi3 / rPhi3);
        DADnup = (w1*DbetaDn - beta*Dw1Dnup) / (r*w1*w1);
        DADndn = (w1*DbetaDn - beta*Dw1Dndn) / (r*w1*w1);
        DgDnup = -0.25*pow(1.0 + 4.0*A*t*t, -1.25) * (4.0*(DADnup*t*t + 2.0*A*t*DtDnup));
        DgDndn = -0.25*pow(1.0 + 4.0*A*t*t, -1.25) * (4.0*(DADndn*t*t + 2.0*A*t*DtDndn));
        DgDDn = -0.25*pow(1.0 + 4.0*A*t*t, -1.25) * (4.0*2.0*A*t*DtDDn);
        DH1Dnup = 3*r*phi*phi*DphiDnup*log(1.0 + w1*(1.0 - g)) + rPhi3*(Dw1Dnup*(1.0 - g) - w1*DgDnup) / (1.0 + w1*(1.0 - g));
        DH1Dndn = 3*r*phi*phi*DphiDndn*log(1.0 + w1*(1.0 - g)) + rPhi3*(Dw1Dndn*(1.0 - g) - w1*DgDndn) / (1.0 + w1*(1.0 - g));
        DH1DDn = rPhi3*(-w1*DgDDn) / (1.0 + w1*(1.0 - g));
        Dec1Dnup = Dec_lsda1Dnup + DH1Dnup;
        Dec1Dndn = Dec_lsda1Dndn + DH1Dndn;
        Dec1DDn = DH1DDn;
        // if (i == 4) {
        //     printf("point 4, Dec_lsda1Dnup %10.8f, DH1Dnup %10.8f, DphiDnup %10.8f, Dw1Dnup %10.8f, DgDnup %10.8f\n", Dec_lsda1Dnup, DH1Dnup, DphiDnup, Dw1Dnup, DgDnup);
        // }

        // compute variation of f_c and epsilon_c
        double DfcDalpha;
        if (alphaP > 2.5) {
            DfcDalpha = fc*(c2c/(1.0 - alphaP)/(1.0 - alphaP));
        }
        else if (alphaP > 0.0) {
            double alpha2 = alphaP *alphaP; double alpha3 = alpha2*alphaP;
            double alpha4 = alpha3*alphaP; double alpha5 = alpha4*alphaP;
            double alpha6 = alpha5*alphaP;
            DfcDalpha = (-0.64) + (-0.4352)*alphaP*2.0 + (-1.535685604549)*alpha2*3.0 + 3.061560252175*alpha3*4.0 
                + (-1.915710236206)*alpha4*5.0 + 0.516884468372*alpha5*6.0 + (-0.051848879792)*alpha6*7.0;
        }
        else {
            DfcDalpha = fc*(-c1c/(1.0 - alphaP)/(1.0 - alphaP));
        }
        DfcDnup = DfcDalpha*alphaP_dadnup_dadndn_daddn_dadtau[1][i];
        DfcDndn = DfcDalpha*alphaP_dadnup_dadndn_daddn_dadtau[2][i];
        DfcDDn = DfcDalpha*alphaP_dadnup_dadndn_daddn_dadtau[3][i];
        DfcDtau = DfcDalpha*alphaP_dadnup_dadndn_daddn_dadtau[4][i];
        DepsiloncDnup = Dec1Dnup + fc*(Dec0Dnup - Dec1Dnup) + DfcDnup*(ec0 - ec1);
        DepsiloncDndn = Dec1Dndn + fc*(Dec0Dndn - Dec1Dndn) + DfcDndn*(ec0 - ec1);
        DepsiloncDDn = Dec1DDn + fc*(Dec0DDn -Dec1DDn) + DfcDDn*(ec0 - ec1);
        DepsiloncDtau = DfcDtau*(ec0 - ec1);
        vc[i] = epsilonc[i] + rho[i]*DepsiloncDnup;
        vc[length+i] = epsilonc[i] + rho[i]*DepsiloncDndn;
        v2c[i] = rho[i]*DepsiloncDDn;
        v3c[i] = rho[i]*DepsiloncDtau;
        // if (i == 4) 
        //     printf("point 4, Dec1Dnup %10.8f, fc %10.8f, Dec0Dnup %10.8f, Dec1Dnup %10.8f, DfcDnup %10.8f, ec0 %10.8f, ec1 %10.8f\n", Dec1Dnup, fc, Dec0Dnup, Dec1Dnup, DfcDnup, ec0, ec1);
    }
}