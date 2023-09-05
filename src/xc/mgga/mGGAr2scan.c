/**
 * @file    MGGAr2scan.c
 * @brief   This file contains the functions for r2scan functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Furness, James W., Aaron D. Kaplan, Jinliang Ning, John P. Perdew, and Jianwei Sun. 
 * "Accurate and numerically efficient r2SCAN meta-generalized gradient approximation." 
 * The journal of physical chemistry letters 11, no. 19 (2020): 8208-8215.
 * Physical review letters 115, no. 3 (2015): 036402.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "mGGAr2scan.h"
#include "isddft.h"

void r2scanx(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x) {
    double **p_dpdn_dpddn = (double**)malloc(sizeof(double*)*3); // variable p, and dp/dn, dp/d|\nabla n|
    double *normDrho = (double*)malloc(DMnd * sizeof(double));
    int i;
    for (i = 0; i < 3; i++) {
        p_dpdn_dpddn[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(p_dpdn_dpddn[i] != NULL);
    }
    double **alpha_dadn_daddn_dadtau = (double**)malloc(sizeof(double*)*4); // variable \alpha, and d\alpha/dn, d\alpha/d|\nabla n|, d\alpha/d\tau
    for (i = 0; i < 4; i++) {
        alpha_dadn_daddn_dadtau[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alpha_dadn_daddn_dadtau[i] != NULL);
    }

    for (i = 0; i < DMnd; i++) {
        normDrho[i] = sqrt(sigma[i]);
    }

    basic_r2scanx_variables(DMnd, rho, normDrho, tau, p_dpdn_dpddn, alpha_dadn_daddn_dadtau);

    Calculate_r2scanx(DMnd, rho, p_dpdn_dpddn, alpha_dadn_daddn_dadtau, ex, vx, v2x, v3x);

    for (i = 0; i < DMnd; i++) {
        v2x[i] = v2x[i] / normDrho[i];
    }

    for (i = 0; i < 3; i++) {
        free(p_dpdn_dpddn[i]);
    }
    free(p_dpdn_dpddn);
    free(normDrho);
    for (i = 0; i < 4; i++) {
        free(alpha_dadn_daddn_dadtau[i]);
    }
    free(alpha_dadn_daddn_dadtau);
}

void basic_r2scanx_variables(int length, double *rho, double *normDrho, double *tau, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau) {
    int i;
    double threeMPi2_1o3 = pow(3.0*M_PI*M_PI, 1.0/3.0);
    double threeMPi2_2o3 = threeMPi2_1o3*threeMPi2_1o3;
    double eta = 0.001;
    for (i = 0; i < length; i++) {
        double s = normDrho[i] / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0));
        p_dpdn_dpddn[0][i] = s*s;
        double tauw = normDrho[i]*normDrho[i] / (8*rho[i]);
        double tauUnif = 3.0/10.0 * threeMPi2_2o3 * pow(rho[i], 5.0/3.0);
        alpha_dadn_daddn_dadtau[0][i] = (tau[i] - tauw) / (tauUnif + eta*tauw);

        double dsdn = -2.0*normDrho[i] / (3.0 * threeMPi2_1o3 * pow(rho[i], 7.0/3.0)); // ds/dn
        p_dpdn_dpddn[1][i] = 2*s*dsdn;
        double dsddn = 1.0 / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0)); // ds/d|\nabla n|
        p_dpdn_dpddn[2][i] = 2*s*dsddn;
        double DtauwDn = -normDrho[i]*normDrho[i] / (8*rho[i]*rho[i]);
        double DtauwDDn = normDrho[i] / (4*rho[i]);
        double DtauUnifDn = threeMPi2_2o3 / 2.0 * pow(rho[i], 2.0/3.0);
        alpha_dadn_daddn_dadtau[1][i] = (-DtauwDn*(tauUnif + eta*tauw) - (tau[i] - tauw)*(DtauUnifDn + eta*DtauwDn)) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/dn
        alpha_dadn_daddn_dadtau[2][i] = (-DtauwDDn*(tauUnif + eta*tauw) - (tau[i] - tauw)*eta*DtauwDDn) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/d|\nabla n|
        alpha_dadn_daddn_dadtau[3][i] = 1.0 / (tauUnif + eta*tauw); // d\alpha/d\tau
    }
}

void Calculate_r2scanx(int length, double *rho, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx, double *v2x, double *v3x) {
    // constants for h_x^1
    double k0 = 0.174;
    double k1 = 0.065;
    double mu_ak = 10.0/81.0;
    double eta = 0.001;
    double Ceta = 20.0/27.0 + eta*5.0/3.0;
    double C2 = -0.162742;
    double dp2 = 0.361;
    // constant h_x^0
    double hx0 = 1.0 + k0;
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
        double p = p_dpdn_dpddn[0][i];
        double p14 = pow(p, 0.25);
        double x = (Ceta*C2*exp(-p_dpdn_dpddn[0][i]*p_dpdn_dpddn[0][i] / pow(dp2, 4.0)) + mu_ak) * p_dpdn_dpddn[0][i];
        double hx1 = 1.0 + k1 - k1/(1.0 + x/k1);
        // interpolate and extrapolate h_x to get F_x
        // switching function f_x
        double fx;
        double alpha = alpha_dadn_daddn_dadtau[0][i];
        if (alpha > 2.5) {
            fx = -dx*exp(c2x / (1.0 - alpha));
        }
        else if (alpha > 0.0) {
            double alpha2 = alpha *alpha; double alpha3 = alpha2*alpha;
            double alpha4 = alpha3*alpha; double alpha5 = alpha4*alpha;
            double alpha6 = alpha5*alpha; double alpha7 = alpha6*alpha;
            fx = 1.0 + (-0.667)*alpha + (-0.4445555)*alpha2 + (-0.663086601049)*alpha3 + 1.451297044490*alpha4 
                + (-0.887998041597)*alpha5 + 0.234528941479*alpha6 + (-0.023185843322)*alpha7;
        }
        else {
            fx = exp(-c1x*alpha / (1.0 - alpha));
        }
        double gx = 1.0 - exp(-a1/p14);
        double Fx = (hx1 + fx*(hx0 - hx1))*gx;
        epsilonx[i] = epsilon_xUnif*Fx;

        double DxDp = (Ceta*C2 * exp(-p*p / pow(dp2, 4.0)) + mu_ak) + Ceta*C2*exp(-p*p / pow(dp2, 4.0))*(-2.0*p/ pow(dp2, 4.0)) * p;
        double DxDn = DxDp*p_dpdn_dpddn[1][i];
        double DxDDn = DxDp*p_dpdn_dpddn[2][i];

        double DgxDn = -exp(-a1/p14)*(a1/4.0/p14/p)*p_dpdn_dpddn[1][i];
        double DgxDDn = -exp(-a1/p14)*(a1/4.0/p14/p)*p_dpdn_dpddn[2][i];
        double Dhx1Dx = 1.0 / ((1.0 + x/k1)*(1.0 + x/k1));
        double Dhx1Dn = DxDn * Dhx1Dx;
        double Dhx1DDn = DxDDn * Dhx1Dx;

        double DfxDalpha;
        if (alpha > 2.5) {
            DfxDalpha = -dx*exp(c2x/(1.0 - alpha)) * (c2x/(1.0 - alpha)/(1.0 - alpha));
        }
        else if (alpha > 0.0) {
            double alpha2 = alpha *alpha; double alpha3 = alpha2*alpha;
            double alpha4 = alpha3*alpha; double alpha5 = alpha4*alpha;
            double alpha6 = alpha5*alpha;
            DfxDalpha = (-0.667) + (-0.4445555)*alpha*2.0 + (-0.663086601049)*alpha2*3.0 + 1.451297044490*alpha3*4.0
                + (-0.887998041597)*alpha4*5.0 + 0.234528941479*alpha5*6.0 + (-0.023185843322)*alpha6*7.0;
        }
        else {
            DfxDalpha = exp(-c1x*alpha/(1.0 - alpha)) * (-c1x/(1.0 - alpha)/(1 - alpha));
        }
        double DfxDn = DfxDalpha*alpha_dadn_daddn_dadtau[1][i];
        double DfxDDn = DfxDalpha*alpha_dadn_daddn_dadtau[2][i];
        double DfxDtau = DfxDalpha*alpha_dadn_daddn_dadtau[3][i];
        double DFxDn = (hx1 + fx*(hx0 - hx1))*DgxDn + gx*(1.0 - fx)*Dhx1Dn + gx*(hx0 - hx1)*DfxDn;
        double DFxDDn = (hx1 + fx*(hx0 - hx1))*DgxDDn + gx*(1.0 - fx)*Dhx1DDn + gx*(hx0 - hx1)*DfxDDn;
        double DFxDtau = gx*(hx0 - hx1)*DfxDtau;

        double Depsilon_xUnifDn = -pow(3.0*M_PI*M_PI, 1.0/3.0) / (4.0*M_PI) * pow(rho[i], -2.0/3.0);
        vx[i] = (epsilon_xUnif + rho[i]*Depsilon_xUnifDn)*Fx + rho[i]*epsilon_xUnif*DFxDn;
        v2x[i] = rho[i]*epsilon_xUnif*DFxDDn;
        v3x[i] = rho[i]*epsilon_xUnif*DFxDtau;
    }
}

void r2scanc(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c) {
    double **s_dsdn_dsddn = (double**)malloc(sizeof(double*)*3); // variable s, and ds/dn, ds/d|\nabla n|
    double **p_dpdn_dpddn = (double**)malloc(sizeof(double*)*3); // variable p, and dp/dn, dp/d|\nabla n|
    double *normDrho = (double*)malloc(DMnd * sizeof(double));
    int i;
    for (i = 0; i < 3; i++) {
        s_dsdn_dsddn[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(s_dsdn_dsddn[i] != NULL);
    }
    for (i = 0; i < 3; i++) {
        p_dpdn_dpddn[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(p_dpdn_dpddn[i] != NULL);
    }
    double **alpha_dadn_daddn_dadtau = (double**)malloc(sizeof(double*)*4); // variable \alpha, and d\alpha/dn, d\alpha/d|\nabla n|, d\alpha/d\tau
    for (i = 0; i < 4; i++) {
        alpha_dadn_daddn_dadtau[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alpha_dadn_daddn_dadtau[i] != NULL);
    }

    for (i = 0; i < DMnd; i++) {
        normDrho[i] = sqrt(sigma[i]);
    }

    basic_r2scanc_variables(DMnd, rho, normDrho, tau, s_dsdn_dsddn, p_dpdn_dpddn, alpha_dadn_daddn_dadtau);

    Calculate_r2scanc(DMnd, rho, s_dsdn_dsddn, p_dpdn_dpddn, alpha_dadn_daddn_dadtau, ec, vc, v2c, v3c);

    for (i = 0; i < DMnd; i++) {
        v2c[i] = v2c[i] / normDrho[i];
    }

    for (i = 0; i < 3; i++) {
        free(s_dsdn_dsddn[i]);
    }
    for (i = 0; i < 3; i++) {
        free(p_dpdn_dpddn[i]);
    }
    free(s_dsdn_dsddn);
    free(p_dpdn_dpddn);
    free(normDrho);
    for (i = 0; i < 4; i++) {
        free(alpha_dadn_daddn_dadtau[i]);
    }
    free(alpha_dadn_daddn_dadtau);
}

void basic_r2scanc_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau) {
    int i;
    double threeMPi2_1o3 = pow(3.0*M_PI*M_PI, 1.0/3.0);
    double threeMPi2_2o3 = threeMPi2_1o3*threeMPi2_1o3;
    double eta = 0.001;
    for (i = 0; i < length; i++) {
        double s = normDrho[i] / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0));
        s_dsdn_dsddn[0][i] = s;
        p_dpdn_dpddn[0][i] = s*s;
        double tauw = normDrho[i]*normDrho[i] / (8*rho[i]);
        double tauUnif = 3.0/10.0 * threeMPi2_2o3 * pow(rho[i], 5.0/3.0);
        alpha_dadn_daddn_dadtau[0][i] = (tau[i] - tauw) / (tauUnif + eta*tauw);

        double dsdn = -2.0*normDrho[i] / (3.0 * threeMPi2_1o3 * pow(rho[i], 7.0/3.0)); // ds/dn
        s_dsdn_dsddn[1][i] = dsdn;
        p_dpdn_dpddn[1][i] = 2*s*dsdn;
        double dsddn = 1.0 / (2.0 * threeMPi2_1o3 * pow(rho[i], 4.0/3.0)); // ds/d|\nabla n|
        s_dsdn_dsddn[2][i] = dsddn;
        p_dpdn_dpddn[2][i] = 2*s*dsddn;
        double DtauwDn = -normDrho[i]*normDrho[i] / (8*rho[i]*rho[i]);
        double DtauwDDn = normDrho[i] / (4*rho[i]);
        double DtauUnifDn = threeMPi2_2o3 / 2.0 * pow(rho[i], 2.0/3.0);
        alpha_dadn_daddn_dadtau[1][i] = (-DtauwDn*(tauUnif + eta*tauw) - (tau[i] - tauw)*(DtauUnifDn + eta*DtauwDn)) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/dn
        alpha_dadn_daddn_dadtau[2][i] = (-DtauwDDn*(tauUnif + eta*tauw) - (tau[i] - tauw)*eta*DtauwDDn) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/d|\nabla n|
        alpha_dadn_daddn_dadtau[3][i] = 1.0 / (tauUnif + eta*tauw); // d\alpha/d\tau
    }
}

void Calculate_r2scanc(int length, double *rho, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau, double *epsilonc, double *vc, double *v2c, double *v3c) {
    // constants for epsilon_c^0
    double b1c = 0.0285764;
    double b2c = 0.0889;
    double b3c = 0.125541;
    double betaConst = 0.06672455060314922;
    double betaRsInf = betaConst*0.1/0.1778;
    double f0 = -0.9;
    // constants for epsilon_c LSDA1
    double AA = 0.0310907;
	double alpha1 = 0.21370;
	double beta1 = 7.5957;
	double beta2 = 3.5876;
	double beta3 = 1.6382;
	double beta4 = 0.49294;
    // constants for H1 of epsilon_c^1
    double r = 0.031091;
    double eta = 0.001;
    double dp2 = 0.361;
    // constants for switching function f_c
    double c1c = 0.64;
    double c2c = 1.5;
    double dc = 0.7;
    int i;
    for (i = 0; i < length; i++) {
        double zeta = 0.0;
        double phi = 1.0;
        double dx = 1.0;
        double s = s_dsdn_dsddn[0][i];
        double p = p_dpdn_dpddn[0][i];
        double alpha = alpha_dadn_daddn_dadtau[0][i];
        double rs = pow(0.75/(M_PI*rho[i]), 1.0/3.0);
        // epsilon_c^0 (\alpha approach 0)
        double ecLDA0 = -b1c / (1.0 + b2c*sqrt(rs) + b3c*rs);
        double cx0 = -3.0/(4.0*M_PI) * pow(9.0*M_PI/4.0, 1.0/3.0);
        double Gc = 1.0;
        double w0 = exp(-ecLDA0/b1c) - 1.0;
        double chiInf0 = pow(3.0*M_PI*M_PI/16.0, 2.0/3.0) * (betaRsInf*1.0/(cx0 - f0)); // \xi_{r_s->\inf}(\zeta=0), 0.128026
        double gInf0s = pow(1.0 + 4.0*chiInf0*s*s, -0.25);
        double H0 = b1c*log(1.0 + w0*(1.0 - gInf0s));
        double ec0 = (ecLDA0 + H0)*Gc;
        // epsilon_c^1 (\alpha approach 1)
        double sqrRs = sqrt(rs);
        double beta = betaConst * (1.0 + 0.1*rs) / (1.0 + 0.1778*rs);
        // epsilon_c LSDA1
        double ec_q0 = -2.0*AA*(1.0 + alpha1*rs);
        double ec_q1 = 2.0*AA*(beta1*sqrRs + beta2*rs + beta3*rs*sqrRs + beta4*rs*rs);
        double ec_q1p = AA * (beta1/sqrRs + 2.0*beta2 + 3.0*beta3*sqrRs + 4.0*beta4*rs);
        double ec_den = 1.0 / (ec_q1*ec_q1 + ec_q1);
        double ec_log = -log(ec_q1*ec_q1*ec_den);
        double ec_lsda1 = ec_q0*ec_log;
        double declsda1_drs = -2.0 * AA * alpha1 * ec_log - ec_q0*ec_q1p*ec_den;
        // H1
        double rPhi3 = r*phi*phi*phi;
        double w1 = exp(-ec_lsda1/rPhi3) - 1.0;
        double t = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * s/(phi*sqrRs);
        double y = beta/(r*w1) * t*t;
        double deltafc2 = 1.0*(-0.64) + 2.0*(-0.4352) + 3.0*(-1.535685604549) + 4.0*3.061560252175 
            + 5.0*(-1.915710236206) + 6.0*0.516884468372 + 7.0*(-0.051848879792);
        double ec_lsda0 = ecLDA0;
        double declsda0_drs = b1c*(0.5*b2c/sqrRs + b3c) / pow(1.0 + b2c*sqrRs + b3c*rs, 2.0);

        double deltayPart1 = deltafc2 / (27.0*r*w1);
        double deltayPart2 = 20.0*rs*(declsda0_drs - declsda1_drs) - 45.0*eta*(ec_lsda0 - ec_lsda1);
        double deltayPart3 = p*exp(-p*p / pow(dp2, 4.0));
        double deltay = deltayPart1 * deltayPart2 * deltayPart3;

        double g = pow(1.0 + 4.0*(y - deltay), -0.25);
        double H1 = rPhi3 * log(1.0 + w1*(1.0 - g));
        double ec1 = ec_lsda1 + H1;
        // interpolate and extrapolate epsilon_c
        double fc;
        if (alpha > 2.5) {
            fc = -dc*exp(c2c / (1.0 - alpha));
        }
        else if (alpha > 0.0) {
            double alpha2 = alpha *alpha; double alpha3 = alpha2*alpha;
            double alpha4 = alpha3*alpha; double alpha5 = alpha4*alpha;
            double alpha6 = alpha5*alpha; double alpha7 = alpha6*alpha;
            fc = 1.0 + (-0.64)*alpha + (-0.4352)*alpha2 + (-1.535685604549)*alpha3 + 3.061560252175*alpha4
                + (-1.915710236206)*alpha5 + 0.516884468372*alpha6 + (-0.051848879792)*alpha7;
        }
        else {
            fc = exp(-c1c*alpha / (1.0 - alpha));
        }
        epsilonc[i] = ec1 + fc*(ec0 - ec1);
        // compute variation of epsilon_c^0
        double DzetaDn = 0.0; // no spin
        double DrsDn = -4.0*M_PI/9.0 * pow(4.0*M_PI/3.0*rho[i], -4.0/3.0);
        double DdxDn = (4.0/3.0*pow(1.0 + zeta, 1.0/3.0) - 4.0/3.0*pow(1.0 - zeta, 1.0/3.0))*DzetaDn; // when there is no spin, it should be 0
        double DGcDn = -2.3631*DdxDn*(1.0 - pow(zeta, 12.0)) + (1.0 - 2.3631*(dx - 1))*(12.0*pow(zeta, 11.0)*DzetaDn); // when there is no spin, it should be 0
        double DgInf0sDs = -0.25*pow(1.0 + 4.0*chiInf0*s*s, -1.25) * (4.0*chiInf0*2.0*s);
        double DgInf0sDn = DgInf0sDs * s_dsdn_dsddn[1][i];
        double DgInf0sDDn = DgInf0sDs * s_dsdn_dsddn[2][i];
        double DecLDA0Dn = b1c*(0.5*b2c/sqrRs + b3c) / pow(1.0 + b2c*sqrRs + b3c*rs, 2.0) * DrsDn;
        double Dw0Dn = (w0 + 1.0) * (-DecLDA0Dn/b1c);
        double DH0Dn = b1c*(Dw0Dn*(1.0 - gInf0s) - w0*DgInf0sDn) / (1.0 + w0*(1.0 - gInf0s));
        double DH0DDn = b1c*(-w0*DgInf0sDDn) / (1.0 + w0*(1.0 - gInf0s));
        double Dec0Dn = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0 + H0)*DGcDn;
        double Dec0DDn = DH0DDn*Gc;
        // compute variation of epsilon_c^1
        double dec0log_drs = -ec_q1p * ec_den;
        double dec0q0_drs = -2.0 * AA * alpha1;
        double dec0q1p_drs = AA * ((-0.5)*beta1/sqrRs/rs + 3.0*0.5*beta3/sqrRs + 4.0*beta4);
        double dec0den_drs = -(2.0*ec_q1*ec_q1p + ec_q1p) / pow(ec_q1*ec_q1 + ec_q1, 2.0);
        double d2ecrs0_drs2 = -2.0 * AA * alpha1 * dec0log_drs 
            - (dec0q0_drs * ec_q1p * ec_den + ec_q0 * dec0q1p_drs * ec_den + ec_q0 * ec_q1p * dec0den_drs);
        double d2eclsda1_drs2 = d2ecrs0_drs2;
        double Ddeclsda1_drsDn = d2eclsda1_drs2*DrsDn;
        double Dec_lsda1Dn = (-rs * 1.0/3.0 * declsda1_drs) / rho[i];
        double DbetaDn = 0.066725*(0.1*(1.0+0.1778*rs) - 0.1778*(1.0+0.1*rs)) / pow(1.0+0.1778*rs, 2.0) * DrsDn;
        double DphiDn = 0.5*(2.0/3.0*pow(1.0 + zeta, -1.0/3.0) - 2.0/3.0*pow(1.0 - zeta, -1.0/3.0)) * DzetaDn; // no spin, it should be 0
        double DtDn = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * (phi*sqrRs*s_dsdn_dsddn[1][i] - s*(DphiDn*sqrRs + phi*DrsDn/(2.0*sqrRs))) / (phi*phi*rs);
        double DtDDn = t*s_dsdn_dsddn[2][i]/s;
        double Dw1Dn = (w1 + 1.0) * (-(rPhi3*Dec_lsda1Dn - r*ec_lsda1*(3.0*phi*phi*DphiDn)) / rPhi3 / rPhi3);
        double DyDn = (w1*DbetaDn - beta*Dw1Dn) / (r*w1*w1) * (t*t) + beta/(r*w1) * (2*t)*DtDn;
        double DyDDn = beta/(r*w1) * (2*t)*DtDDn;
        double Declsda0Dn = declsda0_drs*DrsDn;
        double d2eclsda0_drs2 = b1c*((0.5*b2c*(-0.5)/rs/sqrRs)*pow(1.0 + b2c*sqrRs + b3c*rs, 2.0) 
            - (0.5*b2c/sqrRs + b3c)*2.0*(1 + b2c*sqrRs + b3c*rs)*(0.5*b2c/sqrRs + b3c))
            / pow(1.0 + b2c*sqrRs + b3c*rs, 4.0);
        double Ddeclsda0_drsDn = d2eclsda0_drs2*DrsDn;

        double d_deltayPart1_dn = 0.0 + 0.0 + deltafc2 / (27.0*r) * (-1.0)/w1/w1 * Dw1Dn;
        double d_deltayPart2_dn = 20.0*(declsda0_drs - declsda1_drs)*DrsDn + 20.0*rs*(Ddeclsda0_drsDn - Ddeclsda1_drsDn)
            - 45.0*eta*(Declsda0Dn - Dec_lsda1Dn);
        double d_deltayPart3_dp = exp(-p*p/pow(dp2, 4.0)) + p*exp(-p*p/pow(dp2, 4.0))*(-2.0*p/pow(dp2, 4.0));
        double DdeltayDn = d_deltayPart1_dn*deltayPart2*deltayPart3 + deltayPart1*d_deltayPart2_dn*deltayPart3
            + deltayPart1*deltayPart2*d_deltayPart3_dp*p_dpdn_dpddn[1][i];
        double DdeltayDDn = deltayPart1*deltayPart2*d_deltayPart3_dp*p_dpdn_dpddn[2][i];

        double DgDn = -0.25*pow(1.0 + 4.0*(y - deltay), -1.25) * (4.0*(DyDn - DdeltayDn));
        double DgDDn = -0.25*pow(1.0 + 4.0*(y - deltay), -1.25) * (4.0*(DyDDn - DdeltayDDn));
        double DH1Dn = 3*r*phi*phi*DphiDn*log(1.0 + w1*(1.0 - g)) + rPhi3*(Dw1Dn*(1.0 - g) - w1*DgDn) / (1.0 + w1*(1.0 - g));
        double DH1DDn = rPhi3*(-w1*DgDDn) / (1.0 + w1*(1.0 - g));
        double Dec1Dn = Dec_lsda1Dn + DH1Dn;
        double Dec1DDn = DH1DDn;
        // compute variation of f_c and epsilon_c
        double DfcDalpha;
        if (alpha > 2.5) {
            DfcDalpha = fc*(c2c/(1.0 - alpha)/(1.0 - alpha));
        }
        else if (alpha > 0) {
            double alpha2 = alpha *alpha; double alpha3 = alpha2*alpha;
            double alpha4 = alpha3*alpha; double alpha5 = alpha4*alpha;
            double alpha6 = alpha5*alpha;
            DfcDalpha = (-0.64) + (-0.4352)*alpha*2.0 + (-1.535685604549)*alpha2*3.0 + 3.061560252175*alpha3*4.0 
                + (-1.915710236206)*alpha4*5.0 + 0.516884468372*alpha5*6.0 + (-0.051848879792)*alpha6*7.0;
        }
        else {
            DfcDalpha = fc*(-c1c/(1.0 - alpha)/(1.0 - alpha));
        }
        double DfcDn = DfcDalpha*alpha_dadn_daddn_dadtau[1][i];
        double DfcDDn = DfcDalpha*alpha_dadn_daddn_dadtau[2][i];
        double DfcDtau = DfcDalpha*alpha_dadn_daddn_dadtau[3][i];
        double DepsiloncDn = Dec1Dn + fc*(Dec0Dn - Dec1Dn) + DfcDn*(ec0 - ec1);
        double DepsiloncDDn = Dec1DDn + fc*(Dec0DDn - Dec1DDn) + DfcDDn*(ec0 - ec1);
        double DepsiloncDtau = DfcDtau*(ec0 - ec1);
        vc[i] = epsilonc[i] + rho[i]*DepsiloncDn;
        v2c[i] = rho[i]*DepsiloncDDn;
        v3c[i] = rho[i]*DepsiloncDtau;
    }
}

void r2scanx_spin(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x) { 
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

    r2scanx(DMnd, rho_up, sigma_up, tau_up, ex_up, vx_up, v2x_up, v3x_up);
    r2scanx(DMnd, rho_dn, sigma_dn, tau_dn, ex_dn, vx_dn, v2x_dn, v3x_dn);

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

void r2scanc_spin(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c) { 
    double **s_dsdn_dsddn_correlation = (double**)malloc(sizeof(double*)*3); // variable s, and ds/dn, ds/d|\nabla n|. n = n_up + n_dn
    double **p_dpdn_dpddn_correlation = (double**)malloc(sizeof(double*)*3); // variable p, and dp/dn, dp/d|\nabla n|. n = n_up + n_dn
    double *normDrho = (double*)malloc(3*DMnd * sizeof(double));
    int i;
    for (i = 0; i < 3; i++) {
        s_dsdn_dsddn_correlation[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(s_dsdn_dsddn_correlation[i] != NULL);
    }
    for (i = 0; i < 3; i++) {
        p_dpdn_dpddn_correlation[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(p_dpdn_dpddn_correlation[i] != NULL);
    }
    double **alpha_dadnup_dadndn_daddn_dadtau_correlation = (double**)malloc(sizeof(double*)*5); // variable \alpha, and d\alpha/dn_up, d\alpha/dn_dn, d\alpha/d|\nabla n|, d\alpha/d\tau. n = n_up + n_dn, tau = tau_up + tau_dn
    for (i = 0; i < 5; i++) {
        alpha_dadnup_dadndn_daddn_dadtau_correlation[i] = (double*)malloc(sizeof(double)*DMnd);
        assert(alpha_dadnup_dadndn_daddn_dadtau_correlation[i] != NULL);
    }
    double **zeta_dzetadnup_dzetadndn = (double**)malloc(sizeof(double*)*3);
    for (i = 0; i < 3; i++) {
        zeta_dzetadnup_dzetadndn[i] = (double*)malloc(sizeof(double*)*DMnd);
        assert(zeta_dzetadnup_dzetadndn[i] != NULL);
    }

    for (i = 0; i < 3*DMnd; i++) {
        normDrho[i] = sqrt(sigma[i]);
    }

    basic_r2scanc_spin_variables(DMnd, rho, normDrho, tau, s_dsdn_dsddn_correlation, p_dpdn_dpddn_correlation, alpha_dadnup_dadndn_daddn_dadtau_correlation, zeta_dzetadnup_dzetadndn);

    Calculate_r2scanc_spin(DMnd, rho, s_dsdn_dsddn_correlation, p_dpdn_dpddn_correlation, alpha_dadnup_dadndn_daddn_dadtau_correlation, zeta_dzetadnup_dzetadndn, ec, vc, v2c, v3c);

    for (i = 0; i < DMnd; i++) {
        v2c[i] = v2c[i] / normDrho[i];
    }

    for (i = 0; i < 3; i++) {
        free(s_dsdn_dsddn_correlation[i]);
    }
    free(s_dsdn_dsddn_correlation);
    for (i = 0; i < 3; i++) {
        free(p_dpdn_dpddn_correlation[i]);
    }
    free(p_dpdn_dpddn_correlation);
    free(normDrho);
    for (i = 0; i < 5; i++) {
        free(alpha_dadnup_dadndn_daddn_dadtau_correlation[i]);
    }
    free(alpha_dadnup_dadndn_daddn_dadtau_correlation);
    for (i = 0; i < 3; i++) {
        free(zeta_dzetadnup_dzetadndn[i]);
    }
    free(zeta_dzetadnup_dzetadndn);
}

void basic_r2scanc_spin_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn) {
    int i;
    double threeMPi2_1o3 = pow(3.0*M_PI*M_PI, 1.0/3.0);
    double threeMPi2_2o3 = threeMPi2_1o3*threeMPi2_1o3;
    double theRho, theNormDrho, theTau;
    double theZeta;
    double ds, DdsDnup, DdsDndn;
    double tauw, tauUnif, DtauwDn, DtauwDDn, DtauUnifDnup, DtauUnifDndn;
    double eta = 0.001;
    for (i = 0; i < length; i++) {
        theRho = rho[i];
        theNormDrho = normDrho[i];
        theTau = tau[i];
        s_dsdn_dsddn[0][i] = theNormDrho / (2.0 * threeMPi2_1o3 * pow(theRho, 4.0/3.0));
        p_dpdn_dpddn[0][i] = s_dsdn_dsddn[0][i]*s_dsdn_dsddn[0][i];
        zeta_dzetadnup_dzetadndn[0][i] = (rho[length+i] - rho[length*2+i]) / theRho;
        theZeta = zeta_dzetadnup_dzetadndn[0][i];
        tauw = theNormDrho*theNormDrho / (8.0*theRho);
        ds = (pow(1.0+theZeta, 5.0/3.0) + pow(1-theZeta, 5.0/3.0)) / 2.0;
        tauUnif = 3.0/10.0 * threeMPi2_2o3 * pow(theRho, 5.0/3.0) * ds;
        alpha_dadnup_dadndn_daddn_dadtau[0][i] = (tau[i] - tauw) / (tauUnif + eta*tauw);

        s_dsdn_dsddn[1][i] = -2.0*theNormDrho / (3.0 * threeMPi2_1o3 * pow(theRho, 7.0/3.0)); // ds/dn
        p_dpdn_dpddn[1][i] = 2*s_dsdn_dsddn[0][i]*s_dsdn_dsddn[1][i];
        s_dsdn_dsddn[2][i] = 1.0 / (2.0 * threeMPi2_1o3 * pow(theRho, 4.0/3.0)); // ds/d|\nabla n|
        p_dpdn_dpddn[2][i] = 2*s_dsdn_dsddn[0][i]*s_dsdn_dsddn[2][i];
        zeta_dzetadnup_dzetadndn[1][i] = 2.0*rho[length*2+i] / (theRho*theRho);
        zeta_dzetadnup_dzetadndn[2][i] = -2.0*rho[length+i] / (theRho*theRho);
        DtauwDn = -theNormDrho*theNormDrho / (8*theRho*theRho);
        DtauwDDn = theNormDrho / (4*theRho);
        DdsDnup = 5.0/3.0 * (pow(1.0+theZeta, 2.0/3.0) - pow(1.0-theZeta, 2.0/3.0)) * zeta_dzetadnup_dzetadndn[1][i] / 2.0;
        DdsDndn = 5.0/3.0 * (pow(1.0+theZeta, 2.0/3.0) - pow(1.0-theZeta, 2.0/3.0)) * zeta_dzetadnup_dzetadndn[2][i] / 2.0;
        DtauUnifDnup = threeMPi2_2o3 / 2.0 * pow(theRho, 2.0/3.0) * ds + 3.0/10.0 * threeMPi2_2o3 * pow(theRho, 5.0/3.0) * DdsDnup;
        DtauUnifDndn = threeMPi2_2o3 / 2.0 * pow(theRho, 2.0/3.0) * ds + 3.0/10.0 * threeMPi2_2o3 * pow(theRho, 5.0/3.0) * DdsDndn;
        alpha_dadnup_dadndn_daddn_dadtau[1][i] = (-DtauwDn*(tauUnif + eta*tauw) - (theTau - tauw)*(DtauUnifDnup + eta*DtauwDn)) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/dn
        alpha_dadnup_dadndn_daddn_dadtau[2][i] = (-DtauwDn*(tauUnif + eta*tauw) - (theTau - tauw)*(DtauUnifDndn + eta*DtauwDn)) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/dn
        alpha_dadnup_dadndn_daddn_dadtau[3][i] = (-DtauwDDn*(tauUnif + eta*tauw) - (theTau - tauw)*eta*DtauwDDn) / ((tauUnif + eta*tauw)*(tauUnif + eta*tauw)); // d\alpha/d|\nabla n|
        alpha_dadnup_dadndn_daddn_dadtau[4][i] = 1.0 / (tauUnif + eta*tauw); // d\alpha/d\tau
        
    }
}

void Calculate_r2scanc_spin(int length, double *rho, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc, double *v2c, double *v3c) {
    // constants for epsilon_c^0
    double b1c = 0.0285764;
    double b2c = 0.0889;
    double b3c = 0.125541;
    double betaConst = 0.06672455060314922;
    double betaRsInf = betaConst*0.1/0.1778;
    double f0 = -0.9;
    // constants for epsilon_c LSDA1
    double AA0 = 0.0310907 ;
	double alpha10 = 0.21370 ;
	double beta10 = 7.5957 ;
	double beta20 = 3.5876 ;
	double beta30 = 1.6382 ;
	double beta40 = 0.49294 ;
    double AAac = 0.0168869 ;
	double alpha1ac = 0.11125 ;
	double beta1ac = 10.357 ;
	double beta2ac = 3.6231 ;
	double beta3ac = 0.88026 ;
	double beta4ac = 0.49671 ;
    double AA1 = 0.01554535 ;
	double alpha11 = 0.20548 ;
	double beta11 = 14.1189 ;
	double beta21 = 6.1977 ;
	double beta31 = 3.3662 ;
	double beta41 = 0.62517 ;
    // constants for H1 of epsilon_c^1
    double r = 0.031091;
    double eta = 0.001;
    double dp2 = 0.361;
    // constants for switching function f_c
    double c1c = 0.64;
    double c2c = 1.5;
    double dc = 0.7;
    int i;
    for (i = 0; i < length; i++) {
        double zeta, phi, dx, s, p, alpha, rs;
        zeta = zeta_dzetadnup_dzetadndn[0][i];
        phi = (pow(1.0 + zeta, 2.0/3.0) + pow(1.0 - zeta, 2.0/3.0)) / 2.0;
        dx = (pow(1.0 + zeta, 4.0/3.0) + pow(1.0 - zeta, 4.0/3.0)) / 2.0;
        s = s_dsdn_dsddn[0][i];
        p = p_dpdn_dpddn[0][i];
        alpha = alpha_dadnup_dadndn_daddn_dadtau[0][i];
        rs = pow(0.75/(M_PI*rho[i]), 1.0/3.0);
        // epsilon_c^0 (\alpha approach 0)
        double ecLDA0, cx0, Gc, w0, xiInf0, gInf0s, H0, ec0;
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
        double ecrs0_q0, ecrs0_q1, ecrs0_q1p, ecrs0_den, ecrs0_log, ecrs0, Decrs0_Drs;
        ecrs0_q0 = -2.0*AA0*(1.0 + alpha10*rs);
        ecrs0_q1 = 2.0*AA0*(beta10*sqrRs + beta20*rs + beta30*rs*sqrRs + beta40*rs*rs);
        ecrs0_q1p = AA0*(beta10*rsmHalf + 2.0*beta20 + 3.0*beta30*sqrRs + 4.0*beta40*rs);
        ecrs0_den = 1.0 / (ecrs0_q1*ecrs0_q1 + ecrs0_q1);
        ecrs0_log = -log(ecrs0_q1*ecrs0_q1*ecrs0_den);
        ecrs0 = ecrs0_q0*ecrs0_log;
        Decrs0_Drs = -2.0*AA0*alpha10*ecrs0_log - ecrs0_q0*ecrs0_q1p*ecrs0_den;
        double ac_q0, ac_q1, ac_q1p, ac_den, ac_log, ac, Dac_Drs;
        ac_q0 = -2.0*AAac*(1.0 + alpha1ac*rs);
        ac_q1 = 2.0*AAac*(beta1ac*sqrRs + beta2ac*rs + beta3ac*rs*sqrRs + beta4ac*rs*rs);
        ac_q1p = AAac*(beta1ac*rsmHalf + 2.0*beta2ac + 3.0*beta3ac*sqrRs + 4.0*beta4ac*rs);
        ac_den = 1.0 / (ac_q1*ac_q1 + ac_q1);
        ac_log = -log(ac_q1*ac_q1*ac_den);
        ac = ac_q0*ac_log;
        Dac_Drs = -2.0*AAac*alpha1ac*ac_log - ac_q0*ac_q1p*ac_den;
        double ecrs1_q0, ecrs1_q1, ecrs1_q1p, ecrs1_den, ecrs1_log, ecrs1, Decrs1_Drs;
        ecrs1_q0 = -2.0*AA1*(1.0 + alpha11*rs);
        ecrs1_q1 = 2.0*AA1*(beta11*sqrRs + beta21*rs + beta31*rs*sqrRs + beta41*rs*rs);
        ecrs1_q1p = AA1*(beta11*rsmHalf + 2.0*beta21 + 3.0*beta31*sqrRs + 4.0*beta41*rs);
        ecrs1_den = 1.0 / (ecrs1_q1*ecrs1_q1 + ecrs1_q1);
        ecrs1_log = -log(ecrs1_q1*ecrs1_q1*ecrs1_den);
        ecrs1 = ecrs1_q0*ecrs1_log;
        Decrs1_Drs = -2.0*AA1*alpha11*ecrs1_log - ecrs1_q0*ecrs1_q1p*ecrs1_den;
        double f_zeta, fp_zeta, zeta4, gcrs, ec_lsda1, declsda1_drs;
        f_zeta = (pow(1.0+zeta, 4.0/3.0) + pow(1.0-zeta, 4.0/3.0) - 2.0 ) / (pow(2.0, 4.0/3.0) - 2.0);
	    fp_zeta = (pow(1.0+zeta, 1.0/3.0) - pow(1.0-zeta, 1.0/3.0)) * 4.0/3.0 / (pow(2.0, 4.0/3.0) - 2.0);
	    zeta4 = pow(zeta, 4.0);
        gcrs = ecrs1 - ecrs0 + ac/1.709921;
	    ec_lsda1 = ecrs0 + f_zeta * (zeta4 * gcrs - ac/1.709921);
        declsda1_drs = Decrs0_Drs + f_zeta * (zeta4 * (Decrs1_Drs - Decrs0_Drs + Dac_Drs/1.709921) - Dac_Drs/1.709921);
        double rPhi3, w1, t, y, deltafc2, ds, ec_lsda0, declsda0_drs;
        rPhi3 = r*phi*phi*phi;
        w1 = exp(-ec_lsda1/rPhi3) - 1.0;
        t = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * s/(phi*sqrRs);
        y = beta / (r*w1) * t*t;
        deltafc2 = 1.0*(-0.64) + 2.0*(-0.4352) + 3.0*(-1.535685604549) + 4.0*3.061560252175 
            + 5.0*(-1.915710236206) + 6.0*0.516884468372 + 7.0*(-0.051848879792);
        ds = (pow(1.0 + zeta, 5.0/3.0) + pow(1.0 - zeta, 5.0/3.0)) / 2.0;
        ec_lsda0 = ecLDA0 * Gc;
        declsda0_drs = b1c*(0.5*b2c/sqrRs + b3c) / pow(1.0 + b2c*sqrRs + b3c*rs, 2.0) * Gc;
        double deltayPart1, deltayPart2, deltayPart3, deltay, g, H1, ec1;
        deltayPart1 = deltafc2 / (27.0*r*ds*pow(phi, 3.0)*w1);
        deltayPart2 = 20.0*rs*(declsda0_drs - declsda1_drs) - 45.0*eta*(ec_lsda0 - ec_lsda1);
        deltayPart3 = p*exp(-p*p / pow(dp2, 4.0));
        deltay = deltayPart1 * deltayPart2 * deltayPart3;

        g = pow(1.0 + 4.0*(y - deltay), -0.25);
        H1 = rPhi3 * log(1.0 + w1*(1.0 - g));
        ec1 = ec_lsda1 + H1;
        // interpolate and extrapolate epsilon_c
        double fc;
        if (alpha > 2.5) {
            fc = -dc*exp(c2c / (1.0 - alpha));
        }
        else if (alpha > 0.0) {
            double alpha2, alpha3, alpha4, alpha5, alpha6, alpha7;
            alpha2 = alpha *alpha; alpha3 = alpha2*alpha;
            alpha4 = alpha3*alpha; alpha5 = alpha4*alpha;
            alpha6 = alpha5*alpha; alpha7 = alpha6*alpha;
            fc = 1.0 + (-0.64)*alpha + (-0.4352)*alpha2 + (-1.535685604549)*alpha3 + 3.061560252175*alpha4
                + (-1.915710236206)*alpha5 + 0.516884468372*alpha6 + (-0.051848879792)*alpha7;
        }
        else {
            fc = exp(-c1c*alpha / (1.0 - alpha));
        }
        epsilonc[i] = ec1 + fc*(ec0 - ec1);
        // compute variation of epsilon_c^0
        double DzetaDnup, DzetaDndn, DrsDn, DdxDnup, DdxDndn, DGcDnup, DGcDndn, DgInf0sDs, DgInf0sDn, DgInf0sDDn, DecLDA0Dn, Dw0Dn, DH0Dn, DH0DDn, Dec0Dnup, Dec0Dndn, Dec0DDn;
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
        Dec0Dnup = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0 + H0)*DGcDnup;
        // if (i==4) {
        //     printf("point 4, DecLDA0Dn %10.8f, DH0Dn %10.8f, Gc %10.8f, ecLDA0 %10.8f, DGcDnup %10.8f\n", DecLDA0Dn, DH0Dn, Gc, ecLDA0, DGcDnup);
        // }
        Dec0Dndn = (DecLDA0Dn + DH0Dn)*Gc + (ecLDA0 + H0)*DGcDndn;
        Dec0DDn = DH0DDn*Gc;
        // compute variation of epsilon_c^1
        double DgcrsDrs, Dec_lsda1_Drs, DdsDnup, DdsDndn;
        DgcrsDrs = Decrs1_Drs - Decrs0_Drs + Dac_Drs/1.709921;
        Dec_lsda1_Drs = Decrs0_Drs + f_zeta * (zeta4 * DgcrsDrs - Dac_Drs/1.709921);
        DdsDnup = (5.0/3.0*pow(1.0 + zeta, 2.0/3.0) - 5.0/3.0*pow(1.0 - zeta, 2.0/3.0))/2.0*DzetaDnup; 
        DdsDndn = (5.0/3.0*pow(1.0 + zeta, 2.0/3.0) - 5.0/3.0*pow(1.0 - zeta, 2.0/3.0))/2.0*DzetaDndn; 
        double dec0log_drs, dec0q0_drs, dec0q1p_drs, dec0den_drs, d2ec0_drs2;
        dec0log_drs = -ecrs0_q1p * ecrs0_den;
        dec0q0_drs = -2.0 * AA0 * alpha10;
        dec0q1p_drs = AA0 * ((-0.5)*beta10/sqrRs/rs + 3.0*0.5*beta30/sqrRs + 4.0*beta40);
        dec0den_drs = -(2.0*ecrs0_q1*ecrs0_q1p + ecrs0_q1p) / pow(ecrs0_q1*ecrs0_q1 + ecrs0_q1, 2.0);
        d2ec0_drs2 = -2.0 * AA0 * alpha10 * dec0log_drs - (dec0q0_drs * ecrs0_q1p * ecrs0_den 
            + ecrs0_q0 * dec0q1p_drs * ecrs0_den + ecrs0_q0 * ecrs0_q1p * dec0den_drs);
        double daclog_drs, dacq0_drs, dacq1p_drs, dacden_drs, d2ac_drs2;
        daclog_drs = -ac_q1p * ac_den;
        dacq0_drs = -2.0 * AAac * alpha1ac;
        dacq1p_drs = AAac * ((-0.5)*beta1ac/sqrRs/rs + 3.0*0.5*beta3ac/sqrRs + 4.0*beta4ac);
        dacden_drs = -(2.0*ac_q1*ac_q1p + ac_q1p) / pow(ac_q1*ac_q1 + ac_q1, 2.0);
        d2ac_drs2 = -2.0 * AAac * alpha1ac * daclog_drs - (dacq0_drs * ac_q1p * ac_den 
            + ac_q0 * dacq1p_drs * ac_den + ac_q0 * ac_q1p * dacden_drs);
        double dec1log_drs, dec1q0_drs, dec1q1p_drs, dec1den_drs, d2ec1_drs2;
        dec1log_drs = -ecrs1_q1p * ecrs1_den;
        dec1q0_drs = -2.0 * AA1 * alpha11;
        dec1q1p_drs = AA1 * ((-0.5)*beta11/sqrRs/rs + 3.0*0.5*beta31/sqrRs + 4.0*beta41);
        dec1den_drs = -(2.0*ecrs1_q1*ecrs1_q1p + ecrs1_q1p) / pow(ecrs1_q1*ecrs1_q1 + ecrs1_q1, 2.0);
        d2ec1_drs2 = -2.0 * AA1 * alpha11 * dec1log_drs - (dec1q0_drs * ecrs1_q1p * ecrs1_den 
            + ecrs1_q0 * dec1q1p_drs * ecrs1_den + ecrs1_q0 * ecrs1_q1p * dec1den_drs);
        double d2eclsda1_drs2, d2eclsda1_drsdzeta, Ddeclsda1_drsDnup, Ddeclsda1_drsDndn;
        d2eclsda1_drs2 = d2ec0_drs2 + f_zeta * (zeta4 * (d2ec1_drs2 - d2ec0_drs2 + d2ac_drs2/1.709921) - d2ac_drs2/1.709921);
        d2eclsda1_drsdzeta = fp_zeta* (zeta4 * (Decrs1_Drs - Decrs0_Drs + Dac_Drs/1.709921) - Dac_Drs/1.709921)
            + f_zeta*(4.0*pow(zeta, 3.0) * (Decrs1_Drs - Decrs0_Drs + Dac_Drs/1.709921));
        Ddeclsda1_drsDnup = d2eclsda1_drs2*DrsDn + d2eclsda1_drsdzeta*DzetaDnup;
        Ddeclsda1_drsDndn = d2eclsda1_drs2*DrsDn + d2eclsda1_drsdzeta*DzetaDndn;
        double Dfzeta4_Dzeta, Dec_lsda1_Dzeta, Dec_lsda1Dnup, Dec_lsda1Dndn, DbetaDn, DphiDnup, DphiDndn, DtDnup, DtDndn, DtDDn, Dw1Dnup, Dw1Dndn;
        Dfzeta4_Dzeta = 4.0 * (zeta*zeta*zeta) * f_zeta + fp_zeta * zeta4;
        Dec_lsda1_Dzeta = Dfzeta4_Dzeta * gcrs - fp_zeta * ac/1.709921;
        Dec_lsda1Dnup = (- rs * 1.0/3.0 * Dec_lsda1_Drs - zeta * Dec_lsda1_Dzeta + Dec_lsda1_Dzeta) / rho[i];
        Dec_lsda1Dndn = (- rs * 1.0/3.0 * Dec_lsda1_Drs - zeta * Dec_lsda1_Dzeta - Dec_lsda1_Dzeta) / rho[i];
        DbetaDn = 0.066725*(0.1*(1.0 + 0.1778*rs) - 0.1778*(1.0 + 0.1*rs)) / (1.0 + 0.1778*rs) / (1.0 + 0.1778*rs) * DrsDn;
        DphiDnup = 0.5*(2.0/3.0*pow(1.0 + zeta, -1.0/3.0) - 2.0/3.0*pow(1.0 - zeta, -1.0/3.0)) * DzetaDnup;
        DphiDndn = 0.5*(2.0/3.0*pow(1.0 + zeta, -1.0/3.0) - 2.0/3.0*pow(1.0 - zeta, -1.0/3.0)) * DzetaDndn;
        DtDnup = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * (phi*sqrRs*s_dsdn_dsddn[1][i] - s*(DphiDnup*sqrRs + phi*DrsDn/(2.0*sqrRs))) / (phi*phi*rs);
        DtDndn = pow(3.0*M_PI*M_PI/16.0, 1.0/3.0) * (phi*sqrRs*s_dsdn_dsddn[1][i] - s*(DphiDndn*sqrRs + phi*DrsDn/(2.0*sqrRs))) / (phi*phi*rs);
        DtDDn = t*s_dsdn_dsddn[2][i]/s;
        Dw1Dnup = (w1 + 1.0) * (-(rPhi3*Dec_lsda1Dnup - r*ec_lsda1*(3.0*phi*phi*DphiDnup)) / rPhi3 / rPhi3);
        Dw1Dndn = (w1 + 1.0) * (-(rPhi3*Dec_lsda1Dndn - r*ec_lsda1*(3.0*phi*phi*DphiDndn)) / rPhi3 / rPhi3);
        double DyDnup, DyDndn, DyDDn, Declsda0Dnup, Declsda0Dndn, d2eclsda0_drs2, d2eclsda0_drsdGc, Ddeclsda0_drsDnup, Ddeclsda0_drsDndn;
        DyDnup = (w1*DbetaDn - beta*Dw1Dnup) / (r*w1*w1) * (t*t) + beta/(r*w1) * (2.0*t)*DtDnup; // new variable
        DyDndn = (w1*DbetaDn - beta*Dw1Dndn) / (r*w1*w1) * (t*t) + beta/(r*w1) * (2.0*t)*DtDndn;
        DyDDn = beta/(r*w1) * (2.0*t)*DtDDn; // new variable
        Declsda0Dnup = declsda0_drs*DrsDn + ecLDA0*DGcDnup;
        Declsda0Dndn = declsda0_drs*DrsDn + ecLDA0*DGcDndn;
        d2eclsda0_drs2 = b1c*((0.5*b2c*(-0.5)*pow(rs, -1.5))*pow(1.0 + b2c*sqrRs + b3c*rs, 2.0) 
            - (0.5*b2c/sqrRs + b3c)*2.0*(1.0 + b2c*sqrRs + b3c*rs)*(0.5*b2c/sqrRs + b3c)) 
            / pow(1.0 + b2c*sqrRs + b3c*rs, 4.0) * Gc;
        d2eclsda0_drsdGc = b1c*(0.5*b2c/sqrRs + b3c)/pow(1.0 + b2c*sqrRs + b3c*rs, 2.0);
        Ddeclsda0_drsDnup = d2eclsda0_drs2*DrsDn + d2eclsda0_drsdGc*DGcDnup;
        Ddeclsda0_drsDndn = d2eclsda0_drs2*DrsDn + d2eclsda0_drsdGc*DGcDndn;
        double d_deltayPart1_dnup, d_deltayPart1_dndn, d_deltayPart2_dnup, d_deltayPart2_dndn, d_deltayPart3_dp, DdeltayDnup, DdeltayDndn, DdeltayDDn;
        d_deltayPart1_dnup =  deltafc2 / (27.0*rPhi3*w1) * (-1) / (ds*ds) * DdsDnup
            + deltafc2 / (27.0*r*ds*w1) * (-3)/pow(phi, 4.0) * DphiDnup 
            + deltafc2 / (27.0*ds*rPhi3) * (-1)/(w1*w1) * Dw1Dnup;
        d_deltayPart1_dndn =  deltafc2 / (27.0*rPhi3*w1) * (-1) / (ds*ds) * DdsDndn 
            + deltafc2 / (27.0*r*ds*w1) * (-3)/pow(phi, 4.0) * DphiDndn 
            + deltafc2 / (27.0*ds*rPhi3) * (-1)/(w1*w1) * Dw1Dndn;
        d_deltayPart2_dnup = 20.0*(declsda0_drs - declsda1_drs)*DrsDn 
            + 20.0*rs*(Ddeclsda0_drsDnup - Ddeclsda1_drsDnup) 
            - 45.0*eta*(Declsda0Dnup - Dec_lsda1Dnup);
        d_deltayPart2_dndn = 20.0*(declsda0_drs - declsda1_drs)*DrsDn 
            + 20.0*rs*(Ddeclsda0_drsDndn - Ddeclsda1_drsDndn) 
            - 45.0*eta*(Declsda0Dndn - Dec_lsda1Dndn);
        d_deltayPart3_dp = exp(-p*p/pow(dp2, 4.0)) + p*exp(-p*p/pow(dp2, 4.0))*(-2*p/pow(dp2, 4.0));
        DdeltayDnup = d_deltayPart1_dnup*deltayPart2*deltayPart3 
            + deltayPart1*d_deltayPart2_dnup*deltayPart3 
            + deltayPart1*deltayPart2*d_deltayPart3_dp*p_dpdn_dpddn[1][i]; // new variable
        DdeltayDndn = d_deltayPart1_dndn*deltayPart2*deltayPart3 
            + deltayPart1*d_deltayPart2_dndn*deltayPart3 
            + deltayPart1*deltayPart2*d_deltayPart3_dp*p_dpdn_dpddn[1][i]; // new variable
        DdeltayDDn = deltayPart1*deltayPart2*d_deltayPart3_dp*p_dpdn_dpddn[2][i];
        double DgDnup, DgDndn, DgDDn, DH1Dnup, DH1Dndn, DH1DDn, Dec1Dnup, Dec1Dndn, Dec1DDn;
        DgDnup = -0.25*pow(1.0 + 4.0*(y - deltay), -1.25) * (4.0*(DyDnup - DdeltayDnup)); // new formula
        DgDndn = -0.25*pow(1.0 + 4.0*(y - deltay), -1.25) * (4.0*(DyDndn - DdeltayDndn));
        DgDDn = -0.25*pow(1.0 + 4.0*(y - deltay), -1.25) * (4.0*(DyDDn - DdeltayDDn));
        DH1Dnup = 3*r*phi*phi*DphiDnup*log(1.0 + w1*(1.0 - g)) + rPhi3*(Dw1Dnup*(1.0 - g) - w1*DgDnup) / (1.0 + w1*(1.0 - g));
        DH1Dndn = 3*r*phi*phi*DphiDndn*log(1.0 + w1*(1.0 - g)) + rPhi3*(Dw1Dndn*(1.0 - g) - w1*DgDndn) / (1.0 + w1*(1.0 - g));
        DH1DDn = rPhi3*(-w1*DgDDn) / (1.0 + w1*(1.0 - g));
        Dec1Dnup = Dec_lsda1Dnup + DH1Dnup;
        Dec1Dndn = Dec_lsda1Dndn + DH1Dndn;
        Dec1DDn = DH1DDn;
        double DfcDalpha, DfcDnup, DfcDndn, DfcDDn, DfcDtau, DepsiloncDnup, DepsiloncDndn, DepsiloncDDn, DepsiloncDtau;
        if (alpha > 2.5) {
            DfcDalpha = fc*(c2c/(1.0 - alpha)/(1.0 - alpha));
        }
        else if (alpha > 0.0) {
            double alpha2 = alpha *alpha; double alpha3 = alpha2*alpha;
            double alpha4 = alpha3*alpha; double alpha5 = alpha4*alpha;
            double alpha6 = alpha5*alpha;
            DfcDalpha = (-0.64) + (-0.4352)*alpha*2.0 + (-1.535685604549)*alpha2*3.0 + 3.061560252175*alpha3*4.0 
                + (-1.915710236206)*alpha4*5.0 + 0.516884468372*alpha5*6.0 + (-0.051848879792)*alpha6*7.0;
        }
        else {
            DfcDalpha = fc*(-c1c/(1.0 - alpha)/(1.0 - alpha));
        }
        DfcDnup = DfcDalpha*alpha_dadnup_dadndn_daddn_dadtau[1][i];
        DfcDndn = DfcDalpha*alpha_dadnup_dadndn_daddn_dadtau[2][i];
        DfcDDn = DfcDalpha*alpha_dadnup_dadndn_daddn_dadtau[3][i];
        DfcDtau = DfcDalpha*alpha_dadnup_dadndn_daddn_dadtau[4][i];
        DepsiloncDnup = Dec1Dnup + fc*(Dec0Dnup - Dec1Dnup) + DfcDnup*(ec0 - ec1);
        DepsiloncDndn = Dec1Dndn + fc*(Dec0Dndn - Dec1Dndn) + DfcDndn*(ec0 - ec1);
        DepsiloncDDn = Dec1DDn + fc*(Dec0DDn -Dec1DDn) + DfcDDn*(ec0 - ec1);
        DepsiloncDtau = DfcDtau*(ec0 - ec1);
        vc[i] = epsilonc[i] + rho[i]*DepsiloncDnup;
        vc[length+i] = epsilonc[i] + rho[i]*DepsiloncDndn;
        v2c[i] = rho[i]*DepsiloncDDn;
        v3c[i] = rho[i]*DepsiloncDtau;
    }
}