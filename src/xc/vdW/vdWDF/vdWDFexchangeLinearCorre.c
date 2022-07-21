/**
 * @file    vdWDFexchangeLinearCorre.c
 * @brief   Calculate the XC potential without nonlinear correlation part of vdF-DF1 and vdW-DF2 functional.
 * Zhang-Yang revised PBE exchange + LDA PW91 correlation for vdW-DF1
 * PW86 exchange + LDA PW91 correlation for vdW-DF2
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Dion, Max, Henrik Rydberg, Elsebeth Schröder, David C. Langreth, and Bengt I. Lundqvist. 
 * "Van der Waals density functional for general geometries." 
 * Physical review letters 92, no. 24 (2004): 246401.
 * Román-Pérez, Guillermo, and José M. Soler. 
 * "Efficient implementation of a van der Waals density functional: application to double-wall carbon nanotubes." 
 * Physical review letters 103, no. 9 (2009): 096102.
 * Lee, Kyuho, Éamonn D. Murray, Lingzhu Kong, Bengt I. Lundqvist, and David C. Langreth. 
 * "Higher-accuracy van der Waals density functional." Physical Review B 82, no. 8 (2010): 081101.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <errno.h> 
#include <time.h>

#include "tools.h"
#include "parallelization.h"
#include "vdWDFexchangeLinearCorre.h"
#include "gradVecRoutines.h"

/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/** ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif

/**
* @brief the function to compute the XC potential without nonlinear correlation part of vdF-DF1 and vdW-DF2 functional.
* called by Calculate_Vxc_GGA in exchangeCorrelation.c
*/
void Calculate_Vxc_vdWExchangeLinearCorre(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) { // this function doesn not have spin polarization
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    int DMnd, i;
    DMnd = pSPARC->Nd_d;
    double *vdWDFVx1 = (double*)malloc(DMnd * sizeof(double)); assert(vdWDFVx1 != NULL); // d(n\epsilon_x)/dn in dmcomm_phi
    double *vdWDFVx2 = (double*)malloc(DMnd * sizeof(double)); assert(vdWDFVx2 != NULL); // d(n\epsilon_x)/d|grad n| / |grad n| in dmcomm_phi
    double *vdWDFex = (double*)malloc(DMnd * sizeof(double)); assert(vdWDFex != NULL); // \epsilon_x
    double *vdWDFVcLinear = (double*)malloc(DMnd * sizeof(double)); assert(vdWDFVcLinear != NULL); // d(n\epsilon_cl)/dn in dmcomm_phi
    double *vdWDFecLinear = (double*)malloc(DMnd * sizeof(double)); assert(vdWDFecLinear != NULL); // \epsilon_cl

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    Drho_x = (double *) malloc(DMnd * sizeof(double)); assert(Drho_x != NULL);
    Drho_y = (double *) malloc(DMnd * sizeof(double)); assert(Drho_y != NULL);
    Drho_z = (double *) malloc(DMnd * sizeof(double)); assert(Drho_z != NULL);
    DDrho_x = (double *) malloc(DMnd * sizeof(double)); assert(DDrho_x != NULL);
    DDrho_y = (double *) malloc(DMnd * sizeof(double)); assert(DDrho_y != NULL);
    DDrho_z = (double *) malloc(DMnd * sizeof(double)); assert(DDrho_z != NULL);
    sigma = (double *) malloc(DMnd * sizeof(double)); assert(sigma != NULL);
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
            sigma[i] = Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i]; // |grad n|^2
        }
    }
    // exchange part
    if (pSPARC->vdWDFFlag == 1) // Zhang-Yang revPBE for vdWDF1
        Calculate_Vx_ZhangYangrevPBE(DMnd, xc_cst, rho, sigma, vdWDFVx1, vdWDFVx2, vdWDFex);
    else // GGA PW86 exchange for vdWDF2
        Calculate_Vx_PW86(DMnd, rho, sigma, vdWDFVx1, vdWDFVx2, vdWDFex);
    // correlation part LDA PW91
    Calculate_Vc_PW91(DMnd, rho, vdWDFVcLinear, vdWDFecLinear);

    for(i = 0; i < DMnd; i++){
        pSPARC->e_xc[i] = vdWDFex[i] + vdWDFecLinear[i]; // /epsilon of vdWDF without nonlinear correlation
        pSPARC->XCPotential[i] = vdWDFVx1[i] + vdWDFVcLinear[i]; // local part
        pSPARC->Dxcdgrho[i] = vdWDFVx2[i];
    }

    double temp1, temp2, temp3;
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
        pSPARC->XCPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i]; // gradient part
    }

    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);

    free(vdWDFVx1);
    free(vdWDFVx2);
    free(vdWDFex);
    free(vdWDFVcLinear);
    free(vdWDFecLinear);
}

/**
* @brief the function to compute the potential and energy density of Zhang-Yang revPBE GGA exchange
*/
void Calculate_Vx_ZhangYangrevPBE(int DMnd, XCCST_OBJ *xc_cst, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex) {
    // the variables below comes from Quantum Espresso
    double agrho, kf, dsg, s1, s2, f1, f2, f3, fx, exunif, ex, dxunif, ds, dfx1, dfx;
    
    xc_cst->kappa = 1.245; // Zhang-Yang revPBE
    xc_cst->mu_divkappa = xc_cst->mu/xc_cst->kappa;

    for(int i = 0; i < DMnd; i++){
        // the code below comes from Quantum Espresso
        agrho = sqrt(sigma[i]);
        kf = pow(3.0*M_PI*M_PI*rho[i], 1.0/3.0);
        dsg = 0.5 / kf;
        s1 = agrho * dsg / rho[i];
        s2 = s1 * s1;
        f1 = s2 * xc_cst->mu_divkappa;
        f2 = 1.0 + f1;
        f3 = xc_cst->kappa / f2;
        fx = 1.0 + xc_cst->kappa - f3; // here is different from QE, 1 added
        exunif = - 3.0/(4.0*M_PI) * kf;
        ex = exunif * fx;
        vdWDFex[i] = ex;
        dxunif = exunif * 1.0/3.0;
        ds = - 4.0/3.0 * s1;
        dfx1 = f2 * f2;
        dfx = 2.0 * xc_cst->mu * s1 / dfx1;
        vdWDFVx1[i] = ex + dxunif*fx + exunif*dfx*ds;
        vdWDFVx2[i] = exunif * dfx * dsg / agrho;
    }
}

/**
* @brief the function to compute the potential and energy density of PW86 GGA exchange
*/
void Calculate_Vx_PW86(int DMnd, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex) {
    double s, s_2, s_3, s_4, s_5, s_6, fs, grad_rho, df_ds;
    double a = 1.851;
    double b = 17.33;
    double c = 0.163;
    double s_prefactor = 6.18733545256027; // 2*(3\pi^2)^(1/3)
    double Ax = -0.738558766382022; // -3/4 * (3/pi)^(1/3)
    double four_thirds = 4.0/3.0;
    
    for (int i = 0; i < DMnd; i++) {
        grad_rho = sqrt(sigma[i]);
        s = grad_rho / (s_prefactor*pow(rho[i], four_thirds));
        s_2 = s*s;
        s_3 = s_2*s;
        s_4 = s_3*s;
        s_5 = s_3*s_2;
        s_6 = s_5*s;

        fs = pow(1.0 + a*s_2 + b*s_4 + c*s_6, 1.0/15.0);
        vdWDFex[i] = Ax * pow(rho[i], 1.0/3.0) * fs; // \epsilon_x, not n\epsilon_x
        df_ds = (1.0/(15.0*pow(fs, 14.0))) * (2.0*a*s + 4.0*b*s_3 + 6.0*c*s_5);
        vdWDFVx1[i] = Ax*four_thirds * (pow(rho[i], 1.0/3.0)*fs - grad_rho/(s_prefactor*rho[i])*df_ds);
        vdWDFVx2[i] = Ax * df_ds/(s_prefactor*grad_rho);
    }
}

/**
* @brief the function to compute the potential and energy density of PW91 LDA correlation
*/
void Calculate_Vc_PW91(int DMnd, double *rho, double *vdWDFVcLinear, double *vdWDFecLinear) {
    int i;
    double p, A, alpha1, beta1, beta2, beta3, beta4, C3, C31; // parameters (constants)
    double rs, G1, G2, rho_cbrt, rs_sqrt, rs_pow_1p5, rs_pow_p, rs_pow_pplus1;

    C31 = 0.6203504908993999; // (3/4pi)^(1/3)
    // correlation parameters
    p = 1.0; 
    A = 0.031091; // GGA 0.0310907
    alpha1 = 0.21370;
    beta1 = 7.5957;
    beta2 = 3.5876;
    beta3 = 1.6382;
    beta4 = 0.49294;

    for (i = 0; i < DMnd; i++) {
        rho_cbrt = cbrt(rho[i]);
        rs = C31 / rho_cbrt; // rs = (3/(4*pi*rho))^(1/3)
        rs_sqrt = sqrt(rs); // rs^0.5
        rs_pow_1p5 = rs * rs_sqrt; // rs^1.5
        rs_pow_p = rs; // rs^p, where p = 1, pow function is slow (~100x slower than add/sub)
        rs_pow_pplus1 = rs_pow_p * rs; // rs^(p+1)
        G1 = log(1.0+1.0/(2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1 )));
        G2 = 2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1);

        // ec = -2.0 * A * (1 + alpha1 * rs) * log(1.0 + 1.0 / (2.0 * A * (beta1 * rs_sqrt + beta2 * rs + beta3 * rs_pow_1p5 + beta4 * rs_pow_pplus1)));
        vdWDFecLinear[i] = -2.0 * A * (1 + alpha1 * rs) * G1;
        
        vdWDFVcLinear[i] = -2.0*A*(1.0+alpha1*rs) * G1 
	        -(rs/3.0)*( -2.0*A*alpha1 * G1 + (2.0*A*(1.0+alpha1*rs) * (A*(beta1/rs_sqrt + 2.0*beta2 + 3.0*beta3*rs_sqrt + 2.0*(p+1.0)*beta4*rs_pow_p))) / (G2 * (G2 + 1.0)) );
    }
}

/**
* @brief the function to compute the XC energy o vdF-DF1 and vdW-DF2 functional without nonlinear correlation part.
*/
void Calculate_Exc_GGA_vdWDF_ExchangeLinearCorre(SPARC_OBJ *pSPARC, double * electronDens) {
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