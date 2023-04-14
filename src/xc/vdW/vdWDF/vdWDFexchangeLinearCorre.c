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
 * Thonhauser, T., S. Zuluaga, C. A. Arter, K. Berland, E. Schröder, and P. Hyldgaard. 
 * "Spin signature of nonlocal correlation binding in metal-organic frameworks." 
 * Physical review letters 115, no. 13 (2015): 136402.
 * vdW-DF feature developed in Quantum Espresso:
 * Thonhauser, Timo, Valentino R. Cooper, Shen Li, Aaron Puzder, Per Hyldgaard, and David C. Langreth. 
 * "Van der Waals density functional: Self-consistent potential and the nature of the van der Waals bond." 
 * Physical Review B 76, no. 12 (2007): 125112.
 * Sabatini, Riccardo, Emine Küçükbenli, Brian Kolb, Timo Thonhauser, and Stefano De Gironcoli. 
 * "Structural evolution of amino acid crystals under stress from a non-empirical density functional." 
 * Journal of Physics: Condensed Matter 24, no. 42 (2012): 424209.
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
void Calculate_Vxc_vdWExchangeLinearCorre(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho) { 
    // this function have spin polarization
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    int DMnd, i;
    DMnd = pSPARC->Nd_d;
    // if (pSPARC->countPotentialCalculate == 3) { // jsut for debugging. should be commented after that
    //     char folderRoute[L_STRING];
    //     find_folder_route(pSPARC, folderRoute);
    //     // char *kernelFileRoute = "kernel_d2.txt"; // name of the file of the output kernels and 2nd derivative of kernels
    //     char kernelFileRoute[L_STRING]; // name of the file of the output kernels and 2nd derivative of kernels
    //     snprintf(kernelFileRoute,       L_STRING, "%srhoUp.txt"  ,     folderRoute);
    //     print_variables(rho + DMnd, kernelFileRoute, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    //     snprintf(kernelFileRoute,       L_STRING, "%srhoDn.txt"  ,     folderRoute);
    //     print_variables(rho + 2*DMnd, kernelFileRoute, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz);
    // }
    double *vdWDFVx1 = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(vdWDFVx1 != NULL); // d(n\epsilon_x)/dn in dmcomm_phi
    // It has two Nd_d: d(E_x)/d(n_up) and d(E_x)/d(n_dn)
    double *vdWDFVx2 = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(vdWDFVx2 != NULL); // d(n\epsilon_x)/d|grad n| / |grad n| in dmcomm_phi
    // It has only 2*pSPARC->Nd_d: 1st and 2nd Nd_d are for d(E_xup)/d(|grad nup|) and d(E_xdn)/d(|grad ndn|)
    // but for pSPARC->Dxcdgrho. It must has 3*pSPARC->Nd_d: 1st Nd_d is for d(E_c)/d(|grad n|); in vdW it should be zero. 2nd and 3rd Nd_d are for d(E_xup)/d(|grad nup|) and d(E_xdn)/d(|grad ndn|)
    double *vdWDFex = (double*)malloc(DMnd * sizeof(double)); assert(vdWDFex != NULL); // \epsilon_x, 
    // it should have only one Nd_d. The two length Nd_d vectors will be generated in sub exchange function
    double *vdWDFVcLinear = pSPARC->vdWDFVcLinear;
    // It has two Nd_d: d(E_c)/d(n_up) and d(E_c)/d(n_dn)
    double *vdWDFecLinear = pSPARC->vdWDFecLinear;
    // it has only one Nd_d

    double *Drho_x, *Drho_y, *Drho_z, *DDrho_x, *DDrho_y, *DDrho_z, *sigma, *lapcT;
    Drho_x = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(Drho_x != NULL); // in vdWDF we only need the grad of n_up and n_dn. 
    Drho_y = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(Drho_y != NULL);
    Drho_z = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(Drho_z != NULL);
    DDrho_x = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(DDrho_x != NULL);
    DDrho_y = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(DDrho_y != NULL);
    DDrho_z = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(DDrho_z != NULL);
    sigma = (double *) malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(sigma != NULL);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspin, 0.0, rho + (pSPARC->Nspin - 1) * DMnd, pSPARC->Drho[0], 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspin, 0.0, rho + (pSPARC->Nspin - 1) * DMnd, pSPARC->Drho[1], 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspin, 0.0, rho + (pSPARC->Nspin - 1) * DMnd, pSPARC->Drho[2], 2, pSPARC->dmcomm_phi);
    memcpy(Drho_x, pSPARC->Drho[0], DMnd * pSPARC->Nspin * sizeof(double));
    memcpy(Drho_y, pSPARC->Drho[1], DMnd * pSPARC->Nspin * sizeof(double));
    memcpy(Drho_z, pSPARC->Drho[2], DMnd * pSPARC->Nspin * sizeof(double));
    
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        lapcT = (double *) malloc(6 * sizeof(double));
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(i = 0; i < DMnd * pSPARC->Nspin; i++){
            sigma[i] = Drho_x[i] * (lapcT[0] * Drho_x[i] + lapcT[1] * Drho_y[i]) + Drho_y[i] * (lapcT[3] * Drho_y[i] + lapcT[4] * Drho_z[i]) +
                       Drho_z[i] * (lapcT[5] * Drho_z[i] + lapcT[2] * Drho_x[i]); 
        }
        free(lapcT);
    } else {
        for(i = 0; i < DMnd * pSPARC->Nspin; i++){
            sigma[i] = Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i]; // |grad n|^2
        }
    }

    if (pSPARC->spin_typ == 0) { // spin-unpolarized case
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
    }
    else if (pSPARC->spin_typ == 1) { // spin-polarized case. We will compose new version of these functions for preventing messy structure
        // exchange part
        if (pSPARC->vdWDFFlag == 1) // Zhang-Yang revPBE for vdWDF1
            Calculate_Vx_SZhangYangrevPBE(DMnd, xc_cst, rho, sigma, vdWDFVx1, vdWDFVx2, vdWDFex);
        else // GGA PW86 exchange for vdWDF2
            Calculate_Vx_SPW86(DMnd, rho, sigma, vdWDFVx1, vdWDFVx2, vdWDFex);
        // correlation part LDA PW91
        Calculate_Vc_SPW91(DMnd, xc_cst, rho, vdWDFVcLinear, vdWDFecLinear);

        for(i = 0; i < DMnd; i++){
            pSPARC->e_xc[i] = vdWDFex[i] + vdWDFecLinear[i]; // /epsilon of vdWDF without nonlinear correlation
            pSPARC->XCPotential[i] = vdWDFVx1[i] + vdWDFVcLinear[i]; // local part
            pSPARC->XCPotential[DMnd + i] = vdWDFVx1[DMnd + i] + vdWDFVcLinear[DMnd + i]; // local part
            pSPARC->Dxcdgrho[i] = 0.0;
            pSPARC->Dxcdgrho[DMnd + i] = vdWDFVx2[i];
            pSPARC->Dxcdgrho[2*DMnd + i] = vdWDFVx2[DMnd + i];
        }
    }
    else if (pSPARC->spin_typ == 2)
        assert(pSPARC->Nspin <= 2 && pSPARC->spin_typ <= 1);

    double temp1, temp2, temp3;
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(i = 0; i < DMnd * pSPARC->Nspin; i++){
            temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * vdWDFVx2[i];
            temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * vdWDFVx2[i];
            temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * vdWDFVx2[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
        for(i = 0; i < DMnd * pSPARC->Nspin; i++){
            Drho_x[i] *= vdWDFVx2[i];
            Drho_y[i] *= vdWDFVx2[i];
            Drho_z[i] *= vdWDFVx2[i];
        }
    }

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspin, 0.0, Drho_x, DDrho_x, 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspin, 0.0, Drho_y, DDrho_y, 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, pSPARC->Nspin, 0.0, Drho_z, DDrho_z, 2, pSPARC->dmcomm_phi);
    
    for(i = 0; i < DMnd * pSPARC->Nspin; i++){
        //if(pSPARC->electronDens[i] != 0.0)
        pSPARC->XCPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i]; // gradient part
    }

    free(Drho_x); free(Drho_y); free(Drho_z);
    free(DDrho_x); free(DDrho_y); free(DDrho_z);
    free(sigma);

    free(vdWDFVx1);
    free(vdWDFVx2);
    free(vdWDFex);
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
        if (sigma[i] < 1E-14) sigma[i] = 1E-14;
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
* @brief the function to compute the potential and energy density of Zhang-Yang revPBE GGA exchange, spin-polarized case
*/
void Calculate_Vx_SZhangYangrevPBE(int DMnd, XCCST_OBJ *xc_cst, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex) {
    int i;

    // the variables below comes from Quantum Espresso
    double arho, agrho, kf, dsg, s1, s2, f1, f2, f3, fx, exunif, ex, dxunif, ds, dfx1, dfx;
    double *exUpDn = (double*)malloc(sizeof(double)*DMnd*2); // row 1: rho_up*epsilon(2*rho_up); row 2: rho_dn*epsilon(2*rho_dn);
    
    xc_cst->kappa = 1.245; // Zhang-Yang revPBE
    xc_cst->mu_divkappa = xc_cst->mu/xc_cst->kappa;

    for(i = 0; i < 2*DMnd; i++){
        // the code below comes from Quantum Espresso
        if (sigma[i] < 1E-14) sigma[i] = 1E-14;
        arho = 2.0*rho[DMnd + i];
        agrho = 2.0*sqrt(sigma[i]);  // arho and agrho multiplied by 2 to compute epsilon(2*rho_up dn)
        kf = pow(3.0*M_PI*M_PI*arho, 1.0/3.0);
        dsg = 0.5 / kf;
        s1 = agrho * dsg / arho;
        s2 = s1 * s1;
        f1 = s2 * xc_cst->mu_divkappa;
        f2 = 1.0 + f1;
        f3 = xc_cst->kappa / f2;
        fx = 1.0 + xc_cst->kappa - f3; // here is different from QE, 1 added
        exunif = - 3.0/(4.0*M_PI) * kf;
        ex = exunif * fx;
        exUpDn[i] = rho[DMnd + i]*ex;
        dxunif = exunif * 1.0/3.0;
        ds = - 4.0/3.0 * s1;
        dfx1 = f2 * f2;
        dfx = 2.0 * xc_cst->mu * s1 / dfx1;
        vdWDFVx1[i] = ex + dxunif*fx + exunif*dfx*ds; // they do not need to modify from unpolarized case: E_x[n_up, n_dn] = (E_x[2n_up] + E_x[2n_dn])/2 = (2n_up\epsilon_x[2n_up] + 2n_dn\epsilon_x[2n_dn])/2.
        vdWDFVx2[i] = exunif * dfx * dsg / (agrho / 2.0); // dE_x/d n_up = d(E_x[2n_up]/2)/d (2n_up) * d (2n_up)/d n_up = d(E_x[2n_up])/d (2n_up) /2 * 2 = d(E_x[2n_up])/d (2n_up)
    }
    for (i = 0; i < DMnd; i++) {
        vdWDFex[i] = (exUpDn[i] + exUpDn[DMnd + i]) / rho[i]; // (2.0*rho[DMnd + i]*exUpDn[i] + 2.0*rho[2*DMnd + i]*exUpDn[DMnd + i]) / 2.0 / rho[i]
    }
    free(exUpDn);
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
        if (sigma[i] < 1E-14) sigma[i] = 1E-14;
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
* @brief the function to compute the potential and energy density of PW86 GGA exchange, spin-polarized case
*/
void Calculate_Vx_SPW86(int DMnd, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex) {
    double s, s_2, s_3, s_4, s_5, s_6, fs, grad_rho, df_ds;
    double *exUpDn = (double*)malloc(sizeof(double)*DMnd*2); // row 1: rho_up*epsilon(2*rho_up); row 2: rho_dn*epsilon(2*rho_dn);
    double a = 1.851;
    double b = 17.33;
    double c = 0.163;
    double s_prefactor = 6.18733545256027; // 2*(3\pi^2)^(1/3)
    double Ax = -0.738558766382022; // -3/4 * (3/pi)^(1/3)
    double four_thirds = 4.0/3.0;
    int i;
    double two_pow13 = pow(2.0, 1.0/3.0);
    
    for (i = 0; i < 2*DMnd; i++) {
        if (sigma[i] < 1E-14) sigma[i] = 1E-14;
        grad_rho = sqrt(sigma[i]);
        s = grad_rho / (two_pow13 * s_prefactor*pow(rho[DMnd + i], four_thirds));
        s_2 = s*s;
        s_3 = s_2*s;
        s_4 = s_3*s;
        s_5 = s_3*s_2;
        s_6 = s_5*s;

        fs = pow(1.0 + a*s_2 + b*s_4 + c*s_6, 1.0/15.0);
        exUpDn[i] = Ax * two_pow13 * pow(rho[DMnd + i], 1.0/3.0 + 1.0) * fs; // \epsilon_x, not n\epsilon_x
        df_ds = (1.0/(15.0*pow(fs, 14.0))) * (2.0*a*s + 4.0*b*s_3 + 6.0*c*s_5);
        vdWDFVx1[i] = Ax*four_thirds * (two_pow13 * pow(rho[DMnd + i], 1.0/3.0)*fs - grad_rho/(s_prefactor*rho[DMnd + i])*df_ds);
        vdWDFVx2[i] = Ax * df_ds/(s_prefactor*grad_rho);
    }
    for (i = 0; i < DMnd; i++) {
        vdWDFex[i] = (exUpDn[i] + exUpDn[DMnd + i]) / rho[i];
    }
    free(exUpDn);
}

/**
* @brief the function to compute the potential and energy density of PW91 LDA correlation
*/
void Calculate_Vc_PW91(int DMnd, double *rho, double *vdWDFVcLinear, double *vdWDFecLinear) {
    int i;
    double p, A, alpha1, beta1, beta2, beta3, beta4, C31; // parameters (constants)
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
* @brief the function to compute the potential and energy density of PW91 LDA correlation, spin-polarized case
*/
void Calculate_Vc_SPW91(int DMnd, XCCST_OBJ *xc_cst, double *rho, double *vdWDFVcLinear, double *vdWDFecLinear) {
    int i;
    double rhom1_3, rhotot_inv, zeta, zetp, zetm, zetpm1_3, zetmm1_3, rhotmo6, rhoto6;
    double rs, sqr_rs, rsm1_2;
    double ec0_q0, ec0_q1, ec0_q1p, ec0_den, ec0_log, ecrs0, decrs0_drs, mac_q0, mac_q1, mac_q1p, mac_den, mac_log, macrs, dmacrs_drs;
    double ec1_q0, ec1_q1, ec1_q1p, ec1_den, ec1_log, ecrs1, decrs1_drs, zetp_1_3, zetm_1_3, f_zeta, fp_zeta, zeta4;
    double gcrs, ecrs, dgcrs_drs, decrs_drs, dfzeta4_dzeta, decrs_dzeta, vxcadd;
    for (i = 0; i < DMnd; i++) {
        rhom1_3 = pow(rho[i],-xc_cst->third);
        rhotot_inv = pow(rhom1_3,3.0);
        zeta = (rho[DMnd+i] - rho[2*DMnd+i]) * rhotot_inv;
        zetp = 1.0 + zeta * xc_cst->alpha_zeta;
        zetm = 1.0 - zeta * xc_cst->alpha_zeta;
        zetpm1_3 = pow(zetp,-xc_cst->third);
        zetmm1_3 = pow(zetm,-xc_cst->third);
        rhotmo6 = sqrt(rhom1_3);
        rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;
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

        vdWDFecLinear[i] = ecrs;
        vxcadd = ecrs - rs * xc_cst->third * decrs_drs - zeta * decrs_dzeta;
        vdWDFVcLinear[i] = vxcadd + decrs_dzeta;
        vdWDFVcLinear[DMnd+i] = vxcadd - decrs_dzeta;
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

/**
 * @brief the function to compute the XC energy o vdF-DF1 and vdW-DF2 functional without nonlinear correlation part, spin-polarized case. It is similar to Calculate_Exc_GSGA_PBE
 * it is also similar to Calculate_Exc_GGA_PBE. 
 */
void Calculate_Exc_GSGA_vdWDF_ExchangeLinearCorre(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int i;
    double Exc = 0.0;
    
    for (i = 0; i < pSPARC->Nd_d; i++) {
        //if(electronDens[i] != 0)
        Exc += electronDens[i] * pSPARC->e_xc[i]; 
        #ifdef DEBUG
        if (isnan(electronDens[i])) // for debugging
            printf("i = %d, electronDens is NaN\n", i);
        #endif
    }
    
    Exc *= pSPARC->dV;
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
}

void find_folder_route(SPARC_OBJ *pSPARC, char *folderRoute) { // used for returning the folder route of the input files
    strncpy(folderRoute, pSPARC->filename, L_STRING);
    int indexEndFolderRoute = 0;
    while (folderRoute[indexEndFolderRoute] != '\0') {
        indexEndFolderRoute++;
    } // get the length of the char filename
    while ((indexEndFolderRoute > -1) && (folderRoute[indexEndFolderRoute] != '/')) {
        indexEndFolderRoute--;
    } // find the last '/'. If there is no '/', indexEndFolderRoute should be at the beginning
    folderRoute[indexEndFolderRoute + 1] = '\0'; // cut the string. Now it contains only the folder position
}

void print_variables(double *variable, char *outputFileName, int Nx, int Ny, int Nz) {
    printf("begin printing variable\n");
    FILE *outputFile = NULL;
    outputFile = fopen(outputFileName,"w");
    int xIndex, yIndex, zIndex;
    int globalIndex = 0;
    fprintf(outputFile, "%d %d %d\n", Nx, Ny, Nz);
    for (zIndex = 0; zIndex < Nz; zIndex++) {
        for (yIndex = 0; yIndex < Ny; yIndex++) {
            fprintf(outputFile, "%d %d\n", yIndex, zIndex);
            for (xIndex = 0; xIndex < Nx; xIndex++) {
                if ((xIndex + 1) % 4 == 0 || (xIndex == Nx - 1)) // new line every 4 values
                    fprintf(outputFile, "%12.9f\n", variable[globalIndex]);
                else
                    fprintf(outputFile, "%12.9f ", variable[globalIndex]);
                globalIndex++;
            }
        }
    }
    fclose(outputFile);
}