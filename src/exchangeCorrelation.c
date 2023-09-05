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
#include <assert.h>

#include "exchangeCorrelation.h"
#include "initialization.h"
#include "isddft.h"
#include "gradVecRoutines.h"
#include "tools.h"
#include "vdWDFnonlinearCorre.h"
#include "mGGAtauTransferTauVxc.h"
#include "mGGAscan.h"
#include "mGGArscan.h"
#include "mGGAr2scan.h"
#include "cyclix_gradVec.h"

#define max(x,y) ((x)>(y)?(x):(y))

/**
* @brief  Calculate exchange correlation potential
**/
void Calculate_Vxc(SPARC_OBJ *pSPARC)
{   
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    
    int ncol = pSPARC->Nspdentd; 
    int DMnd = pSPARC->Nd_d;
    int sz = DMnd * ncol;
    double *rho = (double *)malloc(sz * sizeof(double) );
    double *sigma, *Drho_x, *Drho_y, *Drho_z, *tau;
    sigma = Drho_x = Drho_y = Drho_z = tau = NULL;

    // add core electron density if needed
    add_rho_core(pSPARC, pSPARC->electronDens, rho, ncol);
    // calculate sigma
    if (pSPARC->isgradient) {
        sigma  = (double *)malloc(sz * sizeof(double) );
        Drho_x = (double *)malloc(sz * sizeof(double) );
        Drho_y = (double *)malloc(sz * sizeof(double) );
        Drho_z = (double *)malloc(sz * sizeof(double) );
        calculate_square_norm_of_gradient(pSPARC, rho, pSPARC->mag, DMnd, ncol, sigma, Drho_x, Drho_y, Drho_z);

        if (pSPARC->ixc[3]) { // used for vdW-DF
            memcpy(pSPARC->Drho[0], Drho_x + (pSPARC->Nspin - 1)*DMnd, DMnd*pSPARC->Nspin*sizeof(double));
            memcpy(pSPARC->Drho[1], Drho_y + (pSPARC->Nspin - 1)*DMnd, DMnd*pSPARC->Nspin*sizeof(double));
            memcpy(pSPARC->Drho[2], Drho_z + (pSPARC->Nspin - 1)*DMnd, DMnd*pSPARC->Nspin*sizeof(double));
        }
    }

    if (pSPARC->ixc[2]) { // metaGGA
        if (pSPARC->countPotentialCalculate == 0) { // 1st SCF, calculate PBE
            pSPARC->ixc[0] = 2; pSPARC->ixc[1] = 3; 
            pSPARC->ixc[2] = 1; pSPARC->ixc[3] = 0;
            pSPARC->xcoption[0] = 1; pSPARC->xcoption[1] = 1;
        } 
        else {
            tau = pSPARC->KineticTauPhiDomain;
        }
    }

    if (pSPARC->spin_typ == 0) {
        double *ex = (double *)malloc(DMnd * sizeof(double) );
        double *ec = (double *)malloc(DMnd * sizeof(double) );
        double *vx = (double *)malloc(DMnd * sizeof(double) );
        double *vc = (double *)malloc(DMnd * sizeof(double) );
        double *v2x, *v2c, *v3x, *v3c;
        v2x = v2c = NULL;
        if (pSPARC->isgradient) {
            v2x = (double *)malloc(DMnd * sizeof(double) );
            v2c = (double *)malloc(DMnd * sizeof(double) );
        }
        v3x = v3c = NULL;
        if (pSPARC->ixc[2]) { // for metaGGA, d(n\epsilon)/d\tau
            v3x = (double *)malloc(DMnd * sizeof(double) );
            v3c = (double *)malloc(DMnd * sizeof(double) );
        }

        // iexch
        switch (pSPARC->ixc[0])
        {
        case 1:
            slater(DMnd, rho, ex ,vx);
            break;
        case 2:
            pbex(DMnd, rho, sigma, pSPARC->xcoption[0], ex, vx, v2x);
            break;
        case 3:
            rPW86x(DMnd, rho, sigma, ex, vx, v2x);
            break;
        case 4:
            scanx(DMnd, rho, sigma, tau, ex, vx, v2x, v3x);
            break;
        case 5:
            rscanx(DMnd, rho, sigma, tau, ex, vx, v2x, v3x);
            break;
        case 6:
            r2scanx(DMnd, rho, sigma, tau, ex, vx, v2x, v3x);
            break;
        default:
            memset(ex, 0, sizeof(double) * DMnd);
            memset(vx, 0, sizeof(double) * DMnd);
            if (pSPARC->isgradient) {
                memset(v2x, 0, sizeof(double) * DMnd);
            }
            break;
        }

        // icorr
        switch (pSPARC->ixc[1])
        {
        case 1:
            pz(DMnd, rho, ec, vc);
            break;
        case 2:
            pw(DMnd, rho, ec, vc);
            if (pSPARC->ixc[3]) {
                memcpy(pSPARC->vdWDFecLinear, ec, DMnd*sizeof(double));
                memcpy(pSPARC->vdWDFVcLinear, vc, DMnd*sizeof(double));
                memset(v2c, 0, sizeof(double) * DMnd);
            }
            break;
        case 3:
            pbec(DMnd, rho, sigma, pSPARC->xcoption[1], ec, vc, v2c);
            break;
        case 4:
            scanc(DMnd, rho, sigma, tau, ec, vc, v2c, v3c);
            break;
        case 5:
            rscanc(DMnd, rho, sigma, tau, ec, vc, v2c, v3c);
            break;
        case 6:
            r2scanc(DMnd, rho, sigma, tau, ec, vc, v2c, v3c);
            break;
        default:
            memset(ec, 0, sizeof(double) * DMnd);
            memset(vc, 0, sizeof(double) * DMnd);
            if (pSPARC->isgradient) {
                memset(v2c, 0, sizeof(double) * DMnd);
            }
            break;
        }

        if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"PBE0") == 0) {
            for (int i = 0; i < DMnd; i++) {
                ex[i] *= (1-pSPARC->exx_frac);
                vx[i] *= (1-pSPARC->exx_frac);
                v2x[i] *= (1-pSPARC->exx_frac);
            }
        }

        if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"HSE") == 0) {
            for (int i = 0; i < DMnd; i++) {
                double e_xc_sr, Dxcdgrho_sr, XCPotential_sr;
                // Use the same strategy as \rho for \grho here. 
                // Without this threshold, numerical issue will make simulation fail. 
                if (sigma[i] < 1E-14) sigma[i] = 1E-14;
                pbexsr(rho[i], sigma[i], pSPARC->hyb_range_pbe, &e_xc_sr, &XCPotential_sr, &Dxcdgrho_sr);
                ex[i] -=  pSPARC->exx_frac * e_xc_sr / rho[i];
                vx[i] -= pSPARC->exx_frac * XCPotential_sr;
                v2x[i] -= pSPARC->exx_frac * Dxcdgrho_sr;
            }
        }
        
        for (int i = 0; i < DMnd; i++) {
            pSPARC->e_xc[i] = ex[i] + ec[i];
            pSPARC->XCPotential[i] = vx[i] + vc[i];
            if (pSPARC->isgradient) {
                pSPARC->Dxcdgrho[i] = v2x[i] + v2c[i];
            }
            if (pSPARC->ixc[2] && (pSPARC->countPotentialCalculate)) {
                pSPARC->vxcMGGA3[i] = v3x[i] + v3c[i];
            }
        }

        if (pSPARC->ixc[3] != 0)
            Calculate_nonLinearCorr_E_V_vdWDF(pSPARC, rho);

        if (pSPARC->isgradient) {
            double *DDrho_x = (double *) malloc(DMnd * sizeof(double));
            double *DDrho_y = (double *) malloc(DMnd * sizeof(double));
            double *DDrho_z = (double *) malloc(DMnd * sizeof(double));

            Drho_times_v2xc(pSPARC, DMnd, 1, Drho_x, Drho_y, Drho_z, pSPARC->Dxcdgrho);
            if (pSPARC->CyclixFlag) {
                Gradient_vectors_dir_with_rotfac(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, Drho_y, DMnd, DDrho_x, DMnd, 0, pSPARC->dmcomm_phi);
                Gradient_vectors_dir_with_rotfac(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, Drho_x, DMnd, DDrho_y, DMnd, 1, pSPARC->dmcomm_phi);
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DMnd, DDrho_z, DMnd, 2, pSPARC->dmcomm_phi);
            } else {
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_x, DMnd, DDrho_x, DMnd, 0, pSPARC->dmcomm_phi);
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_y, DMnd, DDrho_y, DMnd, 1, pSPARC->dmcomm_phi);
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, Drho_z, DMnd, DDrho_z, DMnd, 2, pSPARC->dmcomm_phi);
            }

            for(int i = 0; i < DMnd; i++){
                pSPARC->XCPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i];
            }
            free(DDrho_x);
            free(DDrho_y);
            free(DDrho_z);
        }

        free(ex);
        free(ec);
        free(vx);
        free(vc);
        if (pSPARC->isgradient) {
            free(v2x);
            free(v2c);
        }
        if (pSPARC->ixc[2]) {
            free(v3x);
            free(v3c);
        }
    } else {
        double *ex = (double *)malloc(DMnd * sizeof(double) );
        double *ec = (double *)malloc(DMnd * sizeof(double) );
        double *vx = (double *)malloc(DMnd*2 * sizeof(double) );
        double *vc = (double *)malloc(DMnd*2 * sizeof(double) );
        double *v2x, *v2c, *v3x, *v3c;
        v2x = v2c = v3x = v3c = NULL;
        if (pSPARC->isgradient) {
            v2x = (double *)malloc(DMnd*2 * sizeof(double) );
            v2c = (double *)malloc(DMnd * sizeof(double) );
        }
        if (pSPARC->ixc[2]) { // for metaGGA, d(n\epsilon)/d\tau
            v3x = (double *)malloc(DMnd*2 * sizeof(double) );
            v3c = (double *)malloc(DMnd * sizeof(double) );
        }

        // iexch
        switch (pSPARC->ixc[0])
        {
        case 1:
            slater_spin(DMnd, rho, ex ,vx);
            break;
        case 2:
            pbex_spin(DMnd, rho, sigma, pSPARC->xcoption[0], ex, vx, v2x);
            break;
        case 3:
            rPW86x_spin(DMnd, rho, sigma + DMnd, ex, vx, v2x);
            break;
        case 4:
            scanx_spin(DMnd, rho, sigma, tau, ex, vx, v2x, v3x);
            break;
        case 5:
            rscanx_spin(DMnd, rho, sigma, tau, ex, vx, v2x, v3x);
            break;
        case 6:
            r2scanx_spin(DMnd, rho, sigma, tau, ex, vx, v2x, v3x);
            break;
        default:
            memset(ex, 0, sizeof(double) * DMnd);
            memset(vx, 0, sizeof(double) * DMnd*2);
            if (pSPARC->isgradient) {
                memset(v2x, 0, sizeof(double) * DMnd*2);
            }
            break;
        }

        // icorr
        switch (pSPARC->ixc[1])
        {
        case 1:
            pz_spin(DMnd, rho, ec, vc);
            break;
        case 2:
            pw_spin(DMnd, rho, ec, vc);
            if (pSPARC->ixc[3]) {
                memcpy(pSPARC->vdWDFecLinear, ec, DMnd*sizeof(double));
                memcpy(pSPARC->vdWDFVcLinear, vc, DMnd*2*sizeof(double));
                memset(v2c, 0, sizeof(double) * DMnd);
            }
            break;
        case 3:
            pbec_spin(DMnd, rho, sigma, pSPARC->xcoption[1], ec, vc, v2c);
            break;
        case 4:
            scanc_spin(DMnd, rho, sigma, tau, ec, vc, v2c, v3c);
            break;
        case 5:
            rscanc_spin(DMnd, rho, sigma, tau, ec, vc, v2c, v3c);
            break;
        case 6:
            r2scanc_spin(DMnd, rho, sigma, tau, ec, vc, v2c, v3c);
            break;
        default:
            memset(ec, 0, sizeof(double) * DMnd);
            memset(vc, 0, sizeof(double) * DMnd*2);
            if (pSPARC->isgradient) {
                memset(v2c, 0, sizeof(double) * DMnd);
            }
            break;
        }

        if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"PBE0") == 0) {
            for (int i = 0; i < DMnd; i++) {
                ex[i] *= (1-pSPARC->exx_frac);
                for(int spn_i = 0; spn_i < 2; spn_i++) {
                    vx[i + spn_i*DMnd] *= (1-pSPARC->exx_frac);
                    v2x[i + spn_i*DMnd] *= (1-pSPARC->exx_frac);
                }
            }
        }

        if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0) && strcmpi(pSPARC->XC,"HSE") == 0) {
            for (int i = 0; i < DMnd; i++) {
                for(int spn_i = 0; spn_i < 2; spn_i++) {
                    double e_xc_sr, Dxcdgrho_sr, XCPotential_sr;
                    // Use the same strategy as \rho for \grho here. 
                    // Without this threshold, numerical issue will make simulation fail. 
                    if (sigma[DMnd + spn_i*DMnd + i] < 1E-14) sigma[DMnd + spn_i*DMnd + i] = 1E-14;
                    pbexsr(rho[DMnd + spn_i*DMnd + i] * 2.0, sigma[DMnd + spn_i*DMnd + i] * 4.0, pSPARC->hyb_range_pbe, &e_xc_sr, &XCPotential_sr, &Dxcdgrho_sr);
                    ex[i] -= pSPARC->exx_frac * e_xc_sr / 2.0 / rho[i];
                    vx[i+spn_i*DMnd] -= pSPARC->exx_frac * XCPotential_sr;
                    v2x[i+spn_i*DMnd] -= pSPARC->exx_frac * Dxcdgrho_sr * 2.0;
                }
            }
        }        

        for (int i = 0; i < DMnd; i++) {
            pSPARC->e_xc[i] = ex[i] + ec[i];
            pSPARC->XCPotential[i] = vx[i] + vc[i];
            pSPARC->XCPotential[i+DMnd] = vx[i+DMnd] + vc[i+DMnd];
            if (pSPARC->isgradient) {
                pSPARC->Dxcdgrho[i] = v2c[i];
                pSPARC->Dxcdgrho[i+DMnd] = v2x[i];
                pSPARC->Dxcdgrho[i+2*DMnd] = v2x[i+DMnd];
            }
            if ((pSPARC->ixc[2]) && (pSPARC->countPotentialCalculate)) {
                pSPARC->vxcMGGA3[i] = v3x[i] + v3c[i];
                pSPARC->vxcMGGA3[i+DMnd] = v3x[i+DMnd] + v3c[i];
            }
        }

        if (pSPARC->ixc[3])
            Calculate_nonLinearCorr_E_V_SvdWDF(pSPARC, rho);

        if (pSPARC->isgradient) {
            double *DDrho_x = (double *) malloc(3*DMnd * sizeof(double));
            double *DDrho_y = (double *) malloc(3*DMnd * sizeof(double));
            double *DDrho_z = (double *) malloc(3*DMnd * sizeof(double));

            Drho_times_v2xc(pSPARC, DMnd, 3, Drho_x, Drho_y, Drho_z, pSPARC->Dxcdgrho);
            if (pSPARC->CyclixFlag) {
                Gradient_vectors_dir_with_rotfac(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_x, Drho_y, DMnd, DDrho_x, DMnd, 0, pSPARC->dmcomm_phi);
                Gradient_vectors_dir_with_rotfac(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_y, Drho_x, DMnd, DDrho_y, DMnd, 1, pSPARC->dmcomm_phi);
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_z, DMnd, DDrho_z, DMnd, 2, pSPARC->dmcomm_phi);
            } else {
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_x, DMnd, DDrho_x, DMnd, 0, pSPARC->dmcomm_phi);
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_y, DMnd, DDrho_y, DMnd, 1, pSPARC->dmcomm_phi);
                Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, Drho_z, DMnd, DDrho_z, DMnd, 2, pSPARC->dmcomm_phi);
            }

            for(int i = 0; i < DMnd; i++){
                pSPARC->XCPotential[i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i] - DDrho_x[DMnd + i] - DDrho_y[DMnd + i] - DDrho_z[DMnd + i];
                pSPARC->XCPotential[DMnd + i] += -DDrho_x[i] - DDrho_y[i] - DDrho_z[i] - DDrho_x[2*DMnd + i] - DDrho_y[2*DMnd + i] - DDrho_z[2*DMnd + i];
            }
            free(DDrho_x);
            free(DDrho_y);
            free(DDrho_z);
        }

        free(ex);
        free(ec);
        free(vx);
        free(vc);
        if (pSPARC->isgradient) {
            free(v2x);
            free(v2c);
        }
        if (pSPARC->ixc[2]) {
            free(v3x);
            free(v3c);
        }
    }

    // calculate noncollinear xc potentail 
    if (pSPARC->spin_typ == 2) {
        Calculate_Xcpotential_Noncollinear(pSPARC, DMnd, pSPARC->XCPotential, pSPARC->mag, pSPARC->XCPotential_nc);
    }

    if (pSPARC->ixc[2]) {
        if (pSPARC->countPotentialCalculate == 0) { // restore metaGGA labels after 1st SCF
            if (strcmpi(pSPARC->XC, "SCAN") == 0) {
                pSPARC->ixc[0] = 4; pSPARC->ixc[1] = 4; 
                pSPARC->ixc[2] = 1; pSPARC->ixc[3] = 0;
                pSPARC->xcoption[0] = 0; pSPARC->xcoption[1] = 0;
            }
            else if (strcmpi(pSPARC->XC, "RSCAN") == 0) {
                pSPARC->ixc[0] = 5; pSPARC->ixc[1] = 5; 
                pSPARC->ixc[2] = 1; pSPARC->ixc[3] = 0;
                pSPARC->xcoption[0] = 0; pSPARC->xcoption[1] = 0;
            }
            else {
                pSPARC->ixc[0] = 6; pSPARC->ixc[1] = 6; 
                pSPARC->ixc[2] = 1; pSPARC->ixc[3] = 0;
                pSPARC->xcoption[0] = 0; pSPARC->xcoption[1] = 0;
            }
        } 
    }

    free(rho);
    if (pSPARC->isgradient) {
        free(sigma);
        free(Drho_x);
        free(Drho_y);
        free(Drho_z);
    }
}


/**
 * @brief  Calculate exchange correlation energy
 **/
void Calculate_Exc(SPARC_OBJ *pSPARC, double *electronDens)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 
    int DMnd = pSPARC->Nd_d;
    double *rho = (double *)malloc(DMnd * sizeof(double) );

    // add core electron density if needed
    add_rho_core(pSPARC, electronDens, rho, 1);

    double Exc = 0.0;
    if (pSPARC->CyclixFlag) {
        for (int i = 0; i < DMnd; i++) {
            Exc += rho[i] * pSPARC->e_xc[i] * pSPARC->Intgwt_phi[i]; 
        }
    } else {
        for (int i = 0; i < DMnd; i++) {
            Exc += rho[i] * pSPARC->e_xc[i]; 
        }
        Exc *= pSPARC->dV;
    }

    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->Exc = Exc;
    if (pSPARC->ixc[3]) Add_Exc_vdWDF(pSPARC); // the function is in /vdW/vdWDF/vdWDF.c
    free(rho);
}

/**
 * @brief   slater exchange
 */
void slater(int DMnd, double *rho, double *ex, double *vx) {
    // exchange parameter
    double C2 = 0.738558766382022;  // 3/4 * (3/pi)^(1/3)
    double C3 = 0.9847450218426965; // (3/pi)^(1/3)
    
    for (int i = 0; i < DMnd; i++) {
        double rho_cbrt = cbrt(rho[i]);
        ex[i] = - C2 * rho_cbrt;
        vx[i] = - C3 * rho_cbrt;
    }
}

/**
 * @brief   pw correaltion
 *          J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
 */
void pw(int DMnd, double *rho, double *ec, double *vc) {
    // correlation parameters
    double p = 1.0; 
    double A = 0.031091;
    double alpha1 = 0.21370;
    double beta1 = 7.5957;
    double beta2 = 3.5876;
    double beta3 = 1.6382;
    double beta4 = 0.49294;
    double C31 = 0.6203504908993999; // (3/4pi)^(1/3)    
    
    for (int i = 0; i < DMnd; i++) {
        double rho_cbrt = cbrt(rho[i]);
        double rs = C31 / rho_cbrt; // rs = (3/(4*pi*rho))^(1/3)
        double rs_sqrt = sqrt(rs); // rs^0.5
        double rs_pow_1p5 = rs * rs_sqrt; // rs^1.5
        double rs_pow_p = rs; // rs^p, where p = 1, pow function is slow (~100x slower than add/sub)
        double rs_pow_pplus1 = rs_pow_p * rs; // rs^(p+1)
        double G1 = log(1.0+1.0/(2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1 )));
        double G2 = 2.0*A*(beta1*rs_sqrt + beta2*rs + beta3*rs_pow_1p5 + beta4*rs_pow_pplus1);
        
        ec[i] = -2.0*A*(1.0+alpha1*rs) * G1;
        vc[i] = ec[i] - (rs/3.0) * ( -2.0*A*alpha1 * G1 + (2.0*A*(1.0+alpha1*rs) * (A*(beta1/rs_sqrt + 2.0*beta2 + 3.0*beta3*rs_sqrt + 2.0*(p+1.0)*beta4*rs_pow_p))) / (G2 * (G2 + 1.0)) );
    }
}

/**
 * @brief   pz correaltion
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 */
void pz(int DMnd, double *rho, double *ec, double *vc) {
    // parameters
    double A = 0.0311;
    double B = -0.048 ;
    double C = 0.002 ;
    double D = -0.0116 ;
    double gamma1 = -0.1423 ;
    double beta1 = 1.0529 ;
    double beta2 = 0.3334 ; 
    double C31 = 0.6203504908993999; // (3/4pi)^(1/3)    
    
    for (int i = 0; i < DMnd; i++) {
        double rho_cbrt = cbrt(rho[i]);
        double rs = C31 / rho_cbrt; // rs = (3/(4*pi*rho))^(1/3)
        if (rs<1.0) {
            ec[i] = A*log(rs) + B + C*rs*log(rs) + D*rs;
            vc[i] = log(rs)*(A+(2.0/3.0)*C*rs) + (B-(1.0/3.0)*A) + (1.0/3.0)*(2.0*D-C)*rs; 
        } else {
            double sqrtrs = sqrt(rs);
            ec[i] = gamma1/(1.0+beta1*sqrt(rs)+beta2*rs);
            vc[i] = (gamma1 + (7.0/6.0)*gamma1*beta1*sqrtrs 
                    + (4.0/3.0)*gamma1*beta2*rs)/pow(1+beta1*sqrtrs+beta2*rs,2.0) ;		  
        }
    }
}


/**
 * @brief   pbe exchange
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   iflag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 */
void pbex(int DMnd, double *rho, double *sigma, int iflag, double *ex, double *vx, double *v2x) {
    assert(iflag == 1 || iflag == 2 || iflag == 3 || iflag == 4);
    double mu_[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    double kappa_[4] = {0.804, 0.804, 0.804, 1.245};
    
    // parameters 
    double kappa = kappa_[iflag-1];
    double mu = mu_[iflag-1];
    double mu_divkappa = mu / kappa;
    double threefourth_divpi = (3.0/4.0) / M_PI;
    double third = 1.0/3.0;
    double sixpi2_1_3 = pow(6.0*M_PI*M_PI, third);
    double sixpi2m1_3 = 1.0/sixpi2_1_3;
    
    for(int i = 0; i < DMnd; i++){
        double rho_updn = rho[i]/2.0;
        double rho_updnm1_3 = pow(rho_updn, -third);

        // First take care of the exchange part of the functional
        double rhomot = rho_updnm1_3;
        double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);

        // Perdew-Burke-Ernzerhof GGA, exchange part
        double rho_inv = rhomot * rhomot * rhomot;
        double coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
        double ss = (sigma[i]/4.0) * coeffss; // s^2

        double divss, dfxdss;
        if (iflag == 1 || iflag == 2 || iflag == 4) {
            divss = 1.0/(1.0 + mu_divkappa * ss);
            dfxdss = mu * (divss * divss);
        } else if (iflag == 3) {
            divss = exp(-mu_divkappa * ss);
            dfxdss = mu * divss;
        }
        
        double fx = 1.0 + kappa * (1.0 - divss);
        double dssdn = (-8.0/3.0) * (ss * rho_inv);
        double dfxdn = dfxdss * dssdn;
        double dssdg = 2.0 * coeffss;
        double dfxdg = dfxdss * dssdg;

        ex[i] = ex_lsd * fx;
        vx[i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);
        v2x[i] = 0.5 * ex_lsd * rho_updn * dfxdg;
    }
}

/**
* @brief the function to compute the potential and energy density of PW86 GGA exchange
*/
void rPW86x(int DMnd, double *rho, double *sigma, double *vdWDFex, double *vdWDFVx1, double *vdWDFVx2) {
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
 * @brief   pbe correlation
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 */
void pbec(int DMnd, double *rho, double *sigma, int iflag, double *ec, double *vc, double *v2c) {
    assert(iflag == 1 || iflag == 2 || iflag == 3);
    double beta_[3] = {0.066725, 0.046, 0.066725};
    
    // parameters 
    double beta = beta_[iflag-1];
    double gamma = (1.0 - log(2.0)) / (M_PI*M_PI);
    double gamma_inv = 1.0/gamma;
    double phi_zeta_inv = 1.0;
    double phi3_zeta = 1.0;
    double gamphi3inv = gamma_inv;
    double third = 1.0/3.0;
    double twom1_3 = pow(2.0,-third);
    double rsfac = 0.6203504908994000;
    double sq_rsfac = sqrt(rsfac);
    double sq_rsfac_inv = 1.0/sq_rsfac;
    double coeff_tt = 1.0/(16.0 / M_PI * pow(3.0*M_PI*M_PI,third));
    double ec0_aa = 0.031091; 
    double ec0_a1 = 0.21370;  
    double ec0_b1 = 7.5957;
    double ec0_b2 = 3.5876;   
    double ec0_b3 = 1.6382;   
    double ec0_b4 = 0.49294;  

    for(int i = 0; i < DMnd; i++){
        double rho_updn = rho[i]/2.0;
        double rho_updnm1_3 = pow(rho_updn, -third);
        double rhom1_3 = twom1_3 * rho_updnm1_3;
        double rhotot_inv = rhom1_3 * rhom1_3 * rhom1_3;
        double rhotmo6 = sqrt(rhom1_3);
        double rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Then takes care of the LSD correlation part of the functional
        double rs = rsfac * rhom1_3;
        double sqr_rs = sq_rsfac * rhotmo6;
        double rsm1_2 = sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
        double ec0_q1 = 2.0 * ec0_aa * (ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
        double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
        double ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        double ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        double ecrs0 = ec0_q0 * ec0_log;
        double decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        double ecrs = ecrs0;
        double decrs_drs = decrs0_drs;

        // Add LSD correlation functional to GGA exchange functional
        ec[i] = ecrs;
        vc[i] = ecrs - (rs/3.0) * decrs_drs;

        // Eventually add the GGA correlation part of the PBE functional
        // Note : the computation of the potential in the spin-unpolarized
        // case could be optimized much further. Other optimizations are left to do.

        // From ec to bb
        double bb = ecrs * gamphi3inv;
        double dbb_drs = decrs_drs * gamphi3inv;
        // dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        double exp_pbe = exp(-bb);
        double cc = 1.0/(exp_pbe - 1.0);
        double dcc_dbb = cc * cc * exp_pbe;
        double dcc_drs = dcc_dbb * dbb_drs;
        // dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        double coeff_aa = beta * gamma_inv * phi_zeta_inv * phi_zeta_inv;
        double aa = coeff_aa * cc;
        double daa_drs = coeff_aa * dcc_drs;
        //daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        double grrho2 = sigma[i];
        double dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        double tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        double xx = aa * tt;
        double dxx_drs = daa_drs * tt;
        double dxx_dtt = aa;

        // From xx to pade
        double pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        double pade = (1.0 + xx) * pade_den;
        double dpade_dxx = -xx * (2.0 + xx) * pow(pade_den,2);
        double dpade_drs = dpade_dxx * dxx_drs;
        double dpade_dtt = dpade_dxx * dxx_dtt;
        //dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        double coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        double qq = coeff_qq * pade;
        double dqq_drs = coeff_qq * dpade_drs;
        double dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        //dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        double arg_rr = 1.0 + beta * gamma_inv * qq;
        double div_rr = 1.0/arg_rr;
        double rr = gamma * log(arg_rr);
        double drr_dqq = beta * div_rr;
        double drr_drs = drr_dqq * dqq_drs;
        double drr_dtt = drr_dqq * dqq_dtt;
        //drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        double hh = phi3_zeta * rr;
        double dhh_drs = phi3_zeta * drr_drs;
        double dhh_dtt = phi3_zeta * drr_dtt;
        //dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        ec[i] += hh;

        // From hh to the derivative of the energy wrt the density
        double drhohh_drho = hh - third * rs * dhh_drs - (7.0/3.0) * tt * dhh_dtt; //- zeta * dhh_dzeta 
        vc[i] += drhohh_drho;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        v2c[i] = (rho[i] * dtt_dg * dhh_dtt);    
    }
}

/**
 * @brief   slater exchange - spin polarized 
 */
void slater_spin(int DMnd, double *rho, double *ex, double *vx) {
    // parameter 
    double third = 1.0/3.0;
    double threefourth_divpi = (3.0/4.0) / M_PI;
    double sixpi2_1_3 = pow(6.0*M_PI*M_PI, third);

    for(int i = 0; i < DMnd; i++) {
        double rhom1_3 = pow(rho[i],-third);
        double rhotot_inv = pow(rhom1_3,3.0);  

        // First take care of the exchange part of the functional
        double extot = 0.0;
        for(int spn_i = 0; spn_i < 2; spn_i++){
            double rho_updn = rho[DMnd + spn_i*DMnd + i]; 
            double rho_updnm1_3 = pow(rho_updn, -third);
            double rhomot = rho_updnm1_3;
            double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            vx[spn_i*DMnd + i] = (4.0/3.0) * ex_lsd;
            extot += ex_lsd * rho_updn;
        }
        ex[i] = extot * rhotot_inv;
    }
}

/**
 * @brief   pw correaltion - spin polarized 
 *          J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
 */
void pw_spin(int DMnd, double *rho, double *ec, double *vc) {
    // correlation parameters
    double third = 1.0/3.0;
    double alpha_zeta2 = 1.0 - 1.0e-6; 
    double alpha_zeta = 1.0 - 1.0e-6;
    double rsfac = 0.6203504908994000;
    double sq_rsfac = sqrt(rsfac);
    double sq_rsfac_inv = 1.0/sq_rsfac;
    double fsec_inv = 1.0/1.709921;
    double factf_zeta = 1.0/(pow(2.0,(4.0/3.0)) - 2.0);
    double factfp_zeta = 4.0/3.0 * factf_zeta * alpha_zeta2;
    double ec0_aa = 0.031091; double ec1_aa = 0.015545; double mac_aa = 0.016887;
    double ec0_a1 = 0.21370;  double ec1_a1 = 0.20548;  double mac_a1 = 0.11125;
    double ec0_b1 = 7.5957;   double ec1_b1 = 14.1189;  double mac_b1 = 10.357;
    double ec0_b2 = 3.5876;   double ec1_b2 = 6.1977;   double mac_b2 = 3.6231;
    double ec0_b3 = 1.6382;   double ec1_b3 = 3.3662;   double mac_b3 = 0.88026;
    double ec0_b4 = 0.49294;  double ec1_b4 = 0.62517;  double mac_b4 = 0.49671;

    for(int i = 0; i < DMnd; i++) {
        double rhom1_3 = pow(rho[i],-third);
        double rhotot_inv = pow(rhom1_3,3.0);
        double zeta = (rho[DMnd+i] - rho[2*DMnd+i]) * rhotot_inv;
        double zetp = 1.0 + zeta * alpha_zeta;
        double zetm = 1.0 - zeta * alpha_zeta;
        double zetpm1_3 = pow(zetp,-third);
        double zetmm1_3 = pow(zetm,-third);
        double rhotmo6 = sqrt(rhom1_3);
        double rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Then takes care of the LSD correlation part of the functional
        double rs = rsfac * rhom1_3;
        double sqr_rs = sq_rsfac * rhotmo6;
        double rsm1_2 = sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
        double ec0_q1 = 2.0 * ec0_aa *(ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
        double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
        double ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        double ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        double ecrs0 = ec0_q0 * ec0_log;
        double decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        double mac_q0 = -2.0 * mac_aa * (1.0 + mac_a1 * rs);
        double mac_q1 = 2.0 * mac_aa * (mac_b1 * sqr_rs + mac_b2 * rs + mac_b3 * rs * sqr_rs + mac_b4 * rs * rs);
        double mac_q1p = mac_aa * (mac_b1 * rsm1_2 + 2.0 * mac_b2 + 3.0 * mac_b3 * sqr_rs + 4.0 * mac_b4 * rs);
        double mac_den = 1.0/(mac_q1 * mac_q1 + mac_q1);
        double mac_log = -log( mac_q1 * mac_q1 * mac_den );
        double macrs = mac_q0 * mac_log;
        double dmacrs_drs = -2.0 * mac_aa * mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

        double ec1_q0 = -2.0 * ec1_aa * (1.0 + ec1_a1 * rs);
        double ec1_q1 = 2.0 * ec1_aa * (ec1_b1 * sqr_rs + ec1_b2 * rs + ec1_b3 * rs * sqr_rs + ec1_b4 * rs * rs);
        double ec1_q1p = ec1_aa * (ec1_b1 * rsm1_2 + 2.0 * ec1_b2 + 3.0 * ec1_b3 * sqr_rs + 4.0 * ec1_b4 * rs);
        double ec1_den = 1.0/(ec1_q1 * ec1_q1 + ec1_q1);
        double ec1_log = -log( ec1_q1 * ec1_q1 * ec1_den );
        double ecrs1 = ec1_q0 * ec1_log;
        double decrs1_drs = -2.0 * ec1_aa * ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;
        
        // alpha_zeta is introduced in order to remove singularities for fully polarized systems.
        double zetp_1_3 = (1.0 + zeta * alpha_zeta) * pow(zetpm1_3,2.0);
        double zetm_1_3 = (1.0 - zeta * alpha_zeta) * pow(zetmm1_3,2.0);

        double f_zeta = ( (1.0 + zeta * alpha_zeta2) * zetp_1_3 + (1.0 - zeta * alpha_zeta2) * zetm_1_3 - 2.0 ) * factf_zeta;
        double fp_zeta = ( zetp_1_3 - zetm_1_3 ) * factfp_zeta;
        double zeta4 = pow(zeta, 4.0);

        double gcrs = ecrs1 - ecrs0 + macrs * fsec_inv;
        double ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * fsec_inv);
        double dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * fsec_inv;
        double decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * fsec_inv);
        double dfzeta4_dzeta = 4.0 * pow(zeta,3.0) * f_zeta + fp_zeta * zeta4;
        double decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * fsec_inv;
        double vxcadd = ecrs - rs * third * decrs_drs - zeta * decrs_dzeta;

        ec[i] = ecrs;
        vc[i] = vxcadd + decrs_dzeta;
        vc[DMnd+i] = vxcadd - decrs_dzeta;
    }
}

/**
 * @brief   pz correaltion - spin polarized 
 *          J.P. Perdew and A. Zunger, PRB 23, 5048 (1981).
 */
void pz_spin(int DMnd, double *rho, double *ec, double *vc) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) printf("ERROR: LDA_PZ for spin polarized case is not implemented!\n");
    exit(EXIT_FAILURE);
}

/**
 * @brief   pbe exchange - spin polarized
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 * @param   iflag=4  Zhang-Yang Revised PBE: Y. Zhang and W. Yang., Phys. Rev. Lett. 80, 890 (1998)
 */
void pbex_spin(int DMnd, double *rho, double *sigma, int iflag, double *ex, double *vx, double *v2x) {
    assert(iflag == 1 || iflag == 2 || iflag == 3 || iflag == 4);
    double mu_[4] = {0.2195149727645171, 10.0/81.0, 0.2195149727645171, 0.2195149727645171};
    double kappa_[4] = {0.804, 0.804, 0.804, 1.245};
    
    // parameters 
    double kappa = kappa_[iflag-1];
    double mu = mu_[iflag-1];
    double mu_divkappa = mu / kappa;
    double threefourth_divpi = (3.0/4.0) / M_PI;
    double third = 1.0/3.0;
    double sixpi2_1_3 = pow(6.0*M_PI*M_PI, third);
    double sixpi2m1_3 = 1.0/sixpi2_1_3;

    for(int i = 0; i < DMnd; i++) {
        double rhom1_3 = pow(rho[i],-third);
        double rhotot_inv = pow(rhom1_3,3.0);

        // First take care of the exchange part of the functional
        double extot = 0.0;
        for(int spn_i = 0; spn_i < 2; spn_i++){
            double rho_updn = rho[DMnd + spn_i*DMnd + i];
            double rho_updnm1_3 = pow(rho_updn, -third);
            double rhomot = rho_updnm1_3;
            double ex_lsd = -threefourth_divpi * sixpi2_1_3 * (rhomot * rhomot * rho_updn);
            double rho_inv = rhomot * rhomot * rhomot;
            double coeffss = (1.0/4.0) * sixpi2m1_3 * sixpi2m1_3 * (rho_inv * rho_inv * rhomot * rhomot);
            double ss = sigma[DMnd + spn_i*DMnd + i] * coeffss;
            
            double divss, dfxdss;
            if (iflag == 1 || iflag == 2 || iflag == 4) {
                divss = 1.0/(1.0 + mu_divkappa * ss);
                dfxdss = mu * (divss * divss);
            } else if (iflag == 3) {
                divss = exp(-mu_divkappa * ss);
                dfxdss = mu * divss;
            }

			double fx = 1.0 + kappa * (1.0 - divss);
            double ex_gga = ex_lsd * fx;
            double dssdn = (-8.0/3.0) * (ss * rho_inv);
            double dfxdn = dfxdss * dssdn;
            vx[spn_i*DMnd + i] = ex_lsd * ((4.0/3.0) * fx + rho_updn * dfxdn);

            double dssdg = 2.0 * coeffss;
            double dfxdg = dfxdss * dssdg;
            v2x[spn_i*DMnd + i] = ex_lsd * rho_updn * dfxdg; // changed to assuming 2 columns 
            extot += ex_gga * rho_updn;
        }
        ex[i] = extot * rhotot_inv;
    }
}


/**
* @brief the function to compute the potential and energy density of PW86 GGA exchange, spin-polarized case
*/
void rPW86x_spin(int DMnd, double *rho, double *sigma, double *vdWDFex, double *vdWDFVx1, double *vdWDFVx2) {
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
 * @brief   pbe correlation - spin polarized
 *
 * @param   iflag=1  J.P.Perdew, K.Burke, M.Ernzerhof, PRL 77, 3865 (1996)
 * @param   iflag=2  PBEsol: J.P.Perdew et al., PRL 100, 136406 (2008)
 * @param   iflag=3  RPBE: B. Hammer, et al., Phys. Rev. B 59, 7413 (1999)
 */
void pbec_spin(int DMnd, double *rho, double *sigma, int iflag, double *ec, double *vc, double *v2c) {
    assert(iflag == 1 || iflag == 2 || iflag == 3);
    double beta_[3] = {0.066725, 0.046, 0.066725};
    
    // parameters 
    double beta = beta_[iflag-1];
    double gamma = (1.0 - log(2.0)) / (M_PI*M_PI);
    double gamma_inv = 1.0/gamma;    
    double third = 1.0/3.0;
    double alpha_zeta2 = 1.0 - 1.0e-6; 
    double alpha_zeta = 1.0 - 1.0e-6;
    double rsfac = 0.6203504908994000;
    double sq_rsfac = sqrt(rsfac);
    double sq_rsfac_inv = 1.0/sq_rsfac;
    double fsec_inv = 1.0/1.709921;
    double factf_zeta = 1.0/(pow(2.0,(4.0/3.0)) - 2.0);
    double factfp_zeta = 4.0/3.0 * factf_zeta * alpha_zeta2;
    double coeff_tt = 1.0/(16.0 / M_PI * pow(3.0*M_PI*M_PI,third));

    double ec0_aa = 0.031091; double ec1_aa = 0.015545; double mac_aa = 0.016887;
    double ec0_a1 = 0.21370;  double ec1_a1 = 0.20548;  double mac_a1 = 0.11125;
    double ec0_b1 = 7.5957;   double ec1_b1 = 14.1189;  double mac_b1 = 10.357;
    double ec0_b2 = 3.5876;   double ec1_b2 = 6.1977;   double mac_b2 = 3.6231;
    double ec0_b3 = 1.6382;   double ec1_b3 = 3.3662;   double mac_b3 = 0.88026;
    double ec0_b4 = 0.49294;  double ec1_b4 = 0.62517;  double mac_b4 = 0.49671;
    
    for(int i = 0; i < DMnd; i++) {
        double rhom1_3 = pow(rho[i],-third);
        double rhotot_inv = pow(rhom1_3,3.0);
        double zeta = (rho[DMnd + i] - rho[2*DMnd + i]) * rhotot_inv;
        double zetp = 1.0 + zeta * alpha_zeta;
        double zetm = 1.0 - zeta * alpha_zeta;
        double zetpm1_3 = pow(zetp,-third);
        double zetmm1_3 = pow(zetm,-third);
        double rhotmo6 = sqrt(rhom1_3);
        double rhoto6 = rho[i] * rhom1_3 * rhom1_3 * rhotmo6;

        // Then takes care of the LSD correlation part of the functional
        double rs = rsfac * rhom1_3;
        double sqr_rs = sq_rsfac * rhotmo6;
        double rsm1_2 = sq_rsfac_inv * rhoto6;

        // Formulas A6-A8 of PW92LSD
        double ec0_q0 = -2.0 * ec0_aa * (1.0 + ec0_a1 * rs);
        double ec0_q1 = 2.0 * ec0_aa *(ec0_b1 * sqr_rs + ec0_b2 * rs + ec0_b3 * rs * sqr_rs + ec0_b4 * rs * rs);
        double ec0_q1p = ec0_aa * (ec0_b1 * rsm1_2 + 2.0 * ec0_b2 + 3.0 * ec0_b3 * sqr_rs + 4.0 * ec0_b4 * rs);
        double ec0_den = 1.0/(ec0_q1 * ec0_q1 + ec0_q1);
        double ec0_log = -log(ec0_q1 * ec0_q1 * ec0_den);
        double ecrs0 = ec0_q0 * ec0_log;
        double decrs0_drs = -2.0 * ec0_aa * ec0_a1 * ec0_log - ec0_q0 * ec0_q1p * ec0_den;

        double mac_q0 = -2.0 * mac_aa * (1.0 + mac_a1 * rs);
        double mac_q1 = 2.0 * mac_aa * (mac_b1 * sqr_rs + mac_b2 * rs + mac_b3 * rs * sqr_rs + mac_b4 * rs * rs);
        double mac_q1p = mac_aa * (mac_b1 * rsm1_2 + 2.0 * mac_b2 + 3.0 * mac_b3 * sqr_rs + 4.0 * mac_b4 * rs);
        double mac_den = 1.0/(mac_q1 * mac_q1 + mac_q1);
        double mac_log = -log( mac_q1 * mac_q1 * mac_den );
        double macrs = mac_q0 * mac_log;
        double dmacrs_drs = -2.0 * mac_aa * mac_a1 * mac_log - mac_q0 * mac_q1p * mac_den;

        double ec1_q0 = -2.0 * ec1_aa * (1.0 + ec1_a1 * rs);
        double ec1_q1 = 2.0 * ec1_aa * (ec1_b1 * sqr_rs + ec1_b2 * rs + ec1_b3 * rs * sqr_rs + ec1_b4 * rs * rs);
        double ec1_q1p = ec1_aa * (ec1_b1 * rsm1_2 + 2.0 * ec1_b2 + 3.0 * ec1_b3 * sqr_rs + 4.0 * ec1_b4 * rs);
        double ec1_den = 1.0/(ec1_q1 * ec1_q1 + ec1_q1);
        double ec1_log = -log( ec1_q1 * ec1_q1 * ec1_den );
        double ecrs1 = ec1_q0 * ec1_log;
        double decrs1_drs = -2.0 * ec1_aa * ec1_a1 * ec1_log - ec1_q0 * ec1_q1p * ec1_den;
        
        // alpha_zeta is introduced in order to remove singularities for fully polarized systems.
        double zetp_1_3 = (1.0 + zeta * alpha_zeta) * pow(zetpm1_3,2.0);
        double zetm_1_3 = (1.0 - zeta * alpha_zeta) * pow(zetmm1_3,2.0);

        double f_zeta = ( (1.0 + zeta * alpha_zeta2) * zetp_1_3 + (1.0 - zeta * alpha_zeta2) * zetm_1_3 - 2.0 ) * factf_zeta;
        double fp_zeta = ( zetp_1_3 - zetm_1_3 ) * factfp_zeta;
        double zeta4 = pow(zeta, 4.0);

        double gcrs = ecrs1 - ecrs0 + macrs * fsec_inv;
        double ecrs = ecrs0 + f_zeta * (zeta4 * gcrs - macrs * fsec_inv);
        double dgcrs_drs = decrs1_drs - decrs0_drs + dmacrs_drs * fsec_inv;
        double decrs_drs = decrs0_drs + f_zeta * (zeta4 * dgcrs_drs - dmacrs_drs * fsec_inv);
        double dfzeta4_dzeta = 4.0 * pow(zeta,3.0) * f_zeta + fp_zeta * zeta4;
        double decrs_dzeta = dfzeta4_dzeta * gcrs - fp_zeta * macrs * fsec_inv;

        ec[i] = ecrs;
        double vxcadd = ecrs - rs * third * decrs_drs - zeta * decrs_dzeta;
        vc[i] = vxcadd + decrs_dzeta;
        vc[DMnd+i] = vxcadd - decrs_dzeta;

        // Eventually add the GGA correlation part of the PBE functional
        // The definition of phi has been slightly changed, because
        // the original PBE one gives divergent behaviour for fully polarized points
        
        double phi_zeta = ( zetpm1_3 * (1.0 + zeta * alpha_zeta) + zetmm1_3 * (1.0 - zeta * alpha_zeta)) * 0.5;
        double phip_zeta = (zetpm1_3 - zetmm1_3) * third * alpha_zeta;
        double phi_zeta_inv = 1.0/phi_zeta;
        double phi_logder = phip_zeta * phi_zeta_inv;
        double phi3_zeta = phi_zeta * phi_zeta * phi_zeta;
        double gamphi3inv = gamma_inv * phi_zeta_inv * phi_zeta_inv * phi_zeta_inv;        
        
        // From ec to bb
        double bb = ecrs * gamphi3inv;
        double dbb_drs = decrs_drs * gamphi3inv;
        double dbb_dzeta = gamphi3inv * (decrs_dzeta - 3.0 * ecrs * phi_logder);

        // From bb to cc
        double exp_pbe = exp(-bb);
        double cc = 1.0/(exp_pbe - 1.0);
        double dcc_dbb = cc * cc * exp_pbe;
        double dcc_drs = dcc_dbb * dbb_drs;
        double dcc_dzeta = dcc_dbb * dbb_dzeta;

        // From cc to aa
        double coeff_aa = beta * gamma_inv * phi_zeta_inv * phi_zeta_inv;
        double aa = coeff_aa * cc;
        double daa_drs = coeff_aa * dcc_drs;
        double daa_dzeta = -2.0 * aa * phi_logder + coeff_aa * dcc_dzeta;

        // Introduce tt : do not assume that the spin-dependent gradients are collinear
        double grrho2 = sigma[i];
        double dtt_dg = 2.0 * rhotot_inv * rhotot_inv * rhom1_3 * coeff_tt;
        // Note that tt is (the t variable of PBE divided by phi) squared
        double tt = 0.5 * grrho2 * dtt_dg;

        // Get xx from aa and tt
        double xx = aa * tt;
        double dxx_drs = daa_drs * tt;
        double dxx_dzeta = daa_dzeta * tt;
        double dxx_dtt = aa;

        // From xx to pade
        double pade_den = 1.0/(1.0 + xx * (1.0 + xx));
        double pade = (1.0 + xx) * pade_den;
        double dpade_dxx = -xx * (2.0 + xx) * pow(pade_den,2.0);
        double dpade_drs = dpade_dxx * dxx_drs;
        double dpade_dtt = dpade_dxx * dxx_dtt;
        double dpade_dzeta = dpade_dxx * dxx_dzeta;

        // From pade to qq
        double coeff_qq = tt * phi_zeta_inv * phi_zeta_inv;
        double qq = coeff_qq * pade;
        double dqq_drs = coeff_qq * dpade_drs;
        double dqq_dtt = pade * phi_zeta_inv * phi_zeta_inv + coeff_qq * dpade_dtt;
        double dqq_dzeta = coeff_qq * (dpade_dzeta - 2.0 * pade * phi_logder);

        // From qq to rr
        double arg_rr = 1.0 + beta * gamma_inv * qq;
        double div_rr = 1.0/arg_rr;
        double rr = gamma * log(arg_rr);
        double drr_dqq = beta * div_rr;
        double drr_drs = drr_dqq * dqq_drs;
        double drr_dtt = drr_dqq * dqq_dtt;
        double drr_dzeta = drr_dqq * dqq_dzeta;

        // From rr to hh
        double hh = phi3_zeta * rr;
        double dhh_drs = phi3_zeta * drr_drs;
        double dhh_dtt = phi3_zeta * drr_dtt;
        double dhh_dzeta = phi3_zeta * (drr_dzeta + 3.0 * rr * phi_logder);

        // The GGA correlation energy is added
        ec[i] += hh;

        // From hh to the derivative of the energy wrt the density
        double drhohh_drho = hh - third * rs * dhh_drs - zeta * dhh_dzeta - (7.0/3.0) * tt * dhh_dtt; 
        vc[i] += drhohh_drho + dhh_dzeta;
        vc[DMnd + i] += drhohh_drho - dhh_dzeta;

        // From hh to the derivative of the energy wrt to the gradient of the
        // density, divided by the gradient of the density
        // (The v3.3 definition includes the division by the norm of the gradient)
        v2c[i] = rho[i] * dtt_dg * dhh_dtt;
    }
}


/**
 * @brief   calculate square norm of gradient
 */
void calculate_square_norm_of_gradient(SPARC_OBJ *pSPARC, 
        double *rho, double *mag, int DMnd, int ncol, 
        double *sigma, double *Drho_x, double *Drho_y, double *Drho_z)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    if (pSPARC->spin_typ == 2) {        
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, DMnd, Drho_x, DMnd, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, DMnd, Drho_y, DMnd, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, DMnd, Drho_z, DMnd, 2, pSPARC->dmcomm_phi);
        
        double *Dmag_x = (double *) malloc(3 * DMnd * sizeof(double)); // [Dmagx_x Dmagy_x Dmagz_x]
        double *Dmag_y = (double *) malloc(3 * DMnd * sizeof(double)); // [Dmagx_y Dmagy_y Dmagz_y]
        double *Dmag_z = (double *) malloc(3 * DMnd * sizeof(double)); // [Dmagx_z Dmagy_z Dmagz_z]
        
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, mag+DMnd, DMnd, Dmag_x, DMnd, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, mag+DMnd, DMnd, Dmag_y, DMnd, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 3, 0.0, mag+DMnd, DMnd, Dmag_z, DMnd, 2, pSPARC->dmcomm_phi);

        double *Dmnorm_x = (double *) malloc(DMnd * sizeof(double));
        double *Dmnorm_y = (double *) malloc(DMnd * sizeof(double));
        double *Dmnorm_z = (double *) malloc(DMnd * sizeof(double));

        // compute gradient of norm of magnetization |mag|
        for (int i = 0; i < DMnd; i++) {
            double magnorm = mag[i];
            if (magnorm > pSPARC->xc_magtol) {
                Dmnorm_x[i] = (mag[i+DMnd] * Dmag_x[i] + mag[i+2*DMnd] * Dmag_x[i+DMnd] + mag[i+3*DMnd] * Dmag_x[i+2*DMnd]) / magnorm;
                Dmnorm_y[i] = (mag[i+DMnd] * Dmag_y[i] + mag[i+2*DMnd] * Dmag_y[i+DMnd] + mag[i+3*DMnd] * Dmag_y[i+2*DMnd]) / magnorm;
                Dmnorm_z[i] = (mag[i+DMnd] * Dmag_z[i] + mag[i+2*DMnd] * Dmag_z[i+DMnd] + mag[i+3*DMnd] * Dmag_z[i+2*DMnd]) / magnorm;
            } else {
                Dmnorm_x[i] = Dmnorm_y[i] = Dmnorm_z[i] = 0;
            }
        }

        // compute gradient of effective up and down density
        for (int i = 0; i < DMnd; i++) {
            Drho_x[i+DMnd]   = 0.5 * (Drho_x[i] + Dmnorm_x[i]); // Drhoup_x
            Drho_x[i+2*DMnd] = 0.5 * (Drho_x[i] - Dmnorm_x[i]); // Drhodn_x
            Drho_y[i+DMnd]   = 0.5 * (Drho_y[i] + Dmnorm_y[i]); // Drhoup_y
            Drho_y[i+2*DMnd] = 0.5 * (Drho_y[i] - Dmnorm_y[i]); // Drhodn_y
            Drho_z[i+DMnd]   = 0.5 * (Drho_z[i] + Dmnorm_z[i]); // Drhoup_z
            Drho_z[i+2*DMnd] = 0.5 * (Drho_z[i] - Dmnorm_z[i]); // Drhodn_z
        }

        compute_norm_square(pSPARC, sigma, 3*DMnd, Drho_x, Drho_y, Drho_z);

        free(Dmag_x);
        free(Dmag_y);
        free(Dmag_z);
        free(Dmnorm_x);
        free(Dmnorm_y);
        free(Dmnorm_z);

    } else {
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, ncol, 0.0, rho, DMnd, Drho_x, DMnd, 0, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, ncol, 0.0, rho, DMnd, Drho_y, DMnd, 1, pSPARC->dmcomm_phi);
        Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, ncol, 0.0, rho, DMnd, Drho_z, DMnd, 2, pSPARC->dmcomm_phi);
        compute_norm_square(pSPARC, sigma, ncol*DMnd, Drho_x, Drho_y, Drho_z);
    }

    // put min threshold on sigma, otherwise numerical issues happen
    for (int i = 0; i < ncol*DMnd; i++) sigma[i] = max(sigma[i], pSPARC->xc_sigmatol);
}



/**
 * @brief   calculate square norm of a set of vector
 */ 
void compute_norm_square(SPARC_OBJ *pSPARC, double *norm2, int DMnd, double *v1, double *v2, double *v3)
{
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        double lapcT[6];
        lapcT[0] = pSPARC->lapcT[0]; lapcT[1] = 2 * pSPARC->lapcT[1]; lapcT[2] = 2 * pSPARC->lapcT[2];
        lapcT[3] = pSPARC->lapcT[4]; lapcT[4] = 2 * pSPARC->lapcT[5]; lapcT[5] = pSPARC->lapcT[8]; 
        for(int i = 0; i < DMnd; i++){
            norm2[i] = v1[i] * (lapcT[0] * v1[i] + lapcT[1] * v2[i]) 
                    + v2[i] * (lapcT[3] * v2[i] + lapcT[4] * v3[i]) 
                    + v3[i] * (lapcT[5] * v3[i] + lapcT[2] * v1[i]); 
        }
    } else {
        for(int i = 0; i < DMnd; i++){
            norm2[i] = v1[i] * v1[i] + v2[i] * v2[i] + v3[i] * v3[i];
        }
    }
}


/**
 * @brief   add core electron density if needed
 */
void add_rho_core(SPARC_OBJ *pSPARC, double *rho_in, double *rho_out, int ncol)
{
    assert(ncol == 1 || ncol == 3);
    int DMnd = pSPARC->Nd_d;
    for (int n = 0; n < ncol; n++) {
        for(int i = 0; i < DMnd; i++){
            rho_out[i + n*DMnd] = rho_in[i + n*DMnd];
            // for non-linear core correction, use rho+rho_core to evaluate Vxc[rho+rho_core]
            if (pSPARC->NLCC_flag) {
                if (n == 0)
                    rho_out[i + n*DMnd] += pSPARC->electronDens_core[i];
                else
                    rho_out[i + n*DMnd] += 0.5 * pSPARC->electronDens_core[i];
            }
            if(rho_out[i + n*DMnd] < pSPARC->xc_rhotol)
                rho_out[i + n*DMnd] = pSPARC->xc_rhotol;
        }
    }

    if (ncol == 3) {
        for(int i = 0; i < DMnd; i++) {
            rho_out[i] = rho_out[DMnd + i] + rho_out[2*DMnd + i];
        }
    }
}


/**
 * @brief   compute Drho times v2xc
 */
void Drho_times_v2xc(SPARC_OBJ *pSPARC, int DMnd, int ncol, double *Drho_x, double *Drho_y, double *Drho_z, double *v2xc)
{
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        for(int i = 0; i < DMnd*ncol; i++){
            double temp1 = (Drho_x[i] * pSPARC->lapcT[0] + Drho_y[i] * pSPARC->lapcT[1] + Drho_z[i] * pSPARC->lapcT[2]) * v2xc[i];
            double temp2 = (Drho_x[i] * pSPARC->lapcT[3] + Drho_y[i] * pSPARC->lapcT[4] + Drho_z[i] * pSPARC->lapcT[5]) * v2xc[i];
            double temp3 = (Drho_x[i] * pSPARC->lapcT[6] + Drho_y[i] * pSPARC->lapcT[7] + Drho_z[i] * pSPARC->lapcT[8]) * v2xc[i];
            Drho_x[i] = temp1;
            Drho_y[i] = temp2;
            Drho_z[i] = temp3;
        }
    } else {
        for(int i = 0; i < DMnd*ncol; i++){
            Drho_x[i] *= v2xc[i];
            Drho_y[i] *= v2xc[i];
            Drho_z[i] *= v2xc[i];
        }
    }
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int DMnd = pSPARC->Nd_d;
    double *rho = (double *)malloc(DMnd * sizeof(double) );

    // add core electron density if needed
    add_rho_core(pSPARC, pSPARC->electronDens, rho, 1);

    // TODO: Add ExcRho for spin-up and down
    for (int i = 0; i < DMnd; i++) {
        ExcRho[i] = pSPARC->e_xc[i] * rho[i];
    }

    free(rho);
#ifdef DEBUG
    double Exc = 0;
    for (int i = 0; i < pSPARC->Nd_d; i++)
        Exc += ExcRho[i];
    MPI_Allreduce(MPI_IN_PLACE, &Exc, 1, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    Exc *= pSPARC->dV;
    if (!rank) printf("Exchange correlation energy (without hybrid) from Exc energy density: %f\n", Exc);
#endif
}


/**
 * @brief  Calculate noncollinear xc potential 
 *
 * @param DMnd      number of local grid points
 * @param Vxc       exchange correlation of diagonal term (DMnd x 2)
 * @param mag       magnetization (DMnd x 4)
 * @param Vxc_nc    noncollinear xc potential (DMnd x 4)
 **/
void Calculate_Xcpotential_Noncollinear(SPARC_OBJ *pSPARC, int DMnd, double *Vxc, double *mag, double *Vxc_nc)
{
    double *magx = mag+DMnd, *magy = mag+2*DMnd, *magz = mag+3*DMnd, *magn = mag;
    double *Vxcup = Vxc, *Vxcdw = Vxc+DMnd;
    for (int i = 0; i < DMnd; i++) {
        double V11pV22 = Vxcup[i] + Vxcdw[i];
        double V11mV22 = Vxcup[i] - Vxcdw[i];
        double magnorm = magn[i];
        // V11
        Vxc_nc[i] = 0.5 * V11pV22;
        // V22
        Vxc_nc[i+DMnd] = 0.5 * V11pV22;
        // V12
        Vxc_nc[i+2*DMnd] = Vxc_nc[i+3*DMnd] = 0;

        if (magnorm > pSPARC->xc_magtol) {
            // V11
            Vxc_nc[i] += 0.5 * V11mV22 * magz[i] / magnorm;
            // V22
            Vxc_nc[i+DMnd] -= 0.5 * V11mV22 * magz[i] / magnorm;
            // real(V12)
            Vxc_nc[i+2*DMnd] = 0.5 * V11mV22 * magx[i] / magnorm;
            // imag(V12)
            Vxc_nc[i+3*DMnd] = -0.5 * V11mV22 * magy[i] / magnorm;
        }
    }
}