/**
 * @file    vdWDFstress.c
 * @brief   This file contains the functions for computing stress from vdF-DF1 and vdW-DF2 non-linear correlation.
 *
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

/*
 vdW-DF stress is deduced from Soler's paper formula (8). Make derivative on energy 
 E = 1/2*sum_{alpha}(int(theta_alpha*u_alpha)d(r_1)) 
    = 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1)*phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2))
 sigma_xy = dE/dF_xy = 1/2*sum_{alpha}sum_{beta}(intint(d(theta_alpha(r_1)*phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))/dF_xy)d(r_1)d(r_2))
    = 1/2*sum_{alpha}sum_{beta}(intint(d(theta_alpha(r_1))/dF_xy * phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2))
        + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * d(phi_{alphabeta}(|r_2 - r_1|))/dF_xy * theta_beta(r_2))d(r_1)d(r_2))
        + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * phi_{alphabeta}(|r_2 - r_1|) * d(theta_beta(r_2))/dF_xy)d(r_1)d(r_2))

 Structure of functions in vdWDFNonlinearCorre.c
 Calculate_XC_stress_vdWDF // Compute stress of vdWDF, called by Calculate_XC_stress in stress.c and Calculate_XC_pressure in pressure.c
 ├──vdWDF_stress_gradient
 └──vdWDF_stress_kernel
    └──interpolate_dKerneldK
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <errno.h> 

#include "isddft.h"
#include "tools.h"
#include "vdWDFinitialization.h"
#include "vdWDFstress.h"

/**
 * @brief compute the stress term containing grad(rho)
 the sum of 1st term and 3rd term of the formula above is
 1/2*sum_{alpha}sum_{beta}(intint(d(theta_alpha(r_1))/dF_xy * phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2)) 
        + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * phi_{alphabeta}(|r_2 - r_1|)*d(theta_beta(r_2))/dF_xy)d(r_1)d(r_2))
    = 1/2*sum_{alpha}(int(d(theta_alpha(r_1))/dF_xy * [sum_{beta}int(phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_2)])d(r_1)) 
        + 1/2*sum_{beta}(int(d(theta_beta(r_2))/dF_xy * [sum_{alpha}int(phi_{alphabeta}(|r_2 - r_1|)*theta_alpha(r_1))d(r_1)])d(r_2))
    = sum_{alpha}(int(d(theta_alpha(r_1))/dF_xy * u_alpha(r_1))d(r_1))
 That is the term this function compute:
 sum_{alpha}(int(d(theta_alpha(r_1))/dF_xy * u_alpha(r_1))d(r_1))
    = sum_{alpha}(int(u_alpha * dp_alpha/dq0 * dq0/d|gradrho| * d|gradrho|/dF_xy)d(r_1))
    = sum_{alpha}(int(u_alpha * dp_alpha/dq0 * dq0/d|gradrho| * [drho/dx*drho/dy) / |gradrho|)d(r_1))
 Divide it by nnr is necessary! (\Delta V/V)
 */
void vdWDF_stress_gradient(SPARC_OBJ *pSPARC, double *stressGrad) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nnr = pSPARC->Nd;
    int nqs = pSPARC->vdWDFnqs;
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    double *gradRhoLen = pSPARC->gradRhoLen;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double **Drho = pSPARC->Drho;
    double **dpdq0s = pSPARC->vdWDFdpdq0s; // used for solving potential
    double **u = pSPARC->vdWDFu;
    double *prefactor = (double*)malloc(sizeof(double)*DMnd);
    int igrid, q1;
    for (igrid = 0; igrid < DMnd; igrid++) {
        prefactor[igrid] = 0.0;
    }
    for (q1 = 0; q1 < nqs; q1++) {
        for (igrid = 0; igrid < DMnd; igrid++) {
            prefactor[igrid] += u[q1][igrid] * dpdq0s[q1][igrid] * dq0dgradrho[igrid];
        }
    }
    for (igrid = 0; igrid < DMnd; igrid++) {
        if (gradRhoLen[igrid] > 1e-15)
            prefactor[igrid] /= gradRhoLen[igrid];
        else
            prefactor[igrid] = 0.0;
    }

    int row, col;
    double localStressGrad[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (row = 0; row < 3; row++) {
        for (col = 0; col <= row; col++) {
            for (igrid = 0; igrid < DMnd; igrid++) {
                localStressGrad[3*row + col] -= prefactor[igrid] * Drho[row][igrid] * Drho[col][igrid];
            }
            localStressGrad[3*row + col] /= nnr;
            localStressGrad[3*col + row] = localStressGrad[3*row + col];
        }
    }
    MPI_Allreduce(localStressGrad, stressGrad, 9, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    free(prefactor);
}


void spin_vdWDF_stress_gradient(SPARC_OBJ *pSPARC, double *stressGrad) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nnr = pSPARC->Nd; // total number of grid points in the system
    int nqs = pSPARC->vdWDFnqs;
    int DMnd = pSPARC->Nd_d;
    double *gradRhoLen = pSPARC->gradRhoLen;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double **Drho = pSPARC->Drho;
    double **dpdq0s = pSPARC->vdWDFdpdq0s; // used for solving potential
    double **u = pSPARC->vdWDFu;
    double *prefactor = (double*)malloc(2*DMnd*sizeof(double));
    int i, igrid, q1;
    for (i = 0; i < 2*DMnd; i++) {
        prefactor[i] = 0.0;
    }
    for (q1 = 0; q1 < nqs; q1++) {
        for (i = 0; i < 2*DMnd; i++) {
            igrid = i%DMnd;
            prefactor[i] += u[q1][igrid] * dpdq0s[q1][igrid] * dq0dgradrho[i];
        }
    }
    for (i = 0; i < 2*DMnd; i++) {
        if (gradRhoLen[i] > 1e-15) 
            prefactor[i] /= gradRhoLen[i];
        else
            prefactor[i] = 0.0;
    }

    int row, col;
    double localStressGrad[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (row = 0; row < 3; row++) {
        for (col = 0; col <= row; col++) {
            for (i = 0; i < 2*DMnd; i++) {
                localStressGrad[3*row + col] -= prefactor[i] * Drho[row][i] * Drho[col][i];
            }
            localStressGrad[3*row + col] /= nnr;
            localStressGrad[3*col + row] = localStressGrad[3*row + col];
        }
    }
    MPI_Allreduce(localStressGrad, stressGrad, 9, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);
    free(prefactor);
}


void interpolate_dKerneldK(SPARC_OBJ *pSPARC, double **dKerneldLength) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nqs = pSPARC->vdWDFnqs;
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    double *reciLength = pSPARC->vdWDFreciLength;
    double **kernelPhi = pSPARC->vdWDFkernelPhi;
    double **d2Phidk2 = pSPARC->vdWDFd2Phidk2;
    double dk = pSPARC->vdWDFdk;
    int rigrid, timeOfdk, q1, q2, qpair;
    double a, b, adk, bdk, cdk, ddk;
    double largestLength = pSPARC->vdWDFnrpoints * dk;
    for (rigrid = 0; rigrid < DMnd; rigrid++) {
        if (reciLength[rigrid] > largestLength)
            continue;
        timeOfdk = (int)floor(reciLength[rigrid] / dk) + 1;
        a = (dk*((timeOfdk - 1.0) + 1.0) - reciLength[rigrid]) / dk;
        b = (reciLength[rigrid] - dk*(timeOfdk - 1.0)) / dk;
        adk = -1.0 / dk;
        bdk = 1.0 / dk;
        cdk = -((3*a*a - 1.0)/6.0) * dk;
        ddk = ((3*b*b -1.0)/6.0) * dk;
        for (q1 = 0; q1 < nqs; q1++) {
            for (q2 = 0; q2 < q1 + 1; q2++) {
                qpair = kernel_label(q1, q2, nqs);
                dKerneldLength[qpair][rigrid] = adk*kernelPhi[qpair][timeOfdk - 1] + bdk*kernelPhi[qpair][timeOfdk] + 
                    (cdk*d2Phidk2[qpair][timeOfdk - 1] + ddk*d2Phidk2[qpair][timeOfdk]);
            }
        }
    }
}

/**
 * @brief compute the stress term containing derivative of kernel function
 The function computes the 2nd term of the stress formula above
 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * d(phi_{alphabeta}(|r_2 - r_1|))/dF_xy * theta_beta(r_2))d(r_1)d(r_2)) 
  after Fourier Transform
    = 1/2*sum_{alpha}sum_{beta}(int(conj(theta_alpha(k_1)) * d(phi_{alphabeta}(|k_1|))/d|k_1| * d|k_1|/dF_xy * theta_beta(k_1))d(k_1))
 */

void vdWDF_stress_kernel(SPARC_OBJ *pSPARC, double *stressKernel) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nqs = pSPARC->vdWDFnqs;
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    double _Complex **thetaFTs = pSPARC->vdWDFthetaFTs;
    double **reciLatticeGrid = pSPARC->vdWDFreciLatticeGrid;
    double *reciLength = pSPARC->vdWDFreciLength;
    int numberKernel = nqs*(nqs - 1)/2 + nqs;
    double largestLength = pSPARC->vdWDFnrpoints * pSPARC->vdWDFdk;
    double **dKerneldLength = (double**)malloc(sizeof(double*)*numberKernel);
    int q1, q2;
    for (q1 = 0; q1 < numberKernel; q1++) {
        dKerneldLength[q1] = (double*)malloc(sizeof(double)*DMnd);
    }
    interpolate_dKerneldK(pSPARC, dKerneldLength);
    int rigrid, qpair;

    double _Complex *thetaMdKerneldkMconjtheta = (double _Complex*)malloc(sizeof(double _Complex)*DMnd);
    for (rigrid = 0; rigrid < DMnd; rigrid++) {
        thetaMdKerneldkMconjtheta[rigrid] = 0.0;
    }

    double _Complex q1riThetaFTs, q2riThetaFTs;
    double dKernelValue;
    for (q2 = 0; q2 < nqs; q2++) {
        for (q1 = 0; q1 < nqs; q1++) {
            qpair = kernel_label(q1, q2, nqs);
            for (rigrid = 0; rigrid < DMnd; rigrid++) {
                q1riThetaFTs = thetaFTs[q1][rigrid];
                q2riThetaFTs = thetaFTs[q2][rigrid];
                dKernelValue = dKerneldLength[qpair][rigrid];
                thetaMdKerneldkMconjtheta[rigrid] += q1riThetaFTs*dKernelValue*conj(q2riThetaFTs);
            }
        }
    }

    int row, col;
    double localStressKernel[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (row = 0; row < 3; row++) {
        for (col = 0; col <= row; col++) {
            for (rigrid = 0; rigrid < DMnd; rigrid++) {
                if ((reciLength[rigrid] > largestLength) || (reciLength[rigrid] < 1e-15))
                    continue;
                localStressKernel[3*row + col] -= 0.5 * creal(thetaMdKerneldkMconjtheta[rigrid]) * 
                    reciLatticeGrid[row][rigrid] * reciLatticeGrid[col][rigrid] / reciLength[rigrid];
            }
            localStressKernel[3*col + row] = localStressKernel[3*row + col];
        }
    }
    MPI_Allreduce(localStressKernel, stressKernel, 9, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi);

    for (q1 = 0; q1 < numberKernel; q1++) {
        free(dKerneldLength[q1]);
    }
    free(dKerneldLength);
    free(thetaMdKerneldkMconjtheta);
}

// The functions below are for calculating vdWDF stress
void Calculate_XC_stress_vdWDF(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    double stress[9];
    double stressGradArray[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
    double *stressGrad = stressGradArray;
    if (pSPARC->spin_typ == 0)
        vdWDF_stress_gradient(pSPARC, stressGrad); // compute the stress term containing grad(rho)
    else if (pSPARC->spin_typ == 1)
        spin_vdWDF_stress_gradient(pSPARC, stressGrad); // compute the stress term containing grad(rho)
    else if (pSPARC->spin_typ == 2)
        assert(pSPARC->Nspin <= 2 && pSPARC->spin_typ <= 1);

    double stressKernelArray[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
    double *stressKernel = stressKernelArray;
    vdWDF_stress_kernel(pSPARC, stressKernel); // compute the stress term containing derivative of kernel functions
    int sigmaDir;
    for (sigmaDir = 0; sigmaDir < 9; sigmaDir++) {
        stress[sigmaDir] = stressGradArray[sigmaDir] + stressKernelArray[sigmaDir]; // comparing with QE, SPARC has opposite sign of stress
    }
    #ifdef DEBUG
    if (rank == 0) {
        printf("\nvdW-DF stress, which is a part of XC stress (GPa):\n");
        for (int row = 0; row < 3; row++) {
            printf("%18.14f %18.14f %18.14f\n", 
                stress[3*row + 0]*29421.02648438959, stress[3*row + 1]*29421.02648438959, stress[3*row + 2]*29421.02648438959);
        }
    }
    #endif

    // add vdW-DF stress to exchange-correlation stress
    if (pSPARC->Calc_stress == 1) {
        pSPARC->stress_xc[0] += stress[0];
        pSPARC->stress_xc[1] += stress[1];
        pSPARC->stress_xc[2] += stress[2];
        pSPARC->stress_xc[3] += stress[4]; // (2,2)
        pSPARC->stress_xc[4] += stress[5]; // (2,3)
        pSPARC->stress_xc[5] += stress[8]; // (3,3)
    }
    if (pSPARC->Calc_pres == 1) {
        pSPARC->pres_xc += (stress[0] + stress[4] + stress[8]) * (pSPARC->Jacbdet*pSPARC->range_x*pSPARC->range_y*pSPARC->range_z);
    }
}