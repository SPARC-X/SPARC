/**
 * @file    vdWDFnonlinearCorre.c
 * @brief   This file contains the functions for vdF-DF1 and vdW-DF2 functional.
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

#include "isddft.h"
#include "initialization.h"
#include "tools.h"
#include "parallelization.h"
#include "vdWDFnonlinearCorre.h"
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
/** FFT routines **/
#ifdef USE_MKL
    #include "mkl_cdft.h"
#endif
#ifdef USE_FFTW
    #include <fftw3-mpi.h>
#endif


/*
 Structure of functions in vdWDFNonlinearCorre.c

 Calculate_E_V_vdWDF // Compute vdwaDF energy and potential, called by Calculate_Vxc_GGA in exchangeCorrelation.c
 ├──get_q0_Grid
 |  ├──pw
 |  └──saturate_q
 ├──spline_interpolation
 ├──compute_Gvectors
 ├──theta_generate_FT
 |  └──parallel_FFT
 |     ├──compose_FFTInput
 |     └──compose_FFTOutput
 ├──vdWDF_energy
 |  ├──interpolate_kernel
 |  └──kernel_label
 ├──u_generate_iFT
 |  └──parallel_iFFT
 |     ├──compose_FFTInput (the same two functions in parallel_FFT)
 |     └──compose_FFTOutput
 └──vdWDF_potential

*/

#define KF(rho) (pow(3.0*M_PI*M_PI * rho, 1.0/3.0))
#define FS(s) (1.0 - Zab*s*s/9.0)
#define DFSDS(s) ((-2.0/9.0) * s*Zab)
#define DSDRHO(s, rho) (-s*(DKFDRHO(rho)/kfResult + 1.0/rho))
#define DKFDRHO(rho) ((1.0/3.0)*kfResult/rho)
#define DSDGRADRHO(rho) (0.5 / (kfResult*rho))
#define INDEX1D(i, j, k) (i + pSPARC->Nx*j + pSPARC->Nx*pSPARC->Ny*k)

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)>(y)?(y):(x))

/*
Functions below are related to computing q0 and p (spline interpolation) on the grid.
*/
double pw(double rs, double *dq0drhoPosi) { // exchange-correlation energy of LDA_PW
    // parameters of LDA_PW
    double a =0.031091;
    double a1=0.21370;
    double b1=7.5957; double b2=3.5876; double b3=1.6382; double b4=0.49294;
    double rs12 = sqrt(rs); double rs32 = rs*rs12; double rs2 = rs*rs;
    double om   = 2.0*a*(b1*rs12 + b2*rs + b3*rs32 + b4*rs2);
    double dom  = 2.0*a*(0.5*b1*rs12 + b2*rs + 1.5*b3*rs32 + 2.0*b4*rs2);
    double olog = log(1.0 + 1.0/om);
    double ecLDA_PW = -2.0*a*(1.0 + a1*rs)*olog; // energy on every point
    double dq0drho  = -2.0*a*(1.0 + 2.0/3.0*a1*rs)*olog - 2.0/3.0*a*(1.0 + a1*rs)*dom/(om*(om + 1.0));
    *dq0drhoPosi = dq0drho;
    return ecLDA_PW;
}

void saturate_q(double qp, double qCut, double *saturateq0dq0dq) { // (5) of Soler
    int mc = 12;
    double eexp = 0.0;
    double dq0dq = 0.0;
    double qDivqCut = qp/qCut;
    double qDivqCutPower = 1.0;
    for (int m = 1; m <= mc; m++) {
        dq0dq += qDivqCutPower;
        qDivqCutPower *= qDivqCut;
        eexp += qDivqCutPower/m;
    }
    saturateq0dq0dq[0] = qCut*(1.0 - exp(-eexp));
    saturateq0dq0dq[1] = dq0dq*exp(-eexp);
}


void get_q0_Grid(SPARC_OBJ *pSPARC, double* rho) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double Zab = pSPARC->vdWDFZab;
    double *gradRhoLen = pSPARC->gradRhoLen;
    double *q0 = pSPARC->vdWDFq0;
    double *dq0drho = pSPARC->vdWDFdq0drho;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double qCut = pSPARC->vdWDFqmesh[nqs - 1];
    double qMin = pSPARC->vdWDFqmesh[0];

    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, pSPARC->Drho[0], 0, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, pSPARC->Drho[1], 1, pSPARC->dmcomm_phi);
    Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, rho, pSPARC->Drho[2], 2, pSPARC->dmcomm_phi); // optimize this part? in exchange and linear correlation function, it has be solved
    double **Drho = pSPARC->Drho;
    double *Drho_x = pSPARC->Drho[0];
    double *Drho_y = pSPARC->Drho[1];
    double *Drho_z = pSPARC->Drho[2];


    int igrid;
    double rs; // used to compute epsilon_xc^0, exchange-correlation energy of LDA
    double s; // Dion's paper, (12) 2nd term
    double ecLDAPW;
    double kfResult, FsResult, qp, dqxdrho;
    double saturate[2]; double *saturateq0dq0dq = saturate; // first: q0; second: dq0dq

    for (igrid = 0; igrid < DMnd; igrid++) {
        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
            nonCart2Cart_grad(pSPARC, &(Drho_x[igrid]), &(Drho_y[igrid]), &(Drho_z[igrid])); // transfer the gradient to cartesian direction
        }
        gradRhoLen[igrid] = sqrt(Drho_x[igrid]*Drho_x[igrid] + Drho_y[igrid]*Drho_y[igrid] + Drho_z[igrid]*Drho_z[igrid]);

        rs = pow(3.0/(4.0*M_PI*rho[igrid]), 1.0/3.0);
        kfResult = KF(rho[igrid]);
        s = gradRhoLen[igrid] / (2.0*kfResult*rho[igrid]);
        ecLDAPW = pw(rs, dq0drho + igrid); // in this step, a part of dq0drho[igrid] is computed
        FsResult = FS(s);
        qp = -4.0*M_PI/3.0*ecLDAPW + kfResult*FsResult; // energy ratio on every point
        saturate_q(qp, qCut, saturateq0dq0dq); // modify q into [qMin, qCut]
        q0[igrid] = saturate[0]>qMin? saturate[0]:qMin;
        dqxdrho = DKFDRHO(rho[igrid])*FsResult + kfResult*DFSDS(s)*DSDRHO(s, rho[igrid]);
        dq0drho[igrid] = saturate[1]*rho[igrid] * (-4.0*M_PI/3.0*(dq0drho[igrid] - ecLDAPW)/rho[igrid] + dqxdrho);
        dq0dgradrho[igrid] = saturate[1]*rho[igrid]*kfResult*DFSDS(s)*DSDGRADRHO(rho[igrid]);
    }
    // // verify the correctness of result
    // if ((pSPARC->countSCF == 0) && (rank == size - 1)) {
    //     printf("rank %d, (%d, %d, %d)-(%d, %d, %d), q0[0] %.6e, q0[DMnd - 1] %.6e\n",
    //      rank, pSPARC->DMVertices[0], pSPARC->DMVertices[2], pSPARC->DMVertices[4], pSPARC->DMVertices[1], pSPARC->DMVertices[3], pSPARC->DMVertices[5],
    //      q0[0], q0[DMnd - 1]);
    // }
}

/**
 * @brief get the component of "model energy ratios" on every grid point, based on the computed q0. The ps array is its output
 */
void spline_interpolation(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double **ps = pSPARC->vdWDFps;
    double **dpdq0s = pSPARC->vdWDFdpdq0s; // used for solving potential
    double *qmesh = pSPARC->vdWDFqmesh;
    double *q0 = pSPARC->vdWDFq0;
    double **d2ydx2 = pSPARC->vdWDFd2Splineydx2;
    // at here ps[q1][i] is the component coefficient of q1th model energy ratio on ith grid point. In m, it is ps(i, q1)
    int igrid, lowerBound, upperBound, idx, q1;
    double dq, a, b, c, d, e, f;
    for (igrid = 0; igrid < DMnd; igrid++) { // the loop to be optimized
        lowerBound = 0; upperBound = nqs - 1;
        while (upperBound - lowerBound > 1) {
            idx = (upperBound + lowerBound)/2;
            if (q0[igrid] > qmesh[idx]) {
                lowerBound = idx;
            }
            else {
                upperBound = idx;
            }
        }
        dq = qmesh[upperBound] - qmesh[lowerBound];
        a = (qmesh[upperBound] - q0[igrid]) / dq;
        b = (q0[igrid] - qmesh[lowerBound]) / dq;
        c = (a*a*a - a) * (dq*dq) / 6.0;
        d = (b*b*b - b) * (dq*dq) / 6.0;
        e = (3.0*a*a - 1.0)*dq / 6.0;
        f = (3.0*b*b - 1.0)*dq / 6.0;
        for (q1 = 0; q1 < nqs; q1++) {
            double y[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; //20 at here is nqs.
            y[q1] = 1.0;
            ps[q1][igrid] = a*y[lowerBound] + b*y[upperBound];
            ps[q1][igrid] += c*d2ydx2[q1][lowerBound];
            ps[q1][igrid] += d*d2ydx2[q1][upperBound];
            dpdq0s[q1][igrid] = (y[upperBound] - y[lowerBound])/dq - e*d2ydx2[q1][lowerBound] + f*d2ydx2[q1][upperBound];
        }
    }
    // verify the correctness of result
    #ifdef DEBUG
        if ((pSPARC->countSCF == 0) && (rank == size - 1)) {
            printf("vdWDF: rank %d, (%d, %d, %d)-(%d, %d, %d), in 1st SCF ps[2][DMnd - 1] %.6e, dpdq0s[2][DMnd - 1] %.6e\n",
             rank, pSPARC->DMVertices[0], pSPARC->DMVertices[2], pSPARC->DMVertices[4], pSPARC->DMVertices[1], pSPARC->DMVertices[3], pSPARC->DMVertices[5],
             ps[5][DMnd - 1], dpdq0s[5][DMnd - 1]);
        }
    #endif
}
/*
Functions above are related to computing q0 and p (spline interpolation) on the grid.
*/

/*
Functions below are related to parallel FFT
*/
// compute the coordinates of G-vectors in reciprocal space. They are sum of integres multiplying three primary reci lattice vectors
void compute_Gvectors(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;

    // generate the 3D grid in reciprocal space
    // Firstly, compute reciprocal lattice vectors
    int row, col, i, j, k;
    for (col = 0; col < 3; col++) 
        pSPARC->lattice[0*3 + col] = pSPARC->LatUVec[0*3 + col] * pSPARC->range_x;
    for (col = 0; col < 3; col++) 
        pSPARC->lattice[1*3 + col] = pSPARC->LatUVec[1*3 + col] * pSPARC->range_y;
    for (col = 0; col < 3; col++) 
        pSPARC->lattice[2*3 + col] = pSPARC->LatUVec[2*3 + col] * pSPARC->range_z;
    double detLattice = 0.0;
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
            for(k = 0; k < 3; k++){
                if(i != j && j != k && k != i)
                    detLattice += ((i - j) * (j - k) * (k - i)/2) * pSPARC->lattice[3 * i] * pSPARC->lattice[3 * j + 1] * pSPARC->lattice[3 * k + 2];
            }
        }
    }
    pSPARC->detLattice = detLattice;
    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++){
           pSPARC->reciLattice[3*j + i] = (pSPARC->lattice[3 * ((j+1) % 3) + (i+1) % 3] * pSPARC->lattice[3 * ((j+2) % 3) + (i+2) % 3] - pSPARC->lattice[3 * ((j+1) % 3) + (i+2) % 3] * pSPARC->lattice[3 * ((j+2) % 3) + (i+1) % 3])/detLattice*(2*M_PI);
        }
    }
    // Secondly, compose the index of these lattice vectors and make a permutation. This part is moved to initialization functions
    // Thirdly, compute the coordinates of reciprocal lattice vectors, and the length of them
    int igrid, rigrid; 
    double **reciLatticeGrid = pSPARC->vdWDFreciLatticeGrid;
    for (row = 0; row < 3; row++) { // cartesian direction
        for (rigrid = 0; rigrid < DMnd; rigrid++) { // reciprocal grid points // 000 100 200 ... 010 110 210 ... 001 101 201 ... 011 111 211 ...
            reciLatticeGrid[row][rigrid] = pSPARC->timeReciLattice[0][rigrid]*pSPARC->reciLattice[0 + row] 
            + pSPARC->timeReciLattice[1][rigrid]*pSPARC->reciLattice[3 + row] 
            + pSPARC->timeReciLattice[2][rigrid]*pSPARC->reciLattice[6 + row];
        }
    }
    double *reciLength = pSPARC->vdWDFreciLength;
    double largestLength = pSPARC->vdWDFnrpoints * pSPARC->vdWDFdk;
    int signReciPointFurther = 0;
    for (rigrid = 0; rigrid < DMnd; rigrid++) {
        reciLength[rigrid] = sqrt(reciLatticeGrid[0][rigrid]*reciLatticeGrid[0][rigrid] 
            + reciLatticeGrid[1][rigrid]*reciLatticeGrid[1][rigrid] + reciLatticeGrid[2][rigrid]*reciLatticeGrid[2][rigrid]);
        if (reciLength[rigrid] > largestLength) {
            signReciPointFurther = 1;
        }
    }
    if ((signReciPointFurther == 1) && (rank == 0)) printf("WARNING: there is reciprocal grid point further than largest allowed distance (%.e) from center!\n", largestLength); // It can be optimized

    // For debugging
    #ifdef DEBUG
    if ((pSPARC->countSCF == 0) && (rank == size - 1)) { // only output result in 1st step
        printf("vdWDF: rank %d, 2nd reci point (%d %d %d) %12.6e %12.6e %12.6e\n", rank, pSPARC->timeReciLattice[0][1], pSPARC->timeReciLattice[1][1], pSPARC->timeReciLattice[2][1],
            reciLatticeGrid[0][1], reciLatticeGrid[1][1], reciLatticeGrid[2][1]);
    }
    #endif
}

// Not sure whether the dividing method of SPARC on z axis can always be the same as the dividing method of DftiGetValueDM
// the usage of this function is to re-divide data on z axis for fitting the dividing of parallel FFT functions
// For example, the grid has 22 nodes on z direction, and 3 processors are on z axis (0, 0, z). If SPARC divides it into 8+8+7,
// but DftiGetValueDM divides it into 7+8+8, then it is necessary to reorganize data
void compose_FFTInput(int *gridsizes, int length, int *start, double _Complex *FFTInput, int* allStartK, double _Complex *inputData, MPI_Comm zAxisComm) {
    int zAxisrank; int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);
    MPI_Comm_rank(zAxisComm, &zAxisrank);
    int numElesInGlobalPlane = gridsizes[0]*gridsizes[1];
    int *zPlaneInputStart = start;
    int zP;

    int *diffStart = (int*)malloc(sizeof(int)*(sizeZcomm + 1));
    diffStart[0] = 0;
    for (zP = 1; zP < sizeZcomm; zP++) {
    	diffStart[zP] = allStartK[zP] - zPlaneInputStart[zP];
    }
    diffStart[sizeZcomm] = 0;
    int planeInputStart, FFTInputStart, selfLength; // transfer data in inputDataRealSpace to FFTInput. The relationship can be shown by a figure.
    planeInputStart = max(0, diffStart[zAxisrank]*numElesInGlobalPlane);
    FFTInputStart = max(0, -diffStart[zAxisrank]*numElesInGlobalPlane);
    selfLength = length - max(0, diffStart[zAxisrank]) - max(0, -diffStart[zAxisrank + 1]);
    memcpy(FFTInput + FFTInputStart, inputData + planeInputStart, sizeof(double _Complex)*numElesInGlobalPlane*selfLength);
    // printf("zAxisrank %d, FFTInputStart %d, planeInputStart %d, selfLength %d\n", zAxisrank, FFTInputStart, planeInputStart, selfLength);
    
    for (zP = 1; zP < sizeZcomm; zP++) { // attention: not start from 0.
        if (diffStart[zP] > 0) { // processor[rank] send some plane data into processor[rank - 1]
            if (zAxisrank == zP) {
                MPI_Send(inputData, numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP - 1, 0, zAxisComm);
            }
            if (zAxisrank == zP - 1) {
                MPI_Status stat;
                MPI_Recv(FFTInput + numElesInGlobalPlane*(FFTInputStart + selfLength), numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP, 0, zAxisComm, &stat);
            }
        }
        else if (diffStart[zP] < 0) { // processor[rank] receive some plane data from processor[rank - 1]
            if (zAxisrank == zP) {
                MPI_Status stat;
                MPI_Recv(FFTInput, -numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP - 1, 0, zAxisComm, &stat);
            }
            if (zAxisrank == zP - 1) {
                MPI_Send(inputData + numElesInGlobalPlane*(planeInputStart + selfLength), -numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP, 0, zAxisComm);
            }
        }
    }
    free(diffStart);
}

// Like the usage of compose_FFTInput
void compose_FFTOutput(int *gridsizes, int length, int *start, double _Complex *FFTOutput, int* allStartK, int* allLengthK, double _Complex *outputData, MPI_Comm zAxisComm) {
    int zAxisrank; int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);
    MPI_Comm_rank(zAxisComm, &zAxisrank);
    int numElesInGlobalPlane = gridsizes[0]*gridsizes[1];
    int *zPlaneOutputStart = start;
    int zP;

    int *diffStart = (int*)malloc(sizeof(int)*(sizeZcomm + 1));
    diffStart[0] = 0;
    for (zP = 1; zP < sizeZcomm; zP++) {
    	diffStart[zP] = allStartK[zP] - zPlaneOutputStart[zP];
    }
    diffStart[sizeZcomm] = 0;
    int planeOutputStart, FFTOutputStart, FFTLength; // transfer data in planeInput to FFTInput. The relationship can be shown by a figure.
    planeOutputStart = max(0, diffStart[zAxisrank]*numElesInGlobalPlane);
    FFTOutputStart = max(0, -diffStart[zAxisrank]*numElesInGlobalPlane);
    FFTLength = allLengthK[zAxisrank] + min(0, diffStart[zAxisrank]) + min(0, -diffStart[zAxisrank + 1]);
    // printf("zAxisrank %d, FFTOutputStart %d, planeOutputStart %d, FFTLength %d\n", zAxisrank, FFTOutputStart, planeOutputStart, FFTLength);
    memcpy(outputData + planeOutputStart, FFTOutput + FFTOutputStart, sizeof(double _Complex)*numElesInGlobalPlane*FFTLength);
    for (zP = 1; zP < sizeZcomm; zP++) { // attention: not start from 0.
        if (diffStart[zP] > 0) { // processor[rank] send some plane data into processor[rank - 1]
            if (zAxisrank == zP - 1) {
                MPI_Send(FFTOutput + numElesInGlobalPlane*FFTLength, numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP, 0, zAxisComm);
            }
            if (zAxisrank == zP) {
                MPI_Status stat;
                MPI_Recv(outputData, numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP - 1, 0, zAxisComm, &stat);
            }
        }
        else if (diffStart[zP] < 0) { // processor[rank] receive some plane data from processor[rank - 1]
            if (zAxisrank == zP - 1) {
                MPI_Status stat;
                MPI_Recv(outputData + numElesInGlobalPlane*FFTLength, -numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP, 0, zAxisComm, &stat);
            }
            if (zAxisrank == zP) {
                MPI_Send(FFTOutput, -numElesInGlobalPlane*diffStart[zP], MPI_C_DOUBLE_COMPLEX, 
                    zP - 1, 0, zAxisComm);
            }
        }
    }

    free(diffStart);
}

void parallel_FFT(double _Complex *inputDataRealSpace, double _Complex *outputDataReciSpace, int *gridsizes, int *zAxis_Starts, int DMnz, int q, MPI_Comm zAxisComm) {
    #if defined(USE_MKL) // use MKL CDFT
    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);

    // initializa parallel FFT
    DFTI_DESCRIPTOR_DM_HANDLE desc = NULL;
    MKL_LONG localArrayLength, lengthK, startK;
    MKL_LONG dim_sizes[3] = {gridsizes[2], gridsizes[1], gridsizes[0]};
    DftiCreateDescriptorDM(zAxisComm, &desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &localArrayLength);
    DftiGetValueDM(desc, CDFT_LOCAL_NX, &lengthK);
    DftiGetValueDM(desc, CDFT_LOCAL_X_START, &startK);
    int lengthKint = lengthK; int startKint = startK;

    // compose FFT input to prevent the inconsistency division on Z axis 
    double _Complex *FFTInput = (double _Complex*)malloc(sizeof(double _Complex)*(localArrayLength));
    double _Complex *FFTOutput = (double _Complex*)malloc(sizeof(double _Complex)*(localArrayLength));
    int *allStartK = (int*)malloc(sizeof(int)*sizeZcomm);
    int *allLengthK = (int*)malloc(sizeof(int)*sizeZcomm);
    MPI_Allgather(&startKint, 1, MPI_INT, allStartK, 1, MPI_INT, zAxisComm);
    MPI_Allgather(&lengthKint, 1, MPI_INT, allLengthK, 1, MPI_INT, zAxisComm);
    compose_FFTInput(gridsizes, DMnz, zAxis_Starts, FFTInput, allStartK, inputDataRealSpace, zAxisComm);
    
    // make parallel FFT
    /* Set that we want out-of-place transform (default is DFTI_INPLACE) */
    DftiSetValueDM(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    /* Commit descriptor, calculate FFT, free descriptor */
    DftiCommitDescriptorDM(desc);
    DftiComputeForwardDM(desc, FFTInput, FFTOutput);

    // compose FFT output to prevent the inconsistency division on Z axis 
    compose_FFTOutput(gridsizes, DMnz, zAxis_Starts, FFTOutput, allStartK, allLengthK, outputDataReciSpace, zAxisComm);

    DftiFreeDescriptorDM(&desc);
    free(FFTInput);
    free(FFTOutput);
    free(allStartK);
    free(allLengthK);

    #elif defined(USE_FFTW) // use FFTW if MKL is not used

    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);

    // initializa parallel FFT
    const ptrdiff_t N0 = gridsizes[2], N1 = gridsizes[1], N2 = gridsizes[0]; // N0: z; N1:Y; N2:x
    fftw_plan plan;
    fftw_complex *FFTInput, *FFTOutput;
    ptrdiff_t localArrayLength, lengthK, startK;
    fftw_mpi_init();
    /* get local data size and allocate */
    localArrayLength = fftw_mpi_local_size_3d(N0, N1, N2, zAxisComm, &lengthK, &startK);
    FFTInput = fftw_alloc_complex(localArrayLength);
    FFTOutput = fftw_alloc_complex(localArrayLength);
    // compose FFT input to prevent the inconsistency division on Z axis 
    int lengthKint = lengthK; int startKint = startK;
    int *allStartK = (int*)malloc(sizeof(int)*sizeZcomm);
    int *allLengthK = (int*)malloc(sizeof(int)*sizeZcomm);
    MPI_Allgather(&startKint, 1, MPI_INT, allStartK, 1, MPI_INT, zAxisComm);
    MPI_Allgather(&lengthKint, 1, MPI_INT, allLengthK, 1, MPI_INT, zAxisComm);
    // printf("rank %d, DMnz %d\n", rank, DMnz);
    // if (rank == 0) {
    //     for (int i = 0; i < sizeZcomm; i++) {
    //         printf("rank %d, zAxis_Starts %d, startK %d, lengthK %d\n", i, zAxis_Starts[i], allStartK[i], allLengthK[i]);
    //     }
    // }
    compose_FFTInput(gridsizes, DMnz, zAxis_Starts, FFTInput, allStartK, inputDataRealSpace, zAxisComm);
    /* create plan for in-place forward DFT */
    plan = fftw_mpi_plan_dft_3d(N0, N1, N2, FFTInput, FFTOutput, zAxisComm, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    // compose FFT output to prevent the inconsistency division on Z axis 
    compose_FFTOutput(gridsizes, DMnz, zAxis_Starts, FFTOutput, allStartK, allLengthK, outputDataReciSpace, zAxisComm);

    fftw_free(FFTInput);
    fftw_free(FFTOutput);
    fftw_destroy_plan(plan);
    fftw_mpi_cleanup();

    free(allStartK);
    free(allLengthK);
    #endif 
}
/*
Functions above are related to parallel FFT
*/

/*
Functions below are related to generating thetas (ps*rho) and integrating energy.
*/
void theta_generate_FT(SPARC_OBJ *pSPARC, double* rho) { // solve thetas (p(q)*rho) in reciprocal space
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double **ps = pSPARC->vdWDFps;

    // compute theta vectors in real space, then FFT
    int nnr = pSPARC->Nd;
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int phiDims[3];
    phiDims[0] = pSPARC->npNdx_phi;
    phiDims[1] = pSPARC->npNdy_phi;
    phiDims[2] = pSPARC->npNdz_phi;
    int zAxisDims[3];
    zAxisDims[0] = 1;
    zAxisDims[1] = 1;
    zAxisDims[2] = pSPARC->npNdz_phi;
    double *theta = (double*)malloc(sizeof(double)*DMnd); // save theta functions in real space
    double *thetaFTreal = (double*)malloc(sizeof(double)*DMnd);
    double *thetaFTimag = (double*)malloc(sizeof(double)*DMnd);
    double _Complex **thetaFTs = pSPARC->vdWDFthetaFTs;
    double *gatheredTheta;
    double _Complex *gatheredThetaCompl;
    double _Complex *gatheredThetaFFT;
    double *gatheredThetaFFT_real;
    double *gatheredThetaFFT_imag;
    int igrid, rigrid; 
    if (pSPARC->zAxisComm != MPI_COMM_NULL) { // the processors on z axis (0, 0, z) receive the theta vectors from all other processors (x, y, z) on its z plane
        // printf("rank %d. pSPARC->zAxisComm not NULL!\n", rank);
        gatheredTheta = (double*)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * DMnz);
        gatheredThetaCompl = (double _Complex*)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * DMnz);
        gatheredThetaFFT = (double _Complex*)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * DMnz);
        gatheredThetaFFT_real = (double*)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * DMnz);
        gatheredThetaFFT_imag = (double*)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * DMnz);
        assert(gatheredTheta != NULL);
        assert(gatheredThetaCompl != NULL);
        assert(gatheredThetaFFT != NULL);
        assert(gatheredThetaFFT_real != NULL);
        assert(gatheredThetaFFT_imag != NULL);
    }
    int *zAxis_Starts = (int*)malloc(sizeof(int) * (pSPARC->npNdz_phi));

    for (int z = 0; z < pSPARC->npNdz_phi; z++) {
        zAxis_Starts[z] = block_decompose_nstart(gridsizes[2], pSPARC->npNdz_phi, z);
    }
    for (int q1 = 0; q1 < nqs; q1++) {
        for (igrid = 0; igrid < DMnd; igrid++) {
            theta[igrid] = ps[q1][igrid]*rho[igrid]; // compute theta vectors, ps*rho
        }
        D2D(&(pSPARC->gatherThetasSender), &(pSPARC->gatherThetasRecvr), gridsizes, 
            pSPARC->DMVertices, theta, 
            pSPARC->zAxisVertices, gatheredTheta, 
            pSPARC->dmcomm_phi, phiDims, pSPARC->zAxisComm, zAxisDims, MPI_COMM_WORLD);
        // printf("rank %d. D2D for q1 %d finished!\n", rank, q1);
        if (pSPARC->zAxisComm != MPI_COMM_NULL) { // the processors on z axis (0, 0, z) receive the theta vectors from all other processors (x, y, z) on its z plane
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * DMnz; rigrid++) {
                gatheredThetaCompl[rigrid] = gatheredTheta[rigrid];
            }
            parallel_FFT(gatheredThetaCompl, gatheredThetaFFT, gridsizes, zAxis_Starts, DMnz, q1, pSPARC->zAxisComm);
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * DMnz; rigrid++) {
            	gatheredThetaFFT_real[rigrid] = creal(gatheredThetaFFT[rigrid]);
            	gatheredThetaFFT_imag[rigrid] = cimag(gatheredThetaFFT[rigrid]);
            }
        }
        D2D(&(pSPARC->scatterThetasSender), &(pSPARC->scatterThetasRecvr), gridsizes, // scatter the real part of theta results from the processors on z axis (0, 0, z) to all other processors
            pSPARC->zAxisVertices, gatheredThetaFFT_real, 
            pSPARC->DMVertices, thetaFTreal, 
            pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi, phiDims, MPI_COMM_WORLD);
        D2D(&(pSPARC->scatterThetasSender), &(pSPARC->scatterThetasRecvr), gridsizes, // scatter the imaginary part of theta results from the processors on z axis (0, 0, z) to all other processors
            pSPARC->zAxisVertices, gatheredThetaFFT_imag, 
            pSPARC->DMVertices, thetaFTimag, 
            pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi, phiDims, MPI_COMM_WORLD);
        for (rigrid = 0; rigrid < DMnd; rigrid++) {
        	thetaFTs[q1][rigrid] = thetaFTreal[rigrid] + thetaFTimag[rigrid]*I;
            thetaFTs[q1][rigrid] /= nnr;
        }
        // if ((pSPARC->countSCF == 0) && (q1 == 5) && (rank == size - 1)) { // only output result in 1st step
        // 	int localIndex1D = domain_index1D(1, 1, 1, DMnx, DMny, DMnz);
    	//     printf("rank %d, in 1st SCF thetaFTs[5][(1, 1, 1)]=globalthetaFTs[5][(%d, %d, %d)] = %.5e + i%.5e\n", rank, 
    	//     	pSPARC->DMVertices[0] + 1, pSPARC->DMVertices[2] + 1, pSPARC->DMVertices[4] + 1,
    	//         creal(thetaFTs[q1][localIndex1D]), cimag(thetaFTs[q1][localIndex1D]));
    	// }
    }

    free(theta);
    free(zAxis_Starts);
    free(thetaFTreal);
    free(thetaFTimag);
    if (pSPARC->zAxisComm != MPI_COMM_NULL) { // the processors on z axis (0, 0, z) receive the theta vectors from all other processors (x, y, z) on its z plane
        free(gatheredTheta);
        free(gatheredThetaCompl);
        free(gatheredThetaFFT);
        free(gatheredThetaFFT_real);
        free(gatheredThetaFFT_imag);
    }
}

/**
 * @brief compute the value of all kernel functions (210 functions for all model energy ratio pairs) on every reciprocal grid point
 * based on the distance between the reciprocal point and the center O
 */
void interpolate_kernel(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double *reciLength = pSPARC->vdWDFreciLength;
    double **kernelReciPoints = pSPARC->vdWDFkernelReciPoints;
    double **kernelPhi = pSPARC->vdWDFkernelPhi;
    double **d2Phidk2 = pSPARC->vdWDFd2Phidk2;
    double dk = pSPARC->vdWDFdk;
    double largestLength = pSPARC->vdWDFnrpoints * dk;
    int rigrid, timeOfdk, q1, q2, qpair;
    double a, b, c, d;
    for (rigrid = 0; rigrid < DMnd; rigrid++) {
        if (reciLength[rigrid] > largestLength)
            continue;
        timeOfdk = (int)floor(reciLength[rigrid] / dk) + 1;
        a = (dk*((timeOfdk - 1.0) + 1.0) - reciLength[rigrid]) / dk;
        b = (reciLength[rigrid] - dk*(timeOfdk - 1.0)) / dk;
        c = (a*a - 1.0)*a * dk*dk / 6.0;
        d = (b*b - 1.0)*b * dk*dk / 6.0;
        for (q1 = 0; q1 < nqs; q1++) {
            for (q2 = 0; q2 < q1 + 1; q2++) {
                qpair = kernel_label(q1, q2, nqs);
                kernelReciPoints[qpair][rigrid] = a*kernelPhi[qpair][timeOfdk - 1] + b*kernelPhi[qpair][timeOfdk]
                + (c*d2Phidk2[qpair][timeOfdk - 1] + d*d2Phidk2[qpair][timeOfdk]);
            }
        }
    }
}

/*
Functions below are two main function of vdWDF: for energy and potential
*/
void vdWDF_energy(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    interpolate_kernel(pSPARC); // Optimize it?
    double **kernelReciPoints = pSPARC->vdWDFkernelReciPoints;
    double _Complex **thetaFTs = pSPARC->vdWDFthetaFTs;
    double _Complex **uFTs = pSPARC->vdWDFuFTs;
    double _Complex vdWenergyLocal = 0.0;
    int q1, q2, qpair, rigrid, i, j, k;
    double _Complex q1riThetaFTs, q2riThetaFTs, uValue;

    // compose u vectors in reciprocal space
    for (q1 = 0; q1 < nqs; q1++) {
        for (rigrid = 0; rigrid < DMnd; rigrid++) { // initialization
            uFTs[q1][rigrid] = 0.0;
        }
    }
    for (q2 = 0; q2 < nqs; q2++) { // compose vector u s in reciprocal space
        for (q1 = 0; q1 < nqs; q1++) { // loop over q2 and q1 (model energy ratio pairs)
            qpair = kernel_label(q1, q2, nqs);
            for (rigrid = 0; rigrid < DMnd; rigrid++) {
                uFTs[q2][rigrid] += kernelReciPoints[qpair][rigrid] * thetaFTs[q1][rigrid];// like the previous ps array, uFTs[q2][rigrid] at here is u(rigrid, q2) in m
            }
        }
    }
    // integrate for vdWDF energy, Soler's paper (8)
    for (q2 = 0; q2 < nqs; q2++) { // loop over all model energy ratio q
        for (rigrid = 0; rigrid < DMnd; rigrid++) { // loop over all riciprocal grid points
            vdWenergyLocal += conj(thetaFTs[q2][rigrid]) * uFTs[q2][rigrid]; // point multiplication
        }
    }
    double vdWenergyLocalReal = creal(vdWenergyLocal) * 0.5 * pSPARC->detLattice;
    MPI_Allreduce(&vdWenergyLocalReal, &(pSPARC->vdWDFenergy), 1, MPI_DOUBLE,
        MPI_SUM, pSPARC->dmcomm_phi);
    #ifdef DEBUG
        if (rank == size - 1) {
        	printf("vdWDF: at %d SCF vdWDF energy is %.5e\n", pSPARC->countSCF, pSPARC->vdWDFenergy);
        }
    #endif 
}
/*
Functions above are related to generating thetas (ps*rho) and integrating energy.
*/

/*
Functions below are related to generating u vectors, transforming them to real space and computing vdW-DF potential.
*/
void parallel_iFFT(double _Complex *inputDataReciSpace, double _Complex *outputDataRealSpace, int *gridsizes, int *zAxis_Starts, int DMnz, int q, MPI_Comm zAxisComm) {
    #if defined(USE_MKL) // use MKL CDFT
    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);
    // initializa parallel iFFT
    DFTI_DESCRIPTOR_DM_HANDLE desc = NULL;
    MKL_LONG localArrayLength, lengthK, startK;
    MKL_LONG dim_sizes[3] = {gridsizes[2], gridsizes[1], gridsizes[0]};
    DftiCreateDescriptorDM(zAxisComm, &desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &localArrayLength);
    DftiGetValueDM(desc, CDFT_LOCAL_NX, &lengthK);
    DftiGetValueDM(desc, CDFT_LOCAL_X_START, &startK);
    int lengthKint = lengthK; int startKint = startK;
    // compose iFFT input to prevent the inconsistency division on Z axis 
    double _Complex *iFFTInput = (double _Complex*)malloc(sizeof(double _Complex)*(localArrayLength));
    double _Complex *iFFTOutput = (double _Complex*)malloc(sizeof(double _Complex)*(localArrayLength));
    assert(iFFTInput != NULL);
    assert(iFFTOutput != NULL);
    int *allStartK = (int*)malloc(sizeof(int)*sizeZcomm);
    int *allLengthK = (int*)malloc(sizeof(int)*sizeZcomm);
    MPI_Allgather(&startKint, 1, MPI_INT, allStartK, 1, MPI_INT, zAxisComm);
    MPI_Allgather(&lengthKint, 1, MPI_INT, allLengthK, 1, MPI_INT, zAxisComm);
    compose_FFTInput(gridsizes, DMnz, zAxis_Starts, iFFTInput, allStartK, inputDataReciSpace, zAxisComm);
    // make parallel FFT
    /* Set that we want out-of-place transform (default is DFTI_INPLACE) */
    DftiSetValueDM(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    /* Commit descriptor, calculate FFT, free descriptor */
    DftiCommitDescriptorDM(desc);
    DftiComputeBackwardDM(desc, iFFTInput, iFFTOutput);

    compose_FFTOutput(gridsizes, DMnz, zAxis_Starts, iFFTOutput, allStartK, allLengthK, outputDataRealSpace, zAxisComm);

    DftiFreeDescriptorDM(&desc);
    free(iFFTInput);
    free(iFFTOutput);
    free(allStartK);
    free(allLengthK);

    #elif defined(USE_FFTW) // use FFTW if MKL is not used

    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);
    // initializa parallel iFFT
    const ptrdiff_t N0 = gridsizes[2], N1 = gridsizes[1], N2 = gridsizes[0]; // N0: z; N1:Y; N2:x
    fftw_plan plan;
	fftw_complex *iFFTInput, *iFFTOutput;
	ptrdiff_t localArrayLength, lengthK, startK;
	fftw_mpi_init();
	/* get local data size and allocate */
	localArrayLength = fftw_mpi_local_size_3d(N0, N1, N2, zAxisComm, &lengthK, &startK);
	iFFTInput = fftw_alloc_complex(localArrayLength);
	iFFTOutput = fftw_alloc_complex(localArrayLength);
	// compose iFFT input to prevent the inconsistency division on Z axis 
	int lengthKint = lengthK; int startKint = startK;
	int *allStartK = (int*)malloc(sizeof(int)*sizeZcomm);
    int *allLengthK = (int*)malloc(sizeof(int)*sizeZcomm);
    MPI_Allgather(&startKint, 1, MPI_INT, allStartK, 1, MPI_INT, zAxisComm);
    MPI_Allgather(&lengthKint, 1, MPI_INT, allLengthK, 1, MPI_INT, zAxisComm);
    compose_FFTInput(gridsizes, DMnz, zAxis_Starts, iFFTInput, allStartK, inputDataReciSpace, zAxisComm);
	/* create plan for out-place backward DFT */
	plan = fftw_mpi_plan_dft_3d(N0, N1, N2, iFFTInput, iFFTOutput, zAxisComm, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	compose_FFTOutput(gridsizes, DMnz, zAxis_Starts, iFFTOutput, allStartK, allLengthK, outputDataRealSpace, zAxisComm);

	fftw_free(iFFTInput);
	fftw_free(iFFTOutput);
	fftw_destroy_plan(plan);
	fftw_mpi_cleanup();

	free(allStartK);
    free(allLengthK);
    #endif
}

// compute u vectors in reciprocal space, then transform them back to real space by iFFT
// Soler's paper (11)-(12)
void u_generate_iFT(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int nnr = pSPARC->Nd;
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int phiDims[3];
    phiDims[0] = pSPARC->npNdx_phi;
    phiDims[1] = pSPARC->npNdy_phi;
    phiDims[2] = pSPARC->npNdz_phi;
    int zAxisDims[3];
    zAxisDims[0] = 1;
    zAxisDims[1] = 1;
    zAxisDims[2] = pSPARC->npNdz_phi;
    int nqs = pSPARC->vdWDFnqs;
    double _Complex **uFTs = pSPARC->vdWDFuFTs;
    double **u = pSPARC->vdWDFu;
    int rigrid, igrid, q1;
    double *uFTreal = (double*)malloc(sizeof(double)*DMnd);
    double *uFTimag = (double*)malloc(sizeof(double)*DMnd);
    double *gathereduFT_real;
    double *gathereduFT_imag;
    double _Complex *gathereduFT;
    double _Complex *gathereduCompl;
    double *gatheredu;
    if (pSPARC->zAxisComm != MPI_COMM_NULL) {
        gathereduFT_real = (double*)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * DMnz);
        gathereduFT_imag = (double*)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * DMnz);
        gathereduFT = (double _Complex*)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * DMnz);
        gathereduCompl = (double _Complex*)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * DMnz);
        gatheredu = (double*)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * DMnz);
        assert(gathereduFT_real != NULL);
        assert(gathereduFT_imag != NULL);
        assert(gathereduFT != NULL);
        assert(gathereduCompl != NULL);
        assert(gatheredu != NULL);
    }
    int *zAxis_Starts = (int*)malloc(sizeof(int) * (pSPARC->npNdz_phi));

    for (int z = 0; z < pSPARC->npNdz_phi; z++) {
        zAxis_Starts[z] = block_decompose_nstart(gridsizes[2], pSPARC->npNdz_phi, z);
    }

    for (q1 = 0; q1 < nqs; q1++) {
        for (rigrid = 0; rigrid < DMnd; rigrid++) {
            uFTreal[rigrid] = creal(uFTs[q1][rigrid]);
            uFTimag[rigrid] = cimag(uFTs[q1][rigrid]);
        }
        D2D(&(pSPARC->gatherThetasSender), &(pSPARC->gatherThetasRecvr), gridsizes, 
            pSPARC->DMVertices, uFTreal, 
            pSPARC->zAxisVertices, gathereduFT_real, 
            pSPARC->dmcomm_phi, phiDims, pSPARC->zAxisComm, zAxisDims, MPI_COMM_WORLD);
        D2D(&(pSPARC->gatherThetasSender), &(pSPARC->gatherThetasRecvr), gridsizes, 
            pSPARC->DMVertices, uFTimag, 
            pSPARC->zAxisVertices, gathereduFT_imag, 
            pSPARC->dmcomm_phi, phiDims, pSPARC->zAxisComm, zAxisDims, MPI_COMM_WORLD);
        if (pSPARC->zAxisComm != MPI_COMM_NULL) {
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * DMnz; rigrid++) {
                gathereduFT[rigrid] = gathereduFT_real[rigrid] + gathereduFT_imag[rigrid]*I;
            }
            parallel_iFFT(gathereduFT, gathereduCompl, gridsizes, zAxis_Starts, DMnz, q1, pSPARC->zAxisComm);
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * DMnz; rigrid++) {
                gatheredu[rigrid] = creal(gathereduCompl[rigrid]); // MKL original iFFT functions do not divide the iFFT results by N
            }
        }
        D2D(&(pSPARC->scatterThetasSender), &(pSPARC->scatterThetasRecvr), gridsizes, // scatter the real part of theta results from the processors on z axis (0, 0, z) to all other processors
            pSPARC->zAxisVertices, gatheredu, 
            pSPARC->DMVertices, u[q1], 
            pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi, phiDims, MPI_COMM_WORLD);
        // if ((pSPARC->countSCF == 0) && (q1 == 5) && (rank == size - 1)) { // only output result in 1st step
        //     int localIndex1D = domain_index1D(1, 1, 1, DMnx, DMny, DMnz);
        //     fprintf(pSPARC->vdWDFOutput, "rank %d, at 1st SCF u[5][(1, 1, 1)]=globalu[5][(%d, %d, %d)] = %.5e\n", rank, 
        //         pSPARC->DMVertices[0] + 1, pSPARC->DMVertices[2] + 1, pSPARC->DMVertices[4] + 1,
        //         u[q1][localIndex1D]);
        // }
    }

    free(uFTreal);
    free(uFTimag);
    free(zAxis_Starts);
    if (pSPARC->zAxisComm != MPI_COMM_NULL) {
        free(gathereduFT_real);
        free(gathereduFT_imag);
        free(gathereduFT);
        free(gathereduCompl);
        free(gatheredu);
    }
}

void vdWDF_potential(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nnr = pSPARC->Nd;
    int nqs = pSPARC->vdWDFnqs;
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    double **u = pSPARC->vdWDFu;
    double **ps = pSPARC->vdWDFps;
    double **dpdq0s = pSPARC->vdWDFdpdq0s;
    double *dq0drho = pSPARC->vdWDFdq0drho;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double *potential = pSPARC->vdWDFpotential;
    double *hPrefactor = (double*)malloc(sizeof(double)*DMnd);
    int igrid, q1;
    for (igrid = 0; igrid < DMnd; igrid++) { // initialization
        potential[igrid] = 0.0;
        hPrefactor[igrid] = 0.0;
    }
    for (q1 = 0; q1 < nqs; q1++) {
        for (igrid = 0; igrid < DMnd; igrid++) {
            potential[igrid] += u[q1][igrid]*(ps[q1][igrid] + dpdq0s[q1][igrid]*dq0drho[igrid]); // First term of Soler's paper (10)
            hPrefactor[igrid] += u[q1][igrid]*dpdq0s[q1][igrid]*dq0dgradrho[igrid]; // Second term of Soler's paper (10), to multiply diffential coefficient matrix
        }
    }
    int direction;
    double *h = (double*)malloc(sizeof(double)*DMnd);
    double *Dh = (double*)malloc(sizeof(double)*DMnd);
    double **Drho = pSPARC->Drho;
    double *gradRhoLen = pSPARC->gradRhoLen;

    for (direction = 0; direction < 3; direction++) {
        for (igrid = 0; igrid < DMnd; igrid++) {
            h[igrid] = hPrefactor[igrid]*Drho[direction][igrid];
            if (gradRhoLen[igrid] > 1e-15) { // gradRhoLen > 0.0
                h[igrid] /= gradRhoLen[igrid];
            }
        }
        if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){ // non-orthogonal cell
            double *Dh_1, *Dh_2, *Dh_3;
            Dh_1 = (double*)malloc(sizeof(double)*DMnd);
            Dh_2 = (double*)malloc(sizeof(double)*DMnd);
            Dh_3 = (double*)malloc(sizeof(double)*DMnd);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, Dh_1, 0, pSPARC->dmcomm_phi); // nonCart direction
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, Dh_2, 1, pSPARC->dmcomm_phi);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, Dh_3, 2, pSPARC->dmcomm_phi);
            for (igrid = 0; igrid < DMnd; igrid++) {
                Dh[igrid] = pSPARC->gradT[0+direction]*Dh_1[igrid] + pSPARC->gradT[3+direction]*Dh_2[igrid] + pSPARC->gradT[6+direction]*Dh_3[igrid];
            }
            free(Dh_1);
            free(Dh_2);
            free(Dh_3);
        } else { // orthogonal cell
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, Dh, direction, pSPARC->dmcomm_phi); // Soler's paper (10), diffential coefficient matrix
        }
        for (igrid = 0; igrid < DMnd; igrid++) {
            potential[igrid] -= Dh[igrid];
        }
    }
    for (igrid = 0; igrid < DMnd; igrid++) { // add vdWDF potential to the potential of system
        pSPARC->XCPotential[igrid] += potential[igrid];
    }

    free(hPrefactor);
    free(h);
    free(Dh);
}
/*
Functions above are related to generating u vectors, transforming them to real space and computing vdW-DF potential.
*/

// The main function in the file
// The main function to be called by function Calculate_Vxc_GGA in exchangeCorrelation.c, compute the energy and potential of non-linear correlation part of vdW-DF
void Calculate_nonLinearCorr_E_V_vdWDF(SPARC_OBJ *pSPARC, double *rho) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    }
    get_q0_Grid(pSPARC, rho);
    spline_interpolation(pSPARC); // get the component of "model energy ratios" on every grid point
    compute_Gvectors(pSPARC);
    theta_generate_FT(pSPARC, rho);
    vdWDF_energy(pSPARC);
    u_generate_iFT(pSPARC);
    vdWDF_potential(pSPARC);

    pSPARC->countSCF++; // count the time of SCF. To output variables in 1st step. To be deleted in the future.
}

void Add_Exc_vdWDF(SPARC_OBJ *pSPARC) { // add vdW_DF energy into the total xc energy
    pSPARC->Exc += pSPARC->vdWDFenergy;
}

