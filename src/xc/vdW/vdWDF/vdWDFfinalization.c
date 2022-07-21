/**
 * @file    vdWDFfinalization.c
 * @brief   This file contains the functions for finalizing variables of vdF-DF1 and vdW-DF2 functional.
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
#include <complex.h>

#include "vdWDFfinalization.h"
#include "parallelization.h"

void vdWDF_free(SPARC_OBJ *pSPARC) {
	int nqs = pSPARC->vdWDFnqs;
    int numberKernel = nqs*(nqs - 1)/2 + nqs;
    int qpair, q1;
    free(pSPARC->vdWDFqmesh);
    for (qpair = 0; qpair < numberKernel; qpair++) {
        free(pSPARC->vdWDFkernelPhi[qpair]);
        free(pSPARC->vdWDFd2Phidk2[qpair]);
    }
    for (q1 = 0; q1 < nqs; q1++) {
        free(pSPARC->vdWDFd2Splineydx2[q1]);
    }
    free(pSPARC->vdWDFkernelPhi);
    free(pSPARC->vdWDFd2Phidk2);
    free(pSPARC->vdWDFd2Splineydx2);
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        int rank;
        MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
        int size;
        MPI_Comm_size(pSPARC->dmcomm_phi, &size);
        // if (rank == size - 1) fclose(pSPARC->vdWDFOutput);
        int direction;
        for (direction = 0; direction < 3; direction++){
            free(pSPARC->Drho[direction]);
        }
        free(pSPARC->Drho);
        free(pSPARC->gradRhoLen);
        free(pSPARC->vdWDFq0);
        free(pSPARC->vdWDFdq0drho);
        free(pSPARC->vdWDFdq0dgradrho);

        for (q1 = 0; q1 < nqs; q1++) {
            free(pSPARC->vdWDFps[q1]); 
            free(pSPARC->vdWDFdpdq0s[q1]);
            free(pSPARC->vdWDFthetaFTs[q1]);
        }
        free(pSPARC->vdWDFps); 
        free(pSPARC->vdWDFdpdq0s);
        free(pSPARC->vdWDFthetaFTs);
        for (direction = 0; direction < 3; direction++){
            free(pSPARC->timeReciLattice[direction]);
            free(pSPARC->vdWDFreciLatticeGrid[direction]);
        }
        free(pSPARC->timeReciLattice);
        free(pSPARC->vdWDFreciLatticeGrid);
        free(pSPARC->vdWDFreciLength);
        Free_D2D_Target(&pSPARC->gatherThetasSender, &pSPARC->gatherThetasRecvr, pSPARC->dmcomm_phi, pSPARC->zAxisComm);
        Free_D2D_Target(&pSPARC->scatterThetasSender, &pSPARC->scatterThetasRecvr, pSPARC->zAxisComm, pSPARC->dmcomm_phi);
        if (pSPARC->zAxisComm != MPI_COMM_NULL) {
            int FFTrank;
            MPI_Comm_rank(pSPARC->zAxisComm, &FFTrank);
            // if (FFTrank == 0) fclose(pSPARC->vdWDFparallelFFTOut);
            MPI_Comm_free(&pSPARC->zAxisComm);
        }
        for (qpair = 0; qpair < numberKernel; qpair++) {
            free(pSPARC->vdWDFkernelReciPoints[qpair]);
        }
        free(pSPARC->vdWDFkernelReciPoints);
        for (q1 = 0; q1 < nqs; q1++) {
            free(pSPARC->vdWDFuFTs[q1]);
            free(pSPARC->vdWDFu[q1]);
        }
        free(pSPARC->vdWDFuFTs);
        free(pSPARC->vdWDFu);
        free(pSPARC->vdWDFpotential);
    }
}