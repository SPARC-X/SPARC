/**
 * @file    vdWDFinitialization.c
 * @brief   This file contains the functions for initializing vdF-DF1 and vdW-DF2 functional.
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

/*
 Structure of functions in vdWDFinitializeFinalize.c
 vdWDF_initial_read_kernel // READ needed kernel functions (phi), its 2nd derivatives and 2nd derivatives of spline functions of model energy ratios
 ├──vdWDF_read_kernel
 ├──read_spline_d2_qmesh
 └──domain_index1D

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
#include "tools.h"
#include "parallelization.h"
#include "vdWDFinitialization.h"
#include "gradVecRoutines.h"
#include "exchangeCorrelation.h"

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

// initialize all needed variables for vdW-DF calculations, and read kernel functions and 2nd derivative of spline functions
void vdWDF_initial_read_kernel(SPARC_OBJ *pSPARC) {
	// Initialization, set parameters and grids of model kernel functions
    int nrpoints = 1024; pSPARC->vdWDFnrpoints = nrpoints; // radial points for composing Phi functions in real and reciprocal space
    double rMax = 100.0; // max radius in real space and minimum k point 2*pi/r_max in reciprocal space
    double vdWdr = rMax / nrpoints; pSPARC->vdWDFdr = vdWdr;
    double vdWdk = 2.0 * M_PI / rMax; pSPARC->vdWDFdk = vdWdk;
    int nqs = 20;
    pSPARC->vdWDFnqs = nqs;
    if (pSPARC->vdWDFFlag == 1) { // vdWDF1
    	pSPARC->vdWDFZab = -0.8491;
    }
    else { // vdWDF2
    	pSPARC->vdWDFZab = -1.887;
    }
    double qmesh[20] = { //model energy ratios
    1.0e-5           , 0.0449420825586261, 0.0975593700991365, 0.159162633466142,
    0.231286496836006, 0.315727667369529 , 0.414589693721418 , 0.530335368404141,
    0.665848079422965, 0.824503639537924 , 1.010254382520950 , 1.227727621364570,
    1.482340921174910, 1.780437058359530 , 2.129442028133640 , 2.538050036534580,
    3.016440085356680, 3.576529545442460 , 4.232271035198720 , 5.0};
    double*qmeshPointer = qmesh;
    pSPARC->vdWDFqmesh = (double*)malloc(sizeof(double)*nqs);
    memcpy(pSPARC->vdWDFqmesh, qmeshPointer, sizeof(double)*nqs);
    int numberKernel = nqs*(nqs - 1)/2 + nqs;
    pSPARC->vdWDFkernelPhi = (double**)malloc(sizeof(double*)*numberKernel); // kernal Phi, index 0, reciprocal
    pSPARC->vdWDFd2Phidk2 = (double**)malloc(sizeof(double*)*numberKernel); // 2nd derivative of kernals
    pSPARC->vdWDFd2Splineydx2 = (double**)malloc((sizeof(double*)*nqs));
    int q1, q2, qpair;
    for (q1 = 0; q1 < nqs; q1++) {
        for (q2 = 0; q2 < q1 + 1; q2++) { // q2 is smaller than q1
            qpair = kernel_label(q1, q2, nqs);
            pSPARC->vdWDFkernelPhi[qpair] = (double*)malloc(sizeof(double)*(1 + nrpoints));
            pSPARC->vdWDFd2Phidk2[qpair] = (double*)malloc(sizeof(double)*(1 + nrpoints));
        }
        pSPARC->vdWDFd2Splineydx2[q1] = (double*)malloc(sizeof(double)*(nqs));
    }

    char folderRoute[L_STRING];
    char kernelFileRoute[L_STRING]; // name of the file of the output kernels and 2nd derivative of kernels
    char splineD2FileRoute[L_STRING]; //name of the file of the 2nd derivatives of spline functions
    strncpy(folderRoute, pSPARC->filename,sizeof(folderRoute));
    int indexEndFolderRoute = 0;
    while (folderRoute[indexEndFolderRoute] != '\0') {
        indexEndFolderRoute++;
    } // get the length of the char filename
    while ((indexEndFolderRoute > -1) && (folderRoute[indexEndFolderRoute] != '/')) {
        indexEndFolderRoute--;
    } // find the last '/'. If there is no '/', indexEndFolderRoute should be at the beginning
    folderRoute[indexEndFolderRoute + 1] = '\0'; // cut the string. Now it contains only the folder position
    snprintf(kernelFileRoute,       L_STRING, "%skernel_d2.txt"  ,     folderRoute);
    snprintf(splineD2FileRoute,       L_STRING, "%sspline_d2.txt"  ,     folderRoute);

    vdWDF_read_kernel(pSPARC, kernelFileRoute); // kernelFileRoute can be changed.
    read_spline_d2_qmesh(pSPARC, splineD2FileRoute);
    pSPARC->zAxisComm = MPI_COMM_NULL;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
    	int rank;
    	MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    	int size;
    	MPI_Comm_size(pSPARC->dmcomm_phi, &size);
        int DMnx, DMny, DMnz, DMnd;
        DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
        DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
        DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
        DMnd = DMnx * DMny * DMnz;
        pSPARC->Drho = (double**)malloc(sizeof(double*)*3);
        int direction; // now only consider 3D cases
        for (direction = 0; direction < 3; direction++) {
            pSPARC->Drho[direction] = (double*)malloc(sizeof(double)*DMnd);
        }
        pSPARC->gradRhoLen = (double*)malloc(sizeof(double)*DMnd);
        pSPARC->vdWDFq0 = (double*)malloc(sizeof(double)*DMnd);
        pSPARC->vdWDFdq0drho = (double*)malloc(sizeof(double)*DMnd);
        pSPARC->vdWDFdq0dgradrho = (double*)malloc(sizeof(double)*DMnd);
        pSPARC->vdWDFps = (double**)malloc(sizeof(double*)*nqs);
        pSPARC->vdWDFdpdq0s = (double**)malloc(sizeof(double*)*nqs);
        pSPARC->vdWDFthetaFTs = (double _Complex**)malloc(sizeof(double*)*nqs);
        for (q1 = 0; q1 < nqs; q1++) {
            pSPARC->vdWDFps[q1] = (double*)malloc(sizeof(double)*DMnd); 
            // at here ps[q1][i] is the component coefficient of q1th model energy ratio on ith grid point. In m, it is ps(i, q1)
            pSPARC->vdWDFdpdq0s[q1] = (double*)malloc(sizeof(double)*DMnd); 
            pSPARC->vdWDFthetaFTs[q1] = (double _Complex*)malloc(sizeof(double _Complex)*DMnd); 
            // likewise, thetaFTs[q1][i] is the theta function of q1th model energy ratio on ith grid point. Im m, it is thetasFFT(q1, i)
        }
        pSPARC->countSCF = 0; // count the time of SCF. To output variables in 1st step. To be deleted in the future.

        // compose the index of these reciprocal lattice vectors and make a permutation.
        pSPARC->timeReciLattice = (int**)malloc(sizeof(int*)*3); // reciprocal lattice grid point i = timeReciLattice[0][i]*pSPARC->reciLattice0(0 1 2)+timeReciLattice[1][i]*pSPARC->reciLattice1(3 4 5)+timeReciLattice[2][i]*pSPARC->reciLattice2(6 7 8)
        pSPARC->vdWDFreciLatticeGrid = (double**)malloc(sizeof(double*)*3);
        int i, j, k, localIndex1D;
        for (direction = 0; direction < 3; direction++) {
            pSPARC->timeReciLattice[direction] = (int*)malloc(sizeof(int)*DMnd); // 000 100 200 ... 010 110 210 ... 001 101 201 ... 011 111 211 ...
            pSPARC->vdWDFreciLatticeGrid[direction] = (double*)malloc(sizeof(double*)*DMnd);
        }
        for (i = pSPARC->DMVertices[0]; i < pSPARC->DMVertices[1] + 1; i++) { // timeReciLattice[direction][index of reci grid point]
            for (j = pSPARC->DMVertices[2]; j < pSPARC->DMVertices[3] + 1; j++) {
                for (k = pSPARC->DMVertices[4]; k < pSPARC->DMVertices[5] + 1; k++) {
                    localIndex1D = domain_index1D(i - pSPARC->DMVertices[0], j - pSPARC->DMVertices[2], k - pSPARC->DMVertices[4], DMnx, DMny, DMnz);
                    pSPARC->timeReciLattice[0][localIndex1D] = (i < pSPARC->Nx/2 + 1) ? i:(i - pSPARC->Nx); // in 1D: 0 1 2 3 4 5 -> 0 1 2 3 -2 -1 
                    //in m: reciLabelSeq(reciLabelSeq(:)) = 1:ngm; vecLength = vecLength(reciLabelSeq(:)). Get the corresponding vector time of every index of the (array after FFT)
                    pSPARC->timeReciLattice[1][localIndex1D] = (j < pSPARC->Ny/2 + 1) ? j:(j - pSPARC->Ny);
                    pSPARC->timeReciLattice[2][localIndex1D] = (k < pSPARC->Nz/2 + 1) ? k:(k - pSPARC->Nz);
                }
            }
        }
        pSPARC->vdWDFreciLength = (double*)malloc(sizeof(double)*DMnd);

        // set the zAxixComm, the communicator contains processors on z axis (0, 0, z)
        int gridsizes[3];
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        int coord_dmcomm[3];
        int rank_dmcomm;
        MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm);
        MPI_Cart_coords(pSPARC->dmcomm_phi, rank_dmcomm, 3, coord_dmcomm);
        MPI_Group comm_group;
        MPI_Comm_group(pSPARC->dmcomm_phi, &comm_group);
        int *zAxis_ranks = (int*)malloc(sizeof(int) * (pSPARC->npNdz_phi));
        for (int index = 0; index < pSPARC->npNdz_phi; index++) {
            int zAxis_coords[3] = {0, 0, index};
            MPI_Cart_rank(pSPARC->dmcomm_phi, zAxis_coords, zAxis_ranks + index);
        }
        MPI_Group zAxis_group = MPI_GROUP_EMPTY;
        MPI_Group_incl(comm_group, pSPARC->npNdz_phi, zAxis_ranks, &zAxis_group);
        MPI_Comm TempzAxisComm = MPI_COMM_NULL;
        MPI_Comm_create_group(pSPARC->dmcomm_phi, zAxis_group, 0, &TempzAxisComm);
        
        int phiDims[3];
        phiDims[0] = pSPARC->npNdx_phi;
        phiDims[1] = pSPARC->npNdy_phi;
        phiDims[2] = pSPARC->npNdz_phi;
        int zAxisDims[3];
        zAxisDims[0] = 1;
        zAxisDims[1] = 1;
        zAxisDims[2] = pSPARC->npNdz_phi;
        int periods[3] = {1, 1, 1};
        int zAxisLabel = 0;
        if ((coord_dmcomm[0] == 0) && (coord_dmcomm[1] == 0)) { // to verify whether zAxisComm is set
            zAxisLabel = 1;
            MPI_Cart_create(TempzAxisComm, 3, zAxisDims, periods, 1, &(pSPARC->zAxisComm));
            int FFTrank;
            MPI_Comm_rank(pSPARC->zAxisComm, &FFTrank);
            pSPARC->zAxisVertices[0] = 0; 
            pSPARC->zAxisVertices[1] = gridsizes[0] - 1;
            pSPARC->zAxisVertices[2] = 0; 
            pSPARC->zAxisVertices[3] = gridsizes[1] - 1;
            pSPARC->zAxisVertices[4] = block_decompose_nstart(gridsizes[2], pSPARC->npNdz_phi, coord_dmcomm[2]);
            pSPARC->zAxisVertices[5] = pSPARC->zAxisVertices[4] + pSPARC->Nz_d - 1;
            #ifdef DEBUG
            if (FFTrank == 0) {
                // char vdWDFparallelFFTRoute[L_STRING];
                // snprintf(vdWDFparallelFFTRoute,       L_STRING, "%svdWDFparallelOutput.txt"  ,     folderRoute);
                // pSPARC->vdWDFparallelFFTOut = fopen(vdWDFparallelFFTRoute,"w");
                printf("vdWDF: rank %d, dmcomm_phi coords %d %d %d, FFTrank %d, zAxisVertices[4] %d, zAxisVertices[5] %d\n", 
                    rank_dmcomm, coord_dmcomm[0], coord_dmcomm[1], coord_dmcomm[2], FFTrank, pSPARC->zAxisVertices[4], pSPARC->zAxisVertices[5]);
            }
            #endif
        }
        if (TempzAxisComm != MPI_COMM_NULL) {
            MPI_Comm_free(&TempzAxisComm);
        }
        MPI_Group_free(&comm_group);
        if (zAxis_group != MPI_GROUP_EMPTY) {
            MPI_Group_free(&zAxis_group);
        }
        // compose the sender and receiver for gathering thetas to do parallel FFT
        Set_D2D_Target(&pSPARC->gatherThetasSender, &pSPARC->gatherThetasRecvr, gridsizes, 
                        pSPARC->DMVertices, pSPARC->zAxisVertices,
                       pSPARC->dmcomm_phi, phiDims,
                       pSPARC->zAxisComm, zAxisDims, MPI_COMM_WORLD);
        // compose the sender and receiver for scattering thetasFT after parallel FFT
        Set_D2D_Target(&pSPARC->scatterThetasSender, &pSPARC->scatterThetasRecvr, gridsizes,
                        pSPARC->zAxisVertices, pSPARC->DMVertices,
                       pSPARC->zAxisComm, zAxisDims,
                       pSPARC->dmcomm_phi, phiDims, MPI_COMM_WORLD);
        free(zAxis_ranks);

        int rigrid;
        pSPARC->vdWDFkernelReciPoints = (double**)malloc(sizeof(double*)*numberKernel);
        for (q1 = 0; q1 < nqs; q1++) {
            for (q2 = 0; q2 < q1 + 1; q2++) {
                qpair = kernel_label(q1, q2, nqs);
                pSPARC->vdWDFkernelReciPoints[qpair] = (double*)malloc(sizeof(double)*DMnd);
            }
        }
        pSPARC->vdWDFuFTs = (double _Complex**)malloc(sizeof(double _Complex*)*nqs);
        pSPARC->vdWDFu = (double**)malloc(sizeof(double)*nqs);
        for (q1 = 0; q1 < nqs; q1++) {
            pSPARC->vdWDFuFTs[q1] = (double _Complex*)malloc(sizeof(double _Complex)*DMnd);
            pSPARC->vdWDFu[q1] = (double*)malloc(sizeof(double)*DMnd);
        }
        pSPARC->vdWDFpotential = (double*)malloc(sizeof(double)*DMnd);
        // if (rank == size - 1) {
    	// 	char vdWDFoutputRoute[L_STRING];
        //     snprintf(vdWDFoutputRoute,       L_STRING, "%svdWDFoutput.txt"  ,     folderRoute);
    	// 	pSPARC->vdWDFOutput = fopen(vdWDFoutputRoute,"w");
    	// 	fprintf(pSPARC->vdWDFOutput, "                    vdWDF Energy and Stress Calculation                    \n");
    	// }
    }
}

int domain_index1D(int locali, int localj, int localk, int DMnx, int DMny, int DMnz) {
    int DMnd = DMnx * DMny * DMnz;
    if (locali > DMnx - 1) {
        printf("mistake on local X index!\n");
        return 0;
    }
    if (localj > DMny - 1) {
        printf("mistake on local Y index!\n");
        return 0;
    }
    if (localk > DMnz - 1) {
        printf("mistake on local Z index!\n");
        return 0;
    }
    return locali + localj*DMnx + localk*DMnx*DMny;
}

int kernel_label(int firstq, int secondq, int nqs) { // 1-D index of the kernel function of (q1, q2)
    // int answer = firstq*nqs + secondq;
    if (firstq < secondq) {
        int med = firstq;
        firstq = secondq;
        secondq = med;
    }
    int answer = (1+firstq)*firstq/2 + secondq;
    return answer;
}

void vdWDF_read_kernel(SPARC_OBJ *pSPARC, char *kernelFile) {
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // rank 0 reads the two files and save them in its own vdWDFkernelPhi, vdWDFd2Phidk2
    int nrpoints = pSPARC->vdWDFnrpoints; 
    int nqs = pSPARC->vdWDFnqs;
    int q1, q2, qpair;
    if (rank == 0) {
        errno = 0;
    	FILE *kernelRead = fopen(kernelFile, "r");
    	if(kernelRead==NULL)
    	{
            int errNum = errno;
            printf("open fail errno = %d\n", errNum);
        	printf("rank %d, kernel file cannot be opened!\n", rank);
            if (errNum == 2) {
                printf("Please make sure there are files kernel_d2.txt and spline_d2.txt in the input folder!\n");
                printf("Further information can be found in manual VDWDF_GEN_KERNEL page\n");
            }
        	exit(EXIT_FAILURE);
    	}
    	int readIn;
    	char *readLines = (char*)malloc(sizeof(char)*40);
    	double *aLine = (double*)malloc(sizeof(double)*2);
    	for (q1 = 0; q1 < nqs; q1++) {
    	    for (q2 = 0; q2 <= q1; q2++) {
    	        fgets(readLines, 39, kernelRead); // first q 0, second q 0
    	        fgets(readLines, 39, kernelRead); // Kernel function
    	        qpair = kernel_label(q1, q2, nqs);
    	        int ir;
    	        for (ir = 0; ir <= nrpoints; ir++) {
    	            readIn = fscanf(kernelRead, "%lf %lf\n", aLine, aLine + 1);
    	            pSPARC->vdWDFkernelPhi[qpair][ir] = aLine[1];
    	        }
    	        fgets(readLines, 39, kernelRead); // 2nd derivative of Kernel function
    	        for (ir = 0; ir <= nrpoints; ir++) {
    	            readIn = fscanf(kernelRead, "%lf %lf\n", aLine, aLine + 1);
    	            pSPARC->vdWDFd2Phidk2[qpair][ir] = aLine[1];
    	        }
    	    }
    	}
    	fclose(kernelRead);
    	free(readLines);
    	free(aLine);
    }
    // then rank 0 broadcast the kernel functions and the 2nd derivative of kernels to all processors
    for (q1 = 0; q1 < nqs; q1++) {
	    for (q2 = 0; q2 <= q1; q2++) {
	        qpair = kernel_label(q1, q2, nqs);
	        MPI_Bcast(pSPARC->vdWDFkernelPhi[qpair], nrpoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	        MPI_Bcast(pSPARC->vdWDFd2Phidk2[qpair], nrpoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    }
	}
}

void read_spline_d2_qmesh(SPARC_OBJ *pSPARC, char *splineD2QmeshFile) {
	int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // rank 0 reads the two files and save them in its own vdWDFd2Splineydx2
    int nqs = pSPARC->vdWDFnqs; int readIn;
    int q1, q2;
    if (rank == 0) {
        errno = 0;
    	FILE *splineD2Read = fopen(splineD2QmeshFile, "r");
    	if(splineD2Read == NULL) {
            int errNum = errno;
            printf("open fail errno = %d\n", errNum);
    	    printf("splineD2 file cannot be opened!\n");
            if (errNum == 2) {
                printf("Please make sure there are files kernel_d2.txt and spline_d2.txt in the input folder!\n");
                printf("Further information can be found in manual VDWDF_GEN_KERNEL page\n");
            }
    	    exit(EXIT_FAILURE);
    	}
    	double **d2Splineydx2 = pSPARC->vdWDFd2Splineydx2;
    	double *aLine = (double*)malloc(sizeof(double)*nqs);
    	char *readLines = (char*)malloc(sizeof(char)*40);
    	fgets(readLines, 39, splineD2Read);
    	for (q1 = 0; q1 < nqs; q1++) {
    	    readIn = fscanf(splineD2Read, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", aLine, aLine+1, aLine+2, aLine+3, aLine+4, aLine+5,
    	        aLine+6, aLine+7, aLine+8, aLine+9, aLine+10, aLine+11, aLine+12, aLine+13, aLine+14, aLine+15, aLine+16, aLine+17, aLine+18, aLine+19);
    	    for (q2 = 0; q2 < nqs; q2++) {
    	        d2Splineydx2[q1][q2] = aLine[q2];
    	    }
    	}
    	fclose(splineD2Read);
        free(aLine);
        free(readLines);
    }
    // then rank 0 broadcast the 2nd derivative of spline functions to all processors
    for (q1 = 0; q1 < nqs; q1++) {
    	MPI_Bcast(pSPARC->vdWDFd2Splineydx2[q1], nqs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}
