/**
 * @file    vdWDFNonlinearCorre.c
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
#include "vdWDF.h"
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

 vdWDF_initial_read_kernel // READ needed kernel functions (phi), its 2nd derivatives and 2nd derivatives of spline functions of model energy ratios
 ├──vdWDF_read_kernel
 ├──read_spline_d2_qmesh
 └──domain_index1D

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

 Calculate_XC_stress_vdWDF // Compute stress of vdWDF, called by Calculate_XC_stress in stress.c and Calculate_XC_pressure in pressure.c
 ├──vdWDF_stress_gradient
 └──vdWDF_stress_kernel
    └──interpolate_dKerneldK

 vdWDF_free // Free space allocated for vdWDF functions, called by Free_SPARC in finalization.c
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

/*
The functions below are for initializing vdW-DF variables
*/

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

/*
The functions above are for initializing vdW-DF variables
*/

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

/**
 * @brief compute the stress term containing grad(rho)
 deduced from Soler's paper formula (8). Make derivative on energy 
 E = 1/2*sum_{alpha}(int(theta_alpha*u_alpha)d(r_1)) = 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1)*phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2))
 d^2E/dxdy = 1/2*sum_{alpha}sum_{beta}(intint(d^2(theta_alpha(r_1)*phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))/dxdy)d(r_1)d(r_2))
           = 1/2*sum_{alpha}sum_{beta}(intint(d^2(theta_alpha(r_1))/dxdy * phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2))
            + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * d^2(phi_{alphabeta}(|r_2 - r_1|))/dxdy * theta_beta(r_2))d(r_1)d(r_2))
            + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * phi_{alphabeta}(|r_2 - r_1|)*d^2(theta_beta(r_2))/dxdy)d(r_1)d(r_2))
 the sum of 1st term and 3rd term is
 1/2*sum_{alpha}sum_{beta}(intint(d^2(theta_alpha(r_1))/dxdy * phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2)) + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * phi_{alphabeta}(|r_2 - r_1|)*d^2(theta_beta(r_2))/dxdy)d(r_1)d(r_2))
           = 1/2*sum_{alpha}(int(d^2(theta_alpha(r_1))/dxdy * [sum_{beta}int(phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_2)])d(r_1)) + 1/2*sum_{beta}(int(d^2(theta_beta(r_2))/dxdy * [sum_{alpha}int(phi_{alphabeta}(|r_2 - r_1|)*theta_alpha(r_1))d(r_1)])d(r_2))
           = 1/2*sum_{alpha}(int(d^2(theta_alpha(r_1))/dxdy * u_alpha(r_1))d(r_1)) + 1/2*sum_{beta}(int(d^2(theta_beta(r_2)/dxdy * u_beta(r_2))d(r_2))
           = sum_{alpha}(int(d^2(theta_alpha(r_1))/dxdy * u_alpha(r_1))d(r_1))
 That is the term this function compute:
 sum_{alpha}(int(d^2(theta_alpha(r_1))/dxdy * u_alpha(r_1))d(r_1))
           = sum_{alpha}(int(u_alpha * dp_alpha/dq0 * dq0/d(gradrho) * [drho/dx*drho/dy) / |gradrho|])d(r_1))
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
        if (gradRhoLen[igrid] > 1e-15) {
            prefactor[igrid] /= gradRhoLen[igrid];
        }
    }

    int row, col, dir;
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
    for (dir = 0; dir < 9; dir++) {
        MPI_Allreduce(&(localStressGrad[dir]), &(stressGrad[dir]), 1, MPI_DOUBLE,
            MPI_SUM, pSPARC->dmcomm_phi);
    }
    free(prefactor);
}



void interpolate_dKerneldK(SPARC_OBJ *pSPARC, double **dKerneldLength) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nnr = pSPARC->Nd;
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
 * @brief compute the stress term containing grad(rho)
 deduced from Soler's paper formula (8). Make derivative on energy 
 E = 1/2*sum_{alpha}(int(theta_alpha*u_alpha)d(r_1)) = 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1)*phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2))
 d^2E/dxdy = 1/2*sum_{alpha}sum_{beta}(intint(d^2(theta_alpha(r_1)*phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))/dxdy)d(r_1)d(r_2))
           = 1/2*sum_{alpha}sum_{beta}(intint(d^2(theta_alpha(r_1))/dxdy * phi_{alphabeta}(|r_2 - r_1|)*theta_beta(r_2))d(r_1)d(r_2))
            + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * d^2(phi_{alphabeta}(|r_2 - r_1|))/dxdy * theta_beta(r_2))d(r_1)d(r_2))
            + 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * phi_{alphabeta}(|r_2 - r_1|)*d^2(theta_beta(r_2))/dxdy)d(r_1)d(r_2))
 The function computes the 2nd term
 1/2*sum_{alpha}sum_{beta}(intint(theta_alpha(r_1) * d^2(phi_{alphabeta}(|r_2 - r_1|))/dxdy * theta_beta(r_2))d(r_1)d(r_2)) 
  after Fourier Transform
            = 1/2*sum_{alpha}sum_{beta}(int(conj(theta_alpha(k_1)) * d^2(phi_{alphabeta}(|k_1|))/dxdy * theta_beta(k_1))d(k_1))
 */

void vdWDF_stress_kernel(SPARC_OBJ *pSPARC, double *stressKernel) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nnr = pSPARC->Nd;
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

    for (int dir = 0; dir < 9; dir++) {
        MPI_Allreduce(&(localStressKernel[dir]), &(stressKernel[dir]), 1, MPI_DOUBLE,
            MPI_SUM, pSPARC->dmcomm_phi);
    }

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
    vdWDF_stress_gradient(pSPARC, stressGrad); // compute the stress term containing grad(rho)

    double stressKernelArray[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
    double *stressKernel = stressKernelArray;
    vdWDF_stress_kernel(pSPARC, stressKernel); // compute the stress term containing derivative of kernel functions

    for (int sigmaDir = 0; sigmaDir < 9; sigmaDir++) {
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