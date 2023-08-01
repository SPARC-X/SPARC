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
 Structure of functions in vdWDFinitializeFinalize.c
 vdWDF_initial_read_kernel // READ needed kernel functions (phi), its 2nd derivatives and 2nd derivatives of spline functions of model energy ratios
 ├──vdWDF_read_kernel
 ├──read_spline_d2_qmesh
 ├──vdWDF_Setup_Comms
 └──domain_index1D

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>
#include <errno.h> 
#include <time.h>

#include "isddft.h"
#include "tools.h"
#include "parallelization.h"
#include "vdWDFinitialization.h"
#include "vdWDFparallelization.h"
#include "vdWDFreadKernel.h"
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
    if (pSPARC->ixc[3] == 1) { // vdWDF1
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

    vdWDF_read_kernel(pSPARC);
    read_spline_d2_qmesh(pSPARC);

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
            pSPARC->Drho[direction] = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(pSPARC->Drho[direction] != NULL);
        }
        pSPARC->gradRhoLen = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double));
        pSPARC->vdWDFecLinear = (double*)malloc(DMnd * sizeof(double)); assert(pSPARC->vdWDFecLinear != NULL); // \epsilon_cl
        pSPARC->vdWDFVcLinear = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(pSPARC->vdWDFVcLinear != NULL); // d(n\epsilon_cl)/dn in dmcomm_phi
        pSPARC->vdWDFq0 = (double*)malloc(sizeof(double)*DMnd);
        pSPARC->vdWDFdq0drho = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(pSPARC->vdWDFdq0drho != NULL);
        pSPARC->vdWDFdq0dgradrho = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(pSPARC->vdWDFdq0dgradrho != NULL);
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
        pSPARC->countPotentialCalculate = 0; // count the time of SCF. To output variables in 1st step. To be deleted in the future.

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
        
        int phiDims[3];
        phiDims[0] = pSPARC->npNdx_phi;
        phiDims[1] = pSPARC->npNdy_phi;
        phiDims[2] = pSPARC->npNdz_phi;

        vdWDF_Setup_Comms(pSPARC, gridsizes, phiDims); // how many processors are used to make FFT/iFFT?

        pSPARC->vdWDFkernelReciPoints = (double**)malloc(sizeof(double*)*numberKernel);
        for (q1 = 0; q1 < nqs; q1++) {
            for (q2 = 0; q2 < q1 + 1; q2++) {
                qpair = kernel_label(q1, q2, nqs);
                pSPARC->vdWDFkernelReciPoints[qpair] = (double*)malloc(sizeof(double)*DMnd); assert(pSPARC->vdWDFkernelReciPoints[qpair] != NULL);
            }
        }
        pSPARC->vdWDFuFTs = (double _Complex**)malloc(sizeof(double _Complex*)*nqs);
        pSPARC->vdWDFu = (double**)malloc(sizeof(double)*nqs);
        for (q1 = 0; q1 < nqs; q1++) {
            pSPARC->vdWDFuFTs[q1] = (double _Complex*)malloc(sizeof(double _Complex)*DMnd); assert(pSPARC->vdWDFuFTs[q1] != NULL);
            pSPARC->vdWDFu[q1] = (double*)malloc(sizeof(double)*DMnd); assert(pSPARC->vdWDFu[q1] != NULL);
        }
        pSPARC->vdWDFpotential = (double*)malloc(DMnd * pSPARC->Nspin * sizeof(double)); assert(pSPARC->vdWDFpotential != NULL);
        // if (rank == size - 1) {
    	// 	char vdWDFoutputRoute[L_STRING];
        //     snprintf(vdWDFoutputRoute,       L_STRING, "%svdWDFoutput.txt"  ,     folderRoute);
    	// 	pSPARC->vdWDFOutput = fopen(vdWDFoutputRoute,"w");
    	// 	fprintf(pSPARC->vdWDFOutput, "                    vdWDF Energy and Stress Calculation                    \n");
    	// }
        
    }
}


int domain_index1D(int locali, int localj, int localk, int DMnx, int DMny, int DMnz) {
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

void read_spline_d2_qmesh(SPARC_OBJ *pSPARC) {
    double **d2Splineydx2 = pSPARC->vdWDFd2Splineydx2;
    d2Splineydx2[0][0]= 0.000000000; d2Splineydx2[0][1]=733.243286433; d2Splineydx2[0][2]=-180.928502951; d2Splineydx2[0][3]=44.644286263; d2Splineydx2[0][4]=-11.016021597; d2Splineydx2[0][5]= 2.718214176; d2Splineydx2[0][6]=-0.670722025; d2Splineydx2[0][7]= 0.165501320; d2Splineydx2[0][8]=-0.040837614; d2Splineydx2[0][9]= 0.010076721; d2Splineydx2[0][10]=-0.002486441; d2Splineydx2[0][11]= 0.000613532; d2Splineydx2[0][12]=-0.000151390; d2Splineydx2[0][13]= 0.000037356; d2Splineydx2[0][14]=-0.000009217; d2Splineydx2[0][15]= 0.000002274; d2Splineydx2[0][16]=-0.000000561; d2Splineydx2[0][17]= 0.000000138; d2Splineydx2[0][18]=-0.000000032; d2Splineydx2[0][19]= 0.000000000; 
    d2Splineydx2[1][0]= 0.000000000; d2Splineydx2[1][1]=-1513.892465312; d2Splineydx2[1][2]=908.307460137; d2Splineydx2[1][3]=-224.125760198; d2Splineydx2[1][4]=55.303252025; d2Splineydx2[1][5]=-13.646131894; d2Splineydx2[1][6]= 3.367196482; d2Splineydx2[1][7]=-0.830859048; d2Splineydx2[1][8]= 0.205015288; d2Splineydx2[1][9]=-0.050587724; d2Splineydx2[1][10]= 0.012482571; d2Splineydx2[1][11]=-0.003080087; d2Splineydx2[1][12]= 0.000760014; d2Splineydx2[1][13]=-0.000187534; d2Splineydx2[1][14]= 0.000046274; d2Splineydx2[1][15]=-0.000011418; d2Splineydx2[1][16]= 0.000002816; d2Splineydx2[1][17]=-0.000000692; d2Splineydx2[1][18]= 0.000000159; d2Splineydx2[1][19]= 0.000000000; 
    d2Splineydx2[2][0]= 0.000000000; d2Splineydx2[2][1]=945.177072923; d2Splineydx2[2][2]=-1337.429120463; d2Splineydx2[2][3]=720.135986823; d2Splineydx2[2][4]=-177.694263866; d2Splineydx2[2][5]=43.846234584; d2Splineydx2[2][6]=-10.819101559; d2Splineydx2[2][7]= 2.669623963; d2Splineydx2[2][8]=-0.658732342; d2Splineydx2[2][9]= 0.162542854; d2Splineydx2[2][10]=-0.040107609; d2Splineydx2[2][11]= 0.009896592; d2Splineydx2[2][12]=-0.002441994; d2Splineydx2[2][13]= 0.000602564; d2Splineydx2[2][14]=-0.000148683; d2Splineydx2[2][15]= 0.000036687; d2Splineydx2[2][16]=-0.000009049; d2Splineydx2[2][17]= 0.000002222; d2Splineydx2[2][18]=-0.000000512; d2Splineydx2[2][19]= 0.000000000; 
    d2Splineydx2[3][0]= 0.000000000; d2Splineydx2[3][1]=-199.203429035; d2Splineydx2[3][2]=738.622986179; d2Splineydx2[3][3]=-987.818887409; d2Splineydx2[3][4]=528.356610610; d2Splineydx2[3][5]=-130.372513940; d2Splineydx2[3][6]=32.169546192; d2Splineydx2[3][7]=-7.937867200; d2Splineydx2[3][8]= 1.958676548; d2Splineydx2[3][9]=-0.483305367; d2Splineydx2[3][10]= 0.119256075; d2Splineydx2[3][11]=-0.029426554; d2Splineydx2[3][12]= 0.007261031; d2Splineydx2[3][13]=-0.001791666; d2Splineydx2[3][14]= 0.000442095; d2Splineydx2[3][15]=-0.000109085; d2Splineydx2[3][16]= 0.000026908; d2Splineydx2[3][17]=-0.000006608; d2Splineydx2[3][18]= 0.000001522; d2Splineydx2[3][19]= 0.000000000; 
    d2Splineydx2[4][0]= 0.000000000; d2Splineydx2[4][1]=41.983674039; d2Splineydx2[4][2]=-155.670546637; d2Splineydx2[4][3]=541.407748909; d2Splineydx2[4][4]=-721.283775457; d2Splineydx2[4][5]=385.613004308; d2Splineydx2[4][6]=-95.150388526; d2Splineydx2[4][7]=23.478452063; d2Splineydx2[4][8]=-5.793331166; d2Splineydx2[4][9]= 1.429510170; d2Splineydx2[4][10]=-0.352733042; d2Splineydx2[4][11]= 0.087037225; d2Splineydx2[4][12]=-0.021476521; d2Splineydx2[4][13]= 0.005299352; d2Splineydx2[4][14]=-0.001307618; d2Splineydx2[4][15]= 0.000322648; d2Splineydx2[4][16]=-0.000079587; d2Splineydx2[4][17]= 0.000019545; d2Splineydx2[4][18]=-0.000004502; d2Splineydx2[4][19]= 0.000000000; 
    d2Splineydx2[5][0]= 0.000000000; d2Splineydx2[5][1]=-8.848386267; d2Splineydx2[5][2]=32.808780046; d2Splineydx2[5][3]=-114.105899496; d2Splineydx2[5][4]=395.111672021; d2Splineydx2[5][5]=-526.238559571; d2Splineydx2[5][6]=281.328429233; d2Splineydx2[5][7]=-69.418066937; d2Splineydx2[5][8]=17.128976373; d2Splineydx2[5][9]=-4.226591787; d2Splineydx2[5][10]= 1.042915685; d2Splineydx2[5][11]=-0.257340472; d2Splineydx2[5][12]= 0.063499014; d2Splineydx2[5][13]=-0.015668442; d2Splineydx2[5][14]= 0.003866197; d2Splineydx2[5][15]=-0.000953965; d2Splineydx2[5][16]= 0.000235313; d2Splineydx2[5][17]=-0.000057790; d2Splineydx2[5][18]= 0.000013311; d2Splineydx2[5][19]= 0.000000000; 
    d2Splineydx2[6][0]= 0.000000000; d2Splineydx2[6][1]= 1.864866316; d2Splineydx2[6][2]=-6.914705905; d2Splineydx2[6][3]=24.048706961; d2Splineydx2[6][4]=-83.272861957; d2Splineydx2[6][5]=288.256914277; d2Splineydx2[6][6]=-383.914104895; d2Splineydx2[6][7]=205.240935168; d2Splineydx2[6][8]=-50.643402853; d2Splineydx2[6][9]=12.496309523; d2Splineydx2[6][10]=-3.083476680; d2Splineydx2[6][11]= 0.760850907; d2Splineydx2[6][12]=-0.187740709; d2Splineydx2[6][13]= 0.046325199; d2Splineydx2[6][14]=-0.011430769; d2Splineydx2[6][15]= 0.002820486; d2Splineydx2[6][16]=-0.000695724; d2Splineydx2[6][17]= 0.000170860; d2Splineydx2[6][18]=-0.000039355; d2Splineydx2[6][19]= 0.000000000; 
    d2Splineydx2[7][0]= 0.000000000; d2Splineydx2[7][1]=-0.393035099; d2Splineydx2[7][2]= 1.457328120; d2Splineydx2[7][3]=-5.068452280; d2Splineydx2[7][4]=17.550404176; d2Splineydx2[7][5]=-60.752389591; d2Splineydx2[7][6]=210.295485094; d2Splineydx2[7][7]=-280.081020244; d2Splineydx2[7][8]=149.731619642; d2Splineydx2[7][9]=-36.946424587; d2Splineydx2[7][10]= 9.116566648; d2Splineydx2[7][11]=-2.249521796; d2Splineydx2[7][12]= 0.555071713; d2Splineydx2[7][13]=-0.136964475; d2Splineydx2[7][14]= 0.033796061; d2Splineydx2[7][15]=-0.008339011; d2Splineydx2[7][16]= 0.002056967; d2Splineydx2[7][17]=-0.000505163; d2Splineydx2[7][18]= 0.000116355; d2Splineydx2[7][19]= 0.000000000; 
    d2Splineydx2[8][0]= 0.000000000; d2Splineydx2[8][1]= 0.082835208; d2Splineydx2[8][2]=-0.307143251; d2Splineydx2[8][3]= 1.068215790; d2Splineydx2[8][4]=-3.698884360; d2Splineydx2[8][5]=12.804039238; d2Splineydx2[8][6]=-44.321411238; d2Splineydx2[8][7]=153.419115908; d2Splineydx2[8][8]=-204.330483982; d2Splineydx2[8][9]=109.235298810; d2Splineydx2[8][10]=-26.953917544; d2Splineydx2[8][11]= 6.650905691; d2Splineydx2[8][12]=-1.641117514; d2Splineydx2[8][13]= 0.404947313; d2Splineydx2[8][14]=-0.099920978; d2Splineydx2[8][15]= 0.024655008; d2Splineydx2[8][16]=-0.006081601; d2Splineydx2[8][17]= 0.001493557; d2Splineydx2[8][18]=-0.000344014; d2Splineydx2[8][19]= 0.000000000; 
    d2Splineydx2[9][0]= 0.000000000; d2Splineydx2[9][1]=-0.017458165; d2Splineydx2[9][2]= 0.064732832; d2Splineydx2[9][3]=-0.225134797; d2Splineydx2[9][4]= 0.779568685; d2Splineydx2[9][5]=-2.698550986; d2Splineydx2[9][6]= 9.341082588; d2Splineydx2[9][7]=-32.334273486; d2Splineydx2[9][8]=111.925476933; d2Splineydx2[9][9]=-149.067386907; d2Splineydx2[9][10]=79.691586980; d2Splineydx2[9][11]=-19.663977547; d2Splineydx2[9][12]= 4.852105781; d2Splineydx2[9][13]=-1.197261733; d2Splineydx2[9][14]= 0.295425007; d2Splineydx2[9][15]=-0.072894663; d2Splineydx2[9][16]= 0.017980779; d2Splineydx2[9][17]=-0.004415829; d2Splineydx2[9][18]= 0.001017107; d2Splineydx2[9][19]= 0.000000000; 
    d2Splineydx2[10][0]= 0.000000000; d2Splineydx2[10][1]= 0.003679444; d2Splineydx2[10][2]=-0.013642948; d2Splineydx2[10][3]= 0.047448912; d2Splineydx2[10][4]=-0.164300172; d2Splineydx2[10][5]= 0.568740636; d2Splineydx2[10][6]=-1.968705902; d2Splineydx2[10][7]= 6.814699952; d2Splineydx2[10][8]=-23.589165924; d2Splineydx2[10][9]=81.654181171; d2Splineydx2[10][10]=-108.750712919; d2Splineydx2[10][11]=58.138249234; d2Splineydx2[10][12]=-14.345670123; d2Splineydx2[10][13]= 3.539807798; d2Splineydx2[10][14]=-0.873449567; d2Splineydx2[10][15]= 0.215519372; d2Splineydx2[10][16]=-0.053161727; d2Splineydx2[10][17]= 0.013055781; d2Splineydx2[10][18]=-0.003007164; d2Splineydx2[10][19]= 0.000000000; 
    d2Splineydx2[11][0]= 0.000000000; d2Splineydx2[11][1]=-0.000775472; d2Splineydx2[11][2]= 0.002875358; d2Splineydx2[11][3]=-0.010000228; d2Splineydx2[11][4]= 0.034627541; d2Splineydx2[11][5]=-0.119866518; d2Splineydx2[11][6]= 0.414920101; d2Splineydx2[11][7]=-1.436251086; d2Splineydx2[11][8]= 4.971600425; d2Splineydx2[11][9]=-17.209254585; d2Splineydx2[11][10]=59.570041409; d2Splineydx2[11][11]=-79.338061730; d2Splineydx2[11][12]=42.414213654; d2Splineydx2[11][13]=-10.465747709; d2Splineydx2[11][14]= 2.582429138; d2Splineydx2[11][15]=-0.637201652; d2Splineydx2[11][16]= 0.157177241; d2Splineydx2[11][17]=-0.038600545; d2Splineydx2[11][18]= 0.008890939; d2Splineydx2[11][19]= 0.000000000; 
    d2Splineydx2[12][0]= 0.000000000; d2Splineydx2[12][1]= 0.000163437; d2Splineydx2[12][2]=-0.000606004; d2Splineydx2[12][3]= 0.002107626; d2Splineydx2[12][4]=-0.007298024; d2Splineydx2[12][5]= 0.025262802; d2Splineydx2[12][6]=-0.087447642; d2Splineydx2[12][7]= 0.302701100; d2Splineydx2[12][8]=-1.047803500; d2Splineydx2[12][9]= 3.626984401; d2Splineydx2[12][10]=-12.554850060; d2Splineydx2[12][11]=43.458764249; d2Splineydx2[12][12]=-57.880337150; d2Splineydx2[12][13]=30.942887685; d2Splineydx2[12][14]=-7.635174954; d2Splineydx2[12][15]= 1.883941760; d2Splineydx2[12][16]=-0.464708098; d2Splineydx2[12][17]= 0.114125847; d2Splineydx2[12][18]=-0.026286830; d2Splineydx2[12][19]= 0.000000000; 
    d2Splineydx2[13][0]= 0.000000000; d2Splineydx2[13][1]=-0.000034446; d2Splineydx2[13][2]= 0.000127720; d2Splineydx2[13][3]=-0.000444199; d2Splineydx2[13][4]= 0.001538115; d2Splineydx2[13][5]=-0.005324332; d2Splineydx2[13][6]= 0.018430270; d2Splineydx2[13][7]=-0.063796608; d2Splineydx2[13][8]= 0.220832725; d2Splineydx2[13][9]=-0.764415131; d2Splineydx2[13][10]= 2.646032157; d2Splineydx2[13][11]=-9.159272086; d2Splineydx2[13][12]=31.704930318; d2Splineydx2[13][13]=-42.226045534; d2Splineydx2[13][14]=22.574054681; d2Splineydx2[13][15]=-5.570036646; d2Splineydx2[13][16]= 1.373949657; d2Splineydx2[13][17]=-0.337422930; d2Splineydx2[13][18]= 0.077719286; d2Splineydx2[13][19]= 0.000000000; 
    d2Splineydx2[14][0]= 0.000000000; d2Splineydx2[14][1]= 0.000007260; d2Splineydx2[14][2]=-0.000026918; d2Splineydx2[14][3]= 0.000093618; d2Splineydx2[14][4]=-0.000324169; d2Splineydx2[14][5]= 0.001122143; d2Splineydx2[14][6]=-0.003884317; d2Splineydx2[14][7]= 0.013445613; d2Splineydx2[14][8]=-0.046542150; d2Splineydx2[14][9]= 0.161106213; d2Splineydx2[14][10]=-0.557671089; d2Splineydx2[14][11]= 1.930385173; d2Splineydx2[14][12]=-6.682051457; d2Splineydx2[14][13]=23.130001359; d2Splineydx2[14][14]=-30.805501080; d2Splineydx2[14][15]=16.468294779; d2Splineydx2[14][16]=-4.062200914; d2Splineydx2[14][17]= 0.997620055; d2Splineydx2[14][18]=-0.229783786; d2Splineydx2[14][19]= 0.000000000; 
    d2Splineydx2[15][0]= 0.000000000; d2Splineydx2[15][1]=-0.000001530; d2Splineydx2[15][2]= 0.000005673; d2Splineydx2[15][3]=-0.000019730; d2Splineydx2[15][4]= 0.000068320; d2Splineydx2[15][5]=-0.000236496; d2Splineydx2[15][6]= 0.000818634; d2Splineydx2[15][7]=-0.002833711; d2Splineydx2[15][8]= 0.009808925; d2Splineydx2[15][9]=-0.033953710; d2Splineydx2[15][10]= 0.117531176; d2Splineydx2[15][11]=-0.406835576; d2Splineydx2[15][12]= 1.408266231; d2Splineydx2[15][13]=-4.874730472; d2Splineydx2[15][14]=16.873938067; d2Splineydx2[15][15]=-22.472696530; d2Splineydx2[15][16]=12.010248111; d2Splineydx2[15][17]=-2.949549921; d2Splineydx2[15][18]= 0.679375625; d2Splineydx2[15][19]= 0.000000000; 
    d2Splineydx2[16][0]= 0.000000000; d2Splineydx2[16][1]= 0.000000322; d2Splineydx2[16][2]=-0.000001195; d2Splineydx2[16][3]= 0.000004157; d2Splineydx2[16][4]=-0.000014395; d2Splineydx2[16][5]= 0.000049829; d2Splineydx2[16][6]=-0.000172485; d2Splineydx2[16][7]= 0.000597059; d2Splineydx2[16][8]=-0.002066728; d2Splineydx2[16][9]= 0.007154002; d2Splineydx2[16][10]=-0.024763664; d2Splineydx2[16][11]= 0.085719721; d2Splineydx2[16][12]=-0.296719842; d2Splineydx2[16][13]= 1.027099297; d2Splineydx2[16][14]=-3.555316550; d2Splineydx2[16][15]=12.306770931; d2Splineydx2[16][16]=-16.382794982; d2Splineydx2[16][17]= 8.720599280; d2Splineydx2[16][18]=-2.008632756; d2Splineydx2[16][19]= 0.000000000; 
    d2Splineydx2[17][0]= 0.000000000; d2Splineydx2[17][1]=-0.000000068; d2Splineydx2[17][2]= 0.000000251; d2Splineydx2[17][3]=-0.000000873; d2Splineydx2[17][4]= 0.000003022; d2Splineydx2[17][5]=-0.000010461; d2Splineydx2[17][6]= 0.000036210; d2Splineydx2[17][7]=-0.000125340; d2Splineydx2[17][8]= 0.000433865; d2Splineydx2[17][9]=-0.001501828; d2Splineydx2[17][10]= 0.005198597; d2Splineydx2[17][11]=-0.017995006; d2Splineydx2[17][12]= 0.062289930; d2Splineydx2[17][13]=-0.215617342; d2Splineydx2[17][14]= 0.746362018; d2Splineydx2[17][15]=-2.583541087; d2Splineydx2[17][16]= 8.942958497; d2Splineydx2[17][17]=-11.829616593; d2Splineydx2[17][18]= 5.938696353; d2Splineydx2[17][19]= 0.000000000; 
    d2Splineydx2[18][0]= 0.000000000; d2Splineydx2[18][1]= 0.000000013; d2Splineydx2[18][2]=-0.000000050; d2Splineydx2[18][3]= 0.000000174; d2Splineydx2[18][4]=-0.000000602; d2Splineydx2[18][5]= 0.000002082; d2Splineydx2[18][6]=-0.000007208; d2Splineydx2[18][7]= 0.000024952; d2Splineydx2[18][8]=-0.000086373; d2Splineydx2[18][9]= 0.000298980; d2Splineydx2[18][10]=-0.001034923; d2Splineydx2[18][11]= 0.003582397; d2Splineydx2[18][12]=-0.012400510; d2Splineydx2[18][13]= 0.042924513; d2Splineydx2[18][14]=-0.148583718; d2Splineydx2[18][15]= 0.514324324; d2Splineydx2[18][16]=-1.780339823; d2Splineydx2[18][17]= 6.162667676; d2Splineydx2[18][18]=-7.378559242; d2Splineydx2[18][19]= 0.000000000; 
    d2Splineydx2[19][0]= 0.000000000; d2Splineydx2[19][1]=-0.000000002; d2Splineydx2[19][2]= 0.000000007; d2Splineydx2[19][3]=-0.000000024; d2Splineydx2[19][4]= 0.000000083; d2Splineydx2[19][5]=-0.000000287; d2Splineydx2[19][6]= 0.000000994; d2Splineydx2[19][7]=-0.000003440; d2Splineydx2[19][8]= 0.000011907; d2Splineydx2[19][9]=-0.000041217; d2Splineydx2[19][10]= 0.000142673; d2Splineydx2[19][11]=-0.000493865; d2Splineydx2[19][12]= 0.001709521; d2Splineydx2[19][13]=-0.005917528; d2Splineydx2[19][14]= 0.020483595; d2Splineydx2[19][15]=-0.070904211; d2Splineydx2[19][16]= 0.245435778; d2Splineydx2[19][17]=-0.849578891; d2Splineydx2[19][18]= 2.940827534; d2Splineydx2[19][19]= 0.000000000; 
}