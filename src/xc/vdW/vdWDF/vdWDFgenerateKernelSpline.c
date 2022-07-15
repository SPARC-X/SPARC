/**
 * @file    vdWDFgenerateKernelSpline.c
 * @brief   This file contains the functions for generating the needed model kernel functions \Phi in reciprocal space
 *  and the value of spline functions at model energy ratios.
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
#include "tools.h"
#include "parallelization.h"
#include "vdWDFinitialization.h"
#include "vdWDFgenerateKernelSpline.h"

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

/*
 Structure of functions in vdWDF.c
 vdWDF_generate_kernel // GENERATE needed kernel functions (phi), its 2nd derivatives and 2nd derivatives of spline functions of model energy ratios
 ├──kernel_label
 ├──prepare_Gauss_quad
 ├──phi_value
 ├──h_function
 ├──radial_FT
 ├──d2_of_kernel
 ├──spline_d2_qmesh
 ├──print_kernel
 └──print_d2ydx2
 */

/*
The functions below are for generating kernel functions and 2nd derivative of spline functions
not directly rely on parameters in SPARC
*/

// called in initialization
void vdWDF_generate_kernel(char *filename) { // let the function unrelated to pSPARC
    printf("begin generating kernel and 2nd derivative of spline functions.\n");
    // generate the needed files into the current test folder
    double t1,t2;
    t1 = MPI_Wtime();
    // Initialization, set parameters and grids of model kernel functions
    int nrpoints = 1024; // radial points for composing Phi functions in real and reciprocal space
    double rMax = 100.0; // max radius in real space and minimum k point 2*pi/r_max in reciprocal space
    double vdWdr = rMax / nrpoints; 
    double vdWdk = 2.0 * M_PI / rMax; 
    int nqs = 20; // number of model energy ratio, they are fixed values

    double qmesh[20] = { //model energy ratios
    1.0e-5           , 0.0449420825586261, 0.0975593700991365, 0.159162633466142,
    0.231286496836006, 0.315727667369529 , 0.414589693721418 , 0.530335368404141,
    0.665848079422965, 0.824503639537924 , 1.010254382520950 , 1.227727621364570,
    1.482340921174910, 1.780437058359530 , 2.129442028133640 , 2.538050036534580,
    3.016440085356680, 3.576529545442460 , 4.232271035198720 , 5.0};
    double*qmeshPointer = qmesh;
    int numberKernel = nqs*(nqs - 1)/2 + nqs;
    double **vdWDFkernelPhi = (double**)malloc(sizeof(double*)*numberKernel); // kernal Phi, index 0, reciprocal
    double **vdWDFd2Phidk2 = (double**)malloc(sizeof(double*)*numberKernel); // 2nd derivative of kernals
    double **vdWDFd2Splineydx2 = (double**)malloc((sizeof(double*)*nqs)); // 2nd derivative of spline function at qmesh points
    int q1, q2, qpair;
    for (q1 = 0; q1 < nqs; q1++) {
        for (q2 = 0; q2 < q1 + 1; q2++) { // q2 is smaller than q1
            qpair = kernel_label(q1, q2, nqs);
            vdWDFkernelPhi[qpair] = (double*)calloc((1 + nrpoints), sizeof(double));
            vdWDFd2Phidk2[qpair] = (double*)calloc((1 + nrpoints), sizeof(double));
        }
        vdWDFd2Splineydx2[q1] = (double*)calloc(nqs, sizeof(double));
    }
    // Generate Wab(a, b) function
    int nIntegratePoints = 256; // Number of imagine frequency points integrated for building kernel
    double aMin = 0.0; double aMax = 64.0; // scope of integration of imagine frequency (a, b)
    double *aPoints = (double*)malloc(sizeof(double)*nIntegratePoints); // imagine frequency points
    double *aPoints2 = (double*)malloc(sizeof(double)*nIntegratePoints); // imagine frequency points
    double *weights = (double*)malloc(sizeof(double)*nIntegratePoints);
    int intpoint, intpoint2;

    prepare_Gauss_quad(nIntegratePoints, aMin, aMax, aPoints, aPoints2, weights); // Gaussian quadrature integration points and weights
    double sin_a[256]; double cos_a[256];
    for (intpoint = 0; intpoint < nIntegratePoints; intpoint++) {
        aPoints[intpoint] = tan(aPoints[intpoint]);
        aPoints2[intpoint] = aPoints[intpoint]*aPoints[intpoint];
        weights[intpoint] = weights[intpoint]*(1.0 + aPoints2[intpoint]);
        sin_a[intpoint] = sin(aPoints[intpoint]);
        cos_a[intpoint] = cos(aPoints[intpoint]);
    }
    double **WabMatrix = (double **)malloc(sizeof(double*)*nIntegratePoints);
    int intpointa, intpointb;
    for (intpointa = 0; intpointa < nIntegratePoints; intpointa++) {
        WabMatrix[intpointa] = (double *)malloc(sizeof(double)*nIntegratePoints);
    }
    for (intpointa = 0; intpointa < nIntegratePoints; intpointa++) {
        for (intpointb = 0; intpointb < nIntegratePoints; intpointb++) {
            double coef1 = 2.0*weights[intpointa]*weights[intpointb];
            double part1 = (3.0 - aPoints2[intpointa])*aPoints[intpointb]*sin_a[intpointa]*cos_a[intpointb] + 
                (3.0 - aPoints2[intpointb])*aPoints[intpointa]*cos_a[intpointa]*sin_a[intpointb];
            double part2 = (aPoints2[intpointa] + aPoints2[intpointb] - 3.0)*sin_a[intpointa]*sin_a[intpointb];
            double part3 = 3.0 * aPoints[intpointa]*aPoints[intpointb]*cos_a[intpointa]*cos_a[intpointb];
            double coef2 = aPoints[intpointa]*aPoints[intpointb];
            WabMatrix[intpointa][intpointb] = coef1*(part1 + part2 - part3)/coef2;
        }
    }
    // Compute kernal function Phi in reciprocal space
    double *realKernel = (double*)malloc(sizeof(double)*nrpoints);
    for (q1 = 0; q1 < nqs; q1++) {
        for (q2 = 0; q2 < q1 + 1; q2++) {
            qpair = kernel_label(q1, q2, nqs);
            printf("generating kernel: q1 %d q2 %d, qpair %d\n", q1, q2, qpair);
            phi_value(nrpoints, nIntegratePoints, WabMatrix, aPoints, aPoints2, qmesh[q1], qmesh[q2], vdWdr, realKernel); //model kernel functions in real space
            radial_FT(realKernel, nrpoints, vdWdr, vdWdk, vdWDFkernelPhi[qpair]); // transform the function to reciprocal space for future use
            d2_of_kernel(vdWDFkernelPhi[qpair], nrpoints, vdWdk, vdWDFd2Phidk2[qpair]); // compute 2nd order derivative of kernel function for spline interpolation
        }
    }
    // print computed kernel and 2nd derivative of kernel
    char folderRoute[L_STRING];
    strncpy(folderRoute, filename,sizeof(folderRoute));
    int indexEndFolderRoute = 0;
    while (folderRoute[indexEndFolderRoute] != '\0') {
        indexEndFolderRoute++;
    } // get the length of the char filename
    while ((indexEndFolderRoute > -1) && (folderRoute[indexEndFolderRoute] != '/')) {
        indexEndFolderRoute--;
    } // find the last '/'. If there is no '/', indexEndFolderRoute should be at the beginning
    folderRoute[indexEndFolderRoute + 1] = '\0'; // cut the string. Now it contains only the folder position
    // char *kernelFileRoute = "kernel_d2.txt"; // name of the file of the output kernels and 2nd derivative of kernels
    char kernelFileRoute[L_STRING]; // name of the file of the output kernels and 2nd derivative of kernels
    snprintf(kernelFileRoute,       L_STRING, "%skernel_d2.txt"  ,     folderRoute);

    print_kernel(kernelFileRoute, vdWDFkernelPhi, vdWDFd2Phidk2, nrpoints, nqs);

    // Compute the 2nd derivative of spline function at qmesh points, then print them. Used in spline interpolation
    spline_d2_qmesh(qmeshPointer, nqs, vdWDFd2Splineydx2);
    // char *d2splineFileRoute = "spline_d2.txt"; //name of the file of the 2nd derivatives of spline functions
    char d2splineFileRoute[L_STRING]; //name of the file of the 2nd derivatives of spline functions
    snprintf(d2splineFileRoute,       L_STRING, "%sspline_d2.txt"  ,     folderRoute);
    print_d2ydx2(d2splineFileRoute, nqs, vdWDFd2Splineydx2);

    t2 = MPI_Wtime();
    printf("end of generating kernel function and 2nd derivative of spline functions.\n");
    printf("Generation spent %.3f s\n", t2 - t1);

    for (qpair = 0; qpair < numberKernel; qpair++) {
        free(vdWDFkernelPhi[qpair]);
        free(vdWDFd2Phidk2[qpair]);
    }
    for (q1 = 0; q1 < nqs; q1++) {
        free(vdWDFd2Splineydx2[q1]);
    }
    free(realKernel);
    free(vdWDFkernelPhi);
    free(vdWDFd2Phidk2);
    free(vdWDFd2Splineydx2);

    free(aPoints);
    free(aPoints2);
    free(weights);
    for (intpointa = 0; intpointa < nIntegratePoints; intpointa++) {
        free(WabMatrix[intpointa]);
    }
    free(WabMatrix);

}

// find suitable Gaussian quadrature integration points and weights
void prepare_Gauss_quad(int nIntegratePoints, double aMin, double aMax, double *aPoints, double *aPoints2, double *weights) {
    int npoints = nIntegratePoints / 2;
    double midPoint = 0.5 * (atan(aMin) + atan(aMax));
    double lengthScope = 0.5 * (atan(aMax) - atan(aMin));
    int ipoint, iPoly;
    for (ipoint = 0; ipoint < npoints; ipoint++) {
        double root = cos((M_PI*(ipoint + 1 - 0.25) / (nIntegratePoints + 0.5)));
        int rootFlag = 1;
        double dpdx;
        while (rootFlag) { // for all Npoints
            double poly1 = 1.0;
            double poly2 = 0.0;
            double poly3;
            for (iPoly = 1; iPoly < nIntegratePoints + 1; iPoly++) {
                poly3 = poly2;
                poly2 = poly1;
                poly1 = ((2.0*iPoly - 1.0)*root*poly2 - ((double)iPoly - 1.0)*poly3)/(double)iPoly;
            }
            dpdx = nIntegratePoints * (root*poly1 - poly2) / (root*root - 1.0);
            double last_root = root;
            root = last_root - poly1/dpdx;
            if (fabs(root - last_root) <= 1e-14) {
                rootFlag = 0;
            }
        }
        aPoints[ipoint] = midPoint - lengthScope*root;
        aPoints[nIntegratePoints - ipoint - 1] = midPoint + lengthScope*root;
        weights[ipoint] = 2.0*lengthScope / ((1.0 - root*root) * dpdx*dpdx);
        weights[nIntegratePoints - ipoint - 1] = weights[ipoint];
    }
}

// calculate model kernel functions in real space
void phi_value(int nrpoints, int nIntegratePoints, double **WabMatrix, double *aPoints, double *aPoints2,
 double qmesh1, double qmesh2, double vdWdr, double *realKernel) {
    int ir, ii1, ii2;
    double *nu = (double*)malloc(sizeof(double)*nIntegratePoints);
    double *nu1 = (double*)malloc(sizeof(double)*nIntegratePoints);
    for (ir = 0; ir < nrpoints; ir++) { // r-r', radial points
        double d1 = (ir + 1)*vdWdr*qmesh1;
        double d2 = (ir + 1)*vdWdr*qmesh2;
        for (ii1 = 0; ii1 < nIntegratePoints; ii1++) {
            nu[ii1] = aPoints2[ii1] / (2.0 * h_function(aPoints[ii1] / d1));
            nu1[ii1] = aPoints2[ii1] / (2.0 * h_function(aPoints[ii1] / d2));
        }
        realKernel[ir] = 0.0;
        double w, y, x, z, T;
        for (ii1 = 0; ii1 < nIntegratePoints; ii1++) { // a in M Dion's paper (14)
            w = nu[ii1];
            y = nu1[ii1];
            for (ii2 = 0; ii2 < nIntegratePoints; ii2++) { // b in M Dion's paper (14)
                x = nu[ii2];
                z = nu1[ii2];
                T = (1.0/(w + x) + 1.0/(y + z)) * (1.0/((w + y)*(x + z)) + 1.0/((w + z)*(y + x)));
                realKernel[ir] += T * WabMatrix[ii1][ii2]; // integrate on 2-D surface ab
            }
        }
        realKernel[ir] *= 1.0 / (M_PI*M_PI);
    }
    free(nu);
    free(nu1);
}

// in Dion's paper (6)
double h_function(double aDivd) {
    double g1 = 4.0*M_PI/9.0;
    return 1.0 - exp(-g1*aDivd*aDivd);
}

// transform calculated kernel function in real space, spherical coordinate into reciprocal space, spherical coordinate
void radial_FT(double* realKernel, int nrpoints, double vdWdr, double vdWdk, double* reciKernel) {
    int ik = 0; // radial point in reciprocal space
    int ir; // radial point in real space
    double r; // the distance between the center and radial point in real space
    double k; // the distance between the center and radial point in reciprocal space
    // reciKernel[0] = 0.0; // calloc
    // handle ik = 0 seperately
    for (ir = 1; ir < nrpoints + 1; ir++) {//r_i = 1, Nr_points
        r = ir*vdWdr;
        reciKernel[0] += 4*M_PI*realKernel[ir - 1]*r*r;
    }
    reciKernel[0] -= 4*M_PI*0.5*(nrpoints*vdWdr)*(nrpoints*vdWdr)*realKernel[nrpoints - 1];
    reciKernel[0] *= vdWdr;
    // compute reciKernel
    for (ik = 1; ik < nrpoints + 1; ik++) {
        // reciKernel[ik] = 0.0; // calloc
        k = ik*vdWdk;
        for (ir = 1; ir < nrpoints + 1; ir++) {
            r = ir*vdWdr;
            reciKernel[ik] += 4*M_PI*realKernel[ir - 1]*r*sin(k*r)/k; // spherical Bessel function
        }
        reciKernel[ik] -= 4*M_PI*0.5*realKernel[ir - 1]*r*sin(k*r)/k;
        reciKernel[ik] *= vdWdr;
    }
}

// compute the 2nd order derivative of the kernel function in recirpocal space
void d2_of_kernel(double* reciKernel, int nrpoints, double vdWdk, double* d2reciKerneldk2) { // set_up_splines in QE, d2ForSplines in m
    double* tempArray = (double*)malloc(sizeof(double)*(nrpoints + 1));
    // d2reciKerneldk2[0] = 0.0; // calloc
    tempArray[0] = 0.0;
    int ik;
    double temp1, temp2;
    for (ik = 1; ik < nrpoints; ik++) {
        temp1 = 0.5;
        temp2 = temp1*d2reciKerneldk2[ik - 1] + 2.0;
        d2reciKerneldk2[ik] = (temp1 - 1.0)/temp2;
        tempArray[ik] = (reciKernel[ik + 1] - reciKernel[ik])/vdWdk - (reciKernel[ik] - reciKernel[ik - 1])/vdWdk;
        tempArray[ik] = (6.0*tempArray[ik]/(2*vdWdk) - temp1*tempArray[ik - 1]) / temp2;
    }
    // d2reciKerneldk2[nrpoints] = 0.0; // calloc
    for (ik = nrpoints - 1; ik > -1; ik--) {
        d2reciKerneldk2[ik] = d2reciKerneldk2[ik]*d2reciKerneldk2[ik + 1] + tempArray[ik];
    }
    free(tempArray);
}

// compute the 2nd derivative of spline function at qmesh points. It is only related to qmesh and independent from kernel functions
void spline_d2_qmesh(double* qmesh, int nqs, double** d2ydx2) { // initialize_spline_interpolation in QE, splineFunc_2Deri_atQmesh in m
    int q1, q2;
    double* y = (double*)malloc(sizeof(double)*nqs);
    double* tempArray = (double*)malloc(sizeof(double)*nqs);
    double temp1, temp2;

    for (q1 = 0; q1 < nqs; q1++) {
        for (int yq = 0; yq < nqs; yq++) {
            y[yq] = 0.0;
            tempArray[yq] = 0.0;
        }
        y[q1] = 1.0;
        for (q2 = 1; q2 < nqs - 1; q2++) {
            temp1 = (qmesh[q2] - qmesh[q2 - 1])/(qmesh[q2 + 1] - qmesh[q2 - 1]);
            temp2 = temp1*d2ydx2[q1][q2 - 1] + 2.0;
            d2ydx2[q1][q2] = (temp1 - 1.0) / temp2;
            tempArray[q2] = (y[q2 + 1] - y[q2])/(qmesh[q2 + 1] - qmesh[q2]) - (y[q2] - y[q2-1])/(qmesh[q2] - qmesh[q2 - 1]);
            tempArray[q2] = (6.0*tempArray[q2]/(qmesh[q2 + 1] - qmesh[q2 - 1]) - temp1*tempArray[q2 - 1]) / temp2;
        }
        for (q2 = nqs - 2; q2 > -1; q2--){
            d2ydx2[q1][q2] = d2ydx2[q1][q2]*d2ydx2[q1][q2 + 1] + tempArray[q2];
        }
    }
    free(y);
    free(tempArray);
}

// Print model kernel functions (vdWDFkernelPhi) and their 2nd derivatives (vdWDFd2Phidk2)
void print_kernel(char* outputName, double **vdWDFkernelPhi, double ** vdWDFd2Phidk2, int nrpoints, int nqs) {
    FILE *outputFile = NULL;
    outputFile = fopen(outputName,"w");
    int q1, q2, index, kernelLabel;
    for (q1 = 0; q1 < nqs; q1++) {
        for (q2 = 0; q2 <= q1; q2++) {
            fprintf(outputFile, "first q %d, second q %d\n", q1, q2);
            kernelLabel = kernel_label(q1, q2, 20);
            fprintf(outputFile, "Kernel function\n");
            for (index = 0; index < nrpoints + 1; index++) {
                fprintf(outputFile, "%5d %12.9f\n", index, vdWDFkernelPhi[kernelLabel][index]);
            }
            fprintf(outputFile, "2nd derivative of Kernel function\n");
            for (index = 0; index < nrpoints + 1; index++) {
                fprintf(outputFile, "%5d %12.9f\n", index, vdWDFd2Phidk2[kernelLabel][index]);
            }
        }
    }
    fclose(outputFile);
}

// Print 2nd derivatives of spline functions (vdWDFd2Splineydx2)
void print_d2ydx2(char* outputName2, int nqs, double **vdWDFd2Splineydx2) {
    FILE *outputFile2 = NULL;
    outputFile2 = fopen(outputName2,"w");
    fprintf(outputFile2, "dy2_dx2 of spline function\n");
    for (int q1 = 0; q1 < nqs; q1++) {
        for (int q2 = 0; q2 < nqs; q2++) {
            fprintf(outputFile2, "%12.9f ", vdWDFd2Splineydx2[q1][q2]);
        }
        fprintf(outputFile2, "\n");
    }
    fclose(outputFile2);
}

