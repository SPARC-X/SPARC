/**
 * @file    vdWDFgenerateKernelSpline.h
 * @brief   This file contains the function declarations for generating the needed model kernel functions \Phi in reciprocal space
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
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef vdWDF_GENKERNEL
#define vdWDF_GENKERNEL

#include "isddft.h"
#include "exchangeCorrelation.h"

/**
 * @brief generate kernel functions, their 2nd derivatives and 2nd derivative of spline functions
 * The function does not have input variables
 */
void vdWDF_generate_kernel(char *filename);

/**
 * @brief find suitable Gaussian quadrature integration points and weights
 nIntegrationPoints: Number of imagine frequency points a or b in M Dion's paper (14) for integration in the a-b plane, 256
 * @param [aMin, aMax]: scope of integration of imagine frequency (a, b)
 * @param aPoints, aPoints2: arrays of imagine frequency points a or b selected for Gaussian quadrature integration, and their square
 * @param weights: arrays of waight corresponding to imagine frequency points selected for Gaussian quadrature integration
 */
void prepare_Gauss_quad(int nIntegratePoints, double aMin, double aMax, double *aPoints, double *aPoints2, double *weights);

/**
 * @brief calculate model kernel functions in real space
 * @param nrpoints: number of radial points of kernel functions in real and reciprocal space, 1024
 * @param WabMatrix: Wab in M Dion's paper (14), values in all integration points in ab plane
 * @param qmesh1, qmesh2: 1st model energy ratio q1 and 2nd model energy ratio q2, in array of model energy ratios qmesh[20]
 * @param vdWdr: distance between radial points of kernel functions in real space
 * @param realKernel: the array of this model kernel function in real space
 */
void phi_value(int nrpoints, int nIntegratePoints, double **WabMatrix, double *aPoints, double *aPoints2,
 double qmesh1, double qmesh2, double vdWdr, double *realKernel);

/**
 * in Dion's paper (6)
 * @param aDivd: see description after Dion's paper (16)
 */
double h_function(double aDivd);

/**
 * @brief transform calculated kernel function in real space, spherical coordinate into reciprocal space, spherical coordinate
 * @param vdWdk: distance between radial points of kernel functions in reciprocal space
 * @param reciKernel: the array of this model kernel function in reciprocal space
 */
void radial_FT(double* realKernel, int nrpoints, double vdWdr, double vdWdk, double* reciKernel);

/**
 * @brief compute the 2nd order derivative of the kernel function in recirpocal space
 * @param d2reciKerneldk2: the array of the 2nd derivative of this model kernel function in reciprocal space
 */
void d2_of_kernel(double* reciKernel, int nrpoints, double vdWdk, double* d2reciKerneldk2);

/**
 * @brief compute the 2nd derivative of spline function at qmesh points. It is only related to qmesh and independent from kernel functions
 * initialize_spline_interpolation in QE, splineFunc_2Deri_atQmesh in msparc
 * @param qmeshPointer: array of model energy ratios, qmesh[20]
 * @param d2ydx2: 2nd derivative of the spline function in a direction (symmetry of d2y/dx2 and d2y/dz2)
 */
void spline_d2_qmesh(double* qmeshPointer, int nqs, double** d2ydx2);

/**
 * @brief Print model kernel functions (vdWDFkernelPhi) and their 2nd derivatives (vdWDFd2Phidk2)
 * @param outputName: name of the file of the output kernels and 2nd derivative of kernels
 * @param nqs: the total number of model energy ratios q
 */
void print_kernel(char* outputName, double **vdWDFkernelPhi, double ** vdWDFd2Phidk2, int nrpoints, int nqs);

/**
 * @brief Print 2nd derivatives of spline functions (vdWDFd2Splineydx2)
 * @param outputName: name of the file of the 2nd derivatives of spline functions
 */
void print_d2ydx2(char* outputName2, int nqs, double **vdWDFd2Splineydx2);

#endif