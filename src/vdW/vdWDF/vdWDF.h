/**
 * @file    d3Correction.h
 * @brief   This file contains the function declarations for  vdF-DF1 and vdW-DF2 functional.
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

#ifndef vdW_DF
#define vdW_DF

#include "isddft.h"
#include "exchangeCorrelation.h"

int domain_index1D(int locali, int localj, int localk, int DMnx, int DMny, int DMnz);

/**
 * @brief If there is a kernel function file, read it.
 TO BE MODIFIED.
 */
void vdWDF_initial_read_kernel(SPARC_OBJ *pSPARC);

/**
 * @brief compute the index of the kernel function in the large table vdWDFkernel
 firstq: index of the first energy ratio q
 secondq: index of the second energy ratio q
 * @param nqs: total number of q s, 20
 */
int kernel_label(int firstq, int secondq, int nqs);

/**
 * @brief read kernel function and its 2nd derivative for every model energy ratios (p1, p2)
 * @param kernelFile: the route of the kernel file
 */
void vdWDF_read_kernel(SPARC_OBJ *pSPARC, char *kernelFile);

/**
 * @brief If there is a spline_d2_qmesh function file, read it.
 * @param kernelFile: the route of the spline_d2_qmesh function file
 */
void read_spline_d2_qmesh(SPARC_OBJ *pSPARC, char *splineD2QmeshFile);

/**
 * @brief compute exchange-correlation energy of LDA_PW on every grid point, called by get_q0_Grid
 * @param dq0drhoPosi: dq0/drho, an output of the function
 */
double pw(double rs, double *dq0drhoPosi);

/**
 * @brief modify qp to let it less than qCut based on (5) of Soler, called by get_q0_Grid
 * @param qp: the input energy ratio, to be modified
 * @param qCut: largest accepted energy ratio, which is the largest model energy ratio, qmesh[19]
 * @param saturateq0dq0dq: the array to save two output values, first: modified q0; second: dq0dq, used to compute dq0/drho and dq0/dgrad(rho)
 */
void saturate_q(double qp, double qCut, double *saturateq0dq0dq);

/**
 * @brief compute q0, the energy ratio on every grid point
 rho: the electron density on real grid
 */
void get_q0_Grid(SPARC_OBJ *pSPARC, double* rho);

/**
 * @brief get the component of "model energy ratios" on every grid point, based on the computed q0. The ps array is its output
 */
void spline_interpolation(SPARC_OBJ *pSPARC);

/**
 * @brief compute the coordinates of G-vectors in reciprocal space. They are sum of integres multiplying three primary reci lattice vectors
 */
void compute_Gvectors(SPARC_OBJ *pSPARC);

/**
 * @brief Not sure whether the dividing method of SPARC on z axis can always be the same as the dividing method of DftiGetValueDM
 the usage of this function is to re-divide data on z axis for fitting the dividing of parallel FFT functions
 For example, the global grid has 22 nodes on z direction, and 3 processors are on z axis (0, 0, z). If SPARC divides it into 8+8+7,
 but DftiGetValueDM divides it into 7+8+8, then it is necessary to reorganize data
 * @param gridsizes: a 3-entries array containing Nx, Ny and Nz of the system
 * @param length: the length of grids arranged on the z-axis processor (0, 0, z), DMnz
 * @param start: a length npNdz_phi array saving all start index on z dir of all z-axis processors in dmcomm_phi, got by function block_decompose_nstart(gridsizes[2], pSPARC->npNdz_phi, z)
 * @param FFTInput: a double _Complex array allocated for saving the FFT input array. The size of the space is got by function DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &localArrayLength)
 * @param allStartK: a length npNdz_phi array saving all start index on z dir of all z-axis processors for FFT, got by function DftiGetValueDM(desc, CDFT_LOCAL_X_START, &startK); then gather
 * @param inputData: a double _Complex array saving the data to compute their FFT or iFFT. The length is Nx*Ny*DMnz
 * @param zAxisComm: the cartesion topology communicator linking all z-axis processors (0, 0, z)
 */
void compose_FFTInput(int *gridsizes, int length, int *start, double _Complex *FFTInput, int* allStartK, double _Complex *inputData, MPI_Comm zAxisComm);

/**
 * @brief Like the usage of compose_FFTInput
 * @param FFTOutput: a double _Complex array saving the FFT result array
 * @param allLengthK: a length npNdz_phi array saving allocated length on z dir of all z-axis processors for FFT, got by function DftiGetValueDM(desc, CDFT_LOCAL_NX, &lengthK); then gather
 * @param outputData: a double _Complex array allocated for saving the rearranged FFT output array. The length is Nx*Ny*DMnz
 */
void compose_FFTOutput(int *gridsizes, int length, int *start, double _Complex *FFTOutput, int* allStartK, int* allLengthK, double _Complex *outputData, MPI_Comm zAxisComm);

/**
 * @brief compose parallel FFT on data gatheredThetaCompl
 * @param q: the model energy ratio index of the theta vector to compute FFT
 */
void parallel_FFT(double _Complex *inputDataRealSpace, double _Complex *outputDataReciSpace, int *gridsizes, int *zAxis_Starts, int DMnz, int q, MPI_Comm zAxisComm);

/**
 * @brief generating thetas (ps*rho) in real space, then using FFT to transform thetas to reciprocal space. array vdWDFthetaFTs is the output.
 * also, computing the reciprocal lattice vectors
 * @param rho: the electron density on real grid
 */
void theta_generate_FT(SPARC_OBJ *pSPARC, double* rho);

/**
 * @brief compute the value of all kernel functions (210 functions for all model energy ratio pairs) on every reciprocal grid point
 * based on the distance between the reciprocal point and the center O
 */
void interpolate_kernel(SPARC_OBJ *pSPARC);

/**
 * @brief compute the arrays u, which is thetas (ps*rho) * Kernel function(Phi), and integrate vdW_DF energy in reciprocal space
 */
void vdWDF_energy(SPARC_OBJ *pSPARC);

/**
 * @brief compose parallel inverse FFT on data gathereduFT
 * @param gathereduFT: the data to compute its inverse FFT
 * @param gatheredu: the double _Complex array allocated for saving the result of inverse FFT
 */
void parallel_iFFT(double _Complex *inputDataReciSpace, double _Complex *outputDataRealSpace, int *gridsizes, int *zAxis_Starts, int DMnz, int q, MPI_Comm zAxisComm);

/**
 * @brief compute u vectors in reciprocal space, then transform them back to real space by inverse FFT
 Soler's paper (11)-(12)
 */
void u_generate_iFT(SPARC_OBJ *pSPARC);

/**
 * @brief compute the potential of vdW_DF, and add onto Vxc.
 */
void vdWDF_potential(SPARC_OBJ *pSPARC);

/**
 * @brief The main function to be called by function Calculate_Vxc_GGA in exchangeCorrelation.c, compute the energy and potential of non-linear correlation part of vdW-DF
 */
void Calculate_nonLinearCorr_E_V_vdWDF(SPARC_OBJ *pSPARC, double *rho);

/**
 * @brief add the computed vdWDF energy into Exc
 */
void Add_Exc_vdWDF(SPARC_OBJ *pSPARC);

/**
 * @brief compute the stress term containing grad(rho)
 */
void vdWDF_stress_gradient(SPARC_OBJ *pSPARC, double *stressGrad);

/**
 * @brief compute the value of all derivative of kernel functions (210 functions for all model energy ratio pairs) on every reciprocal grid point
 * based on the distance between the reciprocal point and the center O, called by vdWDF_stress_kernel
 */
void interpolate_dKerneldK(SPARC_OBJ *pSPARC, double **dKerneldLength);

/**
 * @brief compute the stress term containing derivative of kernel functions
 */
void vdWDF_stress_kernel(SPARC_OBJ *pSPARC, double *stressKernel);

/*
 Functions below is for calculating vdWDF stress, called by Calculate_XC_stress in stress.c
*/
void Calculate_XC_stress_vdWDF(SPARC_OBJ *pSPARC);

/**
 * @brief free space allocated during process of vdW-DF1 and vdW-DF2
 */
void vdWDF_free(SPARC_OBJ *pSPARC);

// functions below are in file vdWDFExchangeLinearCorre.c

/**
* @brief the function to compute the XC potential of vdF-DF1 and vdW-DF2 functional without nonlinear correlation part. Called by Calculate_Vxc_GGA in exchangeCorrelation.c
* @param xc_cst: the structure saving constants needed for Zhang-Yang revPBE
* @param rho: the array saving the electron density
*/
void Calculate_Vxc_vdWExchangeLinearCorre(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);

/**
* @brief the function to compute the potential and energy density of Zhang-Yang revPBE GGA exchange
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param xc_cst: the structure saving constants needed for Zhang-Yang revPBE
* @param sigma: the array saving square of the |grad rho|
* @param vdWDFVx1: d(n*epsilon_x)/dn
* @param vdWDFVx2: d(n*epsilon_x)/d|grad n| / |grad n|
* @param vdWDFex: epsilon_x
*/
void Calculate_Vx_ZhangYangrevPBE(int DMnd, XCCST_OBJ *xc_cst, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex);

/**
* @brief the function to compute the potential and energy density of PW86 GGA exchange
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param rho: the array saving the electron density
* @param sigma: the array saving square of the |grad rho|
* @param vdWDFVx1: d(n*epsilon_x)/dn
* @param vdWDFVx2: d(n*epsilon_x)/d|grad n| / |grad n|
* @param vdWDFex: epsilon_x
*/
void Calculate_Vx_PW86(int DMnd, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex);

/**
* @brief the function to compute the potential and energy density of PW91 LDA correlation
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param rho: the array saving the electron density
* @param vdWDFVcLinear: d(n*epsilon_clinear)/dn
* @param vdWDFecLinear: epsilon_c linear part
*/
void Calculate_Vc_PW91(int DMnd, double *rho, double *vdWDFVcLinear, double *vdWDFecLinear);

/**
* @brief the function to compute the XC energy o vdF-DF1 and vdW-DF2 functional without nonlinear correlation part.
* @param electronDens: the array saving the electron density
*/
void Calculate_Exc_GGA_vdWDF_ExchangeLinearCorre(SPARC_OBJ *pSPARC, double * electronDens);

// functions below are in file vdWDFGenerateKernelSpline.c

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