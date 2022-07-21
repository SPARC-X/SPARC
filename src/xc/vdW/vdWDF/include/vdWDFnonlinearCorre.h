/**
 * @file    vdWDFnonlinearCorre.h
 * @brief   This file contains the function declarations for non-linear correlation part of vdF-DF1 and vdW-DF2 functional.
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

#ifndef vdWDF_NONLCORRE
#define vdWDF_NONLCORRE

#include "isddft.h"
#include "exchangeCorrelation.h"
#include "vdWDFinitialization.h"

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


#endif