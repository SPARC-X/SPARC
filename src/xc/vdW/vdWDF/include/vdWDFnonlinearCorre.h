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
 * Lee, Kyuho, Éamonn D. Murray, Lingzhu Kong, Bengt I. Lundqvist, and David C. Langreth.
 * "Higher-accuracy van der Waals density functional." Physical Review B 82, no. 8 (2010): 081101.
 * Thonhauser, T., S. Zuluaga, C. A. Arter, K. Berland, E. Schröder, and P. Hyldgaard.
 * "Spin signature of nonlocal correlation binding in metal-organic frameworks."
 * Physical review letters 115, no. 13 (2015): 136402.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef vdWDF_NONLCORRE
#define vdWDF_NONLCORRE

#include "isddft.h"
#include "exchangeCorrelation.h"
#include "vdWDFinitialization.h"

/**
 * @brief modify qp to let it less than qCut based on (5) of Soler, called by get_q0_Grid
 * @param qp: the input energy ratio, to be modified
 * @param qCut: largest accepted energy ratio, which is the largest model energy ratio, qmesh[19]
 * @param saturateq0dq0dq: the array to save two output values, first: modified q0; second: dq0dq, used to compute dq0/drho and dq0/dgrad(rho)
 */
void saturate_q(double qp, double qCut, double *saturateq0dq0dq);

/**
 * @brief compute q0, the energy ratio on every grid point
 * @param rho: the electron density on real grid
 */
void get_q0_Grid(SPARC_OBJ *pSPARC, double* rho);

/**
 * @brief compute q0, the energy ratio on every grid point, spin-polarized case
 * @param rho: the electron density on real grid
 */
void spin_get_q0_Grid(SPARC_OBJ *pSPARC, double* rho);

/**
 * @brief get the component of "model energy ratios" on every grid point, based on the computed q0. The ps array is its output
 */
void spline_interpolation(SPARC_OBJ *pSPARC);

/**
 * @brief compute the coordinates of G-vectors in reciprocal space. They are sum of integres multiplying three primary reci lattice vectors
 */
void compute_Gvectors(SPARC_OBJ *pSPARC);


/**
 * @brief compose parallel FFT on data gatheredThetaCompl
 * @param inputDataRealSpace: a double _Complex array saving the data to compute their FFT or iFFT. The length is Nx*Ny*DMnz
 * @param outputDataReciSpace: a double _Complex array saving the result of FFT or iFFT. The length is Nx*Ny*DMnz
 * @param gridsizes: a 3-entries array containing Nx, Ny and Nz of the system
 * @param DMnz: the length of grids arranged on the z-axis processor (0, 0, z), it should be equal to data distribution lengthK from FFT modules
 * @param q: the model energy ratio index of the theta vector to compute FFT
 * @param zAxisComm: the cartesion topology communicator linking all z-axis processors (0, 0, z)
 */
void parallel_FFT(double _Complex *inputDataRealSpace, double _Complex *outputDataReciSpace, int *gridsizes, int DMnz, int q, MPI_Comm zAxisComm);

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
 * @param inputDataReciSpace: a double _Complex array saving the data to compute their FFT or iFFT. The length is Nx*Ny*DMnz
 * @param outputDataRealSpace: a double _Complex array saving the result of FFT or iFFT. The length is Nx*Ny*DMnz
 * @param gridsizes: a 3-entries array containing Nx, Ny and Nz of the system
 * @param DMnz: the length of grids arranged on the z-axis processor (0, 0, z), it should be equal to data distribution lengthK from FFT modules
 * @param q: the model energy ratio index of the theta vector to compute FFT
 * @param zAxisComm: the cartesion topology communicator linking all z-axis processors (0, 0, z)
 */
void parallel_iFFT(double _Complex *inputDataReciSpace, double _Complex *outputDataRealSpace, int *gridsizes, int DMnz, int q, MPI_Comm zAxisComm);

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
 * @brief compute the potential of vdW_DF, and add onto Vxc, spin-polarized case
 */
void spin_vdWDF_potential(SPARC_OBJ *pSPARC);

/**
 * @brief The main function to be called by function Calculate_Vxc_GGA in exchangeCorrelation.c, compute the energy and potential of non-linear correlation part of vdW-DF
 */
void Calculate_nonLinearCorr_E_V_vdWDF(SPARC_OBJ *pSPARC, double *rho);

/**
 * @brief The main function to be called by function Calculate_Vxc_GGA in exchangeCorrelation.c, spin-polarized case, compute the energy and potential of non-linear correlation part of vdW-DF
 */
void Calculate_nonLinearCorr_E_V_SvdWDF(SPARC_OBJ *pSPARC, double *rho);

/**
 * @brief add the computed vdWDF energy into Exc
 */
void Add_Exc_vdWDF(SPARC_OBJ *pSPARC);

// the two functions below are used for debugging
/**
 * @brief find the route of the folder containing input files
 * @param folderRoute: the char array containing the folder route
 */
void find_folder_route(SPARC_OBJ *pSPARC, char *folderRoute);

/**
 * @brief print the variable array in an output file
 * @param variable: the variable array to be printed
 * @param outputFileName: the designated name of the output file
 * @param Nx: the number of grids on 1st dir
 * @param Ny: the number of grids on 2nd dir
 * @param Nz: the number of grids on 3rd dir
 */
void print_variables(double *variable, char *outputFileName, int Nx, int Ny, int Nz);


#endif