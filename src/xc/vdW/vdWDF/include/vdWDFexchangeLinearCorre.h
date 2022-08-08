/**
 * @file    vdWDFexchangeLinearCorre.h
 * @brief   This file contains the declarations of functions of exchange and linear correlation of vdF-DF1 and vdW-DF2.
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

#ifndef vdWDF_EXCLCORRE
#define vdWDF_EXCLCORRE

#include "isddft.h"
#include "exchangeCorrelation.h"

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
* @brief the function to compute the potential and energy density of Zhang-Yang revPBE GGA exchange, spin-polarized case
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param xc_cst: the structure saving constants needed for Zhang-Yang revPBE
* @param sigma: the array saving square of the |grad rho|
* @param vdWDFVx1: d(n*epsilon_x)/dn
* @param vdWDFVx2: d(n*epsilon_x)/d|grad n| / |grad n|
* @param vdWDFex: epsilon_x
*/
void Calculate_Vx_SZhangYangrevPBE(int DMnd, XCCST_OBJ *xc_cst, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex);

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
* @brief the function to compute the potential and energy density of PW86 GGA exchange, spin-polarized case
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param rho: the array saving the electron density
* @param sigma: the array saving square of the |grad rho|
* @param vdWDFVx1: d(n*epsilon_x)/dn
* @param vdWDFVx2: d(n*epsilon_x)/d|grad n| / |grad n|
* @param vdWDFex: epsilon_x
*/
void Calculate_Vx_SPW86(int DMnd, double *rho, double *sigma, double *vdWDFVx1, double *vdWDFVx2, double *vdWDFex);

/**
* @brief the function to compute the potential and energy density of PW91 LDA correlation
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param rho: the array saving the electron density
* @param vdWDFVcLinear: d(n*epsilon_clinear)/dn
* @param vdWDFecLinear: epsilon_c linear part
*/
void Calculate_Vc_PW91(int DMnd, double *rho, double *vdWDFVcLinear, double *vdWDFecLinear);

/**
* @brief the function to compute the potential and energy density of PW91 LDA correlation, spin-polarized case
* @param DMnd: the number of grid points in the area of this process in dmcomm_phi
* @param rho: the array saving the electron density
* @param vdWDFVcLinear: d(n*epsilon_clinear)/dn
* @param vdWDFecLinear: epsilon_c linear part
*/
void Calculate_Vc_SPW91(int DMnd, XCCST_OBJ *xc_cst, double *rho, double *vdWDFVcLinear, double *vdWDFecLinear);

/**
* @brief the function to compute the XC energy o vdF-DF1 and vdW-DF2 functional without nonlinear correlation part.
* @param electronDens: the array saving the electron density
*/
void Calculate_Exc_GGA_vdWDF_ExchangeLinearCorre(SPARC_OBJ *pSPARC, double * electronDens);

/**
 * @brief the function to compute the XC energy o vdF-DF1 and vdW-DF2 functional without nonlinear correlation part, spin-polarized case. It is similar to Calculate_Exc_GSGA_PBE
 * it is also similar to Calculate_Exc_GGA_PBE. 
 * @param electronDens: the array saving the electron density
 */
void Calculate_Exc_GSGA_vdWDF_ExchangeLinearCorre(SPARC_OBJ *pSPARC, double *electronDens);

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