/**
 * @file    d3correction.h
 * @brief   This file contains the function declarations for the DFT-D3 method.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * S.Grimme, J.Antony, S.Ehrlich, H.Krieg, A consistent and accurate ab
 * initio parametrization of density functional dispersion correction
 * (DFT-D) for the 96 elements H-Pu
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef D3_CORRECTION
#define D3_CORRECTION 

/**
 * @brief calculate how many image cells need to be considered
 * @param rLimit  cut-off radius of a calculation
 * @param periodicType  3*1 array for recording type of boundary 1: P 0: D
 * @param inputLatt  (lattice vectors * side length) tensor
 */
void d3_set_criteria(int *nImage, double rLimit, int *periodicType, double *inputLatt);

// double *****copyC6();

double solve_cos(double latF1, double latF2, double latF3, double latS1, double latS2, double latS3, double latT1, double latT2, double latT3);

/**
 * @brief calculate CNs of all atoms in the system
 */
void d3_CN(SPARC_OBJ *pSPARC, double K1);

/**
 * @brief calculate C6 by interpolation
 * @param atomMaxci array saving the number of examples of all atom types
 * @param atomicNumbers array saving the atomic numbers of atoms in the current system
 * @param CN array saving CNs of atoms in the current system
 * @param atomI the index of 1st atoms in the atom pair
 * @param atomJ the index of 2nd atoms in the atom pair
 * @param c6ab array saving C6 coefficients of all example atom pairs
 * @param k3 coefficient, -4.0
 */
double d3_getC6(int *atomMaxci, int *atomicNumbers, double *CN, int atomI, int atomJ, double *****c6ab, double k3);

/**
 * @brief The main function of DFT-D3. D3 energy, force are computed here. Called by Calculate_electronicGroundState in electronicGroundState.c
 */
void d3_energy_gradient(SPARC_OBJ *pSPARC);

/**
 * @brief calculate C6 and dC6/dCNi; dC6/dCNj between atom i and atom j by interpolation
 * @param atomMaxci array saving the number of examples of all atom types
 * @param atomicNumbers array saving the atomic numbers of atoms in the current system
 * @param CN array saving CNs of atoms in the current system
 * @param atomI the index of 1st atoms in the atom pair
 * @param atomJ the index of 2nd atoms in the atom pair
 * @param c6ab array saving C6 coefficients of all example atom pairs
 * @param k3 coefficient, -4.0
 */
void d3_comp_dC6_dCNij(double *C6dC6pairIJ, int *atomMaxci, int *atomicNumbers, double *CN, int atomI, int atomJ, double *****c6ab, double k3);

/**
 * @brief used in d3_energy_gradient
 */
double d3_dAng(double abDist2, double bcDist2, double acDist2, double tDenomin);

/**
 * @brief used in d3_energy_gradient
 */
int d3_count_image(int imageX, int imageY, int imageZ, int *nImage);

#endif
