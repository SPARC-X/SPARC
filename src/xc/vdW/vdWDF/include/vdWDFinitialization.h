/**
 * @file    vdWDFinitialization.h
 * @brief   This file contains the function declarations for initializing variables needed by vdWDF.
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

#ifndef vdWDF_INITIAL
#define vdWDF_INITIAL

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

// /**
//  * @brief read kernel function and its 2nd derivative for every model energy ratios (p1, p2)
//  * @param kernelFile: the route of the kernel file
//  */
// void vdWDF_read_kernel(SPARC_OBJ *pSPARC, char *kernelFile);

/**
 * @brief If there is a spline_d2_qmesh function file, read it.
 * @param kernelFile: the route of the spline_d2_qmesh function file
 */
void read_spline_d2_qmesh(SPARC_OBJ *pSPARC);

void vdWDF_Setup_Comms(SPARC_OBJ *pSPARC, int *gridsizes, int *phiDims);

#endif