/**
 * @file    printing.h
 * @brief   This file contains function declarations for printing properties.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef PRINTING_H
#define PRINTING_H

#include "isddft.h"

/**
 * @brief   Print initial electron density guess and converged density.
 */
void printElecDens(SPARC_OBJ *pSPARC);

/**
 * @brief   Print converged density in cube format.
 */
void printDens_cube(SPARC_OBJ *pSPARC, double *rho, char *fname, char *rhoname);

/**
 * @brief   Print final eigenvalues and occupation numbers.
 */
void printEigen(SPARC_OBJ *pSPARC);

/**
 * @brief   Print Kohn-Sham orbitals
 */
void print_orbitals(SPARC_OBJ *pSPARC);

/**
 * @brief   Print real Kohn-Sham orbitals
 */
void print_orbital_real(
    double *x, int *gridsizes, int *DMVertices, double dV, int Nspinor,
    char *fname, int spin_index, int kpt_index, double *kpt_vec, int band_index, MPI_Comm comm
);

/**
 * @brief   Print complex Kohn-Sham orbitals
 */
void print_orbital_complex(
    double _Complex *x, int *gridsizes, int *DMVertices, double dV, int Nspinor,
    char *fname, int spin_index, int kpt_index, double *kpt_vec, int band_index, MPI_Comm comm
);

/**
 * @brief   Print Energy density
 */
void printEnergyDensity(SPARC_OBJ *pSPARC);

/**
 * @brief   Gather Energy Density from dmcomm
 */
void GatherEnergyDensity_dmcomm(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv);

/**
 * @brief   Gather Energy Density from dmcomm_phi
 */
void GatherEnergyDensity_dmcomm_phi(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv);

#endif // PRINTING_H