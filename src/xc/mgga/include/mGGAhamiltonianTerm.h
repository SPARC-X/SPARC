/**
 * @file    MGGAhamiltonianTerm.h
 * @brief   This file contains the function declarations for metaGGA functionals.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MGGA_HAMIL
#define MGGA_HAMIL

#include "isddft.h"

void mGGA_potential(const SPARC_OBJ *pSPARC, double *x, int ldx, int ncol, int DMnd, int *DMVertices, double *Hx, int ldhx, int spin, MPI_Comm comm);

void mGGA_potential_kpt(const SPARC_OBJ *pSPARC, double _Complex *x, int ldx, int ncol, int DMnd, int *DMVertices, double _Complex *Hx, int ldhx, int spin, int kpt, MPI_Comm comm);

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult
 *          
 * @param x               the vector saving vectors (possible wave functions psi) to be multiplied by Hamiltonian operator
 * @param ncol            the number of vectors to be multiplied
 * @param DMnd            the length of the vector (possible wave functions psi) in this processor in this communicator
 * @param DMVertices      the vector saving indexes of the space saved in this processor in this communicator
 * @param vxcMGGA3_dm     the vector saving vxcMGGA3 (d(n epsilon)/d(tau))
 * @param mGGAterm        the vector saving the mGGA term in Hamiltonian, which is also the result of the function
 * @param spin            number of spin (or whether there is spin polarization). Now mGGA or SCAN odes not have it so it should be equal to 1
 * @param comm            communicator controlling the multiplication H.x
 */
void compute_mGGA_term_hamil(const SPARC_OBJ *pSPARC, double *x, int ldx, int ncol, int DMnd, int *DMVertices, double *vxcMGGA3_dm, double *Hx, int ldhx, MPI_Comm comm);

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult_kpt
 *          
 * @param x               the vector saving vectors (possible wave functions psi) to be multiplied by Hamiltonian operator
 * @param ncol            the number of vectors to be multiplied
 * @param DMnd            the length of the vector (possible wave functions psi) in this processor in this communicator
 * @param DMVertices      the vector saving indexes of the space saved in this processor in this communicator
 * @param vxcMGGA3_dm     the vector saving vxcMGGA3 (d(n epsilon)/d(tau))
 * @param mGGAterm        the vector saving the mGGA term in Hamiltonian, which is also the result of the function
 * @param spin            number of spin (or whether there is spin polarization). Now mGGA or SCAN odes not have it so it should be equal to 1
 * @param kpt             the index of the k-point in this multiplication
 * @param comm            communicator controlling the multiplication H.x
 */
void compute_mGGA_term_hamil_kpt(const SPARC_OBJ *pSPARC, double _Complex *x,  int ldx, int ncol, int DMnd, int *DMVertices, double *vxcMGGA3_dm,  double _Complex *Hx, int ldhx, int kpt, MPI_Comm comm);

#endif