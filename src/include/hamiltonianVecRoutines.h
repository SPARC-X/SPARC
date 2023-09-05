/**
 * @file    hamiltonianVecRoutines.h
 * @brief   This file contains function declarations for performing Hamiltonian-
 *          vector multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
 
#ifndef HAMILTONIANVECROUTINES_H
#define HAMILTONIANVECROUTINES_H

#include "isddft.h"



/**
 * @brief   Calculate (Hamiltonian + c * I) times a bunch of vectors in a matrix-free way.
 *          
 *          This function simply calls the Hamiltonian_vec_mult multiple times. For some
 *          reason it is more efficient than calling it ones and do the multiplication 
 *          together. TODO: think of a more efficient way!
 */
void Hamiltonian_vectors_mult(
    const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
    int ncol, double c, double *x, const int ldi, double *Hx, const int ldo, int spin, MPI_Comm comm);


/**
 * @brief   Calculate (Hamiltonian + c * I) times a bunch of vectors in a matrix-free way with a Bloch wavevector.
 *          
 *          This function simply calls the Hamiltonian_vec_mult (or the sequential version) 
 *          multiple times. For some reason it is more efficient than calling it ones and do  
 *          the multiplication together. TODO: think of a more efficient way!
 */
void Hamiltonian_vectors_mult_kpt(
    const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
    int ncol, double c, double _Complex *x, const int ldi, double _Complex *Hx, const int ldo, int spin, int kpt, MPI_Comm comm
);


/**
 * @brief   Apply off-diagonal effective potential
 */
void off_diagonal_effective_potential(const SPARC_OBJ *pSPARC, 
    int DMnd, int ncol, double *Veff_loc, double _Complex *x, const int ldi, double _Complex *Hx, const int ldo);

#endif // HAMILTONIANVECROUTINES_H


