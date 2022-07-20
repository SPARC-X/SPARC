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
void Hamiltonian_vectors_mult(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
                              ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
                              int ncol, double c, double *x, double *Hx, int spin, MPI_Comm comm);



/**
 * @brief   Calculate (Hamiltonian + c * I) times vectors in a matrix-free way ON A SINGLE PROCESS.
 *          
 *          Note: this function is to take advantage of the band parallelization scheme when a single
 *                process contains the whole vector and therefore no communication is needed.
 */
/*void Hamiltonian_vec_mult_seq_orth(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
                              ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
                              int ncol, double c, double *x, double *Hx);

void Hamiltonian_vec_mult_seq_nonorth(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
                              ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
                              int ncol, double c, double *x, double *Hx);
*/
/**
 * @brief   Calculate (Hamiltonian + c * I) times vectors in a matrix-free way.
 */
/*void Hamiltonian_vec_mult(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
                          ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
                          int ncol, double c, double *x, double *Hx, MPI_Comm comm);

void Hamiltonian_vec_mult_nonorth(const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
                          ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
                          int ncol, double c, double *x, double *Hx, MPI_Comm comm, MPI_Comm comm2);
*/


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
    int ncol, double c, double complex *x, double complex *Hx, int spin, int kpt, MPI_Comm comm
);




#endif // HAMILTONIANVECROUTINES_H


