/**
 * @file    cyclix_gradVec.c
 * @brief   This file contains the functions for performing gradient matrix times vector.
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#ifndef CYCLIX_GRADVEC_H
#define CYCLIX_GRADVEC_H 

#include "isddft.h"

/*
@ brief: function to calculate the gradients in cartesian directions for cyclix systems
*/

void Gradient_vectors_dir_cyclix(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                                 const int ncol, const double c, const double *X, const int ldi,
                                 double *DX, const int ldo, const int dir, MPI_Comm comm, const int* dims);

void Gradient_vectors_dir_with_rotfac(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                                 const int ncol, const double c, const double *X1, const double *X2, const int ldi,
                                 double *DX, const int ldo, const int dir, MPI_Comm comm);

/*
@ brief: function to calculate the gradients in cartesian directions for cyclix systems
*/

void Gradient_vectors_dir_kpt_cyclix(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                                     const int ncol, const double c, const double _Complex *X, const int ldi, 
                                     double _Complex *DX, const int ldo, const int dir, const double *kpt_vec, MPI_Comm comm, const int *dims);


/*
@ brief: function to calculate the gradients in cartesian directions (x and y) for cyclix systems with rotational factors
*/


void Gradient_vec_dir_rotfac(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                             const int ncol, const double c, const double *x, const double *y, const int ldi, 
                             double *Dx, const int ldo, const int dir, MPI_Comm comm, const int* dims, const int vecdir);


#endif // CYCLIX_GRADVEC_H
