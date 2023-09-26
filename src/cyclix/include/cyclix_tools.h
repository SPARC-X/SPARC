/**
 * @file    cyclix_tools.h
 * @brief   This file contains the functions for performing transformations related to cyclix geometry
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */
 
#ifndef CYCLIX_TOOLS_H
#define CYCLIX_TOOLS_H 

#include "isddft.h"


void init_cyclix(SPARC_OBJ *pSPARC);

void free_cyclix(SPARC_OBJ *pSPARC);

/*
@ brief: function to calculate rotational matrices in cyclix symmetry
*/
void RotMat_cyclix(SPARC_OBJ *pSPARC, double ty, double tz);

/*
@ brief: function to determine cell typ corresponding to the given cyclix symmetry
*/
void CellTyp_cyclix(SPARC_OBJ *pSPARC);

/*
@ brief: function to determine cell parameters like vacuum, inner and outer radii
*/

void CellParm_cyclix(SPARC_OBJ *pSPARC);

/*
@ brief: function to store laplacian coefficients
*/

void LapStencil_cyclix(SPARC_OBJ *pSPARC);

/*
@ brief: function to convert non-cartesian to cartesian coordinates
*/

void nonCart2Cart_coord_cyclix(const SPARC_OBJ *pSPARC, double *x, double *y, double *z);

/*
@ brief: function to convert cartesian to non-cartesian coordinates
*/

void Cart2nonCart_coord_cyclix(const SPARC_OBJ *pSPARC, double *x, double *y, double *z);

/*
@brief: function to calculate integration weights
*/

void Integration_weights_cyclix(SPARC_OBJ *pSPARC, double *Intg_wt, int ipos_x, int Nx, int Ny, int Nz);

/*
@brief: function to calculate distance between two points
*/

void CalculateDistance_cyclix(SPARC_OBJ *pSPARC, double x, double y, double z, double xref, double yref, double zref, double *d);

/*
@ brief: function to precondition the laplacian for Poisson equation
*/

void Jacobi_preconditioner_cyclix(SPARC_OBJ *pSPARC, int N, double c, double *r, double *f, MPI_Comm comm);


/*
@ brief: function to normalize the eigenvectors for cyclix system
*/
void NormalizeEigfunc_cyclix(SPARC_OBJ *pSPARC, int spn_i);

/*
@ brief: function to normalize the eigenvectors for cyclix system
*/
void NormalizeEigfunc_kpt_cyclix(SPARC_OBJ *pSPARC, int spn_i, int kpt);

/*
@ brief: generalized eigenvalue problem solver for cyclix
*/
int generalized_eigenvalue_problem_cyclix(SPARC_OBJ *pSPARC, double *Hp_local, double *Mp_local, double *eig_val);

/*
@ brief: generalized eigenvalue problem solver for cyclix complex case
*/
int generalized_eigenvalue_problem_cyclix_kpt(SPARC_OBJ *pSPARC, double _Complex *Hp_local, double _Complex *Mp_local, double *eig_val);

#endif // CYCLIX_TOOLS_H
