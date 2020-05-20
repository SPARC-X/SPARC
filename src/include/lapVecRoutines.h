/**
 * @file    lapVecRoutines.h
 * @brief   This file contains function declarations for performing laplacian-
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
 
 
#ifndef LAPVECROUTINES_H
#define LAPVECROUTINES_H

#include "isddft.h"



/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 *
 *          General routine, for both orthogonal and non-orthogonal.
 */
void Lap_vec_mult(
	const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
	const int ncol, const double c, double *x, double *Lapx, MPI_Comm comm
);



/* Non-orthogonal subroutines */

/*void Lap_plus_diag_vec_mult_nonorth(
        const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, 
        double *Veff_loc, int ncol, double c, double *x, 
        double *Hx, MPI_Comm comm, MPI_Comm comm2
);


void Lap_plus_diag_vec_mult_seq_nonorth(
        const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc, 
        int ncol, double c, double *x, double *Hx
);


void Lap_vec_mult_nonorth(
	const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
	const int ncol, const double c, double *x, double *Lapx, 
	MPI_Comm comm, MPI_Comm comm2
);
*/





/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 */
void Lap_vec_mult_kpt(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, double complex *x, double complex *Lapx, int kpt, MPI_Comm comm
); 

/**
 * @brief   Calculate the residual of the poisson equation: r = b - A * x, where A = -(Lap + c).
 *
 *          The vector x is assumed to be stored domain-wisely among the processors. The
 *          structure pSPARC contains the description of the distribution info of x, and
 *          in this case the info of Laplacian operator such as boundary conditions, 
 *          finite-difference order and coefficients etc.
 */
void poisson_residual(
    SPARC_OBJ *pSPARC, int N, double c, double *x, double *b, 
    double *r, MPI_Comm comm, double *time_info
);


#endif // LAPVECROUTINES_H



