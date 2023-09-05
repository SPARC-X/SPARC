/**
 * @file    linearSolver.h
 * @brief   This file declares the functions for solving linear equations.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef LINEARSOLVER_H
#define LINEARSOLVER_H

#include "isddft.h"


/**
 * @brief   Alternating Anderson-Richardson (AAR) method for solving a general linear
 *          system Ax = b. 
 *
 *          AAR() assumes that x and  b is distributed among the given communicator. 
 *          The residual r = b - Ax is calculated by calling res_fun(). The function
 *          precond_fun() takes a residual vector r and precondition the residual by 
 *          applying inv(M)*r, where M is the preconditioner. The input c is a parameter
 *          for the res_fun() and precond_fun().
 */
void AAR(
    SPARC_OBJ *pSPARC, 
    void (*res_fun)(SPARC_OBJ *,int,double,double *,double*,double*,MPI_Comm,double*),  
    void (*precond_fun)(SPARC_OBJ *,int,double,double *,double*,MPI_Comm), 
    double c, 
    int N, double *x, double *b, double omega, double beta, int m, int p, double tol, 
    int max_iter, MPI_Comm comm
);


/**
 * @brief   Conjugate Gradient (CG) method for solving a general linear system Ax = b. 
 *
 *          CG() assumes that x and  b is distributed among the given communicator. 
 *          Ax is calculated by calling function Ax().
 */
void CG(SPARC_OBJ *pSPARC, 
    void (*Ax)(const SPARC_OBJ *, const int, const int *, const int, const double, double *, const int, double *, const int, MPI_Comm),
    int DMnd, int *DMVertices, double *x, double *b, double tol, int max_iter, MPI_Comm comm
);

#endif //LINEARSOLVER_H


