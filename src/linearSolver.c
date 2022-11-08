/**
 * @file    linearSolver.c
 * @brief   This file contains functions for solving linear equations.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>

#include "linearSolver.h"
#include "mixing.h"
#include "tools.h" // for Vector2Norm
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
    void (*precond_fun)(SPARC_OBJ *,int,double,double *,double*,MPI_Comm), double c, 
    int N, double *x, double *b, double omega, double beta, int m, int p, double tol, 
    int max_iter, MPI_Comm comm
)
{
#define X(i,j) X[(j)*N+(i)]
#define F(i,j) F[(j)*N+(i)]
    if (comm == MPI_COMM_NULL) return;
    
    int rank, i, iter_count, i_hist;
    double *r, *x_old, *f, *f_old, *X, *F, b_2norm, r_2norm;

#ifdef DEBUG
    double t1, t2;
#endif
    double tt1, tt2, ttot, t_anderson, ttot2, ttot3, ttot4;
    
    ttot = ttot2 = ttot3 = ttot4 = 0.0;
    
    MPI_Comm_rank(comm, &rank);
    
    // allocate memory for storing x, residual and preconditioned residual in the local domain
    r = (double *)malloc( N * sizeof(double) );  // residual vector, r = b - Ax
    x_old   = (double *)malloc( N * sizeof(double) );
    f       = (double *)malloc( N * sizeof(double) ); // preconditioned residual vector, f = inv(M) * r
    f_old   = (double *)malloc( N * sizeof(double) );    
    assert(r != NULL && x_old != NULL && f != NULL && f_old != NULL);

    // allocate memory for storing X, F history matrices
    X = (double *)calloc( N * m, sizeof(double) ); 
    F = (double *)calloc( N * m, sizeof(double) ); 
    assert(X != NULL && F != NULL);

    // initialize x_old as x0 (initial guess vector)
    for (i = 0; i < N; i++) 
        x_old[i] = x[i];

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    Vector2Norm(b, N, &b_2norm, comm); // b_2norm = ||b||
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, (t2-t1)*1e3);
#endif
    // find initial residual vector r = b - Ax, and its 2-norm
    res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);
    // Vector2Norm(r, N, &r_2norm, comm); // r_2norm = ||r||

    // replace the abs tol by scaled tol: tol * ||b||
    tol *= b_2norm; 
    r_2norm = tol + 1.0; // to reduce communication
    iter_count = 0;
    while (r_2norm > tol && iter_count < max_iter) {  
        // *** calculate preconditioned residual f *** //
        precond_fun(pSPARC, N, c, r, f, comm); // f = inv(M) * r     
        // *** store residual & iteration history *** //
        if (iter_count > 0) {
            i_hist = (iter_count - 1) % m;
            for (i = 0; i < N; i++) {
                X(i, i_hist) = x[i] - x_old[i];
                F(i, i_hist) = f[i] - f_old[i];
            }
        }
        
        for (i = 0; i < N; i++) {
            x_old[i] = x[i];
            f_old[i] = f[i];
        }
        
        if((iter_count+1) % p == 0 && iter_count > 0) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
            tt1 = MPI_Wtime();
            // AndersonExtrapolation(N, m, x, x, f, X, F, beta, comm);
            AndersonExtrapolation(N, m, x, x_old, f, X, F, beta, comm);
            tt2 = MPI_Wtime();
            ttot += (tt2-tt1);
            ttot4 += t_anderson;

            tt1 = MPI_Wtime();
            // update residual r = b - Ax
            res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
            Vector2Norm(r, N, &r_2norm, comm); // r_2norm = ||r||
        } else {
            /***********************
             *  Richardson update  *
             ***********************/
            for (i = 0; i < N; i++)
                // x[i] = x[i] + omega * f[i];  
                x[i] = x_old[i] + omega * f[i];  
            tt1 = MPI_Wtime();   
            // update residual r = b - Ax
            res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);  
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
        }
        iter_count++;
    }
#ifdef DEBUG
    if (rank == 0) printf("\niter_count = %d, r_2norm = %.3e, tol*||rhs|| = %.3e\n\n", iter_count, r_2norm, tol);    
    if (rank == 0) 
        printf("Anderson update took %.3f ms, out of which F'*F took %.3f ms; b-Ax took %.3f ms, out of which Lap took %.3f ms\n", 
        ttot * 1e3, ttot4*1e3, ttot2 * 1e3, ttot3 * 1e3);
#endif    
    // deallocate memory
    free(x_old);
    free(f_old);
    free(f);
    free(r);
    free(X);
    free(F);
#undef X
#undef F
}



/**
 * @brief   Conjugate Gradient (CG) method for solving a general linear system Ax = b. 
 *
 *          CG() assumes that x and  b is distributed among the given communicator. 
 *          Ax is calculated by calling function Ax().
 */
void CG(SPARC_OBJ *pSPARC, 
    void (*Ax)(const SPARC_OBJ *, const int, const int *, const int, const double, double *, double *, MPI_Comm),
    int N, int DMnd, int *DMVertices, double *x, double *b, double tol, int max_iter, MPI_Comm comm
)
{
    int i, j, iter_count = 0, sqrt_n = (int)sqrt(N);
    double *r, *d, *q, delta_new, delta_old, alpha, beta, err, b_2norm;

    r = (double *)calloc( DMnd , sizeof(double) );
    d = (double *)calloc( DMnd , sizeof(double) );
    q = (double *)calloc( DMnd , sizeof(double) );    
    assert(r != NULL && d != NULL && q != NULL);
    /********************************************************************/

    Vector2Norm(b, DMnd, &b_2norm, comm); 
    tol *= b_2norm;

    Ax(pSPARC, DMnd, DMVertices, 1, 0.0, x, r, comm);

    for (i = 0; i < DMnd; ++i){
        r[i] = b[i] + r[i];
        d[i] = r[i];
    }
    Vector2Norm(r, DMnd, &delta_new, comm);

    err = tol + 1.0;
    while(iter_count < max_iter && err > tol){
        Ax(pSPARC, DMnd, DMVertices, 1, 0.0, d, q, comm);
        VectorDotProduct(d, q, DMnd, &alpha, comm);

        alpha = - delta_new * delta_new / alpha;

        for (j = 0; j < DMnd; ++j)
            x[j] = x[j] + alpha * d[j];
        
        // Restart every sqrt_n cycles.
        if ((iter_count % sqrt_n)==0) {
            Ax(pSPARC, N, DMVertices, 1, 0.0, x, r, comm);
            for (j = 0; j < DMnd; ++j){
                r[j] = b[j] + r[j];
            }
        } else {
            for (j = 0; j < DMnd; ++j)
                r[j] = r[j] + alpha * q[j];
        }

        delta_old = delta_new;

        Vector2Norm(r, DMnd, &delta_new, comm);

        err = delta_new;
        beta = delta_new * delta_new / (delta_old * delta_old);
        for (j = 0; j < DMnd; ++j)
            d[j] = r[j] + beta * d[j];

        iter_count++;
    }

    if (fabs(err) > tol) {
        printf("WARNING: CG failed!\n");
    }

    free(r);
    free(d);
    free(q);
}
