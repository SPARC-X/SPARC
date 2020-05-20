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

void CG(SPARC_OBJ *pSPARC,
    void (*Ax)(const SPARC_OBJ *, const int, const int *, const int, const double, double *, double *, MPI_Comm),
    int N, int DMnd, double* x, double *b, double tol, int max_iter, MPI_Comm comm
);

/**
 * @brief   Conjugate Gradient (CG) method for solving a general linear system Ax = b. 
 *
 *          CG() assumes that x and  b is distributed among the given communicator. 
 *          Ax is calculated by calling function Ax().
 */
void CG(SPARC_OBJ *pSPARC, 
    void (*Ax)(const SPARC_OBJ *, const int, const int *, const int, const double, double *, double *, MPI_Comm),
    int N, int DMnd, double* x, double *b, double tol, int max_iter, MPI_Comm comm
)
{
    if (comm == MPI_COMM_NULL) return;

    int iter_count = 0, sqrt_n = (int)sqrt(N), rank, size;
    double *r, *d, *q, delta_new, delta_old, alpha, beta, err, b_2norm;

    double t1, t2, tt1, tt2, tt, t;

    tt = 0;
    t=0;
    r = (double *)malloc( DMnd * sizeof(double) );
    d = (double *)malloc( DMnd * sizeof(double) );
    q = (double *)malloc( DMnd * sizeof(double) );    
    assert(r != NULL && d != NULL && q != NULL);
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    t1 = MPI_Wtime();
    Vector2Norm(b, DMnd, &b_2norm, comm); 
    t2 = MPI_Wtime();
    t += t2 - t1;
#ifdef DEBUG
    if (rank == 0) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, t*1e3);
#endif

    tt1 = MPI_Wtime();
    Ax(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, x, r, comm);
    tt2 = MPI_Wtime();
    tt += (tt2 - tt1);

    for (int i = 0; i < DMnd; ++i){
        r[i] = b[i] + r[i];
        d[i] = r[i];
    }

    t1 = MPI_Wtime();
    Vector2Norm(r, DMnd, &delta_new, comm);
    t2 = MPI_Wtime();
    t += t2 - t1;

    err = delta_new / b_2norm;

    while(iter_count < max_iter && err > tol){
        tt1 = MPI_Wtime();
        Ax(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, d, q, comm);
        tt2 = MPI_Wtime();
        tt += (tt2 - tt1);

        t1 = MPI_Wtime();
        VectorDotProduct(d, q, DMnd, &alpha, comm);
        t2 = MPI_Wtime();
        t += t2 - t1;

        alpha = - delta_new * delta_new / alpha;

        for (int j = 0; j < DMnd; ++j)
            x[j] = x[j] + alpha * d[j];
        
        // Restart every sqrt_n cycles.
        if ((iter_count % sqrt_n)==0)
        {
            tt1 = MPI_Wtime();
            Ax(pSPARC, N, pSPARC->DMVertices, 1, 0.0, x, r, comm);
            tt2 = MPI_Wtime();
            tt += (tt2 - tt1);

            for (int j = 0; j < DMnd; ++j){
                r[j] = b[j] + r[j];
            }
        }
        else
        {
            for (int j = 0; j < DMnd; ++j)
                r[j] = r[j] + alpha * q[j];
        }

        delta_old = delta_new;

        t1 = MPI_Wtime();
        Vector2Norm(r, DMnd, &delta_new, comm);
        t2 = MPI_Wtime();
        t += t2 - t1;

        err = delta_new / b_2norm;
        beta = delta_new * delta_new / (delta_old * delta_old);
        for (int j = 0; j < DMnd; ++j)
            d[j] = r[j] + beta * d[j];

        iter_count++;
    }

#ifdef DEBUG
    if (rank == 0) printf("\niter_count = %d, err = %.3e, tol = %.3e\n\n", iter_count, err, tol);    
    if (rank == 0) printf("Dot product and 2norm took %.3f ms, Ax took %.3f ms\n", t * 1e3, tt * 1e3);
#endif

    free(r);
    free(d);
    free(q);
}





/**
 * @brief   Alternating Anderson-Richardson (AAR) method for solving a general complex 
 *          linear system Ax = b. 
 *
 *          AAR_complex() assumes that x and  b is distributed among the given communicator. 
 *          The residual r = b - Ax is calculated by calling res_fun(). The function
 *          precond_fun() takes a residual vector r and precondition the residual by 
 *          applying inv(M)*r, where M is the preconditioner. The input c is a parameter
 *          for the res_fun() and precond_fun().
 */
void AAR_complex(
    SPARC_OBJ *pSPARC, 
    void (*res_fun)(SPARC_OBJ *, int, double complex, double complex *, double complex *, double complex *, MPI_Comm, double *),
    void (*precond_fun)(SPARC_OBJ *,int,double complex,double complex *,double complex *, MPI_Comm),
    double complex c, 
    int N, double complex *x, double complex *b, 
    double omega, double beta, int m, int p, double tol, 
    int max_iter, MPI_Comm comm
)
{
#define X(i,j) X[(j)*N+(i)]
#define F(i,j) F[(j)*N+(i)]
    if (comm == MPI_COMM_NULL) return;
    
    int rank, i, iter_count, i_hist;
    double complex *r, *x_old, *f, *f_old, *X, *F;
    double b_2norm, r_2norm;
    
    double t1, t2, tt1, tt2, ttot, t_anderson, ttot2, ttot3, ttot4;
    
    ttot = ttot2 = ttot3 = ttot4 = 0.0;
    
    MPI_Comm_rank(comm, &rank);
    
    // allocate memory for storing x, residual and preconditioned residual in the local domain
    r       = (double complex *)malloc( N * sizeof(double complex) ); // residual vector, r = b - Ax
    x_old   = (double complex *)malloc( N * sizeof(double complex) );
    f       = (double complex *)malloc( N * sizeof(double complex) ); // preconditioned residual vector, f = inv(M) * r
    f_old   = (double complex *)malloc( N * sizeof(double complex) );    
    assert(r != NULL && x_old != NULL && f != NULL && f_old != NULL);

    // allocate memory for storing X, F history matrices
    X = (double complex *)calloc( N * m, sizeof(double complex) ); 
    F = (double complex *)calloc( N * m, sizeof(double complex) ); 
    assert(X != NULL && F != NULL);

    // initialize x_old as x0 (initial guess vector)
    for (i = 0; i < N; i++) 
        x_old[i] = x[i];

    t1 = MPI_Wtime();
    Vector2Norm_complex(b, N, &b_2norm, comm); // b_2norm = ||b||
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (rank == 0) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, (t2-t1)*1e3);
#endif
    // find initial residual vector r = b - Ax, and its 2-norm
    res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);

    //Vector2Norm(r, N, &r_2norm, comm); // r_2norm = ||r||

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
            AndersonExtrapolation_complex(N, m, x, x_old, f, X, F, beta, comm);
            tt2 = MPI_Wtime();
            ttot += (tt2-tt1);
            ttot4 += t_anderson;

            tt1 = MPI_Wtime();
            // update residual r = b - Ax
            res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
                    
            Vector2Norm_complex(r, N, &r_2norm, comm); // r_2norm = ||r||
        } else {
            /***********************
             *  Richardson update  *
             ***********************/
            for (i = 0; i < N; i++)
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
 * @brief   Alternating Anderson-Richardson (AAR) method for solving a general linear
 *          system Ax = b. 
 *
 *          AAR_gen() assumes that x and  b is distributed among the given communicator. 
 *          The residual r = b - Ax is calculated by calling res_fun(). The function
 *          precond_fun() takes a residual vector r and precondition the residual by 
 *          applying inv(M)*r, where M is the preconditioner. This function takes in
 *          any user-provided struct AAR_buff and use it for both the res_fun() and
 *          the precond_fun(), so that user doesn't have to modify this funtion. 
 */
void AAR_gen(
    const void *AAR_buff, 
    void (*res_fun)(const void*,int,double*,double*,double*,MPI_Comm,double*),  
    void (*precond_fun)(const void*,int,double*,double*,MPI_Comm),
    int N, double *x, double *b, double omega, double beta, int m, int p, 
    double tol, int max_iter, MPI_Comm comm
)
{
#define X(i,j) X[(j)*N+(i)]
#define F(i,j) F[(j)*N+(i)]
    if (comm == MPI_COMM_NULL) return;
    
    int rank, i, iter_count, i_hist;
    double *r, *x_old, *f, *f_old, *X, *F, b_2norm, r_2norm;

    double t1, t2, tt1, tt2, ttot, t_anderson, ttot2, ttot3, ttot4;
    
    ttot = 0.0;
    ttot2 = 0.0;
    ttot3 = 0.0;
    ttot4 = 0.0;
    
    MPI_Comm_rank(comm, &rank);
    
    // allocate memory for storing x, residual and preconditioned residual in the local domain
    r = (double *)malloc( N * sizeof(double) );  // residual vector, r = b - Ax
    x_old   = (double *)malloc( N * sizeof(double) );
    f       = (double *)malloc( N * sizeof(double) ); // preconditioned residual vector, f = inv(M) * r
    f_old   = (double *)malloc( N * sizeof(double) );    
    if (r == NULL || x_old == NULL || f == NULL || f_old == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    // allocate memory for storing X, F history matrices
    X = (double *)calloc( N * m, sizeof(double) ); 
    F = (double *)calloc( N * m, sizeof(double) ); 
    if (X == NULL || F == NULL) {
        printf("\nMemory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // initialize x_old as x0 (initial guess vector)
    for (i = 0; i < N; i++) 
        x_old[i] = x[i];

    t1 = MPI_Wtime();
    Vector2Norm(b, N, &b_2norm, comm); // b_2norm = ||b||
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (rank == 0) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, (t2-t1)*1e3);
#endif
    // find initial residual vector r = b - Ax, and its 2-norm
    //res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);
    res_fun(AAR_buff, N, x, b, r, comm, &t_anderson);
    //Vector2Norm(r, N, &r_2norm, comm); // r_2norm = ||r||

    // replace the abs tol by scaled tol: tol * ||b||
    tol *= b_2norm; 
    r_2norm = tol + 1.0; // to reduce communication
    iter_count = 0;
    while (r_2norm > tol && iter_count < max_iter) {
        //if (rank == 0) printf("iteration #%d, r_2norm = %.3e\n", iter_count, r_2norm);       
        // *** calculate preconditioned residual f *** //
        //precond_fun(pSPARC, N, c, r, f, comm); // f = inv(M) * r
        precond_fun(AAR_buff, N, r, f, comm); // f = inv(M) * r

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
            AndersonExtrapolation(N, m, x, x, f, X, F, beta, comm);
            tt2 = MPI_Wtime();
            ttot += (tt2-tt1);
            ttot4 += t_anderson;

            tt1 = MPI_Wtime();
            // update residual r = b - Ax
            //res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);
            res_fun(AAR_buff, N, x, b, r, comm, &t_anderson);
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
                    
            Vector2Norm(r, N, &r_2norm, comm); // r_2norm = ||r||
        } else {
            /***********************
             *  Richardson update  *
             ***********************/
            for (i = 0; i < N; i++)
                x[i] = x[i] + omega * f[i];  
            tt1 = MPI_Wtime();   
            // update residual r = b - Ax
            //res_fun(pSPARC, N, c, x, b, r, comm, &t_anderson);  
            res_fun(AAR_buff, N, x, b, r, comm, &t_anderson);
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            ttot3 += t_anderson;
        }
        iter_count++;
    }
#ifdef DEBUG
    if (rank == 0) printf("\niter_count = %d, r_2norm = %.3e, tol*||rhs|| = %.3e\n\n", iter_count, r_2norm, tol);    
    if (rank == 0) printf("Anderson update took %.3f ms, out of which F'*F took %.3f ms; b-Ax took %.3f ms, out of which Lap took %.3f ms\n", ttot * 1e3, ttot4*1e3, ttot2 * 1e3, ttot3 * 1e3);
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


void  AndersonExtrapolation_complex(
        const int N, const int m, double complex *x_kp1, const double complex *x_k, 
        const double complex *f_k, const double complex *X, const double complex *F, 
        const double beta, MPI_Comm comm
) 
{
    unsigned i;
    double complex *f_wavg = (double complex *)malloc( N * sizeof(double complex) );
    
    // find the weighted average vectors
    AndersonExtrapWtdAvg_complex(N, m, x_k, f_k, X, F, x_kp1, f_wavg, comm);
    
    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] += beta * f_wavg[i];
    
    free(f_wavg);
}




void  AndersonExtrapWtdAvg_complex(
        const int N, const int m, const double complex *x_k, const double complex *f_k, 
        const double complex *X, const double complex *F, double complex *x_wavg, double complex *f_wavg, 
        MPI_Comm comm
) 
{
    double complex *Gamma = (double complex *)calloc( m , sizeof(double complex) );
    assert(Gamma != NULL); 
    
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k
    AndersonExtrapCoeff_complex(N, m, f_k, F, Gamma, comm); 
    
    unsigned i;
    
    // find weighted average x_{k+1} = x_k - X*Gamma
    for (i = 0; i < N; i++) x_wavg[i] = x_k[i];
    double complex alpha, beta;
    alpha = -1.0; beta = 1.0;
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, m, &alpha, X, 
        N, Gamma, 1, &beta, x_wavg, 1);

    // find weighted average f_{k+1} = f_k - F*Gamma
    for (i = 0; i < N; i++) f_wavg[i] = f_k[i];

    alpha = -1.0; beta = 1.0;
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, m, &alpha, F, 
        N, Gamma, 1, &beta, f_wavg, 1);

    free(Gamma);
}




void AndersonExtrapCoeff_complex(
    const int N, const int m, const double complex *f, const double complex *F, 
    double complex *Gamma, MPI_Comm comm
) 
{
// #define FtF(i,j) FtF[(j)*m+(i)]
    int matrank;
    double complex *FtF;
    double *s;

    FtF = (double complex *)malloc( m * m * sizeof(double complex) );
    s   = (double *)malloc( m * sizeof(double) );
    assert(FtF != NULL && s != NULL);  

    //# If mkl-11.3 or later version is available, one may use cblas_zgemmt #
    // calculate F^T * F, only update the LOWER part of the matrix
    //cblas_zgemmt(CblasColMajor, CblasLower, CblasTrans, CblasNoTrans, 
    //             m, N, 1.0, F, N, F, N, 0.0, FtF_Ftf, m);
    //// copy the lower half of the matrix to the upper half (LOCAL)
    //for (j = 0; j < m; j++)
    //    for (i = 0; i < j; i++)
    //        FtF_Ftf(i,j) = FtF_Ftf(j,i);
    
    //#   Otherwise use cblas_dgemm instead    #
    double complex alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, m, N, 
                &alpha, F, N, F, N, &beta, FtF, m);

    // calculate F^T * f using CBLAS  (LOCAL)
    alpha = 1.0; beta = 0.0;
    cblas_zgemv(CblasColMajor, CblasConjTrans, N, m, 
                &alpha, F, N, f, 1, &beta, Gamma, 1);

    // Sum the local results of F^T * F and F^T * f (GLOBAL)
    MPI_Allreduce(MPI_IN_PLACE, FtF, m*m, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, Gamma, m, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);

    // find inv(F^T * F) * (F^T * f) by solving (F^T * F) * x = F^T * f (LOCAL)
    LAPACKE_zgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF, m, Gamma, m, s, -1.0, &matrank);

    free(s);
    free(FtF);
// #undef FtF 
}

