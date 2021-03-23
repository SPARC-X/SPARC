/**
 * @file    mixing.c
 * @brief   This file contains the functions for mixing.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
/* BLAS, LAPACK, LAPACKE routines */
#ifdef USE_MKL
    #define MKL_Complex16 double complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "tools.h" 
#include "mixing.h"
#include "lapVecRoutines.h"
#include "isddft.h"
#include "linearSolver.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))



/**
 * @brief   Anderson extrapolation update.
 *
 *          x_{k+1} = (x_k - X * Gamma) + beta * P * (f_k - F * Gamma),
 *          where P is the preconditioner, and Gamma = inv(F^T * F) * F^T * f.
 *          Expanding above equation gives: 
 *          x_{k+1} = x_k + beta * P * f - (X + beta * P * F) * inv(F^T * F) * F^T * f          
 */
void  AndersonExtrapolation(
        const int N, const int m, double *x_kp1, const double *x_k, 
        const double *f_k, const double *X, const double *F, 
        const double beta, MPI_Comm comm
) 
{
    unsigned i;
    double *f_wavg = (double *)malloc( N * sizeof(double) );
    
    // find the weighted average vectors
    AndersonExtrapWtdAvg(N, m, x_k, f_k, X, F, x_kp1, f_wavg, comm);
    
    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] += beta * f_wavg[i];
    
    free(f_wavg);
}



/**
 * @brief   Anderson extrapolation weighted average vectors.
 *
 *          Find x_wavg := x_k - X * Gamma and f_wavg := (f_k - F * Gamma),
 *          where Gamma = inv(F^T * F) * F^T * f.
 */
void  AndersonExtrapWtdAvg(
        const int N, const int m, const double *x_k, const double *f_k, 
        const double *X, const double *F, double *x_wavg, double *f_wavg, 
        MPI_Comm comm
) 
{
    double *Gamma = (double *)calloc( m , sizeof(double) );
    assert(Gamma != NULL); 
    
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k
    AndersonExtrapCoeff(N, m, f_k, F, Gamma, comm); 
    
    unsigned i;
    
    // find weighted average x_{k+1} = x_k - X*Gamma
    // memcpy(x_wavg, x_k, N*sizeof(*x_k)); // copy x_k into x_wavg
    for (i = 0; i < N; i++) x_wavg[i] = x_k[i];
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, -1.0, X, 
        N, Gamma, 1, 1.0, x_wavg, 1);

    // find weighted average f_{k+1} = f_k - F*Gamma
    // memcpy(f_wavg, f_k, N*sizeof(*f_k)); // copy f_k into f_wavg
    for (i = 0; i < N; i++) f_wavg[i] = f_k[i];
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, m, -1.0, F, 
        N, Gamma, 1, 1.0, f_wavg, 1);

    free(Gamma);
}



/**
 * @brief   Anderson extrapolation coefficiens.
 *
 *          Gamma = inv(F^T * F) * F^T * f.         
 */
void AndersonExtrapCoeff(
    const int N, const int m, const double *f, const double *F, 
    double* Gamma, MPI_Comm comm
) 
{
// #define FtF(i,j) FtF[(j)*m+(i)]
    // int i, j;
    int matrank;
    double *FtF, *s;
    
    FtF = (double *)malloc( m * m * sizeof(double) );
    s   = (double *)malloc( m * sizeof(double) );
    assert(FtF != NULL && s != NULL);  

    //# If mkl-11.3 or later version is available, one may use cblas_dgemmt #
    // calculate F^T * F, only update the LOWER part of the matrix
    //cblas_dgemmt(CblasColMajor, CblasLower, CblasTrans, CblasNoTrans, 
    //             m, N, 1.0, F, N, F, N, 0.0, FtF_Ftf, m);
    //// copy the lower half of the matrix to the upper half (LOCAL)
    //for (j = 0; j < m; j++)
    //    for (i = 0; i < j; i++)
    //        FtF_Ftf(i,j) = FtF_Ftf(j,i);
    
    //#   Otherwise use cblas_dgemm instead    #
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, N, 
                1.0, F, N, F, N, 0.0, FtF, m);

    // calculate F^T * f using CBLAS  (LOCAL)
    cblas_dgemv(CblasColMajor, CblasTrans, N, m, 
                1.0, F, N, f, 1, 0.0, Gamma, 1);

    // Sum the local results of F^T * F and F^T * f (GLOBAL)
    MPI_Allreduce(MPI_IN_PLACE, FtF, m*m, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, Gamma, m, MPI_DOUBLE, MPI_SUM, comm);

    // find inv(F^T * F) * (F^T * f) by solving (F^T * F) * x = F^T * f (LOCAL)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF, m, Gamma, m, s, -1.0, &matrank);

    free(s);
    free(FtF);
// #undef FtF 
}


/**
 * @brief   Perform mixing and preconditioner.
 *
 *          Note that this is only done in "phi-domain".
 */
void Mixing(SPARC_OBJ *pSPARC, int iter_count)
{
    // apply pulay mixing
    Mixing_periodic_pulay(pSPARC, iter_count);
    
    // TODO: implement other mixing schemes
}


/**
 * @brief   Perform pulay mixing.
 *
 *          Note that this is only done in "phi-domain".
 */
void Mixing_periodic_pulay(SPARC_OBJ *pSPARC, int iter_count)
{
#define R(i,j) (*(R+(j)*N+(i)))
#define F(i,j) (*(F+(j)*N+(i)))

    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1, t2;
    int N, p, m, i, i_hist;
    double omega, beta, *g_k, *f_k, *Pf, *x_k,
           *x_km1, *x_kp1, *f_km1, *R, *F;
    int sindx_rho = (pSPARC->Nspin == 2) * pSPARC->Nd_d;

    double precond_tol = pSPARC->TOL_PRECOND;

    N = pSPARC->Nd_d * pSPARC->Nspin;
    m = pSPARC->MixingHistory;
    p = pSPARC->PulayFrequency;
    beta = pSPARC->MixingParameter;
    omega = pSPARC->MixingParameterSimple;

    // g_k = g(x_k), not mixed (or x^{out}_k)
    // Note that x_kp1 points to the same location as g_k!
    // the unmixed g_k will be overwritten in the end
    if (pSPARC->MixingVariable == 0) {        // density mixing
        g_k = x_kp1 = pSPARC->electronDens + sindx_rho; 
    } else if (pSPARC->MixingVariable == 1) { // potential mixing
        g_k = x_kp1 = pSPARC->Veff_loc_dmcomm_phi;
    } else {
        exit(EXIT_FAILURE);
    }
    x_k   = pSPARC->mixing_hist_xk;   // the current mixed var x (x^{in}_k)
    x_km1 = pSPARC->mixing_hist_xkm1; // x_{k-1}
    f_k   = pSPARC->mixing_hist_fk;   // f_k = g(x_k) - x_k
    f_km1 = pSPARC->mixing_hist_fkm1; // f_{k-1}
    R     = pSPARC->mixing_hist_Xk;   // [x_{k-m+1} - x_{k-m}, ... , x_k - x_{k-1}]
    F     = pSPARC->mixing_hist_Fk;   // [f_{k-m+1} - f_{k-m}, ... , f_k - f_{k-1}]
    Pf    = pSPARC->mixing_hist_Pfk;  // the preconditioned residual

    // update old residual f_{k-1}
    if (iter_count > 0) {
        for (i = 0; i < N; i++) f_km1[i] = f_k[i];
    }
    
    // compute current residual     
    for (i = 0; i < N; i++) f_k[i] = g_k[i] - x_k[i];     

    // *** store residual & iteration history *** //
    if (iter_count > 0) {
        i_hist = (iter_count - 1) % m;
        if (pSPARC->PulayRestartFlag && iter_count % (p+1) == 0) {
            // TODO: check if this is necessary!
            for (i = 0; i < N*(m-1); i++) {
                R[N+i] = F[N+i] = 0.0; // set all cols to 0 (except for 1st col)
            }
            
            // set 1st cols of R and F
            for (i = 0; i < N; i++) {
                // R[i] = Dx[i];
                R[i] = x_k[i] - x_km1[i];
                // F[i] = f[i] - f_old[i];
                F[i] = f_k[i] - f_km1[i];
            }
        } else {
            for (i = 0; i < N; i++) {
                // R(i, i_hist) = Dx[i];
                R(i, i_hist) = x_k[i] - x_km1[i];
                // F(i, i_hist) = f[i] - f_old[i];
                F(i, i_hist) = f_k[i] - f_km1[i];
            }
        }
    }
    
    // update x_{k+1} 
    if((iter_count+1) % p == 0 && iter_count > 0) {
        /***********************************
         *  Anderson extrapolation update  *
         ***********************************/    
        double *f_wavg = (double *)malloc( N * sizeof(double) );
        double temp1, temp2;
        assert(f_wavg != NULL);
        
        AndersonExtrapWtdAvg(
            N, m, x_k, f_k, R, F, x_kp1, f_wavg, pSPARC->dmcomm_phi
        ); 
        
        // apply preconditioner if required, Pf = P * f_wavg
        if (pSPARC->MixingPrecond != 0) { // apply preconditioner
            if(pSPARC->spin_typ == 0) {
                if (pSPARC->MixingPrecond == 1) { // kerker preconditioner
                    double k_TF = pSPARC->precond_kerker_kTF;
                    double idiemac = pSPARC->precond_kerker_thresh;
                    Kerker_precond(
                        pSPARC, f_wavg, 1.0, 
                        k_TF, idiemac, precond_tol, N, 
                        pSPARC->DMVertices, Pf, pSPARC->dmcomm_phi
                    );
                }
            } else {
                for (i = 0; i < pSPARC->Nd_d; i++){
                    temp1 = f_wavg[i] + f_wavg[pSPARC->Nd_d+i]; // weighted average of residual of total density
                    temp2 = f_wavg[i] - f_wavg[pSPARC->Nd_d+i]; // weighted average of residual of magnetization density
                    f_wavg[i] = temp1;
                    f_wavg[pSPARC->Nd_d+i] = temp2;
                }
                if (pSPARC->MixingPrecond == 1) { // kerker preconditioner
                    double k_TF = pSPARC->precond_kerker_kTF;
                    double idiemac = pSPARC->precond_kerker_thresh;
                    Kerker_precond(
                        pSPARC, f_wavg, 1.0, 
                        k_TF, idiemac, precond_tol, pSPARC->Nd_d, 
                        pSPARC->DMVertices, Pf, pSPARC->dmcomm_phi
                    ); // preconditioned weighted average of residual of total density
                }

                for (i = 0; i < pSPARC->Nd_d; i++) {
                    temp1 = Pf[i];
                    Pf[i] = (temp1 + f_wavg[pSPARC->Nd_d+i])/2.0;
                    Pf[pSPARC->Nd_d+i] = (temp1 - f_wavg[pSPARC->Nd_d+i])/2.0; 
                }
            }
            
        } else {
            Pf = f_wavg;
        }
        
        // add beta * f to x_{k+1}
        for (i = 0; i < N; i++)
            x_kp1[i] += beta * Pf[i];
        
        free(f_wavg);
    } else {
        /***********************
         *  Richardson update  *
         ***********************/
        // apply preconditioner if required, Pf = P * f_k
        if (pSPARC->MixingPrecond != 0) { // apply preconditioner
            if(pSPARC->spin_typ == 0) {
                if (pSPARC->MixingPrecond == 1) { // kerker preconditioner
                    double k_TF = pSPARC->precond_kerker_kTF;
                    double idiemac = pSPARC->precond_kerker_thresh;
                    Kerker_precond(
                        pSPARC, f_k, 1.0, 
                        k_TF, idiemac, precond_tol, N, 
                        pSPARC->DMVertices, Pf, pSPARC->dmcomm_phi
                    );
                }
            } else {
                double *f_temp = (double *)malloc( N * sizeof(double) );
                double temp;
                assert(f_temp != NULL);

                for (i = 0; i < pSPARC->Nd_d; i++){
                    f_temp[i] = f_k[i] + f_k[pSPARC->Nd_d+i]; // residual of total density
                    f_temp[pSPARC->Nd_d+i] = f_k[i] - f_k[pSPARC->Nd_d+i]; // residual of magnetization density
                }

                if (pSPARC->MixingPrecond == 1) { // kerker preconditioner
                    double k_TF = pSPARC->precond_kerker_kTF;
                    double idiemac = pSPARC->precond_kerker_thresh;
                    Kerker_precond(
                        pSPARC, f_temp, 1.0, 
                        k_TF, idiemac, precond_tol, pSPARC->Nd_d, 
                        pSPARC->DMVertices, Pf, pSPARC->dmcomm_phi
                    ); // preconditioned residual of total density
                }

                for (i = 0; i < pSPARC->Nd_d; i++) {
                    temp = Pf[i];
                    Pf[i] = (temp + f_temp[pSPARC->Nd_d+i])/2.0;
                    Pf[pSPARC->Nd_d+i] = (temp - f_temp[pSPARC->Nd_d+i])/2.0; 
                }

                free(f_temp);
            }
            
        } else {
            Pf = f_k;
        }
        for (i = 0; i < N; i++)
            x_kp1[i] = x_k[i] + omega * Pf[i];
    }


    // for density mixing, need to check if rho < 0
    if (pSPARC->MixingVariable == 0) {
        int neg_flag = 0;
        for (i = 0; i < N; i++) {
            if (x_kp1[i] < 0.0) {
                x_kp1[i] = 0.0;
                neg_flag = 1;
            }
        }

        if(pSPARC->spin_typ != 0) {
            for (i = 0; i < pSPARC->Nd_d; i++)
                pSPARC->electronDens[i] = x_kp1[i] + x_kp1[pSPARC->Nd_d + i];
        }
        
        MPI_Allreduce(MPI_IN_PLACE, &neg_flag, 1, MPI_INT,
                      MPI_SUM, pSPARC->dmcomm_phi);
        
        if (neg_flag > 0) {
            if (rank == 0) printf("WARNING: The density after mixing has negative components!\n");
        }
        
        // consider doing this every time, not just when density is negative
        if (neg_flag > 0 || 1) { // Warning: forcing if cond to be always true
            double int_rho = 0.0;
            for (i = 0; i < pSPARC->Nd_d; i++) {
                int_rho += pSPARC->electronDens[i];
            }
            int_rho *= pSPARC->dV;

            // use MPI_Reduce/MPI_Allreduce to find the global sum (based on the local sums)
            t1 = MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE, &int_rho, 1, MPI_DOUBLE,
                          MPI_SUM, pSPARC->dmcomm_phi);
            t2 = MPI_Wtime();

#ifdef DEBUG    
            if (rank == 0) printf("time taken by All reduce for int_rho(%.15f) in mixing: %.3f ms\n",int_rho, (t2-t1)*1e3);  
#endif

            /*  Scale electron density so that PosCharge + NegCharge = NetCharge  */
            double scal_fac = pSPARC->PosCharge / int_rho;
            
            int len;
            if (pSPARC->spin_typ != 0) { // for spin polarized
                len = 3 * pSPARC->Nd_d;
            } else {
                len = pSPARC->Nd_d;      // for spin unpolarized
            }
            for (i = 0; i < len; i++) {
                pSPARC->electronDens[i] *= scal_fac;
            }    
        }
    }
    
    // update x_k and change in mixed x
    for (i = 0; i < N; i++) {
        // update x_{k-1} = x_k
        x_km1[i] = x_k[i];
        // update x_k = x_{k+1};
        x_k[i] = x_kp1[i];
    }

#undef R
#undef F
}



/**
 * @brief   Perform Kerker preconditioner.
 *
 *          Apply Kerker preconditioner in real space. For given 
 *          function f, this function returns 
 *          Pf := a * (L - lambda_TF^2)^-1 * (L - idemac*lambda_TF^2)f, 
 *          where L is the discrete Laplacian operator, c is the 
 *          inverse of diemac (dielectric macroscopic constant).
 *          When c is 0, it's the original Kerker preconditioner.
 *          The result is written in Pf.
 */
void Kerker_precond(
    SPARC_OBJ *pSPARC, double *f, const double a, 
    const double lambda_TF, const double idiemac, const double tol, 
    const int DMnd, const int *DMVertices, double *Pf, MPI_Comm comm
)
{
    if (comm == MPI_COMM_NULL) return;

    #ifdef DEBUG
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) printf("Start applying Kerker preconditioner ...\n");
    #endif

    // declare the Jacobi preconditioner, defined in electrostatics.c
    void Jacobi_preconditioner(SPARC_OBJ *pSPARC, int N, double c, double *r,  
                               double *f, MPI_Comm comm);
    // function pointer of the preconditioner
    void (*precond_fun) (
        SPARC_OBJ*, int, double, double*, double*, MPI_Comm
    ) = Jacobi_preconditioner;
     // function pointer that applies b + (Laplacian + c) * x
    void (*res_fun) (
        SPARC_OBJ*, int, double, double*, double*, double*, MPI_Comm, double*
    ) = poisson_residual; // poisson_residual is defined in lapVecRoutines.c

    double *Lf;
    Lf = (double *)malloc(DMnd * sizeof(double));
    // find Lf = (L - idemac*lambda_TF^2)f
    Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, -lambda_TF*lambda_TF*idiemac, f, Lf, comm);

    //** solve -(L - lambda_TF^2) * Pf = Lf for Pf **//
    double omega = 0.6, beta = 0.6; 
    int    m = 7, p = 6;
    //double omega = 0.1, beta = 0.1; 
    //int    m = 7, p = 1;
    
    double Lf_2norm = 0.0;
    Vector2Norm(Lf, DMnd, &Lf_2norm, comm); // Lf_2norm = ||Lf||

    AAR(pSPARC, res_fun, precond_fun, -lambda_TF*lambda_TF, DMnd, Pf, Lf, 
        omega, beta, m, p, tol, 1000, comm);
    
    int i;
    if (fabs(lambda_TF) < 1e-14) {
        // in this case the result will be shifted by a constant
        double shift = 0.0;
        for (i = 0; i < DMnd; i++) {
            shift += Pf[i]; 
        }
        MPI_Allreduce(MPI_IN_PLACE, &shift, 1, MPI_DOUBLE, MPI_SUM, comm);
        #ifdef DEBUG
        if (!rank) printf("Kerker precond: Sum of Pf before shift = %.3e\n",shift);
        #endif
        
        shift /= pSPARC->Nd;
        for (i = 0; i < DMnd; i++) {
            Pf[i] -= shift; 
        }
    }
    
    for (i = 0; i < DMnd; i++) {
        Pf[i] *= -a; // note the '-' sign 
    }
    
    free(Lf);
}

