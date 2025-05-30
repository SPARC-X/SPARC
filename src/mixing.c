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
#include <string.h>
#include <mpi.h>
/* BLAS, LAPACK, LAPACKE routines */
#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
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
#include "electronDensity.h"

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
    // use Nspden = 0, opt = 0 here. default as previous
    double *mix_Gamma = (double *)malloc(m * sizeof(double));
    AndersonExtrapWtdAvg(N, m, 0, 0, x_k, f_k, X, F, x_kp1, f_wavg, comm, mix_Gamma);
    
    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] += beta * f_wavg[i];
    
    free(f_wavg); free(mix_Gamma);
}



/**
 * @brief   Anderson extrapolation weighted average vectors.
 *
 *          Find x_wavg := x_k - X * Gamma and f_wavg := (f_k - F * Gamma),
 *          where Gamma = inv(F^T * F) * F^T * f.
 */
void  AndersonExtrapWtdAvg(
        const int N, const int m, const int Nspden, const int opt, 
        const double *x_k, const double *f_k, 
        const double *X, const double *F, double *x_wavg, double *f_wavg, 
        MPI_Comm comm, double *mix_Gamma
) 
{
    double *Gamma = (double *)calloc( m , sizeof(double) );
    assert(Gamma != NULL); 
    
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k
    AndersonExtrapCoeff(N, m, Nspden, opt, f_k, F, Gamma, comm); 
    
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

    memcpy(mix_Gamma, Gamma, m*sizeof(double)); // output gamma
    free(Gamma);
}



/**
 * @brief   Anderson extrapolation coefficiens.
 *
 *          Gamma = inv(F^T * F) * F^T * f.         
 */
void AndersonExtrapCoeff(
    const int N, const int m, const int Nspden, const int opt, 
    const double *f, const double *F, double* Gamma, MPI_Comm comm
) 
{
    int matrank;
    double *FtF, *s;
    
    FtF = (double *)malloc( m * m * sizeof(double) );
    s   = (double *)malloc( m * sizeof(double) );
    assert(FtF != NULL && s != NULL);  
    
    compute_FtF(F, m, N, Nspden, opt, FtF);
    compute_Ftf(F, f, m, N, Nspden, opt, Gamma);

    // Sum the local results of F^T * F and F^T * f (GLOBAL)
    MPI_Allreduce(MPI_IN_PLACE, FtF, m*m, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, Gamma, m, MPI_DOUBLE, MPI_SUM, comm);

    // find inv(F^T * F) * (F^T * f) by solving (F^T * F) * x = F^T * f (LOCAL)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF, m, Gamma, m, s, -1.0, &matrank);

    free(s);
    free(FtF);
}

void compute_FtF(const double *F, int m, int N, int Nspden, int opt, double *FtF)
{
#define FtF(i,j) FtF[i+j*m]
    if (Nspden == 4) {
        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                FtF(i,j) = dotprod_nc(F+i*N, F+j*N, N, opt);
                if (j != i) FtF(j,i) = FtF(i,j);
            }
        }
    } else {
        // calculate F^T * F using CBLAS  (LOCAL)
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, N, 
                1.0, F, N, F, N, 0.0, FtF, m);
    }
#undef FtF
}

void compute_Ftf(const double *F, const double *f, int m, int N, int Nspden, int opt, double *Ftf)
{
    if (Nspden == 4) {
        for (int i = 0; i < m; i++) {
            Ftf[i] = dotprod_nc(F+i*N, f, N, opt);
        }
    } else {
        // calculate F^T * f using CBLAS  (LOCAL)
        cblas_dgemv(CblasColMajor, CblasTrans, N, m, 
                    1.0, F, N, f, 1, 0.0, Ftf, 1);
    }
}

double dotprod_nc(const double *vec1, const double *vec2, int N, int opt)
{    
    double dotprod = 0;
    if (opt == 1) { // potential mixing        
        for (int i = 0; i < N/2; i++) {
            dotprod += vec1[i] * vec2[i];
        }
        double sum_temp = 0;
        for (int i = N/2; i < N; i++) {
            sum_temp += vec1[i] * vec2[i];
        }
        dotprod += 2*sum_temp;
    } else { // density mixing
        for (int i = 0; i < N; i++) {
            dotprod += vec1[i] * vec2[i];
        }
        dotprod *= 0.5;
    }
    return dotprod;
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
    
    double *g_k = NULL, *x_kp1 = NULL;
    double precond_tol = pSPARC->TOL_PRECOND;

    int DMnd = pSPARC->Nd_d;
    int N = DMnd * pSPARC->Nspden;
    int m = pSPARC->MixingHistory;
    int p = pSPARC->PulayFrequency;
    double beta = pSPARC->MixingParameter;
    double omega = pSPARC->MixingParameterSimple;
    double beta_mag = pSPARC->MixingParameterMag;
    double omega_mag = pSPARC->MixingParameterSimpleMag;
    
    // flag for Pulay (Anderson extrapolation) mixing, otherwise apply simple mixing 
    int Pulay_mixing_flag = (int) ((iter_count+1) % p == 0 && iter_count > 0);
    
    // decide which mixing parameter to use
    double amix, amix_mag;
    if (Pulay_mixing_flag) { // pulay mixing
        amix = beta; amix_mag = beta_mag;
    } else { // simple (linear) mixing
        amix = omega; amix_mag = omega_mag;
    }

    // g_k = g(x_k), not mixed (or x^{out}_k)
    // Note that x_kp1 points to the same location as g_k!
    // the unmixed g_k will be overwritten in the end
    if (pSPARC->MixingVariable == 0) {        // density mixing
        if (pSPARC->spin_typ == 0) {
            g_k = x_kp1 = pSPARC->electronDens; 
        } else {
            g_k = x_kp1 = (double *) malloc(N*sizeof(double));
            memcpy(g_k, pSPARC->electronDens, sizeof(double)*DMnd); // rho
            if (pSPARC->spin_typ == 1) {
                memcpy(g_k+DMnd, pSPARC->mag, sizeof(double)*DMnd); // magz
            } else {
                memcpy(g_k+DMnd, pSPARC->mag+DMnd, sizeof(double)*DMnd*3); // magx magy magz
            }
        }
    } else if (pSPARC->MixingVariable == 1) { // potential mixing
        g_k = x_kp1 = pSPARC->Veff_loc_dmcomm_phi;
    } else {
        exit(EXIT_FAILURE);
    }
    double *x_k   = pSPARC->mixing_hist_xk;   // the current mixed var x (x^{in}_k)
    double *x_km1 = pSPARC->mixing_hist_xkm1; // x_{k-1}
    double *f_k   = pSPARC->mixing_hist_fk;   // f_k = g(x_k) - x_k
    double *f_km1 = pSPARC->mixing_hist_fkm1; // f_{k-1}
    double *R     = pSPARC->mixing_hist_Xk;   // [x_{k-m+1} - x_{k-m}, ... , x_k - x_{k-1}]
    double *F     = pSPARC->mixing_hist_Fk;   // [f_{k-m+1} - f_{k-m}, ... , f_k - f_{k-1}]
    double *Pf    = pSPARC->mixing_hist_Pfk;  // the preconditioned residual

    // update old residual f_{k-1}
    if (iter_count > 0) {
        for (int i = 0; i < N; i++) f_km1[i] = f_k[i];
    }
    
    // compute current residual     
    for (int i = 0; i < N; i++) f_k[i] = g_k[i] - x_k[i];     

    // *** store residual & iteration history *** //
    if (iter_count > 0) {
        int i_hist = (iter_count - 1) % m;
        if (pSPARC->PulayRestartFlag && i_hist == 0) {
            // TODO: check if this is necessary!
            for (int i = 0; i < N*(m-1); i++) {
                R[N+i] = F[N+i] = 0.0; // set all cols to 0 (except for 1st col)
            }
            
            // set 1st cols of R and F
            for (int i = 0; i < N; i++) {
                // R[i] = Dx[i];
                R[i] = x_k[i] - x_km1[i];
                // F[i] = f[i] - f_old[i];
                F[i] = f_k[i] - f_km1[i];
            }
        } else {
            for (int i = 0; i < N; i++) {
                // R(i, i_hist) = Dx[i];
                R(i, i_hist) = x_k[i] - x_km1[i];
                // F(i, i_hist) = f[i] - f_old[i];
                F(i, i_hist) = f_k[i] - f_km1[i];
            }
        }
    }
    
    double *f_wavg, *x_wavg;
    x_wavg = (double *)malloc( N * sizeof(double) );
    f_wavg = (double *)malloc( N * sizeof(double) );
    assert(x_wavg != NULL);
    assert(f_wavg != NULL);
    
    // if ((iter_count+1) % p == 0 && iter_count > 0) {
    if (Pulay_mixing_flag) {
        // Anderson extrapolation
        // find weighted average x_wavg, f_wavg
        AndersonExtrapWtdAvg(
            N, m, pSPARC->Nspden, pSPARC->MixingVariable, x_k, f_k, R, F, x_wavg, f_wavg, pSPARC->dmcomm_phi,
        pSPARC->mix_Gamma); 
    } else {
        // Simple (linear) mixing
        for (int i = 0; i < N; i++) {
            x_wavg[i] = x_k[i];
            f_wavg[i] = f_k[i];
        }
    }
    
    // calculate total density and magnetization density for spin-calculation   
    double sum_f[4] = {0,0,0,0};
    if (pSPARC->spin_typ > 0) {
        for (int n = 0; n < pSPARC->Nspden; n++) {
            VectorSum(f_wavg+n*DMnd,  DMnd, sum_f+n,  pSPARC->dmcomm_phi);
        }
    }

    // apply preconditioner if required, Pf = amix * (P * f_tot)
    if (pSPARC->MixingPrecond != 0) { // apply preconditioner
        if (pSPARC->MixingPrecond == 1) { // kerker preconditioner
            double k_TF = pSPARC->precond_kerker_kTF;
            double idiemac = pSPARC->precond_kerker_thresh;
            // precondition the residual of total density/potential
            Kerker_precond(
                pSPARC, f_wavg, amix, 
                k_TF, idiemac, precond_tol, DMnd, 
                pSPARC->DMVertices, Pf, pSPARC->dmcomm_phi
            ); 
        }
    } else {
        // un-preconditioned, i.e., P = amix * I
        for (int i = 0; i < DMnd; i++) {
            Pf[i] = amix * f_wavg[i];
        } 
    }

    // Apply preconditioner for magnetization density
    if (pSPARC->MixingPrecondMag == 1) { // Kerker preconditioner
        double k_TF_mag = pSPARC->precond_kerker_kTF_mag;
        double idiemac_mag = pSPARC->precond_kerker_thresh_mag;
        for (int n = 1; n < pSPARC->Nspden; n++) {
            Kerker_precond(
                pSPARC, f_wavg+n*DMnd, amix_mag, 
                k_TF_mag, idiemac_mag, precond_tol, DMnd, 
                pSPARC->DMVertices, Pf+n*DMnd, pSPARC->dmcomm_phi
            ); // precondition the residual of magnetization density
        }
    } else {
        // un-preconditioned, i.e., P = amix_mag * I
        for (int n = 1; n < pSPARC->Nspden; n++) {
            for (int i = 0; i < DMnd; i++) {
                Pf[i+n*DMnd] = amix_mag * f_wavg[i+n*DMnd];
            } 
        }
    }
    
    // shift Pf so that the integral is preserved
    // note that mixing param is already applied shift Pf
    double sum_Pf[4] = {0,0,0,0};
    if (pSPARC->spin_typ > 0) {
        for (int n = 0; n < pSPARC->Nspden; n++) {
            VectorSum(Pf+n*DMnd,  DMnd, sum_Pf+n,  pSPARC->dmcomm_phi);
            double shift = (sum_f[n] - sum_Pf[n]) / pSPARC->Nd;
            VectorShift(Pf+n*DMnd, DMnd, shift, pSPARC->dmcomm_phi);
        }
    }
    
    // find x_{k+1} := x_wavg + Pf (mixing param is included in Pf)
    for (int i = 0; i < N; i++) x_kp1[i] = x_wavg[i] + Pf[i];

    // for density mixing, need to check if rho < 0
    if (pSPARC->MixingVariable == 0) {
        int neg_flag = 0;
        for (int i = 0; i < DMnd; i++) {
            if (x_kp1[i] < 0.0) {
                x_kp1[i] = 0.0;
                neg_flag = 1;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &neg_flag, 1, MPI_INT,
                      MPI_SUM, pSPARC->dmcomm_phi);
        
        if (neg_flag > 0) {
            if (rank == 0) printf("WARNING: The density after mixing has negative components!\n");
        }

        // scale electron density so that PosCharge + NegCharge = NetCharge 
        double int_rho = 0.0;
        if (pSPARC->CyclixFlag) {
            VectorSum_wt(x_kp1, pSPARC->Intgwt_phi, DMnd, &int_rho, pSPARC->dmcomm_phi);
        } else {
            VectorSum(x_kp1, DMnd, &int_rho, pSPARC->dmcomm_phi);
            int_rho *= pSPARC->dV;
        }
        double scal_fac = -pSPARC->NegCharge / int_rho;
        for (int i = 0; i < DMnd; i++) {
            x_kp1[i] *= scal_fac;
        }

        if (pSPARC->spin_typ == 1) {
            memcpy(pSPARC->electronDens, x_kp1, sizeof(double)*DMnd); // rho
            memcpy(pSPARC->mag, x_kp1+DMnd, sizeof(double)*DMnd); // magz
            Calculate_diagonal_Density(pSPARC, DMnd, pSPARC->mag, pSPARC->electronDens, pSPARC->electronDens+DMnd, pSPARC->electronDens+2*DMnd); 
        } else if (pSPARC->spin_typ == 2) {
            memcpy(pSPARC->electronDens, x_kp1, sizeof(double)*DMnd); // rho
            memcpy(pSPARC->mag+DMnd, x_kp1+DMnd, sizeof(double)*DMnd*3); // magx magy magz
            Calculate_Magnorm(pSPARC, DMnd, pSPARC->mag+DMnd, pSPARC->mag+2*DMnd, pSPARC->mag+3*DMnd, pSPARC->mag); 
            Calculate_diagonal_Density(pSPARC, DMnd, pSPARC->mag, pSPARC->electronDens, pSPARC->electronDens+DMnd, pSPARC->electronDens+2*DMnd); 
        }
    }    

    free(x_wavg);
    free(f_wavg);
    
    // update x_km1 and x_k
    for (int i = 0; i < N; i++) {
        // update x_{k-1} = x_k
        x_km1[i] = x_k[i];
        // update x_k = x_{k+1};
        x_k[i] = x_kp1[i];
    }    

    if (pSPARC->MixingVariable == 0) {
        if (pSPARC->spin_typ) free(x_kp1);
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
    Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, -lambda_TF*lambda_TF*idiemac, f, DMnd, Lf, DMnd, comm);

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

