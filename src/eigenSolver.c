/**
 * @file    eigenSolver.c
 * @brief   This file contains the functions for eigenSolver.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *          Alfredo Metere (GPU support), Lawrence Livermore National Laboratory <metere1@llnl.gov>, <alfredo.metere@xsilico.com>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/* ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif

#include "eigenSolver.h"
#include "tools.h" 
#include "linearSolver.h" // Lanczos
#include "lapVecRoutines.h"
#include "hamiltonianVecRoutines.h"
#include "occupation.h"
#include "isddft.h"
#include "parallelization.h"
#include "linearAlgebra.h"
#include "cyclix_tools.h"

#ifdef SPARCX_ACCEL
#include "accel.h"
#endif

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
int CheFSI_use_EVA = -1;
#endif


/*
 * @brief Set up initial guess for Lanczos.
 *        The eigvecs of Laplacian can be used as good initial guess for the Hamiltonian. 
 *        On the other hand, one can also use random vectors as initial guess.
 *
 * @param x0         Output vector.
 * @param gridsizes  Global grid sizes, [Nx,Ny,Nz].
 * @param DMVert     Local domain vertices owned by the current process.
 * @param RandFlag   Flag that specifies whether random vectors are used.
 * @param comm       The communicator where the vector is distributed.
 */
void init_guess_Lanczos(
    double *x0, double cellsizes[3], int gridsizes[3], double meshes[3], 
    int DMVert[6], int RandFlag, MPI_Comm comm
) {
    if (comm == MPI_COMM_NULL) return;

    int DMnx = DMVert[1] - DMVert[0] + 1;
    int DMny = DMVert[3] - DMVert[2] + 1;
    int DMnz = DMVert[5] - DMVert[4] + 1;
    int DMnd = DMnx * DMny * DMnz;

    if (RandFlag == 1) {
        SetRandMat(x0, DMnd, 1, 0.0, 1.0, comm); // TODO: change for FixRandSeed = 1
    } else {
        // set x0 to be eigenvector corresponding to max eigval of the 
        // periodic discrete Laplacian, let x, y, z be max eigvec of 1D 
        // discrete Laplacian, then x0 = kron(z,kron(y,x))
        int Nx = gridsizes[0];
        int Ny = gridsizes[1];
        int Nz = gridsizes[2];
        double Lx = cellsizes[0];
        double Ly = cellsizes[1];
        double Lz = cellsizes[2];

        // x0 = c1 * cos(2 \pi nx/L) + c2 * sin(2 \pi nx/L)
        double c1 = 0.0001;
        double c2 = 1.0;
        double kx = 2.0 * M_PI * (Nx / 2) / Lx;
        double ky = 2.0 * M_PI * (Ny / 2) / Ly;
        double kz = 2.0 * M_PI * (Nz / 2) / Lz;
        double dx = meshes[0];
        double dy = meshes[1];
        double dz = meshes[2];
        
        // use Kronecker product
        double *vx_1d, *vy_1d, *vz_1d, *kron_vy_vx;
        vx_1d = (double*)malloc(DMnx * sizeof(double));
        vy_1d = (double*)malloc(DMny * sizeof(double));
        vz_1d = (double*)malloc(DMnz * sizeof(double));
        kron_vy_vx = (double*)malloc(DMnx * DMny * sizeof(double));
        for (int ii = 0; ii < DMnx; ii++) {
            int i_global = ii + DMVert[0]; // global coord
            double vx = c1 * cos(kx * i_global * dx) + c2 * sin(kx * i_global * dx);
            vx_1d[ii] = vx;
        }
        for (int jj = 0; jj < DMny; jj++) {
            int j_global = jj + DMVert[2]; // global coord
            double vy = c1 * cos(ky * j_global * dy) + c2 * sin(ky * j_global * dy);
            vy_1d[jj] = vy;
        }
        for (int kk = 0; kk < DMnz; kk++) {
            int k_global = kk + DMVert[4]; // global coord
            double vz = c1 * cos(kz * k_global * dz) + c2 * sin(kz * k_global * dz);
            vz_1d[kk] = vz;
        }
        
        memset(kron_vy_vx, 0, sizeof(double)*DMnx*DMny);
        memset(x0, 0, sizeof(double)*DMnd);
        // dger finds A := alpha*x*y'+ A, which is equivalent to kron(x,y) if A is row-major
        cblas_dger(CblasRowMajor, DMny, DMnx, 1.0, vy_1d, 1, vx_1d, 1, kron_vy_vx, DMnx);
        cblas_dger(CblasRowMajor, DMnz, DMnx*DMny, 1.0, vz_1d, 1, kron_vy_vx, 1, x0, DMnx*DMny);
        
        free(vx_1d);
        free(vy_1d);
        free(vz_1d);
        free(kron_vy_vx);
    }
}


/*
 * @ brief: Main function of Chebyshev filtering 
 */
void eigSolve_CheFSI(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error) {
    // Set up for CheFSI function
    if(pSPARC->spincomm_index < 0) return; 
    
    int count, spn_i;    
    double t1, t2, lambda_cutoff = 0.0;
    double *x0 = pSPARC->Lanczos_x0;
    if (pSPARC->elecgs_Count > 0 || pSPARC->usefock > 1) pSPARC->rhoTrigger = pSPARC->Nchefsi;

    if (SCFcount == 0) {
        pSPARC->npl_max = pSPARC->ChebDegree; 
        pSPARC->npl_min = max(pSPARC->ChebDegree / 4, 12);
        t1 = MPI_Wtime();
        double cellsizes[3] = {pSPARC->range_x, pSPARC->range_y, pSPARC->range_z};
        double meshes[3] = {pSPARC->delta_x, pSPARC->delta_y, pSPARC->delta_z};
        int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
        int RandFlag = (pSPARC->cell_typ != 0 || pSPARC->chefsibound_flag == 0 || 
            pSPARC->chefsibound_flag == 1);
        // set up initial guess for Lanczos
        init_guess_Lanczos(x0, cellsizes, gridsizes, meshes, pSPARC->DMVertices_kptcomm, 
            RandFlag, pSPARC->kptcomm_topo);
        t2 = MPI_Wtime();
#ifdef DEBUG    
        if (!rank) printf("\nTime for setting up initial guess for Lanczos: %.3f ms\n", (t2-t1)*1e3);
#endif
        count = 0;
    } else {
        count = pSPARC->rhoTrigger + (SCFcount-1) * pSPARC->Nchefsi;
    }   

    while(count < pSPARC->rhoTrigger + SCFcount*pSPARC->Nchefsi){
        // perform CheFSI algorithm, including
        // 1) Find Chebyshev filtering bounds
        // 2) Chebyshev filtering,          3) Projection, 
        // 4) Solve projected eigenproblem, 5) Subspace rotation
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++)
            CheFSI(pSPARC, lambda_cutoff, x0, count, 0, spn_i);
        
        t1 = MPI_Wtime();
        
        int indx0, ns;
        if (pSPARC->CyclixFlag) {
            // Find sorted eigenvalues
            for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                indx0 = spn_i*pSPARC->Nstates;
                memcpy(pSPARC->lambda_sorted + indx0, pSPARC->lambda + indx0, sizeof(double)*pSPARC->Nstates);
                qsort(pSPARC->lambda_sorted + indx0, pSPARC->Nstates, sizeof(pSPARC->lambda_sorted[0]), cmp);
            }
        }

        // ** calculate fermi energy ** //
        // find global minimum and global maximum eigenvalue
        double eigmin_g = pSPARC->lambda_sorted[0];
        double eigmax_g = pSPARC->lambda_sorted[pSPARC->Nstates-1];
        for(spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            if(pSPARC->lambda_sorted[spn_i*pSPARC->Nstates] < eigmin_g)
                eigmin_g = pSPARC->lambda_sorted[spn_i*pSPARC->Nstates];
            if(pSPARC->lambda_sorted[(spn_i+1)*pSPARC->Nstates - 1] > eigmax_g)
                eigmax_g = pSPARC->lambda_sorted[(spn_i+1)*pSPARC->Nstates - 1];
        }
        
        if (pSPARC->npspin != 1) { // find min/max over processes with the same rank in spincomm
            MPI_Allreduce(MPI_IN_PLACE, &eigmin_g, 1, MPI_DOUBLE, MPI_MIN, pSPARC->spin_bridge_comm);
            MPI_Allreduce(MPI_IN_PLACE, &eigmax_g, 1, MPI_DOUBLE, MPI_MAX, pSPARC->spin_bridge_comm);
        }
        
        pSPARC->Efermi = Calculate_occupation(pSPARC, eigmin_g-1.0, eigmax_g+1.0, 1e-12, 100); 

        if (pSPARC->CyclixFlag) {
            // Find occupations corresponding to sorted eigenvalues
            for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
                for (ns = 0; ns < pSPARC->Nstates; ns++) {
                    pSPARC->occ_sorted[ns+spn_i*pSPARC->Nstates] = smearing_function(pSPARC->Beta, pSPARC->lambda_sorted[ns+spn_i*pSPARC->Nstates], pSPARC->Efermi, pSPARC->elec_T_type);
                }
            }
        }

        t2 = MPI_Wtime();
#ifdef DEBUG
        if (!rank) {
            printf("rank = %d, Efermi = %16.12f"
                    " calculate fermi energy took %.3f ms\n", 
                    rank, pSPARC->Efermi, (t2-t1)*1e3);
        }
#endif
        count++;
    }

    // adjust chebyshev polynomial degree if CheFSI optimization is turned on
    if (pSPARC->CheFSI_Optmz) {
        //if (max(dEtot,dEband) > 1e-2) {
        if (error > 1e-2) {
            pSPARC->ChebDegree = pSPARC->npl_max;            
#ifdef DEBUG
            if(!rank) 
                printf("************************************************************************\n"
                       "Original Chebyshev polyn degree: %d, for next SCF chebdegree: %d\n"
                       "************************************************************************\n", 
                       pSPARC->npl_max, pSPARC->ChebDegree);
#endif
        } else {
            double log_TOL = log(pSPARC->TOL_SCF), log_err0 = 0.0;
            pSPARC->ChebDegree = pSPARC->npl_min + (int)((pSPARC->npl_max - pSPARC->npl_min)/(log_err0 - log_TOL) * (log(error) - log_TOL));
            pSPARC->ChebDegree = min(pSPARC->npl_max, pSPARC->ChebDegree);
            pSPARC->ChebDegree = max(pSPARC->npl_min, pSPARC->ChebDegree);
#ifdef DEBUG
            if(!rank) printf("************************************************************************\n"
                             "Original Chebyshev polyn degree: %d, for next SCF chebdegree: %d\n"
                             "************************************************************************\n",
                             pSPARC->npl_max, pSPARC->ChebDegree);
#endif
        }
    }
}


/**
 * @brief   Apply Chebyshev-filtered subspace iteration steps.
 */
void CheFSI(SPARC_OBJ *pSPARC, double lambda_cutoff, double *x0, int count, int k, int spn_i)
{
    int rank, rank_spincomm, nproc_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    
    // determine the constants for performing chebyshev filtering
    Chebyshevfilter_constants(pSPARC, x0, &lambda_cutoff, &pSPARC->eigmin[spn_i], &pSPARC->eigmax[spn_i], count, k, spn_i);
    
#ifdef DEBUG
            if (!rank && spn_i == 0) {
                printf("\n Chebfilt %d, in Chebyshev filtering, lambda_cutoff = %f,"
                       " lowerbound = %f, upperbound = %f\n\n",
                       count+1, lambda_cutoff, pSPARC->eigmin[spn_i], pSPARC->eigmax[spn_i]);
            }
#endif

    #ifdef USE_EVA_MODULE
    if (CheFSI_use_EVA == -1)
    {
        char *use_EVA_p = getenv("EXT_VEC_ACCEL");
        if (use_EVA_p != NULL) CheFSI_use_EVA = atoi(use_EVA_p);
        else CheFSI_use_EVA = 0;
        if (CheFSI_use_EVA != 1) CheFSI_use_EVA = 0;
        if (pSPARC->cell_typ != 0) CheFSI_use_EVA = 0;
    }
    #endif
    
    double t1, t2, t3, t_temp;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;    
    
    // ** Chebyshev filtering ** //
    t1 = MPI_Wtime();
    #ifdef USE_EVA_MODULE
    if (CheFSI_use_EVA == 1)
    {
        EVA_Chebyshev_Filtering(
            pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Nband_bandcomm, 
            pSPARC->ChebDegree, lambda_cutoff, pSPARC->eigmax[spn_i], pSPARC->eigmin[spn_i],
            pSPARC->dmcomm, pSPARC->Xorb + spn_i*size_s, pSPARC->Yorb
        );
    } else {
    #endif
        #ifdef SPARCX_ACCEL
		if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && pSPARC->spin_typ <= 1 && pSPARC->usefock <=1 && pSPARC->Nd_d_dmcomm == pSPARC->Nd)
		{
    		ACCEL_ChebyshevFiltering(pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Xorb + spn_i*DMnd, DMndsp,
                           pSPARC->Yorb + spn_i*DMnd, DMndsp, pSPARC->Nband_bandcomm, 
                           pSPARC->ChebDegree, lambda_cutoff, pSPARC->eigmax[spn_i], pSPARC->eigmin[spn_i], k, spn_i, 
                           pSPARC->dmcomm);
		}
		else
		#endif // SPARCX_ACCEL
        {
            ChebyshevFiltering(pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Xorb + spn_i*DMnd, DMndsp,
                           pSPARC->Yorb + spn_i*DMnd, DMndsp, pSPARC->Nband_bandcomm, 
                           pSPARC->ChebDegree, lambda_cutoff, pSPARC->eigmax[spn_i], pSPARC->eigmin[spn_i], k, spn_i, 
                           pSPARC->dmcomm, &t_temp);
        }
    #ifdef USE_EVA_MODULE
    }
    #endif
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) 
        printf("Total time for Chebyshev filtering (%d columns, degree = %d): %.3f ms\n", 
                pSPARC->Nband_bandcomm, pSPARC->ChebDegree, (t2-t1)*1e3);
    #endif
    
    t1 = MPI_Wtime();
    // ** calculate projected Hamiltonian and overlap matrix ** //
    #ifdef USE_DP_SUBEIG
    if (pSPARC->StandardEigenFlag == 1) {
        DP_Project_Hamiltonian_std(
            pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Yorb + spn_i*DMnd, DMndsp, pSPARC->Xorb + spn_i*DMnd, DMndsp, 
            spn_i
        );
    } else {
        DP_Project_Hamiltonian(
            pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Yorb + spn_i*DMnd, DMndsp, pSPARC->Xorb + spn_i*DMnd, DMndsp, 
            pSPARC->Hp, pSPARC->Mp, spn_i
        );
    }
    #else
    // allocate memory for block cyclic format of the wavefunction
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        pSPARC->Xorb_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));
        pSPARC->Yorb_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));
        assert(pSPARC->Xorb_BLCYC != NULL && pSPARC->Yorb_BLCYC != NULL);
    } else {
        pSPARC->Xorb_BLCYC = pSPARC->Xorb + spn_i*DMnd;
        pSPARC->Yorb_BLCYC = pSPARC->Yorb + spn_i*DMnd;
    }
    Project_Hamiltonian(pSPARC, pSPARC->DMVertices_dmcomm, pSPARC->Yorb + spn_i*DMnd, DMndsp, pSPARC->Xorb + spn_i*DMnd, DMndsp,
                        pSPARC->Hp, pSPARC->Mp, k, spn_i, pSPARC->dmcomm);
    #endif
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("Total time for projection: %.3f ms\n", (t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    // ** solve the subspace eigenvalue problem Hp * Q = Mp * Q * Lambda **//
    // ** or Hp * Q = Q * Lambda, if StandardEigenFlag = 1 ** //
    #ifdef USE_DP_SUBEIG
    DP_Solve_Generalized_EigenProblem(pSPARC, spn_i);
    #else
    Solve_Generalized_EigenProblem(pSPARC, k, spn_i);
    #endif    
    t3 = MPI_Wtime();

    // if eigvals are calculated in root process, then bcast the eigvals
    // SPARCX_ACCEL_NOTE Need to add this to propagate GPU calculated eigenvalues back to the other MPI tasks
    #ifdef SPARCX_ACCEL
    if (pSPARC->useACCEL == 1 && nproc_kptcomm > 1 && (!pSPARC->useHIP || pSPARC->useLAPACK == 1)) 
        MPI_Bcast(pSPARC->lambda, pSPARC->Nstates * pSPARC->Nspin_spincomm, MPI_DOUBLE, 0, pSPARC->kptcomm); 
    #else
    if (pSPARC->useLAPACK == 1 && nproc_kptcomm > 1) {
        MPI_Bcast(pSPARC->lambda, pSPARC->Nstates * pSPARC->Nspin_spincomm, 
                  MPI_DOUBLE, 0, pSPARC->kptcomm); // TODO: bcast in blacscomm if possible
    }
    #endif //SPARCX_ACCEL
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) {
        // print eigenvalues
        printf("    first calculated eigval = %.15f\n"
               "    last  calculated eigval = %.15f\n",
               pSPARC->lambda[0],
               pSPARC->lambda[pSPARC->Nstates-1]);
        int neig_print = min(20,pSPARC->Nstates - pSPARC->Nelectron/2 + 10);
        neig_print = min(neig_print, pSPARC->Nstates);
        printf("The last %d eigenvalues of kpoints #%d and spin #%d are (Nelectron = %d):\n", neig_print, 1, spn_i, pSPARC->Nelectron);
        int i;
        for (i = 0; i < neig_print; i++) {
            printf("lambda[%4d] = %18.14f\n", 
                    pSPARC->Nstates - neig_print + i + 1, 
                    pSPARC->lambda[pSPARC->Nstates - neig_print + i]);
        }
        printf("==subpsace eigenproblem: bcast eigvals took %.3f ms\n", (t2-t3)*1e3);
        printf("Total time for solving subspace eigenvalue problem: %.3f ms\n", 
                (t2-t1)*1e3);
    }
    #endif

    t1 = MPI_Wtime();
    // ** subspace rotation ** //
    #ifdef USE_DP_SUBEIG
    DP_Subspace_Rotation(pSPARC, pSPARC->Xorb + spn_i*DMnd);
    #else
    // find Y * Q, store the result in Xorb (band+domain) and Xorb_BLCYC (block cyclic format)
    Subspace_Rotation(pSPARC, pSPARC->Yorb_BLCYC, pSPARC->Q, 
                        pSPARC->Xorb_BLCYC, pSPARC->Xorb + spn_i*DMnd, k, spn_i);
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        free(pSPARC->Xorb_BLCYC); pSPARC->Xorb_BLCYC = NULL;
        free(pSPARC->Yorb_BLCYC); pSPARC->Yorb_BLCYC = NULL;
    }
    #endif
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank) printf("Total time for subspace rotation: %.3f ms\n", (t2-t1)*1e3);
    #endif

    if (pSPARC->CyclixFlag) {
        // Rescale wavefunction to make L2-norm for cyclix systems
        t1 = MPI_Wtime();
        NormalizeEigfunc_cyclix(pSPARC, spn_i);
        t2 = MPI_Wtime();
#ifdef DEBUG
        if(!rank) printf("Total time for normalizing psi: %.3f ms\n", (t2-t1)*1e3);
#endif
    }
}



/**
 * @brief   Solve standard eigenproblem Hp * x = lambda * x.
 *
 *          Note: Hp = Psi' * H * Psi, where Psi' * Psi = I. Also note that Hp is
 *                distributed block cyclically.
 *          
 *          TODO: At some point it is better to use ELPA (https://elpa.mpcdf.mpg.de/) 
 *                for solving subspace eigenvalue problem, which can provide up to 
 *                3x speedup.
 */
void Solve_standard_EigenProblem(SPARC_OBJ *pSPARC, int k, int spn_i) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm,  &rank_kptcomm);
    #ifdef DEBUG
    if (!rank && spn_i == 0) printf("Start solving standard eigenvalue problem ...\n");
    #endif
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int nproc_dmcomm;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    double t1, t2;
    #ifdef DEBUG    
    double st = MPI_Wtime();
    #endif
    
    #ifdef SPARCX_ACCEL // SPARCX_ACCEL_NOTE
    if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && !pSPARC->useHIP) {
		int info = 0;
		t1 = MPI_Wtime();
		if (!pSPARC->bandcomm_index) {
			info = DSYEV(LAPACK_COL_MAJOR, 'V', 'U', pSPARC->Nstates,pSPARC->Hp, 
						  pSPARC->Nstates, pSPARC->lambda + spn_i*pSPARC->Nstates);
		}
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if(!rank_spincomm && spn_i == 0) {
			printf("==standard eigenproblem: "
				   "info = %d, solving standard eigenproblem using LAPACKE_dsyevd: %.3f ms\n", 
				   info, (t2 - t1)*1e3);
		}
		#endif

		int ONE = 1;
		t1 = MPI_Wtime();
		// distribute eigenvectors to block cyclic format
		pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp, &ONE, &ONE, 
				  pSPARC->desc_Hp_BLCYC, pSPARC->Q, &ONE, &ONE, 
				  pSPARC->desc_Q_BLCYC, &pSPARC->ictxt_blacs_topo);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if(!rank_spincomm && spn_i == 0) {
			printf("==standard eigenproblem: "
				   "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
				   (t2 - t1)*1e3);
		}
		#endif
	}
	else
	#endif //SPARCX_ACCEL
    { // SPARCX_ACCEL_NOTE Enclose the if block below and locate at bottom
        if (pSPARC->useLAPACK == 1) {
            int info = 0;
            t1 = MPI_Wtime();
            if (!pSPARC->bandcomm_index) {
                info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', pSPARC->Nstates,pSPARC->Hp, 
                            pSPARC->Nstates, pSPARC->lambda + spn_i*pSPARC->Nstates);
            }
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank_spincomm && spn_i == 0) {
                printf("==standard eigenproblem: "
                    "info = %d, solving standard eigenproblem using LAPACKE_dsyevd: %.3f ms\n", 
                    info, (t2 - t1)*1e3);
            }
            #endif

            int ONE = 1;
            t1 = MPI_Wtime();
            // distribute eigenvectors to block cyclic format
            pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp, &ONE, &ONE, 
                    pSPARC->desc_Hp_BLCYC, pSPARC->Q, &ONE, &ONE, 
                    pSPARC->desc_Q_BLCYC, &pSPARC->ictxt_blacs_topo);
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank_spincomm && spn_i == 0) {
                printf("==standard eigenproblem: "
                    "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
                    (t2 - t1)*1e3);
            }
            #endif
        } else {
            int nprow, npcol, myrow, mycol;
            Cblacs_gridinfo(pSPARC->ictxt_blacs_topo, &nprow, &npcol, &myrow, &mycol);

            int N = pSPARC->Nstates, ONE = 1, info;
            // TODO: This function is not actually called. Apply subgrid eigen sovler to it if needed.
            t1 = MPI_Wtime();
            automem_pdsyev_ ( "V", "U", &N, pSPARC->Hp, &ONE, &ONE, pSPARC->desc_Hp_BLCYC, 
                    pSPARC->lambda + spn_i*N, pSPARC->Q, &ONE, &ONE, pSPARC->desc_Q_BLCYC, 
                    &info);
            
            if (info != 0 && !rank) {
                printf("\nError in solving standard eigenproblem! info = %d\n", info);
            }
            
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank && spn_i == 0) {
                printf("rank = %d, info = %d, time for solving standard "
                    "eigenproblem in %d x %d process grid: %.3f ms\n", 
                        rank, info, nprow, npcol, (t2 - t1)*1e3);
                printf("rank = %d, after calling pdsygv, Nstates = %d\n", rank, N);
            }
            #endif
        }
    }

    #ifdef DEBUG    
    double et = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Solve_standard_EigenProblem used %.3lf ms\n", rank, 1000.0 * (et - st));
    #endif
        
#else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("[FATAL] Subspace eigenproblem are using ScaLAPACK routines but ScaLAPACK is not compiled\n");
    if (rank == 0) printf("\nPlease turn on USE_MKL or USE_SCALAPACK!\n");
    exit(255);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
} 


/**
 * @brief   Find Chebyshev filtering bounds and cutoff constants.
 */
void Chebyshevfilter_constants(
    SPARC_OBJ *pSPARC, double *x0, double *lambda_cutoff, double *eigmin, 
    double *eigmax, int count, int k, int spn_i
)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double t1, t2;
    double temp;
    double eigmin_lap;
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx;         
    sdims[1] = pSPARC->npNdy;         
    sdims[2] = pSPARC->npNdz; 
    rdims[0] = pSPARC->npNdx_kptcomm; 
    rdims[1] = pSPARC->npNdy_kptcomm; 
    rdims[2] = pSPARC->npNdz_kptcomm;
    
    int sg  = pSPARC->spin_start_indx + spn_i;
    // ** find smallest and largest eigenvalue of the Hamiltonian ** //
    if (count == 0) 
    {
        t1 = MPI_Wtime();
        
        if (pSPARC->chefsibound_flag == 0 || pSPARC->chefsibound_flag == 1) { // 0 - default, 1 - always call Lanczos on H
            // estimate the min and max eigenval of H using Lanczos
            if (pSPARC->spin_typ == 0 && pSPARC->is_phi_eq_kpt_topo) {
                Lanczos(pSPARC, pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_dmcomm_phi, 
                        pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->nlocProj_kptcomm, 
                        eigmin, eigmax, x0, pSPARC->TOL_LANCZOS, pSPARC->TOL_LANCZOS, 1000, k, spn_i,
                        pSPARC->kptcomm_topo, &pSPARC->req_veff_loc);
                *eigmax *= 1.01; // add 1% buffer
            } else {
                D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, 
                    pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                    pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, 
                    pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
                    sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm, sizeof(double));
                // If exchange-correlation is SCAN, GGA_PBE will be the exc used in 1st SCF; it is unnecessary to transform a zero vector
                Lanczos(pSPARC, pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, 
                        pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->nlocProj_kptcomm, 
                        eigmin, eigmax, x0, pSPARC->TOL_LANCZOS, pSPARC->TOL_LANCZOS, 
                        1000, k, spn_i, pSPARC->kptcomm_topo, &pSPARC->req_veff_loc);
                *eigmax *= 1.01; // add 1% buffer
            }
        } else {
            if (pSPARC->cell_typ == 0 && pSPARC->BC == 2) {
                // estimate the max eigenval of H by max eigval of -0.5*Lap calculated using 
                // explicit formula
                // Warning: not safe for extreme cases, find max eigval of H 
                //          instead if max eigval of -0.5*Lap < max eigval of H
                *eigmax = pSPARC->MaxEigVal_mhalfLap; 
                // if anything goes wrong, use Lanczos.
                *eigmin = -2.0; // TODO: tune this value
            } else {
                // find min eigval of Lap, estimate eigmax of H by -0.5 * eigmin_lap   
                Lanczos_laplacian(pSPARC, pSPARC->DMVertices_kptcomm, &eigmin_lap, &temp,
                                  x0, pSPARC->TOL_LANCZOS, 1e10, 1000, k, spn_i, pSPARC->kptcomm_topo);
                *eigmax = -0.5 * eigmin_lap;
                // *eigmax += 10 * pSPARC->TOL_LANCZOS; 
                *eigmax *= 1.01; // add 1% buffer
                *eigmin = -2.0; // TODO: tune this value
            }
        }

        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (rank == 0) {
            printf("rank = %3d, Lanczos took %.3f ms, eigmin = %.12f, eigmax = %.12f\n", 
                   rank, (t2-t1)*1e3, *eigmin, *eigmax);
        }
        #endif

        *eigmin -= 0.1; // for safety

    } else if (count >= pSPARC->rhoTrigger) {
        *eigmin = pSPARC->lambda_sorted[spn_i*pSPARC->Nstates]; // take previous eigmin
        
        if (pSPARC->chefsibound_flag == 1 || ((count == pSPARC->rhoTrigger) && (strcmpi(pSPARC->XC, "SCAN") == 0))) { // 1 - always call Lanczos on H; the other condition is for SCAN
        //the first SCF is PBE, the second is SCAN, so it is necessary to do Lanczos again in 2nd SCF
            t1 = MPI_Wtime();
            // estimate both max eigenval of H using Lanczos
            if (pSPARC->spin_typ == 0 && pSPARC->is_phi_eq_kpt_topo) {
                Lanczos(pSPARC, pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_dmcomm_phi, 
                        pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->nlocProj_kptcomm, 
                        &temp, eigmax, x0, 1e10, pSPARC->TOL_LANCZOS, 1000, k, spn_i,
                        pSPARC->kptcomm_topo, &pSPARC->req_veff_loc);
                // *eigmax += 10 * pSPARC->TOL_LANCZOS;
                *eigmax *= 1.01; // add 1% buffer
            } else {
                D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
                    pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
                    sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm, sizeof(double));
                if (strcmpi(pSPARC->XC, "SCAN") == 0) { // transfer vxcMGGA3 of this spin to kptcomm, it is moved from file mgga/mgga.c to here.
                    // printf("rank %d, joined SCAN Lanczos, pSPARC->countPotentialCalculate %d\n", rank, pSPARC->countPotentialCalculate);
                    D2D(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, 
                    pSPARC->DMVertices_dmcomm, pSPARC->vxcMGGA3_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, // processors in dmcomm does not save vxcMGGA3 of both spins; they just saved which their spincomm needs 
                    pSPARC->DMVertices_kptcomm, pSPARC->vxcMGGA3_loc_kptcomm, 
                    pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL,
                    sdims, pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm, sizeof(double));
                }
                Lanczos(pSPARC, pSPARC->DMVertices_kptcomm, pSPARC->Veff_loc_kptcomm_topo, 
                        pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->nlocProj_kptcomm, 
                        &temp, eigmax, x0, 1e10, pSPARC->TOL_LANCZOS, 
                        1000, k, spn_i, pSPARC->kptcomm_topo, &pSPARC->req_veff_loc);
                // *eigmax += 10 * pSPARC->TOL_LANCZOS;
                *eigmax *= 1.01; // add 1% buffer
            }
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if (rank == 0) {
                printf("rank = %3d, Lanczos took %.3f ms, eigmin = %.12f, eigmax = %.12f\n", 
                   rank, (t2-t1)*1e3, *eigmin, *eigmax);
            }
            #endif
        } 
    }

    if (pSPARC->elecgs_Count == 0 && count == 0) {
        *lambda_cutoff = 0.5 * (*eigmin + *eigmax);
    } else{
        //*lambda_cutoff = pSPARC->Efermi + log(1e6-1) / pSPARC->Beta + 0.1;
        *lambda_cutoff = pSPARC->lambda_sorted[(spn_i+1)*pSPARC->Nstates-1] + 0.1;
    }
}



/**
 * @brief   Perform Chebyshev filtering.
 */
void ChebyshevFiltering(
    SPARC_OBJ *pSPARC, int *DMVertices, double *X, int ldi, double *Y, int ldo, int ncol, 
    int m, double a, double b, double a0, int k, int spn_i, MPI_Comm comm, 
    double *time_info
) 
{   
    if (comm == MPI_COMM_NULL || pSPARC->bandcomm_index < 0) return;
    // a0: minimum eigval, b: maxinum eigval, a: cutoff eigval
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    #ifdef DEBUG   
    if(!rank && spn_i == 0) printf("Start Chebyshev filtering ... \n");
    #endif

    double t1, t2;
    *time_info = 0.0;

    double e, c, sigma, sigma1, sigma2, gamma, vscal, vscal2, *Ynew;
    int i, j, DMnd, len_tot, DMndspe;
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);
    DMndspe = DMnd * pSPARC->Nspinor_eig;
    
    len_tot = DMndspe * ncol;    
    e = 0.5 * (b - a);
    c = 0.5 * (b + a);
    sigma = sigma1 = e / (a0 - c);
    gamma = 2.0 / sigma1;

    t1 = MPI_Wtime();
    // find Y = (H - c*I)X
    int sg  = pSPARC->spin_start_indx + spn_i;
    Hamiltonian_vectors_mult(
        pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, -c, X, ldi, Y, ldo, spn_i, comm
    );
    t2 = MPI_Wtime();
    *time_info += t2 - t1;
        
    // scale Y by (sigma1 / e)
    vscal = sigma1 / e;
    for (int n = 0; n < ncol; n++)  {
        for (i = 0; i < DMndspe; i++) {
            Y[i+n*ldo] *= vscal;
        }
    }
   
    Ynew = (double *)malloc( len_tot * sizeof(double));

    for (j = 1; j < m; j++) {
        sigma2 = 1.0 / (gamma - sigma);
        
        t1 = MPI_Wtime();
        // Ynew = (H - c*I)Y
        Hamiltonian_vectors_mult(
            pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm, 
            pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, -c, Y, ldo, Ynew, DMndspe, spn_i, comm
        );
        t2 = MPI_Wtime();
        *time_info += t2 - t1;

        // Ynew = (2*sigma2/e) * Ynew - (sigma*sigma2) * X, then update X and Y
        vscal = 2.0 * sigma2 / e; vscal2 = sigma * sigma2;

        for (int n = 0; n < ncol; n++)  {
            for (i = 0; i < DMndspe; i++) {
                Ynew[i+n*DMndspe] *= vscal;
                Ynew[i+n*DMndspe] -= vscal2 * X[i+n*ldi];
                X[i+n*ldi] = Y[i+n*ldo];
                Y[i+n*ldo] = Ynew[i+n*DMndspe];
            }
        }
        sigma = sigma2;
    } 
    free(Ynew);
}

#ifdef USE_DP_SUBEIG
static int calc_block_spos(const int len, const int nblk, const int iblk)
{
	if (iblk < 0 || iblk > nblk) return -1;
	int rem = len % nblk;
	int bs0 = len / nblk;
	int bs1 = bs0 + 1;
	if (iblk < rem) return (bs1 * iblk);
    else return (bs0 * iblk + rem);
}

/**
 * @brief   Initialize domain parallelization data structures for calculating projected Hamiltonian,  
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI().
 */
void init_DP_CheFSI(SPARC_OBJ *pSPARC)
{
    int proc_active = (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) ? 0 : 1;
    
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) malloc(sizeof(struct DP_CheFSI_s));
    
    // Split the kpt_comm for all active processes in pSPARC->kptcomm
    int nproc_kpt, rank_kpt;
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kpt);
    MPI_Comm_split(pSPARC->kptcomm, proc_active, rank_kpt, &DP_CheFSI->kpt_comm);
    if (proc_active == 0)
    {
        MPI_Comm_free(&DP_CheFSI->kpt_comm);
        free(DP_CheFSI);
        pSPARC->DP_CheFSI = NULL;
        return;
    } else {
        MPI_Comm_size(DP_CheFSI->kpt_comm, &nproc_kpt);
        MPI_Comm_rank(DP_CheFSI->kpt_comm, &rank_kpt);
    }

    // Get the original band parallelization parameters
    int nproc_row, rank_row;
    MPI_Comm_size(pSPARC->blacscomm, &nproc_row);
    MPI_Comm_rank(pSPARC->blacscomm, &rank_row);
    int Ns_bp = pSPARC->band_end_indx - pSPARC->band_start_indx + 1;
    int Ns_dp = pSPARC->Nstates;
    int Nd_bp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_eig;
    int Ndsp_bp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    
    // The number of bands on each process could be different, we need to gather them
    int *Ns_bp_displs = (int*) malloc(sizeof(int) * (nproc_row + 1));
    assert(Ns_bp_displs != NULL);
    MPI_Allgather(
        &pSPARC->band_start_indx, 1, MPI_INT, Ns_bp_displs, 1, 
        MPI_INT, pSPARC->blacscomm
    );
    Ns_bp_displs[nproc_row] = pSPARC->Nstates;
    
    // Nd_bp is the same for all processes in blacscomm, partition it in the same way
    int *Nd_dp_displs = (int*) malloc(sizeof(int) * (nproc_row + 1));
    assert(Nd_dp_displs != NULL);
    for (int i = 0; i <= nproc_row; i++)
        Nd_dp_displs[i] = calc_block_spos(Nd_bp, nproc_row, i);
    int Nd_dp = Nd_dp_displs[rank_row + 1] - Nd_dp_displs[rank_row];
    
    // Calculate MPI_Alltoallv send and recv parameters
    int *bp2dp_sendcnts = (int*) malloc(sizeof(int) * nproc_row);
    int *bp2dp_sdispls  = (int*) malloc(sizeof(int) * nproc_row);
    int *dp2bp_sendcnts = (int*) malloc(sizeof(int) * nproc_row);
    int *dp2bp_sdispls  = (int*) malloc(sizeof(int) * nproc_row);
    assert(bp2dp_sendcnts != NULL && bp2dp_sdispls != NULL);
    assert(dp2bp_sendcnts != NULL && dp2bp_sdispls != NULL);
    bp2dp_sendcnts[0] = Ns_bp * Nd_dp_displs[1];
    dp2bp_sendcnts[0] = Nd_dp * Ns_bp_displs[1];
    bp2dp_sdispls[0]  = 0;
    dp2bp_sdispls[0]  = 0;
    for (int i = 1; i < nproc_row; i++)
    {
        bp2dp_sendcnts[i] = Ns_bp * (Nd_dp_displs[i + 1] - Nd_dp_displs[i]);
        dp2bp_sendcnts[i] = Nd_dp * (Ns_bp_displs[i + 1] - Ns_bp_displs[i]);
        bp2dp_sdispls[i]  = bp2dp_sdispls[i - 1] + bp2dp_sendcnts[i - 1];
        dp2bp_sdispls[i]  = dp2bp_sdispls[i - 1] + dp2bp_sendcnts[i - 1];
    }

    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int ZERO = 0, info;
    // descriptors for Hp_local, Mp_local and eig_vecs (set block size to Ns_dp)
    descinit_(DP_CheFSI->desc_Hp_local, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
    	&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
    assert(info == 0);
    descinit_(DP_CheFSI->desc_Mp_local, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
    	&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
    assert(info == 0);
    descinit_(DP_CheFSI->desc_eig_vecs, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
    	&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
    assert(info == 0);
    #endif
    
    // Copy parameters and pointers to DP_CheFSI
    DP_CheFSI->nproc_row = nproc_row;
    DP_CheFSI->nproc_kpt = nproc_kpt;
    DP_CheFSI->rank_row  = rank_row;
    DP_CheFSI->rank_kpt  = rank_kpt;
    DP_CheFSI->Ns_bp     = Ns_bp;
    DP_CheFSI->Ns_dp     = Ns_dp;
    DP_CheFSI->Nd_bp     = Nd_bp;
    DP_CheFSI->Ndsp_bp   = Ndsp_bp;
    DP_CheFSI->Nd_dp     = Nd_dp;
    DP_CheFSI->Ns_bp_displs   = Ns_bp_displs;
    DP_CheFSI->Nd_dp_displs   = Nd_dp_displs;
    DP_CheFSI->bp2dp_sendcnts = bp2dp_sendcnts;
    DP_CheFSI->bp2dp_sdispls  = bp2dp_sdispls;
    DP_CheFSI->dp2bp_sendcnts = dp2bp_sendcnts;
    DP_CheFSI->dp2bp_sdispls  = dp2bp_sdispls;
    size_t Nd_Ns_bp_msize  = sizeof(double) * Nd_bp * Ns_bp;
    size_t Nd_Ns_dp_msize  = sizeof(double) * Nd_dp * Ns_dp;
    size_t Ns_dp_2_msize   = sizeof(double) * Ns_dp * Ns_dp;
    DP_CheFSI->Y_packbuf  = (double*) malloc(Nd_Ns_bp_msize);
    DP_CheFSI->HY_packbuf = (double*) malloc(Nd_Ns_bp_msize);
    DP_CheFSI->Y_dp       = (double*) malloc(Nd_Ns_dp_msize);
    DP_CheFSI->HY_dp      = (double*) malloc(Nd_Ns_dp_msize);
    DP_CheFSI->Mp_local   = (double*) malloc(Ns_dp_2_msize);
    DP_CheFSI->Hp_local   = (double*) malloc(Ns_dp_2_msize);
    DP_CheFSI->eig_vecs   = (double*) malloc(Ns_dp_2_msize);
    assert(DP_CheFSI->Y_dp != NULL && DP_CheFSI->HY_dp != NULL);
    assert(DP_CheFSI->Y_packbuf  != NULL);
    assert(DP_CheFSI->HY_packbuf != NULL);
    assert(DP_CheFSI->Mp_local   != NULL);
    assert(DP_CheFSI->Hp_local   != NULL);
    assert(DP_CheFSI->eig_vecs   != NULL);
    pSPARC->DP_CheFSI = (void*) DP_CheFSI;
}

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix with domain parallelization
 *          data partitioning. 
 *
 *          Hp = Y' * H * Y,  Mp = Y' * Y. The tall-skinny column-matrices Y and HY 
 *          are partitioned by rows only in this function. MPI_Alltoallv is used to 
 *          convert the data layout from band parallelization to domain parallelization
 *          in each original domain parallelization part (blacscomm). Then we need 2 
 *          MPI_Reduce to get the final Hp and Mp on rank 0 of each kpt_comm.
 */
void DP_Project_Hamiltonian(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, int ldi, double *HY, int ldo, double *Hp, double *Mp, int spn_i)
{
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    if (DP_CheFSI == NULL) return;
    
    double st, et, st0, et0;
    int rank_kpt = DP_CheFSI->rank_kpt;
    
    st0 = MPI_Wtime();
    
    // Calculate H * Y, copied from Project_Hamiltonian
    int sg = pSPARC->spin_start_indx + spn_i;
    double *Veff_loc_sg = pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm;
    st = MPI_Wtime();
    #ifdef SPARCX_ACCEL
	if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && pSPARC->spin_typ <= 1 && pSPARC->usefock <=1 && pSPARC->Nd_d_dmcomm == pSPARC->Nd)
	{
	}
	else
	#endif // SPARCX_ACCEL
    {
        Hamiltonian_vectors_mult(
            pSPARC, pSPARC->Nd_d_dmcomm, DMVertices, 
            Veff_loc_sg, pSPARC->Atom_Influence_nloc, 
            pSPARC->nlocProj, pSPARC->Nband_bandcomm, 
            0.0, Y, ldi, HY, ldo, spn_i, pSPARC->dmcomm
        );
    }
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian, rank 0, calc HY used %.3lf ms\n", 1000.0 * (et - st));
    #endif
    
    // Use MPI_Alltoallv to convert from band parallelization to domain parallelization
    // Note: Y and HY are stored in column major!
    st = MPI_Wtime();
    double *Y_dp  = DP_CheFSI->Y_dp;
    double *HY_dp = DP_CheFSI->HY_dp;
    BP2DP(
        pSPARC->blacscomm, DP_CheFSI->nproc_row,
        DP_CheFSI->Ndsp_bp, DP_CheFSI->Ns_bp, DP_CheFSI->Nd_dp_displs,
        DP_CheFSI->bp2dp_sendcnts, DP_CheFSI->bp2dp_sdispls,
        DP_CheFSI->dp2bp_sendcnts, DP_CheFSI->dp2bp_sdispls,
        sizeof(double),  Y,  DP_CheFSI->Y_packbuf,  Y_dp
    );
    BP2DP(
        pSPARC->blacscomm, DP_CheFSI->nproc_row,
        DP_CheFSI->Ndsp_bp, DP_CheFSI->Ns_bp, DP_CheFSI->Nd_dp_displs,
        DP_CheFSI->bp2dp_sendcnts, DP_CheFSI->bp2dp_sdispls,
        DP_CheFSI->dp2bp_sendcnts, DP_CheFSI->dp2bp_sdispls,
        sizeof(double), HY, DP_CheFSI->HY_packbuf, HY_dp
    );
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian, rank 0, convert BP to DP used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    // Local dgemm, Y and HY are Nd_dp-by-Ns_dp column-major matrices, we need Y^T * (H)Y
    st = MPI_Wtime();
    int Nd_dp = DP_CheFSI->Nd_dp;
    int Ns_dp = DP_CheFSI->Ns_dp;
    double *Mp_local = DP_CheFSI->Mp_local;
    double *Hp_local = DP_CheFSI->Hp_local;
    // SPARCX_ACCEL_NOTE: ADD HERE THE MATRIX PROJECTION GPU STUFF
    #ifdef SPARCX_ACCEL
	if (pSPARC->useACCEL == 1)
	{
    	ACCEL_DGEMM(
        CblasColMajor, CblasTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        1.0, Y_dp, Nd_dp, Y_dp, Nd_dp, 
        0.0, Mp_local, Ns_dp
	    );
    	ACCEL_DGEMM(
        CblasColMajor, CblasTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        1.0, Y_dp, Nd_dp, HY_dp, Nd_dp, 
        0.0, Hp_local, Ns_dp
	    );
	}
	else
	#endif // SPARCX_ACCEL
	{ // Need to enclose CPU-only original statements between brackets
		cblas_dgemm(
			CblasColMajor, CblasTrans, CblasNoTrans,
			Ns_dp, Ns_dp, Nd_dp,
			1.0, Y_dp, Nd_dp, Y_dp, Nd_dp, 
			0.0, Mp_local, Ns_dp
		);
		cblas_dgemm(
			CblasColMajor, CblasTrans, CblasNoTrans,
			Ns_dp, Ns_dp, Nd_dp,
			1.0, Y_dp, Nd_dp, HY_dp, Nd_dp, 
			0.0, Hp_local, Ns_dp
		);
	} // End
    et = MPI_Wtime();
    #ifdef DEBUG
    #ifdef SPARCX_ACCEL // SPARCX_ACCEL_NOTE ADD DEBUG LINE FOR SPARCX_ACCEL
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian, rank 0, local %s for Hp & Mp used %.3lf ms\n", STR_DGEMM, 1000.0 * (et - st));
    #else
	if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian, rank 0, local cblas_dgemm for Hp & Mp used %.3lf ms\n", 1000.0 * (et - st));
	#endif // SPARCX_ACCEL
    #endif // DEBUG

    // Reduce to Mp & Hp
    st = MPI_Wtime();
    int Ns_dp_2 = Ns_dp * Ns_dp;
    MPI_Request req0, req1;
    MPI_Status  sta0, sta1;
    if (rank_kpt == 0)
    {
        MPI_Ireduce(
            MPI_IN_PLACE, Mp_local, Ns_dp_2, MPI_DOUBLE, 
            MPI_SUM, 0, DP_CheFSI->kpt_comm, &req0
        );
        MPI_Ireduce(
            MPI_IN_PLACE, Hp_local, Ns_dp_2, MPI_DOUBLE, 
            MPI_SUM, 0, DP_CheFSI->kpt_comm, &req1
        );
    } else {
        MPI_Ireduce(
            Mp_local, NULL, Ns_dp_2, MPI_DOUBLE, 
            MPI_SUM, 0, DP_CheFSI->kpt_comm, &req0
        );
        MPI_Ireduce(
            Hp_local, NULL, Ns_dp_2, MPI_DOUBLE, 
            MPI_SUM, 0, DP_CheFSI->kpt_comm, &req1
        );
    }
    MPI_Wait(&req0, &sta0);
    MPI_Wait(&req1, &sta1);
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian, rank 0, reduce for Hp & Mp used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    et0 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian, rank 0, DP_Project_Hamiltonian used %.3lf ms\n", 1000.0 * (et0 - st0));
    #endif
}

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix with domain parallelization
 *          data partitioning for standard eigenvalue problem.
 */
void DP_Project_Hamiltonian_std(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, int ldi, double *HY, int ldo, int spn_i)
{
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    if (DP_CheFSI == NULL) return;
    
    double st, et, st0, et0;
    int rank_kpt = DP_CheFSI->rank_kpt;
    double *Y_dp  = DP_CheFSI->Y_dp;
    double *HY_dp = DP_CheFSI->HY_dp;
    int Nd_dp = DP_CheFSI->Nd_dp;
    int Ns_dp = DP_CheFSI->Ns_dp;
    double *Mp_local = DP_CheFSI->Mp_local;
    double *Hp_local = DP_CheFSI->Hp_local;
    int Ns_dp_2 = Ns_dp * Ns_dp;
    MPI_Request req0, req1;
    MPI_Status  sta0, sta1;
    int sg = pSPARC->spin_start_indx + spn_i;
    double *Veff_loc_sg = pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm;

    st0 = MPI_Wtime();
    /* Calculate Mp = Y' * Y */
    st = MPI_Wtime();
    BP2DP(
        pSPARC->blacscomm, DP_CheFSI->nproc_row,
        DP_CheFSI->Ndsp_bp, DP_CheFSI->Ns_bp, DP_CheFSI->Nd_dp_displs,
        DP_CheFSI->bp2dp_sendcnts, DP_CheFSI->bp2dp_sdispls,
        DP_CheFSI->dp2bp_sendcnts, DP_CheFSI->dp2bp_sdispls,
        sizeof(double),  Y,  DP_CheFSI->Y_packbuf,  Y_dp
    );
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, convert BP to DP (Y) used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    // Local dgemm, Y and HY are Nd_dp-by-Ns_dp column-major matrices, we need Y^T * (H)Y
    st = MPI_Wtime();
	#ifdef SPARCX_ACCEL
	ACCEL_DGEMM(
		CblasColMajor, CblasTrans, CblasNoTrans,
		Ns_dp, Ns_dp, Nd_dp,
		1.0, Y_dp, Nd_dp, Y_dp, Nd_dp,
		0.0, Mp_local, Ns_dp
	);
	#else
    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        1.0, Y_dp, Nd_dp, Y_dp, Nd_dp, 
        0.0, Mp_local, Ns_dp
    );
	#endif

    et = MPI_Wtime();
    #ifdef DEBUG
	#ifdef SPARCX_ACCEL
	if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, local %s for Mp used %.3lf ms\n", STR_DGEMM, 1000.0 * (et - st));
	#else
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, local dgemm for Mp used %.3lf ms\n", 1000.0 * (et - st));
    #endif //SPARCX_ACCEL
	#endif //DEBUG

    st = MPI_Wtime();
    MPI_Iallreduce(MPI_IN_PLACE, Mp_local, Ns_dp_2,
               MPI_DOUBLE, MPI_SUM, DP_CheFSI->kpt_comm, &req0);
    MPI_Wait(&req0, &sta0);
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_std, rank 0, reduce for Mp used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    /* Orthogonalization of orbitals using Choleskey */
    st = MPI_Wtime();
    LAPACKE_dpotrf (LAPACK_COL_MAJOR, 'U', Ns_dp, Mp_local, Ns_dp);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 
        Nd_dp, Ns_dp, 1.0, Mp_local, Ns_dp, Y_dp, Nd_dp);

    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_std, Orthogonalization using DP used %.3lf ms\n", 1000.0 * (et - st));
    #endif
    st = MPI_Wtime();
    DP2BP(
        pSPARC->blacscomm, DP_CheFSI->nproc_row,
        DP_CheFSI->Ndsp_bp, DP_CheFSI->Ns_bp, DP_CheFSI->Nd_dp_displs,
        DP_CheFSI->bp2dp_sendcnts, DP_CheFSI->bp2dp_sdispls,
        DP_CheFSI->dp2bp_sendcnts, DP_CheFSI->dp2bp_sdispls,
        sizeof(double), Y_dp, DP_CheFSI->Y_packbuf, HY
    );
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_std, rank 0, redist orthogonalized orbital used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    /* Calculate Hp = Y' * HY */
    // Calculate H * Y, copied from Project_Hamiltonian
    st = MPI_Wtime();
    Hamiltonian_vectors_mult(
        pSPARC, pSPARC->Nd_d_dmcomm, DMVertices, 
        Veff_loc_sg, pSPARC->Atom_Influence_nloc, 
        pSPARC->nlocProj, pSPARC->Nband_bandcomm, 
        0.0, HY, ldo, Y, ldi, spn_i, pSPARC->dmcomm
    );
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, calc HY used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    st = MPI_Wtime();
    BP2DP(
        pSPARC->blacscomm, DP_CheFSI->nproc_row,
        DP_CheFSI->Ndsp_bp, DP_CheFSI->Ns_bp, DP_CheFSI->Nd_dp_displs,
        DP_CheFSI->bp2dp_sendcnts, DP_CheFSI->bp2dp_sdispls,
        DP_CheFSI->dp2bp_sendcnts, DP_CheFSI->dp2bp_sdispls,
        sizeof(double), Y, DP_CheFSI->HY_packbuf, HY_dp
    );
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, convert BP to DP used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    st = MPI_Wtime();
	#ifdef SPARCX_ACCEL
	    ACCEL_DGEMM(
        CblasColMajor, CblasTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        1.0, Y_dp, Nd_dp, HY_dp, Nd_dp, 
        0.0, Hp_local, Ns_dp
    );
	#else 
    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans,
        Ns_dp, Ns_dp, Nd_dp,
        1.0, Y_dp, Nd_dp, HY_dp, Nd_dp, 
        0.0, Hp_local, Ns_dp
    );
	#endif //SPARCX_ACCEL

    et = MPI_Wtime();
    #ifdef DEBUG
	#ifdef SPARCX_ACCEL
	if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, local %s for Hp used %.3lf ms\n", STR_DGEMM, 1000.0 * (et - st));
    #else
    if (rank_kpt == 0 && spn_i == 0) printf("DP_Project_Hamiltonian_std, rank 0, local dgemm for Hp used %.3lf ms\n", 1000.0 * (et - st));
    #endif // SPARCX_ACCEL
	#endif // DEBUG

    st = MPI_Wtime();
    MPI_Iallreduce(MPI_IN_PLACE, Hp_local, Ns_dp_2,
       MPI_DOUBLE, MPI_SUM, DP_CheFSI->kpt_comm, &req1);
    MPI_Wait(&req1, &sta1);
    et = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_std, rank 0, allreduce for Hp used %.3lf ms\n", 1000.0 * (et - st));
    #endif

    et0 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0 && spn_i == 0) 
        printf("DP_Project_Hamiltonian_std, rank 0, DP_Project_Hamiltonian used %.3lf ms\n", 1000.0 * (et0 - st0));
    #endif
}


/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x using domain parallelization
 *          data partitioning. 
 *
 *          Rank 0 of each kpt_comm will solve this generalized eigenproblem locally and 
 *          then broadcast the obtained eigenvectors to all other processes in this kpt_comm.
 */
void DP_Solve_Generalized_EigenProblem(SPARC_OBJ *pSPARC, int spn_i)
{
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    if (DP_CheFSI == NULL) return;
    
    #ifdef SPARCX_ACCEL // SPARCX_ACCEL_NOTE -- ADDS GPU Eigensolver
	if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && !pSPARC->useHIP)
	{
		int Ns_dp = DP_CheFSI->Ns_dp;
		int rank_kpt = DP_CheFSI->rank_kpt;
		double *eig_vecs = DP_CheFSI->eig_vecs;
		double st = MPI_Wtime();
		if (rank_kpt == 0)
		{
			double *Hp_local = DP_CheFSI->Hp_local;
			double *Mp_local = DP_CheFSI->Mp_local; 
			double *eig_val  = pSPARC->lambda + spn_i * Ns_dp;
			int info = 0;
			if (pSPARC->StandardEigenFlag == 0)
				info = DSYGV( LAPACK_COL_MAJOR, 1, 'V', 'U', Ns_dp, 
							Hp_local, Ns_dp, Mp_local, Ns_dp, eig_val);
			else 
				info = DSYEV(LAPACK_COL_MAJOR,'V','U', Ns_dp, Hp_local, Ns_dp, eig_val);
			
			copy_mat_blk(sizeof(double), Hp_local, Ns_dp, Ns_dp, Ns_dp, eig_vecs, Ns_dp);
		}
		double et0 = MPI_Wtime();
		MPI_Bcast(eig_vecs, Ns_dp * Ns_dp, MPI_DOUBLE, 0, DP_CheFSI->kpt_comm);
		double et1 = MPI_Wtime();
		#ifdef DEBUG
		if (pSPARC->StandardEigenFlag == 0) {
			if (rank_kpt == 0) printf("DP_Solve_Generalized_EigenProblem rank 0 used %.3lf ms, %s used %.3lf ms\n", 1000.0 * (et1 - st), STR_DSYGV, 1000.0 * (et0 - st));
		} else {
			if (rank_kpt == 0) printf("DP_Solve_Generalized_EigenProblem rank 0 used %.3lf ms, %s used %.3lf ms\n", 1000.0 * (et1 - st), STR_DSYEV, 1000.0 * (et0 - st));
		}
		#endif
	}
	else
	#endif // SPARCX_ACCEL
    { // Enclose in brackets original CPU if(useLAPACK...) block and put it at the bottom

        if (pSPARC->useLAPACK == 1)
        {
            int Ns_dp = DP_CheFSI->Ns_dp;
            int rank_kpt = DP_CheFSI->rank_kpt;
            double *eig_vecs = DP_CheFSI->eig_vecs;
            double st = MPI_Wtime();
            int info = 0;
            if (rank_kpt == 0)
            {
                double *Hp_local = DP_CheFSI->Hp_local;
                double *Mp_local = DP_CheFSI->Mp_local; 
                double *eig_val  = pSPARC->lambda + spn_i * Ns_dp;
                if (pSPARC->CyclixFlag) {
                    info = generalized_eigenvalue_problem_cyclix(pSPARC, Hp_local, Mp_local, eig_val);
                } else if (pSPARC->StandardEigenFlag == 0) {
                    info = LAPACKE_dsygvd( LAPACK_COL_MAJOR, 1, 'V', 'U', Ns_dp, 
                                Hp_local, Ns_dp, Mp_local, Ns_dp, eig_val);
                } else {
                    info = LAPACKE_dsyevd(LAPACK_COL_MAJOR,'V','U', Ns_dp, Hp_local, Ns_dp, eig_val);
                }
                
                copy_mat_blk(sizeof(double), Hp_local, Ns_dp, Ns_dp, Ns_dp, eig_vecs, Ns_dp);
            }
            double et0 = MPI_Wtime();
            MPI_Bcast(eig_vecs, Ns_dp * Ns_dp, MPI_DOUBLE, 0, DP_CheFSI->kpt_comm);
            double et1 = MPI_Wtime();
            #ifdef DEBUG
            if (pSPARC->StandardEigenFlag == 0) {
                if (rank_kpt == 0) printf("DP_Solve_Generalized_EigenProblem, info = %d, rank 0 used %.3lf ms, LAPACKE_dsygvd used %.3lf ms\n", info, 1000.0 * (et1 - st), 1000.0 * (et0 - st));
            } else {
                if (rank_kpt == 0) printf("DP_Solve_Generalized_EigenProblem, info = %d, rank 0 used %.3lf ms, LAPACKE_dsyevd used %.3lf ms\n", info, 1000.0 * (et1 - st), 1000.0 * (et0 - st));
            }
            #endif
        } else {
            #if defined(USE_MKL) || defined(USE_SCALAPACK)
            int rank_dmcomm = -1;
            if (pSPARC->dmcomm != MPI_COMM_NULL) 
                MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
            // Hp and Mp is only correct at the first blacscomm
            if (rank_dmcomm == 0) {
                int ONE = 1;
                // Step 1: redistribute DP_CheFSI->Hp_local and DP_CheFSI->Mp_local on rank_kpt == 0 to ScaLAPACK format 
                pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, DP_CheFSI->Hp_local, &ONE, &ONE, 
                        DP_CheFSI->desc_Hp_local, pSPARC->Hp, &ONE, &ONE, 
                        pSPARC->desc_Hp_BLCYC, &pSPARC->ictxt_blacs_topo);
                pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, DP_CheFSI->Mp_local, &ONE, &ONE, 
                        DP_CheFSI->desc_Mp_local, pSPARC->Mp, &ONE, &ONE, 
                        pSPARC->desc_Mp_BLCYC, &pSPARC->ictxt_blacs_topo);

                // Step 2: use scalapack to solve the generalized eigenproblem
                Solve_Generalized_EigenProblem(pSPARC, 0, spn_i); // the 2nd arg is not used

                // Step 3: redistribute the obtained eigenvectors from ScaLAPACK format to DP_CheFSI->eig_vecs on rank_kpt == 0
                pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Q, &ONE, &ONE, 
                        pSPARC->desc_Q_BLCYC, DP_CheFSI->eig_vecs, &ONE, &ONE, 
                        DP_CheFSI->desc_eig_vecs, &pSPARC->ictxt_blacs_topo);
            }
            int Ns_dp = DP_CheFSI->Ns_dp;
            MPI_Bcast(DP_CheFSI->eig_vecs, Ns_dp * Ns_dp, MPI_DOUBLE, 0, DP_CheFSI->kpt_comm);
            
            if (pSPARC->npNd > 1 && pSPARC->bandcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
                double *eig_val  = pSPARC->lambda + spn_i * Ns_dp;
                MPI_Bcast(eig_val, Ns_dp, MPI_DOUBLE, 0, pSPARC->dmcomm);
            }

            #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
            int rank_kpt = DP_CheFSI->rank_kpt;
            if (rank_kpt == 0) printf("[FATAL] Subspace eigenproblem should be solved using ScaLAPACK but ScaLAPACK is not compiled\n");
            exit(255);
            #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
        }
    }
}

/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals, using domain parallelization
 *          data partitioning. 
 *
 *          This is just to perform a matrix-matrix multiplication: PsiQ = YPsi* Q, here 
 *          Psi == Y, Q == eigvec. Note that Y and YQ are in domain parallelization data 
 *          layout. We use MPI_Alltoallv to convert the obtained YQ back to the band + domain 
 *          parallelization format in SPARC and copy the transformed YQ to Psi_rot. 
 */
void DP_Subspace_Rotation(SPARC_OBJ *pSPARC, double *Psi_rot)
{
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    if (DP_CheFSI == NULL) return;
    
    int rank_kpt = DP_CheFSI->rank_kpt;
    double st, et0, et1;
    
    // Psi == Y, Q == eig_vecs, we store Psi * Q in HY
    st = MPI_Wtime();
    int Nd_dp = DP_CheFSI->Nd_dp;
    int Ns_dp = DP_CheFSI->Ns_dp;
    double *Y_dp     = DP_CheFSI->Y_dp;
    double *YQ_dp    = DP_CheFSI->HY_dp;
    double *eig_vecs = DP_CheFSI->eig_vecs;
    // SPARCX_ACCEL_NOTE: ADD HERE THE LINES FOR GPU-ACCELERATED ROTATION
	#ifdef SPARCX_ACCEL
	if (pSPARC->useACCEL == 1)
	{
		ACCEL_DGEMM(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        Nd_dp, Ns_dp, Ns_dp, 
        1.0, Y_dp, Nd_dp, eig_vecs, Ns_dp,
        0.0, YQ_dp, Nd_dp
    	);
	}
	else
	#endif // SPARCX_ACCEL
	{ // SPARCX_ACCEL_NOTE Brackets now needed to enclose the original CPU-only cblas call
		cblas_dgemm(
			CblasColMajor, CblasNoTrans, CblasNoTrans,
			Nd_dp, Ns_dp, Ns_dp, 
			1.0, Y_dp, Nd_dp, eig_vecs, Ns_dp,
			0.0, YQ_dp, Nd_dp
		);
	} // SPARCX_ACCEL_NOTE End.

    et0 = MPI_Wtime();
    
    // Redistribute Psi * Q back into band + domain format using MPI_Alltoallv
    DP2BP(
        pSPARC->blacscomm, DP_CheFSI->nproc_row,
        DP_CheFSI->Ndsp_bp, DP_CheFSI->Ns_bp, DP_CheFSI->Nd_dp_displs,
        DP_CheFSI->bp2dp_sendcnts, DP_CheFSI->bp2dp_sdispls,
        DP_CheFSI->dp2bp_sendcnts, DP_CheFSI->dp2bp_sdispls,
        sizeof(double), YQ_dp, DP_CheFSI->Y_packbuf, Psi_rot
    );
    // Synchronize here to prevent some processes run too fast and enter next DP_Project_Hamiltonian too earlier
    MPI_Barrier(DP_CheFSI->kpt_comm);
    et1 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_kpt == 0) printf("DP_Subspace_Rotation rank 0 used %.3lf ms, redist PsiQ used %.3lf ms\n\n", 1000.0 * (et1 - st), 1000.0 * (et1 - et0));
    #endif
}

/**
 * @brief   Free domain parallelization data structures for calculating projected Hamiltonian, 
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI().
 */
void free_DP_CheFSI(SPARC_OBJ *pSPARC)
{
    DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
    if (DP_CheFSI == NULL) return;
    
    free(DP_CheFSI->Ns_bp_displs);
    free(DP_CheFSI->Nd_dp_displs);
    free(DP_CheFSI->bp2dp_sendcnts);
    free(DP_CheFSI->bp2dp_sdispls);
    free(DP_CheFSI->dp2bp_sendcnts);
    free(DP_CheFSI->dp2bp_sdispls);
    free(DP_CheFSI->Y_packbuf);
    free(DP_CheFSI->HY_packbuf);
    free(DP_CheFSI->Y_dp);
    free(DP_CheFSI->HY_dp);
    free(DP_CheFSI->Mp_local);
    free(DP_CheFSI->Hp_local);
    free(DP_CheFSI->eig_vecs);
    MPI_Comm_free(&DP_CheFSI->kpt_comm);
    
    free(DP_CheFSI);
    pSPARC->DP_CheFSI = NULL;
}

#endif  // End of #ifdef USE_DP_SUBEIG

/**
 * @brief   Calculate projected Hamiltonian and overlap matrix.
 *
 *          Hp = Y' * H * Y, 
 *          Mp = Y' * Y.
 */
void Project_Hamiltonian(SPARC_OBJ *pSPARC, int *DMVertices, double *Y, int ldi, double *HY, int ldo,
                         double *Hp, double *Mp, int k, int spn_i, MPI_Comm comm) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->bandcomm_index < 0 || comm == MPI_COMM_NULL) return;

    int nproc_dmcomm, rank;
    MPI_Comm_size(comm, &nproc_dmcomm);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2, t3, t4;
#ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
#endif

    int sg  = pSPARC->spin_start_indx + spn_i;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndspe = DMnd * pSPARC->Nspinor_eig;
    int ONE = 1;
    double alpha = 1.0, beta = 0.0;
    
    /* Calculate Mp = Y' * Y */
    t3 = MPI_Wtime();
    t1 = MPI_Wtime();
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        // distribute orbitals into block cyclic format
        pdgemr2d_(&DMndspe, &pSPARC->Nstates, Y, &ONE, &ONE, pSPARC->desc_orbitals,
                  pSPARC->Yorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, &pSPARC->ictxt_blacs); 
    } 
    t2 = MPI_Wtime();  
    #ifdef DEBUG  
    if(!rank && spn_i == 0) 
        printf("rank = %2d, Distribute orbital to block cyclic format took %.3f ms\n", 
                rank, (t2 - t1)*1e3);          
    #endif

    t1 = MPI_Wtime();
    // perform matrix multiplication using ScaLAPACK routines
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) { 
        if (pSPARC->CyclixFlag) {
            #ifdef DEBUG    
            if (!rank && spn_i == 0) printf("rank = %d, STARTING PDGEMM ...\n",rank);
            #endif   
            pdgemm_("T", "N", &pSPARC->Nstates, &pSPARC->Nstates, &DMndspe, &alpha, 
                pSPARC->Yorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC,
                pSPARC->Yorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, &beta, Mp, 
                &ONE, &ONE, pSPARC->desc_Mp_BLCYC);
        } else {
            #ifdef DEBUG    
            if (!rank && spn_i == 0) printf("rank = %d, STARTING PDSYRK ...\n",rank);
            #endif   
            // perform matrix multiplication using ScaLAPACK routines
            pdsyrk_("U", "T", &pSPARC->Nstates, &DMndspe, &alpha, pSPARC->Yorb_BLCYC, &ONE, &ONE,
                pSPARC->desc_orb_BLCYC, &beta, Mp, &ONE, &ONE, pSPARC->desc_Mp_BLCYC);
        }
    } else {
        if (pSPARC->CyclixFlag) {
            #ifdef DEBUG    
            if (!rank && spn_i == 0) printf("rank = %d, STARTING DGEMM ...\n",rank);
            #endif   
            cblas_dgemm(
                CblasColMajor, CblasTrans, CblasNoTrans,
                pSPARC->Nstates, pSPARC->Nstates, DMndspe,
                alpha, pSPARC->Yorb_BLCYC, DMndspe, pSPARC->Yorb_BLCYC, DMndspe, 
                beta, Mp, pSPARC->Nstates
            );
        } else {
            #ifdef DEBUG    
            if (!rank && spn_i == 0) printf("rank = %d, STARTING DSYRK ...\n",rank);
            #endif   
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, pSPARC->Nstates, DMndspe, alpha, 
                pSPARC->Yorb_BLCYC, DMndspe, beta, Mp, pSPARC->Nstates);
        }
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) 
        printf("rank = %2d, Psi'*Psi in block cyclic format in each blacscomm took %.3f ms\n", 
                rank, (t2 - t1)*1e3); 
    #endif
    
    t1 = MPI_Wtime();
    if (nproc_dmcomm > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Mp, pSPARC->nr_Mp_BLCYC*pSPARC->nc_Mp_BLCYC, 
                      MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    t2 = MPI_Wtime();
    t4 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, Allreduce to sum Psi'*Psi over dmcomm took %.3f ms\n", 
                     rank, (t2 - t1)*1e3); 
    if(!rank && spn_i == 0) printf("rank = %2d, Distribute data + matrix mult took %.3f ms\n", 
                     rank, (t4 - t3)*1e3);
    #endif

    if (pSPARC->StandardEigenFlag == 1){
        // Orthogonalization using Choleskey 
        t1 = MPI_Wtime();
        Chol_orth(pSPARC->Yorb_BLCYC, pSPARC->desc_orb_BLCYC, Mp, pSPARC->desc_Mp_BLCYC, &DMndspe, &pSPARC->Nstates);
        t2 = MPI_Wtime();
        // update Yorb
        pdgemr2d_(&DMndspe, &pSPARC->Nstates, pSPARC->Yorb_BLCYC, &ONE, &ONE, 
              pSPARC->desc_orb_BLCYC, Y, &ONE, &ONE, 
              pSPARC->desc_orbitals, &pSPARC->ictxt_blacs);
        t3 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank && spn_i == 0) printf("Orthogonalization of orbitals took: %.3f ms\n", (t2 - t1)*1e3); 
        if(!rank && spn_i == 0) printf("Updating orbitals took: %.3f ms\n", (t3 - t2)*1e3);
        #endif
    }

    /* Calculate Hp = Y' * HY */
    // first find HY
    double *HY_BLCYC;
    t1 = MPI_Wtime();
    // save HY in Xorb
    Hamiltonian_vectors_mult(
        pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * DMnd, pSPARC->Atom_Influence_nloc, 
        pSPARC->nlocProj, pSPARC->Nband_bandcomm, 0.0, Y, ldi, HY, ldo, spn_i, pSPARC->dmcomm
    );
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, finding HY took %.3f ms\n", rank, (t2 - t1)*1e3);   
    #endif
    
    t1 = MPI_Wtime();
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        // distribute HY
        HY_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));
        pdgemr2d_(&DMndspe, &pSPARC->Nstates, HY, &ONE, &ONE, 
                  pSPARC->desc_orbitals, HY_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, 
                  &pSPARC->ictxt_blacs);
    } else {
        HY_BLCYC = HY;
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, distributing HY into block cyclic form took %.3f ms\n", 
                     rank, (t2 - t1)*1e3);  
    #endif
    
    t1 = MPI_Wtime();
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        // perform matrix multiplication Y' * HY using ScaLAPACK routines
        pdgemm_("T", "N", &pSPARC->Nstates, &pSPARC->Nstates, &DMndspe, &alpha, 
                pSPARC->Yorb_BLCYC, &ONE, &ONE, pSPARC->desc_orb_BLCYC, HY_BLCYC, 
                &ONE, &ONE, pSPARC->desc_orb_BLCYC, &beta, Hp, &ONE, &ONE, 
                pSPARC->desc_Hp_BLCYC);
    } else {
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans,
            pSPARC->Nstates, pSPARC->Nstates, DMndspe,
            1.0, pSPARC->Yorb_BLCYC, DMndspe, HY_BLCYC, DMndspe, 
            0.0, Hp, pSPARC->Nstates
        );
    }

    if (nproc_dmcomm > 1) {
        // sum over all processors in dmcomm
        MPI_Allreduce(MPI_IN_PLACE, Hp, pSPARC->nr_Hp_BLCYC*pSPARC->nc_Hp_BLCYC, 
                      MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm);
    }
    
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, finding Y'*HY took %.3f ms\n",rank,(t2-t1)*1e3); 
    #endif
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        free(HY_BLCYC);
    }

    #ifdef DEBUG
    et = MPI_Wtime();
    if (rank == 0 && spn_i == 0) printf("Rank 0, Project_Hamiltonian used %.3lf ms\n", 1000.0 * (et - st)); 
    #endif
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}



/**
 * @brief   Solve generalized eigenproblem Hp * x = lambda * Mp * x.
 *
 *          Note: Hp = Psi' * H * Psi, Mp = Psi' * Psi. Also note that both Hp and 
 *                Mp are distributed block cyclically.
 *          
 *          TODO: At some point it is better to use ELPA (https://elpa.mpcdf.mpg.de/) 
 *                for solving subspace eigenvalue problem, which can provide up to 
 *                3x speedup.
 */
void Solve_Generalized_EigenProblem(SPARC_OBJ *pSPARC, int k, int spn_i) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank, rank_spincomm, rank_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_rank(pSPARC->kptcomm,  &rank_kptcomm);
    #ifdef DEBUG
    if (!rank && spn_i == 0) printf("Start solving generalized eigenvalue problem ...\n");
    #endif
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int nproc_dmcomm;
    MPI_Comm_size(pSPARC->dmcomm, &nproc_dmcomm);

    double t1, t2;
    #ifdef DEBUG    
    double st = MPI_Wtime();
    #endif

    #ifdef SPARCX_ACCEL // SPARCX_ACCEL_NOTE
    if (pSPARC->useACCEL == 1 && pSPARC->cell_typ < 20 && !pSPARC->useHIP) {
		int info = 0;
		t1 = MPI_Wtime();
		if (!pSPARC->bandcomm_index) {
			if (pSPARC->StandardEigenFlag == 0)
				info = DSYGV(LAPACK_COL_MAJOR,1,'V','U',pSPARC->Nstates,pSPARC->Hp,
							  pSPARC->Nstates,pSPARC->Mp,pSPARC->Nstates,
							  pSPARC->lambda + spn_i*pSPARC->Nstates);
			else 
				info = DSYEV(LAPACK_COL_MAJOR,'V','U',pSPARC->Nstates,pSPARC->Hp,
							  pSPARC->Nstates, pSPARC->lambda + spn_i*pSPARC->Nstates);
		}
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (pSPARC->StandardEigenFlag == 0) {
			if(!rank_spincomm && spn_i == 0) {
				printf("==generalized eigenproblem: "
					   "info = %d, solving generalized eigenproblem using %s: %.3f ms\n", info, STR_DSYGV, (t2 - t1)*1e3);
			}
		} else {
			if(!rank_spincomm && spn_i == 0) {
				printf("==standard eigenproblem: "
					   "info = %d, solving standard eigenproblem using %s: %.3f ms\n", info, STR_DSYEV, (t2 - t1)*1e3);
			}
		}
		#endif

		int ONE = 1;
		t1 = MPI_Wtime();
		// distribute eigenvectors to block cyclic format
		pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp, &ONE, &ONE, 
				  pSPARC->desc_Hp_BLCYC, pSPARC->Q, &ONE, &ONE, 
				  pSPARC->desc_Q_BLCYC, &pSPARC->ictxt_blacs_topo);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if(!rank_spincomm && spn_i == 0) {
			printf("==generalized eigenproblem: "
				   "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
				   (t2 - t1)*1e3);
		}
		#endif
	}
	else
	#endif // SPARCX_ACCEL
    { // SPARCX_ACCEL_NOTE Enclose the whole IF+ELSE statement from CPU-only and place it at the bottom.
        if (pSPARC->useLAPACK == 1) {
            int info = 0;
            t1 = MPI_Wtime();
            if (!pSPARC->bandcomm_index) {
                if (pSPARC->CyclixFlag) {
                    info = generalized_eigenvalue_problem_cyclix(pSPARC, 
                                pSPARC->Hp, pSPARC->Mp, pSPARC->lambda + spn_i*pSPARC->Nstates);
                } else if (pSPARC->StandardEigenFlag == 0) {
                    info = LAPACKE_dsygvd(LAPACK_COL_MAJOR,1,'V','U',pSPARC->Nstates,pSPARC->Hp,
                                pSPARC->Nstates,pSPARC->Mp,pSPARC->Nstates,
                                pSPARC->lambda + spn_i*pSPARC->Nstates);
                } else {
                    info = LAPACKE_dsyevd(LAPACK_COL_MAJOR,'V','U',pSPARC->Nstates,pSPARC->Hp,
                                pSPARC->Nstates, pSPARC->lambda + spn_i*pSPARC->Nstates);
                }
            }
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if (pSPARC->StandardEigenFlag == 0) {
                if(!rank_spincomm && spn_i == 0) {
                    printf("==generalized eigenproblem: "
                        "info = %d, solving generalized eigenproblem using LAPACKE_dsygvd: %.3f ms\n", 
                        info, (t2 - t1)*1e3);
                }
            } else {
                if(!rank_spincomm && spn_i == 0) {
                    printf("==standard eigenproblem: "
                        "info = %d, solving standard eigenproblem using LAPACKE_dsyevd: %.3f ms\n", 
                        info, (t2 - t1)*1e3);
                }
            }
            #endif

            int ONE = 1;
            t1 = MPI_Wtime();
            // distribute eigenvectors to block cyclic format
            pdgemr2d_(&pSPARC->Nstates, &pSPARC->Nstates, pSPARC->Hp, &ONE, &ONE, 
                    pSPARC->desc_Hp_BLCYC, pSPARC->Q, &ONE, &ONE, 
                    pSPARC->desc_Q_BLCYC, &pSPARC->ictxt_blacs_topo);
            t2 = MPI_Wtime();
            #ifdef DEBUG
            if(!rank_spincomm && spn_i == 0) {
                printf("==generalized eigenproblem: "
                    "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
                    (t2 - t1)*1e3);
            }
            #endif
        } else {
            int ONE = 1, il = 1, iu = 1, *ifail, info, N, M, NZ;
            double vl = 0.0, vu = 0.0, abstol, orfac;

            ifail = (int *)malloc(pSPARC->Nstates * sizeof(int));
            N = pSPARC->Nstates;
            orfac = pSPARC->eig_paral_orfac;
            #ifdef DEBUG
            if(!rank) printf("rank = %d, orfac = %.3e\n", rank, orfac);
            #endif
            // this setting yields the most orthogonal eigenvectors
            // abstol = pdlamch_(&pSPARC->ictxt_blacs_topo, "U");
            abstol = -1.0;

            t1 = MPI_Wtime();
            if (pSPARC->StandardEigenFlag == 0)
                pdsygvx_subcomm_ (
                    &ONE, "V", "A", "U", &N, pSPARC->Hp, &ONE, &ONE, 
                    pSPARC->desc_Hp_BLCYC, pSPARC->Mp, &ONE, &ONE, 
                    pSPARC->desc_Mp_BLCYC, &vl, &vu, &il, &iu, &abstol, 
                    &M, &NZ, pSPARC->lambda + spn_i*N, &orfac, pSPARC->Q, 
                    &ONE, &ONE, pSPARC->desc_Q_BLCYC, ifail, &info,
                    pSPARC->blacscomm, pSPARC->eig_paral_subdims, pSPARC->eig_paral_blksz);
            else
                pdsyevx_subcomm_ ("V", "A", "U", &N, pSPARC->Hp, &ONE, &ONE, 
                        pSPARC->desc_Hp_BLCYC, &vl, &vu, &il, &iu, &abstol, 
                        &M, &NZ, pSPARC->lambda + spn_i*N, &orfac, 
                        pSPARC->Q, &ONE, &ONE, pSPARC->desc_Q_BLCYC, ifail, &info,
                        pSPARC->blacscomm, pSPARC->eig_paral_subdims, pSPARC->eig_paral_blksz);
                
            t2 = MPI_Wtime();

            if (info != 0 && !rank) {
                printf("\nError in solving generalized eigenproblem! info = %d\n", info);
            }
                        
            #ifdef DEBUG
            if(!rank && spn_i == 0) {
                printf("rank = %d, info = %d, ifail[0] = %d, time for solving generalized eigenproblem: %.3f ms\n", 
                        rank, info, ifail[0], (t2 - t1)*1e3);
                printf("rank = %d, after calling pdsygvx, Nstates = %d\n", rank, N);
            }
            #endif
            free(ifail);
        }
    } // SPARCX_ACCEL_NOTE END

    #ifdef DEBUG    
    double et = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Solve_Generalized_EigenProblem used %.3lf ms\n", rank, 1000.0 * (et - st));
    #endif
        
#else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("[FATAL] Subspace eigenproblem are using ScaLAPACK routines but ScaLAPACK is not compiled\n");
    if (rank == 0) printf("\nPlease turn on USE_MKL or USE_SCALAPACK!\n");
    exit(255);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
} 



/**
 * @brief   Perform subspace rotation, i.e. rotate the orbitals.
 *
 *          This is just to perform a matrix-matrix multiplication: Psi = Psi * Q.
 *          Note that Psi, Q and PsiQ are distributed block cyclically, Psi_rot is
 *          the band + domain parallelization format of PsiQ.
 */
void Subspace_Rotation(SPARC_OBJ *pSPARC, double *Psi, double *Q, double *PsiQ, double *Psi_rot, int k, int spn_i)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    #ifdef DEBUG
    double st = MPI_Wtime();
    #endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int ONE = 1;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndspe = DMnd * pSPARC->Nspinor_eig;

    double alpha = 1.0, beta = 0.0;
    double t1, t2;

    t1 = MPI_Wtime();
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        // perform matrix multiplication Psi * Q using ScaLAPACK routines
        pdgemm_("N", "N", &DMndspe, &pSPARC->Nstates, &pSPARC->Nstates, &alpha, 
                Psi, &ONE, &ONE, pSPARC->desc_orb_BLCYC, Q, &ONE, &ONE, 
                pSPARC->desc_Q_BLCYC, &beta, PsiQ, &ONE, &ONE, pSPARC->desc_orb_BLCYC);
    } else {
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            DMndspe, pSPARC->Nstates, pSPARC->Nstates, 
            1.0, Psi, DMndspe, Q, pSPARC->Nstates,
            0.0, PsiQ, DMndspe
        );
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, subspace rotation using ScaLAPACK took %.3f ms\n", 
                     rank, (t2 - t1)*1e3); 
    #endif
    t1 = MPI_Wtime();
    if (pSPARC->npband > 1 || pSPARC->Nspinor_eig != pSPARC->Nspinor_spincomm) {
        // distribute rotated orbitals from block cyclic format back into 
        // original format (band + domain)
        pdgemr2d_(&DMndspe, &pSPARC->Nstates, PsiQ, &ONE, &ONE, 
                  pSPARC->desc_orb_BLCYC, Psi_rot, &ONE, &ONE, 
                  pSPARC->desc_orbitals, &pSPARC->ictxt_blacs);
    }
    t2 = MPI_Wtime();    
    #ifdef DEBUG
    if(!rank && spn_i == 0) 
        printf("rank = %2d, Distributing orbital back into band + domain format took %.3f ms\n", 
                rank, (t2 - t1)*1e3); 
    #endif
    
    #ifdef DEBUG
    double et = MPI_Wtime();
    if (rank == 0) printf("rank = %d, Subspace_Rotation used %.3lf ms\n\n", rank, 1000.0 * (et - st));
    #endif
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}



/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Hamiltonian corresponding to the given k-point.  
 */
void Lanczos(const SPARC_OBJ *pSPARC, int *DMVertices, double *Veff_loc,
             ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
             double *eigmin, double *eigmax, double *x0, double TOL_min, double TOL_max, 
             int MAXIT, int k, int spn_i, MPI_Comm comm, MPI_Request *req_veff_loc) 
{
    double t1, t2, ts, te;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
    if (rank == 0 && spn_i == 0) printf("\nStart Lanczos algorithm ...\n");
    #endif

    ts = MPI_Wtime();
    if (comm == MPI_COMM_NULL) {
        t1 = MPI_Wtime();
        // receive computed eigmin and eigmax from root processors in the Cart topology
        double Bbuf[2];

        // the non-root processes do nothing
        MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm_inter);
        // this is inefficient since many proc already have the info.
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm); 
        
        *eigmin = Bbuf[0]; *eigmax = Bbuf[1];

        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && spn_i == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
        #endif
        return;
    }

    double vscal, err_eigmin, err_eigmax, eigmin_pre, eigmax_pre;
    double *V_j, *V_jm1, *V_jp1, *a, *b, *d, *e;
    int i, j, DMnd, DMndspe;
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);
    DMndspe = DMnd * pSPARC->Nspinor_eig;
    
    V_j   = (double*)malloc( DMndspe * sizeof(double));
    V_jm1 = (double*)malloc( DMndspe * sizeof(double));
    V_jp1 = (double*)malloc( DMndspe * sizeof(double));
    a     = (double*)malloc( (MAXIT+1) * sizeof(double));
    b     = (double*)malloc( (MAXIT+1) * sizeof(double));
    d     = (double*)malloc( (MAXIT+1) * sizeof(double));
    e     = (double*)malloc( (MAXIT+1) * sizeof(double));

    /* set random initial guess vector V_jm1 with unit 2-norm */
#ifdef DEBUG
    srand(rank+1);
    //srand(rank+(int)MPI_Wtime());
#else
    srand(rank+1); 
    //srand(rank+1+(int)MPI_Wtime());
#endif    
    double rand_min = -1.0, rand_max = 1.0;
    for (i = 0; i < DMndspe; i++) {
        //V_jm1[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        //TODO: [1,...,1] might be a better guess for it's closer to the eigvec for Lap 
        //      with zero (or ~= zero) eigval, and since min eig is harder to converge
        //      this will make the overall convergence faster.
        //V_jm1[i] = 1.0 - 1e-3 + 2e-3 * (double) rand() / RAND_MAX; 
        V_jm1[i] = x0[i];
    }

    Vector2Norm(V_jm1, DMndspe, &vscal, comm); // find norm of V_jm1
    vscal = 1.0 / vscal;
    // scale the random guess vector s.t. V_jm1 has unit 2-norm
    for (i = 0; i < DMndspe; i++) 
        V_jm1[i] *= vscal;

    // calculate V_j = H * V_jm1, TODO: check if Veff_loc is available
    // TODO: remove if not using nonblocking communication
    t1 = MPI_Wtime();
    MPI_Wait(req_veff_loc, MPI_STATUS_IGNORE);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank && spn_i == 0) printf("Wait for veff to be bcasted took %.3f ms\n", (t2-t1)*1e3);
#endif
    t1 = MPI_Wtime();
    Hamiltonian_vectors_mult(
        pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
        nlocProj, 1, 0.0, V_jm1, DMndspe, V_j, DMndspe, spn_i, comm
    );
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, One H*x took %.3f ms\n", rank, (t2-t1)*1e3);   
#endif
    // find dot product of V_jm1 and V_j, and store the value in a[0]
    VectorDotProduct(V_jm1, V_j, DMndspe, &a[0], comm);

    // orthogonalize V_jm1 and V_j
    for (i = 0; i < DMndspe; i++)
        V_j[i] -= a[0] * V_jm1[i];
    
    // find norm of V_j
    Vector2Norm(V_j, DMndspe, &b[0], comm); 
    
    if (!b[0]) {
        // if ||V_j|| = 0, pick an arbitrary vector with unit norm that's orthogonal to V_jm1
        rand_min = -1.0, rand_max = 1.0;
        for (i = 0; i < DMndspe; i++) {
            V_j[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        }
        // orthogonalize V_j and V_jm1
        VectorDotProduct(V_j, V_jm1, DMndspe, &a[0], comm);
        for (i = 0; i < DMndspe; i++)
            V_j[i] -= a[0] * V_jm1[i];
        // find norm of V_j
        Vector2Norm(V_j, DMndspe, &b[0], comm);
    }

    // scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (i = 0; i < DMndspe; i++) 
        V_j[i] *= vscal;

    eigmin_pre = *eigmin = 0.0;
    eigmax_pre = *eigmax = 0.0;
    err_eigmin = TOL_min + 1.0;
    err_eigmax = TOL_max + 1.0;
    j = 0;
    while ((err_eigmin > TOL_min || err_eigmax > TOL_max) && j < MAXIT) 
    {
        //t1 = MPI_Wtime();        
        // V_{j+1} = H * V_j
        Hamiltonian_vectors_mult(
            pSPARC, DMnd, DMVertices, Veff_loc, Atom_Influence_nloc, 
            nlocProj, 1, 0.0, V_j, DMndspe, V_jp1, DMndspe, spn_i, comm
        );

        // a[j+1] = <V_j, V_{j+1}>
        VectorDotProduct(V_j, V_jp1, DMndspe, &a[j+1], comm);

        for (i = 0; i < DMndspe; i++) {
            // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
            V_jp1[i] -= (a[j+1] * V_j[i] + b[j] * V_jm1[i]);    
            // update V_{j-1}, i.e., V_{j-1} := V_j
            V_jm1[i] = V_j[i];
        }
        
        Vector2Norm(V_jp1, DMndspe, &b[j+1], comm);
        if (!b[j+1]) {
            break;
        }
        
        vscal = 1.0 / b[j+1];
        // update V_j := V_{j+1} / ||V_{j+1}||
        for (i = 0; i < DMndspe; i++)
            V_j[i] = V_jp1[i] * vscal;

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        for (i = 0; i < j+2; i++) {
            d[i] = a[i];
            e[i] = b[i];
        }
        
        if (!LAPACKE_dsterf(j+2, d, e)) {
            *eigmin = d[0];
            *eigmax = d[j+1];
        } else {
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }
        
        err_eigmin = fabs(*eigmin - eigmin_pre);
        err_eigmax = fabs(*eigmax - eigmax_pre);

        eigmin_pre = *eigmin;
        eigmax_pre = *eigmax;

        j++;
    }
    te = MPI_Wtime();    

#ifdef DEBUG
    if (rank == 0 && spn_i == 0) {
        printf("    Lanczos iter %d, eigmin  = %.9f, eigmax = %.9f, err_eigmin = %.3e, err_eigmax = %.3e, taking %.3f ms.\n",j,*eigmin, *eigmax,err_eigmin,err_eigmax,1e3*(te-ts));
    }
#endif
    
    if (pSPARC->kptcomm_inter != MPI_COMM_NULL) {
        t1 = MPI_Wtime();

        // broadcast the computed eigmin and eigmax from root to processors not in the Cart topology
        int rank_kptcomm = -1;
        MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

        double Bbuf[2];
        Bbuf[0] = *eigmin; Bbuf[1] = *eigmax;
        if (!rank_kptcomm) {
            // the root process will broadcast the values
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_ROOT, pSPARC->kptcomm_inter);
        } else {
            // the non-root processes do nothing
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kptcomm_inter);
        }
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm);

        t2 = MPI_Wtime();
#ifdef DEBUG
        if(!rank && spn_i == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
#endif
    }
    
    free(V_j); free(V_jm1); free(V_jp1);
    free(a); free(b); free(d); free(e);
}



/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for the Lap.  
 */
void Lanczos_laplacian(
    const SPARC_OBJ *pSPARC, const int *DMVertices, double *eigmin, 
    double *eigmax, double *x0, const double TOL_min, const double TOL_max, 
    const int MAXIT, int k, int spn_i, MPI_Comm comm
) 
{
    double t1, t2, ts, te;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
    if (rank == 0 && spn_i == 0) printf("\nStart Lanczos algorithm ...\n");
    #endif
    ts = MPI_Wtime();

    if (comm == MPI_COMM_NULL) {
        t1 = MPI_Wtime();
        // receive computed eigmin and eigmax from root processors in the Cart topology
        double Bbuf[2];
        // the non-root processes do nothing
        MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm_inter);
        // this is inefficient since many proc already have the info.
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm); 
        
        *eigmin = Bbuf[0]; *eigmax = Bbuf[1];

        t2 = MPI_Wtime();
        #ifdef DEBUG
        if (!rank && spn_i == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
        #endif
        return;
    }

    double vscal, err_eigmin, err_eigmax, eigmin_pre, eigmax_pre;
    double *V_j, *V_jm1, *V_jp1, *a, *b, *d, *e;
    int i, j, DMnd, DMndspe;
    DMnd = (1 - DMVertices[0] + DMVertices[1]) * 
           (1 - DMVertices[2] + DMVertices[3]) * 
           (1 - DMVertices[4] + DMVertices[5]);
    DMndspe = DMnd * pSPARC->Nspinor_eig;

    V_j   = (double*)malloc( DMndspe * sizeof(double));
    V_jm1 = (double*)malloc( DMndspe * sizeof(double));
    V_jp1 = (double*)malloc( DMndspe * sizeof(double));
    a     = (double*)malloc( (MAXIT+1) * sizeof(double));
    b     = (double*)malloc( (MAXIT+1) * sizeof(double));
    d     = (double*)malloc( (MAXIT+1) * sizeof(double));
    e     = (double*)malloc( (MAXIT+1) * sizeof(double));

    /* set random initial guess vector V_jm1 with unit 2-norm */
#ifdef DEBUG
    srand(rank+1); 
    //srand(rank+(int)MPI_Wtime());
#else
    srand(rank+1); 
    //srand(rank+1+(int)MPI_Wtime());
#endif    
    double rand_min = -1.0, rand_max = 1.0;
    for (i = 0; i < DMndspe; i++) {
        //V_jm1[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        //TODO: [1,...,1] might be a better guess for it's closer to the eigvec for Lap 
        //      with zero (or ~= zero) eigval, and since min eig is harder to converge
        //      this will make the overall convergence faster.
        //V_jm1[i] = 1.0 - 1e-3 + 2e-3 * (double) rand() / RAND_MAX; 
        V_jm1[i] = x0[i];
    }

    Vector2Norm(V_jm1, DMndspe, &vscal, comm); // find norm of V_jm1
    vscal = 1.0 / vscal;
    // scale the random guess vector s.t. V_jm1 has unit 2-norm
    for (i = 0; i < DMndspe; i++) 
        V_jm1[i] *= vscal;

    // calculate V_j = H * V_jm1
    t1 = MPI_Wtime();
    Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, V_jm1, DMndspe, V_j, DMndspe, comm);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank && spn_i == 0) printf("rank = %2d, One H*x took %.3f ms\n", rank, (t2-t1)*1e3);   
#endif
    // find dot product of V_jm1 and V_j, and store the value in a[0]
    VectorDotProduct(V_jm1, V_j, DMndspe, &a[0], comm);

    // orthogonalize V_jm1 and V_j
    for (i = 0; i < DMndspe; i++)
        V_j[i] -= a[0] * V_jm1[i];
    
    // find norm of V_j
    Vector2Norm(V_j, DMndspe, &b[0], comm); 
    
    if (!b[0]) {
        // if ||V_j|| = 0, pick an arbitrary vector with unit norm that's orthogonal to V_jm1
        rand_min = -1.0, rand_max = 1.0;
        for (i = 0; i < DMndspe; i++) {
            V_j[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
        }
        // orthogonalize V_j and V_jm1
        VectorDotProduct(V_j, V_jm1, DMndspe, &a[0], comm);
        for (i = 0; i < DMndspe; i++)
            V_j[i] -= a[0] * V_jm1[i];
        // find norm of V_j
        Vector2Norm(V_j, DMndspe, &b[0], comm);
    }

    // scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    for (i = 0; i < DMndspe; i++) 
        V_j[i] *= vscal;

    t1 = MPI_Wtime();
    eigmin_pre = *eigmin = 0.0;
    eigmax_pre = *eigmax = 0.0;
    err_eigmin = TOL_min + 1.0;
    err_eigmax = TOL_max + 1.0;
    j = 0;
    while ((err_eigmin > TOL_min || err_eigmax > TOL_max) && j < MAXIT) {
        // V_{j+1} = H * V_j
		Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, V_j, DMndspe, V_jp1, DMndspe, comm);

        // a[j+1] = <V_j, V_{j+1}>
        VectorDotProduct(V_j, V_jp1, DMndspe, &a[j+1], comm);

        for (i = 0; i < DMndspe; i++) {
            // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
            V_jp1[i] -= (a[j+1] * V_j[i] + b[j] * V_jm1[i]);    
            // update V_{j-1}, i.e., V_{j-1} := V_j
            V_jm1[i] = V_j[i];
        }

        Vector2Norm(V_jp1, DMndspe, &b[j+1], comm);
        if (!b[j+1]) {
            break;
        }
        
        vscal = 1.0 / b[j+1];
        // update V_j := V_{j+1} / ||V_{j+1}||
        for (i = 0; i < DMndspe; i++)
            V_j[i] = V_jp1[i] * vscal;

        // solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
        for (i = 0; i < j+2; i++) {
            d[i] = a[i];
            e[i] = b[i];
        }        
        
        if (!LAPACKE_dsterf(j+2, d, e)) {
            *eigmin = d[0];
            *eigmax = d[j+1];
        } else {
            if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
            break;
        }        
        
        err_eigmin = fabs(*eigmin - eigmin_pre);
        err_eigmax = fabs(*eigmax - eigmax_pre);

        eigmin_pre = *eigmin;
        eigmax_pre = *eigmax;

        j++;
    }
    te = MPI_Wtime();
#ifdef DEBUG
    if (rank == 0 && spn_i == 0) {
        printf("    Lanczos iter %d, eigmin  = %.9f, eigmax = %.9f, err_eigmin = %.3e, err_eigmax = %.3e, taking %.3f ms\n",j,*eigmin, *eigmax,err_eigmin,err_eigmax,1e3*(te-ts));
    }
#endif
    
    if (pSPARC->kptcomm_inter != MPI_COMM_NULL) {
        t1 = MPI_Wtime();

        // broadcast the computed eigmin and eigmax from root to processors not in the Cart topology
        int rank_kptcomm = -1;
        MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);

        double Bbuf[2];
        Bbuf[0] = *eigmin; Bbuf[1] = *eigmax;
        if (!rank_kptcomm) {
            // the root process will broadcast the values
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_ROOT, pSPARC->kptcomm_inter);
        } else {
            // the non-root processes do nothing
            MPI_Bcast(Bbuf, 2, MPI_DOUBLE, MPI_PROC_NULL, pSPARC->kptcomm_inter);
        }
        // MPI_Bcast(Bbuf, 2, MPI_DOUBLE, 0, pSPARC->kptcomm);

        t2 = MPI_Wtime();
#ifdef DEBUG
        if(!rank && spn_i == 0) printf("rank = %d, inter-communicator Bcast took %.3f ms\n",rank,(t2-t1)*1e3);
#endif
    }
    
    free(V_j); free(V_jm1); free(V_jp1);
    free(a); free(b); free(d); free(e);
}



/**
 * @brief   Calculate Chebyshev polynomial degree based on effective mesh size.
 *
 *          The concept of effective mesh size used here since the Chebyshev polynomial
 *          degree is directly related to the spectral width of the Hamiltonian, which
 *          can be approximated by the maximum eigenvalue the -0.5*Laplacian, which is
 *          preportional to h_eff^-2. We use a cubic polynomial p(x) to fit four data
 *          points (h, npl) = {(0.1,50), (0.2,35), (0.4,20), (0.7,14)}. If h > 0.7, we
 *          fix npl to be 14.
 *
 * @param h   Effective mesh size.
 */
int Mesh2ChebDegree(double h) 
{
    // the relation between h and npl is fit with a cubic polynomial
    // p(x) = p3 * x^3 + p2 * x^2 + p1 * x + p0.
    double p3, p2, p1, p0, npl;
    p3 = -700. / 3.;
    p2 = 1240. / 3.;
    p1 = -773. / 3.;
    p0 = 1078. / 15.;
    if (h > 0.7) {
        npl = 14;
    } else {
        npl = ((p3 * h + p2) * h + p1) * h + p0;
    }
    return round(npl);
}


/**
 * @brief   Orthogonalization of dense matrix A by Choleskey factorization
 */
void Chol_orth(double *A, const int *descA, double *z, const int *descz, const int *m, const int *n)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int ONE = 1, info;
    double alpha = 1.0;

    pdpotrf_("U", n, z, &ONE, &ONE, descz, &info);  
    pdtrsm_("R", "U", "N", "N", m, n, &alpha, z, &ONE, &ONE, descz, A, &ONE, &ONE, descA);
#endif //(#ifdef USE_MKL)    
}
