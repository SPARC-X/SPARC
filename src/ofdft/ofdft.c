/***
 * @file    ofdft.c
 * @brief   This file contains the functions for Orbital Free DFT.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <time.h>
/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/** ScaLAPACK routines */
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

#include "ofdft.h"
#include "tools.h" 
#include "lapVecRoutines.h"
#include "electrostatics.h"
#include "exchangeCorrelation.h"
#include "lapVecRoutines.h"
#include "parallelization.h"
#include "electronicGroundState.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL (1e-14)


/**
 * @brief   Initialize OFDFT variables
 */
void init_OFDFT(SPARC_OBJ *pSPARC) {
    pSPARC->OFDFT_Cf = 0.3*pow(3*M_PI*M_PI,2.0/3);
    pSPARC->OFDFT_lambda =  0.2;
}

/**
 * @brief   OFDFT NLCG algorithm to solve for electron density
 */
void OFDFT_NLCG_TETER(SPARC_OBJ *pSPARC) {
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return;
    int i, iter, iter2, maxit, DMnd, rank, s_iter;
    int *DMVertices;
    double tol1, tol2, eta, s, cst, deltaNew, v1, v2, xi, time1, time2;
    double *u, *F, *r, *d, *r_old;
    MPI_Comm comm;
    comm = pSPARC->dmcomm_phi;
    MPI_Comm_rank(comm, &rank);
    FILE *output_fp;
    
    if (!rank && pSPARC->Verbosity) {
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"===================================================================\n");
        
        if(pSPARC->RelaxFlag >= 1)
            fprintf(output_fp,"               Orbital Free DFT NLCG (OFDFT-NLCG#%d)                \n",
                    pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
        else
            fprintf(output_fp,"               Orbital Free DFT NLCG (OFDFT-NLCG#%d)                \n",1);
        
        fprintf(output_fp,"===================================================================\n");
        fprintf(output_fp,"Iteration    Free Energy (Ha/atom)   NLCG Error        Timing (sec)\n");
        fclose(output_fp);
    }
    
    iter = iter2 = 0;
    maxit = 1500; 
    DMnd = pSPARC->Nd_d;
    tol1 = pSPARC->OFDFT_tol * pSPARC->OFDFT_tol * pSPARC->Nd;
    // Tolerance of brents method to find step length s
    tol2 = 1E-2;
    DMVertices = pSPARC->DMVertices;
    cst = sqrt(pSPARC->Nelectron / pSPARC->dV);
    
    F = (double *) calloc(DMnd, sizeof(double));
    r = (double *) calloc(DMnd, sizeof(double));
    d = (double *) calloc(DMnd, sizeof(double));
    r_old = (double *) calloc(DMnd, sizeof(double));
    pSPARC->OFDFT_u = (double *) calloc(DMnd, sizeof(double));
    assert(F != NULL && r != NULL && d != NULL && r_old != NULL && pSPARC->OFDFT_u != NULL);

    time1 = MPI_Wtime();
    u = pSPARC->OFDFT_u;
    // Be careful! Esc is only correct in rank 0.
    // Could be replaced by using MPI_Allreduce in Esc part.
    MPI_Bcast(&pSPARC->Esc, 1, MPI_DOUBLE, 0, comm);
    for (i = 0; i < DMnd; i++) {
        u[i] = sqrt(fabs(pSPARC->electronDens[i]));
    }

    HamiltonianVecRoutines_OFDFT(pSPARC, DMnd, DMVertices, u, F, comm);
    VectorDotProduct(F, u, DMnd, &eta, comm);
    eta *= (pSPARC->dV / pSPARC->Nelectron);
    for (i = 0; i < DMnd; i++) 
        r[i] = -2 * (F[i] - eta * u[i]);

    s = Brent_Emin(pSPARC, tol2, u, r, &s_iter);
    extend_normalized_vector(u, r, s, u, DMnd, cst, comm);
    for (i = 0; i < DMnd; i++) r_old[i] = r[i];

    while (iter < maxit) {
        HamiltonianVecRoutines_OFDFT(pSPARC, DMnd, DMVertices, u, F, comm);
        VectorDotProduct(F, u, DMnd, &eta, comm);
        eta *= (pSPARC->dV / pSPARC->Nelectron);
        for (i = 0; i < DMnd; i++) 
            r[i] = -2 * (F[i] - eta * u[i]);
        
        VectorDotProduct(r, r, DMnd, &deltaNew, comm);
        
        time2 = MPI_Wtime();
        if (!rank && pSPARC->Verbosity) {
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"%-6d      %18.10E       %.3E         %.3f\n", 
                                iter+1, pSPARC->Etot/pSPARC->n_atom, sqrt(deltaNew/pSPARC->Nd), time2 - time1);
            fclose(output_fp);
        }
        time1 = MPI_Wtime();

    #ifdef DEBUG
        if (!rank) printf(" iter%-5d  s %.6f s-iter %d Energy(Ha/atom) %.6E   error %.3E\n",
                        iter + 1, s, s_iter, pSPARC->Etot/pSPARC->n_atom, sqrt(deltaNew/pSPARC->Nd));
    #endif

        if (deltaNew < tol1) break;
        VectorDotProduct(r_old, r, DMnd, &v1, comm);
        VectorDotProduct(r_old, r_old, DMnd, &v2, comm);
        xi = (deltaNew - v1) / v2;

        if (iter2 == 30 || xi <= 0) {
            for (i = 0; i < DMnd; i++) d[i] = r[i];
            iter2 = 0;
        } else {
            for (i = 0; i < DMnd; i++) 
                d[i] = xi * d[i] + r[i];
        }
        s = Brent_Emin(pSPARC, tol2, u, d, &s_iter);
        extend_normalized_vector(u, d, s, u, DMnd, cst, comm);

        for (i = 0; i < DMnd; i++) r_old[i] = r[i];
        iter2 ++;
        iter ++;
    }
    if (iter == maxit) {
        if (!rank) printf("WARNING: OFDFT-NLCG %d did not converge to desired accuracy!\n", 
                           pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
        // write to .out file
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"WARNING: OFDFT-NLCG %d did not converge to desired accuracy!\n", 
                            pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
        fclose(output_fp);
    }

    ofdftTotalEnergy(pSPARC, u);

    free(F);
    free(r);
    free(d);
    free(r_old);
    free(pSPARC->OFDFT_u);
    return;
}

/**
 * @brief   Calculate Hamiltonian times a vector in a matrix-free way.
 *          
 *          The Hamiltonian includes the TFW kinetic functional. 
 *          TODO: add WGC kinetic functional. 
 */
void HamiltonianVecRoutines_OFDFT(
        SPARC_OBJ *pSPARC, int DMnd, int *DMVertices,
        double *u, double *Hu, MPI_Comm comm) {
    
    int i, nproc;
    double cst;
    MPI_Comm_size(comm, &nproc);

    for (i = 0; i < DMnd; i ++)
        pSPARC->electronDens[i] = u[i] * u[i];
    
    // solve the poisson equation for electrostatic potential, "phi"
    Calculate_elecstPotential(pSPARC);
    
    // calculate xc potential (LDA, PW92), "Vxc"
    Calculate_Vxc(pSPARC);
    
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);

    // calculate Vk TFW kinetic functional
    // Vk = (5/3)*Cf*(rho.^(2/3))
    cst = (5.0 / 3.0) * pSPARC->OFDFT_Cf;
    for (i = 0; i < DMnd; i++) 
        pSPARC->Veff_loc_dmcomm_phi[i] +=  cst * pow(pSPARC->electronDens[i], (2.0/3));

    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else 
        dims[0] = dims[1] = dims[2] = 1;
    
    cst = -0.5 * pSPARC->OFDFT_lambda;

    if (pSPARC->cell_typ == 0)
        Lap_plus_diag_vec_mult_orth(
            pSPARC, DMnd, DMVertices, 1, cst, 1.0, 0.0, 
            pSPARC->Veff_loc_dmcomm_phi, u, DMnd, Hu, DMnd, comm, dims);
    else
        Lap_plus_diag_vec_mult_nonorth(
            pSPARC, DMnd, DMVertices, 1, cst, 1.0, 0.0,
            pSPARC->Veff_loc_dmcomm_phi, u, DMnd, Hu, DMnd, comm, pSPARC->comm_dist_graph_phi, dims);
    return;
}

/**
 * @brief   Brents algorithm to find a minimal point within an interval
 * 
 *          argmin_s Energy(u + s * d)
 *          return the s and s_iter is the number of iterations to get s. 
 *          TODO: make it more general. Input option for interval, input constraint function, etc. 
 */
double Brent_Emin(SPARC_OBJ *pSPARC, double tol, double *U, double *D, int *s_iter) {

    int iter, maxit, DMnd, rank;
    double ax, bx, cx, cgold, zeps, tol1, tol2;
    double a, b, d, e, etemp, fu, fv, fw, fx, p, q, r, u, v, w, x, xm;
    MPI_Comm comm;

    maxit = 100;
    ax = 0.0;               // lowerbound of interval
    bx = 0.5;               // start point, middle of interval
    cx = 1.0;               // upperbound of interval
    cgold = 0.3819660;
    zeps = 1e-10;
    DMnd = pSPARC->Nd_d;
    comm = pSPARC->dmcomm_phi;
    MPI_Comm_rank(comm, &rank);

    a = min(ax,cx);
    b = max(ax,cx);
    v = bx;
    // This is the start point in MATLAB fminbnd function
    // middle point seems to be slightly better
    // v = a + cgold * (b-a);
    w = v;
    x = v;
    e = d = 0;    

    fx = energy_constraint(pSPARC, U, D, x, DMnd, comm);
    fv = fx;
    fw = fx;
    for (iter = 1; iter < maxit; iter++) {
        xm = 0.5 * (a + b);
        tol1 = tol * fabs(x) + zeps;
        tol2 = 2 * tol1;
        if (fabs(x - xm) <= tol2 - 0.5*(b - a)) break;
        if (fabs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2 * (q - r);
            if (q > 0) p = -p;
            q = fabs(q);
            etemp = e;
            e = d;
            if(fabs(p) >= fabs(0.5 * q * etemp) || p <= q * (a-x) || p >= q * (b - x)) goto one;
            d = p/q;
            u = x+d;
            if(((u-a) < tol2) || ((b-u) < tol2))
            d = (((xm-x) >= 0) ? fabs(tol1) : -fabs(tol1));
            goto two;
        }

        one:if(x >= xm) e = a-x;
            else e = b - x;
            d = cgold * e;

        two:if (fabs(d) >= tol1) 
                u = x + d;
            else 
                u = x + ((d >= 0) ? fabs(tol1) : -fabs(tol1));
        fu = energy_constraint(pSPARC, U, D, u, DMnd, comm);
        if (fu <= fx) {
            if (u >= x) 
                a=x;
            else 
                b=x;

            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if (u < x) 
                a=u;
            else 
                b=u;

            if(fu <= fw || w == x) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if(fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }
    }
    *s_iter = iter;
    if(iter == maxit && !rank) 
        printf("WARNING: Brents method finidng min s fails to converge within %d iterations!", maxit);
    
    return x;
}

/**
 * @brief   Compute the energy with sqrt(rho) = u + s * d
 *          
 *          return Energy per atom in Ha/atom
 */
double energy_constraint(SPARC_OBJ *pSPARC, double *u, double *d, double s, int DMnd, MPI_Comm comm) {
    int i;
    double cst, *nu;

    cst = sqrt(pSPARC->Nelectron / pSPARC->dV);
    nu = (double *) calloc(sizeof(double), DMnd);
    extend_normalized_vector(u, d, s, nu, DMnd, cst, comm);
    for (i = 0; i < DMnd; i ++)
        pSPARC->electronDens[i] = nu[i] * nu[i];
    
    Calculate_Vxc(pSPARC);
    // solve the poisson equation for electrostatic potential, "phi"
    Calculate_elecstPotential(pSPARC);
    ofdftTotalEnergy(pSPARC, nu);
    free(nu);
    return (pSPARC->Etot/pSPARC->n_atom);
}

/**
 * @brief   Compute the energy with sqrt(rho) = u
 */
void ofdftTotalEnergy(SPARC_OBJ *pSPARC, double *u) {

    int i, DMnd, rank;
    double E1, E2, Et1, Et2;
    double *Lapu;
    MPI_Comm comm;

    Et1 = Et2 = 0;
    DMnd = pSPARC->Nd_d;
    comm = pSPARC->dmcomm_phi;
    Lapu = (double *) calloc(DMnd, sizeof(double));
    MPI_Comm_rank(comm, &rank);

    // calculate exchange correlation energy
    Calculate_Exc(pSPARC, pSPARC->electronDens);

    // Eele = 0.5 * integral of phi * (rho + b)
    VectorDotProduct(pSPARC->psdChrgDens, pSPARC->elecstPotential, DMnd, &E1, comm);
    VectorDotProduct(pSPARC->electronDens, pSPARC->elecstPotential, DMnd, &E2, comm);
    E1 *= 0.5 * pSPARC->dV;
    E2 *= 0.5 * pSPARC->dV;
    pSPARC->OFDFT_Eele = E1 + E2;

    // Et1 = S.ofdft_Cf*sum(rho.^(5/3))*S.dV;
    for (i = 0; i < DMnd; i++)
        Et1 += pow(pSPARC->electronDens[i], (5.0/3));
    Et1 *= (pSPARC->OFDFT_Cf * pSPARC->dV);
    MPI_Allreduce(MPI_IN_PLACE, &Et1, 1, MPI_DOUBLE, MPI_SUM, comm);

    // Et2 = integral of -0.5 * dot(u,Lap(u))
    Lap_vec_mult(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, u, DMnd, Lapu, DMnd, comm);
    VectorDotProduct(u, Lapu, DMnd, &Et2, comm);
    Et2 *= (-0.5 * pSPARC->dV);

    pSPARC->OFDFT_Ek = Et1 + pSPARC->OFDFT_lambda * Et2;

    pSPARC->Etot = pSPARC->Exc + pSPARC->OFDFT_Eele + pSPARC->OFDFT_Ek + pSPARC->Esc;
    // if(!rank) printf("Etot %f, Ek %f, Exc %f, Eele %f, Esc %f\n", pSPARC->Etot, pSPARC->OFDFT_Ek, pSPARC->Exc, pSPARC->OFDFT_Eele, pSPARC->Esc);
    free(Lapu);

    return;
}

/**
 * @brief   Set up sub-communicators for OFDFT.
 * 
 * Note:    Part of the codes are copied and modified from the code in parallelization.c file
 */
void Setup_Comms_OFDFT(SPARC_OBJ *pSPARC) {
    int i, dims[3] = {0, 0, 0}, periods[3], ierr;
    int nproc, rank;
    int npNd, gridsizes[3], minsize, coord_dmcomm[3], rank_dmcomm;
    int DMnx, DMny, DMnz, DMnd;
#ifdef DEBUG
    double t1, t2;
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Set up OFDFT communicators.\n");
#endif

    //------------------------------------------------//
    //               set up poisson domain            //
    //------------------------------------------------//
    // some variables are reused here
    npNd = pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi;
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    minsize = pSPARC->order/2;

    if (npNd == 0)  {
        // when user does not provide domain decomposition parameters
        npNd = nproc; // try to use all processors
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution."
                   "Please check if your gridsizes are less than MINSIZE\n");
        }
        pSPARC->npNdx_phi = dims[0];
        pSPARC->npNdy_phi = dims[1];
        pSPARC->npNdz_phi = dims[2];
		pSPARC->useDefaultParalFlag = 1;
    } else if (npNd < 0 || npNd > nproc || pSPARC->Nx / pSPARC->npNdx_phi < minsize ||
               pSPARC->Ny / pSPARC->npNdy_phi < minsize || pSPARC->Nz / pSPARC->npNdz_phi < minsize) {
        // when domain decomposition parameters are not reasonable
        npNd = nproc;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution."
                   "Please check if your gridsizes are less than MINSIZE\n");
        }
        pSPARC->npNdx_phi = dims[0];
        pSPARC->npNdy_phi = dims[1];
        pSPARC->npNdz_phi = dims[2];
		if (!rank) printf("WARNING: Default parallelization used because parallelization parameters are unreasonable.\n");
			pSPARC->useDefaultParalFlag = 1;
    } else {
        dims[0] = pSPARC->npNdx_phi;
        dims[1] = pSPARC->npNdy_phi;
        dims[2] = pSPARC->npNdz_phi;
		if (!rank) printf("WARNING: Default parallelization not used. This could result in degradation of performance.\n");
            pSPARC->useDefaultParalFlag = 0;
    }

    // recalculate number of processors in dmcomm_phi
    npNd = pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi;

    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &pSPARC->dmcomm_phi); // 1 is to reorder rank

#ifdef DEBUG
    if (rank == 0) {
        printf("========================================================================\n"
                   "Poisson domain decomposition:"
                   "np total = %d, {Nx, Ny, Nz} = {%d, %d, %d}\n"
                   "nproc used = %d = {%d, %d, %d}, nodes/proc = {%.2f, %.2f, %.2f}\n\n",
                   nproc,pSPARC->Nx,pSPARC->Ny,pSPARC->Nz,dims[0]*dims[1]*dims[2],dims[0],dims[1],dims[2],pSPARC->Nx/(double)dims[0],pSPARC->Ny/(double)dims[1],pSPARC->Nz/(double)dims[2]);
    }
#endif

    // find the vertices of the domain in each processor
    // pSPARC->DMVertices[6] = [xs,xe,ys,ye,zs,ze]
    // int Nx_dist, Ny_dist, Nz_dist;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm);
        MPI_Cart_coords(pSPARC->dmcomm_phi, rank_dmcomm, 3, coord_dmcomm);

        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;

        // find size of distributed domain
        pSPARC->Nx_d = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->Ny_d = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->Nz_d = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->Nd_d = pSPARC->Nx_d * pSPARC->Ny_d * pSPARC->Nz_d;

        // find corners of the distributed domain
        pSPARC->DMVertices[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->DMVertices[1] = pSPARC->DMVertices[0] + pSPARC->Nx_d - 1;
        pSPARC->DMVertices[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->DMVertices[3] = pSPARC->DMVertices[2] + pSPARC->Ny_d - 1;
        pSPARC->DMVertices[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->DMVertices[5] = pSPARC->DMVertices[4] + pSPARC->Nz_d - 1;
    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSPARC->Nx_d = 0;
        pSPARC->Ny_d = 0;
        pSPARC->Nz_d = 0;
        pSPARC->Nd_d = 0;
        pSPARC->DMVertices[0] = 0;
        pSPARC->DMVertices[1] = 0;
        pSPARC->DMVertices[2] = 0;
        pSPARC->DMVertices[3] = 0;
        pSPARC->DMVertices[4] = 0;
        pSPARC->DMVertices[5] = 0;
    }

    // Set up a 26 neighbor communicator for nonorthogonal systems
    // TODO: Modify the communicator based on number of non zero enteries in off diagonal of pSPARC->lapcT
    // TODO: Take into account more than 26 Neighbors
    if(pSPARC->cell_typ != 0) {
        if(pSPARC->dmcomm_phi != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;
            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_dmcomm[0] + i;
                            ncoords[1] = coord_dmcomm[1] + j;
                            ncoords[2] = coord_dmcomm[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->dmcomm_phi,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->dmcomm_phi,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->comm_dist_graph_phi); // creates a distributed graph topology (adjacent, cartesian cubical)
            //pSPARC->dmcomm_phi = pSPARC->comm_dist_graph_phi;
            free(neighb);
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up dmcomm_phi took %.3f ms\n",(t2-t1)*1000);
#endif

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\n--set up dmcomm_phi took %.3f ms\n",(t2-t1)*1000);
#endif

    /* allocate memory for storing atomic forces*/
    pSPARC->forces = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
    assert(pSPARC->forces != NULL);

    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        /* allocate memory for electrostatics calculation */
        DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
        DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
        DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
        DMnd = DMnx * DMny * DMnz;
        // allocate memory for electron density (sum of atom potential) and charge density
        pSPARC->electronDens_at = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->electronDens_core = (double *)calloc( DMnd * (2*pSPARC->Nspin-1), sizeof(double) );
        pSPARC->psdChrgDens = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->psdChrgDens_ref = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->Vc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->electronDens_core != NULL);
        assert(pSPARC->electronDens_at != NULL && pSPARC->psdChrgDens != NULL &&
               pSPARC->psdChrgDens_ref != NULL && pSPARC->Vc != NULL);
        // allocate memory for electron density
        pSPARC->electronDens = (double *)malloc( DMnd * (2*pSPARC->Nspin-1) * sizeof(double) );
        assert(pSPARC->electronDens != NULL);
        // allocate memory for charge extrapolation arrays
        if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
            pSPARC->delectronDens = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens != NULL);
            pSPARC->delectronDens_0dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_0dt != NULL);
            pSPARC->delectronDens_1dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_1dt != NULL);
            pSPARC->delectronDens_2dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_2dt != NULL);
            pSPARC->atom_pos_nm = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_nm != NULL);
            pSPARC->atom_pos_0dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_0dt != NULL);
            pSPARC->atom_pos_1dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_1dt != NULL);
            pSPARC->atom_pos_2dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_2dt != NULL);
        }
        // allocate memory for electrostatic potential
        pSPARC->elecstPotential = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->elecstPotential != NULL);

        // allocate memory for XC potential
        pSPARC->XCPotential = (double *)malloc( DMnd * pSPARC->Nspin * sizeof(double) );
        assert(pSPARC->XCPotential != NULL);

        // allocate memory for exchange-correlation energy density
        pSPARC->e_xc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->e_xc != NULL);

        // if GGA then allocate for xc energy per particle for each grid point and der. wrt. grho
        if(strcmp(pSPARC->XC,"GGA_PBE") == 0 || strcmp(pSPARC->XC,"GGA_RPBE") == 0 || strcmp(pSPARC->XC,"GGA_PBEsol") == 0
         || strcmp(pSPARC->XC,"PBE0") == 0 || strcmp(pSPARC->XC,"HF") == 0){
            pSPARC->Dxcdgrho = (double *)malloc( DMnd * (2*pSPARC->Nspin - 1) * sizeof(double) );
            assert(pSPARC->Dxcdgrho != NULL);
        }

        pSPARC->Veff_loc_dmcomm_phi = (double *)malloc(DMnd * pSPARC->Nspin * sizeof(double));
        assert(pSPARC->Veff_loc_dmcomm_phi != NULL);

        // initialize electrostatic potential as random guess vector
        if (pSPARC->FixRandSeed == 1) {
            SeededRandVec(pSPARC->elecstPotential, pSPARC->DMVertices, gridsizes, -1.0, 1.0, 0);
        } else {
            srand(rank+1);
            double rand_min = -1.0, rand_max = 1.0;
            for (i = 0; i < DMnd; i++) {
                pSPARC->elecstPotential[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX; // or 1.0
            }
        }
    }

    // parallelization summary
    #ifdef DEBUG
    if (rank == 0) {
        printf("\n");
        printf("-----------------------------------------------\n");
        printf("Parallelization summary\n");
        printf("Total number of processors: %d\n", nproc);
        printf("-----------------------------------------------\n");
        printf("== Phi domain ==\n");
        printf("Total number of processors used for Phi domain: %d\n", pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx_phi,pSPARC->npNdy_phi, pSPARC->npNdz_phi);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSPARC->Nd_d,pSPARC->Nx_d,pSPARC->Ny_d,pSPARC->Nz_d);
        printf("-----------------------------------------------\n");
    }
    #endif
}


/**
 * @brief   Free all allocated memory for OFDFT
 */
void Free_OFDFT(SPARC_OBJ *pSPARC) {
    // Free all variables in dmcomm_phi
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        if (pSPARC->cell_typ != 0) {
            MPI_Comm_free(&pSPARC->comm_dist_graph_phi);
        }
        MPI_Comm_free(&pSPARC->dmcomm_phi);
    }
}

/**
 * @brief   Free all allocated memory within each relax step
 */
void Free_NLCGvar_OFDFT(SPARC_OBJ *pSPARC) {
    int i;
    
    // free atom influence struct components
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        for (i = 0; i < pSPARC->Ntypes; i++) {
            free(pSPARC->Atom_Influence_local[i].coords);
            free(pSPARC->Atom_Influence_local[i].atom_spin);
            free(pSPARC->Atom_Influence_local[i].atom_index);
            free(pSPARC->Atom_Influence_local[i].xs);
            free(pSPARC->Atom_Influence_local[i].xe);
            free(pSPARC->Atom_Influence_local[i].ys);
            free(pSPARC->Atom_Influence_local[i].ye);
            free(pSPARC->Atom_Influence_local[i].zs);
            free(pSPARC->Atom_Influence_local[i].ze);
        }
        free(pSPARC->Atom_Influence_local);
    }
    return;
}


/**
 * @brief   Compute normalized vector with a direction 
 * 
 *          nu = c * normalize (u + s * d)
 */
void extend_normalized_vector(
        double *u, double *d, double s, double *nu, 
        int len, double c, MPI_Comm comm) {

    int i;
    double u_2norm;

    for (i = 0; i < len; i++) {
        nu[i] = u[i] + s * d[i];
    }    
    Vector2Norm(nu, len, &u_2norm, comm);
    for (i = 0; i < len; i++) {
        nu[i] *= (c / u_2norm);
        nu[i] = fabs(nu[i]);
    }
}
