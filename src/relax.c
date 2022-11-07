/**
 * @file    relaxation.c
 * @brief   This file contains the functions for performing structural relaxation.
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "relax.h"
#include "electronicGroundState.h"
#include "isddft.h"
#include "md.h"
#include "orbitalElecDensInit.h"
#include "initialization.h"
#include "electrostatics.h"
#include "eigenSolver.h" // Mesh2ChebDegree
#include "stress.h"
#include "tools.h"

#define SIGN(a,b) ((b)>=(0)?fabs(a):-fabs(a))
#define min(a,b) ((a)<(b)?(a):(b))

/**
@brief Main function of relaxation
**/
void main_Relax(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // For Dirichlet BC fix the atom closest to the center of the domain

    // Check if RestartFlag == 1 and .restart file not present
    if(pSPARC->RestartFlag == 1) {
        if(rank == 0){
            FILE *rst_fp = NULL;
            rst_fp = fopen(pSPARC->restart_Filename,"r");
            if(rst_fp == NULL)
                pSPARC->RestartFlag = 0;
        }
        MPI_Bcast(&pSPARC->RestartFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (pSPARC->RelaxFlag == 1) {
        if (strcmpi(pSPARC->RelaxMeth,"NLCG") == 0)
            NLCG(pSPARC);
        else if (strcmpi(pSPARC->RelaxMeth,"LBFGS") == 0)
            LBFGS(pSPARC);
        else if (strcmpi(pSPARC->RelaxMeth,"FIRE") == 0)
            FIRE(pSPARC);
        else {
            if (rank == 0) {
                printf("\nCannot recognize RelaxMeth = \"%s\"\n",pSPARC->RelaxMeth);
                printf("RelaxMeth (Relaxation Method) must be one of the following:\tNLCG\t FIRE\n");
            }
            exit(EXIT_FAILURE);
        }
    } else if (pSPARC->RelaxFlag == 2 || pSPARC->RelaxFlag == 3) {
        Relax_Cell(pSPARC);
    }
}

/* TODO: Add a preconditioner to accelerate geometry optimization
   Reference: A universal preconditioner for simulating condensed phase materials ( https://doi.org/10.1063/1.4947024)
*/
/*
 @brief   Performs relaxation of atom positions using NonLinear Conjugate Gradient (NLCG).
          Reference: An Introduction to the Conjugate Gradient Method Without
                     Agonizing Pain, Jonathan Richard Shewchuk (Section B5)
 */
void NLCG(SPARC_OBJ *pSPARC) {
    double t_init, t_acc;
    t_init = MPI_Wtime();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) {
        printf(GRN "Starting NLCG for structural relaxation\n"RESET);
    }
#endif
    // create and initialize the variables
    int jmax = 6, n = 30; // These values usually work well
    double sigma = pSPARC->NLCG_sigma, cgtol = pSPARC->TOL_RELAX, sectol = cgtol * 1e-2 ; //TODO: verify cgtol &sectol and sigma

    int iter, j, atmc, check = (pSPARC->PrintRelaxout == 1 && !rank), check1 = (pSPARC->Printrestart == 1 && !rank), szatm = 3 * pSPARC->n_atom;
    double *r, *s, *y, *F, delta_old, delta_mid, delta_new, delta_d, alpha, beta, eta, eta_prev, err, err2; //Init_pos
    r = (double *)malloc(szatm * sizeof(double)); // residual of current iteration
    s = (double *)malloc(szatm * sizeof(double)); // originally invM*r but here it is residual of previous iteration
    y = (double *)malloc(szatm * sizeof(double));
    F = (double *)malloc(szatm * sizeof(double));

    // Check whether the restart has to be performed
    if(pSPARC->RestartFlag != 0){
        RestartRelax(pSPARC); // collects atomic positions and previous search direction
        int atm;
        if(pSPARC->cell_typ != 0){
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
            }
        }
        Calculate_electronicGroundState(pSPARC);
        err = delta_new = 0.0;
        for(atmc = 0; atmc < szatm; atmc++){
            r[atmc] = pSPARC->forces[atmc];
            s[atmc] = r[atmc];
            delta_new += r[atmc] * s[atmc];
            if (fabs(pSPARC->forces[atmc]) > err)
                err = fabs(pSPARC->forces[atmc]); // defined as supremum norm of force vector
        }
    } else {
        Calculate_electronicGroundState(pSPARC);
        pSPARC->d = (double *)malloc(szatm * sizeof(double)); // search direction
        if (pSPARC->d == NULL) {
            printf("\nCannot allocate memory for search direction array in NLCG!\n");
            exit(EXIT_FAILURE);
        }
        err = delta_new = 0.0;
        for(atmc = 0; atmc < szatm; atmc++){
            r[atmc] = pSPARC->forces[atmc];
            s[atmc] = r[atmc];
            delta_new += r[atmc] * s[atmc];
            pSPARC->d[atmc] = s[atmc];
            if (fabs(pSPARC->forces[atmc]) > err)
                err = fabs(pSPARC->forces[atmc]);
        }
    }

    pSPARC->elecgs_Count++;
    pSPARC->RelaxCount++;

    int imax = pSPARC->Relax_Niter + pSPARC->restartCount + pSPARC->RelaxCount;
    FILE *output_relax, *output_fp;
    if(check){
        output_relax = fopen(pSPARC->RelaxFilename,"a");
   	    if(output_relax == NULL){
       	    printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
        	exit(EXIT_FAILURE);
   	    }
   	    
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
        fclose(output_fp);

        if(pSPARC->RestartFlag == 0){
            fprintf(output_relax,":RELAXSTEP: %d\n", pSPARC->RelaxCount);
            Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
        }
        fclose(output_relax);

    }

    iter = pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0);
    t_acc = (MPI_Wtime() - t_init)/60;

    while (iter < imax && err > cgtol && (t_acc + 1.2 * (MPI_Wtime() - t_init)/60) < pSPARC->TWtime) {
        t_init = MPI_Wtime();
        if (check){
			output_relax = fopen(pSPARC->RelaxFilename,"a+");
			if (output_relax == NULL) {
			    printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
			    exit(EXIT_FAILURE);
			}
			fprintf(output_relax,":RELAXSTEP: %d\n", iter);
		}

#ifdef DEBUG
        if(!rank)
            printf(":RelaxStep: %d\n",iter);
#endif
        eta = delta_d = 0.0;
        for(atmc = 0; atmc < szatm; atmc++){
            if (fabs(pSPARC->d[atmc]) > delta_d)
                delta_d = fabs(pSPARC->d[atmc]);
            y[atmc] = pSPARC->atom_pos[atmc];
            F[atmc] = pSPARC->forces[atmc];
            pSPARC->atom_pos[atmc] += sigma * pSPARC->d[atmc];
        }

        // Charge extrapolation (for better rho_guess)
        pSPARC->Relax_fac = sigma;
        elecDensExtrapolation(pSPARC);
        Check_atomlocation(pSPARC);
        Calculate_electronicGroundState(pSPARC);
        pSPARC->elecgs_Count++;
        eta_prev = 0.0;
        for(atmc = 0; atmc < szatm; atmc++){
            eta_prev -= pSPARC->forces[atmc] * pSPARC->d[atmc];
            pSPARC->atom_pos[atmc] = y[atmc];
            pSPARC->forces[atmc] = F[atmc];
        }
        j = 0;
        alpha = -sigma;
        //TODO: Update/modify the secant method in future
        do {
#ifdef DEBUG
            if(!rank)
            printf(":SecantStep: %d\n",j);
#endif
            eta = 0.0;
            for(atmc = 0; atmc < szatm; atmc++)
                eta -= pSPARC->forces[atmc] * pSPARC->d[atmc];
            alpha *= eta/(eta_prev - eta);
            err2 = fabs(alpha) * delta_d;
            for(atmc = 0; atmc < szatm; atmc++)
                pSPARC->atom_pos[atmc] += alpha * pSPARC->d[atmc];
           
            pSPARC->Relax_fac = alpha;
            elecDensExtrapolation(pSPARC);
            Check_atomlocation(pSPARC);
            Calculate_electronicGroundState(pSPARC);
            pSPARC->elecgs_Count++;
            eta_prev = eta;
            j++;
        } while (j < jmax && err2 > sectol);
#ifdef DEBUG
    if(!rank)
        printf("Secant method iterations = %d, err = %E\n", j, err2);
#endif
        delta_old = delta_new;
        delta_mid = delta_new = err = 0.0;
        for(atmc = 0; atmc < szatm; atmc++){
            r[atmc] = pSPARC->forces[atmc];
            delta_mid += r[atmc] * s[atmc];
            s[atmc] = r[atmc];
            delta_new += r[atmc] * s[atmc];
            if (fabs(pSPARC->forces[atmc]) > err)
                err = fabs(pSPARC->forces[atmc]);
        }
        beta = (delta_new - delta_mid)/delta_old;
        if (iter % n == 0 || beta <= 0) {
            for(atmc = 0; atmc < szatm; atmc++)
                pSPARC->d[atmc] = s[atmc];
        } else {
            for(atmc = 0; atmc < szatm; atmc++){
                pSPARC->d[atmc] = s[atmc] + beta * pSPARC->d[atmc];
            }
        }
        if(check){
           	Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
            fclose(output_relax);
        }
        if(check1 && !(iter % pSPARC->Printrestart_fq)) // printrestart_fq is the frequency at which the restart file is written
            PrintRelax(pSPARC);
        if(access("SPARC.stop", F_OK ) != -1 ){ // If a .stop file exists in the folder then the run will be terminated
            pSPARC->RelaxCount++;
            break;
        }
#ifdef DEBUG
        if (!rank) printf("Time taken by RelaxStep %d: %.3f s.\n", iter, (MPI_Wtime() - t_init));
#endif

        if(!rank){
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
            fclose(output_fp);
        }

        iter++;
        pSPARC->RelaxCount++;
        t_acc += (MPI_Wtime() - t_init)/60;
    }

    if(check1){
        pSPARC->RelaxCount--;
        PrintRelax(pSPARC);
    }
    free(r);
    free(s);
    free(y);
    free(F);
    free(pSPARC->d);
}

/*
@brief: function to perform Limited memory version of BFGS for structural relaxation (based on the implementation in VTST)
*/
void LBFGS(SPARC_OBJ *pSPARC) {
    double t_init, t_acc;
    t_init = MPI_Wtime();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0){
        printf(GRN "Starting L-BFGS for structural relaxation\n"RESET);
    }
#endif
    double lbfgs_tol = pSPARC->TOL_RELAX;
    int n = 3 * pSPARC->n_atom;

    int m = pSPARC->L_history;
    double finit_stp = pSPARC->L_finit_stp;
    double maxmov = pSPARC->L_maxmov;
    int autoscale = pSPARC->L_autoscale;
    int lineopt = pSPARC->L_lineopt; // Needed only if autoscale = 0
    double icurv = pSPARC->L_icurv; // Needed only if autoscale = 0

    double damp = 2.0;
    pSPARC->isFD = 1; // Never change here
    pSPARC->isReset = 1; // Never change here
    pSPARC->step = 0; // Never change here

    double *alpha, *xold; // Init_pos
    pSPARC->deltaX = (double *)calloc( m*n , sizeof(double) );
    pSPARC->deltaG = (double *)calloc( m*n , sizeof(double) );
    pSPARC->iys = (double *) calloc(m , sizeof(double));
    alpha = (double *) malloc(m * sizeof(double));
    xold = (double *) malloc(n * sizeof(double));
    pSPARC->fold = (double *) malloc(n * sizeof(double));
    pSPARC->d = (double *) malloc(n * sizeof(double));
    pSPARC->atom_disp = (double *) malloc(n * sizeof(double));

    int i, j, k, maxmov_flag, s_pos, bound;
    double curv, fnorm, dnorm, fp1, fp2, stp_sz, favg, beta, a1, a2;

    int check = (pSPARC->PrintRelaxout == 1 && !rank), check1 = (pSPARC->Printrestart == 1 && !rank);
    int iter;
    double err;

    // Check whether the restart has to be performed
    if(pSPARC->RestartFlag != 0){
        RestartRelax(pSPARC); // collects atomic positions
        int atm;
        if(pSPARC->cell_typ != 0){
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
            }
        }
        Calculate_electronicGroundState(pSPARC);
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->forces[i]) > err)
                err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
        }
    } else {
        Calculate_electronicGroundState(pSPARC);
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->forces[i]) > err)
                err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
        }
    }

    pSPARC->elecgs_Count++;
    pSPARC->RelaxCount++;

    int imax = pSPARC->Relax_Niter + pSPARC->restartCount + pSPARC->RelaxCount;
    FILE *output_relax, *output_fp;
    if(check){
        output_relax = fopen(pSPARC->RelaxFilename,"a");
        if(output_relax == NULL){
            printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
            exit(EXIT_FAILURE);
        }
        
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
        fclose(output_fp);

        if(pSPARC->RestartFlag == 0){
            fprintf(output_relax,":RELAXSTEP: %d\n", pSPARC->RelaxCount);
            Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
        }
        fclose(output_relax);

    }

    iter = pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0);
    t_acc = (MPI_Wtime() - t_init)/60.0;

    // TODO: Compute Hessian matrix directly later using perturbation theory
    while (iter < imax && err > lbfgs_tol && (t_acc + 1.2 * (MPI_Wtime() - t_init)/60.0) < pSPARC->TWtime) {
        t_init = MPI_Wtime();
        if (check){
            output_relax = fopen(pSPARC->RelaxFilename,"a+");
            if (output_relax == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_relax,":RELAXSTEP: %d\n", iter);
        }

#ifdef DEBUG
        if(!rank)
            printf(":RelaxStep: %d\n",iter);
#endif
        maxmov_flag = 0;
        if(autoscale){
            if(pSPARC->isFD){
            // Take the finite difference pSPARC->step down the forces
                fnorm = norm(n, pSPARC->forces);
                for(i = 0; i < n; i++){
                    pSPARC->d[i] = pSPARC->forces[i]/fnorm; // normalized force direction
                    xold[i] = pSPARC->atom_pos[i];
                    pSPARC->fold[i] = pSPARC->forces[i];
                    pSPARC->atom_pos[i] += pSPARC->d[i] * finit_stp; // finite difference pSPARC->step in the direction of the force
                }
                pSPARC->Relax_fac = finit_stp; // Needed for charge extrapolation
                pSPARC->isFD = 0;
            } else{
                for(i = 0; i < n; i++){
                    pSPARC->d[i] = pSPARC->atom_disp[i];
                }
                dnorm = norm(n, pSPARC->d);
                fp1 = dotproduct(n, pSPARC->fold, 0, pSPARC->d, 0)/dnorm;
                fp2 = dotproduct(n, pSPARC->forces, 0, pSPARC->d, 0)/dnorm;
                curv = (fp1 - fp2)/dnorm;
                icurv = 1.0/(curv * damp); // To be used as guess for inverse Hessian
                if(icurv < 0){
                    pSPARC->isReset = 1;
                    maxmov_flag = 1;
                }
                if(pSPARC->isReset == 1){
                    pSPARC->step = 0;
                    pSPARC->isReset = 0;
                } else{
                    if(pSPARC->step < m){
                        s_pos = pSPARC->step * n;
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->forces[i];
                        }
                        pSPARC->iys[pSPARC->step] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                    } else{
                        s_pos = (m-1) * n;
                        for(i = 0; i < s_pos; i++){
                            pSPARC->deltaX[i] = pSPARC->deltaX[i + n];
                            pSPARC->deltaG[i] = pSPARC->deltaG[i + n];
                        }
                        for(i = 0; i < m-1; i++)
                            pSPARC->iys[i] = pSPARC->iys[i + 1];
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->forces[i];
                        }
                        pSPARC->iys[m-1] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                    }
                    pSPARC->step++;
                }
                for(i = 0; i < n; i++){
                    xold[i] = pSPARC->atom_pos[i];
                    pSPARC->fold[i] = pSPARC->forces[i];
                }
                if(pSPARC->step < m)
                    bound = pSPARC->step;
                else
                    bound = m;

                // Perform rank two BFGS update
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = -pSPARC->forces[i];

                for(i = 0; i < bound; i++){
                    j = bound - i - 1;
                    s_pos = j * n;
                    alpha[j] = dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->d, 0);
                    alpha[j] *= pSPARC->iys[j];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] -= alpha[j] * pSPARC->deltaG[s_pos + k];
                }
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = icurv * pSPARC->d[i];
                for(i = 0; i < bound; i++){
                    s_pos = i * n;
                    beta = dotproduct(n, pSPARC->deltaG, s_pos, pSPARC->d, 0);
                    beta *= pSPARC->iys[i];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] += pSPARC->deltaX[s_pos + k] * (alpha[i] - beta);
                }
                for(i = 0; i < n; i++){
                    pSPARC->d[i] = -pSPARC->d[i];
                }

                stp_sz = norm(n, pSPARC->d);
                if(stp_sz > maxmov){
                    pSPARC->isReset = 1;
                    stp_sz = maxmov;
                    fnorm = norm(n, pSPARC->forces);
                    for(i = 0; i < n; i++)
                        pSPARC->d[i] = stp_sz * pSPARC->forces[i]/fnorm; //  Take a steepest descent pSPARC->step
                }
                if(maxmov_flag){
                    fnorm = norm(n, pSPARC->forces);
                    for(i = 0; i < n; i++){
                        pSPARC->d[i] = pSPARC->forces[i]/fnorm;
                        pSPARC->atom_pos[i] += maxmov * pSPARC->d[i];
                    }
                    pSPARC->Relax_fac = maxmov;
                    maxmov_flag = 0;
                } else{
                    for(i = 0; i < n; i++)
                        pSPARC->atom_pos[i] += pSPARC->d[i];
                    pSPARC->Relax_fac = 1.0;
                }
            }
        } else{
            if(pSPARC->isFD){
                pSPARC->isFD = 0;
                a1 = fabs(dotproduct(n, pSPARC->forces, 0, pSPARC->fold, 0));
                a2 = dotproduct(n, pSPARC->fold, 0, pSPARC->fold, 0);
                if(a1 > 0.5 * a2 || a2 == 0)
                    pSPARC->isReset = 1;
                if(lineopt == 0)
                    pSPARC->isReset = 0;
                if(a2 == 0)
                    pSPARC->isReset = 1;
                if(pSPARC->isReset){
                    pSPARC->step = 0;
                    pSPARC->isReset = 0;
                } else{
                    if(pSPARC->step < m){
                        s_pos = pSPARC->step * n;
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->forces[i];
                        }
                        pSPARC->iys[pSPARC->step] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                    } else{
                        s_pos = (m-1) * n;
                        for(i = 0; i < s_pos; i++){
                            pSPARC->deltaX[i] = pSPARC->deltaX[i + n];
                            pSPARC->deltaG[i] = pSPARC->deltaG[i + n];
                        }
                        for(i = 0; i < m-1; i++)
                            pSPARC->iys[i] = pSPARC->iys[i + 1];
                        for(i = 0; i < n; i++){
                            pSPARC->deltaX[s_pos + i] = pSPARC->atom_disp[i];
                            pSPARC->deltaG[s_pos + i] = pSPARC->fold[i] - pSPARC->forces[i];
                        }
                        pSPARC->iys[m-1] = 1.0/dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->deltaG, s_pos);
                    }
                    pSPARC->step++;
                }
                for(i = 0; i < n; i++){
                    xold[i] = pSPARC->atom_pos[i];
                    pSPARC->fold[i] = pSPARC->forces[i];
                }
                if(pSPARC->step < m)
                    bound = pSPARC->step;
                else
                    bound = m;

                // Perform rank 2 BFGS update

                for(i = 0; i < n; i++)
                    pSPARC->d[i] = -pSPARC->forces[i];
                for(i = 0; i < bound; i++){
                    j = bound - i - 1;
                    s_pos = j * n;
                    alpha[j] = dotproduct(n, pSPARC->deltaX, s_pos, pSPARC->d, 0);
                    alpha[j] *= pSPARC->iys[j];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] -= alpha[j] * pSPARC->deltaG[s_pos + k];
                }
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = icurv * pSPARC->d[i];
                for(i = 0; i < bound; i++){
                    s_pos = i * n;
                    beta = dotproduct(n, pSPARC->deltaG, s_pos, pSPARC->d, 0);
                    beta *= pSPARC->iys[i];
                    for(k = 0; k < n; k++)
                        pSPARC->d[k] += pSPARC->deltaX[s_pos + k] * (alpha[i] - beta);
                }
                for(i = 0; i < n; i++)
                    pSPARC->d[i] = -pSPARC->d[i];

                if(lineopt){
                    dnorm = norm(n, pSPARC->d);
                    for(i = 0; i < n; i++){
                        pSPARC->d[i] /= dnorm;
                        pSPARC->atom_pos[i] += pSPARC->d[i] * finit_stp; // finite difference pSPARC->step along search direction
                    }
                    pSPARC->Relax_fac = finit_stp;
                } else{
                    stp_sz = dnorm = norm(n, pSPARC->d);
                    if(stp_sz > maxmov){
                        stp_sz = maxmov;
                        for(i = 0; i < n; i++)
                            pSPARC->d[i] = stp_sz * pSPARC->d[i]/dnorm;
                    }
                    for(i = 0; i < n; i++)
                        pSPARC->atom_pos[i] += pSPARC->d[i];
                    pSPARC->Relax_fac = 1.0;
                    pSPARC->isFD = 1;
                }
            } else{
                pSPARC->isFD = 1;
                fp1 = dotproduct(n, pSPARC->fold, 0, pSPARC->d, 0);
                fp2 = dotproduct(n, pSPARC->forces, 0, pSPARC->d, 0);
                curv = (fp1 - fp2)/finit_stp;
                if(curv < 0)
                    stp_sz = maxmov;
                else{
                    favg = 0.5 * (fp1 + fp2);
                    stp_sz = favg/curv;
                    if(fabs(stp_sz) > maxmov)
                        stp_sz = SIGN(maxmov, stp_sz) - SIGN(finit_stp, stp_sz);
                    else
                        stp_sz -= 0.5 * finit_stp;
                }
                for(i = 0; i < n; i++)
                    pSPARC->atom_pos[i] += pSPARC->d[i] * stp_sz;
                pSPARC->Relax_fac = stp_sz;
            }
        }

        // Store the distance the atoms have moved between two relaxation iterations
        for(i = 0; i < n; i++)
            pSPARC->atom_disp[i] = pSPARC->atom_pos[i] - xold[i];

        elecDensExtrapolation(pSPARC);
        Check_atomlocation(pSPARC);
        Calculate_electronicGroundState(pSPARC);
        pSPARC->elecgs_Count++;
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->forces[i]) > err)
                err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
        }
        if(check){
            Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
            fclose(output_relax);
        }
        if(check1 && !(iter % pSPARC->Printrestart_fq)) // printrestart_fq is the frequency at which the restart file is written
            PrintRelax(pSPARC);
        if(access("SPARC.stop", F_OK ) != -1 ){ // If a .stop file exists in the folder then the run will be terminated
            pSPARC->RelaxCount++;
            break;
        }
#ifdef DEBUG
        if (!rank) printf("Time taken by RelaxStep %d: %.3f s.\n", iter, (MPI_Wtime() - t_init));
#endif
        if(!rank){
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
            fclose(output_fp);
        }


        pSPARC->RelaxCount++;
        iter++;
        t_acc += (MPI_Wtime() - t_init)/60.0;
    }


    if(check1){
        pSPARC->RelaxCount--;
        PrintRelax(pSPARC);
    }
    free(pSPARC->deltaX);
    free(pSPARC->deltaG);
    free(pSPARC->iys);
    free(alpha);
    free(xold);
    free(pSPARC->fold);
    free(pSPARC->atom_disp);
    free(pSPARC->d);
}

double norm(int n, double *vec){
    double vecnorm = 0.0;
    int i;
    for(i = 0; i < n; i++)
        vecnorm += pow(vec[i], 2.0);
    vecnorm = pow(vecnorm, 0.5);

    return vecnorm;
}

double dotproduct(int n, double *vec1, int s1, double *vec2, int s2){
    double dp = 0.0;
    int i;
    for(i = 0; i < n; i++)
        dp += vec1[s1 + i] * vec2[s2 + i];

    return dp;
}
/**
 * @brief   Performs relaxation of atom positions using FIRE algorithm.
 */
void FIRE(SPARC_OBJ *pSPARC) {
    double t_init, t_acc;
    t_init = MPI_Wtime();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) {
        printf(GRN "Starting FIRE for structural relaxation\n"RESET);
    }
#endif
    double fire_tol = pSPARC->TOL_RELAX;
    // user controlled variables
    pSPARC->FIRE_dt *= pSPARC->fs2atu; // convert from femto second to atomic unit of time
    double mass = pSPARC->FIRE_mass * pSPARC->amu2au; // convert from amu to atomic unit of mass
    double maxmov = pSPARC->FIRE_maxmov;
    // Internal variables
    double fDec = 0.5;
    double fInc = 1.1;
    int nMIN = 5;
    double alphaStart = 0.1;
    double fAlpha = 0.99;
    double dtMax = 10.0 * pSPARC->FIRE_dt;

    ///////////////////////////////////////////////////////////////////////////
    int check = (pSPARC->PrintRelaxout == 1 && !rank), check1 = (pSPARC->Printrestart == 1 && !rank);
    int n = 3 * pSPARC->n_atom, i, iter;
    double err, Pw, fnorm, vnorm;
    double *acc;
    pSPARC->d = (double *) malloc(n * sizeof(double)); // displacement vector
    pSPARC->FIRE_vel = (double *) malloc(n * sizeof(double));
    acc = (double *) malloc(n * sizeof(double));

    // Check whether the restart has to be performed
    if(pSPARC->RestartFlag != 0){
        RestartRelax(pSPARC); // collects atomic positions
        int atm;
        if(pSPARC->cell_typ != 0){
            for(atm = 0; atm < pSPARC->n_atom; atm++){
                Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
            }
        }
        Calculate_electronicGroundState(pSPARC);
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->forces[i]) > err)
                err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
        }
        for(i = 0; i < n; i++){
            acc[i] = pSPARC->forces[i]/mass;
        }
    } else {
        Calculate_electronicGroundState(pSPARC);
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->forces[i]) > err)
                err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
        }
        // Initialize
        pSPARC->FIRE_resetIter = 0;
        pSPARC->FIRE_alpha = alphaStart;
        pSPARC->FIRE_dtNow = pSPARC->FIRE_dt;
        for(i = 0; i < n; i++){
            pSPARC->FIRE_vel[i] = 0.0; // TODO: Is this the best initialization of velocities?
            acc[i] = pSPARC->forces[i]/mass;
        }
    }

    pSPARC->elecgs_Count++;
    pSPARC->RelaxCount++;

    int imax = pSPARC->Relax_Niter + pSPARC->restartCount + pSPARC->RelaxCount;
    FILE *output_relax, *output_fp;
    if(check){
        output_relax = fopen(pSPARC->RelaxFilename,"a");
        if(output_relax == NULL){
            printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
            exit(EXIT_FAILURE);
        }
        
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
        fclose(output_fp);

        if(pSPARC->RestartFlag == 0){
            fprintf(output_relax,":RELAXSTEP: %d\n", pSPARC->RelaxCount);
            Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
        }
        fclose(output_relax);

    }

    iter = pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0);
    t_acc = (MPI_Wtime() - t_init)/60.0;

    while (iter < imax && err > fire_tol && (t_acc + 1.2 * (MPI_Wtime() - t_init)/60.0) < pSPARC->TWtime) {
        t_init = MPI_Wtime();
        if (check){
            output_relax = fopen(pSPARC->RelaxFilename,"a+");
            if (output_relax == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_relax,":RELAXSTEP: %d\n", iter);
        }

#ifdef DEBUG
        if(!rank)
            printf(":RelaxStep: %d\n",iter);
#endif

        // Verlet step
        for(i = 0; i < n; i++){
            pSPARC->FIRE_vel[i] += 0.5 * pSPARC->FIRE_dtNow * acc[i];
            pSPARC->d[i] = pSPARC->FIRE_vel[i] * pSPARC->FIRE_dtNow;
            if(pSPARC->d[i] > maxmov)
                pSPARC->d[i] = maxmov;
            else if(pSPARC->d[i] < -maxmov)
                pSPARC->d[i] = -maxmov;
            pSPARC->atom_pos[i] += pSPARC->d[i];
        }
        pSPARC->Relax_fac = 1.0;
        elecDensExtrapolation(pSPARC);
        Check_atomlocation(pSPARC);
        Calculate_electronicGroundState(pSPARC);
        pSPARC->elecgs_Count++;

        for(i = 0; i < n; i++){
            acc[i] = pSPARC->forces[i]/mass;
            pSPARC->FIRE_vel[i] += 0.5 * acc[i] * pSPARC->FIRE_dtNow;
        }

        // Check Power
        Pw = dotproduct(n, pSPARC->forces, 0, pSPARC->FIRE_vel, 0);
        if(Pw < 0.0){
            for(i = 0; i < n; i++)
                pSPARC->FIRE_vel[i] = 0.0; // Reset velocity
            pSPARC->FIRE_resetIter = iter;
            pSPARC->FIRE_dtNow *= fDec; // decrease dt
            pSPARC->FIRE_alpha = alphaStart; // reset alpha
        } else if(Pw >= 0.0 && (iter - pSPARC->FIRE_resetIter) > nMIN){
            pSPARC->FIRE_dtNow = min(pSPARC->FIRE_dtNow * fInc, dtMax); // update dt
            pSPARC->FIRE_alpha *= fAlpha; // update alpha
        }

        fnorm = norm(n, pSPARC->forces);
        vnorm = norm(n, pSPARC->FIRE_vel);
        for(i = 0; i < n; i++){
            pSPARC->FIRE_vel[i] = (1.0 - pSPARC->FIRE_alpha) * pSPARC->FIRE_vel[i] + (pSPARC->FIRE_alpha * vnorm) * (pSPARC->forces[i]/fnorm); // modified velocity
        }

        // Compute error
        err = 0.0;
        for(i = 0; i < n; i++){
            if (fabs(pSPARC->forces[i]) > err)
                err = fabs(pSPARC->forces[i]); // defined as supremum norm of force vector
        }
        // Print stuff
        if(check){
            Print_fullRelax(pSPARC, output_relax); // prints the QOI in the output_relax file
            fclose(output_relax);
        }
        if(check1 && !(iter % pSPARC->Printrestart_fq)) // printrestart_fq is the frequency at which the restart file is written
            PrintRelax(pSPARC);
        if(access("SPARC.stop", F_OK ) != -1 ){ // If a .stop file exists in the folder then the run will be terminated
            pSPARC->RelaxCount++;
            break;
        }

#ifdef DEBUG
        if (!rank) printf("Time taken by RelaxStep %d: %.3f s.\n", iter, (MPI_Wtime() - t_init));
#endif

        if(!rank){
            output_fp = fopen(pSPARC->OutFilename,"a");
            if (output_fp == NULL) {
                printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
                exit(EXIT_FAILURE);
            }
            fprintf(output_fp,"Relax step time                    :  %.3f (sec)\n", (MPI_Wtime() - t_init));
            fclose(output_fp);
        }

        pSPARC->RelaxCount++;
        iter++;
        t_acc += (MPI_Wtime() - t_init)/60.0;
    }

    if(check1){
        pSPARC->RelaxCount--;
        PrintRelax(pSPARC);
    }

    free(pSPARC->d);
    free(pSPARC->FIRE_vel);
    free(acc);

}



/**
 * @brief   Performs cell relaxation using Brent's algorithm.
 */
void Relax_Cell(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) {
        printf(GRN "Starting volume relaxation ...\n" RESET);
    }
#endif

    // relax cell in the periodic dims
    pSPARC->cellrelax_dims[0] = 1 - pSPARC->BCx;
    pSPARC->cellrelax_dims[1] = 1 - pSPARC->BCy;
    pSPARC->cellrelax_dims[2] = 1 - pSPARC->BCz;

    pSPARC->cellrelax_ndim = pSPARC->cellrelax_dims[0]
                           + pSPARC->cellrelax_dims[1]
                           + pSPARC->cellrelax_dims[2];

    double max_dilatation = pSPARC->max_dilatation;

    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    double V  = Lx * Ly * Lz * pSPARC->Jacbdet;
    double V_lwbd = V / pow(max_dilatation,pSPARC->cellrelax_ndim);
    double V_upbd = V * pow(max_dilatation,pSPARC->cellrelax_ndim);

    //BrentsFun(SPARC_OBJ *pSPARC, double x1, double x2, double tol_x, double tol_f, int max_iter)
    double optVol = BrentsFun(pSPARC, V_lwbd, V_upbd, 1e-4, pSPARC->TOL_RELAX_CELL, pSPARC->Relax_Niter);
#ifdef DEBUG
    if (!rank) printf(RED "optVol  = %.15f\n" RESET, optVol);
    if (!rank) printf(RED "optCELL = %18.15f %18.15f %18.15f\n" RESET,
                        pSPARC->range_x, pSPARC->range_y, pSPARC->range_z);
#endif
    (void) optVol; // suppress unused-variable warning
}



/**
 * @brief Function for Brent's method to find optimal volume.
 *
 *        This function takes in a volume, and finds the maximum
 *        principle stress of the system. sigma_max = f(vol).
 *        The root of this function gives the optimal volume.
 **/
double volrelax_constraint(SPARC_OBJ *pSPARC, double vol)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    reinitialize_cell_mesh(pSPARC, vol);
    
    if (pSPARC->RelaxFlag == 2) {
        // turn stress calculation on
        pSPARC->Calc_stress = 1;
        Calculate_electronicGroundState(pSPARC);
        pSPARC->elecgs_Count++;
        pSPARC->RelaxCount++;
        
        // print the result in .geopt file
        if (pSPARC->PrintRelaxout == 1) {
            if (rank == 0) PrintCellRelax(pSPARC);
        }
    } else if (pSPARC->RelaxFlag == 3) { // Perform atomic relaxation before for this cell for full relaxation
        pSPARC->RelaxFlag = 1;
        pSPARC->Calc_stress = 1;
        pSPARC->StressCount = pSPARC->elecgs_Count;
        main_Relax(pSPARC);

        if(pSPARC->cell_typ != 0){
            for(int atm = 0; atm < pSPARC->n_atom; atm++){
                Cart2nonCart_coord(pSPARC, &pSPARC->atom_pos[3*atm], &pSPARC->atom_pos[3*atm+1], &pSPARC->atom_pos[3*atm+2]);
            }
        }
        
        pSPARC->RelaxFlag = 3;
    }
    

    // Convert stress from cartesian to lattice coordinates
    double *stress_lc = (double *) calloc(6, sizeof(double));
    double *stress_tmp;
    int i, k, l;

    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        stress_tmp = (double *) malloc(9 * sizeof(double));
        stress_tmp[0] = pSPARC->stress[0]; stress_tmp[1] = pSPARC->stress[1]; stress_tmp[2] = pSPARC->stress[2];
        stress_tmp[3] = pSPARC->stress[1]; stress_tmp[4] = pSPARC->stress[3]; stress_tmp[5] = pSPARC->stress[4];
        stress_tmp[6] = pSPARC->stress[2]; stress_tmp[7] = pSPARC->stress[4]; stress_tmp[8] = pSPARC->stress[5];
       for(k = 0; k < 3; k++){
            for(l = 0; l < 3; l++){
                stress_lc[0] += pSPARC->LatUVec[k] * stress_tmp[3*k+l] * pSPARC->gradT[l];
                stress_lc[1] += pSPARC->LatUVec[k] * stress_tmp[3*k+l] * pSPARC->gradT[3+l];
                stress_lc[2] += pSPARC->LatUVec[k] * stress_tmp[3*k+l] * pSPARC->gradT[6+l];
                stress_lc[3] += pSPARC->LatUVec[3+k] * stress_tmp[3*k+l] * pSPARC->gradT[3+l];
                stress_lc[4] += pSPARC->LatUVec[3+k] * stress_tmp[3*k+l] * pSPARC->gradT[6+l];
                stress_lc[5] += pSPARC->LatUVec[6+k] * stress_tmp[3*k+l] * pSPARC->gradT[6+l];
            }
       }
       free(stress_tmp);
    } else{
        for(i = 0; i < 6; i++){
            stress_lc[i] = pSPARC->stress[i];
        }
    }



    double max_P_stress = 0.0;

    // only root process has stress value
    if (rank == 0) {
        double stress_diag[3];
        if(pSPARC->BC == 2) {
            stress_diag[0] = stress_lc[0] * CONST_HA_BOHR3_GPA;
            stress_diag[1] = stress_lc[3] * CONST_HA_BOHR3_GPA;
            stress_diag[2] = stress_lc[5] * CONST_HA_BOHR3_GPA;
        } else {
            stress_diag[0] = stress_lc[0];
            stress_diag[1] = stress_lc[3];
            stress_diag[2] = stress_lc[5];
        }

        if (pSPARC->cellrelax_dims[0] == 1) max_P_stress += stress_diag[0];
        if (pSPARC->cellrelax_dims[1] == 1) max_P_stress += stress_diag[1];
        if (pSPARC->cellrelax_dims[2] == 1) max_P_stress += stress_diag[2];

        max_P_stress /= pSPARC->cellrelax_ndim;
    }

    // broadcast the max_P_stress value
    MPI_Bcast(&max_P_stress, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // free memory
    free(stress_lc);

    // return max pressure stress
    return max_P_stress;
}



/**
 * @brief Reinitialize all parameters after updating vol and mesh size.
 *
 *        Once volume is updated, the cell dimensions and mesh sizes will
 *        be updated. All variables related to cell lengths and mesh size
 *        have to be updated.
 *
 * @param pSPARC  Pointer to the sturcture whose fields will be updated.
 * @param vol     New volume.
 **/
void reinitialize_cell_mesh(SPARC_OBJ *pSPARC, double vol)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


#ifdef DEBUG
        double t1, t2;
#endif

    int p;

    if (vol <= 0.0) {
        if (rank == 0) {
            printf("ERROR: new volume during relaxation is invalid: %.3f\n", vol);
            exit(EXIT_FAILURE);
        }
    }

    // store previous volume
    double vol_old = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;

    // scaling factor
    double scal = pow(vol / vol_old, 1.0/pSPARC->cellrelax_ndim);

    // calculate new length corresponding to volume vol
    if (pSPARC->cellrelax_dims[0] == 1) pSPARC->range_x *= scal;
    if (pSPARC->cellrelax_dims[1] == 1) pSPARC->range_y *= scal;
    if (pSPARC->cellrelax_dims[2] == 1) pSPARC->range_z *= scal;

    // update mesh size
    if (pSPARC->cellrelax_dims[0] == 1) pSPARC->delta_x *= scal;
    if (pSPARC->cellrelax_dims[1] == 1) pSPARC->delta_y *= scal;
    if (pSPARC->cellrelax_dims[2] == 1) pSPARC->delta_z *= scal;

    pSPARC->dV = pSPARC->delta_x * pSPARC->delta_y * pSPARC->delta_z * pSPARC->Jacbdet;

    // calculate atom positions
    int atm, count = 0;
    for(int ityp = 0; ityp < pSPARC->Ntypes; ityp++){
        for(atm = 0; atm < pSPARC->nAtomv[ityp]; atm++) {
            if (pSPARC->cellrelax_dims[0] == 1) pSPARC->atom_pos[3*count  ] *= scal;
            if (pSPARC->cellrelax_dims[1] == 1) pSPARC->atom_pos[3*count+1] *= scal;
            if (pSPARC->cellrelax_dims[2] == 1) pSPARC->atom_pos[3*count+2] *= scal;
            count++;
        }
    }

#ifdef DEBUG
    if (!rank) {
        printf("Volume: %12.6f\n", vol);
        printf("CELL  : %12.6f\t%12.6f\t%12.6f\n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
        printf("COORD : \n");
        for (int i = 0; i < 3 * pSPARC->n_atom; i++) {
            printf("%12.6f\t",pSPARC->atom_pos[i]);
            if (i%3==2 && i>0) printf("\n");
        }
        printf("\n");
    }
#endif



    int FDn = pSPARC->order / 2;

    // 1st derivative weights including mesh
    double dx_inv, dy_inv, dz_inv;
    dx_inv = 1.0 / pSPARC->delta_x;
    dy_inv = 1.0 / pSPARC->delta_y;
    dz_inv = 1.0 / pSPARC->delta_z;
    for (p = 1; p < FDn + 1; p++) {
        pSPARC->D1_stencil_coeffs_x[p] = pSPARC->FDweights_D1[p] * dx_inv;
        pSPARC->D1_stencil_coeffs_y[p] = pSPARC->FDweights_D1[p] * dy_inv;
        pSPARC->D1_stencil_coeffs_z[p] = pSPARC->FDweights_D1[p] * dz_inv;
    }

    // 2nd derivative weights including mesh
    double dx2_inv, dy2_inv, dz2_inv;
    dx2_inv = 1.0 / (pSPARC->delta_x * pSPARC->delta_x);
    dy2_inv = 1.0 / (pSPARC->delta_y * pSPARC->delta_y);
    dz2_inv = 1.0 / (pSPARC->delta_z * pSPARC->delta_z);

    // Stencil coefficients for mixed derivatives
    if (pSPARC->cell_typ == 0) {
        for (p = 0; p < FDn + 1; p++) {
            pSPARC->D2_stencil_coeffs_x[p] = pSPARC->FDweights_D2[p] * dx2_inv;
            pSPARC->D2_stencil_coeffs_y[p] = pSPARC->FDweights_D2[p] * dy2_inv;
            pSPARC->D2_stencil_coeffs_z[p] = pSPARC->FDweights_D2[p] * dz2_inv;
        }
    } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
        for (p = 0; p < FDn + 1; p++) {
            pSPARC->D2_stencil_coeffs_x[p] = pSPARC->lapcT[0] * pSPARC->FDweights_D2[p] * dx2_inv;
            pSPARC->D2_stencil_coeffs_y[p] = pSPARC->lapcT[4] * pSPARC->FDweights_D2[p] * dy2_inv;
            pSPARC->D2_stencil_coeffs_z[p] = pSPARC->lapcT[8] * pSPARC->FDweights_D2[p] * dz2_inv;
            pSPARC->D2_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_12 d/dx(df/dy)
            pSPARC->D2_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // 2*T_13 d/dx(df/dz)
            pSPARC->D2_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // 2*T_23 d/dy(df/dz)
            pSPARC->D1_stencil_coeffs_xy[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dy_inv; // d/dx(2*T_12 df/dy) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
            pSPARC->D1_stencil_coeffs_yx[p] = 2 * pSPARC->lapcT[1] * pSPARC->FDweights_D1[p] * dx_inv; // d/dy(2*T_12 df/dx) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
            pSPARC->D1_stencil_coeffs_xz[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dz_inv; // d/dx(2*T_13 df/dz) used in d/dx(2*T_12 df/dy + 2*T_13 df/dz)
            pSPARC->D1_stencil_coeffs_zx[p] = 2 * pSPARC->lapcT[2] * pSPARC->FDweights_D1[p] * dx_inv; // d/dz(2*T_13 df/dx) used in d/dz(2*T_13 df/dz + 2*T_23 df/dy)
            pSPARC->D1_stencil_coeffs_yz[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dz_inv; // d/dy(2*T_23 df/dz) used in d/dy(2*T_12 df/dx + 2*T_23 df/dz)
            pSPARC->D1_stencil_coeffs_zy[p] = 2 * pSPARC->lapcT[5] * pSPARC->FDweights_D1[p] * dy_inv; // d/dz(2*T_23 df/dy) used in d/dz(2*T_12 df/dx + 2*T_23 df/dy)
        }
    }

    // maximum eigenvalue of -0.5 * Lap (only with periodic boundary conditions)
    if(pSPARC->cell_typ == 0) {
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        pSPARC->MaxEigVal_mhalfLap = pSPARC->D2_stencil_coeffs_x[0]
                                   + pSPARC->D2_stencil_coeffs_y[0]
                                   + pSPARC->D2_stencil_coeffs_z[0];
        double scal_x, scal_y, scal_z;
        scal_x = (pSPARC->Nx - pSPARC->Nx % 2) / (double) pSPARC->Nx;
        scal_y = (pSPARC->Ny - pSPARC->Ny % 2) / (double) pSPARC->Ny;
        scal_z = (pSPARC->Nz - pSPARC->Nz % 2) / (double) pSPARC->Nz;
        for (int p = 1; p < FDn + 1; p++) {
            pSPARC->MaxEigVal_mhalfLap += 2.0 * (pSPARC->D2_stencil_coeffs_x[p] * cos(M_PI*p*scal_x)
                                               + pSPARC->D2_stencil_coeffs_y[p] * cos(M_PI*p*scal_y)
                                               + pSPARC->D2_stencil_coeffs_z[p] * cos(M_PI*p*scal_z));
        }
        pSPARC->MaxEigVal_mhalfLap *= -0.5;
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("Max eigenvalue of -0.5*Lap is %.13f, time taken: %.3f ms\n",
            pSPARC->MaxEigVal_mhalfLap, (t2-t1)*1e3);
#endif
    }

    double h_eff = 0.0;
    if (fabs(pSPARC->delta_x - pSPARC->delta_y) < 1E-12 &&
        fabs(pSPARC->delta_y - pSPARC->delta_z) < 1E-12) {
        h_eff = pSPARC->delta_x;
    } else {
        // find effective mesh s.t. it has same spectral width
        h_eff = sqrt(3.0 / (dx2_inv + dy2_inv + dz2_inv));
    }

    // find Chebyshev polynomial degree based on max eigenvalue (spectral width)
    if (pSPARC->ChebDegree < 0) {
        pSPARC->ChebDegree = Mesh2ChebDegree(h_eff);
#ifdef DEBUG
        if (!rank && h_eff < 0.1) {
            printf("#WARNING: for mesh less than 0.1, the default Chebyshev polynomial degree might not be enought!\n");
        }
        if (!rank) printf("h_eff = %.2f, npl = %d\n", h_eff,pSPARC->ChebDegree);
#endif
    } else {
#ifdef DEBUG
        if (!rank) printf("Chebyshev polynomial degree (provided by user): npl = %d\n",pSPARC->ChebDegree);
#endif
    }

    // default Kerker tolerance
    if (pSPARC->TOL_PRECOND < 0.0) { // kerker tol not provided by user
        pSPARC->TOL_PRECOND = (h_eff * h_eff) * 1e-3;
    }


    // re-calculate k-point grid
    Calculate_kpoints(pSPARC);

    // re-calculate local k-points array
    if (pSPARC->Nkpts >= 1 && pSPARC->kptcomm_index != -1) {
        Calculate_local_kpoints(pSPARC);
    }

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // re-calculate pseudocharge density cutoff ("rb")
    Calculate_PseudochargeCutoff(pSPARC);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Calculating rb for all atom types took %.3f ms\n",(t2-t1)*1000);
#endif


    // write reinitialized parameters into output file
    if (rank == 0) {
        write_output_reinit(pSPARC);
    }

}




/**
 * @brief   Write the re-initialized parameters into the output file.
 */
void write_output_reinit(SPARC_OBJ *pSPARC) {
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    FILE *output_fp = fopen(pSPARC->OutFilename,"a");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
        exit(EXIT_FAILURE);
    }
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                         Reinitialized parameters                          \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"CELL: %.15g %.15g %.15g \n",pSPARC->range_x,pSPARC->range_y,pSPARC->range_z);
    fprintf(output_fp,"CHEB_DEGREE: %d\n",pSPARC->ChebDegree);
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                             Reinitialization                              \n");
    fprintf(output_fp,"***************************************************************************\n");
    if ( (fabs(pSPARC->delta_x-pSPARC->delta_y) <=1e-12) && (fabs(pSPARC->delta_x-pSPARC->delta_z) <=1e-12)
        && (fabs(pSPARC->delta_y-pSPARC->delta_z) <=1e-12) ) {
        fprintf(output_fp,"Mesh spacing                       :  %.6g (Bohr)\n",pSPARC->delta_x);
    } else {
        fprintf(output_fp,"Mesh spacing in x-direction        :  %.6g (Bohr)\n",pSPARC->delta_x);
        fprintf(output_fp,"Mesh spacing in y-direction        :  %.6g (Bohr)\n",pSPARC->delta_y);
        fprintf(output_fp,"Mesh spacing in z direction        :  %.6g (Bohr)\n",pSPARC->delta_z);
    }

    fclose(output_fp);
}



/**
 * @brief   Write the re-initialized parameters into the output file.
 */
void PrintCellRelax(SPARC_OBJ *pSPARC) {
    FILE *output_fp = fopen(pSPARC->RelaxFilename,"a");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",pSPARC->RelaxFilename);
        exit(EXIT_FAILURE);
    }
    
    fprintf(output_fp,":RELAXSTEP: %d\n", pSPARC->RelaxCount);
    fprintf(output_fp,":CELL: %18.10E %18.10E %18.10E\n", pSPARC->range_x, pSPARC->range_y, pSPARC->range_z);
    fprintf(output_fp,":VOLUME: %18.10E\n", pSPARC->range_x*pSPARC->range_y*pSPARC->range_z*pSPARC->Jacbdet);
    fprintf(output_fp,":LATVEC:\n");
    fprintf(output_fp,"%18.10E %18.10E %18.10E \n",pSPARC->LatVec[0],pSPARC->LatVec[1],pSPARC->LatVec[2]);
    fprintf(output_fp,"%18.10E %18.10E %18.10E \n",pSPARC->LatVec[3],pSPARC->LatVec[4],pSPARC->LatVec[5]);
    fprintf(output_fp,"%18.10E %18.10E %18.10E \n",pSPARC->LatVec[6],pSPARC->LatVec[7],pSPARC->LatVec[8]);
    fprintf(output_fp,":STRESS:\n");

    PrintStress(pSPARC, pSPARC->stress, output_fp);

    fclose(output_fp);
}



/**
 * @brief   Find root of a function using Brent's method.
 *
 * @ref     W.H. Press, Numerical recepies 3rd edition: The art of scientific
 *          computing, Cambridge university press, 2007.
 */
double BrentsFun(SPARC_OBJ *pSPARC, double x1, double x2, double tol_x, double tol_f, int max_iter)
{
#define EPSILON 1e-16
//#define SIGN(a,b) ((b) > 0.0 ? fabs((a)) : -fabs((a)))
    // if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return 0.0;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // declare function f, whose root will be found using Brent's method
    double volrelax_constraint(SPARC_OBJ *pSPARC, double vol);

    int iter;
    double tol1q, eq;
    double a = x1, b = x2, c = b, d, e = 0.0, min1, min2;
    double fa = volrelax_constraint(pSPARC,a), fb = volrelax_constraint(pSPARC,b), fc, p, q, r, s, tol1, xm;

    // if the given upperbound and lowerbound are not good enough, expand the bounds automatically
    int ext_count = 0;
    while (fa * fb > 0.0 && ext_count++ < 3) {   
        double w = b - a;
        a -= w/2.0;
        b += w/2.0;
        c = b;
        fa = volrelax_constraint(pSPARC,a);
        fb = volrelax_constraint(pSPARC,b);
#ifdef DEBUG
        if (rank == 0) 
            printf("Volume relaxation: expanded bounds = (%10.6E,%10.6E), (fa,fb) = (%10.6E,%10.6E)\n", 
                a,b,fa,fb);
#endif
    }

    if (fa * fb > 0.0) {   
        if (rank == 0) printf("Cannot find root in Brent's method for volume relaxation!\n" 
                              "  original bounds (%f,%f)\n"
                              "  expanded to     (%f,%f)", x1,x2,a,b);
        exit(EXIT_FAILURE);
    }

    fc = fb;
    for (iter = 1; iter <= max_iter; iter++) {
        if((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a; fc = fa;
            e = d = b-a;
        }
        if ( fabs(fc) < fabs(fb) ) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        tol1 = 2.0 * EPSILON * fabs(b) + 0.5 * tol_x;
        xm = 0.5*(c-b);
        if(fabs(xm) <= tol1 || fabs(fb) < tol_f) return b;
        if(fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
            // attempt inverse quadratic interpolation
            s = fb / fa;
            if( a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc; r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if(p > 0.0) {
                //check whether in bounds
                q = -q;
            }
            p = fabs(p);
            tol1q = tol1 * q;
            min1 = 3.0 * xm * q - fabs(tol1q);
            eq = e * q;
            min2 = fabs(eq);
            if(2.0 * p < (min1 < min2 ? min1 : min2)) {
                //accept interpolation
                e = d; d = p / q;
            } else {
                // Bounds decreasing too slowly, use bisection
                d = xm; e = d;
            }
        } else {
            d = xm; e = d;
        }
        // move last best guess to a
        a = b; fa = fb;

        if (fabs(d) > tol1) {
            // evaluate new trial root
            b += d;
        } else {
            b += SIGN(tol1, xm);
        }
        fb = volrelax_constraint(pSPARC,b);
    }
    printf("WARNING: Maximum iterations exceeded in brents root finding method for volume relaxation\n");
    return b;
#undef EPSILON
//#undef SIGN
}



/**
@ brief: function to write the output of the relaxation
**/
void Print_fullRelax(SPARC_OBJ *pSPARC, FILE *output_relax) {
    int atm;
    fprintf(output_relax,":E(Ha): %.15E\n", pSPARC->Etot);
    // Print atomic position
    fprintf(output_relax,":R(Bohr):\n");
    for(atm = 0; atm < pSPARC->n_atom; atm++){
        fprintf(output_relax,"%20.15f %20.15f %20.15f\n", pSPARC->atom_pos[3 * atm], pSPARC->atom_pos[3 * atm + 1], pSPARC->atom_pos[3 * atm + 2]);
    }
    // Print Forces
    fprintf(output_relax,":F(Ha/Bohr):\n");
    for(atm = 0; atm < pSPARC->n_atom; atm++){
        fprintf(output_relax,"%20.15f %20.15f %20.15f\n", pSPARC->forces[3 * atm], pSPARC->forces[3 * atm + 1], pSPARC->forces[3 * atm + 2]);
    }
    
    if(pSPARC->Calc_stress == 1) {
        fprintf(output_relax,":CELL: %18.10E %18.10E %18.10E\n", pSPARC->range_x, pSPARC->range_y, pSPARC->range_z);
        fprintf(output_relax,":VOLUME: %18.10E\n", pSPARC->range_x*pSPARC->range_y*pSPARC->range_z*pSPARC->Jacbdet);
        fprintf(output_relax,":LATVEC:\n");
        fprintf(output_relax,"%18.10E %18.10E %18.10E \n",pSPARC->LatVec[0],pSPARC->LatVec[1],pSPARC->LatVec[2]);
        fprintf(output_relax,"%18.10E %18.10E %18.10E \n",pSPARC->LatVec[3],pSPARC->LatVec[4],pSPARC->LatVec[5]);
        fprintf(output_relax,"%18.10E %18.10E %18.10E \n",pSPARC->LatVec[6],pSPARC->LatVec[7],pSPARC->LatVec[8]);
        fprintf(output_relax,":STRESS:\n");
        PrintStress(pSPARC, pSPARC->stress, output_relax);
    }
}

/**
@ brief: function to write the restart file for relaxation
**/
void PrintRelax(SPARC_OBJ *pSPARC) {
    FILE *relaxout;
    relaxout = fopen(pSPARC->restart_Filename,"w");
    if(relaxout == NULL){
        printf("\nCannot open file \"%s\"\n",pSPARC->restart_Filename);
        exit(EXIT_FAILURE);
    }
    // Print relax Count
    fprintf(relaxout,":RELAXSTEP: %d\n", pSPARC->RelaxCount + pSPARC->restartCount + (pSPARC->RestartFlag == 0));
    fprintf(relaxout,":E(Ha): %.15E\n", pSPARC->Etot);
    // Print atomic position
    int atm;
    fprintf(relaxout,":R(Bohr):\n");
    for(atm = 0; atm < pSPARC->n_atom; atm++){
        fprintf(relaxout,"%20.15f %20.15f %20.15f\n", pSPARC->atom_pos[3 * atm], pSPARC->atom_pos[3 * atm + 1], pSPARC->atom_pos[3 * atm + 2]);
    }
    
    if (strcmpi(pSPARC->RelaxMeth,"NLCG") == 0){
        // Print search direction
        fprintf(relaxout,":D:\n");
        for(atm = 0; atm < pSPARC->n_atom; atm++){
            fprintf(relaxout,"%20.15f %20.15f %20.15f\n", pSPARC->d[3 * atm], pSPARC->d[3 * atm + 1], pSPARC->d[3 * atm + 2]);
        }
    } else if(strcmpi(pSPARC->RelaxMeth,"LBFGS") == 0){
        fprintf(relaxout,":ISFD: %d\n", pSPARC->isFD);
        fprintf(relaxout,":ISRESET: %d\n", pSPARC->isReset);
        fprintf(relaxout,":STEP: %d\n", pSPARC->step);
        int n = 3 * pSPARC->n_atom;
        fprintf(relaxout,":DX:\n");
        for(atm = 0; atm < pSPARC->L_history * n; atm++){
            fprintf(relaxout,"%20.15f\n", pSPARC->deltaX[atm]);
        }
        fprintf(relaxout,":DG:\n");
        for(atm = 0; atm < pSPARC->L_history * n; atm++){
            fprintf(relaxout,"%20.15f\n", pSPARC->deltaG[atm]);
        }
        fprintf(relaxout,":IYS:\n");
        for(atm = 0; atm < pSPARC->L_history; atm++){
            fprintf(relaxout,"%20.15f\n", pSPARC->iys[atm]);
        }
        fprintf(relaxout,":FOLD:\n");
        for(atm = 0; atm < n; atm++){
            fprintf(relaxout,"%20.15f\n", pSPARC->fold[atm]);
        }
        fprintf(relaxout,":RDISP:\n");
        for(atm = 0; atm < n; atm++){
            fprintf(relaxout,"%20.15f\n", pSPARC->atom_disp[atm]);
        }
    } else if(strcmpi(pSPARC->RelaxMeth,"FIRE") == 0) {
        fprintf(relaxout,":FIRE_alpha: %20.15f\n", pSPARC->FIRE_alpha);
        fprintf(relaxout,":FIRE_dtNow: %20.15f\n", pSPARC->FIRE_dtNow);
        fprintf(relaxout,":FIRE_resetIter: %d\n", pSPARC->FIRE_resetIter);
        int n = 3 * pSPARC->n_atom;
        fprintf(relaxout,":FIRE_V:\n");
        for(atm = 0; atm < n; atm++){
            fprintf(relaxout,"%20.15f\n", pSPARC->FIRE_vel[atm]);
        }
    }
    fclose(relaxout);
}

/**
@ brief: function to restart the relaxation
**/
void RestartRelax(SPARC_OBJ *pSPARC) {
    int rank, position, l_buff = 0;
#ifdef DEBUG
    double t1, t2;
#endif
    char *buff;
    FILE *rst_fp = NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Open the restart file
    if(!rank){
        char rst_Filename[L_STRING];
        strncpy(rst_Filename, pSPARC->restart_Filename, sizeof(rst_Filename));
        rst_fp = fopen(rst_Filename,"r");
        if (rst_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",rst_Filename);
            print_usage();
            exit(EXIT_FAILURE);
        }
    }
#ifdef DEBUG
    if(!rank)
        printf("Reading .restart file for relaxation\n");
#endif
    // Allocate memory for dynamic variables
    if (strcmpi(pSPARC->RelaxMeth,"NLCG") == 0){
        pSPARC->d = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
        if (pSPARC->d == NULL) {
            printf("\nCannot allocate memory for search direction array in NLCG!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Allocate memory for Pack and Unpack to be used later for broadcasting
    int n = 3 * pSPARC->n_atom;
    if(strcmpi(pSPARC->RelaxMeth,"NLCG") == 0)
        l_buff = 1 * sizeof(int) + (2 * n) * sizeof(double);
    else if(strcmpi(pSPARC->RelaxMeth,"LBFGS") == 0)
        l_buff = 4 * sizeof(int) + ((3 + 2 * pSPARC->L_history) * n + pSPARC->L_history) * sizeof(double);
    else if(strcmpi(pSPARC->RelaxMeth,"FIRE") == 0)
        l_buff = 2 * sizeof(int) + ((2 + 2 * n )) * sizeof(double);
    buff = (char *)malloc( l_buff*sizeof(char) );
    if (buff == NULL) {
        printf("\nmemory cannot be allocated for buffer\n");
        exit(EXIT_FAILURE);
    }
    if(!rank){
        char str[L_STRING];
        int atm;
        while (fscanf(rst_fp,"%s",str) != EOF){
            if (strcmpi(str, ":RELAXSTEP:") == 0)
                fscanf(rst_fp,"%d",&pSPARC->restartCount);
            else if (strcmpi(str,":R(Bohr):") == 0)
                for(atm = 0; atm < pSPARC->n_atom; atm++)
                    fscanf(rst_fp,"%lf %lf %lf", &pSPARC->atom_pos[3 * atm], &pSPARC->atom_pos[3 * atm + 1], &pSPARC->atom_pos[3 * atm + 2]);
            else if (strcmpi(str,":D:") == 0)
                for(atm = 0; atm < pSPARC->n_atom; atm++)
                    fscanf(rst_fp,"%lf %lf %lf", &pSPARC->d[3 * atm], &pSPARC->d[3 * atm + 1], &pSPARC->d[3 * atm + 2]);
            else if (strcmpi(str,":ISFD:") == 0)
                fscanf(rst_fp,"%d", &pSPARC->isFD);
            else if (strcmpi(str,":ISRESET:") == 0)
                fscanf(rst_fp,"%d", &pSPARC->isReset);
            else if (strcmpi(str,":STEP:") == 0)
                fscanf(rst_fp,"%d", &pSPARC->step);
            else if (strcmpi(str,":DX:") == 0)
                for(atm = 0; atm < pSPARC->L_history * n; atm++)
                    fscanf(rst_fp,"%lf", &pSPARC->deltaX[atm]);
            else if (strcmpi(str,":DG:") == 0)
                for(atm = 0; atm < pSPARC->L_history * n; atm++)
                    fscanf(rst_fp,"%lf", &pSPARC->deltaG[atm]);
            else if (strcmpi(str,":IYS:") == 0)
                for(atm = 0; atm < pSPARC->L_history; atm++)
                    fscanf(rst_fp,"%lf", &pSPARC->iys[atm]);
            else if (strcmpi(str,":FOLD:") == 0)
                for(atm = 0; atm < n; atm++)
                    fscanf(rst_fp,"%lf", &pSPARC->fold[atm]);
            else if (strcmpi(str,":RDISP:") == 0)
                for(atm = 0; atm < n; atm++)
                    fscanf(rst_fp,"%lf", &pSPARC->atom_disp[atm]);
            else if (strcmpi(str,":FIRE_alpha:") == 0)
                fscanf(rst_fp,"%lf", &pSPARC->FIRE_alpha);
            else if (strcmpi(str,":FIRE_dtNow:") == 0)
                fscanf(rst_fp,"%lf", &pSPARC->FIRE_dtNow);
            else if (strcmpi(str,":FIRE_resetIter:") == 0)
                fscanf(rst_fp,"%d", &pSPARC->FIRE_resetIter);
            else if (strcmpi(str,":FIRE_V:") == 0)
                for(atm = 0; atm < n; atm++)
                    fscanf(rst_fp,"%lf", &pSPARC->FIRE_vel[atm]);

        }
        fclose(rst_fp);
        // Pack the variables
        position = 0;
        MPI_Pack(&pSPARC->restartCount, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
        MPI_Pack(pSPARC->atom_pos, n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        if(strcmpi(pSPARC->RelaxMeth,"NLCG") == 0){
            MPI_Pack(pSPARC->d, n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        } else if(strcmpi(pSPARC->RelaxMeth,"LBFGS") == 0){
            MPI_Pack(&pSPARC->isFD, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->isReset, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->step, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->deltaX, pSPARC->L_history * n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->deltaG, pSPARC->L_history * n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->iys, pSPARC->L_history, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->fold, n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->atom_disp, n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        } else if(strcmpi(pSPARC->RelaxMeth,"FIRE") == 0){
            MPI_Pack(&pSPARC->FIRE_alpha, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->FIRE_dtNow, 1, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(&pSPARC->FIRE_resetIter, 1, MPI_INT, buff, l_buff, &position, MPI_COMM_WORLD);
            MPI_Pack(pSPARC->FIRE_vel, n, MPI_DOUBLE, buff, l_buff, &position, MPI_COMM_WORLD);
        }
        // broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
    } else{
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        // broadcast the packed buffer
        MPI_Bcast(buff, l_buff, MPI_PACKED, 0, MPI_COMM_WORLD);
#ifdef DEBUG
        t2 = MPI_Wtime();
        if (rank == 0) printf(GRN "MPI_Bcast (.restart relax) packed buff of length %d took %.3f ms\n" RESET, l_buff,(t2-t1)*1000);
#endif
        // unpack the variables
        position = 0;
        MPI_Unpack(buff, l_buff, &position, &pSPARC->restartCount, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buff, l_buff, &position, pSPARC->atom_pos, n, MPI_DOUBLE, MPI_COMM_WORLD);
        if(strcmpi(pSPARC->RelaxMeth,"NLCG") == 0){
            MPI_Unpack(buff, l_buff, &position, pSPARC->d, n, MPI_DOUBLE, MPI_COMM_WORLD);
        } else if(strcmpi(pSPARC->RelaxMeth,"LBFGS") == 0){
            MPI_Unpack(buff, l_buff, &position, &pSPARC->isFD, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->isReset, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->step, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->deltaX, pSPARC->L_history * n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->deltaG, pSPARC->L_history * n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->iys, pSPARC->L_history, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->fold, n, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->atom_disp, n, MPI_DOUBLE, MPI_COMM_WORLD);
        } else if(strcmpi(pSPARC->RelaxMeth,"FIRE") == 0){
            MPI_Unpack(buff, l_buff, &position, &pSPARC->FIRE_alpha, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->FIRE_dtNow, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, &pSPARC->FIRE_resetIter, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buff, l_buff, &position, pSPARC->FIRE_vel, n, MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
   free(buff);
}
