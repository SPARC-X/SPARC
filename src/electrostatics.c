/**
 * @file    electrostatics.c
 * @brief   This file contains functions for electrostatics.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "electrostatics.h"
#include "tools.h"
#include "linearSolver.h"
#include "lapVecRoutines.h"
#include "nlocVecRoutines.h"
#include "gradVecRoutines.h"
#include "initialization.h"
#include "isddft.h"

#include <mpi.h>
// #include <cblas.h> 

#ifndef TEMP_TOL
#define TEMP_TOL 1e-12
#endif

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))



/**
 * @brief    Calculate pseudocharge density cutoff ("rb") using bisection.
 */
void Calculate_PseudochargeCutoff(SPARC_OBJ *pSPARC) {
// Info: Working in the four upper quadrants only (from symmetry) for nonorthogonal and 1 quadrant for orthogonal systems
#define b(i,j,k) b[(k)*(xlen)*(ylen)+(j)*(xlen)+(i)]
#define V_temp(i,j,k) V_temp[(k)*(xlen_ex)*(ylen_ex)+(j)*(xlen_ex)+(i)]
// TODO: rb_max depends on finite difference order as well, improve this function
#define RB_MAX(h) (h) < 1.5 ? (h)*10+5.5 : 20*(h)-9.5
#define RB_Y(y,L,typ) (typ) < 20 ? (y) : acos(1 - (y)*(y)/(2*(L)*(L))) + pSPARC->twist*y
//#define RB_MAX(h) (h) < 1.5 ? (h)*30+5.5 : 40*(h)-9.5 
    double Rbmax_x, Rbmax_y, Rbmax_z;
    int rank, nx, ny, nz, nxp, nyp, nzp, FDn, rlen_ex, rlen, i, j, k, ityp, xlen, ylen, zlen, xlen_ex, ylen_ex, zlen_ex;
    int count, Ncube_x, Ncube_y, Ncube_z, len_interp, count_interp;
    double w2_diag, dx, dy, dz, x, y, z, r;
    double *R, *V_temp, *b, *temp3, Bint, val, rchrg; 
    double rb_cur_x, rb_cur_y, rb_cur_z, rb_prev_x, rb_prev_y, rb_prev_z, error_cur, error_prev;
    double *Lap_wt, *Lap_stencil;
    double pos_atm_x, pos_atm_y, pos_atm_z, xin;
#ifdef DEBUG
    double t2, t3, t31, t4, t5;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    dx = pSPARC->delta_x;
    dy = pSPARC->delta_y;
    dz = pSPARC->delta_z;
    
    // if Rbmax too small, change the function RB_MAX() or override Rbmax later!
    pos_atm_x = pos_atm_y = pos_atm_z = 0.0;
    Rbmax_x = RB_MAX(dx);
    Rbmax_y = RB_MAX(dy);
    Rbmax_z = RB_MAX(dz);
    
    /******************************************************
     *  If Rbmax is not big enough, override Rbmax below  *
     ******************************************************/
    // Rbmax_x = 15;
    // Rbmax_y = 15;
    // Rbmax_z = 15;

#ifdef DEBUG
    if (rank == 0) printf("Rbmax_x = %f, Rbmax_y = %f, Rbmax_z = %f\n",Rbmax_x,Rbmax_y,Rbmax_z);
#endif

    nx = ceil(Rbmax_x / dx - TEMP_TOL);
    ny = ceil(Rbmax_y / dy - TEMP_TOL);
    nz = ceil(Rbmax_z / dz - TEMP_TOL);   
    FDn = pSPARC->order / 2;
    w2_diag = pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0] + pSPARC->D2_stencil_coeffs_z[0];
    
#ifdef DEBUG
    t31 = MPI_Wtime();
#endif

    rchrg = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        if (rchrg < pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1]) {
            rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        }
    }

    // find radius of the grids
    count = 0; count_interp = 0;
    if(pSPARC->cell_typ == 0) { 
        nxp = nx + 2 * FDn;
        nyp = ny + 2 * FDn;
        nzp = nz + 2 * FDn;
        rlen = nx * ny * nz;
        rlen_ex = nxp * nyp * nzp;
        xlen = nx;
        ylen = ny;
        zlen = nz; 
        xlen_ex = nxp;
        ylen_ex = nyp;
        zlen_ex = nzp;
        R = (double *)malloc(sizeof(double) * rlen_ex);
        for (k = 0; k < nzp; k++) {
            z = (k - FDn) * dz;
            z *= z;
            for (j = 0; j < nyp; j++) {
                y = (j - FDn) * dy;
                y *= y;
                for (i = 0; i < nxp; i++) {
                    x = (i - FDn) * dx;
                    x *= x;
                    r = sqrt(x + y + z);
                    R[count] = r;
                    if (r <= rchrg) count_interp++;
                    count++;
                }
            }
        }
        Lap_wt = (double *)malloc((3*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt;
    } else {
        nxp = nx + FDn;
        nyp = ny + FDn;
        nzp = nz + 2*FDn; 
        xlen = 2 * nx + 1;
        ylen = 2 * ny + 1;
        zlen = nz + 1;
        xlen_ex = 2 * nxp + 1;
        ylen_ex = 2 * nyp + 1;
        zlen_ex = nzp + 1; 
        rlen = xlen * ylen * zlen;
        rlen_ex = xlen_ex * ylen_ex * zlen_ex;
        R = (double *)malloc(sizeof(double) * rlen_ex);
        for (k = -FDn; k <= nzp - FDn; k++) { 
            z = k * dz; // length along lattice direction 3
            for (j = -nyp; j <= nyp; j++) {
                y = j * dy; // length along lattice direction 2
                for (i = -nxp; i <= nxp; i++) {
                    x = pos_atm_x + i * dx; // length along lattice direction 1
                    CalculateDistance(pSPARC, x, y, z, pos_atm_x, pos_atm_y, pos_atm_z, &r);
                    R[count] = r;
                    if (r <= rchrg) count_interp++;                                                         
                    count++;
                }
            }
        }
        Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt+5;
    }
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, 1.0);

#ifdef DEBUG    
    if (rank == 0) {
        printf("rlen_ex = %d, nxp = %d, nyp = %d, nzp = %d\n",
                rlen_ex,(nxp),(nyp),(nzp));    
    }
#endif

#ifdef DEBUG
    t3 = MPI_Wtime();
    if (rank == 0) printf("time spent on qsort: %.3f ms.\n", (t3-t31)*1000);       
#endif
    Ncube_x = Ncube_y = Ncube_z = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        #ifdef DEBUG
        //t1 = MPI_Wtime();
        t2 = MPI_Wtime();
        #endif
        temp3 = (double *)malloc( rlen_ex * sizeof(double) );
        if (temp3 == NULL) {
            printf("\nMemory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
        // apply cubic spline interpolation
        len_interp = rlen_ex;
		if (pSPARC->psd[ityp].is_r_uniform == 1) {
			SplineInterpUniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
			                    R, temp3, len_interp, pSPARC->psd[ityp].SplinerVlocD);
		} else {
			SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
			                    R, temp3, len_interp, pSPARC->psd[ityp].SplinerVlocD);
		}

        V_temp = (double *)malloc( rlen_ex * sizeof(double) );
        assert(V_temp != NULL);

        for (i = 0; i < len_interp; i++) {
            if (R[i] < TEMP_TOL) {
                V_temp[i] = pSPARC->psd[ityp].Vloc_0;
            } else if (R[i] > rchrg) {
            	V_temp[i] = -pSPARC->Znucl[ityp] / R[i];
            } else {
                V_temp[i] =  temp3[i]/R[i];
            }
        }
        
        free(temp3); 
        temp3 = NULL;

#ifdef DEBUG
        t4 = MPI_Wtime();
        if (rank == 0)
            printf("time spent on vectorized spline interp: %.3f ms.\n", (t4-t2)*1000);
#endif
        /* calculate pseudocharge density */
        b = (double *)malloc( rlen * sizeof(double) );
        if (b == NULL) {
            printf("\nMemory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
        
        xin = pos_atm_x - nx * dx;
        Calc_lapV(pSPARC, V_temp, FDn, xlen_ex, ylen_ex, zlen_ex, xlen, ylen, zlen, Lap_wt, w2_diag, xin, 1.0, b);
 
        free(V_temp); V_temp = NULL;
        
        /* go over different radii by BISECTION */
        double temp, lowerbd_x, lowerbd_y, lowerbd_z, upperbd_x, upperbd_y, upperbd_z, rtol_x, rtol_y, rtol_z, rc;
        int errflag = 0;
        //lloc = pSPARC->localPsd[ityp];
        rc = 0.0;
        for (i = 0; i <= pSPARC->psd[ityp].lmax; i++) {
            rc = max(rc,pSPARC->psd[ityp].rc[i]);
        }
        
        rb_cur_x = rb_cur_z = rc;
        rb_cur_y = RB_Y(rc,pos_atm_x, pSPARC->cell_typ);
        rb_prev_x = rb_prev_z = rb_cur_x;
        rb_prev_y = rb_cur_y;
        error_cur = 100;
        error_prev = 100;
        lowerbd_x = lowerbd_z = rc;
        lowerbd_y = RB_Y(rc,pos_atm_x, pSPARC->cell_typ);
        upperbd_x = Rbmax_x;
        upperbd_y = Rbmax_y;
        upperbd_z = Rbmax_z;
        rtol_x = 1.0 * dx;
        rtol_y = 1.0 * dy;
        rtol_z = 1.0 * dz;
        count = 0;
        while (count < 20) {
            if (error_cur < pSPARC->TOL_PSEUDOCHARGE) {
                if (error_prev >= pSPARC->TOL_PSEUDOCHARGE) {
                    if (fabs(rb_cur_x - rb_prev_x) < rtol_x 
                        && fabs(rb_cur_y - rb_prev_y) < rtol_y
                        && fabs(rb_cur_z - rb_prev_z) < rtol_z) {                  
                        break;
                    } else {
                        // update error
                        error_prev = error_cur;
                    }
                    temp = rb_cur_x;
                    // update upper bond
                    upperbd_x = min(upperbd_x, rb_cur_x);
                    if (fabs(rb_cur_x - rb_prev_x) >= rtol_x) {
                        // update rb
                        rb_cur_x = 0.5 * (rb_cur_x + rb_prev_x);
                    }
                    rb_prev_x = temp; 
                    temp = rb_cur_y;
                    // update upper bond
                    upperbd_y = min(upperbd_y, rb_cur_y);
                    if (fabs(rb_cur_y - rb_prev_y) >= rtol_y) {
                        // update rb
                        rb_cur_y = 0.5 * (rb_cur_y + rb_prev_y);     
                    }
                    rb_prev_y = temp; 
                    temp = rb_cur_z;
                    // update upper bond
                    upperbd_z = min(upperbd_z, rb_cur_z);
                    if (fabs(rb_cur_z - rb_prev_z) >= rtol_z) {
                        // update rb
                        rb_cur_z = 0.5 * (rb_cur_z + rb_prev_z);     
                    }
                    rb_prev_z = temp; 
                } else {
                    if (fabs(rb_cur_x - lowerbd_x) < rtol_x 
                        && fabs(rb_cur_y - lowerbd_y) < rtol_y
                        && fabs(rb_cur_z - lowerbd_z) < rtol_z) {                  
                        break;
                    } else {
                        // update error
                        error_prev = error_cur;
                    }
                    rb_prev_x = rb_cur_x;
                    // update upper bond
                    upperbd_x = min(upperbd_x, rb_cur_x);
                    if (fabs(rb_cur_x - lowerbd_x) >= rtol_x) { 
                        // update rb
                        rb_cur_x = 0.5 * (lowerbd_x + rb_cur_x);
                    }
                    rb_prev_y = rb_cur_y;
                    // update upper bond
                    upperbd_y = min(upperbd_y, rb_cur_y);	
                    if (fabs(rb_cur_y - lowerbd_y) >= rtol_y) { 
                        // update rb
                        rb_cur_y = 0.5 * (lowerbd_y + rb_cur_y);
                    }
                    rb_prev_z = rb_cur_z;
                    // update upper bond
                    upperbd_z = min(upperbd_z, rb_cur_z);
                    if (fabs(rb_cur_z - lowerbd_z) >= rtol_z) { 
                        // update rb
                        rb_cur_z = 0.5 * (lowerbd_z + rb_cur_z);
                    }
                }
            } else {
                if (error_prev >= pSPARC->TOL_PSEUDOCHARGE) {
                    if (fabs(upperbd_x - rb_cur_x) < rtol_x
                        && fabs(upperbd_y - rb_cur_y) < rtol_y
                        && fabs(upperbd_z - rb_cur_z) < rtol_z) {
                        rb_cur_x = upperbd_x;
                        rb_cur_y = upperbd_y;
                        rb_cur_z = upperbd_z;
                        errflag = 1;
                        break;
                    } else {
                        error_prev = error_cur;
                    }
                    rb_prev_x = rb_cur_x;
                    // update lower bound
                    lowerbd_x = max(lowerbd_x, rb_cur_x);
                    if (fabs(upperbd_x - rb_cur_x) >= rtol_x) { 
                        // update rb
                        rb_cur_x = 0.5 * (upperbd_x + rb_cur_x);
                    }
                    rb_prev_y = rb_cur_y;
                    // update lower bound
                    lowerbd_y = max(lowerbd_y, rb_cur_y);
                    if (fabs(upperbd_y - rb_cur_y) >= rtol_y) { 
                        // update rb
                        rb_cur_y = 0.5 * (upperbd_y + rb_cur_y);
                    }
                    rb_prev_z = rb_cur_z;
                    // update lower bound
                    lowerbd_z = max(lowerbd_z, rb_cur_z);
                    if (fabs(upperbd_z - rb_cur_z) >= rtol_z) { 
                        // update rb
                        rb_cur_z = 0.5 * (upperbd_z + rb_cur_z);
                    }
                } else {
                    if (fabs(rb_prev_x - rb_cur_x) < rtol_x 
                        && fabs(rb_prev_y - rb_cur_y) < rtol_y 
                        && fabs(rb_prev_z - rb_cur_z) < rtol_z) {
                        rb_cur_x = rb_prev_x;
                        rb_cur_y = rb_prev_y;
                        rb_cur_z = rb_prev_z;
                        error_cur = error_prev;
                        break;
                    } else {
                        error_prev = error_cur;
                    }
                    temp = rb_cur_x;
                    // update lower bound
                    lowerbd_x = max(lowerbd_x, rb_cur_x);
                    if (fabs(rb_prev_x - rb_cur_x) >= rtol_x) {
                        // update rb
                        rb_cur_x = 0.5 * (rb_cur_x + rb_prev_x);
                    }
                    rb_prev_x = temp;
                    temp = rb_cur_y;
                    // update lower bound
                    lowerbd_y = max(lowerbd_y, rb_cur_y);
                    if (fabs(rb_prev_y - rb_cur_y) >= rtol_y) {
                        // update rb
                        rb_cur_y = 0.5 * (rb_cur_y + rb_prev_y);
                    }
                    rb_prev_y = temp;
                    temp = rb_cur_z;
                    // update lower bound
                    lowerbd_z = max(lowerbd_z, rb_cur_z);
                    if (fabs(rb_prev_z - rb_cur_z) >= rtol_z) {
                        // update rb
                        rb_cur_z = 0.5 * (rb_cur_z + rb_prev_z);
                    }
                    rb_prev_z = temp;
                }
            }
            
            Ncube_x=ceil(rb_cur_x/dx-TEMP_TOL); 
            Ncube_y=ceil(rb_cur_y/dy-TEMP_TOL); 
            Ncube_z=ceil(rb_cur_z/dz-TEMP_TOL);
            
            // integrate b over the cell of length rb_cur
            Bint = 0.0;
            if(pSPARC->cell_typ == 0){
                for(k = 0; k < Ncube_z; k++)
                    for(j = 0; j < Ncube_y; j++)
                        for(i = 0; i < Ncube_x; i++) {
                            val = b(i,j,k);
                            if (k == 0) val *= 0.5;
                            if (k == Ncube_z-1) val *= 0.5;
                            if(j == 0) val *= 0.5;
                            if(j == Ncube_y-1) val *= 0.5;
                            if(i == 0) val *= 0.5;
                            if(i == Ncube_x-1) val *= 0.5;
                            Bint += val;
                        }
                Bint = Bint / (-4*M_PI) * 8.0 * pSPARC->dV;
            } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {    
                for(k = 0; k <= Ncube_z; k++) //for(k = nz - Ncube_z; k <= nz + Ncube_z; k++)
                    for(j = ny - Ncube_y; j <= ny + Ncube_y; j++)
                        for(i = nx - Ncube_x; i <= nx + Ncube_x; i++) {
                            val = b(i,j,k);
                            if (k == 0) val *= 0.5; //if (k == nz - Ncube_z) val *= 0.5;
                            if (k == Ncube_z) val *= 0.5; //if (k == nz + Ncube_z) val *= 0.5;
                            if (j == ny - Ncube_y) val *= 0.5;
                            if (j == ny + Ncube_y) val *= 0.5;
                            if (i == nx - Ncube_x) val *= 0.5;
                            if (i == nx + Ncube_x) val *= 0.5;
                            Bint += val;
                        }
                Bint = Bint / (-4*M_PI) * 2 * pSPARC->dV; 
            }
            //error_cur = fabs(Bint + pSPARC->Znucl[ityp]) / fabs(pSPARC->Znucl[ityp]); 
            error_cur = fabs(Bint + pSPARC->Znucl[ityp]); 
            count++;
#ifdef DEBUG
            if (rank==0) {
                printf("Z = %d,rb_x = %f,rb_y = %f,rb_z = %f,error = %.3E,Bint = %.13f\n", 
                        pSPARC->Znucl[ityp],rb_cur_x, rb_cur_y, rb_cur_z, error_cur, Bint); 
            }
#endif
        }
        
        // after rb search is done, update rb projected to grid
        Ncube_x=ceil(rb_cur_x/dx-TEMP_TOL); 
        Ncube_y=ceil(rb_cur_y/dy-TEMP_TOL); 
        Ncube_z=ceil(rb_cur_z/dz-TEMP_TOL);
        
        if (rank == 0 && count >= 20) {
            printf("\nWARNING: Searching for pseudocharge radius not converged!\n");
        }
        
        // evaluate the error again if necessary
        if (errflag == 1) {     
            // after rb search is done, update rb projected to grid
            Bint = 0.0;
            // integrate b over cuboidal domain with rb_cur
            if(pSPARC->cell_typ == 0){
                for(k = 0; k < Ncube_z; k++)
                    for(j = 0; j < Ncube_y; j++)
                        for(i = 0; i < Ncube_x; i++) {
                            val = b(i,j,k);
                            if (k == 0) val *= 0.5;
                            if (k == Ncube_z-1) val *= 0.5;
                            if(j == 0) val *= 0.5;
                            if(j == Ncube_y-1) val *= 0.5;
                            if(i == 0) val *= 0.5;
                            if(i == Ncube_x-1) val *= 0.5;
                            Bint += val;
                        }
                Bint = Bint / (-4*M_PI) * 8.0 * pSPARC->dV;
            } else if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {    
                for(k = 0; k <= Ncube_z; k++) 
                    for(j = ny - Ncube_y; j <= ny + Ncube_y; j++)
                        for(i = nx - Ncube_x; i <= nx + Ncube_x; i++) {
                            val = b(i,j,k);
                            if (k == 0) val *= 0.5; 
                            if (k == Ncube_z) val *= 0.5; 
                            if (j == ny - Ncube_y) val *= 0.5;
                            if (j == ny + Ncube_y) val *= 0.5;
                            if (i == nx - Ncube_x) val *= 0.5;
                            if (i == nx + Ncube_x) val *= 0.5;
                            Bint += val;
                        }
                Bint = Bint / (-4*M_PI) * 2 * pSPARC->dV; 
            }
            //error_cur = fabs(Bint + pSPARC->Znucl[ityp]) / fabs(pSPARC->Znucl[ityp]);
            error_cur = fabs(Bint + pSPARC->Znucl[ityp]);
        }

        if (rb_cur_x == Rbmax_x || rb_cur_y == Rbmax_y || rb_cur_z == Rbmax_z) {
            if (rank == 0 && error_cur > pSPARC->TOL_PSEUDOCHARGE) {
                printf("\nerror = %.2e > TOL_PSEUDOCHARGE = %.2e\n",error_cur,pSPARC->TOL_PSEUDOCHARGE);
                printf("\nWARNING: upperbond for pseudocharge radius (Rbmax_?) is not big enough! Rbmax_x = %.1f, Rbmax_y = %.1f, Rbmax_z = %.1f\n\n",Rbmax_x,Rbmax_y,Rbmax_z);
            }
        }

        pSPARC->CUTOFF_x[ityp] = Ncube_x * dx;
        pSPARC->CUTOFF_y[ityp] = Ncube_y * dy;
        pSPARC->CUTOFF_z[ityp] = Ncube_z * dz;

#ifdef DEBUG
        if (rank == 0) {
            printf("dx = %f, dy = %f, dz = %f, Ncube_x = %d, Ncube_y = %d, Ncube_z = %d\n",dx,dy,dz,Ncube_x,Ncube_y,Ncube_z);  
            printf("ityp = %d, converged in %d iters, err_cur = %.2E, TOL = %.2E, rb = {%f, %f, %f}, after proj to grid rb = {%f, %f, %f}\n", 
                    ityp, count, error_cur, pSPARC->TOL_PSEUDOCHARGE, rb_cur_x, rb_cur_y, rb_cur_z, pSPARC->CUTOFF_x[ityp], pSPARC->CUTOFF_y[ityp], pSPARC->CUTOFF_z[ityp]);
        }
#endif

#ifdef DEBUG
        t5 = MPI_Wtime();
        if (rank == 0) printf("time for finding rb using bisection: %.3f ms.\n", (t5 - t4) * 1000);        
#endif    
        // deallocate memory
        free(b);
        b = NULL;
    }
    
    // deallocate memory
    free(Lap_wt);
    free(R);

#undef b
#undef V_temp
#undef RB_MAX
#undef RB_Y
}



/**
 * @brief   Find the list of all atoms that influence the processor domain.
 */
void GetInfluencingAtoms(SPARC_OBJ *pSPARC) {
    // processors that are not in the dmcomm_phi will remain idle
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return;
    }
    
    int nproc, rank, coords[3], ityp, i, atmcount, atmcount2, count_overlap_local;
    int pp, qq, rr, ppmin, ppmax, qqmin, qqmax, rrmin, rrmax;
    double Lx, Ly, Lz, rbx, rby, rbz, x0, y0, z0, x0_i, y0_i, z0_i;
    double DMxs, DMxe, DMys, DMye, DMzs, DMze;
    int rb_xl, rb_xr, rb_yl, rb_yr, rb_zl, rb_zr;
    int isRbOut[3] = {0,0,0}; // check rb-region of atom is out of the domain
    MPI_Comm grid_comm = pSPARC->dmcomm_phi;
    MPI_Comm_size(grid_comm, &nproc);
    MPI_Comm_rank(grid_comm, &rank);
    MPI_Cart_coords(grid_comm, rank, 3, coords);    
#ifdef DEBUG
    if (rank == 0) {
        printf("Finding atoms that influence the local process domain ... \n");
    }
#endif
    Lx = pSPARC->range_x;
    Ly = pSPARC->range_y;
    Lz = pSPARC->range_z;
    
    DMxs = pSPARC->xin + pSPARC->DMVertices[0] * pSPARC->delta_x;
    DMxe = pSPARC->xin + (pSPARC->DMVertices[1]+1) * pSPARC->delta_x; // note that this is the actual edge
    DMys = pSPARC->DMVertices[2] * pSPARC->delta_y;
    DMye = (pSPARC->DMVertices[3]+1) * pSPARC->delta_y; // note that this is the actual edge
    DMzs = pSPARC->DMVertices[4] * pSPARC->delta_z;
    DMze = (pSPARC->DMVertices[5]+1) * pSPARC->delta_z; // note that this is the actual edge
    
    pSPARC->Atom_Influence_local = (ATOM_INFLUENCE_OBJ *)malloc(sizeof(ATOM_INFLUENCE_OBJ) * pSPARC->Ntypes);
    
    // find which atoms have influence on the distributed domain owned by current process
    atmcount = 0; atmcount2 = 0;
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rbx = pSPARC->CUTOFF_x[ityp];
        rby = pSPARC->CUTOFF_y[ityp];
        rbz = pSPARC->CUTOFF_z[ityp];
        
        // first loop over all atoms of each type to find number of influencing atoms
        count_overlap_local = 0;
        for (i = 0; i < pSPARC->nAtomv[ityp]; i++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount];
            y0 = pSPARC->atom_pos[3*atmcount+1];
            z0 = pSPARC->atom_pos[3*atmcount+2];
            atmcount++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                ppmax = floor((rbx + Lx - x0) / Lx + TEMP_TOL);
                ppmin = -floor((rbx + x0) / Lx + TEMP_TOL);
            }
            if (pSPARC->BCy == 0) {
                qqmax = floor((rby + Ly - y0) / Ly + TEMP_TOL);
                qqmin = -floor((rby + y0) / Ly + TEMP_TOL);
            }
            if (pSPARC->BCz == 0) {
                rrmax = floor((rbz + Lz - z0) / Lz + TEMP_TOL);
                rrmin = -floor((rbz + z0) / Lz + TEMP_TOL);
            }
            // check how many of it's images interacts with the local distributed domain
            for (rr = rrmin; rr <= rrmax; rr++) {
                z0_i = z0 + Lz * rr; // z coord of image atom
                if ((z0_i < DMzs - rbz) || (z0_i >= DMze + rbz)) continue;
                for (qq = qqmin; qq <= qqmax; qq++) {
                    y0_i = y0 + Ly * qq; // y coord of image atom
                    if ((y0_i < DMys - rby) || (y0_i >= DMye + rby)) continue;
                    for (pp = ppmin; pp <= ppmax; pp++) {
                        x0_i = x0 + Lx * pp; // x coord of image atom
                        if ((x0_i < DMxs - rbx) || (x0_i >= DMxe + rbx)) continue;
                        count_overlap_local++;
                    }
                }
            }

        } // end for loop over atoms of each type, for the first time

        pSPARC->Atom_Influence_local[ityp].n_atom = count_overlap_local;

        // now that the number of atoms of this type that have influence is determined, 
        // memories for storing atom positions, overlap domain corners, and so on, can be allocated
        pSPARC->Atom_Influence_local[ityp].coords = (double *)malloc(sizeof(double) * count_overlap_local * 3);
        pSPARC->Atom_Influence_local[ityp].atom_index = (int *)malloc(sizeof(int) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].atom_spin = (double *)malloc(sizeof(double) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].xs = (int *)malloc(sizeof(int) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].ys = (int *)malloc(sizeof(int) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].zs = (int *)malloc(sizeof(int) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].xe = (int *)malloc(sizeof(int) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].ye = (int *)malloc(sizeof(int) * count_overlap_local);
        pSPARC->Atom_Influence_local[ityp].ze = (int *)malloc(sizeof(int) * count_overlap_local);
        
        // TODO: Simply remove this if anything goes wrong
        // when there's no atom of this type that have influence, go to next type of atom
        if (pSPARC->Atom_Influence_local[ityp].n_atom == 0)
        {
            atmcount2 = atmcount;
            continue;
        }
                
        // loop over atoms of this type again to find overlapping region and atom info
        count_overlap_local = 0;
        for (i = 0; i < pSPARC->nAtomv[ityp]; i++) {
            // get atom positions
            x0 = pSPARC->atom_pos[3*atmcount2];
            y0 = pSPARC->atom_pos[3*atmcount2+1];
            z0 = pSPARC->atom_pos[3*atmcount2+2];
            atmcount2++;
            ppmin = ppmax = qqmin = qqmax = rrmin = rrmax = 0;
            if (pSPARC->BCx == 0) {
                ppmax = floor((rbx + Lx - x0) / Lx + TEMP_TOL);
                ppmin = -floor((rbx + x0) / Lx + TEMP_TOL);
            }
            if (pSPARC->BCy == 0) {
                qqmax = floor((rby + Ly - y0) / Ly + TEMP_TOL);
                qqmin = -floor((rby + y0) / Ly + TEMP_TOL);
            }
            if (pSPARC->BCz == 0) {
                rrmax = floor((rbz + Lz - z0) / Lz + TEMP_TOL);
                rrmin = -floor((rbz + z0) / Lz + TEMP_TOL);
            }
            
            // check if this image interacts with the local distributed domain
            for (rr = rrmin; rr <= rrmax; rr++) {
                z0_i = z0 + Lz * rr; // z coord of image atom
                if ((z0_i < DMzs - rbz) || (z0_i >= DMze + rbz)) continue;
                for (qq = qqmin; qq <= qqmax; qq++) {
                    y0_i = y0 + Ly * qq; // y coord of image atom
                    if ((y0_i < DMys - rby) || (y0_i >= DMye + rby)) continue;
                    for (pp = ppmin; pp <= ppmax; pp++) {
                        x0_i = x0 + Lx * pp; // x coord of image atom
                        if ((x0_i < DMxs - rbx) || (x0_i >= DMxe + rbx)) continue;
                        
                        // store coordinates of the overlapping atom
                        pSPARC->Atom_Influence_local[ityp].coords[count_overlap_local*3  ] = x0_i;
                        pSPARC->Atom_Influence_local[ityp].coords[count_overlap_local*3+1] = y0_i;
                        pSPARC->Atom_Influence_local[ityp].coords[count_overlap_local*3+2] = z0_i;
                        
                        // record the original atom index this image atom corresponds to
                        pSPARC->Atom_Influence_local[ityp].atom_index[count_overlap_local] = atmcount2-1;
                        
                        // record the spin on this image atom for spin polarized calculation
                        pSPARC->Atom_Influence_local[ityp].atom_spin[count_overlap_local] = pSPARC->atom_spin[atmcount2-1];
                        
                        // find start & end nodes of the rb-region of the image atom
                        // This way, we try to make sure all points inside rb-region
                        // is strictly less that rb distance away from the image atom
                        rb_xl = ceil((x0_i - pSPARC->xin - rbx)/pSPARC->delta_x);
                        rb_xr = floor((x0_i - pSPARC->xin + rbx)/pSPARC->delta_x);
                        rb_yl = ceil((y0_i - rby)/pSPARC->delta_y);
                        rb_yr = floor((y0_i + rby)/pSPARC->delta_y);
                        rb_zl = ceil((z0_i - rbz)/pSPARC->delta_z);
                        rb_zr = floor((z0_i + rbz)/pSPARC->delta_z);
                        
                        // TODO: check if rb-region is out of fundamental doamin for BC == 1!
                        if (pSPARC->BCx == 1 && (rb_xl < 0 || rb_xr >= pSPARC->Nx))
                            isRbOut[0] = 1;
                        if (pSPARC->BCy == 1 && (rb_yl < 0 || rb_yr >= pSPARC->Ny))
                            isRbOut[1] = 1;
                        if (pSPARC->BCz == 1 && (rb_zl < 0 || rb_zr >= pSPARC->Nz))
                            isRbOut[2] = 1;

                        // find overlap of rb-region of the image and the local dist. domain
                        pSPARC->Atom_Influence_local[ityp].xs[count_overlap_local] = max(pSPARC->DMVertices[0], rb_xl);
                        pSPARC->Atom_Influence_local[ityp].xe[count_overlap_local] = min(pSPARC->DMVertices[1], rb_xr);
                        pSPARC->Atom_Influence_local[ityp].ys[count_overlap_local] = max(pSPARC->DMVertices[2], rb_yl);
                        pSPARC->Atom_Influence_local[ityp].ye[count_overlap_local] = min(pSPARC->DMVertices[3], rb_yr);
                        pSPARC->Atom_Influence_local[ityp].zs[count_overlap_local] = max(pSPARC->DMVertices[4], rb_zl);
                        pSPARC->Atom_Influence_local[ityp].ze[count_overlap_local] = min(pSPARC->DMVertices[5], rb_zr);
                        
                        count_overlap_local++;
                    }
                }
            }

        }
    } 
    if (pSPARC->BCx == 1 || pSPARC->BCy == 1 || pSPARC->BCz == 1) {
        if (rank == 0) { 
            MPI_Reduce(MPI_IN_PLACE, isRbOut, 3, MPI_INT, MPI_LAND, 0, pSPARC->dmcomm_phi);
        } else {
            MPI_Reduce(isRbOut, isRbOut, 3, MPI_INT, MPI_LAND, 0, pSPARC->dmcomm_phi);
        }        
        #ifdef DEBUG
        if (!rank) printf("Is Rb region out? isRbOut = [%d, %d, %d]\n", isRbOut[0], isRbOut[1], isRbOut[2]);
        #endif
    }

}


/**
 * @brief   Calculate pseudocharge density for given atom positions
 */
void Generate_PseudoChargeDensity(SPARC_OBJ *pSPARC) {
#define electronDens_at(s,i,j,k) electronDens_at[s+(k)*DMnx*DMny+(j)*DMnx+(i)]
#define electronDens_core(s,i,j,k) electronDens_core[s+(k)*DMnx*DMny+(j)*DMnx+(i)]
#define psdChrgDens(i,j,k) psdChrgDens[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define psdChrgDens_ref(i,j,k) psdChrgDens_ref[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define Vc(i,j,k) Vc[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define VJ(i,j,k) VJ[(k)*nxp*nyp+(j)*nxp+(i)]
#define VJ_ref(i,j,k) VJ_ref[(k)*nxp*nyp+(j)*nxp+(i)]
#define rho_J(i,j,k) rho_J[(k)*nxp*nyp+(j)*nxp+(i)]
#define rho_c_J(i,j,k) rho_c_J[(k)*nxp*nyp+(j)*nxp+(i)]
    int nproc, rank, ityp, iat, i, j, k, ip, jp, kp, i_global, j_global, k_global, 
        i_DM, j_DM, k_DM,dI, dJ, dK, FDn, count, count_interp, DMnx, DMny, DMnz, 
        DMnd, nx, ny, nz, nd, nxp, nyp, nzp, nd_ex, icor, jcor, kcor, len_interp;
    int spn_i;
    double x0_i, y0_i, z0_i, x0_i_shift, y0_i_shift, z0_i_shift, x, y, z, *R,
           *VJ, *VJ_ref, Esc, *rho_J; 
    double spin, spin_frac;
    double inv_4PI = 0.25 / M_PI, w2_diag, rchrg;
    double *Lap_wt, *Lap_stencil;
    double xin;
#ifdef DEBUG
    double t1, t2;
    double tt1, tt2, ttot, ttot2;
    ttot = 0;
    ttot2 = 0;
#endif
    
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        double vt[2] = {0.0, 0.0};
        MPI_Bcast(vt, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        pSPARC->PosCharge = vt[0]; 
        pSPARC->NegCharge = vt[1];
#ifdef DEBUG
        if (!rank) printf("rank = %d, PosCharge = %f, NegCharge = %f\n", rank, pSPARC->PosCharge, pSPARC->NegCharge);
#endif
        return; 
    }  
    
    MPI_Comm_size(pSPARC->dmcomm_phi, &nproc);
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
#ifdef DEBUG
    if (rank == 0) {
        printf("Calculating pseudocharge density ... \n");
    }
#endif
   
    FDn = pSPARC->order / 2;
    w2_diag = (pSPARC->D2_stencil_coeffs_x[0] + pSPARC->D2_stencil_coeffs_y[0] + pSPARC->D2_stencil_coeffs_z[0]) * -inv_4PI;
    if(pSPARC->cell_typ == 0){
        Lap_wt = (double *)malloc((3*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt;
    } else{
        Lap_wt = (double *)malloc((5*(FDn+1))*sizeof(double));
        Lap_stencil = Lap_wt+5;
        //nonCart2Cart_coord(pSPARC, &delta_x, &delta_y, &delta_z);
    }
    Lap_stencil_coef_compact(pSPARC, FDn, Lap_stencil, -inv_4PI);

    DMnx = pSPARC->Nx_d;
    DMny = pSPARC->Ny_d;
    DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d;
    
    // initialize to zero at the beginning of each relax/MD step
    for (i = 0; i < DMnd; i++) {
        pSPARC->electronDens_at[i] = pSPARC->electronDens_core[i] =
        pSPARC->psdChrgDens[i] = pSPARC->psdChrgDens_ref[i] = pSPARC->Vc[i] = 0.0;    
    }
    
    for (spn_i = 1; spn_i < 2*pSPARC->Nspin-1; spn_i++) {
        for (i = 0; i < DMnd; i++) {
            pSPARC->electronDens_at[spn_i*DMnd + i] = 0.0;
            pSPARC->electronDens_core[spn_i*DMnd + i] = 0.0;
        }
    }

    // calculate pseudocharge density bJ and (self + correction) energy
    Esc = 0.0; // Esc = Eself + Ec
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        rchrg = pSPARC->psd[ityp].RadialGrid[pSPARC->psd[ityp].size-1];
        for (iat = 0; iat < pSPARC->Atom_Influence_local[ityp].n_atom; iat++) {      
            // coordinates of the image atom
            x0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3];
            y0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 1];
            z0_i = pSPARC->Atom_Influence_local[ityp].coords[iat * 3 + 2];
            
            // spin of the image atom
            spin = pSPARC->Atom_Influence_local[ityp].atom_spin[iat];
            spin_frac = spin/pSPARC->Znucl[ityp];
            
            // number of finite-difference nodes in each direction of overlap rb region
            nx = pSPARC->Atom_Influence_local[ityp].xe[iat] - pSPARC->Atom_Influence_local[ityp].xs[iat] + 1;
            ny = pSPARC->Atom_Influence_local[ityp].ye[iat] - pSPARC->Atom_Influence_local[ityp].ys[iat] + 1;
            nz = pSPARC->Atom_Influence_local[ityp].ze[iat] - pSPARC->Atom_Influence_local[ityp].zs[iat] + 1;
            nd = nx * ny * nz;
            
            // number of finite-difference nodes in each direction of extended rb (+ order/2) region
            nxp = nx + pSPARC->order;
            nyp = ny + pSPARC->order;
            nzp = nz + pSPARC->order;
            nd_ex = nxp * nyp * nzp; // total number of nodes

            // radii^2 of the finite difference grids of the extended-rb-region
            R  = (double *)malloc(sizeof(double) * nd_ex);
            if (R == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            } 
            
            // left corner of the extended-rb-region
            icor = pSPARC->Atom_Influence_local[ityp].xs[iat] - FDn;
            jcor = pSPARC->Atom_Influence_local[ityp].ys[iat] - FDn;
            kcor = pSPARC->Atom_Influence_local[ityp].zs[iat] - FDn;
            
            // relative coordinate of image atoms
            x0_i_shift =  x0_i - pSPARC->delta_x * icor; 
            y0_i_shift =  y0_i - pSPARC->delta_y * jcor;
            z0_i_shift =  z0_i - pSPARC->delta_z * kcor; 
            
            // find distance between atom and finite-difference grids
            count = 0; 
            count_interp = 0;
            
            if(pSPARC->cell_typ == 0) {    
                for (k = 0; k < nzp; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < nyp; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nxp; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            } else if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20) {
                for (k = 0; k < nzp; k++) {
                    z = k * pSPARC->delta_z - z0_i_shift; 
                    for (j = 0; j < nyp; j++) {
                        y = j * pSPARC->delta_y - y0_i_shift;
                        for (i = 0; i < nxp; i++) {
                            x = i * pSPARC->delta_x - x0_i_shift;
                            R[count] = sqrt(pSPARC->metricT[0] * (x*x) + pSPARC->metricT[1] * (x*y) + pSPARC->metricT[2] * (x*z) 
                                          + pSPARC->metricT[4] * (y*y) + pSPARC->metricT[5] * (y*z) + pSPARC->metricT[8] * (z*z) );
                            //R[count] = sqrt((x*x) + (y*y) + (z*z) );                   
                            if (R[count] <= rchrg) count_interp++;
                            count++;
                        }
                    }
                }
            }    
            //Calc_dist(pSPARC, nxp, nyp, nzp, x0_i_shift, y0_i_shift, z0_i_shift, R, rchrg, &count_interp);
            #ifdef DEBUG
            tt1 = MPI_Wtime();
            #endif
            VJ_ref = (double *)malloc( nd_ex * sizeof(double) );
            if (VJ_ref == NULL) {
               printf("\nMemory allocation failed!\n");
               exit(EXIT_FAILURE);
            }
            // Calculate pseudopotential reference
            Calculate_Pseudopot_Ref(R, nd_ex, pSPARC->REFERENCE_CUTOFF, -pSPARC->Znucl[ityp], VJ_ref);
            #ifdef DEBUG
            tt2 = MPI_Wtime();
            ttot += (tt2 - tt1);
            #endif
            
            VJ = (double *)malloc( nd_ex * sizeof(double) );
            if (VJ == NULL) {
                printf("\nMemory allocation failed!\n");
                exit(EXIT_FAILURE);
            } 

            len_interp = nd_ex;
            if (pSPARC->psd[ityp].is_r_uniform == 1) {
				SplineInterpUniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
				         R, VJ, len_interp, pSPARC->psd[ityp].SplinerVlocD);
            } else {
				SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rVloc, pSPARC->psd[ityp].size, 
				         R, VJ, len_interp, pSPARC->psd[ityp].SplinerVlocD);
            }

            for (i = 0; i < nd_ex; i++) {
                // rearrange VJ back to original order
                if (fabs(R[i]) < TEMP_TOL) {
                    VJ[i] = pSPARC->psd[ityp].Vloc_0;
                } else if (R[i] > rchrg) {
            		VJ[i] = -pSPARC->Znucl[ityp] / R[i];
                } else {
                    VJ[i] = VJ[i] / R[i];
                }
            }
            #ifdef DEBUG
            tt1 = MPI_Wtime();
            #endif
            // calculate the sum of atomic charge densities as initial electron density guess, only in the very 
            // first MD/Relaxation step, and only if restart_flag is off
			rho_J = (double *)malloc( nd_ex * sizeof(double) );
            assert(rho_J != NULL);

            len_interp = nd_ex;
			if (pSPARC->psd[ityp].is_r_uniform == 1) {
				SplineInterpUniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rhoIsoAtom, pSPARC->psd[ityp].size, 
				         R, rho_J, len_interp, pSPARC->psd[ityp].SplineFitIsoAtomDen);
            } else {
				SplineInterpNonuniform(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rhoIsoAtom, pSPARC->psd[ityp].size, 
				         R, rho_J, len_interp, pSPARC->psd[ityp].SplineFitIsoAtomDen);
			}
            
            double *rho_c_J = (double *)malloc( nd_ex * sizeof(double) );
            assert(rho_c_J != NULL);
            SplineInterpMain(pSPARC->psd[ityp].RadialGrid,pSPARC->psd[ityp].rho_c_table, pSPARC->psd[ityp].size, 
                         R, rho_c_J, len_interp, pSPARC->psd[ityp].SplineRhocD,pSPARC->psd[ityp].is_r_uniform);

            for (i = 0; i < nd_ex; i++) {
                if (R[i] > rchrg) {
                    rho_J[i] = 0.0;
                    rho_c_J[i] = 0.0;
                }
            }

            for (k = 0; k < nz; k++) {
                kp = k + FDn;
                k_global = k + pSPARC->Atom_Influence_local[ityp].zs[iat];// global coord 
                k_DM = k_global - pSPARC->DMVertices[4]; // local coord 
                if (k_DM < 0 || k_DM >= DMnz) continue;
                for (j = 0; j < ny; j++) {
                    jp = j + FDn;
                    j_global = j + pSPARC->Atom_Influence_local[ityp].ys[iat];
                    j_DM = j_global - pSPARC->DMVertices[2]; // local coord 
                    if (j_DM < 0 || j_DM >= DMny) continue;
                    for (i = 0; i < nx; i++) {
                        ip = i + FDn;
                        i_global = i + pSPARC->Atom_Influence_local[ityp].xs[iat];
                        i_DM = i_global - pSPARC->DMVertices[0]; // local coord 
                        if (i_DM < 0 || i_DM >= DMnx) continue;
                        pSPARC->electronDens_at(0,i_DM,j_DM,k_DM) += rho_J(ip,jp,kp);
                        pSPARC->electronDens_core(0,i_DM,j_DM,k_DM) += rho_c_J(ip,jp,kp);
                        for(spn_i = 1; spn_i < 2*pSPARC->Nspin - 1; spn_i++) {
                            pSPARC->electronDens_at(spn_i*DMnd,i_DM,j_DM,k_DM) += (0.5 + (1.5 - spn_i) * spin_frac) * rho_J(ip,jp,kp);
                            pSPARC->electronDens_core(spn_i*DMnd,i_DM,j_DM,k_DM) += 0.5 * rho_c_J(ip,jp,kp);
                        }
                    }
                }
            }

            free(rho_J); rho_J = NULL;
            free(rho_c_J); rho_c_J = NULL;
            #ifdef DEBUG
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            #endif
            
            free(R); R = NULL;

            dK = pSPARC->Atom_Influence_local[ityp].zs[iat] - pSPARC->DMVertices[4];
            dJ = pSPARC->Atom_Influence_local[ityp].ys[iat] - pSPARC->DMVertices[2];
            dI = pSPARC->Atom_Influence_local[ityp].xs[iat] - pSPARC->DMVertices[0];
            
            // calculate pseudocharge density bJ and add to b
            double *bJ = (double*)malloc(nd * sizeof(double));
            double *bJ_ref = (double*)malloc(nd * sizeof(double));
                
            xin = pSPARC->xin + pSPARC->Atom_Influence_local[ityp].xs[iat] * pSPARC->delta_x;
            Calc_lapV(pSPARC, VJ, FDn, nxp, nyp, nzp, nx, ny, nz, Lap_wt, w2_diag, xin, -inv_4PI, bJ);
            Calc_lapV(pSPARC, VJ_ref, FDn, nxp, nyp, nzp, nx, ny, nz, Lap_wt, w2_diag, xin, -inv_4PI, bJ_ref);

            for (k = 0, kp = FDn, k_DM = dK; k < nz; k++, kp++, k_DM++) {
                int kshift_DM = k_DM * DMnx * DMny;
                int kshift_p = kp * nxp * nyp;
                int kshift = k * nx * ny;  
                for (j = 0, jp = FDn, j_DM = dJ; j < ny; j++, jp++, j_DM++) {
                    int jshift_DM = kshift_DM + j_DM * DMnx;
                    int jshift_p = kshift_p + jp * nxp;
                    int jshift = kshift + j * nx;
                    for (i = 0, ip = FDn, i_DM = dI; i < nx; i++, ip++, i_DM++) {
                        int ishift_DM = jshift_DM + i_DM;
                        int ishift_p = jshift_p + ip;
                        int ishift = jshift + i;
                        pSPARC->psdChrgDens[ishift_DM] += bJ[ishift];
                        pSPARC->psdChrgDens_ref[ishift_DM] += bJ_ref[ishift];
                        pSPARC->Vc[ishift_DM] += (VJ_ref[ishift_p] -  VJ[ishift_p]);
                        double bJvJ = bJ_ref[ishift] * VJ_ref[ishift_p];
                        Esc -= bJvJ;
                    }
                }
            }
            
            free(bJ); bJ = NULL;
            free(bJ_ref); bJ_ref = NULL;
            free(VJ); VJ = NULL;
            free(VJ_ref); VJ_ref = NULL;
        }
    
    }

    /*  Calculate integral of b and Esc  */
    double int_b = 0.0, int_rho = 0.0;
    // find integral of b, Esc locally
    for (k = 0; k < DMnz; k++) {
        for (j = 0; j < DMny; j++) {
            for (i = 0; i < DMnx; i++) {
                int_b += pSPARC->psdChrgDens(i,j,k);
                int_rho += pSPARC->electronDens_at(0,i,j,k);
                Esc += (pSPARC->psdChrgDens(i,j,k) + pSPARC->psdChrgDens_ref(i,j,k)) * pSPARC->Vc(i,j,k);
            }
        }
    }
    int_b *= pSPARC->dV;
    int_rho *= pSPARC->dV; 
    Esc *= pSPARC->dV * 0.5; 
    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    MPI_Reduce(&Esc, &pSPARC->Esc, 1, MPI_DOUBLE,
               MPI_SUM , 0, pSPARC->dmcomm_phi);
    // find integral rho to get negative charge of the system
    double vt[2],vsum[2]={0,0};
    vt[0] = int_b; vt[1] = int_rho;
    MPI_Allreduce(vt, vsum, 2, MPI_DOUBLE,
                  MPI_SUM, pSPARC->dmcomm_phi);
    pSPARC->PosCharge = -vsum[0];
    pSPARC->NegCharge = -vsum[1];

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("the global sum of int_b = %.13f, sum_int_rho = %.13f\n", -pSPARC->PosCharge, -pSPARC->NegCharge);    
#endif

    /*  Scale electron density so that PosCharge + NegCharge = NetCharge  */
    double Nelectron_check = 0.0, scal_fac = (pSPARC->NetCharge - pSPARC->PosCharge) / pSPARC->NegCharge;
    for (k = 0; k < DMnz; k++) {
        for (j = 0; j < DMny; j++) {
            for (i = 0; i < DMnx; i++) {
                pSPARC->electronDens_at(0,i,j,k) *= scal_fac;
                Nelectron_check += pSPARC->electronDens_at(0,i,j,k);
                for(spn_i = 1; spn_i < 2*pSPARC->Nspin - 1; spn_i++)
                    pSPARC->electronDens_at(spn_i*DMnd,i,j,k) *= scal_fac;
            }
        }
    }
    Nelectron_check *= pSPARC->dV;
    // Find net magnetization in the beginning
    double int_rhoup = 0.0, int_rhodn = 0.0;
    double spn_int[2], spn_sum[2] = {0.0,0.0};
    if(pSPARC->spin_typ != 0 && pSPARC->elecgs_Count == 0) {
        for (k = 0; k < DMnz; k++) {
            for (j = 0; j < DMny; j++) {
                for (i = 0; i < DMnx; i++) {
                    int_rhoup += pSPARC->electronDens_at(DMnd,i,j,k);
                    int_rhodn += pSPARC->electronDens_at(2*DMnd,i,j,k);
                }
            }
        }
        int_rhoup *= pSPARC->dV;
        int_rhodn *= pSPARC->dV;
        spn_int[0] = int_rhoup; spn_int[1] = int_rhodn;
        MPI_Allreduce(spn_int, spn_sum, 2, MPI_DOUBLE,
                      MPI_SUM, pSPARC->dmcomm_phi);
        pSPARC->netM = spn_sum[0] - spn_sum[1];          
#ifdef DEBUG
        if(!rank) {
            printf("Net magnetization and total charge are : %.15f, %.15f\n", pSPARC->netM, spn_sum[0] + spn_sum[1]);
        }
#endif                          
    }

#ifdef DEBUG
    if(!rank) {
        printf("PosCharge = %.12f, NegCharge = %.12f, scal_fac = %.12f\n",pSPARC->PosCharge, pSPARC->NegCharge, scal_fac);
    }
#endif
    MPI_Reduce(&Nelectron_check, &pSPARC->NegCharge, 1, MPI_DOUBLE,
               MPI_SUM , 0, pSPARC->dmcomm_phi);
    pSPARC->NegCharge *= -1;
#ifdef DEBUG
    if (rank == 0) printf("After scaling, int_rho = %.13f, PosCharge + NegCharge - NetCharge = %.3e\n", -pSPARC->NegCharge, -pSPARC->NetCharge + pSPARC->PosCharge + pSPARC->NegCharge);
#endif
 
#ifdef DEBUG 
    if (rank == 0) {
        printf("--Calculate Vref took %.3f ms\n", ttot * 1e3);
        printf("--Calculate rho_guess took %.3f ms\n", ttot2 * 1e3);
    }
    if (rank == 0) {
        printf("\n integral of b = %.13f,\n int{b} + Nelectron + NetCharge = %.3e,\n Esc = %.13f,\n MPI_Reduce took %.3f ms\n",
                          -pSPARC->PosCharge, -pSPARC->PosCharge + pSPARC->Nelectron + pSPARC->NetCharge,pSPARC->Esc,(t2-t1)*1e3);
    }
#endif

    // Broadcast PosCharge and NegCharge
    if (!rank) {
        vt[0] = pSPARC->PosCharge; 
        vt[1] = pSPARC->NegCharge;
    }
    MPI_Bcast(vt, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank) {
        pSPARC->PosCharge = vt[0]; 
        pSPARC->NegCharge = vt[1];
    }
    free(Lap_wt);
    
#undef psdChrgDens
#undef psdChrgDens_ref
#undef electronDens_at
#undef electronDens_core
#undef Vc
#undef VJ
#undef VJ_ref
#undef rho_J
#undef rho_c_J
}



/*
 @ brief: function to evaluate bJ/bJ_ref
*/
void Calc_lapV(const SPARC_OBJ *pSPARC, const double *VJ, const double FDn,
               const int nxp, const int nyp, const int nzp,
               const int nx, const int ny, const int nz,
               const double *Lap_wt, const double w2_diag, double xin, double coef, double *bJ)
{                   
    int nxny = nx * ny;
    int nxny_ex = nxp * nyp;
    int nxexny = nxp * ny;
    int nxnyex = nx * nyp;                   

    double *dV1, *dV2;

    switch(pSPARC->cell_typ) {
        case 0  :
            stencil_3axis_thread_v2(
                VJ, FDn, nx, nxp, nxny, nxny_ex, 
                0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 
                Lap_wt, w2_diag, 0.0, VJ, bJ);
            break; /* optional */

        case 11  :
            dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // dV/dy
            if(dV1 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }
            
            Calc_DX(VJ, dV1, FDn, nxp, nxp, nxp, nxny_ex, nxexny, 
                    0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0);

            stencil_4comp(VJ, dV1, FDn, 1, nx, nxp, nxp, nxny, nxny_ex, nxexny,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);

            free(dV1); dV1 = NULL;                                                     
            break; /* optional */
          
        case 12  :
            dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // dV/dz
            if(dV1 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }
            
            Calc_DX(VJ, dV1, FDn, nxny_ex, nxp, nxp, nxny_ex, nxexny, 
                    0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            
            stencil_4comp(VJ, dV1, FDn, 1, nx, nxp, nxp, nxny, nxny_ex, nxexny,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);
                          
            free(dV1); dV1 = NULL;
            break; /* optional */
        
        case 13  :
            dV1 = (double *)malloc( (nxnyex*nz) * sizeof(double) ); // dV/dz
            if(dV1 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }

            Calc_DX(VJ, dV1, FDn, nxny_ex, nxp, nx, nxny_ex, nxnyex, 
                    0, nx, 0, nyp, 0, nz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            
            stencil_4comp(VJ, dV1, FDn, nx, nx, nxp, nx, nxny, nxny_ex, nxnyex,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 0, FDn, 0,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);
            
            free(dV1); dV1 = NULL;
            break; /* optional */
        
        case 14  :
            dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // 2*T_12*dV/dy + 2*T_13*dV/dz
            if(dV1 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }
            
            Calc_DX1_DX2(VJ, dV1, FDn, nxp, nxny_ex, nxp, nxp, nxny_ex, nxexny, 
                         0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
            
            stencil_4comp(VJ, dV1, FDn, 1, nx, nxp, nxp, nxny, nxny_ex, nxexny,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);
                          
            free(dV1); dV1 = NULL;
            break; /* optional */
          
        case 15  :
            dV1 = (double *)malloc( (nxny*nzp) * sizeof(double) ); // 2*T_13*dV/dx + 2*T_23*dV/dy
            if(dV1 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }

            Calc_DX1_DX2(VJ, dV1, FDn, 1, nxp, nxp, nx, nxny_ex, nxny, 
                         0, nx, 0, ny, 0, nzp, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy);
            // lap*V = T_11*d2/dx2 + T_22*d2/dy2 + T_33*d2/dz2 + d/dz(2*T_13*dV/dx + 2*T_23*dV/dy)
            
            stencil_4comp(VJ, dV1, FDn, nxny, nx, nxp, nx, nxny, nxny_ex, nxny,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 0, 0, FDn,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);
                          
            free(dV1); dV1 = NULL;
            break; /* optional */
          
        case 16  :
            dV1 = (double *)malloc( (nxnyex*nz) * sizeof(double) ); // 2*T_12*dV/dx + 2*T_23*dV/dz
            if(dV1 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }

            // 2*T_12*dV/dx + 2*T_23*dV/dz
            Calc_DX1_DX2(VJ, dV1, FDn, 1, nxny_ex, nxp, nx, nxny_ex, nxnyex, 
                         0, nx, 0, nyp, 0, nz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz);
            // lap*V = T_11*d2/dx2 + T_22*d2/dy2 + T_33*d2/dz2 + d/dy(2*T_12*dV/dx + 2*T_23*dV/dz)
            stencil_4comp(VJ, dV1, FDn, nx, nx, nxp, nx, nxny, nxny_ex, nxnyex,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 0, FDn, 0,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);
           
            free(dV1); dV1 = NULL;
            break; /* optional */
        
        case 17  :
            dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // 2*T_12*dV/dy + 2*T_13*dV/dz
            dV2 = (double *)malloc( (nxnyex*nz) * sizeof(double) ); // dV/dz
            if(dV1 == NULL || dV2 == NULL){
                printf("\nMemory allocation failed in pseudocharge radius!\n");
                exit(EXIT_FAILURE);
            }

            // 2*T_12*dV/dy + 2*T_13*dV/dz
            Calc_DX1_DX2(VJ, dV1, FDn, nxp, nxny_ex, nxp, nxp, nxny_ex, nxexny, 
                         0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
            
            Calc_DX(VJ, dV2, FDn, nxny_ex, nxp, nx, nxny_ex, nxnyex, 
                    0, nx, 0, nyp, 0, nz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
            
            stencil_5comp(VJ, dV1, dV2, FDn, 1, nx, nx, nxp, nxp, nx, nxny, nxny_ex, nxexny, nxnyex,
                          0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0, 0, FDn, 0,
                          Lap_wt, w2_diag, 0.0, VJ, bJ);

            free(dV1); dV1 = NULL;
            free(dV2); dV2 = NULL;
            break; /* optional */
          
        /* you can have any number of case statements */
        default : /* Optional */
        printf("cell_typ = %d is not implemented!\n", pSPARC->cell_typ); 
        exit(EXIT_FAILURE); break;
    }

    // if(pSPARC->cell_typ == 0){
    //    stencil_3axis_thread_v2(
    //            VJ, FDn, nx, nxp, nxny, nxny_ex, 
    //            0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 
    //            Lap_wt, w2_diag, 0.0, VJ, bJ);
    // } else if(pSPARC->cell_typ == 11){
    //    dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // dV/dy
    //    if(dV1 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }
       
    //    Calc_DX(VJ, dV1, FDn, nxp, nxp, nxp, nxny_ex, nxexny, 
    //            0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_y, 0.0);
       
    //    stencil_4comp(VJ, dV1, FDn, 1, nx, nxp, nxp, nxny, nxny_ex, nxexny,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);
       
    //    free(dV1); dV1 = NULL;                                                     
    // } else if(pSPARC->cell_typ == 12){
    //    dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // dV/dz
    //    if(dV1 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }
       
    //    Calc_DX(VJ, dV1, FDn, nxny_ex, nxp, nxp, nxny_ex, nxexny, 
    //            0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
       
    //    stencil_4comp(VJ, dV1, FDn, 1, nx, nxp, nxp, nxny, nxny_ex, nxexny,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);
                     
    //    free(dV1); dV1 = NULL;
    // } else if(pSPARC->cell_typ == 13){
    //    dV1 = (double *)malloc( (nxnyex*nz) * sizeof(double) ); // dV/dz
    //    if(dV1 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }

    //    Calc_DX(VJ, dV1, FDn, nxny_ex, nxp, nx, nxny_ex, nxnyex, 
    //            0, nx, 0, nyp, 0, nz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
       
    //    stencil_4comp(VJ, dV1, FDn, nx, nx, nxp, nx, nxny, nxny_ex, nxnyex,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 0, FDn, 0,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);
       
    //    free(dV1); dV1 = NULL;
    // } else if(pSPARC->cell_typ == 14){
    //    dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // 2*T_12*dV/dy + 2*T_13*dV/dz
    //    if(dV1 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }
       
    //    Calc_DX1_DX2(VJ, dV1, FDn, nxp, nxny_ex, nxp, nxp, nxny_ex, nxexny, 
    //                 0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
       
    //    stencil_4comp(VJ, dV1, FDn, 1, nx, nxp, nxp, nxny, nxny_ex, nxexny,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);
                     
    //    free(dV1); dV1 = NULL; 
    // } else if(pSPARC->cell_typ == 15) {
    //    dV1 = (double *)malloc( (nxny*nzp) * sizeof(double) ); // 2*T_13*dV/dx + 2*T_23*dV/dy
    //    if(dV1 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }

    //    Calc_DX1_DX2(VJ, dV1, FDn, 1, nxp, nxp, nx, nxny_ex, nxny, 
    //                 0, nx, 0, ny, 0, nzp, FDn, FDn, 0, pSPARC->D1_stencil_coeffs_zx, pSPARC->D1_stencil_coeffs_zy);
    //    // lap*V = T_11*d2/dx2 + T_22*d2/dy2 + T_33*d2/dz2 + d/dz(2*T_13*dV/dx + 2*T_23*dV/dy)
       
    //    stencil_4comp(VJ, dV1, FDn, nxny, nx, nxp, nx, nxny, nxny_ex, nxny,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 0, 0, FDn,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);
                     
    //    free(dV1); dV1 = NULL;
    // } else if(pSPARC->cell_typ == 16){
    //    dV1 = (double *)malloc( (nxnyex*nz) * sizeof(double) ); // 2*T_12*dV/dx + 2*T_23*dV/dz
    //    if(dV1 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }

    //    // 2*T_12*dV/dx + 2*T_23*dV/dz
    //    Calc_DX1_DX2(VJ, dV1, FDn, 1, nxny_ex, nxp, nx, nxny_ex, nxnyex, 
    //                 0, nx, 0, nyp, 0, nz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_yx, pSPARC->D1_stencil_coeffs_yz);
    //    // lap*V = T_11*d2/dx2 + T_22*d2/dy2 + T_33*d2/dz2 + d/dy(2*T_12*dV/dx + 2*T_23*dV/dz)
    //    stencil_4comp(VJ, dV1, FDn, nx, nx, nxp, nx, nxny, nxny_ex, nxnyex,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, 0, FDn, 0,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);
      
    //    free(dV1); dV1 = NULL;
    // } else if(pSPARC->cell_typ == 17) {
    //    dV1 = (double *)malloc( (nxexny*nz) * sizeof(double) ); // 2*T_12*dV/dy + 2*T_13*dV/dz
    //    dV2 = (double *)malloc( (nxnyex*nz) * sizeof(double) ); // dV/dz
    //    if(dV1 == NULL || dV2 == NULL){
    //        printf("\nMemory allocation failed in pseudocharge radius!\n");
    //        exit(EXIT_FAILURE);
    //    }

    //    // 2*T_12*dV/dy + 2*T_13*dV/dz
    //    Calc_DX1_DX2(VJ, dV1, FDn, nxp, nxny_ex, nxp, nxp, nxny_ex, nxexny, 
    //                 0, nxp, 0, ny, 0, nz, 0, FDn, FDn, pSPARC->D1_stencil_coeffs_xy, pSPARC->D1_stencil_coeffs_xz);
       
    //    Calc_DX(VJ, dV2, FDn, nxny_ex, nxp, nx, nxny_ex, nxnyex, 
    //            0, nx, 0, nyp, 0, nz, FDn, 0, FDn, pSPARC->D1_stencil_coeffs_z, 0.0);
       
    //    stencil_5comp(VJ, dV1, dV2, FDn, 1, nx, nx, nxp, nxp, nx, nxny, nxny_ex, nxexny, nxnyex,
    //                  0, nx, 0, ny, 0, nz, FDn, FDn, FDn, FDn, 0, 0, 0, FDn, 0,
    //                  Lap_wt, w2_diag, 0.0, VJ, bJ);

    //    free(dV1); dV1 = NULL;
    //    free(dV2); dV2 = NULL;
    // }
}    



/**
 * @brief   Calculate pseudopotential reference.
 *
 * @ref     Linear scaling solution of the all-electron Coulomb problem in solids.
 *          J.E.Pask, N. Sukumar and S.E. Mousavi
 */
void Calculate_Pseudopot_Ref(double *R, int len, double rcut, int Znucl, double *Vref) {
    int i;
    double R2, C3, C6;
    for (i = 0; i < len; i++) {
        if (R[i] <= rcut) {
            R2 = R[i]*R[i];
            C3 = rcut * rcut * rcut;
            C6 = C3 * C3;
            // use horner's rule 
            Vref[i] = R2 * ( R2*R[i] * (5.6/C6 + R[i] * (1.8*R[i]/(C6*rcut*rcut) - 6.0/(C6 * rcut) )) - 2.8/C3) + 2.4/rcut; 
            //Vref[i] = pow(R[i],2) * ( pow(R[i],3) * (5.6 / pow(rcut,6) + R[i]*(1.8*R[i]/pow(rcut,8) - 6/pow(rcut,7) ) ) - 2.8/pow(rcut,3)) + 2.4/rcut;
            // // calculate directly
            // Vref[i] = (9.0*pow(R[i],7) - 30.0*pow(R[i],6)*rcut + 28.0*pow(R[i],5)*rcut*rcut - 14.0*pow(R[i],2)*pow(rcut,5) + 12.0*pow(rcut,7))/(5.0*pow(rcut,8));
            Vref[i] *= Znucl;
        } else {
            Vref[i] = Znucl / R[i];
        }
    }
}



/**
 * @brief   Calculate the right-hand-side of the poisson equation.
 *          
 *          The RHS of the poisson equation includes the boundary
 *          conditions, i.e., f = -4 * pi * ( rho + b - d), where
 *          d is the BC term. For periodic systems, d = 0.
 */
void poisson_RHS(SPARC_OBJ *pSPARC, double *rhs) {

    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    #ifdef DEBUG
    double t1, t2;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("Start calculating rhs of the poisson problem\n");
    #endif

	int DMnd = pSPARC->Nd_d;

	for (int i = 0; i < DMnd; i++) {
		rhs[i] = 4.0 * M_PI * (pSPARC->psdChrgDens[i] + pSPARC->electronDens[i]);
	}
    
    // for charged systems, add a uniform background charge so that total charge is 0
    if (fabs((double)pSPARC->NetCharge) > TEMP_TOL) {
		double Lx = pSPARC->range_x;
		double Ly = pSPARC->range_y;
		double Lz = pSPARC->range_z;
        double unif_bkgd_chrg = pSPARC->NetCharge / (Lx*Ly*Lz*pSPARC->Jacbdet);
        double rhs_shift = 4.0 * M_PI * unif_bkgd_chrg;
        for (int i = 0; i < DMnd; i++) {
			rhs[i] += rhs_shift;
		}
    }

    int n_periodic = 3 - (pSPARC->BCx + pSPARC->BCy + pSPARC->BCz);

    // for Dirichlet BC
    if (n_periodic == 0) {
        #ifdef DEBUG
        t1 = MPI_Wtime();
        if (rank == 0) printf("Start calculating Multipole Expansion for Dirichlet BC\n");
        #endif

        // multipole expansion for Dirichlet BC!
        double *d_cor = (double *)malloc( DMnd * sizeof(double) );
        MultipoleExpansion_phi(pSPARC, rhs, d_cor);
        
        for (int i = 0; i < DMnd; i++) rhs[i] -= d_cor[i];
        free(d_cor);
		
		#ifdef DEBUG
        t2 = MPI_Wtime();
		if(!rank)
			printf("Multipole Expansion for Dirichlet BC took %.3f ms\n",(t2-t1)*1e3);
		#endif
    } else if (n_periodic == 2) {
		#ifdef DEBUG
		t1 = MPI_Wtime();
		if (rank == 0) printf("Start calculating Surface BC\n");
		#endif

		// multipole expansion for Dirichlet BC!
		double *d_cor = (double *)malloc( DMnd * sizeof(double) );
		PartrialDipole_surface(pSPARC, rhs, d_cor);

		for (int i = 0; i < DMnd; i++) rhs[i] -= d_cor[i];
		free(d_cor);

		#ifdef DEBUG
		t2 = MPI_Wtime();
		if(!rank)
			printf("Surface BC took %.3f ms\n",(t2-t1)*1e3);
		#endif
    } else if (n_periodic == 1) {
    	#ifdef DEBUG
		t1 = MPI_Wtime();
		if (rank == 0) printf("Start calculating Wire BC\n");
		#endif

		// multipole expansion for Dirichlet BC!
		double *d_cor = (double *)malloc( DMnd * sizeof(double) );
		PartrialDipole_wire(pSPARC, rhs, d_cor);

		for (int i = 0; i < DMnd; i++) rhs[i] -= d_cor[i];
		free(d_cor);

		#ifdef DEBUG
		t2 = MPI_Wtime();
		if(!rank)
			printf("Wire BC took %.3f ms\n",(t2-t1)*1e3);
		#endif
    }
}



/**
 * @brief   Solve Poisson equation for electrostatic potential.
 */
void Calculate_elecstPotential(SPARC_OBJ *pSPARC) {
#define psdChrgDens(i,j,k) psdChrgDens[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define electronDens(i,j,k) electronDens[(k)*DMnx*DMny+(j)*DMnx+(i)]
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef DEBUG
    if (rank == 0) {
        printf("Start calculating electrostatic potential ... \n");
    }
#endif
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) {
        return; 
    } 

    int DMnd = pSPARC->Nd_d;
    
    //t1 = MPI_Wtime();
    // calculate right hand side of the possoin equation
    double *rhs = (double *)malloc( DMnd * sizeof(double) );
    // rhs = -4 * pi * ( rho + b - d)
    poisson_RHS(pSPARC, rhs);

    // declare functions for linear solver
    void Jacobi_preconditioner(SPARC_OBJ *pSPARC, int N, double c, double *r, double *f, MPI_Comm comm);
    
    // function pointer that applies b + Laplacian * x
    void (*residule_fptr) (SPARC_OBJ*, int, double, double*, double*, double*, MPI_Comm, double*) = poisson_residual; // poisson_residual is defined in lapVecRoutines.c
    void (*Jacobi_fptr) (SPARC_OBJ*, int, double, double*, double*, MPI_Comm) = Jacobi_preconditioner;
    
#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
    double int_b = 0.0, int_rho = 0.0;
    // find integral of b, rho locally
    for (int i = 0; i < DMnd; i++) {
    	int_b += pSPARC->psdChrgDens[i];
        int_rho += pSPARC->electronDens[i];
    }
    double vt[2],vsum[2]={0,0};
    vt[0] = int_b; vt[1] = int_rho;
    MPI_Allreduce(vt, vsum, 2, MPI_DOUBLE,
                  MPI_SUM, pSPARC->dmcomm_phi); 
    int_b   = vsum[0];
    int_rho = vsum[1];
    t2 = MPI_Wtime();
    if(!rank) 
    printf("rank = %d, int_b = %18.14f, int_rho = %18.14f, int_b + int_rho = %.3e, checking this took %.3f ms\n",
    	rank,int_b*pSPARC->dV,int_rho*pSPARC->dV,(int_b+int_rho)*pSPARC->dV,(t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif    

    // call linear solver to solve the poisson equation
    // solve -Laplacian phi = 4 * M_PI * (rho + b) 
    // TODO: use real preconditioner instead of Jacobi preconditioner!
    if(pSPARC->POISSON_SOLVER == 0) {
        double omega, beta;
        int m, p;
        omega = 0.6, beta = 0.6; //omega = 0.6, beta = 0.6;
        m = 7, p = 6; //m = 9, p = 8; //m = 9, p = 9;
        AAR(pSPARC, residule_fptr, Jacobi_fptr, 0.0, DMnd, pSPARC->elecstPotential, rhs, 
        omega, beta, m, p, pSPARC->TOL_POISSON, pSPARC->MAXIT_POISSON, pSPARC->dmcomm_phi);
    } else {
        if (rank == 0) printf("Please provide a valid poisson solver!\n");
        exit(EXIT_FAILURE);
        // CG(pSPARC, Lap_vec_mult, pSPARC->Nd, DMnd, pSPARC->elecstPotential, rhs, pSPARC->TOL_POISSON, pSPARC->MAXIT_POISSON, pSPARC->dmcomm_phi);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
#endif

    // shift the electrostatic potential so that its integral is zero for periodic systems
    if (pSPARC->BC == 2 || pSPARC->BC == 0) {
        double phi_shift = 0.0;
        VectorSum  (pSPARC->elecstPotential, DMnd, &phi_shift, pSPARC->dmcomm_phi);
        phi_shift /= (double)pSPARC->Nd;
        VectorShift(pSPARC->elecstPotential, DMnd, -phi_shift, pSPARC->dmcomm_phi);
    }
    free(rhs);
    
#undef psdChrgDens
#undef electronDens  
}


void Jacobi_preconditioner(SPARC_OBJ *pSPARC, int N, double c, double *r, double *f, MPI_Comm comm) {
    int i;
    double m_inv;
    // TODO: m_inv can be calculated in advance and stored into pSPARC
    m_inv =  pSPARC->D2_stencil_coeffs_x[0] 
           + pSPARC->D2_stencil_coeffs_y[0] 
           + pSPARC->D2_stencil_coeffs_z[0] + c;
    if (fabs(m_inv) < 1e-14) {
        m_inv = 1.0;
    }
    m_inv = - 1.0 / m_inv;
    for (i = 0; i < N; i++)
        f[i] = m_inv * r[i];
}



/**
 * @brief   Perform multipole expansion to find boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          So that when discretized in finite difference with Dirichlet BC, the equation will be
 *                                  - DiscreteLaplacian phi = f - d.
 *          It is required that f decays to zero on the boundary.
 *
 *          Note that this is only done in "phi-domain".
 */
void MultipoleExpansion_phi(SPARC_OBJ *pSPARC, double *f, double *d_cor)
{
#define d_cor(i,j,k) d_cor[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define phi(i,j,k) phi[(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int LMAX = 6, l, m, i, j, k, p, DMnx, DMny, DMnz, DMnd, count, index, Q_len,
        FDn, Nx, Ny, Nz, i_global, j_global, k_global, nbr_i, is, ie, js, je,
        ks, ke, nx_cor, ny_cor, nz_cor, nd_cor, is_phi, ie_phi, js_phi, je_phi,
        ks_phi, ke_phi, nx_phi, ny_phi, nz_phi, nd_phi, i_phi, j_phi, k_phi,
        i_DM, j_DM, k_DM, DMCorVert[6][6];
    double *Qlm, Lx, Ly, Lz, *r_pos_x, *r_pos_y, *r_pos_z, *r_pos_r, *r_pow_l,
           *Ylm, *phi, x, y, z, r, x2, y2, z2;
    
    FDn = pSPARC->order / 2;
    
    DMnx = pSPARC->Nx_d;
    DMny = pSPARC->Ny_d;
    DMnz = pSPARC->Nz_d;
    DMnd = pSPARC->Nd_d;
    Lx = pSPARC->range_x;
    Ly = pSPARC->range_y;
    Lz = pSPARC->range_z;
    Nx = pSPARC->Nx;
    Ny = pSPARC->Ny;
    Nz = pSPARC->Nz;

    /* find multipole moments Qlm */
    r_pos_x = (double *)malloc( sizeof(double) * DMnd );
    r_pos_y = (double *)malloc( sizeof(double) * DMnd );
    r_pos_z = (double *)malloc( sizeof(double) * DMnd );
    r_pos_r = (double *)malloc( sizeof(double) * DMnd );
    
    // find distance between the center of the domain and finite-difference grids
    count = 0; 
    for (k = 0; k < DMnz; k++) {
        k_global = k + pSPARC->DMVertices[4]; // global coord
        z = k_global * pSPARC->delta_z - Lz/2.0; 
        z2 = z * z;
        for (j = 0; j < DMny; j++) {
            j_global = j + pSPARC->DMVertices[2]; // global coord
            y = j_global * pSPARC->delta_y - Ly/2.0; 
            y2 = y * y;
            for (i = 0; i < DMnx; i++) {
                i_global = i + pSPARC->DMVertices[0]; // global coord
                x = i_global * pSPARC->delta_x - Lx/2.0;
                x2 = x * x; 
                r_pos_x[count] = x;
                r_pos_y[count] = y;
                r_pos_z[count] = z;
                r_pos_r[count] = sqrt(x2 + y2 + z2);
                count++;
            }
        }
    }    

    Ylm = (double *)malloc( sizeof(double) * DMnd );
    Q_len = (LMAX+1)*(LMAX+1);
    Qlm = (double *)calloc( Q_len, sizeof(double) );
    r_pow_l = (double *)malloc( sizeof(double) * DMnd );
    for (i = 0; i < DMnd; i++) r_pow_l[i] = 1.0; // init to 1
    index = 0;
    for (l = 0; l <= LMAX; l++) {
        // find r^l
        if (l) {
            for (i = 0; i < DMnd; i++) r_pow_l[i] *= r_pos_r[i];
        }
        for (m = -l; m <= l; m++) {
            RealSphericalHarmonic(DMnd, r_pos_x, r_pos_y, r_pos_z, r_pos_r, l, m, Ylm);
            Qlm[index] = 0.0;
            for (i = 0; i < DMnd; i++) 
                Qlm[index] += r_pow_l[i] * f[i] * Ylm[i];
            Qlm[index] *= pSPARC->dV;
            index++;
        }
    } 
    free(r_pos_x); free(r_pos_y); free(r_pos_z); free(r_pos_r); free(Ylm); free(r_pow_l);

    // do allreduce to sum over all phi process
    MPI_Allreduce(MPI_IN_PLACE, Qlm, Q_len, MPI_DOUBLE, MPI_SUM, pSPARC->dmcomm_phi); 

    #ifdef DEBUG
    // Calculate dipole moment based on the Qlm values
    // dipole moment vector = (\int {x * rho} dV, \int {y * rho} dV, \int {z * rho} dV)
    // dipole moment value is defined as the length of the dipole moment vector
    // Note:
    //   1. Qlm is defined using the spherical harmonics, which contains a factor of sqrt(3/(4*pi))
    //   2. Qlm is calculated using f = 4*pi*(rho+b), which includes another factor of 4*pi
    double dipole_moment = sqrt(Qlm[1] * Qlm[1] + Qlm[2] * Qlm[2] + Qlm[3] * Qlm[3]);
    const double Debye2eBohr = 0.3934303; // 1 Debye = 0.3934303 e*Bohr
    const double eBohr2Debye = 1.0 / Debye2eBohr;
    dipole_moment = dipole_moment / (4.0*M_PI*sqrt(3.0/4.0/M_PI)) * eBohr2Debye;
    if (rank == 0) printf("Dipole moment: %.6f (Debye)\n", dipole_moment);
    #endif

	/* find "charge correction" (boudary correction) */
    // define the “correction domain” which contributes to the charge correction. i.e. 0 to FDn-1 and
    // nx-FDn nx-1 in each direction.
    DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
    DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;  
    DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
    DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
    DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
    DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

    for (i = 0; i < DMnd; i++) d_cor[i] = 0.0; // init correction to 0
       
    // find correction contribution from each side
    for (nbr_i = 0; nbr_i < 6; nbr_i++) {
        is = max(DMCorVert[nbr_i][0], pSPARC->DMVertices[0]);
        ie = min(DMCorVert[nbr_i][1], pSPARC->DMVertices[1]);
        js = max(DMCorVert[nbr_i][2], pSPARC->DMVertices[2]);
        je = min(DMCorVert[nbr_i][3], pSPARC->DMVertices[3]);
        ks = max(DMCorVert[nbr_i][4], pSPARC->DMVertices[4]);
        ke = min(DMCorVert[nbr_i][5], pSPARC->DMVertices[5]);
        nx_cor = ie - is + 1;
        ny_cor = je - js + 1;
        nz_cor = ke - ks + 1;
        nd_cor = nx_cor * ny_cor * nz_cor;
        if (nd_cor <= 0) continue;
        
        // find the region of phi that have contribution to the correction domain
        is_phi = is; ie_phi = ie;
        js_phi = js; je_phi = je;
        ks_phi = ks; ke_phi = ke;
        switch (nbr_i) {
            case 0:
                is_phi = is - FDn; ie_phi = -1; break;
            case 1:
                is_phi = Nx; ie_phi = ie + FDn; break;
            case 2:
                js_phi = js - FDn; je_phi = -1; break;
            case 3:
                js_phi = Ny; je_phi = je + FDn; break;
            case 4:
                ks_phi = ks - FDn; ke_phi = -1; break;
            case 5:
                ks_phi = Nz; ke_phi = ke + FDn; break; 
        }
        
        nx_phi = ie_phi - is_phi + 1;
        ny_phi = je_phi - js_phi + 1;
        nz_phi = ke_phi - ks_phi + 1;
        nd_phi = nx_phi * ny_phi * nz_phi;
        
        // calculate electrostatic potential "phi" inside
        phi = (double *)calloc( nd_phi , sizeof(double) );
        Ylm = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_x = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_y = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_z = (double *)malloc( sizeof(double) * nd_phi );
        r_pos_r = (double *)malloc( sizeof(double) * nd_phi );
        r_pow_l = (double *)malloc( sizeof(double) * nd_phi );
    
        count = 0;
        for (k = 0; k < nz_phi; k++) {
            z = (k + ks_phi) * pSPARC->delta_z - Lz*0.5; 
            for (j = 0; j < ny_phi; j++) {
                y = (j + js_phi) * pSPARC->delta_y - Ly*0.5;
                for (i = 0; i < nx_phi; i++) {
                    x = (i + is_phi) * pSPARC->delta_x - Lx*0.5;
                    r = sqrt(x * x + y * y + z * z);
                    r_pos_x[count] = x;
                    r_pos_y[count] = y;
                    r_pos_z[count] = z;
                    r_pos_r[count] = r;
                    count++;
                }
            }
        } 
        
        for (i = 0; i < nd_phi; i++) r_pow_l[i] = 1.0; // init r_pow_l to 1
        index = 0;
        for (l = 0; l <= LMAX; l++) {
            // find r^(l+1)
            for (i = 0; i < nd_phi; i++) r_pow_l[i] *= r_pos_r[i];
            for (m = -l; m <= l; m++) {
                RealSphericalHarmonic(nd_phi, r_pos_x, r_pos_y, r_pos_z, r_pos_r, l, m, Ylm);
                for (i = 0; i < nd_phi; i++) 
                    phi[i] += 1.0 / ((2*l+1) * r_pow_l[i]) * Ylm[i] * Qlm[index];
                index++;
            }
        } 
        free(Ylm); free(r_pos_x); free(r_pos_y); free(r_pos_z); free(r_pos_r); free(r_pow_l);

        // calculate the correction "d_cor"
        for (k = ks; k <= ke; k++) {
            k_phi = k - ks_phi; k_DM = k - pSPARC->DMVertices[4];
            for (j = js; j <= je; j++) {
                j_phi = j - js_phi; j_DM = j - pSPARC->DMVertices[2];
                for (i = is; i <= ie; i++) {
                    i_phi = i - is_phi; i_DM = i - pSPARC->DMVertices[0];
                    for (p = 1; p <= FDn; p++) {
                        switch (nbr_i) {
                            case 0:
                                if ((i-p) < 0) 
                                    d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi);
                                break;
                            case 1:
                                if ((i+p) >= Nx) 
                                    d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi);
                                break;
                            case 2:
                                if ((j-p) < 0) 
                                    d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi);
                                break;
                            case 3:
                                if ((j+p) >= Ny) 
                                    d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi);
                                break;
                            case 4:
                                if ((k-p) < 0) 
                                    d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p);
                                break;
                            case 5:
                                if ((k+p) >= Nz) 
                                    d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p);
                                break;
                        }
                    }
                }
            }
        }
        free(phi);
    }
    free(Qlm);
#undef d_cor
#undef phi
}



/**
 * @brief   Use partial dipole to correct boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          with periodic BCs in 2 directions, and Dirichlet BC in the other direction (surface).
 *          So that when discretized in finite difference with Dirichlet BC, the equation will be
 *                                  - DiscreteLaplacian phi = f - d.
 *          It is required that f decays to zero on the Dirichlet boundary.
 *
 *          Note that this is only done in "phi-domain".
 */
void PartrialDipole_surface(SPARC_OBJ *pSPARC, double *f, double *d_cor) {
#define d_cor(i,j,k) d_cor[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define f(i,j,k) f[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define phi(i,j,k) phi[(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]
	if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

	int FDn = pSPARC->order / 2;
	int DMnx = pSPARC->Nx_d;
	int DMny = pSPARC->Ny_d;
	int DMnz = pSPARC->Nz_d;
	int DMnd = pSPARC->Nd_d;
	int Nx = pSPARC->Nx;
	int Ny = pSPARC->Ny;
	int Nz = pSPARC->Nz;

	// define the “correction domain” which contributes to the charge correction, 
	// i.e. 0 to FDn-1 and, nx-FDn nx-1 in each direction.
	int DMCorVert[6][6];
	DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
	DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;  
	DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
	DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
	DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
	DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

	// init correction to 0
	for (int i = 0; i < DMnd; i++) d_cor[i] = 0.0; 

	//** Find rho_av = int (rho + b) dXdY **//
	int gridsizes[3], DMsizes[3];
	gridsizes[0] = pSPARC->Nx;
	gridsizes[1] = pSPARC->Ny;
	gridsizes[2] = pSPARC->Nz;
	DMsizes[0] = DMnx;
	DMsizes[1] = DMny;
	DMsizes[2] = DMnz;
	double cellsizes[3], meshsizes[3];
	cellsizes[0] = pSPARC->range_x;
	cellsizes[1] = pSPARC->range_y;
	cellsizes[2] = pSPARC->range_z;
	meshsizes[0] = pSPARC->delta_x;
	meshsizes[1] = pSPARC->delta_y;
	meshsizes[2] = pSPARC->delta_z;

	// find Dirichlet direction and call it the Z direction
	int dir_Z = (pSPARC->BCx == 1) ? 0 : (pSPARC->BCy == 1 ? 1 : 2);
	int dir_X = (dir_Z + 1) % 3;
	int dir_Y = (dir_X + 1) % 3;

	// once we find the direction, we assume that direction is the Z
	// direction, the other two directions are then called X, Y
	int NX = gridsizes[dir_X]; 
	int NY = gridsizes[dir_Y]; 
	// int NZ = gridsizes[dir_Z];
	int NXY = NX * NY;
	// int DMnX = DMsizes[dir_X];
	// int DMnY = DMsizes[dir_Y];
	int DMnZ = DMsizes[dir_Z];
	double LX = cellsizes[dir_X];  
	double LY = cellsizes[dir_Y];  
	// double LZ = cellsizes[dir_Z];
	// double dX = meshsizes[dir_X]; 
	// double dY = meshsizes[dir_Y]; 
	double dZ = meshsizes[dir_Z];
	double A_XY = LX * LY; // area of (X,Y) surface, neglecting the Jacobian

	// Find rho_av = int (rho + b) dXdY / int (1) dXdY locally
	double *rho_av = (double *)calloc( sizeof(double), DMnZ);
	
	// first find sum
	int ind_orig[3], k_new;
	for (int k = 0; k < DMnz; k++) {
		for (int j = 0; j < DMny; j++) {
			for (int i = 0; i < DMnx; i++) {
				ind_orig[0] = i;
				ind_orig[1] = j;
				ind_orig[2] = k;
				k_new = ind_orig[dir_Z];
				rho_av[k_new] += f(i,j,k); // note here f = 4*pi* (rho + b)
			}
		}
	}

	// find average, note here we assume mesh is uniform, otherwise
	// use rho_av = int (rho + b) dXdY / int (1) dXdY
	for (int k = 0; k < DMnZ; k++) {
		rho_av[k] /= (NXY*4*M_PI);
	}

	// Create sub-comm slices of the Cartesian topology
	int remain_dims[3]; // which dimensions to keep
	remain_dims[dir_X] = 1;
	remain_dims[dir_Y] = 1;
	remain_dims[dir_Z] = 0;
	MPI_Comm XY_comm;
	MPI_Cart_sub(pSPARC->dmcomm_phi, remain_dims, &XY_comm);

	// sum over processors in the sub-slices in the X-Y plane
	MPI_Allreduce(MPI_IN_PLACE, rho_av, DMnZ, MPI_DOUBLE, 
		MPI_SUM, XY_comm);
	
	//** evaluate P(0) = int_0^{LZ} rho_av(Z)*Z dZ **//
	// find P0 locally
	double P0 = 0.0;
	for (int k = 0; k < DMnZ; k++) {
		double Z = (k + pSPARC->DMVertices[2*dir_Z]) * dZ;
		P0 += rho_av[k] * Z * dZ;
	}

	// create sub-communicators in the Z direction
	remain_dims[dir_X] = 0;
	remain_dims[dir_Y] = 0;
	remain_dims[dir_Z] = 1;
	MPI_Comm Z_comm;
	MPI_Cart_sub(pSPARC->dmcomm_phi, remain_dims, &Z_comm);

	// sum over processors in the Z direction
	MPI_Allreduce(MPI_IN_PLACE, &P0, 1, MPI_DOUBLE, 
		MPI_SUM, Z_comm);

	MPI_Comm_free(&XY_comm);
	MPI_Comm_free(&Z_comm);

	int nbr_BCs[6];
	nbr_BCs[0] = nbr_BCs[1] = pSPARC->BCx;
	nbr_BCs[2] = nbr_BCs[3] = pSPARC->BCy;
	nbr_BCs[4] = nbr_BCs[5] = pSPARC->BCz;

	// find correction contribution from each side (in Dirichlet BC side)
	for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
		// skip if BC is periodic in this side
		if (nbr_BCs[nbr_i] == 0) continue; 

		int is = max(DMCorVert[nbr_i][0], pSPARC->DMVertices[0]);
		int ie = min(DMCorVert[nbr_i][1], pSPARC->DMVertices[1]);
		int js = max(DMCorVert[nbr_i][2], pSPARC->DMVertices[2]);
		int je = min(DMCorVert[nbr_i][3], pSPARC->DMVertices[3]);
		int ks = max(DMCorVert[nbr_i][4], pSPARC->DMVertices[4]);
		int ke = min(DMCorVert[nbr_i][5], pSPARC->DMVertices[5]);
		int nx_cor = ie - is + 1;
		int ny_cor = je - js + 1;
		int nz_cor = ke - ks + 1;
		int nd_cor = nx_cor * ny_cor * nz_cor;
		if (nd_cor <= 0) continue;

		// find the region of phi that have contribution to the correction domain
		int is_phi = is, ie_phi = ie;
		int js_phi = js, je_phi = je;
		int ks_phi = ks, ke_phi = ke;
		switch (nbr_i) {
			case 0:
				is_phi = is - FDn; ie_phi = -1; break;
			case 1:
				is_phi = Nx; ie_phi = ie + FDn; break;
			case 2:
				js_phi = js - FDn; je_phi = -1; break;
			case 3:
				js_phi = Ny; je_phi = je + FDn; break;
			case 4:
				ks_phi = ks - FDn; ke_phi = -1; break;
			case 5:
				ks_phi = Nz; ke_phi = ke + FDn; break; 
		}

		int nx_phi = ie_phi - is_phi + 1;
		int ny_phi = je_phi - js_phi + 1;
		int nz_phi = ke_phi - ks_phi + 1;
		int nd_phi = nx_phi * ny_phi * nz_phi;

		// calculate electrostatic potential "phi" inside
		double *phi = (double *)calloc( nd_phi , sizeof(double) );

		int count = 0;
		for (int k = 0; k < nz_phi; k++) {
			double z = (k + ks_phi) * pSPARC->delta_z; 
			for (int j = 0; j < ny_phi; j++) {
				double y = (j + js_phi) * pSPARC->delta_y;
				for (int i = 0; i < nx_phi; i++) {
					double x = (i + is_phi) * pSPARC->delta_x;
					double coords[3] = {x,y,z};
					double Z = coords[dir_Z];
					// phi = phi_av = 2*pi*int_0^{Z_cell} rho_av(Z') |Z-Z'| dZ'
					// note: int_0^{Z_cell} rho_av(Z') dZ' = NetCharge/A_XY
					double phi_av = -2.0 * M_PI * (pSPARC->NetCharge/A_XY*Z - P0);
					if (Z <= 0.0) phi_av *= -1.0; 
					phi[count] = phi_av;
					count++;
				}
			}
		} 

		// calculate the correction "d_cor"
		for (int k = ks; k <= ke; k++) {
			int k_phi = k - ks_phi, k_DM = k - pSPARC->DMVertices[4];
			for (int j = js; j <= je; j++) {
				int j_phi = j - js_phi, j_DM = j - pSPARC->DMVertices[2];
				for (int i = is; i <= ie; i++) {
					int i_phi = i - is_phi, i_DM = i - pSPARC->DMVertices[0];
					for (int p = 1; p <= FDn; p++) {
						switch (nbr_i) {
							case 0:
								if ((i-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi);
								break;
							case 1:
								if ((i+p) >= Nx) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi);
								break;
							case 2:
								if ((j-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi);
								break;
							case 3:
								if ((j+p) >= Ny) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi);
								break;
							case 4:
								if ((k-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p);
								break;
							case 5:
								if ((k+p) >= Nz) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p);
								break;
						}
					}
				}
			}
		}
		free(phi);
	}
	free(rho_av);
#undef d_cor
#undef phi
#undef f
}



/**
 * @brief   Use partial dipole to correct boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          with periodic BCs in 1 directions, and Dirichlet BC in the other two directions (wire).
 *          So that when discretized in finite difference with Dirichlet BC, the equation will be
 *                                  - DiscreteLaplacian phi = f - d.
 *          It is required that f decays to zero on the Dirichlet boundary.
 *
 *          Note that this is only done in "phi-domain".
 */
void PartrialDipole_wire(SPARC_OBJ *pSPARC, double *f, double *d_cor) {
#define d_cor(i,j,k) d_cor[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define f(i,j,k) f[(k)*DMnx*DMny+(j)*DMnx+(i)]
#define phi(i,j,k) phi[(k)*nx_phi*ny_phi+(j)*nx_phi+(i)]
	if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 

	int FDn = pSPARC->order / 2;
	int DMnx = pSPARC->Nx_d;
	int DMny = pSPARC->Ny_d;
	int DMnz = pSPARC->Nz_d;
	int DMnd = pSPARC->Nd_d;
	int Nx = pSPARC->Nx;
	int Ny = pSPARC->Ny;
	int Nz = pSPARC->Nz;

	// define the “correction domain” which contributes to the charge correction. i.e. 0 to FDn-1 and
	// nx-FDn nx-1 in each direction.
	int DMCorVert[6][6];
	DMCorVert[0][0]=0;      DMCorVert[0][1]=FDn-1; DMCorVert[0][2]=0;       DMCorVert[0][3]=Ny-1;  DMCorVert[0][4]=0;       DMCorVert[0][5]=Nz-1;
	DMCorVert[1][0]=Nx-FDn; DMCorVert[1][1]=Nx-1;  DMCorVert[1][2]=0;       DMCorVert[1][3]=Ny-1;  DMCorVert[1][4]=0;       DMCorVert[1][5]=Nz-1;  
	DMCorVert[2][0]=0;      DMCorVert[2][1]=Nx-1;  DMCorVert[2][2]=0;       DMCorVert[2][3]=FDn-1; DMCorVert[2][4]=0;       DMCorVert[2][5]=Nz-1;
	DMCorVert[3][0]=0;      DMCorVert[3][1]=Nx-1;  DMCorVert[3][2]=Ny-FDn;  DMCorVert[3][3]=Ny-1;  DMCorVert[3][4]=0;       DMCorVert[3][5]=Nz-1;
	DMCorVert[4][0]=0;      DMCorVert[4][1]=Nx-1;  DMCorVert[4][2]=0;       DMCorVert[4][3]=Ny-1;  DMCorVert[4][4]=0;       DMCorVert[4][5]=FDn-1;
	DMCorVert[5][0]=0;      DMCorVert[5][1]=Nx-1;  DMCorVert[5][2]=0;       DMCorVert[5][3]=Ny-1;  DMCorVert[5][4]=Nz-FDn;  DMCorVert[5][5]=Nz-1;

	// init correction to 0
	for (int i = 0; i < DMnd; i++) d_cor[i] = 0.0; 

	//** Find rho_av = int (rho + b) dZ **//
	int gridsizes[3], DMsizes[3];
	gridsizes[0] = pSPARC->Nx;
	gridsizes[1] = pSPARC->Ny;
	gridsizes[2] = pSPARC->Nz;
	DMsizes[0] = DMnx;
	DMsizes[1] = DMny;
	DMsizes[2] = DMnz;
	double meshsizes[3];
	meshsizes[0] = pSPARC->delta_x;
	meshsizes[1] = pSPARC->delta_y;
	meshsizes[2] = pSPARC->delta_z;

	// find Periodic direction and call it the Z direction
	int dir_Z = (pSPARC->BCx == 0) ? 0 : (pSPARC->BCy == 0 ? 1 : 2);
	int dir_X = (dir_Z + 1) % 3;
	int dir_Y = (dir_X + 1) % 3;
	
	int nbr_BCs[6];
	nbr_BCs[0] = nbr_BCs[1] = pSPARC->BCx;
	nbr_BCs[2] = nbr_BCs[3] = pSPARC->BCy;
	nbr_BCs[4] = nbr_BCs[5] = pSPARC->BCz;

	// once we find the direction, we assume that direction is the Z
	// direction, the other two directions are then called X, Y
	int NX = gridsizes[dir_X]; 
	int NY = gridsizes[dir_Y]; 
	int NZ = gridsizes[dir_Z];
	// int NXY = NX * NY;
	int DMnX = DMsizes[dir_X];
	int DMnY = DMsizes[dir_Y];
	// int DMnZ = DMsizes[dir_Z];
	// double LX = cellsizes[dir_X];  
	// double LY = cellsizes[dir_Y];  
	// double LZ = cellsizes[dir_Z];
	double dX = meshsizes[dir_X]; 
	double dY = meshsizes[dir_Y]; 
	// double dZ = meshsizes[dir_Z];
	// double A_XY = LX * LY; // area of (X,Y) surface, neglecting the Jacobian

	// Find rho_av = int (rho + b) dZ / int (1) dZ locally
	double *rho_av = (double *)calloc( sizeof(double), DMnX*DMnY);
	
	// first find sum
	int ind_orig[3], i_new, j_new;
	for (int k = 0; k < DMnz; k++) {
		for (int j = 0; j < DMny; j++) {
			for (int i = 0; i < DMnx; i++) {
				ind_orig[0] = i;
				ind_orig[1] = j;
				ind_orig[2] = k;
				i_new = ind_orig[dir_X];
				j_new = ind_orig[dir_Y];
				rho_av[j_new*DMnX+i_new] += f(i,j,k); // note here f = 4*pi* (rho + b)
			}
		}
	}

	// find average, note here we assume mesh is uniform, otherwise
	// use rho_av = int (rho + b) dZ / int (1) dZ	
	for (int i = 0; i < DMnX*DMnY; i++) {
		rho_av[i] /= (NZ*4*M_PI);
	}

	// Create sub-comm slices of the Cartesian topology
	int remain_dims[3]; // which dimensions to keep
	remain_dims[dir_X] = 1;
	remain_dims[dir_Y] = 1;
	remain_dims[dir_Z] = 0;
	MPI_Comm XY_comm;
	MPI_Cart_sub(pSPARC->dmcomm_phi, remain_dims, &XY_comm);

	// create sub-communicators in the Z direction
	remain_dims[dir_X] = 0;
	remain_dims[dir_Y] = 0;
	remain_dims[dir_Z] = 1;
	MPI_Comm Z_comm;
	MPI_Cart_sub(pSPARC->dmcomm_phi, remain_dims, &Z_comm);

	// sum over processors in the sub-slices in the Z direction
	MPI_Allreduce(MPI_IN_PLACE, rho_av, DMnX*DMnY, MPI_DOUBLE, 
		MPI_SUM, Z_comm);
	
	//** evaluate V_av(X,Y) = int rho_av(X,Y)*ln((X-X')^2+(Y-Y')^2) dX'dY' **//
	double *V_av =  (double *)calloc(2 * FDn * (NX+NY), sizeof(double));
	double *V_av_nbr[6];
	V_av_nbr[dir_X*2]   = &V_av[0];
	V_av_nbr[dir_X*2+1] = &V_av[FDn*NY];
	V_av_nbr[dir_Y*2]   = &V_av[FDn*2*NY];
	V_av_nbr[dir_Y*2+1] = &V_av[FDn*(2*NY+NX)];

	int DMVert_phi[6][6];
	for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
		DMVert_phi[nbr_i][0] = DMCorVert[nbr_i][0];
		DMVert_phi[nbr_i][1] = DMCorVert[nbr_i][1];
		DMVert_phi[nbr_i][2] = DMCorVert[nbr_i][2];
		DMVert_phi[nbr_i][3] = DMCorVert[nbr_i][3];
		DMVert_phi[nbr_i][4] = DMCorVert[nbr_i][4];
		DMVert_phi[nbr_i][5] = DMCorVert[nbr_i][5];
		switch (nbr_i) {
			case 0:
				DMVert_phi[nbr_i][0] -= FDn; DMVert_phi[nbr_i][1] -= FDn; break;
			case 1:
				DMVert_phi[nbr_i][0] += FDn; DMVert_phi[nbr_i][1] += FDn; break;
			case 2:
				DMVert_phi[nbr_i][2] -= FDn; DMVert_phi[nbr_i][3] -= FDn; break;
			case 3:
				DMVert_phi[nbr_i][2] += FDn; DMVert_phi[nbr_i][3] += FDn; break;
			case 4:
				DMVert_phi[nbr_i][4] -= FDn; DMVert_phi[nbr_i][5] -= FDn; break;
			case 5:
				DMVert_phi[nbr_i][4] += FDn; DMVert_phi[nbr_i][5] += FDn; break; 
		}
	}

	// find V_av locally
	for (int jp = 0; jp < DMnY; jp++) {
		double Yp = (jp + pSPARC->DMVertices[2*dir_Y]) * dY;
		for (int ip = 0; ip < DMnX; ip++) {
			double Xp = (ip + pSPARC->DMVertices[2*dir_X]) * dX;
			double rho_av_xp_yp = rho_av[jp*DMnX+ip];
			for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
				if (nbr_BCs[nbr_i] == 0) continue;
				int Is_phi_full = DMVert_phi[nbr_i][dir_X*2];
				int Ie_phi_full = DMVert_phi[nbr_i][dir_X*2+1];
				int Js_phi_full = DMVert_phi[nbr_i][dir_Y*2];
				int Je_phi_full = DMVert_phi[nbr_i][dir_Y*2+1];
				int nX_phi_full = Ie_phi_full - Is_phi_full + 1;
				// int nY_phi = je_phi - js_phi + 1;
				for (int j = Js_phi_full; j <= Je_phi_full; j++) {
					double Y = j * dY;
					for (int i = Is_phi_full; i <= Ie_phi_full; i++) {
						double X = i * dX;
						double r2 = (X-Xp)*(X-Xp) + (Y-Yp)*(Y-Yp);
						V_av_nbr[nbr_i][(j- Js_phi_full)*nX_phi_full+(i-Is_phi_full)] -= rho_av_xp_yp * log(r2) * (dX*dY); 
					}
				}
			}
		}
	}

	// sum over processors in the sub-slices in the X-Y plane
	MPI_Allreduce(MPI_IN_PLACE, V_av, 2*FDn*(NX+NY), MPI_DOUBLE, 
		MPI_SUM, XY_comm);

	MPI_Comm_free(&XY_comm);
	MPI_Comm_free(&Z_comm);


	// find correction contribution from each side (in Dirichlet BC side)
	for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
		// skip if BC is periodic in this side
		if (nbr_BCs[nbr_i] == 0) continue; 

		int is = max(DMCorVert[nbr_i][0], pSPARC->DMVertices[0]);
		int ie = min(DMCorVert[nbr_i][1], pSPARC->DMVertices[1]);
		int js = max(DMCorVert[nbr_i][2], pSPARC->DMVertices[2]);
		int je = min(DMCorVert[nbr_i][3], pSPARC->DMVertices[3]);
		int ks = max(DMCorVert[nbr_i][4], pSPARC->DMVertices[4]);
		int ke = min(DMCorVert[nbr_i][5], pSPARC->DMVertices[5]);
		int nx_cor = ie - is + 1;
		int ny_cor = je - js + 1;
		int nz_cor = ke - ks + 1;
		int nd_cor = nx_cor * ny_cor * nz_cor;
		if (nd_cor <= 0) continue;

		// find the region of phi that have contribution to the correction domain
		int is_phi = is, ie_phi = ie;
		int js_phi = js, je_phi = je;
		int ks_phi = ks, ke_phi = ke;
		switch (nbr_i) {
			case 0:
				is_phi = is - FDn; ie_phi = -1; break;
			case 1:
				is_phi = Nx; ie_phi = ie + FDn; break;
			case 2:
				js_phi = js - FDn; je_phi = -1; break;
			case 3:
				js_phi = Ny; je_phi = je + FDn; break;
			case 4:
				ks_phi = ks - FDn; ke_phi = -1; break;
			case 5:
				ks_phi = Nz; ke_phi = ke + FDn; break; 
		}

		int nx_phi = ie_phi - is_phi + 1;
		int ny_phi = je_phi - js_phi + 1;
		int nz_phi = ke_phi - ks_phi + 1;
		int nd_phi = nx_phi * ny_phi * nz_phi;

		// calculate electrostatic potential "phi" inside
		double *phi = (double *)calloc( nd_phi , sizeof(double) );
		int nX_phi_full = DMVert_phi[nbr_i][dir_X*2+1] - DMVert_phi[nbr_i][dir_X*2] + 1;

		int count = 0;
		for (int k = ks_phi; k <= ke_phi; k++) {
			for (int j = js_phi; j <= je_phi; j++) {
				for (int i = is_phi; i <= ie_phi; i++) {
					int inds[3] = {i,j,k};
					int I_phi_full = inds[dir_X] - DMVert_phi[nbr_i][dir_X*2];
					int J_phi_full = inds[dir_Y] - DMVert_phi[nbr_i][dir_Y*2];
					phi[count] = V_av_nbr[nbr_i][J_phi_full*nX_phi_full+I_phi_full];
					count++;
				}
			}
		}

		// calculate the correction "d_cor"
		for (int k = ks; k <= ke; k++) {
			int k_phi = k - ks_phi, k_DM = k - pSPARC->DMVertices[4];
			for (int j = js; j <= je; j++) {
				int j_phi = j - js_phi, j_DM = j - pSPARC->DMVertices[2];
				for (int i = is; i <= ie; i++) {
					int i_phi = i - is_phi, i_DM = i - pSPARC->DMVertices[0];
					for (int p = 1; p <= FDn; p++) {
						switch (nbr_i) {
							case 0:
								if ((i-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi-p,j_phi,k_phi);
								break;
							case 1:
								if ((i+p) >= Nx) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_x[p] * phi(i_phi+p,j_phi,k_phi);
								break;
							case 2:
								if ((j-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi-p,k_phi);
								break;
							case 3:
								if ((j+p) >= Ny) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_y[p] * phi(i_phi,j_phi+p,k_phi);
								break;
							case 4:
								if ((k-p) < 0) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi-p);
								break;
							case 5:
								if ((k+p) >= Nz) 
									d_cor(i_DM,j_DM,k_DM) -= pSPARC->D2_stencil_coeffs_z[p] * phi(i_phi,j_phi,k_phi+p);
								break;
						}
					}
				}
			}
		}
		free(phi);
	}
	free(rho_av);
	free(V_av);
#undef d_cor
#undef phi
#undef f
}
