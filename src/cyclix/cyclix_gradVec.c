/**
 * @file    cyclix_gradVec.c
 * @brief   This file contains the functions for performing gradient matrix times vector.
 *
 * @author  Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          
 * Copyright (c) 2017 Material Physics & Mechanics Group at Georgia Tech.
 */


#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

#include "initialization.h"
#include "cyclix_gradVec.h"
#include "gradVecRoutines.h"
#include "gradVecRoutinesKpt.h"
#include "isddft.h"
#include "assert.h"


/*
@ brief: function to calculate the gradients in cartesian directions for cyclix systems
*/

void Gradient_vectors_dir_cyclix(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                                 const int ncol, const double c, const double *X, const int ldi,
                                 double *DX, const int ldo, const int dir, MPI_Comm comm, const int* dims)
{
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
    
    double *DX1 = (double *)malloc( DMnd * sizeof(double));
    double *DX2 = (double *)malloc( DMnd * sizeof(double));

    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;

    double xc, yc, zc;

    switch (dir) {
        case 0:
            for (int n = 0; n < ncol; n++){
                Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX1, DMnd, 0, comm, dims);
                Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX2, DMnd, 1, comm, dims);

                int count = 0;
                for(int k = 0; k < DMnz; k++) {
                    double z = (k + DMVertices[4]) * dz;
                    for(int j = 0; j < DMny; j++) {
                        double y = (j + DMVertices[2]) * dy;
                        for(int i = 0; i < DMnx; i++) {
                            double x = pSPARC->xin + (i + DMVertices[0]) * dx;
                            xc = x; yc = y; zc = z;
                            nonCart2Cart_coord(pSPARC, &xc, &yc, &zc);
                            DX[n*(unsigned)ldo+count] = DX1[count] * xc/x - DX2[count] * yc/(x*x);
                            count++;  
                        }
                    }
                }
            }
            break;
        case 1:
            for (int n = 0; n < ncol; n++){
                Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX1, DMnd, 0, comm, dims);  
                Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX2, DMnd, 1, comm, dims);

                int count = 0;
                for(int k = 0; k < DMnz; k++) {
                    double z = (k + DMVertices[4]) * dz;
                    for(int j = 0; j < DMny; j++) {
                        double y = (j + DMVertices[2]) * dy;
                        for(int i = 0; i < DMnx; i++) {
                            double x = pSPARC->xin + (i + DMVertices[0]) * dx;
                            xc = x; yc = y; zc = z;
                            nonCart2Cart_coord(pSPARC, &xc, &yc, &zc);
                            DX[n*(unsigned)ldo+count] = DX1[count] * yc/x + DX2[count] * xc/(x*x);
                            count++;  
                        }
                    }
                }
            }
            break;
        case 2:
            for (int n = 0; n < ncol; n++){
                Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX1, DMnd, 1, comm, dims);
                Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX2, DMnd, 2, comm, dims);
                int count = 0;
                for(int k = 0; k < DMnz; k++) {
                    for(int j = 0; j < DMny; j++) {
                        for(int i = 0; i < DMnx; i++) {
                            DX[n*(unsigned)ldo+count] = -pSPARC->twist * DX1[count] + DX2[count];
                            count++;
                        }
                    }
                }
            }
            break;
        default: printf("gradient dir must be either 0, 1 or 2!\n");
                 break;
    }

    free(DX1);
    free(DX2);
}



void Gradient_vectors_dir_with_rotfac(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                                 const int ncol, const double c, const double *X1, const double *X2, const int ldi,
                                 double *DX, const int ldo, const int dir, MPI_Comm comm)
{
    assert(dir == 0 || dir == 1); 
    int nproc;
    MPI_Comm_size(comm, &nproc);

    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else 
        dims[0] = dims[1] = dims[2] = 1;

    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;

    double *DX1 = (double *)malloc( DMnd * sizeof(double));
    double *DX2 = (double *)malloc( DMnd * sizeof(double));

    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;

    double xc, yc, zc;

    for (int n = 0; n < ncol; n++){
        Gradient_vec_dir(pSPARC, DMnd, DMVertices, 1, c, X1+n*(unsigned)ldi, ldi, DX1, DMnd, 0, comm, dims);
        Gradient_vec_dir_rotfac(pSPARC, DMnd, DMVertices, 1, c, X1+n*(unsigned)ldi, X2+n*(unsigned)ldi, ldi, DX2, DMnd, 1, comm, dims, dir);

        int count = 0;
        for(int k = 0; k < DMnz; k++) {
            double z = (k + DMVertices[4]) * dz;
            for(int j = 0; j < DMny; j++) {
                double y = (j + DMVertices[2]) * dy;
                for(int i = 0; i < DMnx; i++) {
                    double x = pSPARC->xin + (i + DMVertices[0]) * dx;
                    xc = x; yc = y; zc = z;
                    nonCart2Cart_coord(pSPARC, &xc, &yc, &zc);
                    if (dir == 0)
                        DX[n*(unsigned)ldo+count] = DX1[count] * xc/x - DX2[count] * yc/(x*x);
                    else
                        DX[n*(unsigned)ldo+count] = DX1[count] * yc/x + DX2[count] * xc/(x*x);
                    count++;  
                }
            }
        }
    }

    free(DX1);
    free(DX2);
}




/*
@ brief: function to calculate the gradients in cartesian directions for cyclix systems
*/

void Gradient_vectors_dir_kpt_cyclix(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                                     const int ncol, const double c, const double _Complex *X, const int ldi, 
                                     double _Complex *DX, const int ldo, const int dir, const double *kpt_vec, MPI_Comm comm, const int *dims)
{
    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;

    double _Complex *DX1 = (double _Complex *)malloc( DMnd * sizeof(double _Complex));
    double _Complex *DX2 = (double _Complex *)malloc( DMnd * sizeof(double _Complex));

    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;

    double xc, yc, zc;

    switch (dir) {
        case 0:
            for (int n = 0; n < ncol; n++){
                Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX1, DMnd, 0, &kpt_vec[0], comm, dims);
                Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX2, DMnd, 1, &kpt_vec[1], comm, dims);
                int count = 0;
                for(int k = 0; k < DMnz; k++) {
                    double z = (k + DMVertices[4]) * dz;
                    for(int j = 0; j < DMny; j++) {
                        double y = (j + DMVertices[2]) * dy;
                        for(int i = 0; i < DMnx; i++) {
                            double x = pSPARC->xin + (i + DMVertices[0]) * dx;
                            xc = x; yc = y; zc = z;
                            nonCart2Cart_coord(pSPARC, &xc, &yc, &zc);
                            DX[n*ldo+count] = DX1[count] * xc/x - DX2[count] * yc/(x*x);
                            count++;  
                        }
                    }
                }
            }
            break;
        case 1:
            for (int n = 0; n < ncol; n++){
                Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX1, DMnd, 0, &kpt_vec[0], comm, dims);
                Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX2, DMnd, 1, &kpt_vec[1], comm, dims);
                int count = 0;
                for(int k = 0; k < DMnz; k++) {
                    double z = (k + DMVertices[4]) * dz;
                    for(int j = 0; j < DMny; j++) {
                        double y = (j + DMVertices[2]) * dy;
                        for(int i = 0; i < DMnx; i++) {
                            double x = pSPARC->xin + (i + DMVertices[0]) * dx;
                            xc = x; yc = y; zc = z;
                            nonCart2Cart_coord(pSPARC, &xc, &yc, &zc);
                            DX[n*ldo+count] = DX1[count] * yc/x + DX2[count] * xc/(x*x);
                            count++;  
                        }
                    }
                }
            }
            break;
        case 2:
            for (int n = 0; n < ncol; n++){
                Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX1, DMnd, 1, &kpt_vec[1], comm, dims);
                Gradient_vec_dir_kpt(pSPARC, DMnd, DMVertices, 1, c, X+n*(unsigned)ldi, ldi, DX2, DMnd, 2, &kpt_vec[2], comm, dims);
                int count = 0;
                for(int k = 0; k < DMnz; k++) {
                    for(int j = 0; j < DMny; j++) {
                        for(int i = 0; i < DMnx; i++) {
                            DX[n*ldo+count] = -pSPARC->twist * DX1[count] + DX2[count];
                            count++;
                        }
                    }
                }
            }
            break;
        default: printf("gradient dir must be either 0, 1 or 2!\n");
                 break;
    }

    free(DX1);
    free(DX2);
}




/*
@ brief: function to calculate the gradients in cartesian directions (x and y) for cyclix systems with rotational factors
*/
void Gradient_vec_dir_rotfac(const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices,
                             const int ncol, const double c, const double *x, const double *y, const int ldi, 
                             double *Dx, const int ldo, const int dir, MPI_Comm comm, const int* dims, const int vecdir)
{

    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nproc = dims[0] * dims[1] * dims[2];
    int periods[3];
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    
    int FDn = pSPARC->order / 2;
    
    int isDir[3], exDir[3];
    isDir[0] = (int)(dir == 0); isDir[1] = (int)(dir == 1); isDir[2] = (int)(dir == 2);
    exDir[0] = isDir[0] * FDn; exDir[1] = isDir[1] * FDn; exDir[2] = isDir[2] * FDn;
    
    // The user has to make sure DMnd = DMnx * DMny * DMnz
    int DMnx = DMVertices[1] - DMVertices[0] + 1;
    int DMny = DMVertices[3] - DMVertices[2] + 1;
    int DMnz = DMVertices[5] - DMVertices[4] + 1;
    
    int DMnxny = DMnx * DMny;
    
    int DMnx_ex = DMnx + pSPARC->order * isDir[0];
    int DMny_ex = DMny + pSPARC->order * isDir[1];
    int DMnz_ex = DMnz + pSPARC->order * isDir[2];
    int DMnxny_ex = DMnx_ex * DMny_ex;
    int DMnd_ex = DMnxny_ex * DMnz_ex;
    
    int DMnx_in = DMnx - FDn;
    int DMny_in = DMny - FDn;
    int DMnz_in = DMnz - FDn;
    
    double w1_diag = c;
    
    // set up send buffer based on the ordering of the neighbors
    int istart[6] = {0,    DMnx_in, 0,    0,        0,    0}, 
          iend[6] = {FDn,  DMnx,     DMnx, DMnx,     DMnx, DMnx}, 
        jstart[6] = {0,    0,        0,    DMny_in, 0,    0},  
          jend[6] = {DMny, DMny,     FDn,  DMny,     DMny, DMny}, 
        kstart[6] = {0,    0,        0,    0,        0,    DMnz_in}, 
          kend[6] = {DMnz, DMnz,     DMnz, DMnz,     FDn,  DMnz};
    
    int count, n, k, j, i, nshift, kshift, jshift;
    int nbrcount, nbr_i;
    
    MPI_Request request;
    double *x_in, *x_out;
    x_in = x_out = NULL;

    double rotf1, rotf2, rotf3, rotf4;
    rotf1 = rotf2 = rotf3 = rotf4 = 0;
    if(vecdir == 0){
        rotf1 = cos(pSPARC->range_y);
        rotf2 = -sin(pSPARC->range_y);
        rotf3 = cos(pSPARC->range_y);
        rotf4 = sin(pSPARC->range_y);
    } else if(vecdir == 1){
        rotf1 = cos(pSPARC->range_y);
        rotf2 = sin(pSPARC->range_y);
        rotf3 = cos(pSPARC->range_y);
        rotf4 = -sin(pSPARC->range_y);
    }

    if(nproc > 1){
        int nd_in = ncol * pSPARC->order * (isDir[0] * DMny * DMnz + DMnx * isDir[1] * DMnz + DMnxny * isDir[2]);
        int nd_out = nd_in;
        x_in  = (double *)calloc( nd_in, sizeof(double));
        x_out = (double *)malloc( nd_out * sizeof(double)); // no need to init x_out

        int sendcounts[6], sdispls[6], recvcounts[6], rdispls[6];
        // set up parameters for MPI_Ineighbor_alltoallv
        // TODO: do this in Initialization to save computation time!
        sendcounts[0] = sendcounts[1] = recvcounts[0] = recvcounts[1] = ncol * FDn * (DMny * DMnz * isDir[0]);
        sendcounts[2] = sendcounts[3] = recvcounts[2] = recvcounts[3] = ncol * FDn * (DMnx * DMnz * isDir[1]);
        sendcounts[4] = sendcounts[5] = recvcounts[4] = recvcounts[5] = ncol * FDn * (DMnxny * isDir[2]);
    
        rdispls[0] = sdispls[0] = 0;
        rdispls[1] = sdispls[1] = sdispls[0] + sendcounts[0];
        rdispls[2] = sdispls[2] = sdispls[1] + sendcounts[1];
        rdispls[3] = sdispls[3] = sdispls[2] + sendcounts[2];
        rdispls[4] = sdispls[4] = sdispls[3] + sendcounts[3];
        rdispls[5] = sdispls[5] = sdispls[4] + sendcounts[4];

        count = 0;
        for (nbr_i = dir*2; nbr_i < dir*2+2; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
            nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)) * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
            //if(rank == 0)
            //    printf("nbrcount = %d, nbr_i = %d, DMvertices[2] %d, DMvertices[3] %d\n", nbrcount, nbr_i,pSPARC->DMVertices[2], pSPARC->DMVertices[3]);
            // TODO: Start loop over n here
            if ((pSPARC->DMVertices[2] == 0 && jstart[nbrcount] == 0) ) {
                for (n = 0; n < ncol; n++) {
                    nshift = n * ldi;
                    for (k = kstart[nbrcount]; k < kend[nbrcount]; k++) {
                        kshift = nshift + k * DMnxny;
                        for (j = jstart[nbrcount]; j < jend[nbrcount]; j++) {
                            jshift = kshift + j * DMnx;
                            for (i = istart[nbrcount]; i < iend[nbrcount]; i++) {
                                x_out[count++] = x[jshift+i] * rotf1 + y[jshift+i] * rotf2;
                            }
                        }
                    }
                }
            } else if ((pSPARC->DMVertices[3] == (pSPARC->Ny-1) && jstart[nbrcount] == DMny_in)){
                for (n = 0; n < ncol; n++) {
                    nshift = n * ldi;
                    for (k = kstart[nbrcount]; k < kend[nbrcount]; k++) {
                        kshift = nshift + k * DMnxny;
                        for (j = jstart[nbrcount]; j < jend[nbrcount]; j++) {
                            jshift = kshift + j * DMnx;
                            for (i = istart[nbrcount]; i < iend[nbrcount]; i++) {
                                x_out[count++] = x[jshift+i] * rotf3 + y[jshift+i] * rotf4;
                            }
                        }
                    }
                }
            } else {
                for (n = 0; n < ncol; n++) {
                    nshift = n * ldi;
                    for (k = kstart[nbrcount]; k < kend[nbrcount]; k++) {
                        kshift = nshift + k * DMnxny;
                        for (j = jstart[nbrcount]; j < jend[nbrcount]; j++) {
                            jshift = kshift + j * DMnx;
                            for (i = istart[nbrcount]; i < iend[nbrcount]; i++) {
                                x_out[count++] = x[jshift+i];
                            }
                        }
                    }
                }
            }
            
        }    

        // first transfer info. to/from neighbor processors
        MPI_Ineighbor_alltoallv(x_out, sendcounts, sdispls, MPI_DOUBLE, 
                                x_in, recvcounts, rdispls, MPI_DOUBLE, 
                                comm, &request); // non-blocking
    }                             
 
    // while the non-blocking communication is undergoing, compute Dx which only requires values from local memory
    int pshift = 0, pshift_ex = 0;
    double *D1_stencil_coeffs_dim;
    D1_stencil_coeffs_dim = (double *)malloc((FDn + 1) * sizeof(double));
    double *x_ex = (double *)malloc(ncol * DMnd_ex * sizeof(double));
    D1_stencil_coeffs_dim[0] = 0.0;
    
    int p;
    switch (dir) {
        case 0:
            pshift = 1; pshift_ex = 1;
            for (p = 1; p <= FDn; p++) {
                // stencil coeff
                D1_stencil_coeffs_dim[p] = pSPARC->D1_stencil_coeffs_x[p];
            }
            break;
        case 1:
            pshift = DMnx; pshift_ex = DMnx_ex; 
            for (p = 1; p <= FDn; p++) {
                // stencil coeff
                D1_stencil_coeffs_dim[p] = pSPARC->D1_stencil_coeffs_y[p];
            }
            break;
        case 2:
            pshift = DMnxny; pshift_ex = DMnxny_ex;
            for (p = 1; p <= FDn; p++) {
                // stencil coeff
                D1_stencil_coeffs_dim[p] = pSPARC->D1_stencil_coeffs_z[p];
            }
            break;
        default: printf("gradient dir must be either 0, 1 or 2!\n");
                 break;
    }
    
    int DMnz_exDir = DMnz+exDir[2];
    int DMny_exDir = DMny+exDir[1];
    int DMnx_exDir = DMnx+exDir[0];
    for (n = 0; n < ncol; n++){
        nshift = n * DMnd_ex;
        count = 0;
        for (k = exDir[2]; k < DMnz_exDir; k++){
            kshift = nshift + k * DMnxny_ex;
            for (j = exDir[1]; j < DMny_exDir; j++){
                jshift = kshift + j * DMnx_ex;
                for (i = exDir[0]; i < DMnx_exDir; i++){
                    x_ex[jshift+i] = x[count++ + n*ldi]; // this saves index calculation time
                }
            }
        }
    }                

    int overlap_flag = (int) (nproc > 1 && DMnx > pSPARC->order 
                          && DMny > pSPARC->order && DMnz > pSPARC->order);
                          
    //overlap_flag = 1;                                      
    if (overlap_flag) {
        for (n = 0; n < ncol; n++) {
            Calc_DX(x+n*ldi, Dx+n*ldo, FDn, pshift, DMnx, DMnx, DMnxny, DMnxny,
                    FDn, DMnx_in, FDn, DMny_in, FDn, DMnz_in, FDn, FDn, FDn,
                    D1_stencil_coeffs_dim, w1_diag);
        }    
    } 
    
    if(nproc > 1) {
        // set up start and end indices for copy receive buffer
        istart[0] = 0;             iend[0] = exDir[0];        
        jstart[0] = exDir[1];      jend[0] = DMny+exDir[1];   
        kstart[0] = exDir[2];      kend[0] = DMnz+exDir[2];
        istart[1] = DMnx+exDir[0]; iend[1] = DMnx+2*exDir[0]; 
        jstart[1] = exDir[1];      jend[1] = DMny+exDir[1];   
        kstart[1] = exDir[2];      kend[1] = DMnz+exDir[2]; 
        istart[2] = exDir[0];      iend[2] = DMnx+exDir[0];   
        jstart[2] = 0;             jend[2] = exDir[1];        
        kstart[2] = exDir[2];      kend[2] = DMnz+exDir[2];
        istart[3] = exDir[0];      iend[3] = DMnx+exDir[0];   
        jstart[3] = DMny+exDir[1]; jend[3] = DMny+2*exDir[1]; 
        kstart[3] = exDir[2];      kend[3] = DMnz+exDir[2];
        istart[4] = exDir[0];      iend[4] = DMnx+exDir[0];   
        jstart[4] = exDir[1];      jend[4] = DMny+exDir[1];   
        kstart[4] = 0;             kend[4] = exDir[2];
        istart[5] = exDir[0];      iend[5] = DMnx+exDir[0];   
        jstart[5] = exDir[1];      jend[5] = DMny+exDir[1];   
        kstart[5] = DMnz+exDir[2]; kend[5] = DMnz+2*exDir[2];

        // make sure receive buffer is ready
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // copy receive buffer into extended domain
        count = 0;
        for (nbrcount = dir*2; nbrcount < dir*2+2; nbrcount++) {
            for (n = 0; n < ncol; n++) {
                nshift = n * DMnd_ex;
                for (k = kstart[nbrcount]; k < kend[nbrcount]; k++) {
                    kshift = nshift + k * DMnxny_ex;
                    for (j = jstart[nbrcount]; j < jend[nbrcount]; j++) {
                        jshift = kshift + j * DMnx_ex;
                        for (i = istart[nbrcount]; i < iend[nbrcount]; i++) {
                            x_ex[jshift+i] = x_in[count++];
                        }
                    }
                }
            }
        }
        free(x_out);
        free(x_in);
    } else {
        int istart_in[6], iend_in[6], jstart_in[6], jend_in[6], kstart_in[6], kend_in[6];
        istart_in[0] = 0;             iend_in[0] = exDir[0];        
        jstart_in[0] = exDir[1];      jend_in[0] = DMny+exDir[1];   
        kstart_in[0] = exDir[2];      kend_in[0] = DMnz+exDir[2];
        istart_in[1] = DMnx+exDir[0]; iend_in[1] = DMnx+2*exDir[0]; 
        jstart_in[1] = exDir[1];      jend_in[1] = DMny+exDir[1];   
        kstart_in[1] = exDir[2];      kend_in[1] = DMnz+exDir[2]; 
        istart_in[2] = exDir[0];      iend_in[2] = DMnx+exDir[0];   
        jstart_in[2] = 0;             jend_in[2] = exDir[1];        
        kstart_in[2] = exDir[2];      kend_in[2] = DMnz+exDir[2];
        istart_in[3] = exDir[0];      iend_in[3] = DMnx+exDir[0];   
        jstart_in[3] = DMny+exDir[1]; jend_in[3] = DMny+2*exDir[1]; 
        kstart_in[3] = exDir[2];      kend_in[3] = DMnz+exDir[2];
        istart_in[4] = exDir[0];      iend_in[4] = DMnx+exDir[0];   
        jstart_in[4] = exDir[1];      jend_in[4] = DMny+exDir[1];   
        kstart_in[4] = 0;             kend_in[4] = exDir[2];
        istart_in[5] = exDir[0];      iend_in[5] = DMnx+exDir[0];   
        jstart_in[5] = exDir[1];      jend_in[5] = DMny+exDir[1];   
        kstart_in[5] = DMnz+exDir[2]; kend_in[5] = DMnz+2*exDir[2];

        int nshift1, kshift1, jshift1, kp, jp, ip;
        // copy the extended part from x into x_ex
        for (nbr_i = dir * 2; nbr_i < dir * 2 + 2; nbr_i++) {
            // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
            nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)); // * (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
            if (periods[nbr_i / 2]) {
                if (jstart[nbrcount] == 0) {
                    for (n = 0; n < ncol; n++){
                        nshift = n * DMnd_ex; nshift1 = n * ldi;
                        for (k = kstart[nbrcount], kp = kstart_in[nbr_i]; k < kend[nbrcount]; k++, kp++){
                            kshift = nshift + kp * DMnxny_ex; kshift1 = nshift1 + k * DMnxny;
                            for (j = jstart[nbrcount], jp = jstart_in[nbr_i]; j < jend[nbrcount]; j++, jp++){
                                jshift = kshift + jp * DMnx_ex; jshift1 = kshift1 + j * DMnx;
                                for (i = istart[nbrcount], ip = istart_in[nbr_i]; i < iend[nbrcount]; i++, ip++){
                                    x_ex[jshift+ip] = x[jshift1+i] * rotf1 + y[jshift1+i] * rotf2;
                                }
                            }
                        }
                    }   
                } else if (jstart[nbrcount] == DMny_in){
                    for (n = 0; n < ncol; n++){
                        nshift = n * DMnd_ex; nshift1 = n * ldi;
                        for (k = kstart[nbrcount], kp = kstart_in[nbr_i]; k < kend[nbrcount]; k++, kp++){
                            kshift = nshift + kp * DMnxny_ex; kshift1 = nshift1 + k * DMnxny;
                            for (j = jstart[nbrcount], jp = jstart_in[nbr_i]; j < jend[nbrcount]; j++, jp++){
                                jshift = kshift + jp * DMnx_ex; jshift1 = kshift1 + j * DMnx;
                                for (i = istart[nbrcount], ip = istart_in[nbr_i]; i < iend[nbrcount]; i++, ip++){
                                    x_ex[jshift+ip] = x[jshift1+i] * rotf3 + y[jshift1+i] * rotf4;
                                }
                            }
                        }
                    }  
                } else {
                    for (n = 0; n < ncol; n++){
                        nshift = n * DMnd_ex; nshift1 = n * ldi;
                        for (k = kstart[nbrcount], kp = kstart_in[nbr_i]; k < kend[nbrcount]; k++, kp++){
                            kshift = nshift + kp * DMnxny_ex; kshift1 = nshift1 + k * DMnxny;
                            for (j = jstart[nbrcount], jp = jstart_in[nbr_i]; j < jend[nbrcount]; j++, jp++){
                                jshift = kshift + jp * DMnx_ex; jshift1 = kshift1 + j * DMnx;
                                for (i = istart[nbrcount], ip = istart_in[nbr_i]; i < iend[nbrcount]; i++, ip++){
                                    x_ex[jshift+ip] = x[jshift1+i];
                                }
                            }
                        }
                    }  
                }
                             
            } else {
                for (n = 0; n < ncol; n++){
                    nshift = n * DMnd_ex;
                    for (kp = kstart_in[nbr_i]; kp < kend_in[nbr_i]; kp++){
                        kshift = nshift + kp * DMnxny_ex;
                        for (jp = jstart_in[nbr_i]; jp < jend_in[nbr_i]; jp++){
                            jshift = kshift + jp * DMnx_ex;
                            for (ip = istart_in[nbr_i]; ip < iend_in[nbr_i]; ip++){
                                x_ex[jshift+ip] = 0.0;
                            }
                        }
                    }
                }                
            }
        }
    }   

    // calculate dx
    if (overlap_flag) {
        // first calculate dx(0:DMnx, 0:DMny, [0:FDn,DMnz-FDn:DMnz])

        for (n = 0; n < ncol; n++) {
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    0, DMnx, 0, DMny, 0, FDn, exDir[0], exDir[1], exDir[2], D1_stencil_coeffs_dim, w1_diag);
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    0, DMnx, 0, DMny, DMnz_in, DMnz, exDir[0], exDir[1], DMnz_in+exDir[2], D1_stencil_coeffs_dim, w1_diag);        
        }
        
        // then calculate dx(0:DMnx, [0:FDn,DMny-FDn:DMny], FDn:DMnz-FDn)

        for (n = 0; n < ncol; n++) {
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    0, DMnx, 0, FDn, FDn, DMnz_in, exDir[0], exDir[1], FDn+exDir[2], D1_stencil_coeffs_dim, w1_diag);
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    0, DMnx, DMny_in, DMny, FDn, DMnz_in, exDir[0], DMny_in+exDir[1], FDn+exDir[2], D1_stencil_coeffs_dim, w1_diag);        
        } 
        
        // finally calculate dx([0:FDn,DMnx-FDn:DMnx], FDn:DMny-FDn, FDn:DMnz-FDn)
       
        for (n = 0; n < ncol; n++) {
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    0, FDn, FDn, DMny_in, FDn, DMnz_in, exDir[0], FDn+exDir[1], FDn+exDir[2], D1_stencil_coeffs_dim, w1_diag);
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    DMnx_in, DMnx, FDn, DMny_in, FDn, DMnz_in, DMnx_in+exDir[0], FDn+exDir[1], FDn+exDir[2], D1_stencil_coeffs_dim, w1_diag);        
        }
           
    } else {
        for (n = 0; n < ncol; n++) {
            Calc_DX(x_ex+n*DMnd_ex, Dx+n*ldo, FDn, pshift_ex, DMnx_ex, DMnx, DMnxny_ex, DMnxny,
                    0, DMnx, 0, DMny, 0, DMnz, exDir[0], exDir[1], exDir[2], D1_stencil_coeffs_dim, w1_diag);    
        }
    }
    
    free(x_ex);
    free(D1_stencil_coeffs_dim);

}
