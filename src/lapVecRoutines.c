/**
 * @file    lapVecRoutines.c
 * @brief   This file contains functions for performing laplacian-vector 
 *          multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

#include "lapVecRoutines.h"
#include "lapVecOrth.h"
#include "lapVecOrthKpt.h"
#include "lapVecNonOrth.h"
#include "lapVecNonOrthKpt.h"
#include "isddft.h"

/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 */
void Lap_vec_mult(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, double *x, double *Lapx, MPI_Comm comm
) 
{
    int dims[3], periods[3], my_coords[3];
    MPI_Cart_get(comm, 3, dims, periods, my_coords);
    
    if(pSPARC->cell_typ == 0) {
        Lap_vec_mult_orth(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, dims);
    } else {
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph; // pSPARC->comm_dist_graph_phi
        else    
            comm2 = pSPARC->comm_dist_graph_phi;
        
        //comm2 = pSPARC->comm_dist_graph_phi;
        // TODO: make the second communicator general rather than only for phi
        Lap_vec_mult_nonorth(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, comm2, dims);       
    }
}



/**
 * @brief   Calculate (Lap + c * I) times vectors in a matrix-free way.
 */
void Lap_vec_mult_kpt(
    const SPARC_OBJ *pSPARC, const int DMnd, const int *DMVertices, 
    const int ncol, const double c, double complex *x, double complex *Lapx, int kpt, MPI_Comm comm
) 
{
    int dims[3], periods[3], my_coords[3];
    MPI_Cart_get(comm, 3, dims, periods, my_coords);
    
    if(pSPARC->cell_typ == 0) {
        Lap_vec_mult_orth_kpt(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, dims, kpt);
    } else {
        MPI_Comm comm2;
        comm2 = pSPARC->kptcomm_topo_dist_graph;
        // TODO: make the second communicator general rather than only for phi
        //Lap_vec_mult_nonorth(pSPARC, DMnd, DMVertices, ncol, c, x, Lapx, comm, comm2);
        Lap_vec_mult_nonorth_kpt(pSPARC, DMnd, DMVertices, ncol, 1.0, c, x, Lapx, comm, comm2, dims, kpt);       
    }
}

/**
 * @brief   Calculate the residual of the poisson equation: r = b - (-(Lap+c) * x).
 *
 *          The vector x is assumed to be stored domain-wisely among the processors. The
 *          structure pSPARC contains the description of the distribution info of x, and
 *          in this case the info of Laplacian operator such as boundary conditions, 
 *          finite-difference order and coefficients etc.
 */
void poisson_residual(SPARC_OBJ *pSPARC, int N, double c, double *x, double *b, double *r, MPI_Comm comm, double *time_info) 
{
    int i;
    double t1 = MPI_Wtime();
    Lap_vec_mult(pSPARC, N, pSPARC->DMVertices, 1, c, x, r, comm);
    double t2 = MPI_Wtime();
    *time_info = t2 - t1;
    
    // Calculate residual once Lx is obtained
    for (i = 0; i < N; i++) r[i] += b[i];
}


