/**
 * @file    dssq.h
 * @brief   This file contains the data structure declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef DSSQ_H
#define DSSQ_H 

#include <mpi.h>

/**
 * @brief   This structure type is for SQ communication.
 * 
 *          Variables required for SQ communication, to get all nodes within PR domain 
 */
typedef struct _SQIND
{
    int *send_neighs;
    int *rec_neighs;
    int *send_counts;
    int *rec_counts;
    int send_layers[6];
    int rec_layers[6];

    int *x_scounts;
    int *y_scounts;
    int *z_scounts;
    int *x_rcounts;
    int *y_rcounts;
    int *z_rcounts;

    int *scounts;
    int *sdispls; 
    int *rcounts;
    int *rdispls;
    int n_in;
    int n_out;
    
}SQIND;


/**
 * @brief   This structure type is for SQ method.
 */
typedef struct _SQ_OBJ
{
    int nloc[3];                      // Number of finite difference intervals on each direction within Rcut distance
    int coords[3];                    // Coordinates of each process
    int rem[3];                       // Remainder of total 

    MPI_Comm dmcomm_SQ;             // SQ Communicator topology
    int DMnx_SQ;
    int DMny_SQ;
    int DMnz_SQ;
    int DMnd_SQ;
    int DMVertices_SQ[6];

    int Nx_loc;
    int Ny_loc;
    int Nz_loc;
    int Nd_loc;                       // Total number of finite difference nodes within Rcut domain

    int DMnx_PR;
    int DMny_PR;
    int DMnz_PR;
    int DMnd_PR;
    int DMVertices_PR[6];             // Domain vertices of PR (process+Rcut) domain

    int Nx_ex;
    int Ny_ex;
    int Nz_ex;
    int Nd_ex;

    MPI_Comm SQ_dist_graph_comm;      // SQ Communicator for transferring data in PR domain
    SQIND *SqInd;                     // Communicator information for SQ_dist_graph_comm

    double **lanczos_vec_all;
    double **w_all;
    double *lanczos_vec;
    double *w;

    double *mineig;
    double *maxeig;
    double *lambda_max;
    double *Veff_loc_SQ;
    double **gnd;
    double **gwt;
    double *Veff_PR;                // Veff operator within PR (process+Rcut) domain
    double *x_ex;
    int forceFlag;
} SQ_OBJ;


#endif // DSSQ_H