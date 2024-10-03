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

// declare datatypes (defined elsewhere)
typedef struct _D2DEXT_OBJ D2Dext_OBJ;
typedef struct _KRON_LAP KRON_LAP;
typedef struct _MPEXP_OBJ MPEXP_OBJ;

/**
 * @brief   This structure type is for SQ method.
 */
typedef struct _SQ_OBJ
{
    int nloc[3];                      // Number of finite difference intervals on each direction within Rcut distance

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

    MPI_Comm dmcomm_d2dext_sq;      // SQ Communicator for transferring data in PR domain
    D2Dext_OBJ *d2dext_dmcomm_sq;
    D2Dext_OBJ *d2dext_dmcomm_sq_ext;

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

    // exact exchange
    KRON_LAP *kron_lap;
    MPEXP_OBJ *MpExp_exx;
    double **Dn;
    double **Dn_exx;
    double *basis;
    double **exxPot;
    double *erfcR;
} SQ_OBJ;


#endif // DSSQ_H