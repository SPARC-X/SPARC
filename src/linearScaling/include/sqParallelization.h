/**
 * @file    sqParallelization.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQPARALLELIZATION_H
#define SQPARALLELIZATION_H 

/**
 * @brief   Set up sub-communicators for SQ method.
 * 
 * Note:    Part of the codes are copied and modified from the code in parallelization.c file
 */
void Setup_Comms_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Create communication topology for SQ type communication
 * 
 * @param DMnxnynz  Number of FD nodes of current processor in 3 directions
 * @param np        Number of processors in 3 directions for current cartesian topology
 * @param cart      Cartesian topology, where the SQ communicator is built on
 * @param comm_sq   The SQ communicator 
 * 
 * Note:            SQ type communication is for each process getting all values within their 
 *                  P.R. domain (Process + Rcut). It is different from Laplacian type communication 
 *                  for orthogonal systems, but similar to Lap communication for unorthogonal systems. 
 */
void Comm_topologies_sq(SPARC_OBJ *pSPARC, int DMnxnynz[3], int np[3], MPI_Comm cart, MPI_Comm *comm_sq);

/**
 * @brief   Transferring Veff to PR domain use SQ type communication 
 */
void Transfer_Veff_PR(SPARC_OBJ *pSPARC, double *Veff_PR, MPI_Comm comm_sq);

/**
 * @brief   Create a D2D object for communication from SQ domain to phi domain 
 */
void create_D2D_sq2phi(SPARC_OBJ *pSPARC);

/**
 * @brief   Transferring electron density from SQ domain to phi domain using D2D
 */
void TransferDensity_sq2phi(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv);

/**
 * @brief   Transferring Veff from phi domain to SQ domain using D2D
 */
void TransferVeff_phi2sq(SPARC_OBJ *pSPARC, double *Veff_send, double *Veff_recv);

void TransferVeff_sq2sqext(SPARC_OBJ *pSPARC, double *Veff_send, double *Veff_recv);
#endif