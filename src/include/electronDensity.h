/**
 * @file    electronDensity.c
 * @brief   This file contains the functions for calculating electron density.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef ELECTRONDENSITY_H
#define ELECTRONDENSITY_H

#include "isddft.h"
#if USE_PCE
#include <libpce.h>
#include "hamstruct.h"
#endif

/*
@ brief: Main function responsible to find electron density
*/
#if USE_PCE
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error,
                        Hybrid_Decomp *hd, Chebyshev_Info *cheb, Eig_Info *Eigvals,
                        Our_Hamiltonian_Struct *ham_struct, 
                        Psi_Info *Psi1, Psi_Info *Psi2, Psi_Info *Psi3,
                        MPI_Comm kptcomm, MPI_Comm dmcomm, MPI_Comm blacscomm);
#else
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error);
#endif

/*
@ brief: calculate electron density from psi with no kpts
*/ 
 
void CalculateDensity_psi(SPARC_OBJ *pSPARC, double *rho);

/*
@ brief: calculate electron density from psi with no kpts but spin polarized
*/ 
 
void CalculateDensity_psi_spin(SPARC_OBJ *pSPARC, double *rho);

/*
@ brief: calculate electron density from psi with kpts
*/ 
 
void CalculateDensity_psi_kpt(SPARC_OBJ *pSPARC, double *rho);

/*
@ brief: calculate electron density from psi with kpts and spin polarized
*/ 
 
void CalculateDensity_psi_kpt_spin(SPARC_OBJ *pSPARC, double *rho);

#endif // ELECTRONDENSITY_H
