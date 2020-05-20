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

/*
@ brief: Main function responsible to find electron density
*/
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error);

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
