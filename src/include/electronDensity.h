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
@ brief: calculate electron density from psi
*/ 
void CalculateDensity_psi(SPARC_OBJ *pSPARC, double *rho);

/*
@ brief: calculate magx and magy from psi
*/ 
void Calculate_Magx_Magy_psi(SPARC_OBJ *pSPARC, double *mag);

/*
@ brief: calculate magz
*/ 
void Calculate_Magz(SPARC_OBJ *pSPARC, int DMnd, double *magz, double *rhoup, double *rhodw);

/*
@ brief: calculate norm of magnetization
*/ 
void Calculate_Magnorm(SPARC_OBJ *pSPARC, int DMnd, double *magx, double *magy, double *magz, double *magnorm);

/*
@ brief: calculate diagnoal density
*/ 
void Calculate_diagonal_Density(SPARC_OBJ *pSPARC, int DMnd, double *magnorm, double *rho_tot, double *rho11, double *rho22);
#endif // ELECTRONDENSITY_H
