/**
 * @file    structRelaxation.h
 * @brief   Declares the functions for performing structural relaxation.
 *
 * @authors Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef RELAX_H
#define RELAX_H

#include "isddft.h"

/** 
@brief Main function of relaxation
**/
void main_Relax(SPARC_OBJ *pSPARC);


/**
 * @brief   Performs relaxation of atom positions using Nonlinear Conjugate Gradient (NLCG).
 *
 * @param SPARC    
 */
void NLCG(SPARC_OBJ *pSPARC);


/*
 @ brief: implementation of Limited memory BFGS (L-BFGS) algorithm (based on the implementation provided by Edmond Chow)
*/
void LBFGS(SPARC_OBJ *pSPARC);


double norm(int n, double *vec);


double dotproduct(int n, double *vec1, int s1, double *vec2, int s2);


/**
 * @brief   Performs relaxation of atom positions using FIRE algorithm.
*/
void FIRE(SPARC_OBJ *pSPARC);


/**
 * @brief   Performs cell relaxation using Brent's algorithm.
 */
void Relax_Cell(SPARC_OBJ *pSPARC);


/**
 * @brief Function for Brent's method to find optimal volume.
 *
 *        This function takes in a volume, and finds the maximum
 *        principle stress of the system. sigma_max = f(vol). 
 *        The root of this function gives the optimal volume.
 **/
double volrelax_constraint(SPARC_OBJ *pSPARC, double vol);


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
void reinitialize_cell_mesh(SPARC_OBJ *pSPARC, double vol);


/**
 * @brief   Write the re-initialized parameters into the output file.
 */
void write_output_reinit(SPARC_OBJ *pSPARC);


/**
 * @brief   Write the re-initialized parameters into the output file.
 */
void PrintCellRelax(SPARC_OBJ *pSPARC);


/**
 * @brief   Find root of a function using Brent's method.
 *
 * @ref     W.H. Press, Numerical recepies 3rd edition: The art of scientific 
 *          computing, Cambridge university press, 2007.
 */
double BrentsFun(SPARC_OBJ *pSPARC, double x1, double x2, double tol_x, double tol_f, int max_iter);


/**
@ brief: function to write the output of the relaxation  
**/
void Print_fullRelax(SPARC_OBJ *pSPARC, FILE *output_relax);


/**
@ brief: function to write the restart file for relaxation
**/
void PrintRelax(SPARC_OBJ *pSPARC);


/**
@ brief: function to restart the relaxation
**/
void RestartRelax(SPARC_OBJ *pSPARC);


#endif // RELAX_H
