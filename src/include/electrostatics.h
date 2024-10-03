/**
 * @file    electrostatics.h
 * @brief   This file declares the functions for electrostatics.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef ELECTROSTATICS_H
#define ELECTROSTATICS_H

#include "isddft.h"

/**
 * @brief    Calculate pseudocharge density cutoff ("rb").
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure.
 */
void Calculate_PseudochargeCutoff(SPARC_OBJ *pSPARC);


/**
 * @brief   Find the list of all atoms that influence the processor domain.
 *
 * @param pSPARC 
 */
void GetInfluencingAtoms(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate pseudocharge density for given atom positions.  
 */
void Generate_PseudoChargeDensity(SPARC_OBJ *pSPARC);


/*
 @ brief: function to evaluate bJ and bJ_ref
*/


void Calc_lapV(const SPARC_OBJ *pSPARC, const double *VJ, const double FDn,
               const int nxp, const int nyp, const int nzp,
               const int nx, const int ny, const int nz,
               const double *Lap_wt, const double w2_diag, double xin, double coef, double *bJ);



/**
 * @brief   Calculate pseudopotential reference.
 *
 * @ref     Linear scaling solution of the all-electron Coulomb problem in solids.
 *          J.E.Pask, N. Sukumar and S.E. Mousavi
 */
void Calculate_Pseudopot_Ref(double *R, int len, double rcut, int Znucl, double *Vref);


/**
 * @brief   Calculate the right-hand-side of the poisson equation.
 *          
 *          The RHS of the poisson equation includes the boundary
 *          conditions, i.e., f = -4 * pi * ( rho + b - d), where
 *          d is the BC term. For periodic systems, d = 0.
 */
void poisson_RHS(SPARC_OBJ *pSPARC, double *f);


/**
 * @brief   Solve Poisson equation for electrostatic potential.
 */
void Calculate_elecstPotential(SPARC_OBJ *pSPARC);


/**
 * @brief   Perform multipole expansion to find boundary condition for the poisson equation
 *                                      -D2 phi(x) = f. 
 *          It is required that f decays to zero on the boundary.
 *
 *          Note that this is only done in "phi-domain".
 */
void MultipoleExpansion_phi(SPARC_OBJ *pSPARC, 
    int DMnx, int DMny, int DMnz, int *DMVertices, double *f, double *d_cor, MPI_Comm comm);
 

/**
 * @brief   Use partial dipole to correct boundary condition for the poisson equation
 *                                      -D2 phi(x) = f.
 *          with periodic BCs in 2 directions, and Dirichlet BC in the other direction (surface).
 *          So that when discretized in finite difference with Dirichlet BC, the equation will be
 *                                  - DiscreteLaplacian phi = f - d.
 *          It is required that f decays to zero on the boundary.
 *
 *          Note that this is only done in "phi-domain".
 */
void PartrialDipole_surface(SPARC_OBJ *pSPARC, double *f, double *d_cor);


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
void PartrialDipole_wire(SPARC_OBJ *pSPARC, double *f, double *d_cor);



void Jacobi_preconditioner(SPARC_OBJ *pSPARC, int N, double c, double *r, double *f, MPI_Comm comm);


void init_multipole_expansion(SPARC_OBJ *pSPARC, MPEXP_OBJ *MpExp, 
        int Nx, int Ny, int Nz, int DMnx, int DMny, int DMnz, int *DMVertices, MPI_Comm comm);

void free_multipole_expansion(MPEXP_OBJ *MpExp, MPI_Comm comm);

void apply_multipole_expansion(SPARC_OBJ *pSPARC, MPEXP_OBJ *MpExp, 
    int Nx, int Ny, int Nz, int DMnx, int DMny, int DMnz, 
    int *DMVertices, double *rhs, double *d, MPI_Comm comm);

#endif // ELECTROSTATICS_H

