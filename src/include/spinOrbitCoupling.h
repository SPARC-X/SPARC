/**
 * @file    spinOrbitCoupling.h
 * @brief   This file contains function declarations for spin-orbit 
 *          coupling (SOC) calculation.  
 *
 * @authors Xin Jing <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef SPINORBITCOUPLING_H
#define SPINORBITCOUPLING_H 

#include "isddft.h"

//////////////////////////////////////////////////////////////////////////////////////////
/////////////////////      Vloc functions      ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief   Calculate nonlocal spin-orbit (SO) projectors. 
 */
void CalculateNonlocalProjectors_SOC(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ *nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, int *DMVertices, MPI_Comm comm);

/**
 * @brief   Extract 3 different components from Chiso for further calculation
 */
void CreateChiSOMatrix(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ *nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, MPI_Comm comm);

/**
 * @brief   Calculate indices for storing nonlocal inner product in an array for SOC. 
 *
 *          We will store the inner product < Chi_Jlm, x_n > in a continuous array "alpha",
 *          the dimensions are in the order: <lm>, n, J. Here we find out the sizes of the 
 *          inner product corresponding to atom J, and the total number of inner products
 *          corresponding to each vector x_n.
 */
void CalculateNonlocalInnerProductIndexSOC(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate Vnl SO term 1 times vectors in a matrix-free way with Bloch factor
 * 
 *          0.5*sum_{J,n,lm} m*gamma_{Jln} (sum_{J'} ChiSO_{J'lmn}>)(sum_{J'} <ChiSO_{J'lmn}|x>)
 */
void Vnl_vec_mult_SOC1(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                      NLOC_PROJ_OBJ *nlocProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int spinor, int kpt, MPI_Comm comm);

/**
 * @brief   Calculate Vnl SO term 1 times vectors in a matrix-free way with Bloch factor
 * 
 *          0.5*sum_{J,n,lm} sqrt(l*(l+1)-m*(m+sigma))*gamma_{Jln} *
 *          (sum_{J'} ChiSO_{J'lm+sigma,n}>)(sum_{J'} <ChiSO_{J'lmn}|x_sigma'>)
 */
void Vnl_vec_mult_SOC2(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
        NLOC_PROJ_OBJ *nlocProj, int ncol, double _Complex *xos, int ldi, double _Complex *Hx, int ldo, int spinor, int kpt, MPI_Comm comm);


#endif 

