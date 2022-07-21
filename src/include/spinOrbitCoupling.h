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
        NLOC_PROJ_OBJ *nlocProj, int ncol, double complex *x, double complex *Hx, int spinor, int kpt, MPI_Comm comm);

/**
 * @brief   Calculate Vnl SO term 1 times vectors in a matrix-free way with Bloch factor
 * 
 *          0.5*sum_{J,n,lm} sqrt(l*(l+1)-m*(m+sigma))*gamma_{Jln} *
 *          (sum_{J'} ChiSO_{J'lm+sigma,n}>)(sum_{J'} <ChiSO_{J'lmn}|x_sigma'>)
 */
void Vnl_vec_mult_SOC2(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
        NLOC_PROJ_OBJ *nlocProj, int ncol, double complex *xos, double complex *Hx, 
        int spinor, int kpt, MPI_Comm comm);


//////////////////////////////////////////////////////////////////////////////////////////
///////////////////      Forces functions      ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief    Calculate nonlocal force components with kpts in case of spinor wave function.
 */
void Calculate_nonlocal_forces_kpt_spinor_linear(SPARC_OBJ *pSPARC);
/**
 * @brief   Calculate <Psi_n, Chi_Jlm> for spinor force
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_psi_Chi_kpt(SPARC_OBJ *pSPARC, double complex *beta, int spn_i, int kpt, char *option);
/**
 * @brief   Calculate <Chi_Jlm, DPsi_n> for spinor force
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_Dpsi_kpt(SPARC_OBJ *pSPARC, double complex *dpsi, double complex *beta, int spn_i, int kpt, char *option);
/**
 * @brief   Compute nonlocal forces using stored integrals
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_force_nloc_by_integrals(SPARC_OBJ *pSPARC, double *force_nloc, double complex *alpha, char *option);


//////////////////////////////////////////////////////////////////////////////////////////
///////////////////      Stress functions      ///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief    Calculate nonlocal + kinetic components of stress in terms of spinor wavefunction.
 */
void Calculate_nonlocal_kinetic_stress_kpt_spinor(SPARC_OBJ *pSPARC);
/**
 * @brief   Calculate <ChiSC_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_StXmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double complex *dpsi_xi, double complex *beta, int spn_i, int kpt, int dim2, char *option);
/**
 * @brief   Compute nonlocal Energy with spin-orbit coupling
 */
double Compute_Nonlocal_Energy_by_integrals(SPARC_OBJ *pSPARC, double complex *alpha, double complex *alpha_so1, double complex *alpha_so2);
/**
 * @brief   Compute nonlocal stress tensor with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_stress_tensor_nloc_by_integrals(SPARC_OBJ *pSPARC, double *stress_nl, double complex *alpha, char *option);


//////////////////////////////////////////////////////////////////////////////////////////
///////////////////      Pressure functions      /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief    Calculate nonlocal pressure components.
 */
void Calculate_nonlocal_pressure_kpt_spinor(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate <ChiSC_Jlm, (x-RJ')_beta, DPsi_n> for spinor stress
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_Integral_Chi_XmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double complex *dpsi_xi, double complex *beta, int spn_i, int kpt, int dim2, char *option);

/**
 * @brief   Compute nonlocal pressure with spin-orbit coupling
 * 
 *          Note: avail options are "SC", "SO1", "SO2"
 */
void Compute_pressure_nloc_by_integrals(SPARC_OBJ *pSPARC, double *pressure_nloc, double complex *alpha, char *option);

#endif 

