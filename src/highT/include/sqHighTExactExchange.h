/**
 * @file    sqHighTExactExchange.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQHIGHTEXACTEXCHANGE_H
#define SQHIGHTEXACTEXCHANGE_H 

#include "isddft.h"

/**
 * @brief   claculate exact exchange energy using SQ 
 */
void exact_exchange_energy_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   claculate exact exchange energy in SQ by solving poisson's equations
 */
void evaluate_exact_exchange_energy_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   claculate exact exchange energy in SQ by using the basis
 */
void evaluate_exact_exchange_energy_Acc_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   claculate exact exchange potential using SQ 
 */
void exact_exchange_potential_SQ(SPARC_OBJ *pSPARC, const double *x, const int indx, double *Hx);

/**
 * @brief   claculate exact exchange potential in SQ by using the basis
 */
void evaluate_exact_exchange_potential_ACC_SQ(SPARC_OBJ *pSPARC, const double *x, const int indx, double *Hx);

/**
 * @brief   claculate exact exchange potential in SQ by solving poisson's equations
 */
void evaluate_exact_exchange_potential_SQ(SPARC_OBJ *pSPARC, const double *x, const int indx, double *Hx);

/**
 * @brief   calculate poisson's right hand side for hybrid functional
 */
void poisson_RHS_hybrid(SPARC_OBJ *pSPARC, double *rhs, int Nd);

/**
 * @brief   calculate poisson's right hand side for hybrid functional
 * 
 * Note:    this is used when rhs and density matrix are eccentric
 */
void poisson_RHS_hybrid_eccentric(SPARC_OBJ *pSPARC, 
                    const double *Dn, const int cDx, const int cDy, const int cDz, const int indx, 
                    const int Nx, const int Ny, const int Nz, const double *x, double *rhs);

/**
 * @brief   collect column of density matrix
 */
void collect_col_of_Density_matrix(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate the basis, i.e. the solution of poissons equation with e_n as rhs
 */
void calculate_basis(SPARC_OBJ *pSPARC);

/**
 * @brief   Restrict column of density matrix onto current local domain
 */
void restrict_Dn(const double *Dn, const int cDx, const int cDy, const int cDz, 
                const int Nx, const int Ny, const int Nz, double *Dn_res);

/**
 * @brief   Compute explicit exact exchange potential matrix 
 */
void compute_exx_potential_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute explicit exact exchange potential matrix for each node
 */
void compute_exx_potential_node_SQ(SPARC_OBJ *pSPARC, int indx, double *exxPot);

/**
 * @brief   Compute erfc(\mu R) in HSE
 */
void calculate_erfcR(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute exact exchange stress contribution in SQ
 */
void exact_exchange_stress_node_SQ(SPARC_OBJ *pSPARC, double *Dn, 
                double *gradx_Dn, double *grady_Dn, double *gradz_Dn, double *s_exx);

/**
 * @brief   Compute exact exchange stress using basis in SQ for each node
 */
void evaluate_exact_exchange_stress_node_ACC_SQ(SPARC_OBJ *pSPARC, 
                double *Dn, double *gradx_Dn, double *grady_Dn, double *gradz_Dn, double *s_exx);

/**
 * @brief   Compute exact exchange stress by solving poisson's equations in SQ for each node
 */
void evaluate_exact_exchange_stress_node_SQ(SPARC_OBJ *pSPARC, double *Dn, 
                double *gradx_Dn, double *grady_Dn, double *gradz_Dn, double *s_exx);

/**
 * @brief   Compute exact exchange stress in SQ
 */
void Calculate_exact_exchange_stress_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute exact exchange pressure in SQ
 */
void exact_exchange_pressure_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute exact exchange pressure in SQ
 */
void Calculate_exact_exchange_pressure_SQ(SPARC_OBJ *pSPARC);


/**
 * @brief   Free SCF variables for SQ-hybrid
 */
void free_exx_SQ_highT(SPARC_OBJ *pSPARC);

#endif // SQ_H