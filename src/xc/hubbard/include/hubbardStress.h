#ifndef HUBBARDSTRESS_H
#define HUBBARDSTRESS_H

#include "isddft.h"

/**
 * @brief Calculate hubbard stress
 */
void Calculate_hubbard_stresses(SPARC_OBJ *pSPARC);

/**
 * @brief Calculate gamma point hubbard stress
 */
void Calculate_hubbard_stress_linear(SPARC_OBJ *pSPARC);

/**
 * @brief Calculate <\Psi | V_U | \Psi>
 */
double Compute_Hubbard_Energy(SPARC_OBJ *pSPARC);

/**
 * @brief Calculate <Orb_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 */
void Compute_Integral_Orb_StXmRjp_beta_Dpsi(SPARC_OBJ *pSPARC, double *dpsi_xi, double *beta, int dim2);

/**
 * @brief Compute hubbard stress tensor
 */
void Compute_stress_tensor_hubbard_by_integrals(SPARC_OBJ *pSPARC,double *stress_hub,double *alpha);

/**
 * @brief Calculate hubbard stress for k-points
 */
void Calculate_hubbard_stress_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief Calculate <orb_Jlm, ST(x-RJ')_beta, DPsi_n> for spinor stress
 */
void Compute_Integral_Orb_StXmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int kpt, int dim2);

/**
 * @brief Compute hubbard stress tensor with k-points
 */
void Compute_stress_tensor_hubbard_by_integrals_kpt(SPARC_OBJ *pSPARC, double *stress_hub, double _Complex *alpha);

#endif