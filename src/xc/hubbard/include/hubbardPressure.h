#ifndef HUBBARDPRESSURE_H
#define HUBBARDPRESSURE_H

#include "isddft.h"

/**
 * @brief Calculate hubbard pressure
 */
void Calculate_hubbard_pressure(SPARC_OBJ *pSPARC);

/**
 * @brief Calculate gamma point hubbard pressure
 */
void Calculate_hubbard_pressure_linear(SPARC_OBJ *pSPARC);

/**
 * @brief   Calculate <Orb_Jlm, (x-RJ')_beta, DPsi_n> for spinor psi
 */
void Compute_Integral_Orb_XmRjp_beta_Dpsi(SPARC_OBJ *pSPARC, double *dpsi_xi, double *beta, int dim2);

/**
 * @brief Compute hubbard pressure for gamma point using integrals
 */
void Compute_pressure_hubbard_by_integrals(SPARC_OBJ *pSPARC,double *pressure_hub,double *alpha);

/**
 * @brief Calculate hubbard pressure for k-points using integrals
 */
void Calculate_hubbard_pressure_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief Calculate <orb_Jlm, (x-RJ')_beta, DPsi_n> for spinor pressure
 */
void Compute_Integral_Orb_XmRjp_beta_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi_xi, double _Complex *beta, int kpt, int dim2);

/**
 * @brief Compute hubbard pressure with k-points by integrals
 */
void Compute_pressure_hubbard_by_integrals_kpt(SPARC_OBJ *pSPARC, double *pressure_hub, double _Complex *alpha);


#endif