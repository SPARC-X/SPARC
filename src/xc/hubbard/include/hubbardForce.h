#ifndef HUBBARDFORCE_H
#define HUBBARDFORCE_H

#include "isddft.h"

void Calculate_hubbard_forces(SPARC_OBJ *pSPARC);
void Calculate_hubbard_forces_linear(SPARC_OBJ *pSPARC);
void Compute_Integral_psi_Orb(SPARC_OBJ *pSPARC, double *beta, double *Xorb);
void Compute_Integral_Orb_Dpsi(SPARC_OBJ *pSPARC, double *dpsi, double *beta);
void Compute_force_hubbard_by_integrals(SPARC_OBJ *pSPARC, double *force_hub, double *alpha);

void Calculate_hubbard_forces_kpt(SPARC_OBJ *pSPARC);
void Compute_Integral_psi_Orb_kpt(SPARC_OBJ *pSPARC, double _Complex *beta, double _Complex *Xorb_kpt, int kpt);
void Compute_Integral_Orb_Dpsi_kpt(SPARC_OBJ *pSPARC, double _Complex *dpsi, double _Complex *beta, int kpt);
void Compute_force_hubbard_by_integrals_kpt(SPARC_OBJ *pSPARC, double *force_hub, double _Complex *alpha);

#endif