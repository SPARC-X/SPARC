#ifndef MGGAR2SCANATOM_H
#define MGGAR2SCANATOM_H

#include "isddftAtom.h"
void r2scanx(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

void basic_r2scanx_variables(int length, double *rho, double *normDrho, double *tau, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau);

void Calculate_r2scanx(int length, double *rho, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx, double *v2x, double *v3x);

void r2scanc(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

void basic_r2scanc_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau);

void Calculate_r2scanc(int length, double *rho, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadn_daddn_dadtau, double *epsilonc, double *vc, double *v2c, double *v3c);

void r2scanx_spin(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

void r2scanc_spin(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

void basic_r2scanc_spin_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn);

void Calculate_r2scanc_spin(int length, double *rho, double **s_dsdn_dsddn, double **p_dpdn_dpddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc, double *v2c, double *v3c);

#endif