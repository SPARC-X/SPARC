#ifndef MGGARSCANATOM_H
#define MGGARSCANATOM_H

#include "isddftAtom.h"

void rscanx(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

void basic_rscan_variables(int length, double *rho, double *normDrho, double *tau, double **p_dpdn_dpddn, double **alphaP_dadn_daddn_dadtau);

void Calculate_rscanx(int length, double *rho, double **p_dpdn_dpddn, double **alphaP_dadn_daddn_dadtau, double *epsilonx, double *vx, double *v2x, double *v3x);

void rscanc(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

void Calculate_rscanc(int length, double *rho, double **s_dsdn_dsddn, double **alphaP_dadn_daddn_dadtau, double *epsilonc, double *vc, double *v2c, double *v3c);

void rscanx_spin(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

void rscanc_spin(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

void basic_rscanc_spin_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alphaP_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn);

void Calculate_rscanc_spin(int length, double *rho, double **s_dsdn_dsddn, double **alphaP_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc, double *v2c, double *v3c);

#endif