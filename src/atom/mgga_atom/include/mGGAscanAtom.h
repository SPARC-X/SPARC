#ifndef MGGASCANATOM_H
#define MGGASCANATOM_H

#include "isddftAtom.h"

void scanx(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

void basic_MGGA_variables(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau);

void Calculate_scanx(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx1, double *vx2, double *vx3);

void scanc(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

void Calculate_scanc(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonc, double *vc1, double *vc2, double *vc3);

void scanx_spin(int DMnd, double *rho, double *sigma, double *tau, double *ex, double *vx, double *v2x, double *v3x);

void basic_MGSGA_variables_exchange(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau);

void Calculate_scanx_spin(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadn_daddn_dadtau, double *epsilonx, double *vx1, double *vx2, double *vx3);

void scanc_spin(int DMnd, double *rho, double *sigma, double *tau, double *ec, double *vc, double *v2c, double *v3c);

void basic_MGSGA_variables_correlation(int length, double *rho, double *normDrho, double *tau, double **s_dsdn_dsddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn);

void Calculate_scanc_spin(int length, double *rho, double **s_dsdn_dsddn, double **alpha_dadnup_dadndn_daddn_dadtau, double **zeta_dzetadnup_dzetadndn, double *epsilonc, double *vc1, double *vc2, double *vc3);

#endif