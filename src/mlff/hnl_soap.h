#ifndef HNL_SOAP_H
#define HNL_SOAP_H

#include "mlff_types.h"
#include "isddft.h"

double compute_fcut(double r, double sigma_atom, double rcut);

double compute_der_fcut(double r, double sigma_atom, double rcut);

double compute_hnl(int n, int l, double r, double rcut, double sigma_atom);

double compute_d_hnl(int n, int l, double r, double rcut, double sigma_atom);

void compute_hnl_soap(SPARC_OBJ *pSPARC, MLFF_Obj *mlff_str);

#endif
