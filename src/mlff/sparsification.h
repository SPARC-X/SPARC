#ifndef SPARSIFICATION_H
#define SPARSIFICATION_H

#include "mlff_types.h"
void mlff_CUR_sparsify(int kernel_typ, double **X3, int n_descriptor, 
	int size_X3, double xi_3, dyArray *highrank_ID_descriptors, int N_low_min);

double mlff_kernel_eval(int kernel_typ, double *X3_i, double *X3_j, double xi_3, int size_X3);
#endif
