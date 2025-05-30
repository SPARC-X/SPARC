#ifndef EXXTOOLSATOM_H
#define EXXTOOLSATOM_H

#include "isddftAtom.h"
void exx_initialization_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void densityMatrix(SPARC_ATOM_OBJ *pSPARC_ATOM, double *denMat);
double Wigner3j(int j1, int j2, int j);
double factorial(int n);
// void findMinMax(int *arr, int n, int *min, int *max);

#endif