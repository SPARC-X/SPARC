#ifndef SPECTRALFUNCTIONSATOM_H
#define SPECTRALFUNCTIONSATOM_H

#include "isddftAtom.h"

void set_grid_grad_lap(SPARC_ATOM_OBJ *pSPARC_ATOM);

void chebD(int N, double R, double* D, double* r);

void clencurt(int N, double* w);

#endif