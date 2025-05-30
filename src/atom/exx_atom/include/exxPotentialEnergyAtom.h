#ifndef EXXPOTENTIALENERGYATOM_H
#define EXXPOTENTIALENERGYATOM_H

#include "isddftAtom.h"

void exxOperatorAtom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void evaluateExxPotentialAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, int l, double spin, double *VexxL);
void evaluateExxEnergyAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *Exx);

#endif