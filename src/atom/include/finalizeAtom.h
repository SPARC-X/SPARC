#ifndef FINALIZEATOM_H
#define FINALIZEATOM_H

#include "isddftAtom.h"
#include "isddft.h"

void printResultsAtom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void evaluateEnergyAtom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void copyAtomSolution(SPARC_ATOM_OBJ *pSPARC_ATOM, SPARC_OBJ *pSPARC, int ityp);
void Finalize_Atom(SPARC_ATOM_OBJ *pSPARC_ATOM);

#endif