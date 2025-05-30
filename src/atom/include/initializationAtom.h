#ifndef INITIALIZATIONATOM_H
#define INITIALIZATIONATOM_H

#include "atomicStates.h"
#include "readfiles.h"
#include "toolsAtom.h"
#include "isddftAtom.h"
#include "spectralFunctionsAtom.h"
#include "isddft.h"

void Initialize_Atom(SPARC_ATOM_OBJ *pSPARC_ATOM, SPARC_OBJ *pSPARC, SPARC_INPUT_OBJ *pSPARC_Input, int ityp);
void set_defaults_Atom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void copy_PSP_atom(SPARC_ATOM_OBJ *pSPARC_ATOM, SPARC_OBJ *pSPARC, int ityp);
void init_rho_NLCC(SPARC_ATOM_OBJ *pSPARC_ATOM);
void xc_decomposition_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void Calculate_SplineDerivRadFun_atom(SPARC_ATOM_OBJ *pSPARC_ATOM);
void setValence(SPARC_ATOM_OBJ *pSPARC_ATOM);

#endif