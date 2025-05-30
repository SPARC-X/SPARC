#ifndef HUBBARDINITIALIZATION_H
#define HUBBARDINITIALIZATION_H

#include "isddft.h"

void bcast_SPARC_Atom_Soln(SPARC_OBJ *pSPARC);
void set_rc_loc_orbitals(SPARC_OBJ *pSPARC);
void Calculate_SplineDerivLocOrb(SPARC_OBJ *pSPARC);
void init_occ_mat(SPARC_OBJ *pSPARC);

#endif