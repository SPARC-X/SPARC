#ifndef OCCUPATIONMATRIX_H
#define OCCUPATIONMATRIX_H

#include "isddft.h"

// void init_occ_mat(SPARC_OBJ *pSPARC);
void RefreshOccMatHistory(SPARC_OBJ *pSPARC);
void init_occ_mat_scf(SPARC_OBJ *pSPARC);
void occMatExtrapolation(SPARC_OBJ *pSPARC);
void CalculateOccMatAtomIndex(SPARC_OBJ *pSPARC);
void Calculate_Occupation_Matrix(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj);
void Calculate_occMat(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj);
void Calculate_occMat_kpt(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj);
// void Mixing_Occ_Mat(SPARC_OBJ *pSPARC, int iter_count);
void mixing_rho_mn(SPARC_OBJ *pSPARC, int iter_count);
// void mixing_rho_mn_kpt(SPARC_OBJ *pSPARC, int iter_count);
void communicate_mix_gamma(SPARC_OBJ *pSPARC);
void print_Occ_mat(SPARC_OBJ *pSPARC);


#endif