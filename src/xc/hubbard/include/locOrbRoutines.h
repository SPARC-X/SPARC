#ifndef LOCORBROUTINES_H
#define LOCORBROUTINES_H

#include "isddft.h"

void GetInfluencingAtoms_loc(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ **Atom_Influence_loc, int *DMVertices, MPI_Comm comm);

void CalculateLocalProjectors(SPARC_OBJ *pSPARC, LOC_PROJ_OBJ **locProj,
    ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, int *DMVertices, MPI_Comm comm);

void CalculateLocalProjectors_kpt(SPARC_OBJ *pSPARC, LOC_PROJ_OBJ **locProj,
    ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, int *DMVertices, MPI_Comm comm);

void CalculateLocalInnerProductIndex(SPARC_OBJ *pSPARC);

void Vhub_vec_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
    LOC_PROJ_OBJ *locProj, int ncol, double *x, int ldi, double *Hx, int ldo, int spin, MPI_Comm comm);

void Vhub_vec_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
    LOC_PROJ_OBJ *locProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int kpt, int spin, MPI_Comm comm);

#endif