#ifndef LOCORBROUTINES_H
#define LOCORBROUTINES_H

#include "isddft.h"

void GetInfluencingAtoms_loc(SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ **Atom_Influence_loc, int *DMVertices, MPI_Comm comm);

void CalculateLocalProjectors(SPARC_OBJ *pSPARC, LOC_PROJ_OBJ **locProj,
    ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, int *DMVertices, MPI_Comm comm);

void CalculateLocalProjectors_kpt(SPARC_OBJ *pSPARC, LOC_PROJ_OBJ **locProj,
    ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, int *DMVertices, MPI_Comm comm);

void CalculateLocalInnerProductIndex(SPARC_OBJ *pSPARC);

// void Vhub_vec_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
//     LOC_PROJ_OBJ *locProj, int ncol, double *x, int ldi, double *Hx, int ldo, int spin, MPI_Comm comm);
void Vhub_vec_mult(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
    LOC_PROJ_OBJ *locProj, int ncol, double *x, int ldi, double *Hx, int ldo, int spin, MPI_Comm comm);

// void Vhub_vec_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
//     LOC_PROJ_OBJ *locProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int kpt, int spin, MPI_Comm comm);
void Vhub_vec_mult_kpt(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc,
    LOC_PROJ_OBJ *locProj, int ncol, double _Complex *x, int ldi, double _Complex *Hx, int ldo, int spin, int kpt, MPI_Comm comm);

/**
 * @brief   out += scale * Phi~^H x for every U atom touching the local domain, without the sum over the domain
 *          (occupation matrix). kpt = -1: Gamma point (real x, out); otherwise complex, as pairs of doubles.
 */
void Vhub_inner_product(const SPARC_OBJ *pSPARC, ATOM_LOC_INFLUENCE_OBJ *Atom_Influence_loc, LOC_PROJ_OBJ *locProj,
    int ncol, const double *x, int ldi, double scale, double *out, int kpt, MPI_Comm comm);

/**
 * @brief   Projections of the bands of this process for the forces, stress and pressure: out += scale * sum_R w_R Phi_R^T x
 *          (Bloch factor at a k-point, times e . (x - R) if e != NULL; complex conjugate added if conj_out).
 */
void Vhub_inner_product_bands(const SPARC_OBJ *pSPARC, const double *x, int kpt, const double *e, double scale, int conj_out,
    double *out);

/**
 * @brief   Free the weighted (stress/pressure) folded orbitals; call after the stress or pressure.
 */
void Vhub_free_weighted(void);

/**
 * @brief   Free the folded orbitals and work space of the Hubbard product; call whenever the Hubbard projectors change.
 */
void Vhub_free_plans(void);

#endif