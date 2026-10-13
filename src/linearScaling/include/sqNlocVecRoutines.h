/**
 * @file    sqNlocVecRoutines.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQNLOCVECROUTINES_H
#define SQNLOCVECROUTINES_H 

/**
 * @brief   The nonlocal projectors of grid point nd: the rows of the PR-domain projectors (nlocProj_kptcomm) at the
 *          nodes within the point's Rcut sphere and box, positions relative to the point's Rcut box, into
 *          pSQ->nlocProj_pt / pSQ->Atom_Influence_nloc_pt (one point at a time, grow-only buffers). Called at the
 *          start of each Lanczos run and in the force step.
 */
void GetNonlocalProjectorsForOnePoint(SPARC_OBJ *pSPARC, int nd);

/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
// void Vnl_vec_mult_SQ(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
//                   NLOC_PROJ_OBJ *nlocProj, const double *x, double *Hx);
void Vnl_vec_mult_SQ(const SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
                  NLOC_PROJ_OBJ *nlocProj, const double *x, double *Hx);

/**
 * @brief   Calculate Vnl times vectors in a matrix-free way for force calculation.
 */
// void Vnl_vec_mult_J_SQ(const SPARC_OBJ *pSPARC, int DMnd, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
//                   NLOC_PROJ_OBJ *nlocProj, double *x, double *Vx);
void Vnl_vec_mult_J_SQ(const SPARC_OBJ *pSPARC, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
                  NLOC_PROJ_OBJ *nlocProj, double *x, double *Vx);

/**
 * @brief   Calculate Vnl times vectors in a matrix-free way for stress calculation.
 */
// void Vnl_vec_mult_dir_SQ(const SPARC_OBJ *pSPARC, int DMnd, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
//                   NLOC_PROJ_OBJ *nlocProj, int dir, double *x, double *Vx);
void Vnl_vec_mult_dir_SQ(const SPARC_OBJ *pSPARC, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc,
                  NLOC_PROJ_OBJ *nlocProj, int dir, double *x, double *Vx);
                  
#endif