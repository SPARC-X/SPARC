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
 * @brief   This function gets the overlapping nodes of each Rcut cuboidal or spherical domain of 
 *          each FD node in current domain and atoms and their images' rc spherical domain. In current 
 *          implementation, only nloc_mem_flag in SQDFT is implemented, i.e., saving nonlocal projector 
 *          chi matrix for each FD node. 
 * 
 * @TODO     Implement the nloc_mem_flag option if memory issue happens.
 */
void GetNonlocalProjectorsForNode(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ *nlocProj, NLOC_PROJ_OBJ ***nlocProj_SQ, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, ATOM_NLOC_INFLUENCE_OBJ ***Atom_Influence_nloc_SQ, MPI_Comm comm);

/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
void Vnl_vec_mult_SQ(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, const double *x, double *Hx);

/**
 * @brief   Calculate Vnl times vectors in a matrix-free way for force calculation.
 */
void Vnl_vec_mult_J_SQ(const SPARC_OBJ *pSPARC, int DMnd, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, double *x, double *Vx);
                  
/**
 * @brief   Calculate Vnl times vectors in a matrix-free way for stress calculation.
 */
void Vnl_vec_mult_dir_SQ(const SPARC_OBJ *pSPARC, int DMnd, int nd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, int dir, double *x, double *Vx);
                  
#endif