/**
 * @file    finalization.h
 * @brief   This file contains the function declarations for finalization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef FINALIZATION_H
#define FINALIZATION_H 

#include "isddft.h"


/**
 * @brief   Finalize parameters and objects.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 */
void Finalize(SPARC_OBJ *pSPARC);

/**
 * @brief   Deallocates basic arrays in SPARC.
 */
void Free_basic(SPARC_OBJ *pSPARC);

/**
 * @brief   Deallocates dynamic arrays in SPARC.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 */
void Free_SPARC(SPARC_OBJ *pSPARC);

/*
@ brief: function clears the scf variables after every relax/MD step
*/
void Free_scfvar(SPARC_OBJ *pSPARC);


#endif // FINALIZATION_H 

