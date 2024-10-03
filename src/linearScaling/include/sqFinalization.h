/**
 * @file    sqFinalization.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQFINALIZATION_H
#define SQFINALIZATION_H 

#include "isddft.h"

/**
 * @brief   Free all allocated memory spaces and communicators 
 */
void Free_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Free SCF variables for SQ methods.
 */
void Free_scfvar_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Free SCF variables for SQ-hybrid
 */
void free_exx_SQ(SPARC_OBJ *pSPARC);

#endif // SQFINALIZATION_H