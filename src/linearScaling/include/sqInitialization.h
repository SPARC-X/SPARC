/**
 * @file    sqInitialization.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQINITIALIZATION_H
#define SQINITIALIZATION_H 

#include "isddft.h"
/**
 * @brief   Initialize SQ variables and create necessary communicators
 */
void init_SQ(SPARC_OBJ *pSPARC);

/**
 * @brief   Allocate what the Lanczos runs use (init_SQ, init_SQ_HighT): Veff of the box, 1D Laplacians, eigensolver
 *          workspace, JIT kernel tables, the current point's nonlocal projectors
 */
void init_SQ_lanczos(SPARC_OBJ *pSPARC);

double memory_estimation_SQ(const SPARC_OBJ *pSPARC);

#endif // SQINITIALIZATION_H