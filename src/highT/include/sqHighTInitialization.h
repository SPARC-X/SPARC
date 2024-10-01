/**
 * @file    sqHighTInitialization.h
 * @brief   This file contains the function declarations for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef SQHIGHTINITIALIZATION_H
#define SQHIGHTINITIALIZATION_H 

#include "isddft.h"
/**
 * @brief   Initialize SQ variables and create necessary communicators
 */
void init_SQ_HighT(SPARC_OBJ *pSPARC);

void init_exx_SQ_highT(SPARC_OBJ *pSPARC);

double memory_estimation_SQ_highT(const SPARC_OBJ *pSPARC);

#endif // SQHIGHTINITIALIZATION_H