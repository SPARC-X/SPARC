/***
 * @file    exactExchangeFinalization.h
 * @brief   This file contains the function declarations for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGEFINALIZATION_H
#define EXACTEXCHANGEFINALIZATION_H

#include "isddft.h"


/**
 * @brief   Memory free of all variables for exact exchange.
 */
void free_exx(SPARC_OBJ *pSPARC);

void free_singularity_removal_const(SPARC_OBJ *pSPARC);

#endif // EXACTEXCHANGEFINALIZATION_H 