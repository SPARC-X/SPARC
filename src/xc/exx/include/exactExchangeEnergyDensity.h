/***
 * @file    exactExchangeEenrgyDensity.h
 * @brief   This file contains the function declarations for Exact Exchange energy density.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGEENERGYDENSITY_H
#define EXACTEXCHANGEENERGYDENSITY_H

#include "isddft.h"

/**
 * @brief   Compute exact exchange energy density
 */
void computeExactExchangeEnergyDensity(SPARC_OBJ *pSPARC, double *Exxrho);

#endif