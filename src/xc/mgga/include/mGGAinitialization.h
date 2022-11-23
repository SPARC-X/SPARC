/**
 * @file    MGGAinitialization.h
 * @brief   This file contains the function initializing variables needed by MGGA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MGGA_INITIAL
#define MGGA_INITIAL

#include "isddft.h"

/**
 * @brief   allocate space to MGGA variables
 */
void initialize_MGGA(SPARC_OBJ *pSPARC);

#endif