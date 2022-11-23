/**
 * @file    MGGAinitialization.h
 * @brief   This file contains the declaration of function freeing variables needed by MGGA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MGGA_FINAL
#define MGGA_FINAL

#include "isddft.h"

/**
 * @brief   free space allocated to MGGA variables
 */
void free_MGGA(SPARC_OBJ *pSPARC);

#endif