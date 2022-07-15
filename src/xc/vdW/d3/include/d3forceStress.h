/**
 * @file    d3forceStress.h
 * @brief   This file contains the declarations of functions for adding forces and stresses from DFT-D3 into total forces and stresses.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * S.Grimme, J.Antony, S.Ehrlich, H.Krieg, A consistent and accurate ab
 * initio parametrization of density functional dispersion correction
 * (DFT-D) for the 96 elements H-Pu
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef D3_FORCESTR
#define D3_FORCESTR

#include "isddft.h"

/**
 * @brief The function to compute stress of DFT-D3. Called by Calculate_XC_stress in stress.c
 */
void d3_grad_cell_stress(SPARC_OBJ *pSPARC);

/**
 * @brief The function to add atomic force of DFT-D3 to total atomic force. Called by Calculate_electronicGroundState in electronicGroundState.c
 */
void add_d3_forces(SPARC_OBJ *pSPARC);

#endif