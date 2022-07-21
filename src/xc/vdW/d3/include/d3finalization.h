/**
 * @file    d3finalization.h
 * @brief   This file contains the declaration of function freeing variables and coefficients of DFT-D3.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * S.Grimme, J.Antony, S.Ehrlich, H.Krieg, A consistent and accurate ab
 * initio parametrization of density functional dispersion correction
 * (DFT-D) for the 96 elements H-Pu
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef D3_FINAL
#define D3_FINAL 

/**
 * @brief function freeing variables and coefficients of DFT-D3.
 */
void free_D3_coefficients(SPARC_OBJ *pSPARC);

#endif