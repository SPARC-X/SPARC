/**
 * @file    orbitalElecDensInit.h
 * @brief   This file declares the functions for electron density initialization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef ORBITALELECDENSINIT_H
#define ORBITALELECDENSINIT_H

#include "isddft.h"

/**
 * @brief   Initialze electron density.
 */
void Init_electronDensity(SPARC_OBJ *pSPARC);

/*
 @ brief: function to perform charge extrapolation to provide better rho_guess for future relax/MD steps
*/
void elecDensExtrapolation(SPARC_OBJ *pSPARC);

/**
 * @brief   initialize Kohn-Sham orbitals.
 */
void Init_orbital(SPARC_OBJ *pSPARC);

#endif // ORBITALELECDENSINIT_H

