/**
 * @file    copyC6.h
 * @brief   This file contains the declaration of function returning the array saving all C6 coefficient of DFT-D3.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * S.Grimme, J.Antony, S.Ehrlich, H.Krieg, A consistent and accurate ab
 * initio parametrization of density functional dispersion correction
 * (DFT-D) for the 96 elements H-Pu
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef COPYC6_H
#define COPYC6_H 

/**
 * @brief return the array containing all C6 coefficients
 */
double *****copyC6();

#endif