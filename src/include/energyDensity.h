/**
 * @file    energyDensity.h
 * @brief   This file contains the declarations of functions for 
 *          calculating energy densities
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#ifndef ENERGYDENSITY_H
#define ENERGYDENSITY_H

#include "isddft.h"

/**
 * @brief   compute kinetic energy density \tau
 */
void compute_Kinetic_Density_Tau(SPARC_OBJ *pSPARC, double *Krho);


#endif