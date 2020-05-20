/**
 * @file    ExtVecAccel.h
 * @brief   External Vectorization and Acceleration (EVA) module
 *          Module header file
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */

#ifndef __EXT_VEC_ACCEL_H__
#define __EXT_VEC_ACCEL_H__

// EVA_buff structure for storing intermediate variables in EVA module
#include "EVA_buff.h"

// Functions for multiplying Laplacian matrix with multiple vectors
#include "EVA_Lap_MV_orth.h"
#include "EVA_Lap_MV_nonorth.h"

// Functions for multiplying Vnl operator with multiple vectors
#include "EVA_Vnl_MV.h"

// Functions for multiplying Hamiltonian matrix with multiple vectors
#include "EVA_Hamil_MV.h"

// Functions for Chebyshev Filtering Subspace Iteration 
#include "EVA_CheFSI.h"

#endif
