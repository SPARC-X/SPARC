/**
 * @file    mGGAtauTransferTauVxc.h
 * @brief   This file contains the function declarations for computing kinetic density tau and 
 * transferring tau and Vxc from phi domain to psi domain.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MGGA_EXC
#define MGGA_EXC

#include "isddft.h"


/**
 * @brief   compute kinetic energy density \tau
 */
void compute_Kinetic_Density_Tau(SPARC_OBJ *pSPARC, double *Krho);

/**
 * @brief   compute the kinetic energy density tau and transfer it to phi domain for computing Vxc of metaGGA
 *          
 */
void compute_Kinetic_Density_Tau_Transfer_phi(SPARC_OBJ *pSPARC);


/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from phi-domain to psi-domain.
 *          
 * @param vxcMGGA3_phi_domain               the vxcMGGA3 vector in phi-domain
 * @param vxcMGGA3_psi_domain               the vxcMGGA3 vector in psi-domain
 */
void Transfer_vxcMGGA3_phi_psi(SPARC_OBJ *pSPARC, double *vxcMGGA3_phi_domain, double *vxcMGGA3_psi_domain);


#endif 