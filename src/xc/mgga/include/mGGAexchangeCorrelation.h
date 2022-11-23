/**
 * @file    MGGAexchangeCorrelation.h
 * @brief   This file contains the function declarations for computing metaGGA potentials.
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
 * @brief   the main function in the file, compute epsilon and XCPotential in phi domain, then transfer potential Vxc3 to psi domain
 *          
 * @param rho               electron density vector
 */
void Calculate_transfer_Vxc_MGGA(SPARC_OBJ *pSPARC,  double *rho);

/**
 * @brief   compute epsilon and XCPotential; vxcMGGA3 of metaGGA functional
 *          
 * @param rho               electron density vector
 */
void Calculate_Vxc_MGGA(SPARC_OBJ *pSPARC,  double *rho);

/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from phi-domain to psi-domain.
 *          
 * @param vxcMGGA3_phi_domain               the vxcMGGA3 vector in phi-domain
 * @param vxcMGGA3_psi_domain               the vxcMGGA3 vector in psi-domain
 */
void Transfer_vxcMGGA3_phi_psi(SPARC_OBJ *pSPARC, double *vxcMGGA3_phi_domain, double *vxcMGGA3_psi_domain);

/**
 * @brief   the function to compute the exchange-correlation energy of metaGGA functional
 *          
 * @param rho               electron density vector
 */
void Calculate_Exc_MGGA(SPARC_OBJ *pSPARC,  double *rho);

#endif 