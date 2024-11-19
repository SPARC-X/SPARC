/**
 * @file    electronicGroundState.h
 * @brief   This file declares the functions for electronic Ground-state calculation.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef GROUNDSTATE_H
#define GROUNDSTATE_H

#include "isddft.h"


/**
 * @brief   Calculate properties for fixed atom positions using DFT or MLFF
 * 
 * @param pSPARC  
 */
void Calculate_Properties(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate the ground state energy and forces for fixed atom positions.  
 *
 * @param pSPARC  
 */
void Calculate_electronicGroundState(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate electronic ground state electron density and energy 
 *          for fixed atom positions.  
 *
 * @param pSPARC   
 */
void Calculate_EGS_elecDensEnergy(SPARC_OBJ *pSPARC);


///**
// * @brief   Calculate electronic ground state forces for fixed atom positions.  
// *
// * @param pSPARC   
// */
//void Calculate_EGS_Forces(SPARC_OBJ *pSPARC);


/**
 * @brief   KS-DFT self-consistent field (SCF) calculations.
 */
void scf(SPARC_OBJ *pSPARC);

/**
 * @brief   KS-DFT self-consistent field (SCF) calculations.
 */
void scf_loop(SPARC_OBJ *pSPARC);


/**
 * @ brief Evaluate SCF error.
 *
 *         Depending on whether it's density mixing or potential mixing, 
 *         the scf error is defines as
 *           scf error := || rho  -  rho_old || / ||  rho ||, or
 *           scf error := || Veff - Veff_old || / || Veff ||.
 *         Note that Veff and Veff_old here are shifted so that their 
 *         mean value is 0.
 */
void Evaluate_scf_error(SPARC_OBJ *pSPARC, double *error, int *scf_conv);


/**
 * @brief Evaluate the scf error defined in Quantum Espresso.
 *
 *        Find the scf error defined in Quantum Espresso. QE implements 
 *        Eq.(A.7) in the reference paper below, with a slight modification: 
 *          conv_thr := 4 \pi e^2 \Omega \sum_G |\Delta \rho(G)|^2 / G^2
 *        This is equivalent to 2 * Eq.(A.6) in the reference paper.
 *
 * @ref   P Giannozzi et al, J. Phys.:Condens. Matter 21(2009) 395502.
 */
void Evaluate_QE_scf_error(SPARC_OBJ *pSPARC, double *scf_error, int *scf_conv);


/**
* @ brief find net magnetization of the system
**/
void Calculate_magnetization(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate Veff_loc = phi + Vxc (in phi-domain).
 */
void Calculate_Veff_loc_dmcomm_phi(SPARC_OBJ *pSPARC);


/**
 * @brief   Update mixing_hist_xk.
 */
void Update_mixing_hist_xk(SPARC_OBJ *pSPARC);


/**
 * @brief   shift each column of Veff with mean value
 */
void shifting_Veff_mean(SPARC_OBJ *pSPARC, double *Veff, int ncol, double *Veff_mean, int option, int dir);

/**
 * @brief   Transfer Veff_loc from phi-domain to psi-domain.
 *
 *          For now, we just gather Veff_loc in one process and create a global 
 *          Veff_loc vector and then broadcast to all processes.
 *
 *          TODO: use DD2DD (Domain Decomposition to Domain Decomposition) to 
 *          do the transmision instead.
 */
void Transfer_Veff_loc(SPARC_OBJ *pSPARC, double *Veff_phi_domain, double *Veff_psi_domain);


/**
 * @brief   Calculate Veff_loc = phi + Vxc, and transmit Veff_loc from the "phi-domain" to "psi-domain".
 *
 *          Both phi and Vxc are calculated in a Cartesian topology dmcomm_phi ("phi-domain"), but in 
 *          order to perform Chebyshev filtering, we need Veff_loc to be in dmcomm ("psi-domain"). 
 *
 *          Ideally we want each process in the dmcomm to only obtain the part of Veff_loc corresponding
 *          to the domain distributed in dmcomm. However, direct transmition of domain-distributed data
 *          between two different Cartesian topology can be messy, since one vector of length Nd is not
 *          formidable, we will simply assemble Veff_loc as the whole vector and broadcast.
 *          
 */
//void Calculate_Veff_loc(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate electron density with given states in psi-domain.
 *
 *          Note that here rho is distributed in psi-domain, which needs
 *          to be transmitted to phi-domain for solving the poisson 
 *          equation.
 */


/**
 * @brief   Transfer electron density from psi-domain to phi-domain.
 */
void TransferDensity(SPARC_OBJ *pSPARC, double *rho_send, double *rho_recv);

/**
 * @brief   Calculate magnetization of atoms
 */
void CalculateAtomMag(SPARC_OBJ *pSPARC);

#endif // GROUNDSTATE_H
