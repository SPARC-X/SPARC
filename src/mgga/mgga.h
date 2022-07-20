/**
 * @file    mgga.h
 * @brief   This file contains the function declarations for metaGGA functionals.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef MGGA_H
#define MGGA_H 

/**
 * @brief   allocate space to variables, initialize countSCF
 */
void initialize_MGGA(SPARC_OBJ *pSPARC);

/**
 * @brief   compute the kinetic energy density tau and transfer it to phi domain for computing Vxc of metaGGA
 *          
 */
void compute_Kinetic_Density_Tau_Transfer_phi(SPARC_OBJ *pSPARC);

/**
 * @brief   the main function in the file, compute needed xc energy density and its potentials, then transfer them to needed communicators
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
 * @brief   the function to compute the exchange-correlation energy of metaGGA functional
 *          
 * @param rho               electron density vector
 */
void Calculate_Exc_MGGA(SPARC_OBJ *pSPARC,  double *rho);

/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from phi-domain to psi-domain.
 *          
 * @param vxcMGGA3_phi_domain               the vxcMGGA3 vector in phi-domain
 * @param vxcMGGA3_psi_domain               the vxcMGGA3 vector in psi-domain
 */
void Transfer_vxcMGGA3_phi_psi(SPARC_OBJ *pSPARC, double *vxcMGGA3_phi_domain, double *vxcMGGA3_psi_domain);

/**
 * @brief   Transfer vxcMGGA3 (d(n epsilon)/d(tau)) from psi-domain to k-point topology.
 *          
 * @param vxcMGGA3_psi_domain               the vxcMGGA3 vector in psi-domain
 * @param vxcMGGA3_kpt_topo               the vxcMGGA3 vector in k-point topology
 */
void Transfer_vxcMGGA3_psi_kptTopo(SPARC_OBJ *pSPARC, double *vxcMGGA3_psi_domain, double *vxcMGGA3_kpt_topo);

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult
 *          
 * @param x               the vector saving vectors (possible wave functions psi) to be multiplied by Hamiltonian operator
 * @param ncol            the number of vectors to be multiplied
 * @param colLength       the length of the vector (possible wave functions psi) in this processor in this communicator
 * @param DMVertices      the vector saving indexes of the space saved in this processor in this communicator
 * @param vxcMGGA3_dm     the vector saving vxcMGGA3 (d(n epsilon)/d(tau))
 * @param mGGAterm        the vector saving the mGGA term in Hamiltonian, which is also the result of the function
 * @param spin            number of spin (or whether there is spin polarization). Now mGGA or SCAN odes not have it so it should be equal to 1
 * @param comm            communicator controlling the multiplication H.x
 */
void compute_mGGA_term_hamil(const SPARC_OBJ *pSPARC, double *x, int ncol, int colLength, int *DMVertices, double *vxcMGGA3_dm, double *mGGAterm, int spin, MPI_Comm comm);

/**
 * @brief   the function to compute the mGGA term in Hamiltonian, called by Hamiltonian_vectors_mult_kpt
 *          
 * @param x               the vector saving vectors (possible wave functions psi) to be multiplied by Hamiltonian operator
 * @param ncol            the number of vectors to be multiplied
 * @param colLength       the length of the vector (possible wave functions psi) in this processor in this communicator
 * @param DMVertices      the vector saving indexes of the space saved in this processor in this communicator
 * @param vxcMGGA3_dm     the vector saving vxcMGGA3 (d(n epsilon)/d(tau))
 * @param mGGAterm     the vector saving the mGGA term in Hamiltonian, which is also the result of the function
 * @param spin            number of spin (or whether there is spin polarization). Now mGGA or SCAN odes not have it so it should be equal to 1
 * @param kpt             the index of the k-point in this multiplication
 * @param comm            communicator controlling the multiplication H.x
 */
void compute_mGGA_term_hamil_kpt(const SPARC_OBJ *pSPARC, double _Complex *x, int ncol, int colLength, int *DMVertices, double *vxcMGGA3_dm, double _Complex *mGGAterm, int spin, int kpt, MPI_Comm comm);

/**
 * @brief   compute the metaGGA psi stress term, gamma point
 */
void Calculate_XC_stress_mGGA_psi_term(SPARC_OBJ *pSPARC);

/**
 * @brief   compute the metaGGA psi stress term, k-point
 */
void Calculate_XC_stress_mGGA_psi_term_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   free space allocated to MGGA variables
 */
void free_MGGA(SPARC_OBJ *pSPARC);

#endif // MGGA_H 