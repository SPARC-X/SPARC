/***
 * @file    exactExchangeKpt.h
 * @brief   This file contains the function declarations for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGEKPT_H
#define EXACTEXCHANGEKPT_H

#include "isddft.h"


/**
 * @brief   Create ACE operator in dmcomm for each k-point
 *
 *          Using occupied + extra orbitals to construct the ACE operator for each k-point.
 *          Note: there is no band parallelization when usign ACE, so there is only 1 dmcomm
 *          for each k-point. 
 */
void ACE_operator_kpt(SPARC_OBJ *pSPARC, 
    double _Complex *psi_outer, double _Complex *psi, double *occ_outer, int spn_i, double _Complex *Xi_kpt);


/**
 * @brief   Evaluating Exact Exchange potential in k-point case
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ncol, int DMnd, double _Complex *Hx, int spin, int kpt, MPI_Comm comm);


/**
 * @brief   Evaluate Exact Exchange potential using non-ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param dims            3 dimensions of comm processes grid
 * @param occ_outer       Full set of occ_outer occupations
 * @param psi_outer       Full set of psi_outer orbitals
 * @param Hx              Result of Hx plus fock operator times X 
 * @param kpt_k           Local index of each k-point 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, double _Complex *X, 
        int ncol, int DMnd, int *dims, double *occ_outer, 
        double _Complex *psi_outer, double _Complex *Hx, int kpt_k, MPI_Comm comm);


/**
 * @brief   Evaluate Exact Exchange potential using ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param Xi              Xi of ACE operator 
 * @param Hx              Result of Hx plus fock operator times X 
 * @param kpt_k           Local index of each k-point 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_ACE_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ncol, int DMnd, double _Complex *Xi, 
        double _Complex *Hx, int spin, int kpt, MPI_Comm comm);


/**
 * @brief   Evaluate Exact Exchange Energy in k-point case.
 */
void evaluate_exact_exchange_energy_kpt(SPARC_OBJ *pSPARC);


/**
 * @brief   Solving Poisson's equation using FFT 
 *          
 *          This function only works for solving Poisson's equation with complex right hand side
 *          Note: only FFT method is supported in current version. 
 *          TODO: add method in real space. 
 */
void poissonSolve_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_FFT_const, int ncol, int DMnd, 
                int *dims, double _Complex *Vi, int *kpt_k_list, int *kpt_q_list, MPI_Comm comm);


/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space - in k-point case. 
 * 
 * @param rhs_loc_order     complete RHS of poisson's equations without parallelization. 
 * @param pois_FFT_const    constant for solving possion's equations
 * @param ncolp             Number of poisson's equations to be solved.
 * @param Vi_loc            complete solutions of poisson's equations without parallelization. 
 * @param kpt_k_list        List of global index of k Bloch wave vector
 * @param kpt_q_list        List of global index of q Bloch wave vector
 * Note:                    Assuming the RHS is periodic with Bloch wave vector (k - q). 
 * Note:                    This function is complete localized. 
 */
void pois_fft_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs_loc_order, double *pois_FFT_const, 
                    int ncolp, double _Complex *Vi_loc, int *kpt_k_list, int *kpt_q_list);


/**
 * @brief   Transfer complex vectors from dmcomm to kptcomm_topo for Lancozs algorithm
 *
 *          Used to transfer psi_outer_kpt in case of no-ACE method and transfer 
 *          Xi_kpt (of ACE operator) in case of ACE method from dmcomm to kptcomm_topo
 */
void Transfer_dmcomm_to_kptcomm_topo_complex(SPARC_OBJ *pSPARC, int ncols, double _Complex *psi_outer_kpt, double _Complex *psi_outer_kptcomm_topo_kpt);


/**
 * @brief   Gather psi_outer_kpt and occupations in other bandcomms and kptcomms
 * 
 *          In need of communication between k-point topology and dmcomm, additional 
 *          communication for occupations is required. 
 */
void gather_psi_occ_outer_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_outer_kpt, double *occ_outer);


/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 */
void compute_pois_fft_const_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   Find out the unique Bloch vector shifts (k-q)
 */
void find_k_shift(SPARC_OBJ *pSPARC);

/**
 * @brief   Find out the positive shift exp(i*(k-q)*r) and negative shift exp(-i*(k-q)*r)
 *          for each unique Bloch shifts
 */
void kshift_phasefactor(SPARC_OBJ *pSPARC);

/**
 * @brief   Find out the k-point for hartree-fock exact exchange in local process
 */
void find_local_kpthf(SPARC_OBJ *pSPARC);


/**
 * @brief   Apply phase factor by Bloch wave vector shifts. 
 * 
 * @param vec           vectors to be applied phase factors
 * @param ncol          number of columns of vectors
 * @param NorP          "N" is for negative phase factor, "P" is for positive.
 * @param kpt_k_list    list of global k-point index for k
 * @param kpt_q_list    list of global k-point index for q
 * Note:                Assuming the shifts is introduced by wave vector (k - q)
 */
void apply_phase_factor(SPARC_OBJ *pSPARC, double _Complex *vec, int ncol, char *NorP, int *kpt_k_list, int *kpt_q_list);


/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE_kpt(SPARC_OBJ *pSPARC);

#endif // EXACTEXCHANGEPOTENTIAL_KPT_H 