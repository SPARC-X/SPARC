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
    double _Complex *psi, double *occ_outer, double _Complex *Xi_kpt);

void solving_for_Xi_kpt(SPARC_OBJ *pSPARC, 
    double _Complex *psi, double *occ_outer, double _Complex *Xi_kpt);
    
void calculate_ACE_operator_kpt(SPARC_OBJ *pSPARC, double _Complex *psi, double _Complex *Xi_kpt);

/**
 * @brief   Evaluating Exact Exchange potential in k-point case
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, 
        double _Complex *X, int ldx, int ncol, int DMnd, double _Complex *Hx, int ldhx, int spin, int kpt, MPI_Comm comm);


/**
 * @brief   Evaluate Exact Exchange potential using non-ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param occ_outer       Full set of occ_outer occupations
 * @param psi_outer       Full set of psi_outer orbitals
 * @param Hx              Result of Hx plus fock operator times X 
 * @param kpt_k           Local index of each k-point 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_kpt(SPARC_OBJ *pSPARC, double _Complex *X, int ldx, 
        int ncol, int DMnd, double *occ_outer, 
        double _Complex *psi_outer, int ldpo, double _Complex *Hx, int ldhx, int kpt_k, MPI_Comm comm);


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
        double _Complex *X, int ldx, int ncol, int DMnd, double _Complex *Xi, int ldxi,
        double _Complex *Hx, int ldhx, int kpt, MPI_Comm comm);


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
void poissonSolve_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_const, int ncol, int DMnd, 
                double _Complex *sol, int kpt_k, int kpt_q, MPI_Comm comm);


/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space - in k-point case. 
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * @param kpt_k_list        List of global index of k Bloch wave vector
 * @param kpt_q_list        List of global index of q Bloch wave vector
 * Note:                    Assuming the RHS is periodic with Bloch wave vector (k - q). 
 * Note:                    This function is complete localized. 
 */
void pois_fft_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_const, 
                    int ncol, double _Complex *sol, int kpt_k, int kpt_q);


/**
 * @brief   Solve Poisson's equation using Kronecker product Lapacian
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * @param kpt_k_list        List of global index of k Bloch wave vector
 * @param kpt_q_list        List of global index of q Bloch wave vector
 * Note:                    Assuming the RHS is periodic with Bloch wave vector (k - q). 
 * Note:                    This function is complete localized. 
 */
void pois_kron_kpt(SPARC_OBJ *pSPARC, double _Complex *rhs, double *pois_const, 
                    int ncol, double _Complex *sol, int kpt_k, int kpt_q);


/**
 * @brief   Gather psi_outer_kpt and occupations in other bandcomms and kptcomms
 * 
 *          In need of communication between k-point topology and dmcomm, additional 
 *          communication for occupations is required. 
 */
void gather_psi_occ_outer_kpt(SPARC_OBJ *pSPARC, double _Complex *psi_outer_kpt, double *occ_outer);


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
void apply_phase_factor(SPARC_OBJ *pSPARC, double _Complex *vec, int ncol, char *NorP, int kpt_k, int kpt_q);

/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE_kpt(SPARC_OBJ *pSPARC);

/**
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm_kpt(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double _Complex *vec);

/**
 * @brief   Gather orbitals shape vectors across kpt_bridge_comm
 */
void gather_kptbridgecomm_kpt(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double _Complex *vec);

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief   transfer orbitals in a cyclic rotation way to save memory
 */
void transfer_orbitals_kptbridgecomm(SPARC_OBJ *pSPARC, 
        void *sendbuff, void *recvbuff, int shift, MPI_Request *reqs, int unit_size);

/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi_kpt(SPARC_OBJ *pSPARC, 
    int ncol, double _Complex *psi, int ldp, double _Complex *psi_storage, int ldps, double *occ, double _Complex *Xi_kpt, int ldxi, 
    int kpt_k, int kpt_q, int shift);

#ifdef __cplusplus
}
#endif

#endif // EXACTEXCHANGEPOTENTIAL_KPT_H 