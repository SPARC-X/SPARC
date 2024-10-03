/***
 * @file    exactExchange.h
 * @brief   This file contains the function declarations for Exact Exchange.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#ifndef EXACTEXCHANGE_H
#define EXACTEXCHANGE_H

#include "isddft.h"


/**
 * @brief   Outer loop of SCF using Vexx (exact exchange potential)
 */
void Exact_Exchange_loop(SPARC_OBJ *pSPARC);

/**
 * @brief   Evaluating Exact Exchange potential
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ldx, int ncol, int DMnd, double *Hx, int ldhx, int spin, MPI_Comm comm);

/**
 * @brief   Evaluate Exact Exchange potential using non-ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param occ_outer       Full set of occ_outer occupations
 * @param psi_outer       Full set of psi_outer orbitals
 * @param Hx              Result of Hx plus fock operator times X 
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ldx, int ncol, int DMnd, 
                                    double *occ_outer, double *psi_outer, int ldpo, double *Hx, int ldhx, MPI_Comm comm);

/**
 * @brief   Evaluate Exact Exchange potential using ACE operator
 *          
 * @param X               The vectors premultiplied by the Fock operator
 * @param ncol            Number of columns of vector X
 * @param DMnd            Number of FD nodes in comm
 * @param Xi              Xi of ACE operator 
 * @param Hx              Result of Hx plus Vx times X
 * @param spin            Local spin index
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential_ACE(SPARC_OBJ *pSPARC, double *X, int ldx, 
    int ncol, int DMnd, double *Xi, int ldxi, double *Hx, int ldhx, MPI_Comm comm);


/**
 * @brief   Evaluate Exact Exchange Energy
 */
void exact_exchange_energy(SPARC_OBJ *pSPARC);


/**
 * @brief   Evaluate Exact Exchange Energy
 */
void evaluate_exact_exchange_energy(SPARC_OBJ *pSPARC);


/**
 * @brief   Solving Poisson's equation using FFT or KRON
 *          
 *          This function only works for solving Poisson's equation with real right hand side
 *          option: 0 - solve poissons equation,       1 - solve with pois_const_stress
 *                  2 - solve with pois_const_stress2, 3 - solve with pois_const_press
 */
void poissonSolve(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, 
                int ncol, int DMnd, double *Vi, MPI_Comm comm);

#ifdef __cplusplus
extern "C" {
#endif
void parallel_info_dp2bp(int gridsizes[3], int DMnd, int ncol, 
    int (*DMVertices)[6], int *sendcounts, int *sdispls, int *recvcounts, int *rdispls, int *kq_shift, int sing_size, MPI_Comm cart_comm);

#ifdef __cplusplus
}
#endif

/**
 * @brief   Solve Poisson's equation using kronecker product of Laplacian
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const        constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_kron(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, int ncol, double *sol);

/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs               complete RHS of poisson's equations without parallelization. 
 * @param pois_const    constant for solving possion's equations
 * @param ncol              Number of poisson's equations to be solved.
 * @param sol               complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_fft(SPARC_OBJ *pSPARC, double *rhs, double *pois_const, int ncol, double *sol);

/**
 * @brief   Gather psi_outers in other bandcomms
 *
 *          The default comm is blacscomm
 */
void gather_psi_occ_outer(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer);

/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE(SPARC_OBJ *pSPARC);

/**
 * @brief   Create ACE operator in dmcomm
 *
 *          Using occupied + extra orbitals to construct the ACE operator 
 *          Due to the symmetry of ACE operator, only half Poisson's 
 *          equations need to be solved.
 */
void ACE_operator(SPARC_OBJ *pSPARC, double *psi, double *occ, double *Xi);

void solving_for_Xi(SPARC_OBJ *pSPARC, double *psi, double *occ, double *Xi);

void calculate_ACE_operator(SPARC_OBJ *pSPARC, double *psi, double *Xi);

/**
 * @brief   Gather orbitals shape vectors across blacscomm
 */
void gather_blacscomm(SPARC_OBJ *pSPARC, int Nrow, int Ncol, double *vec);

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief   Solve half of poissons equation locally and apply to Xi
 */
void solve_half_local_poissons_equation_apply2Xi(SPARC_OBJ *pSPARC, int ncol, double *psi, int ldp, double *occ, double *Xi, int ldxi);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief   transfer orbitals in a cyclic rotation way to save memory
 */
void transfer_orbitals_blacscomm(SPARC_OBJ *pSPARC, void *sendbuff, void *recvbuff, int shift, MPI_Request *reqs, int unit_size);

#ifdef __cplusplus
}
#endif

/**
 * @brief   Sovle all pair of poissons equations by remote orbitals and apply to Xi
 */
void solve_allpair_poissons_equation_apply2Xi(
    SPARC_OBJ *pSPARC, int ncol, double *psi, int ldp, double *psi_storage, int ldps, double *occ, double *Xi, int ldxi, int shift);

#endif // EXACTEXCHANGEPOTENTIAL_H 