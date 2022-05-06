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
 * @brief   Initialization of all variables for exact exchange.
 */
void init_exx(SPARC_OBJ *pSPARC);

/**
 * @brief   Memory free of all variables for exact exchange.
 */
void free_exx(SPARC_OBJ *pSPARC);

/**
 * @brief   Create ACE operator in dmcomm
 *
 *          Using occupied + extra orbitals to construct the ACE operator 
 *          Due to the symmetry of ACE operator, only Ns_occ(Ns_occ+1)/2
 *          Poisson's equations need to be solved.
 *          Note: there is no band parallelization when usign ACE, so there 
 *          is only 1 dmcomm for each k-point. 
 */
void ACE_operator(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer, int spn_i, double *Xi);


/**
 * @brief   Evaluating Exact Exchange potential
 *          
 *          This function basically prepares different variables for kptcomm_topo and dmcomm
 */
void exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ncol, int DMnd, double *Hx, int spn_i, MPI_Comm comm);


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
 * @param comm            Communicator where the operation happens. dmcomm or kptcomm_topo
 */
void evaluate_exact_exchange_potential(SPARC_OBJ *pSPARC, double *X, int ncol, int DMnd, int *dims, 
                                    double *occ_outer, double *psi_outer, double *Hx, MPI_Comm comm);


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
void evaluate_exact_exchange_potential_ACE(SPARC_OBJ *pSPARC, 
    double *X, int ncol, int DMnd, double *Xi, double *Hx, int spin, MPI_Comm comm);


/**
 * @brief   Evaluate Exact Exchange Energy
 */
void evaluate_exact_exchange_energy(SPARC_OBJ *pSPARC);


/**
 * @brief   Solving Poisson's equation using FFT or CG
 *          
 *          This function only works for solving Poisson's equation with real right hand side
 */
void poissonSolve(SPARC_OBJ *pSPARC, double *rhs, double *pois_FFT_const, 
                    int ncol, int DMnd, int *dims, double *Vi, MPI_Comm comm);


/**
 * @brief   Rearrange Vi after receiving from root comm
 *          
 *          Vi was ordered in continuous way and now is ordered in block-separated way
 *          Note: using unit_size to control whether it's double or double _Complex 
 */
void rearrange_Vi(void *Vi_full, int ncol, int **DMVertices, int size_comm, 
                    int Nx, int Ny, int Nd, void *Vi_full_order, int unit_size);
                    
/**
 * @brief   Rearrange rhs after receiving from other comms
 *          
 *          Vi was ordered in block-separated way and now is ordered in continuous way
 *          Note: using unit_size to control whether it's double or double _Complex 
 */
void rearrange_rhs(void *rhs_full, int ncol, int **DMVertices, int *displs, int size_comm, 
                    int Nx, int Ny, int Nd, void *rhs_full_order, int unit_size);

/**
 * @brief   preprocessing RHS of Poisson's equation depends on the method for Exact Exchange
 */
void poisson_RHS_local(SPARC_OBJ *pSPARC, double *rhs, double *f, int Nd, int ncolp);

/**
 * @brief   Solve Poisson's equation using FFT in Fourier Space
 * 
 * @param rhs_loc_order     complete RHS of poisson's equations without parallelization. 
 * @param pois_FFT_const    constant for solving possion's equations
 * @param ncolp             Number of poisson's equations to be solved.
 * @param Vi_loc            complete solutions of poisson's equations without parallelization. 
 * Note:                    This function is complete localized. 
 */
void pois_fft(SPARC_OBJ *pSPARC, double *rhs_loc_order, double *pois_FFT_const, int ncolp, double *Vi_loc);


/**
 * @brief   Solve Poisson's equation using linear solver (e.g. CG) in Real Space
 */
void pois_linearsolver(SPARC_OBJ *pSPARC, double *rhs_loc_order, int ncolp, double *Vi_loc);

/**
 * @brief   preprocessing RHS of Poisson's equation by multipole expansion on single process
 */
void MultipoleExpansion_phi_local(SPARC_OBJ *pSPARC, double *rhs, double *d_cor, int Nd, int ncolp);



/**
 * @brief   Transfer vectors from dmcomm to kptcomm_topo for Lancozs algorithm
 *
 *          Used to transfer psi_outer in case of no-ACE method and transfer 
 *          Xi (of ACE operator) in case of ACE method from dmcomm to kptcomm_topo
 */
void Transfer_dmcomm_to_kptcomm_topo(SPARC_OBJ *pSPARC, int ncols, double *psi_outer, double *psi_outer_kptcomm);


/**
 * @brief   Gather psi_outers in other bandcomms
 *
 *          The default comm is blacscomm
 */
void gather_psi_occ_outer(SPARC_OBJ *pSPARC, double *psi_outer, double *occ_outer);


/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 *          Auxiliary function - Method by Gygi and Baldereschi
 *          DOI: 10.1103/PhysRevB.34.4405
 */
void compute_pois_fft_const(SPARC_OBJ *pSPARC);


/**
 * @brief   Compute constant coefficients for solving Poisson's equation using FFT
 * 
 *          Spherical Cutoff - Method by James Spencer and Ali Alavi 
 *          DOI: 10.1103/PhysRevB.77.193110
 * 
 *          Note: using Finite difference approximation of G2
 */
void compute_pois_fft_const_FD(SPARC_OBJ *pSPARC);

/**
 * @brief   Create communicators between k-point topology and dmcomm.
 * 
 * Notes:   Occupations are only correct within all dmcomm processes.
 *          When not using ACE method, this communicator might be required. 
 */
void create_kpttopo_dmcomm_inter(SPARC_OBJ *pSPARC);


/**
 * @brief   Compute the coefficient for G = 0 term in auxiliary function method
 * 
 *          Note: Using QE's method. Not the original one by Gygi 
 */
void auxiliary_constant(SPARC_OBJ *pSPARC);

/**
 * @brief   Estimation of Ecut by (pi/h)^2/2
 */
double ecut_estimate(double hx, double hy, double hz);


/**
 * @brief   Allocate memory space for ACE operator and check its size for each outer loop
 */
void allocate_ACE(SPARC_OBJ *pSPARC);

/**
 * @brief   Compute exact exchange energy density
 */
void computeExactExchangeEnergyDensity(SPARC_OBJ *pSPARC, double *Exxrho);
#endif // EXACTEXCHANGEPOTENTIAL_H 