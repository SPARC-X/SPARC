/**
 * @file    hamiltonianVecRoutines.c
 * @brief   This file contains functions for performing Hamiltonian-vector 
 *          multiply routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *          Xin Jing <xjing30@gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

#include "hamiltonianVecRoutines.h"
#include "lapVecRoutines.h"
#include "lapVecRoutinesKpt.h"
#include "nlocVecRoutines.h"
#include "isddft.h"
#include "exactExchange.h"
#include "exactExchangeKpt.h"
#include "mGGAhamiltonianTerm.h"
#include "spinOrbitCoupling.h"

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
#endif



/**
 * @brief   Calculate (Hamiltonian + c * I) times a bunch of vectors in a matrix-free way.
 *          
 *          This function simply calls the Hamiltonian_vec_mult (or the sequential version) 
 *          multiple times. For some reason it is more efficient than calling it ones and do  
 *          the multiplication together. TODO: think of a more efficient way!
 */
void Hamiltonian_vectors_mult(
    const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
    int ncol, double c, double *x, const int ldi, double *Hx, const int ldo, int spin, MPI_Comm comm
)
{
    unsigned i;
    int nproc;
    MPI_Comm_size(comm, &nproc);
    
    // TODO: make dims an input parameter
    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else 
        dims[0] = dims[1] = dims[2] = 1;
        
    // first find (-0.5 * Lap + Veff + c) * x
    if (pSPARC->cell_typ == 0) { // orthogonal cell
        for (i = 0; i < ncol; i++) {
            Lap_plus_diag_vec_mult_orth(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*(unsigned)ldi, ldi, Hx+i*(unsigned)ldo, ldo, comm, dims
            );
        }
        // Lap_plus_diag_vec_mult_orth(.., ncol,..) slower than the for loop above
    } else {  // non-orthogonal cell
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph; // pSPARC->comm_dist_graph_phi
        else    
            comm2 = pSPARC->comm_dist_graph_psi;
  
        for (i = 0; i < ncol; i++) {
            Lap_plus_diag_vec_mult_nonorth(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*(unsigned)ldi, ldi, Hx+i*(unsigned)ldo, ldo, comm, comm2, dims
            );
        }
    }

    // adding Exact Exchange potential  
    if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0)){
        exact_exchange_potential((SPARC_OBJ *)pSPARC, x, ldi, ncol, DMnd, Hx, ldo, spin, comm);
    }

    // adding metaGGA term
    if(pSPARC->ixc[2] && pSPARC->countPotentialCalculate > 1) {
        mGGA_potential(pSPARC, x, ldi, ncol, DMnd, DMVertices, Hx, ldo, spin, comm);
    }

    // apply nonlocal projectors
    #ifdef USE_EVA_MODULE
    t1 = MPI_Wtime();
    #endif
    Vnl_vec_mult(pSPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, x, ldi, Hx, ldo, comm);
    #ifdef USE_EVA_MODULE
    t2 = MPI_Wtime();
    EVA_buff_timer_add(0.0, 0.0, 0.0, 0.0, 0.0, t2 - t1);
    EVA_buff_rhs_add(0, ncol);
    #endif
}



/**
 * @brief   Calculate (Hamiltonian + c * I) times a bunch of vectors in a matrix-free way with a Bloch wavevector.
 *          
 *          This function simply calls the Hamiltonian_vec_mult (or the sequential version) 
 *          multiple times. For some reason it is more efficient than calling it ones and do  
 *          the multiplication together. TODO: think of a more efficient way!
 */
void Hamiltonian_vectors_mult_kpt(
    const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
    ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, NLOC_PROJ_OBJ *nlocProj, 
    int ncol, double c, double _Complex *x, const int ldi, double _Complex *Hx, const int ldo, int spin, int kpt, MPI_Comm comm
)
{
    unsigned i;
    int nproc;
    MPI_Comm_size(comm, &nproc);
    
    // TODO: make dims an input parameter
    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else 
        dims[0] = dims[1] = dims[2] = 1;
    int spinor;

    // first find (-0.5 * Lap + Veff + c) * x
    if (pSPARC->cell_typ == 0) { // orthogonal cell
        for (i = 0; i < ncol; i++) {
            for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
                int shift = (pSPARC->spin_typ == 2) * spinor * DMnd;
                Lap_plus_diag_vec_mult_orth_kpt(
                    pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc+shift,
                    x+i*(unsigned)ldi+spinor*DMnd, ldi,
                    Hx+i*(unsigned)ldo+spinor*DMnd, ldo, comm, dims, kpt
                );
            }
        }
    } else {  // non-orthogonal cell
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph;
        else    
            comm2 = pSPARC->comm_dist_graph_psi;

        for (i = 0; i < ncol; i++) {
            for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
                int shift = (pSPARC->spin_typ == 2) * spinor * DMnd;
                Lap_plus_diag_vec_mult_nonorth_kpt(
                    pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc+shift,
                    x+i*(unsigned)ldi+spinor*DMnd, ldi,
                    Hx+i*(unsigned)ldo+spinor*DMnd, ldo, comm, comm2, dims, kpt
                );
            }
        }
    }

    if (pSPARC->spin_typ == 2) {
        // apply off-diagonal effective potenital in case of non-collinear
        off_diagonal_effective_potential(pSPARC, DMnd, ncol, Veff_loc+2*DMnd, x, ldi, Hx, ldo);
    }

    #ifdef USE_EVA_MODULE
    t1 = MPI_Wtime();
    #endif

    // apply nonlocal projectors
    for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
        // Apply scalar-relativistic part
        Vnl_vec_mult_kpt(pSPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, 
                            x+spinor*DMnd, ldi, Hx+spinor*DMnd, ldo, kpt, comm);
        
        if (pSPARC->SOC_Flag == 0) continue;
        // Apply spin-orbit onto the same spinor
        Vnl_vec_mult_SOC1(pSPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, 
                            x+spinor*DMnd, ldi, Hx+spinor*DMnd, ldo, spinor, kpt, comm);

        // Apply spin-orbit onto the opposite spinor
        Vnl_vec_mult_SOC2(pSPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, 
                            x+(1-spinor)*DMnd, ldi, Hx+spinor*DMnd, ldo, spinor, kpt, comm);
    }

    #ifdef USE_EVA_MODULE
    t2 = MPI_Wtime();
    EVA_buff_timer_add(0.0, 0.0, 0.0, 0.0, 0.0, t2 - t1);
    EVA_buff_rhs_add(0, ncol);
    #endif

    // apply Exact Exchange potential  
    if ((pSPARC->usefock > 0) && (pSPARC->usefock % 2 == 0)) {
        for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
            exact_exchange_potential_kpt((SPARC_OBJ *)pSPARC, x+spinor*DMnd, ldi, ncol, DMnd, Hx+spinor*DMnd, ldo, spinor+spin, kpt, comm);
        }   
    }

    // apply metaGGA term 
    if(pSPARC->ixc[2] && pSPARC->countPotentialCalculate > 1) {
        for (spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
            mGGA_potential_kpt(pSPARC, x+spinor*DMnd, ldi, ncol, DMnd, DMVertices, Hx+spinor*DMnd, ldo, spinor+spin, kpt, comm);
        }   
    }
}


/**
 * @brief   Apply off-diagonal effective potential
 */
void off_diagonal_effective_potential(const SPARC_OBJ *pSPARC, 
    int DMnd, int ncol, double *Veff_loc, double _Complex *x, const int ldi, double _Complex *Hx, const int ldo)
{
    for (int n = 0; n < ncol; n++) {
        for (int spinor = 0; spinor < pSPARC->Nspinor_eig; spinor++) {
            double spinorfac = (spinor == 0) ? 1.0 : -1.0; 
            for (int i = 0; i < DMnd; i++) {
                Hx[i+n*ldo+spinor*DMnd] += (Veff_loc[i] + I*spinorfac*Veff_loc[i+DMnd]) * x[i+n*ldi+(1-spinor)*DMnd];      
            }
        }
    }
}

