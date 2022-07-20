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
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

#include "hamiltonianVecRoutines.h"
#include "lapVecRoutines.h"
#include "lapVecOrth.h"
#include "lapVecOrthKpt.h"
#include "lapVecNonOrth.h"
#include "lapVecNonOrthKpt.h"
#include "nlocVecRoutines.h"
#include "isddft.h"
#include "exactExchange.h"
#include "exactExchangeKpt.h"
#include "mgga.h"

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
    int ncol, double c, double *x, double *Hx, int spin, MPI_Comm comm
)
{
    unsigned i, j;
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
                x+i*(unsigned)DMnd, Hx+i*(unsigned)DMnd, comm, dims
            );
        }
        // Lap_plus_diag_vec_mult_orth(
        //         pSPARC, DMnd, DMVertices, ncol, -0.5, 1.0, c, Veff_loc,
        //         x, Hx, comm, dims
        // ); // slower than the for loop above
    } else {  // non-orthogonal cell
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph; // pSPARC->comm_dist_graph_phi
        else    
            comm2 = pSPARC->comm_dist_graph_psi;
  
        for (i = 0; i < ncol; i++) {
            Lap_plus_diag_vec_mult_nonorth(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*(unsigned)DMnd, Hx+i*(unsigned)DMnd, comm, comm2, dims
            );
        }
    }

    // adding Exact Exchange potential  
    if (pSPARC->usefock > 1){
        exact_exchange_potential((SPARC_OBJ *)pSPARC, x, ncol, DMnd, Hx, spin, comm);
    }

    // adding metaGGA term
    if(pSPARC->mGGAflag == 1 && pSPARC->countSCF > 1) {
        // ATTENTION: now SCAN does not have polarized spin!
        int Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
        double *vxcMGGA3_dm = (Lanczos_flag == 1) ? pSPARC->vxcMGGA3_loc_kptcomm : pSPARC->vxcMGGA3_loc_dmcomm;
        double *mGGAterm = (double *)malloc(DMnd*ncol * sizeof(double));
        
        compute_mGGA_term_hamil(pSPARC, x, ncol, DMnd, DMVertices, vxcMGGA3_dm, mGGAterm, spin, comm);

        for (i = 0; i < ncol; i++) {
            for (j = 0; j < DMnd; j++) {
                Hx[j+i*(unsigned)DMnd] -= 0.5*(mGGAterm[j+i*(unsigned)DMnd]);
            }
        }
        free(mGGAterm);
    }

    // apply nonlocal projectors
    #ifdef USE_EVA_MODULE
    t1 = MPI_Wtime();
    #endif
    Vnl_vec_mult(pSPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, x, Hx, comm);
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
    int ncol, double c, double complex *x, double complex *Hx, int spin, int kpt, MPI_Comm comm
)
{
    unsigned i, j;
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
            Lap_plus_diag_vec_mult_orth_kpt(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*(unsigned)DMnd, Hx+i*(unsigned)DMnd, comm, dims, kpt
            );
        }
        // Lap_plus_diag_vec_mult_orth(
        //         pSPARC, DMnd, DMVertices, ncol, -0.5, 1.0, c, Veff_loc,
        //         x, Hx, comm, dims
        // ); // slower than the for loop above
    } else {  // non-orthogonal cell
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph;
        else    
            comm2 = pSPARC->comm_dist_graph_psi;
  
        for (i = 0; i < ncol; i++) {
            Lap_plus_diag_vec_mult_nonorth_kpt(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*(unsigned)DMnd, Hx+i*(unsigned)DMnd, comm, comm2, dims, kpt
            );
        }
    }

    // adding Exact Exchange potential  
    if (pSPARC->usefock > 1){
        exact_exchange_potential_kpt((SPARC_OBJ *)pSPARC, x, ncol, DMnd, Hx, spin, kpt, comm);
    }

    // adding metaGGA term
    if(pSPARC->mGGAflag == 1 && pSPARC->countSCF > 1) {
        // ATTENTION: now SCAN does not have polarized spin!
        int Lanczos_flag = (comm == pSPARC->kptcomm_topo) ? 1 : 0;
        double *vxcMGGA3_dm = (Lanczos_flag == 1) ? pSPARC->vxcMGGA3_loc_kptcomm : pSPARC->vxcMGGA3_loc_dmcomm;
        double _Complex *mGGAterm = (double _Complex *)malloc(DMnd*ncol * sizeof(double _Complex));
        compute_mGGA_term_hamil_kpt(pSPARC, x, ncol, DMnd, DMVertices, vxcMGGA3_dm, mGGAterm, spin, kpt, comm);
        for (i = 0; i < ncol; i++) {
            for (j = 0; j < DMnd; j++) {
                // Hx[j+i*(unsigned)DMnd] -= 0.5*(Dvxc3Dx_x[j] + Dvxc3Dx_y[j] + Dvxc3Dx_z[j]);
                Hx[j+i*(unsigned)DMnd] -= 0.5*(mGGAterm[j+i*(unsigned)DMnd]);
            }
        }
        free(mGGAterm);
    }

    // apply nonlocal projectors
    #ifdef USE_EVA_MODULE
    t1 = MPI_Wtime();
    #endif
    
    Vnl_vec_mult_kpt(pSPARC, DMnd, Atom_Influence_nloc, nlocProj, ncol, x, Hx, kpt, comm);
    
    #ifdef USE_EVA_MODULE
    t2 = MPI_Wtime();
    EVA_buff_timer_add(0.0, 0.0, 0.0, 0.0, 0.0, t2 - t1);
    EVA_buff_rhs_add(0, ncol);
    #endif
}



