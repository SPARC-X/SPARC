/***
 * @file    sqFinalization.c
 * @brief   This file contains the functions for SQ method.
 *
 * @authors Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "sqFinalization.h"
#include "parallelization.h"
#include "tools.h"
#include "electrostatics.h"
#include "kroneckerLaplacian.h"

/**
 * @brief   Free all allocated memory spaces and communicators 
 */
void Free_SQ(SPARC_OBJ *pSPARC) {
    
    int i;
    SQ_OBJ  *pSQ = pSPARC->pSQ;
    
    Free_D2D_Target(&pSPARC->d2d_s2p_sq, &pSPARC->d2d_s2p_phi, pSQ->dmcomm_SQ, pSPARC->dmcomm_phi);
    // free communicators 
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        MPI_Comm_free(&pSPARC->dmcomm_phi);
    }

    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) {
        free(pSPARC->pSQ);
        return;
    }

    // Following variables are all only in dmcomm_SQ    
    free(pSQ->maxeig);
    free(pSQ->mineig);
    free(pSQ->lambda_max);

    for (i = 0; i < pSQ->DMnd_SQ; i++) {
        free(pSQ->gnd[i]);
        free(pSQ->gwt[i]);
    }
    free(pSQ->gnd);
    free(pSQ->gwt);

    free(pSQ->Veff_PR);
    free(pSQ->Veff_loc_SQ);
    free(pSQ->x_ex);

    if (pSPARC->sqHighTFlag == 1) {
        for (i = 0; i < pSQ->DMnd_SQ; i++) {
            free(pSQ->lanczos_vec_all[i]);
            free(pSQ->w_all[i]);
        }
        free(pSQ->lanczos_vec_all);
        free(pSQ->w_all);
    } else {
        free(pSQ->lanczos_vec);
        free(pSQ->w);
    }

    if (pSPARC->usefock) {
        for (int i = 0; i < pSQ->DMnd_SQ; i++)
            free(pSQ->Dn[i]);
        free(pSQ->Dn);
        for (int i = 0; i < pSQ->DMnd_PR; i++)
            free(pSQ->Dn_exx[i]);
        free(pSQ->Dn_exx);
        free_exx_SQ(pSPARC);
    }

    free_D2Dext_Target(pSQ->d2dext_dmcomm_sq, pSQ->dmcomm_d2dext_sq);
    free_D2Dext_Target(pSQ->d2dext_dmcomm_sq_ext, pSQ->dmcomm_d2dext_sq);
    free(pSQ->d2dext_dmcomm_sq);
    free(pSQ->d2dext_dmcomm_sq_ext);
    if (pSQ->dmcomm_d2dext_sq != MPI_COMM_NULL)
        MPI_Comm_free(&pSQ->dmcomm_d2dext_sq);
    MPI_Comm_free(&pSPARC->spincomm);
    MPI_Comm_free(&pSPARC->spin_bridge_comm);
    MPI_Comm_free(&pSQ->dmcomm_SQ);
    free(pSPARC->pSQ);
}

/**
 * @brief   Free SCF variables for SQ methods.
 */
void Free_scfvar_SQ(SPARC_OBJ *pSPARC) {
    SQ_OBJ *pSQ = pSPARC->pSQ;
    int i, j, nd, ityp, n_atom, iat;
    
    // free atom influence struct components
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        for (i = 0; i < pSPARC->Ntypes; i++) {
            free(pSPARC->Atom_Influence_local[i].coords);
            free(pSPARC->Atom_Influence_local[i].atom_spin);
            free(pSPARC->Atom_Influence_local[i].atom_index);
            free(pSPARC->Atom_Influence_local[i].xs);
            free(pSPARC->Atom_Influence_local[i].xe);
            free(pSPARC->Atom_Influence_local[i].ys);
            free(pSPARC->Atom_Influence_local[i].ye);
            free(pSPARC->Atom_Influence_local[i].zs);
            free(pSPARC->Atom_Influence_local[i].ze);
        }
        free(pSPARC->Atom_Influence_local);
    }

    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    // deallocate nonlocal projectors in kptcomm_topo
    for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
        // if (! pSPARC->nlocProj_kptcomm[ityp].nproj) continue;
        if (! pSPARC->nlocProj_kptcomm[ityp].nproj) {
            free( pSPARC->nlocProj_kptcomm[ityp].Chi );                    
            continue;
        }
        for (iat = 0; iat < pSPARC->Atom_Influence_nloc_kptcomm[ityp].n_atom; iat++) {
            free( pSPARC->nlocProj_kptcomm[ityp].Chi[iat] );
        }
        free( pSPARC->nlocProj_kptcomm[ityp].Chi );
    }
    free(pSPARC->nlocProj_kptcomm);
    


    for (i = 0; i < pSPARC->Ntypes; i++) {
        if (pSPARC->Atom_Influence_nloc_kptcomm[i].n_atom > 0) {
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].coords);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].atom_index);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].xs);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].xe);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].ys);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].ye);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].zs);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].ze);
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].ndc);
            for (j = 0; j < pSPARC->Atom_Influence_nloc_kptcomm[i].n_atom; j++) {
                free(pSPARC->Atom_Influence_nloc_kptcomm[i].grid_pos[j]);
            }
            free(pSPARC->Atom_Influence_nloc_kptcomm[i].grid_pos);
        }
    }
    free(pSPARC->Atom_Influence_nloc_kptcomm);

    for (nd = 0; nd < pSQ->DMnd_SQ; nd++) {
        for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
            n_atom = pSPARC->Atom_Influence_nloc_SQ[nd][ityp].n_atom;
            if (n_atom > 0){
                free(pSPARC->Atom_Influence_nloc_SQ[nd][ityp].coords);
                free(pSPARC->Atom_Influence_nloc_SQ[nd][ityp].atom_index);
                free(pSPARC->Atom_Influence_nloc_SQ[nd][ityp].ndc);
                for (iat = 0; iat < n_atom; iat++) {
                    free(pSPARC->Atom_Influence_nloc_SQ[nd][ityp].grid_pos[iat]);
                    free(pSPARC->nlocProj_SQ[nd][ityp].Chi[iat]);
                }
                free(pSPARC->Atom_Influence_nloc_SQ[nd][ityp].grid_pos);
                free(pSPARC->nlocProj_SQ[nd][ityp].Chi);
            }
        }
        free(pSPARC->nlocProj_SQ[nd]);
        free(pSPARC->Atom_Influence_nloc_SQ[nd]);
    }
    free(pSPARC->nlocProj_SQ);
    free(pSPARC->Atom_Influence_nloc_SQ);
    
    return;
}


/**
 * @brief   Free SCF variables for SQ-hybrid
 */
void free_exx_SQ(SPARC_OBJ *pSPARC)
{
    SQ_OBJ *pSQ = pSPARC->pSQ;    
    if (pSQ->dmcomm_SQ == MPI_COMM_NULL) return;
    
    free_kron_Lap(pSQ->kron_lap);
    free(pSQ->kron_lap);
    free_multipole_expansion(pSQ->MpExp_exx, MPI_COMM_SELF);
    free(pSQ->MpExp_exx);
    if (pSPARC->ExxAcc == 1) {
        free(pSQ->basis);
        int nExxPot = (pSPARC->SQ_highT_hybrid_gauss_mem == 1) ? pSQ->DMnd_SQ : 1;        
        for (int i = 0; i < nExxPot; i++) {
            free(pSQ->exxPot[i]);
        }
        free(pSQ->exxPot);
    }
    if (strcmpi(pSPARC->XC,"HSE") == 0) {
        free(pSQ->erfcR);
    }
}
