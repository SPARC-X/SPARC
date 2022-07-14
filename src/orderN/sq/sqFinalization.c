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

/**
 * @brief   Free all allocated memory spaces and communicators 
 */
void Free_SQ(SPARC_OBJ *pSPARC) {
    
    int i, j, k, DMnx, DMny, DMnz;
    SQ_OBJ  *pSQ = pSPARC->pSQ;
    SQIND *SqInd = pSPARC->pSQ->SqInd;
    
    // Free variables in all processors
    free(pSPARC->forces);
    free(pSPARC->FDweights_D1);
    free(pSPARC->FDweights_D2);
    free(pSPARC->localPsd);
    free(pSPARC->Mass);
    free(pSPARC->atomType);
    free(pSPARC->Znucl);
    free(pSPARC->nAtomv);
    free(pSPARC->psdName);
    free(pSPARC->atom_pos);
    free(pSPARC->IsFrac);
    free(pSPARC->IsSpin);
    free(pSPARC->mvAtmConstraint);
    free(pSPARC->atom_spin);
    free(pSPARC->D1_stencil_coeffs_x);
    free(pSPARC->D1_stencil_coeffs_y);
    free(pSPARC->D1_stencil_coeffs_z);
    free(pSPARC->D2_stencil_coeffs_x);
    free(pSPARC->D2_stencil_coeffs_y);
    free(pSPARC->D2_stencil_coeffs_z);
    free(pSPARC->CUTOFF_x);
    free(pSPARC->CUTOFF_y);
    free(pSPARC->CUTOFF_z); 
    
    free(pSPARC->kptWts); 
    free(pSPARC->k1); 
    free(pSPARC->k2); 
    free(pSPARC->k3);

    // free preconditioner coeff arrays
    if (pSPARC->MixingPrecond == 2 || pSPARC->MixingPrecond == 3) {
        free(pSPARC->precondcoeff_a);
        free(pSPARC->precondcoeff_lambda_sqr);
    }
    // free psd struct components
    for (i = 0; i < pSPARC->Ntypes; i++) {
        free(pSPARC->psd[i].rVloc);
        free(pSPARC->psd[i].UdV);
        free(pSPARC->psd[i].rhoIsoAtom);
        free(pSPARC->psd[i].RadialGrid);
        free(pSPARC->psd[i].SplinerVlocD);
        free(pSPARC->psd[i].SplineFitUdV);
        free(pSPARC->psd[i].SplineFitIsoAtomDen);
        free(pSPARC->psd[i].SplineRhocD);
        free(pSPARC->psd[i].rc);
        free(pSPARC->psd[i].Gamma);
        free(pSPARC->psd[i].rho_c_table);
        free(pSPARC->psd[i].ppl);
    }
    free(pSPARC->psd);
    
    if(pSPARC->MDFlag == 1) {
      free(pSPARC->ion_vel);
      free(pSPARC->ion_accel);
    }

    Free_D2D_Target(&pSPARC->d2d_s2p_sq, &pSPARC->d2d_s2p_phi, pSQ->dmcomm_SQ, pSPARC->dmcomm_phi);
    
    // Free all variables in dmcomm_phi
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        free(pSPARC->electronDens_at);
        free(pSPARC->electronDens_core);
        free(pSPARC->electronDens);
        free(pSPARC->psdChrgDens);
        free(pSPARC->psdChrgDens_ref);
        free(pSPARC->Vc);
        free(pSPARC->XCPotential);
        free(pSPARC->e_xc);
        if(strcmpi(pSPARC->XC,"GGA_PBE") == 0 || strcmpi(pSPARC->XC,"GGA_RPBE") == 0 || strcmpi(pSPARC->XC,"GGA_PBEsol") == 0
            || strcmpi(pSPARC->XC,"PBE0") == 0 || strcmpi(pSPARC->XC,"HF") == 0){
            free(pSPARC->Dxcdgrho);
        }    
        free(pSPARC->elecstPotential);
        free(pSPARC->Veff_loc_dmcomm_phi);
        free(pSPARC->mixing_hist_xk);
        free(pSPARC->mixing_hist_fk);
        free(pSPARC->mixing_hist_fkm1);
        free(pSPARC->mixing_hist_xkm1);
        free(pSPARC->mixing_hist_Xk);
        free(pSPARC->mixing_hist_Fk);

        // for using QE scf error definition
        if (pSPARC->scf_err_type == 1) {
            free(pSPARC->rho_dmcomm_phi_in);
            free(pSPARC->phi_dmcomm_phi_in);
        }

        // for denstiy mixing, extra memory is created to store potential history
        if (pSPARC->MixingVariable == 0) { 
            free(pSPARC->Veff_loc_dmcomm_phi_in);
        }
        if (pSPARC->MixingPrecond != 0) {
           free(pSPARC->mixing_hist_Pfk);
        }

        // free MD and relax stuff
        if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
            free(pSPARC->delectronDens);
            free(pSPARC->delectronDens_0dt);
            free(pSPARC->delectronDens_1dt);
            free(pSPARC->delectronDens_2dt);
            free(pSPARC->atom_pos_nm);
            free(pSPARC->atom_pos_0dt);
            free(pSPARC->atom_pos_1dt);
            free(pSPARC->atom_pos_2dt);
        }
        MPI_Comm_free(&pSPARC->dmcomm_phi);
    }
    
    if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) {
        free(pSPARC->pSQ);
        return;
    }
    // Following variables are all only in dmcomm_SQ
    DMnx = pSQ->Nx_d_SQ;
    DMny = pSQ->Ny_d_SQ;
    DMnz = pSQ->Nz_d_SQ;

    free(pSQ->maxeig);
    free(pSQ->mineig);
    free(pSQ->lambda_max);
    free(pSQ->chi);
    free(pSQ->zee);

    for (i = 0; i < pSQ->Nd_d_SQ; i++) {
        free(pSQ->gnd[i]);
        free(pSQ->gwt[i]);
        free(pSQ->Ci[i]);
    }
    free(pSQ->gnd);
    free(pSQ->gwt);
    free(pSQ->Ci);

    for (k = 0; k < DMnz+2*pSQ->nloc[2]; k++) {
        for (j = 0; j < DMny+2*pSQ->nloc[1]; j++) {
            free(pSQ->Veff_PR[k][j]);
        }
        free(pSQ->Veff_PR[k]);
    }
    free(pSQ->Veff_PR);
    free(pSQ->Veff_loc_SQ);

    if (pSPARC->SQ_typ_dm == 2) { 
        if (pSPARC->SQ_gauss_mem == 1) {
            for (i = 0; i < pSQ->Nd_d_SQ; i++) {
                free(pSQ->lanczos_vec_all[i]);
                free(pSQ->w_all[i]);
            }
            free(pSQ->lanczos_vec_all);
            free(pSQ->w_all);
        } else {
            free(pSQ->lanczos_vec);
            free(pSQ->w);
        }
    }

    if (pSPARC->SQ_typ_dm == 1) {
        for (i = 0; i < pSQ->Nd_d_SQ; i++) {
            for (k = 0; k < 2*pSQ->nloc[2]+1; k++) {
                for (j = 0; j < 2*pSQ->nloc[1]+1; j++) {
                    free(pSQ->vec[i][k][j]);
                }
                free(pSQ->vec[i][k]);
            }
            free(pSQ->vec[i]);
        }
        free(pSQ->vec);
    }

    MPI_Comm_free(&pSQ->SQ_dist_graph_comm);
    
    free(pSQ->SqInd->scounts);
	free(pSQ->SqInd->sdispls);
	free(pSQ->SqInd->rcounts);
	free(pSQ->SqInd->rdispls);

	free(pSQ->SqInd->x_scounts);
	free(pSQ->SqInd->y_scounts);
	free(pSQ->SqInd->z_scounts);
	free(pSQ->SqInd->x_rcounts);
	free(pSQ->SqInd->y_rcounts);
	free(pSQ->SqInd->z_rcounts);

    free(pSQ->SqInd);
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

    for (nd = 0; nd < pSQ->Nd_d_SQ; nd++) {
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


