/**
 * @file    finalization.c
 * @brief   This file contains the functions for finalization.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "finalization.h"
#include "parallelization.h"
#include "isddft.h"
#include "tools.h"
#include "eigenSolver.h"     // free_GTM_CheFSI()
#include "eigenSolverKpt.h"  // free_GTM_CheFSI_kpt()
#include "exactExchangeFinalization.h"
#include "d3finalization.h"
#include "vdWDFfinalization.h"
#include "MGGAfinalization.h"
#include "sqFinalization.h"
/* ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
    #include <mkl_service.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // include ScaLAPACK function declarations
#endif

/**
 * @brief   Finalize parameters and objects.
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 */
void Finalize(SPARC_OBJ *pSPARC)
{
    if (pSPARC->SQFlag == 1) {
        Free_SQ(pSPARC);
    } else {
        Free_SPARC(pSPARC);
    }
    
    FILE *output_fp;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) {
        output_fp = fopen(pSPARC->OutFilename,"a");
        if (output_fp == NULL) {
            printf("\nCannot open file \"%s\"\n",pSPARC->OutFilename);
            exit(EXIT_FAILURE);
        }
        
        double t_end, t_wall;
        t_end = MPI_Wtime();
        t_wall = t_end - pSPARC->time_start;
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"                               Timing info                                 \n");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"Total walltime                     :  %.3f sec\n", t_wall);
        fprintf(output_fp,"___________________________________________________________________________\n");
        fprintf(output_fp,"\n");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"*             Material Physics & Mechanics Group, Georgia Tech            *\n");
        fprintf(output_fp,"*                       PI: Phanish Suryanarayana                         *\n");
        fprintf(output_fp,"*               List of contributors: See the documentation               *\n");
        fprintf(output_fp,"*         Citation: See README.md or the documentation for details        *\n");
        fprintf(output_fp,"*  Acknowledgements: U.S. DOE (DE-SC0019410)                              *\n");
        fprintf(output_fp,"*      {Preliminary developments: U.S. NSF (1333500,1663244,1553212)}     *\n");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"                                                                           \n");
        fclose(output_fp);
    }
}



/**
 * @brief   Deallocates dynamic arrays in SPARC.
 */
void Free_SPARC(SPARC_OBJ *pSPARC) {
    int i;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // free initial guess vector for Lanczos
    if (pSPARC->isGammaPoint && pSPARC->kptcomm_topo != MPI_COMM_NULL) 
        free(pSPARC->Lanczos_x0);
        
    if (pSPARC->isGammaPoint != 1 && pSPARC->kptcomm_topo != MPI_COMM_NULL) 
        free(pSPARC->Lanczos_x0_complex);    

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
            || strcmp(pSPARC->XC,"PBE0") == 0 || strcmp(pSPARC->XC,"HF") == 0 || strcmp(pSPARC->XC,"HSE") == 0 || strcmp(pSPARC->XC,"SCAN") == 0
            || strcmpi(pSPARC->XC,"vdWDF1") == 0 || strcmpi(pSPARC->XC,"vdWDF2") == 0){
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
        free(pSPARC->mixing_hist_Pfk);
        //free(pSPARC->forces);
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
    }
    
    if (pSPARC->isGammaPoint){
        if (pSPARC->dmcomm != MPI_COMM_NULL) {
            free(pSPARC->Xorb);
            free(pSPARC->Yorb);
        }
    } else {
        if (pSPARC->dmcomm != MPI_COMM_NULL) {
            free(pSPARC->Xorb_kpt);
            free(pSPARC->Yorb_kpt);
        }
    }

    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->isGammaPoint) {
        free(pSPARC->Hp);
        free(pSPARC->Mp);
        free(pSPARC->Q);
    } else {
        free(pSPARC->Hp_kpt);
        free(pSPARC->Mp_kpt);
        free(pSPARC->Q_kpt);
    }
    #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    free(pSPARC->forces);
    free(pSPARC->lambda);
    free(pSPARC->occ);
    free(pSPARC->eigmin);
    free(pSPARC->eigmax);
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
    if(pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20){
        free(pSPARC->D2_stencil_coeffs_xy);
        free(pSPARC->D2_stencil_coeffs_yz);
        free(pSPARC->D2_stencil_coeffs_xz);
        free(pSPARC->D1_stencil_coeffs_xy);
        free(pSPARC->D1_stencil_coeffs_yx);
        free(pSPARC->D1_stencil_coeffs_xz);
        free(pSPARC->D1_stencil_coeffs_zx);
        free(pSPARC->D1_stencil_coeffs_yz);
        free(pSPARC->D1_stencil_coeffs_zy);
    }
    free(pSPARC->CUTOFF_x);
    free(pSPARC->CUTOFF_y);
    free(pSPARC->CUTOFF_z); 
    free(pSPARC->IP_displ);
    if (pSPARC->SOC_Flag) 
        free(pSPARC->IP_displ_SOC); 
    
    // free preconditioner coeff arrays
    if (pSPARC->MixingPrecond == 2 || pSPARC->MixingPrecond == 3) {
        free(pSPARC->precondcoeff_a);
        free(pSPARC->precondcoeff_lambda_sqr);
    }

    // free k-points arrays    
    //if (pSPARC->Nkpts >= 1) {
        // if (pSPARC->BC != 1) {
            free(pSPARC->kptWts); 
            free(pSPARC->k1); 
            free(pSPARC->k2); 
            free(pSPARC->k3);
        // }
    // }

    if (pSPARC->usefock > 0) {
        free(pSPARC->k1_hf);
        free(pSPARC->k2_hf);
        free(pSPARC->k3_hf);
        free(pSPARC->kpthf_ind);
        free(pSPARC->kpthf_ind_red);
        free(pSPARC->kpthfred2kpthf);
        free(pSPARC->kpthf_pn);
        free(pSPARC->kpts_hf_red_list);
        free(pSPARC->k1_shift);
        free(pSPARC->k2_shift);
        free(pSPARC->k3_shift);
        free(pSPARC->Kptshift_map);
        free_exx(pSPARC);
    }
    
    if (pSPARC->Nkpts >= 1 && pSPARC->kptcomm_index != -1) {
        //if (pSPARC->BC != 1) {
            free(pSPARC->kptWts_loc); 
            free(pSPARC->k1_loc); 
            free(pSPARC->k2_loc); 
            free(pSPARC->k3_loc);
            //free(pSPARC->lambdakpt);
        //}
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
        if (pSPARC->psd[i].pspsoc == 1) {
            free(pSPARC->psd[i].ppl_soc);
            free(pSPARC->psd[i].Gamma_soc);
            free(pSPARC->psd[i].UdV_soc);
            free(pSPARC->psd[i].SplineFitUdV_soc);
        }
    }
    // then free the psd struct itself
    free(pSPARC->psd);
	if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
		free(pSPARC->Veff_loc_dmcomm);
	}
    
    //if (pSPARC->npkpt > 1 && pSPARC->kptcomm_topo != MPI_COMM_NULL) {
    free(pSPARC->Veff_loc_kptcomm_topo);
    //}
      
    // free MD stuff
    if(pSPARC->MDFlag == 1) {
      free(pSPARC->ion_vel);
      free(pSPARC->ion_accel);
    }
    
    // free D2D targets between phi comm and psi comm
    Free_D2D_Target(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, pSPARC->dmcomm_phi, 
                   (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? 
                   pSPARC->dmcomm : MPI_COMM_NULL);
                   
    // free D2D targets between psi comm and kptcomm_topo comm
    if (((pSPARC->chefsibound_flag == 0 || pSPARC->chefsibound_flag == 1) &&
            pSPARC->spincomm_index >=0 && pSPARC->kptcomm_index >= 0
            && (pSPARC->spin_typ != 0 || !pSPARC->is_phi_eq_kpt_topo || !pSPARC->isGammaPoint))
            || (pSPARC->usefock != 0) )
    {
        Free_D2D_Target(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo,
                       pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL, pSPARC->kptcomm_topo);
    }
                     
    // free communicators
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        if (pSPARC->cell_typ != 0) {
            MPI_Comm_free(&pSPARC->comm_dist_graph_phi);
        }
        MPI_Comm_free(&pSPARC->dmcomm_phi);
    }
    if (pSPARC->blacscomm != MPI_COMM_NULL) {
        MPI_Comm_free(&pSPARC->blacscomm);
    }
    
    if (pSPARC->dmcomm != MPI_COMM_NULL) {
        if (pSPARC->cell_typ != 0) {
            MPI_Comm_free(&pSPARC->comm_dist_graph_psi);
        }
        MPI_Comm_free(&pSPARC->dmcomm);
    }
    if (pSPARC->bandcomm != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->bandcomm);
    if (pSPARC->kptcomm_inter != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->kptcomm_inter);
    if (pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        if (pSPARC->cell_typ != 0) {
            MPI_Comm_free(&pSPARC->kptcomm_topo_dist_graph);
        }
        MPI_Comm_free(&pSPARC->kptcomm_topo);
    }
    if (pSPARC->kptcomm_topo_excl != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->kptcomm_topo_excl);
    if (pSPARC->kptcomm != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->kptcomm);
    if (pSPARC->kpt_bridge_comm != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->kpt_bridge_comm);
    if (pSPARC->spincomm != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->spincomm);
    if (pSPARC->spin_bridge_comm != MPI_COMM_NULL)
        MPI_Comm_free(&pSPARC->spin_bridge_comm);     
    if (pSPARC->d3Flag == 1) {
        free_D3_coefficients(pSPARC); // this function is moved from electronicGroundState.c
    }
    if (pSPARC->vdWDFFlag != 0){
        vdWDF_free(pSPARC);
    }
    if(pSPARC->mGGAflag == 1) {
        free_MGGA(pSPARC);
    }

    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    Cblacs_gridexit(pSPARC->ictxt_blacs);
    Cblacs_gridexit(pSPARC->ictxt_blacs_topo);
    Cblacs_exit(1); // is this necessary
    #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    #ifdef USE_DP_SUBEIG
    if (pSPARC->isGammaPoint) 
    {
        free_DP_CheFSI(pSPARC);
    } else {
        free_DP_CheFSI_kpt(pSPARC);
    }
    #endif

    // free the memory allocated by the Intel MKL memory management software
    #ifdef USE_MKL
    mkl_thread_free_buffers();
    mkl_free_buffers();
    #endif
}

/*
@ brief: function clears the scf variables after every relax/MD step
*/
void Free_scfvar(SPARC_OBJ *pSPARC) {	
	int i, j;
	
	int iat, ityp;
	if (pSPARC->isGammaPoint){
        // deallocate nonlocal projectors in psi-domain
        if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                // if (! pSPARC->nlocProj[ityp].nproj) continue;
                if (! pSPARC->nlocProj[ityp].nproj) {
                    free( pSPARC->nlocProj[ityp].Chi );                    
                    continue;
                }
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    free( pSPARC->nlocProj[ityp].Chi[iat] );
                }
                free( pSPARC->nlocProj[ityp].Chi );
            }
            free(pSPARC->nlocProj);
        }
        
        // deallocate nonlocal projectors in kptcomm_topo
        if (pSPARC->kptcomm_topo != MPI_COMM_NULL && pSPARC->kptcomm_index >= 0) {
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
        }
    } else{
        // deallocate nonlocal projectors in psi-domain
        if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                // if (! pSPARC->nlocProj[ityp].nproj) continue;
                if (! pSPARC->nlocProj[ityp].nproj) {
                    free(pSPARC->nlocProj[ityp].Chi_c);
                    continue;
                }
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                    free( pSPARC->nlocProj[ityp].Chi_c[iat] );
                }
                free(pSPARC->nlocProj[ityp].Chi_c);
            }
            if (pSPARC->SOC_Flag == 1) {
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                    // if (! pSPARC->nlocProj[ityp].nproj) continue;
                    if (! pSPARC->nlocProj[ityp].nprojso) continue;
                    for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                        free( pSPARC->nlocProj[ityp].Chiso[iat] );
                    }
                    free( pSPARC->nlocProj[ityp].Chiso );
                }
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                    // if (! pSPARC->nlocProj[ityp].nproj) continue;
                    if (! pSPARC->nlocProj[ityp].nprojso_ext) continue;
                    for (iat = 0; iat < pSPARC->Atom_Influence_nloc[ityp].n_atom; iat++) {
                        free( pSPARC->nlocProj[ityp].Chisowt0[iat] );
                        free( pSPARC->nlocProj[ityp].Chisowtl[iat] );
                        free( pSPARC->nlocProj[ityp].Chisowtnl[iat] );
                    }
                    free( pSPARC->nlocProj[ityp].Chisowt0 );
                    free( pSPARC->nlocProj[ityp].Chisowtl );
                    free( pSPARC->nlocProj[ityp].Chisowtnl );
                }
            }
            free(pSPARC->nlocProj);
        }
        
        // deallocate nonlocal projectors in kptcomm_topo
        if (pSPARC->kptcomm_topo != MPI_COMM_NULL && pSPARC->kptcomm_index >= 0) {
            for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                // if (! pSPARC->nlocProj_kptcomm[ityp].nproj) continue;
                if (! pSPARC->nlocProj_kptcomm[ityp].nproj) {
                    free(pSPARC->nlocProj_kptcomm[ityp].Chi_c);
                    continue;
                }
                for (iat = 0; iat < pSPARC->Atom_Influence_nloc_kptcomm[ityp].n_atom; iat++) {
                    free( pSPARC->nlocProj_kptcomm[ityp].Chi_c[iat] );
                }
                free(pSPARC->nlocProj_kptcomm[ityp].Chi_c);
            }
            if (pSPARC->SOC_Flag == 1) {
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                    // if (! pSPARC->nlocProj[ityp].nproj) continue;
                    if (! pSPARC->nlocProj_kptcomm[ityp].nprojso) continue;
                    for (iat = 0; iat < pSPARC->Atom_Influence_nloc_kptcomm[ityp].n_atom; iat++) {
                        free( pSPARC->nlocProj_kptcomm[ityp].Chiso[iat] );
                    }
                    free( pSPARC->nlocProj_kptcomm[ityp].Chiso );
                }
                for (ityp = 0; ityp < pSPARC->Ntypes; ityp++) { 
                    // if (! pSPARC->nlocProj[ityp].nproj) continue;
                    if (! pSPARC->nlocProj_kptcomm[ityp].nprojso_ext) continue;
                    for (iat = 0; iat < pSPARC->Atom_Influence_nloc_kptcomm[ityp].n_atom; iat++) {
                        free( pSPARC->nlocProj_kptcomm[ityp].Chisowt0[iat] );
                        free( pSPARC->nlocProj_kptcomm[ityp].Chisowtl[iat] );
                        free( pSPARC->nlocProj_kptcomm[ityp].Chisowtnl[iat] );
                    }
                    free( pSPARC->nlocProj_kptcomm[ityp].Chisowt0 );
                    free( pSPARC->nlocProj_kptcomm[ityp].Chisowtl );
                    free( pSPARC->nlocProj_kptcomm[ityp].Chisowtnl );
                }
            }
            free(pSPARC->nlocProj_kptcomm);
        }
    }    

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
    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
        for (i = 0; i < pSPARC->Ntypes; i++) {
            if (pSPARC->Atom_Influence_nloc[i].n_atom > 0) {
                free(pSPARC->Atom_Influence_nloc[i].coords);
                free(pSPARC->Atom_Influence_nloc[i].atom_index);
                free(pSPARC->Atom_Influence_nloc[i].xs);
                free(pSPARC->Atom_Influence_nloc[i].xe);
                free(pSPARC->Atom_Influence_nloc[i].ys);
                free(pSPARC->Atom_Influence_nloc[i].ye);
                free(pSPARC->Atom_Influence_nloc[i].zs);
                free(pSPARC->Atom_Influence_nloc[i].ze);
                free(pSPARC->Atom_Influence_nloc[i].ndc);
                for (j = 0; j < pSPARC->Atom_Influence_nloc[i].n_atom; j++) {
                    free(pSPARC->Atom_Influence_nloc[i].grid_pos[j]);
                }
                free(pSPARC->Atom_Influence_nloc[i].grid_pos);
            }
        }
        free(pSPARC->Atom_Influence_nloc);
    }
	if (pSPARC->kptcomm_topo != MPI_COMM_NULL && pSPARC->kptcomm_index >= 0) {
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
    }
}

