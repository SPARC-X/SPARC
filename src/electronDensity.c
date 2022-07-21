/**
 * @file    electronDensity.c
 * @brief   This file contains the functions for calculating electron density.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>


#include "electronicGroundState.h"
#include "electronDensity.h"
#include "eigenSolver.h"
#include "eigenSolverKpt.h" 
#include "isddft.h"

#if USE_PCE /* Include PCE Headers */
#include <libpce.h>
#include "hamstruct.h"
#endif

/*
@ brief: Main function responsible to find electron density
*/
#if USE_PCE /* PCE Requires some additional data for reuse */
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error,
                        Hybrid_Decomp *hd, Chebyshev_Info *cheb, Eig_Info *Eigvals,
                        Our_Hamiltonian_Struct *ham_struct, 
                        Psi_Info *Psi1, Psi_Info *Psi2, Psi_Info *Psi3,
                        MPI_Comm kptcomm, MPI_Comm dmcomm, MPI_Comm blacscomm)
#else
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error)
#endif
{
    int i;
    double *rho = (double *) calloc(pSPARC->Nd_d_dmcomm * (2*pSPARC->Nspin-1), sizeof(double));
    double t1 = MPI_Wtime();
    
    // Currently only involves Chebyshev filtering eigensolver
    if (pSPARC->isGammaPoint){

        #if USE_PCE
            eigSolve_CheFSI(rank, pSPARC, SCFcount, error,
                            hd, cheb, Eigvals,
                            ham_struct, 
                            Psi1, Psi2, Psi3,
                            kptcomm, dmcomm, blacscomm);
        #else
            eigSolve_CheFSI(rank, pSPARC, SCFcount, error);
        #endif

        if(pSPARC->spin_typ == 0) {
        #if USE_PCE
            PCE_Density_Calculate(rho, Eigvals, hd, Psi1, pSPARC->dV, ham_struct->communication_device, ham_struct->compute_device, blacscomm);
        #else /* USE_PCE */
         CalculateDensity_psi(pSPARC, rho);
        #endif /* USE_PCE */
        }
        else
            CalculateDensity_psi_spin(pSPARC, rho);     
    }
    else{
        eigSolve_CheFSI_kpt(rank, pSPARC, SCFcount, error);
        if(pSPARC->spin_typ == 0)
            CalculateDensity_psi_kpt(pSPARC, rho);
        else
            CalculateDensity_psi_kpt_spin(pSPARC, rho);
    }

    double t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("rank = %d, Calculating density took %.3f ms\n",rank,(t2-t1)*1e3);       
    if(!rank) printf("rank = %d, starting to transfer density...\n",rank);
#endif
    
    // transfer density from psi-domain to phi-domain
    t1 = MPI_Wtime();    
    for (i = 0; i < 2*pSPARC->Nspin-1; i++)
        TransferDensity(pSPARC, rho + i*pSPARC->Nd_d_dmcomm, pSPARC->electronDens + i*pSPARC->Nd_d);
    t2 = MPI_Wtime();
    
#ifdef DEBUG
    if(!rank) printf("rank = %d, Transfering density took %.3f ms\n", rank, (t2 - t1) * 1e3);
#endif

    free(rho);
}


/**
 * @brief   Calculate electron density with given states in psi-domain.
 *
 *          Note that here rho is distributed in psi-domain, which needs
 *          to be transmitted to phi-domain for solving the poisson 
 *          equation.
 */
void CalculateDensity_psi(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, Nd, count, nstart, nend;
    double g_nk;
    // Ns = pSPARC->Nstates;
    Nd = pSPARC->Nd_d_dmcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    
    t1 = MPI_Wtime();
    // calculate rho based on local bands
    count = 0;
    for (n = nstart; n <= nend; n++) {
        // g_nk = 2.0 * smearing_FermiDirac(pSPARC->Beta,pSPARC->lambda[n],pSPARC->Efermi);
        g_nk = 2.0 * pSPARC->occ[n];
        for (i = 0; i < Nd; i++, count++) {
            rho[i] += g_nk * pSPARC->Xorb[count] * pSPARC->Xorb[count];
        }
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    // sum over all band groups
    if (pSPARC->npband > 1) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    double vscal = 1.0 / pSPARC->dV;
    // scale electron density by 1/dV
    // TODO: this can be done in phi-domain over more processes!
    //       Perhaps right after transfer to phi-domain is complete.
    for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
        rho[i] *= vscal; 
    }
    
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}


/**
 * @brief   Calculate electron density with given states in psi-domain with spin on.
 *
 *          Note that here rho is distributed in psi-domain, which needs
 *          to be transmitted to phi-domain for solving the poisson 
 *          equation.
 */
void CalculateDensity_psi_spin(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, Ns, Nd, count, nstart, nend, spn_i, sg;
    double g_nk;
    Ns = pSPARC->Nstates;
    Nd = pSPARC->Nd_d_dmcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;

    t1 = MPI_Wtime();
    
    // calculate rho based on local bands
    count = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        sg = spn_i + pSPARC->spin_start_indx;
        for (n = nstart; n <= nend; n++) {
            // g_nk = 2.0 * smearing_FermiDirac(pSPARC->Beta,pSPARC->lambda[n],pSPARC->Efermi);
            g_nk = pSPARC->occ[n+spn_i*Ns];
            for (i = 0; i < Nd; i++, count++) {
                rho[(sg+1)*Nd + i] += g_nk * pSPARC->Xorb[count] * pSPARC->Xorb[count];
            }
        }
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
    // sum over spin comm
    t1 = MPI_Wtime();
    if(pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(rho, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    // sum over all band groups
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, 3*pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    } // TODO: can be made only 2*Nd

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    double vscal = 1.0 / pSPARC->dV;
    // scale electron density by 1/dV
    // TODO: this can be done in phi-domain over more processes!
    //       Perhaps right after transfer to phi-domain is complete.
    for (i = 0; i < 2*Nd; i++) {
        rho[Nd+i] *= vscal;
    }    
    
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    for (i = 0; i < Nd; i++) {
        rho[i] = rho[Nd+i] + rho[2*Nd+i]; 
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Calculate rho: forming total rho took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}



void CalculateDensity_psi_kpt(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, k, Ns, count, nstart, nend, spinor;
    double g_nk, occfac;
    Ns = pSPARC->Nstates;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    occfac = 2.0/pSPARC->Nspinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    
    t1 = MPI_Wtime();
    
    // calculate rho based on local bands
    count = 0;
    
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (n = nstart; n <= nend; n++) {
            g_nk = occfac * (pSPARC->kptWts_loc[k] / pSPARC->Nkpts) * pSPARC->occ[k*Ns+n];
            for (spinor = 0; spinor < pSPARC->Nspinor; spinor ++) {
                for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
                    rho[i] += g_nk * (pow(creal(pSPARC->Xorb_kpt[count]), 2.0) 
                                    + pow(cimag(pSPARC->Xorb_kpt[count]), 2.0));
                    count++;
                }
            }
        }
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    
    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(rho, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
    }
    
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
    
    t1 = MPI_Wtime();
    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index == 0) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    double vscal = 1.0 / pSPARC->dV;
    // scale electron density by 1/dV
    // TODO: this can be done in phi-domain over more processes!
    //       Perhaps right after transfer to phi-domain is complete.
    for (i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
        rho[i] *= vscal;
    }
    
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}




void CalculateDensity_psi_kpt_spin(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;
    
    int i, n, k, Ns, Nd, Nk, count, nstart, nend, sg, spn_i;
    double g_nk;
    Ns = pSPARC->Nstates;
    Nd = pSPARC->Nd_d_dmcomm;
    Nk = pSPARC->Nkpts_kptcomm;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    
    t1 = MPI_Wtime();
    // calculate rho based on local bands
    count = 0;
    for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        sg = spn_i + pSPARC->spin_start_indx;
        for (k = 0; k < Nk; k++) {
            for (n = nstart; n <= nend; n++) {
                g_nk = (pSPARC->kptWts_loc[k] / pSPARC->Nkpts) * pSPARC->occ[spn_i*Nk*Ns+k*Ns+n];
                for (i = 0; i < pSPARC->Nd_d_dmcomm; i++, count++) {
                    rho[i+(sg+1)*Nd] += g_nk * (pow(creal(pSPARC->Xorb_kpt[count]), 2.0) + pow(cimag(pSPARC->Xorb_kpt[count]), 2.0));
                }
            }
        }
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    // sum over spin comm group
    t1 = MPI_Wtime();
    if(pSPARC->npspin > 1) {
        if (pSPARC->spincomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
        else
            MPI_Reduce(rho, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->spin_bridge_comm);
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    // sum over all k-point groups
    if (pSPARC->spincomm_index == 0 &&  pSPARC->npkpt > 1) {    
        if (pSPARC->kptcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
        else
            MPI_Reduce(rho, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->kpt_bridge_comm);
    }
    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
    
    t1 = MPI_Wtime();
    // sum over all band groups (only in the first k point group)
    if (pSPARC->npband > 1 && pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0) {
        if (pSPARC->bandcomm_index == 0)
            MPI_Reduce(MPI_IN_PLACE, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
        else
            MPI_Reduce(rho, rho, 3*Nd, MPI_DOUBLE, MPI_SUM, 0, pSPARC->blacscomm);
    }

    t2 = MPI_Wtime();

#ifdef DEBUG
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    double vscal = 1.0 / pSPARC->dV;
    // scale electron density by 1/dV
    // TODO: this can be done in phi-domain over more processes!
    //       Perhaps right after transfer to phi-domain is complete.
    for (i = 0; i < 2*Nd; i++) {
        rho[Nd+i] *= vscal; 
    }
    
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
#endif

    t1 = MPI_Wtime();
    for (i = 0; i < Nd; i++) {
        rho[i] = rho[Nd+i] + rho[2*Nd+i]; 
    }
    t2 = MPI_Wtime();
#ifdef DEBUG
    if (!rank) printf("rank = %d, --- Calculate rho: forming total rho took %.3f ms\n", rank, (t2-t1)*1e3);
#endif
}

