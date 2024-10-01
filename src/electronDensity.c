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
#include <assert.h>

#include "electronicGroundState.h"
#include "electronDensity.h"
#include "eigenSolver.h"
#include "eigenSolverKpt.h" 
#include "isddft.h"


/*
@ brief: Main function responsible to find electron density
*/
void Calculate_elecDens(int rank, SPARC_OBJ *pSPARC, int SCFcount, double error){
    int i, DMnd;
    DMnd = pSPARC->Nd_d_dmcomm;
    double *rho = (double *) calloc(DMnd * (2*pSPARC->Nspinor-1), sizeof(double)); 
    double *mag = (double *) calloc(DMnd * pSPARC->Nmag, sizeof(double));   

#ifdef DEBUG
    double t1 = MPI_Wtime();
#endif
    
    // Currently only involves Chebyshev filtering eigensolver
    if (pSPARC->isGammaPoint){
        eigSolve_CheFSI(rank, pSPARC, SCFcount, error);
    } else {
        eigSolve_CheFSI_kpt(rank, pSPARC, SCFcount, error);
    }

    CalculateDensity_psi(pSPARC, rho + (pSPARC->Nspinor-1)*DMnd);

    if (pSPARC->Nspinor > 1) {
        // calculate total electron density
        for (i = 0; i < DMnd; i++) {
            rho[i] = rho[DMnd+i] + rho[2*DMnd+i]; 
        }
    }

    if (pSPARC->spin_typ == 1) {
        Calculate_Magz(pSPARC, DMnd, mag, rho+DMnd, rho+2*DMnd); // magz
    }

    if (pSPARC->spin_typ == 2) {
        // magx, magy
        Calculate_Magx_Magy_psi(pSPARC, mag+DMnd); 
        // magz
        Calculate_Magz(pSPARC, DMnd, mag+3*DMnd, rho+DMnd, rho+2*DMnd); 
        // magnorm
        Calculate_Magnorm(pSPARC, DMnd, mag+DMnd, mag+2*DMnd, mag+3*DMnd, mag); 
        // update rhod11 rhod22
        Calculate_diagonal_Density(pSPARC, DMnd, mag, rho, rho+DMnd, rho+2*DMnd); 
    }

#ifdef DEBUG
    double t2 = MPI_Wtime();
    if(!rank) printf("rank = %d, Calculating density and magnetization took %.3f ms\n",rank,(t2-t1)*1e3);       
    if(!rank) printf("rank = %d, starting to transfer density and magnetization ...\n",rank);
#endif

    // transfer density from psi-domain to phi-domain
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    for (i = 0; i < pSPARC->Nspdentd; i++)
        TransferDensity(pSPARC, rho + i*DMnd, pSPARC->electronDens + i*pSPARC->Nd_d);
    
    for (i = 0; i < pSPARC->Nmag; i++)
        TransferDensity(pSPARC, mag + i*DMnd, pSPARC->mag + i*pSPARC->Nd_d);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("rank = %d, Transfering density and magnetization took %.3f ms\n", rank, (t2 - t1) * 1e3);
#endif

    free(rho);
    free(mag);
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
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, n, k, Ns, count, nstart, nend, spinor, DMnd;
    double g_nk;
    Ns = pSPARC->Nstates;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    DMnd = pSPARC->Nd_d_dmcomm;
    int Nspinor = pSPARC->Nspinor;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif

    // calculate rho based on local bands
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (n = nstart; n <= nend; n++) {
            double woccfac = pSPARC->occfac * (pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
            for (spinor = 0; spinor < pSPARC->Nspinor_spincomm; spinor ++) {
                int spinor_g = spinor + pSPARC->spinor_start_indx;
                double *occ = pSPARC->occ + k*Ns; 
                if (pSPARC->spin_typ == 1) occ += spinor*Ns*pSPARC->Nkpts_kptcomm;
                g_nk = woccfac * occ[n];

                if (pSPARC->isGammaPoint) {
                    for (i = 0; i < DMnd; i++) {
                        rho[i+spinor_g*DMnd] += g_nk * pSPARC->Xorb[count] * pSPARC->Xorb[count];
                        count++;
                    }
                } else {
                    for (i = 0; i < DMnd; i++) {
                        rho[i+spinor_g*DMnd] += g_nk * (creal(pSPARC->Xorb_kpt[count]) * creal(pSPARC->Xorb_kpt[count])
                                                      + cimag(pSPARC->Xorb_kpt[count]) * cimag(pSPARC->Xorb_kpt[count]));
                        count++;
                    }
                }
            }
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over spin comm group
    if(pSPARC->npspin > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, rho, Nspinor*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, rho, Nspinor*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over all band groups 
    if (pSPARC->npband) {
        MPI_Allreduce(MPI_IN_PLACE, rho, Nspinor*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate rho: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    if (!pSPARC->CyclixFlag) {
        double vscal = 1.0 / pSPARC->dV;
        // scale electron density by 1/dV        
        for (i = 0; i < Nspinor*DMnd; i++) {
            rho[i] *= vscal; 
        }
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("rank = %d, --- Scale rho: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
}



/**
 * @brief   Calculate off-diagonal electron density with given states in psi-domain.
 *
 */
void Calculate_Magx_Magy_psi(SPARC_OBJ *pSPARC, double *mag)
{
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

    int i, n, k, Ns, count, nstart, nend, DMnd;    
    Ns = pSPARC->Nstates;
    nstart = pSPARC->band_start_indx;
    nend = pSPARC->band_end_indx;
    DMnd = pSPARC->Nd_d_dmcomm;    

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
    double t1, t2;
    t1 = MPI_Wtime();
#endif    

    // calculate rho based on local bands
    count = 0;
    for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
        for (n = nstart; n <= nend; n++) {
            double woccfac = pSPARC->occfac * (pSPARC->kptWts_loc[k] / pSPARC->Nkpts);
            double g_nk = woccfac * pSPARC->occ[n + k*Ns];
            for (i = 0; i < DMnd; i++) {
                double _Complex rho_odd = pSPARC->Xorb_kpt[count] * conj(pSPARC->Xorb_kpt[count+DMnd]);
                mag[i] += 2 * g_nk * creal(rho_odd);      // magx
                mag[i+DMnd] -= 2 * g_nk * cimag(rho_odd); // magy
                count ++;
            }
            count += DMnd;
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: sum over local bands took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over spin comm group
    if(pSPARC->npspin > 1) {        
        MPI_Allreduce(MPI_IN_PLACE, mag, 2*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: reduce over all spin_comm took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    // sum over all k-point groups
    if (pSPARC->npkpt > 1) {            
        MPI_Allreduce(MPI_IN_PLACE, mag, 2*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: reduce over all kpoint groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // sum over all band groups 
    if (pSPARC->npband) {
        MPI_Allreduce(MPI_IN_PLACE, mag, 2*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, --- Calculate magx, magy: reduce over all band groups took %.3f ms\n", rank, (t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif

    if (!pSPARC->CyclixFlag) {
        double vscal = 1.0 / pSPARC->dV;
        // scale mag by 1/dV
        for (i = 0; i < 2*DMnd; i++) {
            mag[i] *= vscal; 
        }
        
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (!rank) printf("rank = %d, --- Scale mag: scale by 1/dV took %.3f ms\n", rank, (t2-t1)*1e3);
    #endif
    }
}

/*
@ brief: calculate magz
*/ 
void Calculate_Magz(SPARC_OBJ *pSPARC, int DMnd, double *magz, double *rhoup, double *rhodw)
{
    for (int i = 0; i < DMnd; i++) {
        magz[i] = rhoup[i] - rhodw[i];
    }
}

/*
@ brief: calculate norm of magnetization
*/ 
void Calculate_Magnorm(SPARC_OBJ *pSPARC, int DMnd, double *magx, double *magy, double *magz, double *magnorm)
{
    for (int i = 0; i < DMnd; i++) {
        magnorm[i] = sqrt(magx[i]*magx[i] + magy[i]*magy[i] + magz[i]*magz[i]);
    }
}


void Calculate_diagonal_Density(SPARC_OBJ *pSPARC, int DMnd, double *magnorm, double *rho_tot, double *rho11, double *rho22)
{
    for (int i = 0; i < DMnd; i++) {
        rho11[i] = 0.5*(rho_tot[i] + magnorm[i]);
        rho22[i] = 0.5*(rho_tot[i] - magnorm[i]);
    }
}