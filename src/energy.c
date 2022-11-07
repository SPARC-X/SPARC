/**
 * @file    energy.c
 * @brief   This file contains functions for calculating system energies.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * @Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "energy.h"
#include "exchangeCorrelation.h"
#include "occupation.h"
#include "tools.h"
#include "isddft.h"
#include "sqEnergy.h"

#define TEMP_TOL (1e-14)


/**
 * @brief   Calculate free energy.
 */
void Calculate_Free_Energy(SPARC_OBJ *pSPARC, double *electronDens)
{
    //if (pSPARC->dmcomm_phi == MPI_COMM_NULL) return; 
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double Etot, Eband, Entropy, E1, E2, E3;
    
    Etot = Eband = Entropy = E1 = E2 = E3 = 0.0; // initialize energies

    #ifdef DEBUG
    double dEtot = 0.0, dEband = 0.0; // this is for temp use
    #endif

    // exchange-correlation energy
    Calculate_Exc(pSPARC, electronDens);
    
    // band structure energy
    if (pSPARC->SQFlag == 1) {
        Eband = Calculate_Eband_SQ(pSPARC);
    } else {
        Eband = Calculate_Eband(pSPARC);
    }
    
    pSPARC->Eband = Eband;
    #ifdef DEBUG
    // find changes in Eband from previous SCF step
    dEband = fabs(Eband - pSPARC->Eband) / pSPARC->n_atom;
    #endif
    
    // calculate entropy
    if (pSPARC->SQFlag == 1) {
        pSPARC->Entropy = Calculate_electronicEntropy_SQ(pSPARC);
    } else {
        pSPARC->Entropy = Calculate_electronicEntropy(pSPARC);
    }
    
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        VectorDotProduct(pSPARC->psdChrgDens, pSPARC->elecstPotential, pSPARC->Nd_d, &E1, pSPARC->dmcomm_phi);
        VectorDotProduct(electronDens, pSPARC->elecstPotential, pSPARC->Nd_d, &E2, pSPARC->dmcomm_phi);
        if (pSPARC->spin_typ == 0)
            VectorDotProduct(electronDens, pSPARC->XCPotential, pSPARC->Nd_d, &E3, pSPARC->dmcomm_phi);
        else
            VectorDotProduct(electronDens + pSPARC->Nd_d, pSPARC->XCPotential, 2*pSPARC->Nd_d, &E3, pSPARC->dmcomm_phi);

        E1 *= 0.5 * pSPARC->dV;
        E2 *= 0.5 * pSPARC->dV;
        E3 *= pSPARC->dV;

        if (pSPARC->mGGAflag == 1) {
            double Emgga;
            if (pSPARC->spin_typ == 0)
                VectorDotProduct(pSPARC->KineticTauPhiDomain, pSPARC->vxcMGGA3, pSPARC->Nd_d, &Emgga, pSPARC->dmcomm_phi);
            else
                VectorDotProduct(pSPARC->KineticTauPhiDomain + pSPARC->Nd_d, pSPARC->vxcMGGA3, 2*pSPARC->Nd_d, &Emgga, pSPARC->dmcomm_phi);
            Emgga *= pSPARC->dV;
            E3 += Emgga;
        }
    }
    
    if ((pSPARC->usefock == 0 ) || (pSPARC->usefock%2 == 1)) {
        // calculate total free energy
        Etot = Eband + E1 - E2 - E3 + pSPARC->Exc + pSPARC->Esc + pSPARC->Entropy;
        pSPARC->Exc_corr = E3;
        pSPARC->Etot = Etot;
        MPI_Bcast(&pSPARC->Etot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #ifdef DEBUG
        // find change in Etot from previous SCF step
        dEtot = fabs(Etot - pSPARC->Etot) / pSPARC->n_atom;
        if(!rank) printf("Etot    = %18.12f\nEband   = %18.12f\nE1      = %18.12f\nE2      = %18.12f\n"
                        "E3      = %18.12f\nExc     = %18.12f\nEsc     = %18.12f\nEntropy = %18.12f\n"
                        "dE = %.3e, dEband = %.3e\n", 
                Etot, Eband, E1, E2, E3, pSPARC->Exc, pSPARC->Esc, pSPARC->Entropy, dEtot, dEband); 
    #endif
    } else {
        // add the Exact exchange correction term    
        pSPARC->Exc += pSPARC->Eexx;
        // calculate total free energy
        Etot = Eband + E1 - E2 - E3 + pSPARC->Exc + pSPARC->Esc + pSPARC->Entropy - 2*pSPARC->Eexx;
        pSPARC->Exc_corr = E3;
        pSPARC->Etot = Etot;
        MPI_Bcast(&pSPARC->Etot, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #ifdef DEBUG
        // find change in Etot from previous SCF step
        dEtot = fabs(Etot - pSPARC->Etot) / pSPARC->n_atom;
        if(!rank) printf("Etot    = %18.12f\nEband   = %18.12f\nE1      = %18.12f\nE2      = %18.12f\n"
                        "E3      = %18.12f\nExc     = %18.12f\nEsc     = %18.12f\nEntropy = %18.12f\n"
                        "Eexx    = %18.12f\n, dE = %.3e, dEband = %.3e\n", 
                Etot, Eband, E1, E2, E3, pSPARC->Exc, pSPARC->Esc, pSPARC->Entropy, pSPARC->Eexx, dEtot, dEband); 
    #endif
    }
}


/**
 * @brief   Calculate band energy.
 */
double Calculate_Eband(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int n, Ns, k, spn_i, Nk;
    double Eband, occfac; 
    
    Eband = 0.0; // initialize energies
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    occfac = 2.0/pSPARC->Nspin/pSPARC->Nspinor;

    if (pSPARC->isGammaPoint) { // for gamma-point systems
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (n = 0; n < Ns; n++) {
                // Eband += 2.0 * smearing_FermiDirac(pSPARC->Beta, pSPARC->lambda[n], pSPARC->Efermi) * pSPARC->lambda[n];
                Eband += occfac * pSPARC->occ[n+spn_i*Ns] * pSPARC->lambda[n+spn_i*Ns];
            }
        }
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find Eband
            MPI_Allreduce(MPI_IN_PLACE, &Eband, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }    
    } else { // for k-points
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (k = 0; k < Nk; k++) {
                for (n = 0; n < Ns; n++) {
                    //Eband += 2.0 * pSPARC->kptWts_loc[k] * smearing_FermiDirac(pSPARC->Beta, pSPARC->lambda[n+k*Ns], pSPARC->Efermi)
                    //         * pSPARC->lambda[n+k*Ns];
                    Eband += occfac * pSPARC->kptWts_loc[k] * pSPARC->occ[n+k*Ns+spn_i*Nk*Ns] * pSPARC->lambda[n+k*Ns+spn_i*Nk*Ns];
                }
            }
        }    
        Eband /= pSPARC->Nkpts;
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find Eband
            MPI_Allreduce(MPI_IN_PLACE, &Eband, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }
        if (pSPARC->npkpt != 1) { // sum over processes with the same rank in kptcomm to find Eband
            MPI_Allreduce(MPI_IN_PLACE, &Eband, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
    }  
    return Eband;
}


/**
 * @brief   Calculate electronic entropy.  
 */
double Calculate_electronicEntropy(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0) return 0.0;
    int k, Ns = pSPARC->Nstates, Nk = pSPARC->Nkpts_kptcomm, spn_i;
    double occfac = 2.0/pSPARC->Nspin/pSPARC->Nspinor;
    double Entropy = 0.0;
    if (pSPARC->isGammaPoint) { // for gamma-point systems
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            Entropy += Calculate_entropy_term (
                pSPARC->lambda+spn_i*Ns, pSPARC->occ+spn_i*Ns, pSPARC->Efermi, 0, Ns-1, 
                pSPARC->Beta, pSPARC->elec_T_type
            );
        }    
        Entropy *= -occfac / pSPARC->Beta;
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find Eband
            MPI_Allreduce(MPI_IN_PLACE, &Entropy, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }
    } else { 
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (k = 0; k < pSPARC->Nkpts_kptcomm; k++) {
                double Entropy_k = Calculate_entropy_term (
                    pSPARC->lambda+k*Ns+spn_i*Nk*Ns, pSPARC->occ+k*Ns+spn_i*Nk*Ns, pSPARC->Efermi, 0, Ns-1, 
                    pSPARC->Beta, pSPARC->elec_T_type
                );
                Entropy += Entropy_k * pSPARC->kptWts_loc[k]; // multiply by the kpoint weights
            }
        }    
        Entropy *= -occfac / (pSPARC->Nkpts * pSPARC->Beta);
 
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find Eband
            MPI_Allreduce(MPI_IN_PLACE, &Entropy, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }

        if (pSPARC->npkpt != 1) { // sum over processes with the same rank in kptcomm to find Eband
            MPI_Allreduce(MPI_IN_PLACE, &Entropy, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
    }   
    return Entropy;
}



/**
 * @brief   Calculate entropy term for the provided eigenvalues.
 *
 *          There are several choices of entropy terms depending on 
 *          what smearing method is used. 
 *          For Fermi-Dirac smearing: 
 *              S(occ) = -[occ * ln(occ) + (1-occ) * ln(1-occ)],
 *          For Gaussian smearing:
 *              S(lambda) = 1/sqrt(pi) * exp(-((lambda - lambda_f)/sigma)^2),
 *          For Methfessel-Paxton with Hermite polynomial of degree 2N,
 *              S(lambda) = 1/2 * A_N * H_2N(x) * exp(-x^2),
 *          where x = (lambda - lambda_f)/sigma, H_n are the Hermite polynomial
 *          of degree n, A_N is given in ref: M. Methfessel and A.T. Paxton (1989). 
 *          Note: when N = 0, MP is equivalent to gaussian smearing. Currently 
 *          not implemented. 
 *
 * @param lambda     Eigenvalue.
 * @param lambda_f   Fermi level.
 * @param occ        Occupation number corresponding to the eigenvalue (only 
 *                   used in Fermi-Dirac smearing).
 * @param beta       beta := 1/sigma.   
 * @param type       Smearing type. type = 0: Fermi-Dirac, type = 1: Gassian (MP0),
 *                   type > 1: Methfessel-Paxton (N = type - 1).
 */
double Calculate_entropy_term(
    const double *lambda,    const double *occ,
    const double lambda_f,   const int n_start,       
    const int n_end,         const double beta,       
    const int type
) 
{
    int n;
    double Entropy_term = 0.0;
    const double c = 0.5 / sqrt(M_PI);
    switch (type) {
        case 0: // fermi-dirac 
            for (n = n_start; n <= n_end; n++) {
                double g_nk = occ[n];
                if (g_nk > TEMP_TOL && (1.0-g_nk) > TEMP_TOL) 
                    Entropy_term += -(g_nk * log(g_nk) + (1.0 - g_nk) * log(1.0 - g_nk));
            }
            break;
        case 1: // gaussian
            for (n = n_start; n <= n_end; n++) {
                double x = beta * (lambda[n] - lambda_f);
                Entropy_term += c * exp(-x*x);
            }
            break;
        default: 
            printf("Methfessel-Paxton with N = %d is not implemented\n",type-1);
    }
    return Entropy_term;
}



/**
 * @brief   Calculate self consistent correction to free energy.
 */
double Calculate_Escc(
    SPARC_OBJ *pSPARC,      const int DMnd,
    const double *Veff_out, const double *Veff_in,
    const double *rho_out,  MPI_Comm comm
)
{
    int size_comm, nproc;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_size(comm, &size_comm);
    } else {
        size_comm = 0;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    double Escc_in, Escc_out, Escc;
    if (comm != MPI_COMM_NULL) {
        VectorDotProduct(rho_out, Veff_out, DMnd, &Escc_out, comm);
        VectorDotProduct(rho_out, Veff_in , DMnd, &Escc_in , comm);
        Escc = (Escc_out - Escc_in) * pSPARC->dV;
    }
    if (size_comm < nproc)
        MPI_Bcast(&Escc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return Escc;
}
