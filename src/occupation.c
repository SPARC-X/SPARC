/**
 * @file    occupation.c
 * @brief   This file contains functions for calculating occupations.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "occupation.h"
#include "isddft.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


/**
 * @brief   Smearing function.
 *
 *          Currently there are two types of smearing functions implemented, 
 *          Gaussian smearing function and Fermi-Dirac smearing function.
 */
double smearing_function(double beta, double lambda, double lambda_f, int type)
{
    return (type) ? 
           0.5 * (1.0 - erf(beta * (lambda - lambda_f))) : // Gaussian
           1.0 / (1.0 + exp(beta * (lambda - lambda_f)));  // Fermi-Dirac
}



/**
 * @brief   Find occupation, and return fermi level.
 */
double Calculate_occupation(SPARC_OBJ *pSPARC, double x1, double x2, double tol, int max_iter)
{
    double Efermi, g_nk;
    int n, Ns, k, Nk, spn_i;
    
    Ns = pSPARC->Nstates;
    Nk = pSPARC->Nkpts_kptcomm;
    
    // find fermi level using Brent's method
    Efermi = Calculate_FermiLevel(pSPARC, x1, x2, tol, max_iter, occ_constraint);

    // find occupations
    if (pSPARC->isGammaPoint) { // for gamma-point systems
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (n = 0; n < Ns; n++) {
                g_nk = smearing_function(
                    pSPARC->Beta, pSPARC->lambda[n+spn_i*Ns], Efermi, pSPARC->elec_T_type
                );
                pSPARC->occ[n+spn_i*Ns] = g_nk;
            }
        }
    } else { // for k-points
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (k = 0; k < Nk; k++) {
                for (n = 0; n < Ns; n++) {
                    g_nk = smearing_function(
                        pSPARC->Beta, pSPARC->lambda[n+k*Ns+spn_i*Nk*Ns], Efermi, pSPARC->elec_T_type
                    );
                    pSPARC->occ[n+k*Ns+spn_i*Nk*Ns] = g_nk;
                }
            }
        }    
    }
    return Efermi;
}



/**
 * @brief   Find fermi level using Brent's method.
 *
 * @ref     W.H. Press, Numerical recepies 3rd edition: The art of scientific 
 *          computing, Cambridge university press, 2007.
 */
double Calculate_FermiLevel(SPARC_OBJ *pSPARC, double x1, double x2, double tol, int max_iter, 
    double (*constraint)(SPARC_OBJ*, double)) 
{
#define EPSILON 1e-16
#define SIGN(a,b) ((b) > 0.0 ? fabs((a)) : -fabs((a)))
    if (pSPARC->SQFlag == 1) {
        if (pSPARC->pSQ->dmcomm_SQ == MPI_COMM_NULL) return 0.0;
    } else {
        if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return 0.0;
    }
    // let all root processors in each kptcomm work together to find fermi energy.
    int iter;
    double tol1q, eq;
    double a = min(x1,x2), b = max(x1,x2), c = b, d, e = 0.0, min1, min2;
    double fa = constraint(pSPARC,a), fb = constraint(pSPARC,b), fc, p, q, r, s, tol1, xm;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // if the given upperbound and lowerbound are not good enough, expand the bounds automatically
    int ext_count = 0;
    while (fa * fb > 0.0 && ext_count++ < 10) {   
        double w = b - a;
        a -= w/2.0;
        b += w/2.0;
        c = b;
        fa = constraint(pSPARC,a);
        fb = constraint(pSPARC,b);
#ifdef DEBUG
        if (rank == 0) 
            printf("Fermi level calculation: expanded bounds = (%10.6E,%10.6E), (fa,fb) = (%10.6E,%10.6E)\n", 
                a,b,fa,fb);
#endif
    }

    if (fa * fb > 0.0) {   
        if (rank == 0) printf("Cannot find root in Brent's method!\n" 
                              "  original bounds (%f,%f)\n"
                              "  expanded to     (%f,%f)\n", x1,x2,a,b);
        if (rank == 0) printf("Fermi level calculation failed, consider increasing number of states!\n");
        exit(EXIT_FAILURE);
    }

    fc = fb;
    for (iter = 1; iter <= max_iter; iter++) {
        if((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a; fc = fa;
            e = d = b-a;
        }
        if ( fabs(fc) < fabs(fb) ) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        tol1 = 2.0 * EPSILON * fabs(b) + 0.5 * tol; 
        xm = 0.5*(c-b);
        if(fabs(xm) <= tol1 || fabs(fb) < EPSILON) return b;
        if(fabs(e) >= tol1 && fabs(fa) > fabs(fb)) {
            // attempt inverse quadratic interpolation
            s = fb / fa;
            if( a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc; r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if(p > 0.0) {
                //check whether in bounds
                q = -q;
            }
            p = fabs(p);
            tol1q = tol1 * q;
            min1 = 3.0 * xm * q - fabs(tol1q);
            eq = e * q;
            min2 = fabs(eq);
            if(2.0 * p < (min1 < min2 ? min1 : min2)) { 
                //accept interpolation
                e = d; d = p / q;
            } else {
                // Bounds decreasing too slowly, use bisection
                d = xm; e = d;
            }
        } else {
            d = xm; e = d;
        }
        // move last best guess to a
        a = b; fa = fb;
        
        if (fabs(d) > tol1) {
            // evaluate new trial root
            b += d;
        } else {
            b += SIGN(tol1, xm);    
        }
        fb = constraint(pSPARC,b);
    }
    printf("Maximum iterations exceeded in brents root finding method...exiting\n");
    exit(EXIT_FAILURE);
    return 0.0;
#undef EPSILON
#undef SIGN
}



/**
 * @brief   With given fermi energy guess, this function returns the difference  
 *          between the negative charges of the system and the average integral 
 *          of the occupations in the Brillouin zone.
 *
 */
double occ_constraint(SPARC_OBJ *pSPARC, double lambda_f)
{
    if (pSPARC->kptcomm_index < 0) return 0.0;
    int Ns = pSPARC->Nstates, n, k, spn_i;
    int Nk = pSPARC->Nkpts_kptcomm;
    // TODO: confirm whether to use number of electrons or NegCharge
    double g = 0.0, Ne = pSPARC->NegCharge; 
    if (pSPARC->isGammaPoint) { // for gamma-point systems
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (n = 0; n < Ns; n++) {
                g += (2.0/pSPARC->Nspin) * smearing_function(
                    pSPARC->Beta, pSPARC->lambda[n+spn_i*Ns], lambda_f, pSPARC->elec_T_type
                );
            }
        }
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find g
            MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }
    } else { // for k-points
        for (spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
            for (k = 0; k < Nk; k++) {
                // TODO: each k-point group should access its local kpoints!
                for (n = 0; n < Ns; n++) {
                    g += (2.0/pSPARC->Nspin/pSPARC->Nspinor) * pSPARC->kptWts_loc[k] * smearing_function(
                        pSPARC->Beta, pSPARC->lambda[n+k*Ns+spn_i*Nk*Ns], lambda_f, pSPARC->elec_T_type
                    );
                }
            }
        }    
        g *= 1.0 / pSPARC->Nkpts; // find average
        if (pSPARC->npspin != 1) { // sum over processes with the same rank in spincomm to find g
            MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);
        }
        
        if (pSPARC->npkpt != 1) { // sum over processes with the same rank in kptcomm to find g
            MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
    }
    return g + Ne;
    // return g - pSPARC->Nelectron; // this will work even when Ns = Nelectron/2
}
