/**
 * @file    vdWDFnonlinearCorre.c
 * @brief   This file contains the functions for vdF-DF1 and vdW-DF2 functional.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * Reference:
 * Dion, Max, Henrik Rydberg, Elsebeth Schröder, David C. Langreth, and Bengt I. Lundqvist. 
 * "Van der Waals density functional for general geometries." 
 * Physical review letters 92, no. 24 (2004): 246401.
 * Román-Pérez, Guillermo, and José M. Soler. 
 * "Efficient implementation of a van der Waals density functional: application to double-wall carbon nanotubes." 
 * Physical review letters 103, no. 9 (2009): 096102.
 * Lee, Kyuho, Éamonn D. Murray, Lingzhu Kong, Bengt I. Lundqvist, and David C. Langreth. 
 * "Higher-accuracy van der Waals density functional." Physical Review B 82, no. 8 (2010): 081101.
 * Thonhauser, T., S. Zuluaga, C. A. Arter, K. Berland, E. Schröder, and P. Hyldgaard. 
 * "Spin signature of nonlocal correlation binding in metal-organic frameworks." 
 * Physical review letters 115, no. 13 (2015): 136402.
 * vdW-DF feature developed in Quantum Espresso:
 * Thonhauser, Timo, Valentino R. Cooper, Shen Li, Aaron Puzder, Per Hyldgaard, and David C. Langreth. 
 * "Van der Waals density functional: Self-consistent potential and the nature of the van der Waals bond." 
 * Physical Review B 76, no. 12 (2007): 125112.
 * Sabatini, Riccardo, Emine Küçükbenli, Brian Kolb, Timo Thonhauser, and Stefano De Gironcoli. 
 * "Structural evolution of amino acid crystals under stress from a non-empirical density functional." 
 * Journal of Physics: Condensed Matter 24, no. 42 (2012): 424209.
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <errno.h>
#include <time.h>

#include "isddft.h"
#include "initialization.h"
#include "tools.h"
#include "parallelization.h"
#include "vdWDFnonlinearCorre.h"
#include "vdWDFparallelization.h"
#include "gradVecRoutines.h"

/** BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
#endif
/** ScaLAPACK routines */
#ifdef USE_MKL
#include "blacs.h" // Cblacs_*
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
#include "blacs.h"     // Cblacs_*
#include "scalapack.h" // ScaLAPACK functions
#endif
/** FFT routines **/
#ifdef USE_MKL
#include "mkl_cdft.h"
#endif
#ifdef USE_FFTW
#include <fftw3-mpi.h>
#endif

/*
 Structure of functions in vdWDFNonlinearCorre.c

 ├──get_q0_Grid
 |  ├──pw
 |  └──saturate_q
 ├──spline_interpolation
 ├──compute_Gvectors
 ├──theta_generate_FT
 |  └──parallel_FFT
 ├──vdWDF_energy
 |  ├──interpolate_kernel
 |  └──kernel_label
 ├──u_generate_iFT
 |  └──parallel_iFFT
 └──vdWDF_potential

*/

#define KF(rho) (pow(3.0 * M_PI * M_PI * rho, 1.0 / 3.0))
#define FS(s) (1.0 - Zab * s * s / 9.0)
#define DFSDS(s) ((-2.0 / 9.0) * s * Zab)
#define DSDRHO(s, rho, kfResult) (-s * (DKFDRHO(rho, kfResult) / kfResult + 1.0 / rho))
#define DKFDRHO(rho, kfResult) ((1.0 / 3.0) * kfResult / rho)
#define DSDGRADRHO(rho, kfResult) (0.5 / (kfResult * rho))
#define INDEX1D(i, j, k) (i + pSPARC->Nx * j + pSPARC->Nx * pSPARC->Ny * k)

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) > (y) ? (y) : (x))

void saturate_q(double qp, double qCut, double *saturateq0dq0dq)
{ // (5) of Soler
    int mc = 12;
    double eexp = 0.0;
    double dq0dq = 0.0;
    double qDivqCut = qp / qCut;
    double qDivqCutPower = 1.0;
    for (int m = 1; m <= mc; m++)
    {
        dq0dq += qDivqCutPower;
        qDivqCutPower *= qDivqCut;
        eexp += qDivqCutPower / m;
    }
    saturateq0dq0dq[0] = qCut * (1.0 - exp(-eexp));
    saturateq0dq0dq[1] = dq0dq * exp(-eexp);
}

void get_q0_Grid(SPARC_OBJ *pSPARC, double *rho)
{
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double Zab = pSPARC->vdWDFZab;
    double *gradRhoLen = pSPARC->gradRhoLen;
    double *q0 = pSPARC->vdWDFq0;
    double *dq0drho = pSPARC->vdWDFdq0drho;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double qCut = pSPARC->vdWDFqmesh[nqs - 1];
    double qMin = pSPARC->vdWDFqmesh[0];

    double *Drho_x = pSPARC->Drho[0]; // computed in Calculate_Vxc_vdWExchangeLinearCorre
    double *Drho_y = pSPARC->Drho[1];
    double *Drho_z = pSPARC->Drho[2];

    int igrid;
    double s; // Dion's paper, (12) 2nd term
    double ecLDAPW;
    double kfResult, FsResult, qp, dqxdrho;
    double saturate[2];
    double *saturateq0dq0dq = saturate; // first: q0; second: dq0dq

    for (igrid = 0; igrid < DMnd; igrid++)
    {
        if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20)
        {
            nonCart2Cart_grad(pSPARC, &(Drho_x[igrid]), &(Drho_y[igrid]), &(Drho_z[igrid])); // transfer the gradient to cartesian direction
        }
        gradRhoLen[igrid] = sqrt(Drho_x[igrid] * Drho_x[igrid] + Drho_y[igrid] * Drho_y[igrid] + Drho_z[igrid] * Drho_z[igrid]);

        kfResult = KF(rho[igrid]);
        s = gradRhoLen[igrid] / (2.0 * kfResult * rho[igrid]);
        ecLDAPW = pSPARC->vdWDFecLinear[igrid]; // to prevent repeatly computing epsilon_cl and V_cl; they are computed in Calculate_Vc_PW91, vdWDFexchangeLinearCorre.c
        dq0drho[igrid] = pSPARC->vdWDFVcLinear[igrid];
        FsResult = FS(s);
        qp = -4.0 * M_PI / 3.0 * ecLDAPW + kfResult * FsResult; // energy ratio on every point
        saturate_q(qp, qCut, saturateq0dq0dq);                  // modify q into [qMin, qCut]
        q0[igrid] = saturate[0] > qMin ? saturate[0] : qMin;
        dqxdrho = DKFDRHO(rho[igrid], kfResult) * FsResult + kfResult * DFSDS(s) * DSDRHO(s, rho[igrid], kfResult);
        dq0drho[igrid] = saturate[1] * rho[igrid] * (-4.0 * M_PI / 3.0 * (dq0drho[igrid] - ecLDAPW) / rho[igrid] + dqxdrho);
        dq0dgradrho[igrid] = saturate[1] * rho[igrid] * kfResult * DFSDS(s) * DSDGRADRHO(rho[igrid], kfResult);
    }
    // // verify the correctness of result
    // if ((pSPARC->countPotentialCalculate == 0) && (rank == size - 1)) {
    //     printf("rank %d, (%d, %d, %d)-(%d, %d, %d), q0[0] %.6e, q0[DMnd - 1] %.6e\n",
    //      rank, pSPARC->DMVertices[0], pSPARC->DMVertices[2], pSPARC->DMVertices[4], pSPARC->DMVertices[1], pSPARC->DMVertices[3], pSPARC->DMVertices[5],
    //      q0[0], q0[DMnd - 1]);
    // }
}

void spin_get_q0_Grid(SPARC_OBJ *pSPARC, double *rho)
{ // rho has 3 DMnd cols: 1st n_up+n_dn; 2nd n_up; 3rd n_dn
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double Zab = pSPARC->vdWDFZab;
    double *gradRhoLen = pSPARC->gradRhoLen;
    double *q0 = pSPARC->vdWDFq0;
    double *dq0drho = pSPARC->vdWDFdq0drho;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double qCut = pSPARC->vdWDFqmesh[nqs - 1];
    double qMin = pSPARC->vdWDFqmesh[0];

    double *Drho_x = pSPARC->Drho[0]; // computed in Calculate_Vxc_vdWExchangeLinearCorre, two DMnd columns for n_up and n_dn
    double *Drho_y = pSPARC->Drho[1]; // currently they are directional gradient
    double *Drho_z = pSPARC->Drho[2];
    for (int i = 0; i < 2 * DMnd; i++)
    {
        if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20)
            nonCart2Cart_grad(pSPARC, &(Drho_x[i]), &(Drho_y[i]), &(Drho_z[i])); // transfer the gradient to cartesian direction
        gradRhoLen[i] = sqrt(Drho_x[i] * Drho_x[i] + Drho_y[i] * Drho_y[i] + Drho_z[i] * Drho_z[i]);
    }

    int igrid;
    double fac = pow(2.0, -1.0 / 3.0);
    double sUp = 0.0; double sDn = 0.0; // Dion's paper, (12) 2nd term
    double rho_up, rho_dn;
    double kfResult_up = 0.0; double kfResult_dn = 0.0; double FsResult_up = 0.0; double FsResult_dn = 0.0;
    double qx_up, qx_dn, q0x_up, q0x_dn, dqxdrho_up, dqxdrho_dn, dqcdrho_up, dqcdrho_dn;
    double qx, qc, q;
    // double saturate_up[2]; // double saturate_dn[2]; // double saturate[2];
    double *saturateq0dq0dq_up = (double *)malloc(sizeof(double)*2);
    double *saturateq0dq0dq_dn = (double *)malloc(sizeof(double)*2);
    double *saturateq0dq0dq = (double *)malloc(sizeof(double)*2);
    // double *saturateq0dq0dq = saturate; // first: q0; second: dq0dq

    double epsr = 1.0E-12; // lower bound of electron density
    for (igrid = 0; igrid < DMnd; igrid++)
    {
        q0[igrid] = qCut;
        dq0drho[igrid] = 0.0;
        dq0drho[DMnd + igrid] = 0.0;
        dq0dgradrho[igrid] = 0.0;
        dq0dgradrho[DMnd + igrid] = 0.0;
        if (rho[igrid] < epsr)
            continue; // if the electron density is smaller than epsr, this point will not be computed
        q0x_up = 0.0;
        q0x_dn = 0.0;
        dqxdrho_up = 0.0;
        dqxdrho_dn = 0.0;
        rho_up = rho[DMnd + igrid];
        rho_dn = rho[2 * DMnd + igrid];
        if (rho_up > epsr / 2.0)
        { // q0x_up(dn) is only computed for grids whose rho_up(dn) is more than epsr/2
            sUp = gradRhoLen[igrid] / (2.0 * KF(rho_up) * rho_up);
            kfResult_up = KF(2.0 * rho_up);
            FsResult_up = FS(fac * sUp); // maybe fac*sUp can be replaced by gradRhoLen[igrid] / (2.0 * KF(2.0*rho_up) * rho_up)
            qx_up = kfResult_up * FsResult_up;
            saturate_q(qx_up, 4.0 * qCut, saturateq0dq0dq_up);
            q0x_up = saturateq0dq0dq_up[0];
            dqxdrho_up += 2.0 * saturateq0dq0dq_up[1] * rho_up * (DKFDRHO((2.0 * rho_up), kfResult_up) * FsResult_up + kfResult_up * DFSDS(fac * sUp) * DSDRHO(fac * sUp, (2.0 * rho_up), kfResult_up)) + q0x_up * rho_dn / rho[igrid];
            dqxdrho_dn += -q0x_up * rho_up / rho[igrid];
        }
        if (rho_dn > epsr / 2.0)
        {
            sDn = gradRhoLen[DMnd + igrid] / (2.0 * KF(rho_dn) * rho_dn);
            kfResult_dn = KF(2.0 * rho_dn);
            FsResult_dn = FS(fac * sDn); // maybe fac*sDn can be replaced by gradRhoLen[igrid] / (2.0 * KF(2.0*rho_dn) * rho_dn)
            qx_dn = kfResult_dn * FsResult_dn;
            saturate_q(qx_dn, 4.0 * qCut, saturateq0dq0dq_dn);
            q0x_dn = saturateq0dq0dq_dn[0];
            dqxdrho_dn += 2.0 * saturateq0dq0dq_dn[1] * rho_dn * (DKFDRHO((2.0 * rho_dn), kfResult_dn) * FsResult_dn + kfResult_dn * DFSDS(fac * sDn) * DSDRHO(fac * sDn, (2.0 * rho_dn), kfResult_dn)) + q0x_dn * rho_up / rho[igrid];
            dqxdrho_up += -q0x_dn * rho_dn / rho[igrid]; // the formula of qdxdrho_up(dn) from reference [3] Supp (2a) and (2b)
        }
        qx = (rho_up * q0x_up + rho_dn * q0x_dn) / rho[igrid];
        qc = -4.0 * M_PI / 3.0 * pSPARC->vdWDFecLinear[igrid];
        q = qx + qc;
        saturate_q(q, qCut, saturateq0dq0dq);
        if (saturateq0dq0dq[0] < qMin)
            saturateq0dq0dq[0] = qMin;
        q0[igrid] = saturateq0dq0dq[0];
        dqcdrho_up = -4.0 * M_PI / 3.0 * (pSPARC->vdWDFVcLinear[igrid] - pSPARC->vdWDFecLinear[igrid]);
        dqcdrho_dn = -4.0 * M_PI / 3.0 * (pSPARC->vdWDFVcLinear[DMnd + igrid] - pSPARC->vdWDFecLinear[igrid]);
        dq0drho[igrid] = saturateq0dq0dq[1] * (dqxdrho_up + dqcdrho_up);
        dq0drho[DMnd + igrid] = saturateq0dq0dq[1] * (dqxdrho_dn + dqcdrho_dn); // Dq0Drho at here is (n_up + n_dn)*[d(q0)/d(n_up), d(q0)/d(n_dn)]
        if (rho_up > epsr / 2.0)                                                // Dq0Dgradrho at here is (n_up + n_dn)*d(q0)/d(|grad n_up|)
            dq0dgradrho[igrid] = 2.0 * saturateq0dq0dq[1] * saturateq0dq0dq_up[1] * rho_up * kfResult_up * DFSDS(fac * sUp) * DSDGRADRHO(2.0 * rho_up, kfResult_up);
        if (rho_dn > epsr / 2.0)
            dq0dgradrho[DMnd + igrid] = 2.0 * saturateq0dq0dq[1] * saturateq0dq0dq_dn[1] * rho_dn * kfResult_dn * DFSDS(fac * sDn) * DSDGRADRHO(2.0 * rho_dn, kfResult_dn);
        // if ((rank == 0) && (igrid == 17)) { // for debugging
        //     printf("rank 0 17th point, rho %12.9f up %12.9f dn %12.9f, |grad rho| up %12.9f dn %12.9f\n", rho[igrid], rho[igrid+DMnd], rho[igrid+2*DMnd], gradRhoLen[igrid], gradRhoLen[igrid+DMnd]);
        //     printf("rank 0 17th point, ecLinear %12.9f, VcLinear %12.9f %12.9f\n", pSPARC->vdWDFecLinear[igrid], pSPARC->vdWDFVcLinear[igrid], pSPARC->vdWDFVcLinear[DMnd + igrid]);
        //     printf("rank 0 17th point, kfResult_up %12.9f, FsResult_up %12.9f, kfResult_dn %12.9f, FsResult_dn %12.9f\n", kfResult_up, FsResult_up, kfResult_dn, FsResult_dn);
        //     printf("rank 0 17th point, qx %12.9f, q0x_up %12.9f, q0x_dn %12.9f, dqxdrho_up %12.9f, dqxdrho_dn %12.9f\n", qx, q0x_up, q0x_dn, dqxdrho_up, dqxdrho_dn);
        //     printf("rank 0 17th point, qc %12.9f, dqcdrho_up %12.9f, dqcdrho_dn %12.9f\n", qc, dqcdrho_up, dqcdrho_dn);
        //     printf("rank 0 17th point, q0 %12.9f, dq0drho %12.9f %12.9f, dq0dgradrho %12.9f %12.9f\n", q0[igrid], dq0drho[igrid], dq0drho[DMnd + igrid], dq0dgradrho[igrid], dq0dgradrho[DMnd + igrid]);
        // }
    }
    free(saturateq0dq0dq_up);
    free(saturateq0dq0dq_dn);
    free(saturateq0dq0dq);
}

/**
 * @brief get the component of "model energy ratios" on every grid point, based on the computed q0. The ps array is its output
 */
void spline_interpolation(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double **ps = pSPARC->vdWDFps;
    double **dpdq0s = pSPARC->vdWDFdpdq0s; // used for solving potential
    double *qmesh = pSPARC->vdWDFqmesh;
    double *q0 = pSPARC->vdWDFq0;
    double **d2ydx2 = pSPARC->vdWDFd2Splineydx2;
    // at here ps[q1][i] is the component coefficient of q1th model energy ratio on ith grid point. In m, it is ps(i, q1)
    int igrid, lowerBound, upperBound, idx, q1;
    double dq, a, b, c, d, e, f;
    for (igrid = 0; igrid < DMnd; igrid++)
    { // the loop to be optimized
        lowerBound = 0;
        upperBound = nqs - 1;
        while (upperBound - lowerBound > 1)
        {
            idx = (upperBound + lowerBound) / 2;
            if (q0[igrid] > qmesh[idx])
            {
                lowerBound = idx;
            }
            else
            {
                upperBound = idx;
            }
        }
        dq = qmesh[upperBound] - qmesh[lowerBound];
        a = (qmesh[upperBound] - q0[igrid]) / dq;
        b = (q0[igrid] - qmesh[lowerBound]) / dq;
        c = (a * a * a - a) * (dq * dq) / 6.0;
        d = (b * b * b - b) * (dq * dq) / 6.0;
        e = (3.0 * a * a - 1.0) * dq / 6.0;
        f = (3.0 * b * b - 1.0) * dq / 6.0;
        for (q1 = 0; q1 < nqs; q1++)
        {
            double y[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // 20 at here is nqs.
            y[q1] = 1.0;
            ps[q1][igrid] = a * y[lowerBound] + b * y[upperBound];
            ps[q1][igrid] += c * d2ydx2[q1][lowerBound];
            ps[q1][igrid] += d * d2ydx2[q1][upperBound];
            dpdq0s[q1][igrid] = (y[upperBound] - y[lowerBound]) / dq - e * d2ydx2[q1][lowerBound] + f * d2ydx2[q1][upperBound];
        }
    }
    // verify the correctness of result
    // #ifdef DEBUG
    //     if ((pSPARC->countPotentialCalculate == 3) && (rank == size - 1)) {
    //         printf("vdWDF: rank %d, (%d, %d, %d)-(%d, %d, %d), in 3rd SCF ps[5][DMnd - 1] %.6e, dpdq0s[5][DMnd - 1] %.6e\n",
    //          rank, pSPARC->DMVertices[0], pSPARC->DMVertices[2], pSPARC->DMVertices[4], pSPARC->DMVertices[1], pSPARC->DMVertices[3], pSPARC->DMVertices[5],
    //          ps[5][DMnd - 1], dpdq0s[5][DMnd - 1]);
    //     }
    // #endif
}
/*
Functions above are related to computing q0 and p (spline interpolation) on the grid.
*/

/*
Functions below are related to parallel FFT
*/
// compute the coordinates of G-vectors in reciprocal space. They are sum of integres multiplying three primary reci lattice vectors
void compute_Gvectors(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;

    // generate the 3D grid in reciprocal space
    // Firstly, compute reciprocal lattice vectors
    int row, col, i, j, k;
    for (col = 0; col < 3; col++)
        pSPARC->lattice[0 * 3 + col] = pSPARC->LatUVec[0 * 3 + col] * pSPARC->range_x;
    for (col = 0; col < 3; col++)
        pSPARC->lattice[1 * 3 + col] = pSPARC->LatUVec[1 * 3 + col] * pSPARC->range_y;
    for (col = 0; col < 3; col++)
        pSPARC->lattice[2 * 3 + col] = pSPARC->LatUVec[2 * 3 + col] * pSPARC->range_z;
    double detLattice = 0.0;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            for (k = 0; k < 3; k++)
            {
                if (i != j && j != k && k != i)
                    detLattice += ((i - j) * (j - k) * (k - i) / 2) * pSPARC->lattice[3 * i] * pSPARC->lattice[3 * j + 1] * pSPARC->lattice[3 * k + 2];
            }
        }
    }
    pSPARC->detLattice = detLattice;
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            pSPARC->reciLattice[3 * j + i] = (pSPARC->lattice[3 * ((j + 1) % 3) + (i + 1) % 3] * pSPARC->lattice[3 * ((j + 2) % 3) + (i + 2) % 3] - pSPARC->lattice[3 * ((j + 1) % 3) + (i + 2) % 3] * pSPARC->lattice[3 * ((j + 2) % 3) + (i + 1) % 3]) / detLattice * (2 * M_PI);
        }
    }
    // Secondly, compose the index of these lattice vectors and make a permutation. This part is moved to initialization functions
    // Thirdly, compute the coordinates of reciprocal lattice vectors, and the length of them
    int rigrid;
    double **reciLatticeGrid = pSPARC->vdWDFreciLatticeGrid;
    for (row = 0; row < 3; row++)
    { // cartesian direction
        for (rigrid = 0; rigrid < DMnd; rigrid++)
        { // reciprocal grid points // 000 100 200 ... 010 110 210 ... 001 101 201 ... 011 111 211 ...
            reciLatticeGrid[row][rigrid] = pSPARC->timeReciLattice[0][rigrid] * pSPARC->reciLattice[0 + row] + pSPARC->timeReciLattice[1][rigrid] * pSPARC->reciLattice[3 + row] + pSPARC->timeReciLattice[2][rigrid] * pSPARC->reciLattice[6 + row];
        }
    }
    double *reciLength = pSPARC->vdWDFreciLength;
    double largestLength = pSPARC->vdWDFnrpoints * pSPARC->vdWDFdk;
    int signReciPointFurther = 0;
    for (rigrid = 0; rigrid < DMnd; rigrid++)
    {
        reciLength[rigrid] = sqrt(reciLatticeGrid[0][rigrid] * reciLatticeGrid[0][rigrid] + reciLatticeGrid[1][rigrid] * reciLatticeGrid[1][rigrid] + reciLatticeGrid[2][rigrid] * reciLatticeGrid[2][rigrid]);
        if (reciLength[rigrid] > largestLength)
        {
            signReciPointFurther = 1;
        }
    }
    if ((signReciPointFurther == 1) && (rank == 0))
        printf("WARNING: there is reciprocal grid point further than largest allowed distance (%.e) from center!\n", largestLength); // It can be optimized

    // // For debugging
    // #ifdef DEBUG
    // if ((pSPARC->countPotentialCalculate == 3) && (rank == size - 1)) { // only output result in 1st step
    //     printf("vdWDF: rank %d, 2nd reci point (%d %d %d) %12.6e %12.6e %12.6e\n", rank, pSPARC->timeReciLattice[0][1], pSPARC->timeReciLattice[1][1], pSPARC->timeReciLattice[2][1],
    //         reciLatticeGrid[0][1], reciLatticeGrid[1][1], reciLatticeGrid[2][1]);
    // }
    // #endif
}



void parallel_FFT(double _Complex *inputDataRealSpace, double _Complex *outputDataReciSpace, int *gridsizes, int DMnz, int q, MPI_Comm zAxisComm)
{
#if defined(USE_MKL) // use MKL CDFT
    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);

    // initializa parallel FFT
    DFTI_DESCRIPTOR_DM_HANDLE desc = NULL;
    MKL_LONG localArrayLength;
    MKL_LONG dim_sizes[3] = {gridsizes[2], gridsizes[1], gridsizes[0]};
    DftiCreateDescriptorDM(zAxisComm, &desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &localArrayLength);
    // MKL_LONG lengthK, startK;
    // DftiGetValueDM(desc, CDFT_LOCAL_NX, &lengthK);
    // DftiGetValueDM(desc, CDFT_LOCAL_X_START, &startK);
    // int lengthKint = lengthK;
    // int startKint = startK;

    // compose FFT input to prevent the inconsistency division on Z axis
    double _Complex *FFTInput = (double _Complex *)malloc(sizeof(double _Complex) * (localArrayLength));
    double _Complex *FFTOutput = (double _Complex *)malloc(sizeof(double _Complex) * (localArrayLength));
    assert(FFTInput != NULL);
    assert(FFTOutput != NULL);

    // the decomposition of space in zAxisComm should be consistent with the data distribution designated by DFT module. DMnz == lengthK
    // Relative code can be found in function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    int numElesInGlobalPlane = gridsizes[0] * gridsizes[1];
    memcpy(FFTInput, inputDataRealSpace, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);

    // make parallel FFT
    /* Set that we want out-of-place transform (default is DFTI_INPLACE) */
    DftiSetValueDM(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    /* Commit descriptor, calculate FFT, free descriptor */
    DftiCommitDescriptorDM(desc);
    DftiComputeForwardDM(desc, FFTInput, FFTOutput);

    // the decomposition of space in zAxisComm should be consistent with the data distribution designated by DFT module. DMnz == lengthK
    // Relative code can be found in function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    memcpy(outputDataReciSpace, FFTOutput, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);

    DftiFreeDescriptorDM(&desc);
    free(FFTInput);
    free(FFTOutput);

#elif defined(USE_FFTW) // use FFTW if MKL is not used

    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);

    // initializa parallel FFT
    const ptrdiff_t N0 = gridsizes[2], N1 = gridsizes[1], N2 = gridsizes[0]; // N0: z; N1:Y; N2:x
    fftw_plan plan;
    fftw_complex *FFTInput, *FFTOutput;
    ptrdiff_t localArrayLength, lengthK, startK;
    fftw_mpi_init();
    /* get local data size and allocate */
    localArrayLength = fftw_mpi_local_size_3d(N0, N1, N2, zAxisComm, &lengthK, &startK);
    FFTInput = fftw_alloc_complex(localArrayLength);
    FFTOutput = fftw_alloc_complex(localArrayLength);
    // // compose FFT input to prevent the inconsistency division on Z axis
    // int lengthKint = lengthK;
    // int startKint = startK;
    // printf("rank %d, DMnz %d\n", rank, DMnz);
    // if (rank == 0) {
    //     for (int i = 0; i < sizeZcomm; i++) {
    //         printf("rank %d, startK %d, lengthK %d\n", i, allStartK[i], allLengthK[i]);
    //     }
    // }

    // the decomposition of space in zAxisComm should be consistent with the DFT module. DMnz == lengthK
    // Relative code can be found in function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    int numElesInGlobalPlane = gridsizes[0] * gridsizes[1];
    memcpy(FFTInput, inputDataRealSpace, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);

    /* create plan for in-place forward DFT */
    plan = fftw_mpi_plan_dft_3d(N0, N1, N2, FFTInput, FFTOutput, zAxisComm, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    // // compose FFT output to prevent the inconsistency division on Z axis

    // the decomposition of space in zAxisComm is consistent with the DFT module. Relative code can be found in 
    // function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    memcpy(outputDataReciSpace, FFTOutput, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);

    fftw_free(FFTInput);
    fftw_free(FFTOutput);
    fftw_destroy_plan(plan);
    fftw_mpi_cleanup();

#endif
}
/*
Functions above are related to parallel FFT
*/

/*
Functions below are related to generating thetas (ps*rho) and integrating energy.
*/
void theta_generate_FT(SPARC_OBJ *pSPARC, double *rho)
{ // solve thetas (p(q)*rho) in reciprocal space
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double **ps = pSPARC->vdWDFps;

    // compute theta vectors in real space, then FFT
    int nnr = pSPARC->Nd;
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int phiDims[3];
    phiDims[0] = pSPARC->npNdx_phi;
    phiDims[1] = pSPARC->npNdy_phi;
    phiDims[2] = pSPARC->npNdz_phi;
    int *zAxisDims = pSPARC->newzAxisDims;
    double *theta = (double *)malloc(sizeof(double) * DMnd); // save theta functions in real space
    double *thetaFTreal = (double *)malloc(sizeof(double) * DMnd);
    double *thetaFTimag = (double *)malloc(sizeof(double) * DMnd);
    double _Complex **thetaFTs = pSPARC->vdWDFthetaFTs;
    double *gatheredTheta = NULL;
    double _Complex *gatheredThetaCompl = NULL;
    double _Complex *gatheredThetaFFT = NULL;
    double *gatheredThetaFFT_real = NULL;
    double *gatheredThetaFFT_imag = NULL;
    int igrid, rigrid;
    int FFTrank = -1; int FFTsize = -1; int FFTDMnz = -1;
    if (pSPARC->zAxisComm != MPI_COMM_NULL)
    { // the processors on z axis (0, 0, z) receive the theta vectors from all other processors (x, y, z) on its z plane
        // printf("rank %d. pSPARC->zAxisComm not NULL!\n", rank);
        MPI_Comm_rank(pSPARC->zAxisComm, &FFTrank);
        MPI_Comm_size(pSPARC->zAxisComm, &FFTsize);
        FFTDMnz = pSPARC->zAxisVertices[5] - pSPARC->zAxisVertices[4] + 1;
        gatheredTheta = (double *)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gatheredThetaCompl = (double _Complex *)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gatheredThetaFFT = (double _Complex *)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gatheredThetaFFT_real = (double *)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gatheredThetaFFT_imag = (double *)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        assert(gatheredTheta != NULL);
        assert(gatheredThetaCompl != NULL);
        assert(gatheredThetaFFT != NULL);
        assert(gatheredThetaFFT_real != NULL);
        assert(gatheredThetaFFT_imag != NULL);
    }
    
    for (int q1 = 0; q1 < nqs; q1++)
    {
        for (igrid = 0; igrid < DMnd; igrid++)
        {
            theta[igrid] = ps[q1][igrid] * rho[igrid]; // compute theta vectors, ps*rho
        }
        D2D_AnyDMVert(&(pSPARC->gatherThetasSender), &(pSPARC->gatherThetasRecvr), gridsizes,
            pSPARC->DMVertices, theta,
            pSPARC->zAxisVertices, gatheredTheta,
            pSPARC->dmcomm_phi, phiDims, pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi);
        // printf("rank %d. D2D for q1 %d finished!\n", rank, q1);
        if (pSPARC->zAxisComm != MPI_COMM_NULL)
        { // the processors on z axis (0, 0, z) receive the theta vectors from all other processors (x, y, z) on its z plane
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * FFTDMnz; rigrid++)
            {
                gatheredThetaCompl[rigrid] = (double _Complex)gatheredTheta[rigrid];
            }
            parallel_FFT(gatheredThetaCompl, gatheredThetaFFT, gridsizes, FFTDMnz, q1, pSPARC->zAxisComm);
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * FFTDMnz; rigrid++)
            {
                gatheredThetaFFT_real[rigrid] = creal(gatheredThetaFFT[rigrid]);
                gatheredThetaFFT_imag[rigrid] = cimag(gatheredThetaFFT[rigrid]);
            }
        }
        D2D_AnyDMVert(&(pSPARC->scatterThetasSender), &(pSPARC->scatterThetasRecvr), gridsizes, // scatter the real part of theta results from the processors on z axis (0, 0, z) to all other processors
            pSPARC->zAxisVertices, gatheredThetaFFT_real,
            pSPARC->DMVertices, thetaFTreal,
            pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi, phiDims, pSPARC->dmcomm_phi);
        D2D_AnyDMVert(&(pSPARC->scatterThetasSender), &(pSPARC->scatterThetasRecvr), gridsizes, // scatter the imaginary part of theta results from the processors on z axis (0, 0, z) to all other processors
            pSPARC->zAxisVertices, gatheredThetaFFT_imag,
            pSPARC->DMVertices, thetaFTimag,
            pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi, phiDims, pSPARC->dmcomm_phi);
        for (rigrid = 0; rigrid < DMnd; rigrid++)
        {
            thetaFTs[q1][rigrid] = thetaFTreal[rigrid] + thetaFTimag[rigrid] * I;
            thetaFTs[q1][rigrid] /= nnr;
        }
        // if ((pSPARC->countPotentialCalculate == 3) && (q1 == 5) && (rank == size - 1)) { // only output result in 3rd SCF
        // 	int localIndex1D = domain_index1D(1, 1, 1, DMnx, DMny, DMnz);
        //     printf("rank %d, in 3rd SCF thetaFTs[5][(1, 1, 1)]=globalthetaFTs[5][(%d, %d, %d)] = %.5e + i%.5e\n", rank,
        //     	pSPARC->DMVertices[0] + 1, pSPARC->DMVertices[2] + 1, pSPARC->DMVertices[4] + 1,
        //         creal(thetaFTs[q1][localIndex1D]), cimag(thetaFTs[q1][localIndex1D]));
        // }
    }

    free(theta);
    free(thetaFTreal);
    free(thetaFTimag);
    if (pSPARC->zAxisComm != MPI_COMM_NULL)
    { // the processors on z axis (0, 0, z) receive the theta vectors from all other processors (x, y, z) on its z plane
        free(gatheredTheta);
        free(gatheredThetaCompl);
        free(gatheredThetaFFT);
        free(gatheredThetaFFT_real);
        free(gatheredThetaFFT_imag);
    }
}

/**
 * @brief compute the value of all kernel functions (210 functions for all model energy ratio pairs) on every reciprocal grid point
 * based on the distance between the reciprocal point and the center O
 */
void interpolate_kernel(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    double *reciLength = pSPARC->vdWDFreciLength;
    double **kernelReciPoints = pSPARC->vdWDFkernelReciPoints;
    double **kernelPhi = pSPARC->vdWDFkernelPhi;
    double **d2Phidk2 = pSPARC->vdWDFd2Phidk2;
    double dk = pSPARC->vdWDFdk;
    double largestLength = pSPARC->vdWDFnrpoints * dk;
    int rigrid, timeOfdk, q1, q2, qpair;
    double a, b, c, d;
    for (rigrid = 0; rigrid < DMnd; rigrid++)
    {
        if (reciLength[rigrid] > largestLength)
            continue;
        timeOfdk = (int)floor(reciLength[rigrid] / dk) + 1;
        a = (dk * ((timeOfdk - 1.0) + 1.0) - reciLength[rigrid]) / dk;
        b = (reciLength[rigrid] - dk * (timeOfdk - 1.0)) / dk;
        c = (a * a - 1.0) * a * dk * dk / 6.0;
        d = (b * b - 1.0) * b * dk * dk / 6.0;
        for (q1 = 0; q1 < nqs; q1++)
        {
            for (q2 = 0; q2 < q1 + 1; q2++)
            {
                qpair = kernel_label(q1, q2, nqs);
                kernelReciPoints[qpair][rigrid] = a * kernelPhi[qpair][timeOfdk - 1] + b * kernelPhi[qpair][timeOfdk] + (c * d2Phidk2[qpair][timeOfdk - 1] + d * d2Phidk2[qpair][timeOfdk]);
            }
        }
    }
}

/*
Functions below are two main function of vdWDF: for energy and potential
*/
void vdWDF_energy(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int nqs = pSPARC->vdWDFnqs;
    interpolate_kernel(pSPARC); // Optimize it?
    double **kernelReciPoints = pSPARC->vdWDFkernelReciPoints;
    double _Complex **thetaFTs = pSPARC->vdWDFthetaFTs;
    double _Complex **uFTs = pSPARC->vdWDFuFTs;
    double _Complex vdWenergyLocal = 0.0;
    int q1, q2, qpair, rigrid;

    // compose u vectors in reciprocal space
    for (q1 = 0; q1 < nqs; q1++)
    {
        for (rigrid = 0; rigrid < DMnd; rigrid++)
        { // initialization
            uFTs[q1][rigrid] = 0.0;
        }
    }
    for (q2 = 0; q2 < nqs; q2++)
    { // compose vector u s in reciprocal space
        for (q1 = 0; q1 < nqs; q1++)
        { // loop over q2 and q1 (model energy ratio pairs)
            qpair = kernel_label(q1, q2, nqs);
            for (rigrid = 0; rigrid < DMnd; rigrid++)
            {
                uFTs[q2][rigrid] += kernelReciPoints[qpair][rigrid] * thetaFTs[q1][rigrid]; // like the previous ps array, uFTs[q2][rigrid] at here is u(rigrid, q2) in m
            }
        }
    }
    // if ((pSPARC->countPotentialCalculate == 3) && (rank == size - 1)) { // only output result in 3rd step, for debugging
    // 	int localIndex1D = domain_index1D(1, 1, 1, DMnx, DMny, DMnz);
    //     printf("rank %d, in 3rd SCF uFTs[5][(1, 1, 1)]=globaluFTs[5][(%d, %d, %d)] = %.5e + i%.5e\n", rank,
    //     	pSPARC->DMVertices[0] + 1, pSPARC->DMVertices[2] + 1, pSPARC->DMVertices[4] + 1,
    //         creal(uFTs[5][localIndex1D]), cimag(uFTs[5][localIndex1D]));
    // }
    // integrate for vdWDF energy, Soler's paper (8)
    for (q2 = 0; q2 < nqs; q2++)
    { // loop over all model energy ratio q
        for (rigrid = 0; rigrid < DMnd; rigrid++)
        {                                                                    // loop over all riciprocal grid points
            vdWenergyLocal += conj(thetaFTs[q2][rigrid]) * uFTs[q2][rigrid]; // point multiplication
        }
    }
    double vdWenergyLocalReal = creal(vdWenergyLocal) * 0.5 * pSPARC->detLattice;
    MPI_Allreduce(&vdWenergyLocalReal, &(pSPARC->vdWDFenergy), 1, MPI_DOUBLE,
                  MPI_SUM, pSPARC->dmcomm_phi);
#ifdef DEBUG
    if (rank == size - 1)
    {
        printf("vdWDF: at %d countPotentialCalculate vdWDF energy is %.5e\n", pSPARC->countPotentialCalculate, pSPARC->vdWDFenergy);
    }
#endif
}
/*
Functions above are related to generating thetas (ps*rho) and integrating energy.
*/

/*
Functions below are related to generating u vectors, transforming them to real space and computing vdW-DF potential.
*/
void parallel_iFFT(double _Complex *inputDataReciSpace, double _Complex *outputDataRealSpace, int *gridsizes, int DMnz, int q, MPI_Comm zAxisComm)
{
#if defined(USE_MKL) // use MKL CDFT
    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);
    // initializa parallel iFFT
    DFTI_DESCRIPTOR_DM_HANDLE desc = NULL;
    MKL_LONG localArrayLength;
    MKL_LONG dim_sizes[3] = {gridsizes[2], gridsizes[1], gridsizes[0]};
    DftiCreateDescriptorDM(zAxisComm, &desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &localArrayLength);
    // MKL_LONG lengthK, startK;
    // DftiGetValueDM(desc, CDFT_LOCAL_NX, &lengthK);
    // DftiGetValueDM(desc, CDFT_LOCAL_X_START, &startK);
    // int lengthKint = lengthK;
    // int startKint = startK;
    // compose iFFT input to prevent the inconsistency division on Z axis
    double _Complex *iFFTInput = (double _Complex *)malloc(sizeof(double _Complex) * (localArrayLength));
    double _Complex *iFFTOutput = (double _Complex *)malloc(sizeof(double _Complex) * (localArrayLength));
    assert(iFFTInput != NULL);
    assert(iFFTOutput != NULL);

    // the decomposition of space in zAxisComm should be consistent with the data distribution designated by DFT module. DMnz == lengthK
    // Relative code can be found in function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    int numElesInGlobalPlane = gridsizes[0] * gridsizes[1];
    memcpy(iFFTInput, inputDataReciSpace, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);
    // make parallel FFT
    /* Set that we want out-of-place transform (default is DFTI_INPLACE) */
    DftiSetValueDM(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    /* Commit descriptor, calculate FFT, free descriptor */
    DftiCommitDescriptorDM(desc);
    DftiComputeBackwardDM(desc, iFFTInput, iFFTOutput);

    // the decomposition of space in zAxisComm should be consistent with the data distribution designated by DFT module. DMnz == lengthK
    // Relative code can be found in function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    memcpy(outputDataRealSpace, iFFTOutput, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);

    DftiFreeDescriptorDM(&desc);
    free(iFFTInput);
    free(iFFTOutput);

#elif defined(USE_FFTW) // use FFTW if MKL is not used

    int rank;
    MPI_Comm_rank(zAxisComm, &rank);
    int sizeZcomm;
    MPI_Comm_size(zAxisComm, &sizeZcomm);
    // initializa parallel iFFT
    const ptrdiff_t N0 = gridsizes[2], N1 = gridsizes[1], N2 = gridsizes[0]; // N0: z; N1:Y; N2:x
    fftw_plan plan;
    fftw_complex *iFFTInput, *iFFTOutput;
    ptrdiff_t localArrayLength, lengthK, startK;
    fftw_mpi_init();
    /* get local data size and allocate */
    localArrayLength = fftw_mpi_local_size_3d(N0, N1, N2, zAxisComm, &lengthK, &startK);
    iFFTInput = fftw_alloc_complex(localArrayLength);
    iFFTOutput = fftw_alloc_complex(localArrayLength);
    // compose iFFT input to prevent the inconsistency division on Z axis
    // int lengthKint = lengthK;
    // int startKint = startK;

    // the decomposition of space in zAxisComm should be consistent with the data distribution designated by DFT module. DMnz == lengthK
    // Relative code can be found in function vdWDF_Setup_Comms, file vdWDFparallelization.c.
    int numElesInGlobalPlane = gridsizes[0] * gridsizes[1];
    memcpy(iFFTInput, inputDataReciSpace, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);
    /* create plan for out-place backward DFT */
    plan = fftw_mpi_plan_dft_3d(N0, N1, N2, iFFTInput, iFFTOutput, zAxisComm, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);


    memcpy(outputDataRealSpace, iFFTOutput, sizeof(double _Complex) * numElesInGlobalPlane * DMnz);

    fftw_free(iFFTInput);
    fftw_free(iFFTOutput);
    fftw_destroy_plan(plan);
    fftw_mpi_cleanup();

#endif
}

// compute u vectors in reciprocal space, then transform them back to real space by iFFT
// Soler's paper (11)-(12)
void u_generate_iFT(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(pSPARC->dmcomm_phi, &rank);
    int size;
    MPI_Comm_size(pSPARC->dmcomm_phi, &size);

    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    int gridsizes[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    int phiDims[3];
    phiDims[0] = pSPARC->npNdx_phi;
    phiDims[1] = pSPARC->npNdy_phi;
    phiDims[2] = pSPARC->npNdz_phi;
    int *zAxisDims = pSPARC->newzAxisDims;
    int nqs = pSPARC->vdWDFnqs;
    double _Complex **uFTs = pSPARC->vdWDFuFTs;
    double **u = pSPARC->vdWDFu;
    int rigrid, q1;
    double *uFTreal = (double *)malloc(sizeof(double) * DMnd);
    double *uFTimag = (double *)malloc(sizeof(double) * DMnd);
    double *gathereduFT_real = NULL;
    double *gathereduFT_imag = NULL;
    double _Complex *gathereduFT = NULL;
    double _Complex *gathereduCompl = NULL;
    double *gatheredu = NULL;
    int FFTrank = -1; int FFTsize = -1; int FFTDMnz = -1;
    if (pSPARC->zAxisComm != MPI_COMM_NULL)
    {
        MPI_Comm_rank(pSPARC->zAxisComm, &FFTrank);
        MPI_Comm_size(pSPARC->zAxisComm, &FFTsize);
        FFTDMnz = pSPARC->zAxisVertices[5] - pSPARC->zAxisVertices[4] + 1;
        gathereduFT_real = (double *)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gathereduFT_imag = (double *)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gathereduFT = (double _Complex *)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gathereduCompl = (double _Complex *)malloc(sizeof(double _Complex) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        gatheredu = (double *)malloc(sizeof(double) * gridsizes[0] * gridsizes[1] * FFTDMnz);
        assert(gathereduFT_real != NULL);
        assert(gathereduFT_imag != NULL);
        assert(gathereduFT != NULL);
        assert(gathereduCompl != NULL);
        assert(gatheredu != NULL);
    }

    for (q1 = 0; q1 < nqs; q1++)
    {
        for (rigrid = 0; rigrid < DMnd; rigrid++)
        {
            uFTreal[rigrid] = creal(uFTs[q1][rigrid]);
            uFTimag[rigrid] = cimag(uFTs[q1][rigrid]);
        }
        D2D_AnyDMVert(&(pSPARC->gatherThetasSender), &(pSPARC->gatherThetasRecvr), gridsizes,
            pSPARC->DMVertices, uFTreal,
            pSPARC->zAxisVertices, gathereduFT_real,
            pSPARC->dmcomm_phi, phiDims, pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi);
        D2D_AnyDMVert(&(pSPARC->gatherThetasSender), &(pSPARC->gatherThetasRecvr), gridsizes,
            pSPARC->DMVertices, uFTimag,
            pSPARC->zAxisVertices, gathereduFT_imag,
            pSPARC->dmcomm_phi, phiDims, pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi);
        if (pSPARC->zAxisComm != MPI_COMM_NULL)
        {
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * FFTDMnz; rigrid++)
            {
                gathereduFT[rigrid] = gathereduFT_real[rigrid] + gathereduFT_imag[rigrid] * I;
            }
            parallel_iFFT(gathereduFT, gathereduCompl, gridsizes, FFTDMnz, q1, pSPARC->zAxisComm);
            for (rigrid = 0; rigrid < gridsizes[0] * gridsizes[1] * FFTDMnz; rigrid++)
            {
                gatheredu[rigrid] = creal(gathereduCompl[rigrid]); // MKL original iFFT functions do not divide the iFFT results by N
            }
        }
        D2D_AnyDMVert(&(pSPARC->scatterThetasSender), &(pSPARC->scatterThetasRecvr), gridsizes, // scatter the real part of theta results from the processors on z axis (0, 0, z) to all other processors
            pSPARC->zAxisVertices, gatheredu,
            pSPARC->DMVertices, u[q1],
            pSPARC->zAxisComm, zAxisDims, pSPARC->dmcomm_phi, phiDims, pSPARC->dmcomm_phi);
        // if ((pSPARC->countPotentialCalculate == 0) && (q1 == 5) && (rank == size - 1)) { // only output result in 1st step
        //     int localIndex1D = domain_index1D(1, 1, 1, DMnx, DMny, DMnz);
        //     fprintf(pSPARC->vdWDFOutput, "rank %d, at 1st SCF u[5][(1, 1, 1)]=globalu[5][(%d, %d, %d)] = %.5e\n", rank,
        //         pSPARC->DMVertices[0] + 1, pSPARC->DMVertices[2] + 1, pSPARC->DMVertices[4] + 1,
        //         u[q1][localIndex1D]);
        // }
    }

    free(uFTreal);
    free(uFTimag);
    if (pSPARC->zAxisComm != MPI_COMM_NULL)
    {
        free(gathereduFT_real);
        free(gathereduFT_imag);
        free(gathereduFT);
        free(gathereduCompl);
        free(gatheredu);
    }
}

void vdWDF_potential(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nqs = pSPARC->vdWDFnqs;
    int DMnx, DMny, DMnz, DMnd;
    DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
    DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
    DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
    DMnd = DMnx * DMny * DMnz;
    double **u = pSPARC->vdWDFu;
    double **ps = pSPARC->vdWDFps;
    double **dpdq0s = pSPARC->vdWDFdpdq0s;
    double *dq0drho = pSPARC->vdWDFdq0drho;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double *potential = pSPARC->vdWDFpotential;
    double *hPrefactor = (double *)malloc(sizeof(double) * DMnd);
    int igrid, q1;
    for (igrid = 0; igrid < DMnd; igrid++)
    { // initialization
        potential[igrid] = 0.0;
        hPrefactor[igrid] = 0.0;
    }
    for (q1 = 0; q1 < nqs; q1++)
    {
        for (igrid = 0; igrid < DMnd; igrid++)
        {
            potential[igrid] += u[q1][igrid] * (ps[q1][igrid] + dpdq0s[q1][igrid] * dq0drho[igrid]); // First term of Soler's paper (10)
            hPrefactor[igrid] += u[q1][igrid] * dpdq0s[q1][igrid] * dq0dgradrho[igrid];              // Second term of Soler's paper (10), to multiply diffential coefficient matrix
        }
    }
    int direction;
    double *h = (double *)malloc(sizeof(double) * DMnd);
    double *Dh = (double *)malloc(sizeof(double) * DMnd);
    double **Drho = pSPARC->Drho;
    double *gradRhoLen = pSPARC->gradRhoLen;

    for (direction = 0; direction < 3; direction++)
    {
        for (igrid = 0; igrid < DMnd; igrid++)
        {
            h[igrid] = hPrefactor[igrid] * Drho[direction][igrid];
            if (gradRhoLen[igrid] > 1e-15)
            { // gradRhoLen > 0.0
                h[igrid] /= gradRhoLen[igrid];
            }
        }
        if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20)
        { // non-orthogonal cell
            double *Dh_1, *Dh_2, *Dh_3;
            Dh_1 = (double *)malloc(sizeof(double) * DMnd);
            Dh_2 = (double *)malloc(sizeof(double) * DMnd);
            Dh_3 = (double *)malloc(sizeof(double) * DMnd);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, DMnd, Dh_1, DMnd, 0, pSPARC->dmcomm_phi); // nonCart direction
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, DMnd, Dh_2, DMnd, 1, pSPARC->dmcomm_phi);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, DMnd, Dh_3, DMnd, 2, pSPARC->dmcomm_phi);
            for (igrid = 0; igrid < DMnd; igrid++)
            {
                Dh[igrid] = pSPARC->gradT[0 + direction] * Dh_1[igrid] + pSPARC->gradT[3 + direction] * Dh_2[igrid] + pSPARC->gradT[6 + direction] * Dh_3[igrid];
            }
            free(Dh_1);
            free(Dh_2);
            free(Dh_3);
        }
        else
        {                                                                                                         // orthogonal cell
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, h, DMnd, Dh, DMnd, direction, pSPARC->dmcomm_phi); // Soler's paper (10), diffential coefficient matrix
        }
        for (igrid = 0; igrid < DMnd; igrid++)
        {
            potential[igrid] -= Dh[igrid];
        }
    }
    for (igrid = 0; igrid < DMnd; igrid++)
    { // add vdWDF potential to the potential of system
        pSPARC->XCPotential[igrid] += potential[igrid];
    }

    free(hPrefactor);
    free(h);
    free(Dh);
}

void spin_vdWDF_potential(SPARC_OBJ *pSPARC)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nqs = pSPARC->vdWDFnqs;
    int DMnd = pSPARC->Nd_d;
    double **u = pSPARC->vdWDFu;
    double **ps = pSPARC->vdWDFps;
    double **dpdq0s = pSPARC->vdWDFdpdq0s;
    double *dq0drho = pSPARC->vdWDFdq0drho;
    double *dq0dgradrho = pSPARC->vdWDFdq0dgradrho;
    double *potential = pSPARC->vdWDFpotential;
    double *hPrefactor = (double *)malloc(2 * DMnd * sizeof(double));
    int i, igrid, q1;
    for (i = 0; i < 2 * DMnd; i++)
    { // initialization
        potential[i] = 0.0;
        hPrefactor[i] = 0.0;
    }
    for (q1 = 0; q1 < nqs; q1++)
    {
        for (i = 0; i < 2 * DMnd; i++)
        {
            igrid = i % DMnd;
            potential[i] += u[q1][igrid] * (ps[q1][igrid] + dpdq0s[q1][igrid] * dq0drho[i]); // First term of Soler's paper (10)
            hPrefactor[i] += u[q1][igrid] * dpdq0s[q1][igrid] * dq0dgradrho[i];              // Second term of Soler's paper (10), to multiply diffential coefficient matrix
        }
    }
    int direction;
    double *h = (double *)malloc(2 * DMnd * sizeof(double));
    double *Dh = (double *)malloc(2 * DMnd * sizeof(double));
    double **Drho = pSPARC->Drho;
    double *gradRhoLen = pSPARC->gradRhoLen;
    double *Dh_1 = (double *)malloc(2 * DMnd * sizeof(double));
    double *Dh_2 = (double *)malloc(2 * DMnd * sizeof(double));
    double *Dh_3 = (double *)malloc(2 * DMnd * sizeof(double));
    for (direction = 0; direction < 3; direction++)
    {
        for (i = 0; i < 2 * DMnd; i++)
        {
            h[i] = hPrefactor[i] * Drho[direction][i];
            if (gradRhoLen[i] > 1e-15)
            { // gradRhoLen > 0.0
                h[i] /= gradRhoLen[i];
            }
        }
        if (pSPARC->cell_typ > 10 && pSPARC->cell_typ < 20)
        {                                                                                                   // non-orthogonal cell
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, h, DMnd, Dh_1, DMnd, 0, pSPARC->dmcomm_phi); // nonCart direction
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, h, DMnd, Dh_2, DMnd, 1, pSPARC->dmcomm_phi);
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, h, DMnd, Dh_3, DMnd, 2, pSPARC->dmcomm_phi);
            for (i = 0; i < 2 * DMnd; i++)
            {
                Dh[i] = pSPARC->gradT[0 + direction] * Dh_1[i] + pSPARC->gradT[3 + direction] * Dh_2[i] + pSPARC->gradT[6 + direction] * Dh_3[i];
            }
        }
        else
        {                                                                                                         // orthogonal cell
            Gradient_vectors_dir(pSPARC, DMnd, pSPARC->DMVertices, 2, 0.0, h, DMnd, Dh, DMnd, direction, pSPARC->dmcomm_phi); // Soler's paper (10), diffential coefficient matrix
        }
        for (i = 0; i < 2 * DMnd; i++)
        {
            potential[i] -= Dh[i];
        }
    }
    for (i = 0; i < 2 * DMnd; i++)
    { // add vdWDF potential to the potential of system
        pSPARC->XCPotential[i] += potential[i];
    }
    free(hPrefactor);
    free(h);
    free(Dh);
    free(Dh_1);
    free(Dh_2);
    free(Dh_3);
}

/*
Functions above are related to generating u vectors, transforming them to real space and computing vdW-DF potential.
*/

// The main function in the file
// The main function to be called by function Calculate_Vxc_GGA in exchangeCorrelation.c, compute the energy and potential of non-linear correlation part of vdW-DF
void Calculate_nonLinearCorr_E_V_vdWDF(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL)
    {
        return;
    }
    get_q0_Grid(pSPARC, rho);
    spline_interpolation(pSPARC); // get the component of "model energy ratios" on every grid point
    compute_Gvectors(pSPARC);
    theta_generate_FT(pSPARC, rho);
    vdWDF_energy(pSPARC);
    u_generate_iFT(pSPARC);
    vdWDF_potential(pSPARC);

    pSPARC->countPotentialCalculate++; // count the time of SCF. To output variables in 1st step. To be deleted in the future.
}

// The main function in the file, spin-polarized case
// The main function to be called by function Calculate_Vxc_GGA in exchangeCorrelation.c, compute the energy and potential of non-linear correlation part of vdW-DF
void Calculate_nonLinearCorr_E_V_SvdWDF(SPARC_OBJ *pSPARC, double *rho)
{
    if (pSPARC->dmcomm_phi == MPI_COMM_NULL)
    {
        return;
    }
    spin_get_q0_Grid(pSPARC, rho);

    spline_interpolation(pSPARC); // get the component of "model energy ratios" on every grid point
    compute_Gvectors(pSPARC);
    theta_generate_FT(pSPARC, rho);
    vdWDF_energy(pSPARC);
    u_generate_iFT(pSPARC); // functions above should be similar to spin-unpolarized case

    spin_vdWDF_potential(pSPARC);
    pSPARC->countPotentialCalculate++; // count the time of SCF. To output variables in 1st step. To be deleted in the future.
}

void Add_Exc_vdWDF(SPARC_OBJ *pSPARC)
{ // add vdW_DF energy into the total xc energy
    pSPARC->Exc += pSPARC->vdWDFenergy;
}

void find_folder_route(SPARC_OBJ *pSPARC, char *folderRoute) { // used for returning the folder route of the input files
    strncpy(folderRoute, pSPARC->filename, L_STRING);
    int indexEndFolderRoute = 0;
    while (folderRoute[indexEndFolderRoute] != '\0') {
        indexEndFolderRoute++;
    } // get the length of the char filename
    while ((indexEndFolderRoute > -1) && (folderRoute[indexEndFolderRoute] != '/')) {
        indexEndFolderRoute--;
    } // find the last '/'. If there is no '/', indexEndFolderRoute should be at the beginning
    folderRoute[indexEndFolderRoute + 1] = '\0'; // cut the string. Now it contains only the folder position
}

void print_variables(double *variable, char *outputFileName, int Nx, int Ny, int Nz) {
    printf("begin printing variable\n");
    FILE *outputFile = NULL;
    outputFile = fopen(outputFileName,"w");
    int xIndex, yIndex, zIndex;
    int globalIndex = 0;
    fprintf(outputFile, "%d %d %d\n", Nx, Ny, Nz);
    for (zIndex = 0; zIndex < Nz; zIndex++) {
        for (yIndex = 0; yIndex < Ny; yIndex++) {
            fprintf(outputFile, "%d %d\n", yIndex, zIndex);
            for (xIndex = 0; xIndex < Nx; xIndex++) {
                if ((xIndex + 1) % 4 == 0 || (xIndex == Nx - 1)) // new line every 4 values
                    fprintf(outputFile, "%12.9f\n", variable[globalIndex]);
                else
                    fprintf(outputFile, "%12.9f ", variable[globalIndex]);
                globalIndex++;
            }
        }
    }
    fclose(outputFile);
}