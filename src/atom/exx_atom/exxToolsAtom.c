#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif


#include "isddftAtom.h"
#include "exxToolsAtom.h"

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

/**
 * @brief Initialise and allocate memory for exact exchange variables
 */
void exx_initialization_atom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    // Change SCF tolerance to 1e-6 for appropriate outer loop convergence
    pSPARC_ATOM->SCF_tol = 1e-6;

    if (pSPARC_ATOM->MAXIT_FOCK < 0) pSPARC_ATOM->MAXIT_FOCK = 30;
    if (pSPARC_ATOM->MINIT_FOCK < 0) pSPARC_ATOM->MINIT_FOCK = 1;
    if (pSPARC_ATOM->TOL_FOCK < 0) pSPARC_ATOM->TOL_FOCK = 0.2*pSPARC_ATOM->SCF_tol;
    if (pSPARC_ATOM->TOL_SCF_INIT < 0) pSPARC_ATOM->TOL_SCF_INIT = max(10*pSPARC_ATOM->TOL_FOCK, 1e-3);

    int lmax = pSPARC_ATOM->max_l;
    int Nd = pSPARC_ATOM->Nd;
    pSPARC_ATOM->EXXL = (EXX_POT_OBJ_ATOM *)calloc((lmax + 1), sizeof(EXX_POT_OBJ_ATOM));
    for (int l = 0; l <= lmax; l++) {
        pSPARC_ATOM->EXXL[l].VexxUp = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
        pSPARC_ATOM->EXXL[l].VexxDw = NULL;
        if (pSPARC_ATOM->spinFlag) 
            pSPARC_ATOM->EXXL[l].VexxDw = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
    }
}

/**
 * @brief Calculate density matrix from the outer loop wavefunctions
 *        to check for convergence
 *        \sum_{nl} | \tilde{R}_{nl} >< \tilde{R}_{nl} | * g_{nl}
 */
void densityMatrix(SPARC_ATOM_OBJ *pSPARC_ATOM, double *denMat) {
    int Nd = pSPARC_ATOM->Nd;
    int nspinor = pSPARC_ATOM->nspinor;
    double *w = pSPARC_ATOM->w;
    double *r = pSPARC_ATOM->r;
    double *int_scale = pSPARC_ATOM->int_scale;
    int val_len = pSPARC_ATOM->val_len;
    int *occ = pSPARC_ATOM->occ;

    // Copy \tilde{R}_{nl} = r*R_{nl} into orbitals
    double *orbitals = (double *)malloc(nspinor*val_len*(Nd-1)*sizeof(double));
    for (int jj = 0; jj < val_len; jj++) {
        // Spin up part
        memcpy(orbitals+jj*(Nd-1),pSPARC_ATOM->orbitals+jj*(Nd-1),sizeof(double)*(Nd-1));
        // Spin dw part
        if (pSPARC_ATOM->spinFlag){
            memcpy(orbitals+jj*(Nd-1)+val_len*(Nd-1),pSPARC_ATOM->orbitals+jj*(Nd-1)+val_len*(Nd-1),sizeof(double)*(Nd-1));
        }
    }

    int lmin, lmax;
    lmin = pSPARC_ATOM->min_l; lmax = pSPARC_ATOM->max_l;
    memset(denMat, 0, (Nd-1)*(Nd-1)*sizeof(double));

    int jstart, jstop;
    double *orbital;
    double *wt_orbital = (double *)malloc(sizeof(double)*(Nd - 1));
    for (int i = lmin; i <= lmax; i++) {
        // store the start and stop indices
        if (i == 0) {
            jstart = 0;
            jstop = pSPARC_ATOM->lcount0;
        } else if (i == 1) {
            jstart = pSPARC_ATOM->lcount0;
            jstop = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1;
        } else if (i == 2) {
            jstart = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1;
            jstop = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1 + pSPARC_ATOM->lcount2;
        } else {
            jstart = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1 + pSPARC_ATOM->lcount2;
            jstop = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1 + pSPARC_ATOM->lcount2 + pSPARC_ATOM->lcount3;
        }

        for (int j = jstart; j < jstop; j++) {
            orbital = orbitals+j*(Nd - 1);
            // Calculate wt_orbital
            for (int jj = 0; jj < Nd - 1; jj++){
                wt_orbital[jj] = (w[jj+1]/int_scale[jj+1])*orbital[jj];
            }

            double *V = (double *)calloc((Nd-1)*(Nd-1),sizeof(double));

            // Outer product |orbital><wt_orbital| * occ[j]
            cblas_dger(CblasColMajor, Nd-1, Nd-1, occ[j], orbital, 1, wt_orbital, 1, V, Nd-1);

            for (int col = 0; col < Nd - 1; col++) {
                for (int row = 0; row < Nd - 1; row++) {
                    denMat[col*(Nd - 1) + row] += V[col * (Nd - 1) + row];
                }
            }

            free(V);
        }
    }

    if (pSPARC_ATOM->spinFlag) {
        for (int i = lmin; i <= lmax; i++) {
            // store the start and stop indices
            if (i == 0) {
                jstart = 0;
                jstop = pSPARC_ATOM->lcount0;
            } else if (i == 1) {
                jstart = pSPARC_ATOM->lcount0;
                jstop = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1;
            } else if (i == 2) {
                jstart = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1;
                jstop = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1 + pSPARC_ATOM->lcount2;
            } else {
                jstart = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1 + pSPARC_ATOM->lcount2;
                jstop = pSPARC_ATOM->lcount0 + pSPARC_ATOM->lcount1 + pSPARC_ATOM->lcount2 + pSPARC_ATOM->lcount3;
            }
            jstart += val_len; jstop += val_len;

            for (int j = jstart; j < jstop; j++) {
                orbital = orbitals+j*(Nd - 1);
                // Calculate wt_orbital
                for (int jj = 0; jj < Nd - 1; jj++){
                    wt_orbital[jj] = (w[jj+1]/int_scale[jj+1])*orbital[jj];
                }

                double *V = (double *)calloc((Nd-1)*(Nd-1),sizeof(double));

                // Outer product |orbital><wt_orbital| * occ[j]
                cblas_dger(CblasColMajor, Nd-1, Nd-1, occ[j], orbital, 1, wt_orbital, 1, V, Nd-1);

                for (int col = 0; col < Nd - 1; col++) {
                    for (int row = 0; row < Nd - 1; row++) {
                        denMat[col*(Nd - 1) + row] += V[col * (Nd - 1) + row];
                    }
                }

                free(V);
            }
        }
    }

    free(wt_orbital); free(orbitals);

}

/**
 * @brief Calculate Wigner coefficient
 */
double Wigner3j(int j1, int j2, int j) {
    int J = j1 + j2 + j;
    double W3j;
    int g;

    if (J % 2 == 1) {
        return 0.0;
    } else {
        g = J / 2;
        W3j = (g % 2 == 0) ? 1.0 : -1.0;
        W3j *= sqrt(factorial(2*g - 2*j1) * factorial(2*g - 2*j2) * factorial(2*g - 2*j) / factorial(2*g + 1));
        W3j *= factorial(g) / (factorial(g - j1) * factorial(g - j2) * factorial(g - j));

        return W3j;
    }
}

/**
 * @brief Calculate factorial of an integer
 */
double factorial(int n) {
    if (n == 0) return 1.0;
    double result = 1.0;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}