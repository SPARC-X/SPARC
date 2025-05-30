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
#include "exxPotentialEnergyAtom.h"

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

/**
 * @brief Calculate and store the exact exchange potential operator for
 *        each value of azimuthal quantum number, "l" in the system. 
 */
void exxOperatorAtom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    int lmin = pSPARC_ATOM->min_l;
    int lmax = pSPARC_ATOM->max_l;
    int Nd = pSPARC_ATOM->Nd;
    double spin;
    double *VexxL;

    for (int l = lmin; l <= lmax; l++) {
        spin = 0.5;
        VexxL = pSPARC_ATOM->EXXL[l].VexxUp;
        memset(VexxL, 0, (Nd-1)*(Nd-1)*sizeof(double));
        evaluateExxPotentialAtom(pSPARC_ATOM, l, spin, VexxL);

        if (pSPARC_ATOM->spinFlag) {
            spin = -0.5;
            VexxL = pSPARC_ATOM->EXXL[l].VexxDw;
            memset(VexxL, 0, (Nd-1)*(Nd-1)*sizeof(double));
            evaluateExxPotentialAtom(pSPARC_ATOM, l, spin, VexxL);
        }
    }
}

/**
 * @brief Calculate the exact exchange potential operator for a particular
 *        azimuthal quantum number, "l".
 *        Refer to Eq. 13 and 25 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void evaluateExxPotentialAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, int l, double spin, double *VexxL) {
    int Nd = pSPARC_ATOM->Nd;
    int nspinor = pSPARC_ATOM->nspinor;
    double *w = pSPARC_ATOM->w;
    double *r = pSPARC_ATOM->r;
    double *int_scale = pSPARC_ATOM->int_scale;
    int val_len = pSPARC_ATOM->val_len;
    int *occ = pSPARC_ATOM->occ;
    int *orbital_l = pSPARC_ATOM->orbital_l;

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

    int jstart, jstop;
    double *orbital;
    double wigner_const;
    double *wt_orbital = (double *)malloc(sizeof(double)*(Nd - 1));
    if (spin == 0.5) {
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

                double *V = (double *)calloc((Nd-1)*(Nd - 1), sizeof(double));
                for (int k = abs(l - i); k <= abs(l + i); k += 2) {
                    wigner_const = pow(Wigner3j(l, i, k), 2);
                    double *term = (double *)calloc((Nd-1)*(Nd-1),sizeof(double));
                    
                    // Outer product |orbital><wt_orbital| * wigner_const
                    cblas_dger(CblasColMajor, Nd-1, Nd-1, wigner_const, orbital, 1, wt_orbital, 1, term, Nd-1);

                    for (int cols = 0; cols < Nd - 1; cols++) {
                        for (int rows = 0; rows < Nd - 1; rows++) {
                            double ratio = pow(min(r[rows+1],r[cols+1]),k)/pow(max(r[rows+1],r[cols+1]),k+1);
                            term[cols * (Nd - 1) + rows] *= ratio;
                            V[cols * (Nd - 1) + rows] += term[cols * (Nd - 1) + rows];
                        }
                    }

                    free(term);
                }

                for (int col = 0; col < Nd - 1; col++) {
                    for (int row = 0; row < Nd - 1; row++) {
                        VexxL[col*(Nd - 1) + row] += occ[j]*V[col * (Nd - 1) + row];
                    }
                }

                free(V);
            }
        }

        if (pSPARC_ATOM->spinFlag == 0) {
            cblas_dscal((Nd-1)*(Nd - 1), -0.5, VexxL, 1);
        } else {
            cblas_dscal((Nd-1)*(Nd - 1), -1, VexxL, 1);
        }
    } else {
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

                double *V = (double *)calloc((Nd-1)*(Nd - 1), sizeof(double));
                for (int k = abs(l - i); k <= abs(l + i); k += 2) {
                    wigner_const = pow(Wigner3j(l, i, k), 2);
                    double *term = (double *)calloc((Nd-1)*(Nd-1),sizeof(double));
                    
                    // Outer product |orbital><wt_orbital| * wigner_const
                    cblas_dger(CblasColMajor, Nd-1, Nd-1, wigner_const, orbital, 1, wt_orbital, 1, term, Nd-1);

                    for (int cols = 0; cols < Nd - 1; cols++) {
                        for (int rows = 0; rows < Nd - 1; rows++) {
                            double ratio = pow(min(r[rows+1],r[cols+1]),k)/pow(max(r[rows+1],r[cols+1]),k+1);
                            term[cols * (Nd - 1) + rows] *= ratio;
                            V[cols * (Nd - 1) + rows] += term[cols * (Nd - 1) + rows];
                        }
                    }

                    free(term);
                }

                for (int col = 0; col < Nd - 1; col++) {
                    for (int row = 0; row < Nd - 1; row++) {
                        VexxL[col*(Nd - 1) + row] += occ[j]*V[col * (Nd - 1) + row];
                    }
                }

                free(V);
            }
        }

        cblas_dscal((Nd-1)*(Nd - 1), -1, VexxL, 1);
    }

    free(wt_orbital); free(orbitals);
}

/**
 * @brief Calculate the exact exchange energy for the atom 
 *        Refer to Eq. 12 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void evaluateExxEnergyAtom(SPARC_ATOM_OBJ *pSPARC_ATOM, double *Exx) {
    // Initialise Exx
    *Exx = 0.0;
    
    int Nd = pSPARC_ATOM->Nd;
    int nspinor = pSPARC_ATOM->nspinor;
    double *w = pSPARC_ATOM->w;
    double *r = pSPARC_ATOM->r;
    double *int_scale = pSPARC_ATOM->int_scale;
    int val_len = pSPARC_ATOM->val_len;
    int *occ = pSPARC_ATOM->occ;
    int *orbital_l = pSPARC_ATOM->orbital_l;

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

    int jstart, jstop;
    double *orbital;
    double wigner_const;
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

        double *VexxL = pSPARC_ATOM->EXXL[i].VexxUp;

        for (int j = jstart; j < jstop; j++) {
            orbital = orbitals+j*(Nd - 1);
            // Calculate wt_orbital
            for (int jj = 0; jj < Nd - 1; jj++){
                wt_orbital[jj] = (w[jj+1]/int_scale[jj+1])*orbital[jj];
            }

            double *VexxL_orbital = (double *)malloc((Nd-1)*sizeof(double));
            cblas_dgemv(CblasColMajor, CblasNoTrans, Nd - 1, Nd - 1, 1.0, VexxL, Nd - 1, orbital, 1, 0.0, VexxL_orbital, 1);
            double wt_orb_dot_VexxL_orb = cblas_ddot(Nd-1, wt_orbital, 1, VexxL_orbital, 1);

            *Exx += occ[j]*wt_orb_dot_VexxL_orb;

            free(VexxL_orbital);
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
    
            double *VexxL = pSPARC_ATOM->EXXL[i].VexxDw;
    
            for (int j = jstart; j < jstop; j++) {
                orbital = orbitals+j*(Nd - 1);
                // Calculate wt_orbital
                for (int jj = 0; jj < Nd - 1; jj++){
                    wt_orbital[jj] = (w[jj+1]/int_scale[jj+1])*orbital[jj];
                }
    
                double *VexxL_orbital = (double *)malloc((Nd-1)*sizeof(double));
                cblas_dgemv(CblasColMajor, CblasNoTrans, Nd - 1, Nd - 1, 1.0, VexxL, Nd - 1, orbital, 1, 0.0, VexxL_orbital, 1);
                double wt_orb_dot_VexxL_orb = cblas_ddot(Nd-1, wt_orbital, 1, VexxL_orbital, 1);
    
                *Exx += occ[j]*wt_orb_dot_VexxL_orb;
    
                free(VexxL_orbital);
            }
        }
    }

    free(wt_orbital); free(orbitals);

    *Exx *= 0.5;
}