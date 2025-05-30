#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif


#include "isddftAtom.h"
#include "mGGAtoolsAtom.h"

void initialize_MGGA_atom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    if (!pSPARC_ATOM) {
        fprintf(stderr, "Error: NULL pointer passed to initialize_MGGA_atom\n");
        return;
    }

    int Nd = pSPARC_ATOM->Nd;
    int nspinor = pSPARC_ATOM->nspinor;
    int nspden = pSPARC_ATOM->nspden;

    if (Nd <= 1 || nspden <= 0) {
        fprintf(stderr, "Error: Invalid dimensions Nd=%d, nspden=%d\n", Nd, nspden);
        return;
    }

    pSPARC_ATOM->tau = (double *)malloc(sizeof(double) * nspden * (Nd - 1));
    if (!pSPARC_ATOM->tau) {
        fprintf(stderr, "Error: Memory allocation failed for pSPARC_ATOM->tau\n");
        exit(EXIT_FAILURE);
    }
    memset(pSPARC_ATOM->tau, 0, sizeof(double) * nspden * (Nd - 1));

    pSPARC_ATOM->vxcMGGA3 = (double *)malloc(sizeof(double) * nspinor * (Nd - 1));
    if (!pSPARC_ATOM->vxcMGGA3) {
        fprintf(stderr, "Error: Memory allocation failed for pSPARC_ATOM->vxcMGGA3\n");
        free(pSPARC_ATOM->tau);  // Avoid leak
        exit(EXIT_FAILURE);
    }
    memset(pSPARC_ATOM->vxcMGGA3, 0, sizeof(double) * nspinor * (Nd - 1));
}

/**
 * @brief calculate kinetic energy density
 * Eq A2 of Sala, Fabiano and Constantin, Phys. Rev. B, 91, 035126 (2015)
 * (2l + 1) factor disappears due to equal smearing
 * Eq 17 of Lehtola,S.; J. Chem. Theory Comput. 2023, 19, 2502−2517.
 */
void kineticEnergyDensityAtom(SPARC_ATOM_OBJ *pSPARC_ATOM) {
    int Nd = pSPARC_ATOM->Nd;
    int nspinor = pSPARC_ATOM->nspinor;
    double *r = pSPARC_ATOM->r; // r size id (Nd+1)
    int *occ = pSPARC_ATOM->occ;
    int *orbital_l = pSPARC_ATOM->orbital_l;
    int val_len = pSPARC_ATOM->val_len;
    double *tau = (double *)calloc(nspinor*(Nd - 1), sizeof(double));

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

    // Extract gradient matrix
    double *D = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
    for (int j = 1; j < Nd; j++) {  // Iterate column-first
        for (int i = 1; i < Nd; i++) {
            D[(i - 1) + (j - 1) * (Nd - 1)] = pSPARC_ATOM->grad[i + j * (Nd+1)];  // Proper column-major indexing
        }
    }

    // Perform dR_{nl}/dr = ((D*orbitals - orbitals./r)./r) where r has the first and last points removed
    // D is (Nd-1)x(Nd-1), orbitals is (Nd-1)x(2Nd-2) both stored in column major format
    double *dR_nl_dr = (double *)malloc(nspinor*val_len*(Nd-1)*sizeof(double));
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                Nd - 1, nspinor * val_len, Nd - 1, 
                1.0, D, Nd - 1, orbitals, Nd - 1, 
                0.0, dR_nl_dr, Nd - 1); // D*orbitals
    for (int j = 0; j < nspinor*val_len; j++) {
        for (int i = 0; i < Nd - 1; i++) {
            dR_nl_dr[i + j * (Nd - 1)] = (dR_nl_dr[i + j * (Nd - 1)] - orbitals[i + j * (Nd - 1)]/r[i+1])/r[i+1];
        }
    }

    int start, stop, l;
    for (int spinor = 0; spinor < nspinor; spinor++) {
        if (spinor == 0) {
            start = 0;
            stop = val_len;
        } else {
            start = val_len;
            stop = 2*val_len;
        }

        for (int i = start; i < stop; i++) {
            l = orbital_l[i];
            for (int jj = 0; jj < Nd - 1; jj++) {
                tau[spinor*(Nd - 1) + jj] += 0.5*occ[i]*pow(dR_nl_dr[i*(Nd-1)+jj],2);
                tau[spinor*(Nd - 1) + jj] += 0.5*occ[i]*l*(l+1)
                                            *pow(orbitals[i*(Nd-1)+jj],2)/pow(r[jj+1],4);
            }
        }
    }

    if (pSPARC_ATOM->spinFlag == 0){
        for (int jj = 0; jj < Nd - 1; jj++) {
            pSPARC_ATOM->tau[jj] = (0.25/M_PI)*tau[jj];
        }
    } else {
        for (int jj = 0; jj < Nd - 1; jj++) {
            double tau_up = (0.25/M_PI) * tau[jj];
            double tau_dw = (0.25/M_PI) * tau[(Nd-1) + jj];
            pSPARC_ATOM->tau[jj] = tau_up + tau_dw; // Total kinetic energy density
            pSPARC_ATOM->tau[(Nd - 1) + jj] = tau_up; // Spin-up component
            pSPARC_ATOM->tau[2*(Nd - 1) + jj] = tau_dw; // Spin-down component
        }
    }

    // for (int jj = 0; jj < pSPARC_ATOM->nspden*(Nd - 1); jj++) {
    //     if (pSPARC_ATOM->tau[jj] < pSPARC_ATOM->xc_rhotol)
    //         pSPARC_ATOM->tau[jj] = pSPARC_ATOM->xc_rhotol; // Might not be required, test
    // }

    free(tau); free(D); free(orbitals); free(dR_nl_dr);
}