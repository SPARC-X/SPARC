#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef USE_MKL
    #define MKL_Complex16 double _Complex
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#include "mGGApotentialAtom.h"
#include "isddftAtom.h"

void mGGA_hamiltonian_term(SPARC_ATOM_OBJ *pSPARC_ATOM, int l, double spin, double *VmGGA) {
    int Nd = pSPARC_ATOM->Nd;
    int nspinor = pSPARC_ATOM->nspinor;
    double *r = pSPARC_ATOM->r;
    double *V3;

    // Extract gradient matrix
    double *D = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
    for (int j = 1; j < Nd; j++) {  // Iterate column-first
        for (int i = 1; i < Nd; i++) {
            D[(i - 1) + (j - 1) * (Nd - 1)] = pSPARC_ATOM->grad[i + j * (Nd+1)];  // Proper column-major indexing
        }
    }

    // Extract laplacian matrix
    double *L = (double *)malloc((Nd - 1) * (Nd - 1) * sizeof(double));
    for (int j = 1; j < Nd; j++) {  // Iterate column-first
        for (int i = 1; i < Nd; i++) {
            L[(i - 1) + (j - 1) * (Nd - 1)] = pSPARC_ATOM->laplacian[i + j * (Nd+1)];  // Proper column-major indexing
        }
    }

    if (spin == 0.5) {
        V3 = pSPARC_ATOM->vxcMGGA3;
    } else {
        V3 = pSPARC_ATOM->vxcMGGA3 + (Nd-1);
    }

    // matrix vector product
    double *v3grad = (double *)malloc(sizeof(double)*(Nd-1));
    cblas_dgemv(CblasColMajor, CblasNoTrans, Nd - 1, Nd - 1, 1.0, D, Nd - 1, V3, 1, 0.0, v3grad, 1);

    // VmGGA = 0.5*r(2:Nd).*(term1 + term2 + term3)
    // term1: l*(l+1)*(V3./(r.^3)) + v3grad./(r.^2)
    // term2: -(V3./r(2:Nd)).*L
    // term3: -(1./r(2:Nd)).*v3grad.*D
    for (int col = 0; col < Nd - 1; col++) {
        for (int row = 0; row < Nd - 1; row++) {
            if (row == col) {
                VmGGA[col*(Nd - 1) + row] += l*(l+1)*V3[row]/pow(r[row+1],3) + v3grad[row]/pow(r[row+1],2);
            }
            VmGGA[col*(Nd - 1) + row] += -(V3[row]/r[row+1])*L[col*(Nd - 1) + row]
                                        -(1/r[row+1])*v3grad[row]*D[col*(Nd - 1) + row];

            VmGGA[col*(Nd - 1) + row] *= 0.5*r[row+1];
        }
    }
    free(D);
    free(L);
    free(v3grad);
}