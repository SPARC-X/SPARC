#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "spectralFunctionsAtom.h"
#include "isddftAtom.h"

/**
 * @brief Set up the grid, gradient and laplacian for
 *        modified Chebyshev grid
 *        Refer to Eq. 14-16 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void set_grid_grad_lap(SPARC_ATOM_OBJ *pSPARC_ATOM){
    int N = pSPARC_ATOM->Nd;
    double R = pSPARC_ATOM->Rmax;
    double alpha = pSPARC_ATOM->alpha;
    double beta = pSPARC_ATOM->beta;

    pSPARC_ATOM->x = (double*)malloc((N+1) * sizeof(double));

    // Allocate memory for the differentiation matrix D (column-major order)
    pSPARC_ATOM->D = (double*)malloc((N + 1) * (N + 1) * sizeof(double));

    chebD(N, pSPARC_ATOM->xmax, pSPARC_ATOM->D, pSPARC_ATOM->x);

    pSPARC_ATOM->w = (double *)malloc((N + 1) * sizeof(double));
    clencurt(N, pSPARC_ATOM->w);
    for (int i = 0; i <= N; ++i) {
        pSPARC_ATOM->w[i] *= pSPARC_ATOM->xmax;
    }

    pSPARC_ATOM->r = (double *)malloc((N + 1) * sizeof(double));
    // Compute r grid
    for (int i = 0; i <= N; ++i) {
        double xi = pSPARC_ATOM->x[i];
        pSPARC_ATOM->r[i] = (1.0 / beta) * log(1.0 - xi / alpha);
    }
    pSPARC_ATOM->r[0] = 2 * R;

    pSPARC_ATOM->int_scale = (double *)malloc((N + 1) * sizeof(double));
    // Compute int_scale = -beta * (alpha - x)
    for (int i = 0; i <= N; ++i) {
        pSPARC_ATOM->int_scale[i] = -beta * (alpha - pSPARC_ATOM->x[i]);
    }


    pSPARC_ATOM->grad = (double *)malloc((N + 1) * (N + 1) * sizeof(double));
    pSPARC_ATOM->laplacian = (double *)malloc((N + 1) * (N + 1) * sizeof(double));

    // Compute Gradient matrix: grad = int_scale .* D (column-major order)
    for (int j = 0; j <= N; ++j) {  // Columns
        for (int i = 0; i <= N; ++i) {  // Rows
            pSPARC_ATOM->grad[j * (N + 1) + i] =
                pSPARC_ATOM->int_scale[i] * pSPARC_ATOM->D[j * (N + 1) + i];
        }
    }

    // Compute Laplacian matrix: L = (int_scale.^2) .* D^2 + beta * int_scale .* D
    for (int j = 0; j <= N; ++j) {  // Columns
        for (int i = 0; i <= N; ++i) {  // Rows
            double D2_ij = 0.0;
            for (int k = 0; k <= N; ++k) {
                D2_ij += pSPARC_ATOM->D[k * (N + 1) + i] * pSPARC_ATOM->D[j * (N + 1) + k];
            }
            pSPARC_ATOM->laplacian[j * (N + 1) + i] =
                (pSPARC_ATOM->int_scale[i] * pSPARC_ATOM->int_scale[i]) * D2_ij +
                beta * pSPARC_ATOM->int_scale[i] * pSPARC_ATOM->D[j * (N + 1) + i];
        }
    }

}

/**
 * @brief Calculate the Chebyshev grid and differentiation matrix
 *        for given number, N (resulting in N+1 grid points) and
 *        scaling factor R
 *        Refer to Eq. 20, 21 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void chebD(int N, double R, double* D, double* r){

    // Generate Chebyshev grid points r 
    for (int i = 0; i <= N; i++) {
        r[i] = cos(M_PI * i / N);
        r[i] = R * r[i] + R;
    }

    // Coefficients c (same as in MATLAB)
    double* c = (double*)malloc((N + 1) * sizeof(double));
    for (int i = 0; i <= N; i++) {
        c[i] = (i == 0 || i == N) ? 2.0 : 1.0; // Boundary points (2 for first/last)
        c[i] *= pow(-1.0, i); // Alternating sign
    }

    // Construct the differentiation matrix D (in column-major order)
    for (int j = 0; j <= N; j++) {  // Iterate over columns
        for (int i = 0; i <= N; i++) {  // Iterate over rows
            if (i == j) {
                D[j * (N + 1) + i] = 0.0; // Diagonal entries
            } else {
                double diff = r[i] - r[j];  // Difference X - X'
                D[j * (N + 1) + i] = (c[i] / c[j]) / diff;  // Off-diagonal entries
            }
        }
    }

    // Final step to adjust diagonal entries 
    for (int i = 0; i <= N; i++) {
        double sum = 0.0;
        for (int j = 0; j <= N; j++) {
            if (i != j) {
                sum += D[j * (N + 1) + i];  // Sum off-diagonal elements in column
            }
        }
        D[i * (N + 1) + i] = -sum;  // Diagonal entry
    }

    // Free the coefficient array c
    free(c);
}

/**
 * @brief Calculate the Clenshaw-curtis quadrature weights for the 
 *        Chebyshev grid points.
 *        Refer to Eq. 22 of Bhowmik, S. et al. Comput. Phys. Commun. 308 (2025) 109448
 */
void clencurt(int N, double* w){
    double* theta = (double*)malloc((N + 1) * sizeof(double));
    double* v = (double*)malloc((N - 1) * sizeof(double));

    for (int i = 0; i <= N; i++){
        theta[i] = M_PI * i / N;
    }

    for (int i = 0; i <= N; i++){
        w[i] = 0.0;
    }

    int ii_start = 1, ii_end = N - 1;

    for (int i = 0; i < N - 1; i++) {
        v[i] = 1.0;
    }

    if (N % 2 == 0){
         w[0] = 1.0 / (N * N - 1);
        w[N] = w[0];
        for (int k = 1; k <= N / 2 - 1; k++) {
            for (int i = ii_start; i <= ii_end; i++) {
                v[i - 1] -= 2.0 * cos(2 * k * theta[i]) / (4 * k * k - 1);
            }
        }
        for (int i = ii_start; i <= ii_end; i++) {
            v[i - 1] -= cos(N * theta[i]) / (N * N - 1);
        }
    }else{
        w[0] = 1.0 / (N * N);
        w[N] = w[0];
        for (int k = 1; k <= (N - 1) / 2; k++) {
            for (int i = ii_start; i <= ii_end; i++) {
                v[i - 1] -= 2.0 * cos(2 * k * theta[i]) / (4 * k * k - 1);
            }
        }
    }

    for (int i = ii_start; i <= ii_end; i++) {
        w[i] = 2.0 * v[i - 1] / N;
    }

    free(theta);
    free(v);
}
